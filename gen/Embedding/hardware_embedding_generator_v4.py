"""
hardware_embedding_generator_v4.py

Generate physics-aware hardware embeddings (24-dim) with explicit bucketed identity,
TVM constraint logs, and physical performance specs. Designed as a drop-in replacement
for previous v2/v3 generators: call `generate(name: str, tvm_config: dict) -> List[float]`.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import argparse
import json


@dataclass
class HardwarePerfSpec:
    sm_count: int          # GPU SM 数或 CPU 核数
    peak_flops_32: float   # FP32 峰值算力 (TFLOPS)
    mem_bandwidth: float   # 显存/内存带宽 (GB/s)
    l2_cache_mb: float     # L2 缓存 (MB)
    vram_gb: float         # 显存/内存容量 (GB)
    tensor_core_gen: int   # Tensor Core 代数 (0=None, 1=Volta, 2=Turing, 3=Ampere, 4=Ada/Hopper)


# 手工维护的小型性能库；数值按官方规格近似，少数条目标记 TODO 以便后续校正
PERF_DB: Dict[str, HardwarePerfSpec] = {
    # HPC / 高端 GPU
    "nvidia/nvidia-a100": HardwarePerfSpec(108, 19.5, 1555, 40, 80, 3),
    "nvidia/nvidia-v100": HardwarePerfSpec(80, 15.7, 900, 6, 32, 1),
    "nvidia/rtx-4090": HardwarePerfSpec(128, 82.6, 1008, 72, 24, 4),
    "nvidia/geforce-rtx-3090": HardwarePerfSpec(82, 35.6, 936, 6, 24, 3),
    # 中端 GPU
    "nvidia/geforce-rtx-3060": HardwarePerfSpec(28, 12.7, 360, 3, 12, 3),
    # 边缘/嵌入式 GPU
    "nvidia/jetson-agx-xavier": HardwarePerfSpec(8, 1.4, 137, 0.5, 32, 1),
    "nvidia/jetson-orin": HardwarePerfSpec(16, 5.3, 204, 3, 64, 3),  # TODO: verify L2/VRAM
    # CPU
    "aws/cpu/c5.18xlarge": HardwarePerfSpec(36, 2.3, 120, 36, 128, 0),
    # 可选：树莓派（占位，待校正）
    "raspberry-pi/4b-aarch64": HardwarePerfSpec(4, 0.1, 30, 1, 4, 0),  # TODO: rough guess
}


def _log2(x: float, eps: float = 1e-6) -> float:
    x = max(x, eps)
    return math.log2(x)


def _log10(x: float, eps: float = 1e-6) -> float:
    x = max(x, eps)
    return math.log10(x)


class EmbeddingV4Generator:
    def __init__(self):
        self.perf_db: Dict[str, HardwarePerfSpec] = PERF_DB
        # 方便不区分大小写匹配
        self._perf_db_lower = {k.lower(): v for k, v in self.perf_db.items()}

    def _get_perf(self, name: str) -> HardwarePerfSpec:
        perf = self._perf_db_lower.get(name.lower())
        if perf is None:
            # 兜底：用极小规格，提示未命中
            print(f"[WARN] hardware {name} not found in PERF_DB, using fallback perf spec.")
            perf = HardwarePerfSpec(sm_count=1, peak_flops_32=1.0, mem_bandwidth=1.0, l2_cache_mb=1.0, vram_gb=1.0, tensor_core_gen=0)
        return perf

    def _get(self, cfg: Dict[str, Any], key: str, default: Any) -> Any:
        return cfg.get(key, default)

    def generate(self, name: str, tvm_config: Dict[str, Any]) -> List[float]:
        # -------- A. Identity & Arch (0–7) --------
        cfg = tvm_config or {}
        kind = str(self._get(cfg, "kind", "")).lower()
        arch = str(self._get(cfg, "arch", "")).lower()
        mcpu = str(self._get(cfg, "mcpu", "")).lower()
        mtriple = str(self._get(cfg, "mtriple", "")).lower()
        mattr = self._get(cfg, "mattr", []) or []
        mattr_str = " ".join(mattr).lower()

        perf = self._get_perf(name)

        is_gpu = kind == "cuda"
        is_gpu_hpc = 1.0 if (is_gpu and perf.peak_flops_32 > 10.0) else 0.0
        is_gpu_edge = 1.0 if (is_gpu and perf.peak_flops_32 <= 10.0) else 0.0
        is_cpu_x86 = 1.0 if (kind == "llvm" and "aarch64" not in mtriple) else 0.0
        is_cpu_arm = 1.0 if (kind == "llvm" and ("aarch64" in mtriple or "neon" in mattr_str)) else 0.0
        has_tensor_core = 1.0 if perf.tensor_core_gen > 0 else 0.0
        # Reserved flags to avoid strong priors in small data; force to 0.0
        #     is_ampere_plus = 1.0 if perf.tensor_core_gen >= 3 else 0.0
        #    has_avx512 = 1.0 if (kind == "llvm" and ("avx512" in mcpu or "avx512" in mattr_str)) else 0.0
        #   is_arm_neon = 1.0 if (kind == "llvm" and ("neon" in mcpu or "neon" in mattr_str)) else 0.0
        is_ampere_plus = 0.0
        has_avx512 = 0.0
        is_arm_neon = 0.0

        # -------- B. Constraints (8–11) --------
        mtpb = float(self._get(cfg, "max_threads_per_block", 1024))
        warp = float(self._get(cfg, "thread_warp_size", 32))
        smem = float(self._get(cfg, "max_shared_memory_per_block", 49152))
        regs = float(self._get(cfg, "registers_per_block", 65536))

        # -------- C. Scale & Performance (12–19) --------
        sm_count = float(perf.sm_count)
        flops = float(perf.peak_flops_32)
        bw = float(perf.mem_bandwidth)
        l2 = float(perf.l2_cache_mb)
        vram = float(perf.vram_gb)
        tc_gen = float(perf.tensor_core_gen)

        # -------- D. Environment (20–23) --------
        name_l = name.lower()
        is_unified = 1.0 if any(tag in name_l for tag in ["jetson", "raspberry", "apple"]) else 0.0
        is_cloud = 1.0 if any(tag in name_l for tag in ["aws/", "gcp/", "azure/"]) else 0.0

        vec: List[float] = [
            # A. Identity
            is_gpu_hpc,        # 0
            is_gpu_edge,       # 1
            is_cpu_x86,        # 2
            is_cpu_arm,        # 3
            has_tensor_core,   # 4
            is_ampere_plus,    # 5
            has_avx512,        # 6
            is_arm_neon,       # 7
            # B. Constraints (log2)
            _log2(mtpb),                   # 8
            _log2(warp),                   # 9
            _log2(smem / 1024.0),          # 10 (KB)
            _log2(regs),                   # 11
            # C. Scale & Performance (log10)
            _log10(sm_count),              # 12
            _log10(flops),                 # 13
            _log10(bw),                    # 14
            _log10(l2),                    # 15
            _log10(vram),                  # 16
            tc_gen / 5.0,                  # 17 (normalized)
            0.0,                           # 18 reserved
            0.0,                           # 19 reserved
            # D. Environment
            is_unified,                    # 20
            is_cloud,                      # 21
            0.0,                           # 22 reserved_env_1
            0.0,                           # 23 reserved_env_2
        ]

        assert len(vec) == 24, f"Embedding length {len(vec)} != 24"
        return vec


def _default_tvm_config(name: str) -> Dict[str, Any]:
    """Best-effort default TVM config by name for offline export."""
    name_l = name.lower()
    # 默认先填 CPU-safe 的最小值，GPU 再覆盖
    cfg = {
        "max_threads_per_block": 1,
        "thread_warp_size": 1,
        "max_shared_memory_per_block": 1,
        "registers_per_block": 1,
    }
    if "nvidia" in name_l or "rtx" in name_l:
        cfg["kind"] = "cuda"
        # GPU 复写合理的 CUDA 约束默认值
        cfg["max_threads_per_block"] = 1024
        cfg["thread_warp_size"] = 32
        cfg["max_shared_memory_per_block"] = 49152
        cfg["registers_per_block"] = 65536
        if "v100" in name_l:
            cfg["arch"] = "sm_70"
        elif "a100" in name_l:
            cfg["arch"] = "sm_80"
        elif "3090" in name_l or "3060" in name_l:
            cfg["arch"] = "sm_86"
        elif "4090" in name_l:
            cfg["arch"] = "sm_86"
        elif "xavier" in name_l:
            cfg["arch"] = "sm_72"
        elif "orin" in name_l:
            cfg["arch"] = "sm_87"
        else:
            cfg["arch"] = "sm_86"
    else:
        cfg["kind"] = "llvm"
        cfg["mtriple"] = "x86_64-linux-gnu"
        cfg["mcpu"] = "generic"
        if "c5.18xlarge" in name_l or "xeon" in name_l:
            cfg["mcpu"] = "skylake-avx512"
        if "raspberry" in name_l or "aarch64" in name_l:
            cfg["mtriple"] = "aarch64-linux-gnu"
            cfg["mcpu"] = "cortex-a72"
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default="/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json",
        help="Where to write the generated hardware embeddings JSON.",
    )
    args = parser.parse_args()

    gen = EmbeddingV4Generator()

    sample_v100 = {
        "kind": "cuda",
        "arch": "sm_70",
        "max_threads_per_block": 1024,
        "thread_warp_size": 32,
        "max_shared_memory_per_block": 49152,
        "registers_per_block": 65536,
    }
    sample_4090 = {
        "kind": "cuda",
        "arch": "sm_86",
        "max_threads_per_block": 1024,
        "thread_warp_size": 32,
        "max_shared_memory_per_block": 49152,
        "registers_per_block": 65536,
    }
    sample_xavier = {
        "kind": "cuda",
        "arch": "sm_72",
        "max_threads_per_block": 1024,
        "thread_warp_size": 32,
        "max_shared_memory_per_block": 49152,
        "registers_per_block": 65536,
        "mtriple": "aarch64-linux-gnu",
        "mcpu": "carmel",
        "mattr": ["+neon"],
    }

    v_v100 = gen.generate("nvidia/nvidia-v100", sample_v100)
    v_4090 = gen.generate("nvidia/rtx-4090", sample_4090)
    v_xavier = gen.generate("nvidia/jetson-agx-xavier", sample_xavier)

    print("v100 len:", len(v_v100), "head:", v_v100[:16], "perf[12:17]", v_v100[12:17])
    print("4090 len:", len(v_4090), "head:", v_4090[:16], "perf[12:17]", v_4090[12:17])
    print("xavier len:", len(v_xavier), "head:", v_xavier[:16], "perf[12:17]", v_xavier[12:17])

    # 导出全量 PERF_DB 到 JSON
    entries = []
    for name in sorted(PERF_DB.keys()):
        tvm_cfg = _default_tvm_config(name)
        vec = gen.generate(name, tvm_cfg)
        entries.append({"hardware_name": name, "vector": vec})
    with open(args.output_path, "w", encoding="utf-8") as fout:
        json.dump(entries, fout, ensure_ascii=False, indent=2)
    print(f"Exported {len(entries)} embeddings to {args.output_path}")
