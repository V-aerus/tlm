#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_hw_emb_v5.py

目的
----
把你们现有的 v4-universe 硬件库 (/gen/Embedding/hardware_embeddings_v4_universe.json)
“程序化地”转换成 emb-v5（你们刚刚讨论敲定的新 schema），并输出 JSON。

为什么要这样写？
- 你可以在一个 Python 文件里直接维护“每个硬件的缺失字段/近似值”的备注；
- 输出过程可复现（同一份输入 + 同一份表 => 同一份 emb-v5 JSON）；
- 后续你要增添新硬件、修正某个数字，只改这里的 MANUAL_OVERRIDES / TABLES 即可。

输入/输出
---------
输入：v4-universe 列表（每项含 hardware_name, vector[24]）
输出：
1) emb-v5 向量 JSON（列表格式，兼容你们 v4 的写法）：
   [
     {"hardware_name": "...", "vector": [..24 floats..]},
     ...
   ]
2) 可选：meta JSON（包含 schema + 每个硬件的字段来源/备注，方便你写论文/复核）

注意（非常重要）
--------------
- v4 里只有：GPU 的 TVM 约束（log2）+ 一小套 “性能库 PERF_DB” 产生的 spec（log10）。
  v5 新增的字段（例如 CPU L3 / peak_matmul / Jetson SoC CPU 核数等）需要额外补齐。
- 本脚本的目标是做到“可复现 + 可审计”：
  - 尽量使用官方来源（NVIDIA/Intel ARK/AMD/AWS/Ampere 官方页面或 PDF）。
  - 如果某字段拿不到可追溯的官方来源，宁可置 0 并在 meta 里标注 TODO。

用法示例
--------
python gen_hw_emb_v5.py \
  --v4-universe /mnt/data/hardware_embeddings_v4_universe.json \
  --out /mnt/data/hardware_embeddings_v5.json \
  --out-meta /mnt/data/hardware_embeddings_v5_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# 0) emb-v5 schema (24-dim)
# -----------------------------
DIM_V5 = 24

IDX = {
    # A: Identity & Arch (0-3)
    "is_gpu_hpc": 0,
    "is_gpu_edge": 1,
    "is_cpu_x86": 2,
    "is_cpu_arm": 3,

    # B: TVM constraints / ABI-ish (4-10)
    "num_cores": 4,                    # CPU / SoC CPU 核数；GPU-only 设备可置 0
    "vector_unit_bytes": 5,            # CPU SIMD 宽度（bytes）；ARM NEON=16, AVX2=32, AVX-512=64
    "cache_line_bytes": 6,             # CPU cache line bytes（一般 64）
    "max_shared_memory_per_block": 7,  # CUDA shared memory bytes（CPU N/A=0）
    "max_threads_per_block": 8,        # CUDA threads-per-block（CPU N/A=0）
    "registers_per_block": 9,          # CUDA registers-per-block（CPU N/A=0）
    "warp_size": 10,                   # CUDA warp size（CPU N/A=0）

    # C: Scale / Performance (11-22)
    "sm_count": 11,                    # GPU SM count（CPU N/A=0）
    "peak_fp32": 12,                   # TFLOPS
    "peak_matmul": 13,                 # TFLOPS（TensorCore/AMX/SVE2 等；没有就 0）
    "mem_bandwidth": 14,               # GB/s
    "llc_mb": 15,                      # GPU L2 (MB) / CPU L3 (MB)
    "mid_cache_kb": 16,                # CPU L2 (KB)；GPU N/A=0
    "device_mem_gb": 17,               # GPU VRAM / UMA 可用内存 / CPU RAM（慎用）
    "matmul_accel_ratio": 18,          # log10(peak_matmul)-log10(peak_fp32)
    "lowp_level": 19,                  # 0=FP32 only, 1=FP16/BF16, 2=INT8
    "compute_bw_ratio": 20,            # log10(peak_fp32)-log10(mem_bw)
    "cache_bw_ratio": 21,              # log10(llc_mb)-log10(mem_bw)
    "bw_per_sm": 22,                   # log10(mem_bw)-log10(sm_count) (GPU-only)

    # D: Environment (23)
    "mem_is_uma": 23,                  # Jetson/RPi/Apple 等统一内存
}


# -----------------------------
# 1) v4-universe parsing helpers
# -----------------------------
def load_v4_universe(path: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    v4-universe 文件格式（你们 repo 里实际用的）是 list[{"hardware_name", "vector"}]。
    这里返回：保持原顺序的 names list，以及 name->vector(24) 的 map。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names: List[str] = []
    mp: Dict[str, List[float]] = {}
    for item in data:
        name = item["hardware_name"]
        vec = item["vector"]
        if len(vec) != 24:
            raise ValueError(f"v4 vector dim mismatch for {name}: got {len(vec)}")
        names.append(name)
        mp[name] = vec
    return names, mp


def inv_log2(x: float) -> float:
    return float(2.0 ** x)


def inv_log2_na0(x: float) -> float:
    """
    v4 里 CUDA 约束（threads/warp/shared-mem/regs）采用 log2 编码。
    对 CPU-only/N/A 的情况，v4 常用 0 或负数作为占位。

    这里做“语义保护”：
      - 若 log2 值 <= 0，直接输出 0（保持 N/A=0 语义）
      - 否则输出 2**x
    """
    if x <= 0.0:
        return 0.0
    return float(2.0 ** x)


def inv_log10(x: float) -> float:
    return float(10.0 ** x)


def safe_log10(x: float, eps: float = 1e-12) -> float:
    return math.log10(max(x, eps))


def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if den <= eps:
        return 0.0
    return float(num / den)


# -----------------------------
# 2) Identity inference (A-seg)
# -----------------------------
def infer_identity(name: str) -> Tuple[float, float, float, float]:
    """
    你们讨论里 A 段是“类别/ABI”，更像 bucket token 的语义：
      - is_gpu_hpc: 离散 GPU（RTX/GTX/A100/V100...）
      - is_gpu_edge: SoC/边缘 GPU（Jetson；RPi 这里也先算 edge，便于你们把 UMA 设备聚类）
      - is_cpu_x86
      - is_cpu_arm

    注意：这里是“启发式”，你也可以改成从 canonical target 里解析 keys。
    """
    n = name.lower()

    is_cpu_x86 = 1.0 if (n.startswith("intel/") or n.startswith("amd/") or n.startswith("aws/cpu")) else 0.0
    is_cpu_arm = 1.0 if (
        n.startswith("raspberry-pi/")
        or n.startswith("aws/graviton")
        or n.startswith("ampere/")
        or ("aarch64" in n)
        or ("arm" in n and not n.startswith("nvidia/"))
    ) else 0.0

    # Jetson / SoC：既有 GPU（cuda）又常见 ARM CPU
    is_jetson = 1.0 if "jetson" in n else 0.0
    # 注意：RPi 在你们的 TLM/TVM 场景里不参与 CUDA schedule，因此不要把它当 GPU（会污染 valid mask）
    is_gpu_edge = 1.0 if is_jetson else 0.0

    # 其他 nvidia/ 都视为离散 GPU（hpc）
    is_gpu_hpc = 1.0 if (n.startswith("nvidia/") and not is_gpu_edge) else 0.0

    # SoC 同时标记 ARM CPU（对齐你们“一个设备有 cuda+llvm 两个 target”的讨论）
    if is_jetson:
        is_cpu_arm = 1.0

    return is_gpu_hpc, is_gpu_edge, is_cpu_x86, is_cpu_arm


# -----------------------------
# 3) Manual tables / overrides
# -----------------------------
# 3.1 CPU / SoC 侧：vector width (bytes)
VECTOR_UNIT_BYTES: Dict[str, Tuple[int, str]] = {
    # x86：常见近似
    "aws/cpu/c5.18xlarge": (64, "from logs: skylake-avx512 => 512-bit = 64B"),
    "intel/xeon-gold-6226": (64, "APPROX: skylake-avx512 class => assume AVX-512 (64B)"),
    "intel/xeon-platinum-8480c": (
        64,
        "official: Intel ARK (Instruction Set Extensions includes AVX-512) => 512-bit = 64B; "
        "src=https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-platinum-8480c-processor-105m-cache-2-00-ghz/specifications.html",
    ),
    "intel/xeon-platinum-8380": (
        64,
        "official: Intel ARK (Instruction Set Extensions includes AVX-512) => 512-bit = 64B; "
        "src=https://www.intel.com/content/www/us/en/products/sku/212285/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz/specifications.html",
    ),
    "intel/core-i5-12400": (32, "APPROX: Alder Lake (no AVX-512) => AVX2 256-bit = 32B"),
    "intel/core-i7-12700k": (32, "APPROX: Alder Lake (no AVX-512) => AVX2 32B"),
    "intel/core-i7-10510u": (32, "APPROX: Comet Lake U => AVX2 32B"),
    "amd/ryzen-7-5800h": (32, "APPROX: Zen3 mobile => AVX2 32B"),
    "amd/epyc-9654": (
        64,
        "official: AMD EPYC 9004 tuning guide (mentions AVX-512) => 64B; "
        "src=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-9004-tg-58011.pdf",
    ),
    "amd/epyc-7763": (
        32,
        "official: AMD EPYC 7003 microarchitecture overview (Zen 3) => x86 SIMD width set to AVX2=32B; "
        "src=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-7003-series-microarchitecture-overview.pdf "
        "(TODO: add explicit ISA list reference)",
    ),
    # ARM：NEON 通常 128-bit
    "raspberry-pi/4b-aarch64": (16, "APPROX: ARM NEON 128-bit = 16B"),
    "nvidia/jetson-agx-xavier": (16, "APPROX: ARMv8 NEON 128-bit = 16B"),
    "nvidia/jetson-orin": (16, "APPROX: ARMv8 NEON 128-bit = 16B"),
    "nvidia/jetson-orin-nano": (16, "APPROX: ARMv8 NEON 128-bit = 16B"),
    "nvidia/jetson-orin-nx": (16, "APPROX: ARMv8 NEON 128-bit = 16B"),
    "nvidia/jetson-agx-orin": (16, "APPROX: ARMv8 NEON 128-bit = 16B"),
    "aws/graviton3": (
        32,
        "official: AWS Graviton technical guide (SIMD includes 2x256b SVE) => 32B; "
        "src=https://aws.github.io/graviton/",
    ),
    "ampere/altra-max": (
        16,
        "official: Ampere Altra Max product brief (2x128b vector units) => 16B; "
        "src=https://amperecomputing.com/assets/documents/Ampere_Altra_Max_Product_Brief.pdf",
    ),
}

# 3.2 CPU / SoC：cache line bytes（几乎都 64）
CACHE_LINE_BYTES_DEFAULT = 64

# 3.3 SoC CPU 核数（因为 v4 的 sm_count 对 Jetson 是 GPU SM，不是 CPU cores）
SOC_CPU_CORES: Dict[str, Tuple[int, str]] = {
    "nvidia/jetson-agx-xavier": (8, "APPROX: AGX Xavier CPU is 8-core Carmel"),
    "nvidia/jetson-orin": (12, "APPROX: Orin CPU is typically 12-core ARM (varies by SKU)"),
    "nvidia/jetson-orin-nano": (
        6,
        "official: NVIDIA Jetson Orin technical specs (CPU: 6-core Arm Cortex-A78AE); "
        "src=https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
    ),
    "nvidia/jetson-orin-nx": (
        8,
        "official: NVIDIA Jetson Orin NX datasheet (CPU: 8-core Arm Cortex-A78AE); "
        "src=https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/jetson_orin_nx_series_modules_ds-10712.pdf",
    ),
    "nvidia/jetson-agx-orin": (
        12,
        "official: NVIDIA Jetson AGX Orin technical brief (CPU: 12-core Arm Cortex-A78AE); "
        "src=https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx-orin/Jetson_AGX_Orin_Technical_Brief.pdf",
    ),
    # RPi v4-universe 里 sm_count=4 已经是 CPU cores，这里可不写
}

# 3.4 CPU LLC(L3) MB（v4 的 l2_cache_mb 对 CPU 更像 L2 total；v5 想要 llc_mb=CPU L3）
CPU_L3_MB: Dict[str, Tuple[float, str]] = {
    "aws/cpu/c5.18xlarge": (24.75, "APPROX: Intel Xeon Platinum 8124M has 24.75MB L3 (check your exact AWS SKU)"),
    "intel/xeon-gold-6226": (19.25, "APPROX: common spec for Xeon Gold 6226 L3"),
    "intel/xeon-platinum-8480c": (105.0, "official: Intel ARK (Cache = 105 MB)"),
    "intel/xeon-platinum-8380": (60.0, "official: Intel ARK (Cache = 60 MB)"),
    "amd/epyc-9654": (384.0, "official: AMD EPYC 9004 series datasheet (L3 cache = 384 MB)"),
    "amd/epyc-7763": (256.0, "official: AMD EPYC 7003 press release SKU table (L3 cache = 256 MB)"),
    "aws/graviton3": (32.0, "official: AWS Graviton technical guide (SLC/LLC = 32 MB); src=https://aws.github.io/graviton/"),
    "ampere/altra-max": (16.0, "official: Ampere Altra Max product brief (System Level Cache = 16 MB)"),
    "intel/core-i5-12400": (18.0, "APPROX: Intel ARK i5-12400 L3=18MB"),
    "intel/core-i7-12700k": (25.0, "APPROX: Intel ARK i7-12700K L3=25MB"),
    "intel/core-i7-10510u": (8.0, "APPROX: Intel ARK i7-10510U L3=8MB"),
    "amd/ryzen-7-5800h": (16.0, "APPROX: Zen3 5800H L3=16MB"),
    "raspberry-pi/4b-aarch64": (0.0, "RPi typically no large shared L3; set 0"),
    # Jetson：CPU L3 视 SKU/SoC 而定；这里先置 0，后续你可补
    "nvidia/jetson-agx-xavier": (0.0, "TODO: Xavier CPU LLC unclear; set 0"),
    "nvidia/jetson-orin": (0.0, "TODO: Orin CPU LLC unclear; set 0"),
    "nvidia/jetson-orin-nano": (4.0, "official: NVIDIA Jetson Orin technical specs (CPU: 1.5MB L2 + 4MB L3)"),
    "nvidia/jetson-orin-nx": (4.0, "official: NVIDIA Jetson Orin technical specs (CPU: 2MB L2 + 4MB L3)"),
    "nvidia/jetson-agx-orin": (6.0, "official: NVIDIA Jetson AGX Orin technical brief (CPU: 3MB L2 + 6MB L3)"),
}

# 3.5 CPU peak_fp32 (TFLOPS) via *base frequency* (GHz)
#
# 设计动机：
# - 统一使用 base frequency（SKU 固定规格），保证可复现、跨平台不漂；
# - turbo / all-core / AVX 频率依赖功耗与散热策略，属于运行环境因素，不进入 embedding 数值。
#
# 说明：
# - 下面 base_freq_ghz 建议你们最终用 datasheet / Intel ARK / 官方 brief 逐项核对；
# - 对 hybrid core（如 12700K），用 P/E base frequency 加权求和，避免“混在一起”。
CPU_BASE_FREQ_GHZ: Dict[str, Tuple[float, str]] = {
    # x86 / server
    "aws/cpu/c5.18xlarge": (3.0, "Xeon Platinum 8124M base 3.0GHz (verify AWS SKU; c5.18xlarge is 18C/36T)"),
    "intel/xeon-gold-6226": (2.7, "Intel ARK: Processor Base Frequency 2.70 GHz"),
    "intel/xeon-platinum-8480c": (2.0, "Intel ARK: Processor Base Frequency 2.00 GHz"),
    "intel/xeon-platinum-8380": (2.3, "official: Intel ARK: Processor Base Frequency 2.30 GHz"),
    "amd/epyc-9654": (2.4, "official: AMD EPYC 9004 series datasheet: base freq 2.40 GHz; src=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/data-sheets/amd-epyc-9004-series-processors-data-sheet.pdf"),
    "amd/epyc-7763": (2.45, "official: AMD EPYC 7003 press release SKU table: base freq 2.45 GHz"),
    "aws/graviton3": (2.6, "official: AWS Graviton technical guide: Frequency 2600MHz; src=https://aws.github.io/graviton/"),
    "ampere/altra-max": (3.0, "official: Ampere Altra Max product brief: consistent frequency up to 3.0 GHz"),
    # x86 / desktop
    "intel/core-i5-12400": (2.5, "Intel ARK: Performance-core Base Frequency 2.50 GHz"),
    "intel/core-i7-12700k": (3.6, "Intel ARK: Performance-core Base Frequency 3.60 GHz (hybrid; see CPU_HYBRID_TOPOLOGY)"),
    # x86 / mobile
    "intel/core-i7-10510u": (1.8, "Intel ARK: Processor Base Frequency 1.80 GHz"),
    "amd/ryzen-7-5800h": (3.2, "Common spec: base frequency 3.2 GHz (verify)"),
    # ARM
    "raspberry-pi/4b-aarch64": (1.5, "Raspberry Pi 4: 1.5GHz base clock (official brief)"),
}

# 可选：混合架构 CPU（P/E cores）更可解释的 base 频率聚合方式。
# 若某个 CPU 在此表中，则 peak_fp32 用该表计算；否则回退到单一 base_freq_ghz * num_cores。
CPU_HYBRID_TOPOLOGY: Dict[str, Tuple[int, float, int, float, str]] = {
    # name: (p_cores, p_base_ghz, e_cores, e_base_ghz, note)
    "intel/core-i7-12700k": (8, 3.6, 4, 2.7, "Intel ARK: 8P@3.6GHz + 4E@2.7GHz (base)"),
}

# CPU FP32 峰值推导假设：每核每周期可持续发射的 vector FMA 单元数量。
# 这是一个 micro-arch 相关的常数近似；用于提供“可解释的上限”，不追求 cycle-accurate。
DEFAULT_X86_FMA_UNITS_PER_CORE = 2
DEFAULT_ARM_FMA_UNITS_PER_CORE = 1


# 3.6 peak_matmul (TFLOPS) & lowp_level
# 说明：
# - 这些是“最容易缺/最容易被老师追问来源”的字段，所以我刻意把它们单独写成表 + NOTE。
# - peak_matmul：GPU 用 dense Tensor TFLOPS（FP16/BF16/TF32 一类，具体可选）；CPU 如果未来要严谨，请按 ISA/AMX/SVE2 推导。
# - lowp_level：是“能力层级”的粗量化（0/1/2），便于 router；不是精确性能。
# 口径：统一用 dense Tensor TFLOPS（FP16/BF16/TF32）。如果只拿到 sparse marketing 值，则按 /2 近似 dense。
PEAK_MATMUL_TFLOPS: Dict[str, Tuple[float, str]] = {
    # Data center
    "nvidia/nvidia-h100-sxm": (989.5, "official: NVIDIA H100 product page lists BF16/FP16 Tensor 1,979 TFLOPS (with sparsity) => dense ~= /2"),
    "nvidia/nvidia-a100": (312.0, "APPROX: A100 Tensor TFLOPS (FP16/BF16 dense) order-of-magnitude"),
    "nvidia/nvidia-v100": (125.0, "APPROX: V100 Tensor TFLOPS (FP16 dense)"),
    "nvidia/nvidia-a40": (
        149.7,
        "official: NVIDIA A40 datasheet (FP16/BF16 Tensor TFLOPS); src=https://resources.nvidia.com/en-us-brief/rtx-a40-datasheet",
    ),
    "nvidia/nvidia-a10": (
        125.0,
        "official: NVIDIA A10 product page (FP16/BF16 Tensor TFLOPS, non-sparsity value); "
        "src=https://www.nvidia.com/en-us/data-center/products/a10-gpu/",
    ),
    "nvidia/nvidia-t4": (
        65.0,
        "official: NVIDIA T4 datasheet (FP16/FP32 mixed-precision Tensor = 65 TFLOPS); "
        "src=https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
    ),
    # Consumer
    # 统一规则：marketing Tensor TFLOPS 多为 sparse 值，这里用 /2 近似 dense
    "nvidia/rtx-4090": (660.5, "APPROX: 1321 Tensor-TFLOPs (sparse) / 2 => dense"),
    "nvidia/geforce-rtx-3090": (142.5, "APPROX: 285 Tensor-TFLOPs (sparse) / 2 => dense"),
    "nvidia/geforce-rtx-3060": (50.5, "APPROX: 101 Tensor-TFLOPs (sparse) / 2 => dense"),
    "nvidia/geforce-rtx-2080-ti": (110.0, "APPROX: RTX2080Ti Tensor TFLOPS"),
    # Edge SoC
    "nvidia/jetson-orin": (39.0, "SPEC: Jetson AGX Orin 64GB dense FP16 Tensor TFLOPS ~= 39 (v4 mem_gb=64 indicates this SKU)"),
    "nvidia/jetson-agx-xavier": (11.0, "SPEC: Xavier FP16 TFLOPS ~= 11 (official spec)"),
    "nvidia/jetson-orin-nano": (0.0, "TODO: Orin Nano Tensor TFLOPS not provided in official spec table; set 0"),
    "nvidia/jetson-orin-nx": (0.0, "TODO: Orin NX Tensor TFLOPS not provided in official spec table; set 0"),
    "nvidia/jetson-agx-orin": (0.0, "TODO: AGX Orin Tensor TFLOPS not provided in official spec table; set 0"),
    # No tensor core
    "nvidia/geforce-gtx-1060": (0.0, "no tensor core"),
    "nvidia/geforce-gtx-950": (0.0, "no tensor core"),
    "nvidia/geforce-gtx-1080-ti": (0.0, "no tensor core"),
    "nvidia/geforce-gtx-980-ti": (0.0, "no tensor core"),
    # CPUs (unknown => 0)
    "aws/cpu/c5.18xlarge": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "intel/xeon-gold-6226": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "intel/core-i7-12700k": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "intel/core-i5-12400": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "intel/core-i7-10510u": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "amd/ryzen-7-5800h": (0.0, "CPU peak_matmul not modeled here (set 0)"),
    "raspberry-pi/4b-aarch64": (0.0, "CPU peak_matmul not modeled here (set 0)"),
}

LOWP_LEVEL: Dict[str, Tuple[int, str]] = {
    # NVIDIA tensor core => generally supports FP16/BF16; INT8 varies by gen (这里用粗规则)
    "nvidia/nvidia-h100-sxm": (2, "tensor core + int8 supported => 2"),
    "nvidia/nvidia-a100": (2, "tensor core + int8 supported => 2"),
    "nvidia/nvidia-v100": (1, "APPROX: treat Volta as lowp=1 (FP16/BF16)"),
    "nvidia/nvidia-a40": (2, "tensor core + int8 => 2"),
    "nvidia/nvidia-a10": (2, "tensor core + int8 => 2"),
    "nvidia/nvidia-t4": (2, "tensor core + int8 => 2"),
    "nvidia/rtx-4090": (2, "tensor core + int8 => 2"),
    "nvidia/geforce-rtx-3090": (2, "tensor core + int8 => 2"),
    "nvidia/geforce-rtx-3060": (2, "tensor core + int8 => 2"),
    "nvidia/geforce-rtx-2080-ti": (2, "tensor core + int8 => 2"),
    "nvidia/jetson-orin": (2, "tensor core + int8 => 2"),
    "nvidia/jetson-agx-xavier": (2, "tensor core => 2 (APPROX)"),
    "nvidia/jetson-orin-nano": (2, "tensor core + int8 => 2"),
    "nvidia/jetson-orin-nx": (2, "tensor core + int8 => 2"),
    "nvidia/jetson-agx-orin": (2, "tensor core + int8 => 2"),
    "nvidia/geforce-gtx-1060": (0, "no tensor => 0"),
    "nvidia/geforce-gtx-950": (0, "no tensor => 0"),
    "nvidia/geforce-gtx-1080-ti": (0, "no tensor => 0"),
    "nvidia/geforce-gtx-980-ti": (0, "no tensor => 0"),
    # CPUs：若你未来加入 AMX/i8mm/VNNI，可改成 1/2
    "aws/cpu/c5.18xlarge": (0, "CPU lowp level not modeled => 0"),
    "intel/xeon-gold-6226": (0, "CPU lowp level not modeled => 0"),
    "intel/core-i7-12700k": (0, "CPU lowp level not modeled => 0"),
    "intel/core-i5-12400": (0, "CPU lowp level not modeled => 0"),
    "intel/core-i7-10510u": (0, "CPU lowp level not modeled => 0"),
    "amd/ryzen-7-5800h": (0, "CPU lowp level not modeled => 0"),
    "raspberry-pi/4b-aarch64": (0, "CPU lowp level not modeled => 0"),
}

# 3.7 v5 额外硬件（不依赖 v4-universe）
# 说明：你要求“不要改 v4”，因此把新增硬件的 raw spec 放在 v5 内部。
# 这些 raw spec 只在 v4-universe 中缺失时生效。
EXTRA_V5_PERF: Dict[str, Dict[str, float]] = {
    # CPU (raw spec)
    "intel/xeon-platinum-8480c": {"num_cores": 56, "mem_bandwidth": 307.2, "l2_mb": 112.0},
    "intel/xeon-platinum-8380": {"num_cores": 40, "mem_bandwidth": 204.8, "l2_mb": 50.0},
    "amd/epyc-9654": {"num_cores": 96, "mem_bandwidth": 460.8, "l2_mb": 96.0},
    "amd/epyc-7763": {"num_cores": 64, "mem_bandwidth": 204.8, "l2_mb": 32.0},
    "aws/graviton3": {"num_cores": 64, "mem_bandwidth": 307.2, "l2_mb": 64.0},
    "ampere/altra-max": {"num_cores": 128, "mem_bandwidth": 204.8, "l2_mb": 128.0},
    # GPU / Jetson (raw spec)
    "nvidia/nvidia-h100-sxm": {"sm_count": 132, "peak_fp32": 67.0, "mem_bandwidth": 3350.0, "l2_mb": 50.0, "vram_gb": 80.0},
    "nvidia/nvidia-a40": {"sm_count": 84, "peak_fp32": 37.4, "mem_bandwidth": 696.0, "l2_mb": 0.0, "vram_gb": 48.0},
    "nvidia/nvidia-a10": {"sm_count": 72, "peak_fp32": 31.2, "mem_bandwidth": 600.0, "l2_mb": 0.0, "vram_gb": 24.0},
    # NOTE: NVIDIA T4 官方资料存在“datasheet=300GB/s vs product brief=320GB/s”的口径差异。
    # v5 这里优先对齐 datasheet（更可审计），见 EXTRA_V5_PERF_NOTE。
    # NOTE: NVIDIA official materials differ on T4 memory bandwidth:
    # - datasheet (Mar19): 300 GB/s
    # - product page: 320+ GB/s
    # We choose the datasheet value (more conservative + stable PDF reference).
    "nvidia/nvidia-t4": {"sm_count": 40, "peak_fp32": 8.1, "mem_bandwidth": 300.0, "l2_mb": 0.0, "vram_gb": 16.0},
    "nvidia/geforce-gtx-1080-ti": {"sm_count": 28, "peak_fp32": 11.34, "mem_bandwidth": 484.4, "l2_mb": 0.0, "vram_gb": 11.0},
    "nvidia/geforce-gtx-980-ti": {"sm_count": 22, "peak_fp32": 6.06, "mem_bandwidth": 336.6, "l2_mb": 0.0, "vram_gb": 6.0},
    # Jetson Orin (model the current "Super" spec table on NVIDIA official page)
    # - Orin Nano 8GB (Super): 1024 CUDA cores @ 1020MHz => 2.08896 TFLOPS; mem_bw=102 GB/s
    # - Orin NX 16GB (Super): 1024 CUDA cores @ 1173MHz => 2.402304 TFLOPS; mem_bw=102.4 GB/s
    "nvidia/jetson-orin-nano": {"sm_count": 8, "peak_fp32": 2.08896, "mem_bandwidth": 102.0, "l2_mb": 0.0, "vram_gb": 8.0},
    "nvidia/jetson-orin-nx": {"sm_count": 8, "peak_fp32": 2.402304, "mem_bandwidth": 102.4, "l2_mb": 0.0, "vram_gb": 16.0},
    "nvidia/jetson-agx-orin": {"sm_count": 16, "peak_fp32": 5.325, "mem_bandwidth": 204.8, "l2_mb": 0.0, "vram_gb": 64.0},
}

EXTRA_V5_PERF_NOTE: Dict[str, str] = {
    # CPU
    "intel/xeon-platinum-8480c": (
        "official: Intel ARK (cores/base/cache/channels/speed) + Intel cache hierarchy doc (4th Gen: 2MB L2/core); "
        "mem_bw=8*38.4(DDR5-4800), L2_total=56*2=112MB; "
        "src_ark=https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-platinum-8480c-processor-105m-cache-2-00-ghz/specifications.html; "
        "src_cache=https://www.intel.com/content/www/us/en/support/articles/000093753/processors.html"
    ),
    "intel/xeon-platinum-8380": (
        "official: Intel ARK (cores/base/cache/channels/speed) + Intel cache hierarchy doc (3rd Gen: 1.25MB L2/core); "
        "mem_bw=8*25.6(DDR4-3200), L2_total=40*1.25=50MB; "
        "src_ark=https://www.intel.com/content/www/us/en/products/sku/212285/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz/specifications.html; "
        "src_cache=https://www.intel.com/content/www/us/en/support/articles/000093753/processors.html"
    ),
    "amd/epyc-9654": (
        "official: AMD EPYC 9004 series datasheet (9654 row includes cores/base/L3/mem_bw) + AMD 58011 tuning guide (Zen4 L2 up to 1MB/core); "
        "L2_total=96*1=96MB; "
        "src_ds=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/data-sheets/amd-epyc-9004-series-processors-data-sheet.pdf; "
        "src_l2=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-9004-tg-58011.pdf"
    ),
    "amd/epyc-7763": (
        "official: AMD EPYC 7003 press release SKU table (cores/base/L3/DDR4-3200, 8ch) + AMD EPYC 7003 microarchitecture overview (Zen3 L2=512KB/core); "
        "mem_bw=8*25.6(DDR4-3200), L2_total=64*0.5=32MB; "
        "src_sku=https://www.amd.com/en/newsroom/press-releases/2021-03-15-amd-launches-3rd-gen-amd-epyc-processors.html; "
        "src_l2=https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-7003-series-microarchitecture-overview.pdf"
    ),
    "aws/graviton3": (
        "official: AWS Graviton technical guide (64C, 2.6GHz, L2=1MB/core, SLC/LLC=32MB, DRAM=8x DDR5, SIMD includes 2x256b SVE); "
        "official(Arm): Graviton3E spec table lists memory channels = 8 x DDR5-4800; "
        "derived: mem_bw=8*38.4(DDR5-4800) => 307.2GB/s; "
        "src_aws=https://aws.github.io/graviton/ ; "
        "src_arm=https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/leading-hpc-performance-with-graviton4"
    ),
    "ampere/altra-max": (
        "official: Ampere Altra Max product brief (128C@3.0GHz, L2=1MB/core, SLC=16MB, 8x DDR4-3200); "
        "mem_bw=8*25.6=204.8GB/s, L2_total=128*1=128MB; "
        "src=https://amperecomputing.com/assets/documents/Ampere_Altra_Max_Product_Brief.pdf"
    ),
    # GPU / Jetson
    "nvidia/nvidia-h100-sxm": (
        "official: NVIDIA H100 product page (mem/throughput) + NVIDIA Hopper architecture docs (SM/L2); "
        "src=https://www.nvidia.com/en-us/data-center/h100/ ; "
        "src_arch=https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
    ),
    "nvidia/nvidia-a40": (
        "official: NVIDIA RTX A40 datasheet / product brief; "
        "src_ds=https://resources.nvidia.com/en-us-brief/rtx-a40-datasheet"
    ),
    "nvidia/nvidia-a10": (
        "official: NVIDIA A10 product page (FP32/tensor/mem); "
        "src=https://www.nvidia.com/en-us/data-center/products/a10-gpu/ ; "
        "src_ds=https://resources.nvidia.com/en-us-brief/nvidia-a10-datasheet "
        "(TODO: confirm CUDA cores/SM count fields in datasheet)"
    ),
    "nvidia/nvidia-t4": (
        "official: NVIDIA T4 datasheet (Mar19): FP32=8.1 TFLOPS, mixed-precision=65 TFLOPS, memory=16GB GDDR6, mem_bw=300 GB/s; "
        "NOTE: NVIDIA product page lists mem_bw as 320+ GB/s; we use the datasheet number for consistency; "
        "src_ds=https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf ; "
        "src_page=https://www.nvidia.com/en-us/data-center/tesla-t4/"
    ),
    "nvidia/geforce-gtx-1080-ti": (
        "official: NVIDIA GeForce 10-series specs page; L2 not found => 0; "
        "src=https://www.nvidia.com/en-us/geforce/10-series/10-series-specs/"
    ),
    "nvidia/geforce-gtx-980-ti": (
        "official: NVIDIA GeForce 900-series specs page; L2 not found => 0; "
        "src=https://www.nvidia.com/en-us/geforce/900-series/900-series-specs/"
    ),
    "nvidia/jetson-orin-nano": (
        "official: NVIDIA Jetson Orin technical specifications table (Orin Nano 8GB (Super): 1024 CUDA cores, GPU max 1020MHz, mem_bw=102GB/s); "
        "=> peak_fp32 ~= 1024*2*1.020GHz/1000=2.08896; "
        "src=https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/"
    ),
    "nvidia/jetson-orin-nx": (
        "official: NVIDIA Jetson Orin technical specifications table (Orin NX 16GB (Super): 1024 CUDA cores, GPU max 1173MHz, mem_bw=102.4GB/s); "
        "=> peak_fp32 ~= 1024*2*1.173GHz/1000=2.402304; "
        "src=https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/"
    ),
    "nvidia/jetson-agx-orin": (
        "official: NVIDIA Jetson AGX Orin technical brief (CUDA cores, FP32 TFLOPS, mem bw, mem size); "
        "src=https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx-orin/Jetson_AGX_Orin_Technical_Brief.pdf"
    ),
}


# -----------------------------
# 4) v5 builder
# -----------------------------
@dataclass
class V5Record:
    name: str
    vector: List[float]
    notes: Dict[str, str]

def estimate_cpu_peak_fp32_tflops(
    *,
    num_cores: int,
    vector_unit_bytes: int,
    base_freq_ghz: float,
    fma_units_per_core: int,
) -> float:
    """粗略估计 CPU FP32 峰值（TFLOPS）。

    公式（TFLOPS）：
      cores * base_freq_GHz * (vector_bytes/4) * 2(FMA) * fma_units / 1000

    注意：
    - 这是“可解释的上限”估计：假设持续满吞吐发射向量 FMA；
    - 不建模 SMT、AVX 降频、前端/内存瓶颈等。
    """
    if num_cores <= 0 or vector_unit_bytes <= 0 or base_freq_ghz <= 0 or fma_units_per_core <= 0:
        return 0.0
    lanes_fp32 = float(vector_unit_bytes) / 4.0
    flops_per_cycle = lanes_fp32 * 2.0 * float(fma_units_per_core)
    return float(num_cores) * float(base_freq_ghz) * flops_per_cycle / 1000.0


def build_v5_for_one(name: str, v4_vec: Optional[List[float]]) -> V5Record:
    """
    核心转换逻辑：
    - 能从 v4 反推的字段，尽量从 v4 来（保证与你们现有库一致）
    - v5 新加字段靠 manual table，缺失就 0 并写 NOTE
    """
    notes: Dict[str, str] = {}
    n_low = name.lower()
    extra_perf = EXTRA_V5_PERF.get(n_low)

    # ---- A: identity ----
    is_gpu_hpc, is_gpu_edge, is_cpu_x86, is_cpu_arm = infer_identity(name)

    # ---- B: constraints / ABI-ish ----
    # CPU 侧：num_cores 直接用 v4 的 sm_count 槽（对 CPU v4 定义就是 cores）
    # GPU 侧：v4 的 sm_count 是 SM，不是 cores，所以这里置 0；Jetson/RPi 用 SOC_CPU_CORES/heuristic
    if v4_vec is not None:
        v4_sm_or_cores = inv_log10(v4_vec[12])
    else:
        v4_sm_or_cores = float((extra_perf or {}).get("sm_count") or (extra_perf or {}).get("num_cores") or 0.0)
        notes["v4_sm_or_cores"] = "from v5 EXTRA_V5_PERF (v4 missing)"

    if is_cpu_x86 or (is_cpu_arm and not n_low.startswith("nvidia/")):
        num_cores = int(round(v4_sm_or_cores))
        if v4_vec is not None:
            notes["num_cores"] = "from v4 (CPU uses sm_count slot as cores)"
        else:
            notes["num_cores"] = "from v5 EXTRA_V5_PERF (CPU cores)"
    elif n_low in SOC_CPU_CORES:
        num_cores = int(SOC_CPU_CORES[n_low][0])
        notes["num_cores"] = SOC_CPU_CORES[n_low][1]
    else:
        num_cores = 0
        notes["num_cores"] = "N/A(GPU-only) => 0"

    # vector_unit_bytes / cache_line_bytes
    if n_low in VECTOR_UNIT_BYTES:
        vector_unit_bytes = int(VECTOR_UNIT_BYTES[n_low][0])
        notes["vector_unit_bytes"] = VECTOR_UNIT_BYTES[n_low][1]
    else:
        vector_unit_bytes = 0
        if is_cpu_x86 or is_cpu_arm:
            notes["vector_unit_bytes"] = "unknown CPU SIMD width => 0 (TODO)"
        else:
            notes["vector_unit_bytes"] = "N/A(GPU) => 0"

    cache_line_bytes = CACHE_LINE_BYTES_DEFAULT if (is_cpu_x86 or is_cpu_arm) else 0
    if is_cpu_x86 or is_cpu_arm:
        notes["cache_line_bytes"] = f"default {CACHE_LINE_BYTES_DEFAULT}B (common CPU)"
    else:
        notes["cache_line_bytes"] = "GPU-only => 0"

    # GPU constraints from v4 B segment:
    # v4[8]=log2(max_threads_per_block), v4[9]=log2(warp), v4[10]=log2(shared_mem_kb), v4[11]=log2(regs)
    if is_gpu_hpc or is_gpu_edge:
        if v4_vec is not None:
            # NOTE: v4 对 CPU-only/N/A 往往用 0 或负数占位；这里做 N/A=0 语义保护。
            max_threads_per_block = int(round(inv_log2_na0(v4_vec[8])))
            warp_size = int(round(inv_log2_na0(v4_vec[9])))
            shared_mem_kb = float(inv_log2_na0(v4_vec[10]))
            max_shared_memory_per_block = int(round(shared_mem_kb * 1024.0))
            registers_per_block = int(round(inv_log2_na0(v4_vec[11])))
            notes["gpu_constraints"] = "from v4 B-seg inverse (log2->raw, with N/A<=0 => 0)"
        else:
            max_threads_per_block = 1024
            warp_size = 32
            max_shared_memory_per_block = 49152
            registers_per_block = 65536
            notes["gpu_constraints"] = "default CUDA constraints (v5 extra)"
    else:
        max_threads_per_block = 0
        warp_size = 0
        max_shared_memory_per_block = 0
        registers_per_block = 0
        notes["gpu_constraints"] = "CPU-only => 0"

    # ---- C: scale / perf ----
    # GPU SM count:
    if is_gpu_hpc or is_gpu_edge:
        sm_count = int(round(v4_sm_or_cores))
        if v4_vec is not None:
            notes["sm_count"] = "from v4 (GPU sm_count)"
        else:
            notes["sm_count"] = "from v5 EXTRA_V5_PERF (v4 missing)"
    else:
        sm_count = 0
        notes["sm_count"] = "CPU-only => 0"

    if v4_vec is not None:
        peak_fp32_v4 = inv_log10(v4_vec[13])      # TFLOPS (from v4 PERF_DB)
        mem_bandwidth = inv_log10(v4_vec[14])     # GB/s
        v4_l2_mb = inv_log10(v4_vec[15])          # GPU: L2; CPU: L2(total) in v4 PERF_DB
        v4_mem_gb = inv_log10(v4_vec[16])
    else:
        peak_fp32_v4 = float((extra_perf or {}).get("peak_fp32") or 0.0)
        mem_bandwidth = float((extra_perf or {}).get("mem_bandwidth") or 0.0)
        v4_l2_mb = float((extra_perf or {}).get("l2_mb") or 0.0)
        v4_mem_gb = float((extra_perf or {}).get("vram_gb") or 0.0)
        notes["perf_source"] = EXTRA_V5_PERF_NOTE.get(n_low, "v5 extra perf (no v4)")

    # ---- D: env ----
    mem_is_uma = 1.0 if ("jetson" in n_low or n_low.startswith("raspberry-pi/")) else 0.0
    notes["mem_is_uma"] = "heuristic: Jetson/RPi => 1 else 0"

    # peak_fp32:
    # - GPU/Jetson：沿用 v4 PERF_DB（更贴近你们已有数据口径）
    # - CPU-only：优先用 base frequency 公式推导（可复现，避免 env 指纹化）
    peak_fp32 = peak_fp32_v4
    cpu_like = bool(is_cpu_x86 or (is_cpu_arm and not n_low.startswith("nvidia/")))
    if cpu_like:
        # Hybrid topology takes priority (P/E base freq differ).
        if n_low in CPU_HYBRID_TOPOLOGY and vector_unit_bytes > 0:
            p_cores, p_base, e_cores, e_base, note = CPU_HYBRID_TOPOLOGY[n_low]
            fma_units = DEFAULT_X86_FMA_UNITS_PER_CORE if is_cpu_x86 else DEFAULT_ARM_FMA_UNITS_PER_CORE
            peak_fp32 = (
                estimate_cpu_peak_fp32_tflops(
                    num_cores=p_cores,
                    vector_unit_bytes=vector_unit_bytes,
                    base_freq_ghz=p_base,
                    fma_units_per_core=fma_units,
                )
                + estimate_cpu_peak_fp32_tflops(
                    num_cores=e_cores,
                    vector_unit_bytes=vector_unit_bytes,
                    base_freq_ghz=e_base,
                    fma_units_per_core=fma_units,
                )
            )
            notes["peak_fp32"] = "CPU derived from base freq formula (hybrid P/E; see cpu_* notes)"
            notes["cpu_base_freq_ghz"] = f"P:{p_base}GHz E:{e_base}GHz"
            notes["cpu_peak_fp32_model"] = note
            notes["cpu_peak_fp32_assumption"] = (
                f"fma_units_per_core={fma_units}, lanes_fp32=vector_bytes/4; "
                "does not model SMT/AVX downclock"
            )
        else:
            base_entry = CPU_BASE_FREQ_GHZ.get(n_low)
            if base_entry is not None and num_cores > 0 and vector_unit_bytes > 0:
                base_freq_ghz = float(base_entry[0])
                if base_freq_ghz > 0:
                    fma_units = DEFAULT_X86_FMA_UNITS_PER_CORE if is_cpu_x86 else DEFAULT_ARM_FMA_UNITS_PER_CORE
                    peak_fp32 = estimate_cpu_peak_fp32_tflops(
                        num_cores=num_cores,
                        vector_unit_bytes=vector_unit_bytes,
                        base_freq_ghz=base_freq_ghz,
                        fma_units_per_core=fma_units,
                    )
                    notes["peak_fp32"] = "CPU derived from base freq formula (see cpu_* notes)"
                    notes["cpu_base_freq_ghz"] = f"{base_freq_ghz} GHz ({base_entry[1]})"
                    notes["cpu_peak_fp32_assumption"] = (
                        f"fma_units_per_core={fma_units}, lanes_fp32=vector_bytes/4; "
                        "does not model SMT/AVX downclock"
                    )
                else:
                    notes["peak_fp32"] = "CPU base_freq_ghz <= 0; fallback to v4 PERF_DB"
                    peak_fp32 = peak_fp32_v4
            else:
                notes["peak_fp32"] = "CPU base_freq_ghz missing or missing num_cores/vector_unit_bytes; fallback to v4 PERF_DB"
                peak_fp32 = peak_fp32_v4
    else:
        if v4_vec is not None:
            notes["peak_fp32"] = "from v4 PERF_DB (legacy v4 generator)"
        else:
            notes["peak_fp32"] = f"from v5 EXTRA_V5_PERF ({EXTRA_V5_PERF_NOTE.get(n_low, 'see perf_source')})"

    if v4_vec is not None:
        notes["mem_bandwidth"] = "from v4 PERF_DB (legacy v4 generator)"
    else:
        notes["mem_bandwidth"] = f"from v5 EXTRA_V5_PERF ({EXTRA_V5_PERF_NOTE.get(n_low, 'see perf_source')})"

    # device_mem_gb policy (v5):
    # - GPU: VRAM (from v4 PERF_DB)
    # - UMA/SoC (Jetson/RPi): shared memory capacity (from v4 PERF_DB),作为“可行域约束”
    # - CPU-only non-UMA: 不建模整机 RAM，置 0，避免把环境配置指纹化进 embedding
    if is_gpu_hpc:
        device_mem_gb = float(v4_mem_gb)
        notes["device_mem_gb"] = "GPU VRAM from v4 PERF_DB" if v4_vec is not None else "GPU VRAM from v5 EXTRA_V5_PERF"
    elif mem_is_uma > 0:
        device_mem_gb = float(v4_mem_gb)
        notes["device_mem_gb"] = (
            "UMA/SoC shared mem from v4 PERF_DB (kept as feasibility constraint)"
            if v4_vec is not None
            else "UMA/SoC shared mem from v5 EXTRA_V5_PERF (kept as feasibility constraint)"
        )
    else:
        device_mem_gb = 0.0
        notes["device_mem_gb"] = "CPU-only non-UMA => 0 (do not embed host RAM to avoid fingerprinting)"

    # llc_mb: GPU 用 v4 L2；CPU 用 CPU_L3_MB override（否则 0）
    if is_gpu_hpc or is_gpu_edge:
        llc_mb = float(v4_l2_mb)
        if llc_mb <= 0.0:
            notes["llc_mb"] = "GPU L2 not available from official source => 0"
        else:
            if v4_vec is not None:
                notes["llc_mb"] = "GPU: from v4 l2_cache_mb (treated as LLC role)"
            else:
                notes["llc_mb"] = "GPU: from v5 EXTRA_V5_PERF l2_mb (treated as LLC role)"
    else:
        if n_low in CPU_L3_MB:
            llc_mb = float(CPU_L3_MB[n_low][0])
            notes["llc_mb"] = CPU_L3_MB[n_low][1]
        else:
            llc_mb = 0.0
            notes["llc_mb"] = "CPU L3 unknown => 0 (TODO)"

    # mid_cache_kb: CPU 用 v4 L2(total) -> KB；GPU N/A=0；Jetson/RPi 可后续补
    if is_cpu_x86 or (is_cpu_arm and not (is_gpu_hpc or is_gpu_edge)):
        mid_cache_kb = float(v4_l2_mb * 1024.0)
        if v4_vec is not None:
            notes["mid_cache_kb"] = "CPU: from v4 l2_cache_mb (treated as mid-level cache) *1024"
        else:
            notes["mid_cache_kb"] = "CPU: from v5 EXTRA_V5_PERF l2_mb (total) *1024"
    else:
        mid_cache_kb = 0.0
        notes["mid_cache_kb"] = "N/A(GPU/SoC) => 0"

    # peak_matmul + lowp_level from manual tables
    if n_low in PEAK_MATMUL_TFLOPS:
        peak_matmul = float(PEAK_MATMUL_TFLOPS[n_low][0])
        notes["peak_matmul"] = PEAK_MATMUL_TFLOPS[n_low][1]
    else:
        peak_matmul = 0.0
        notes["peak_matmul"] = "missing => 0 (TODO)"

    if n_low in LOWP_LEVEL:
        lowp_level = int(LOWP_LEVEL[n_low][0])
        notes["lowp_level"] = LOWP_LEVEL[n_low][1]
    else:
        lowp_level = 0
        notes["lowp_level"] = "missing => 0 (TODO)"

    # derived ratios
    if peak_matmul > 0 and peak_fp32 > 0:
        matmul_accel_ratio = safe_log10(peak_matmul) - safe_log10(peak_fp32)
        notes["matmul_accel_ratio"] = "derived: log10(peak_matmul) - log10(peak_fp32)"
    else:
        matmul_accel_ratio = 0.0
        notes["matmul_accel_ratio"] = "peak_matmul<=0 or peak_fp32<=0 => ratio=0"

    if peak_fp32 > 0 and mem_bandwidth > 0:
        compute_bw_ratio = safe_log10(peak_fp32) - safe_log10(mem_bandwidth)
        notes["compute_bw_ratio"] = "derived: log10(peak_fp32) - log10(mem_bandwidth)"
    else:
        compute_bw_ratio = 0.0
        notes["compute_bw_ratio"] = "peak_fp32<=0 or mem_bandwidth<=0 => 0"

    if llc_mb > 0 and mem_bandwidth > 0:
        cache_bw_ratio = safe_log10(llc_mb) - safe_log10(mem_bandwidth)
        notes["cache_bw_ratio"] = "derived: log10(llc_mb) - log10(mem_bandwidth)"
    else:
        cache_bw_ratio = 0.0
        notes["cache_bw_ratio"] = "llc_mb<=0 or mem_bandwidth<=0 => 0"

    if (is_gpu_hpc or is_gpu_edge) and sm_count > 0 and mem_bandwidth > 0:
        bw_per_sm = safe_log10(mem_bandwidth) - safe_log10(sm_count)
        notes["bw_per_sm"] = "derived: log10(mem_bandwidth) - log10(sm_count) (GPU-only)"
    else:
        bw_per_sm = 0.0
        notes["bw_per_sm"] = "GPU-only; sm_count<=0 or mem_bandwidth<=0 => 0"

    # ---- pack vector (24) ----
    vec = [0.0] * DIM_V5

    # A
    vec[IDX["is_gpu_hpc"]] = float(is_gpu_hpc)
    vec[IDX["is_gpu_edge"]] = float(is_gpu_edge)
    vec[IDX["is_cpu_x86"]] = float(is_cpu_x86)
    vec[IDX["is_cpu_arm"]] = float(is_cpu_arm)

    # B
    vec[IDX["num_cores"]] = float(num_cores)
    vec[IDX["vector_unit_bytes"]] = float(vector_unit_bytes)
    vec[IDX["cache_line_bytes"]] = float(cache_line_bytes)
    vec[IDX["max_shared_memory_per_block"]] = float(max_shared_memory_per_block)
    vec[IDX["max_threads_per_block"]] = float(max_threads_per_block)
    vec[IDX["registers_per_block"]] = float(registers_per_block)
    vec[IDX["warp_size"]] = float(warp_size)

    # C
    vec[IDX["sm_count"]] = float(sm_count)
    vec[IDX["peak_fp32"]] = float(peak_fp32)
    vec[IDX["peak_matmul"]] = float(peak_matmul)
    vec[IDX["mem_bandwidth"]] = float(mem_bandwidth)
    vec[IDX["llc_mb"]] = float(llc_mb)
    vec[IDX["mid_cache_kb"]] = float(mid_cache_kb)
    vec[IDX["device_mem_gb"]] = float(device_mem_gb)
    vec[IDX["matmul_accel_ratio"]] = float(matmul_accel_ratio)
    vec[IDX["lowp_level"]] = float(lowp_level)
    vec[IDX["compute_bw_ratio"]] = float(compute_bw_ratio)
    vec[IDX["cache_bw_ratio"]] = float(cache_bw_ratio)
    vec[IDX["bw_per_sm"]] = float(bw_per_sm)

    # D
    vec[IDX["mem_is_uma"]] = float(mem_is_uma)

    # sanity
    if len(vec) != DIM_V5:
        raise AssertionError("bad dim")

    return V5Record(name=name, vector=vec, notes=notes)


def build_v5_all(v4_universe_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    names, mp = load_v4_universe(v4_universe_path)

    out_list: List[Dict[str, Any]] = []
    notes_map: Dict[str, Dict[str, str]] = {}

    seen = set()
    for name in names:
        if name.lower() in EXTRA_V5_PERF:
            rec = build_v5_for_one(name, None)
        else:
            rec = build_v5_for_one(name, mp[name])
        out_list.append({"hardware_name": rec.name, "vector": rec.vector})
        notes_map[rec.name] = rec.notes
        seen.add(name.lower())

    # Append v5-only extras (do NOT require v4-universe changes)
    for name in sorted(EXTRA_V5_PERF.keys()):
        if name.lower() in seen:
            continue
        rec = build_v5_for_one(name, None)
        out_list.append({"hardware_name": rec.name, "vector": rec.vector})
        notes_map[rec.name] = rec.notes

    meta = {
        "schema": {
            "version": "emb-v5",
            "dim": DIM_V5,
            "idx": IDX,
            "units": {
                "peak_fp32": "TFLOPS",
                "peak_matmul": "TFLOPS",
                "mem_bandwidth": "GB/s",
                "llc_mb": "MB",
                "mid_cache_kb": "KB",
                "device_mem_gb": "GB",
                "max_shared_memory_per_block": "bytes",
            },
            "cpu_peak_fp32_definition": (
                "CPU peak_fp32 is derived from base frequency (GHz): "
                "cores * base_freq_GHz * (vector_bytes/4) * 2(FMA) * fma_units / 1000. "
                "Turbo/all-core/AVX downclock are treated as environment factors and are not embedded."
            ),
            "peak_matmul_definition": (
                "dense Tensor TFLOPS (FP16/BF16/TF32). "
                "If only sparse marketing TFLOPS is available, approximate dense by /2."
            ),
            "notes": [
                "GPU specs mostly come from v4 PERF_DB via inverse(log10).",
                "If a hardware is listed in EXTRA_V5_PERF, v5 will override v4-universe and use v5-only raw specs.",
                "CPU peak_fp32 uses base frequency tables + a simple throughput formula (see cpu_* notes per hardware).",
                "CPU L3 and GPU peak_matmul are filled by manual tables (APPROX/TODO).",
                "CUDA constraints inverse(log2) preserves N/A semantics: log2<=0 => raw 0.",
                "device_mem_gb is set to 0 for CPU-only non-UMA (avoid fingerprinting host RAM); UMA/SoC keeps shared mem, GPU keeps VRAM.",
                "You can replace heuristics with canonical target parsing if you prefer.",
            ],
        },
        "notes_per_hardware": notes_map,
    }

    return out_list, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-universe", required=True, help="Path to hardware_embeddings_v4_universe.json (list format)")
    ap.add_argument("--out", required=True, help="Output JSON path for emb-v5 embeddings")
    ap.add_argument("--out-meta", default="", help="Optional meta JSON path (schema + notes)")
    args = ap.parse_args()

    out_list, meta = build_v5_all(args.v4_universe)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)

    if args.out_meta:
        with open(args.out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {len(out_list)} embeddings to: {args.out}")
    if args.out_meta:
        print(f"[OK] wrote meta to: {args.out_meta}")


if __name__ == "__main__":
    main()
