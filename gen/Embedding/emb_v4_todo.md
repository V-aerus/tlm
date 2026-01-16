System / Prompt:

请编写一个名为 hardware_embedding_generator_v4.py 的 Python 脚本，用于生成“物理感知（Physics-Aware）”的 24 维硬件特征向量。
这个脚本将取代 v2 / v3 版本，核心目标是：

能明显区分 同一架构不同规模的 GPU（例如 RTX 4090 vs RTX 3060 vs 3090）。

保留一部分 TVM target 的约束信息（threads / shared mem / regs）。

同时引入 真实物理规格（SM 数、TFLOPS、带宽、L2、显存等）。

该脚本会被 EdgeTLM 项目的数据管线调用，因此：

接口尽量保持和 v3 类似：generate(name: str, tvm_config: dict) -> List[float]。

内部可以使用一个 手工维护的小型 PERF_DB 来补充 TVM target 中没有的物理参数。

1. 数据结构定义

定义一个 @dataclass：

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class HardwarePerfSpec:
    sm_count: int          # 计算单元数量（GPU: SM 数；CPU: 核心数）
    peak_flops_32: float   # FP32 峰值算力 (TFLOPS)
    mem_bandwidth: float   # 内存/显存带宽 (GB/s)
    l2_cache_mb: float     # L2 缓存大小 (MB)
    vram_gb: float         # 显存/内存容量 (GB)
    tensor_core_gen: int   # Tensor Core 代数 (0=None, 1=Volta, 2=Turing, 3=Ampere, 4=Ada/Hopper)


定义一个全局字典 PERF_DB: Dict[str, HardwarePerfSpec]，键是 硬件名字（和我们 TVM tag / hw_name 一致），例如：

高端 / HPC GPU：

"nvidia/nvidia-a100": (108 SM, 19.5 TFLOPS, 1555 GB/s, 40 MB L2, 80 GB, Gen 3)

"nvidia/nvidia-v100": (80 SM, 15.7 TFLOPS, 900 GB/s, 6 MB L2, 32 GB, Gen 1)

"nvidia/rtx-4090": (128 SM, 82.6 TFLOPS, 1008 GB/s, 72 MB L2, 24 GB, Gen 4)

"nvidia/geforce-rtx-3090": (82 SM, 35.6 TFLOPS, 936 GB/s, 6 MB L2, 24 GB, Gen 3)

中端 GPU：

"nvidia/geforce-rtx-3060": (28 SM, 12.7 TFLOPS, 360 GB/s, 3 MB L2, 12 GB, Gen 3)

（注意：不需要 T4，可不要写 nvidia/nvidia-t4。如果想预留，可以用注释标出来。）

边缘/嵌入式 GPU：

"nvidia/jetson-agx-xavier": (8 SM, 1.4 TFLOPS, 137 GB/s, 0.5 MB L2, 32 GB, Gen 1)

"nvidia/jetson-orin"（或我们将来实际使用的名称）: (16 SM, 5.3 TFLOPS, 204 GB/s, 3 MB L2, 64 GB, Gen 3)

CPU：

"aws/cpu/c5.18xlarge": (36 cores, 2.3 TFLOPS, 120 GB/s, 36 MB L2, 128 GB, Gen 0)

如果方便，可以预留一个 "raspberry-pi/4b-aarch64" 的条目，参数可以先粗略估计并写 TODO 注释。

数值可以按 NVIDIA / AWS 官方规格大致填写，不需要 100% 精确；如果不确定，请在注释里写 # TODO: check spec。

2. 生成器类逻辑（EmbeddingV4Generator）

编写一个 EmbeddingV4Generator 类：

class EmbeddingV4Generator:
    def __init__(self):
        self.perf_db: Dict[str, HardwarePerfSpec] = PERF_DB


实现两个辅助函数：

import math

def _log2(x: float, eps: float = 1e-6) -> float:
    x = max(x, eps)
    return math.log2(x)

def _log10(x: float, eps: float = 1e-6) -> float:
    x = max(x, eps)
    return math.log10(x)


实现 generate(self, name: str, tvm_config: Dict[str, Any]) -> List[float]：

返回一个 长度固定为 24 的 List[float]。

tvm_config 的结构和 v3 一致：来自我们 TVM target JSON，包含字段：

"kind": "cuda" 或 "llvm"

"arch": "sm_70" / "sm_72" / "sm_86" 等（CUDA 才有）

"mtriple": "aarch64-linux-gnu"、"x86_64-linux-gnu" 等

"mcpu": "skylake-avx512"、"carmel" 等

"mattr": 如有则为字符串列表，例如 ["+neon"]

"max_threads_per_block", "thread_warp_size", "max_shared_memory_per_block", "registers_per_block" 等

允许 tvm_config 缺少某些字段，此时用合理的默认值（例如 max_threads_per_block=1024，thread_warp_size=32，其它用 1 或很小的正数）。

3. 特征维度设计（共 24 维）

向量维度的语义固定如下（按照下标）：

A. Identity & Arch（0–7，离散/布尔特征）

从 tvm_config 和 PERF_DB 里提取：

[0] is_gpu_hpc

如果 kind == "cuda" 且 perf.peak_flops_32 > 10.0 → 1.0

否则 0.0

[1] is_gpu_edge

如果 kind == "cuda" 且 perf.peak_flops_32 <= 10.0 → 1.0

否则 0.0
（注意 3060 会走 HPC 路线，这是刻意设计：算力足够强，算训练用 GPU。）

[2] is_cpu_x86

如果 kind == "llvm" 且 mtriple 中不含 "aarch64" → 1.0

否则 0.0

[3] is_cpu_arm

如果 kind == "llvm" 且 ("aarch64" 在 mtriple 里，或 "neon" 出现在 mattr 里) → 1.0

否则 0.0

[4] has_tensor_core

如果 perf.tensor_core_gen > 0 → 1.0，否则 0.0。

[5] is_ampere_plus

如果 perf.tensor_core_gen >= 3 → 1.0，否则 0.0。

[6] has_avx512

如果 kind == "llvm" 且 "avx512" 出现在 mcpu 或 mattr 字符串里 → 1.0，否则 0.0。

[7] is_arm_neon

如果 kind == "llvm" 且 "neon" 出现在 mattr 或 mcpu 中 → 1.0，否则 0.0。

B. Constraints（8–11，TVM 线程/寄存器约束，用 log2）

从 tvm_config 读取，缺省值合理填充：

[8] log2(max_threads_per_block)

[9] log2(thread_warp_size)

[10] log2(max_shared_memory_per_block / 1024.0) # 单位转成 KB 再取 log2

[11] log2(registers_per_block)

若字段缺失，给默认值（如 1024, 32, 49152, 65536 等），确保不会报错。

C. Scale & Performance（12–19，物理规模 + 性能，用 log10）

从 perf = PERF_DB.get(name) 里取；如果找不到，使用一个 fallback（例如 sm_count=1, flops=1, 带宽=1, 等），并在代码里打一个 logger.warning 或 print 提醒。

[12] log10(sm_count)

[13] log10(peak_flops_32) # 单位仍然是 TFLOPS

[14] log10(mem_bandwidth) # GB/s

[15] log10(l2_cache_mb) # MB

[16] log10(vram_gb) # GB

`[17] tensor_core_gen / 5.0 # 归一化到 [0,1] 左右

`[18] 0.0 # 预留

`[19] 0.0 # 预留

D. Environment（20–23，环境/平台特征）

适度编码“统一内存/嵌入式环境”等：

[20] is_unified_memory

如果 name 包含 "jetson"、"raspberry-pi" 或 "apple" → 1.0，否则 0.0。

[21] is_cloud_instance

补充：v4 universe
----------------
为 routing preprocess 统计使用，不影响 KV/aligner。
请用 gen/scripts/inspect_hw_emb_universe.py 生成
gen/Embedding/hardware_embeddings_v4_universe.json，
其中会补齐更多硬件并输出每维 mean/std/near-constant 维度清单。

如果 name 包含 "aws/"、"gcp/" 或 "azure/" → 1.0，否则 0.0。

[22] reserved_env_1 = 0.0（预留）

[23] reserved_env_2 = 0.0（预留）

要求：最终向量长度必须始终为 24，即使某些字段缺失也要按上述规则填默认值。

4. 主程序 / 测试代码

在 if __name__ == "__main__": 中写一段简单测试：

从一个简单的 tvm_config 示例构造字典，比如：

V100 的 CUDA config（kind="cuda", arch="sm_70", max_threads_per_block=1024, ...）

4090 的 CUDA config（arch="sm_86"）

Xavier 的 CUDA + ARM host config。

实例化：

gen = EmbeddingV4Generator()
vec_v100   = gen.generate("nvidia/nvidia-v100", sample_tvm_cfg_v100)
vec_4090   = gen.generate("nvidia/rtx-4090", sample_tvm_cfg_4090)
vec_xavier = gen.generate("nvidia/jetson-agx-xavier", sample_tvm_cfg_xavier)


打印：

每个向量的 前 16 维；

len(vec_*)；

额外打印 [12:17] 这几维，用来肉眼检查：

4090 在 [12–16] 上明显大于 V100；

HPC GPU 明显大于 Xavier / Orin；

CPU 在 [12–16] 上形态不同（例如 sm_count 取核心数）。

5. 其他实现细节要求

使用标准库，不依赖外部包。

所有常数（阈值、默认值）在代码里用清晰名字或注释标明含义。

对缺失字段要有鲁棒处理，不允许抛异常。

在 PERF_DB 里对不完全确定的数字加 # TODO 注释，方便后续手工校正。
