用一套统一的语义来描述不同类型的硬件
统一硬件特征向量 V2.0 规则回顾
首先，我们回顾一下最终确定的向量结构。这个结构旨在用一套统一的语义来描述不同类型的硬件。

语义部分	特征名称	类型	备注
1. 平台与类别	
kind	One-Hot	[is_llvm, is_cuda, ...]
keys	Multi-Hot	[has_cpu, has_gpu, ...]
2. 微架构	
mcpu / arch	One-Hot	CPU微架构或GPU SM架构
mattr	Multi-Hot	SIMD等指令集属性
3. 计算资源
compute_unit_count	数值	CPU核心数 或 GPU SM数
max_threads_per_block	数值	CPU填超线程数
thread_warp_size	数值	GPU填32，CPU填1
peak_fp32_flops_g	数值	GFLOPS单位的FP32算力
4. 内存与缓存
max_shared_memory_per_block	数值	KB单位, CPU填0
registers_per_block	数值	CPU填0
l1_cache_size_kb	数值	L1数据缓存总大小
l2_cache_size_kb	数值	L2缓存总大小
l3_cache_size_kb	数值	L3缓存总大小, GPU填0
memory_bandwidth_gb_s	数值	GB/s单位

NVIDIA V100 (数据中心GPU) 的初步 Embedding
---

NVIDIA V100的初步 Embedding
- 原始数据来源: TVM tag.cc 文件 () 和 NVIDIA官方技术规格。
{
  "hardware_name": "NVIDIA V100",
  "vector": {
    "kind": [0, 1, 0],              // [llvm, cuda, metal]
    "keys": [0, 1],                 // [cpu, gpu]
    "arch": [0, 0, 1, 0, 0],        // [icelake-server, cortex-a72, sm_70, sm_72, sm_86]
    "mattr": [0, 0, 0],             // [avx2, avx512, neon]
    "compute_unit_count": 80,
    "max_threads_per_block": 1024,
    "thread_warp_size": 32,
    "peak_fp32_flops_g": 15700,
    "max_shared_memory_per_block": 48,
    "registers_per_block": 65536,
    "l1_cache_size_kb": 10240,
    "l2_cache_size_kb": 6144,
    "l3_cache_size_kb": 0,
    "memory_bandwidth_gb_s": 900
  }
}
特征解释:
- kind: [0, 1, 0] -> 这是一个 CUDA 设备 ()。
- keys: [0, 1] -> 它的通用标签是 GPU ()。
- arch: [..., 1, 0, 0] -> 其计算架构是 sm_70 
- mattr: [0, 0, 0] -> 不适用CPU的SIMD指令集。
- compute_unit_count: 80 -> V100拥有 80个SM (流式多处理器)。
- max_threads_per_block: 1024 -> 每个线程块最多支持1024个线程 ()。
- thread_warp_size: 32 -> 使用NVIDIA标准的32线程Warp ()。
- peak_fp32_flops_g: 15700 -> V100的FP32（单精度）理论峰值算力约为 15.7 TFLOPS。
- max_shared_memory_per_block: 48 -> 每个SM最大可配置 48KB 的共享内存 (源数据为49152字节) 
- registers_per_block: 65536 -> 每个线程块可用的寄存器总数 ()。
- l1_cache_size_kb: 10240 -> 每个SM有128KB的L1数据缓存/共享内存，总计 80 * 128 = 10240 KB。
- l2_cache_size_kb: 6144 -> 拥有 6MB 的L2缓存。
- l3_cache_size_kb: 0 -> GPU通常没有L3缓存。
- memory_bandwidth_gb_s: 900 -> 采用HBM2内存，理论峰值带宽约为 900 GB/s。

Intel Xeon Platinum 8351N (服务器CPU) 的初步 Embedding
- 原始数据来源:  lscpu 输出和 Intel 官方技术规格。
{
  "hardware_name": "Intel Xeon Platinum 8351N",
  "vector": {
    "kind": [1, 0, 0],              // [llvm, cuda, metal]"keys": [1, 0],                 // [cpu, gpu]"arch": [1, 0, 0, 0, 0],        // [icelake-server, cortex-a72, sm_70, sm_72, sm_86]"mattr": [1, 1, 0],             // [avx2, avx512, neon]"compute_unit_count": 36,
    "max_threads_per_block": 2,
    "thread_warp_size": 1,
    "peak_fp32_flops_g": 2764,
    "max_shared_memory_per_block": 0,
    "registers_per_block": 0,
    "l1_cache_size_kb": 1728,
    "l2_cache_size_kb": 46080,
    "l3_cache_size_kb": 55296,
    "memory_bandwidth_gb_s": 204.8
  }
}
特征解释:
- kind: [1, 0, 0] -> 这是一个使用 LLVM 后端的CPU设备。
- keys: [1, 0] -> 它的通用标签是 CPU。
- arch: [1, 0, ...] -> 型号为Platinum 8351N，属于 Ice Lake-SP 微架构。在TVM中，这通常被标记为 icelake-server。
- mattr: [1, 1, 0] -> 同时支持 AVX2 和 AVX512 指令集。
- compute_unit_count: 36 -> lscpu 显示 cpu cores : 36，拥有 36个物理核心。
- max_threads_per_block: 2 -> flags 中有 ht (Hyper-Threading)，表示每个物理核心支持 2个逻辑线程。
- thread_warp_size: 1 -> CPU线程独立执行，没有Warp概念，因此设为 1。
- peak_fp32_flops_g: 2764 -> 估算值约为 2.76 TFLOPS。计算方式：36核 * 2.4GHz * 16 (AVX512 FP32宽度) * 2 (FMA指令) ≈ 2764.8 GFLOPS。
- max_shared_memory_per_block: 0 -> CPU没有类似GPU的片上共享内存，此值为 0。
- registers_per_block: 0 -> CPU不以这种方式暴露寄存器文件，此值为 0。
- l1_cache_size_kb: 1728 -> Ice Lake架构每个核心有48KB的L1数据缓存，总计 36 * 48 = 1728 KB。
- l2_cache_size_kb: 46080 -> 每个核心有1.25MB的L2缓存，总计 36 * 1.25 * 1024 = 46080 KB。
- l3_cache_size_kb: 55296 -> lscpu 显示 cache size : 55296 KB，这是共享的L3缓存大小。
- memory_bandwidth_gb_s: 204.8 -> 该CPU支持8通道DDR4-3200内存，理论峰值带宽为 8 * 3200 * 8 / 1024 / 1024 / 1024 * 1000 * 1000 * 1000 ≈ 204.8 GB/s

## 附录：聚类友好硬件Embedding（建议版）

本附录在不改变“统一语义”的前提下，给出更适合“聚类/可视化/相似度检索”的Embedding规则，重点解决：
- 降维“arch”超长one-hot（几十维过长）；
- 缩放“寄存器/缓存/带宽/算力”等大数值，避免淹没类别信号；
- 提升跨CPU/GPU/Metal的可比性与可解释性。

### 设计目标
- 用“属性刻画”替代“大one-hot”：`arch`不再使用全量one-hot，而是数值化（如CUDA计算能力或CPU代际索引）+ 小规模`vendor` one-hot；
- 用“对数缩放 + 单位化/比值化”处理跨度巨大的物理量（寄存器、缓存、带宽、算力），保证尺度一致；
- 控制总维度在约20–32，便于KMeans/层次聚类/UMAP等方法。

### 特征结构（建议）
- 类别特征（≤ 12 维）
  - kind one-hot: `[llvm, cuda, metal]`（3）
  - keys one-hot: `[cpu, gpu]`（2，设备级互斥）
  - vendor one-hot: `[nvidia, intel, amd, apple, arm]`（5）
  - ISA multi-hot: `[avx2, avx512, neon]`（3）
  - cpu_arch_family one-hot: `[x86, aarch64]`（2，仅当keys=cpu时生效，其余设备置0）
- 架构数值化（1 维）
  - `arch_cc_or_gen`: CUDA用计算能力（如`sm_86 → 0.86`），CPU/Metal用简洁的代际索引映射到[0,1]区间（如`skylake-avx512 → 0.60`, `cascadelake → 0.62`, `icelake-server → 0.65`, `apple-latest → 0.80`, `carmel → 0.55`）。
- 资源与性能（对数/比值化，建议 10–16 维）
  - log1p原始量：
    - `log_cu = log1p(compute_unit_count)`
    - `log_mtpb = log1p(max_threads_per_block)`
    - `log_warp = log1p(thread_warp_size)`
    - `log_flops_g = log1p(peak_fp32_flops_g)`
    - `log_bw_gbs = log1p(memory_bandwidth_gb_s)`
    - `log_shm_kb = log1p(max_shared_memory_per_block)`（单位KB）
  - 单位化/比值化（先单位化再log1p）：
    - `log_gflops_per_cu = log1p(peak_fp32_flops_g / compute_unit_count)`
    - `log_bw_per_cu = log1p(memory_bandwidth_gb_s / compute_unit_count)`
    - `log_regs_per_thread = log1p(registers_per_block / max_threads_per_block)`
    - `log_l1_per_cu = log1p(l1_cache_size_kb / compute_unit_count)`
    - `log_l2_per_cu = log1p(l2_cache_size_kb / compute_unit_count)`
    - `log_l3_per_cu = log1p(l3_cache_size_kb / compute_unit_count)`
    - `log_roofline_ai = log1p(peak_fp32_flops_g / max(memory_bandwidth_gb_s, 1e-6))`

说明：`log1p(x) = ln(1+x)`，用于压缩数量级，同时保留可比性。CPU/GPU/Metal的差异通过“每CU/每线程/比值”自然对齐。

### 建议的聚类前处理
- 对数值特征在聚类前做标准化（z-score）；
- 保存标准化统计（均值/方差）到`vector_info_v2.json`，确保复现；
- 类别one-hot与数值同样参与标准化，以免被大数值淹没。

### 示例A：NVIDIA V100（基于公开规格与tag.cc）
原始关键参数：`compute_unit_count=80, max_threads_per_block=1024, thread_warp_size=32, peak_fp32_flops_g=15700, max_shared_memory_per_block=48KB, registers_per_block=65536, l1=10240KB, l2=6144KB, l3=0KB, memory_bw=900GB/s, arch=sm_70`。

```json
{
  "hardware_name": "NVIDIA V100",
  "categorical": {
    "kind": [0, 1, 0],
    "keys": [0, 1],
    "vendor": [1, 0, 0, 0, 0],
    "cpu_arch_family": [0, 0],
    "isa": [0, 0, 0]
  },
  "arch_numeric": {
    "arch_cc_or_gen": 0.70
  },
  "numeric": {
    "log_cu": 4.39,
    "log_mtpb": 6.93,
    "log_warp": 3.50,
    "log_flops_g": 9.66,
    "log_bw_gbs": 6.80,
    "log_shm_kb": 3.89,
    "log_gflops_per_cu": 5.29,
    "log_bw_per_cu": 2.50,
    "log_regs_per_thread": 4.17,
    "log_l1_per_cu": 4.86,
    "log_l2_per_cu": 4.36,
    "log_l3_per_cu": 0.00,
    "log_roofline_ai": 2.91
  }
}
```

要点：不再用`arch`大one-hot；通过“每CU/每线程/比值 + log1p”让V100在“高性能GPU簇”中与A100/4090自然接近，但`roofline_ai`等特征可区分其算力/带宽比差异。

### 示例B：Intel Xeon Platinum 8351N（Ice Lake-SP）
原始关键参数（参考lscpu与公开规格）：`compute_unit_count=36, max_threads_per_block=2, thread_warp_size=1, peak_fp32_flops_g≈2764, max_shared_memory_per_block=0, registers_per_block=0, l1=1728KB, l2=46080KB, l3=55296KB, memory_bw≈204.8GB/s, mcpu=icelake-server`。

```json
{
  "hardware_name": "Intel Xeon Platinum 8351N",
  "categorical": {
    "kind": [1, 0, 0],
    "keys": [1, 0],
    "vendor": [0, 1, 0, 0, 0],
    "cpu_arch_family": [1, 0],
    "isa": [1, 1, 0]
  },
  "arch_numeric": {
    "arch_cc_or_gen": 0.65
  },
  "numeric": {
    "log_cu": 3.61,
    "log_mtpb": 1.10,
    "log_warp": 0.69,
    "log_flops_g": 7.92,
    "log_bw_gbs": 5.33,
    "log_shm_kb": 0.00,
    "log_gflops_per_cu": 4.36,
    "log_bw_per_cu": 1.90,
    "log_regs_per_thread": 0.00,
    "log_l1_per_cu": 3.89,
    "log_l2_per_cu": 7.15,
    "log_l3_per_cu": 7.34,
    "log_roofline_ai": 2.67
  }
}
```

要点：CPU不具备GPU样式的共享内存/寄存器暴露，改用缓存与带宽、每核算力等度量；与GPU在同一尺度下聚类，CPU样本会在“CPU簇”内部按缓存/带宽/算力比值分层。

---

实施建议：
- 生成Embedding时按本附录输出“扁平向量”与“解释字段”（方便检查）；
- 聚类脚本中完成标准化，并把统计写回`vector_info_v2.json`；
- 4090等未在`tag.cc`中定义的设备，按`vendor=nvidia`、`arch_cc_or_gen≈0.86`（当前以sm_86近似）与估算物理量填充，即可自然落入“高性能GPU簇”。

