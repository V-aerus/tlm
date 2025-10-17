# 硬件特征向量生成器 V2.0 总结（更新：设备级聚类版）

## 主要改进（相对旧版）

- **设备级互斥 keys**: `keys=[cpu,gpu]` 仅表示“设备类型”，避免把节点(host+device)信息混入设备embedding。
- **新增 vendor 与 cpu_arch_family**: `vendor=[nvidia,intel,amd,apple,arm]`；`cpu_arch_family=[x86,aarch64]`（仅CPU生效）。
- **移除 arch 大one-hot**: 使用单一数值`arch_cc_or_gen`表达“代际位置”。CUDA取`sm_XX→XX/100`，CPU取微架构代际映射（如`skylake-avx512→0.60`, `icelake-server→0.65`）。
- **数值特征改为“对数+单位化/比值化”派生（13维）**: 用`log1p`压缩量级；用“每CU/每线程”等派生量对齐CPU/GPU/Metal的可比性。
- **不内置Min-Max归一化**: 建议在聚类脚本中做z-score标准化（并保存均值/方差），生成器仅输出工程化后的数值（含`log1p/比值`）。

## 向量结构（29维）

- **kind**: 3维 - `[llvm, cuda, metal]`
- **keys**: 2维 - `[cpu, gpu]`（设备级互斥）
- **vendor**: 5维 - `[nvidia, intel, amd, apple, arm]`
- **cpu_arch_family**: 2维 - `[x86, aarch64]`（仅CPU生效，其他设备全0）
- **mattr**: 3维 - `[avx2, avx512, neon]`
- **arch_cc_or_gen**: 1维 - 架构代际数值（0~1）
- **numeric_derived**: 13维 - 对数与单位化/比值化派生特征（见下）

合计：3 + 2 + 5 + 2 + 3 + 1 + 13 = 29 维

## 13个派生数值特征（来源与理由）

说明：以下特征均为“先构造单位化/比值化”，再取`log1p(x)=ln(1+x)`，用于压缩数量级差异，提升聚类可比性与稳健性。

1) log_cu
- **来源**: `compute_unit_count`（GPU=SM数，CPU=核心数）
- **理由**: 基本并行度量纲；对数压缩避免“核数/SM数”过大对聚类的主导效应。

2) log_mtpb
- **来源**: `max_threads_per_block`（GPU=线程块上限；CPU≈逻辑线程/核心，通常=2）
- **理由**: 线程并行粒度；跨设备差异大，用对数稳定尺度。

3) log_warp
- **来源**: `thread_warp_size`（GPU=Warp大小32；CPU=1）
- **理由**: 硬件调度/执行粒度的差异指标；对数压缩使其与其他量纲可比。

4) log_flops_g
- **来源**: `peak_fp32_flops_g`（GFLOPS）
- **理由**: 计算峰值能力；量级极大（GPU~万GFLOPS），对数化避免吞噬其他特征。

5) log_bw_gbs
- **来源**: `memory_bandwidth_gb_s`（GB/s）
- **理由**: 内存系统吞吐能力；与算力搭配用于分析Roofline位置。

6) log_shm_kb
- **来源**: `max_shared_memory_per_block`（KB；CPU=0）
- **理由**: 片上共享内存规模（主要针对GPU）；对数化与其他缓存量统一尺度。

7) log_gflops_per_cu
- **来源**: `peak_fp32_flops_g / compute_unit_count`
- **理由**: 单CU的平均算力（每SM/每核）；剔除“数量优势”，对齐不同并行度设备的可比性。

8) log_bw_per_cu
- **来源**: `memory_bandwidth_gb_s / compute_unit_count`
- **理由**: 单CU可分摊的带宽供给；衡量带宽资源是否成为瓶颈（与7)配对）。

9) log_regs_per_thread
- **来源**: `registers_per_block / max_threads_per_block`
- **理由**: 每线程可用寄存器规模（主要针对GPU线程块）；比“总寄存器”更公平可比。

10) log_l1_per_cu
- **来源**: `l1_cache_size_kb / compute_unit_count`
- **理由**: 单CU的L1容量；缓存资源按CU归一化，避免“总量随并行度膨胀”。

11) log_l2_per_cu
- **来源**: `l2_cache_size_kb / compute_unit_count`
- **理由**: 单CU的L2容量；对比不同架构的中层缓存供给。

12) log_l3_per_cu
- **来源**: `l3_cache_size_kb / compute_unit_count`（GPU通常=0）
- **理由**: 单CU的LLC（CPU常见）；有助于区分CPU族内的缓存层级差异。

13) log_roofline_ai
- **来源**: `peak_fp32_flops_g / max(memory_bandwidth_gb_s, 1e-6)`（单位近似FLOP/Byte）
- **理由**: 近似算术强度（Roofline AI），反映“算力受限/带宽受限”的硬件倾向，聚类可解释性强。

## 其它字段说明

- **arch_cc_or_gen (0~1)**: 仅表示“代际位置”的单调刻度，不比较CPU与GPU的强弱。CUDA取`sm_86→0.86`；CPU取代际映射（如`skylake-avx512→0.60`,`icelake-server→0.65`）。可配置化维护映射表。
- **cpu_arch_family**: 仅CPU生效的二值one-hot `[x86,aarch64]`；GPU/Metal全0。用于区分CPU ISA家族，避免与`keys`语义混淆。
- **mattr**: 指令集能力Multi-Hot（如AVX2/AVX512/NEON），仅在适用设备上置1。
- **标准化建议**: 在聚类/可视化前对全部数值做z-score，并在`vector_info_v2.json`持久化均值/方差，保证复现与可解释。

## 文件输出

- `hardware_embeddings_v2.json` - 所有硬件的Embedding列表
- `vector_info_v2.json` - 维度分解、类别列表、数值特征名及统计信息（建议扩展）
- `hardware_embedding_generator_v2.py` - 生成器代码（设备级聚类版）

## 符合规范与适用性

- ✅ 与`MT_MoSLoRA_embedding.md`附录“聚类友好规则”一致
- ✅ 向量紧凑（29维），可解释性强，便于KMeans/层次聚类/UMAP
- ✅ 跨CPU/GPU/Metal可比：通过“每CU/每线程/比值+对数”实现
- ✅ 新硬件友好：4090等缺失TVM定义也能通过近似CC与估算物理量自然落位
