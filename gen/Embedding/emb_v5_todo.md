v4:
总结 emb_v4以及 universe 版本的问题：1) 维度语义不正交：把不同“轴”的信息塞进同一段，导致结构不自洽 你现在把 embedding 分成“四段”，但段内混进了不
  同层级/不同维度的属性：比如在 identity/Ark 这段里同时放了“设备大类 one-hot（0~3）”和“has TensorCore 这类 GPU-specific 特征”，老师的核心意见是：这俩不是一个维度，放一起会显得“没道理/
  不顺眼”，而且 CPU 场景下某些位永远为 0，本质上也说明分段逻辑有问题。 一句话： 分段是可以的，但每段必须对应清晰且一致的语义轴；否则后面做相似度/路由就会被“结构噪声”污染。 2) 特征粒度
  过“指纹化”：过于具体到单卡，反而破坏训练目标（信息泄露） 文档里明确提到：有些位“太具体到单个硬件”，在训练 aligner 时你本来想遮掉信息让它补全，但这些位把信息直接暴露，导致训练目标被
  短路。 典型例子就是你提的 is AmperePlus：老师直说 “so what？设计不对”，因为它既不像“架构大类”，又不像“系统性可泛化特征”，还会带来强指纹。 3) 类别体系不完整/不一致：有“一个架构位”，
  却没有“体系化架构表示” 老师的质疑点是：你如果要放“架构代际”这种轴（Ampere/Volta/Hopper…），那就应该是一套一致的类别/编码体系；你现在只有一个 AmperePlus 会显得非常突兀——等价于“只给一
  个架构开了特殊通道”。 这会直接带来两个后果： 语义不统一（embedding 像是东拼西凑） 泛化不自然：新硬件落点很难解释（你到底想表达“代际”，还是想表达“某种性能分界”？） 4) 字段来源不清、
  可解释性不足：不知道“为什么要有这个位” 你老师反复追问“这些东西你是哪来的？自己想的还是抄的？”尤其像 cloud 这种字段，你自己也说不清具体语义，老师认为：如果是你自己设计的，你必须讲得
  出含义；如果是引用/借鉴的，就要明确来源。 同时老师给了一个非常明确的方向：embedding 的字段应该尽量来自“编译器/target 描述体系”（TVM / GCC / LLVM 的 target options），至少这样每个字
  段都有“工程上站得住”的出处，而不是拍脑袋。 5) 数值归一化方案过度 heuristic：规则太多、太随意、难写进论文 这一段你老师讲得非常直接： 你用“收集 N 个硬件算均值方差”做归一化会依赖样本集
  合（选 4 个 vs 200 个结果就变了），这在方法学上不稳。 如果某一维所有硬件都几乎不变/恒为 0，那不是靠 mask/置零来处理，而是设计时就该删掉这个维度。 最大的问题是：你在归一化后又不断
  加“if / clip / log10 / log2 ……”的补丁，整体显得 ad-hoc，解释性很差。老师甚至点名：“你把这个写到 paper 里，你觉得能写吗？感觉很 tricky。” 老师建议的“更像 AI 圈会做的”方式是：定一个统
  一目标范围（例如 0~1 或 -1~1），规则尽量少且统一（要么统一 z-score，要么每维按最大值/理论上界缩放），避免一堆特例。 6) 目标函数未显式化：你想同时控制“强度”和“方向”，但 embedding 设
  计没对齐这个目标 你自己也意识到冲突：embedding 想表达很多信息，但最后在路由打分（dot/cos）时出现尺度冲突，于是你才做大量“数值处理”去让向量更可分。 老师的隐含观点是：“可分”不是靠不断
  调归一化规则堆出来的，而应该是 embedding 的语义结构 + 简洁归一化 + 路由训练共同实现。 一句话总括你现在的 embedding 主要问题 结构上：语义轴混杂、类别体系不完整；特征上：过度指纹化导
  致信息泄露；数值上：归一化太 heuristic 且样本依赖、难以解释与复现。 或许我们需要一个更自洽的 embedding 分段模板（identity / GPU-specific / CPU-specific / environment）以及一套尽量少
  规则的归一化方案，保证后面路由的 cos/dot 不再被尺度问题绑架。


  于是我们得到了 v5 版本的设计思路：
  A. Identity & Arch（0–4） 硬件类别/架构标志 0：is_gpu_hpc（高性能离散 GPU） 1：is_gpu_edge（边缘/SoC GPU） 2：is_cpu_x86（x86 CPU） 3：is_cpu_arm（ARM/AArch64 CPU） B. TVM 约束（4–10） 4：num_cores（CPU 核数） 5：vector_unit_bytes（cpu SIMD 向量宽度）（与cpu的avx-512等架构相关） 6：cache_line_bytes（cpu典型 64B） 7：max_shared_memory_per_block（GPU 共享内存） 8：max_threads_per_block（GPU 线程块上限) 9：registers_per_block（每个线程块的寄存器数目） 10：warp_size（GPU warp） C. 规模/性能（11–22） - 11: sm_count - 12: peak_flops_32 - 13: peak_matmul - 14: mem_bandwidth - 15: llc_mb ( gpu L2，cpu L3) - 16: mid_cache_kb（cpu_l2） - 17: device_mem_gb - 18: 矩阵加速强度: peak_matmul / peak_fp32 - 19: 低精度友好度: 0=FP32 only，1=FP16/BF16，2=INT8 - 20: compute_bw_ratio:log10(peak_fp32) - log10(mem_bandwidth) - 21: cache_bw_ratio:log10(llc_mb) - log10(mem_bandwidth) - 22: reserved 0 D. 环境（23–24） - 23: mem_is_uma（jetson/raspberry/apple为1，其余0） - 24: reserved 0 。

  C 段推荐维度（建议 9–11 维）
C1) 能力轴（capability，2–3 维）
能力轴不要用“突兀的一两个 flag”（老师讨厌那种 AmperePlus），而要么体系化，要么用连续可解释的 proxy。
(C_cap_1) 矩阵加速强度（连续，比 flag 更稳）
- 定义：matmul_accel_ratio = peak_matmul / peak_fp32（log 或直接比值）
- GPU：peak_matmul 可以用 TensorCore FP16/BF16/TF32 的峰值（如果你们有 perf_db）
- CPU：如果没有 AMX / SVE2-Matmul，就基本为 0 或很小（可用 BF16/INT8 dot 的峰值近似）
- 意义：这比“has_tensor_core”更体系化：它表达的是“矩阵乘法加速相对 FP32 的优势”。
(C_cap_2) 低精度友好度（可选，连续/分桶都行）
- 定义：lowp_level（例如 0=FP32 only, 1=BF16/FP16, 2=INT8）
- 意义：很多张量优化策略（layout/pack/tensorize）会因为 dtype 能力完全不同。
如果你现在拿不到 CPU 的 BF16/INT8/AMX 信息，可以先只做 matmul_accel_ratio 这一维，后续数据齐了再加第二维。

---
C2) 性能规格轴（scale/spec，6–7 维）
这些维度是你们现在 C 段已有的核心，但建议 CPU/GPU 都系统化补齐，并且把“缓存层级”说清楚。
(C_spec_1) 计算吞吐（FP32）
- 定义：log10(peak_fp32)
- GPU/CPU 都适用
- 意义：同类硬件差异（3090 vs 4090；不同 Xeon）最直接的“上限刻画”。
(C_spec_2) 矩阵吞吐（FP16/BF16/TF32 Tensor 或 AMX）
- 定义：log10(peak_matmul)（没有就 0）
- 意义：和 capability 轴呼应；也能区分“同样 FP32 但 tensor 强很多”的卡。
(C_spec_3) 内存带宽（HBM/GDDR/DDR）
- 定义：log10(mem_bandwidth)
- 意义：大量张量算子是 memory-bound，带宽决定 tile/融合收益上限。
(C_spec_4) LLC/片上共享缓存规模（统一口径）
- 定义：log10(llc_mb)
- GPU：用 L2（更接近 LLC 的角色）
- CPU：用 L3（LLC）
- 意义：决定 blocking / cache reuse 的合理粒度。这对 schedule 很关键，也比“微架构代号”更不指纹。
(C_spec_5) 中层私有缓存规模（建议补上，CPU 更重要）
- 定义：log10(mid_cache_kb)
- CPU：L2（可用 per-core 或总量，建议 per-core 更稳）
- GPU：若不好定义可置 0（N/A）
- 意义：CPU 的 tile 很多时候被 L2 卡住；你之前问 L1/L2，就是这里最该体现的地方。
(C_spec_6) 设备内存容量（VRAM / UMA 可用内存）
- 定义：log10(device_mem_gb)
- GPU：VRAM
- Jetson/UMA：共享内存容量（如果你们能拿到）
- CPU：如果数据来源不稳定（机器 RAM 变动），建议先置 0 或只用于 UMA 设备
- 意义：更偏“可行域/批大小/是否能放下”，对某些任务影响大，但确实容易受环境影响——所以要谨慎。
(C_spec_7) 并行计算单元规模（GPU 专用但很有用）
- 定义：log10(sm_count)（CPU 置 0）
- 意义：同样 TFLOPS 的卡，SM 数/频率分解不同，会影响 occupancy / 并行颗粒度偏好；在你们张量 schedule 场景里这维很常用，也相对不指纹。

---
C3) 派生比值（强烈推荐 2 维，路由更稳）
这两维的好处是：把“谁更 compute-bound / memory-bound”显式化，这比靠 router 自己在高维里猜要稳定，解释也非常自然（roofline）。
(C_ratio_1) Roofline 倾向：算力/带宽比
- 定义：compute_bw_ratio = log10(peak_fp32) - log10(mem_bandwidth)
- 意义：同样算子在不同硬件上最优 tile/fuse 策略会随这个比值改变。
(C_ratio_2) Cache 相对带宽：缓存/带宽比（或缓存/算力比）
- 定义：cache_bw_ratio = log10(llc_mb) - log10(mem_bandwidth)（或 llc_mb / peak_fp32）
- 意义：表达“更依赖 cache blocking 还是 streaming”，对 CPU/GPU 都有用。
这两维非常“少规则、可解释”，不会像 mask/clip 那样被说成补丁。


我们不直接把 ISA 的所有细粒度开关展开成 embedding 维度，而是把这些官方 target/ISA 表当作数据来源，抽象为少量与张量优化直接相关的“能力轴”（SIMD 宽度、矩阵加速单元、低精度支持、内存层级/带宽等）。这样既避免 embedding 指纹化，也让归一化和路由相似度更稳定、更可解释。

你列的这套实际上已经把 roofline 的两条主轴都覆盖了：
- 算力上限：peak_fp32 +（可选）peak_matmul
- 带宽上限：mem_bandwidth
- 片上复用能力：llc(L2/L3) + L2(per-core)
- 容量可行域：device_mem_gb
- 并行资源规模（GPU）：sm_count
- 两条解释性比值：compute_bw_ratio、cache_bw_ratio
- 矩阵/低精度能力：matmul_accel_ratio + lowp_level
从“张量优化”角度，很多“代际/ISA”差异最终都会体现在这些量上（或者可以用这些量做 proxy），所以 你不需要再把 ISA 表展开成几十维。