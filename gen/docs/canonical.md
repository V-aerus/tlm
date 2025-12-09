# EdgeTLM 硬件 Target Canonical 规范（草案）

> 目的：统一 TLM 看到的硬件 target 语法，避免同一类硬件用多种字符串形式描述，
> 为后续的 bucket 设计、KV side-channel 注入提供一个稳定的“文本世界”。

## 0. 总体设计思路

1. **TLM-base 只看到 canonical 的硬件字符串**  
   - 所有 `text` 字段中的 TVM target 串，都应先通过 `canonicalize_target(hw_id, raw_target)` 标准化后再写入。
   - `gen_state` / `make_dataset` / aligner 训练数据，都使用这一套规则。

2. **区分两层信息：**
   - 文本侧（discrete）：保留**类型级别**的信息（GPU-HPC / GPU-EDGE / CPU-X86 / CPU-ARM 等），尽量避免具体型号 / 易 OOV 字符串。
   - 连续侧（continuous）：由 `hw_emb` + KV-aligner 表达**细粒度差异**（sm_70 vs sm_86、carmel vs cortex-a72 等）。

3. **未来 bucket 规则基于 canonical 表达**  
   - bucket 化（例如 `-arch=sm_86 -> -arch=[HW_GPU_HPC]`，`-mcpu=cortex-a72 -> -mcpu=[HW_CPU_ARM]`）都是在 canonical target 之上进行。
   - 因此 canonical 层必须先收敛，再做 bucket 层设计。

---

## 1. GPU 目标硬件的 canonical 规则

### 1.1 GPU-HPC（V100 / A40 / 4090 等）

**适用硬件：**

- NVIDIA V100、A40、A100、4090 等高性能数据中心 / 高端桌面 GPU。

**canonical 形式：**

```text
cuda -keys=cuda,gpu -arch=sm_XX \
-max_num_threads=1024 \
-max_shared_memory_per_block=49152 \
-max_threads_per_block=1024 \
-registers_per_block=65536 \
-thread_warp_size=32
sm_XX 根据具体硬件决定：V100=sm_70，A40=sm_80，4090=sm_86。

4090 的历史短 target（带 -model=4090 + 一串数字）统一被替换为上述 canonical 形式。

说明：

4090 的所有样本必须先用离线脚本（例如 patch_4090_canonical.py）做一次修复，确保 text 中见到的都是 canonical 版。

后续新数据（eval/gen_eval）一律在 make_dataset / gen_state 中构造时就用 canonical 形式，不再出现 -model=4090 的旧样式。

2. GPU-EDGE（Jetson 等嵌入式 GPU）
适用硬件：

NVIDIA Jetson AGX Xavier 等嵌入式 GPU 平台。

canonical 形式（设备端 GPU）：

text
复制代码
cuda -keys=cuda,gpu -arch=sm_72 \
-max_num_threads=1024 \
-max_shared_memory_per_block=49152 \
-max_threads_per_block=1024 \
-registers_per_block=65536 \
-thread_warp_size=32
arch 根据 Jetson 型号设定（示例为 AGX Xavier 的 sm_72）。

说明：

与 GPU-HPC 一样，保证所有 Jetson GPU target 串在 dataset 中形式一致（不要出现带 -model=xxx 的变种）。

这部分在未来 bucket 化时，-arch=sm_72 将被映射到 [HW_GPU_EDGE]。

3. CPU-X86 目标硬件（服务器 CPU）
适用硬件：

Xeon 等 x86 服务器 CPU。

canonical 形式：

text
复制代码
llvm -keys=cpu -mcpu=skylake-avx512 -model=xeon \
<num_cores and other numeric fields...>
细节字段：num-cores、缓存类字段等保持原来的数值形式（按原 TLM 数据集习惯）。

mcpu 统一为 skylake-avx512（或选定的一种代表 CPU 微架构）。

说明：

后续 bucket 可能会定义：

-mcpu=skylake-avx512 -> -mcpu=[HW_CPU_X86]

-model=xeon -> -model=[MODEL_CPU_SERVER]

但在 canonical 层只做统一文本形式，暂不做 bucket。

4. CPU-ARM（Jetson host / Raspberry Pi 等）
这是本次新增的关键规则。

适用硬件：

Jetson host（例如 mcpu=carmel，mtriple=aarch64-linux-gnu）。

Raspberry Pi 4B（例如 mcpu=cortex-a72，mtriple=aarch64-linux-gnu）。

未来所有 ARM CPU 目标（只要是 kind=llvm + mtriple=aarch64-* 且不属于 x86）。

4.1 canonical 目标形式（简化版）
统一写成：

text
复制代码
llvm -keys=cpu -mtriple=aarch64-linux-gnu -mcpu=ARM_CPU_GENERIC \
<num-cores and other numeric fields...>
在实现中，ARM_CPU_GENERIC 可以直接替换为 bucket token [HW_CPU_ARM]（见 4.2），也可以先用一个中立占位符（如 arm_cpu_generic）再在 bucket 阶段替换。

4.2 bucket-aware 默契写法（推荐）
由于我们后续计划将 ARM CPU 统一 bucket 成 [HW_CPU_ARM]，在 canonical 规则中，可以直接约定：

规则：任何 ARM CPU 的 mcpu=... 在 canonical 后统一替换为：

text
复制代码
-mcpu=[HW_CPU_ARM]
举例：

Jetson host 原始形式（逻辑上）：

text
复制代码
llvm -keys=cpu -mtriple=aarch64-linux-gnu -mcpu=carmel -num-cores=8 ...
canonical 后：

text
复制代码
llvm -keys=cpu -mtriple=aarch64-linux-gnu -mcpu=[HW_CPU_ARM] -num-cores=8 ...
Raspberry Pi 4B 原始形式：

text
复制代码
llvm -keys=cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -num-cores=4 ...
canonical 后：

text
复制代码
llvm -keys=cpu -mtriple=aarch64-linux-gnu -mcpu=[HW_CPU_ARM] -num-cores=4 ...
注意：

ARM 具体微架构差异（carmel vs cortex-a72）不再用文本表达，而是由 hw_emb 在 KV side-channel 中表达。

文本侧只表示“这是 ARM CPU 一类”，不会因为出现新 CPU 名称而 OOV。

5. host 字段的处理（GPU + host 组合）
对于像 Jetson / Raspberry Pi 这种 GPU + host 的 target，通常有两层：

device 端：cuda ... arch=sm_XX ...；

host 端：llvm ... mtriple=aarch64-linux-gnu mcpu=... ...。

canonical 规则：

device 端按 GPU-HPC / GPU-EDGE 规则 canonical；

host 端按 CPU-ARM 规则 canonical，并统一将 mcpu=... 替换为 [HW_CPU_ARM]；

两者组合成完整 target 串时，确保格式在所有样本中一致（字段顺序可以固定为：device target → host target → 数值字段）。

6. canonical 实现位置（代码层面约定）
make_dataset.py / 数据生成脚本

所有 text / text_student 的 target 段在写入前必须调用：

python
复制代码
canonical_target = canonicalize_target(hw_id, raw_target_string)
canonicalize_target 内部根据 hw_id / kind / mtriple 决定走 GPU-HPC / GPU-EDGE / CPU-X86 / CPU-ARM 等分支，并应用上述规则。

gen_state_* / eval 生成脚本

用于 eval / gen_eval 的 target（无论是 for_train 还是 for_eval），同样通过 canonicalize_target 构造。

这样 base TLM 在训练 / 推理时看到的 target 形式始终一致。

历史数据修正脚本

patch_4090_canonical.py：一次性修复已有数据中 4090 的短 target；

未来如有必要，可增加 patch_arm_cpu_canonical.py：

扫描 hw_id 属于 ARM CPU 的样本；

在 text 中将已有 mcpu=... 替换为 mcpu=[HW_CPU_ARM]。

7. 与 KV-aligner / hw_emb 的衔接（说明）
canonical 层只是保证 “文本世界干净统一”：

GPU-HPC / GPU-EDGE / CPU-X86 / CPU-ARM 这些类型的信息留在文本中（可进一步 bucket）；

具体型号 / 细粒度硬件差异从文本中剥离，交由 hw_emb 表达。

在 KV-aligner 阶段：

对于 ARM 系列硬件（Jetson host / Raspberry Pi 等），它们在文本中都表现为 mcpu=[HW_CPU_ARM]；

但 hw_emb 不同，例如：

Jetson host：hw_emb 中编码 8 核 ARM、带 GPU、边缘设备等属性；

Raspberry Pi：hw_emb 中编码 4 核 ARM、无大 GPU、低功耗等属性；

KV-aligner 将这些连续信息映射为额外的 K/V slot，使得 TLM 在 schedule 生成时能区分不同 ARM 设备，而不会在文本层被 OOV 拖垮。
