# Emb-v5 硬件规格信息源与搜索流程（可复用）

本文件用于记录：为 `gen/Embedding/gen_hw_emb_v5.py` 补齐新硬件（CPU/GPU/Jetson）的**官方信息源**与**可复现的搜索/推导流程**。

## 1. 总体原则（对齐 codex_todo.md）

- 优先使用**官方来源**：NVIDIA / Intel / AMD / AWS / Ampere 的官网页面或官方 PDF（datasheet / product brief / tuning guide / architecture blog）。
- 若官方资料**无法查到某字段**：宁可在 embedding 中将该字段置 0，并在 meta 里标注 `TODO`（不要用第三方猜数）。
- 若需要推导：在 meta 里写清楚**推导公式**与使用的官方字段（例如：内存通道数、内存速率、CUDA cores 等）。

## 2. 常用官方信息源（按厂商）

### Intel（Xeon / Core）

- Intel ARK（SKU 规格，最常用）
  - 示例：
    - Xeon Platinum 8480C: `https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-platinum-8480c-processor-105m-cache-2-00-ghz/specifications.html`
    - Xeon Platinum 8380: `https://www.intel.com/content/www/us/en/products/sku/212285/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz/specifications.html`
  - 典型可直接取字段：
    - `# of Cores`（num_cores）
    - `Processor Base Frequency`（base_freq_GHz，用于 CPU peak_fp32 推导）
    - `Cache`（通常是 L3/LLC）
    - `Max # of Memory Channels` + `Memory Types/Speed`（用于 mem_bandwidth 推导）
    - `Instruction Set Extensions`（是否 AVX-512）

- Intel 官方“缓存层级差异”说明（用于 L2 per-core / L2 total 推导）
  - `https://www.intel.com/content/www/us/en/support/articles/000093753/processors.html`
  - 用法：
    - 找到对应代际（3rd Gen / 4th Gen Xeon）的 `Mid-Level Cache`（L2）大小（例如 1.25MB/core 或 2MB/core）
    - `L2_total_MB = num_cores * L2_per_core_MB`

### AMD（EPYC）

- AMD EPYC 9004 系列 datasheet（SKU 表常含 cores/base/L3/mem bw）
  - `https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/data-sheets/amd-epyc-9004-series-processors-data-sheet.pdf`

- AMD EPYC 9004 tuning guide（用于补齐“微结构/缓存/指令集”类字段）
  - `https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-9004-tg-58011.pdf`
  - 用法：
    - 搜索 `L2` 获取 per-core L2（例如 “Up to a 1MB private unified L2 cache”）
    - 搜索 `AVX-512` 佐证 vector ISA 宽度（vector_unit_bytes=64）

- AMD EPYC 7003 press release（SKU 表用于 cores/base/L3、内存通道/速率）
  - `https://www.amd.com/en/newsroom/press-releases/2021-03-15-amd-launches-3rd-gen-amd-epyc-processors.html`

- AMD EPYC 7003 microarchitecture overview（用于 L2 per-core）
  - `https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-7003-series-microarchitecture-overview.pdf`
  - 用法：
    - 搜索 `L2` 获取 “512KB private unified L2” 等描述

### AWS（Graviton）

- AWS Graviton Technical Guide（AWS 官方技术手册，推荐优先使用）
  - `https://aws.github.io/graviton/`
  - 典型字段（可直接引用）：
    - cores / frequency
    - L2 per core / SLC(LLC)
    - ISA（NEON / SVE 等）
    - DRAM 通道数量（例如 8x DDR5）

- Arm 官方博客（用于补齐“DDR5-4800”这类 AWS 文档不一定显式写出的字段）
  - `https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/leading-hpc-performance-with-graviton4`
  - 用法：
    - 博客内含 Graviton3E 的规格表（包含 “8 x DDR5-4800”），可用来为 mem_bandwidth 推导提供“官方(Arm)依据”

### Ampere（Altra / Altra Max）

- Ampere Altra Max product brief（官方 PDF，较全）
  - `https://amperecomputing.com/assets/documents/Ampere_Altra_Max_Product_Brief.pdf`
  - 典型字段：
    - cores / frequency
    - L2 per core / System Level Cache (SLC)
    - 内存通道与 DDR4-3200（用于 mem_bandwidth 推导）
    - vector unit 描述（用于 vector_unit_bytes）

### NVIDIA（Data Center GPU / GeForce / Jetson）

- NVIDIA 官方产品页（常含 peak_fp32、tensor TFLOPS、mem bw、显存）
  - H100: `https://www.nvidia.com/en-us/data-center/h100/`
  - A10: `https://www.nvidia.com/en-us/data-center/products/a10-gpu/`
  - Tesla T4: `https://www.nvidia.com/en-us/data-center/tesla-t4/`

- NVIDIA resources（datasheet / product brief PDF，适合补齐“CUDA cores/显存/带宽/算力口径”）
  - RTX A40 datasheet: `https://resources.nvidia.com/en-us-brief/rtx-a40-datasheet`
  - A10 datasheet（若需要核对 CUDA cores/SM 信息）：
    - `https://resources.nvidia.com/en-us-brief/nvidia-a10-datasheet`（TODO：逐字段映射）

- NVIDIA Developer Blog / Whitepaper（架构细节：SM/L2 等）
  - Hopper in-depth: `https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/`

- GeForce 官方规格页（消费级：CUDA cores / clocks / memory bandwidth 等；部分字段如 L2 可能缺失）
  - 10-series: `https://www.nvidia.com/en-us/geforce/10-series/10-series-specs/`
  - 900-series: `https://www.nvidia.com/en-us/geforce/900-series/900-series-specs/`

- Jetson 官方规格/brief（SoC/UMA：CPU 核数、CUDA cores、内存带宽/容量、功耗档等）
  - Jetson Orin 技术规格表：`https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/`
  - Jetson Orin NX datasheet：
    - `https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/jetson_orin_nx_series_modules_ds-10712.pdf`
  - Jetson AGX Orin technical brief：
    - `https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx-orin/Jetson_AGX_Orin_Technical_Brief.pdf`
  - Jetson Orin Nano Super Dev Kit（用于某些 Orin Nano 配置的官方可得规格）：
    - `https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/`

- Tesla T4 官方 datasheet PDF（用于精确 mem_bw=300 / mixed-precision=65）
  - `https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf`
  - 注意：NVIDIA 产品页常写 “320+ GB/s”，datasheet (Mar19) 写 “300 GB/s”；需要在 meta 里解释你采用的口径。

## 3. 可复现的搜索/推导流程（建议模板）

下面是一套“从字段出发”的可复用流程。建议你/学生以后补新硬件时也按这个做，避免 ad-hoc。

### 3.1 CPU：num_cores / base_freq / cache / mem_bandwidth / L2_total

1) 先找 SKU 官方规格页（Intel ARK / AMD datasheet / Ampere brief / AWS guide）
2) 抽取：
   - num_cores（核心数）
   - base_freq（用于 peak_fp32 推导；CPU 统一用 base frequency）
   - LLC/L3（对应 `llc_mb` 的 CPU 语义）
   - memory channels + memory speed（用于 mem_bandwidth 推导）
3) L2_total：
   - 若官方 SKU 页直接给 total L2：直接用
   - 否则用“官方微结构/代际文档”找 per-core L2，再乘以核心数：
     - `L2_total_MB = num_cores * L2_per_core_MB`
4) mem_bandwidth（理论值）：
   - DDR4-3200：每通道 `25.6 GB/s`
   - DDR5-4800：每通道 `38.4 GB/s`
   - 总带宽：`mem_bw = channels * per_channel_bw`

### 3.2 GPU：SM count / peak_fp32 / peak_matmul / mem_bandwidth / vram

1) 先找官方产品页或 datasheet（NVIDIA）
2) 抽取：
   - peak_fp32（若无，使用 CUDA cores + boost clock 推导）
   - peak_matmul（Tensor TFLOPS；若只有 sparse marketing 值，dense ~= /2）
   - mem_bandwidth、vram_gb
3) SM count：
   - 最理想：官方架构文档直接给 SM count
   - 次优：用官方给出的 CUDA cores + 官方架构每 SM CUDA cores 推导（需要有官方架构文档支撑）
4) L2（GPU `llc_mb`）：
   - 若官方未给：置 0 并在 meta 标注 TODO（不要用第三方猜）

### 3.3 Jetson（SoC/UMA）：CPU cores / CUDA cores / mem bw / UMA mem

1) 先用 Jetson 官方规格表 / datasheet / technical brief
2) 抽取：
   - CPU cores（写入 v5 的 `num_cores`）
   - CUDA cores（推导 `sm_count`：Ampere 通常 `SM = CUDA_cores / 128`）
   - Memory bandwidth / Memory size（写入 `mem_bandwidth`、`device_mem_gb`，并置 `mem_is_uma=1`）
3) Tensor/TOPS：
   - 若官方只给 TOPS（INT8）但不给 FP16/BF16 Tensor TFLOPS：`peak_matmul` 置 0 并 TODO

## 4. 本次补齐的“实际搜索记录”（可复用）

下面记录的是“我这次为 emb-v5 补齐/修正字段时”的搜索路径与关键依据，方便你以后复核或让学生复现。

### 4.1 Tesla T4：mem_bandwidth 300 vs 320+ 的冲突如何处理

1) 先找官方 datasheet（最稳定的 PDF 口径）
   - 搜索：`NVIDIA T4 tensor core datasheet 951643`
   - 结果：官方 PDF `t4-tensor-core-datasheet-951643.pdf`
   - 在 datasheet 表格中找到：
     - Memory Bandwidth (GB/sec) = 300
     - Mixed precision TFLOPS = 65

2) 再查官方产品页（会出现不同口径）
   - 页面：`https://www.nvidia.com/en-us/data-center/tesla-t4/`
   - 发现写法通常是 `320+ GB/s`

3) 最终取值策略
   - embedding 里取 `mem_bandwidth=300`（datasheet 明确给出，且可稳定引用）
   - meta 里写清楚“产品页口径不同”，避免评审/老师追问时解释不清

### 4.2 Jetson Orin Nano / Orin NX：用官方规格表推导 peak_fp32

1) 打开 Jetson Orin 官方规格表
   - `https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/`

2) 在页面表格中定位字段
   - CUDA cores（例如 1024）
   - GPU Max Frequency（MHz）
   - Memory bandwidth（GB/s）

3) 用同一套公式推导 peak_fp32（TFLOPS）
   - `peak_fp32 = cuda_cores * 2(FMA) * (gpu_max_freq_MHz/1000) / 1000`
   - 例：Orin Nano 8GB (Super)
     - 1024 cores, 1020MHz => 1024*2*1.020/1000 = 2.08896 TFLOPS
   - 例：Orin NX 16GB (Super)
     - 1024 cores, 1173MHz => 1024*2*1.173/1000 = 2.402304 TFLOPS

### 4.3 Graviton3：AWS 文档不写 MT/s 时，如何“去 ASSUME”

1) 先用 AWS 官方 Technical Guide 拿到“通道数/缓存/ISA”等字段
   - `https://aws.github.io/graviton/`
   - 可直接引用：`8x DDR5`、SLC/LLC、SVE 256b 等

2) 再补齐 DDR5 速率（用于 mem_bandwidth 推导）
   - 搜索：`Graviton3 DDR5-4800 memory channels`
   - 选用 Arm 官方博客（包含 Graviton3E 的规格表，明确写出 `8 x DDR5-4800`）
     - `https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/leading-hpc-performance-with-graviton4`

3) 推导 mem_bandwidth 并写入 meta
   - per-channel bw(DDR5-4800) = 38.4 GB/s
   - `mem_bw = channels * per_channel_bw = 8 * 38.4 = 307.2 GB/s`
   - meta 中同时保留 `src_aws` + `src_arm`，保证可审计、可复现
