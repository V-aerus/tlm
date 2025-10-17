# MT-MoSLoRA 编译测试性能基准记录

## 📊 概述

本文档记录MT-MoSLoRA模型在不同硬件上的bert_base编译测试性能，用于跟踪模型迭代优化效果。

## 🎯 测试配置

### 测试模型
- **工作负载**: bert_base
- **输入形状**: [1, 128] (batch_size=1, sequence_length=128)
- **后端**: graph
- **测试框架**: TVM Relay

### 硬件配置

| 硬件类型 | TVM Target | 特殊说明 |
|----------|------------|----------|
| **V100** | `nvidia/nvidia-v100` | 标准GPU target |
| **Xavier** | `nvidia/jetson-agx-xavier` | 嵌入式GPU target |
| **RTX 4090** | `cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32` | 完整target字符串 |
| **Xeon** | `llvm -mcpu=skylake-avx512 -model=xeon` | CPU target |

## 📈 性能记录

### 版本记录格式
```
## 版本 X.X (YYYY-MM-DD)
- **模型路径**: /path/to/model
- **训练数据**: 描述
- **关键改进**: 改进点描述

### 硬件性能对比

| 硬件 | 延迟 (ms) | 相对性能 | 备注 |
|------|-----------|----------|------|
| V100 | X.XX | 1.00x | 基准 |
| Xavier | X.XX | X.XXx | |
| RTX 4090 | X.XX | X.XXx | |
| Xeon | X.XX | X.XXx | |

### 详细结果
- **V100**: X.XX ms
- **Xavier**: X.XX ms  
- **RTX 4090**: X.XX ms
- **Xeon**: X.XX ms
```

---

## 版本 1.0 (2024-09-23)
- **模型路径**: `/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1`
- **训练数据**: 多硬件混合训练数据 (V100, Xavier, RTX 4090, Xeon)
- **关键改进**: 首次多硬件MT-MoSLoRA模型，支持硬件感知推理

### 硬件性能对比

| 硬件 | 延迟 (ms) | 相对性能 | 备注 |
|------|-----------|----------|------|
| V100 | 16.0538 | 1.00x | 基准 |
| Xavier | 36.6987 | 0.44x | 比V100慢56% |
| RTX 4090 | 9.6796 | 1.66x | 比V100快66% |
| Xeon | 35.4667 | 0.45x | 比V100慢55% |

### 详细结果
- **V100**: 16.0538 ms
- **Xavier**: 36.6987 ms
- **RTX 4090**: 9.6796 ms
- **Xeon**: 35.4667 ms

### 性能分析

#### 当前测试结果总结
- **RTX 4090** 表现最佳：9.68ms，比V100快66%
- **V100** 作为基准：16.05ms
- **Xeon CPU** 相对较慢：35.47ms，比V100慢55%
- **Xavier** 最慢：36.70ms，比V100慢56%

#### 硬件性能排序（从快到慢）
1. **RTX 4090**: 9.68ms (最快)
2. **V100**: 16.05ms (基准)
3. **Xeon**: 35.47ms
4. **Xavier**: 36.70ms (最慢)

#### 关键发现
- **RTX 4090的sm_86架构优化效果显著**：比V100快66%
- **GPU vs CPU性能差异明显**：Xavier和Xeon都比V100慢55-56%
- **Xavier vs Xeon性能接近**：Xavier(36.70ms) vs Xeon(35.47ms)，差异仅3.5%
- **多硬件MT-MoSLoRA模型能够有效利用不同硬件的特性**

### 测试命令
```bash
# V100编译测试
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target=nvidia/nvidia-v100 --backend=graph

# Xavier编译测试
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/walker/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_programs_v1/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target=nvidia/jetson-agx-xavier --backend=graph

# RTX 4090编译测试
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --backend=graph

# Xeon编译测试
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target="llvm -mcpu=skylake-avx512 -model=xeon" --backend=graph
```

---

## 📝 记录说明

### 如何添加新版本记录

1. **复制版本记录格式**：复制上面的版本记录格式
2. **更新版本信息**：修改版本号、日期、模型路径等
3. **填写性能数据**：将实际测试结果填入表格
4. **添加改进说明**：描述本次版本的关键改进点

### 性能数据收集

1. **运行编译测试命令**：使用对应硬件的编译测试命令
2. **记录延迟数据**：从输出中提取端到端执行延迟
3. **计算相对性能**：以V100为基准计算相对性能
4. **更新文档**：将数据填入对应版本记录

### 注意事项

- **硬件一致性**：确保每次测试使用相同的硬件配置
- **环境一致性**：保持测试环境的一致性（CUDA版本、驱动版本等）
- **多次测试**：建议进行多次测试取平均值
- **详细记录**：记录任何可能影响性能的因素

## 🔄 版本历史

| 版本 | 日期 | 主要改进 | 状态 |
|------|------|----------|------|
| 1.0 | 2024-09-23 | 首次多硬件MT-MoSLoRA模型 | 测试中 |

## 📊 性能趋势分析

### 硬件性能对比趋势
- 待数据积累后添加性能趋势图表

### 模型优化效果
- 待多版本数据后添加优化效果分析

---

**最后更新**: 2024-09-23  
**维护者**: MT-MoSLoRA开发团队
