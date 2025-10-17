# MT-MoSLoRA: Multi-Task MoSLoRA with Hardware-Aware Dual-Track Architecture

## 概述

MT-MoSLoRA是基于MTLoRA模式的MoSLoRA扩展，实现了HA（硬件无关）+ HS（硬件专属）双轨制架构，专门为Gen-Edge框架设计。

## 核心设计理念

### 1. 双轨制架构
- **HA-MoSLoRA (Hardware-Agnostic)**: 共享的硬件无关模块，学习跨平台的通用优化知识
- **HS-MoSLoRA (Hardware-Specific)**: 硬件专属专家模块，学习特定硬件的优化策略

### 2. 前向传播机制
```
final_output = base_output + ha_delta + hs_delta
```
- `base_output`: 冻结的基础模型输出
- `ha_delta`: HA模块的通用知识修正
- `hs_delta`: HS模块的硬件特定修正

## 架构对比

### 原始MoSLoRA
```
Input → MoSLoRA → Output
```
- 单一MoSLoRA模块
- 通过W矩阵实现硬件感知

### MT-MoSLoRA
```
Input → Base Linear (frozen)
     → HA-MoSLoRA (shared)
     → HS-MoSLoRA (hardware-specific)
     → Combined Output
```
- HA模块：所有数据都更新
- HS模块：只在对应硬件数据时更新

## 关键组件

### 1. MTMoSLoRALinear类
```python
class MTMoSLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, ha_config, hs_config, hardware_types):
        # Base linear layer (frozen)
        self.base_linear = nn.Linear(in_features, out_features, bias=True)
        
        # HA (Hardware-Agnostic) MoSLoRA module
        self.ha_moslora = self._create_moslora_module(in_features, out_features, ha_config)
        
        # HS (Hardware-Specific) MoSLoRA modules
        self.hs_experts = nn.ModuleDict()
        for hw_type in hardware_types:
            self.hs_experts[hw_type] = self._create_moslora_module(in_features, out_features, hs_config)
```

### 2. 硬件路由机制
```python
def route_hardware(self, hardware_id: str) -> str:
    return self.hardware_router.get(hardware_id, 'v100')  # 默认路由
```

### 3. 前向传播逻辑
```python
def forward(self, x: torch.Tensor, hardware_id: str = None):
    # Base model output (frozen)
    base_output = self.base_linear(x)
    
    # HA adaptation - always applied
    ha_delta = self.ha_moslora(x)
    
    # HS adaptation - only for specific hardware
    hs_delta = torch.zeros_like(base_output)
    if hardware_id is not None:
        hw_type = self.route_hardware(hardware_id)
        if hw_type in self.hs_experts:
            hs_delta = self.hs_experts[hw_type](x)
    
    # Final output: base + HA + HS
    return base_output + ha_delta + hs_delta
```

## 使用方法

### 1. 训练脚本
```bash
bash run_mt_moslora.sh
```

### 2. 关键参数
```bash
--use_mt_moslora=true          # 启用MT-MoSLoRA
--use_mixer=true               # 启用MoSLoRA的W矩阵
--defuse_gpt2_attn=true        # 切开GPT-2的融合层
--ha_lora_r=16                 # HA模块的LoRA rank
--ha_lora_alpha=32             # HA模块的LoRA alpha
--hs_lora_r=16                 # HS模块的LoRA rank
--hs_lora_alpha=32             # HS模块的LoRA alpha
--hardware_types=v100,xavier,i7 # 支持的硬件类型
--target_modules=q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj
```

### 3. 数据格式
训练数据需要包含`hardware_id`字段：
```json
{
    "text": "张量程序优化内容...",
    "hardware_id": "v100"  // 或 "xavier", "i7"
}
```

## 训练流程

### 1. 数据预处理
```python
def extract_hardware_id_function(example):
    text_data = example['text']
    if "sm_70" in text_data or "v100" in text_data.lower():
        example['hardware_id'] = 'v100'
    elif "xavier" in text_data.lower():
        example['hardware_id'] = 'xavier'
    elif "i7" in text_data.lower():
        example['hardware_id'] = 'i7'
    else:
        example['hardware_id'] = 'v100'  # 默认
    return example
```

### 2. 硬件感知训练
```python
class HardwareAwareTrainer(transformers.Trainer):
    def training_step(self, model, inputs):
        if 'hardware_id' in inputs:
            model.current_hardware_id = inputs['hardware_id'][0]
        return super().training_step(model, inputs)
```

### 3. 模型保存
训练完成后，模型会保存为标准的PEFT格式，包含：
- `adapter_model.bin`: 适配器权重
- `adapter_config.json`: 适配器配置

## 参数效率

### 参数量对比
- **基础模型**: 冻结，不参与训练
- **HA模块**: 每个目标模块一个MoSLoRA适配器
- **HS模块**: 每个硬件类型 × 每个目标模块一个MoSLoRA适配器

### 示例计算
假设：
- 目标模块数: 6 (q_proj, k_proj, v_proj, attn.c_proj, mlp.c_fc, mlp.c_proj)
- 硬件类型数: 3 (v100, xavier, i7)
- LoRA rank: 16
- 隐藏维度: 768

参数量：
- HA模块: 6 × (16 × 768 + 768 × 16) = 147,456
- HS模块: 3 × 6 × (16 × 768 + 768 × 16) = 442,368
- 总计: 589,824 参数

相比全量微调（~117M参数），参数量减少99.5%。

## 测试

运行测试脚本验证功能：
```bash
python test_mt_moslora.py
```

测试包括：
1. MTMoSLoRALinear基本功能
2. 硬件路由机制
3. 模型集成
4. 前向传播一致性

## 与Gen-Edge框架的集成

MT-MoSLoRA完美契合Gen-Edge框架的设计理念：

1. **知识解耦**: HA模块学习通用知识，HS模块学习硬件特定知识
2. **模块化设计**: 每个硬件类型都有独立的专家模块
3. **零样本泛化**: 新硬件可以通过HA模块获得基础能力
4. **持续学习**: 可以为新硬件添加新的HS专家模块

## 扩展性

### 添加新硬件类型
1. 在`hardware_types`参数中添加新硬件
2. 更新`hardware_router`字典
3. 重新训练或增量训练

### 自定义硬件路由
```python
def _build_hardware_router(self) -> Dict[str, str]:
    return {
        'nvidia/nvidia-v100': 'v100',
        'nvidia/rtx-4090': 'v100',  # 映射到现有类型
        'nvidia/jetson-agx-xavier': 'xavier',
        'intel/i7': 'i7',
        'amd/ryzen': 'i7',  # 映射到现有类型
    }
```

## 性能优势

1. **训练效率**: 只训练LoRA参数，大幅减少训练时间
2. **内存效率**: 相比全量微调，内存占用显著降低
3. **硬件感知**: 真正的硬件特定优化
4. **可解释性**: 清晰的HA/HS分离，便于分析
5. **可扩展性**: 易于添加新硬件类型

## 注意事项

1. **数据质量**: 确保训练数据包含准确的硬件标识
2. **硬件平衡**: 尽量保证各硬件类型的数据量平衡
3. **超参数调优**: HA和HS模块的超参数可能需要分别调优
4. **内存管理**: 多个HS专家模块会增加内存占用

## 多硬件TLM模型基础性能测试

### 测试流程概述

多硬件TLM模型测试包含三个主要步骤：
1. **生成Sketch** - 为每个硬件生成测试提示
2. **生成张量程序** - 使用多硬件TLM模型生成优化程序  
3. **性能测量** - 在具体硬件上测量生成程序的执行延迟

### 四个硬件配置

| 硬件类型 | TVM Target | 特殊说明 |
|----------|------------|----------|
| **V100** | `nvidia/nvidia-v100` | 标准GPU target |
| **Xavier** | `nvidia/jetson-agx-xavier` | 嵌入式GPU target |
| **RTX 4090** | `cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32` | **特殊处理**: 使用sm_86架构，完整target字符串 |
| **Xeon** | `llvm -mcpu=skylake-avx512 -model=xeon` | CPU target |

### 步骤1: 生成Sketch (为每个硬件)

#### 1.1 V100 Sketch生成
```bash
python make_dataset.py \
--for_type=for_gen_eval_sketch \
--target=nvidia/nvidia-v100 \
--dataset_path=/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/v100 \
--tokenizer_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval \
--keep_cnt=64
```

#### 1.2 Xavier Sketch生成
```bash
python make_dataset.py \
--for_type=for_gen_eval_sketch \
--target=nvidia/jetson-agx-xavier \
--dataset_path=/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/xavier \
--tokenizer_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval \
--keep_cnt=64
```

#### 1.3 RTX 4090 Sketch生成 (特殊处理)
```bash
python make_dataset.py \
--for_type=for_gen_eval_sketch \
--target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
--dataset_path=/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/4090 \
--tokenizer_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval \
--keep_cnt=64
```

#### 1.4 Xeon Sketch生成
```bash
python make_dataset.py \
--for_type=for_gen_eval_sketch \
--target="llvm -mcpu=skylake-avx512 -model=xeon" \
--dataset_path=/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/xeon \
--tokenizer_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval \
--keep_cnt=64
```

### 步骤2: 生成张量程序 (使用多硬件TLM模型)

#### 2.1 V100程序生成
```bash
CUDA_VISIBLE_DEVICES=0 python gen_state.py \
--model_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
--sketch_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/0_merge.json \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/gen_eval.json \
--allow_repeat=True \
--target=nvidia/nvidia-v100 \
--keep_cnt=32
```

#### 2.2 Xavier程序生成
```bash
CUDA_VISIBLE_DEVICES=0 python gen_state.py \
--model_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
--sketch_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/0_merge.json \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/gen_eval.json \
--allow_repeat=True \
--target=nvidia/jetson-agx-xavier \
--keep_cnt=32
```

#### 2.3 RTX 4090程序生成 (特殊处理)
```bash
CUDA_VISIBLE_DEVICES=0 python gen_state.py \
--model_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
--sketch_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/0_merge.json \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/gen_eval.json \
--allow_repeat=True \
--target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
--keep_cnt=32
```

#### 2.4 Xeon程序生成
```bash
CUDA_VISIBLE_DEVICES=0 python gen_state.py \
--model_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
--sketch_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/0_merge.json \
--save_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/gen_eval.json \
--allow_repeat=True \
--target="llvm -mcpu=skylake-avx512 -model=xeon" \
--keep_cnt=32
```

### 步骤3: 性能测量 (在具体硬件上执行)

**重要说明**: 性能测量步骤需要在**对应的具体硬件**上执行，确保测量结果的准确性。

#### 3.1 V100性能测量 (在V100硬件上执行)
```bash
CUDA_VISIBLE_DEVICES=0 python measure_programs.py \
--batch-size=64 \
--target=nvidia/nvidia-v100 \
--to-measure-path=/root/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_v1/gen_train.json \
--measured-path=/root/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_v1/measured_results.json
```

#### 3.2 Xavier性能测量 (在Xavier硬件上执行)
```bash
CUDA_VISIBLE_DEVICES=0 python measure_programs.py \--batch-size=64 \--target=nvidia/jetson-agx-xavier \--to-measure-path=/home/walker/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_programs_v1/gen_eval.json \--measured-path=/home/walker/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_programs_v1/measured_results.json
```

```bash
CUDA_VISIBLE_DEVICES=0 python measure_programs.py \--batch-size=64 \--target=nvidia/jetson-agx-xavier \--to-measure-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/gen_eval.json \--measured-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/measured_results.json
```

#### 3.3 RTX 4090性能测量 (在RTX 4090硬件上执行)
```bash
CUDA_VISIBLE_DEVICES=1 python measure_programs.py \
--batch-size=64 \
--target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
--to-measure-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/gen_eval.json \
--measured-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/measured_results.json
```

#### 3.4 Xeon性能测量 (在Xeon硬件上执行)
```bash
CUDA_VISIBLE_DEVICES=1 python measure_programs.py \
--batch-size=64 \
--target="llvm -mcpu=skylake-avx512 -model=xeon" \
--to-measure-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/gen_eval.json \
--measured-path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/measured_results.json
```

### 步骤4: 数据后处理 (生成测量记录索引)

**重要说明**: 在性能测量完成后，需要运行postprocess.py来整理测量结果并生成重复测量避免索引。

#### 4.1 更新utils.json配置
在运行postprocess.py之前，需要先更新`utils.json`文件，添加新的测量记录路径：

```json
{
  "v100": {
    "measure_records": [
      "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/measured_results.json"
    ]
  },
  "xavier": {
    "measure_records": [
      "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/measured_results.json"
    ]
  },
  "4090": {
    "measure_records": [
      "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/measured_results.json"
    ]
  },
  "xeon": {
    "measure_records": [
      "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/measured_results.json"
    ]
  }
}
```

#### 4.2 运行postprocess.py
为每个硬件运行postprocess.py命令：

```bash
# V100硬件
python postprocess.py --target=nvidia/nvidia-v100

# Xavier硬件  
python postprocess.py --target=nvidia/jetson-agx-xavier

# RTX 4090硬件 (特殊处理)
python postprocess.py --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32"

# Xeon CPU硬件
python postprocess.py --target="llvm -mcpu=skylake-avx512 -model=xeon"
```

**输出文件**: 每个命令会生成对应的`measured_{硬件}.pkl`文件，用于后续的重复测量避免机制。

### 步骤5: bert_base编译测试 (端到端性能评估)

**重要说明**: 编译测试步骤使用测量结果中性能最好的张量程序来编译bert_base模型，并测量端到端执行延迟。

#### 5.1 V100编译测试
```bash
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/root/tlm/tlm_dataset/gen/gen_data/mutil_to_measure_v1/v100_measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target=nvidia/nvidia-v100 --backend=graph
```

#### 5.2 Xavier编译测试
```bash
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target=nvidia/jetson-agx-xavier --backend=graph
```

#### 5.3 RTX 4090编译测试 (特殊处理)
```bash
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32" --backend=graph
```

#### 5.4 Xeon编译测试
```bash
CUDA_VISIBLE_DEVICES=0 TLM_LOG_FILE=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/measured_results.json python tune_relay.py --workload=bert_base --input-shape=[1,128] --target="llvm -mcpu=skylake-avx512 -model=xeon" --backend=graph
```

### 执行策略

#### 并行执行策略
1. **步骤1**: 可以并行执行4个sketch生成命令
2. **步骤2**: 可以并行执行4个程序生成命令（使用不同GPU）
3. **步骤3**: 需要在对应硬件上分别执行测量
4. **步骤4**: 可以并行执行4个postprocess.py命令

#### 串行执行策略
```bash
# 按硬件顺序执行
# V100 → Xavier → RTX 4090 → Xeon
```

**重要提醒**: 在运行postprocess.py之前，必须先更新`utils.json`文件，添加新的测量记录路径，否则postprocess.py无法找到测量结果文件。

### 关键参数说明

- `--keep_cnt=64`: 每个硬件测试64个任务
- `--keep_cnt=32`: 每个任务生成32个候选程序
- `CUDA_VISIBLE_DEVICES=0/1`: 使用不同GPU避免冲突
- 使用多硬件tokenizer和模型确保跨硬件兼容性

### RTX 4090特殊处理说明

RTX 4090需要特殊处理的原因：
1. **TVM tag.cc中未定义**: `nvidia/nvidia-rtx-4090`在TVM的tag.cc文件中不存在
2. **使用sm_86架构**: 虽然RTX 4090实际是sm_89，但当前TVM版本支持sm_86
3. **完整target字符串**: 需要包含所有硬件参数以确保正确识别

### 推理流程说明

`gen_state.py`采用**LLM生成 + TVM验证**的两阶段流程：
1. **LLM生成**: 使用多硬件TLM模型生成张量程序候选
2. **TVM验证**: 每个生成的程序都会被TVM验证和构建
3. **质量保证**: 只保留验证通过的程序，确保在目标硬件上可执行

### 预期结果

测试完成后，你将获得：
- 四种硬件的张量程序生成结果
- 各硬件上的实际执行性能数据
- 多硬件TLM模型在不同硬件上的表现评估

## 未来改进方向

1. **动态硬件路由**: 基于硬件特征自动路由
2. **知识蒸馏**: 从HA模块向HS模块蒸馏知识
3. **多任务学习**: 同时优化多个任务
4. **自适应rank**: 根据硬件复杂度调整LoRA rank

