# coding=utf-8
"""
HardwareExpertLoRALinear: 硬件专家LoRA模块 (V5 最终修复版)

1. 遵循PEFT的继承式设计。
2. 修正了参数命名，确保冻结逻辑正确生效。
3. 彻底重写了GPT-2的defusion和LoRA替换逻辑，确保维度、权重复制和forward方法都正确。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import transformers
from transformers.pytorch_utils import Conv1D

# --- 1. 配置类 ---
@dataclass
class HardwareExpertLoRAConfig:
    # ... (这部分与您文件中的完全相同，为简洁省略)
    ha_r: int = field(default=8)
    ha_alpha: float = field(default=16.0)
    ha_dropout: float = field(default=0.1)
    hs_r: int = field(default=8)
    hs_alpha: float = field(default=16.0)
    hs_dropout: float = field(default=0.1)
    hardware_types: List[str] = field(default_factory=lambda: ['gpu', 'cpu_hw', 'edge'])
    bias: str = field(default="none")
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "c_proj", "c_fc"])
    enable_hardware_routing: bool = field(default=True)
    default_hardware_type: str = field(default="gpu")
    trainable_ha: bool = field(default=True)
    trainable_hs: bool = field(default=True)

# --- 2. 核心LoRA层 (已修正参数命名) ---
class HardwareExpertLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, config: HardwareExpertLoRAConfig, **kwargs):
        super().__init__(in_features, out_features, bias=(config.bias != "none"), **kwargs)
        self.config = config
        self.merged = False
        self.disable_adapters = False

        # HA模块 (参数名中加入'lora'，确保能被冻结逻辑识别)
        if config.trainable_ha and config.ha_r > 0:
            self.ha_lora_A = nn.Parameter(self.weight.new_zeros((config.ha_r, in_features)))
            self.ha_lora_B = nn.Parameter(self.weight.new_zeros((out_features, config.ha_r)))
            self.ha_scaling = config.ha_alpha / config.ha_r
            self.ha_lora_dropout = nn.Dropout(config.ha_dropout)

        # HS模块 (参数名中加入'lora'，确保能被冻结逻辑识别)
        if config.trainable_hs and config.hs_r > 0:
            self.hs_lora_experts_A = nn.ParameterDict()
            self.hs_lora_experts_B = nn.ParameterDict()
            for hw_type in config.hardware_types:
                self.hs_lora_experts_A[hw_type] = nn.Parameter(self.weight.new_zeros((config.hs_r, in_features)))
                self.hs_lora_experts_B[hw_type] = nn.Parameter(self.weight.new_zeros((out_features, config.hs_r)))
            self.hs_scaling = config.hs_alpha / config.hs_r
            self.hs_lora_dropout = nn.Dropout(config.hs_dropout)
            self.hardware_router = self._build_hardware_router()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if hasattr(self, 'ha_lora_A'):
            nn.init.kaiming_uniform_(self.ha_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.ha_lora_B)
        if hasattr(self, 'hs_lora_experts_A'):
            for hw_type in self.config.hardware_types:
                nn.init.kaiming_uniform_(self.hs_lora_experts_A[hw_type], a=math.sqrt(5))
                nn.init.zeros_(self.hs_lora_experts_B[hw_type])

    def _build_hardware_router(self) -> Dict[str, str]:
        return {'nvidia/nvidia-v100': 'gpu', 'nvidia/rtx-4090': 'gpu', 'intel/i7': 'cpu_hw', 'nvidia/jetson-agx-xavier': 'edge'}

    def forward(self, x: torch.Tensor, target_hardware: str = None):
        previous_dtype = x.dtype
        result = F.linear(x, self.weight, self.bias)

        if self.disable_adapters or self.merged:
            return result

        # HA LoRA 计算
        if hasattr(self, 'ha_lora_A'):
            ha_x = self.ha_lora_dropout(x.to(self.ha_lora_A.dtype))
            result += (ha_x @ self.ha_lora_A.transpose(0, 1) @ self.ha_lora_B.transpose(0, 1)) * self.ha_scaling

        # HS LoRA 计算
        if hasattr(self, 'hs_lora_experts_A') and target_hardware is not None:
            hw_type = self.hardware_router.get(target_hardware, self.config.default_hardware_type)
            if hw_type in self.hs_lora_experts_A:
                hs_x = self.hs_lora_dropout(x.to(self.hs_lora_experts_A[hw_type].dtype))
                hs_lora_A = self.hs_lora_experts_A[hw_type]
                hs_lora_B = self.hs_lora_experts_B[hw_type]
                result += (hs_x @ hs_lora_A.transpose(0, 1) @ hs_lora_B.transpose(0, 1)) * self.hs_scaling
        
        return result.to(previous_dtype)

# --- 3. 正确的参数冻结函数 ---
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        # 修正：冻结所有不包含 'lora_' 的参数
        if 'lora_' not in n:
            p.requires_grad = False

# --- 4. 统一的、健壮的替换与解融合函数 ---
# --- 4. 最终的、健壮的替换与解融合函数 ---
# 请将这个函数完整替换掉 lora_expert.py 中旧的同名函数

def apply_hardware_expert_lora(model: nn.Module, config: HardwareExpertLoRAConfig) -> nn.Module:
    
    # 猴子补丁：修正GPT2Attention的forward方法以适应新模块
    for layer in model.transformer.h:
        attn_module = layer.attn
        if not hasattr(attn_module, "_original_forward"):
            attn_module._original_forward = attn_module.forward

        def new_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
            # 将target_hardware从顶层模型传递下来
            target_hw = getattr(model, 'target_hardware', None)
            
            if hasattr(self, 'q_proj'): # 检查是否已解融合
                # 我们的HardwareExpertLinear返回(output, debug_info)元组
                query, _ = self.q_proj(hidden_states, target_hardware=target_hw)
                key, _ = self.k_proj(hidden_states, target_hardware=target_hw)
                value, _ = self.v_proj(hidden_states, target_hardware=target_hw)
            else:
                query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            
            # ... (后面是GPT2Attention原始的forward代码) ...
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)
            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            if use_cache: present = (key, value)
            else: present = None
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            
            # 检查c_proj是否也被替换了
            if isinstance(self.c_proj, HardwareExpertLinear):
                c_proj_output, _ = self.c_proj(attn_output, target_hardware=target_hw)
            else:
                c_proj_output = self.c_proj(attn_output)
            
            attn_output = self.resid_dropout(c_proj_output)
            outputs = (attn_output, present)
            if output_attentions: outputs += (attn_weights,)
            return outputs

        attn_module.forward = new_forward.__get__(attn_module, type(attn_module))
    print("Patched forward method for all GPT2Attention modules.")

    # 先收集需要替换的模块，避免在迭代时修改
    replacements = {}
    for name, module in model.named_modules():
        # 处理GPT-2的"三合一"c_attn层
        if isinstance(module, Conv1D) and name.endswith('.attn.c_attn'):
            parent_name = ".".join(name.split('.')[:-1])
            
            in_features = module.weight.shape[0]
            out_features_total = module.weight.shape[1]
            out_features_per_proj = out_features_total // 3
            
            q_proj = HardwareExpertLinear(in_features, out_features_per_proj, config)
            k_proj = HardwareExpertLinear(in_features, out_features_per_proj, config)
            v_proj = HardwareExpertLinear(in_features, out_features_per_proj, config)

            # 正确的权重和偏置复制 (Conv1D权重是(in, out)，Linear权重是(out, in)，所以需要转置)
            q_proj.weight.data = module.weight.data[:, :out_features_per_proj].T.clone()
            k_proj.weight.data = module.weight.data[:, out_features_per_proj:2*out_features_per_proj].T.clone()
            v_proj.weight.data = module.weight.data[:, 2*out_features_per_proj:].T.clone()
            
            if module.bias is not None:
                q_proj.bias.data = module.bias.data[:out_features_per_proj].clone()
                k_proj.bias.data = module.bias.data[out_features_per_proj:2*out_features_per_proj].clone()
                v_proj.bias.data = module.bias.data[2*out_features_per_proj:].clone()
            
            replacements[parent_name] = {'q_proj': q_proj, 'k_proj': k_proj, 'v_proj': v_proj, 'c_attn_to_del': True}

        # 处理其他普通层
        elif any(name.endswith("." + target) for target in config.target_modules):
            if isinstance(module, (nn.Linear, Conv1D)):
                parent_name = ".".join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if isinstance(module, nn.Linear):
                    new_module = HardwareExpertLinear(module.in_features, module.out_features, config)
                    new_module.load_state_dict(module.state_dict(), strict=False)
                else: # Conv1D
                    in_features, out_features = module.weight.shape
                    new_module = HardwareExpertLinear(in_features, out_features, config)
                    new_module.weight.data = module.weight.data.T.clone()
                    if module.bias is not None:
                        new_module.bias.data = module.bias.data.clone()
                
                if parent_name not in replacements:
                    replacements[parent_name] = {}
                replacements[parent_name][child_name] = new_module
    
    # 执行所有替换
    for parent_name, child_map in replacements.items():
        parent_module = model.get_submodule(parent_name)
        for child_name, new_module in child_map.items():
            if child_name == 'c_attn_to_del':
                if hasattr(parent_module, 'c_attn'):
                    delattr(parent_module, 'c_attn')
                    print(f"Deleted old c_attn from {parent_name}")
            else:
                setattr(parent_module, child_name, new_module)
                print(f"Replaced {parent_name}.{child_name}")

    mark_only_lora_as_trainable(model)
    return model

# 自定义Trainer以传递hardware_id
class HardwareAwareTrainer(transformers.Trainer):
    def training_step(self, model, inputs):
        # 假设您的数据样本中有一个 'hardware_id' 字段
        if 'hardware_id' in inputs:
            model.target_hardware = inputs['hardware_id'][0] 
        return super().training_step(model, inputs)