#!/usr/bin/env python3
"""
测试标准LoRA配置
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# MoSLoRA integration
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

from peft import LoraConfig, get_peft_model

def test_standard_lora():
    """测试标准LoRA配置"""
    
    print("=== 测试标准LoRA配置 ===")
    
    # 加载基础模型
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    config = AutoConfig.from_pretrained(base_model_path)
    
    print(f"模型类型: {getattr(config, 'model_type', 'unknown')}")
    
    # 检查c_attn层的结构
    for name, module in model.named_modules():
        if name.endswith("c_attn"):
            print(f"找到c_attn层: {name}")
            print(f"  类型: {type(module)}")
            print(f"  权重形状: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  偏置形状: {module.bias.shape}")
            break
    
    # 测试标准LoRA配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        enable_lora=[True, True, True]  # 为GPT-2的融合层启用LoRA
    )
    
    print(f"\nLoRA配置: {lora_config}")
    
    try:
        peft_model = get_peft_model(model, lora_config)
        print("✅ 标准LoRA配置成功")
        
        # 检查可训练参数
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"可训练参数: {trainable}/{total} ({100.0 * trainable / total:.2f}%)")
        
    except Exception as e:
        print(f"❌ 标准LoRA配置失败: {e}")
        import traceback
        traceback.print_exc()

def test_moslora_config():
    """测试MoSLoRA配置"""
    
    print("\n=== 测试MoSLoRA配置 ===")
    
    # 加载基础模型
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    config = AutoConfig.from_pretrained(base_model_path)
    
    # 测试MoSLoRA配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "attn.c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        lora_use_mixer=True  # 启用MoSLoRA mixer
    )
    
    print(f"LoRA配置: {lora_config}")
    
    try:
        peft_model = get_peft_model(model, lora_config)
        print("✅ MoSLoRA配置成功")
        
        # 检查可训练参数
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"可训练参数: {trainable}/{total} ({100.0 * trainable / total:.2f}%)")
        
    except Exception as e:
        print(f"❌ MoSLoRA配置失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_standard_lora()
    test_moslora_config()
