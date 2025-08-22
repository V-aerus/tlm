#!/usr/bin/env python3
"""
测试PEFT适配器的保存和加载功能
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MoSLoRA integration: ensure the local customized PEFT is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    print("✓ Successfully imported custom PEFT library")
except Exception as e:
    print(f"✗ Failed to import custom PEFT library: {e}")
    sys.exit(1)

def test_peft_save_load():
    """测试PEFT适配器的保存和加载"""
    
    # 基础模型路径
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    adapter_output_path = "./test_adapter_output"
    
    print(f"1. 加载基础模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"2. 应用MoSLoRA配置")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        lora_use_mixer=True,  # 启用MoSLoRA mixer
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"   可训练参数: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    
    print(f"3. 保存PEFT适配器到: {adapter_output_path}")
    peft_model.save_pretrained(adapter_output_path)
    tokenizer.save_pretrained(adapter_output_path)
    
    # 检查保存的文件
    print(f"4. 检查保存的文件:")
    if os.path.exists(adapter_output_path):
        files = os.listdir(adapter_output_path)
        for file in files:
            file_path = os.path.join(adapter_output_path, file)
            size = os.path.getsize(file_path)
            print(f"   {file}: {size:,} bytes")
    
    print(f"5. 测试加载PEFT适配器")
    # 重新加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # 加载适配器
    loaded_peft_model = PeftModel.from_pretrained(base_model, adapter_output_path)
    
    print(f"6. 验证模型结构")
    # 检查是否成功应用了LoRA
    has_lora = False
    for name, module in loaded_peft_model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            has_lora = True
            print(f"   发现LoRA层: {name}")
            break
    
    if has_lora:
        print("✓ PEFT适配器加载成功!")
    else:
        print("✗ PEFT适配器加载失败!")
    
    # 清理测试文件
    import shutil
    if os.path.exists(adapter_output_path):
        shutil.rmtree(adapter_output_path)
        print(f"7. 清理测试文件: {adapter_output_path}")

if __name__ == "__main__":
    test_peft_save_load()
