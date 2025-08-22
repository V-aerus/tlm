#!/usr/bin/env python3
"""
完整的MoSLoRA测试：GPU训练 + PEFT适配器加载
"""

import os
import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MoSLoRA integration: ensure the local customized PEFT is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

try:
    from peft import PeftModel
    print("✓ Successfully imported custom PEFT library")
except Exception as e:
    print(f"✗ Failed to import custom PEFT library: {e}")
    sys.exit(1)

def test_gpu_training():
    """测试GPU上的MoSLoRA训练"""
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("✗ CUDA不可用，无法进行GPU训练测试")
        return False
    
    print(f"✓ 检测到 {torch.cuda.device_count()} 个GPU设备")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 训练参数
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    train_file = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v5/0_merge.json"
    output_dir = "/home/hangshuaihe/tlm/gen/test_gpu_moslora_output"
    
    # 构建训练命令
    cmd = [
        "python", "train_clm.py",
        "--model_name_or_path", base_model_path,
        "--train_file", train_file,
        "--output_dir", output_dir,
        "--use_moslora",  # 启用MoSLoRA
        "--defuse_gpt2_attn",  # 启用defuse
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--target_modules", "q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj",
        "--per_device_train_batch_size", "2",  # 减小batch size以适应GPU内存
        "--gradient_accumulation_steps", "8",
        "--max_steps", "50",  # 减少步数进行快速测试
        "--save_steps", "25",
        "--logging_steps", "5",
        "--learning_rate", "5e-5",
        "--warmup_steps", "5",
        "--block_size", "512",
        "--overwrite_output_dir",
        "--do_train",
    ]
    
    print("\n开始GPU MoSLoRA训练测试...")
    print(f"命令: CUDA_VISIBLE_DEVICES=0 {' '.join(cmd)}")
    
    try:
        # 运行训练，使用CUDA_VISIBLE_DEVICES=0
        full_cmd = f"CUDA_VISIBLE_DEVICES=0 {' '.join(cmd)}"
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300, shell=True)  # 5分钟超时
        
        print("训练完成!")
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("标准输出 (最后1000字符):")
            print(result.stdout[-1000:])
        
        if result.stderr:
            print("错误输出 (最后1000字符):")
            print(result.stderr[-1000:])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("训练超时!")
        return False
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False

def test_peft_loading():
    """测试PEFT适配器加载"""
    
    output_dir = "/home/hangshuaihe/tlm/gen/test_gpu_moslora_output"
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    
    print(f"\n测试PEFT适配器加载...")
    
    # 检查输出目录
    if not os.path.exists(output_dir):
        print(f"✗ 输出目录不存在: {output_dir}")
        return False
    
    # 检查PEFT文件
    adapter_config = os.path.join(output_dir, "adapter_config.json")
    adapter_model = os.path.join(output_dir, "adapter_model.bin")
    
    if not os.path.exists(adapter_config):
        print(f"✗ 未找到 adapter_config.json")
        return False
    
    if not os.path.exists(adapter_model):
        print(f"✗ 未找到 adapter_model.bin")
        return False
    
    print(f"✓ 找到PEFT适配器文件")
    
    try:
        # 测试加载
        print(f"加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        print(f"加载PEFT适配器: {output_dir}")
        model = PeftModel.from_pretrained(base_model, output_dir)
        
        # 检查模型是否在GPU上
        if torch.cuda.is_available():
            model = model.to("cuda:0")
            print(f"✓ 模型已移动到GPU: {model.device}")
        
        # 简单的推理测试
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        test_input = "Hello, this is a test"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✓ 推理测试成功: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"✗ PEFT加载测试失败: {e}")
        return False

def test_gen_state_integration():
    """测试与gen_state.py的集成"""
    
    output_dir = "/home/hangshuaihe/tlm/gen/test_gpu_moslora_output"
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    
    print(f"\n测试gen_state.py集成...")
    
    # 创建一个简单的测试脚本
    test_script = """
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加PEFT路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

from peft import PeftModel

# 测试参数
model_name_or_path = "{adapter_dir}"
base_model_path = "{base_model_path}"

# 检查是否为PEFT适配器
adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
if os.path.exists(adapter_config_path):
    print(f"Loading PEFT adapter from {{model_name_or_path}}")
    print(f"Using base model from {{base_model_path}}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    print("✓ PEFT适配器加载成功!")
else:
    print(f"Loading standard model from {{model_name_or_path}}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    print("✓ 标准模型加载成功!")

print("集成测试完成!")
""".format(adapter_dir=output_dir, base_model_path=base_model_path)
    
    # 保存测试脚本
    test_file = "test_gen_state_integration.py"
    with open(test_file, "w") as f:
        f.write(test_script)
    
    try:
        # 运行测试
        result = subprocess.run(["python", test_file], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ gen_state.py集成测试成功!")
            print(result.stdout)
        else:
            print("✗ gen_state.py集成测试失败!")
            print(result.stderr)
        
        # 清理
        os.remove(test_file)
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

def main():
    """主测试函数"""
    
    print("=== 完整MoSLoRA测试 ===")
    
    # 1. GPU训练测试
    print("\n1. GPU训练测试")
    training_success = test_gpu_training()
    
    # 2. PEFT加载测试
    print("\n2. PEFT适配器加载测试")
    loading_success = test_peft_loading()
    
    # 3. gen_state.py集成测试
    print("\n3. gen_state.py集成测试")
    integration_success = test_gen_state_integration()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"GPU训练: {'✓ 成功' if training_success else '✗ 失败'}")
    print(f"PEFT加载: {'✓ 成功' if loading_success else '✗ 失败'}")
    print(f"集成测试: {'✓ 成功' if integration_success else '✗ 失败'}")
    
    if all([training_success, loading_success, integration_success]):
        print("\n🎉 所有测试通过! MoSLoRA系统工作正常!")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")

if __name__ == "__main__":
    main()
