#!/usr/bin/env python3
"""
测试MoSLoRA训练和PEFT适配器保存功能
"""

import os
import sys
import subprocess

def test_moslora_training():
    """测试MoSLoRA训练"""
    
    # 训练参数
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    train_file = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v5/0_merge.json"
    output_dir = "./test_moslora_training_output"
    
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
        "--per_device_train_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--max_steps", "100",  # 只训练100步进行测试
        "--save_steps", "50",
        "--logging_steps", "10",
        "--learning_rate", "5e-5",
        "--warmup_steps", "10",
        "--block_size", "512",
        "--overwrite_output_dir",
        "--do_train",
        "--device", "cuda:0",  # 明确指定使用单个GPU
    ]
    
    print("开始MoSLoRA训练测试...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10分钟超时
        
        print("训练完成!")
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("标准输出:")
            print(result.stdout[-1000:])  # 只显示最后1000个字符
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr[-1000:])  # 只显示最后1000个字符
        
        # 检查输出目录
        if os.path.exists(output_dir):
            print(f"\n检查输出目录: {output_dir}")
            files = os.listdir(output_dir)
            for file in files:
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file}: {size:,} bytes")
            
            # 特别检查PEFT文件
            adapter_config = os.path.join(output_dir, "adapter_config.json")
            adapter_model = os.path.join(output_dir, "adapter_model.bin")
            
            if os.path.exists(adapter_config):
                print(f"✓ 找到 adapter_config.json")
            else:
                print(f"✗ 未找到 adapter_config.json")
                
            if os.path.exists(adapter_model):
                print(f"✓ 找到 adapter_model.bin")
            else:
                print(f"✗ 未找到 adapter_model.bin")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("训练超时!")
        return False
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_moslora_training()
    if success:
        print("\n✓ MoSLoRA训练测试成功!")
    else:
        print("\n✗ MoSLoRA训练测试失败!")
