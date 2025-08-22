#!/usr/bin/env python3
"""
分析MoSLoRA中的W矩阵（Mixer矩阵）
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
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

def load_moslora_model(adapter_path, base_model_path):
    """加载MoSLoRA模型"""
    print(f"加载基础模型: {base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        print("✓ 基础模型加载成功")
    except Exception as e:
        print(f"✗ 基础模型加载失败: {e}")
        return None
    
    print(f"加载MoSLoRA适配器: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✓ MoSLoRA适配器加载成功")
    except Exception as e:
        print(f"✗ MoSLoRA适配器加载失败: {e}")
        return None
    
    return model

def extract_w_matrices(model):
    """提取所有W矩阵（Mixer矩阵）"""
    w_matrices = {}
    
    print("\n=== 提取W矩阵 ===")
    print("正在遍历模型的所有模块...")
    
    module_count = 0
    lora_modules = 0
    
    for name, module in model.named_modules():
        module_count += 1
        
        # 检查是否是LoRA模块
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules += 1
            print(f"找到LoRA模块: {name}")
            
            # 检查是否有lora_AB（Mixer矩阵）
            if hasattr(module, 'lora_AB') and module.lora_AB is not None:
                try:
                    w_matrix = module.lora_AB.weight.data.cpu().numpy()
                    w_matrices[name] = w_matrix
                    print(f"  ✓ 找到W矩阵: {name}")
                    print(f"    形状: {w_matrix.shape}")
                    print(f"    数据类型: {w_matrix.dtype}")
                    print(f"    数值范围: [{w_matrix.min():.4f}, {w_matrix.max():.4f}]")
                    print(f"    均值: {w_matrix.mean():.4f}")
                    print(f"    标准差: {w_matrix.std():.4f}")
                    print()
                except Exception as e:
                    print(f"  ✗ 提取W矩阵失败: {e}")
            else:
                print(f"  - 没有lora_AB属性或为None")
        
        # 每1000个模块打印一次进度
        if module_count % 1000 == 0:
            print(f"已检查 {module_count} 个模块...")
    
    print(f"\n模块统计:")
    print(f"  总模块数: {module_count}")
    print(f"  LoRA模块数: {lora_modules}")
    print(f"  包含W矩阵的模块数: {len(w_matrices)}")
    
    return w_matrices

def analyze_w_matrix_statistics(w_matrices):
    """分析W矩阵的统计特性"""
    if not w_matrices:
        print("没有找到W矩阵，跳过统计分析")
        return None
        
    print("=== W矩阵统计特性 ===")
    
    all_values = []
    for name, w_matrix in w_matrices.items():
        all_values.extend(w_matrix.flatten())
    
    all_values = np.array(all_values)
    
    print(f"总体统计:")
    print(f"  总参数数量: {len(all_values):,}")
    print(f"  数值范围: [{all_values.min():.4f}, {all_values.max():.4f}]")
    print(f"  均值: {all_values.mean():.4f}")
    print(f"  标准差: {all_values.std():.4f}")
    print(f"  中位数: {np.median(all_values):.4f}")
    print(f"  偏度: {np.mean(((all_values - all_values.mean()) / all_values.std())**3):.4f}")
    print(f"  峰度: {np.mean(((all_values - all_values.mean()) / all_values.std())**4):.4f}")
    
    return all_values

def visualize_w_matrices(w_matrices, save_dir="./w_matrix_analysis"):
    """可视化W矩阵"""
    if not w_matrices:
        print("没有找到W矩阵，跳过可视化")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== 可视化W矩阵 ===")
    print(f"保存到: {save_dir}")
    
    # 1. 热力图
    for i, (name, w_matrix) in enumerate(w_matrices.items()):
        plt.figure(figsize=(10, 8))
        plt.imshow(w_matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='权重值')
        plt.title(f'W矩阵热力图: {name}\n形状: {w_matrix.shape}')
        plt.xlabel('列索引')
        plt.ylabel('行索引')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/w_matrix_heatmap_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. 权重分布直方图
    plt.figure(figsize=(12, 8))
    for name, w_matrix in w_matrices.items():
        plt.hist(w_matrix.flatten(), bins=50, alpha=0.7, label=name, density=True)
    
    plt.xlabel('权重值')
    plt.ylabel('密度')
    plt.title('W矩阵权重分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/w_matrix_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 奇异值分解分析
    plt.figure(figsize=(15, 5))
    
    for i, (name, w_matrix) in enumerate(w_matrices.items()):
        # 计算奇异值
        U, S, Vt = np.linalg.svd(w_matrix)
        
        plt.subplot(1, len(w_matrices), i+1)
        plt.plot(S, 'o-', markersize=4)
        plt.title(f'{name}\n奇异值谱')
        plt.xlabel('奇异值索引')
        plt.ylabel('奇异值大小')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/w_matrix_singular_values.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_w_matrices_to_file(w_matrices, save_dir="./w_matrix_analysis"):
    """保存W矩阵到文件"""
    if not w_matrices:
        print("没有找到W矩阵，跳过文件保存")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== 保存W矩阵到文件 ===")
    
    # 保存为numpy文件
    for name, w_matrix in w_matrices.items():
        safe_name = name.replace('.', '_').replace('/', '_')
        np.save(f'{save_dir}/w_matrix_{safe_name}.npy', w_matrix)
        print(f"保存: w_matrix_{safe_name}.npy")
    
    # 保存统计信息
    stats = {}
    for name, w_matrix in w_matrices.items():
        stats[name] = {
            'shape': w_matrix.shape,
            'min': float(w_matrix.min()),
            'max': float(w_matrix.max()),
            'mean': float(w_matrix.mean()),
            'std': float(w_matrix.std()),
            'norm': float(np.linalg.norm(w_matrix))
        }
    
    import json
    with open(f'{save_dir}/w_matrix_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"保存统计信息: w_matrix_stats.json")

def main():
    """主函数"""
    
    # 配置路径
    adapter_path = "/home/hangshuaihe/tlm/gen/test_direct_training"
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    
    print("=== MoSLoRA W矩阵分析 ===")
    
    # 1. 加载模型
    model = load_moslora_model(adapter_path, base_model_path)
    
    if model is None:
        print("❌ 模型加载失败，退出分析")
        return
    
    # 2. 提取W矩阵
    w_matrices = extract_w_matrices(model)
    
    if not w_matrices:
        print("❌ 未找到W矩阵！可能的原因：")
        print("   - 模型不是MoSLoRA训练的")
        print("   - lora_use_mixer=False")
        print("   - 使用了不同的PEFT实现")
        print("   - W矩阵存储在不同的属性名中")
        
        # 尝试查找其他可能的属性
        print("\n=== 调试信息 ===")
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                print(f"LoRA模块 {name} 的属性:")
                for attr in dir(module):
                    if 'lora' in attr.lower() and not attr.startswith('_'):
                        print(f"  - {attr}: {getattr(module, attr)}")
                break
        return
    
    # 3. 分析统计特性
    all_values = analyze_w_matrix_statistics(w_matrices)
    
    # 4. 可视化
    visualize_w_matrices(w_matrices)
    
    # 5. 保存到文件
    save_w_matrices_to_file(w_matrices)
    
    print(f"\n✅ 分析完成！")
    print(f"找到 {len(w_matrices)} 个W矩阵")
    print(f"结果保存在: ./w_matrix_analysis/")

if __name__ == "__main__":
    main()

