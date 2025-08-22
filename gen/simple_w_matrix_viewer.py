#!/usr/bin/env python3
"""
简化的W矩阵查看器
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM

# MoSLoRA integration
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

from peft import PeftModel

def main():
    """主函数"""
    
    # 配置路径
    adapter_path = "/home/hangshuaihe/tlm/gen/test_direct_training"
    base_model_path = "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v4"
    
    print("=== MoSLoRA W矩阵详细分析 ===")
    
    # 加载模型
    print("加载模型...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 提取W矩阵
    w_matrices = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_AB') and module.lora_AB is not None:
            w_matrix = module.lora_AB.weight.data.cpu().numpy()
            w_matrices[name] = w_matrix
    
    print(f"\n找到 {len(w_matrices)} 个W矩阵")
    
    # 分析每个W矩阵
    for i, (name, w_matrix) in enumerate(w_matrices.items()):
        print(f"\n--- W矩阵 {i+1}: {name} ---")
        print(f"形状: {w_matrix.shape}")
        print(f"数值范围: [{w_matrix.min():.6f}, {w_matrix.max():.6f}]")
        print(f"均值: {w_matrix.mean():.6f}")
        print(f"标准差: {w_matrix.std():.6f}")
        print(f"L2范数: {np.linalg.norm(w_matrix):.6f}")
        
        # 显示矩阵的前几行和列
        print("矩阵内容 (前4x4):")
        print(w_matrix[:4, :4])
        
        # 奇异值分析
        U, S, Vt = np.linalg.svd(w_matrix)
        print(f"奇异值: {S[:5]}")  # 显示前5个奇异值
        print(f"条件数: {S[0] / S[-1]:.2f}")
        
        # 检查是否接近正交
        orthogonality = np.abs(U.T @ U - np.eye(U.shape[1])).max()
        print(f"正交性误差: {orthogonality:.6f}")
        
        # 检查是否接近单位矩阵
        identity_error = np.abs(w_matrix - np.eye(w_matrix.shape[0])).max()
        print(f"单位矩阵误差: {identity_error:.6f}")
    
    # 总体统计
    print(f"\n=== 总体统计 ===")
    all_values = np.concatenate([w.flatten() for w in w_matrices.values()])
    print(f"总参数数量: {len(all_values):,}")
    print(f"全局数值范围: [{all_values.min():.6f}, {all_values.max():.6f}]")
    print(f"全局均值: {all_values.mean():.6f}")
    print(f"全局标准差: {all_values.std():.6f}")
    
    # 检查W矩阵的分布特征
    print(f"\n=== 分布特征 ===")
    print(f"接近0的参数比例: {np.mean(np.abs(all_values) < 0.01):.2%}")
    print(f"接近±0.25的参数比例: {np.mean(np.abs(all_values) > 0.24):.2%}")
    
    # 检查W矩阵的对称性
    print(f"\n=== 对称性分析 ===")
    for name, w_matrix in w_matrices.items():
        symmetry_error = np.abs(w_matrix - w_matrix.T).max()
        print(f"{name}: 对称性误差 = {symmetry_error:.6f}")

if __name__ == "__main__":
    main()
