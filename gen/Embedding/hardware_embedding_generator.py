#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件特征向量生成器
根据MT_MoSLoRA_embedding.md中的规范，将硬件配置转换为统一的特征向量
"""

import json
import numpy as np
from typing import Dict, List, Any, Union
from pathlib import Path


class HardwareEmbeddingGenerator:
    """硬件特征向量生成器"""
    
    def __init__(self):
        """初始化特征编码规则"""
        # 1. 平台与类别 - One-Hot编码
        self.kind_categories = ['llvm', 'cuda', 'metal']
        
        # 2. 通用标签 - Multi-Hot编码
        self.keys_categories = ['cpu', 'gpu']
        
        # 3. 微架构 - One-Hot编码
        self.arch_categories = [
            'icelake-server', 'cortex-a72', 'sm_70', 'sm_72', 'sm_86',
            'skylake-avx512', 'cascadelake', 'carmel', 'apple-latest',
            'sm_37', 'sm_35', 'sm_20', 'sm_80', 'sm_75', 'sm_61', 'sm_52',
            'sm_50', 'sm_30', 'sm_21', 'sm_53', 'sm_62'
        ]
        
        # 4. 指令集属性 - Multi-Hot编码
        self.mattr_categories = ['avx2', 'avx512', 'neon']
        
        # 计算向量维度
        self.vector_dim = (
            len(self.kind_categories) +      # kind: 3
            len(self.keys_categories) +      # keys: 2
            len(self.arch_categories) +      # arch: 21
            len(self.mattr_categories) +     # mattr: 3
            10  # 数值特征: compute_unit_count, max_threads_per_block, thread_warp_size,
                # peak_fp32_flops_g, max_shared_memory_per_block, registers_per_block,
                # l1_cache_size_kb, l2_cache_size_kb, l3_cache_size_kb, memory_bandwidth_gb_s
        )
        
        # 数值特征的归一化范围（用于Min-Max归一化）
        self.normalization_ranges = {
            'compute_unit_count': (1, 80),           # 1-80个核心/SM
            'max_threads_per_block': (1, 1024),      # 1-1024个线程
            'thread_warp_size': (1, 32),             # 1-32个线程
            'peak_fp32_flops_g': (100, 20000),       # 100-20000 GFLOPS
            'max_shared_memory_per_block': (0, 50),  # 0-50 KB
            'registers_per_block': (0, 65536),       # 0-65536个寄存器
            'l1_cache_size_kb': (0, 20000),          # 0-20000 KB
            'l2_cache_size_kb': (0, 100000),         # 0-100000 KB
            'l3_cache_size_kb': (0, 100000),         # 0-100000 KB
            'memory_bandwidth_gb_s': (50, 1000)      # 50-1000 GB/s
        }
    
    def create_hardware_embedding(self, hardware_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个硬件配置创建特征向量
        
        Args:
            hardware_config: 硬件配置字典，格式为:
                {'name': 'nvidia/v100', 'config': {'kind': 'cuda', 'arch': 'sm_70', ...}}
        
        Returns:
            包含硬件名称和特征向量的字典
        """
        config = hardware_config['config']
        name = hardware_config['name']
        
        # 初始化特征向量
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        offset = 0
        
        # 1. 平台与类别 - One-Hot编码
        kind = config.get('kind', 'unknown')
        if kind in self.kind_categories:
            kind_idx = self.kind_categories.index(kind)
            vector[offset + kind_idx] = 1.0
        offset += len(self.kind_categories)
        
        # 2. 通用标签 - Multi-Hot编码
        keys = config.get('keys', [])
        for key in keys:
            if key in self.keys_categories:
                key_idx = self.keys_categories.index(key)
                vector[offset + key_idx] = 1.0
        offset += len(self.keys_categories)
        
        # 3. 微架构 - One-Hot编码
        arch = self._extract_arch(config)
        if arch in self.arch_categories:
            arch_idx = self.arch_categories.index(arch)
            vector[offset + arch_idx] = 1.0
        offset += len(self.arch_categories)
        
        # 4. 指令集属性 - Multi-Hot编码
        mattr = config.get('mattr', [])
        for attr in mattr:
            # 处理"+neon"格式的属性
            clean_attr = attr.replace('+', '').replace('-', '')
            if clean_attr in self.mattr_categories:
                mattr_idx = self.mattr_categories.index(clean_attr)
                vector[offset + mattr_idx] = 1.0
        offset += len(self.mattr_categories)
        
        # 5. 数值特征 - 归一化处理
        numerical_features = self._extract_numerical_features(config)
        for i, (feature_name, value) in enumerate(numerical_features.items()):
            if feature_name in self.normalization_ranges:
                min_val, max_val = self.normalization_ranges[feature_name]
                # Min-Max归一化
                normalized_value = (value - min_val) / (max_val - min_val)
                vector[offset + i] = np.clip(normalized_value, 0.0, 1.0)
        
        return {
            'hardware_name': name,
            'vector': vector.tolist(),
            'config': config
        }
    
    def _extract_arch(self, config: Dict[str, Any]) -> str:
        """提取架构信息"""
        # 优先使用arch字段（GPU）
        if 'arch' in config:
            return config['arch']
        
        # 其次使用mcpu字段（CPU）
        if 'mcpu' in config:
            return config['mcpu']
        
        return 'unknown'
    
    def _extract_numerical_features(self, config: Dict[str, Any]) -> Dict[str, float]:
        """提取数值特征"""
        features = {}
        
        # 计算单元数量
        if 'num-cores' in config:
            features['compute_unit_count'] = float(config['num-cores'])
        elif 'arch' in config and config.get('kind') == 'cuda':
            # 对于CUDA GPU，根据架构估算SM数量
            features['compute_unit_count'] = self._estimate_sm_count(config['arch'])
        else:
            features['compute_unit_count'] = 1.0
        
        # 每块最大线程数
        features['max_threads_per_block'] = float(config.get('max_threads_per_block', 1))
        
        # Warp大小
        features['thread_warp_size'] = float(config.get('thread_warp_size', 1))
        
        # 峰值FP32算力（GFLOPS）
        features['peak_fp32_flops_g'] = self._estimate_peak_flops(config)
        
        # 每块最大共享内存（KB）
        shared_mem_bytes = config.get('max_shared_memory_per_block', 0)
        features['max_shared_memory_per_block'] = shared_mem_bytes / 1024.0
        
        # 每块寄存器数量
        features['registers_per_block'] = float(config.get('registers_per_block', 0))
        
        # 缓存大小（KB）
        features['l1_cache_size_kb'] = self._estimate_l1_cache(config)
        features['l2_cache_size_kb'] = self._estimate_l2_cache(config)
        features['l3_cache_size_kb'] = self._estimate_l3_cache(config)
        
        # 内存带宽（GB/s）
        features['memory_bandwidth_gb_s'] = self._estimate_memory_bandwidth(config)
        
        return features
    
    def _estimate_sm_count(self, arch: str) -> float:
        """根据GPU架构估算SM数量"""
        sm_counts = {
            'sm_20': 8, 'sm_21': 8, 'sm_30': 8, 'sm_35': 8, 'sm_37': 8,
            'sm_50': 16, 'sm_52': 16, 'sm_53': 16, 'sm_60': 20, 'sm_61': 20,
            'sm_62': 20, 'sm_70': 80, 'sm_72': 64, 'sm_75': 20, 'sm_80': 108,
            'sm_86': 84
        }
        return float(sm_counts.get(arch, 8))
    
    def _estimate_peak_flops(self, config: Dict[str, Any]) -> float:
        """估算峰值FP32算力"""
        kind = config.get('kind', '')
        arch = config.get('arch', '')
        mcpu = config.get('mcpu', '')
        
        if kind == 'cuda':
            # GPU算力估算（基于架构和SM数量）
            sm_count = self._estimate_sm_count(arch)
            base_flops_per_sm = {
                'sm_20': 200, 'sm_21': 200, 'sm_30': 200, 'sm_35': 200, 'sm_37': 200,
                'sm_50': 300, 'sm_52': 300, 'sm_53': 300, 'sm_60': 400, 'sm_61': 400,
                'sm_62': 400, 'sm_70': 500, 'sm_72': 500, 'sm_75': 400, 'sm_80': 600,
                'sm_86': 500
            }
            return sm_count * base_flops_per_sm.get(arch, 200)
        
        elif kind == 'llvm':
            # CPU算力估算
            cores = config.get('num-cores', 1)
            mattr = config.get('mattr', [])
            
            # 基础频率假设
            base_freq = 2.4  # GHz
            
            # 根据指令集调整
            if 'avx512' in str(mattr):
                flops_per_cycle = 32  # AVX512 FP32
            elif 'avx2' in str(mattr):
                flops_per_cycle = 16  # AVX2 FP32
            elif 'neon' in str(mattr):
                flops_per_cycle = 8   # NEON FP32
            else:
                flops_per_cycle = 4   # 基础FP32
            
            return cores * base_freq * flops_per_cycle * 2  # FMA指令
        
        elif kind == 'metal':
            # Metal GPU算力估算
            return 1000.0  # 假设1 TFLOPS
        
        return 100.0  # 默认值
    
    def _estimate_l1_cache(self, config: Dict[str, Any]) -> float:
        """估算L1缓存大小"""
        kind = config.get('kind', '')
        
        if kind == 'cuda':
            # GPU L1缓存（每个SM约128KB）
            arch = config.get('arch', '')
            sm_count = self._estimate_sm_count(arch)
            return sm_count * 128.0
        
        elif kind == 'llvm':
            # CPU L1缓存（每个核心约48KB）
            cores = config.get('num-cores', 1)
            return cores * 48.0
        
        return 0.0
    
    def _estimate_l2_cache(self, config: Dict[str, Any]) -> float:
        """估算L2缓存大小"""
        kind = config.get('kind', '')
        
        if kind == 'cuda':
            # GPU L2缓存（通常6MB）
            return 6144.0
        
        elif kind == 'llvm':
            # CPU L2缓存（每个核心约1.25MB）
            cores = config.get('num-cores', 1)
            return cores * 1280.0
        
        return 0.0
    
    def _estimate_l3_cache(self, config: Dict[str, Any]) -> float:
        """估算L3缓存大小"""
        kind = config.get('kind', '')
        
        if kind == 'llvm':
            # CPU L3缓存（共享，约1.5MB每核心）
            cores = config.get('num-cores', 1)
            return cores * 1536.0
        
        # GPU通常没有L3缓存
        return 0.0
    
    def _estimate_memory_bandwidth(self, config: Dict[str, Any]) -> float:
        """估算内存带宽"""
        kind = config.get('kind', '')
        
        if kind == 'cuda':
            # GPU内存带宽（基于架构）
            arch = config.get('arch', '')
            bandwidth_map = {
                'sm_20': 200, 'sm_21': 200, 'sm_30': 200, 'sm_35': 200, 'sm_37': 200,
                'sm_50': 300, 'sm_52': 300, 'sm_53': 300, 'sm_60': 400, 'sm_61': 400,
                'sm_62': 400, 'sm_70': 900, 'sm_72': 800, 'sm_75': 400, 'sm_80': 900,
                'sm_86': 500
            }
            return float(bandwidth_map.get(arch, 200))
        
        elif kind == 'llvm':
            # CPU内存带宽（基于核心数）
            cores = config.get('num-cores', 1)
            return cores * 25.6  # 假设每核心25.6 GB/s
        
        elif kind == 'metal':
            # Metal GPU内存带宽
            return 400.0
        
        return 100.0
    
    def process_hardware_list(self, hardware_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理硬件配置列表
        
        Args:
            hardware_list: 硬件配置列表
            
        Returns:
            包含特征向量的硬件列表
        """
        embeddings = []
        
        for hardware in hardware_list:
            try:
                embedding = self.create_hardware_embedding(hardware)
                embeddings.append(embedding)
            except Exception as e:
                print(f"警告：处理硬件 {hardware.get('name', 'unknown')} 时出错: {e}")
                continue
        
        return embeddings
    
    def save_embeddings(self, embeddings: List[Dict[str, Any]], output_file: str):
        """保存特征向量到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)
        print(f"硬件特征向量已保存到: {output_file}")
    
    def get_vector_info(self) -> Dict[str, Any]:
        """获取向量信息"""
        return {
            'vector_dimension': self.vector_dim,
            'feature_breakdown': {
                'kind': len(self.kind_categories),
                'keys': len(self.keys_categories),
                'arch': len(self.arch_categories),
                'mattr': len(self.mattr_categories),
                'numerical': 10
            },
            'categories': {
                'kind': self.kind_categories,
                'keys': self.keys_categories,
                'arch': self.arch_categories,
                'mattr': self.mattr_categories
            }
        }


def main():
    """主函数，演示硬件特征向量生成器的使用"""
    # 加载硬件配置
    config_file = "tvm_hardware_config_final.json"
    if not Path(config_file).exists():
        print(f"错误：找不到文件 {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        hardware_list = json.load(f)
    
    print(f"加载了 {len(hardware_list)} 个硬件配置")
    
    # 创建特征向量生成器
    generator = HardwareEmbeddingGenerator()
    
    # 显示向量信息
    vector_info = generator.get_vector_info()
    print(f"\n特征向量维度: {vector_info['vector_dimension']}")
    print("特征分解:")
    for feature, dim in vector_info['feature_breakdown'].items():
        print(f"  {feature}: {dim}")
    
    # 生成特征向量
    print("\n正在生成硬件特征向量...")
    embeddings = generator.process_hardware_list(hardware_list)
    
    print(f"成功生成 {len(embeddings)} 个硬件特征向量")
    
    # 显示几个示例
    print("\n前3个硬件特征向量示例:")
    for i, embedding in enumerate(embeddings[:3]):
        print(f"\n{i+1}. {embedding['hardware_name']}")
        print(f"   向量维度: {len(embedding['vector'])}")
        print(f"   向量前10维: {embedding['vector'][:10]}")
        print(f"   非零特征数: {sum(1 for x in embedding['vector'] if x > 0)}")
    
    # 查找特定硬件
    print("\n查找特定硬件:")
    v100 = next((emb for emb in embeddings if 'v100' in emb['hardware_name'].lower()), None)
    if v100:
        print(f"V100特征向量: {v100['vector'][:15]}...")
    
    xavier = next((emb for emb in embeddings if 'xavier' in emb['hardware_name'].lower()), None)
    if xavier:
        print(f"Xavier特征向量: {xavier['vector'][:15]}...")
    
    # 保存结果
    output_file = "hardware_embeddings.json"
    generator.save_embeddings(embeddings, output_file)
    
    # 保存向量信息
    info_file = "vector_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(vector_info, f, indent=2, ensure_ascii=False)
    print(f"向量信息已保存到: {info_file}")


if __name__ == "__main__":
    main()
