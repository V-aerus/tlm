#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件特征向量生成器 V2.0
根据MT_MoSLoRA_embedding.md中的规范，生成扁平化的纯数值特征向量
修复了归一化、估算不准确、向量过长等问题
"""

import json
from typing import Dict, List, Any, Union
from pathlib import Path


class HardwareEmbeddingGeneratorV2:
    """硬件特征向量生成器 V2.0"""
    
    def __init__(self):
        """初始化特征编码规则"""
        # 1. 平台与类别 - One-Hot编码
        self.kind_categories = ['llvm', 'cuda', 'metal']

        # 2. 设备类型 - 设备级互斥 One-Hot
        self.keys_categories = ['cpu', 'gpu']

        # 3. 厂商 - One-Hot
        self.vendor_categories = ['nvidia', 'intel', 'amd', 'apple', 'arm']

        # 4. CPU 架构家族 - One-Hot（仅对 CPU 有意义）
        self.cpu_arch_family_categories = ['x86', 'aarch64']

        # 5. 指令集属性 - Multi-Hot编码
        self.mattr_categories = ['avx2', 'avx512', 'neon']

        # 6. 数值特征名称（对数/比值化后的派生特征，固定顺序）
        self.numeric_feature_names = [
            'log_cu', 'log_mtpb', 'log_warp', 'log_flops_g', 'log_bw_gbs', 'log_shm_kb',
            'log_gflops_per_cu', 'log_bw_per_cu', 'log_regs_per_thread',
            'log_l1_per_cu', 'log_l2_per_cu', 'log_l3_per_cu', 'log_roofline_ai'
        ]

        # 7. 计算向量维度（+1 为 arch_cc_or_gen 数值）
        self.vector_dim = (
            len(self.kind_categories) +
            len(self.keys_categories) +
            len(self.vendor_categories) +
            len(self.cpu_arch_family_categories) +
            len(self.mattr_categories) +
            1 +  # arch_cc_or_gen
            len(self.numeric_feature_names)
        )
        
        # 硬件规格数据库（真实数据，覆盖主要硬件）
        self.hardware_specs = {
            'nvidia/nvidia-v100': {
                'compute_unit_count': 80,  # SM count
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 15700,
                'max_shared_memory_per_block': 48,  # KB
                'registers_per_block': 65536,
                'l1_cache_size_kb': 10240,  # 128 KB/SM * 80
                'l2_cache_size_kb': 6144,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 900
            },
            'nvidia/nvidia-a100': {
                'compute_unit_count': 108,  # SM count
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 19580,
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 13824,  # 128 KB/SM * 108
                'l2_cache_size_kb': 40960,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 1555
            },
            'nvidia/nvidia-t4': {
                'compute_unit_count': 40,
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 8100,
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 5120,  # 128 KB/SM * 40
                'l2_cache_size_kb': 6144,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 300
            },
            'nvidia/tesla-k80': {
                'compute_unit_count': 13,  # per GPU (K80 dual, 但TVM视作single)
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 4113,  # per GPU
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 208,  # 16 KB/SM * 13
                'l2_cache_size_kb': 1536,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 240  # per GPU
            },
            'nvidia/jetson-agx-xavier': {
                'compute_unit_count': 8,  # SM
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 1410,
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 1024,  # 128 KB/SM * 8
                'l2_cache_size_kb': 512,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 137
            },
            'nvidia/jetson-nano': {
                'compute_unit_count': 1,  # SM
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 472,
                'max_shared_memory_per_block': 48,
                'registers_per_block': 32768,
                'l1_cache_size_kb': 128,  # 128 KB/SM * 1
                'l2_cache_size_kb': 128,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 25
            },
            'raspberry-pi/4b-aarch64': {
                'compute_unit_count': 4,  # cores
                'max_threads_per_block': 1,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 48,  # 4*1.5GHz*8 FLOPS/cycle (NEON)
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 128,  # 32 KB data/core * 4
                'l2_cache_size_kb': 1024,  # 1 MB shared
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 13
            },
            'apple/m1-gpu': {
                'compute_unit_count': 8,
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 2600,
                'max_shared_memory_per_block': 32,  # 32768 bytes -> 32 KB
                'registers_per_block': 0,  # 不适用
                'l1_cache_size_kb': 8,  # 估算
                'l2_cache_size_kb': 768,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 68
            },
            'apple/m2-gpu': {
                'compute_unit_count': 10,
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 3500,
                'max_shared_memory_per_block': 32,
                'registers_per_block': 0,
                'l1_cache_size_kb': 10,
                'l2_cache_size_kb': 960,
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 100
            },
            'aws/cpu/c5.large': {
                'compute_unit_count': 1,  # 实际2 vCPU，但num-cores=1
                'max_threads_per_block': 2,  # HT
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 192,  # 估算: 2 vCPU * 3GHz * 32 FLOPS/cycle (AVX512)
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 64,  # 32K data + 32K inst per core
                'l2_cache_size_kb': 1024,  # 1MB per core
                'l3_cache_size_kb': 25344,  # ~25MB shared
                'memory_bandwidth_gb_s': 128  # DDR4-2666估算
            },
            'aws/cpu/c5.xlarge': {
                'compute_unit_count': 2,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 384,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 128,
                'l2_cache_size_kb': 2048,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.2xlarge': {
                'compute_unit_count': 4,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 768,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 256,
                'l2_cache_size_kb': 4096,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.4xlarge': {
                'compute_unit_count': 8,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 1536,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 512,
                'l2_cache_size_kb': 8192,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.9xlarge': {
                'compute_unit_count': 18,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 3456,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 1152,
                'l2_cache_size_kb': 18432,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.12xlarge': {
                'compute_unit_count': 24,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 4608,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 1536,
                'l2_cache_size_kb': 24576,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.18xlarge': {
                'compute_unit_count': 36,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 6912,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 2304,
                'l2_cache_size_kb': 36864,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            'aws/cpu/c5.24xlarge': {
                'compute_unit_count': 48,
                'max_threads_per_block': 2,
                'thread_warp_size': 1,
                'peak_fp32_flops_g': 9216,
                'max_shared_memory_per_block': 0,
                'registers_per_block': 0,
                'l1_cache_size_kb': 3072,
                'l2_cache_size_kb': 49152,
                'l3_cache_size_kb': 25344,
                'memory_bandwidth_gb_s': 128
            },
            # 额外补充的常见GPU规格（参考公开资料，便于避免走估算分支）
            'nvidia/rtx-4090': {
                'compute_unit_count': 128,   # SM count (16384 CUDA cores / 128 per SM)
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 82600,  # ~82.6 TFLOPS
                'max_shared_memory_per_block': 48,  # KB
                'registers_per_block': 65536,
                'l1_cache_size_kb': 16384,   # 128 KB/SM * 128 SM
                'l2_cache_size_kb': 73728,   # 72 MB
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 1008 # 21 Gbps * 384-bit / 8
            },
            'nvidia/geforce-rtx-3090': {
                'compute_unit_count': 82,    # 10496 CUDA cores / 128 per SM ≈ 82
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 35600,  # ~35.6 TFLOPS
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 10496,   # 128 KB/SM * 82 SM
                'l2_cache_size_kb': 6144,    # 6 MB
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 936  # 19.5 Gbps * 384-bit / 8
            },
            'nvidia/geforce-rtx-3080': {
                'compute_unit_count': 68,    # 8704 CUDA cores / 128 per SM = 68
                'max_threads_per_block': 1024,
                'thread_warp_size': 32,
                'peak_fp32_flops_g': 29800,  # ~29.8 TFLOPS
                'max_shared_memory_per_block': 48,
                'registers_per_block': 65536,
                'l1_cache_size_kb': 8704,    # 128 KB/SM * 68 SM
                'l2_cache_size_kb': 5120,    # 5 MB
                'l3_cache_size_kb': 0,
                'memory_bandwidth_gb_s': 760  # 19 Gbps * 320-bit / 8
            }
        }

        # 基于TVM tag.cc导出的配置自动扩充hardware_specs（估算补全）
        self._maybe_expand_specs_from_tvm_config("tvm_hardware_config_final.json")
    
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
        vector = [0.0] * self.vector_dim
        offset = 0
        
        # 1. 平台与类别 - One-Hot编码
        kind = config.get('kind', 'unknown')
        if kind in self.kind_categories:
            kind_idx = self.kind_categories.index(kind)
            vector[offset + kind_idx] = 1.0
        offset += len(self.kind_categories)
        
        # 2. 设备类型 - 设备级互斥 One-Hot
        # CPU: llvm 视为 CPU；GPU: cuda/metal 视为 GPU
        if kind == 'llvm':
            vector[offset + self.keys_categories.index('cpu')] = 1.0
        elif kind in ('cuda', 'metal'):
            vector[offset + self.keys_categories.index('gpu')] = 1.0
        offset += len(self.keys_categories)

        # 3. 厂商 - One-Hot + 架构数值（arch_cc_or_gen）
        vendor, arch_cc_or_gen = self._parse_vendor_and_cc(name, config)
        if vendor in self.vendor_categories:
            vector[offset + self.vendor_categories.index(vendor)] = 1.0
        offset += len(self.vendor_categories)

        # 4. CPU 架构家族 - 仅在 CPU 时生效
        cpu_arch_family = self._infer_cpu_arch_family(config)
        if kind == 'llvm' and cpu_arch_family in self.cpu_arch_family_categories:
            vector[offset + self.cpu_arch_family_categories.index(cpu_arch_family)] = 1.0
        offset += len(self.cpu_arch_family_categories)

        # 5. 指令集属性 - Multi-Hot编码
        mattr = config.get('mattr', [])
        for attr in mattr:
            # 处理"+neon"格式的属性
            clean_attr = attr.replace('+', '').replace('-', '')
            if clean_attr in self.mattr_categories:
                mattr_idx = self.mattr_categories.index(clean_attr)
                vector[offset + mattr_idx] = 1.0
        offset += len(self.mattr_categories)
        
        # 6. 架构数值（arch_cc_or_gen）
        vector[offset] = round(float(arch_cc_or_gen), 4)
        offset += 1

        # 7. 数值特征 - 对数/比值化派生数值
        raw_features = self._extract_numerical_features(config, name)
        processed = self._postprocess_features(raw_features)
        for i, feat_name in enumerate(self.numeric_feature_names):
            value = float(processed.get(feat_name, 0.0))
            vector[offset + i] = round(value, 6)
        
        return {
            'hardware_name': name,
            'vector': vector
        }

    def _maybe_expand_specs_from_tvm_config(self, config_file: str) -> None:
        """如果存在由tag.cc解析得到的TVM配置文件，则基于其中的字段对hardware_specs进行自动扩充。
        - 优先保留self.hardware_specs中的“真实规格”；
        - 对缺失的硬件，使用现有估算逻辑生成一份静态规格，便于后续复用与稳定输出；
        """
        try:
            cfg_path = Path(config_file)
            if not cfg_path.exists():
                return
            with open(cfg_path, 'r', encoding='utf-8') as f:
                hardware_list = json.load(f)
        except Exception:
            return

        added = 0
        for item in hardware_list:
            name = item.get('name')
            config = item.get('config', {})
            if not name or not isinstance(config, dict):
                continue
            if name in self.hardware_specs:
                continue
            try:
                # 使用内部估算逻辑构建一份稳定的规格表项
                feats = self._extract_numerical_features(config, name)
                spec = {
                    'compute_unit_count': int(feats.get('compute_unit_count', 1.0)),
                    'max_threads_per_block': int(feats.get('max_threads_per_block', 1.0)),
                    'thread_warp_size': int(feats.get('thread_warp_size', 1.0)),
                    'peak_fp32_flops_g': float(feats.get('peak_fp32_flops_g', 100.0)),
                    'max_shared_memory_per_block': int(feats.get('max_shared_memory_per_block', 0.0)),  # KB
                    'registers_per_block': int(feats.get('registers_per_block', 0.0)),
                    'l1_cache_size_kb': int(feats.get('l1_cache_size_kb', 0.0)),
                    'l2_cache_size_kb': int(feats.get('l2_cache_size_kb', 0.0)),
                    'l3_cache_size_kb': int(feats.get('l3_cache_size_kb', 0.0)),
                    'memory_bandwidth_gb_s': float(feats.get('memory_bandwidth_gb_s', 100.0)),
                }
                self.hardware_specs[name] = spec
                added += 1
            except Exception:
                continue
        if added:
            print(f"已基于TVM配置自动扩充 {added} 个hardware_specs 项（估算值）。")
    
    def _extract_arch(self, config: Dict[str, Any]) -> str:
        """提取架构信息"""
        # 优先使用arch字段（GPU）
        if 'arch' in config:
            return config['arch']
        
        # 其次使用mcpu字段（CPU）
        if 'mcpu' in config:
            return config['mcpu']
        
        return 'unknown'

    def _parse_vendor_and_cc(self, name: str, config: Dict[str, Any]) -> (str, float):
        """解析厂商与架构数值（GPU用CC，CPU/Metal用代际索引）。返回 (vendor, arch_cc_or_gen[0-1])"""
        lname = str(name).lower()
        vendor = 'unknown'
        if 'nvidia' in lname:
            vendor = 'nvidia'
        elif 'intel' in lname or 'xeon' in lname or 'i7' in lname:
            vendor = 'intel'
        elif 'apple' in lname or config.get('kind') == 'metal':
            vendor = 'apple'
        elif 'raspberry' in lname or 'arm' in lname or 'carmel' in str(config.get('mcpu', '')):
            vendor = 'arm'

        kind = config.get('kind', '')
        arch_cc_or_gen = 0.5
        if kind == 'cuda':
            arch = str(config.get('arch', ''))
            if arch.startswith('sm_'):
                try:
                    cc_int = int(arch.split('_')[1])  # 86
                    arch_cc_or_gen = max(0.0, min(1.0, cc_int / 100.0))
                except Exception:
                    arch_cc_or_gen = 0.70  # 缺省靠近Volta
            else:
                arch_cc_or_gen = 0.70
        elif kind == 'llvm':
            # CPU 代际索引（可按需扩展）
            uarch = str(config.get('mcpu', '')).lower()
            cpu_gen_map = {
                'skylake-avx512': 0.60,
                'cascadelake': 0.62,
                'icelake-server': 0.65,
                'sapphirerapids': 0.72,
                'apple-latest': 0.80,
                'carmel': 0.55,
            }
            arch_cc_or_gen = cpu_gen_map.get(uarch, 0.50)
        elif kind == 'metal':
            arch_cc_or_gen = 0.80  # Apple GPU 近似

        return vendor, float(arch_cc_or_gen)

    def _infer_cpu_arch_family(self, config: Dict[str, Any]) -> str:
        """推断CPU架构家族：x86 或 aarch64。非CPU返回'unknown'。"""
        kind = config.get('kind', '')
        if kind != 'llvm':
            return 'unknown'
        mtriple = str(config.get('mtriple', '')).lower()
        mcpu = str(config.get('mcpu', '')).lower()
        keys = [str(k).lower() for k in config.get('keys', [])]
        if 'aarch64' in mtriple or 'aarch64' in keys:
            return 'aarch64'
        if 'x86' in mtriple or 'x86_64' in mtriple or 'skylake' in mcpu or 'cascadelake' in mcpu or 'icelake' in mcpu:
            return 'x86'
        if 'x86' in keys:
            return 'x86'
        return 'unknown'

    def _log1p(self, x: float) -> float:
        import math
        try:
            return float(math.log1p(max(0.0, float(x))))
        except Exception:
            return 0.0

    def _postprocess_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        """对原始数值做对数与比值化，返回固定名称字典。"""
        cu = max(float(raw.get('compute_unit_count', 1.0)), 1.0)
        mtpb = max(float(raw.get('max_threads_per_block', 1.0)), 1.0)
        flops = float(raw.get('peak_fp32_flops_g', 0.0))
        bw = float(raw.get('memory_bandwidth_gb_s', 0.0))
        shm_kb = float(raw.get('max_shared_memory_per_block', 0.0))
        regs = float(raw.get('registers_per_block', 0.0))
        l1 = float(raw.get('l1_cache_size_kb', 0.0))
        l2 = float(raw.get('l2_cache_size_kb', 0.0))
        l3 = float(raw.get('l3_cache_size_kb', 0.0))

        out = {}
        out['log_cu'] = self._log1p(cu)
        out['log_mtpb'] = self._log1p(mtpb)
        out['log_warp'] = self._log1p(float(raw.get('thread_warp_size', 1.0)))
        out['log_flops_g'] = self._log1p(flops)
        out['log_bw_gbs'] = self._log1p(bw)
        out['log_shm_kb'] = self._log1p(shm_kb)

        out['log_gflops_per_cu'] = self._log1p(flops / cu)
        out['log_bw_per_cu'] = self._log1p(bw / cu)
        out['log_regs_per_thread'] = self._log1p(regs / mtpb)
        out['log_l1_per_cu'] = self._log1p(l1 / cu)
        out['log_l2_per_cu'] = self._log1p(l2 / cu)
        out['log_l3_per_cu'] = self._log1p(l3 / cu)
        out['log_roofline_ai'] = self._log1p(flops / max(bw, 1e-6))

        return out
    
    def _extract_numerical_features(self, config: Dict[str, Any], hardware_name: str) -> Dict[str, float]:
        """提取数值特征"""
        # 首先检查是否有预定义的规格数据
        if hardware_name in self.hardware_specs:
            return self.hardware_specs[hardware_name].copy()
        
        # 否则使用估算逻辑
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
        missing_specs = []
        
        for hardware in hardware_list:
            try:
                embedding = self.create_hardware_embedding(hardware)
                embeddings.append(embedding)
                
                # 记录缺失规格的硬件
                if hardware['name'] not in self.hardware_specs:
                    missing_specs.append(hardware['name'])
                    
            except Exception as e:
                print(f"警告：处理硬件 {hardware.get('name', 'unknown')} 时出错: {e}")
                continue
        
        if missing_specs:
            print(f"\n注意：以下 {len(missing_specs)} 个硬件使用了估算值，建议添加真实规格数据：")
            for name in missing_specs[:10]:  # 只显示前10个
                print(f"  - {name}")
            if len(missing_specs) > 10:
                print(f"  ... 还有 {len(missing_specs) - 10} 个")
        
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
                'vendor': len(self.vendor_categories),
                'cpu_arch_family': len(self.cpu_arch_family_categories),
                'mattr': len(self.mattr_categories),
                'arch_cc_or_gen': 1,
                'numerical': len(self.numeric_feature_names)
            },
            'categories': {
                'kind': self.kind_categories,
                'keys': self.keys_categories,
                'vendor': self.vendor_categories,
                'cpu_arch_family': self.cpu_arch_family_categories,
                'mattr': self.mattr_categories,
                'numeric_feature_names': self.numeric_feature_names
            },
            'hardware_specs_count': len(self.hardware_specs)
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
    generator = HardwareEmbeddingGeneratorV2()
    
    # 显示向量信息
    vector_info = generator.get_vector_info()
    print(f"\n特征向量维度: {vector_info['vector_dimension']}")
    print("特征分解:")
    for feature, dim in vector_info['feature_breakdown'].items():
        print(f"  {feature}: {dim}")
    print(f"预定义规格数据: {vector_info['hardware_specs_count']} 个硬件")
    
    # 生成特征向量
    print("\n正在生成硬件特征向量...")
    embeddings = generator.process_hardware_list(hardware_list)
    
    print(f"成功生成 {len(embeddings)} 个硬件特征向量")
    
    # 显示几个示例
    print("\n前3个硬件特征向量示例:")
    for i, embedding in enumerate(embeddings[:3]):
        print(f"\n{i+1}. {embedding['hardware_name']}")
        print(f"   向量维度: {len(embedding['vector'])}")
        print(f"   向量: {embedding['vector']}")
        print(f"   非零特征数: {sum(1 for x in embedding['vector'] if x > 0)}")
    
    # 查找特定硬件
    print("\n查找特定硬件:")
    v100 = next((emb for emb in embeddings if 'v100' in emb['hardware_name'].lower()), None)
    if v100:
        print(f"V100特征向量: {v100['vector']}")
    
    xavier = next((emb for emb in embeddings if 'xavier' in emb['hardware_name'].lower()), None)
    if xavier:
        print(f"Xavier特征向量: {xavier['vector']}")
    
    rpi = next((emb for emb in embeddings if 'raspberry-pi' in emb['hardware_name'].lower()), None)
    if rpi:
        print(f"Raspberry Pi特征向量: {rpi['vector']}")
    
    # 保存结果
    output_file = "hardware_embeddings_v2.json"
    generator.save_embeddings(embeddings, output_file)
    
    # 保存向量信息
    info_file = "vector_info_v2.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(vector_info, f, indent=2, ensure_ascii=False)
    print(f"向量信息已保存到: {info_file}")


if __name__ == "__main__":
    main()
