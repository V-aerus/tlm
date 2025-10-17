#!/usr/bin/env python3
"""
测试修复后的measure_programs.py脚本

该脚本用于验证修复后的measure_programs.py是否能正确处理数据完整性
"""

import subprocess
import os
import json

def test_measure_programs_fix():
    """测试修复后的measure_programs.py"""
    
    # 测试配置
    test_configs = [
        {
            "name": "Jetson Xavier",
            "target": "nvidia/jetson-agx-xavier",
            "to_measure_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/gen_eval.json",
            "measured_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/measured_results.json",
            "batch_size": 64
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {config['name']}")
        print(f"{'='*60}")
        
        # 检查文件是否存在
        if not os.path.exists(config['to_measure_path']):
            print(f"❌ 原始数据文件不存在: {config['to_measure_path']}")
            continue
            
        # 运行修复后的measure_programs.py（仅检查，不实际测量）
        cmd = [
            "python", "measure_programs.py",
            "--target", config['target'],
            "--to-measure-path", config['to_measure_path'],
            "--measured-path", config['measured_path'],
            "--batch-size", str(config['batch_size'])
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        print("注意：这只是检查逻辑，不会实际进行测量")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            # 运行命令
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
            
            print(f"返回码: {result.returncode}")
            if result.stdout:
                print("标准输出:")
                print(result.stdout)
            if result.stderr:
                print("错误输出:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("❌ 命令执行超时")
        except Exception as e:
            print(f"❌ 执行出错: {e}")

def analyze_before_after():
    """分析修复前后的差异"""
    print(f"\n{'='*60}")
    print("修复前后对比分析")
    print(f"{'='*60}")
    
    # 运行检查脚本
    try:
        result = subprocess.run(["python", "check_measurement_integrity.py"], 
                              capture_output=True, text=True, timeout=60)
        print("检查结果:")
        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 检查脚本执行出错: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("measure_programs.py 修复验证工具")
    print("=" * 80)
    
    # 分析当前状态
    analyze_before_after()
    
    # 测试修复后的逻辑
    test_measure_programs_fix()
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

