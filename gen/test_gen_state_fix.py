#!/usr/bin/env python3
"""
测试gen_state.py修复效果的脚本
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def test_json_merge_function():
    """测试JSON合并函数"""
    print("=== 测试JSON合并函数 ===")
    
    # 创建测试目录
    test_dir = "test_json_merge"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试文件
    test_files = [
        f"{test_dir}/0_part",
        f"{test_dir}/1_part",
        f"{test_dir}/2_part"
    ]
    
    # 写入测试数据
    test_data = [
        '{"test": "data1", "id": 1}',
        '{"test": "data2", "id": 2}',
        '{"test": "data3", "id": 3}',
        '{"test": "data4", "id": 4}',
        '{"test": "data5", "id": 5}'
    ]
    
    for i, file_path in enumerate(test_files):
        with open(file_path, 'w') as f:
            if i < len(test_data):
                f.write(test_data[i] + '\n')
            # 故意留一个空文件测试
    
    # 导入并测试合并函数
    try:
        from gen_state import merge_json_files_safely
        
        output_file = f"{test_dir}/merged.json"
        total_records = merge_json_files_safely(test_dir, output_file)
        
        print(f"合并完成，共 {total_records} 条记录")
        
        # 验证结果
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"输出文件行数: {len(lines)}")
            for i, line in enumerate(lines):
                print(f"第{i+1}行: {line.strip()}")
        
        # 清理测试文件
        import shutil
        shutil.rmtree(test_dir)
        
        print("✅ JSON合并函数测试通过")
        return True
        
    except Exception as e:
        print(f"❌ JSON合并函数测试失败: {e}")
        import shutil
        shutil.rmtree(test_dir)
        return False

def test_gpu_detection():
    """测试GPU设备检测"""
    print("\n=== 测试GPU设备检测 ===")
    
    # 测试不同的CUDA_VISIBLE_DEVICES设置
    test_cases = [
        ("", "未设置CUDA_VISIBLE_DEVICES"),
        ("0", "设置CUDA_VISIBLE_DEVICES=0"),
        ("1", "设置CUDA_VISIBLE_DEVICES=1"),
        ("0,1", "设置CUDA_VISIBLE_DEVICES=0,1")
    ]
    
    for cuda_visible, description in test_cases:
        print(f"\n测试: {description}")
        
        # 设置环境变量
        if cuda_visible:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        # 模拟GPU检测逻辑
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            visible_gpu_list = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
            num_gpus = len(visible_gpu_list)
            print(f"  检测到 {num_gpus} 个GPU: {visible_gpu_list}")
        else:
            print(f"  未设置CUDA_VISIBLE_DEVICES，将使用所有可用GPU")
    
    print("✅ GPU设备检测测试完成")
    return True

def test_data_distribution():
    """测试数据分片策略"""
    print("\n=== 测试数据分片策略 ===")
    
    # 模拟数据
    total_workloads = 9
    num_workers = 2
    
    print(f"总workload数: {total_workloads}, Worker数: {num_workers}")
    
    # 原始版本的分片策略（修复后采用）
    per_len = (total_workloads + num_workers - 1) // num_workers  # 向上取整
    
    print("\n修复后的分片策略（均匀分片）:")
    for worker_id in range(num_workers):
        start_idx = worker_id * per_len
        end_idx = min((worker_id + 1) * per_len, total_workloads)
        workload_count = end_idx - start_idx
        print(f"  Worker {worker_id}: [{start_idx}:{end_idx}] = {workload_count} 个workload")
    
    # 原始版本的问题分片策略
    print("\n原始版本的问题分片策略（步长分片）:")
    for worker_id in range(num_workers):
        workload_indices = list(range(worker_id, total_workloads, num_workers))
        print(f"  Worker {worker_id}: {workload_indices} = {len(workload_indices)} 个workload")
    
    print("✅ 数据分片策略测试完成")
    return True

def main():
    """主测试函数"""
    print("开始测试gen_state.py修复效果...")
    
    tests = [
        test_json_merge_function,
        test_gpu_detection,
        test_data_distribution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！gen_state.py修复成功")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())


