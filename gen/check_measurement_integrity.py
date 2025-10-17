#!/usr/bin/env python3
"""
检查测量结果完整性的脚本

该脚本用于验证gen_eval.json和measured_results.json之间的数据完整性，
确保每个张量句子都被正确测量。

使用方法:
python check_measurement_integrity.py
"""

import json
import os
from collections import defaultdict

# 硬编码的检查路径
CHECK_PATHS = [
    {
        "name": "Jetson Xavier",
        "gen_eval_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/gen_eval.json",
        "measured_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xavier_eval/measured_results.json"
    },
    {
        "name": "V100",
        "gen_eval_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/gen_eval.json",
        "measured_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_v100_eval/measured_results.json"
    },
    {
        "name": "RTX 4090",
        "gen_eval_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/gen_eval_v2.json",
        "measured_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_4090_eval/measured_results_v2.json"
    },
    {
        "name": "Xeon",
        "gen_eval_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/gen_eval.json",
        "measured_path": "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_eval/multi_xeon_eval/measured_results.json"
    }
]

def load_json_lines(file_path):
    """加载JSON行文件，处理空行和格式错误"""
    if not os.path.exists(file_path):
        return []
    
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data = json.loads(line)
                lines.append((line_num, data))
            except json.JSONDecodeError as e:
                print(f"警告: {file_path} 第{line_num}行JSON格式错误: {e}")
                continue
    return lines

def analyze_measurement_status(measured_lines):
    """分析测量状态"""
    stats = {
        'total': len(measured_lines),
        'unmeasured': 0,  # 未测量: r字段为[[0], 0, 0, timestamp]
        'measured_success': 0,  # 测量成功: 有实际延迟值
        'measured_failed': 0,  # 测量失败: 值为1e+10
        'invalid_format': 0  # 格式无效
    }
    
    for line_num, data in measured_lines:
        try:
            if 'r' not in data:
                stats['invalid_format'] += 1
                continue
                
            r_field = data['r']
            if not isinstance(r_field, list) or len(r_field) < 4:
                stats['invalid_format'] += 1
                continue
                
            # 检查是否为未测量状态: [[0], 0, 0, timestamp]
            if (isinstance(r_field[0], list) and len(r_field[0]) == 1 and r_field[0][0] == 0 and
                r_field[1] == 0 and r_field[2] == 0):
                stats['unmeasured'] += 1
            else:
                # 检查测量结果
                if len(r_field) >= 2 and isinstance(r_field[1], (int, float)):
                    if r_field[1] >= 1e9:  # 测量失败（超时或错误）
                        stats['measured_failed'] += 1
                    else:
                        stats['measured_success'] += 1
                else:
                    stats['invalid_format'] += 1
        except Exception as e:
            print(f"警告: 分析第{line_num}行时出错: {e}")
            stats['invalid_format'] += 1
    
    return stats

def check_data_integrity(gen_eval_path, measured_path):
    """检查数据完整性"""
    print(f"\n检查路径:")
    print(f"  原始数据: {gen_eval_path}")
    print(f"  测量结果: {measured_path}")
    
    # 加载数据
    gen_eval_lines = load_json_lines(gen_eval_path)
    measured_lines = load_json_lines(measured_path)
    
    print(f"\n数据统计:")
    print(f"  原始数据行数: {len(gen_eval_lines)}")
    print(f"  测量结果行数: {len(measured_lines)}")
    
    if len(gen_eval_lines) == 0:
        print("  错误: 原始数据文件为空或不存在")
        return False
    
    if len(measured_lines) == 0:
        print("  错误: 测量结果文件为空或不存在")
        return False
    
    # 分析测量状态
    measured_stats = analyze_measurement_status(measured_lines)
    
    print(f"\n测量状态分析:")
    print(f"  总记录数: {measured_stats['total']}")
    print(f"  未测量: {measured_stats['unmeasured']}")
    print(f"  测量成功: {measured_stats['measured_success']}")
    print(f"  测量失败: {measured_stats['measured_failed']}")
    print(f"  格式无效: {measured_stats['invalid_format']}")
    
    # 检查完整性
    missing_count = len(gen_eval_lines) - len(measured_lines)
    if missing_count > 0:
        print(f"\n⚠️  数据不完整:")
        print(f"  缺失记录数: {missing_count}")
        print(f"  缺失比例: {missing_count/len(gen_eval_lines)*100:.1f}%")
        return False
    elif missing_count < 0:
        print(f"\n⚠️  测量结果多于原始数据:")
        print(f"  多余记录数: {-missing_count}")
        return False
    else:
        print(f"\n✅ 数据行数匹配")
    
    # 检查测量完成度
    total_measured = measured_stats['measured_success'] + measured_stats['measured_failed']
    completion_rate = total_measured / len(gen_eval_lines) * 100 if len(gen_eval_lines) > 0 else 0
    
    print(f"\n测量完成度:")
    print(f"  已完成测量: {total_measured}/{len(gen_eval_lines)} ({completion_rate:.1f}%)")
    print(f"  成功率: {measured_stats['measured_success']}/{total_measured} ({measured_stats['measured_success']/total_measured*100:.1f}%)" if total_measured > 0 else "  成功率: N/A")
    
    if completion_rate < 100:
        print(f"  ⚠️  测量未完成，还有 {len(gen_eval_lines) - total_measured} 条记录未测量")
        return False
    else:
        print(f"  ✅ 所有记录都已测量")
        return True

def main():
    """主函数"""
    print("=" * 80)
    print("测量结果完整性检查工具")
    print("=" * 80)
    
    all_passed = True
    
    for check_config in CHECK_PATHS:
        print(f"\n{'='*60}")
        print(f"检查: {check_config['name']}")
        print(f"{'='*60}")
        
        gen_eval_path = check_config['gen_eval_path']
        measured_path = check_config['measured_path']
        
        # 检查文件是否存在
        if not os.path.exists(gen_eval_path):
            print(f"❌ 原始数据文件不存在: {gen_eval_path}")
            all_passed = False
            continue
            
        if not os.path.exists(measured_path):
            print(f"❌ 测量结果文件不存在: {measured_path}")
            all_passed = False
            continue
        
        # 检查数据完整性
        is_complete = check_data_integrity(gen_eval_path, measured_path)
        if not is_complete:
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 所有检查都通过！")
    else:
        print("❌ 发现问题，请检查上述报告")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
