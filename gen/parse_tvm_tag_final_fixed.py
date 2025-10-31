#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVM硬件配置文件解析器 - 最终修复版本
能够正确解析tag.cc文件中的所有TVM_REGISTER_TARGET_TAG宏
修复了Xavier的kind字段和host字段截断问题
"""

import re
import json
from typing import List, Dict, Any, Union
from pathlib import Path


def parse_tvm_tag_file(file_path: str) -> List[Dict[str, Any]]:
    """
    解析TVM硬件配置文件，提取所有硬件标签和配置信息
    
    Args:
        file_path: tag.cc文件路径
        
    Returns:
        包含硬件信息的字典列表，格式为:
        [{'name': 'nvidia/v100', 'config': {'kind': 'cuda', 'arch': 'sm_70', ...}}, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    hardware_list = []
    
    # 1. 解析直接的TVM_REGISTER_TARGET_TAG调用
    direct_pattern = r'TVM_REGISTER_TARGET_TAG\("([^"]+)"\)'
    direct_matches = re.findall(direct_pattern, content)
    
    print(f"找到 {len(direct_matches)} 个直接的TVM_REGISTER_TARGET_TAG调用")
    
    for name in direct_matches:
        print(f"处理: {name}")
        try:
            config = extract_config_for_hardware_fixed(content, name)
            if config:
                hardware_list.append({
                    'name': name,
                    'config': config,
                    'type': 'direct'
                })
        except Exception as e:
            print(f"警告：解析 {name} 时出错: {e}")
    
    # 2. 解析宏定义的TVM_REGISTER_CUDA_TAG调用
    cuda_macro_pattern = r'TVM_REGISTER_CUDA_TAG\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*(\d+)\);'
    cuda_matches = re.findall(cuda_macro_pattern, content)
    
    print(f"找到 {len(cuda_matches)} 个CUDA宏调用")
    
    for name, arch, shared_mem, reg_per_block in cuda_matches:
        config = {
            'kind': 'cuda',
            'keys': ['cuda', 'gpu'],
            'arch': arch,
            'max_shared_memory_per_block': int(shared_mem),
            'max_threads_per_block': 1024,
            'thread_warp_size': 32,
            'registers_per_block': int(reg_per_block)
        }
        hardware_list.append({
            'name': name,
            'config': config,
            'type': 'cuda_macro'
        })
    
    # 3. 解析AWS C5宏定义
    aws_macro_pattern = r'TVM_REGISTER_TAG_AWS_C5\("([^"]+)",\s*(\d+),\s*"([^"]+)"\);'
    aws_matches = re.findall(aws_macro_pattern, content)
    
    print(f"找到 {len(aws_matches)} 个AWS宏调用")
    
    for name, cores, arch in aws_matches:
        config = {
            'kind': 'llvm',
            'keys': ['x86', 'cpu'],
            'mcpu': arch,
            'num-cores': int(cores)
        }
        hardware_list.append({
            'name': name,
            'config': config,
            'type': 'aws_macro'
        })
    
    # 4. 解析Metal GPU宏定义
    metal_macro_pattern = r'TVM_REGISTER_METAL_GPU_TAG\("([^"]+)",\s*(\d+),\s*(\d+),\s*(\d+)\);'
    metal_matches = re.findall(metal_macro_pattern, content)
    
    print(f"找到 {len(metal_matches)} 个Metal宏调用")
    
    for name, threads_per_block, shared_mem, warp_size in metal_matches:
        config = {
            'kind': 'metal',
            'max_threads_per_block': int(threads_per_block),
            'max_shared_memory_per_block': int(shared_mem),
            'thread_warp_size': int(warp_size),
            'host': {
                'kind': 'llvm',
                'mtriple': 'arm64-apple-macos',
                'mcpu': 'apple-latest'
            }
        }
        hardware_list.append({
            'name': name,
            'config': config,
            'type': 'metal_macro'
        })
    
    return hardware_list


def extract_config_for_hardware_fixed(content: str, hardware_name: str) -> Dict[str, Any]:
    """
    为特定硬件提取完整的配置信息，修复了kind字段和host字段问题
    
    Args:
        content: 文件内容
        hardware_name: 硬件名称
        
    Returns:
        配置字典
    """
    # 找到硬件名称的位置
    pattern = f'TVM_REGISTER_TARGET_TAG\\("{re.escape(hardware_name)}"\\)'
    match = re.search(pattern, content)
    if not match:
        return None
    
    # 从该位置开始查找.set_config
    start_pos = match.end()
    config_start = content.find('.set_config({{', start_pos)
    if config_start == -1:
        return None
    
    # 找到配置的结束位置，需要正确处理嵌套的大括号
    config_start += len('.set_config({{')
    brace_count = 2  # 开始有两个大括号
    config_end = config_start
    
    while config_end < len(content) and brace_count > 0:
        char = content[config_end]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        config_end += 1
    
    if brace_count > 0:
        return None
    
    # 提取配置字符串
    config_str = content[config_start:config_end-1]  # 去掉最后的}}
    
    # 解析配置
    config = parse_config_string_fixed(config_str)
    
    # 修复Xavier的kind字段问题
    if hardware_name == "nvidia/jetson-agx-xavier" and 'arch' in config:
        config['kind'] = 'cuda'
        config['keys'] = ['cuda', 'gpu']
    
    return config


def parse_config_string_fixed(config_str: str) -> Dict[str, Any]:
    """
    修复版本的配置字符串解析函数，能够正确处理复杂的嵌套结构
    
    Args:
        config_str: 配置字符串
        
    Returns:
        配置字典
    """
    config = {}
    
    # 使用递归下降解析器
    i = 0
    while i < len(config_str):
        # 查找下一个键值对的开始
        key_start = config_str.find('{"', i)
        if key_start == -1:
            break
            
        # 找到键的结束位置
        key_end = config_str.find('"', key_start + 2)
        if key_end == -1:
            break
            
        key = config_str[key_start + 2:key_end]
        
        # 找到值的开始位置（跳过逗号和空格）
        value_start = config_str.find(',', key_end) + 1
        while value_start < len(config_str) and config_str[value_start] in ' \t\n':
            value_start += 1
            
        # 找到值的结束位置，需要处理嵌套的大括号
        value_end = find_matching_brace_fixed(config_str, value_start)
        if value_end == -1:
            break
        
        # 提取值字符串
        value_str = config_str[value_start:value_end].strip()
        
        # 解析值
        try:
            parsed_value = parse_value_fixed(value_str)
            config[key] = parsed_value
        except Exception as e:
            print(f"警告：解析键 '{key}' 的值时出错: {e}")
            config[key] = value_str
        
        # 移动到下一个键值对
        i = value_end + 1
    
    return config


def find_matching_brace_fixed(text: str, start: int) -> int:
    """
    找到匹配的大括号位置，正确处理嵌套结构
    
    Args:
        text: 文本
        start: 开始位置
        
    Returns:
        匹配的大括号位置，如果未找到返回-1
    """
    brace_count = 0
    i = start
    in_string = False
    escape_next = False
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            escape_next = False
        elif char == '\\':
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                if brace_count == 0:
                    return i
                brace_count -= 1
            elif char == ',' and brace_count == 0:
                return i
                
        i += 1
    
    return -1


def parse_value_fixed(value_str: str) -> Union[str, int, List[str], Dict[str, Any]]:
    """
    修复版本的值解析函数，能够处理所有类型
    
    Args:
        value_str: 值字符串
        
    Returns:
        解析后的值
    """
    value_str = value_str.strip()
    
    # String类型
    if value_str.startswith('String(') and value_str.endswith(')'):
        return value_str[7:-1].strip('"')
    
    # Integer类型
    elif value_str.startswith('Integer(') and value_str.endswith(')'):
        return int(value_str[8:-1])
    
    # Array<String>类型
    elif value_str.startswith('Array<String>{') and value_str.endswith('}'):
        array_content = value_str[14:-1]
        string_pattern = r'"([^"]*)"'
        strings = re.findall(string_pattern, array_content)
        return strings
    
    # Map<String, ObjectRef>类型 - 递归解析
    elif value_str.startswith('Map<String, ObjectRef>{') and value_str.endswith('}'):
        map_content = value_str[24:-1]
        return parse_config_string_fixed(map_content)
    
    # 直接的数字
    elif value_str.isdigit():
        return int(value_str)
    
    # 直接的字符串（带引号）
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    
    # 其他情况，返回原字符串
    else:
        return value_str


def main():
    """主函数，演示解析器的使用"""
    # 解析tag.cc文件
    tag_file = "tag.cc"
    if not Path(tag_file).exists():
        print(f"错误：找不到文件 {tag_file}")
        return
    
    print("正在解析TVM硬件配置文件...")
    hardware_list = parse_tvm_tag_file(tag_file)
    
    print(f"\n成功解析 {len(hardware_list)} 个硬件配置")
    
    # 统计各类型硬件数量
    type_counts = {}
    for hw in hardware_list:
        hw_type = hw['config'].get('kind', 'unknown')
        type_counts[hw_type] = type_counts.get(hw_type, 0) + 1
    
    print("\n硬件类型统计:")
    for hw_type, count in type_counts.items():
        print(f"  {hw_type}: {count} 个")
    
    # 显示前几个示例
    print("\n前5个硬件配置示例:")
    for i, hw in enumerate(hardware_list[:5]):
        print(f"\n{i+1}. {hw['name']} ({hw['type']})")
        print(f"   类型: {hw['config'].get('kind', 'unknown')}")
        if 'arch' in hw['config']:
            print(f"   架构: {hw['config']['arch']}")
        if 'mcpu' in hw['config']:
            print(f"   CPU: {hw['config']['mcpu']}")
    
    # 保存为JSON文件
    output_file = "tvm_hardware_config_final.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hardware_list, f, indent=2, ensure_ascii=False)
    print(f"\n硬件配置已保存到: {output_file}")
    
    # 查找特定硬件
    print("\n查找特定硬件:")
    v100 = next((hw for hw in hardware_list if hw['name'] == "nvidia/nvidia-v100"), None)
    if v100:
        print(f"V100配置: {json.dumps(v100['config'], indent=2, ensure_ascii=False)}")
    
    xavier = next((hw for hw in hardware_list if hw['name'] == "nvidia/jetson-agx-xavier"), None)
    if xavier:
        print(f"Xavier配置: {json.dumps(xavier['config'], indent=2, ensure_ascii=False)}")
    
    rpi = next((hw for hw in hardware_list if hw['name'] == "raspberry-pi/4b-aarch64"), None)
    if rpi:
        print(f"Raspberry Pi配置: {json.dumps(rpi['config'], indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()






