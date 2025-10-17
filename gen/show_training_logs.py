#!/usr/bin/env python3
"""
展示训练日志文件的工具
"""

import os
import glob
from datetime import datetime

def show_latest_log():
    """显示最新的训练日志文件"""
    
    # 查找所有.log文件
    log_files = glob.glob("*.log")
    
    if not log_files:
        print("❌ 没有找到任何.log文件")
        return
    
    # 按修改时间排序，最新的在前
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print("📋 找到的日志文件:")
    for i, log_file in enumerate(log_files):
        mtime = os.path.getmtime(log_file)
        size = os.path.getsize(log_file)
        dt = datetime.fromtimestamp(mtime)
        print(f"  {i+1}. {log_file} ({size:,} bytes, {dt.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 显示最新的日志文件内容
    latest_log = log_files[0]
    print(f"\n📄 显示最新日志文件: {latest_log}")
    print("=" * 80)
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 如果文件太大，只显示最后的部分
            if len(content) > 10000:
                print("... (文件太大，只显示最后10000字符)")
                content = content[-10000:]
            
            print(content)
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")

def show_specific_log(filename):
    """显示指定的日志文件"""
    if not os.path.exists(filename):
        print(f"❌ 日志文件不存在: {filename}")
        return
    
    print(f"📄 显示日志文件: {filename}")
    print("=" * 80)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 如果文件太大，只显示最后的部分
            if len(content) > 10000:
                print("... (文件太大，只显示最后10000字符)")
                content = content[-10000:]
            
            print(content)
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 显示指定的日志文件
        show_specific_log(sys.argv[1])
    else:
        # 显示最新的日志文件
        show_latest_log()
