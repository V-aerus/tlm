#!/usr/bin/env python3
"""
重新生成all_tasks.pkl文件的脚本
只需要删除现有的all_tasks.pkl，然后运行这个脚本即可
"""

import glob
import os
import pickle
from tqdm import tqdm
from common import register_data_path, NETWORK_INFO_FOLDER

def get_all_tasks():
    """从所有*.task.pkl文件中收集任务并去重"""
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    filenames = glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")
    filenames.sort()

    print(f"找到 {len(filenames)} 个task.pkl文件")
    
    for filename in tqdm(filenames, desc="处理task文件"):
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            task_key = (t.workload_key, str(t.target.kind))

            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    print(f"去重前任务数: {len(all_tasks) + duplication}")
    print(f"去重后任务数: {len(all_tasks)}")
    print(f"重复任务数: {duplication}")
    
    return all_tasks

def regenerate_all_tasks(target_str):
    """重新生成指定硬件的all_tasks.pkl文件"""
    print(f"正在为硬件 '{target_str}' 重新生成all_tasks.pkl...")
    
    # 注册数据路径
    register_data_path(target_str)
    
    # 检查目录是否存在
    if not os.path.exists(NETWORK_INFO_FOLDER):
        print(f"错误: 目录 {NETWORK_INFO_FOLDER} 不存在")
        return False
    
    # 检查是否有task.pkl文件
    task_files = glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")
    if not task_files:
        print(f"错误: 在 {NETWORK_INFO_FOLDER} 中没有找到*.task.pkl文件")
        return False
    
    # 删除现有的all_tasks.pkl（如果存在）
    all_tasks_file = f"{NETWORK_INFO_FOLDER}/all_tasks.pkl"
    if os.path.exists(all_tasks_file):
        print(f"删除现有的 {all_tasks_file}")
        os.remove(all_tasks_file)
    
    # 重新生成all_tasks.pkl
    tasks = get_all_tasks()
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    
    print(f"保存 {len(tasks)} 个任务到 {all_tasks_file}")
    pickle.dump(tasks, open(all_tasks_file, "wb"))
    
    # 检查文件大小
    file_size = os.path.getsize(all_tasks_file)
    print(f"生成的all_tasks.pkl文件大小: {file_size / (1024*1024):.2f} MB")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python regenerate_all_tasks.py <target>")
        print("例如: python regenerate_all_tasks.py 'llvm -mcpu=skylake-avx512 -model=xeon'")
        print("例如: python regenerate_all_tasks.py 'cuda -arch=sm_86 -model=4090'")
        sys.exit(1)
    
    target = sys.argv[1]
    success = regenerate_all_tasks(target)
    
    if success:
        print("✅ all_tasks.pkl 重新生成成功！")
    else:
        print("❌ all_tasks.pkl 重新生成失败！")
        sys.exit(1)













