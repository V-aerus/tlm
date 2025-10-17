#!/usr/bin/env python3
"""
为multi硬件重新生成all_tasks.pkl文件的脚本
这个脚本专门处理已经存在的task.pkl文件，不需要创建新的TVM target
"""

import glob
import os
import pickle
from tqdm import tqdm
from common import register_data_path, NETWORK_INFO_FOLDER

def get_all_tasks(network_info_folder):
    """从所有*.task.pkl文件中收集任务并去重"""
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    filenames = glob.glob(f"{network_info_folder}/*.task.pkl")
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

def regenerate_multi_all_tasks():
    """为multi硬件重新生成all_tasks.pkl文件"""
    print("正在为multi硬件重新生成all_tasks.pkl...")
    
    # 直接设置路径，避免依赖register_data_path
    network_info_folder = "/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/network_info/multi"
    
    # 检查目录是否存在
    if not os.path.exists(network_info_folder):
        print(f"错误: 目录 {network_info_folder} 不存在")
        return False
    
    # 检查是否有task.pkl文件
    task_files = glob.glob(f"{network_info_folder}/*.task.pkl")
    if not task_files:
        print(f"错误: 在 {network_info_folder} 中没有找到*.task.pkl文件")
        return False
    
    print(f"找到 {len(task_files)} 个task.pkl文件")
    
    # 删除现有的all_tasks.pkl（如果存在）
    all_tasks_file = f"{network_info_folder}/all_tasks.pkl"
    if os.path.exists(all_tasks_file):
        print(f"删除现有的 {all_tasks_file}")
        os.remove(all_tasks_file)
    
    # 重新生成all_tasks.pkl
    tasks = get_all_tasks(network_info_folder)
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    
    print(f"保存 {len(tasks)} 个任务到 {all_tasks_file}")
    pickle.dump(tasks, open(all_tasks_file, "wb"))
    
    # 检查文件大小
    file_size = os.path.getsize(all_tasks_file)
    print(f"生成的all_tasks.pkl文件大小: {file_size / (1024*1024):.2f} MB")
    
    # 统计不同硬件的任务数量
    llvm_tasks = sum(1 for t in tasks if str(t.target.kind) == "llvm")
    cuda_tasks = sum(1 for t in tasks if str(t.target.kind) == "cuda")
    print(f"LLVM任务数: {llvm_tasks}")
    print(f"CUDA任务数: {cuda_tasks}")
    
    return True

if __name__ == "__main__":
    success = regenerate_multi_all_tasks()
    
    if success:
        print("✅ multi硬件的all_tasks.pkl重新生成成功！")
    else:
        print("❌ multi硬件的all_tasks.pkl重新生成失败！")
        exit(1)
