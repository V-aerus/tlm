#!/usr/bin/env python3
"""
创建多硬件预训练数据集
"""
import json
import os
import glob
import random
from multiprocessing import Pool
import multiprocessing as mp
from transformers import AutoTokenizer
from datasets import Dataset
import time

def process_file_chunk(args):
    """处理文件块，提取程序文本"""
    files, tokenizer_path = args
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    all_texts = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    if line.strip():
                        record = json.loads(line)
                        if 'i' in record and len(record['i']) > 0:
                            # 提取程序文本
                            program_text = record['i'][0][0] if len(record['i'][0]) > 0 else ''
                            if program_text and len(program_text) > 10:  # 过滤太短的文本
                                all_texts.append(program_text)
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
    
    return all_texts

def create_multi_hardware_dataset():
    """创建多硬件预训练数据集"""
    
    # 配置
    tokenizer_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_hardware_tokenizer'
    save_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_hardware_pretrain'
    
    # 收集所有硬件的数据文件
    hardware_dirs = [
        '/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/v100',
        '/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/4090', 
        '/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/xavier',
        '/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/xeon'
    ]
    
    all_files = []
    for hw_dir in hardware_dirs:
        if os.path.exists(hw_dir):
            files = glob.glob(os.path.join(hw_dir, '*.json'))
            # 采样一半数据
            sampled_files = random.sample(files, len(files) // 2)
            all_files.extend(sampled_files)
            print(f'Added {len(sampled_files)} files from {os.path.basename(hw_dir)}')
    
    print(f'Total files to process: {len(all_files)}')
    
    # 多进程处理
    num_processes = min(16, mp.cpu_count())
    chunk_size = len(all_files) // num_processes
    
    print(f'Using {num_processes} processes, chunk size: {chunk_size}')
    
    # 准备参数
    args_list = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(all_files)
        chunk_files = all_files[start:end]
        args_list.append((chunk_files, tokenizer_path))
    
    # 多进程处理
    print('Processing files with multiple processes...')
    start_time = time.time()
    
    with Pool(num_processes) as pool:
        results = pool.map(process_file_chunk, args_list)
    
    # 合并结果
    print('Merging results...')
    all_texts = []
    for texts in results:
        all_texts.extend(texts)
    
    processing_time = time.time() - start_time
    print(f'Processing completed in {processing_time:.2f} seconds')
    print(f'Total texts collected: {len(all_texts)}')
    
    # 创建数据集
    print('Creating dataset...')
    dataset = Dataset.from_dict({'text': all_texts})
    
    # 保存数据集
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    
    print(f'Multi-hardware pretrain dataset saved to {save_path}')
    print(f'Dataset size: {len(dataset)}')
    
    # 显示一些样本
    print('\nSample texts:')
    for i, sample in enumerate(dataset.select(range(3))):
        print(f'Sample {i+1}: {sample["text"][:100]}...')

if __name__ == '__main__':
    random.seed(42)
    create_multi_hardware_dataset()
