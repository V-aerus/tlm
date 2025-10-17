#!/usr/bin/env python3
"""
多进程创建多硬件tokenizer
"""
import json
import os
import multiprocessing as mp
from collections import Counter
from transformers import AutoTokenizer
import time

def process_chunk(args):
    """处理数据块，提取新词汇"""
    chunk_start, chunk_size, input_file = args
    
    hardware_specific_tokens = set()
    
    with open(input_file, 'r') as f:
        # 跳过前面的行
        for _ in range(chunk_start):
            f.readline()
        
        # 处理指定数量的行
        for i in range(chunk_size):
            line = f.readline()
            if not line:
                break
                
            tokens = line.strip().split()
            for token in tokens:
                # 只收集看起来像硬件特定token的词汇
                if any(keyword in token.lower() for keyword in ['sm_', 'arch=', 'mcpu=', 'model=', 'keys=']):
                    hardware_specific_tokens.add(token)
    
    return hardware_specific_tokens

def create_multi_hardware_tokenizer():
    """创建多硬件tokenizer"""
    
    # 配置
    input_file = '/home/hangshuaihe/tlm/tlm_dataset/gen/dataset/to_measure_programs/multi_hardware_simple/all_programs.txt'
    v100_tokenizer_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100'
    save_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_hardware_tokenizer'
    
    print('Loading V100 tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(v100_tokenizer_path)
    original_vocab_size = len(tokenizer)
    print(f'Original vocabulary size: {original_vocab_size}')
    
    # 计算文件总行数
    print('Counting total lines...')
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f'Total lines: {total_lines}')
    
    # 多进程处理
    num_processes = min(16, mp.cpu_count())  # 使用16个进程或CPU核心数
    chunk_size = total_lines // num_processes
    
    print(f'Using {num_processes} processes, chunk size: {chunk_size}')
    
    # 准备参数
    args_list = []
    for i in range(num_processes):
        start = i * chunk_size
        size = chunk_size if i < num_processes - 1 else total_lines - start
        args_list.append((start, size, input_file))
    
    # 多进程处理
    print('Processing data with multiple processes...')
    start_time = time.time()
    
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk, args_list)
    
    # 合并结果
    print('Merging results...')
    all_new_tokens = set()
    for token_set in results:
        all_new_tokens.update(token_set)
    
    processing_time = time.time() - start_time
    print(f'Processing completed in {processing_time:.2f} seconds')
    print(f'Found {len(all_new_tokens)} hardware-specific tokens')
    
    # 添加新token到tokenizer
    new_tokens = list(all_new_tokens)[:10000]  # 限制新token数量
    tokenizer.add_tokens(new_tokens)
    
    print(f'Added {len(new_tokens)} new tokens to tokenizer')
    print(f'New vocabulary size: {len(tokenizer)}')
    
    # 保存扩展后的tokenizer
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    
    print(f'Extended multi-hardware tokenizer saved to {save_path}')
    
    # 测试tokenizer
    test_texts = [
        'cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024',  # RTX 4090
        'llvm -keys=cpu -mcpu=skylake-avx512 -model=xeon',        # Xeon CPU
        'cuda -keys=cuda,gpu -arch=sm_72 -model=xavier'           # Xavier
    ]
    
    print('\nTesting tokenizer:')
    for test_text in test_texts:
        tokens = tokenizer.tokenize(test_text)
        print(f'Test: {test_text[:50]}... -> {tokens[:5]}...')

if __name__ == '__main__':
    create_multi_hardware_tokenizer()
