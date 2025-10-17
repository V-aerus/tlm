#!/usr/bin/env python3
"""
比较多硬件tokenizer和V100 tokenizer的差异
"""
from transformers import AutoTokenizer
import json

def compare_tokenizers():
    """比较两个tokenizer的差异"""
    
    # 加载两个tokenizer - 对比原始V100和新创建的官方V100
    original_v100_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100'
    official_v100_path = '/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100_v1test'
    
    print("Loading tokenizers...")
    original_tokenizer = AutoTokenizer.from_pretrained(original_v100_path)
    official_tokenizer = AutoTokenizer.from_pretrained(official_v100_path)
    
    print(f"原始V100 tokenizer词汇量: {len(original_tokenizer)}")
    print(f"官方V100 tokenizer词汇量: {len(official_tokenizer)}")
    print(f"差异: {len(official_tokenizer) - len(original_tokenizer)} tokens")
    
    # 获取词汇表
    original_vocab = set(original_tokenizer.get_vocab().keys())
    official_vocab = set(official_tokenizer.get_vocab().keys())
    
    # 找出差异
    only_in_official = official_vocab - original_vocab
    only_in_original = original_vocab - official_vocab
    
    print(f"\n=== 只在官方V100 tokenizer中的token ({len(only_in_official)}个) ===")
    for token in sorted(only_in_official)[:50]:  # 只显示前50个
        token_id = official_tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} (ID: {token_id})")
    if len(only_in_official) > 50:
        print(f"  ... 还有 {len(only_in_official) - 50} 个token")
    
    print(f"\n=== 只在原始V100 tokenizer中的token ({len(only_in_original)}个) ===")
    for token in sorted(only_in_original)[:50]:  # 只显示前50个
        token_id = original_tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} (ID: {token_id})")
    if len(only_in_original) > 50:
        print(f"  ... 还有 {len(only_in_original) - 50} 个token")
    
    # 测试V100特定文本
    print(f"\n=== V100特定文本测试 ===")
    test_texts = [
        "cuda -keys=cuda,gpu -arch=sm_70 -max_num_threads=1024",  # V100
        "conv2d_NCHWc",  # 常见算子
        "auto_scheduler",  # TVM组件
        "data_vec",  # 数据向量化
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # 原始V100 tokenizer
        original_tokens = original_tokenizer.tokenize(text)
        original_unk_count = original_tokens.count('[UNK]')
        print(f"  原始V100: {original_tokens[:8]}... (UNK: {original_unk_count})")
        
        # 官方V100 tokenizer
        official_tokens = official_tokenizer.tokenize(text)
        official_unk_count = official_tokens.count('[UNK]')
        print(f"  官方V100: {official_tokens[:8]}... (UNK: {official_unk_count})")
        
        # 分析差异
        if official_unk_count < original_unk_count:
            print(f"  ✅ 官方更好: UNK数量从{original_unk_count}减少到{official_unk_count}")
        elif official_unk_count == original_unk_count:
            print(f"  ➖ 相同: UNK数量都是{official_unk_count}")
        else:
            print(f"  ❌ 原始更好: UNK数量从{original_unk_count}增加到{official_unk_count}")
    
    # 分析token类型差异
    print(f"\n=== Token类型差异分析 ===")
    
    # 分析只在官方V100中的token
    algorithm_tokens = []
    numeric_tokens = []
    other_tokens = []
    
    for token in list(only_in_official)[:100]:  # 分析前100个
        if any(alg in token.lower() for alg in ['conv', 'matmul', 'pool', 'batch', 'layer']):
            algorithm_tokens.append(token)
        elif token.isdigit() or any(c.isdigit() for c in token):
            numeric_tokens.append(token)
        else:
            other_tokens.append(token)
    
    print(f"算法相关token: {algorithm_tokens[:10]}...")
    print(f"数值相关token: {numeric_tokens[:10]}...")
    print(f"其他token: {other_tokens[:10]}...")
    
    # 保存比较结果
    comparison_result = {
        'original_v100_vocab_size': len(original_tokenizer),
        'official_v100_vocab_size': len(official_tokenizer),
        'only_in_official': list(only_in_official),
        'only_in_original': list(only_in_original),
        'algorithm_tokens': algorithm_tokens,
        'numeric_tokens': numeric_tokens,
        'other_tokens': other_tokens
    }
    
    with open('v100_tokenizer_comparison.json', 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    print(f"\nV100 tokenizer比较结果已保存到 v100_tokenizer_comparison.json")

if __name__ == '__main__':
    compare_tokenizers()
