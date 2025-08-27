#!/usr/bin/env python3
"""
LoRA迭代式监督微调脚本
在固定的基础模型上，不断更新LoRA适配器
"""

import os
import sys
import argparse
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# MoSLoRA integration: ensure the local customized PEFT is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    HfArgumentParser
)
from datasets import Dataset
import json

try:
    from peft import PeftModel
    print("✓ Successfully imported custom PEFT library")
except Exception as e:
    print(f"✗ Failed to import custom PEFT library: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    base_model_path: str = field(
        metadata={"help": "基础模型路径（被冻结的模型）"}
    )
    lora_adapter_path: str = field(
        metadata={"help": "要在此基础上继续训练的LoRA适配器路径"}
    )
    model_type: Optional[str] = field(
        default="gpt2",
        metadata={"help": "模型类型"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "模型数据类型",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "低CPU内存使用模式"}
    )
    # 添加defuse_gpt2_attn参数
    defuse_gpt2_attn: bool = field(
        default=False,
        metadata={"help": "Split GPT-2 fused attention (c_attn) into separate q,k,v projections for MoSLoRA compatibility."}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    train_file: str = field(
        metadata={"help": "训练数据文件路径"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件路径"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "输入序列长度，训练数据集将被截断为这个大小的块"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "覆盖缓存的训练和评估集"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用的进程数"}
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "使用TXT文件时是否保留换行符"}
    )

def load_json_data(file_path: str) -> list:
    """加载JSON格式的训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
    return data

def tokenize_function(examples, tokenizer, block_size):
    """对数据进行tokenize处理"""
    # 假设数据格式为 {"text": "..."}
    texts = examples["text"]
    
    # 对文本进行tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=block_size,
        return_overflowing_tokens=False,
    )
    
    return tokenized

def load_and_prepare_model(model_args: ModelArguments):
    """加载并准备模型"""
    logger.info(f"1. 加载基础模型: {model_args.base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    
    # --- 添加defuse_gpt2_attn处理逻辑 ---
    if model_args.defuse_gpt2_attn:
        logger.info("2. 应用defuse_gpt2_attn转换...")
        
        def _convert_conv1d_to_linear(conv1d_module):
            in_features, out_features = conv1d_module.weight.shape
            linear = torch.nn.Linear(in_features, out_features, bias=conv1d_module.bias is not None)
            with torch.no_grad():
                linear.weight.copy_(conv1d_module.weight.T)
                if conv1d_module.bias is not None:
                    linear.bias.copy_(conv1d_module.bias)
            return linear

        class DefusedQKVLinear(torch.nn.Module):
            def __init__(self, conv1d_qkv):
                super().__init__()
                in_features, out_features_total = conv1d_qkv.weight.shape
                assert out_features_total % 3 == 0
                hidden = out_features_total // 3
                # Create three Linear projections
                self.q_proj = torch.nn.Linear(in_features, hidden, bias=conv1d_qkv.bias is not None)
                self.k_proj = torch.nn.Linear(in_features, hidden, bias=conv1d_qkv.bias is not None)
                self.v_proj = torch.nn.Linear(in_features, hidden, bias=conv1d_qkv.bias is not None)
                # Initialize from fused Conv1D
                with torch.no_grad():
                    W = conv1d_qkv.weight  # (in, 3*hidden)
                    b = conv1d_qkv.bias if conv1d_qkv.bias is not None else None
                    self.q_proj.weight.copy_(W[:, 0:hidden].T)
                    self.k_proj.weight.copy_(W[:, hidden:2*hidden].T)
                    self.v_proj.weight.copy_(W[:, 2*hidden:3*hidden].T)
                    if b is not None:
                        self.q_proj.bias.copy_(b[0:hidden])
                        self.k_proj.bias.copy_(b[hidden:2*hidden])
                        self.v_proj.bias.copy_(b[2*hidden:3*hidden])

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                return torch.cat([q, k, v], dim=2)

        # 应用defuse转换
        for name, module in list(base_model.named_modules()):
            # Replace fused qkv
            if name.endswith("attn") and hasattr(module, "c_attn"):
                setattr(module, "c_attn", DefusedQKVLinear(module.c_attn))
            # Replace c_proj to Linear to allow mixer path
            if name.endswith("attn") and hasattr(module, "c_proj"):
                setattr(module, "c_proj", _convert_conv1d_to_linear(module.c_proj))
            # Replace MLP projections to Linear to enable mixer path cleanly
            if name.endswith("mlp"):
                if hasattr(module, "c_fc"):
                    setattr(module, "c_fc", _convert_conv1d_to_linear(module.c_fc))
                if hasattr(module, "c_proj"):
                    setattr(module, "c_proj", _convert_conv1d_to_linear(module.c_proj))
        
        logger.info("✓ defuse_gpt2_attn转换完成")
    
    logger.info(f"3. 加载LoRA适配器: {model_args.lora_adapter_path}")
    
    # 将LoRA适配器加载到基础模型上
    model = PeftModel.from_pretrained(base_model, model_args.lora_adapter_path)
    
    # 确保只有LoRA参数可训练
    for name, param in model.named_parameters():
        if "lora_" in name or "adapter_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    
    return model

def prepare_dataset(data_args: DataArguments, tokenizer):
    """准备训练数据集"""
    logger.info(f"3. 加载训练数据: {data_args.train_file}")
    
    # 加载训练数据
    train_data = load_json_data(data_args.train_file)
    train_dataset = Dataset.from_list(train_data)
    
    # 设置block_size
    block_size = data_args.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(f"模型的最大长度 ({block_size}) 大于1024，设置为1024")
            block_size = 1024
    
    logger.info(f"使用block_size: {block_size}")
    
    # 对数据进行tokenize
    def tokenize_function_wrapper(examples):
        return tokenize_function(examples, tokenizer, block_size)
    
    tokenized_train_dataset = train_dataset.map(
        tokenize_function_wrapper,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )
    
    # 准备验证数据集（如果有）
    eval_dataset = None
    if data_args.validation_file is not None:
        logger.info(f"4. 加载验证数据: {data_args.validation_file}")
        eval_data = load_json_data(data_args.validation_file)
        eval_dataset = Dataset.from_list(eval_data)
        
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function_wrapper,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation data",
        )
        eval_dataset = tokenized_eval_dataset
    
    return tokenized_train_dataset, eval_dataset

def main():
    """主函数"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果传入JSON配置文件
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 如果传入命令行参数
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("=== LoRA迭代式监督微调 ===")
    logger.info(f"基础模型: {model_args.base_model_path}")
    logger.info(f"LoRA适配器: {model_args.lora_adapter_path}")
    logger.info(f"训练数据: {data_args.train_file}")
    logger.info(f"输出路径: {training_args.output_dir}")
    
    # 检查输入路径
    if not os.path.exists(model_args.base_model_path):
        raise ValueError(f"基础模型路径不存在: {model_args.base_model_path}")
    
    if not os.path.exists(model_args.lora_adapter_path):
        raise ValueError(f"LoRA适配器路径不存在: {model_args.lora_adapter_path}")
    
    if not os.path.exists(data_args.train_file):
        raise ValueError(f"训练数据文件不存在: {data_args.train_file}")
    
    # 创建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 加载tokenizer
    logger.info("5. 加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载并准备模型
    model = load_and_prepare_model(model_args)
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(data_args, tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建Trainer
    logger.info("6. 创建Trainer并开始训练")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    train_result = trainer.train()
    
    # 保存训练指标
    trainer.save_metrics("train", train_result.metrics)
    
    # 保存模型（只保存LoRA适配器）
    logger.info(f"7. 保存LoRA适配器到: {training_args.output_dir}")
    
    # 关键步骤：只保存LoRA适配器，不保存基础模型
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 验证保存的文件
    adapter_config = os.path.join(training_args.output_dir, "adapter_config.json")
    adapter_model = os.path.join(training_args.output_dir, "adapter_model.bin")
    
    if os.path.exists(adapter_config) and os.path.exists(adapter_model):
        logger.info("✓ LoRA适配器保存成功!")
        logger.info(f"  - adapter_config.json: {os.path.getsize(adapter_config):,} bytes")
        logger.info(f"  - adapter_model.bin: {os.path.getsize(adapter_model):,} bytes")
    else:
        logger.error("✗ LoRA适配器保存失败!")
    
    logger.info("=== 训练完成 ===")

if __name__ == "__main__":
    main()
