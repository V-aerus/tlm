#!/usr/bin/env python3
"""
MT-MoSLoRA Iterative Training Script
支持从已有的HA/HS适配器继续训练的迭代训练脚本

使用方法：
python train_mt_moslora_iterative.py \
    --base_model_path /path/to/base/model \
    --ha_adapter_path /path/to/ha_adapter.bin \
    --hs_adapter_paths /path/to/hs_v100_adapter.bin,/path/to/hs_xavier_adapter.bin \
    --output_dir /path/to/output \
    --train_file /path/to/train_data.json
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, List
import json

import datasets
from datasets import load_dataset
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM, GPT2LMHeadModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
    is_torch_tpu_available,
    set_seed,
    EarlyStoppingCallback
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

# MoSLoRA integration: ensure the local customized PEFT is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

try:
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as MoSLoRALinear
except Exception:
    LoraConfig = None
    get_peft_model = None
    MoSLoRALinear = None

# 导入MT-MoSLoRA相关类
from train_mt_moslora import MTMoSLoRALinear, HardwareAwareTrainer, extract_hardware_id_function

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class IterativeModelArguments:
    """
    Arguments for iterative MT-MoSLoRA training
    """
    base_model_path: str = field(
        metadata={"help": "Path to the base model (e.g., V4 model)"}
    )
    ha_adapter_path: str = field(
        metadata={"help": "Path to the HA adapter file (ha_adapter.bin)"}
    )
    hs_adapter_paths: str = field(
        metadata={"help": "Comma-separated paths to HS adapter files (hs_v100_adapter.bin,hs_xavier_adapter.bin,hs_i7_adapter.bin)"}
    )
    adapter_config_path: str = field(
        metadata={"help": "Path to the adapter config file (adapter_config.json)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class IterativeDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: str = field(metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


def load_mt_moslora_adapters(model: nn.Module, adapter_config_path: str, ha_adapter_path: str, hs_adapter_paths: List[str]):
    """
    加载MT-MoSLoRA适配器到模型中
    """
    import torch
    
    # 加载适配器配置
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    logger.info(f"Loading MT-MoSLoRA adapters with config: {adapter_config}")
    
    # 加载HA适配器
    ha_adapters = torch.load(ha_adapter_path, map_location='cpu')
    logger.info(f"Loaded HA adapter with {len(ha_adapters)} modules")
    
    # 加载HS适配器
    hs_adapters = {}
    for hs_path in hs_adapter_paths:
        if os.path.exists(hs_path):
            hw_type = os.path.basename(hs_path).replace('hs_', '').replace('_adapter.bin', '')
            hs_adapters[hw_type] = torch.load(hs_path, map_location='cpu')
            logger.info(f"Loaded HS {hw_type} adapter with {len(hs_adapters[hw_type])} modules")
    
    # 应用适配器到模型
    for name, module in model.named_modules():
        if isinstance(module, MTMoSLoRALinear):
            # 加载HA适配器
            if name in ha_adapters:
                ha_data = ha_adapters[name]
                if ha_data['lora_A'] is not None:
                    module.ha_moslora.lora_A.load_state_dict(ha_data['lora_A'])
                if ha_data['lora_B'] is not None:
                    module.ha_moslora.lora_B.load_state_dict(ha_data['lora_B'])
                if ha_data['lora_AB'] is not None and hasattr(module.ha_moslora, 'lora_AB'):
                    module.ha_moslora.lora_AB.load_state_dict(ha_data['lora_AB'])
                logger.info(f"Loaded HA adapter for {name}")
            
            # 加载HS适配器
            for hw_type, hs_data in hs_adapters.items():
                if hw_type in module.hs_experts and name in hs_data:
                    hs_module_data = hs_data[name]
                    if hs_module_data['lora_A'] is not None:
                        module.hs_experts[hw_type].lora_A.load_state_dict(hs_module_data['lora_A'])
                    if hs_module_data['lora_B'] is not None:
                        module.hs_experts[hw_type].lora_B.load_state_dict(hs_module_data['lora_B'])
                    if hs_module_data['lora_AB'] is not None and hasattr(module.hs_experts[hw_type], 'lora_AB'):
                        module.hs_experts[hw_type].lora_AB.load_state_dict(hs_module_data['lora_AB'])
                    logger.info(f"Loaded HS {hw_type} adapter for {name}")
    
    return adapter_config


def main():
    # 解析参数
    parser = HfArgumentParser((IterativeModelArguments, IterativeDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 设置随机种子
    set_seed(training_args.seed)
    
    # 加载分词器
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_path, use_fast=model_args.use_fast_tokenizer)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path,
        from_tf=bool(".ckpt" in model_args.base_model_path),
        cache_dir=model_args.cache_dir,
        use_auth_token=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # 调整模型大小
    model.resize_token_embeddings(len(tokenizer))

    # 解析HS适配器路径
    hs_adapter_paths = [p.strip() for p in model_args.hs_adapter_paths.split(",") if p.strip()]
    
    # 首先需要将基础模型转换为MT-MoSLoRA架构
    logger.info("Converting base model to MT-MoSLoRA architecture...")
    from train_mt_moslora import apply_mt_moslora_to_model, ModelArguments
    
    # 从adapter_config.json中读取配置
    with open(model_args.adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    # 创建ModelArguments对象用于模型转换
    mt_model_args = ModelArguments(
        use_mt_moslora=True,
        use_mixer=adapter_config['ha_config']['use_mixer'],
        ha_lora_r=adapter_config['ha_config']['r'],
        ha_lora_alpha=adapter_config['ha_config']['alpha'],
        ha_lora_dropout=adapter_config['ha_config']['dropout'],
        hs_lora_r=adapter_config['hs_config']['r'],
        hs_lora_alpha=adapter_config['hs_config']['alpha'],
        hs_lora_dropout=adapter_config['hs_config']['dropout'],
        hardware_types=','.join(adapter_config['hardware_types']),
        target_modules=','.join(adapter_config['target_modules']),
        defuse_gpt2_attn=True  # 添加GPT-2解融合
    )
    
    # 先进行GPT-2解融合（如果需要）
    is_gpt2 = getattr(model.config, "model_type", None) == "gpt2"
    if mt_model_args.defuse_gpt2_attn and is_gpt2:
        logger.info("Defusing GPT-2 fused attention layers...")
        
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

        for name, module in list(model.named_modules()):
            # Replace fused qkv
            if name.endswith("attn") and hasattr(module, "c_attn"):
                setattr(module, "c_attn", DefusedQKVLinear(module.c_attn))
            # Replace c_proj to Linear
            if name.endswith("attn") and hasattr(module, "c_proj"):
                setattr(module, "c_proj", _convert_conv1d_to_linear(module.c_proj))
            # Replace MLP projections to Linear
            if name.endswith("mlp"):
                if hasattr(module, "c_fc"):
                    setattr(module, "c_fc", _convert_conv1d_to_linear(module.c_fc))
                if hasattr(module, "c_proj"):
                    setattr(module, "c_proj", _convert_conv1d_to_linear(module.c_proj))
        
        logger.info("GPT-2 defusion completed")
    
    # 转换模型架构
    model = apply_mt_moslora_to_model(model, mt_model_args)
    logger.info("Model converted to MT-MoSLoRA architecture")
    
    # 验证MT-MoSLoRA模块是否正确创建
    mt_moslora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, MTMoSLoRALinear):
            mt_moslora_count += 1
    logger.info(f"MT-MoSLoRA modules created: {mt_moslora_count}")
    if mt_moslora_count == 0:
        logger.error("No MT-MoSLoRA modules were created! Check apply_mt_moslora_to_model function.")
    
    # 加载MT-MoSLoRA适配器
    logger.info("Loading MT-MoSLoRA adapters...")
    load_mt_moslora_adapters(
        model, 
        model_args.adapter_config_path, 
        model_args.ha_adapter_path, 
        hs_adapter_paths
    )
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 准备数据集
    if training_args.do_train:
        logger.info(f"Loading and tokenizing training data from {data_args.train_file}")
        raw_datasets = load_dataset("json", data_files={"train": data_args.train_file}, cache_dir=model_args.cache_dir)

        # 硬件ID提取
        with training_args.main_process_first(desc="Extracting hardware IDs"):
            raw_datasets = raw_datasets.map(
                extract_hardware_id_function,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running hardware ID extraction",
            )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
        else:
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        with training_args.main_process_first(desc="Running tokenizer on training dataset"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        train_dataset = tokenized_datasets["train"]

    eval_dataset = None
    if training_args.do_eval:
        # 评估逻辑（如果需要可以添加）
        pass

    # 初始化训练器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = HardwareAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 训练
    if training_args.do_train:
        # 检测最后一个检查点
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # 保存MT-MoSLoRA适配器
        logger.info("Saving updated MT-MoSLoRA adapters...")
        from train_mt_moslora import save_mt_moslora_adapters, ModelArguments
        
        # 创建ModelArguments对象用于保存
        save_model_args = ModelArguments(
            use_mt_moslora=True,
            use_mixer=adapter_config.get('ha_config', {}).get('use_mixer', True),
            ha_lora_r=adapter_config.get('ha_config', {}).get('r', 16),
            ha_lora_alpha=adapter_config.get('ha_config', {}).get('alpha', 16),
            ha_lora_dropout=adapter_config.get('ha_config', {}).get('dropout', 0.05),
            hs_lora_r=adapter_config.get('hs_config', {}).get('r', 16),
            hs_lora_alpha=adapter_config.get('hs_config', {}).get('alpha', 32),
            hs_lora_dropout=adapter_config.get('hs_config', {}).get('dropout', 0.05),
            hardware_types=','.join(adapter_config.get('hardware_types', [])),
            target_modules=','.join(adapter_config.get('target_modules', [])) if adapter_config.get('target_modules') else None
        )
        
        save_mt_moslora_adapters(model, training_args.output_dir, save_model_args)
        # 保存分词器
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Updated MT-MoSLoRA adapters and tokenizer saved to {training_args.output_dir}")
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 最终操作
    kwargs = {"finetuned_from": model_args.base_model_path, "tasks": "text-generation"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
