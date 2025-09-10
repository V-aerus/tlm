#!/usr/bin/env python3
"""
MT-MoSLoRA Training Script
基于MTLoRA模式的MoSLoRA训练脚本，实现HA（硬件无关）+ HS（硬件专属）双轨制架构

核心设计：
1. HA-MoSLoRA：共享的硬件无关模块，对所有数据学习通用知识
2. HS-MoSLoRA：硬件专属模块字典，只在特定硬件数据时更新
3. 前向传播：base_output + ha_delta + hs_delta
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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
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
    early_stopping_patience: Optional[int] = field(default=None, metadata={"help": ""})

    # --- MT-MoSLoRA arguments ---
    use_mt_moslora: bool = field(default=False, metadata={"help": "Enable MT-MoSLoRA (HA + HS dual-track)"})
    use_mixer: bool = field(default=True, metadata={"help": "Enable MoSLoRA mixer (W matrix)"})
    
    # HA (Hardware-Agnostic) parameters
    ha_lora_r: Optional[int] = field(default=16, metadata={"help": "HA LoRA rank r"})
    ha_lora_alpha: Optional[int] = field(default=32, metadata={"help": "HA LoRA alpha"})
    ha_lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "HA LoRA dropout"})
    
    # HS (Hardware-Specific) parameters
    hs_lora_r: Optional[int] = field(default=16, metadata={"help": "HS LoRA rank r"})
    hs_lora_alpha: Optional[int] = field(default=32, metadata={"help": "HS LoRA alpha"})
    hs_lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "HS LoRA dropout"})
    
    # Hardware types configuration
    hardware_types: Optional[str] = field(
        default="v100,xavier,i7",
        metadata={"help": "Comma-separated hardware types for HS experts"}
    )
    
    # Target modules
    target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of module suffixes to adapt (e.g. 'q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj,c_attn,c_fc,c_proj'). "
                "If not provided, a broad default covering common architectures will be used."
            )
        },
    )
    
    # Defuse GPT-2 fused attention to enable full MoSLoRA (mixer) on q/k/v
    defuse_gpt2_attn: bool = field(default=False, metadata={"help": "Split GPT-2 fused c_attn into q_proj,k_proj,v_proj and convert c_proj to Linear for MoSLoRA"})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
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
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class MTMoSLoRALinear(nn.Module):
    """
    MT-MoSLoRA Linear Layer: HA (Hardware-Agnostic) + HS (Hardware-Specific) dual-track architecture
    
    Architecture:
    - HA-MoSLoRA: Shared hardware-agnostic module learning universal knowledge
    - HS-MoSLoRA: Hardware-specific expert modules learning device-specific optimizations
    - Forward: base_output + ha_delta + hs_delta
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 ha_config: dict, hs_config: dict, hardware_types: List[str]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hardware_types = hardware_types
        
        # Base linear layer (frozen)
        self.base_linear = nn.Linear(in_features, out_features, bias=True)
        self.base_linear.weight.requires_grad = False
        self.base_linear.bias.requires_grad = False
        
        # HA (Hardware-Agnostic) MoSLoRA module
        self.ha_moslora = self._create_moslora_module(in_features, out_features, ha_config, is_ha=True)
        
        # HS (Hardware-Specific) MoSLoRA modules
        self.hs_experts = nn.ModuleDict()
        for hw_type in hardware_types:
            self.hs_experts[hw_type] = self._create_moslora_module(in_features, out_features, hs_config)
        
        # Hardware router for automatic hardware detection
        self.hardware_router = self._build_hardware_router()
        
    def _create_moslora_module(self, in_features: int, out_features: int, config: dict, is_ha: bool = False):
        """Create a MoSLoRA module with the given configuration"""
        if MoSLoRALinear is None:
            raise ImportError("MoSLoRALinear not available. Please ensure MosLora/peft/src is in the path.")
        
        moslora_module = MoSLoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=config['r'],
            lora_alpha=config['alpha'],
            lora_dropout=config['dropout'],
            lora_use_mixer=config.get('use_mixer', True),
            bias=True
        )
        
        # 对于HA模块，使用更小的初始化，让它更接近原始模型
        if is_ha:
            with torch.no_grad():
                # 减小LoRA参数的初始化范围，让HA模块更温和地学习
                if hasattr(moslora_module, 'lora_A'):
                    moslora_module.lora_A.weight *= 0.1  # 减小10倍
                if hasattr(moslora_module, 'lora_B'):
                    moslora_module.lora_B.weight *= 0.1
                if hasattr(moslora_module, 'lora_AB'):
                    moslora_module.lora_AB.weight *= 0.1
        
        return moslora_module
    
    def _build_hardware_router(self) -> Dict[str, str]:
        """Build hardware routing dictionary"""
        return {
            'nvidia/nvidia-v100': 'v100',
            'nvidia/jetson-agx-xavier': 'xavier', 
            'intel/i7': 'i7',
            'v100': 'v100',
            'xavier': 'xavier',
            'i7': 'i7'
        }
    
    def route_hardware(self, hardware_id: str) -> str:
        """Route hardware_id to internal hardware type"""
        return self.hardware_router.get(hardware_id, 'v100')  # default to v100
    
    def forward(self, x: torch.Tensor, hardware_id: str = None):
        """
        Forward pass: base_output + ha_delta + hs_delta
        
        Args:
            x: Input tensor
            hardware_id: Hardware identifier (e.g., 'v100', 'xavier', 'i7')
        
        Returns:
            Output tensor with hardware-aware adaptations
        """
        # Base model output (frozen)
        base_output = self.base_linear(x)
        
        # HA (Hardware-Agnostic) adaptation - always applied
        ha_delta = self.ha_moslora(x)
        
        # HS (Hardware-Specific) adaptation - only for specific hardware
        hs_delta = torch.zeros_like(base_output)
        if hardware_id is not None:
            hw_type = self.route_hardware(hardware_id)
            if hw_type in self.hs_experts:
                hs_delta = self.hs_experts[hw_type](x)
        
        # Final output: base + HA + HS
        final_output = base_output + ha_delta + hs_delta
        
        return final_output
    
    def get_trainable_parameters(self):
        """Get statistics about trainable parameters"""
        ha_params = sum(p.numel() for p in self.ha_moslora.parameters() if p.requires_grad)
        hs_params = sum(p.numel() for p in self.hs_experts.parameters() if p.requires_grad)
        total_params = ha_params + hs_params
        
        return {
            'ha_params': ha_params,
            'hs_params': hs_params,
            'total_params': total_params,
            'num_hs_experts': len(self.hs_experts)
        }


def save_mt_moslora_adapters(model: nn.Module, output_dir: str, model_args: ModelArguments):
    """
    保存MT-MoSLoRA适配器，分别保存HA和HS模块
    """
    import os
    import json
    import torch
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有MT-MoSLoRA模块
    mt_moslora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, MTMoSLoRALinear):
            mt_moslora_modules[name] = module
    
    logger.info(f"Found {len(mt_moslora_modules)} MT-MoSLoRA modules to save")
    if len(mt_moslora_modules) == 0:
        logger.error("No MT-MoSLoRA modules found! This indicates a serious problem.")
        # 打印所有模块名称用于调试
        all_modules = list(model.named_modules())
        logger.info(f"Total modules in model: {len(all_modules)}")
        for name, module in all_modules[:10]:  # 只显示前10个
            logger.info(f"Module: {name}, Type: {type(module)}")
        return
    
    # 保存HA适配器
    ha_adapters = {}
    for name, module in mt_moslora_modules.items():
        ha_adapters[name] = {
            'lora_A': module.ha_moslora.lora_A.state_dict() if hasattr(module.ha_moslora, 'lora_A') else None,
            'lora_B': module.ha_moslora.lora_B.state_dict() if hasattr(module.ha_moslora, 'lora_B') else None,
            'lora_AB': module.ha_moslora.lora_AB.state_dict() if hasattr(module.ha_moslora, 'lora_AB') else None,
        }
    
    # 保存HA适配器
    ha_adapter_path = os.path.join(output_dir, "ha_adapter.bin")
    torch.save(ha_adapters, ha_adapter_path)
    logger.info(f"HA adapter saved to {ha_adapter_path}")
    
    # 保存HS适配器
    hardware_types = [t.strip() for t in model_args.hardware_types.split(",")]
    for hw_type in hardware_types:
        hs_adapters = {}
        for name, module in mt_moslora_modules.items():
            if hw_type in module.hs_experts:
                hs_adapters[name] = {
                    'lora_A': module.hs_experts[hw_type].lora_A.state_dict() if hasattr(module.hs_experts[hw_type], 'lora_A') else None,
                    'lora_B': module.hs_experts[hw_type].lora_B.state_dict() if hasattr(module.hs_experts[hw_type], 'lora_B') else None,
                    'lora_AB': module.hs_experts[hw_type].lora_AB.state_dict() if hasattr(module.hs_experts[hw_type], 'lora_AB') else None,
                }
        
        hs_adapter_path = os.path.join(output_dir, f"hs_{hw_type}_adapter.bin")
        torch.save(hs_adapters, hs_adapter_path)
        logger.info(f"HS {hw_type} adapter saved to {hs_adapter_path}")
    
    # 保存适配器配置
    adapter_config = {
        "adapter_type": "MT-MoSLoRA",
        "ha_config": {
            "r": model_args.ha_lora_r,
            "alpha": model_args.ha_lora_alpha,
            "dropout": model_args.ha_lora_dropout,
            "use_mixer": model_args.use_mixer
        },
        "hs_config": {
            "r": model_args.hs_lora_r,
            "alpha": model_args.hs_lora_alpha,
            "dropout": model_args.hs_lora_dropout,
            "use_mixer": model_args.use_mixer
        },
        "hardware_types": hardware_types,
        "target_modules": [m.strip() for m in model_args.target_modules.split(",")] if model_args.target_modules else None
    }
    
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    logger.info(f"Adapter config saved to {config_path}")


def apply_mt_moslora_to_model(model: nn.Module, model_args: ModelArguments) -> nn.Module:
    """
    Apply MT-MoSLoRA to the model by replacing target modules with MTMoSLoRALinear
    """
    if not model_args.use_mt_moslora:
        return model
    
    # Parse hardware types
    hardware_types = [t.strip() for t in model_args.hardware_types.split(",")]
    
    # Parse target modules
    if model_args.target_modules:
        target_modules = [m.strip() for m in model_args.target_modules.split(",") if m.strip()]
    else:
        # Default target modules for GPT-2 after defusion
        target_modules = ["q_proj", "k_proj", "v_proj", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    
    # HA and HS configurations
    ha_config = {
        'r': model_args.ha_lora_r,
        'alpha': model_args.ha_lora_alpha,
        'dropout': model_args.ha_lora_dropout,
        'use_mixer': model_args.use_mixer
    }
    
    hs_config = {
        'r': model_args.hs_lora_r,
        'alpha': model_args.hs_lora_alpha,
        'dropout': model_args.hs_lora_dropout,
        'use_mixer': model_args.use_mixer
    }
    
    # Find and replace target modules
    replacements = {}
    for name, module in model.named_modules():
        if any(name.endswith("." + target) for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                # Create MT-MoSLoRA replacement
                new_module = MTMoSLoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    ha_config=ha_config,
                    hs_config=hs_config,
                    hardware_types=hardware_types
                )
                
                # Copy weights from original module
                with torch.no_grad():
                    new_module.base_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        new_module.base_linear.bias.copy_(module.bias)
                
                if parent_name not in replacements:
                    replacements[parent_name] = {}
                replacements[parent_name][child_name] = new_module
    
    # Apply replacements
    for parent_name, child_map in replacements.items():
        parent_module = model.get_submodule(parent_name)
        for child_name, new_module in child_map.items():
            setattr(parent_module, child_name, new_module)
            logger.info(f"Replaced {parent_name}.{child_name} with MT-MoSLoRA")
    
    return model


def extract_hardware_id_function(example):
    """
    从 text 字段中解析硬件信息，并添加 'hardware_id' 字段
    """
    text_data = example['text']
    
    # 硬件检测规则
    if "sm_70" in text_data or "v100" in text_data.lower():
        example['hardware_id'] = 'v100'
    elif "xavier" in text_data.lower() or "jetson" in text_data.lower():
        example['hardware_id'] = 'xavier'
    elif "i7" in text_data.lower() or "intel" in text_data.lower():
        example['hardware_id'] = 'i7'
    else:
        # 默认硬件类型
        example['hardware_id'] = 'v100'
        
    return example


class HardwareAwareTrainer(transformers.Trainer):
    """
    硬件感知的训练器，在每个训练步骤中传递hardware_id
    """
    def training_step(self, model, inputs):
        # 从数据中获取hardware_id并设置到模型上
        if 'hardware_id' in inputs:
            model.current_hardware_id = inputs['hardware_id'][0] if isinstance(inputs['hardware_id'], list) else inputs['hardware_id']
        
        return super().training_step(model, inputs)


def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
    
    # 加载配置
    config_kwargs = {}
    if model_args.config_overrides is not None:
        config_kwargs = {k: v for k, v in [x.split("=") for x in model_args.config_overrides.split(",")]}
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            config.update_from_string(model_args.config_overrides)
    
    # 加载分词器
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # 调整模型大小
    model.resize_token_embeddings(len(tokenizer))

    # GPT-2解融合（如果需要）
    is_gpt2 = getattr(config, "model_type", None) == "gpt2"
    if model_args.use_mt_moslora and model_args.defuse_gpt2_attn and is_gpt2:
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

    # 应用MT-MoSLoRA
    if model_args.use_mt_moslora:
        logger.info("Applying MT-MoSLoRA...")
        model = apply_mt_moslora_to_model(model, model_args)
        
        # 验证MT-MoSLoRA模块是否正确创建
        mt_moslora_count = 0
        for name, module in model.named_modules():
            if isinstance(module, MTMoSLoRALinear):
                mt_moslora_count += 1
        logger.info(f"MT-MoSLoRA modules created: {mt_moslora_count}")
        if mt_moslora_count == 0:
            logger.error("No MT-MoSLoRA modules were created! Check apply_mt_moslora_to_model function.")
        
        # 打印参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 准备数据集
    if training_args.do_train:
        if data_args.dataset_name:
            # 从磁盘加载预处理的数据集
            logger.info(f"Loading pre-processed dataset from {data_args.dataset_name}")
            tokenized_datasets = load_from_disk(data_args.dataset_name, keep_in_memory=True)
            train_dataset = tokenized_datasets["train"]

        elif data_args.train_file:
            # 从原始JSON文件加载并分词
            logger.info(f"Loading and tokenizing SFT data from {data_args.train_file}")
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

            with training_args.main_process_first(desc="Running tokenizer on SFT dataset"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            train_dataset = tokenized_datasets["train"]

        else:
            raise ValueError("For training, you must provide either a `dataset_name` or a `train_file`.")

    eval_dataset = None
    if training_args.do_eval:
        # 评估逻辑（如果需要可以添加）
        pass

    # 初始化训练器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 对于MT-MoSLoRA，禁用Trainer的自动模型保存，我们使用自定义保存逻辑
    if model_args.use_mt_moslora:
        training_args.save_strategy = "no"  # 禁用自动保存
        training_args.save_steps = 0
        logger.info("Disabled Trainer auto-saving for MT-MoSLoRA (using custom adapter saving)")

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
        if model_args.use_mt_moslora:
            logger.info("Saving MT-MoSLoRA adapters...")
            save_mt_moslora_adapters(model, training_args.output_dir, model_args)
            # 保存分词器
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info(f"MT-MoSLoRA adapters and tokenizer saved to {training_args.output_dir}")
        else:
            # 对于非MT-MoSLoRA模型，使用标准保存
            trainer.save_model()
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 最终操作
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
