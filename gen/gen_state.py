from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from transformers import HfArgumentParser
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_record_from_string
from common import register_data_path, load_and_register_tasks
import tvm
from make_dataset import input_to_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import time
import os
import random
import json
from postprocess import check_measured
import math
from multiprocessing import Process, Queue
import subprocess
import shutil
import sys
import glob

# MoSLoRA integration: ensure the local customized PEFT is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PEFT_PATH = os.path.join(_BASE_DIR, "MosLora", "peft", "src")
if os.path.isdir(_LOCAL_PEFT_PATH) and _LOCAL_PEFT_PATH not in sys.path:
    sys.path.append(_LOCAL_PEFT_PATH)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# 导入MT-MoSLoRA相关类
try:
    from train_mt_moslora import MTMoSLoRALinear, apply_mt_moslora_to_model, ModelArguments
except Exception:
    MTMoSLoRALinear = None
    apply_mt_moslora_to_model = None
    ModelArguments = None

from modeling import (
    BasePlusExperts,
    ExpertRegistry,
    FrozenBaseWrapper,
    GatedLoRAExpert,
    PeftDeltaWrapper,
)


@dataclass
class ScriptArguments:
    # 弃用 model_name_or_path，重命名为更清晰的参数
    model_path: str = field(metadata={"help": "Path to the main model (base model or complete standard model)"})
    sketch_path: str = field(metadata={"help": ""})
    save_path: str = field(metadata={"help": ""})
    keep_cnt: int = field(metadata={"help": ""})
    target: str = field(metadata={"help": ""})
    
    # 新增适配器参数
    adapter_path: str = field(default=None, metadata={"help": "Path to a single LoRA/MoSLORA adapter (PEFT)"})
    multi_adapter_dir: str = field(default=None, metadata={"help": "Path to directory containing multiple HA/HS adapter files"})
    edge_expert_dirs: str = field(default=None, metadata={"help": "Comma-separated directories containing EdgeTLM experts"})
    edge_embedding_path: str = field(default="Embedding/hardware_embeddings_v2.json", metadata={"help": "Hardware embedding json path"})
    edge_topk: int = field(default=1, metadata={"help": "Top-K experts to activate during inference"})
    
    # 硬件路由参数
    target_hardware: str = field(default=None, metadata={"help": "Target hardware type for MT-MoSLoRA (e.g., v100, xavier, i7)"})

    # device: str = field(default="cuda:0", metadata={"help": ""})
    allow_repeat: bool = field(default=True, metadata={"help": ""})
    is_build: bool = field(default=False, metadata={"help": ""})


def gen_func(task, states, input, tokenizer, model, device, gen_kwargs):
    if len(states) == 0:
        return []
    tokens = input_to_tokens(task, states, input)
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(tokens, padding=True, max_length=None)
    except Exception as e:
        print(e)
        print(task, states, input, tokenizer, model, device, gen_kwargs)
        raise Exception()
    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = 64

    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = input_ids_all[start : start + batch_size]
            attention_mask = attention_mask_all[start : start + batch_size]

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)[:, :-1]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)[:, :-1]
            gen_kwargs['max_new_tokens'] = min(gen_kwargs['max_new_tokens'], tokenizer.model_max_length - input_ids.shape[-1])

            response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            response = response[:, input_ids.shape[-1]:]
            response_list.extend(response.tolist())
    return [tokenizer.batch_decode(item) for item in response_list]


def extract_hardware_id_from_target(target) -> str:
    """
    从target字符串或tvm.target.Target对象中提取硬件标识符
    例如: "nvidia/nvidia-v100" -> "v100"
    """
    if not target:
        return "v100"  # 默认硬件类型
    
    # 处理tvm.target.Target对象
    if hasattr(target, 'keys'):  # tvm.target.Target对象
        target_str = str(target)
    else:
        target_str = str(target)
    
    # 提取硬件标识符的规则
    target_lower = target_str.lower()
    
    if "v100" in target_lower:
        return "v100"
    elif "xavier" in target_lower:
        return "xavier"
    elif "i7" in target_lower or "intel" in target_lower:
        return "i7"
    else:
        # 如果无法识别，返回默认值
        print(f"Warning: Cannot extract hardware ID from target '{target_str}', using default 'v100'")
        return "v100"


EDGE_EMBEDDING_DEFAULTS = {
    "v100": "nvidia/nvidia-v100",
    "4090": "nvidia/nvidia-a40",
    "xavier": "nvidia/jetson-agx-xavier",
    "xeon": "aws/cpu/c5.18xlarge",
}


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def resolve_hw_embedding(hardware_id: str, embeddings: Dict[str, List[float]]) -> Tuple[str, List[float]]:
    if hardware_id in EDGE_EMBEDDING_DEFAULTS:
        hw_name = EDGE_EMBEDDING_DEFAULTS[hardware_id]
        if hw_name in embeddings:
            return hw_name, embeddings[hw_name]
    for name, vector in embeddings.items():
        if hardware_id in name:
            return name, vector
    raise KeyError(f"No embedding found for hardware id '{hardware_id}'")


def load_model_for_inference(args: ScriptArguments) -> tuple:
    """
    集中的模型加载函数 - 所有模型加载逻辑的唯一入口
    
    Args:
        args: ScriptArguments对象，包含所有必要的参数
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model for inference with mode detection...")
    
    # 1. 初始化 Tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 2. 模式判断 (核心逻辑)
    expert_dirs = []
    if args.edge_expert_dirs:
        expert_dirs = [p.strip() for p in args.edge_expert_dirs.split(",") if p.strip()]

    if args.multi_adapter_dir:
        # 模式 C - MT-MoSLORA
        print("Mode C: Multi-adapter MT-MoSLoRA")
        print(f"Base model: {args.model_path}")
        print(f"Multi-adapter directory: {args.multi_adapter_dir}")
        
        # 检查MT-MoSLoRA类是否可用
        if MTMoSLoRALinear is None or apply_mt_moslora_to_model is None:
            raise ImportError("MT-MoSLoRA classes not available. Please ensure train_mt_moslora.py is accessible.")
        
        # 加载基础模型
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        # 检查适配器配置文件
        adapter_config_path = os.path.join(args.multi_adapter_dir, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}")
        
        # 加载适配器配置
        print("Loading adapter configuration...")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # 创建ModelArguments对象（兼容无HA配置与关闭mixer）
        ha_cfg = adapter_config.get('ha_config', None)
        hs_cfg = adapter_config.get('hs_config', {}) or {}
        use_ha = isinstance(ha_cfg, dict) and (ha_cfg.get('r', 0) or 0) > 0
        model_args = ModelArguments(
            use_mt_moslora=True,
            use_mixer=(hs_cfg.get('use_mixer', False) if not use_ha else ha_cfg.get('use_mixer', False)),
            use_ha=use_ha,
            ha_lora_r=(ha_cfg.get('r', 0) if isinstance(ha_cfg, dict) else 0),
            ha_lora_alpha=(ha_cfg.get('alpha', 0) if isinstance(ha_cfg, dict) else 0),
            ha_lora_dropout=(ha_cfg.get('dropout', 0.0) if isinstance(ha_cfg, dict) else 0.0),
            hs_lora_r=hs_cfg.get('r', 16),
            hs_lora_alpha=hs_cfg.get('alpha', 32),
            hs_lora_dropout=hs_cfg.get('dropout', 0.05),
            hardware_types=','.join(adapter_config.get('hardware_types', [])),
            target_modules=','.join(adapter_config.get('target_modules', [])) if adapter_config.get('target_modules') else None
        )
        
        # 执行模型改造：应用MT-MoSLoRA到模型
        print("Applying MT-MoSLoRA architecture to model...")
        model = apply_mt_moslora_to_model(base_model, model_args)
        
        # 解析目标硬件
        if args.target_hardware:
            target_hardware = args.target_hardware
        else:
            target_hardware = extract_hardware_id_from_target(args.target)
        
        print(f"Target hardware: {target_hardware}")
        
        # 加载 HA 适配器
        # 可选加载 HA 适配器
        ha_adapters = {}
        if model_args.use_ha:
            print("Loading HA adapter...")
            ha_adapter_path = os.path.join(args.multi_adapter_dir, "ha_adapter.bin")
            if os.path.exists(ha_adapter_path):
                ha_adapters = torch.load(ha_adapter_path, map_location='cpu')
                print(f"Loaded HA adapter with {len(ha_adapters)} modules")
            else:
                print(f"Warning: HA adapter not found at {ha_adapter_path}")
        
        # 加载 HS 适配器 (动态选择)
        print(f"Loading HS adapter for {target_hardware}...")
        hs_adapter_path = os.path.join(args.multi_adapter_dir, f"hs_{target_hardware}_adapter.bin")
        if os.path.exists(hs_adapter_path):
            hs_adapters = torch.load(hs_adapter_path, map_location='cpu')
            print(f"Loaded HS {target_hardware} adapter with {len(hs_adapters)} modules")
        else:
            print(f"Warning: HS {target_hardware} adapter not found at {hs_adapter_path}")
            hs_adapters = {}
        
        # 应用适配器权重到模型（兼容 FrozenBaseWrapper）
        print("Applying adapter weights to model...")
        target_model = getattr(model, "base_model", model)
        for name, module in target_model.named_modules():
            if isinstance(module, MTMoSLoRALinear):
                # 加载HA适配器（如启用）
                if model_args.use_ha and name in ha_adapters:
                    ha_data = ha_adapters[name]
                    if ha_data['lora_A'] is not None:
                        module.ha_moslora.lora_A.load_state_dict(ha_data['lora_A'])
                    if ha_data['lora_B'] is not None:
                        module.ha_moslora.lora_B.load_state_dict(ha_data['lora_B'])
                    if ha_data['lora_AB'] is not None and hasattr(module.ha_moslora, 'lora_AB'):
                        module.ha_moslora.lora_AB.load_state_dict(ha_data['lora_AB'])
                
                # 加载HS适配器
                if name in hs_adapters and target_hardware in module.hs_experts:
                    hs_module_data = hs_adapters[name]
                    if hs_module_data['lora_A'] is not None:
                        module.hs_experts[target_hardware].lora_A.load_state_dict(hs_module_data['lora_A'])
                    if hs_module_data['lora_B'] is not None:
                        module.hs_experts[target_hardware].lora_B.load_state_dict(hs_module_data['lora_B'])
                    if hs_module_data['lora_AB'] is not None and hasattr(module.hs_experts[target_hardware], 'lora_AB'):
                        module.hs_experts[target_hardware].lora_AB.load_state_dict(hs_module_data['lora_AB'])
        
        # 设置目标硬件
        print(f"Setting target hardware to {target_hardware}")
        set_target_hardware(model, target_hardware)
    elif expert_dirs:
        if PeftModel is None:
            raise ImportError("PEFT library not available. Please install peft.")

        print("Mode D: EdgeTLM expert gating")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
        base_model.to(device)
        base_model.eval()
        base_model.requires_grad_(False)

        frozen_base = FrozenBaseWrapper(base_model)
        embeddings = load_hardware_embeddings(args.edge_embedding_path)

        hw_id = args.target_hardware or extract_hardware_id_from_target(args.target)
        hw_name, hw_vector = resolve_hw_embedding(hw_id, embeddings)
        hw_emb_tensor = torch.tensor(hw_vector, dtype=torch.float32, device=device)
        print(f"Target hardware: {hw_id} ({hw_name})")

        registry = ExpertRegistry()
        for dir_path in expert_dirs:
            resolved_dir = os.path.abspath(dir_path)
            router_path = os.path.join(resolved_dir, "router.json")
            if not os.path.exists(router_path):
                raise FileNotFoundError(f"router.json not found in {resolved_dir}")
            print(f"Loading Edge expert from {resolved_dir}")

            expert_model = AutoModelForCausalLM.from_pretrained(args.model_path)
            expert_model = PeftModel.from_pretrained(expert_model, resolved_dir)
            expert_model.to(device)
            expert_model.eval()
            expert_model.requires_grad_(False)

            delta_wrapper = PeftDeltaWrapper(expert_model)

            with open(router_path, "r", encoding="utf-8") as rf:
                router_state = json.load(rf)

            r_vector = torch.tensor(router_state["r"], dtype=torch.float32, device=device)
            expert = GatedLoRAExpert(delta_wrapper, r_dim=r_vector.numel(), init_router=r_vector)
            beta_value = float(router_state.get("beta", float(expert.beta.detach().cpu())))
            tau_value = float(router_state.get("tau", float(expert.tau.detach().cpu())))
            with torch.no_grad():
                expert.beta.fill_(beta_value)
                expert.tau.fill_(tau_value)
            expert.to(device)

            expert_name = os.path.basename(resolved_dir.rstrip("/"))
            registry.register(expert_name, expert)

        system = BasePlusExperts(frozen_base, registry)
        topk = max(1, args.edge_topk)

        original_forward = base_model.forward

        def edge_forward(*forward_args, **forward_kwargs):
            input_ids = forward_kwargs.get("input_ids")
            if input_ids is None and forward_args:
                input_ids = forward_args[0]
            if input_ids is None:
                raise ValueError("EdgeTLM forward requires input_ids.")

            attention_mask = forward_kwargs.get("attention_mask")
            base_outputs = original_forward(*forward_args, **forward_kwargs)
            base_logits = base_outputs.logits

            batch_size = input_ids.shape[0]
            hw_batch = hw_emb_tensor.unsqueeze(0).expand(batch_size, -1)

            expert_kwargs = dict(forward_kwargs)
            expert_kwargs["input_ids"] = input_ids
            expert_kwargs["attention_mask"] = attention_mask
            expert_kwargs["base_logits"] = base_logits

            logits, meta = system.forward_multi(
                input_ids,
                hw_emb=hw_batch,
                topk=topk,
                expert_kwargs=expert_kwargs,
                cached_base=base_logits,
            )
            base_outputs.logits = logits
            base_outputs.edge_meta = meta
            return base_outputs

        base_model.forward = edge_forward
        model = base_model
    elif args.adapter_path:
        # 模式 B - 单适配器 MoSLORA
        print("Mode B: Single-adapter MoSLoRA")
        print(f"Base model: {args.model_path}")
        print(f"Adapter path: {args.adapter_path}")
        
        if PeftModel is None:
            raise ImportError("PEFT library not available. Please install peft.")
        
        # 加载基础模型
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        # 使用 PEFT 库加载适配器
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        
    else:
        # 模式 A - 标准推理
        print("Mode A: Standard inference")
        print(f"Loading complete model from {args.model_path}")
        
        # 直接加载完整模型
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    print("Model loading completed successfully!")
    return model, tokenizer


def load_mt_moslora_model(model_path, multi_adapter_dir, target_hardware):
    """
    加载MT-MoSLoRA模型 (保留向后兼容性)
    """
    if MTMoSLoRALinear is None or apply_mt_moslora_to_model is None:
        raise ImportError("MT-MoSLoRA classes not available. Please ensure train_mt_moslora.py is accessible.")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 检查适配器配置文件
    adapter_config_path = os.path.join(multi_adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}")
    
    # 加载适配器配置
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    # 创建ModelArguments对象
    model_args = ModelArguments(
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
    
    # 应用MT-MoSLoRA到模型
    model = apply_mt_moslora_to_model(base_model, model_args)
    
    # 加载适配器权重
    load_mt_moslora_adapters(model, multi_adapter_dir, target_hardware)
    
    # 设置目标硬件
    set_target_hardware(model, target_hardware)
    
    return model


def load_mt_moslora_adapters(model, multi_adapter_dir, target_hardware):
    """
    加载MT-MoSLoRA适配器权重
    """
    import torch
    
    # 加载HA适配器
    ha_adapter_path = os.path.join(multi_adapter_dir, "ha_adapter.bin")
    if os.path.exists(ha_adapter_path):
        ha_adapters = torch.load(ha_adapter_path, map_location='cpu')
        print(f"Loaded HA adapter with {len(ha_adapters)} modules")
    else:
        print(f"Warning: HA adapter not found at {ha_adapter_path}")
        ha_adapters = {}
    
    # 加载目标硬件的HS适配器
    hs_adapter_path = os.path.join(multi_adapter_dir, f"hs_{target_hardware}_adapter.bin")
    if os.path.exists(hs_adapter_path):
        hs_adapters = torch.load(hs_adapter_path, map_location='cpu')
        print(f"Loaded HS {target_hardware} adapter with {len(hs_adapters)} modules")
    else:
        print(f"Warning: HS {target_hardware} adapter not found at {hs_adapter_path}")
        hs_adapters = {}
    
    # 应用适配器权重到模型（兼容 FrozenBaseWrapper）
    target_model = getattr(model, "base_model", model)
    for name, module in target_model.named_modules():
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
            
            # 加载HS适配器
            if name in hs_adapters and target_hardware in module.hs_experts:
                hs_module_data = hs_adapters[name]
                if hs_module_data['lora_A'] is not None:
                    module.hs_experts[target_hardware].lora_A.load_state_dict(hs_module_data['lora_A'])
                if hs_module_data['lora_B'] is not None:
                    module.hs_experts[target_hardware].lora_B.load_state_dict(hs_module_data['lora_B'])
                if hs_module_data['lora_AB'] is not None and hasattr(module.hs_experts[target_hardware], 'lora_AB'):
                    module.hs_experts[target_hardware].lora_AB.load_state_dict(hs_module_data['lora_AB'])


def set_target_hardware(model, target_hardware):
    """
    设置目标硬件，激活对应的HS适配器
    """
    # 兼容 FrozenBaseWrapper
    target_model = getattr(model, "base_model", model)
    for name, module in target_model.named_modules():
        if isinstance(module, MTMoSLoRALinear):
            module.set_active_hardware(target_hardware)


def merge_json_files_safely(tmp_folder, save_path):
    """
    安全地合并多个JSON文件，避免格式错误
    """
    print(f"开始合并JSON文件从 {tmp_folder} 到 {save_path}")
    
    all_records = []
    part_files = glob.glob(f"{tmp_folder}/*_part")
    part_files.sort()  # 确保文件按顺序处理
    
    total_files = len(part_files)
    valid_files = 0
    total_records = 0
    
    for part_file in part_files:
        try:
            if os.path.getsize(part_file) > 0:  # 只处理非空文件
                valid_files += 1
                file_records = 0
                with open(part_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                # 验证JSON格式
                                json.loads(line)
                                all_records.append(line)
                                file_records += 1
                                total_records += 1
                            except json.JSONDecodeError as e:
                                print(f"警告: {part_file} 第{line_num}行JSON格式错误: {e}")
                                print(f"问题行内容: {line[:100]}...")
                                continue
                print(f"文件 {part_file}: {file_records} 条有效记录")
            else:
                print(f"跳过空文件: {part_file}")
        except Exception as e:
            print(f"处理文件 {part_file} 时出错: {e}")
            continue
    
    print(f"合并完成: {valid_files}/{total_files} 个文件有效，共 {total_records} 条记录")
    
    # 写入合并后的文件
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(record + '\n')
        print(f"成功写入 {save_path}，共 {len(all_records)} 条记录")
        return len(all_records)
    except Exception as e:
        print(f"写入文件 {save_path} 时出错: {e}")
        raise


def worker(err_queue, save_path_i, sketch_path, gen_kwargs, model_path, adapter_path, multi_adapter_dir, target_hardware, original_target, device, allow_repeat, keep_cnt, is_build, worker_id, num_workers):
    try:
        # <<< 新增的TVM初始化代码 >>>
        print(f"Initializing TVM environment in worker for target: {original_target}")
        register_data_path(original_target)
        load_and_register_tasks()
        # <<< 结束新增代码 >>>
        
        # 在子进程中重新构建ScriptArguments对象
        script_args = ScriptArguments(
            model_path=model_path,
            sketch_path="",  # 不需要在worker中使用
            save_path="",    # 不需要在worker中使用
            keep_cnt=keep_cnt,
            target=original_target,
            adapter_path=adapter_path,
            multi_adapter_dir=multi_adapter_dir,
            target_hardware=target_hardware,
            allow_repeat=allow_repeat,
            is_build=is_build
        )
        
        # 使用集中的模型加载函数
        print(f"Loading model and tokenizer in worker process for device: {device}")
        model, tokenizer = load_model_for_inference(script_args)
        
        # 更新gen_kwargs中的tokenizer相关参数
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["eos_token_id"] = tokenizer.sep_token_id
        
        # 将模型移动到指定设备
        print(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()
        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            # tag = input(script_args.save_path + ' exist, delete it? [n]')
            # if tag == 'y':
            os.remove(save_path_i)
        
        # 3. 数据读取和分发 (新增/移动的代码)
        print(f"Worker {worker_id}: Reading and processing sketches from {sketch_path}")
        inputs, _ = auto_scheduler.RecordReader(sketch_path).read_lines()
        sketch_dic = {}
        inp_dic = {}
        for inp in tqdm.tqdm(inputs):
            workload_key = inp.task.workload_key
            inp_str = inp.to_json()
            if inp_str in inp_dic:
                inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
            else:
                inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
                inp_dic[inp_str] = inp
            if workload_key not in sketch_dic:
                sketch_dic[workload_key] = []
            sketch_dic[workload_key].append(inp)
        
        sketch_dic_list_full = list(sketch_dic.items())
        
        # <<< 优化的数据分片逻辑 - 采用原始版本的均匀分片策略 >>>
        # 使用数学计算确保数据均匀分布，避免步长分片可能导致的不均匀问题
        per_len = math.ceil(len(sketch_dic_list_full) / num_workers)
        start_idx = worker_id * per_len
        end_idx = min((worker_id + 1) * per_len, len(sketch_dic_list_full))
        my_sketch_chunk = sketch_dic_list_full[start_idx:end_idx]
        
        print(f"Worker {worker_id}: 数据分片 [{start_idx}:{end_idx}]，处理 {len(my_sketch_chunk)} out of {len(sketch_dic_list_full)} workload groups")
        
        # 4. 主循环现在遍历自己的数据块
        total_workloads = len(my_sketch_chunk)
        successful_workloads = 0
        total_generated = 0
        
        print(f"Worker {worker_id}: 开始处理 {total_workloads} 个workload组")
        
        for workload_idx, (workload_key, inputs) in enumerate(tqdm.tqdm(my_sketch_chunk, desc=f"Worker {worker_id}")):
            try:
                def gen_func_inner(task, states, max_new_tokens):
                    max_new_tokens = max(max_new_tokens, 1)
                    gen_kwargs["max_new_tokens"] = max_new_tokens
                    return gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs)

                policy = auto_scheduler.SketchPolicy(inputs[0].task)
                measure_inputs = []
                measure_results = []
                input_set = set()
                
                # from filelock import FileLock
                # lock = FileLock('/root/tlm/gen/my_lock.lock')
                # with lock:
                retry_i = 0
                while retry_i < 5:
                    try:
                        all_state_list = policy.gen_states([inp.state for inp in inputs], gen_func_inner)
                        # measure_inputs_cnt_before = len(measure_inputs)

                        measure_inputs_tmp = []
                        for state in all_state_list:
                            inp = auto_scheduler.MeasureInput(inputs[0].task, state)
                            i_str = inp.to_json()
                            if i_str in input_set:
                                continue
                            if allow_repeat is False and check_measured(i_str):
                                continue
                            
                            input_set.add(i_str)
                            measure_inputs_tmp.append(inp)

                        default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                        if is_build:
                            build_results = builder.build(measure_inputs_tmp)
                        else:
                            build_results = [default_build_result for x in measure_inputs_tmp]
                        for res, inp in zip(build_results, measure_inputs_tmp):
                            if res.error_no == 0:
                                measure_inputs.append(inp)
                                measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                        retry_i += 1
                        # measure_inputs_cnt_after = len(measure_inputs)
                        # if measure_inputs_cnt_before == measure_inputs_cnt_after:
                        #     retry_i += 1
                        # else:
                        #     retry_i = 0
                        if len(measure_inputs) >= keep_cnt:
                            break
                    except Exception as e:
                        print(f"Worker {worker_id}: workload {workload_key} 第{retry_i+1}次重试时出错: {e}")
                        retry_i += 1
                        if retry_i >= 5:
                            print(f"Worker {worker_id}: workload {workload_key} 重试5次后仍然失败，跳过")
                            break
                        continue
                
                if len(measure_inputs) > keep_cnt:
                    measure_inputs, measure_results = zip(
                        *random.sample(list(zip(measure_inputs, measure_results)), keep_cnt)
                    )
                
                if len(measure_inputs) > 0:
                    auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
                    successful_workloads += 1
                    total_generated += len(measure_inputs)
                    print(f"Worker {worker_id}: workload {workload_key} 成功生成 {len(measure_inputs)} 条记录")
                else:
                    print(f"Worker {worker_id}: workload {workload_key} 未能生成任何有效记录")
                    
            except Exception as e:
                print(f"Worker {worker_id}: 处理workload {workload_key} 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Worker {worker_id}: 处理完成！成功处理 {successful_workloads}/{total_workloads} 个workload，共生成 {total_generated} 条记录")
    except Exception as e:
        err_queue.put(e)


def main():
    # 强制使用spawn模式解决CUDA多进程冲突
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # TVM初始化已移至worker进程中，避免在进程间传递TVM对象

    # 不再在主进程中加载模型，改为在worker进程中加载
    print("Model and tokenizer will be loaded in worker processes...")
    
    # 使用默认的gen_kwargs，避免在主进程中初始化CUDA
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": None,  # 将在worker进程中设置
        "eos_token_id": None   # 将在worker进程中设置
    }

    # 不再在主进程中处理inputs，改为传递文件路径给子进程
    # 子进程将重新读取文件并构建TVM对象

    # 修复GPU设备检测，支持CUDA_VISIBLE_DEVICES环境变量
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        # 解析可见的GPU设备
        visible_gpu_list = [int(x.strip()) for x in visible_devices.split(',') if x.strip()]
        num_gpus = len(visible_gpu_list)
        print(f"使用CUDA_VISIBLE_DEVICES={visible_devices}，检测到{num_gpus}个GPU")
        print(f"可见GPU列表: {visible_gpu_list}")
    else:
        # 如果没有设置CUDA_VISIBLE_DEVICES，使用所有可用GPU
        num_gpus = torch.cuda.device_count()
        visible_gpu_list = list(range(num_gpus))
        print(f"未设置CUDA_VISIBLE_DEVICES，使用所有{num_gpus}个GPU")
    
    # filelist = []
    processes = []
    tmp_folder = '.gen_state'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    err_queue = Queue()
    for gpu_i in range(num_gpus):
        save_path_i = f'{tmp_folder}/{gpu_i}_part'
        # filelist.append(save_path_i)
        # 关键修复：当设置了CUDA_VISIBLE_DEVICES时，设备ID应该从0开始
        # 因为CUDA_VISIBLE_DEVICES会重新映射设备ID
        if visible_devices:
            device = f'cuda:{gpu_i}'  # 使用重新映射后的ID
            actual_physical_gpu = visible_gpu_list[gpu_i]
            print(f"Worker {gpu_i}: 使用设备 {device} (物理GPU {actual_physical_gpu})")
        else:
            device = f'cuda:{gpu_i}'
            print(f"Worker {gpu_i}: 使用设备 {device}")
        p = Process(target=worker, args=(err_queue, save_path_i, script_args.sketch_path, gen_kwargs, script_args.model_path, script_args.adapter_path, script_args.multi_adapter_dir, script_args.target_hardware, script_args.target, device, script_args.allow_repeat, script_args.keep_cnt, script_args.is_build, gpu_i, num_gpus))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")


    # 使用安全的JSON合并函数替代简单的cat命令
    try:
        total_records = merge_json_files_safely(tmp_folder, script_args.save_path)
        print(f"推理完成！总共生成 {total_records} 条记录")
    except Exception as e:
        print(f"JSON合并失败: {e}")
        # 如果合并失败，回退到原始方法
        print("回退到原始cat命令...")
        subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True)
    finally:
        shutil.rmtree(tmp_folder)
    


if __name__ == "__main__":
    main()
