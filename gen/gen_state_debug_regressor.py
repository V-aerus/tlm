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
from typing import Optional
import logging
import sys

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
ProtoMixAligner,
)

# 全局日志文件路径，由 main 设置
LOG_FILE_PATH = None


def log_debug(msg: str):
    """简易日志写文件，避免 logging 配置在多进程失效。"""
    if LOG_FILE_PATH:
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

def diagnose_state_failure(worker_id, workload_key, debug_ctx):
    """
    当 TVM 报告 “All states are invalid” 时打印调试信息。
    debug_ctx 可能包含：
      - last_input_json: str
      - last_states_json: List[str]
      - last_prompts: List[str]
      - last_generations: List[str]
      - hw_info: str
    """
    try:
        hw_info = debug_ctx.get("hw_info")
        log_debug("\n" + "=" * 80)
        log_debug(f"[DEBUG][Worker {worker_id}] TVM rejected all states for workload: {workload_key}")
        if hw_info:
            log_debug(f"[DEBUG] HW info: {hw_info}")
        log_debug("=" * 80)

        inp_json = debug_ctx.get("last_input_json")
        if inp_json is not None:
            log_debug("[DEBUG] Last MeasureInput.to_json():")
            log_debug(inp_json)

        states_json = debug_ctx.get("last_states_json") or []
        if states_json:
            log_debug(f"[DEBUG] Dumping {len(states_json)} state(s):")
            for i, s in enumerate(states_json[:5]):
                log_debug(f"  - state[{i}]: {s}")

        prompts = debug_ctx.get("last_prompts") or []
        if prompts:
            log_debug(f"[DEBUG] Dumping {len(prompts)} LLM prompt(s):")
            for i, p in enumerate(prompts[:5]):
                log_debug(f"  --- prompt[{i}] ---")
                log_debug(p)

        generations = debug_ctx.get("last_generations") or []
        if generations:
            log_debug(f"[DEBUG] Dumping {len(generations)} LLM generation(s):")
            for i, g in enumerate(generations[:5]):
                log_debug(f"  --- generation[{i}] ---")
                log_debug(g)

        log_debug("=" * 80 + "\n")
    except Exception as e:
        log_debug(f"[DEBUG] diagnose_state_failure raised error: {e}")


def find_subsequence(seq: List[int], pattern: List[int]) -> int:
    """Return start index of first occurrence of pattern in seq, or -1."""
    n, m = len(seq), len(pattern)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if seq[i : i + m] == pattern:
            return i
    return -1


def generate_with_embeds_greedy(model, embed_layer, embeds, attention_mask, max_new_tokens, eos_token_id=None):
    """
    绕过 PeftModel.generate(inputs_embeds=...) 的潜在问题，手写最简单的 greedy 解码。
    embeds: (B, L, D), attention_mask: (B, L)
    返回: (B, T_new) 的 token id 张量（只包含新生成部分）
    """
    device = embeds.device
    B = embeds.size(0)
    generated = []
    cur_embeds = embeds
    cur_attn = attention_mask

    for _ in range(max_new_tokens):
        outputs = model(inputs_embeds=cur_embeds, attention_mask=cur_attn)
        logits = outputs.logits[:, -1, :]  # (B, vocab)
        next_ids = torch.argmax(logits, dim=-1)  # (B,)
        generated.append(next_ids)

        if eos_token_id is not None and (next_ids == eos_token_id).all():
            break

        next_embeds = embed_layer(next_ids.unsqueeze(1))  # (B,1,D)
        cur_embeds = torch.cat([cur_embeds, next_embeds], dim=1)
        next_attn = torch.ones((B, 1), dtype=cur_attn.dtype, device=device)
        cur_attn = torch.cat([cur_attn, next_attn], dim=1)

    if not generated:
        return torch.zeros((B, 0), dtype=torch.long, device=device)
    return torch.stack(generated, dim=1)


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
    edge_embedding_path: str = field(default="Embedding/hardware_embeddings_v3.json", metadata={"help": "Hardware embedding json path"})
    edge_topk: int = field(default=1, metadata={"help": "Top-K experts to activate during inference"})
    
    # 硬件路由参数
    target_hardware: str = field(default=None, metadata={"help": "Target hardware type for MT-MoSLoRA (e.g., v100, xavier, i7)"})
    hw_injection: bool = field(default=False, metadata={"help": "Enable HwToken injection"})
    hw_aligner_path: str = field(default=None, metadata={"help": "Path to ProtoMix aligner checkpoint"})
    hw_regressor_path: str = field(default=None, metadata={"help": "Optional regressor ckpt to map hw_emb -> embedding (bypass aligner)"})
    hw_prototype_path: str = field(default=None, metadata={"help": "Optional prototype embedding pt (precomputed), bypass aligner/regressor"})
    hw_name: str = field(default=None, metadata={"help": "Hardware name used to lookup prototype embedding (e.g., nvidia/nvidia-a40)"})
    hw_token: str = field(default="[MASK]", metadata={"help": "Token placeholder used for hardware injection"})

    # device: str = field(default="cuda:0", metadata={"help": ""})
    allow_repeat: bool = field(default=True, metadata={"help": ""})
    is_build: bool = field(default=False, metadata={"help": ""})
    debug_only_generate: bool = field(default=False, metadata={"help": "仅调 gen_func 观察生成结果，不走 TVM gen_states/measure"})


def gen_func(task, states, input, tokenizer, model, device, gen_kwargs, hw_injection_ctx=None):
    if len(states) == 0:
        return []
    hw_placeholder = hw_injection_ctx["placeholder"] if hw_injection_ctx else None
    tokens = input_to_tokens(task, states, input, hw_token_placeholder=hw_placeholder)
    tokenizer.padding_side = "left"
    try:
        # 保持与原版 gen_state.py 一致：使用默认 special token 行为
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
        embed_layer = model.get_input_embeddings()
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = torch.tensor(input_ids_all[start : start + batch_size], dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_mask_all[start : start + batch_size], dtype=torch.long, device=device)

            # 与原版保持一致的长度裁剪
            prompt_len = input_ids.shape[-1] - 1  # 去掉最后一个 special token
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            allowed = tokenizer.model_max_length - input_ids.shape[-1]
            if allowed < 1:
                allowed = 1
            gen_kwargs["max_new_tokens"] = min(gen_kwargs.get("max_new_tokens", 16), allowed)
            gen_kwargs["max_new_tokens"] = max(gen_kwargs["max_new_tokens"], 1)

            # 若有多 MASK，占位符后截断 prompt，强制从 MASK 后生成
            if hw_injection_ctx:
                hw_token_ids = hw_injection_ctx["hw_token_ids"]
                if len(hw_token_ids) > 1:
                    span_positions = []
                    for i in range(input_ids.size(0)):
                        pos = find_subsequence(input_ids[i].tolist(), hw_token_ids)
                        if pos < 0 or pos + len(hw_token_ids) > input_ids.size(1):
                            raise ValueError("HwToken span not found or truncated in prompt")
                        span_positions.append(pos)
                    cut_len = span_positions[0] + len(hw_token_ids)
                    input_ids = input_ids[:, :cut_len]
                    attention_mask = attention_mask[:, :cut_len]
                    prompt_len = input_ids.shape[-1]

            if hw_injection_ctx:
                # 用手写 greedy 解码，绕过 PEFT+inputs_embeds 的潜在问题
                embeds = embed_layer(input_ids)
                gen_ids = generate_with_embeds_greedy(
                    model,
                    embed_layer,
                    embeds,
                    attention_mask,
                    gen_kwargs.get("max_new_tokens", 16),
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                gen_ids = response[:, prompt_len:]

            try:
                log_debug(f"[GEN-DEBUG] gen_sequences shape={list(gen_ids.shape)} (prompt_len={prompt_len})")
            except Exception:
                pass
            gen_ids_list = gen_ids.cpu().tolist()
            for ids in gen_ids_list:
                if len(ids) == 0:
                    # 跳过空生成，避免传递无效记录到 TVM
                    continue
                # 返回 token 级别列表，符合 TVM 期望的 Array[Array[String]]
                toks = tokenizer.convert_ids_to_tokens(ids)
                response_list.append(toks)
    return response_list


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
    elif "4090" in target_lower or "sm_86" in target_lower:
        return "4090"
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


def prepare_hw_injection_context(cfg, tokenizer, model, device):
    # 优先使用预计算的 prototype
    proto_path = cfg.get("hw_prototype_path")
    if proto_path:
        ck = torch.load(proto_path, map_location=device)
        hw_name = cfg.get("hw_name")
        if not hw_name or hw_name not in ck:
            raise ValueError(f"hw_name '{hw_name}' not found in prototype file")
        segs = ck[hw_name]
        vectors = []
        for seg in ["arch", "mem", "cons", "host"]:
            info = segs.get(seg, {})
            if not info.get("active", False):
                # 对 inactive 的段，用全零向量占位，保证长度一致
                vec_dim = next((v.get("vec").shape[-1] for v in segs.values() if v.get("active", False)), None)
                if vec_dim is None:
                    raise ValueError(f"No active segments found in prototype for {hw_name}")
                vectors.append(torch.zeros(vec_dim, device=device))
            else:
                vectors.append(info["vec"].to(device))
        hw_vec = torch.stack(vectors, dim=0)  # (4, D)
        hw_token = cfg.get("hw_token", "[MASK]")
        hw_token_ids = tokenizer(hw_token, add_special_tokens=False)["input_ids"]
        if len(hw_token_ids) != hw_vec.size(0):
            raise ValueError(f"hw_token length {len(hw_token_ids)} != prototype vec count {hw_vec.size(0)}")
        return {
            "mode": "prototype",
            "hw_vec_proto": hw_vec,  # (4,D)
            "hw_vec": torch.tensor(cfg["hw_vec"], dtype=torch.float32, device=device),  # 仍保留原始 hw_vec 以备需要
            "hw_token_ids": hw_token_ids,
            "placeholder": hw_token,
            "patch_embedding": True,
        }

    reg_path = cfg.get("hw_regressor_path")
    hw_token = cfg.get("hw_token", "[MASK]")
    hw_token_ids = tokenizer(hw_token, add_special_tokens=False)["input_ids"]
    if not hw_token_ids:
        raise ValueError(f"Tokenizer failed to tokenize hw_token '{hw_token}'")
    if reg_path:
        hw_vec = torch.tensor(cfg["hw_vec"], dtype=torch.float32, device=device)
        ck_reg = torch.load(reg_path, map_location=device)
        in_dim, out_dim = ck_reg["in_dim"], ck_reg["out_dim"]
        hidden = ck_reg.get("hidden", 256)
        reg = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        ).to(device)
        state = ck_reg["state_dict"]
        if any(k.startswith("net.") for k in state.keys()):
            state = {k.replace("net.", ""): v for k, v in state.items()}
        reg.load_state_dict(state, strict=False)
        reg.eval()
        return {
            "mode": "regressor",
            "regressor": reg,
            "hw_vec": hw_vec,
            "hw_token_ids": hw_token_ids,
            "placeholder": hw_token,
            "patch_embedding": True,
        }
    else:
        ckpt = torch.load(cfg["aligner_path"], map_location=device)
        proto = torch.tensor(ckpt["prototype_keys"], dtype=torch.float32, device=device)
        embed_layer = model.get_input_embeddings()
        embed_dim = ckpt.get("embed_dim", embed_layer.embedding_dim)
        cfg_meta = ckpt.get("config", {})
        temperature = cfg_meta.get("temperature", ckpt.get("temperature", 1.0))
        trainable_temp = cfg_meta.get("trainable_temperature", False)
        aligner = ProtoMixAligner(proto, embed_dim=embed_dim, temperature=temperature, trainable_temperature=trainable_temp)
        aligner.load_state_dict(ckpt["state_dict"])
        aligner.to(device)
        aligner.eval()
        hw_vec = torch.tensor(cfg["hw_vec"], dtype=torch.float32, device=device)
        return {
            "mode": "aligner",
            "aligner": aligner,
            "hw_vec": hw_vec,
            "hw_token_ids": hw_token_ids,
            "placeholder": hw_token,
            "patch_embedding": True,
        }


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
            expert = GatedLoRAExpert(delta_wrapper, r_dim=r_vector.numel(), init_router=r_vector, reset_lora=False)
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


def worker(err_queue, save_path_i, sketch_path, gen_kwargs, model_path, adapter_path, multi_adapter_dir, target_hardware, original_target, device, allow_repeat, keep_cnt, is_build, worker_id, num_workers, hw_injection_cfg=None, log_file_path=None, debug_only_generate=False):
    try:
        # 子进程内设置 logging（追加同一个文件）
        global LOG_FILE_PATH
        if log_file_path:
            LOG_FILE_PATH = log_file_path
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
            hw_injection=bool(hw_injection_cfg),
            hw_aligner_path=hw_injection_cfg["aligner_path"] if hw_injection_cfg else None,
            hw_token=hw_injection_cfg["hw_token"] if hw_injection_cfg else "[MASK]",
            allow_repeat=allow_repeat,
            is_build=is_build,
            debug_only_generate=debug_only_generate,
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
        hw_injection_ctx = None
        if hw_injection_cfg:
            hw_injection_ctx = prepare_hw_injection_context(hw_injection_cfg, tokenizer, model, device)
            # 预先将 HwToken 对应的 embedding 覆盖为 aligner/regressor 输出，避免 inputs_embeds 路径
            if hw_injection_ctx.get("patch_embedding", False):
                with torch.no_grad():
                    hw_token_ids = hw_injection_ctx["hw_token_ids"]
                    if hw_injection_ctx["mode"] == "prototype":
                        hw_embed = hw_injection_ctx["hw_vec_proto"].unsqueeze(0)  # (1,4,D)
                    else:
                        hw_vec_tensor = hw_injection_ctx["hw_vec"].unsqueeze(0)
                        if hw_injection_ctx["mode"] == "regressor":
                            hw_embed = hw_injection_ctx["regressor"](hw_vec_tensor)  # (1, D)
                            hw_embed = hw_embed.unsqueeze(1).expand(-1, len(hw_token_ids), -1)  # (1, T, D)
                        else:
                            if len(hw_token_ids) > 1:
                                hw_embed = hw_injection_ctx["aligner"](hw_vec_tensor, split_heads=True)  # (1, T, D)
                            else:
                                hw_embed = hw_injection_ctx["aligner"](hw_vec_tensor)
                                hw_embed = hw_embed.unsqueeze(1)
                    if hw_embed.size(1) != len(hw_token_ids):
                        raise ValueError(
                            f"Aligner/regressor output heads ({hw_embed.size(1)}) != hw_token length ({len(hw_token_ids)})"
                        )
                    emb_weight = model.get_input_embeddings().weight
                    alpha = 0.3  # residual blend ratio
                    for idx, tok_id in enumerate(hw_token_ids):
                        orig = emb_weight.data[tok_id]
                        emb_weight.data[tok_id] = (1 - alpha) * orig + alpha * hw_embed[0, idx].to(device)
                    log_debug(f"[PATCH] Residual-patched embedding rows for hw tokens: {hw_token_ids}, alpha={alpha}")
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
                debug_ctx = {
                    "last_input_json": None,
                    "last_states_json": None,
                    "last_prompts": None,
                    "last_generations": None,
                    "hw_info": target_hardware,
                }

                # Debug 分支：仅调用 gen_func 观察输出，不走 TVM gen_states/measure
                if hw_injection_cfg is not None and getattr(script_args, "debug_only_generate", False):
                    states_dbg = [inp.state for inp in inputs][:1]
                    gens = gen_func(
                        inputs[0].task,
                        states_dbg,
                        inputs[0],
                        tokenizer,
                        model,
                        device,
                        {"max_new_tokens": 16},
                        hw_injection_ctx=hw_injection_ctx,
                    )
                    log_debug(f"[DEBUG-ONLY] workload={workload_key} gen_func output: {gens[:1]}")
                    continue

                def gen_func_inner(task, states, max_new_tokens):
                    max_new_tokens = max(max_new_tokens, 1)
                    gen_kwargs["max_new_tokens"] = max_new_tokens
                    try:
                        debug_ctx["last_input_json"] = inputs[0].to_json()
                    except Exception:
                        debug_ctx["last_input_json"] = None
                    try:
                        debug_ctx["last_states_json"] = [s.to_json() for s in states]
                    except Exception:
                        debug_ctx["last_states_json"] = [str(s) for s in states]
                    try:
                        prompts = input_to_tokens(
                            task,
                            states,
                            inputs[0],
                            hw_token_placeholder=hw_injection_ctx["placeholder"] if hw_injection_ctx else None,
                        )
                        debug_ctx["last_prompts"] = prompts
                    except Exception:
                        debug_ctx["last_prompts"] = None

                    generations = gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs, hw_injection_ctx=hw_injection_ctx)
                    # 将生成结果转为可读字符串，便于日志调试
                    decoded_generations = []
                    try:
                        for g in generations:
                            if isinstance(g, list):
                                decoded_generations.append(tokenizer.convert_tokens_to_string(g))
                            else:
                                decoded_generations.append(str(g))
                    except Exception:
                        decoded_generations = generations
                    debug_ctx["last_generations"] = decoded_generations
                    return generations

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
                        if not all_state_list:
                            log_debug(f"Worker {worker_id}: workload {workload_key} gen_states returned empty list")
                            diagnose_state_failure(worker_id, workload_key, debug_ctx)
                            retry_i += 1
                            continue
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

                        if len(measure_inputs_tmp) == 0:
                            log_debug(f"Worker {worker_id}: workload {workload_key} no measure_inputs generated")
                            diagnose_state_failure(worker_id, workload_key, debug_ctx)
                            retry_i += 1
                            continue

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
                        if len(measure_inputs) >= keep_cnt:
                            break
                    except Exception as e:
                        import traceback
                        msg = str(e)
                        print(f"Worker {worker_id}: workload {workload_key} 第{retry_i+1}次重试时出错: {msg}")
                        log_debug(msg)
                        log_debug(traceback.format_exc())
                        if "All states are invalid" in msg or "Internal error" in msg:
                            diagnose_state_failure(worker_id, workload_key, debug_ctx)
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

    hw_injection_cfg = None
    if script_args.hw_injection:
        if not (script_args.hw_aligner_path or script_args.hw_regressor_path or script_args.hw_prototype_path):
            raise ValueError("--hw_injection requires --hw_aligner_path or --hw_regressor_path or --hw_prototype_path")
        embeddings = load_hardware_embeddings(script_args.edge_embedding_path)
        hw_id = (script_args.target_hardware or extract_hardware_id_from_target(script_args.target)).lower()
        hw_name, hw_vec = resolve_hw_embedding(hw_id, embeddings)
        hw_injection_cfg = {
            "aligner_path": script_args.hw_aligner_path,
            "hw_regressor_path": script_args.hw_regressor_path,
            "hw_prototype_path": script_args.hw_prototype_path,
            "hw_vec": hw_vec,
            "hw_token": script_args.hw_token,
            "hw_name": hw_name,
        }

        # 若未显式指定 adapter_path，且对齐器 ckpt 所在目录包含 LoRA 适配器，则自动启用 LoRA（Mode B）
        if script_args.adapter_path is None and script_args.hw_aligner_path:
            ckpt_dir = os.path.dirname(script_args.hw_aligner_path)
            adapter_file = os.path.join(ckpt_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                script_args.adapter_path = ckpt_dir
                print(f"Detected LoRA adapter at {ckpt_dir}, enabling HwToken injection with LoRA (Mode B).")

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
    import tempfile
    tmp_folder = tempfile.mkdtemp(prefix=".gen_state_")
    err_queue = Queue()

    # 设置日志写入文件（使用简单 file append，log_debug 会直接写）
    log_file = os.path.join(os.path.dirname(script_args.save_path) or ".", "gen_state_debug.log")
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_file
    print(f"[DEBUG] Logging debug info to {log_file}")
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
        p = Process(target=worker, args=(err_queue, save_path_i, script_args.sketch_path, gen_kwargs, script_args.model_path, script_args.adapter_path, script_args.multi_adapter_dir, script_args.target_hardware, script_args.target, device, script_args.allow_repeat, script_args.keep_cnt, script_args.is_build, gpu_i, num_gpus, hw_injection_cfg, log_file, script_args.debug_only_generate))
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
        shutil.rmtree(tmp_folder, ignore_errors=True)
    


if __name__ == "__main__":
    main()
