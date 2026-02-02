import glob
import json
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

# KV injection invariants:
# - attention_mask length == prefix_len + prompt_len (+ step in generate loop)
# - position_ids must be explicitly advanced during generation when prefix_len > 0
# - past_len after prefill should equal prefix_len + (prompt_len - 1)
# - when prefix_len > 0, prompt position_ids should still start at 0 if pos_compensate
# 简易日志文件路径
LOG_FILE_PATH = None


def log_debug(msg: str):
    if LOG_FILE_PATH:
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

import torch
import tqdm
import tvm
from tvm import auto_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from common import load_and_register_tasks, register_data_path
from hw_kv_aligner import HwKVAligner
from make_dataset import input_to_tokens
from postprocess import check_measured

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Path to base or fully-finetuned model"})
    sketch_path: str = field(metadata={"help": "Path to input sketches (measure records)"})
    save_path: str = field(metadata={"help": "Path to save generated measure records"})
    keep_cnt: int = field(metadata={"help": "Max records to keep per workload"})
    target: str = field(metadata={"help": "Target string or shortcut (e.g., 4090)"})

    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Optional tokenizer path (defaults to model_path)"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Optional single LoRA/PEFT adapter path"})
    hardware_embedding_path: str = field(
        default="Embedding/hardware_embeddings_v4_universe.json",
        metadata={"help": "Path to hardware_embeddings_v4_universe.json"},
    )
    target_hardware: Optional[str] = field(
        default=None,
        metadata={"help": "Logical hardware id; if None, inferred from --target"},
    )
    allow_repeat: bool = field(default=True, metadata={"help": "Allow repeated measure inputs"})
    is_build: bool = field(default=False, metadata={"help": "Actually build schedules instead of using placeholders"})
    batch_size: int = field(default=64, metadata={"help": "Generation batch size (default 64; set 1 to disable padding effects)"})
    use_bucket: bool = field(default=False, metadata={"help": "Use bucket tokens when forming text prompts"})
    use_hw_kv: bool = field(default=False, metadata={"help": "Enable HwKVAligner KV injection"})
    hw_kv_aligner_path: Optional[str] = field(default=None, metadata={"help": "Optional HwKVAligner checkpoint path"})
    hw_kv_num_slots: int = field(default=4, metadata={"help": "Number of KV prefix slots (default 4)"})
    hw_kv_mode: Optional[str] = field(
        default=None,
        metadata={"help": "KV mode: None/baseline (no prefix), 'zero' (zero past), or 'real' (aligner output)"},
    )
    pos_compensate: bool = field(default=False, metadata={"help": "When using KV, force prompt position_ids to start at 0 (ignore prefix len)"})
    pos_shift: int = field(default=0, metadata={"help": "Force position_ids += pos_shift when no KV is used"})
    zero_prefix_mask_one: bool = field(default=False, metadata={"help": "In zero mode, force prefix_mask to all 1 instead of 0"})
    debug_logits_compare: bool = field(
        default=False,
        metadata={"help": "Debug: compare baseline vs zero-past logits on last prompt token (one batch)"},
    )
    debug_prepare_inputs: bool = field(
        default=False,
        metadata={"help": "Debug: log prepare_inputs_for_generation outputs (position_ids/cache_position)"},
    )
    debug_forward_trace: int = field(
        default=0,
        metadata={"help": "Debug: trace first N model.forward calls (0 disables logging)"},
    )
    debug_manual_greedy_steps: int = field(
        default=0,
        metadata={"help": "Debug: run manual greedy loop for N steps (0 disables)"},
    )
    debug_manual_greedy_count: int = field(
        default=3,
        metadata={"help": "Debug: number of samples for manual greedy compare"},
    )
    debug_prefill_equiv_kv: bool = field(
        default=False,
        metadata={"help": "Debug: compare KV full forward vs split prefill equivalence"},
    )
    debug_kv_stats: bool = field(
        default=False,
        metadata={"help": "Debug: log KV mean/max stats for last 4 layers (real mode)"},
    )
    debug_prefill_equiv: bool = field(
        default=False,
        metadata={"help": "Debug: compare direct generate vs split prefill (no KV)"},
    )


DEFAULT_HW_NAME_MAP = {
    "v100": "nvidia/nvidia-v100",
    "nvidia-v100": "nvidia/nvidia-v100",
    "rtx-4090": "nvidia/rtx-4090",
    "4090": "nvidia/rtx-4090",
    "geforce-rtx-3090": "nvidia/geforce-rtx-3090",
    "3090": "nvidia/geforce-rtx-3090",
    "jetson-agx-xavier": "nvidia/jetson-agx-xavier",
    "xavier": "nvidia/jetson-agx-xavier",
    "xeon": "aws/cpu/c5.18xlarge",
    "c5.18xlarge": "aws/cpu/c5.18xlarge",
}


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def resolve_hw_embedding(hw_id: str, embeddings: Dict[str, List[float]]) -> Tuple[Optional[str], Optional[List[float]]]:
    if hw_id in embeddings:
        return hw_id, embeddings[hw_id]

    hw_id_norm = hw_id.lower()
    mapped = DEFAULT_HW_NAME_MAP.get(hw_id_norm)
    if mapped and mapped in embeddings:
        return mapped, embeddings[mapped]

    for name, vec in embeddings.items():
        if hw_id_norm in name.lower():
            return name, vec

    return None, None


class DebugUtils:
    @staticmethod
    def hw_vec_stats(name: str, vec: Optional[List[float]]) -> str:
        if vec is None:
            return f"[HW-VEC] {name}: not found"
        if not vec:
            return f"[HW-VEC] {name}: empty vector"
        mean_abs = sum(abs(x) for x in vec) / len(vec)
        max_abs = max(abs(x) for x in vec)
        first8 = vec[:8]
        return f"[HW-VEC] {name}: mean_abs={mean_abs:.6f} max_abs={max_abs:.6f} first8={first8}"

    @staticmethod
    def kv_stats(tag: str, k: torch.Tensor, v: torch.Tensor) -> str:
        k_abs = k.abs()
        v_abs = v.abs()
        k_nz = (k_abs > 1e-12).sum().item()
        v_nz = (v_abs > 1e-12).sum().item()
        return (
            f"[KV-STATS {tag}] "
            f"k_mean_abs={k_abs.mean().item():.3e} k_max_abs={k_abs.max().item():.3e} "
            f"k_nz={k_nz}/{k.numel()} "
            f"v_mean_abs={v_abs.mean().item():.3e} v_max_abs={v_abs.max().item():.3e} "
            f"v_nz={v_nz}/{v.numel()}"
        )


def _build_zero_past(n_layer: int, n_head: int, head_dim: int, batch_size: int, prefix_len: int, device, dtype):
    past = []
    for _ in range(n_layer):
        k = torch.zeros(batch_size, n_head, prefix_len, head_dim, device=device, dtype=dtype)
        v = torch.zeros(batch_size, n_head, prefix_len, head_dim, device=device, dtype=dtype)
        past.append((k, v))
    return tuple(past)


def prepare_hw_kv_context(
    use_hw_kv: bool,
    hw_vec: Optional[List[float]],
    model: torch.nn.Module,
    device: torch.device,
    hw_kv_aligner_path: Optional[str] = None,
    hw_kv_mode: Optional[str] = None,
    hw_kv_num_slots: int = 4,
):
    if not use_hw_kv:
        return None

    mode = hw_kv_mode or "real"

    n_layer = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))
    n_head = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
    hidden_size = getattr(model.config, "hidden_size", None)
    if n_layer is None or n_head is None or hidden_size is None:
        raise ValueError("Model config missing n_layer/n_head/hidden_size for HwKVAligner.")
    head_dim = hidden_size // n_head

    if mode == "zero":
        # Only keep shape info; past will be built on the fly
        return {
            "mode": "zero",
            "n_layer": n_layer,
            "n_head": n_head,
            "head_dim": head_dim,
            "prefix_len": hw_kv_num_slots,
        }
    if mode == "noop":
        # Explicit no-op mode: behave exactly like baseline (no prefix, no past)
        return {"mode": "noop"}

    # mode == real
    if not hw_kv_aligner_path:
        print("[WARN] --use_hw_kv is set but no --hw_kv_aligner_path provided; skip KV injection.")
        log_debug("[WARN] --use_hw_kv is set but no --hw_kv_aligner_path provided; skip KV injection.")
        return None

    if hw_vec is None:
        print("[WARN] No hardware embedding found; using zeros for HwKVAligner.")
        log_debug("[WARN] No hardware embedding found; using zeros for HwKVAligner.")
        hw_vec = [0.0] * 24

    hw_vec_t = torch.tensor(hw_vec, dtype=torch.float32, device=device)
    if hw_vec_t.dim() == 1:
        hw_vec_t = hw_vec_t.unsqueeze(0)

    kv_aligner = HwKVAligner(
        llm_num_layers=n_layer,
        llm_num_heads=n_head,
        llm_head_dim=head_dim,
        hw_dim=hw_vec_t.size(-1),
        num_slots=hw_kv_num_slots,
        backward_depth=min(4, n_layer),
        linker_temperature=1.0,
    )
    if hw_kv_aligner_path:
        try:
            kv_state = torch.load(hw_kv_aligner_path, map_location=device)
            state = kv_state.get("state_dict", kv_state)
            kv_aligner.load_state_dict(state, strict=False)
            print(f"Loaded HwKVAligner checkpoint from {hw_kv_aligner_path}")
            log_debug(f"[INFO] Loaded HwKVAligner checkpoint from {hw_kv_aligner_path}")
        except Exception as e:
            print(f"[WARN] Failed to load HwKVAligner checkpoint: {e}")
            log_debug(f"[WARN] Failed to load HwKVAligner checkpoint: {e}")
    kv_aligner.to(device)
    kv_aligner.eval()
    return {"mode": "real", "kv_aligner": kv_aligner, "hw_vec": hw_vec_t}


def load_model_for_inference(args: ScriptArguments):
    print("Loading model for inference with mode detection...")
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter_path")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    if args.debug_forward_trace and args.debug_forward_trace > 0:
        import inspect
        import functools

        orig_forward = model.forward
        orig_sig = inspect.signature(orig_forward)
        trace_limit = int(args.debug_forward_trace)

        @functools.wraps(orig_forward)
        def _forward_wrapper(*f_args, **f_kwargs):
            if not hasattr(_forward_wrapper, "_count"):
                _forward_wrapper._count = 0
            if _forward_wrapper._count < trace_limit:
                _forward_wrapper._count += 1
                input_ids = f_kwargs.get("input_ids")
                attention_mask = f_kwargs.get("attention_mask")
                position_ids = f_kwargs.get("position_ids")
                past = f_kwargs.get("past_key_values")
                if input_ids is not None:
                    log_debug(f"[FORWARD] input_ids shape={tuple(input_ids.shape)}")
                if attention_mask is not None:
                    log_debug(f"[FORWARD] attention_mask shape={tuple(attention_mask.shape)}")
                if position_ids is None:
                    log_debug("[FORWARD] position_ids=None")
                else:
                    pos_sample = position_ids[0].tolist()
                    log_debug(
                        f"[FORWARD] position_ids shape={tuple(position_ids.shape)} head={pos_sample[:6]} tail={pos_sample[-6:]}"
                    )
                if past is None:
                    log_debug("[FORWARD] past_key_values=None")
                else:
                    try:
                        past_len = past[0][0].shape[2]
                        log_debug(f"[FORWARD] past_len={past_len}")
                    except Exception:
                        log_debug("[FORWARD] past_len=unavailable")
            return orig_forward(*f_args, **f_kwargs)

        # Preserve original signature so generate() validation can see supported kwargs
        _forward_wrapper.__signature__ = orig_sig

        model.forward = _forward_wrapper

    print("Model loading completed successfully!")
    return model, tokenizer


def gen_func(
    task,
    states,
    input_item,
    tokenizer,
    model,
    device,
    gen_kwargs,
    use_bucket: bool = False,
    hw_kv_ctx=None,
    debug_prefill_equiv: bool = False,
    pos_shift: int = 0,
    pos_compensate: bool = False,
    zero_prefix_mask_one: bool = False,
    batch_size: int = 64,
    kv_num_slots: int = 4,
    debug_logits_compare: bool = False,
    debug_prepare_inputs: bool = False,
    debug_manual_greedy_steps: int = 0,
    debug_manual_greedy_count: int = 3,
    debug_prefill_equiv_kv: bool = False,
    debug_kv_stats: bool = False,
):
    if len(states) == 0:
        return []

    debug_prompts = None
    debug_generations = None

    tokens = input_to_tokens(
        task,
        states,
        input_item,
        use_bucket=use_bucket,
    )
    debug_prompts = tokens
    tokenizer.padding_side = "left"
    batch = tokenizer(tokens, padding=True, max_length=None)
    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = torch.tensor(input_ids_all[start : start + batch_size], dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_mask_all[start : start + batch_size], dtype=torch.long, device=device)

            # 回归旧版行为：裁掉最后一个 token（通常是结尾 special token）
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

            gen_kwargs["max_new_tokens"] = min(
                gen_kwargs["max_new_tokens"], tokenizer.model_max_length - input_ids.shape[-1]
            )

            extra_kwargs = {}
            input_ids_for_gen = input_ids
            attention_mask_for_gen = attention_mask
            hw_past = None

            if debug_manual_greedy_steps > 0 and not getattr(gen_func, "_logged_manual_greedy", False):
                try:
                    n_layer = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))
                    n_head = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
                    hidden_size = getattr(model.config, "hidden_size", None)
                    if n_layer is None or n_head is None or hidden_size is None:
                        raise ValueError("Model config missing n_layer/n_head/hidden_size for manual greedy")
                    head_dim = hidden_size // n_head
                    dtype = model.dtype if hasattr(model, "dtype") else torch.float32

                    sample_cnt = min(debug_manual_greedy_count, input_ids.size(0))
                    for i in range(sample_cnt):
                        seq_len = int(attention_mask[i].sum().item())
                        prompt_ids = input_ids[i, -seq_len:].unsqueeze(0)
                        prompt_mask = torch.ones(1, seq_len, device=device, dtype=attention_mask.dtype)

                        # Baseline greedy
                        base_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
                        base_past = base_out.past_key_values
                        base_logits = base_out.logits[:, -1, :]
                        base_tokens = []
                        base_mask = prompt_mask
                        for t in range(debug_manual_greedy_steps):
                            next_token = torch.argmax(base_logits, dim=-1)
                            base_tokens.append(next_token.item())
                            base_mask = torch.cat(
                                [base_mask, torch.ones(1, 1, device=device, dtype=base_mask.dtype)], dim=1
                            )
                            base_out = model(
                                input_ids=next_token.unsqueeze(0),
                                attention_mask=base_mask,
                                past_key_values=base_past,
                                use_cache=True,
                            )
                            base_past = base_out.past_key_values
                            base_logits = base_out.logits[:, -1, :]

                        # Zero-past greedy (slots = kv_num_slots)
                        zero_past = _build_zero_past(
                            n_layer, n_head, head_dim, 1, kv_num_slots, device, dtype
                        )
                        prefix_mask = torch.zeros(1, kv_num_slots, device=device, dtype=prompt_mask.dtype)
                        if zero_prefix_mask_one:
                            prefix_mask = torch.ones(1, kv_num_slots, device=device, dtype=prompt_mask.dtype)
                        full_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

                        if seq_len > 1:
                            pre_ids = prompt_ids[:, :-1]
                            pre_mask = full_mask[:, :-1]
                            pos_prefill = None
                            if pos_compensate:
                                pos_prefill = (
                                    torch.arange(pre_ids.size(1), device=device).unsqueeze(0)
                                )
                            pre_out = model(
                                input_ids=pre_ids,
                                attention_mask=pre_mask,
                                position_ids=pos_prefill,
                                past_key_values=zero_past,
                                use_cache=True,
                            )
                            z_past = pre_out.past_key_values
                            z_input = prompt_ids[:, -1:]
                        else:
                            z_past = zero_past
                            z_input = prompt_ids

                        pos_gen = None
                        if pos_compensate:
                            pos_gen = torch.tensor([[seq_len - 1]], device=device)
                        z_out = model(
                            input_ids=z_input,
                            attention_mask=full_mask,
                            position_ids=pos_gen,
                            past_key_values=z_past,
                            use_cache=True,
                        )
                        z_past = z_out.past_key_values
                        z_logits = z_out.logits[:, -1, :]
                        z_tokens = []
                        z_mask = full_mask
                        for t in range(debug_manual_greedy_steps):
                            next_token = torch.argmax(z_logits, dim=-1)
                            z_tokens.append(next_token.item())
                            z_mask = torch.cat(
                                [z_mask, torch.ones(1, 1, device=device, dtype=z_mask.dtype)], dim=1
                            )
                            pos_next = None
                            if pos_compensate:
                                pos_next = torch.tensor([[seq_len + t]], device=device)
                            z_out = model(
                                input_ids=next_token.unsqueeze(0),
                                attention_mask=z_mask,
                                position_ids=pos_next,
                                past_key_values=z_past,
                                use_cache=True,
                            )
                            z_past = z_out.past_key_values
                            z_logits = z_out.logits[:, -1, :]

                        # Compare token streams
                        diff_pos = None
                        for t, (a, b) in enumerate(zip(base_tokens, z_tokens)):
                            if a != b:
                                diff_pos = t
                                break
                        if diff_pos is None:
                            log_debug(
                                f"[MANUAL-GREEDY] sample={i} steps={debug_manual_greedy_steps} match"
                            )
                        else:
                            tok_a = tokenizer.convert_ids_to_tokens(base_tokens[diff_pos])
                            tok_b = tokenizer.convert_ids_to_tokens(z_tokens[diff_pos])
                            log_debug(
                                f"[MANUAL-GREEDY] sample={i} steps={debug_manual_greedy_steps} diverge_at={diff_pos} tok_base='{tok_a}' tok_zero='{tok_b}'"
                            )
                except Exception as e:
                    log_debug(f"[MANUAL-GREEDY ERROR] {e}")
                gen_func._logged_manual_greedy = True

            if debug_logits_compare and not getattr(gen_func, "_logged_logits", False):
                try:
                    n_layer = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))
                    n_head = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
                    hidden_size = getattr(model.config, "hidden_size", None)
                    if n_layer is None or n_head is None or hidden_size is None:
                        raise ValueError("Model config missing n_layer/n_head/hidden_size for logits compare")
                    head_dim = hidden_size // n_head
                    dtype = model.dtype if hasattr(model, "dtype") else torch.float32

                    base_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    base_logits = base_out.logits[:, -1, :]

                    zero_past = _build_zero_past(
                        n_layer, n_head, head_dim, input_ids.size(0), kv_num_slots, device, dtype
                    )

                    prefix_mask0 = torch.zeros(
                        input_ids.size(0), kv_num_slots, device=attention_mask.device, dtype=attention_mask.dtype
                    )
                    prefix_mask1 = torch.ones(
                        input_ids.size(0), kv_num_slots, device=attention_mask.device, dtype=attention_mask.dtype
                    )
                    full_mask0 = torch.cat([prefix_mask0, attention_mask], dim=1)
                    full_mask1 = torch.cat([prefix_mask1, attention_mask], dim=1)

                    out0 = model(input_ids=input_ids, attention_mask=full_mask0, past_key_values=zero_past)
                    out1 = model(input_ids=input_ids, attention_mask=full_mask1, past_key_values=zero_past)
                    logits0 = out0.logits[:, -1, :]
                    logits1 = out1.logits[:, -1, :]

                    def _kl(p_logits, q_logits):
                        p = torch.log_softmax(p_logits, dim=-1)
                        q = torch.log_softmax(q_logits, dim=-1)
                        return torch.sum(torch.exp(p) * (p - q)).item()

                    kl0 = _kl(base_logits[0], logits0[0])
                    kl1 = _kl(base_logits[0], logits1[0])

                    def _topk(logits, k=5):
                        vals, idx = torch.topk(logits, k)
                        toks = [tokenizer.convert_ids_to_tokens(i.item()) for i in idx]
                        return list(zip(toks, [v.item() for v in vals]))

                    log_debug(
                        f"[LOGITS-COMPARE] slots={kv_num_slots} KL(base||zero_mask0)={kl0:.4f} KL(base||zero_mask1)={kl1:.4f}"
                    )
                    log_debug(f"[LOGITS-COMPARE] base_top5={_topk(base_logits[0])}")
                    log_debug(f"[LOGITS-COMPARE] zero_mask0_top5={_topk(logits0[0])}")
                    log_debug(f"[LOGITS-COMPARE] zero_mask1_top5={_topk(logits1[0])}")
                except Exception as e:
                    log_debug(f"[LOGITS-COMPARE ERROR] {e}")
                gen_func._logged_logits = True

            if hw_kv_ctx is not None:
                mode = hw_kv_ctx.get("mode", "real")
                hw_past = None

                if mode == "noop":
                    # behave exactly like baseline: no prefix, no past
                    pass
                elif mode == "zero":
                    prefix_len = hw_kv_ctx.get("prefix_len", 4)
                    dtype = model.dtype if hasattr(model, "dtype") else torch.float32
                    hw_past = _build_zero_past(
                        hw_kv_ctx["n_layer"],
                        hw_kv_ctx["n_head"],
                        hw_kv_ctx["head_dim"],
                        input_ids.size(0),
                        prefix_len,
                        device,
                        dtype,
                    )
                else:
                    kv_aligner = hw_kv_ctx["kv_aligner"]
                    hw_vec = hw_kv_ctx["hw_vec"].to(device)

                    # 1. HW past (force num_beams=1; beam expansion handled by HF generate)
                    if hw_vec.size(0) != input_ids.size(0):
                        hw_vec = hw_vec.expand(input_ids.size(0), -1)
                    hw_past = kv_aligner(hw_vec, batch_size=input_ids.size(0), num_beams=1)

                if debug_kv_stats and mode == "real" and hw_past is not None and not getattr(gen_func, "_logged_kv_stats", False):
                    try:
                        total_layers = len(hw_past)
                        start = max(0, total_layers - 4)
                        for layer_idx in range(start, total_layers):
                            raw_k = hw_past[layer_idx][0]
                            raw_v = hw_past[layer_idx][1]
                            log_debug(DebugUtils.kv_stats(f"raw L{layer_idx}", raw_k, raw_v))
                            # 当前 gen_state 未对 KV 做额外 scale/mask，这里 post 与 raw 等价
                            log_debug(DebugUtils.kv_stats(f"post L{layer_idx}", raw_k, raw_v))
                    except Exception as e:
                        log_debug(f"[KV-STATS ERROR] {e}")
                    gen_func._logged_kv_stats = True

                if hw_past is not None:
                    # 2. Full mask = HW prefix + prompt
                    prefix_len = hw_past[0][0].shape[2]
                    # Zero-mode 对照：默认 prefix_mask 设为全 0；若 zero_prefix_mask_one=True 则设为全 1
                    if mode == "zero" and not zero_prefix_mask_one:
                        prefix_mask = torch.zeros(
                            input_ids.size(0),
                            prefix_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    else:
                        prefix_mask = torch.ones(
                            input_ids.size(0),
                            prefix_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                    # 3. Split prefill: keep last token for generate, prefill the rest with HW past
                    seq_len = input_ids.size(1)
                    if seq_len > 1:
                        input_ids_prefill = input_ids[:, :-1]
                        mask_prefill = full_attention_mask[:, :-1]
                        if pos_compensate:
                            prompt_pos = attention_mask.long().cumsum(-1) - 1
                            prompt_pos = prompt_pos.masked_fill(attention_mask == 0, 0)
                            pos_prefill = prompt_pos[:, :-1]
                        else:
                            pos_prefill = None
                        with torch.no_grad():
                            outputs_prefill = model(
                                input_ids=input_ids_prefill,
                                attention_mask=mask_prefill,
                                position_ids=pos_prefill,
                                past_key_values=hw_past,
                                use_cache=True,
                            )
                        final_past = outputs_prefill.past_key_values
                        input_ids_for_gen = input_ids[:, -1:]  # only last token drives generation
                    else:
                        final_past = hw_past
                        input_ids_for_gen = input_ids

                    if pos_compensate:
                        prompt_pos = attention_mask.long().cumsum(-1) - 1
                        prompt_pos = prompt_pos.masked_fill(attention_mask == 0, 0)
                        pos_gen = prompt_pos[:, -1:].clone()
                        extra_kwargs["position_ids"] = pos_gen

                    extra_kwargs["past_key_values"] = final_past
                    attention_mask_for_gen = full_attention_mask

                    if debug_prefill_equiv_kv and not getattr(gen_func, "_logged_prefill_equiv_kv", False):
                        try:
                            # Full forward with hw_past
                            full_out = model(
                                input_ids=input_ids,
                                attention_mask=full_attention_mask,
                                past_key_values=hw_past,
                                use_cache=True,
                            )
                            full_logits = full_out.logits[:, -1, :]
                            full_top = torch.argmax(full_logits, dim=-1)

                            # Split prefill + last-token forward (same as generation prefill)
                            split_out = model(
                                input_ids=input_ids_for_gen,
                                attention_mask=full_attention_mask,
                                past_key_values=final_past,
                                use_cache=True,
                            )
                            split_logits = split_out.logits[:, -1, :]
                            split_top = torch.argmax(split_logits, dim=-1)

                            if torch.equal(full_top, split_top):
                                log_debug("[PREFILL-EQUIV-KV] top1 match")
                            else:
                                tok_full = tokenizer.convert_ids_to_tokens(full_top[0].item())
                                tok_split = tokenizer.convert_ids_to_tokens(split_top[0].item())
                                log_debug(
                                    f"[PREFILL-EQUIV-KV] top1 mismatch tok_full='{tok_full}' tok_split='{tok_split}'"
                                )
                        except Exception as e:
                            log_debug(f"[PREFILL-EQUIV-KV ERROR] {e}")
                        gen_func._logged_prefill_equiv_kv = True
                else:
                    input_ids_for_gen = input_ids
                    attention_mask_for_gen = attention_mask
            else:
                input_ids_for_gen = input_ids
                attention_mask_for_gen = attention_mask
                if pos_shift > 0:
                    position_ids = (
                        torch.arange(input_ids_for_gen.size(1), device=device)
                        .unsqueeze(0)
                        .expand(input_ids_for_gen.size(0), -1)
                        + pos_shift
                    )
                    extra_kwargs["position_ids"] = position_ids

            # Debug: compare direct generate vs split prefill (no HW KV) for equivalence
            if debug_prefill_equiv and hw_kv_ctx is None:
                try:
                    # Path P0: direct generate on the full prompt
                    direct_out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )
                    direct_gen = direct_out[:, input_ids.shape[1] :]

                    # Path P1: split prefill without HW KV, then generate from last token
                    if input_ids.shape[1] > 1:
                        prefill_ids = input_ids[:, :-1]
                        prefill_mask = attention_mask[:, :-1]
                        prefill_outputs = model(
                            input_ids=prefill_ids,
                            attention_mask=prefill_mask,
                            use_cache=True,
                        )
                        split_out = model.generate(
                            input_ids=input_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=prefill_outputs.past_key_values,
                            **gen_kwargs,
                        )
                        split_gen = split_out[:, 1:]
                    else:
                        split_gen = direct_gen

                    same_shape = direct_gen.shape == split_gen.shape
                    same_tokens = same_shape and torch.equal(direct_gen, split_gen)
                    if not same_tokens:
                        len1 = direct_gen.shape[1]
                        len2 = split_gen.shape[1]
                        pos = -1
                        tok1 = tok2 = ""
                        min_len = min(len1, len2)
                        for j in range(min_len):
                            if direct_gen[0, j].item() != split_gen[0, j].item():
                                pos = j
                                tok1 = tokenizer.convert_ids_to_tokens(direct_gen[0, j].item())
                                tok2 = tokenizer.convert_ids_to_tokens(split_gen[0, j].item())
                                break
                        log_debug(
                            f"[PREFILL-EQUIV MISMATCH] pos={pos} len1={len1} len2={len2} tok1='{tok1}' tok2='{tok2}' attn_shape={attention_mask.shape}"
                        )
                    else:
                        log_debug(f"[PREFILL-EQUIV OK] len={direct_gen.shape[1]} attn_shape={attention_mask.shape}")
                except Exception as e:
                    log_debug(f"[PREFILL-EQUIV ERROR] {e}")

            if debug_prepare_inputs and not getattr(gen_func, "_logged_prepare_inputs", False):
                try:
                    if hasattr(model, "prepare_inputs_for_generation"):
                        prep = model.prepare_inputs_for_generation(
                            input_ids_for_gen,
                            attention_mask=attention_mask_for_gen,
                            **extra_kwargs,
                        )
                        pos_ids = prep.get("position_ids")
                        cache_pos = prep.get("cache_position")
                        if pos_ids is not None:
                            sample = pos_ids[0].tolist()
                            log_debug(
                                f"[PREPARE-INPUTS] position_ids shape={tuple(pos_ids.shape)} sample_head={sample[:8]} sample_tail={sample[-8:]}"
                            )
                        else:
                            log_debug("[PREPARE-INPUTS] position_ids not present")
                        if cache_pos is not None:
                            cache_sample = cache_pos[0].tolist() if cache_pos.dim() > 0 else cache_pos.tolist()
                            log_debug(
                                f"[PREPARE-INPUTS] cache_position shape={tuple(cache_pos.shape)} sample_head={cache_sample[:8] if isinstance(cache_sample, list) else cache_sample}"
                            )
                        else:
                            log_debug("[PREPARE-INPUTS] cache_position not present")
                    else:
                        log_debug("[PREPARE-INPUTS] model.prepare_inputs_for_generation not available")
                except Exception as e:
                    log_debug(f"[PREPARE-INPUTS ERROR] {e}")
                gen_func._logged_prepare_inputs = True

            if not hasattr(gen_func, "_forward_supports"):
                import inspect

                sig = inspect.signature(model.forward)
                params = sig.parameters.values()
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                supported = {
                    p.name
                    for p in params
                    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                }
                gen_func._forward_supports = {
                    "attention_mask": has_kwargs or "attention_mask" in supported,
                    "position_ids": has_kwargs or "position_ids" in supported,
                }

            supports_attention = gen_func._forward_supports["attention_mask"]
            supports_position = gen_func._forward_supports["position_ids"]
            if not supports_position and "position_ids" in extra_kwargs:
                log_debug("[WARN] model.forward does not accept position_ids; drop it for generate")
                extra_kwargs.pop("position_ids", None)
            if not supports_attention:
                log_debug("[WARN] model.forward does not accept attention_mask; skip passing it to generate")
                attention_mask_for_gen = None

            generate_kwargs = dict(gen_kwargs)
            generate_kwargs.update(extra_kwargs)
            if attention_mask_for_gen is not None:
                generate_kwargs["attention_mask"] = attention_mask_for_gen
            # Patch prepare_inputs_for_generation in KV mode to enforce position_ids per step.
            orig_prepare = None
            if hw_past is not None and hasattr(model, "prepare_inputs_for_generation"):
                orig_prepare = model.prepare_inputs_for_generation
                kv_prefix_len = prefix_len

                def patched_prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, **kwargs):
                    model_inputs = orig_prepare(
                        input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                    attn = model_inputs.get("attention_mask", attention_mask)
                    inp = model_inputs.get("input_ids", input_ids)
                    if attn is not None and attn.dim() == 2 and inp is not None:
                        pos_mask = attn.long()
                        if kv_prefix_len > 0 and pos_mask.shape[1] >= kv_prefix_len:
                            pos_mask[:, :kv_prefix_len] = 0
                        position_ids = pos_mask.cumsum(-1) - 1
                        position_ids = position_ids.masked_fill(position_ids < 0, 0)
                        position_ids = position_ids[:, -inp.shape[1] :].contiguous()
                        model_inputs["position_ids"] = position_ids
                    return model_inputs

                model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation

            try:
                response = model.generate(input_ids=input_ids_for_gen, **generate_kwargs)
            finally:
                if orig_prepare is not None:
                    model.prepare_inputs_for_generation = orig_prepare
            gen_start_idx = input_ids_for_gen.shape[-1]
            response = response[:, gen_start_idx:]
            response_list.extend(response.tolist())

    # Decode to string list (TVM expects decoded schedule strings, not raw tokens)
    decoded = []
    if response_list:
        for ids in response_list:
            try:
                decoded.append(tokenizer.batch_decode([ids], skip_special_tokens=False)[0])
            except Exception:
                decoded.append(str(ids))
    if debug_prompts is not None:
        log_debug("[DEBUG] Prompts:")
        for i, p in enumerate(debug_prompts[:3]):
            log_debug(f"  prompt[{i}]: {p}")
    if decoded:
        log_debug("[DEBUG] Generations (decoded):")
        for i, g in enumerate(decoded[:3]):
            # 使用 decode 保证与旧版一致的可读性
            try:
                log_debug(f"  gen[{i}]: {tokenizer.decode(tokenizer.encode(g), skip_special_tokens=True, clean_up_tokenization_spaces=False)}")
            except Exception:
                log_debug(f"  gen[{i}]: {g}")
    return [tokenizer.batch_decode(item) for item in response_list] if response_list else []


def merge_json_files_safely(tmp_folder, save_path):
    print(f"开始合并JSON文件从 {tmp_folder} 到 {save_path}")

    all_records = []
    part_files = glob.glob(f"{tmp_folder}/*_part")
    part_files.sort()

    total_files = len(part_files)
    valid_files = 0
    total_records = 0

    for part_file in part_files:
        try:
            if os.path.getsize(part_file) > 0:
                valid_files += 1
                file_records = 0
                with open(part_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
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

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(record + "\n")
        print(f"成功写入 {save_path}，共 {len(all_records)} 条记录")
        return len(all_records)
    except Exception as e:
        print(f"写入文件 {save_path} 时出错: {e}")
        raise


def worker(
    err_queue,
    save_path_i,
    sketch_path,
    gen_kwargs,
    model_path,
    tokenizer_path,
    adapter_path,
    target_hardware,
    original_target,
    device,
    allow_repeat,
    keep_cnt,
    is_build,
    worker_id,
    num_workers,
    use_bucket=False,
    hw_kv_cfg=None,
    log_file_path=None,
    debug_prefill_equiv=False,
    pos_shift=0,
    pos_compensate=False,
    zero_prefix_mask_one=False,
    batch_size=64,
    kv_num_slots=4,
    debug_logits_compare=False,
    debug_prepare_inputs=False,
    debug_forward_trace=0,
    debug_manual_greedy_steps=0,
    debug_manual_greedy_count=3,
    debug_prefill_equiv_kv=False,
    debug_kv_stats=False,
):
    try:
        global LOG_FILE_PATH
        if log_file_path:
            LOG_FILE_PATH = log_file_path
        if worker_id == 0 and hw_kv_cfg and hw_kv_cfg.get("hw_vec_stats"):
            for msg in hw_kv_cfg["hw_vec_stats"]:
                print(msg)
                log_debug(msg)
        print(f"Initializing TVM environment in worker for target: {original_target}")
        register_data_path(original_target)
        load_and_register_tasks()

        sketch_path_to_use = sketch_path
        tmp_sanitized = None
        need_sanitize = False
        try:
            with open(sketch_path, "r", encoding="utf-8") as fin:
                first_line = fin.readline()
                if first_line:
                    obj = json.loads(first_line)
                    tgt = obj.get("i", [None])[0][1] if isinstance(obj.get("i"), list) and len(obj["i"]) > 0 else None
                    if isinstance(tgt, str) and " -1 " in tgt:
                        need_sanitize = True
        except Exception:
            pass

        if need_sanitize:
            try:
                import tempfile

                fd, tmp_path = tempfile.mkstemp(prefix=".gen_state_sketch_", suffix=".json")
                os.close(fd)
                with open(sketch_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
                    for line in fin:
                        try:
                            obj = json.loads(line)
                            tgt = obj.get("i", [None])[0][1] if isinstance(obj.get("i"), list) and len(obj["i"]) > 0 else None
                            if isinstance(tgt, str) and " -1 " in tgt:
                                obj["i"][0][1] = tgt.split(" -1 ")[0].strip()
                                line = json.dumps(obj, separators=(",", ":")) + "\n"
                        except Exception:
                            pass
                        fout.write(line)
                sketch_path_to_use = tmp_path
                tmp_sanitized = tmp_path
                print(f"Worker {worker_id}: sanitized target strings for TVM parsing -> {sketch_path_to_use}")
            except Exception as e:
                print(f"Worker {worker_id}: sanitize sketch target failed, fallback to original. Error: {e}")

        script_args = ScriptArguments(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            sketch_path="",
            save_path="",
            keep_cnt=keep_cnt,
            target=original_target,
            adapter_path=adapter_path,
            target_hardware=target_hardware,
            allow_repeat=allow_repeat,
            is_build=is_build,
            use_bucket=use_bucket,
            use_hw_kv=bool(hw_kv_cfg),
            hw_kv_aligner_path=hw_kv_cfg["hw_kv_aligner_path"] if hw_kv_cfg else None,
            debug_forward_trace=debug_forward_trace,
            hardware_embedding_path="",
        )

        print(f"Loading model and tokenizer in worker process for device: {device}")
        model, tokenizer = load_model_for_inference(script_args)

        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        print(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        hw_kv_ctx = None
        if hw_kv_cfg:
            hw_kv_ctx = prepare_hw_kv_context(
                use_hw_kv=True,
                hw_vec=hw_kv_cfg.get("hw_vec"),
                model=model,
                device=device,
                hw_kv_aligner_path=hw_kv_cfg.get("hw_kv_aligner_path"),
                hw_kv_mode=hw_kv_cfg.get("hw_kv_mode"),
                hw_kv_num_slots=hw_kv_cfg.get("hw_kv_num_slots", 4),
            )

        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            os.remove(save_path_i)

        print(f"Worker {worker_id}: Reading and processing sketches from {sketch_path_to_use}")
        inputs, _ = auto_scheduler.RecordReader(sketch_path_to_use).read_lines()
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
        per_len = math.ceil(len(sketch_dic_list_full) / num_workers)
        start_idx = worker_id * per_len
        end_idx = min((worker_id + 1) * per_len, len(sketch_dic_list_full))
        my_sketch_chunk = sketch_dic_list_full[start_idx:end_idx]

        print(f"Worker {worker_id}: 数据分片 [{start_idx}:{end_idx}]，处理 {len(my_sketch_chunk)} out of {len(sketch_dic_list_full)} workload groups")
        total_workloads = len(my_sketch_chunk)
        successful_workloads = 0
        total_generated = 0
        print(f"Worker {worker_id}: 开始处理 {total_workloads} 个workload组")
        debug_ctx = {}

        for workload_idx, (workload_key, inputs_for_key) in enumerate(tqdm.tqdm(my_sketch_chunk, desc=f"Worker {worker_id}")):
            try:

                def gen_func_inner(task, states, max_new_tokens):
                    max_new_tokens = max(max_new_tokens, 1)
                    gen_kwargs["max_new_tokens"] = max_new_tokens
                    return gen_func(
                        task,
                        states,
                        inputs_for_key[0],
                        tokenizer,
                        model,
                        device,
                        gen_kwargs,
                        use_bucket=use_bucket,
                        hw_kv_ctx=hw_kv_ctx,
                        pos_shift=pos_shift,
                        pos_compensate=pos_compensate,
                        zero_prefix_mask_one=zero_prefix_mask_one,
                        batch_size=batch_size,
                        kv_num_slots=kv_num_slots,
                        debug_logits_compare=debug_logits_compare,
                        debug_prepare_inputs=debug_prepare_inputs,
                        debug_manual_greedy_steps=debug_manual_greedy_steps,
                        debug_manual_greedy_count=debug_manual_greedy_count,
                        debug_prefill_equiv_kv=debug_prefill_equiv_kv,
                        debug_kv_stats=debug_kv_stats,
                        debug_prefill_equiv=debug_prefill_equiv,
                    )

                policy = auto_scheduler.SketchPolicy(inputs_for_key[0].task)
                measure_inputs = []
                measure_results = []
                input_set = set()

                retry_i = 0
                while retry_i < 5:
                    try:
                        all_state_list = policy.gen_states([inp.state for inp in inputs_for_key], gen_func_inner)

                        measure_inputs_tmp = []
                        for state in all_state_list:
                            inp = auto_scheduler.MeasureInput(inputs_for_key[0].task, state)
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
                            build_results = [default_build_result for _ in measure_inputs_tmp]
                        for res, inp in zip(build_results, measure_inputs_tmp):
                            if res.error_no == 0:
                                measure_inputs.append(inp)
                                measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                        retry_i += 1
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
                    measure_inputs, measure_results = zip(*random.sample(list(zip(measure_inputs, measure_results)), keep_cnt))

                if len(measure_inputs) > 0:
                    auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
                    successful_workloads += 1
                    total_generated += len(measure_inputs)
                    try:
                        gens_log = debug_ctx.get("last_generations")
                        if gens_log:
                            log_debug(f"[VALID] workload={workload_key} generated={len(measure_inputs)} first_gen='{gens_log[0]}'")
                        else:
                            log_debug(f"[VALID] workload={workload_key} generated={len(measure_inputs)} (no decoded gen cached)")
                    except Exception:
                        pass
                    print(f"Worker {worker_id}: workload {workload_key} 成功生成 {len(measure_inputs)} 条记录")
                else:
                    print(f"Worker {worker_id}: workload {workload_key} 未能生成任何有效记录")

            except Exception as e:
                print(f"Worker {worker_id}: 处理workload {workload_key} 时发生严重错误: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"Worker {worker_id}: 处理完成！成功处理 {successful_workloads}/{total_workloads} 个workload，共生成 {total_generated} 条记录")
        if tmp_sanitized:
            try:
                os.remove(tmp_sanitized)
            except Exception:
                pass
    except Exception as e:
        err_queue.put(e)


def main():
    import tempfile
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    gen_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None,
        "max_new_tokens": 512,
    }

    hw_kv_cfg = None
    # Determine KV mode: None -> baseline; "zero"/"real" -> use prefix
    mode = script_args.hw_kv_mode
    if mode is not None:
        assert mode in ("noop", "zero", "real"), "--hw_kv_mode must be one of: noop, zero, real"
    use_hw_kv_flag = script_args.use_hw_kv or (mode in ("noop", "zero", "real"))

    if use_hw_kv_flag:
        hw_vec = None
        stats = None
        if mode not in ("zero", "noop"):
            embeddings = load_hardware_embeddings(script_args.hardware_embedding_path)
            hw_id_candidate = script_args.target_hardware or str(script_args.target)
            hw_name, hw_vec = resolve_hw_embedding(hw_id_candidate, embeddings)
            if hw_vec is None:
                print(f"[WARN] No embedding found for hw_id={hw_id_candidate}; using zeros.")
            stats = []
            stats.append(DebugUtils.hw_vec_stats(f"target({hw_id_candidate})", hw_vec))
            for label, lookup in (("4090", "4090"), ("v100", "v100")):
                name, vec = resolve_hw_embedding(lookup, embeddings)
                stats.append(DebugUtils.hw_vec_stats(f"{label}({name or lookup})", vec))
            for msg in stats:
                print(msg)
                log_debug(msg)
        hw_kv_cfg = {
            "use_hw_kv": True,
            "hw_vec": hw_vec,
            "hw_kv_aligner_path": script_args.hw_kv_aligner_path,
            "hw_kv_mode": mode or "real",
            "hw_kv_num_slots": script_args.hw_kv_num_slots,
            "hw_vec_stats": stats,
        }

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        visible_gpu_list = [int(x.strip()) for x in visible_devices.split(",") if x.strip()]
        num_gpus = len(visible_gpu_list)
        print(f"使用CUDA_VISIBLE_DEVICES={visible_devices}，检测到{num_gpus}个GPU")
        print(f"可见GPU列表: {visible_gpu_list}")
    else:
        num_gpus = torch.cuda.device_count()
        visible_gpu_list = list(range(num_gpus))
        print(f"未设置CUDA_VISIBLE_DEVICES，使用所有{num_gpus}个GPU")

    processes = []
    tmp_folder = tempfile.mkdtemp(prefix=".gen_state_")
    err_queue = Queue()
    log_file = os.path.join(
        os.path.dirname(script_args.save_path) or ".",
        f"gen_state_debug_kv_{uuid.uuid4().hex[:8]}.log",
    )
    for gpu_i in range(num_gpus):
        save_path_i = f"{tmp_folder}/{gpu_i}_part"
        device = f"cuda:{gpu_i}"
        if visible_devices:
            actual_physical_gpu = visible_gpu_list[gpu_i]
            print(f"Worker {gpu_i}: 使用设备 {device} (物理GPU {actual_physical_gpu})")
        else:
            print(f"Worker {gpu_i}: 使用设备 {device}")
        p = Process(
            target=worker,
            args=(
                err_queue,
                save_path_i,
                script_args.sketch_path,
                gen_kwargs,
                script_args.model_path,
                script_args.tokenizer_path,
                script_args.adapter_path,
                script_args.target_hardware,
                script_args.target,
                device,
                script_args.allow_repeat,
                script_args.keep_cnt,
                script_args.is_build,
                gpu_i,
                num_gpus,
                script_args.use_bucket,
                hw_kv_cfg,
                log_file,
                script_args.debug_prefill_equiv,
                script_args.pos_shift,
                script_args.pos_compensate,
                script_args.zero_prefix_mask_one,
                script_args.batch_size,
                script_args.hw_kv_num_slots,
                script_args.debug_logits_compare,
                script_args.debug_prepare_inputs,
                script_args.debug_forward_trace,
                script_args.debug_manual_greedy_steps,
                script_args.debug_manual_greedy_count,
                script_args.debug_prefill_equiv_kv,
                script_args.debug_kv_stats,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")

    try:
        total_records = merge_json_files_safely(tmp_folder, script_args.save_path)
        print(f"推理完成！总共生成 {total_records} 条记录")
    except Exception as e:
        print(f"JSON合并失败: {e}")
        print("回退到原始cat命令...")
        subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True)
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)


if __name__ == "__main__":
    main()
