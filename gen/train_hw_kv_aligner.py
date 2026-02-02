"""
Train HwKVAligner on bucket化的张量句子：
 - 冻结 base TLM，只训练 HwKVAligner
 - 仅在 schedule 段计算自回归 CE（前面的 workload/target 段不计入 loss）
 - 目标：同一 bucket 内依然能利用硬件 embedding 区分不同硬件的 schedule
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from hw_kv_aligner import HwKVAligner
from modeling.hw_preprocess import apply_preprocess, preprocess_meta_equal, summarize_preprocess


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

# Schedule operators that usually mark the start of the schedule DSL
SCHEDULE_OP_TOKENS: List[str] = [
    "CI",
    "CHW",
    "local",
    "SP",
    "FSP",
    "FFSP",
    "RE",
    "CA",
    "CHR",
    "AN",
    "PPT",
    "SPC",
    "FU",
    "TBS",
    "PRS",
    "PR",
]


def find_schedule_start_idx(tokens: List[str]) -> int:
    """
    Locate the schedule segment start.
    Heuristic:
      1) find the first schedule op token;
      2) if its previous token is a pure integer, include that integer as part of schedule.
    """
    first_op_idx = None
    for i, tok in enumerate(tokens):
        if tok in SCHEDULE_OP_TOKENS:
            first_op_idx = i
            break

    if first_op_idx is None:
        return -1

    start_idx = first_op_idx
    j = first_op_idx - 1
    if j >= 0:
        prev = tokens[j]
        if prev.lstrip("-").isdigit():
            start_idx = j
    return start_idx


class BucketScheduleDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.samples: List[Tuple[str, str, str]] = []
        self.hw_ids_seen = set()
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text")
                    text_full = obj.get("text_full")
                    hw_id = obj.get("hw_id")
                    if not text or not text_full or not hw_id:
                        continue
                    self.samples.append((text, text_full, hw_id))
                    self.hw_ids_seen.add(hw_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, text_full, hw_id = self.samples[idx]
        return {"text": text, "text_full": text_full, "hw_id": hw_id}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]  # student
    texts_full = [item["text_full"] for item in batch]  # teacher
    hw_ids = [item["hw_id"] for item in batch]
    enc_s = tokenizer(texts, return_tensors="pt", padding=True)
    enc_t = tokenizer(texts_full, return_tensors="pt", padding=True)
    return {
        "input_ids_s": enc_s["input_ids"],
        "attention_mask_s": enc_s["attention_mask"],
        "input_ids_t": enc_t["input_ids"],
        "attention_mask_t": enc_t["attention_mask"],
        "hw_ids": hw_ids,
    }


def load_hw_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["hardware_name"]: e["vector"] for e in entries}


def resolve_hw_vec(hw_id: str, db: Dict[str, List[float]]) -> Tuple[str, List[float]]:
    if hw_id in db:
        return hw_id, db[hw_id]
    key = DEFAULT_HW_NAME_MAP.get(hw_id.lower())
    if key and key in db:
        return key, db[key]
    for name, vec in db.items():
        if hw_id.lower() in name.lower():
            return name, vec
    raise KeyError(f"No embedding found for hw_id={hw_id}")


def build_labels_with_schedule_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Mask everything before the detected schedule start token to -100.
    """
    labels = input_ids.clone()
    for b in range(input_ids.size(0)):
        tokens = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in input_ids[b]]
        start = find_schedule_start_idx(tokens)
        if start is None or start < 0:
            # 未找到 schedule，整行忽略（由上层决定是否跳过该样本）
            labels[b, :] = -100
        elif start > 0:
            labels[b, :start] = -100
        else:
            # start == 0, do nothing
            pass
    return labels


def build_schedule_weights(
    labels: torch.Tensor, prefix_len: int = 0, decay: float = 0.0
) -> torch.Tensor:
    """
    Build per-token weights for schedule segment.
    - prefix_len>0: only first N schedule tokens are weighted.
    - decay in (0,1): apply exponential decay within schedule segment.
    """
    weights = torch.zeros_like(labels, dtype=torch.float32)
    B, _ = labels.shape
    for b in range(B):
        idx = (labels[b] != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if prefix_len and prefix_len > 0:
            idx = idx[:prefix_len]
        if decay and 0.0 < decay < 1.0:
            w = decay ** torch.arange(idx.numel(), device=labels.device, dtype=torch.float32)
        else:
            w = torch.ones(idx.numel(), device=labels.device, dtype=torch.float32)
        weights[b, idx] = w
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train HwKVAligner with bucket schedules.")
    parser.add_argument("--model_path", required=True, help="Bucket base model path")
    parser.add_argument("--tokenizer_path", required=True, help="Bucket tokenizer path")
    parser.add_argument("--train_json_paths", required=True, help="Comma-separated JSONL paths")
    parser.add_argument("--hardware_embeddings_path", required=True, help="Path to hardware_embeddings_v4_universe.json")
    parser.add_argument(
        "--preprocess_json",
        default="",
        help="Optional fixed preprocess JSON (e.g., preprocess_v4u_zscore_v1.json).",
    )
    parser.add_argument(
        "--require-preprocess",
        action="store_true",
        help="Fail if preprocess JSON is not found (prevents accidental raw hw_emb training).",
    )
    parser.add_argument(
        "--disable_preprocess",
        action="store_true",
        help="Skip preprocess even if default/explicit preprocess JSON exists.",
    )
    parser.add_argument("--output_dir", required=True, help="Where to save HwKVAligner checkpoints")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hw_kv_aligner_ckpt", default=None, help="Optional existing HwKVAligner checkpoint")
    parser.add_argument("--kd_weight", type=float, default=0.5, help="Weight for KD loss")
    parser.add_argument("--kd_temperature", type=float, default=2.0, help="Temperature for KD")
    parser.add_argument("--kv_scale_init", type=float, default=0.01, help="Initial scale for KV injection")
    parser.add_argument("--kv_scale_warmup_steps", type=int, default=200, help="Freeze kv_scale for first N steps")
    parser.add_argument("--kv_scale_max", type=float, default=2.0, help="Clamp kv_scale to this max")
    parser.add_argument("--kv_scale_reg", type=float, default=0.0, help="Optional L2 reg on (kv_scale-init)")
    parser.add_argument(
        "--linker_init_std",
        type=float,
        default=0.0,
        help="Optional random init std for linker_weights (only when not resuming).",
    )
    parser.add_argument("--monitor_kv_every", type=int, default=50, help="Steps interval to log KV norms")
    parser.add_argument("--monitor_kv_layer", type=int, default=0, help="Layer index to monitor KV norms")
    parser.add_argument(
        "--debug_injection_delta",
        action="store_true",
        help="Log logits delta between injected KV and no-injection on schedule tokens.",
    )
    parser.add_argument(
        "--debug_injection_every",
        type=int,
        default=0,
        help="Steps interval for injection-delta debug (0 means use logging_steps).",
    )
    parser.add_argument(
        "--debug_hw_swap",
        type=str,
        default="",
        help="Optional hw_id to swap in for one sample to compare logits (e.g., v100 or nvidia/nvidia-v100).",
    )
    parser.add_argument(
        "--debug_hw_swap_every",
        type=int,
        default=0,
        help="Steps interval for hw-swap debug (0 means use logging_steps).",
    )
    parser.add_argument(
        "--schedule_prefix_len",
        type=int,
        default=0,
        help=(
            "Only use first N schedule tokens for LM/KD (0 means full schedule). "
            "NOTE: setting a short prefix can make LM/KD ~0 if teacher/student match early."
        ),
    )
    parser.add_argument(
        "--schedule_decay",
        type=float,
        default=0.0,
        help=(
            "Optional exponential decay factor within schedule tokens (e.g., 0.98). "
            "Default 0 disables decay."
        ),
    )
    parser.add_argument(
        "--gain_weight",
        type=float,
        default=0.0,
        help="Optional gain loss weight (0 disables gain loss).",
    )
    parser.add_argument(
        "--gain_margin",
        type=float,
        default=0.0,
        help="Margin for gain loss: penalize if inj loss not better by this margin.",
    )
    parser.add_argument(
        "--gain_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps before enabling gain loss.",
    )
    parser.add_argument(
        "--swap_kd_weight",
        type=float,
        default=0.0,
        help="Optional swap-KD loss weight (0 disables swap loss).",
    )
    parser.add_argument(
        "--swap_kd_margin",
        type=float,
        default=0.0,
        help="Margin for swap-KD loss: penalize if swap KD not worse by this margin.",
    )
    parser.add_argument(
        "--swap_kd_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps before enabling swap-KD loss.",
    )
    parser.add_argument(
        "--swap_hw_prob",
        type=float,
        default=1.0,
        help="Probability to apply swap-KD on a batch (for cheaper debugging).",
    )
    parser.add_argument(
        "--swap_hw_strategy",
        type=str,
        default="permute",
        choices=["permute", "shift"],
        help="How to build swapped hw_vec batch.",
    )
    parser.add_argument(
        "--swap_kd_weight_end",
        type=float,
        default=-1.0,
        help="If >0, linearly ramp swap_kd_weight to this value after warmup.",
    )
    parser.add_argument(
        "--swap_kd_ramp_steps",
        type=int,
        default=0,
        help="Steps to linearly ramp swap_kd_weight after warmup (0 disables ramp).",
    )
    parser.add_argument(
        "--swap_require_diff_hw",
        action="store_true",
        help="Require swapped hw_id to be different for every sample; otherwise skip swap loss.",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Scheduler type for LR warmup/decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for LR scheduler")
    parser.add_argument("--kd_mismatch_threshold", type=float, default=0.2, help="If KD length mismatch ratio exceeds, skip KD for batch")
    return parser.parse_args()


def _load_preprocess_params(path: str) -> Optional[Dict]:
    if not path:
        return None
    p = path
    try:
        params = json.loads(open(p, "r", encoding="utf-8").read())
    except Exception:
        return None
    if not isinstance(params, dict):
        return None
    if "mean" in params and "std" in params and "mask" in params:
        params.setdefault("source", p)
        return params
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    model.eval()
    model.requires_grad_(False)

    n_layer = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))
    n_head = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
    hidden = getattr(model.config, "hidden_size", None)
    if n_layer is None or n_head is None or hidden is None:
        raise ValueError("Model config missing n_layer/n_head/hidden_size")
    head_dim = hidden // n_head

    hw_kv_aligner = HwKVAligner(
        llm_num_layers=n_layer,
        llm_num_heads=n_head,
        llm_head_dim=head_dim,
        hw_dim=24,
        num_slots=4,
        backward_depth=min(4, n_layer),
        linker_temperature=1.0,
        kv_scale_init=args.kv_scale_init,
    ).to(args.device)
    resume_kv_scale = None
    if args.linker_init_std and args.linker_init_std > 0 and not args.hw_kv_aligner_ckpt:
        with torch.no_grad():
            if hasattr(hw_kv_aligner, "linker_weights"):
                hw_kv_aligner.linker_weights.normal_(mean=0.0, std=args.linker_init_std)
        print(f"[INIT] linker_weights normal_(0,{args.linker_init_std})")
    if args.hw_kv_aligner_ckpt:
        state = torch.load(args.hw_kv_aligner_ckpt, map_location=args.device)
        ckpt_meta = state.get("meta", {})
        state = state.get("state_dict", state)
        hw_kv_aligner.load_state_dict(state, strict=False)
        if hasattr(hw_kv_aligner, "kv_scale"):
            resume_kv_scale = hw_kv_aligner.kv_scale.detach().item()
        print(f"Loaded HwKVAligner from {args.hw_kv_aligner_ckpt}")

    hw_db = load_hw_embeddings(args.hardware_embeddings_path)

    preprocess_params = None
    default_preprocess = "gen/Embedding/preprocess_v4u_zscore_v1.json"
    if args.disable_preprocess:
        if args.require_preprocess:
            raise ValueError("disable_preprocess conflicts with require_preprocess.")
        print("[CONFIG] hw preprocess: disabled (raw hw_emb)")
    else:
        preprocess_params = _load_preprocess_params(args.preprocess_json) or _load_preprocess_params(default_preprocess)
        if preprocess_params:
            print(f"[CONFIG] hw preprocess: {summarize_preprocess(preprocess_params)}")
            if args.hw_kv_aligner_ckpt and ckpt_meta:
                init_pre = ckpt_meta.get("preprocess")
                if init_pre and not preprocess_meta_equal(init_pre, preprocess_params):
                    print("[WARN] hw_kv_aligner_ckpt preprocess meta differs from current config.")
        else:
            msg = "[WARN] No preprocess JSON found; training HwKVAligner on raw hw_emb."
            print(msg)
            if args.require_preprocess:
                raise ValueError("require-preprocess enabled but no preprocess JSON found.")

    paths = [p.strip() for p in args.train_json_paths.split(",") if p.strip()]
    dataset = BucketScheduleDataset(paths)
    if args.swap_kd_weight > 0:
        print(f"[INFO] dataset unique hw_ids={len(dataset.hw_ids_seen)} sample={list(sorted(dataset.hw_ids_seen))[:8]}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    # build swap hw pool (canonical names) and cache preprocessed vectors
    swap_hw_pool: List[str] = []
    swap_hw_vec_cache: Dict[str, torch.Tensor] = {}
    if args.swap_kd_weight > 0:
        pool_names = set()
        for hid in sorted(dataset.hw_ids_seen):
            try:
                name, vec = resolve_hw_vec(hid, hw_db)
            except KeyError:
                continue
            pool_names.add(name)
        swap_hw_pool = sorted(pool_names)
        for name in swap_hw_pool:
            vec = torch.tensor(hw_db[name], dtype=torch.float32, device=args.device)
            if preprocess_params:
                vec = apply_preprocess(vec.unsqueeze(0), preprocess_params).squeeze(0)
            swap_hw_vec_cache[name] = vec
        print(f"[INFO] swap hw_pool (canon) size={len(swap_hw_pool)}: {swap_hw_pool}")
    if args.debug_injection_delta and args.debug_injection_every <= 0:
        args.debug_injection_every = max(1, args.logging_steps)
    if args.debug_hw_swap and args.debug_hw_swap_every <= 0:
        args.debug_hw_swap_every = max(1, args.logging_steps)
    data_iter = iter(dataloader)

    optimizer = AdamW(hw_kv_aligner.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    global_step = 0
    saved_ckpts: List[str] = []
    kv_scale_frozen = True
    warned_single_hw = False
    swap_considered = 0
    swap_applied_cnt = 0
    swap_skipped_cnt = 0
    if resume_kv_scale is not None:
        # Respect ckpt kv_scale on resume: avoid warmup reset and L2 pull to init.
        print(f"[RESUME] kv_scale from ckpt={resume_kv_scale:.6f}; disable kv_scale warmup and reg-to-init.")
        args.kv_scale_init = resume_kv_scale
        args.kv_scale_warmup_steps = 0
        args.kv_scale_reg = 0.0
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids_s = batch["input_ids_s"].to(args.device)
        attention_mask_s = batch["attention_mask_s"].to(args.device)
        input_ids_t = batch["input_ids_t"].to(args.device)
        attention_mask_t = batch["attention_mask_t"].to(args.device)
        hw_ids = batch["hw_ids"]

        # hw embedding batch
        hw_vecs = []
        missing = 0
        hw_names: List[str] = []
        for hid in hw_ids:
            try:
                name, vec = resolve_hw_vec(hid, hw_db)
                hw_vecs.append(vec)
                hw_names.append(name)
            except KeyError:
                missing += 1
                hw_vecs.append([0.0] * 24)
                hw_names.append("unknown")
        if missing > 0:
            print(f"[WARN] Missing {missing} hw embeddings in batch; filled with zeros.")
        hw_vec_batch = torch.tensor(hw_vecs, dtype=torch.float32, device=args.device)
        if preprocess_params:
            hw_vec_batch = apply_preprocess(hw_vec_batch, preprocess_params)

        labels_s = build_labels_with_schedule_mask(input_ids_s, tokenizer).to(args.device)
        labels_s[attention_mask_s == 0] = -100
        labels_t = build_labels_with_schedule_mask(input_ids_t, tokenizer).to(args.device)
        labels_t[attention_mask_t == 0] = -100

        valid_mask = (labels_s != -100).any(dim=1) & (labels_t != -100).any(dim=1)
        if valid_mask.sum() == 0:
            print(f"[WARN] batch {global_step}: all samples invalid schedule, skip batch")
            global_step += 1
            continue
        if valid_mask.sum() < labels_s.size(0):
            input_ids_s = input_ids_s[valid_mask]
            attention_mask_s = attention_mask_s[valid_mask]
            input_ids_t = input_ids_t[valid_mask]
            attention_mask_t = attention_mask_t[valid_mask]
            labels_s = labels_s[valid_mask]
            labels_t = labels_t[valid_mask]
            hw_vec_batch = hw_vec_batch[valid_mask]

        # Teacher forward (no KV)
        with torch.no_grad():
            out_t = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                use_cache=False,
            )
            logits_t = out_t.logits
        # kv_scale warmup
        if global_step < args.kv_scale_warmup_steps:
            if kv_scale_frozen:
                hw_kv_aligner.kv_scale.data.fill_(args.kv_scale_init)
                hw_kv_aligner.kv_scale.requires_grad_(False)
        elif kv_scale_frozen:
            hw_kv_aligner.kv_scale.requires_grad_(True)
            kv_scale_frozen = False

        past_key_values = hw_kv_aligner(hw_vec_batch, batch_size=input_ids_s.size(0), num_beams=1)
        # 为 past 长度补 attention_mask，避免维度不匹配
        past_k_len = past_key_values[0][0].shape[2]  # [B, H, T_past, d]
        if past_k_len > 0:
            prefix_mask = torch.ones(
                input_ids_s.size(0),
                past_k_len,
                device=attention_mask_s.device,
                dtype=attention_mask_s.dtype,
            )
            attention_mask_full = torch.cat([prefix_mask, attention_mask_s], dim=1)
        else:
            attention_mask_full = attention_mask_s

        # Position ids follow prompt tokens only (exclude KV prefix length)
        prompt_pos = attention_mask_s.long().cumsum(-1) - 1
        prompt_pos = prompt_pos.masked_fill(attention_mask_s == 0, 0).to(args.device)

        outputs = model(
            input_ids=input_ids_s,
            attention_mask=attention_mask_full,
            position_ids=prompt_pos,
            past_key_values=past_key_values,
            use_cache=False,
        )
        logits = outputs.logits  # [B, T, V]

        shift_logits_s = logits[..., :-1, :].contiguous()
        shift_labels_s = labels_s[..., 1:].contiguous()

        # NOTE: loss focuses on schedule prefix / decay weights to emphasize
        # hardware-critical early tokens and reduce dilution from long suffix tokens.
        sched_weights = build_schedule_weights(
            shift_labels_s, prefix_len=args.schedule_prefix_len, decay=args.schedule_decay
        ).to(shift_logits_s.device)
        if sched_weights.sum().item() <= 0:
            sched_weights = (shift_labels_s != -100).float()

        if args.debug_injection_delta and global_step % args.debug_injection_every == 0:
            with torch.no_grad():
                out_noinj = model(
                    input_ids=input_ids_s,
                    attention_mask=attention_mask_s,
                    position_ids=prompt_pos,
                    use_cache=False,
                )
                logits_noinj = out_noinj.logits
                shift_logits_noinj = logits_noinj[..., :-1, :].contiguous()
                mask = sched_weights > 0
                if mask.any():
                    s = shift_logits_s[mask]
                    n = shift_logits_noinj[mask]
                    mean_delta = (s - n).abs().mean().item()
                    T_dbg = args.kd_temperature
                    p = torch.softmax(s / T_dbg, dim=-1)
                    q = torch.softmax(n / T_dbg, dim=-1)
                    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean().item()
                    print(
                        f"[DEBUG-INJ] step {global_step} "
                        f"mean|Δlogits|={mean_delta:.4e} KL={kl:.4e}"
                    )
                else:
                    print(f"[DEBUG-INJ] step {global_step} no schedule tokens to compare.")

        if args.debug_hw_swap and global_step % args.debug_hw_swap_every == 0:
            try:
                _, alt_vec = resolve_hw_vec(args.debug_hw_swap, hw_db)
                alt_vec = torch.tensor(alt_vec, dtype=torch.float32, device=args.device).unsqueeze(0)
                if preprocess_params:
                    alt_vec = apply_preprocess(alt_vec, preprocess_params)
                # use the first sample for a lightweight debug
                si = 0
                past_alt = hw_kv_aligner(alt_vec, batch_size=1, num_beams=1)
                past_alt_len = past_alt[0][0].shape[2]
                attn_s = attention_mask_s[si : si + 1]
                if past_alt_len > 0:
                    prefix_mask_alt = torch.ones(
                        1,
                        past_alt_len,
                        device=attn_s.device,
                        dtype=attn_s.dtype,
                    )
                    attn_full_alt = torch.cat([prefix_mask_alt, attn_s], dim=1)
                else:
                    attn_full_alt = attn_s
                pos_alt = prompt_pos[si : si + 1]
                out_alt = model(
                    input_ids=input_ids_s[si : si + 1],
                    attention_mask=attn_full_alt,
                    position_ids=pos_alt,
                    past_key_values=past_alt,
                    use_cache=False,
                )
                logits_alt = out_alt.logits[..., :-1, :].contiguous()
                mask = sched_weights[si] > 0
                if mask.any():
                    s = shift_logits_s[si][mask]
                    a = logits_alt[0][mask]
                    mean_delta = (s - a).abs().mean().item()
                    T_dbg = args.kd_temperature
                    p = torch.softmax(s / T_dbg, dim=-1)
                    q = torch.softmax(a / T_dbg, dim=-1)
                    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean().item()
                    orig_hw = hw_ids[si]
                    print(
                        f"[DEBUG-HWSWAP] step {global_step} orig_hw={orig_hw} swap_hw={args.debug_hw_swap} "
                        f"mean|Δlogits|={mean_delta:.4e} KL={kl:.4e}"
                    )
                else:
                    print(f"[DEBUG-HWSWAP] step {global_step} no schedule tokens to compare.")
            except KeyError as e:
                print(f"[DEBUG-HWSWAP] step {global_step} cannot resolve hw '{args.debug_hw_swap}': {e}")
        per_tok = F.cross_entropy(
            shift_logits_s.view(-1, shift_logits_s.size(-1)),
            shift_labels_s.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels_s)
        lm_loss = (per_tok * sched_weights).sum() / (sched_weights.sum() + 1e-8)

        # KD loss on schedule segment
        shift_logits_t = logits_t[..., :-1, :].contiguous()
        shift_labels_t = labels_t[..., 1:].contiguous()
        # schedule-length sanity
        sched_len_s = (shift_labels_s != -100).sum(dim=1)
        sched_len_t = (shift_labels_t != -100).sum(dim=1)
        zero_sched = (sched_len_s == 0).sum().item()
        full_sched = (sched_len_s.float() > 0.9 * shift_labels_s.size(1)).sum().item()
        kd_len_mismatch = (sched_len_s != sched_len_t).sum().item()

        kd_terms = []
        T = args.kd_temperature
        kd_mismatch_count = 0
        for b in range(input_ids_s.size(0)):
            idx_s = (shift_labels_s[b] != -100).nonzero(as_tuple=True)[0]
            idx_t = (shift_labels_t[b] != -100).nonzero(as_tuple=True)[0]
            L = min(idx_s.numel(), idx_t.numel())
            if L == 0:
                continue
            if args.schedule_prefix_len and args.schedule_prefix_len > 0:
                idx_s = idx_s[: args.schedule_prefix_len]
                idx_t = idx_t[: args.schedule_prefix_len]
                L = min(idx_s.numel(), idx_t.numel())
            seq_s = shift_logits_s[b, idx_s[:L], :]
            seq_t = shift_logits_t[b, idx_t[:L], :]
            if idx_s.numel() != idx_t.numel():
                kd_mismatch_count += 1
            p_t = torch.softmax(seq_t / T, dim=-1)
            log_p_s = torch.log_softmax(seq_s / T, dim=-1)
            kd_tok = (p_t * (torch.log(p_t + 1e-12) - log_p_s)).sum(dim=-1)
            if args.schedule_decay and 0.0 < args.schedule_decay < 1.0:
                w = args.schedule_decay ** torch.arange(L, device=seq_s.device, dtype=torch.float32)
            else:
                w = torch.ones(L, device=seq_s.device, dtype=torch.float32)
            kd_i = (kd_tok * w).sum() / (w.sum() + 1e-8) * (T * T)
            kd_terms.append(kd_i)
        mismatch_ratio = kd_mismatch_count / max(1, input_ids_s.size(0))
        if kd_terms and mismatch_ratio <= args.kd_mismatch_threshold:
            kd_loss = torch.stack(kd_terms).mean()
            kd_valid = len(kd_terms)
        else:
            kd_loss = torch.tensor(0.0, device=args.device)
            kd_valid = 0

        # Swap-KD loss: enforce "wrong hardware" should be worse than correct hardware
        swap_loss = torch.tensor(0.0, device=args.device)
        kd_loss_swap = torch.tensor(0.0, device=args.device)
        swap_gap = torch.tensor(0.0, device=args.device)
        swap_applied = False
        swap_skipped_reason = ""
        # swap weight warmup / ramp
        swap_weight_eff = args.swap_kd_weight
        if args.swap_kd_weight_end and args.swap_kd_weight_end > 0 and args.swap_kd_ramp_steps > 0:
            if global_step >= args.swap_kd_warmup_steps:
                t = min(1.0, (global_step - args.swap_kd_warmup_steps) / args.swap_kd_ramp_steps)
                swap_weight_eff = args.swap_kd_weight + t * (args.swap_kd_weight_end - args.swap_kd_weight)
        if swap_weight_eff <= 0:
            swap_weight_eff = 0.0
        if args.swap_kd_weight > 0 and global_step >= args.swap_kd_warmup_steps:
            if torch.rand(1).item() <= args.swap_hw_prob:
                swap_considered += 1
                B = input_ids_s.size(0)
                if len(swap_hw_pool) < 2:
                    swap_skipped_reason = "pool<2"
                else:
                    # sample swap hw from global pool, ensuring different hw
                    swap_names: List[str] = []
                    for i in range(B):
                        orig = hw_names[i]
                        if len(swap_hw_pool) == 2:
                            # fast path
                            a, b = swap_hw_pool[0], swap_hw_pool[1]
                            swap_names.append(b if orig == a else a)
                        else:
                            # random pick different
                            for _ in range(10):
                                pick = swap_hw_pool[torch.randint(len(swap_hw_pool), (1,)).item()]
                                if pick != orig:
                                    swap_names.append(pick)
                                    break
                            else:
                                # fallback if orig unknown or pool degenerate
                                swap_names.append(swap_hw_pool[0])
                    # build swap hw_vec batch
                    hw_vec_swap = torch.stack([swap_hw_vec_cache[n] for n in swap_names], dim=0)
                    if args.swap_require_diff_hw:
                        # verify at least one sample is different; otherwise skip
                        if all(orig == sw for orig, sw in zip(hw_names, swap_names)):
                            swap_skipped_reason = "same_hw_only"
                    if not swap_skipped_reason:
                        past_swap = hw_kv_aligner(hw_vec_swap, batch_size=input_ids_s.size(0), num_beams=1)
                        past_swap_len = past_swap[0][0].shape[2]
                        if past_swap_len > 0:
                            prefix_mask_swap = torch.ones(
                                input_ids_s.size(0),
                                past_swap_len,
                                device=attention_mask_s.device,
                                dtype=attention_mask_s.dtype,
                            )
                            attention_mask_swap = torch.cat([prefix_mask_swap, attention_mask_s], dim=1)
                        else:
                            attention_mask_swap = attention_mask_s
                        outputs_swap = model(
                            input_ids=input_ids_s,
                            attention_mask=attention_mask_swap,
                            position_ids=prompt_pos,
                            past_key_values=past_swap,
                            use_cache=False,
                        )
                        logits_swap = outputs_swap.logits
                        shift_logits_swap = logits_swap[..., :-1, :].contiguous()

                        # KD loss for swapped hw_vec
                        kd_terms_swap = []
                        kd_mismatch_count_swap = 0
                        for b in range(input_ids_s.size(0)):
                            idx_s = (shift_labels_s[b] != -100).nonzero(as_tuple=True)[0]
                            idx_t = (shift_labels_t[b] != -100).nonzero(as_tuple=True)[0]
                            L = min(idx_s.numel(), idx_t.numel())
                            if L == 0:
                                continue
                            if args.schedule_prefix_len and args.schedule_prefix_len > 0:
                                idx_s = idx_s[: args.schedule_prefix_len]
                                idx_t = idx_t[: args.schedule_prefix_len]
                                L = min(idx_s.numel(), idx_t.numel())
                            seq_s = shift_logits_swap[b, idx_s[:L], :]
                            seq_t = shift_logits_t[b, idx_t[:L], :]
                            if idx_s.numel() != idx_t.numel():
                                kd_mismatch_count_swap += 1
                            p_t = torch.softmax(seq_t / T, dim=-1)
                            log_p_s = torch.log_softmax(seq_s / T, dim=-1)
                            kd_tok = (p_t * (torch.log(p_t + 1e-12) - log_p_s)).sum(dim=-1)
                            if args.schedule_decay and 0.0 < args.schedule_decay < 1.0:
                                w = args.schedule_decay ** torch.arange(L, device=seq_s.device, dtype=torch.float32)
                            else:
                                w = torch.ones(L, device=seq_s.device, dtype=torch.float32)
                            kd_i = (kd_tok * w).sum() / (w.sum() + 1e-8) * (T * T)
                            kd_terms_swap.append(kd_i)
                        mismatch_ratio_swap = kd_mismatch_count_swap / max(1, input_ids_s.size(0))
                        if kd_terms_swap and mismatch_ratio_swap <= args.kd_mismatch_threshold:
                            kd_loss_swap = torch.stack(kd_terms_swap).mean()
                            swap_gap = kd_loss_swap - kd_loss
                            swap_loss = torch.relu(args.swap_kd_margin + kd_loss - kd_loss_swap)
                            swap_applied = True
                            swap_applied_cnt += 1
                        else:
                            swap_skipped_reason = "kd_mismatch"
            else:
                swap_skipped_reason = "prob_skip"
        if args.swap_kd_weight > 0 and not swap_applied and args.swap_require_diff_hw and not warned_single_hw:
            # if we keep skipping because batch has only one hw, warn once
            if len(set(hw_ids)) < 2:
                print("[WARN] swap_kd enabled but batch has only one hw_id; swap loss skipped.")
                warned_single_hw = True
        if args.swap_kd_weight > 0 and not swap_applied and swap_skipped_reason:
            swap_skipped_cnt += 1

        gain_loss = torch.tensor(0.0, device=args.device)
        if args.gain_weight > 0 and global_step >= args.gain_warmup_steps:
            with torch.no_grad():
                out_noinj = model(
                    input_ids=input_ids_s,
                    attention_mask=attention_mask_s,
                    position_ids=prompt_pos,
                    use_cache=False,
                )
                logits_noinj = out_noinj.logits
                shift_logits_noinj = logits_noinj[..., :-1, :].contiguous()
            per_tok_noinj = F.cross_entropy(
                shift_logits_noinj.view(-1, shift_logits_noinj.size(-1)),
                shift_labels_s.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels_s)
            lm_loss_noinj = (per_tok_noinj * sched_weights).sum() / (sched_weights.sum() + 1e-8)
            gain_loss = torch.relu(lm_loss_noinj - lm_loss - args.gain_margin)

        l2_reg = sum(p.pow(2).sum() for p in hw_kv_aligner.parameters())
        scale_reg = args.kv_scale_reg * (hw_kv_aligner.kv_scale - args.kv_scale_init).pow(2)
        loss = (
            lm_loss
            + args.kd_weight * kd_loss
            + args.gain_weight * gain_loss
            + swap_weight_eff * swap_loss
            + args.l2_reg * l2_reg
            + scale_reg
        )

        optimizer.zero_grad()
        loss.backward()
        # Single-batch gradient check (after backward)
        if (global_step + 1) % args.logging_steps == 0:
            def _grad_stats(param):
                if param is None or param.grad is None:
                    return "none"
                g = param.grad.detach()
                return f"mean={g.abs().mean().item():.3e} max={g.abs().max().item():.3e}"

            grad_msgs = [
                f"hw_mlp_last.w { _grad_stats(getattr(hw_kv_aligner.hw_mlp[-1], 'weight', None)) }",
                f"hw_mlp_last.b { _grad_stats(getattr(hw_kv_aligner.hw_mlp[-1], 'bias', None)) }",
                f"key_aligner.w { _grad_stats(getattr(hw_kv_aligner.key_aligner, 'weight', None)) }",
                f"value_aligner.w { _grad_stats(getattr(hw_kv_aligner.value_aligner, 'weight', None)) }",
                f"linker_weights { _grad_stats(getattr(hw_kv_aligner, 'linker_weights', None)) }",
                f"kv_scale { _grad_stats(getattr(hw_kv_aligner, 'kv_scale', None)) }",
            ]
            print(f"[GRAD] step {global_step + 1} " + " | ".join(grad_msgs))
        torch.nn.utils.clip_grad_norm_(hw_kv_aligner.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            hw_kv_aligner.kv_scale.clamp_(0.0, args.kv_scale_max)

        global_step += 1
        if global_step % args.logging_steps == 0:
            with torch.no_grad():
                link_soft = torch.softmax(hw_kv_aligner.linker_weights, dim=0).detach().cpu().tolist()
                link_norm = hw_kv_aligner.linker_weights.detach().norm().item()
                kv_scale_val = hw_kv_aligner.kv_scale.detach().item()
                sched_stats = (
                    f"sched_len_s(min/mean/max)={sched_len_s.min().item():.0f}/"
                    f"{sched_len_s.float().mean().item():.1f}/"
                    f"{sched_len_s.max().item():.0f} "
                    f"zero={zero_sched} full90={full_sched} kd_mismatch={kd_len_mismatch}"
                )
            extra = ""
            if args.swap_kd_weight > 0:
                swap_ratio = 0.0
                if swap_considered > 0:
                    swap_ratio = swap_applied_cnt / swap_considered
                swap_info = (
                    f"kd_swap={kd_loss_swap.item():.4f} swap_gap={swap_gap.item():.4f} "
                    f"swap_loss={swap_loss.item():.4f} swap_ratio={swap_ratio:.2f} swap_w={swap_weight_eff:.3f}"
                )
                if not swap_applied:
                    swap_info += f" swap_skip={swap_skipped_reason}"
                extra = " " + swap_info
            print(
                f"step {global_step}: lm_loss={lm_loss.item():.4f} kd_loss={kd_loss.item():.4f} kd_valid={kd_valid} "
                f"gain={gain_loss.item():.4f} l2={l2_reg.item():.4f} loss={loss.item():.4f} "
                f"kv_scale={kv_scale_val:.4f} "
                f"|linker_weights|={link_norm:.4f} "
                f"softmax={['{:.3f}'.format(x) for x in link_soft]} "
                f"{sched_stats}{extra}"
            )
        if args.monitor_kv_every > 0 and global_step % args.monitor_kv_every == 0:
            try:
                layer_idx = max(0, min(args.monitor_kv_layer, len(past_key_values) - 1))
                k_mon, v_mon = past_key_values[layer_idx]
                # shapes: [B, H, T, d]
                k_norm = k_mon.norm(dim=-1)
                v_norm = v_mon.norm(dim=-1)
                kv_scale_val = hw_kv_aligner.kv_scale.detach().item()
                eff_k = kv_scale_val * k_norm.mean().item()
                eff_v = kv_scale_val * v_norm.mean().item()
                print(
                    f"[MONITOR] step {global_step} layer {layer_idx} "
                    f"K mean={k_norm.mean().item():.4f} max={k_norm.max().item():.4f} "
                    f"V mean={v_norm.mean().item():.4f} max={v_norm.max().item():.4f} "
                    f"kv_scale={kv_scale_val:.4f} "
                    f"eff_K={eff_k:.4e} eff_V={eff_v:.4e}"
                )
            except Exception as e:
                print(f"[MONITOR] failed to compute KV norms: {e}")
        if global_step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"hw_kv_aligner_step{global_step}.pt")
            payload = {"state_dict": hw_kv_aligner.state_dict()}
            if preprocess_params:
                payload["meta"] = {"preprocess": preprocess_params}
            torch.save(payload, ckpt_path)
            print(f"Saved HwKVAligner checkpoint to {ckpt_path}")
            saved_ckpts.append(ckpt_path)
            # 保留最近 2 个中间 checkpoint
            if len(saved_ckpts) > 2:
                old = saved_ckpts.pop(0)
                try:
                    os.remove(old)
                    print(f"Removed old checkpoint {old}")
                except OSError:
                    pass

    final_path = os.path.join(args.output_dir, "hw_kv_aligner.pt")
    payload = {"state_dict": hw_kv_aligner.state_dict()}
    if preprocess_params:
        payload["meta"] = {"preprocess": preprocess_params}
    torch.save(payload, final_path)
    print(f"Training finished. Saved final HwKVAligner to {final_path}")


if __name__ == "__main__":
    main()
