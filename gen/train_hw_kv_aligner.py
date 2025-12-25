"""
Train HwKVAligner on bucket化的张量句子：
 - 冻结 base TLM，只训练 HwKVAligner
 - 仅在 schedule 段计算自回归 CE（前面的 workload/target 段不计入 loss）
 - 目标：同一 bucket 内依然能利用硬件 embedding 区分不同硬件的 schedule
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from hw_kv_aligner import HwKVAligner


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train HwKVAligner with bucket schedules.")
    parser.add_argument("--model_path", required=True, help="Bucket base model path")
    parser.add_argument("--tokenizer_path", required=True, help="Bucket tokenizer path")
    parser.add_argument("--train_json_paths", required=True, help="Comma-separated JSONL paths")
    parser.add_argument("--hardware_embeddings_path", required=True, help="Path to hardware_embeddings_v4.json")
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
    parser.add_argument("--monitor_kv_every", type=int, default=50, help="Steps interval to log KV norms")
    parser.add_argument("--monitor_kv_layer", type=int, default=0, help="Layer index to monitor KV norms")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Scheduler type for LR warmup/decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for LR scheduler")
    parser.add_argument("--kd_mismatch_threshold", type=float, default=0.2, help="If KD length mismatch ratio exceeds, skip KD for batch")
    return parser.parse_args()


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
    if args.hw_kv_aligner_ckpt:
        state = torch.load(args.hw_kv_aligner_ckpt, map_location=args.device)
        state = state.get("state_dict", state)
        hw_kv_aligner.load_state_dict(state, strict=False)
        print(f"Loaded HwKVAligner from {args.hw_kv_aligner_ckpt}")

    hw_db = load_hw_embeddings(args.hardware_embeddings_path)

    paths = [p.strip() for p in args.train_json_paths.split(",") if p.strip()]
    dataset = BucketScheduleDataset(paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
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
        for hid in hw_ids:
            try:
                _, vec = resolve_hw_vec(hid, hw_db)
                hw_vecs.append(vec)
            except KeyError:
                missing += 1
                hw_vecs.append([0.0] * 24)
        if missing > 0:
            print(f"[WARN] Missing {missing} hw embeddings in batch; filled with zeros.")
        hw_vec_batch = torch.tensor(hw_vecs, dtype=torch.float32, device=args.device)

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
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(shift_logits_s.view(-1, shift_logits_s.size(-1)), shift_labels_s.view(-1))

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
            seq_s = shift_logits_s[b, idx_s[:L], :]
            seq_t = shift_logits_t[b, idx_t[:L], :]
            if idx_s.numel() != idx_t.numel():
                kd_mismatch_count += 1
            p_t = torch.softmax(seq_t / T, dim=-1)
            log_p_s = torch.log_softmax(seq_s / T, dim=-1)
            kd_i = (p_t * (torch.log(p_t + 1e-12) - log_p_s)).sum(dim=-1).mean() * (T * T)
            kd_terms.append(kd_i)
        mismatch_ratio = kd_mismatch_count / max(1, input_ids_s.size(0))
        if kd_terms and mismatch_ratio <= args.kd_mismatch_threshold:
            kd_loss = torch.stack(kd_terms).mean()
            kd_valid = len(kd_terms)
        else:
            kd_loss = torch.tensor(0.0, device=args.device)
            kd_valid = 0

        l2_reg = sum(p.pow(2).sum() for p in hw_kv_aligner.parameters())
        scale_reg = args.kv_scale_reg * (hw_kv_aligner.kv_scale - args.kv_scale_init).pow(2)
        loss = lm_loss + args.kd_weight * kd_loss + args.l2_reg * l2_reg + scale_reg

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
            print(
                f"step {global_step}: lm_loss={lm_loss.item():.4f} kd_loss={kd_loss.item():.4f} kd_valid={kd_valid} "
                f"l2={l2_reg.item():.4f} loss={loss.item():.4f} "
                f"kv_scale={kv_scale_val:.4f} "
                f"|linker_weights|={link_norm:.4f} "
                f"softmax={['{:.3f}'.format(x) for x in link_soft]} "
                f"{sched_stats}"
            )
        if args.monitor_kv_every > 0 and global_step % args.monitor_kv_every == 0:
            try:
                layer_idx = max(0, min(args.monitor_kv_layer, len(past_key_values) - 1))
                k_mon, v_mon = past_key_values[layer_idx]
                # shapes: [B, H, T, d]
                k_norm = k_mon.norm(dim=-1)
                v_norm = v_mon.norm(dim=-1)
                print(
                    f"[MONITOR] step {global_step} layer {layer_idx} "
                    f"K mean={k_norm.mean().item():.4f} max={k_norm.max().item():.4f} "
                    f"V mean={v_norm.mean().item():.4f} max={v_norm.max().item():.4f} "
                    f"kv_scale={hw_kv_aligner.kv_scale.detach().item():.4f}"
                )
            except Exception as e:
                print(f"[MONITOR] failed to compute KV norms: {e}")
        if global_step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"hw_kv_aligner_step{global_step}.pt")
            torch.save({"state_dict": hw_kv_aligner.state_dict()}, ckpt_path)
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
    torch.save({"state_dict": hw_kv_aligner.state_dict()}, final_path)
    print(f"Training finished. Saved final HwKVAligner to {final_path}")


if __name__ == "__main__":
    main()
