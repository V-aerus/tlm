#!/usr/bin/env python3
"""Train HwToken injection with ProtoMix + LoRA on frozen TLM-Base.

目标：
- 在冻结 Base 的前提下，通过 LoRA + ProtoMixAligner 让模型适配
  “target→[MASK] + HwToken 注入”的新输入格式；
- 直接在原始张量句子数据上，用 LM CE + 硬件分类 + 原型正交等损失训练，
  可选保留蒸馏（KD）和对比学习（CTR）的扩展接口。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)
import time

from modeling.hw_injection import ProtoMixAligner, build_prototype_matrix

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


@dataclass
class TrainHwLoraArgs:
    model_path: str = field(metadata={"help": "Base model path"})
    tokenizer_path: str = field(metadata={"help": "Tokenizer path"})
    dataset_path: str = field(metadata={"help": "JSONL/JSON file with text/text_student/hw_emb"})
    output_dir: str = field(metadata={"help": "Where to save LoRA + aligner"})

    # 数据与 batch
    batch_size: int = field(default=4, metadata={"help": "Training batch size"})
    num_epochs: int = field(default=1, metadata={"help": "Number of epochs"})
    max_length: int = field(default=512, metadata={"help": "Tokenizer max length"})
    sample_fraction: Optional[float] = field(default=None, metadata={"help": "Optional fraction of data to sample (0,1]"})

    # 优化与调度
    lr_lora: float = field(default=5e-5, metadata={"help": "Learning rate for LoRA parameters"})
    lr_aligner: float = field(default=1e-4, metadata={"help": "Learning rate for ProtoMix + classifier"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    warmup_steps: int = field(default=1000, metadata={"help": "Warmup steps"})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "LR scheduler type"})
    grad_accum_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})
    gradient_clip: float = field(default=1.0, metadata={"help": "Gradient clipping norm"})

    # 硬件嵌入与投影
    hardware_embedding_path: str = field(default="Embedding/hardware_embeddings_v2.json", metadata={"help": "Hardware embedding json"})
    prototype_names: str = field(
        default="nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge",
        metadata={"help": "Comma separated hardware names used as prototypes"},
    )
    hw_token: str = field(default="[MASK]", metadata={"help": "Placeholder token used in student text"})

    # 损失权重
    lm_window_size: int = field(default=0, metadata={"help": "Tokens after HwToken included in LM loss (0 = full sequence)"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Weight for hardware classification loss"})
    lambda_ortho: float = field(default=0.1, metadata={"help": "Weight for prototype orthogonality loss"})
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for contrastive loss (default off)"})
    lambda_kd: float = field(default=0.0, metadata={"help": "Weight for KD loss (default off)"})
    ctr_temp: float = field(default=0.1, metadata={"help": "Temperature for contrastive loss"})
    kd_temp: float = field(default=1.0, metadata={"help": "Temperature for KD"})

    # LoRA 配置
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj",
        metadata={"help": "Comma-separated module name fragments for LoRA"},
    )

    # 设备与日志
    device: str = field(default="cuda", metadata={"help": "Device"})
    log_interval: int = field(default=50, metadata={"help": "Log every N steps"})
    save_steps: int = field(default=2000, metadata={"help": "Save checkpoint every N steps"})
    save_total_limit: int = field(default=5, metadata={"help": "Max checkpoints to keep"})


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


class HwLoraDataset(Dataset):
    """直接从 JSON 数据集中提供 text/text_student/hw_emb."""

    def __init__(self, hf_ds, proto_name_to_idx: Dict[str, int]):
        self.data = []
        for item in hf_ds:
            text = item.get("text")
            text_student = item.get("text_student")
            hw_emb = item.get("hw_emb")
            hw_name = item.get("hw_name")
            if text is None or text_student is None or hw_emb is None or hw_name not in proto_name_to_idx:
                continue
            self.data.append(
                {
                    "text": text,
                    "text_student": text_student,
                    "hw_emb": hw_emb,
                    "hw_label": proto_name_to_idx[hw_name],
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class HwLoraCollator:
    def __init__(self, tokenizer, max_length: int, hw_noise_std: float = 0.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hw_noise_std = hw_noise_std

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts_teacher = [b["text"] for b in batch]
        texts_student = [b["text_student"] for b in batch]

        enc_teacher = self.tokenizer(
            texts_teacher,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_student = self.tokenizer(
            texts_student,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        hw = torch.tensor([b["hw_emb"] for b in batch], dtype=torch.float32)
        if self.hw_noise_std > 0:
            hw = hw + torch.randn_like(hw) * self.hw_noise_std

        return {
            "input_ids_teacher": enc_teacher["input_ids"],
            "attention_mask_teacher": enc_teacher["attention_mask"],
            "input_ids_student": enc_student["input_ids"],
            "attention_mask_student": enc_student["attention_mask"],
            "labels_student": enc_student["input_ids"].clone(),
            "hw": hw,
            "hw_label": torch.tensor([b["hw_label"] for b in batch], dtype=torch.long),
        }


def main() -> None:
    parser = HfArgumentParser(TrainHwLoraArgs)
    args: TrainHwLoraArgs = parser.parse_args_into_dataclasses()[0]

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载基础模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_teacher = AutoModelForCausalLM.from_pretrained(args.model_path)
    base_teacher.to(device)
    base_teacher.eval()
    for p in base_teacher.parameters():
        p.requires_grad = False

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft (LoRA) is required for this script.")

    student = AutoModelForCausalLM.from_pretrained(args.model_path)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    student = get_peft_model(student, lora_cfg)
    student.print_trainable_parameters()
    student.to(device)

    embed_layer = student.get_input_embeddings()

    # Hw 对齐器与原型
    hardware_embeddings = load_hardware_embeddings(args.hardware_embedding_path)
    prototype_names = [name.strip() for name in args.prototype_names.split(",") if name.strip()]
    proto_matrix = build_prototype_matrix(hardware_embeddings, prototype_names)
    aligner = ProtoMixAligner(proto_matrix, embed_dim=embed_layer.embedding_dim, temperature=1.0, trainable_temperature=False)
    aligner.to(device)

    num_hw_classes = len(prototype_names)
    cls_head = nn.Linear(embed_layer.embedding_dim, num_hw_classes).to(device)

    proto_name_to_idx = {name: idx for idx, name in enumerate(prototype_names)}

    # 数据集
    raw_ds = load_dataset("json", data_files=args.dataset_path)["train"]
    if args.sample_fraction and 0 < args.sample_fraction < 1:
        raw_ds = raw_ds.shuffle(seed=0).select(range(int(len(raw_ds) * args.sample_fraction)))
    dataset = HwLoraDataset(raw_ds, proto_name_to_idx)

    collator = HwLoraCollator(tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    hw_token_id = tokenizer.convert_tokens_to_ids(args.hw_token)
    if hw_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not know hardware token '{args.hw_token}'")

    # 优化器与调度器
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr_lora},
            {"params": list(aligner.parameters()) + list(cls_head.parameters()), "lr": args.lr_aligner},
        ],
        weight_decay=args.weight_decay,
    )

    dataset_len = len(dataset)
    steps_per_epoch = math.ceil(dataset_len / args.batch_size / max(1, args.grad_accum_steps))
    total_steps = steps_per_epoch * args.num_epochs

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_basename = os.path.basename(args.output_dir.rstrip("/"))
    log_file = os.path.join(log_dir, f"train_hw_injection_lora_{log_basename}_{timestamp}.log")
    log_json_file = os.path.join(log_dir, f"train_hw_injection_lora_{log_basename}_{timestamp}.json")

    training_logs = {
        "config": vars(args),
        "total_steps": total_steps,
        "total_samples": dataset_len,
        "logs": [],
    }

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_checkpoints: List[str] = []

    print(f"[数据集] 样本数: {dataset_len:,}")
    print(f"[LoRA] r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[损失权重] LM, CLS={args.lambda_cls}, ORTH={args.lambda_ortho}, CTR={args.lambda_ctr}, KD={args.lambda_kd}")
    print(f"日志文件: {log_file}")

    global_step = 0
    loss_history: List[float] = []
    loss_window = 50

    student.train()
    aligner.train()
    cls_head.train()

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"Start training at {datetime.now().isoformat()}\n")

    start_time = time.time()

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            student_input_ids = batch["input_ids_student"].to(device)
            student_attn_mask = batch["attention_mask_student"].to(device)
            labels_student = batch["labels_student"].to(device)
            teacher_input_ids = batch["input_ids_teacher"].to(device)
            teacher_attn_mask = batch["attention_mask_teacher"].to(device)
            hw_vec = batch["hw"].to(device)
            hw_labels = batch["hw_label"].to(device)

            # 局部 LM 窗口处理
            if args.lm_window_size and args.lm_window_size > 0:
                ignore_index = -100
                B, L = labels_student.size()
                hw_mask = (student_input_ids == hw_token_id)
                if not torch.all(hw_mask.any(dim=1)):
                    # 训练期简单跳过不含 HwToken 的样本
                    continue
                first_mask_pos = hw_mask.float().argmax(dim=1)
                window = args.lm_window_size
                for i in range(B):
                    start = int(first_mask_pos[i].item()) + 1
                    end = min(start + window, L)
                    if start > 0:
                        labels_student[i, :start] = ignore_index
                    if end < L:
                        labels_student[i, end:] = ignore_index
            else:
                # 全句 LM：仍然要求每条样本至少包含一个 HwToken
                hw_mask = (student_input_ids == hw_token_id)
                if not torch.all(hw_mask.any(dim=1)):
                    continue

            # 注入 HwToken
            embeds = embed_layer(student_input_ids)
            hw_embed = aligner(hw_vec)
            embeds = torch.where(hw_mask.unsqueeze(-1), hw_embed.unsqueeze(1), embeds)

            # Student 前向
            outputs_student = student(
                inputs_embeds=embeds,
                attention_mask=student_attn_mask,
                labels=labels_student,
                output_hidden_states=True,
            )
            loss_lm = outputs_student.loss
            logits_student = outputs_student.logits

            # Teacher 前向（可选 KD，使用同样的 student prompt）
            if args.lambda_kd > 0:
                with torch.no_grad():
                    out_teacher = base_teacher(
                        input_ids=student_input_ids,
                        attention_mask=student_attn_mask,
                        output_hidden_states=False,
                    )
                    logits_teacher = out_teacher.logits.detach()
                log_s = F.log_softmax(logits_student / args.kd_temp, dim=-1)
                log_t = F.log_softmax(logits_teacher / args.kd_temp, dim=-1)
                kd_loss = F.kl_div(log_s, log_t.exp(), reduction="batchmean") * (args.kd_temp ** 2)
            else:
                kd_loss = torch.tensor(0.0, device=device)

            # 硬件分类与正交正则
            logits_hw = cls_head(hw_embed)
            loss_cls = F.cross_entropy(logits_hw, hw_labels)

            if args.lambda_ortho > 0:
                loss_ortho = aligner.get_ortho_loss()
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            # 对比学习（默认关闭）
            if args.lambda_ctr > 0 and hw_vec.size(0) > 1:
                hidden_states = outputs_student.hidden_states[-1]
                B, L, _ = hidden_states.shape
                mask_pos = hw_mask.float().argmax(dim=1)
                z_prog = hidden_states[torch.arange(B, device=device), mask_pos]
                z_prog_norm = F.normalize(z_prog, dim=-1)
                z_hw_norm = F.normalize(hw_embed, dim=-1)
                logits_ctr = torch.matmul(z_prog_norm, z_hw_norm.t()) / args.ctr_temp
                labels_mask = (hw_labels.unsqueeze(0) == hw_labels.unsqueeze(1)).float()
                logits_max, _ = torch.max(logits_ctr, dim=1, keepdim=True)
                logits_ctr = logits_ctr - logits_max.detach()
                exp_logits = torch.exp(logits_ctr)
                log_prob = logits_ctr - torch.log(exp_logits.sum(1, keepdim=True))
                mask_sum = labels_mask.sum(1)
                mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
                mean_log_prob_pos = (labels_mask * log_prob).sum(1) / mask_sum
                loss_ctr = -mean_log_prob_pos.mean()
            else:
                loss_ctr = torch.tensor(0.0, device=device)

            total_loss = (
                loss_lm
                + args.lambda_cls * loss_cls
                + args.lambda_ortho * loss_ortho
                + args.lambda_ctr * loss_ctr
                + args.lambda_kd * kd_loss
            )

            loss = total_loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_val = loss.item() * args.grad_accum_steps
                lm_val = loss_lm.item()
                cls_val = loss_cls.item()
                ortho_val = loss_ortho.item()
                ctr_val = loss_ctr.item()
                kd_val = kd_loss.item()

                loss_history.append(loss_val)
                if len(loss_history) > loss_window:
                    loss_history.pop(0)
                moving_avg = sum(loss_history) / len(loss_history)

                if global_step % args.log_interval == 0:
                    # 估算剩余时间
                    elapsed = time.time() - start_time
                    steps_done = max(global_step, 1)
                    steps_left = max(total_steps - global_step, 0)
                    eta_sec = elapsed / steps_done * steps_left
                    eta_h = int(eta_sec // 3600)
                    eta_m = int((eta_sec % 3600) // 60)
                    eta_s = int(eta_sec % 60)

                    msg = (
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={loss_val:.4f} lm={lm_val:.4f} cls={cls_val:.4f} "
                        f"orth={ortho_val:.4f} ctr={ctr_val:.4f} kd={kd_val:.4f} "
                        f"avg={moving_avg:.4f} "
                        f"ETA={eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                    )
                    # 控制台单行刷新
                    print(msg, end="\r", flush=True)
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss_val,
                        "loss_lm": lm_val,
                        "loss_cls": cls_val,
                        "loss_ortho": ortho_val,
                        "loss_ctr": ctr_val,
                        "loss_kd": kd_val,
                        "moving_avg": moving_avg,
                        "timestamp": datetime.now().isoformat(),
                    }
                    training_logs["logs"].append(log_entry)
                    with open(log_file, "a", encoding="utf-8") as f_log:
                        f_log.write(
                            f"Step {global_step:6d} | Loss: {loss_val:.6f} | "
                            f"LM: {lm_val:.6f} | CLS: {cls_val:.6f} | ORTH: {ortho_val:.6f} | "
                            f"CTR: {ctr_val:.6f} | KD: {kd_val:.6f} | Avg: {moving_avg:.6f} | "
                            f"ETA={eta_h:02d}:{eta_m:02d}:{eta_s:02d}\n"
                        )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt = {
                        "state_dict": aligner.state_dict(),
                        "cls_head_state_dict": cls_head.state_dict(),
                        "lora_config": lora_cfg.__dict__,
                        "prototype_names": prototype_names,
                        "prototype_keys": proto_matrix.tolist(),
                        "hw_token": args.hw_token,
                        "hw_token_id": hw_token_id,
                        "tokenizer_path": args.tokenizer_path,
                        "embed_dim": embed_layer.embedding_dim,
                        "config": vars(args),
                    }
                    torch.save(ckpt, os.path.join(ckpt_dir, "hw_aligner_lora.pt"))
                    saved_checkpoints.append(ckpt_dir)
                    if len(saved_checkpoints) > args.save_total_limit:
                        old = saved_checkpoints.pop(0)
                        if os.path.exists(old):
                            import shutil

                            shutil.rmtree(old)

    # 保存最终模型与对齐器
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_ckpt = {
        "state_dict": aligner.state_dict(),
        "cls_head_state_dict": cls_head.state_dict(),
        "lora_config": lora_cfg.__dict__,
        "prototype_names": prototype_names,
        "prototype_keys": proto_matrix.tolist(),
        "hw_token": args.hw_token,
        "hw_token_id": hw_token_id,
        "tokenizer_path": args.tokenizer_path,
        "embed_dim": embed_layer.embedding_dim,
        "config": vars(args),
    }
    torch.save(final_ckpt, os.path.join(args.output_dir, "hw_aligner_lora.pt"))

    with open(log_json_file, "w", encoding="utf-8") as f_json:
        json.dump(training_logs, f_json, indent=2, ensure_ascii=False)

    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write(f"Training finished at {datetime.now().isoformat()}\n")

    print(f"训练完成，LoRA + 对齐器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
