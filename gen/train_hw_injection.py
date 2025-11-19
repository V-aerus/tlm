#!/usr/bin/env python3
"""Train ProtoMix hardware token aligner with frozen TLM-Base."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm

from modeling.hw_injection import ProtoMixAligner, build_prototype_matrix


@dataclass
class TrainHwArgs:
    model_path: str = field(metadata={"help": "Path to frozen base model"})
    tokenizer_path: str = field(metadata={"help": "Path to tokenizer"})
    dataset_path: str = field(metadata={"help": "JSONL file produced by make_dataset with student text"})
    output_dir: str = field(metadata={"help": "Where to save the trained aligner"})
    hardware_embedding_path: str = field(default="Embedding/hardware_embeddings_v2.json", metadata={"help": "Hardware embedding json"})
    prototype_names: str = field(default="nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge", metadata={"help": "Comma separated hardware names used as prototypes"})
    hw_token: str = field(default="[MASK]", metadata={"help": "Placeholder token used in student text"})
    batch_size: int = field(default=4, metadata={"help": "Training batch size"})
    num_epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    lr: float = field(default=5e-4, metadata={"help": "Learning rate for ProtoMix"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    grad_accum_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})
    max_length: int = field(default=512, metadata={"help": "Tokenizer max length"})
    temperature: float = field(default=1.0, metadata={"help": "Initial softmax temperature"})
    trainable_temperature: bool = field(default=False, metadata={"help": "Learn the temperature parameter"})
    device: str = field(default="cuda", metadata={"help": "Device to run training on"})
    log_interval: int = field(default=20, metadata={"help": "Logging frequency in steps"})
    warmup_steps: int = field(default=1000, metadata={"help": "Number of warmup steps"})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "LR scheduler type: linear|cosine|constant"})
    sample_fraction: float = field(default=None, metadata={"help": "Optional fraction of data to sample (0,1]"})
    hw_noise_std: float = field(default=0.0, metadata={"help": "Std of Gaussian noise added to hw_emb (0 to disable)"})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every N steps"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})
    lm_window_size: int = field(default=64, metadata={"help": "Number of tokens after hw token included in LM loss"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Weight for hardware classification loss"})


def load_hardware_embeddings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


class HwStudentDataset(Dataset):
    def __init__(self, dataset, proto_name_to_idx: Optional[Dict[str, int]] = None):
        self.dataset = dataset
        self.proto_name_to_idx = proto_name_to_idx or {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        hw_name = item.get("hw_name")
        hw_label = None
        if hw_name is not None and hw_name in self.proto_name_to_idx:
            hw_label = self.proto_name_to_idx[hw_name]
        return {
            "text_student": item["text_student"],
            "hw_emb": item["hw_emb"],
            "hw_label": hw_label,
        }


class HwCollator:
    def __init__(self, tokenizer, hw_token_id, max_length, hw_noise_std=0.0):
        self.tokenizer = tokenizer
        self.hw_token_id = hw_token_id
        self.max_length = max_length
        self.hw_noise_std = hw_noise_std

    def __call__(self, batch):
        texts = [b["text_student"] for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        hw = torch.tensor([b["hw_emb"] for b in batch], dtype=torch.float32)
        if self.hw_noise_std > 0:
            hw = hw + torch.randn_like(hw) * self.hw_noise_std
        labels = enc["input_ids"].clone()
        hw_labels = torch.tensor(
            [b["hw_label"] for b in batch], dtype=torch.long
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "hw": hw,
            "hw_label": hw_labels,
        }


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def train():
    parser = HfArgumentParser(TrainHwArgs)
    args: TrainHwArgs = parser.parse_args_into_dataclasses()[0]

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 用于异常处理的变量
    log_file = None
    log_json_file = None
    training_logs = None
    global_step = 0
    aligner = None
    
    def save_logs_on_exit():
        """在退出时保存日志"""
        nonlocal log_file, log_json_file, training_logs, global_step
        if log_file and training_logs is not None:
            try:
                training_logs["final_step"] = global_step
                training_logs["interrupted"] = True
                training_logs["completed_at"] = datetime.now().isoformat()
                if log_json_file:
                    with open(log_json_file, "w", encoding="utf-8") as f:
                        json.dump(training_logs, f, indent=2, ensure_ascii=False)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n训练中断 | 当前步数: {global_step} | 中断时间: {datetime.now().isoformat()}\n")
                print(f"\n[已保存日志] 训练中断，日志已保存到: {log_file}")
            except Exception as e:
                print(f"\n[警告] 保存日志时出错: {e}")
    
    def signal_handler(sig, frame):
        """处理 Ctrl+C 信号"""
        print("\n\n[收到中断信号] 正在保存日志...")
        save_logs_on_exit()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 加载硬件嵌入与原型信息（用于 ProtoMix 和分类头）
    hardware_embeddings = load_hardware_embeddings(args.hardware_embedding_path)
    prototype_names = [name.strip() for name in args.prototype_names.split(",") if name.strip()]
    proto_matrix = build_prototype_matrix(hardware_embeddings, prototype_names)
    proto_name_to_idx = {name: idx for idx, name in enumerate(prototype_names)}

    dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    required_cols = {"text_student", "hw_emb", "hw_name"}
    missing = required_cols - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    def _valid_example(x):
        if x["text_student"] is None or x["hw_emb"] is None:
            return False
        name = x.get("hw_name")
        return name is not None and name in proto_name_to_idx

    dataset = dataset.filter(_valid_example)
    if args.sample_fraction and 0 < args.sample_fraction < 1:
        dataset = dataset.shuffle(seed=0).select(range(int(len(dataset) * args.sample_fraction)))
    hf_dataset = HwStudentDataset(dataset, proto_name_to_idx=proto_name_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    hw_token_id = tokenizer.convert_tokens_to_ids(args.hw_token)
    if hw_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not know token '{args.hw_token}'")

    collator = HwCollator(tokenizer, hw_token_id, args.max_length, hw_noise_std=args.hw_noise_std)
    dataloader = DataLoader(hf_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    freeze_model_parameters(model)
    model.to(device)
    model.eval()

    embed_layer = model.get_input_embeddings()
    embed_dim = embed_layer.embedding_dim

    aligner = ProtoMixAligner(
        proto_matrix,
        embed_dim=embed_dim,
        temperature=args.temperature,
        trainable_temperature=args.trainable_temperature,
    )
    aligner.to(device)

    # 硬件分类头（阶段 2）
    num_hw_classes = len(prototype_names)
    cls_head = nn.Linear(embed_dim, num_hw_classes)
    cls_head.to(device)

    optimizer = torch.optim.AdamW(
        list(aligner.parameters()) + list(cls_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    
    # 直接使用数据集大小计算总步数，避免 DataLoader len() 的缓存问题
    dataset_len = len(hf_dataset)
    steps_per_epoch = math.ceil(dataset_len / args.batch_size / max(1, args.grad_accum_steps))
    total_steps = steps_per_epoch * args.num_epochs
    
    # Loss 移动平均（用于平滑显示）
    loss_history = []
    loss_window = 50  # 移动平均窗口大小
    
    # Checkpoint 管理
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_checkpoints = []  # 保存的 checkpoint 路径列表

    # 设置日志保存
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_basename = os.path.basename(args.output_dir.rstrip("/"))
    log_file = os.path.join(log_dir, f"train_hw_injection_{log_basename}_{timestamp}.log")
    log_json_file = os.path.join(log_dir, f"train_hw_injection_{log_basename}_{timestamp}.json")
    
    training_logs = {
        "config": vars(args),
        "total_steps": total_steps,
        "total_samples": dataset_len,
        "logs": []
    }
    
    print(f"\n[数据集信息]")
    print(f"  数据集大小: {dataset_len:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  每个 epoch 步数: {steps_per_epoch:,}")
    print(f"  总步数: {total_steps:,}")
    print(f"\n训练日志将保存到: {log_file}")
    print(f"训练指标将保存到: {log_json_file}")
    print()

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(args.num_epochs):
        aligner.train()
        cls_head.train()
        optimizer.zero_grad()
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            hw_vec = batch["hw"].to(device)
            hw_labels = batch["hw_label"].to(device)

            embeds = embed_layer(input_ids)
            hw_mask = (input_ids == hw_token_id)
            if not torch.all(hw_mask.any(dim=1)):
                raise ValueError("Each sample must contain at least one hardware token")

            # 局部 LM CE：只在 hw_token 之后的一段窗口内计算 loss
            if args.lm_window_size and args.lm_window_size > 0:
                ignore_index = -100
                B, L = labels.size()
                first_mask_pos = hw_mask.float().argmax(dim=1)  # 每行第一个 [MASK]
                window = args.lm_window_size
                for i in range(B):
                    start = int(first_mask_pos[i].item()) + 1
                    end = min(start + window, L)
                    if start > 0:
                        labels[i, :start] = ignore_index
                    if end < L:
                        labels[i, end:] = ignore_index

            hw_embed = aligner(hw_vec)
            embeds = torch.where(hw_mask.unsqueeze(-1), hw_embed.unsqueeze(1), embeds)

            outputs = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )

            loss_lm = outputs.loss

            # 硬件分类头：使用 [MASK] 位置的隐藏状态
            hidden_states = outputs.hidden_states[-1]  # [B, L, d]
            B, L, _ = hidden_states.shape
            mask_pos = hw_mask.float().argmax(dim=1)  # [B]
            cls_features = hidden_states[torch.arange(B, device=device), mask_pos]
            logits_hw = cls_head(cls_features)
            loss_cls = F.cross_entropy(logits_hw, hw_labels)

            total_loss = loss_lm + args.lambda_cls * loss_cls
            loss = total_loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    loss_value = loss.item() * args.grad_accum_steps
                    loss_lm_value = loss_lm.item()
                    loss_cls_value = loss_cls.item()
                    
                    # 更新移动平均
                    loss_history.append(loss_value)
                    if len(loss_history) > loss_window:
                        loss_history.pop(0)
                    moving_avg_loss = sum(loss_history) / len(loss_history)
                    
                    # 显示瞬时 loss 和移动平均 loss
                    progress.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "lm": f"{loss_lm_value:.4f}",
                        "cls": f"{loss_cls_value:.4f}",
                        "avg": f"{moving_avg_loss:.4f}",
                    })
                    
                    # 记录日志
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss_value,
                        "loss_lm": loss_lm_value,
                        "loss_cls": loss_cls_value,
                        "moving_avg_loss": moving_avg_loss,
                        "timestamp": datetime.now().isoformat()
                    }
                    training_logs["logs"].append(log_entry)
                    
                    # 写入文本日志（包含移动平均）
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"Step {global_step:6d} | Epoch {epoch+1}/{args.num_epochs} | "
                            f"Loss: {loss_value:.6f} | LM: {loss_lm_value:.6f} | "
                            f"CLS: {loss_cls_value:.6f} | Avg: {moving_avg_loss:.6f}\n"
                        )
                
                # 保存 checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    ckpt = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "state_dict": aligner.state_dict(),
                        "cls_head_state_dict": cls_head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "prototype_names": prototype_names,
                        "prototype_keys": proto_matrix.tolist(),
                        "hw_token": args.hw_token,
                        "hw_token_id": hw_token_id,
                        "tokenizer_path": args.tokenizer_path,
                        "embed_dim": embed_dim,
                        "config": vars(args),
                    }
                    torch.save(ckpt, os.path.join(checkpoint_path, "hw_aligner.pt"))
                    
                    saved_checkpoints.append(checkpoint_path)
                    print(f"\n[Checkpoint] 已保存检查点到: {checkpoint_path}")
                    
                    # 清理旧 checkpoint（保留最新的 N 个）
                    if len(saved_checkpoints) > args.save_total_limit:
                        old_checkpoint = saved_checkpoints.pop(0)
                        if os.path.exists(old_checkpoint):
                            shutil.rmtree(old_checkpoint)
                            print(f"[Checkpoint] 已删除旧检查点: {old_checkpoint}")

    aligner.eval()
    cls_head.eval()
    
    # 保存最终训练日志
    training_logs["final_step"] = global_step
    training_logs["completed_at"] = datetime.now().isoformat()
    with open(log_json_file, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2, ensure_ascii=False)
    
    # 写入最终日志条目
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n训练完成 | 总步数: {global_step} | 完成时间: {datetime.now().isoformat()}\n")
    
    print(f"\n训练日志已保存:")
    print(f"  - 文本日志: {log_file}")
    print(f"  - JSON日志: {log_json_file}")
    
    ckpt = {
        "state_dict": aligner.state_dict(),
        "cls_head_state_dict": cls_head.state_dict(),
        "prototype_names": prototype_names,
        "prototype_keys": proto_matrix.tolist(),
        "hw_token": args.hw_token,
        "hw_token_id": hw_token_id,
        "tokenizer_path": args.tokenizer_path,
        "embed_dim": embed_dim,
        "config": vars(args),
    }
    torch.save(ckpt, os.path.join(args.output_dir, "hw_aligner.pt"))
    with open(os.path.join(args.output_dir, "hw_aligner_config.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt["config"], f, indent=2)


if __name__ == "__main__":
    train()
