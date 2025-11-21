#!/usr/bin/env python3
"""Train ProtoMix hardware token aligner with frozen TLM-Base (Stage 3: Contrastive + Ortho)."""

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
    save_total_limit: int = field(default=5, metadata={"help": "Maximum number of checkpoints to keep"})
    lm_window_size: int = field(default=64, metadata={"help": "Number of tokens after hw token included in LM loss"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Weight for hardware classification loss"})
    # 对比学习与正交正则（当前默认关闭 CTR）
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for InfoNCE contrastive loss"})
    lambda_ortho: float = field(default=0.1, metadata={"help": "Weight for prototype orthogonality loss"})
    ctr_temp: float = field(default=0.1, metadata={"help": "Temperature for InfoNCE loss"})


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

    # 加载硬件嵌入与原型信息
    hardware_embeddings = load_hardware_embeddings(args.hardware_embedding_path)
    prototype_names = [name.strip() for name in args.prototype_names.split(",") if name.strip()]
    proto_matrix = build_prototype_matrix(hardware_embeddings, prototype_names)
    proto_name_to_idx = {name: idx for idx, name in enumerate(prototype_names)}

    dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    
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
    dataloader = DataLoader(hf_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True) 
    # 注意：drop_last=True 推荐开启，避免最后一个 batch size=1 导致 Contrastive Loss 报错

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

    # 硬件分类头
    num_hw_classes = len(prototype_names)
    cls_head = nn.Linear(embed_dim, num_hw_classes)
    cls_head.to(device)

    optimizer = torch.optim.AdamW(
        list(aligner.parameters()) + list(cls_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    
    dataset_len = len(hf_dataset)
    steps_per_epoch = math.ceil(dataset_len / args.batch_size / max(1, args.grad_accum_steps))
    total_steps = steps_per_epoch * args.num_epochs
    
    loss_history = []
    loss_window = 50
    
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_checkpoints = []

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_basename = os.path.basename(args.output_dir.rstrip("/"))
    log_file = os.path.join(log_dir, f"train_hw_injection_{log_basename}_{timestamp}.log")
    log_json_file = os.path.join(log_dir, f"train_hw_injection_{log_basename}_{timestamp}.json")
    
    training_logs = {
        "config": vars(args),
        "total_steps": total_steps,
        "logs": []
    }
    
    print(f"\n[数据集信息]")
    print(f"  数据集大小: {dataset_len:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  策略: LM + CLS + InfoNCE + Ortho")
    print(f"  权重: CLS={args.lambda_cls}, CTR={args.lambda_ctr}, Ortho={args.lambda_ortho}")
    print(f"\n训练日志将保存到: {log_file}")
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

            # 1. 局部 LM Loss 准备
            if args.lm_window_size and args.lm_window_size > 0:
                ignore_index = -100
                B, L = labels.size()
                first_mask_pos = hw_mask.float().argmax(dim=1)
                window = args.lm_window_size
                for i in range(B):
                    start = int(first_mask_pos[i].item()) + 1
                    end = min(start + window, L)
                    if start > 0:
                        labels[i, :start] = ignore_index
                    if end < L:
                        labels[i, end:] = ignore_index

            # 2. 注入
            hw_embed = aligner(hw_vec)  # [B, d]
            embeds = torch.where(hw_mask.unsqueeze(-1), hw_embed.unsqueeze(1), embeds)

            outputs = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )

            loss_lm = outputs.loss

            # 3. 提取 Context 表示 (用于对比学习)
            hidden_states = outputs.hidden_states[-1]  # [B, L, d]
            B, L, _ = hidden_states.shape
            mask_pos = hw_mask.float().argmax(dim=1)
            
            # z_prog: 程序上下文在 [MASK] 处的表示
            z_prog = hidden_states[torch.arange(B, device=device), mask_pos] # [B, d]

            # 4. 硬件分类 Loss：直接监督 hw_embed，避免分类头只依赖上下文作弊
            logits_hw = cls_head(hw_embed)
            loss_cls = F.cross_entropy(logits_hw, hw_labels)

            # 5. (修正版) 有监督对比学习 Loss (SupCon)
            if args.lambda_ctr > 0 and B > 1:
                z_prog_norm = F.normalize(z_prog, dim=-1)
                z_hw_norm = F.normalize(hw_embed, dim=-1)
                
                # [B, B] 相似度矩阵
                logits = torch.matmul(z_prog_norm, z_hw_norm.t()) / args.ctr_temp
                
                # 构建 Mask，标记出哪些是同类硬件
                # hw_labels: [B]，labels_mask[i, j] = 1 表示 i 和 j 是同一种硬件
                labels_mask = (hw_labels.unsqueeze(0) == hw_labels.unsqueeze(1)).float()
                
                # 为了数值稳定性，减去最大值
                logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = logits - logits_max.detach()
                
                # 计算 LogSoftmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
                
                # SupCon：只计算正样本位置的 log_prob 平均值
                mask_sum = labels_mask.sum(1)
                mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
                mean_log_prob_pos = (labels_mask * log_prob).sum(1) / mask_sum
                
                loss_ctr = - mean_log_prob_pos.mean()
            else:
                loss_ctr = torch.tensor(0.0, device=device)

            # 6. 正交正则 Loss
            if args.lambda_ortho > 0:
                loss_ortho = aligner.get_ortho_loss()
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            # 总 Loss
            total_loss = (
                loss_lm + 
                args.lambda_cls * loss_cls + 
                args.lambda_ctr * loss_ctr + 
                args.lambda_ortho * loss_ortho
            )
            
            loss = total_loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    loss_val = loss.item() * args.grad_accum_steps
                    lm_val = loss_lm.item()
                    cls_val = loss_cls.item()
                    ctr_val = loss_ctr.item()
                    orth_val = loss_ortho.item()
                    
                    loss_history.append(loss_val)
                    if len(loss_history) > loss_window:
                        loss_history.pop(0)
                    moving_avg = sum(loss_history) / len(loss_history)
                    
                    progress.set_postfix({
                        "loss": f"{loss_val:.2f}",
                        "lm": f"{lm_val:.2f}",
                        "cls": f"{cls_val:.2f}",
                        "ctr": f"{ctr_val:.2f}",
                        "orth": f"{orth_val:.2f}"
                    })
                    
                    log_entry = {
                        "step": global_step,
                        "loss": loss_val,
                        "loss_lm": lm_val,
                        "loss_cls": cls_val,
                        "loss_ctr": ctr_val,
                        "loss_ortho": orth_val,
                        "timestamp": datetime.now().isoformat()
                    }
                    training_logs["logs"].append(log_entry)
                    
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"Step {global_step:5d} | "
                            f"Loss: {loss_val:.4f} | LM: {lm_val:.4f} | CLS: {cls_val:.4f} | "
                            f"CTR: {ctr_val:.4f} | ORTH: {orth_val:.4f}\n"
                        )
                
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    ckpt = {
                        "state_dict": aligner.state_dict(),
                        "cls_head_state_dict": cls_head.state_dict(),
                        "config": vars(args),
                        "prototype_names": prototype_names,
                        "prototype_keys": proto_matrix.tolist(),
                        "hw_token": args.hw_token,
                        "hw_token_id": hw_token_id,
                        "tokenizer_path": args.tokenizer_path,
                        "embed_dim": embed_dim,
                    }
                    torch.save(ckpt, os.path.join(checkpoint_path, "hw_aligner.pt"))
                    saved_checkpoints.append(checkpoint_path)
                    if len(saved_checkpoints) > args.save_total_limit:
                        shutil.rmtree(saved_checkpoints.pop(0))

    aligner.eval()
    
    # 保存最终模型
    ckpt = {
        "state_dict": aligner.state_dict(),
        "cls_head_state_dict": cls_head.state_dict(),
        "config": vars(args),
        "prototype_names": prototype_names,
        "prototype_keys": proto_matrix.tolist(),
        "hw_token": args.hw_token,
        "hw_token_id": hw_token_id,
        "tokenizer_path": args.tokenizer_path,
        "embed_dim": embed_dim,
    }
    torch.save(ckpt, os.path.join(args.output_dir, "hw_aligner.pt"))
    
    # 保存日志
    with open(log_json_file, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)
    print(f"训练结束，日志已保存至 {log_file}")


if __name__ == "__main__":
    train()
