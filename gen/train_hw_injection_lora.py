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
import sys
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
import re

# 确保可以从项目根目录导入 modeling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modeling.hw_injection import ProtoMixAligner, build_prototype_matrix

try:
    from peft import LoraConfig, get_peft_model
    from peft import PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


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
    hw_token: str = field(default="[MASK] [MASK] [MASK] [MASK]", metadata={"help": "Placeholder token used in student text"})

    # 损失权重
    lm_window_size: int = field(default=0, metadata={"help": "Tokens after HwToken included in LM loss (0 = full sequence)"})
    lambda_cls: float = field(default=0.01, metadata={"help": "Weight for hardware classification loss (phase1: weak)"} )
    lambda_ortho: float = field(default=0.1, metadata={"help": "Weight for prototype orthogonality loss"})
    lambda_ctr: float = field(default=0.0, metadata={"help": "Weight for contrastive loss (default off)"})
    lambda_kd: float = field(default=0.0, metadata={"help": "Weight for KD loss (phase1: off)"})
    ctr_temp: float = field(default=0.1, metadata={"help": "Temperature for contrastive loss"})
    kd_temp: float = field(default=1.0, metadata={"help": "Temperature for KD"})
    lambda_geom: float = field(default=1.0, metadata={"help": "Weight for geometry loss (aligner outputs vs base targets)"})
    lambda_geom_cos: float = field(default=1.0, metadata={"help": "Weight for cosine term in geometry loss"})
    hw_target_path: Optional[str] = field(default=None, metadata={"help": "Precomputed hw token targets .pt"})

    # LoRA 配置
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(
        default="mlp.c_fc,mlp.c_proj",
        metadata={"help": "Comma-separated module name fragments for LoRA (phase1: focus on MLP)"},
    )
    last_lora_layers: int = field(default=4, metadata={"help": "Only apply LoRA to the last N transformer blocks (0=disable filter)"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Optional checkpoint dir to resume (expects student adapter + hw_aligner_lora.pt)"})

    # 设备与日志
    device: str = field(default="cuda", metadata={"help": "Device"})
    log_interval: int = field(default=50, metadata={"help": "Log every N steps"})
    save_steps: int = field(default=2000, metadata={"help": "Save checkpoint every N steps"})
    save_total_limit: int = field(default=5, metadata={"help": "Max checkpoints to keep"})
    freeze_aligner: bool = field(default=False, metadata={"help": "If true, aligner params are frozen and excluded from optimizer"})


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def collect_target_modules(model, base_patterns: List[str], last_layers: int) -> List[str]:
    """
    动态选择 LoRA 作用的模块：仅在最后 last_layers 个 block 上，优先匹配给定片段。
    如果无法解析层号或 last_layers<=0，则退回全局匹配 base_patterns。
    """
    # 收集所有模块名，优先使用底层 base_model，避免 PeftModel 包装后前缀变化
    iter_model = getattr(model, "base_model", model)
    all_names = [name for name, _ in iter_model.named_modules()]

    # 允许任意前缀，只要出现 .h.<idx>. 即认为是 transformer block
    layer_pattern = re.compile(r"\.h\.(\d+)\.")
    layer_to_modules = {}
    for name in all_names:
        if not any(pat in name for pat in base_patterns):
            continue
        m = layer_pattern.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        layer_to_modules.setdefault(idx, []).append(name)

    if not layer_to_modules:
        if last_layers > 0:
            sample = all_names[:20]
            raise RuntimeError(
                f"last_layers={last_layers} but no transformer blocks matched patterns {base_patterns}. "
                f"Sample module names: {sample}"
            )
        fallback = []
        for name in all_names:
            if any(pat in name for pat in base_patterns):
                if name not in fallback:
                    fallback.append(name)
        print(f"[LoRA target modules] last_layers=0, using global patterns, resolved={len(fallback)}")
        return fallback

    all_layers = sorted(layer_to_modules.keys())
    if last_layers > 0 and len(all_layers) > last_layers:
        target_layers = all_layers[-last_layers:]
    else:
        target_layers = all_layers

    target_modules = []
    for idx in target_layers:
        for name in layer_to_modules[idx]:
            if name not in target_modules:
                target_modules.append(name)

    print(f"[LoRA target modules] last_layers={last_layers}, resolved_layers={target_layers}, resolved={len(target_modules)} modules")
    print(f"[LoRA target sample] {target_modules[:8]}")
    return target_modules


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
                    "hw_name": hw_name,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class HwLoraCollator:
    def __init__(self, tokenizer, max_length: int, hw_token_id: int, hw_token_ids: list[int], lm_window_size: int, hw_noise_std: float = 0.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hw_token_id = hw_token_id
        self.hw_token_ids = hw_token_ids
        self.lm_window_size = lm_window_size
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

        ignore_index = -100
        labels_student = enc_student["input_ids"].clone()
        labels_teacher = enc_teacher["input_ids"].clone()
        # 默认全保留，具体遮罩分支处理
        if self.lm_window_size == 0:
            if len(self.hw_token_ids) > 1:
                # 多MASK场景：留到训练循环按span遮罩
                pass
            else:
                hw_mask = (enc_student["input_ids"] == self.hw_token_id)
                # 初始化为忽略
                labels_student.fill_(ignore_index)
                labels_teacher.fill_(ignore_index)
                for i in range(hw_mask.size(0)):
                    pos = torch.nonzero(hw_mask[i], as_tuple=False)
                    if len(pos) == 0:
                        continue  # 无 HwToken，整句忽略
                    first = pos[0].item()
                    # student：仅保留 [MASK] 之后的 schedule
                    labels_student[i, first + 1 :] = enc_student["input_ids"][i, first + 1 :]
                    valid_len = int((labels_student[i] != ignore_index).sum().item())
                    if valid_len <= 0:
                        continue
                    # teacher：从真实 prompt 的末尾对齐同样长度的 schedule 片段
                    teacher_len = int(enc_teacher["attention_mask"][i].sum().item())
                    valid_len = min(valid_len, teacher_len)
                    start_t = max(teacher_len - valid_len, 0)
                    labels_teacher[i, start_t:teacher_len] = enc_teacher["input_ids"][i, start_t:teacher_len]
        else:
            # 窗口模式：初始不 mask，留给训练循环按窗口处理
            labels_teacher.fill_(ignore_index)

        hw = torch.tensor([b["hw_emb"] for b in batch], dtype=torch.float32)
        if self.hw_noise_std > 0:
            hw = hw + torch.randn_like(hw) * self.hw_noise_std

        return {
            "input_ids_teacher": enc_teacher["input_ids"],
            "attention_mask_teacher": enc_teacher["attention_mask"],
            "input_ids_student": enc_student["input_ids"],
            "attention_mask_student": enc_student["attention_mask"],
            "labels_student": labels_student,
            "labels_teacher": labels_teacher,
            "hw": hw,
            "hw_label": torch.tensor([b["hw_label"] for b in batch], dtype=torch.long),
            "hw_name": [b["hw_name"] for b in batch],
        }


class HwTargetBank:
    def __init__(self, path: Optional[str], device: torch.device):
        self.targets = None
        if path:
            ck = torch.load(path, map_location="cpu")
            self.targets = {}
            for hw, segs in ck.items():
                self.targets[hw] = {}
                for name, info in segs.items():
                    vec = info["vec"]
                    active = info.get("active", True)
                    if active:
                        self.targets[hw][name] = {"vec": vec.to(device), "active": True}
                    else:
                        self.targets[hw][name] = {"vec": vec, "active": False}

    def get(self, hw_name: str):
        if self.targets is None:
            return None
        return self.targets.get(hw_name, None)

def find_subsequence(seq: torch.Tensor, pattern: list[int]) -> int:
    """Return start index of first occurrence of pattern in seq (CPU list), or -1."""
    seq_list = seq.tolist()
    n, m = len(seq_list), len(pattern)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if seq_list[i : i + m] == pattern:
            return i
    return -1


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

    target_bank = HwTargetBank(args.hw_target_path, device)

    hw_token_ids = tokenizer(args.hw_token, add_special_tokens=False)["input_ids"]
    if not hw_token_ids:
        raise ValueError(f"Tokenizer failed to tokenize hw_token '{args.hw_token}'")
    # 允许 1 或 4 token 的占位符
    if len(hw_token_ids) not in (1, 4):
        raise ValueError(f"Expected 1 or 4 tokens for hw_token, got {len(hw_token_ids)}: {hw_token_ids}")
    hw_token_id = hw_token_ids[0]
    if hw_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not know hardware token '{args.hw_token}'")

    base_teacher = AutoModelForCausalLM.from_pretrained(args.model_path)
    base_teacher.to(device)
    base_teacher.eval()
    for p in base_teacher.parameters():
        p.requires_grad = False

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft (LoRA) is required for this script.")

    lora_cfg = None
    lora_cfg_dict = None
    proto_matrix = None

    if args.resume_from_checkpoint:
        print(f"[Resume] Loading student from {args.resume_from_checkpoint}")
        student = AutoModelForCausalLM.from_pretrained(args.model_path)
        student = PeftModel.from_pretrained(student, args.resume_from_checkpoint)
        student.to(device)
        aligner_ckpt = torch.load(os.path.join(args.resume_from_checkpoint, "hw_aligner_lora.pt"), map_location=device)
        lora_cfg_dict = aligner_ckpt.get("lora_config", {})
        lora_cfg = None  # resume 分支默认不重建 LoraConfig 对象
        proto = torch.tensor(aligner_ckpt["prototype_keys"], dtype=torch.float32, device=device)
        embed_layer_tmp = student.get_input_embeddings()
        embed_dim_tmp = aligner_ckpt.get("embed_dim", embed_layer_tmp.embedding_dim)
        cfg_meta = aligner_ckpt.get("config", {})
        temperature = cfg_meta.get("temperature", aligner_ckpt.get("temperature", 1.0))
        trainable_temp = cfg_meta.get("trainable_temperature", False)
        aligner = ProtoMixAligner(proto, embed_dim=embed_dim_tmp, temperature=temperature, trainable_temperature=trainable_temp)
        aligner.load_state_dict(aligner_ckpt["state_dict"])
        aligner.to(device)
        aligner.eval()
        proto_matrix = proto  # 用于后续保存
        base_patterns = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        target_module_names = collect_target_modules(student, base_patterns, args.last_lora_layers)
        embed_layer = student.get_input_embeddings()
        # 恢复 prototype_names（ckpt 保存）或回退到参数
        prototype_names = aligner_ckpt.get("prototype_names", None)
        if prototype_names is None:
            prototype_names = [name.strip() for name in args.prototype_names.split(",") if name.strip()]
        # 保存 ckpt 中的 lora 配置
        lora_cfg_dict = aligner_ckpt.get("lora_config", {})
        # 打印可训练参数，确认 LoRA 层号
        print("[DEBUG] Trainable parameter names (LoRA):")
        for n, p in student.named_parameters():
            if p.requires_grad:
                print("  ", n)
        # 同时打印 target_module_names 以便核对
        print(f"[DEBUG] target_module_names (len={len(target_module_names)}):")
        for n in target_module_names:
            print("  ", n)
    else:
        student = AutoModelForCausalLM.from_pretrained(args.model_path)
        base_patterns = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        target_module_names = collect_target_modules(student, base_patterns, args.last_lora_layers)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_module_names,
        )
        lora_cfg_dict = lora_cfg.__dict__
        student = get_peft_model(student, lora_cfg)
        student.print_trainable_parameters()
        student.to(device)

        embed_layer = student.get_input_embeddings()
        hardware_embeddings = load_hardware_embeddings(args.hardware_embedding_path)
        prototype_names = [name.strip() for name in args.prototype_names.split(",") if name.strip()]
        proto_matrix = build_prototype_matrix(hardware_embeddings, prototype_names)
        aligner = ProtoMixAligner(proto_matrix, embed_dim=embed_layer.embedding_dim, temperature=1.0, trainable_temperature=False)
        aligner.to(device)

    if lora_cfg_dict is None:
        lora_cfg_dict = {}
    assert (lora_cfg is not None) or (lora_cfg_dict is not None), "LoRA config missing in both lora_cfg and lora_cfg_dict"

    num_hw_classes = len(prototype_names)
    cls_head = nn.Linear(student.get_input_embeddings().embedding_dim, num_hw_classes).to(device)

    proto_name_to_idx = {name: idx for idx, name in enumerate(prototype_names)}

    # 数据集
    raw_ds = load_dataset("json", data_files=args.dataset_path)["train"]
    if args.sample_fraction and 0 < args.sample_fraction < 1:
        raw_ds = raw_ds.shuffle(seed=0).select(range(int(len(raw_ds) * args.sample_fraction)))
    dataset = HwLoraDataset(raw_ds, proto_name_to_idx)

    collator = HwLoraCollator(
        tokenizer,
        max_length=args.max_length,
        hw_token_id=hw_token_id,
        hw_token_ids=hw_token_ids,
        lm_window_size=args.lm_window_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    # 冻结 aligner（Stage1.5）可选
    if args.freeze_aligner:
        for p in aligner.parameters():
            p.requires_grad = False
        aligner.eval()

    # 优化器与调度器
    optimizer_groups = [
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": list(cls_head.parameters()), "lr": args.lr_aligner},
    ]
    if not args.freeze_aligner:
        optimizer_groups.append({"params": list(aligner.parameters()), "lr": args.lr_aligner})

    optimizer = torch.optim.AdamW(
        optimizer_groups,
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
    print(f"[LoRA target modules] last_layers={args.last_lora_layers}, resolved={len(target_module_names)} modules")
    print(f"[LoRA target sample] {target_module_names[:5]}")
    print(f"[损失权重] LM, CLS={args.lambda_cls}, ORTH={args.lambda_ortho}, CTR={args.lambda_ctr}, KD={args.lambda_kd}, GEOM={args.lambda_geom}")
    print(f"日志文件: {log_file}")
    if args.freeze_aligner:
        print("[INFO] Aligner is frozen (parameters not optimized)")

    debug_gen_interval = max(args.log_interval * 5, 500)

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
            labels_teacher = batch.get("labels_teacher")
            if labels_teacher is not None:
                labels_teacher = labels_teacher.to(device)
            hw_vec = batch["hw"].to(device)
            hw_labels = batch["hw_label"].to(device)

            multi_mask = len(hw_token_ids) > 1
            span_positions = []
            if multi_mask:
                for i in range(student_input_ids.size(0)):
                    pos = find_subsequence(student_input_ids[i].cpu(), hw_token_ids)
                    if pos < 0 or pos + len(hw_token_ids) > student_input_ids.size(1):
                        raise ValueError("HwToken span not found or truncated in sample")
                    span_positions.append(pos)

            # 局部 LM 窗口处理
            if args.lm_window_size and args.lm_window_size > 0:
                ignore_index = -100
                B, L = labels_student.size()
                if multi_mask:
                    window = args.lm_window_size
                    for i, pos in enumerate(span_positions):
                        start = pos + len(hw_token_ids)
                        end = min(start + window, L)
                        labels_student[i, :start] = ignore_index
                        if end < L:
                            labels_student[i, end:] = ignore_index
                else:
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
                if multi_mask:
                    # 先保留，后续按 span 屏蔽 prompt
                    pass
                else:
                    hw_mask = (student_input_ids == hw_token_id)
                    if not torch.all(hw_mask.any(dim=1)):
                        continue

            # 全句模式 + 多MASK：按 span 屏蔽 prompt
            if (not args.lm_window_size) or args.lm_window_size == 0:
                if multi_mask:
                    ignore_index = -100
                    for i, pos in enumerate(span_positions):
                        labels_student[i, : pos + len(hw_token_ids)] = ignore_index

            # 若开启 KD，确保 teacher 标签与 schedule 段对齐
            if args.lambda_kd > 0:
                ignore_index = -100
                if labels_teacher is None:
                    labels_teacher = torch.full_like(teacher_input_ids, ignore_index)
                else:
                    labels_teacher = labels_teacher.clone()
                B = labels_student.size(0)
                for i in range(B):
                    valid_len = int((labels_student[i] != ignore_index).sum().item())
                    if valid_len <= 0:
                        labels_teacher[i].fill_(ignore_index)
                        continue
                    teacher_len = int(teacher_attn_mask[i].sum().item())
                    valid_len = min(valid_len, teacher_len)
                    start_t = max(teacher_len - valid_len, 0)
                    labels_teacher[i].fill_(ignore_index)
                    labels_teacher[i, start_t:teacher_len] = teacher_input_ids[i, start_t:teacher_len]

            # 注入 HwToken
            embeds = embed_layer(student_input_ids)
            if multi_mask:
                hw_embed = aligner(hw_vec, split_heads=True)  # (B,4,D)
                for i, pos in enumerate(span_positions):
                    embeds[i, pos : pos + len(hw_token_ids), :] = hw_embed[i]
                hw_embed_for_cls = hw_embed.mean(dim=1)  # 汇聚后做分类
                hw_embed_all = hw_embed
            else:
                hw_mask = (student_input_ids == hw_token_id)
                hw_embed = aligner(hw_vec)
                embeds = torch.where(hw_mask.unsqueeze(-1), hw_embed.unsqueeze(1), embeds)
                hw_embed_for_cls = hw_embed
                hw_embed_all = None

            # Student 前向
            outputs_student = student(
                inputs_embeds=embeds,
                attention_mask=student_attn_mask,
                labels=labels_student,
                output_hidden_states=True,
            )
            loss_lm = outputs_student.loss
            logits_student = outputs_student.logits

            # Teacher 前向（可选 KD，使用“oracle prompt”）
            if args.lambda_kd > 0:
                with torch.no_grad():
                    out_teacher = base_teacher(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attn_mask,
                        output_hidden_states=False,
                    )
                    logits_teacher = out_teacher.logits.detach()
                # Schedule 段对齐：仅取有效 label 对应的 logits，长度不一致则截断
                student_mask_flat = (labels_student != -100).view(-1)
                teacher_mask_flat = (labels_teacher != -100).view(-1)
                student_valid = logits_student.view(-1, logits_student.size(-1))[student_mask_flat]
                teacher_valid = logits_teacher.view(-1, logits_teacher.size(-1))[teacher_mask_flat]
                if student_valid.size(0) == 0 or teacher_valid.size(0) == 0:
                    kd_loss = torch.tensor(0.0, device=device)
                else:
                    min_len = min(student_valid.size(0), teacher_valid.size(0))
                    if student_valid.size(0) != teacher_valid.size(0):
                        student_valid = student_valid[-min_len:]
                        teacher_valid = teacher_valid[-min_len:]
                    log_s = F.log_softmax(student_valid / args.kd_temp, dim=-1)
                    log_t = F.log_softmax(teacher_valid / args.kd_temp, dim=-1)
                    kd_loss = F.kl_div(log_s, log_t.exp(), reduction="batchmean") * (args.kd_temp ** 2)
            else:
                kd_loss = torch.tensor(0.0, device=device)

            # 硬件分类与正交正则
            logits_hw = cls_head(hw_embed_for_cls)
            loss_cls = F.cross_entropy(logits_hw, hw_labels)

            if args.lambda_ortho > 0:
                loss_ortho = aligner.get_ortho_loss()
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            # 几何监督：aligner 输出 vs 预计算的 base embedding 目标
            if args.lambda_geom > 0 and hw_embed_all is not None and target_bank.targets is not None:
                geom_sum = torch.tensor(0.0, device=device)
                geom_cnt = 0
                for i, hw_name in enumerate(batch["hw_name"]):
                    tgt_segs = target_bank.get(hw_name)
                    if not tgt_segs:
                        continue
                    for idx, seg in enumerate(["arch", "mem", "cons", "host"]):
                        info = tgt_segs.get(seg)
                        if not info or not info.get("active", False):
                            continue
                        e_pred = hw_embed_all[i, idx]
                        e_tgt = info["vec"]
                        mse = F.mse_loss(e_pred, e_tgt)
                        cos = 1 - F.cosine_similarity(e_pred.unsqueeze(0), e_tgt.unsqueeze(0), dim=-1).mean()
                        geom_sum = geom_sum + mse + args.lambda_geom_cos * cos
                        geom_cnt += 1
                if geom_cnt > 0:
                    geom_loss = geom_sum / geom_cnt
                else:
                    geom_loss = torch.tensor(0.0, device=device)
            else:
                geom_loss = torch.tensor(0.0, device=device)

            # 对比学习（默认关闭）
            if args.lambda_ctr > 0 and hw_vec.size(0) > 1:
                hidden_states = outputs_student.hidden_states[-1]
                B, L, _ = hidden_states.shape
                if multi_mask:
                    mask_pos = torch.tensor(span_positions, device=device)
                else:
                    mask_pos = (student_input_ids == hw_token_id).float().argmax(dim=1)
                z_prog = hidden_states[torch.arange(B, device=device), mask_pos]
                z_prog_norm = F.normalize(z_prog, dim=-1)
                z_hw_norm = F.normalize(hw_embed_for_cls, dim=-1)
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
                + args.lambda_geom * geom_loss
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

                geom_val = geom_loss.item()

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
                        f"orth={ortho_val:.4f} ctr={ctr_val:.4f} kd={kd_val:.4f} geom={geom_val:.4f} "
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
                        "loss_geom": geom_val,
                        "moving_avg": moving_avg,
                        "timestamp": datetime.now().isoformat(),
                    }
                    training_logs["logs"].append(log_entry)
                    with open(log_file, "a", encoding="utf-8") as f_log:
                        f_log.write(
                            f"Step {global_step:6d} | Loss: {loss_val:.6f} | "
                            f"LM: {lm_val:.6f} | CLS: {cls_val:.6f} | ORTH: {ortho_val:.6f} | "
                            f"CTR: {ctr_val:.6f} | KD: {kd_val:.6f} | GEOM: {geom_val:.6f} | Avg: {moving_avg:.6f} | "
                            f"ETA={eta_h:02d}:{eta_m:02d}:{eta_s:02d}\n"
                        )

                # 低频调试生成：检查模型当前输出的 schedule 是否可读
                if global_step % debug_gen_interval == 0:
                    try:
                        student.eval()
                        with torch.no_grad():
                            # 取当前 batch 第一个样本做一次贪心生成
                            raw_prompt = tokenizer.decode(student_input_ids[0], skip_special_tokens=False)
                            assert raw_prompt.strip() != "", "[DEBUG GEN] got empty prompt"
                            # 如果是多MASK，定位第一个 span，截断到 4-MASK 结束，模拟推理从 4-MASK 开始生成
                            if multi_mask and span_positions:
                                start = span_positions[0] + len(hw_token_ids)
                                prompt_embeds = embeds[:1, :start, :].detach()
                                prompt_attn = student_attn_mask[:1, :start]
                                prompt_len = start
                            else:
                                prompt_embeds = embeds[:1].detach()
                                prompt_attn = student_attn_mask[:1]
                                prompt_len = int(prompt_attn.sum().item())
                            gen_out = student.generate(
                                inputs_embeds=prompt_embeds,
                                attention_mask=prompt_attn,
                                max_new_tokens=80,
                                min_new_tokens=1,
                                do_sample=False,
                                num_beams=1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                            gen_new = gen_out[:, prompt_len:]
                            decoded = tokenizer.batch_decode(gen_new, skip_special_tokens=True)[0]
                            if decoded.strip() == "":
                                decoded = tokenizer.batch_decode(gen_new, skip_special_tokens=False)[0]
                            msg_debug = f"\n[DEBUG GEN] step={global_step} prompt='{raw_prompt[:120]}' text='{decoded[:200]}'"
                            print(msg_debug)
                            with open(log_file, "a", encoding="utf-8") as f_log:
                                f_log.write(msg_debug + "\n")
                    except Exception as e:
                        print(f"\n[DEBUG GEN] failed at step {global_step}: {e}")
                    finally:
                        student.train()

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    # 保存 aligner
                    cfg_to_save = lora_cfg.__dict__ if lora_cfg is not None else lora_cfg_dict
                    ckpt = {
                        "state_dict": aligner.state_dict(),
                        "cls_head_state_dict": cls_head.state_dict(),
                        "lora_config": cfg_to_save,
                        "prototype_names": prototype_names,
                        "prototype_keys": proto_matrix.tolist(),
                        "hw_token": args.hw_token,
                        "hw_token_id": hw_token_id,
                        "tokenizer_path": args.tokenizer_path,
                        "embed_dim": embed_layer.embedding_dim,
                        "config": vars(args),
                    }
                    torch.save(ckpt, os.path.join(ckpt_dir, "hw_aligner_lora.pt"))
                    # 保存 student (LoRA adapter)
                    student.save_pretrained(ckpt_dir)
                    saved_checkpoints.append(ckpt_dir)
                    if len(saved_checkpoints) > args.save_total_limit:
                        old = saved_checkpoints.pop(0)
                        if os.path.exists(old):
                            import shutil

                            shutil.rmtree(old)

    # 保存最终模型与对齐器
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    cfg_to_save = lora_cfg.__dict__ if lora_cfg is not None else lora_cfg_dict
    final_ckpt = {
        "state_dict": aligner.state_dict(),
        "cls_head_state_dict": cls_head.state_dict(),
        "lora_config": cfg_to_save,
        "prototype_names": prototype_names,
        "prototype_keys": proto_matrix.tolist(),
        "hw_token": args.hw_token,
        "hw_token_id": hw_token_id,
        "tokenizer_path": args.tokenizer_path,
        "embed_dim": embed_layer.embedding_dim if 'embed_layer' in locals() else student.get_input_embeddings().embedding_dim,
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
