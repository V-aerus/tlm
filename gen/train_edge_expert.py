#!/usr/bin/env python3
"""Train a single EdgeTLM LoRA expert with gating."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from modeling import (
    BasePlusExperts,
    ExpertRegistry,
    FrozenBaseWrapper,
    GatedLoRAExpert,
    PeftDeltaWrapper,
)
from training import compute_gain_loss, compute_task_loss, entropy_reg, l2r_reg


@dataclass
class TrainConfig:
    base_model_path: str
    tokenizer_path: str
    dataset_jsonl: str
    output_dir: str
    init_expert_dir: str = None
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 5e-5
    router_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gain_margin: float = 0.05
    lambda_gain: float = 0.0
    lambda_router: float = 1e-4
    lambda_entropy: float = 1e-4
    max_length: int = 512
    gradient_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_name: str = "edge_expert"
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class EdgeExpertDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        self.samples: List[Dict] = []
        self.hardware_ids = set()
        def _to_float_or_nan(value) -> float:
            if value is None:
                return float("nan")
            try:
                val = float(value)
            except (TypeError, ValueError):
                return float("nan")
            if math.isnan(val):
                return float("nan")
            return val
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                lat_base = _to_float_or_nan(record.get("lat_base_star", record.get("latency")))
                lat_lora = _to_float_or_nan(record.get("lat_lora_star"))
                tokenized = tokenizer(
                    record["text"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.samples.append(
                    {
                        "input_ids": tokenized["input_ids"][0],
                        "attention_mask": tokenized["attention_mask"][0],
                        "labels": tokenized["input_ids"][0].clone(),
                        "hw_emb": torch.tensor(record["hw_emb"], dtype=torch.float32),
                        "lat_base": lat_base,
                        "lat_lora": lat_lora,
                        "hardware_id": record["hardware_id"],
                        "hardware_name": record.get("hardware_name", record["hardware_id"]),
                    }
                )
                self.hardware_ids.add(record["hardware_id"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train EdgeTLM single expert.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--router-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gain-margin", type=float, default=0.05)
    parser.add_argument("--lambda-gain", type=float, default=0.0)
    parser.add_argument("--lambda-router", type=float, default=1e-4)
    parser.add_argument("--lambda-entropy", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--adapter-name", default="edge_expert")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--init-expert-dir", default=None, help="Optional expert dir to resume (adapter + router).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TrainConfig(
        base_model_path=args.base_model_path,
        dataset_jsonl=args.dataset_jsonl,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path or args.base_model_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        router_learning_rate=args.router_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gain_margin=args.gain_margin,
        lambda_gain=args.lambda_gain,
        lambda_router=args.lambda_router,
        lambda_entropy=args.lambda_entropy,
        max_length=args.max_length,
        gradient_clip=args.gradient_clip,
        device=device,
        adapter_name=args.adapter_name,
        target_modules=args.target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_expert_dir=args.init_expert_dir,
    )


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = EdgeExpertDataset(Path(cfg.dataset_jsonl), tokenizer, cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    hardware_ids = sorted(dataset.hardware_ids)

    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)
    base_model.to(device)
    frozen_base = FrozenBaseWrapper(base_model)

    init_router = None
    init_beta = None
    init_tau = None

    lora_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)
    if cfg.init_expert_dir:
        init_dir = os.path.abspath(cfg.init_expert_dir)
        if not os.path.isdir(init_dir):
            raise FileNotFoundError(f"init_expert_dir not found: {init_dir}")
        lora_model = PeftModel.from_pretrained(lora_model, init_dir, is_trainable=True)
        if hasattr(lora_model, "print_trainable_parameters"):
            lora_model.print_trainable_parameters()

        router_path = os.path.join(init_dir, "router.json")
        if os.path.exists(router_path):
            with open(router_path, "r", encoding="utf-8") as rf:
                router_state = json.load(rf)
            if "r" in router_state:
                init_router = torch.tensor(router_state["r"], dtype=torch.float32)
            init_beta = router_state.get("beta")
            init_tau = router_state.get("tau")
        else:
            print(f"[WARN] router.json not found in init_expert_dir: {init_dir}")
    else:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=[m.strip() for m in cfg.target_modules.split(",") if m.strip()],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(lora_model, lora_config)
        lora_model.print_trainable_parameters()
    lora_model.to(device)

    delta_module = PeftDeltaWrapper(lora_model)

    r_dim = len(dataset[0]["hw_emb"])
    if init_router is not None and init_router.numel() != r_dim:
        print(
            "[WARN] init router dim mismatch: "
            f"{init_router.numel()} (router) vs {r_dim} (hw_emb). Ignore router init."
        )
        init_router = None
    expert = GatedLoRAExpert(
        lora_module=delta_module,
        r_dim=r_dim,
        init_router=init_router,
        reset_lora=(cfg.init_expert_dir is None),
    )
    if init_beta is not None or init_tau is not None:
        with torch.no_grad():
            if init_beta is not None:
                expert.beta.fill_(float(init_beta))
            if init_tau is not None:
                expert.tau.fill_(float(init_tau))
    expert.to(device)
    registry = ExpertRegistry()
    registry.register(cfg.adapter_name, expert)
    system = BasePlusExperts(frozen_base, registry)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in lora_model.parameters() if p.requires_grad], "lr": cfg.learning_rate},
            {"params": [expert.r, expert.beta], "lr": cfg.router_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )

    global_step = 0
    total_steps = len(dataloader) * cfg.num_epochs
    loss_sum = 0.0
    step_count = 0
    last_metrics: Dict[str, float] = {}

    for epoch in range(cfg.num_epochs):
        lora_model.train()
        expert.train()
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            hw_emb = batch["hw_emb"].to(device)
            lat_base = batch["lat_base"].to(device=device, dtype=torch.float32)
            lat_lora = batch["lat_lora"].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                base_outputs = frozen_base(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_outputs.logits if hasattr(base_outputs, "logits") else base_outputs

            logits = system.forward_single(
                input_ids,
                expert_name=cfg.adapter_name,
                hw_emb=hw_emb,
                expert_kwargs={"attention_mask": attention_mask, "base_logits": base_logits},
                cached_base=base_logits,
            )

            task_loss = compute_task_loss(logits, labels, ignore_index=tokenizer.pad_token_id)

            gates = expert.gating_weight(hw_emb)
            g_mean = gates.mean()
            paired_mask = (~torch.isnan(lat_base)) & (~torch.isnan(lat_lora))
            paired_count = int(paired_mask.sum().item())
            if paired_count > 0:
                lat_base_p = lat_base[paired_mask]
                lat_lora_p = lat_lora[paired_mask]
                epsilon = 1e-6
                I_pct = (lat_base_p - lat_lora_p) / (lat_base_p + epsilon)
                g_mean_p = gates[paired_mask].mean()
                gain_loss = compute_gain_loss(
                    I_pct=I_pct,
                    g_mean=g_mean_p,
                    step=global_step,
                    warmup_steps=cfg.warmup_steps,
                    m_target=cfg.gain_margin,
                    lambda_gain=cfg.lambda_gain,
                )
            else:
                gain_loss = torch.zeros((), device=lat_base.device, dtype=lat_base.dtype)

            router_reg = l2r_reg(expert.r, cfg.lambda_router)
            entropy_loss = entropy_reg(gates, cfg.lambda_entropy)

            loss = task_loss + gain_loss + router_reg + entropy_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), cfg.gradient_clip)
            optimizer.step()

            loss_sum += loss.item()
            step_count += 1
            last_metrics = {
                "loss": float(loss.item()),
                "task_loss": float(task_loss.item()),
                "gain_loss": float(gain_loss.item()),
                "router_reg": float(router_reg.item()),
                "entropy_reg": float(entropy_loss.item()),
                "g_mean": float(g_mean.item()),
                "paired_in_batch": paired_count,
            }

            if global_step % 50 == 0:
                print(
                    f"epoch={epoch} step={global_step}/{total_steps} "
                    f"loss={loss.item():.4f} task={task_loss.item():.4f} "
                    f"gain={gain_loss.item():.4f} router={router_reg.item():.4f} "
                    f"entropy={entropy_loss.item():.4f} g_mean={g_mean.item():.4f} "
                    f"paired={paired_count}"
                )
            global_step += 1

    # Save artifacts
    lora_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    router_info = {
        "hardware_dim": len(dataset[0]["hw_emb"]),
        "r": expert.r.detach().cpu().tolist(),
        "beta": float(expert.beta.detach().cpu()),
        "tau": float(expert.tau.detach().cpu()),
        "lambda_router": cfg.lambda_router,
        "lambda_entropy": cfg.lambda_entropy,
        "lambda_gain": cfg.lambda_gain,
        "meta": {
            "hardware_ids": hardware_ids,
            "train_samples": len(dataset),
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "warmup_steps": cfg.warmup_steps,
            "gain_margin": cfg.gain_margin,
        },
    }
    with open(os.path.join(cfg.output_dir, "router.json"), "w", encoding="utf-8") as f:
        json.dump(router_info, f, indent=2)

    metrics_payload = {
        "avg_loss": float(loss_sum / step_count) if step_count else 0.0,
        "final_step": global_step,
        "device": cfg.device,
        "adapter_name": cfg.adapter_name,
        "learning_rate": cfg.learning_rate,
        "router_learning_rate": cfg.router_learning_rate,
        "metrics_last_step": last_metrics,
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)

    print(f"Training complete. Artifacts saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
