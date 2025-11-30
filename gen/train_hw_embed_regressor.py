#!/usr/bin/env python3
"""
将硬件连续向量 (hw_emb) 回归到 Base 模型的目标字符串 embedding（平均池化），用于检验“直接几何对齐是否保持语法”。

用法示例（seen 硬件）：
python gen/train_hw_embed_regressor.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
  --hardware_embedding_path gen/Embedding/hardware_embeddings_v3.json \
  --output_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_embed_regressor.pt

可选：提供 target 文本映射（JSON），否则使用默认映射（基于 EDGE_EMBEDDING_DEFAULTS）。

训练输出：简单的两层 MLP，将 hw_emb -> target_embed_mean。检查训练/验证的 MSE 与余弦。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 默认 target 映射（可通过 --target_map 覆盖）
DEFAULT_TARGETS = {
    "nvidia/nvidia-v100": "cuda -keys=cuda,gpu -arch=sm_70 -max_num_threads=1024 -model=v100 -thread_warp_size=32",
    "nvidia/nvidia-a40": "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -model=4090 -thread_warp_size=32",
    "nvidia/jetson-agx-xavier": "cuda -keys=cuda,gpu -arch=sm_72 -max_num_threads=1024 -model=xavier -thread_warp_size=32",
    "aws/cpu/c5.18xlarge": "llvm -mtriple=x86_64-linux-gnu",
}


class HwRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_hw_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["hardware_name"]: e["vector"] for e in entries}


def build_target_embeddings(
    tokenizer,
    embed_layer: nn.Embedding,
    targets: Dict[str, str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out = {}
    for name, text in targets.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        ids_t = torch.tensor(ids, device=device)
        with torch.no_grad():
            emb = embed_layer(ids_t).mean(dim=0)
        out[name] = emb
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--hardware_embedding_path", required=True)
    parser.add_argument("--target_map", default=None, help="JSON file mapping hardware_name -> target string")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path)
    base.eval().to(device)
    embed_layer = base.get_input_embeddings()

    hw_embs = load_hw_embeddings(args.hardware_embedding_path)
    targets = json.load(open(args.target_map)) if args.target_map else DEFAULT_TARGETS

    target_embs = build_target_embeddings(tok, embed_layer, targets, device)
    # 过滤存在两侧的硬件
    pairs = [(hw_name, hw_embs[hw_name], target_embs[hw_name]) for hw_name in targets if hw_name in hw_embs and hw_name in target_embs]
    if not pairs:
        raise ValueError("No overlapping hardware between embeddings and target map.")

    in_dim = len(pairs[0][1])
    out_dim = embed_layer.embedding_dim
    model = HwRegressor(in_dim, out_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for _, hw_vec, tgt_emb in pairs:
            hw_t = torch.tensor(hw_vec, device=device).unsqueeze(0)
            tgt_t = tgt_emb.unsqueeze(0)
            pred = model(hw_t)
            loss_mse = F.mse_loss(pred, tgt_t)
            loss_cos = 1 - F.cosine_similarity(pred, tgt_t, dim=-1).mean()
            loss = loss_mse + loss_cos
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                cos_vals = []
                for _, hw_vec, tgt_emb in pairs:
                    hw_t = torch.tensor(hw_vec, device=device).unsqueeze(0)
                    pred = model(hw_t)
                    cos = F.cosine_similarity(pred, tgt_emb.unsqueeze(0), dim=-1).item()
                    cos_vals.append(cos)
                print(f"Epoch {epoch+1}: loss={total_loss/len(pairs):.4f}, cos@mean={sum(cos_vals)/len(cos_vals):.4f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden": args.hidden,
        "targets": targets,
        "hardware_embedding_path": args.hardware_embedding_path,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output_path)
    print(f"Saved regressor to {args.output_path}")


if __name__ == "__main__":
    main()
