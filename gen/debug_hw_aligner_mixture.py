#!/usr/bin/env python3
"""
Quick geometry check for ProtoMix hardware aligner.

- 加载训练好的 hw_aligner.pt（ProtoMixAligner）
- 从 hardware_embeddings_v2.json 中读出原型硬件与若干待测硬件（如 3090）的 hw_emb
- 打印：
  - 每个硬件在原型上的 mixing weights（softmax 权重）
  - 对齐后 embedding 之间的余弦相似度

用法示例（在仓库根目录）：

  python gen/debug_hw_aligner_mixture.py \\
    --aligner-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_aligner_proto_cosine_lowlr/hw_aligner.pt \\
    --hardware-embedding-path gen/Embedding/hardware_embeddings_v2.json \\
    --extra-hw "nvidia/geforce-rtx-3090"
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from modeling.hw_injection import ProtoMixAligner


def load_hardware_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aligner-path",
        type=str,
        required=True,
        help="Path to hw_aligner.pt checkpoint",
    )
    parser.add_argument(
        "--hardware-embedding-path",
        type=str,
        default="gen/Embedding/hardware_embeddings_v2.json",
        help="Path to hardware_embeddings_v2.json",
    )
    parser.add_argument(
        "--extra-hw",
        type=str,
        default="nvidia/geforce-rtx-3090",
        help="Comma-separated hardware names to probe (unseen / extra devices)",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.aligner_path, map_location="cpu")
    proto_names: List[str] = ckpt["prototype_names"]
    proto_keys = torch.tensor(ckpt["prototype_keys"], dtype=torch.float32)
    embed_dim: int = ckpt["embed_dim"]
    config = ckpt.get("config", {})
    temperature = float(config.get("temperature", 1.0))
    trainable_temperature = bool(config.get("trainable_temperature", False))

    print("[Checkpoint]")
    print(f"  prototypes: {proto_names}")
    print(f"  embed_dim:  {embed_dim}")
    print(f"  temperature (init): {temperature}")
    print()

    aligner = ProtoMixAligner(
        prototype_keys=proto_keys,
        embed_dim=embed_dim,
        temperature=temperature,
        trainable_temperature=trainable_temperature,
    )
    aligner.load_state_dict(ckpt["state_dict"])
    aligner.eval()

    hw_emb_dict = load_hardware_embeddings(args.hardware_embedding_path)

    # 要检查的硬件集合：原型 + 额外指定
    extra_hw = [name.strip() for name in args.extra_hw.split(",") if name.strip()]
    hw_names = list(dict.fromkeys(proto_names + extra_hw))  # 去重并保持顺序

    # 过滤：确保在 embedding 字典中都有
    missing = [name for name in hw_names if name not in hw_emb_dict]
    if missing:
        print("[Warning] 以下硬件在 hardware_embeddings 中不存在，将被跳过：")
        for name in missing:
            print(f"  - {name}")
        hw_names = [name for name in hw_names if name in hw_emb_dict]

    if not hw_names:
        print("[Error] 没有可用的硬件名称，退出。")
        return

    print("[Hardware to probe]")
    for name in hw_names:
        mark = ""
        if name in proto_names:
            mark = "(prototype)"
        print(f"  - {name} {mark}")
    print()

    # 计算每个硬件的 mixing weights 与对齐后的 embedding
    hw_vecs = []
    for name in hw_names:
        vec = torch.tensor(hw_emb_dict[name], dtype=torch.float32)
        hw_vecs.append(vec)
    hw_mat = torch.stack(hw_vecs, dim=0)  # [H, D_key]

    with torch.no_grad():
        embeds, weights = aligner(hw_mat, return_weights=True)  # [H, D_embed], [H, P]

    # 打印 mixing weights
    print("[Mixing weights over prototypes]")
    header = "hardware".ljust(32) + " | " + "  ".join(
        proto.ljust(24) for proto in proto_names
    )
    print(header)
    print("-" * len(header))
    for i, name in enumerate(hw_names):
        w = weights[i]
        w_str = "  ".join(f"{float(val):5.3f}" for val in w)
        print(f"{name.ljust(32)} | {w_str}")
    print()

    # 打印输出 embedding 之间的余弦相似度（按硬件）
    embeds_norm = F.normalize(embeds, dim=-1)
    sim = embeds_norm @ embeds_norm.t()  # [H, H]

    print("[Cosine similarity between aligned hardware embeddings]")
    header = " ".ljust(24) + " | " + "  ".join(
        name[:20].ljust(22) for name in hw_names
    )
    print(header)
    print("-" * len(header))
    for i, name in enumerate(hw_names):
        row = "  ".join(f"{float(val):5.3f}" for val in sim[i])
        print(f"{name[:20].ljust(24)} | {row}")


if __name__ == "__main__":
    main()

