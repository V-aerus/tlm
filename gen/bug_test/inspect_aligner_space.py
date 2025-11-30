"""
离线检查 aligner 输出在 embedding 空间的位置：
 - 读取 aligner ckpt（ProtoMixAligner）
 - 对指定硬件 hw_vec（来自 hardware_embeddings_v3.json）做前向，得到 4 个 HwToken 向量
 - 与 prototype（hw_token_targets_v1.pt）各段、以及 vocab token 的 embedding 计算余弦相似度

运行示例（4090 + V100）：
CUDA_VISIBLE_DEVICES=3 python gen/bug_test/inspect_aligner_space.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --aligner_ckpt /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_aligner_lora_stage1/hw_aligner_lora.pt \
  --edge_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v3.json \
  --prototype_path /home/hehangshuai/workspace/tlm/gen/Embedding/hw_token_targets_v1.pt \
  --hw_names nvidia/nvidia-a40,nvidia/nvidia-v100 \
  --topk 8
"""
import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from modeling import ProtoMixAligner

# 可选的 canonical target 文本，用于对比
CANONICAL_TARGETS = {
    "nvidia/nvidia-a40": "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -model=4090 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32",
    "nvidia/nvidia-v100": "cuda -keys=cuda,gpu -arch=sm_70 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32",
    "nvidia/jetson-agx-xavier": "cuda -keys=cuda,gpu -arch=sm_72 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32 llvm -keys=arm_cpu,cpu -mcpu=carmel -mtriple=aarch64-linux-gnu -num-cores=8",
    "aws/cpu/c5.18xlarge": "llvm -keys=cpu -mcpu=skylake-avx512 -model=xeon 36 64 64 0 0 0 0 0",
}


def load_edge_emb(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["hardware_name"]: e["vector"] for e in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--aligner_ckpt", required=True)
    parser.add_argument("--edge_embedding_path", required=True)
    parser.add_argument("--prototype_path", required=True)
    parser.add_argument("--hw_names", required=True, help="逗号分隔的硬件名，如 nvidia/nvidia-a40,nvidia/nvidia-v100")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
    embed_weight = model.get_input_embeddings().weight  # (V,D)

    # 加载 aligner
    ckpt = torch.load(args.aligner_ckpt, map_location=device)
    proto = torch.tensor(ckpt["prototype_keys"], dtype=torch.float32, device=device)
    embed_dim = ckpt.get("embed_dim", embed_weight.shape[1])
    cfg_meta = ckpt.get("config", {})
    temperature = cfg_meta.get("temperature", ckpt.get("temperature", 1.0))
    trainable_temp = cfg_meta.get("trainable_temperature", False)
    aligner = ProtoMixAligner(proto, embed_dim=embed_dim, temperature=temperature, trainable_temperature=trainable_temp)
    aligner.load_state_dict(ckpt["state_dict"])
    aligner.to(device).eval()

    # 加载 edge embedding
    edge_map = load_edge_emb(args.edge_embedding_path)

    # 加载 prototype 目标
    proto_targets = torch.load(args.prototype_path, map_location=device)

    hw_list = [h.strip() for h in args.hw_names.split(",") if h.strip()]
    for hw_name in hw_list:
        if hw_name not in edge_map:
            print(f"[WARN] hw_name {hw_name} not in edge embedding file, skip.")
            continue
        hw_vec = torch.tensor(edge_map[hw_name], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            hw_embed = aligner(hw_vec, split_heads=True)[0]  # (4,D)

        print("=" * 80)
        print(f"[HW] {hw_name}")
        # 与 prototype 各段的相似度
        if hw_name in proto_targets:
            segs = proto_targets[hw_name]
            for i, seg in enumerate(["arch", "mem", "cons", "host"]):
                info = segs.get(seg, {})
                if info.get("active", False):
                    tgt = info["vec"].to(device)
                    cos = F.cosine_similarity(hw_embed[i].unsqueeze(0), tgt.unsqueeze(0)).item()
                    print(f"  head{i} vs proto[{seg}] cos={cos:.4f}")
                else:
                    print(f"  head{i} vs proto[{seg}] inactive")
        # 与 canonical target 文本的均值相似度
        if hw_name in CANONICAL_TARGETS:
            text = CANONICAL_TARGETS[hw_name]
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ids_tensor = torch.tensor(ids, device=device)
            txt_embed = embed_weight[ids_tensor].mean(dim=0)
            for i in range(hw_embed.size(0)):
                cos = F.cosine_similarity(hw_embed[i].unsqueeze(0), txt_embed.unsqueeze(0)).item()
                print(f"  head{i} vs canonical_target_mean cos={cos:.4f}")

        # vocab top-k
        emb_norm = F.normalize(embed_weight, dim=-1)
        hw_norm = F.normalize(hw_embed, dim=-1)
        sims = hw_norm @ emb_norm.t()  # (4,V)
        for i in range(hw_embed.size(0)):
            topk_val, topk_idx = torch.topk(sims[i], k=min(args.topk, sims.size(1)))
            top_tokens = tok.convert_ids_to_tokens(topk_idx.tolist())
            pairs = [f"{t}({v:.3f})" for t, v in zip(top_tokens, topk_val.tolist())]
            print(f"  head{i} top{args.topk} tokens: " + ", ".join(pairs))


if __name__ == "__main__":
    main()
