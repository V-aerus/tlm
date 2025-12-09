import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def topk_neighbors(vec: torch.Tensor, emb: torch.Tensor, tokenizer, k: int = 10):
    vec = vec / vec.norm().clamp(min=1e-6)
    emb_norm = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sims = torch.matmul(emb_norm, vec)
    vals, idx = torch.topk(sims, k)
    tokens = tokenizer.convert_ids_to_tokens(idx.tolist())
    return list(zip(tokens, vals.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    emb = model.get_input_embeddings().weight.data

    buckets = [
        "[HW_GPU_HPC]",
        "[HW_GPU_EDGE]",
        "[HW_CPU_X86]",
        "[HW_CPU_ARM]",
        "[MODEL_CPU_SERVER]",
        "[MODEL_GPU_CONSUMER]",
    ]
    for b in buckets:
        if b not in tok.get_vocab():
            print(f"{b} not in tokenizer vocab, skip.")
            continue
        bid = tok.convert_tokens_to_ids(b)
        neighbors = topk_neighbors(emb[bid], emb, tok, k=args.k)
        print(f"\nBucket {b} top-{args.k} neighbors:")
        for t, s in neighbors:
            print(f"{t:30s} {s:.4f}")


if __name__ == "__main__":
    main()
