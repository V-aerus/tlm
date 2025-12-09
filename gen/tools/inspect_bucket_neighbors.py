import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BUCKET_TOKENS = [
    "[HW_GPU_HPC]",
    "[HW_GPU_EDGE]",
    "[HW_CPU_X86]",
    "[HW_CPU_ARM]",
    "[MODEL_CPU_SERVER]",
    "[MODEL_GPU_CONSUMER]",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    emb_weight = model.get_input_embeddings().weight.detach()
    vocab_size, d_model = emb_weight.shape
    num_buckets = len(BUCKET_TOKENS)
    orig_vocab_size = vocab_size - num_buckets

    emb_norm = emb_weight / (emb_weight.norm(dim=-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        # 打印 bucket 间的两两余弦相似度
        print("Pairwise cosine between buckets:")
        for i, bi in enumerate(BUCKET_TOKENS):
            id_i = tokenizer.convert_tokens_to_ids(bi)
            if id_i is None or id_i < 0:
                continue
            vi = emb_norm[id_i]
            for j, bj in enumerate(BUCKET_TOKENS):
                if j <= i:
                    continue
                id_j = tokenizer.convert_tokens_to_ids(bj)
                if id_j is None or id_j < 0:
                    continue
                vj = emb_norm[id_j]
                cos = torch.dot(vi, vj).item()
                print(f"  cos({bi}, {bj}) = {cos:.4f}")

        for bucket in BUCKET_TOKENS:
            bucket_id = tokenizer.convert_tokens_to_ids(bucket)
            if bucket_id is None or bucket_id < 0:
                print(f"[WARN] Bucket token {bucket} not in tokenizer, skip.")
                continue
            bucket_vec = emb_norm[bucket_id]
            orig_vecs = emb_norm[:orig_vocab_size]
            top_k = min(args.top_k, orig_vocab_size)
            sim = torch.matmul(orig_vecs, bucket_vec)
            top_values, top_indices = torch.topk(sim, k=top_k)

            print("=" * 80)
            print(f"Bucket: {bucket} (id={bucket_id})")
            print(f"Top-{top_k} neighbors in original vocab:\n")
            for rank, (idx, score) in enumerate(
                zip(top_indices.tolist(), top_values.tolist()), start=1
            ):
                tok = tokenizer.convert_ids_to_tokens(idx)
                print(f"{rank:2d}. id={idx:5d}  token={tok!r}  cos={score:.4f}")


if __name__ == "__main__":
    main()
