import argparse
import os
from typing import Dict, Iterable, List

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

# 每个 bucket 对应的一组“原型短语”
BUCKET_PHRASES = {
    "[HW_GPU_HPC]": [
        "-arch=sm_70",
        "-arch=sm_86",
    ],
    "[HW_GPU_EDGE]": [
        "-arch=sm_72",
    ],
    "[HW_CPU_X86]": [
        "-mcpu=skylake-avx512",
    ],
    "[MODEL_CPU_SERVER]": [
        "-model=xeon",
    ],
    "[HW_CPU_ARM]": [
        "-mcpu=carmel",
    ],
    "[MODEL_GPU_CONSUMER]": [
        "-model=4090",
    ],
}


def compute_proto_for_phrases(
    phrases: Iterable[str],
    tokenizer: AutoTokenizer,
    emb_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute prototype embedding as the mean of token embeddings for anchor phrases."""
    collected: List[torch.Tensor] = []
    for phrase in phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Phrase {phrase!r} encodes to empty id list")
        for tid in token_ids:
            if tid < 0 or tid >= emb_weight.size(0):
                raise ValueError(f"Token id {tid} out of range for phrase {phrase!r}")
            collected.append(emb_weight[tid])

    if not collected:
        raise ValueError(f"No token embeddings collected for phrases: {phrases}")

    proto = torch.stack(collected, dim=0).mean(dim=0)
    return proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_tokenizer_path", required=True)
    parser.add_argument("--out_model_path", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_tokenizer_path, exist_ok=True)
    os.makedirs(args.out_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    orig_vocab_size = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": BUCKET_TOKENS})
    print(f"Added {added} bucket tokens.")

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    emb_weight = model.get_input_embeddings().weight.data
    lm_head_weight = model.lm_head.weight.data

    assert emb_weight.size(0) == len(tokenizer)
    assert lm_head_weight.size(0) == len(tokenizer)

    base_mean_norm = emb_weight[:orig_vocab_size].norm(dim=-1).mean().item()

    for bucket_token in BUCKET_TOKENS:
        if bucket_token not in BUCKET_PHRASES:
            raise ValueError(f"No phrases defined for bucket {bucket_token}")

        bucket_id = tokenizer.convert_tokens_to_ids(bucket_token)
        if bucket_id is None or bucket_id < 0:
            raise ValueError(f"Bucket token {bucket_token} not found in tokenizer")

        phrases = BUCKET_PHRASES[bucket_token]
        proto = compute_proto_for_phrases(phrases, tokenizer, emb_weight)

        proto_norm = proto.norm().item()
        if proto_norm > 0 and base_mean_norm > 0:
            proto = proto * (base_mean_norm / proto_norm)

        emb_weight[bucket_id] = proto
        lm_head_weight[bucket_id] = proto.clone()
        print(f"Init {bucket_token} from phrases {phrases}")

    model.tie_weights()

    tokenizer.save_pretrained(args.out_tokenizer_path)
    model.save_pretrained(args.out_model_path)
    print(f"Saved tokenizer -> {args.out_tokenizer_path}")
    print(f"Saved model -> {args.out_model_path}")


if __name__ == "__main__":
    main()
