"""
Debug helper to inspect schedule-boundary detection on bucket+KV training data.
It does NOT modify training code. It prints raw text plus the decoded prefix/schedule tokens
so you can visually verify whether the start index is reasonable.
"""
import argparse
import json
import random
from typing import List

import torch
from transformers import AutoTokenizer

# Schedule operators that typically mark the beginning of the schedule DSL segment.
# This list mirrors common ops in TLM schedule sentences (split, fuse, reorder, annotate, etc.).
SCHEDULE_OP_TOKENS: List[str] = [
    "CI",
    "CHW",
    "local",
    "SP",
    "FSP",
    "FFSP",
    "RE",
    "CA",
    "CHR",
    "AN",
    "PPT",
    "SPC",
    "FU",
    "TBS",
    "PRS",
    "PR",
]


def find_schedule_start_idx(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Given a token id sequence, return the start index of the schedule segment.
    If no schedule op is found, return -1.
    """
    ids = input_ids.tolist()
    schedule_ids = {
        tokenizer.convert_tokens_to_ids(tok)
        for tok in SCHEDULE_OP_TOKENS
        if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
    }

    for i, tid in enumerate(ids):
        tok = tokenizer.convert_ids_to_tokens(tid).strip()
        if tid in schedule_ids or tok in SCHEDULE_OP_TOKENS:
            return i
    return -1


def parse_args():
    parser = argparse.ArgumentParser(description="Debug schedule boundary on bucket KV dataset.")
    parser.add_argument("--tokenizer_path", required=True, help="Bucket tokenizer path")
    parser.add_argument(
        "--train_json_paths",
        required=True,
        help="Comma-separated paths to *_bucket_kv.json files",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Max samples to print")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    paths = [p.strip() for p in args.train_json_paths.split(",") if p.strip()]
    lines = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            lines.extend(f.readlines())

    if not lines:
        print("No data lines found.")
        return

    random.shuffle(lines)
    total = 0
    found = 0
    not_found = 0
    max_print = args.num_samples

    for idx, line in enumerate(lines):
        if total >= max_print:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"[WARN] JSON decode failed at shuffled_idx={idx}: {e}")
            continue

        text = obj.get("text")
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"][0]
        start = find_schedule_start_idx(input_ids, tokenizer)

        tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in input_ids]
        prefix = " ".join(tokens[:start]) if start >= 0 else ""
        sched = " ".join(tokens[start : start + 80]) if start >= 0 else ""

        print("=" * 80)
        print(f"sample_idx : {total}")
        print(f"hw_id      : {obj.get('hw_id')}")
        raw_snippet = text[:200].replace("\n", " ")
        print(f"raw text   : {raw_snippet} ...")
        if start >= 0:
            found += 1
            print(f"prefix tokens ({start}): {prefix}")
            print(f"schedule tokens       : {sched}")
        else:
            not_found += 1
            print(f"[WARN] no schedule op found for sample_idx={total}, hw_id={obj.get('hw_id')}")
        total += 1

    print("-" * 80)
    print(f"Total printed: {total}")
    print(f"Found schedule start: {found}")
    print(f"Not found: {not_found}")


if __name__ == "__main__":
    main()
