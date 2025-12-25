import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer

from train_hw_kv_aligner import build_labels_with_schedule_mask


def main():
    parser = argparse.ArgumentParser(description="Sanity check schedule masks and KD alignment.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--train_json_paths", required=True, help="Comma-separated JSONL paths (bucket_kv gpu_only).")
    parser.add_argument("--num_samples", type=int, default=2000, help="Max samples to check")
    parser.add_argument("--log_dir", default="/home/hehangshuai/workspace/tlm/gen/logs", help="Where to dump sampled cases")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    paths = [p.strip() for p in args.train_json_paths.split(",") if p.strip()]

    stats = Counter()
    samples_full = []
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "schedule_full_samples.log")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.num_samples:
                    break
                obj = json.loads(line)
                text = obj.get("text")
                text_full = obj.get("text_full", text)
                if not text or not text_full:
                    continue

                enc_s = tok(text, return_tensors="pt", padding=False, add_special_tokens=True)
                enc_t = tok(text_full, return_tensors="pt", padding=False, add_special_tokens=True)
                labels_s = build_labels_with_schedule_mask(enc_s["input_ids"], tok)
                labels_t = build_labels_with_schedule_mask(enc_t["input_ids"], tok)

                len_s = (labels_s != -100).sum().item()
                len_t = (labels_t != -100).sum().item()
                if len_s == 0:
                    stats["zero_sched_s"] += 1
                if len_t == 0:
                    stats["zero_sched_t"] += 1
                if len_s > 0.9 * enc_s["input_ids"].numel():
                    stats["full_sched_s"] += 1
                    if len(samples_full) < 200:
                        samples_full.append(text.replace("\n", " "))
                if len_t > 0.9 * enc_t["input_ids"].numel():
                    stats["full_sched_t"] += 1
                if len_s != len_t:
                    stats["len_mismatch"] += 1
                stats["checked"] += 1
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1} samples...")

    print("Checked:", stats.get("checked", 0))
    print("zero_sched_s:", stats.get("zero_sched_s", 0))
    print("zero_sched_t:", stats.get("zero_sched_t", 0))
    print("full_sched_s:", stats.get("full_sched_s", 0))
    print("full_sched_t:", stats.get("full_sched_t", 0))
    print("len_mismatch:", stats.get("len_mismatch", 0))

    if samples_full:
        with open(log_path, "w", encoding="utf-8") as fout:
            for s in samples_full:
                fout.write(s + "\n")
        print(f"Sampled {len(samples_full)} full-schedule cases written to {log_path}")


if __name__ == "__main__":
    main()
