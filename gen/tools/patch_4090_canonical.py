import argparse
import json
from typing import Tuple


CANONICAL_TARGET = (
    "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 "
    "-max_shared_memory_per_block=49152 "
    "-max_threads_per_block=1024 "
    "-registers_per_block=65536 "
    "-thread_warp_size=32"
)

KEY_START = "cuda -keys=cuda,gpu -arch=sm_86"
KEY_END = "-thread_warp_size=32"


def patch_text(text: str) -> Tuple[str, bool]:
    """Replace the CUDA target span with the canonical form. Return (text, patched?)."""
    if KEY_START not in text:
        return text, False
    start = text.find(KEY_START)
    end = text.find(KEY_END, start)
    if end == -1:
        return text, False
    end += len(KEY_END)
    return text[:start] + CANONICAL_TARGET + text[end:], True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--only_4090",
        action="store_true",
        default=True,
        help="Only patch samples with hw_id == '4090' (default true).",
    )
    parser.add_argument(
        "--all_hw",
        dest="only_4090",
        action="store_false",
        help="Patch all samples regardless of hw_id.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report stats, do not write output file.",
    )
    parser.add_argument(
        "--show_missed",
        type=int,
        default=0,
        help="If >0, print examples of 4090 samples that failed to match.",
    )
    args = parser.parse_args()

    total = patched = missed = 0
    missed_examples = []
    fout = None
    if not args.dry_run:
        fout = open(args.out_path, "w", encoding="utf-8")
    with open(args.in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            sample = json.loads(line)
            hw_id = sample.get("hw_id")
            if args.only_4090 and hw_id != "4090":
                if fout:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            new_text, ok = patch_text(sample.get("text", ""))
            if hw_id == "4090":
                if ok:
                    patched += 1
                else:
                    missed += 1
                    if len(missed_examples) < args.show_missed:
                        missed_examples.append(sample.get("text", "")[:200])
            sample["text"] = new_text
            if fout:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if fout:
        fout.close()

    print(f"Total lines: {total}")
    print(f"Patched 4090 samples: {patched}")
    print(f"Unmatched 4090 samples: {missed}")
    if missed_examples:
        print("Missed examples (truncated):")
        for ex in missed_examples:
            print(ex)


if __name__ == "__main__":
    main()
