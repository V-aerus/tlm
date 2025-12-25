import argparse
import json
import re
from typing import Dict, List, Optional, Tuple


HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
INT_RE = re.compile(r"^-?\d+$")


def parse_workload_key(text: str) -> Optional[Tuple[str, List[int]]]:
    tokens = text.split()
    hash_idx = None
    for i, tok in enumerate(tokens):
        if HEX32_RE.match(tok):
            hash_idx = i
            break
    if hash_idx is None:
        return None

    shapes: List[int] = []
    for tok in tokens[hash_idx + 1 :]:
        if tok in ("cuda", "llvm"):
            break
        if not INT_RE.match(tok):
            break
        shapes.append(int(tok))

    if not shapes:
        return None
    return tokens[hash_idx], shapes


def build_key(hash_key: str, shapes: List[int]) -> str:
    return f"{hash_key}|{','.join(str(x) for x in shapes)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find matching workloads with different hardware (e.g., v100 vs 4090) in a JSONL dataset."
    )
    parser.add_argument("--in_path", required=True, help="Input JSONL path")
    parser.add_argument("--cnt", type=int, default=5, help="Number of pairs to output")
    parser.add_argument(
        "--hw_ids",
        default="4090,v100",
        help="Comma-separated hw_id whitelist (default: 4090,v100)",
    )
    parser.add_argument("--out_path", default=None, help="Optional output JSONL path for pairs")
    args = parser.parse_args()

    hw_ids = {x.strip() for x in args.hw_ids.split(",") if x.strip()}
    pending: Dict[str, Dict[str, Dict[str, object]]] = {}
    pair_count = 0
    total = 0
    out_f = open(args.out_path, "w", encoding="utf-8") if args.out_path else None

    try:
        with open(args.in_path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                hw_id = sample.get("hw_id")
                if hw_id not in hw_ids:
                    continue

                text = sample.get("text") or sample.get("text_full") or ""
                parsed = parse_workload_key(text)
                if parsed is None:
                    continue
                hash_key, shapes = parsed
                key = build_key(hash_key, shapes)

                entry = {
                    "hw_id": hw_id,
                    "hash": hash_key,
                    "shapes": shapes,
                    "text": sample.get("text"),
                    "text_full": sample.get("text_full"),
                    "labels": sample.get("labels"),
                    "latency": sample.get("latency"),
                }

                slot = pending.setdefault(key, {})
                if hw_id in slot:
                    continue
                slot[hw_id] = entry

                if hw_ids.issubset(slot.keys()):
                    pair_count += 1
                    pair = {"key": key, "entries": slot}
                    if out_f:
                        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    else:
                        print("=" * 80)
                        print(f"pair #{pair_count} key={key}")
                        for hid in sorted(slot.keys()):
                            print(f"[{hid}]")
                            print(slot[hid].get("text") or "")
                    pending.pop(key, None)

                    if pair_count >= args.cnt:
                        break
    finally:
        if out_f:
            out_f.close()

    print(f"Total lines: {total}")
    print(f"Pairs found: {pair_count}")


if __name__ == "__main__":
    main()
