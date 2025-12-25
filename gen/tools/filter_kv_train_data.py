import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Filter KV train data to keep only desired hardware IDs.")
    parser.add_argument("--in_path", required=True, help="输入 JSONL，包含 text/text_full/hw_id/hw_name 等字段")
    parser.add_argument("--out_path", required=True, help="输出 JSONL，过滤后的结果")
    parser.add_argument(
        "--keep_hw_ids",
        default="v100,4090",
        help="逗号分隔的保留 hw_id 列表，默认只保留 v100,4090",
    )
    args = parser.parse_args()

    keep_set = {h.strip().lower() for h in args.keep_hw_ids.split(",") if h.strip()}

    total = 0
    kept = 0
    counter = Counter()
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON decode failed at line {total}: {e}")
                continue
            hw_id = str(obj.get("hw_id", "")).lower()
            hw_name = str(obj.get("hw_name", "")).lower()
            if hw_id in keep_set:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                counter[hw_id] += 1
            else:
                # 可选：打印过滤原因
                continue

    print(f"Total lines: {total}")
    print(f"Kept lines: {kept}")
    print("Kept hw_id counts:")
    for k, v in counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
