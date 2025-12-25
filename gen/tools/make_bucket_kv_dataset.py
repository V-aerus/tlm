import argparse
import json
from typing import Tuple

# 复用现有 bucket 规则，确保与训练保持一致
from make_bucket_clm_dataset import BUCKET_TOKENS, classify_and_replace


def process_line(obj: dict) -> Tuple[dict, bool]:
    """
    返回 (out_obj, keep_flag)。
    当缺少 hw_id/hw_name 时 keep_flag=False。
    """
    hw_id = obj.get("hw_id")
    hw_name = obj.get("hw_name")
    if not hw_id or not hw_name:
        return {}, False

    canonical_text = obj.get("text", "")
    bucket_text = classify_and_replace(canonical_text)
    out_obj = {
        "text": bucket_text,
        "hw_id": hw_id,
        "hw_name": hw_name,
        "text_full": canonical_text,
    }
    return out_obj, True


def main():
    parser = argparse.ArgumentParser(description="Make bucket+hw_id dataset for HwKVAligner.")
    parser.add_argument("--in-path", required=True, help="输入 JSONL 路径（canonical 对齐数据）")
    parser.add_argument("--out-path", required=True, help="输出 JSONL 路径（bucket + hw_id）")
    args = parser.parse_args()

    total = 0
    kept = 0
    skipped = 0

    with open(args.in_path, "r", encoding="utf-8") as fin, open(
        args.out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Line {total} JSON decode failed: {e}")
                skipped += 1
                continue

            out_obj, keep_flag = process_line(obj)
            if not keep_flag:
                skipped += 1
                if total <= 5:
                    print(f"[WARN] Missing hw_id/hw_name at line {total}, skip.")
                continue

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total lines: {total}")
    print(f"Kept lines: {kept}")
    print(f"Skipped (missing hw_id/hw_name or decode fail): {skipped}")


if __name__ == "__main__":
    main()
