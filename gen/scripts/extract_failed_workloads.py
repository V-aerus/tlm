#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REASONS = {
    "no_valid_records": re.compile(r"workload\s+(?P<raw>\[.*\])\s+未能生成任何有效记录"),
    "retry_exhausted": re.compile(r"workload\s+(?P<raw>\[.*\])\s+重试5次后仍然失败"),
    "fatal_error": re.compile(r"处理workload\s+(?P<raw>\[.*\])\s+时发生严重错误"),
}


def parse_workload(raw: str):
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return None, raw
    return obj, raw


def summarize(log_path: Path):
    records = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for reason, pattern in REASONS.items():
                m = pattern.search(line)
                if not m:
                    continue
                raw = m.group("raw")
                obj, raw_fallback = parse_workload(raw)
                records.append((reason, obj, raw_fallback))
                break
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract failed workloads from gen_state_debug_kv log.")
    parser.add_argument("--log-path", required=True, help="Path to gen_state_debug_kv_*.log")
    parser.add_argument("--output-json", help="Optional output JSON path")
    parser.add_argument("--output-csv", help="Optional output CSV path")
    parser.add_argument("--min-count", type=int, default=1, help="Only keep workloads with count >= N")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    records = summarize(log_path)
    counter = Counter()
    meta = defaultdict(dict)

    for reason, obj, raw in records:
        if obj and isinstance(obj, list) and obj:
            wid = obj[0]
            shapes = obj[1:]
            key = (reason, wid, json.dumps(shapes, separators=(",", ":")))
            meta[key]["workload_repr"] = json.dumps(obj, separators=(",", ":"))
        else:
            key = (reason, "unknown", raw)
            meta[key]["workload_repr"] = raw
        counter[key] += 1

    rows = []
    for (reason, wid, shapes), count in counter.most_common():
        if count < args.min_count:
            continue
        rows.append(
            {
                "reason": reason,
                "workload_id": wid,
                "shapes": shapes,
                "count": count,
                "workload_repr": meta[(reason, wid, shapes)].get("workload_repr", ""),
            }
        )

    print(f"failed_records={len(records)} unique={len(rows)}")
    for row in rows[:20]:
        print(f"{row['count']:>3} {row['reason']} {row['workload_repr']}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.output_json}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write("reason,workload_id,shapes,count,workload_repr\n")
            for row in rows:
                f.write(
                    f"{row['reason']},{row['workload_id']},{row['shapes']},{row['count']},\"{row['workload_repr']}\"\n"
                )
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
