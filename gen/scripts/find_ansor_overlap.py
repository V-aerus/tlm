#!/usr/bin/env python3
"""Find overlap between Ansor workload keys and measured records."""

import argparse
import csv
import json
from pathlib import Path

from tvm import auto_scheduler


def norm_key(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        return raw.strip()


def load_ansor_keys(path: Path, target: str) -> set:
    keys = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") != "task":
                continue
            if target and row.get("target") != target:
                continue
            wk = row.get("workload_key")
            if wk:
                keys.add(norm_key(wk))
    return keys


def load_measured_keys(path: Path) -> set:
    keys = set()
    for inp, _ in auto_scheduler.RecordReader(str(path)):
        keys.add(norm_key(inp.task.workload_key))
    return keys


def load_record_dir_keys(dir_path: Path) -> set:
    keys = set()
    for file in sorted(dir_path.glob("*.json")):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                try:
                    wk = obj["i"][0][0]
                except Exception:
                    continue
                keys.add(norm_key(wk))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Report overlap between Ansor summary and measured json.")
    parser.add_argument("--ansor-summary", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--measured-path", default=None, help="AutoScheduler measured JSON (single file).")
    parser.add_argument("--record-dir", default=None, help="Directory of postprocess measure_records/<hw>/*.json")
    parser.add_argument("--dump-overlap", default=None, help="Optional output JSON with overlapping keys.")
    args = parser.parse_args()

    ansor_keys = load_ansor_keys(Path(args.ansor_summary), args.target)
    if args.measured_path:
        measured_keys = load_measured_keys(Path(args.measured_path))
    elif args.record_dir:
        measured_keys = load_record_dir_keys(Path(args.record_dir))
    else:
        raise SystemExit("Provide --measured-path or --record-dir.")
    overlap = ansor_keys & measured_keys

    print(f"ansor_keys={len(ansor_keys)} measured_keys={len(measured_keys)} overlap={len(overlap)}")
    if args.dump_overlap:
        Path(args.dump_overlap).write_text(
            json.dumps(sorted(overlap), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"overlap keys written to: {args.dump_overlap}")


if __name__ == "__main__":
    main()
