#!/usr/bin/env python3
"""Merge per-hardware edge_sft JSONL into a router calibration JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_input(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Invalid --input '{arg}', expect label=path")
    label, path = arg.split("=", 1)
    return label.strip(), Path(path.strip())


def _iter_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build router calibration dataset by merging JSONL files.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input as label=path (repeatable). label is used as router_label.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--label-field", default="router_label", help="Field name for router label.")
    parser.add_argument(
        "--override-hardware-id",
        action="store_true",
        help="If set, overwrite hardware_id with label.",
    )
    parser.add_argument(
        "--limit-per-input",
        type=int,
        default=0,
        help="Optional max records per input (0 means no limit).",
    )
    args = parser.parse_args()

    inputs = [_parse_input(inp) for inp in args.input]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for label, path in inputs:
            if not path.exists():
                raise FileNotFoundError(f"Input not found: {path}")
            count = 0
            for record in _iter_records(path):
                record[args.label_field] = label
                if args.override_hardware_id:
                    record["hardware_id"] = label
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                count += 1
                if args.limit_per_input and count >= args.limit_per_input:
                    break
            print(f"[INFO] {label}: wrote {count} records from {path}")

    print(f"[DONE] wrote {total} records -> {output_path}")


if __name__ == "__main__":
    main()
