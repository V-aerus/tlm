import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_workload_repr(workload_repr: str) -> str:
    try:
        workload_obj = json.loads(workload_repr)
    except json.JSONDecodeError:
        return workload_repr
    return json.dumps(workload_obj, separators=(",", ":"), ensure_ascii=True)


def parse_line(line_str: str) -> Tuple[str, str, str]:
    line_json = json.loads(line_str)
    workload_repr = line_json["i"][0][0]
    target_str = line_json["i"][0][1]
    workload = json.loads(workload_repr)
    workload_id = workload[0]
    return workload_id, target_str, normalize_workload_repr(workload_repr)


def collect_collisions(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, str]]:
    collisions: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                line_str = rec.get("line")
                if not line_str:
                    continue
                workload_id, target_str, workload_repr = parse_line(line_str)
                try:
                    shapes = json.loads(workload_repr)[1:]
                    shape_repr = json.dumps(shapes, separators=(",", ":"), ensure_ascii=True)
                except Exception:
                    shape_repr = workload_repr
                bucket = collisions[(workload_id, target_str)]
                if workload_repr not in bucket:
                    bucket[workload_repr] = shape_repr
    return collisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check collisions when merging EdgeTLM base/lora JSONL by workload_id."
    )
    parser.add_argument("--base_jsonl", required=True, help="Base JSONL path.")
    parser.add_argument("--lora_jsonl", required=True, help="LoRA JSONL path.")
    parser.add_argument("--topk", type=int, default=20, help="How many collision examples to print.")
    args = parser.parse_args()

    paths = [Path(args.base_jsonl), Path(args.lora_jsonl)]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    collisions = collect_collisions(paths)
    conflict_items = [
        (wid, target, variants) for (wid, target), variants in collisions.items() if len(variants) > 1
    ]
    conflict_items.sort(key=lambda x: len(x[2]), reverse=True)

    print(f"Total workload_id groups: {len(collisions)}")
    print(f"Collision groups (same id, different shapes): {len(conflict_items)}")
    if conflict_items:
        print("Top collisions:")
        for wid, target, variants in conflict_items[: args.topk]:
            shapes_list = list(variants.values())
            print(f"  - workload_id={wid} target={target} variants={len(shapes_list)} shapes={shapes_list[:6]}")
        print("Suggested merge key: workload_repr (shape-aware)")


if __name__ == "__main__":
    main()
