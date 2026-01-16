import argparse
import ast
import csv
import glob
import os
import pickle
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import tvm
from tvm import auto_scheduler

# Ensure gen/ is on sys.path so local imports (e.g., common.py) work.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEN_DIR = os.path.dirname(_SCRIPTS_DIR)
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

import common
'''
  python /home/hehangshuai/workspace/tlm/gen/scripts/plot_convergence_by_count.py \
    --target 4090 \
    --records $RUN_ROOT/4090/iter03/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter04/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter05/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter06/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter07/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter08/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter09/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter10/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter11/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter12/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter13/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter14/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter15/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter16/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter17/measure/kv_lora.json \
    --records $RUN_ROOT/4090/iter18/measure/kv_lora.json \
    --normalize first \
    --metric total_weighted \
    --compare-mode union \
    --step 20000 \
    --output-png $RUN_ROOT/4090/iter18/gen/total_converge_count.png \
    --output-csv $RUN_ROOT/4090/iter18/gen/total_converge_count.csv
'''

def resolve_target_string(target_str: str) -> str:
    if not isinstance(target_str, str):
        return target_str
    ts = target_str.strip().lower()
    if ts in ("4090", "rtx-4090", "nvidia/rtx-4090", "3090", "rtx-3090", "geforce-rtx-3090"):
        return (
            "cuda -keys=cuda,gpu "
            "-arch=sm_86 "
            "-max_num_threads=1024 "
            "-max_shared_memory_per_block=49152 "
            "-max_threads_per_block=1024 "
            "-registers_per_block=65536 "
            "-thread_warp_size=32"
        )
    if ts in ("v100", "nvidia-v100"):
        return "nvidia/nvidia-v100"
    if ts in ("xavier", "jetson-agx-xavier"):
        return "nvidia/jetson-agx-xavier"
    return target_str


def parse_task_filename(path: str) -> Optional[Tuple[str, List[int], str]]:
    base = os.path.basename(path)
    if not base.endswith(".task.pkl") or not base.startswith("(("):
        return None
    mid = base[:-len(".task.pkl")]
    if not mid.endswith(")"):
        return None
    cut = mid.rfind("),")
    if cut == -1:
        return None
    key_part = mid[2:cut]
    target_kind = mid[cut + 2:-1].strip()
    if "," not in key_part:
        return None
    name, shape_str = key_part.split(",", 1)
    try:
        shape = ast.literal_eval(shape_str)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(shape, (list, tuple)):
        return None
    return name, list(shape), target_kind


def format_network_id(name: str, shape: List[int]) -> str:
    shape_tag = "x".join(str(x) for x in shape)
    return f"{name}_{shape_tag}"


def parse_shape_arg(shape_str: Optional[str]) -> Optional[List[int]]:
    if not shape_str:
        return None
    parts = [p for p in re.split(r"[x,]", shape_str) if p]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def build_workload_map(
    network_info_dir: str,
    target_kind: str,
    only_network: Optional[str],
    only_shape: Optional[List[int]],
    collision_policy: str,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, float]]:
    wk_to_meta: Dict[str, Dict[str, str]] = {}
    wk_to_weight: Dict[str, float] = {}
    collision_keys = set()
    collisions = 0
    task_files = sorted(glob.glob(os.path.join(network_info_dir, "*.task.pkl")))
    for path in task_files:
        parsed = parse_task_filename(path)
        if not parsed:
            continue
        name, shape, tkind = parsed
        if tkind != target_kind:
            continue
        if only_network and name != only_network:
            continue
        if only_shape and shape != only_shape:
            continue
        try:
            tasks, task_weights = pickle.load(open(path, "rb"))
        except Exception:
            continue
        net_id = format_network_id(name, shape)
        net_shape = ",".join(str(x) for x in shape)
        for task, weight in zip(tasks, task_weights):
            wk = task.workload_key
            meta = {"network_name": name, "network_shape": net_shape, "network_id": net_id}
            if wk in collision_keys:
                collisions += 1
                continue
            if wk in wk_to_meta and wk_to_meta[wk] != meta:
                collisions += 1
                if collision_policy == "merge":
                    wk_to_meta[wk]["network_id"] = f"{wk_to_meta[wk]['network_id']}|{net_id}"
                elif collision_policy == "skip":
                    wk_to_meta.pop(wk, None)
                    wk_to_weight.pop(wk, None)
                    collision_keys.add(wk)
                continue
            wk_to_meta[wk] = meta
            wk_to_weight[wk] = float(weight)
    if collisions:
        print(f"[WARN] workload_key collisions across networks: {collisions} (policy={collision_policy})")
    return wk_to_meta, wk_to_weight


def iter_record_paths(items: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for item in items:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.json"))))
        else:
            paths.append(item)
    return [p for p in paths if os.path.exists(p)]


def measure_iterator(paths: List[str]) -> Iterable[Tuple[auto_scheduler.MeasureInput, auto_scheduler.MeasureResult]]:
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        for inp, res in auto_scheduler.RecordReader(path):
            yield inp, res


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot convergence by measurement count.")
    parser.add_argument("--target", required=True, help="Target string for register_data_path.")
    parser.add_argument(
        "--records",
        action="append",
        help="Repeatable measured json path or directory (order matters).",
    )
    parser.add_argument("--network", default=None, help="Network name (e.g., bert_base).")
    parser.add_argument("--shape", default=None, help="Network shape (e.g., 1,128).")
    parser.add_argument("--network-info-dir", default=None, help="Override network_info dir.")
    parser.add_argument(
        "--metric",
        choices=("total_best", "total_weighted", "avg_best", "avg_weighted"),
        default="total_best",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20000,
        help="Measurement count step size.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("all", "ok"),
        default="all",
        help="Count all records or only successful records.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("intersection", "union"),
        default="intersection",
        help="Use workload intersection or union.",
    )
    parser.add_argument(
        "--collision-policy",
        choices=("keep_first", "merge", "skip"),
        default="skip",
        help="How to handle workload_key collisions across networks.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum workload coverage (0~1) to emit a point.",
    )
    parser.add_argument(
        "--normalize",
        choices=("first", "none"),
        default="first",
        help="Normalize values by the first emitted point or keep raw.",
    )
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--output-png", default=None, help="Output PNG path (optional).")
    args = parser.parse_args()

    if not args.records:
        raise SystemExit("Provide --records (repeatable).")
    if args.shape and not args.network:
        raise SystemExit("--shape requires --network.")
    if args.step <= 0:
        raise SystemExit("--step must be positive.")

    target_str = resolve_target_string(args.target)
    common.register_data_path(args.target)
    tvm.target.Target(target_str)

    network_info_dir = args.network_info_dir or common.NETWORK_INFO_FOLDER
    if not network_info_dir or not os.path.isdir(network_info_dir):
        raise FileNotFoundError(f"network_info dir not found: {network_info_dir}")

    only_shape = parse_shape_arg(args.shape)
    if args.shape and not only_shape:
        raise ValueError(f"Invalid --shape: {args.shape}")

    wk_to_meta, wk_to_weight = build_workload_map(
        network_info_dir,
        tvm.target.Target(target_str).kind.name,
        only_network=args.network,
        only_shape=only_shape,
        collision_policy=args.collision_policy,
    )
    if not wk_to_meta:
        raise RuntimeError("No workload mappings found. Check --network/--shape.")

    records = iter_record_paths(args.records)
    if not records:
        raise FileNotFoundError("No measured json files found.")

    wk_all = set(wk_to_meta.keys())
    wk_best: Dict[str, float] = {}

    points = []
    count_total = 0
    next_step = args.step

    def compute_totals() -> Tuple[float, float, int]:
        if args.compare_mode == "intersection":
            wk_use = sorted(wk for wk in wk_all if wk in wk_best)
        else:
            wk_use = sorted(wk_all)
        total_best = 0.0
        total_weighted = 0.0
        count = 0
        for wk in wk_use:
            best = wk_best.get(wk)
            if best is None:
                continue
            total_best += best
            total_weighted += best * wk_to_weight.get(wk, 1.0)
            count += 1
        return total_best, total_weighted, count

    for inp, res in measure_iterator(records):
        if inp.task.workload_key not in wk_all:
            continue
        is_ok = getattr(res, "error_no", 0) == 0
        if args.count_mode == "all" or is_ok:
            count_total += 1
        if is_ok:
            costs = []
            for item in res.costs:
                try:
                    val = float(item.value) if hasattr(item, "value") else float(item)
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    costs.append(val)
            if costs:
                lat_ms = float(sum(costs) / len(costs)) * 1e3
                wk = inp.task.workload_key
                prev = wk_best.get(wk)
                if prev is None or lat_ms < prev:
                    wk_best[wk] = lat_ms
        if count_total >= next_step:
            total_best, total_weighted, count = compute_totals()
            coverage = count / max(len(wk_all), 1)
            if coverage >= args.min_coverage:
                points.append(
                    {
                        "count": count_total,
                        "total_best": total_best,
                        "total_weighted": total_weighted,
                        "workload_count": count,
                        "coverage": coverage,
                    }
                )
            next_step += args.step

    if not points:
        raise RuntimeError("No points generated; check records and filters.")

    def metric_value(point: Dict[str, float]) -> Optional[float]:
        if args.metric == "total_best":
            return point["total_best"]
        if args.metric == "total_weighted":
            return point["total_weighted"]
        if args.metric == "avg_best":
            return point["total_best"] / point["workload_count"] if point["workload_count"] else None
        if args.metric == "avg_weighted":
            return point["total_weighted"] / point["workload_count"] if point["workload_count"] else None
        return None

    baseline_val = None
    if args.normalize == "first":
        baseline_val = metric_value(points[0])
        if baseline_val is None:
            raise ValueError("Baseline value is empty; increase step or check coverage.")
    else:
        baseline_val = 1.0
    rows = []
    prev_val = None
    for point in points:
        val = metric_value(point)
        if val is None:
            continue
        normalized = val / baseline_val if args.normalize == "first" else val
        profit = None
        if prev_val is not None and prev_val != 0:
            profit = (prev_val - val) / prev_val
        prev_val = val
        rows.append(
            {
                "count": point["count"],
                "metric": args.metric,
                "value": val,
                "normalized": normalized,
                "workload_count": point["workload_count"],
                "coverage": point["coverage"],
                "profit": profit if profit is not None else "",
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["count", "metric", "value", "normalized", "workload_count", "coverage", "profit"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote CSV: {args.output_csv}")

    if not args.output_png:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib not available, skip PNG: {exc}")
        return

    xs = [row["count"] for row in rows]
    ys = [row["normalized"] for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("measured entries")
    ylabel = args.metric if args.normalize == "none" else f"normalized {args.metric}"
    plt.ylabel(ylabel)
    if args.title:
        plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, dpi=200)
    print(f"Wrote PNG: {args.output_png}")


if __name__ == "__main__":
    main()
