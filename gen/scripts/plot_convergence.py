import argparse
import ast
import csv
import glob
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import tvm
from tvm import auto_scheduler

import common


'''
  它会：
  - 按网络汇总 total best（或 weighted）
  - 以 iter00 为 baseline 做归一化
  - 输出 CSV + 折线图 PNG

  ———

  ## 用法示例（你要的 network 级、baseline=iter00）

  python /home/hehangshuai/workspace/tlm/gen/scripts/plot_convergence.py \
    --target 4090 \
    --network bert_base \
    --shape 1,128 \
    --series iter00=$RUN_ROOT/4090/iter00/measure/kv_lora.json \
    --series iter01=$RUN_ROOT/4090/iter01/measure/kv_lora.json \
    --series iter02=$RUN_ROOT/4090/iter02/measure/kv_lora.json \
    --series iter03=$RUN_ROOT/4090/iter03/measure/kv_lora.json \
    --series iter04=$RUN_ROOT/4090/iter04/measure/kv_lora.json \
    --series iter05=$RUN_ROOT/4090/iter05/measure/kv_lora.json \
    --series iter06=$RUN_ROOT/4090/iter06/measure/kv_lora.json \
    --series iter07=$RUN_ROOT/4090/iter07/measure/kv_lora.json \
    --series iter08=$RUN_ROOT/4090/iter08/measure/kv_lora.json \
    --series iter09=$RUN_ROOT/4090/iter09/measure/kv_lora.json \
    --series iter10=$RUN_ROOT/4090/iter10/measure/kv_lora.json \
    --normalize iter00 \
    --metric total_best \
    --compare-mode intersection \
    --output-png $RUN_ROOT/4090/iter10/gen/bert_converge.png \
    --output-csv $RUN_ROOT/4090/iter10/gen/bert_converge.csv

  ### 说明

  - --normalize iter00：iter00 归一化为 1.0
  - 纵轴是 normalized total_best_ms，越低越好
  - compare-mode intersection：确保各迭代使用相同 workload 集合

  ### 1) 只指定网络（不指定 shape）

  会把 bert_base 的所有 shape 合并做趋势（前提是测量里有这些 shape）。

  python /home/hehangshuai/workspace/tlm/gen/scripts/plot_convergence.py \
    --target 4090 \
    --network bert_base \
    --series iter00=... \
    --series iter01=... \
    --normalize iter00 \
    --compare-mode union \
    --output-png ... \
    --output-csv ...

  ### 2) 不指定 network/shape（全量）

  会把 所有网络所有子图 汇总成一个总体趋势（最接近“总收敛趋势”）。

  python /home/hehangshuai/workspace/tlm/gen/scripts/plot_convergence.py \
    --target 4090 \
    --series iter00=... \
    --series iter01=... \
    --normalize iter00 \
    --compare-mode union \
    --output-png ... \
    --output-csv ...

  ### 3) 指定网络+shape（精确）

  你之前用的 bert_base + 1,128 就是最精确的，但如果这个 shape 没有被迭代测量（hold-out），趋势会空/为 0。
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


def collect_measured_stats(paths: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        for inp, res in auto_scheduler.RecordReader(path):
            key = inp.task.workload_key
            entry = stats.setdefault(
                key, {"best_ms": None, "sum_ms": 0.0, "ok": 0, "err": 0, "count": 0}
            )
            entry["count"] += 1
            if getattr(res, "error_no", 0) != 0:
                entry["err"] += 1
                continue
            costs = []
            for item in res.costs:
                try:
                    val = float(item.value) if hasattr(item, "value") else float(item)
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    costs.append(val)
            if not costs:
                entry["err"] += 1
                continue
            lat_ms = float(sum(costs) / len(costs)) * 1e3
            entry["ok"] += 1
            entry["sum_ms"] += lat_ms
            if entry["best_ms"] is None or lat_ms < entry["best_ms"]:
                entry["best_ms"] = lat_ms
    return stats


def parse_series_item(item: str, used: Dict[str, int]) -> Tuple[str, List[str]]:
    if "=" in item:
        name, path = item.split("=", 1)
        label = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_") or "series"
    else:
        path = item
        base = os.path.basename(path.rstrip("/"))
        if base.endswith(".json"):
            base = os.path.splitext(base)[0]
        label = re.sub(r"[^A-Za-z0-9_]+", "_", base.strip()).strip("_") or "series"
    if label in used:
        used[label] += 1
        label = f"{label}_{used[label]}"
    else:
        used[label] = 1
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        paths = [path]
    return label, paths


def sort_labels(labels: List[str], mode: str) -> List[str]:
    if mode == "label":
        return sorted(labels)
    if mode == "numeric":
        def keyfn(x: str) -> Tuple[int, str]:
            nums = re.findall(r"\d+", x)
            return (int(nums[-1]) if nums else -1, x)
        return sorted(labels, key=keyfn)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot convergence trend from measured JSONs.")
    parser.add_argument("--target", required=True, help="Target string for register_data_path.")
    parser.add_argument(
        "--series",
        action="append",
        help="Repeatable: label=path or path. Directory allowed.",
    )
    parser.add_argument("--network", default=None, help="Network name (e.g., bert_base).")
    parser.add_argument("--shape", default=None, help="Network shape (e.g., 1,128).")
    parser.add_argument("--network-info-dir", default=None, help="Override network_info dir.")
    parser.add_argument(
        "--metric",
        choices=("total_best", "total_weighted"),
        default="total_best",
        help="Which metric to plot.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("intersection", "union"),
        default="intersection",
        help="Use workload intersection across series or union.",
    )
    parser.add_argument(
        "--collision-policy",
        choices=("keep_first", "merge", "skip"),
        default="skip",
        help="How to handle workload_key collisions across networks.",
    )
    parser.add_argument(
        "--normalize",
        default=None,
        help="Baseline label for normalization (divide by baseline).",
    )
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--output-png", required=True, help="Output PNG path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--sort-mode",
        choices=("input", "label", "numeric"),
        default="input",
        help="Sort order of series labels.",
    )
    args = parser.parse_args()

    if not args.series:
        raise SystemExit("Provide --series (repeatable label=path or path).")

    if args.shape and not args.network:
        raise SystemExit("--shape requires --network.")

    target_str = resolve_target_string(args.target)
    common.register_data_path(args.target)
    target = tvm.target.Target(target_str)

    used_labels: Dict[str, int] = {}
    series_items: List[Tuple[str, List[str]]] = []
    for item in args.series:
        label, paths = parse_series_item(item, used_labels)
        paths = [p for p in paths if os.path.exists(p)]
        if paths:
            series_items.append((label, paths))
    if not series_items:
        raise FileNotFoundError("No measured JSONL files found in series.")

    network_info_dir = args.network_info_dir or common.NETWORK_INFO_FOLDER
    if not network_info_dir or not os.path.isdir(network_info_dir):
        raise FileNotFoundError(f"network_info dir not found: {network_info_dir}")

    only_shape = parse_shape_arg(args.shape)
    if args.shape and not only_shape:
        raise ValueError(f"Invalid --shape: {args.shape}")

    wk_to_meta, wk_to_weight = build_workload_map(
        network_info_dir,
        target.kind.name,
        only_network=args.network,
        only_shape=only_shape,
        collision_policy=args.collision_policy,
    )
    if not wk_to_meta:
        raise RuntimeError("No workload mappings found. Check --network/--shape.")

    stats_by_label: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, paths in series_items:
        stats_by_label[label] = collect_measured_stats(paths)

    labels = [label for label, _ in series_items]
    labels = sort_labels(labels, args.sort_mode)

    if args.compare_mode == "intersection":
        wk_use = set(wk_to_meta.keys())
        for label in labels:
            wk_use = {wk for wk in wk_use if stats_by_label.get(label, {}).get(wk, {}).get("best_ms") is not None}
    else:
        wk_use = set(wk_to_meta.keys())
    wk_use = sorted(wk_use)

    totals = {}
    for label in labels:
        total_best = 0.0
        total_weighted = 0.0
        count = 0
        stats = stats_by_label.get(label, {})
        for wk in wk_use:
            best = stats.get(wk, {}).get("best_ms")
            if best is None and args.compare_mode == "union":
                continue
            if best is None:
                continue
            total_best += best
            total_weighted += best * wk_to_weight.get(wk, 1.0)
            count += 1
        totals[label] = {
            "total_best": total_best,
            "total_weighted": total_weighted,
            "count": count,
        }

    baseline = args.normalize or labels[0]
    if baseline not in totals:
        raise ValueError(f"Baseline label not found: {baseline}")
    base_val = totals[baseline][args.metric]
    if base_val == 0:
        raise ValueError(f"Baseline value is 0 for metric {args.metric}")

    rows = []
    for idx, label in enumerate(labels):
        val = totals[label][args.metric]
        rows.append(
            {
                "index": idx,
                "label": label,
                "metric": args.metric,
                "value": val,
                "normalized": val / base_val,
                "workload_count": totals[label]["count"],
                "baseline": baseline,
                "compare_mode": args.compare_mode,
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "label",
            "metric",
            "value",
            "normalized",
            "workload_count",
            "baseline",
            "compare_mode",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote CSV: {args.output_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    xs = [row["index"] for row in rows]
    ys = [row["normalized"] for row in rows]
    xticklabels = [row["label"] for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xticks(xs, xticklabels, rotation=45, ha="right")
    plt.ylabel(f"normalized {args.metric} (baseline={baseline})")
    plt.xlabel("iteration")
    if args.title:
        plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, dpi=200)
    print(f"Wrote PNG: {args.output_png}")


if __name__ == "__main__":
    main()
