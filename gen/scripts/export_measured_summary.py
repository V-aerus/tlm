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


def _cost_to_float(val) -> Optional[float]:
    try:
        if hasattr(val, "value"):
            return float(val.value)
        return float(val)
    except (TypeError, ValueError):
        return None


def collect_measured_stats(paths: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        inputs, results = auto_scheduler.RecordReader(path).read_lines()
        for inp, res in zip(inputs, results):
            key = inp.task.workload_key
            entry = stats.setdefault(
                key, {"best_ms": None, "ok": 0, "err": 0, "count": 0}
            )
            entry["count"] += 1
            if getattr(res, "error_no", 0) != 0:
                entry["err"] += 1
                continue
            costs = []
            for item in res.costs:
                val = _cost_to_float(item)
                if val is not None:
                    costs.append(val)
            if not costs:
                entry["err"] += 1
                continue
            lat_ms = float(sum(costs) / len(costs)) * 1e3
            entry["ok"] += 1
            if entry["best_ms"] is None or lat_ms < entry["best_ms"]:
                entry["best_ms"] = lat_ms
    return stats


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "measured"


def _derive_label_from_path(path: str) -> str:
    base = os.path.basename(path.rstrip("/"))
    if base.endswith(".json"):
        base = os.path.splitext(base)[0]
    return _sanitize_label(base)


def _unique_label(label: str, used: Dict[str, int]) -> str:
    if label not in used:
        used[label] = 1
        return label
    used[label] += 1
    return f"{label}_{used[label]}"


def parse_measured_item(item: str, used_labels: Dict[str, int]) -> Tuple[str, List[str]]:
    if "=" in item:
        name, path = item.split("=", 1)
        label = _sanitize_label(name)
    else:
        path = item
        label = _derive_label_from_path(path)
    label = _unique_label(label, used_labels)
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        paths = [path]
    return label, paths


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
    only_network: Optional[str] = None,
    only_shape: Optional[List[int]] = None,
    collision_policy: str = "keep_first",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export measured workload latency summary.")
    parser.add_argument("--target", required=True, help="Target string for register_data_path.")
    parser.add_argument(
        "--measured",
        action="append",
        help="Repeatable measured input: name=path or path. "
             "If path is a directory, all *.json files are used.",
    )
    parser.add_argument("--measured-path", default=None, help="Measured JSONL path.")
    parser.add_argument("--record-dir", default=None, help="Directory with measured JSONL files.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path.")
    parser.add_argument("--network-info-dir", default=None, help="Override network_info dir.")
    parser.add_argument("--only-network", default=None, help="Restrict to a single network name (e.g., bert_base).")
    parser.add_argument("--only-shape", default=None, help="Restrict to one network shape (e.g., 1,128 or 1x128).")
    parser.add_argument(
        "--collision-policy",
        choices=("keep_first", "merge", "skip"),
        default="keep_first",
        help="How to handle workload_key collisions across networks.",
    )
    args = parser.parse_args()

    if not args.measured and not args.measured_path and not args.record_dir:
        raise SystemExit("Provide --measured, --measured-path or --record-dir.")

    target_str = resolve_target_string(args.target)
    common.register_data_path(args.target)
    target = tvm.target.Target(target_str)

    measured_sets: List[Tuple[str, List[str]]] = []
    used_labels: Dict[str, int] = {}
    if args.measured:
        for item in args.measured:
            label, paths = parse_measured_item(item, used_labels)
            measured_sets.append((label, paths))
    if args.measured_path:
        label = _unique_label(_derive_label_from_path(args.measured_path), used_labels)
        measured_sets.append((label, [args.measured_path]))
    if args.record_dir:
        label = _unique_label(_derive_label_from_path(args.record_dir), used_labels)
        paths = sorted(glob.glob(os.path.join(args.record_dir, "*.json")))
        measured_sets.append((label, paths))
    measured_sets = [
        (label, [p for p in paths if os.path.exists(p)])
        for label, paths in measured_sets
    ]
    measured_sets = [(label, paths) for label, paths in measured_sets if paths]
    if not measured_sets:
        raise FileNotFoundError("No measured JSONL files found.")

    network_info_dir = args.network_info_dir or common.NETWORK_INFO_FOLDER
    if not network_info_dir or not os.path.isdir(network_info_dir):
        raise FileNotFoundError(f"network_info dir not found: {network_info_dir}")

    only_shape = parse_shape_arg(args.only_shape)
    if args.only_shape and not only_shape:
        raise ValueError(f"Invalid --only-shape: {args.only_shape}")

    wk_to_meta, wk_to_weight = build_workload_map(
        network_info_dir,
        target.kind.name,
        only_network=args.only_network,
        only_shape=only_shape,
        collision_policy=args.collision_policy,
    )
    stats_by_label: Dict[str, Dict[str, Dict[str, float]]] = {}
    total_by_label: Dict[str, Dict[str, float]] = {}
    total_weighted_by_label: Dict[str, Dict[str, float]] = {}
    for label, paths in measured_sets:
        stats = collect_measured_stats(paths)
        stats_by_label[label] = stats
        total_by_net: Dict[str, float] = {}
        total_weighted_by_net: Dict[str, float] = {}
        for wk, stat in stats.items():
            best_ms = stat["best_ms"]
            if best_ms is None:
                continue
            meta = wk_to_meta.get(wk)
            if not meta:
                continue
            net_id = meta["network_id"]
            total_by_net[net_id] = total_by_net.get(net_id, 0.0) + best_ms
            weight = wk_to_weight.get(wk, 1.0)
            total_weighted_by_net[net_id] = total_weighted_by_net.get(net_id, 0.0) + best_ms * weight
        total_by_label[label] = total_by_net
        total_weighted_by_label[label] = total_weighted_by_net

    out_csv = args.output_csv
    if not out_csv:
        first_label, first_paths = measured_sets[0]
        stem = first_label or os.path.splitext(os.path.basename(first_paths[0]))[0]
        out_csv = os.path.join(os.path.dirname(first_paths[0]), f"{stem}_summary.csv")

    all_keys = set()
    for stats in stats_by_label.values():
        all_keys.update(stats.keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["network_name", "network_shape", "network_id", "workload_key"]
        for label in stats_by_label.keys():
            fieldnames.extend(
                [
                    f"{label}_best_ms",
                    f"{label}_ok",
                    f"{label}_err",
                    f"{label}_count",
                    f"{label}_network_total_best_ms",
                    f"{label}_network_total_weighted_ms",
                ]
            )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for wk in sorted(all_keys):
            meta = wk_to_meta.get(wk, {"network_name": "", "network_shape": "", "network_id": ""})
            net_id = meta["network_id"]
            row = {
                "network_name": meta["network_name"],
                "network_shape": meta["network_shape"],
                "network_id": net_id,
                "workload_key": wk,
            }
            for label, stats in stats_by_label.items():
                stat = stats.get(wk)
                if stat:
                    best_ms = stat["best_ms"]
                    row[f"{label}_best_ms"] = "" if best_ms is None else f"{best_ms:.6f}"
                    row[f"{label}_ok"] = int(stat["ok"])
                    row[f"{label}_err"] = int(stat["err"])
                    row[f"{label}_count"] = int(stat["count"])
                else:
                    row[f"{label}_best_ms"] = ""
                    row[f"{label}_ok"] = 0
                    row[f"{label}_err"] = 0
                    row[f"{label}_count"] = 0
                if net_id:
                    row[f"{label}_network_total_best_ms"] = f"{total_by_label.get(label, {}).get(net_id, 0.0):.6f}"
                    row[f"{label}_network_total_weighted_ms"] = f"{total_weighted_by_label.get(label, {}).get(net_id, 0.0):.6f}"
                else:
                    row[f"{label}_network_total_best_ms"] = ""
                    row[f"{label}_network_total_weighted_ms"] = ""
            writer.writerow(row)

    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
