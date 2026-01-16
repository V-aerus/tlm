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
新脚本：
  gen/scripts/compare_measured_versions.py

  它支持：

  - 同时比较多个 measured.json
  - 输出每个网络总延迟对比（可加权）
  - 输出每个子图延迟对比
  - 支持 baseline 自动算 Δ 和 speedup
  - 支持 --only-network bert_base --only-shape 1,128，避免 workload 冲突

  ———

  ### 示例（对比官方 vs v8）

  python /home/hehangshuai/workspace/tlm/gen/scripts/compare_measured_versions.py \
    --target 4090 \
    --only-network bert_base \
    --only-shape 1,128 \
    --measured v10_gain=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter10/measure/kv_lora.json \
    --measured v8_gain=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter08/measure/kv_lora.json \
    --baseline v8_gain \ 
    --output-network-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter10/gen/compare_network.csv \
    --output-workload-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter10/gen/compare_workload.csv

    python /home/hehangshuai/workspace/tlm/gen/scripts/compare_measured_versions.py \
    --target 4090 \
    --compare-mode union \
    --measured v12_gain=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter12/measure/kv_lora.json \
    --measured v8_gain=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter08/measure/kv_lora.json \
    --baseline v8_gain \
    --output-network-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter12/gen/compare_network.csv \
    --output-workload-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter12/gen/compare_workload.csv

'''

'''
• 用来 对比 4 个 measured 文件（3 个网络、4 种方法），最合适的是：

  gen/scripts/compare_measured_versions.py
  它会同时生成：

  - network 级总延迟对比（更直观）
  - workload 级子图对比（更细）

  建议用这种方式：

  MEAS_DIR=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/3090/iter00/eval_ansor_measure

  python /home/hehangshuai/workspace/tlm/gen/scripts/compare_measured_versions.py \
    --target 3090 \
    --compare-mode intersection \
    --measured mix=$MEAS_DIR/kv_lora_mix_measured.json \
    --measured lora_4090=$MEAS_DIR/kv_lora_4090_measured.json \
    --measured lora_v100=$MEAS_DIR/kv_lora_v100_measured.json \
    --measured official=$MEAS_DIR/official_measured.json \
    --baseline official \
    --output-network-csv $MEAS_DIR/compare_network.csv \
    --output-workload-csv $MEAS_DIR/compare_workload.csv

  说明：

  - intersection 只比较“所有方法都测到”的 workload，更公平
  - baseline 用 official 可以直接看 speedup
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
    if ts in ("2080", "rtx-2080", "2080ti", "rtx-2080-ti", "geforce-rtx-2080", "geforce-rtx-2080-ti"):
        return (
            "cuda -keys=cuda,gpu "
            "-arch=sm_75 "
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


def compute_network_totals(
    stats_by_label: Dict[str, Dict[str, Dict[str, float]]],
    wk_to_meta: Dict[str, Dict[str, str]],
    wk_to_weight: Dict[str, float],
    compare_mode: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    totals: Dict[str, Dict[str, float]] = {}
    totals_weighted: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    nets = {}
    for wk, meta in wk_to_meta.items():
        nets.setdefault(meta["network_id"], meta)

    for net_id, meta in nets.items():
        wk_all = [wk for wk, m in wk_to_meta.items() if m["network_id"] == net_id]
        totals[net_id] = {}
        totals_weighted[net_id] = {}
        counts[net_id] = {}

        if compare_mode == "intersection":
            common = set(wk_all)
            for label, stats in stats_by_label.items():
                common = {wk for wk in common if stats.get(wk, {}).get("best_ms") is not None}
            wk_use = sorted(common)
        else:
            wk_use = sorted(wk_all)

        for label, stats in stats_by_label.items():
            total = 0.0
            total_weighted = 0.0
            count = 0
            for wk in wk_use:
                best = stats.get(wk, {}).get("best_ms")
                if best is None and compare_mode == "union":
                    continue
                if best is None:
                    continue
                total += best
                total_weighted += best * wk_to_weight.get(wk, 1.0)
                count += 1
            totals[net_id][label] = total
            totals_weighted[net_id][label] = total_weighted
            counts[net_id][label] = count
    return totals, totals_weighted, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare measured JSONs across versions.")
    parser.add_argument("--target", required=True, help="Target string for register_data_path.")
    parser.add_argument(
        "--measured",
        action="append",
        help="Repeatable measured input: name=path or path. Directory allowed.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline label for delta/speedup.")
    parser.add_argument("--output-network-csv", default=None, help="Output CSV for network totals.")
    parser.add_argument("--output-workload-csv", default=None, help="Output CSV for per-workload stats.")
    parser.add_argument("--network-info-dir", default=None, help="Override network_info dir.")
    parser.add_argument("--only-network", default=None, help="Restrict to a single network name.")
    parser.add_argument("--only-shape", default=None, help="Restrict to one network shape (e.g., 1,128).")
    parser.add_argument(
        "--collision-policy",
        choices=("keep_first", "merge", "skip"),
        default="skip",
        help="How to handle workload_key collisions across networks.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("intersection", "union"),
        default="intersection",
        help="Use workload intersection across labels or union.",
    )
    args = parser.parse_args()

    if not args.measured:
        raise SystemExit("Provide --measured (repeatable name=path or path).")

    target_str = resolve_target_string(args.target)
    common.register_data_path(args.target)
    target = tvm.target.Target(target_str)

    used_labels: Dict[str, int] = {}
    measured_sets: List[Tuple[str, List[str]]] = []
    for item in args.measured:
        label, paths = parse_measured_item(item, used_labels)
        paths = [p for p in paths if os.path.exists(p)]
        if paths:
            measured_sets.append((label, paths))
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
    if not wk_to_meta:
        raise RuntimeError("No workload mappings found. Check --only-network/--only-shape.")

    stats_by_label: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, paths in measured_sets:
        stats_by_label[label] = collect_measured_stats(paths)

    totals, totals_weighted, counts = compute_network_totals(
        stats_by_label, wk_to_meta, wk_to_weight, args.compare_mode
    )

    labels = [label for label, _ in measured_sets]
    baseline = args.baseline if args.baseline in labels else None

    out_net = args.output_network_csv
    if not out_net:
        first_label, first_paths = measured_sets[0]
        stem = _sanitize_label(first_label)
        out_net = os.path.join(os.path.dirname(first_paths[0]), f"{stem}_compare_network.csv")

    with open(out_net, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["network_name", "network_shape", "network_id", "compare_mode"]
        for label in labels:
            fieldnames.extend(
                [
                    f"{label}_total_best_ms",
                    f"{label}_total_weighted_ms",
                    f"{label}_workload_count",
                ]
            )
        if baseline:
            for label in labels:
                if label == baseline:
                    continue
                fieldnames.extend(
                    [
                        f"{label}_delta_best_ms",
                        f"{label}_speedup_best",
                        f"{label}_delta_weighted_ms",
                        f"{label}_speedup_weighted",
                    ]
                )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for net_id, meta in sorted({m["network_id"]: m for m in wk_to_meta.values()}.items()):
            row = {
                "network_name": meta["network_name"],
                "network_shape": meta["network_shape"],
                "network_id": net_id,
                "compare_mode": args.compare_mode,
            }
            for label in labels:
                row[f"{label}_total_best_ms"] = f"{totals.get(net_id, {}).get(label, 0.0):.6f}"
                row[f"{label}_total_weighted_ms"] = f"{totals_weighted.get(net_id, {}).get(label, 0.0):.6f}"
                row[f"{label}_workload_count"] = counts.get(net_id, {}).get(label, 0)
            if baseline:
                base_best = totals.get(net_id, {}).get(baseline, 0.0)
                base_weighted = totals_weighted.get(net_id, {}).get(baseline, 0.0)
                for label in labels:
                    if label == baseline:
                        continue
                    cur_best = totals.get(net_id, {}).get(label, 0.0)
                    cur_weighted = totals_weighted.get(net_id, {}).get(label, 0.0)
                    row[f"{label}_delta_best_ms"] = f"{cur_best - base_best:.6f}"
                    row[f"{label}_speedup_best"] = "" if cur_best == 0 else f"{base_best / cur_best:.6f}"
                    row[f"{label}_delta_weighted_ms"] = f"{cur_weighted - base_weighted:.6f}"
                    row[f"{label}_speedup_weighted"] = "" if cur_weighted == 0 else f"{base_weighted / cur_weighted:.6f}"
            writer.writerow(row)

    print(f"Wrote network CSV: {out_net}")

    if args.output_workload_csv:
        out_wk = args.output_workload_csv
        with open(out_wk, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["network_name", "network_shape", "network_id", "workload_key"]
            for label in labels:
                fieldnames.extend(
                    [
                        f"{label}_best_ms",
                        f"{label}_mean_ms",
                        f"{label}_ok",
                        f"{label}_err",
                        f"{label}_count",
                    ]
                )
            if baseline:
                for label in labels:
                    if label == baseline:
                        continue
                    fieldnames.extend(
                        [
                            f"{label}_delta_ms",
                            f"{label}_speedup",
                        ]
                    )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for wk in sorted(wk_to_meta.keys()):
                meta = wk_to_meta[wk]
                row = {
                    "network_name": meta["network_name"],
                    "network_shape": meta["network_shape"],
                    "network_id": meta["network_id"],
                    "workload_key": wk,
                }
                for label in labels:
                    stat = stats_by_label.get(label, {}).get(wk, {})
                    best = stat.get("best_ms")
                    ok = stat.get("ok", 0)
                    mean = None
                    if ok:
                        mean = stat.get("sum_ms", 0.0) / ok
                    row[f"{label}_best_ms"] = "" if best is None else f"{best:.6f}"
                    row[f"{label}_mean_ms"] = "" if mean is None else f"{mean:.6f}"
                    row[f"{label}_ok"] = ok
                    row[f"{label}_err"] = stat.get("err", 0)
                    row[f"{label}_count"] = stat.get("count", 0)
                if baseline:
                    base_best = stats_by_label.get(baseline, {}).get(wk, {}).get("best_ms")
                    for label in labels:
                        if label == baseline:
                            continue
                        cur_best = stats_by_label.get(label, {}).get(wk, {}).get("best_ms")
                        if base_best is None or cur_best is None or cur_best == 0:
                            row[f"{label}_delta_ms"] = ""
                            row[f"{label}_speedup"] = ""
                        else:
                            row[f"{label}_delta_ms"] = f"{cur_best - base_best:.6f}"
                            row[f"{label}_speedup"] = f"{base_best / cur_best:.6f}"
                writer.writerow(row)
        print(f"Wrote workload CSV: {out_wk}")


if __name__ == "__main__":
    main()
