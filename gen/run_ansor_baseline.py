import argparse
import ast
import csv
import glob
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tvm
from tvm import auto_scheduler

import common
from common import hold_out_task_files, register_data_path


def resolve_target_string(target_str: str) -> str:
    if not isinstance(target_str, str):
        return target_str
    ts = target_str.strip().lower()
    if ts in ("4090", "rtx-4090", "nvidia/rtx-4090"):
        return (
            "cuda -keys=cuda,gpu "
            "-arch=sm_86 "
            "-max_num_threads=1024 "
            "-max_shared_memory_per_block=49152 "
            "-max_threads_per_block=1024 "
            "-registers_per_block=65536 "
            "-thread_warp_size=32"
        )
    if ts in ("orin", "jetson-orin", "nvidia/jetson-orin"):
        return (
            "cuda -keys=cuda,gpu "
            "-arch=sm_87 "
            "-max_num_threads=1024 "
            "-max_shared_memory_per_block=49152 "
            "-max_threads_per_block=1024 "
            "-registers_per_block=65536 "
            "-thread_warp_size=32"
        )
    return target_str


def parse_task_filename(path: str) -> Optional[Tuple[str, List[int], str]]:
    base = os.path.basename(path)
    if not base.endswith(".task.pkl") or not base.startswith("(("):
        return None
    # Example: ((bert_base,[1,128]),cuda).task.pkl
    # Example: ((resnet_50,[1,3,224,224]),llvm).task.pkl
    mid = base[:-len(".task.pkl")]
    if not mid.endswith(")"):
        return None
    # Split by the last "),"
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


def read_best_latency_ms(log_path: str) -> Optional[float]:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None
    inputs, results = auto_scheduler.RecordReader(log_path).read_lines()
    best = None
    for res in results:
        if getattr(res, "error_no", 0) != 0:
            continue
        costs = []
        for item in res.costs:
            val = _cost_to_float(item)
            if val is not None:
                costs.append(val)
        if not costs:
            continue
        latency = float(np.mean(costs)) * 1e3
        if best is None or latency < best:
            best = latency
    return best


def count_measure_records(log_path: str) -> Tuple[int, int]:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return 0, 0
    total = 0
    ok = 0
    for _inp, res in auto_scheduler.RecordReader(log_path):
        total += 1
        if getattr(res, "error_no", 0) == 0:
            ok += 1
    return total, ok


def round_robin_select(
    tasks_by_net: Dict[str, List[Tuple[auto_scheduler.SearchTask, float]]],
    max_tasks: int,
) -> List[Tuple[str, auto_scheduler.SearchTask, float]]:
    selected = []
    net_ids = list(tasks_by_net.keys())
    while len(selected) < max_tasks:
        progressed = False
        for net_id in net_ids:
            if len(selected) >= max_tasks:
                break
            if tasks_by_net[net_id]:
                task, weight = tasks_by_net[net_id].pop(0)
                selected.append((net_id, task, weight))
                progressed = True
        if not progressed:
            break
    return selected


def ensure_parent(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument(
        "--host_target",
        type=str,
        default="",
        help="Optional host target string; when set, use Target(device, host=host_target).",
    )
    parser.add_argument(
        "--network_names",
        type=str,
        default=None,
        help="Comma-separated network names (e.g., bert_base,resnet_50,mobilenet_v2).",
    )
    parser.add_argument(
        "--task_files",
        type=str,
        default=None,
        help="Comma-separated .task.pkl files. If set, overrides --network_names.",
    )
    parser.add_argument(
        "--one_shape_per_network",
        action="store_true",
        help="Pick only the first shape per network name (sorted by filename).",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=0,
        help="Max number of tasks to tune (0 means no limit).",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="64,1000,10000",
        help="Comma-separated num_measure_trials per task.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Ansor_baseline/ansor",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Ansor_baseline/ansor_summary.csv",
    )
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--min_repeat_ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--builder_timeout",
        type=int,
        default=120,
        help="LocalBuilder compile timeout in seconds.",
    )
    parser.add_argument(
        "--builder_n_parallel",
        type=int,
        default=1,
        help="LocalBuilder parallel compile workers (use 1 for easier debugging).",
    )
    parser.add_argument(
        "--builder_verbose",
        type=int,
        default=2,
        help="LocalBuilder verbose level (2 prints detailed compile stderr).",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Retune even if log exists.")
    args = parser.parse_args()

    resolved_target = resolve_target_string(args.target)
    register_data_path(resolved_target)
    if args.host_target:
        target = tvm.target.Target(resolved_target, host=args.host_target)
    else:
        target = tvm.target.Target(resolved_target)
    print(f"[TARGET] device={target}")
    if getattr(target, "host", None) is not None:
        print(f"[TARGET] host={target.host}")
    if common.NETWORK_INFO_FOLDER is None:
        raise RuntimeError("NETWORK_INFO_FOLDER is not set; call register_data_path first.")

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    out_target = common.HARDWARE_PLATFORM or "unknown_target"

    task_files: List[str] = []
    if args.task_files:
        task_files = [x for x in args.task_files.split(",") if x.strip()]
    elif args.network_names:
        names = [x.strip() for x in args.network_names.split(",") if x.strip()]
        holdout_map = hold_out_task_files(target)
        target_kind = target.kind.name
        if args.one_shape_per_network:
            for name in names:
                if name in holdout_map and os.path.exists(holdout_map[name]):
                    task_files.append(holdout_map[name])
                    continue
                # Fallback: first match under network_info for this name/target kind
                matches = []
                for path in sorted(glob.glob(os.path.join(common.NETWORK_INFO_FOLDER, "*.task.pkl"))):
                    parsed = parse_task_filename(path)
                    if not parsed:
                        continue
                    net_name, _shape, tkind = parsed
                    if net_name == name and tkind == target_kind:
                        matches.append(path)
                if matches:
                    task_files.append(matches[0])
                else:
                    print(f"[WARN] No task file found for network {name} (target={target_kind})")
        else:
            for path in sorted(glob.glob(os.path.join(common.NETWORK_INFO_FOLDER, "*.task.pkl"))):
                parsed = parse_task_filename(path)
                if not parsed:
                    continue
                net_name, _shape, tkind = parsed
                if net_name in names and tkind == target_kind:
                    task_files.append(path)
    else:
        # Default: use hold-out set from common.py
        task_files = list(hold_out_task_files(target).values())

    if not task_files:
        raise RuntimeError("No task files found. Check --network_names or --task_files.")

    tasks_by_net: Dict[str, List[Tuple[auto_scheduler.SearchTask, float]]] = {}
    net_meta: Dict[str, Tuple[str, List[int]]] = {}
    parse_failed = 0
    target_mismatch = 0
    for path in task_files:
        parsed = parse_task_filename(path)
        if not parsed:
            parse_failed += 1
            continue
        name, shape, target_kind = parsed
        if target_kind != target.kind.name:
            target_mismatch += 1
            continue
        net_id = format_network_id(name, shape)
        tasks, task_weights = pickle.load(open(path, "rb"))
        tasks_by_net[net_id] = list(zip(tasks, task_weights))
        net_meta[net_id] = (name, shape)

    if parse_failed or target_mismatch:
        print(f"[WARN] Skipped task files: parse_failed={parse_failed}, target_mismatch={target_mismatch}")

    if args.max_tasks <= 0:
        selected = []
        for net_id, tasks in tasks_by_net.items():
            for task, weight in tasks:
                selected.append((net_id, task, weight))
    else:
        selected = round_robin_select(tasks_by_net, args.max_tasks)
    if not selected:
        raise RuntimeError("Selected task list is empty after filtering.")

    if target.kind.name == "llvm":
        enable_cpu_cache_flush = True
    else:
        enable_cpu_cache_flush = False
    runner = auto_scheduler.LocalRunner(
        repeat=args.repeat,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
        number=args.number,
        timeout=args.timeout,
        min_repeat_ms=args.min_repeat_ms,
    )
    try:
        builder = auto_scheduler.LocalBuilder(
            timeout=args.builder_timeout,
            n_parallel=args.builder_n_parallel,
            verbose=args.builder_verbose,
        )
    except TypeError:
        # Backward compatibility for older TVM builds without verbose arg.
        builder = auto_scheduler.LocalBuilder(
            timeout=args.builder_timeout,
            n_parallel=args.builder_n_parallel,
        )
    print(
        f"[BUILDER] timeout={args.builder_timeout}s "
        f"n_parallel={args.builder_n_parallel} verbose={args.builder_verbose}"
    )

    summary_fields = [
        "row_type",
        "target",
        "network_id",
        "network_name",
        "network_shape",
        "budget",
        "task_idx",
        "workload_key",
        "task_weight",
        "best_latency_ms",
        "tune_time_s",
        "log_path",
        "notes",
    ]
    summary_exists = os.path.exists(args.summary_csv)
    ensure_parent(os.path.dirname(args.summary_csv))

    with open(args.summary_csv, "a", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=summary_fields)
        if not summary_exists:
            writer.writeheader()

        total_tasks_by_net = {k: len(v) for k, v in tasks_by_net.items()}
        selected_tasks_by_net: Dict[str, int] = {}
        for net_id, _task, _weight in selected:
            selected_tasks_by_net[net_id] = selected_tasks_by_net.get(net_id, 0) + 1

        for budget in budgets:
            task_rows = []
            for task_idx, (net_id, task, weight) in enumerate(selected):
                net_name, shape = net_meta[net_id]
                out_dir = os.path.join(args.out_root, out_target, net_id, f"times_{budget}")
                ensure_parent(out_dir)
                log_path = os.path.join(out_dir, f"task_{task_idx}_{task.workload_key}.json")

                existing_total, existing_ok = count_measure_records(log_path)
                if existing_total > 0:
                    remaining = max(budget - existing_total, 0)
                    print(
                        f"[RESUME] {net_id} task_idx={task_idx} budget={budget} "
                        f"done={existing_total} remaining={remaining} ok={existing_ok}"
                    )
                if args.force and os.path.exists(log_path):
                    os.remove(log_path)
                    existing_total, existing_ok = 0, 0

                if existing_total >= budget and not args.force:
                    tune_time_s = 0.0
                    notes = f"skip_full_budget:records={existing_total},ok={existing_ok}"
                    print(
                        f"[SKIP] {net_id} task_idx={task_idx} budget={budget} "
                        f"done={existing_total} remaining=0 ok={existing_ok}"
                    )
                else:
                    load_log_file = log_path if existing_total > 0 else None
                    tuner = auto_scheduler.TaskScheduler(
                        [task],
                        [1.0],
                        load_log_file=load_log_file,
                    )
                    options = auto_scheduler.TuningOptions(
                        num_measure_trials=budget,
                        builder=builder,
                        runner=runner,
                        verbose=args.verbose,
                        measure_callbacks=[auto_scheduler.RecordToFile(log_path)],
                    )
                    start_time = time.time()
                    tuner.tune(options)
                    tune_time_s = time.time() - start_time
                    if load_log_file:
                        notes = f"resume_from:records={existing_total},ok={existing_ok}"
                    else:
                        notes = ""

                best_latency_ms = read_best_latency_ms(log_path)
                row = {
                    "row_type": "task",
                    "target": out_target,
                    "network_id": net_id,
                    "network_name": net_name,
                    "network_shape": str(shape),
                    "budget": budget,
                    "task_idx": task_idx,
                    "workload_key": task.workload_key,
                    "task_weight": float(weight),
                    "best_latency_ms": "" if best_latency_ms is None else f"{best_latency_ms:.6f}",
                    "tune_time_s": f"{tune_time_s:.3f}",
                    "log_path": log_path,
                    "notes": notes,
                }
                writer.writerow(row)
                task_rows.append((net_id, weight, best_latency_ms))

            # Network-level summary (weighted sum over selected tasks)
            net_summary: Dict[str, Tuple[float, float, int]] = {}
            for net_id, weight, best_latency_ms in task_rows:
                if best_latency_ms is None:
                    continue
                if net_id not in net_summary:
                    net_summary[net_id] = (0.0, 0.0, 0)
                total_lat, total_weight, cnt = net_summary[net_id]
                net_summary[net_id] = (
                    total_lat + best_latency_ms * weight,
                    total_weight + weight,
                    cnt + 1,
                )

            for net_id, (lat_sum, w_sum, cnt) in net_summary.items():
                net_name, shape = net_meta[net_id]
                selected_cnt = selected_tasks_by_net.get(net_id, 0)
                total_cnt = total_tasks_by_net.get(net_id, 0)
                row = {
                    "row_type": "network_summary",
                    "target": out_target,
                    "network_id": net_id,
                    "network_name": net_name,
                    "network_shape": str(shape),
                    "budget": budget,
                    "task_idx": "",
                    "workload_key": "",
                    "task_weight": f"{w_sum:.6f}",
                    "best_latency_ms": f"{lat_sum:.6f}",
                    "tune_time_s": "",
                    "log_path": "",
                    "notes": f"covered_tasks={cnt};selected_tasks={selected_cnt}/{total_cnt}",
                }
                writer.writerow(row)


if __name__ == "__main__":
    main()
