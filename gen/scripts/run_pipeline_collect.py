import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import tvm
from tvm import auto_scheduler

import common
from gen.make_dataset import FOR_GEN_EVAL_SKETCH, token_files_and_merge


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


def run_cmd(cmd: List[str], log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.flush()
    subprocess.run(cmd, check=True)


def append_run_log(log_path: str, msg: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{msg.rstrip()}\n")
        f.flush()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_list(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


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


def is_empty_file(path: str) -> bool:
    try:
        return os.path.getsize(path) == 0
    except OSError:
        return True


def read_best_latency_ms(log_path: str) -> Optional[float]:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None
    _inputs, results = auto_scheduler.RecordReader(log_path).read_lines()
    best = None
    for res in results:
        if getattr(res, "error_no", 0) != 0:
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
            continue
        latency = float(sum(costs) / len(costs)) * 1e3
        if best is None or latency < best:
            best = latency
    return best


def compute_overall_latency_ms(task_pkl: str, record_path: str) -> Optional[float]:
    if not os.path.exists(record_path):
        return None
    inputs, results = auto_scheduler.RecordReader(record_path).read_lines()
    input_dict: Dict[str, float] = {}
    for inp, res in zip(inputs, results):
        if getattr(res, "error_no", 0) != 0:
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
            continue
        latency = float(sum(costs) / len(costs))
        if inp.task.workload_key not in input_dict:
            input_dict[inp.task.workload_key] = latency
        else:
            input_dict[inp.task.workload_key] = min(input_dict[inp.task.workload_key], latency)

    tasks, task_weights = common.pickle.load(open(task_pkl, "rb"))
    total = 0.0
    for task, weight in zip(tasks, task_weights):
        if task.workload_key not in input_dict:
            continue
        total += input_dict[task.workload_key] * weight
    return total * 1e3


def load_gen_candidates(gen_path: str) -> Dict[str, List[str]]:
    workload_to_candidates: Dict[str, List[str]] = {}
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "i" not in obj or not isinstance(obj["i"], list):
                continue
            try:
                workload_key = obj["i"][0][0]
            except Exception:
                continue
            cand_key = json.dumps(obj["i"])
            workload_to_candidates.setdefault(workload_key, []).append(cand_key)
    return workload_to_candidates


def load_measured_latency(measured_path: str) -> Dict[str, float]:
    lat_map: Dict[str, float] = {}
    if not os.path.exists(measured_path):
        return lat_map
    with open(measured_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "i" not in obj or "r" not in obj:
                continue
            r = obj["r"]
            if not isinstance(r, list) or len(r) < 2:
                continue
            if r[1] != 0:
                continue
            costs = r[0]
            if not isinstance(costs, list) or not costs:
                continue
            latency = float(sum(costs) / len(costs))
            cand_key = json.dumps(obj["i"])
            if cand_key not in lat_map:
                lat_map[cand_key] = latency
            else:
                lat_map[cand_key] = min(lat_map[cand_key], latency)
    return lat_map


def compute_overall_latency_ms_from_gen(
    task_pkl: str,
    gen_path: str,
    measured_path: str,
    k: int,
) -> Tuple[Optional[float], int, int]:
    if not os.path.exists(gen_path) or not os.path.exists(measured_path):
        return None, 0, 0
    workload_to_candidates = load_gen_candidates(gen_path)
    lat_map = load_measured_latency(measured_path)
    tasks, task_weights = common.pickle.load(open(task_pkl, "rb"))
    total = 0.0
    missing = 0
    covered = 0
    for task, weight in zip(tasks, task_weights):
        cand_list = workload_to_candidates.get(task.workload_key, [])
        best = None
        for cand in cand_list[:k]:
            if cand in lat_map:
                best = lat_map[cand] if best is None else min(best, lat_map[cand])
        if best is None:
            missing += 1
            continue
        covered += 1
        total += best * weight
    return total * 1e3, missing, covered


def build_sketch(
    task_pkl: str,
    keep_cnt: int,
    sketch_path: str,
    regen_sketch: bool,
) -> str:
    if os.path.exists(sketch_path) and not regen_sketch:
        return sketch_path

    tasks, _task_weights = common.pickle.load(open(task_pkl, "rb"))
    files = []
    missing = []
    for task in tasks:
        path = common.get_to_measure_filename(task)
        if os.path.exists(path):
            files.append(path)
        else:
            missing.append(path)
    if missing:
        raise RuntimeError(f"Missing to_measure_programs files: {missing[:3]} (total {len(missing)})")

    out_dir = os.path.dirname(sketch_path)
    ensure_dir(out_dir)
    tmp_dir = os.path.join(out_dir, "sketch_tmp")
    ensure_dir(tmp_dir)
    merged_path = token_files_and_merge(
        FOR_GEN_EVAL_SKETCH,
        files,
        tmp_dir,
        keep_cnt=keep_cnt,
    )
    shutil.move(merged_path, sketch_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return sketch_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--networks", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--sketch_keep_cnt", type=int, required=True)
    parser.add_argument("--gen_keep_cnt_list", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--sketch_root",
        type=str,
        default=None,
        help="Root dir for sketch.json (defaults to save_root).",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Eval_result",
    )
    parser.add_argument("--hw_kv_aligner_path", type=str, default=None)
    parser.add_argument("--hardware_embedding_path", type=str, default=None)
    parser.add_argument("--hw_kv_num_slots", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--regen_sketch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip_measure_if_exists",
        action="store_true",
        help="Skip measure_programs when measured.json exists.",
    )
    args = parser.parse_args()

    common.register_data_path(args.target)
    target = tvm.target.Target(resolve_target_string(args.target))
    if common.NETWORK_INFO_FOLDER is None:
        raise RuntimeError("NETWORK_INFO_FOLDER is not set; call register_data_path first.")

    networks = parse_list(args.networks)
    methods = parse_list(args.methods)
    gen_keep_cnt_list = [int(x) for x in parse_list(args.gen_keep_cnt_list)]
    if not gen_keep_cnt_list:
        raise RuntimeError("--gen_keep_cnt_list is empty")
    gen_keep_cnt_max = max(gen_keep_cnt_list)
    allowed_methods = {"nobucket_nokv", "bucket", "bucket_kv"}
    unknown = [m for m in methods if m not in allowed_methods]
    if unknown:
        raise RuntimeError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed_methods)}")

    holdout_map = common.hold_out_task_files(target)
    missing = [n for n in networks if n not in holdout_map]
    if missing:
        raise RuntimeError(f"Unknown networks (not in common.py hold_out_task_files): {missing}")

    run_log = os.path.join(args.save_root, "run.log")
    ensure_dir(args.save_root)
    sketch_root = args.sketch_root or args.save_root

    results_csv = os.path.join(args.save_root, "results.csv")
    csv_exists = os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "network",
                "target",
                "method",
                "keep_cnt",
                "overall_latency_ms",
                "record_path",
                "gen_path",
                "sketch_path",
                "notes",
            ],
        )
        if not csv_exists:
            writer.writeheader()

        for network in networks:
            task_pkl = holdout_map[network]
            if not os.path.exists(task_pkl):
                # Fallback: dump all network_info for target if missing
                cmd = [sys.executable, "gen/dump_network_info.py", "--target", args.target]
                run_cmd(cmd, run_log)
                if not os.path.exists(task_pkl):
                    raise RuntimeError(f"Task file still missing after dump: {task_pkl}")

            sketch_dir = os.path.join(sketch_root, args.target, network, f"keep_{args.sketch_keep_cnt}")
            sketch_path = os.path.join(sketch_dir, "sketch.json")
            if os.path.exists(sketch_path) and not args.regen_sketch:
                msg = f"[SKIP] sketch exists, skip make_dataset: {sketch_path}"
                print(msg)
                append_run_log(run_log, msg)
            sketch_path = build_sketch(task_pkl, args.sketch_keep_cnt, sketch_path, args.regen_sketch)

            for method in methods:
                method_dir = os.path.join(
                    args.save_root,
                    args.target,
                    network,
                    method,
                    f"keep_{gen_keep_cnt_max}",
                )
                ensure_dir(method_dir)
                method_sketch_path = os.path.join(method_dir, "sketch.json")
                if not os.path.exists(method_sketch_path):
                    shutil.copyfile(sketch_path, method_sketch_path)

                gen_path = os.path.join(method_dir, "gen.json")
                measured_path = os.path.join(method_dir, "measured.json")

                gen_exists = os.path.exists(gen_path)
                if gen_exists and is_empty_file(gen_path):
                    msg = f"[WARN] gen.json is empty, will regenerate: {gen_path}"
                    print(msg)
                    append_run_log(run_log, msg)
                    gen_exists = False
                if args.force or not gen_exists:
                    cmd = [
                        sys.executable,
                        "gen/gen_state_debug_kv.py",
                        "--model_path",
                        args.model_path,
                        "--tokenizer_path",
                        args.tokenizer_path,
                        "--sketch_path",
                        method_sketch_path,
                        "--save_path",
                        gen_path,
                        "--target",
                        args.target,
                        "--keep_cnt",
                        str(gen_keep_cnt_max),
                    ]
                    if method in ("bucket", "bucket_kv"):
                        cmd.append("--use_bucket")
                    if method == "bucket_kv":
                        if not args.hw_kv_aligner_path or not args.hardware_embedding_path:
                            raise RuntimeError("bucket_kv requires --hw_kv_aligner_path and --hardware_embedding_path")
                        cmd.extend(
                            [
                                "--use_hw_kv",
                                "--hw_kv_mode",
                                "real",
                                "--hw_kv_aligner_path",
                                args.hw_kv_aligner_path,
                                "--hardware_embedding_path",
                                args.hardware_embedding_path,
                                "--hw_kv_num_slots",
                                str(args.hw_kv_num_slots),
                                "--pos_compensate",
                            ]
                        )
                    run_cmd(cmd, run_log)

                if not (args.skip_measure_if_exists and os.path.exists(measured_path)):
                    cmd = [
                        sys.executable,
                        "gen/measure_programs.py",
                        "--batch-size",
                        str(args.batch_size),
                        "--target",
                        args.target,
                        "--to-measure-path",
                        gen_path,
                        "--measured-path",
                        measured_path,
                    ]
                    run_cmd(cmd, run_log)

                total_rec, ok_rec = count_measure_records(measured_path)
                for keep_cnt in gen_keep_cnt_list:
                    overall_latency_ms, missing, covered = compute_overall_latency_ms_from_gen(
                        task_pkl,
                        gen_path,
                        measured_path,
                        keep_cnt,
                    )
                    notes = f"records={total_rec} ok={ok_rec}"
                    if missing > 0:
                        notes += f" missing_workloads={missing}"
                        overall_latency_ms = None
                    writer.writerow(
                        {
                            "network": network,
                            "target": args.target,
                            "method": method,
                            "keep_cnt": keep_cnt,
                            "overall_latency_ms": "" if overall_latency_ms is None else f"{overall_latency_ms:.6f}",
                            "record_path": measured_path,
                            "gen_path": gen_path,
                            "sketch_path": method_sketch_path,
                            "notes": notes,
                        }
                    )
                    f_csv.flush()
                time.sleep(0.1)


if __name__ == "__main__":
    main()
