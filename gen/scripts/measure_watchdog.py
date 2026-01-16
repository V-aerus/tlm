#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional, Set, Tuple

'''

  export RUN_ROOT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart
  export GEN_DIR=$RUN_ROOT/4090/iter00/eval_ansor_gen
  export MEASURE_DIR=$RUN_ROOT/4090/iter00/eval_ansor_measure
  mkdir -p "$MEASURE_DIR"
  
  CUDA_VISIBLE_DEVICES=3 python /home/hehangshuai/workspace/tlm/gen/scripts/measure_watchdog.py \
    --repo-root /home/hehangshuai/workspace/tlm \
    --target 4090 \
    --batch-size 64 \
    --job "$GEN_DIR/kv_lora_mix.json=$MEASURE_DIR/kv_lora_mix_measured.json" \
    --job "$GEN_DIR/kv_lora_4090.json=$MEASURE_DIR/kv_lora_4090_measured.json" \
    --job "$GEN_DIR/kv_lora_v100.json=$MEASURE_DIR/kv_lora_v100_measured.json" \
    --job "$GEN_DIR/official.json=$MEASURE_DIR/official_measured.json"

'''
def _key_from_i(obj, mode: str) -> Optional[str]:
    if "i" not in obj:
        return None
    try:
        i = obj["i"]
        workload_repr = i[0][0]
        target_str = i[0][1]
    except Exception:
        return None
    if mode == "workload_key":
        return str(workload_repr)
    if mode == "workload_key_target":
        return json.dumps([workload_repr, target_str], separators=(",", ":"))
    return json.dumps(i, separators=(",", ":"))


def load_unique_i(path: str, mode: str = "i") -> Set[str]:
    items: Set[str] = set()
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = _key_from_i(obj, mode)
                if key is None:
                    continue
                items.add(key)
            except Exception:
                continue
    return items


def format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def run_once(cmd, log_path, progress_interval, total_unique, measured_path, match_mode: str, prefix: str = ""):
    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n{prefix}[RUN] {datetime.now().isoformat()} cmd={' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

        last_check = start
        last_measured = len(load_unique_i(measured_path, match_mode))
        while True:
            rc = proc.poll()
            now = time.time()
            if progress_interval > 0 and now - last_check >= progress_interval:
                measured_now = len(load_unique_i(measured_path, match_mode))
                delta = measured_now - last_measured
                dt = now - last_check
                rate = delta / dt if dt > 0 else 0.0
                remaining = max(total_unique - measured_now, 0)
                eta = format_seconds(remaining / rate) if rate > 0 else "unknown"
                msg = (
                    f"{prefix}[PROGRESS] measured={measured_now}/{total_unique} "
                    f"delta={delta} rate={rate:.2f}/s remaining={remaining} eta={eta}"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
                last_check = now
                last_measured = measured_now
            if rc is not None:
                break
            time.sleep(1)
    return rc, time.time() - start


def find_measure_script(start_dir: str, max_up: int = 4) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        cand_same = os.path.join(cur, "measure_programs.py")
        if os.path.exists(cand_same):
            return cand_same
        cand_gen = os.path.join(cur, "gen", "measure_programs.py")
        if os.path.exists(cand_gen):
            return cand_gen
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def parse_jobs(args) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    if args.job:
        for item in args.job:
            if "=" in item:
                to_path, measured_path = item.split("=", 1)
            elif "," in item:
                to_path, measured_path = item.split(",", 1)
            else:
                raise ValueError(
                    "Invalid --job format. Use --job to_path=measured_path (or to_path,measured_path)."
                )
            jobs.append((to_path.strip(), measured_path.strip()))
    else:
        if not args.to_measure_path or not args.measured_path:
            raise ValueError("Provide --to-measure-path and --measured-path, or use repeated --job.")
        jobs.append((args.to_measure_path, args.measured_path))
    return jobs


def resolve_log_path(base_log_path: Optional[str], measured_path: str, job_idx: int, total_jobs: int, stamp: str) -> str:
    if base_log_path:
        if total_jobs <= 1:
            return base_log_path
        root, ext = os.path.splitext(base_log_path)
        ext = ext or ".log"
        return f"{root}_job{job_idx + 1}{ext}"
    default_dir = os.path.dirname(measured_path)
    suffix = f"_job{job_idx + 1}" if total_jobs > 1 else ""
    return os.path.join(default_dir, f"measure_watchdog_{stamp}{suffix}.log")


def run_job(args, measure_script: str, job_idx: int, total_jobs: int, to_measure_path: str, measured_path: str, log_path: str) -> None:
    prefix = f"[JOB {job_idx + 1}/{total_jobs}] "
    if not os.path.exists(to_measure_path):
        raise FileNotFoundError(f"to_measure_path not found: {to_measure_path}")
    os.makedirs(os.path.dirname(measured_path), exist_ok=True)

    to_set = load_unique_i(to_measure_path, args.match_mode)
    total_unique = len(to_set)
    if total_unique == 0:
        print(f"{prefix}No valid measure inputs found. Skip.")
        return

    restarts = 0
    measured_prev = len(load_unique_i(measured_path, args.match_mode))
    while True:
        measured_now = len(load_unique_i(measured_path, args.match_mode))
        measured_set = load_unique_i(measured_path, args.match_mode)
        overlap = len(to_set & measured_set)
        remaining = len(to_set - measured_set)
        print(f"{prefix}[STATUS] measured={measured_now}/{total_unique} remaining={remaining}")
        if measured_now > 0 and overlap == 0 and total_unique > 0:
            print(f"{prefix}[WARN] zero overlap between to_measure and measured under match_mode={args.match_mode}")
        if remaining == 0:
            print(f"{prefix}All measurements completed.")
            return

        cmd = [sys.executable, measure_script,
               "--batch-size", str(args.batch_size),
               "--target", args.target,
               "--to-measure-path", to_measure_path,
               "--measured-path", measured_path]
        if args.target_host:
            cmd.extend(["--target-host", args.target_host])

        rc, elapsed = run_once(cmd, log_path, args.progress_interval, total_unique, measured_path, args.match_mode, prefix=prefix)
        measured_after = len(load_unique_i(measured_path, args.match_mode))
        delta = measured_after - measured_prev
        measured_prev = measured_after
        print(f"{prefix}[RUN DONE] rc={rc} elapsed={format_seconds(elapsed)} delta_measured={delta}")

        remaining = len(to_set - load_unique_i(measured_path, args.match_mode))
        if remaining == 0:
            print(f"{prefix}All measurements completed.")
            return

        if rc != 0:
            restarts += 1
            if args.max_restarts and restarts > args.max_restarts:
                raise RuntimeError(f"Exceeded max restarts ({args.max_restarts}).")
            print(f"{prefix}[WARN] measure_programs exited with rc={rc}, restarting in {args.sleep_secs}s")
        else:
            print(f"{prefix}[INFO] Still remaining={remaining}, restarting in {args.sleep_secs}s")

        time.sleep(args.sleep_secs)


def main():
    parser = argparse.ArgumentParser(description="Watchdog for gen/measure_programs.py with resume support.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--to-measure-path", default=None, help="Single job to-measure jsonl path.")
    parser.add_argument("--measured-path", default=None, help="Single job measured jsonl path.")
    parser.add_argument("--job", action="append", help="Repeatable job: to_path=measured_path (or to_path,measured_path).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--match-mode",
        default="i",
        choices=("i", "workload_key", "workload_key_target"),
        help="How to match measured entries to to-measure entries.",
    )
    parser.add_argument("--target-host", default=None)
    parser.add_argument("--repo-root", default=None, help="Repo root (default: inferred from this script).")
    parser.add_argument("--sleep-secs", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    measure_script = None
    if args.repo_root:
        candidate = os.path.join(args.repo_root, "gen", "measure_programs.py")
        if os.path.exists(candidate):
            measure_script = candidate
    if measure_script is None:
        measure_script = find_measure_script(script_dir)
    if measure_script is None:
        raise FileNotFoundError(
            "measure_programs.py not found. "
            "Pass --repo-root /path/to/tlm or place measure_watchdog.py under tlm/gen or tlm/gen/scripts."
        )

    jobs = parse_jobs(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_jobs = len(jobs)
    for job_idx, (to_measure_path, measured_path) in enumerate(jobs):
        log_path = resolve_log_path(args.log_path, measured_path, job_idx, total_jobs, stamp)
        run_job(args, measure_script, job_idx, total_jobs, to_measure_path, measured_path, log_path)


if __name__ == "__main__":
    main()
