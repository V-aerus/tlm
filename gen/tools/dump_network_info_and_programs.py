#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import subprocess
import sys
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def log(msg: str, log_path: str) -> None:
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")


def count_task_pkls(folder: str) -> int:
    return len(glob.glob(os.path.join(folder, "*.task.pkl")))


def count_jsons(folder: str) -> int:
    return len(glob.glob(os.path.join(folder, "*.json")))


def run_cmd(cmd, log_path: str) -> int:
    log(f"[CMD] {' '.join(cmd)}", log_path)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        log(f"[WARN] command failed with code {proc.returncode}", log_path)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--max_retries", type=int, default=0)
    parser.add_argument("--sleep_sec", type=int, default=20)
    parser.add_argument("--min_task_pkls", type=int, default=0)
    parser.add_argument("--dump_programs_size", type=int, default=1000)
    parser.add_argument("--programs_max_retries", type=int, default=0)
    parser.add_argument("--programs_sleep_sec", type=int, default=20)
    parser.add_argument("--min_to_measure_jsons", type=int, default=0)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    sys.path.append(os.path.join(REPO_ROOT, "gen"))
    import common  # pylint: disable=import-error

    common.register_data_path(args.target)
    network_info_dir = common.NETWORK_INFO_FOLDER
    if not network_info_dir:
        raise RuntimeError("NETWORK_INFO_FOLDER is not set")
    os.makedirs(network_info_dir, exist_ok=True)
    all_tasks_path = os.path.join(network_info_dir, "all_tasks.pkl")
    to_measure_dir = common.TO_MEASURE_PROGRAM_FOLDER

    attempt = 0
    while True:
        attempt += 1
        before = count_task_pkls(network_info_dir)
        log(f"[INFO] attempt={attempt} task_pkl_count(before)={before}", args.log_path)
        ret = run_cmd(
            [sys.executable, "gen/dump_network_info.py", "--target", args.target],
            args.log_path,
        )
        after = count_task_pkls(network_info_dir)
        log(f"[INFO] task_pkl_count(after)={after}", args.log_path)

        done = ret == 0 and os.path.exists(all_tasks_path)
        if done and args.min_task_pkls > 0 and after < args.min_task_pkls:
            log(
                f"[WARN] task_pkl_count {after} < min_task_pkls {args.min_task_pkls}, retrying",
                args.log_path,
            )
            done = False

        if done:
            log("[INFO] dump_network_info completed", args.log_path)
            break

        if args.max_retries and attempt >= args.max_retries:
            raise RuntimeError("dump_network_info did not complete within max_retries")

        log(f"[INFO] sleep {args.sleep_sec}s before retry", args.log_path)
        time.sleep(args.sleep_sec)

    expected_tasks = 0
    if os.path.exists(all_tasks_path):
        try:
            expected_tasks = len(pickle.load(open(all_tasks_path, "rb")))
        except Exception:
            expected_tasks = 0

    log("[INFO] start dump_programs", args.log_path)
    attempt = 0
    while True:
        attempt += 1
        before = count_jsons(to_measure_dir)
        log(f"[INFO] attempt={attempt} to_measure_count(before)={before}", args.log_path)
        ret = run_cmd(
            [
                sys.executable,
                "gen/dump_programs.py",
                "--target",
                args.target,
                "--size",
                str(args.dump_programs_size),
            ],
            args.log_path,
        )
        after = count_jsons(to_measure_dir)
        log(f"[INFO] to_measure_count(after)={after}", args.log_path)

        done = ret == 0
        if args.min_to_measure_jsons > 0 and after < args.min_to_measure_jsons:
            log(
                f"[WARN] to_measure_count {after} < min_to_measure_jsons {args.min_to_measure_jsons}",
                args.log_path,
            )
            done = False
        if expected_tasks > 0 and after < expected_tasks:
            log(
                f"[WARN] to_measure_count {after} < expected_tasks {expected_tasks}",
                args.log_path,
            )
            done = False

        if done:
            log("[INFO] dump_programs completed", args.log_path)
            break

        if args.programs_max_retries and attempt >= args.programs_max_retries:
            raise RuntimeError("dump_programs did not complete within programs_max_retries")

        log(f"[INFO] sleep {args.programs_sleep_sec}s before retry", args.log_path)
        time.sleep(args.programs_sleep_sec)

    log("[INFO] done", args.log_path)


if __name__ == "__main__":
    main()
