#!/usr/bin/env python3
"""Dump programs for tasks under Orin side-workflow.

This side script reads all_tasks.pkl from a specified network_info folder and
writes to_measure_programs to a specified output folder.
"""

import argparse
import gc
import os
import pickle
import time

from tvm import auto_scheduler
from tqdm import tqdm


def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', "")
    x = x.replace("'", "")
    return x


def get_to_measure_filename(to_measure_folder: str, task) -> str:
    task_key = (task.workload_key, str(task.target.kind))
    return f"{to_measure_folder}/{clean_name(task_key)}.json"


def load_and_register_tasks(network_info_folder: str):
    all_tasks_path = os.path.join(network_info_folder, "all_tasks.pkl")
    if not os.path.exists(all_tasks_path):
        raise FileNotFoundError(f"all_tasks.pkl not found: {all_tasks_path}")

    tasks = pickle.load(open(all_tasks_path, "rb"))
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )
    return tasks


def dump_program(task, size, to_measure_folder: str, max_retry_iter=5):
    filename = get_to_measure_filename(to_measure_folder, task)
    if os.path.exists(filename):
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    policy = auto_scheduler.SketchPolicy(
        task,
        params={
            "evolutionary_search_num_iters": 1,
            "evolutionary_search_population": min(size, 2560),
            "max_innermost_split_factor": 1024,
        },
        verbose=0,
    )

    all_state_str_set = set()
    all_state_list = []
    retry_ct = 0

    while len(all_state_list) < size and retry_ct < max_retry_iter:
        states = policy.sample_initial_population()
        ct_before = len(all_state_list)

        for s in states:
            str_s = str(s)
            if str_s not in all_state_str_set:
                all_state_str_set.add(str_s)
                all_state_list.append(s)

        if len(all_state_list) >= size:
            break

        ct_after = len(all_state_list)
        if ct_before == ct_after:
            retry_ct += 1
        else:
            retry_ct = 0

    measure_inputs = []
    measure_results = []
    for state in all_state_list:
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    auto_scheduler.save_records(filename, measure_inputs, measure_results)


def resolve_data_root(data_root_arg: str) -> str:
    if data_root_arg:
        return data_root_arg
    env_data_root = os.environ.get("DATA_ROOT", "").strip()
    if env_data_root:
        return env_data_root
    tlm_root = os.environ.get("TLM_ROOT", "").strip()
    if tlm_root:
        return os.path.join(tlm_root, "tlm_dataset", "gen")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "tlm_dataset", "gen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-info-dir", type=str, default="")
    parser.add_argument("--to-measure-dir", type=str, default="")
    parser.add_argument("--model", type=str, default="orin")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--max-retry-iter", type=int, default=5)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    network_info_dir = args.network_info_dir or os.path.join(data_root, "dataset", "network_info", args.model)
    to_measure_dir = args.to_measure_dir or os.path.join(data_root, "dataset", "to_measure_programs", args.model)

    print("[NETWORK_INFO]", network_info_dir)
    print("[TO_MEASURE]", to_measure_dir)

    tasks = load_and_register_tasks(network_info_dir)

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)

    for task in tqdm(tasks[start_idx:end_idx]):
        dump_program(task, size=args.size, to_measure_folder=to_measure_dir, max_retry_iter=args.max_retry_iter)
        gc.collect()

    print(f"[DONE] dumped programs for tasks[{start_idx}:{end_idx}] -> {to_measure_dir}")


if __name__ == "__main__":
    main()
