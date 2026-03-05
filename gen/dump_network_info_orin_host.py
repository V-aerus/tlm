#!/usr/bin/env python3
"""Dump network_info for Orin-like targets with explicit host target.

This script is a side-copy for Orin workflow and does not modify the default
pipeline scripts. It builds TVM target by:
  Target(cuda_target, host=host_target)
so we can preserve host information without requiring a C++ target tag.
"""

import argparse
import gc
import glob
import os
import pickle
from typing import Tuple

import tvm
from tvm import auto_scheduler, relay
from tvm.meta_schedule.testing.dataset_collect_models import build_network_keys
from tvm.meta_schedule.testing.relay_workload import get_network
from tqdm import tqdm


def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', "")
    x = x.replace("'", "")
    return x


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


def make_paths(data_root: str, model: str, network_info_dir: str) -> str:
    if network_info_dir:
        return network_info_dir
    return os.path.join(data_root, "dataset", "network_info", model)


def get_relay_ir_filename(network_info_folder: str, network_key) -> str:
    return f"{network_info_folder}/{clean_name(network_key)}.relay.pkl"


def get_task_info_filename(network_info_folder: str, network_key, target_kind: str) -> str:
    network_task_key = (network_key,) + (target_kind,)
    return f"{network_info_folder}/{clean_name(network_task_key)}.task.pkl"


def build_target(cuda_target: str, host_target: str) -> tvm.target.Target:
    if host_target:
        return tvm.target.Target(cuda_target, host=host_target)
    return tvm.target.Target(cuda_target)


def dump_network(network_info_folder: str, network_key, target: tvm.target.Target, hardware_params):
    relay_ir_filename = get_relay_ir_filename(network_info_folder, network_key)
    task_info_filename = get_task_info_filename(network_info_folder, network_key, str(target.kind))

    if os.path.exists(task_info_filename):
        return

    mod, params, inputs = get_network(*network_key)

    if not os.path.exists(relay_ir_filename):
        print(f"Dump relay ir for {network_key}...")
        mod_json = tvm.ir.save_json(mod)
        params_bytes = relay.save_param_dict(params)
        pickle.dump((mod_json, len(params_bytes), inputs), open(relay_ir_filename, "wb"))

    if not os.path.exists(task_info_filename):
        print(f"Dump task info for {(network_key, target)}...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"],
            params,
            target,
            hardware_params=hardware_params,
        )
        pickle.dump((tasks, task_weights), open(task_info_filename, "wb"))


def get_all_tasks(network_info_folder: str) -> Tuple[list, int]:
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    filenames = glob.glob(f"{network_info_folder}/*.task.pkl")
    filenames.sort()

    for filename in tqdm(filenames):
        tasks, _task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            task_key = (t.workload_key, str(t.target.kind))
            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    return all_tasks, duplication


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-target",
        type=str,
        default=(
            "cuda -keys=cuda,gpu -arch=sm_87 "
            "-max_num_threads=1024 "
            "-max_shared_memory_per_block=49152 "
            "-max_threads_per_block=1024 "
            "-registers_per_block=65536 "
            "-thread_warp_size=32"
        ),
    )
    parser.add_argument(
        "--host-target",
        type=str,
        default="llvm -keys=arm_cpu,cpu -mcpu=cortex-a78 -mtriple=aarch64-linux-gnu -num-cores=12",
    )
    parser.add_argument("--model", type=str, default="orin")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--network-info-dir", type=str, default="")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    network_info_folder = make_paths(data_root, args.model, args.network_info_dir)
    os.makedirs(network_info_folder, exist_ok=True)

    target = build_target(args.cuda_target, args.host_target)
    print("[TARGET]", target)
    print("[HOST]", target.host)
    print("[NETWORK_INFO]", network_info_folder)

    if target.kind.name == "llvm":
        hardware_params = auto_scheduler.HardwareParams(target=target)
    elif target.kind.name == "cuda":
        max_shared_memory = target.attrs.get("max_shared_memory_per_block", 49152)
        max_threads = target.attrs.get("max_threads_per_block", 1024)
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(max_shared_memory),
            max_threads_per_block=int(max_threads),
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    else:
        raise NotImplementedError(f"Unsupported target {target}")

    network_keys = build_network_keys()
    for key in tqdm(network_keys):
        dump_network(network_info_folder, key, target, hardware_params)
        gc.collect()

    tasks, duplication = get_all_tasks(network_info_folder)
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    all_tasks_path = os.path.join(network_info_folder, "all_tasks.pkl")
    pickle.dump(tasks, open(all_tasks_path, "wb"))
    print(f"[DONE] all_tasks={len(tasks)} duplication={duplication} -> {all_tasks_path}")


if __name__ == "__main__":
    main()
