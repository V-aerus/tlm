#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional, Tuple


def find_repo_root(start_dir: str, max_up: int = 6) -> Tuple[Optional[str], Optional[str]]:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        utils_path = os.path.join(cur, "tlm_dataset", "gen", "utils.json")
        gen_dir = os.path.join(cur, "gen")
        if os.path.exists(utils_path) and os.path.isdir(gen_dir):
            return cur, utils_path
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None, None


def read_run_root_from_paths_sh(paths_sh: str) -> Optional[str]:
    if not paths_sh or not os.path.exists(paths_sh):
        return None
    run_root = None
    with open(paths_sh, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export RUN_ROOT="):
                run_root = line.split("=", 1)[1].strip().strip('"').strip("'")
                run_root = os.path.expandvars(run_root)
                break
    return run_root


def build_measured_path(run_root: str, hw: str, iter_idx: int, mode: str) -> str:
    iter_name = f"iter{iter_idx:02d}"
    filename = "base_bucketkv.json" if mode == "base" else "kv_lora.json"
    return os.path.join(run_root, hw, iter_name, "measure", filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append measured file to tlm_dataset/gen/utils.json.")
    parser.add_argument("--hardware", required=True, help="Hardware key in utils.json (e.g., 4090, v100).")
    parser.add_argument("--iter", type=int, required=True, help="Iteration index (e.g., 0,1,2,3).")
    parser.add_argument("--mode", choices=("base", "kv_lora"), default="base", help="Measured file type.")
    parser.add_argument("--run-tag", default=None, help="Run tag under gen_data/edge_runs/<RUN_TAG>.")
    parser.add_argument("--run-root", default=None, help="Explicit run root (overrides --run-tag).")
    parser.add_argument("--utils-path", default=None, help="Explicit utils.json path.")
    parser.add_argument("--paths-sh", default=None, help="Optional paths.sh (used to read RUN_ROOT).")
    parser.add_argument("--measured-path", default=None, help="Explicit measured file path (overrides run-root/tag).")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned changes; do not write.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root, utils_path_default = find_repo_root(script_dir)

    utils_path = args.utils_path or utils_path_default
    if not utils_path:
        print("utils.json not found; pass --utils-path explicitly.")
        sys.exit(1)

    if args.measured_path:
        measured_path = args.measured_path
        if os.path.isdir(measured_path):
            filename = "base_bucketkv.json" if args.mode == "base" else "kv_lora.json"
            measured_path = os.path.join(measured_path, filename)
    else:
        run_root = args.run_root
        if run_root is None:
            paths_sh = args.paths_sh or os.environ.get("EDGE_PATHS_SH")
            run_root = read_run_root_from_paths_sh(paths_sh) if paths_sh else None
        if run_root is None:
            run_tag = args.run_tag or os.environ.get("RUN_TAG")
            if not run_tag:
                print("No run root or run tag found. Provide --run-root or --run-tag.")
                sys.exit(1)
            if not repo_root:
                print("Repo root not found; pass --run-root explicitly.")
                sys.exit(1)
            run_root = os.path.join(repo_root, "tlm_dataset", "gen", "gen_data", "edge_runs", run_tag)
        measured_path = build_measured_path(run_root, args.hardware, args.iter, args.mode)

    if not os.path.exists(measured_path):
        print(f"Measured file not found: {measured_path}")
        sys.exit(1)

    with open(utils_path, "r", encoding="utf-8") as f:
        utils = json.load(f)

    hw_key = args.hardware
    utils.setdefault(hw_key, {}).setdefault("measure_records", [])
    if args.mode == "base":
        utils.setdefault(hw_key, {}).setdefault("measure_records_base", [])
        target_list = utils[hw_key]["measure_records_base"]
        list_name = "measure_records_base"
    else:
        utils.setdefault(hw_key, {}).setdefault("measure_records_kv_lora", [])
        target_list = utils[hw_key]["measure_records_kv_lora"]
        list_name = "measure_records_kv_lora"

    before_count = len(target_list)
    if measured_path in target_list:
        print("Already present in utils.json:")
        print(f"  utils_path: {utils_path}")
        print(f"  hardware: {hw_key}")
        print(f"  measured_path: {measured_path}")
        print(f"  list: {list_name}")
        print(f"  count: {before_count}")
        return

    if measured_path not in utils[hw_key]["measure_records"]:
        utils[hw_key]["measure_records"].append(measured_path)
    if measured_path not in target_list:
        target_list.append(measured_path)
    after_count = len(target_list)

    print("Planned update:")
    print(f"  utils_path: {utils_path}")
    print(f"  hardware: {hw_key}")
    print(f"  measured_path: {measured_path}")
    print(f"  list: {list_name}")
    print(f"  count: {before_count} -> {after_count}")

    if args.dry_run:
        print("Dry run: no changes written.")
        return

    with open(utils_path, "w", encoding="utf-8") as f:
        json.dump(utils, f, indent=2)
    print("Updated utils.json successfully.")


if __name__ == "__main__":
    main()
