#!/usr/bin/env python3
import argparse
import signal
import time
from typing import List

import torch

GIB = 1024 ** 3
MIB = 1024 ** 2
_RUNNING = True


def _to_gib(num_bytes: int) -> float:
    return num_bytes / GIB


def _stop_handler(_signum, _frame):
    global _RUNNING
    _RUNNING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reserve GPU memory without launching compute kernels."
    )
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=30.0,
        help="Target reserved memory in GiB.",
    )
    parser.add_argument(
        "--keep-free-gb",
        type=float,
        default=12.0,
        help="Always keep at least this many GiB free.",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help="Allocation chunk size in MiB.",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=30.0,
        help="Print memory heartbeat every N seconds.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error when reserve target cannot be fully satisfied.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in current environment.")

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    torch.cuda.set_device(args.cuda_device)
    device = torch.device(f"cuda:{args.cuda_device}")
    props = torch.cuda.get_device_properties(device)

    free0, total0 = torch.cuda.mem_get_info(device)
    keep_free_bytes = int(args.keep_free_gb * GIB)
    reserve_target_bytes = int(args.reserve_gb * GIB)
    max_reservable_bytes = max(0, free0 - keep_free_bytes)
    reserve_plan_bytes = min(reserve_target_bytes, max_reservable_bytes)

    print(
        f"[GPU] device={args.cuda_device} name={props.name} "
        f"total={_to_gib(total0):.2f}GiB free={_to_gib(free0):.2f}GiB"
    )
    print(
        f"[PLAN] target={args.reserve_gb:.2f}GiB keep_free={args.keep_free_gb:.2f}GiB "
        f"max_reservable={_to_gib(max_reservable_bytes):.2f}GiB "
        f"planned={_to_gib(reserve_plan_bytes):.2f}GiB"
    )

    if args.strict and reserve_plan_bytes < reserve_target_bytes:
        raise SystemExit(
            f"Strict mode: cannot reserve {args.reserve_gb:.2f}GiB "
            f"(only {_to_gib(reserve_plan_bytes):.2f}GiB allowed)."
        )
    if reserve_plan_bytes <= 0:
        raise SystemExit("No memory can be reserved with current keep-free setting.")

    chunk_bytes = max(1, args.chunk_mb) * MIB
    blocks: List[torch.Tensor] = []
    reserved = 0

    while reserved < reserve_plan_bytes:
        cur = min(chunk_bytes, reserve_plan_bytes - reserved)
        try:
            blocks.append(torch.empty(cur, dtype=torch.uint8, device=device))
            reserved += cur
        except RuntimeError as err:
            print(f"[WARN] allocation stopped early: {err}")
            break

    free1, _ = torch.cuda.mem_get_info(device)
    actual_reserved = free0 - free1
    print(
        f"[READY] reserved={_to_gib(actual_reserved):.2f}GiB "
        f"free_now={_to_gib(free1):.2f}GiB blocks={len(blocks)}"
    )

    if args.strict and actual_reserved < reserve_target_bytes:
        raise SystemExit(
            f"Strict mode: actual reserve {_to_gib(actual_reserved):.2f}GiB "
            f"< target {args.reserve_gb:.2f}GiB."
        )

    while _RUNNING:
        time.sleep(max(0.1, args.heartbeat_sec))
        free_now, _ = torch.cuda.mem_get_info(device)
        print(
            f"[HEARTBEAT] reserved={_to_gib(actual_reserved):.2f}GiB "
            f"free={_to_gib(free_now):.2f}GiB"
        )

    print("[EXIT] Releasing reserved GPU memory.")


if __name__ == "__main__":
    main()
