#!/usr/bin/env python3
"""
预计算每个硬件 target 字符串的分段 embedding，供几何监督使用。

思路：
- Teacher prompt 保留完整 target 字符串（cuda/llvm + 8 个整数）。
- 按字符范围切出 ARCH / MEM / CONS / HOST 四段（当前实现：ARCH=整数段之前，CONS=首个 8 整数段，MEM/HOST 默认 inactive）。
- 取每段命中的 token 的 embedding（均值）作为目标，存为 .pt。

输出示例：
{
  "nvidia/nvidia-a40": {
    "arch": {"vec": tensor(D), "active": True},
    "mem":  {"vec": tensor(D), "active": False},
    "cons": {"vec": tensor(D), "active": True},
    "host": {"vec": tensor(D), "active": False},
  },
  ...
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

# 默认的 canonical target 映射（不含 DSL/shape，只含硬件参数 + 8 整数 + Jetson host）
DEFAULT_TARGET_MAP = {
    "nvidia/nvidia-a40": "cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32",
    "nvidia/nvidia-v100": "cuda -keys=cuda,gpu -arch=sm_70 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32",
    "nvidia/jetson-agx-xavier": "cuda -keys=cuda,gpu -arch=sm_72 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32 -1 16 64 49152 12345678 1024 8 32 llvm -keys=arm_cpu,cpu -mcpu=carmel -mtriple=aarch64-linux-gnu -num-cores=8",
    "aws/cpu/c5.18xlarge": "llvm -keys=cpu -mcpu=skylake-avx512 -model=xeon 36 64 64 0 0 0 0 0",
}


def find_cons_span_after_anchor(text: str, anchor: str) -> Tuple[int, int]:
    pos = text.find(anchor)
    if pos == -1:
        return -1, -1
    sub = text[pos:]
    m = re.search(r"(-?\d+(?:\s+-?\d+){7})", sub)
    if not m:
        return -1, -1
    start = pos + m.start(1)
    end = pos + m.end(1)
    return start, end


def split_gpu(text: str) -> Dict[str, Tuple[int, int, bool]]:
    cons_start, cons_end = find_cons_span_after_anchor(text, "-thread_warp_size")
    if cons_start == -1:
        L = len(text)
        return {"arch": (0, L, True), "mem": (0, 0, False), "cons": (0, 0, False), "host": (0, 0, False)}

    mem_start = text.find("-max_shared_memory_per_block")
    if mem_start != -1 and mem_start < cons_start:
        mem_end = cons_start
        mem_active = True
        arch_end = mem_start
    else:
        mem_start = mem_end = 0
        mem_active = False
        arch_end = cons_start

    arch_start = text.find("cuda -keys=cuda,gpu")
    if arch_start == -1:
        arch_start = 0

    return {
        "arch": (arch_start, arch_end, True),
        "mem": (mem_start, mem_end, mem_active),
        "cons": (cons_start, cons_end, True),
        "host": (0, 0, False),
    }


def split_jetson(text: str) -> Dict[str, Tuple[int, int, bool]]:
    cons_start, cons_end = find_cons_span_after_anchor(text, "-thread_warp_size")

    host_start = text.find("llvm -keys=arm_cpu,cpu")
    if host_start != -1:
        host_end = len(text)
        host_active = True
    else:
        host_start = host_end = 0
        host_active = False

    mem_start = text.find("-max_shared_memory_per_block")
    if mem_start != -1 and mem_start < cons_start:
        mem_end = cons_start
        mem_active = True
        arch_end = mem_start
    else:
        mem_start = mem_end = 0
        mem_active = False
        arch_end = cons_start

    arch_start = text.find("cuda -keys=cuda,gpu")
    if arch_start == -1:
        arch_start = 0

    return {
        "arch": (arch_start, arch_end, True),
        "mem": (mem_start, mem_end, mem_active),
        "cons": (cons_start, cons_end, cons_start != -1),
        "host": (host_start, host_end, host_active),
    }


def split_cpu(text: str) -> Dict[str, Tuple[int, int, bool]]:
    arch_start = text.find("llvm -keys=cpu")
    if arch_start == -1:
        arch_start = 0
    model_pos = text.find("-model=xeon")
    arch_end = model_pos + len("-model=xeon") if model_pos != -1 else len(text)
    cons_start, cons_end = find_cons_span_after_anchor(text, "-model=xeon")
    cons_active = cons_start != -1
    return {
        "arch": (arch_start, arch_end, True),
        "mem": (0, 0, False),
        "cons": (cons_start, cons_end, cons_active),
        "host": (0, 0, False),
    }


def split_segments(hw_name: str, text: str) -> Dict[str, Tuple[int, int, bool]]:
    if "cuda -keys=cuda,gpu" in text:
        if "llvm -keys=arm_cpu,cpu" in text:
            return split_jetson(text)
        return split_gpu(text)
    if "llvm -keys=cpu" in text and "arm_cpu" not in text:
        return split_cpu(text)
    L = len(text)
    return {"arch": (0, L, True), "mem": (0, 0, False), "cons": (0, 0, False), "host": (0, 0, False)}


def collect_segment_ids(tokenizer, text: str, start: int, end: int) -> List[int]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    seg_ids = []
    for tid, (s, e) in zip(ids, offsets):
        mid = (s + e) / 2.0
        if start <= mid <= end:
            seg_ids.append(tid)
    return seg_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--target_map", default=None, help="JSON 文件：hardware_name -> target string；默认使用内置 canonical 映射")
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    embed = model.get_input_embeddings()

    if args.target_map:
        targets = json.load(open(args.target_map))
    else:
        targets = DEFAULT_TARGET_MAP

    result = {}
    for hw_name, target_text in targets.items():
        seg_ranges = split_segments(hw_name, target_text)
        seg_out = {}
        for seg_name in ["arch", "mem", "cons", "host"]:
            s, e, active = seg_ranges.get(seg_name, (0, 0, False))
            if not active or s == e:
                seg_out[seg_name] = {"vec": torch.zeros(embed.embedding_dim), "active": False}
                continue
            ids = collect_segment_ids(tok, target_text, s, e)
            if not ids:
                seg_out[seg_name] = {"vec": torch.zeros(embed.embedding_dim), "active": False}
                continue
            with torch.no_grad():
                vec = embed(torch.tensor(ids, dtype=torch.long)).mean(dim=0)
            seg_out[seg_name] = {"vec": vec, "active": True}
        result[hw_name] = seg_out

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"Saved targets for {len(result)} hardware to {out_path}")


if __name__ == "__main__":
    main()
