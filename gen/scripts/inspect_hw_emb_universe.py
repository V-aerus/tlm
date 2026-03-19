#!/usr/bin/env python3
"""Inspect hardware embeddings (v4 universe or arbitrary emb JSON)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen.Embedding.hardware_embedding_generator_v4 import (  # noqa: E402
    EmbeddingV4Generator,
    _default_tvm_config,
)
import torch

from gen.modeling.hw_preprocess import build_preprocess_params, apply_preprocess, summarize_preprocess  # noqa: E402


UNIVERSE_ADDITIONS = [
    # GPU
    "nvidia/geforce-gtx-1060",
    "nvidia/geforce-gtx-950",
    # CPU x86
    "intel/core-i7-10510u",
    "amd/ryzen-7-5800h",
    "intel/xeon-gold-6226",
    "intel/core-i7-12700k",
    "intel/core-i5-12400",
    # ARM
    "raspberry-pi/4b-aarch64",
]

CHECK_NAMES = [
    "nvidia/jetson-orin",
    "raspberry-pi/4b-aarch64",
]

GPU_COS_PAIRS = [
    ("nvidia/rtx-4090", "nvidia/nvidia-v100"),
    ("nvidia/rtx-4090", "nvidia/geforce-gtx-1060"),
    ("nvidia/nvidia-v100", "nvidia/geforce-gtx-950"),
    ("nvidia/jetson-orin", "nvidia/jetson-agx-xavier"),
]


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def _stats(vectors: List[List[float]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    dim = len(vectors[0])
    means = []
    stds = []
    mins = []
    maxs = []
    n = float(len(vectors))
    for i in range(dim):
        vals = [vec[i] for vec in vectors]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std)
        mins.append(min(vals))
        maxs.append(max(vals))
    return means, stds, mins, maxs


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb-json",
        default="",
        help="直接读取 embedding JSON（list 格式），用于 v5/v4 任意版本；提供后将跳过 v4 生成流程",
    )
    parser.add_argument(
        "--v4-json",
        default=str(ROOT / "gen/Embedding/hardware_embeddings_v4.json"),
        help="Path to hardware_embeddings_v4.json (v4 generator input)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Where to write v4 universe json (only used in v4 generator mode)",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=1e-6,
        help="Std threshold to flag near-constant dims.",
    )
    parser.add_argument(
        "--show-preprocess",
        action="store_true",
        help="Also print cosine similarities after routing preprocess.",
    )
    parser.add_argument(
        "--preprocess-type",
        default="zscore_mask",
        choices=[
            "identity",
            "zscore_mask",
            "v5_logmask",
            "v5_router_proc_v1",
            "v5_transform",
            "v5_transform_zscore_valid",
        ],
    )
    parser.add_argument(
        "--preprocess-json",
        default="",
        help="Load/Save preprocess json (for v5_transform_zscore_valid). If file exists -> load; else -> write.",
    )
    parser.add_argument("--preprocess-std-floor", type=float, default=1e-3)
    parser.add_argument("--preprocess-clip", type=float, default=5.0)
    parser.add_argument("--preprocess-mask-mode", default="zero", choices=["zero", "keep_raw"])
    parser.add_argument("--preprocess-fallback-dim", type=int, default=4)
    args = parser.parse_args()

    if args.emb_json:
        emb_path = _resolve_path(args.emb_json)
        base_entries = json.loads(emb_path.read_text(encoding="utf-8"))
        base_map: Dict[str, List[float]] = {e["hardware_name"]: e["vector"] for e in base_entries}
        universe_entries = list(base_entries)
        output_path = None
        print(f"[INFO] emb-json entries: {len(universe_entries)} from {emb_path}")
    else:
        v4_path = _resolve_path(args.v4_json)
        base_entries = json.loads(v4_path.read_text(encoding="utf-8"))
        base_map = {e["hardware_name"]: e["vector"] for e in base_entries}
        base_only = dict(base_map)

        gen = EmbeddingV4Generator()

        added = []
        for name in UNIVERSE_ADDITIONS:
            if name in base_map:
                continue
            vec = gen.generate(name, _default_tvm_config(name))
            base_map[name] = vec
            added.append(name)

        universe_entries = [{"hardware_name": name, "vector": base_map[name]} for name in sorted(base_map.keys())]
        if not args.output_json:
            args.output_json = str(ROOT / "gen/Embedding/hardware_embeddings_v4_universe.json")
        output_path = _resolve_path(args.output_json)
        output_path.write_text(json.dumps(universe_entries, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[INFO] v4 entries: {len(base_entries)}")
        print(f"[INFO] universe entries: {len(universe_entries)} (added {len(added)})")
        if added:
            print("[INFO] added:", ", ".join(sorted(added)))

        print("\n[CHECK] Orin / RPi consistency vs generator")
        for name in CHECK_NAMES:
            base_vec = base_only.get(name)
            if base_vec is None:
                print(f"  - {name}: NOT FOUND in v4")
                continue
            gen_vec = gen.generate(name, _default_tvm_config(name))
            diffs = [abs(a - b) for a, b in zip(base_vec, gen_vec)]
            max_diff = max(diffs) if diffs else 0.0
            mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
            status = "MATCH" if max_diff <= 1e-6 else "DIFF"
            print(f"  - {name}: {status} max_abs={max_diff:.6g} mean_abs={mean_diff:.6g}")

        print("\n[NOTE] Orin VRAM in generator PERF_DB is 64GB.")
        print("       If your platform is AGX Orin 32GB, add 'nvidia/jetson-orin-32gb' to universe.")

    print("\n[STATS] dim\tmean\tstd\tmin\tmax\tflag")
    vectors = list(base_map.values())
    means, stds, mins, maxs = _stats(vectors)
    for i, (mean, std, vmin, vmax) in enumerate(zip(means, stds, mins, maxs)):
        flag = "std<1e-6" if std < args.std_threshold else ""
        print(f"{i}\t{mean:.6g}\t{std:.6g}\t{vmin:.6g}\t{vmax:.6g}\t{flag}")

    print("\n[STATS] near-constant dims:", [i for i, s in enumerate(stds) if s < args.std_threshold])

    print("\n[GPU COSINE] universe vectors")
    for a_name, b_name in GPU_COS_PAIRS:
        a_vec = base_map.get(a_name)
        b_vec = base_map.get(b_name)
        if a_vec is None or b_vec is None:
            print(f"  - {a_name} vs {b_name}: MISSING")
            continue
        print(f"  - {a_name} vs {b_name}: cos={_cos(a_vec, b_vec):.6f}")

    if args.show_preprocess:
        params = None
        if args.preprocess_type == "v5_transform_zscore_valid" and args.preprocess_json:
            pre_path = _resolve_path(args.preprocess_json)
            if pre_path.exists():
                params = json.loads(pre_path.read_text(encoding="utf-8"))
            else:
                params = build_preprocess_params(
                    vectors,
                    preprocess_type=args.preprocess_type,
                    std_floor=args.preprocess_std_floor,
                    clip=args.preprocess_clip,
                    mask_mode=args.preprocess_mask_mode,
                    fallback_min_dim=args.preprocess_fallback_dim,
                    source=str(output_path) if output_path is not None else (args.emb_json or ""),
                )
                pre_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            params = build_preprocess_params(
                vectors,
                preprocess_type=args.preprocess_type,
                std_floor=args.preprocess_std_floor,
                clip=args.preprocess_clip,
                mask_mode=args.preprocess_mask_mode,
                fallback_min_dim=args.preprocess_fallback_dim,
                source=str(output_path) if output_path is not None else (args.emb_json or ""),
            )
        print("\n[PREPROCESS] ", summarize_preprocess(params))
        print("\n[GPU COSINE] after preprocess")
        for a_name, b_name in GPU_COS_PAIRS:
            a_vec = base_map.get(a_name)
            b_vec = base_map.get(b_name)
            if a_vec is None or b_vec is None:
                print(f"  - {a_name} vs {b_name}: MISSING")
                continue
            a_t = apply_preprocess(torch.tensor(a_vec), params).tolist()
            b_t = apply_preprocess(torch.tensor(b_vec), params).tolist()
            print(f"  - {a_name} vs {b_name}: cos={_cos(a_t, b_t):.6f}")


if __name__ == "__main__":
    main()
