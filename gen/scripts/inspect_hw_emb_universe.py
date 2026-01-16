#!/usr/bin/env python3
"""Build and inspect v4 universe embeddings for routing preprocess stats."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v4-json",
        default=str(ROOT / "gen/Embedding/hardware_embeddings_v4.json"),
        help="Path to hardware_embeddings_v4.json",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "gen/Embedding/hardware_embeddings_v4_universe.json"),
        help="Where to write hardware_embeddings_v4_universe.json",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=1e-6,
        help="Std threshold to flag near-constant dims.",
    )
    args = parser.parse_args()

    v4_path = Path(args.v4_json)
    base_entries = json.loads(v4_path.read_text(encoding="utf-8"))
    base_map: Dict[str, List[float]] = {e["hardware_name"]: e["vector"] for e in base_entries}
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
    Path(args.output_json).write_text(json.dumps(universe_entries, ensure_ascii=False, indent=2), encoding="utf-8")

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


if __name__ == "__main__":
    main()
