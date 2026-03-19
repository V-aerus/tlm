#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_preprocess_stability.py

目的
----
评估硬件 embedding preprocess（尤其是 v5_transform_zscore_valid）在“小硬件池”下的稳定性：
  - LOO: leave-one-out（每次去掉 1 个硬件重新估计 mean/std）
  - Bootstrap: 重采样（对同样大小的硬件池做有放回抽样）

核心关注点（你们当前的痛点）
------------------------
v5_transform_zscore_valid 的统计口径是：
  logmask -> valid-only z-score -> (optional clip) -> L2 normalize

其中 z-score 的 mean/std 是由“当前硬件池”估计出来的，因此需要用该脚本回答：
  - 这些 mean/std 对于删掉/替换少量硬件是否敏感？
  - 相似度（cosine）与近邻排序（top-k neighbors）是否稳定？

用法示例
--------
python gen/scripts/inspect_preprocess_stability.py \\
  --emb-json gen/Embedding/hardware_embeddings_v5_draft.json \\
  --preprocess-type v5_transform_zscore_valid \\
  --mode both \\
  --bootstrap-iters 200 \\
  --topk 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen.modeling.hw_preprocess import build_preprocess_params, apply_preprocess, summarize_preprocess  # noqa: E402


DEFAULT_PAIRS = [
    # GPU-GPU
    ("nvidia/rtx-4090", "nvidia/nvidia-a100"),
    ("nvidia/rtx-4090", "nvidia/nvidia-v100"),
    ("nvidia/rtx-4090", "nvidia/geforce-rtx-3090"),
    ("nvidia/geforce-rtx-3090", "nvidia/geforce-rtx-3060"),
    ("nvidia/nvidia-v100", "nvidia/geforce-gtx-1060"),
    ("nvidia/nvidia-v100", "nvidia/geforce-gtx-950"),
    ("nvidia/jetson-orin", "nvidia/jetson-agx-xavier"),
    # CPU-CPU
    ("intel/xeon-gold-6226", "aws/cpu/c5.18xlarge"),
    ("intel/core-i7-12700k", "intel/core-i5-12400"),
    ("amd/ryzen-7-5800h", "intel/core-i7-10510u"),
    ("raspberry-pi/4b-aarch64", "intel/core-i7-10510u"),
    # mixed
    ("nvidia/rtx-4090", "intel/xeon-gold-6226"),
    ("nvidia/jetson-orin", "raspberry-pi/4b-aarch64"),
]

DEFAULT_ANCHORS = [
    "nvidia/rtx-4090",
    "nvidia/nvidia-v100",
    "nvidia/jetson-orin",
    "intel/xeon-gold-6226",
    "raspberry-pi/4b-aarch64",
]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path


def _load_embeddings(path: Path) -> Tuple[List[str], Dict[str, List[float]]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    names = [e["hardware_name"] for e in entries]
    mp = {e["hardware_name"]: e["vector"] for e in entries}
    return names, mp


def _parse_pairs(args_pairs: Optional[List[str]]) -> List[Tuple[str, str]]:
    if not args_pairs:
        return list(DEFAULT_PAIRS)
    out = []
    for item in args_pairs:
        if "," not in item:
            raise ValueError(f"Invalid --pair '{item}', expect A,B")
        a, b = item.split(",", 1)
        out.append((a.strip(), b.strip()))
    return out


def _parse_csv_list(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0).item())


def _safe_std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mean = sum(xs) / float(len(xs))
    var = sum((x - mean) ** 2 for x in xs) / float(len(xs))
    return math.sqrt(max(var, 0.0))


def _summarize_dist(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"count": 0}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mean = sum(xs_sorted) / float(n)
    std = _safe_std(xs_sorted)
    p05 = xs_sorted[int(0.05 * (n - 1))]
    p50 = xs_sorted[int(0.50 * (n - 1))]
    p95 = xs_sorted[int(0.95 * (n - 1))]
    return {
        "count": float(n),
        "mean": float(mean),
        "std": float(std),
        "min": float(xs_sorted[0]),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "max": float(xs_sorted[-1]),
    }


def _build_params(
    vectors: List[List[float]],
    preprocess_type: str,
    *,
    std_floor: float,
    clip: float,
    source: str,
) -> Dict[str, Any]:
    return build_preprocess_params(
        vectors,
        preprocess_type=preprocess_type,
        std_floor=std_floor,
        clip=clip,
        source=source,
    )


def _preprocess_name_to_tensor(
    name_to_vec: Dict[str, List[float]],
    params: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    out = {}
    for name, vec in name_to_vec.items():
        out[name] = apply_preprocess(torch.tensor(vec, dtype=torch.float32), params)
    return out


def _calc_pair_cos(
    pairs: List[Tuple[str, str]],
    emb_map: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    out = {}
    for a, b in pairs:
        out[(a, b)] = _cos(emb_map[a], emb_map[b])
    return out


def _calc_neighbors(
    anchors: List[str],
    names: List[str],
    emb_map: Dict[str, torch.Tensor],
    *,
    topk: int,
) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for a in anchors:
        av = emb_map[a]
        sims = []
        for b in names:
            if b == a:
                continue
            sims.append((b, _cos(av, emb_map[b])))
        sims.sort(key=lambda x: x[1], reverse=True)
        out[a] = sims[: max(topk, 1)]
    return out


def _format_float(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect stability of preprocess params and cosine similarity under resampling.")
    ap.add_argument("--emb-json", default="gen/Embedding/hardware_embeddings_v5_draft.json")
    ap.add_argument(
        "--baseline-preprocess-json",
        default="",
        help="Optional preprocess meta JSON to use as baseline (otherwise compute from full set).",
    )
    ap.add_argument(
        "--preprocess-type",
        default="v5_transform_zscore_valid",
        help="Preprocess type passed to build_preprocess_params (default: v5_transform_zscore_valid).",
    )
    ap.add_argument("--std-floor", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--mode", choices=("loo", "bootstrap", "both"), default="both")
    ap.add_argument("--bootstrap-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair", action="append", default=None, help="Repeatable pair spec: A,B")
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS), help="Comma-separated anchors for top-k neighbors.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--output-json", default="", help="Optional output JSON for downstream plotting.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    emb_path = _resolve_path(args.emb_json)
    names, name_to_vec = _load_embeddings(emb_path)
    print(f"[INFO] emb entries: {len(names)} from {emb_path}")

    pairs = _parse_pairs(args.pair)
    anchors = _parse_csv_list(args.anchors)

    # filter missing pairs/anchors
    missing = []
    for a, b in pairs:
        if a not in name_to_vec or b not in name_to_vec:
            missing.append((a, b))
    if missing:
        print("[WARN] missing pairs (skipped):")
        for a, b in missing:
            print(f"  - {a} vs {b}")
        pairs = [(a, b) for a, b in pairs if a in name_to_vec and b in name_to_vec]

    anchors = [a for a in anchors if a in name_to_vec]
    if not anchors:
        anchors = []

    # ----- baseline -----
    if args.baseline_preprocess_json:
        base_params = json.loads(_resolve_path(args.baseline_preprocess_json).read_text(encoding="utf-8"))
        base_source = f"file:{args.baseline_preprocess_json}"
    else:
        base_params = _build_params(
            list(name_to_vec.values()),
            args.preprocess_type,
            std_floor=args.std_floor,
            clip=args.clip,
            source=str(emb_path),
        )
        base_source = "computed:full"

    print(f"[BASE] {base_source} preprocess={summarize_preprocess(base_params)}")
    base_emb = _preprocess_name_to_tensor(name_to_vec, base_params)
    base_pair_cos = _calc_pair_cos(pairs, base_emb) if pairs else {}
    base_neighbors = _calc_neighbors(anchors, names, base_emb, topk=args.topk) if anchors else {}

    # Print baseline cos for quick check.
    if pairs:
        print("\n[BASE COS]")
        for (a, b), c in base_pair_cos.items():
            print(f"  - {a} vs {b}: cos={_format_float(c)}")

    # ----- resampling runs -----
    out_json: Dict[str, Any] = {
        "meta": {
            "emb_json": str(emb_path),
            "preprocess_type": args.preprocess_type,
            "std_floor": float(args.std_floor),
            "clip": float(args.clip),
            "seed": int(args.seed),
            "n": int(len(names)),
            "baseline_source": base_source,
        },
        "baseline": {
            "pair_cos": {f"{a}|||{b}": float(c) for (a, b), c in base_pair_cos.items()},
            "neighbors": base_neighbors,
        },
        "loo": {},
        "bootstrap": {},
    }

    def run_once(vectors: List[List[float]], tag: str) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], float], Dict[str, List[Tuple[str, float]]]]:
        params = _build_params(
            vectors,
            args.preprocess_type,
            std_floor=args.std_floor,
            clip=args.clip,
            source=tag,
        )
        emb_map = _preprocess_name_to_tensor(name_to_vec, params)
        pair_cos = _calc_pair_cos(pairs, emb_map) if pairs else {}
        neigh = _calc_neighbors(anchors, names, emb_map, topk=args.topk) if anchors else {}
        return params, pair_cos, neigh

    # Accumulators
    def init_pair_acc() -> Dict[Tuple[str, str], List[float]]:
        return {p: [] for p in pairs}

    loo_pair = init_pair_acc()
    boot_pair = init_pair_acc()
    loo_neighbor_votes: Dict[str, Counter] = {a: Counter() for a in anchors}
    boot_neighbor_votes: Dict[str, Counter] = {a: Counter() for a in anchors}

    # Parameter stability: track mean/std per dim across reps (only for preprocess that has these fields).
    def init_param_acc() -> Dict[str, List[List[float]]]:
        return {"mean": [], "std": [], "valid_counts": []}

    loo_param = init_param_acc()
    boot_param = init_param_acc()

    # --- LOO ---
    if args.mode in ("loo", "both"):
        for i, drop_name in enumerate(names):
            sub_names = [n for n in names if n != drop_name]
            vectors = [name_to_vec[n] for n in sub_names]
            params, pair_cos, neigh = run_once(vectors, tag=f"loo:drop={drop_name}")

            if "mean" in params and "std" in params:
                loo_param["mean"].append(list(params.get("mean", [])))
                loo_param["std"].append(list(params.get("std", [])))
            if "valid_counts" in params:
                loo_param["valid_counts"].append(list(params.get("valid_counts", [])))

            for p, c in pair_cos.items():
                loo_pair[p].append(c)
            for a, ns in neigh.items():
                for b, _c in ns:
                    loo_neighbor_votes[a][b] += 1

        print(f"\n[LOO] done: {len(names)} runs")

    # --- Bootstrap ---
    if args.mode in ("bootstrap", "both"):
        iters = max(int(args.bootstrap_iters), 0)
        for t in range(iters):
            sample_names = [names[rng.randrange(len(names))] for _ in range(len(names))]
            vectors = [name_to_vec[n] for n in sample_names]
            params, pair_cos, neigh = run_once(vectors, tag=f"bootstrap:iter={t}")

            if "mean" in params and "std" in params:
                boot_param["mean"].append(list(params.get("mean", [])))
                boot_param["std"].append(list(params.get("std", [])))
            if "valid_counts" in params:
                boot_param["valid_counts"].append(list(params.get("valid_counts", [])))

            for p, c in pair_cos.items():
                boot_pair[p].append(c)
            for a, ns in neigh.items():
                for b, _c in ns:
                    boot_neighbor_votes[a][b] += 1

        print(f"\n[BOOT] done: {iters} runs (n={len(names)})")

    # ----- summarize -----
    def summarize_pairs(title: str, acc: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
        if not pairs:
            return {}
        print(f"\n[{title} COS STATS]")
        out: Dict[str, Any] = {}
        for a, b in pairs:
            xs = acc[(a, b)]
            s = _summarize_dist(xs)
            base = base_pair_cos.get((a, b))
            if base is not None and xs:
                deltas = [x - base for x in xs]
                s["base"] = float(base)
                s["delta_abs_max"] = float(max(abs(d) for d in deltas))
                s["delta_std"] = float(_safe_std(deltas))
            key = f"{a}|||{b}"
            out[key] = s
            if "count" in s and s.get("count", 0) > 0:
                base_s = _format_float(float(s.get("base", float("nan")))) if "base" in s else "n/a"
                print(
                    f"  - {a} vs {b}: base={base_s} "
                    f"mean={_format_float(s.get('mean', float('nan')))} std={_format_float(s.get('std', float('nan')))} "
                    f"min={_format_float(s.get('min', float('nan')))} p05={_format_float(s.get('p05', float('nan')))} "
                    f"p95={_format_float(s.get('p95', float('nan')))} max={_format_float(s.get('max', float('nan')))} "
                    f"delta_abs_max={_format_float(s.get('delta_abs_max', float('nan')))}"
                )
        return out

    def summarize_neighbors(title: str, votes: Dict[str, Counter], total_runs: int) -> Dict[str, Any]:
        if not anchors:
            return {}
        print(f"\n[{title} NEIGHBOR STABILITY] top{args.topk}")
        out: Dict[str, Any] = {}
        for a in anchors:
            ctr = votes[a]
            items = ctr.most_common(max(args.topk * 3, 8))
            out[a] = [{"neighbor": n, "freq": int(c), "rate": float(c) / float(total_runs or 1)} for n, c in items]
            print(f"  - anchor={a}")
            for n, c in items[: max(args.topk, 1)]:
                print(f"      {n}: {c}/{total_runs} ({(float(c)/float(total_runs or 1)):.2%})")
        return out

    def summarize_params(title: str, param_acc: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        means = param_acc.get("mean") or []
        stds = param_acc.get("std") or []
        if not means or not stds:
            return {}
        base_mean = list(base_params.get("mean", [])) if isinstance(base_params.get("mean"), list) else []
        base_std = list(base_params.get("std", [])) if isinstance(base_params.get("std"), list) else []
        dim = min(len(base_mean), len(base_std), len(means[0]), len(stds[0]))
        if dim <= 0:
            return {}

        # Per-dim max |delta| over reps for mean/std.
        mean_max = []
        std_max = []
        mean_std = []
        std_std = []
        for i in range(dim):
            dm = [abs(m[i] - base_mean[i]) for m in means]
            ds = [abs(s[i] - base_std[i]) for s in stds]
            mean_max.append(max(dm) if dm else 0.0)
            std_max.append(max(ds) if ds else 0.0)
            mean_std.append(_safe_std([m[i] for m in means]))
            std_std.append(_safe_std([s[i] for s in stds]))

        # Print top dims by max delta.
        top_k = 8
        idx_mean = sorted(range(dim), key=lambda i: mean_max[i], reverse=True)[:top_k]
        idx_std = sorted(range(dim), key=lambda i: std_max[i], reverse=True)[:top_k]
        print(f"\n[{title} PARAM STABILITY] compare to baseline ({base_source})")
        print(f"  top{top_k} dims by max |mean-mean_base|:")
        for i in idx_mean:
            print(f"    dim {i:2d}: max_abs_delta={mean_max[i]:.6g} rep_std={mean_std[i]:.6g} base_mean={base_mean[i]:.6g}")
        print(f"  top{top_k} dims by max |std-std_base|:")
        for i in idx_std:
            print(f"    dim {i:2d}: max_abs_delta={std_max[i]:.6g} rep_std={std_std[i]:.6g} base_std={base_std[i]:.6g}")

        return {
            "dim": int(dim),
            "mean_max_abs_delta": mean_max,
            "std_max_abs_delta": std_max,
            "mean_rep_std": mean_std,
            "std_rep_std": std_std,
        }

    if args.mode in ("loo", "both"):
        out_json["loo"]["pair_cos"] = summarize_pairs("LOO", loo_pair)
        out_json["loo"]["neighbors"] = summarize_neighbors("LOO", loo_neighbor_votes, total_runs=len(names))
        out_json["loo"]["param_stability"] = summarize_params("LOO", loo_param)

    if args.mode in ("bootstrap", "both"):
        out_json["bootstrap"]["pair_cos"] = summarize_pairs("BOOT", boot_pair)
        out_json["bootstrap"]["neighbors"] = summarize_neighbors("BOOT", boot_neighbor_votes, total_runs=int(args.bootstrap_iters))
        out_json["bootstrap"]["param_stability"] = summarize_params("BOOT", boot_param)

    if args.output_json:
        out_path = _resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

