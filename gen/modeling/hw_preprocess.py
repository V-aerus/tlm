"""Hardware embedding preprocess utilities for routing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import torch

_IDENTITY_DEBUG_PRINTED = False


def build_preprocess_params(
    vectors: List[List[float]],
    *,
    preprocess_type: str = "zscore_mask",
    std_floor: float = 1e-3,
    clip: float = 5.0,
    mask_mode: str = "zero",
    fallback_min_dim: int = 4,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    if preprocess_type == "identity":
        return {"type": "identity"}
    if not vectors:
        raise ValueError("build_preprocess_params requires non-empty vectors")
    dim = len(vectors[0])
    means = []
    stds = []
    mask = []
    n = float(len(vectors))
    for i in range(dim):
        vals = [vec[i] for vec in vectors]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std)
        mask.append(1.0 if std >= std_floor else 0.0)
    nonzero = int(sum(1 for m in mask if m > 0))
    params = {
        "type": preprocess_type,
        "mean": means,
        "std": stds,
        "mask": mask,
        "std_floor": float(std_floor),
        "clip": float(clip),
        "mask_mode": mask_mode,
        "fallback_min_dim": int(fallback_min_dim),
        "mask_nonzero": nonzero,
    }
    if source:
        params["source"] = source
    return params


def _to_tensor(x: List[float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def apply_preprocess(hw_emb: torch.Tensor, params: Optional[Dict[str, Any]]) -> torch.Tensor:
    if params is None or params.get("type", "identity") == "identity":
        return hw_emb
    if "mean" not in params or "std" not in params or "mask" not in params:
        return hw_emb
    mask_nonzero = int(params.get("mask_nonzero", 0))
    fallback_min_dim = int(params.get("fallback_min_dim", 4))
    if mask_nonzero < fallback_min_dim:
        return hw_emb

    device = hw_emb.device
    dtype = hw_emb.dtype
    mean = _to_tensor(params["mean"], device=device, dtype=dtype)
    std = _to_tensor(params["std"], device=device, dtype=dtype)
    mask = _to_tensor(params["mask"], device=device, dtype=dtype)
    std_floor = float(params.get("std_floor", 1e-6))
    clip = float(params.get("clip", 0.0))
    mask_mode = params.get("mask_mode", "zero")

    std = torch.clamp(std, min=std_floor)
    z = (hw_emb - mean) / std
    if mask_mode == "zero":
        z = z * mask
    elif mask_mode == "keep_raw":
        keep = (mask > 0).to(dtype)
        z = z * keep + hw_emb * (1.0 - keep)
    if clip > 0:
        z = torch.clamp(z, -clip, clip)
    return z


def preprocess_meta_equal(a: Dict[str, Any], b: Dict[str, Any], *, tol: float = 1e-6) -> bool:
    ta = (a or {}).get("type", "identity")
    tb = (b or {}).get("type", "identity")
    if ta == "identity" and tb == "identity":
        global _IDENTITY_DEBUG_PRINTED
        if not _IDENTITY_DEBUG_PRINTED:
            print("[ROUTER-PRE] both preprocess are identity; skip consistency mismatch.")
            _IDENTITY_DEBUG_PRINTED = True
        return True
    if a is b:
        return True
    if not a and not b:
        return True
    if not a:
        return b.get("type", "identity") == "identity"
    if not b:
        return a.get("type", "identity") == "identity"
    keys = [
        "type",
        "std_floor",
        "clip",
        "mask_mode",
        "fallback_min_dim",
        "mask_nonzero",
    ]
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if abs(float(av) - float(bv)) > tol:
                return False
        else:
            if av != bv:
                return False
    for key in ["mean", "std", "mask"]:
        if key not in a or key not in b:
            return False
        av = a[key]
        bv = b[key]
        if len(av) != len(bv):
            return False
        for x, y in zip(av, bv):
            if abs(float(x) - float(y)) > tol:
                return False
    return True


def summarize_preprocess(params: Optional[Dict[str, Any]]) -> str:
    if not params or params.get("type", "identity") == "identity":
        return "identity"
    return (
        f"{params.get('type')} mask_nonzero={params.get('mask_nonzero')} "
        f"std_floor={params.get('std_floor')} clip={params.get('clip')}"
    )
