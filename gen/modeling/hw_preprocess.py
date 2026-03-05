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
    if preprocess_type == "v5_logmask":
        return {"type": "v5_logmask"}
    if preprocess_type == "v5_router_proc_v1":
        return {
            "type": "v5_router_proc_v1",
            "segment_weights": {"A": 0.5, "B": 0.1, "C_abs": 1.0, "C_ratio": 1.5, "D": 0.2},
        }
    if preprocess_type == "v5_transform":
        return {"type": "v5_transform"}
    if preprocess_type == "v5_transform_zscore_valid":
        if not vectors:
            raise ValueError("build_preprocess_params requires non-empty vectors")
        # transform + valid-only stats
        dim = len(vectors[0])
        sums = [0.0] * dim
        sq_sums = [0.0] * dim
        counts = [0] * dim
        for vec in vectors:
            t, valid = _transform_v5_list(vec)
            for i in range(dim):
                if valid[i] > 0.5:
                    v = t[i]
                    sums[i] += v
                    sq_sums[i] += v * v
                    counts[i] += 1
        means = []
        stds = []
        mask = []
        for i in range(dim):
            if counts[i] <= 0:
                means.append(0.0)
                stds.append(1.0)
                mask.append(0.0)
                continue
            mean = sums[i] / counts[i]
            var = sq_sums[i] / counts[i] - mean * mean
            var = max(var, 0.0)
            std = math.sqrt(var)
            means.append(mean)
            stds.append(std)
            mask.append(1.0)
        params = {
            "type": "v5_transform_zscore_valid",
            "version": "v5_zscore_v1",
            "mean": means,
            "std": stds,
            "mask": mask,
            "valid_counts": counts,
            "std_floor": float(std_floor),
            "clip": float(clip),
            "l2_normalize": True,
        }
        if source:
            params["source"] = source
        return params
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
    if params.get("type") == "v5_logmask":
        return _apply_v5_logmask(hw_emb)
    if params.get("type") == "v5_router_proc_v1":
        return _apply_v5_router_proc_v1(hw_emb, params)
    if params.get("type") == "v5_transform":
        return _apply_v5_transform(hw_emb)
    if params.get("type") == "v5_transform_zscore_valid":
        return _apply_v5_transform_zscore_valid(hw_emb, params)
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
    if params.get("type") == "v5_logmask":
        return "v5_logmask"
    if params.get("type") == "v5_router_proc_v1":
        return "v5_router_proc_v1(seg_norm + weights)"
    if params.get("type") == "v5_transform":
        return "v5_transform"
    if params.get("type") == "v5_transform_zscore_valid":
        return "v5_transform_zscore_valid"
    return (
        f"{params.get('type')} mask_nonzero={params.get('mask_nonzero')} "
        f"std_floor={params.get('std_floor')} clip={params.get('clip')}"
    )


def _apply_v5_logmask(hw_emb: torch.Tensor) -> torch.Tensor:
    """Field-level transform for emb-v5: log2/log10 + validity mask (no zscore)."""
    # v5 dim = 24
    if hw_emb.shape[-1] < 24:
        return hw_emb

    # indices (emb-v5)
    IDX = {
        "is_gpu_hpc": 0,
        "is_gpu_edge": 1,
        "is_cpu_x86": 2,
        "is_cpu_arm": 3,
        "num_cores": 4,
        "vector_unit_bytes": 5,
        "cache_line_bytes": 6,
        "max_shared_memory_per_block": 7,
        "max_threads_per_block": 8,
        "registers_per_block": 9,
        "warp_size": 10,
        "sm_count": 11,
        "peak_fp32": 12,
        "peak_matmul": 13,
        "mem_bandwidth": 14,
        "llc_mb": 15,
        "mid_cache_kb": 16,
        "device_mem_gb": 17,
        "matmul_accel_ratio": 18,
        "lowp_level": 19,
        "compute_bw_ratio": 20,
        "cache_bw_ratio": 21,
        "bw_per_sm": 22,
        "mem_is_uma": 23,
    }

    h = hw_emb
    out = torch.zeros_like(h)

    # identity & env
    out[..., IDX["is_gpu_hpc"]] = h[..., IDX["is_gpu_hpc"]]
    out[..., IDX["is_gpu_edge"]] = h[..., IDX["is_gpu_edge"]]
    out[..., IDX["is_cpu_x86"]] = h[..., IDX["is_cpu_x86"]]
    out[..., IDX["is_cpu_arm"]] = h[..., IDX["is_cpu_arm"]]
    out[..., IDX["mem_is_uma"]] = h[..., IDX["mem_is_uma"]]

    is_gpu = (h[..., IDX["is_gpu_hpc"]] > 0.5) | (h[..., IDX["is_gpu_edge"]] > 0.5)
    is_cpu = (h[..., IDX["is_cpu_x86"]] > 0.5) | (h[..., IDX["is_cpu_arm"]] > 0.5)
    mem_is_uma = h[..., IDX["mem_is_uma"]] > 0.5

    def _mask_log2(val: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.where(valid & (val > 0), torch.log2(val), torch.zeros_like(val))

    def _mask_log10(val: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.where(valid & (val > 0), torch.log10(val), torch.zeros_like(val))

    # B segment (log2)
    out[..., IDX["num_cores"]] = _mask_log2(h[..., IDX["num_cores"]], is_cpu)
    out[..., IDX["vector_unit_bytes"]] = _mask_log2(h[..., IDX["vector_unit_bytes"]], is_cpu)
    out[..., IDX["cache_line_bytes"]] = _mask_log2(h[..., IDX["cache_line_bytes"]], is_cpu)
    out[..., IDX["max_shared_memory_per_block"]] = _mask_log2(h[..., IDX["max_shared_memory_per_block"]], is_gpu)
    out[..., IDX["max_threads_per_block"]] = _mask_log2(h[..., IDX["max_threads_per_block"]], is_gpu)
    out[..., IDX["registers_per_block"]] = _mask_log2(h[..., IDX["registers_per_block"]], is_gpu)
    out[..., IDX["warp_size"]] = _mask_log2(h[..., IDX["warp_size"]], is_gpu)

    # C absolute specs (log10)
    out[..., IDX["sm_count"]] = _mask_log10(h[..., IDX["sm_count"]], is_gpu)
    out[..., IDX["peak_fp32"]] = _mask_log10(h[..., IDX["peak_fp32"]], is_gpu | is_cpu)
    out[..., IDX["peak_matmul"]] = _mask_log10(h[..., IDX["peak_matmul"]], is_gpu)
    out[..., IDX["mem_bandwidth"]] = _mask_log10(h[..., IDX["mem_bandwidth"]], is_gpu | is_cpu)
    out[..., IDX["llc_mb"]] = _mask_log10(h[..., IDX["llc_mb"]], is_gpu | is_cpu)
    out[..., IDX["mid_cache_kb"]] = _mask_log10(h[..., IDX["mid_cache_kb"]], is_cpu)
    out[..., IDX["device_mem_gb"]] = _mask_log10(
        h[..., IDX["device_mem_gb"]],
        is_gpu & (~mem_is_uma),
    )

    # ratios
    pm = h[..., IDX["peak_matmul"]]
    pf = h[..., IDX["peak_fp32"]]
    valid_ratio = is_gpu & (pm > 0) & (pf > 0)
    out[..., IDX["matmul_accel_ratio"]] = torch.where(
        valid_ratio,
        torch.log10(pm) - torch.log10(pf),
        torch.zeros_like(pm),
    )
    out[..., IDX["compute_bw_ratio"]] = torch.where(
        is_gpu | is_cpu,
        h[..., IDX["compute_bw_ratio"]],
        torch.zeros_like(h[..., IDX["compute_bw_ratio"]]),
    )
    out[..., IDX["cache_bw_ratio"]] = torch.where(
        is_gpu | is_cpu,
        h[..., IDX["cache_bw_ratio"]],
        torch.zeros_like(h[..., IDX["cache_bw_ratio"]]),
    )

    # lowp_level (keep discrete)
    out[..., IDX["lowp_level"]] = torch.where(
        is_gpu, h[..., IDX["lowp_level"]], torch.zeros_like(h[..., IDX["lowp_level"]])
    )

    # bw_per_sm (keep derived ratio)
    out[..., IDX["bw_per_sm"]] = torch.where(
        is_gpu, h[..., IDX["bw_per_sm"]], torch.zeros_like(h[..., IDX["bw_per_sm"]])
    )

    return out


def _apply_v5_router_proc_v1(hw_emb: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    """v5_logmask + segment-wise RMS norm + fixed weights."""
    h = _apply_v5_logmask(hw_emb)

    # segment indices (v5)
    segs = {
        "A": list(range(0, 4)),
        "B": list(range(4, 11)),
        "C_abs": list(range(11, 18)),
        "C_ratio": list(range(18, 23)),
        "D": [23],
    }

    weights = params.get("segment_weights", {})

    # apply RMS per segment
    out = h.clone()
    for name, idxs in segs.items():
        if not idxs:
            continue
        seg = out[..., idxs]
        rms = torch.sqrt((seg * seg).mean(dim=-1, keepdim=True) + 1e-6)
        seg = seg / rms
        w = float(weights.get(name, 1.0))
        seg = seg * w
        out[..., idxs] = seg
    return out


def _transform_v5_list(vec: List[float]) -> Tuple[List[float], List[float]]:
    """Transform v5 raw vector (Python list). Returns (transformed, valid_mask)."""
    if len(vec) < 24:
        return vec, [1.0] * len(vec)

    # indices (v5)
    IDX = {
        "is_gpu_hpc": 0,
        "is_gpu_edge": 1,
        "is_cpu_x86": 2,
        "is_cpu_arm": 3,
        "num_cores": 4,
        "vector_unit_bytes": 5,
        "cache_line_bytes": 6,
        "max_shared_memory_per_block": 7,
        "max_threads_per_block": 8,
        "registers_per_block": 9,
        "warp_size": 10,
        "sm_count": 11,
        "peak_fp32": 12,
        "peak_matmul": 13,
        "mem_bandwidth": 14,
        "llc_mb": 15,
        "mid_cache_kb": 16,
        "device_mem_gb": 17,
        "matmul_accel_ratio": 18,
        "lowp_level": 19,
        "compute_bw_ratio": 20,
        "cache_bw_ratio": 21,
        "bw_per_sm": 22,
        "mem_is_uma": 23,
    }

    out = [0.0] * len(vec)
    valid = [0.0] * len(vec)

    is_gpu = (vec[IDX["is_gpu_hpc"]] > 0.5) or (vec[IDX["is_gpu_edge"]] > 0.5)
    is_cpu = (vec[IDX["is_cpu_x86"]] > 0.5) or (vec[IDX["is_cpu_arm"]] > 0.5)
    mem_is_uma = vec[IDX["mem_is_uma"]] > 0.5

    # A/D: keep 0/1
    for i in [IDX["is_gpu_hpc"], IDX["is_gpu_edge"], IDX["is_cpu_x86"], IDX["is_cpu_arm"], IDX["mem_is_uma"]]:
        out[i] = float(vec[i])
        valid[i] = 1.0

    def log2v(x: float) -> float:
        return math.log2(x)

    def log10v(x: float) -> float:
        return math.log10(x)

    # B: log2
    for i in [IDX["num_cores"], IDX["vector_unit_bytes"], IDX["cache_line_bytes"]]:
        if is_cpu and vec[i] > 0:
            out[i] = log2v(vec[i])
            valid[i] = 1.0
    for i in [IDX["max_shared_memory_per_block"], IDX["max_threads_per_block"], IDX["registers_per_block"], IDX["warp_size"]]:
        if is_gpu and vec[i] > 0:
            out[i] = log2v(vec[i])
            valid[i] = 1.0

    # C_abs: log10
    if is_gpu and vec[IDX["sm_count"]] > 0:
        out[IDX["sm_count"]] = log10v(vec[IDX["sm_count"]])
        valid[IDX["sm_count"]] = 1.0
    if (is_gpu or is_cpu) and vec[IDX["peak_fp32"]] > 0:
        out[IDX["peak_fp32"]] = log10v(vec[IDX["peak_fp32"]])
        valid[IDX["peak_fp32"]] = 1.0
    if is_gpu and vec[IDX["peak_matmul"]] > 0:
        out[IDX["peak_matmul"]] = log10v(vec[IDX["peak_matmul"]])
        valid[IDX["peak_matmul"]] = 1.0
    if (is_gpu or is_cpu) and vec[IDX["mem_bandwidth"]] > 0:
        out[IDX["mem_bandwidth"]] = log10v(vec[IDX["mem_bandwidth"]])
        valid[IDX["mem_bandwidth"]] = 1.0
    if (is_gpu or is_cpu) and vec[IDX["llc_mb"]] > 0:
        out[IDX["llc_mb"]] = log10v(vec[IDX["llc_mb"]])
        valid[IDX["llc_mb"]] = 1.0
    if is_cpu and vec[IDX["mid_cache_kb"]] > 0:
        out[IDX["mid_cache_kb"]] = log10v(vec[IDX["mid_cache_kb"]])
        valid[IDX["mid_cache_kb"]] = 1.0
    if is_gpu and (not mem_is_uma) and vec[IDX["device_mem_gb"]] > 0:
        out[IDX["device_mem_gb"]] = log10v(vec[IDX["device_mem_gb"]])
        valid[IDX["device_mem_gb"]] = 1.0

    # ratios
    pm = vec[IDX["peak_matmul"]]
    pf = vec[IDX["peak_fp32"]]
    if is_gpu and pm > 0 and pf > 0:
        out[IDX["matmul_accel_ratio"]] = log10v(pm) - log10v(pf)
        valid[IDX["matmul_accel_ratio"]] = 1.0
    # keep lowp_level
    if is_gpu:
        out[IDX["lowp_level"]] = vec[IDX["lowp_level"]]
        valid[IDX["lowp_level"]] = 1.0
    # compute/cache ratios
    if (is_gpu or is_cpu) and vec[IDX["peak_fp32"]] > 0 and vec[IDX["mem_bandwidth"]] > 0:
        out[IDX["compute_bw_ratio"]] = log10v(vec[IDX["peak_fp32"]]) - log10v(vec[IDX["mem_bandwidth"]])
        valid[IDX["compute_bw_ratio"]] = 1.0
    if (is_gpu or is_cpu) and vec[IDX["llc_mb"]] > 0 and vec[IDX["mem_bandwidth"]] > 0:
        out[IDX["cache_bw_ratio"]] = log10v(vec[IDX["llc_mb"]]) - log10v(vec[IDX["mem_bandwidth"]])
        valid[IDX["cache_bw_ratio"]] = 1.0
    if is_gpu and vec[IDX["mem_bandwidth"]] > 0 and vec[IDX["sm_count"]] > 0:
        out[IDX["bw_per_sm"]] = log10v(vec[IDX["mem_bandwidth"]]) - log10v(vec[IDX["sm_count"]])
        valid[IDX["bw_per_sm"]] = 1.0

    return out, valid


def _apply_v5_transform(hw_emb: torch.Tensor) -> torch.Tensor:
    # Support both single vector [D] and batched tensor [..., D].
    if hw_emb.ndim == 1:
        vec = hw_emb.detach().cpu().tolist()
        t, _ = _transform_v5_list(vec)
        return torch.tensor(t, dtype=hw_emb.dtype, device=hw_emb.device)

    last_dim = hw_emb.shape[-1]
    flat = hw_emb.reshape(-1, last_dim).detach().cpu().tolist()
    out_rows = []
    for vec in flat:
        t, _ = _transform_v5_list(vec)
        out_rows.append(t)
    out = torch.tensor(out_rows, dtype=hw_emb.dtype, device=hw_emb.device)
    return out.reshape(*hw_emb.shape[:-1], last_dim)


def _apply_v5_transform_zscore_valid(hw_emb: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    mean = params.get("mean", [])
    std = params.get("std", [])
    std_floor = float(params.get("std_floor", 1e-6))
    clip = float(params.get("clip", 0.0))
    l2_normalize = bool(params.get("l2_normalize", True))

    def _apply_one(vec: List[float]) -> List[float]:
        t, valid = _transform_v5_list(vec)
        out = []
        for i, v in enumerate(t):
            if i < len(valid) and valid[i] > 0.5 and i < len(mean) and i < len(std):
                s = max(float(std[i]), std_floor)
                z = (v - float(mean[i])) / s
            else:
                z = 0.0
            out.append(z)

        if clip > 0:
            out = [max(min(x, clip), -clip) for x in out]

        if l2_normalize:
            norm = math.sqrt(sum(x * x for x in out))
            if norm > 0:
                out = [x / norm for x in out]
        return out

    if hw_emb.ndim == 1:
        vec = hw_emb.detach().cpu().tolist()
        out = _apply_one(vec)
        return torch.tensor(out, dtype=hw_emb.dtype, device=hw_emb.device)

    last_dim = hw_emb.shape[-1]
    flat = hw_emb.reshape(-1, last_dim).detach().cpu().tolist()
    out_rows = [_apply_one(vec) for vec in flat]
    out = torch.tensor(out_rows, dtype=hw_emb.dtype, device=hw_emb.device)
    return out.reshape(*hw_emb.shape[:-1], last_dim)
