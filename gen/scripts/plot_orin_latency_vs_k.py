#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tvm import auto_scheduler


def normalize_workload_key(raw: str) -> str:
    if raw is None:
        return ""
    raw = raw.strip()
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        return raw


def parse_int_list(raw: str) -> List[int]:
    values = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            range_part, *step_part = item.split(":")
            start_s, end_s = range_part.split("-", 1)
            start = int(float(start_s))
            end = int(float(end_s))
            step = int(float(step_part[0])) if step_part else 1
            if step <= 0:
                raise ValueError(f"Invalid step in k-values: {item}")
            for value in range(start, end + 1, step):
                values.add(value)
        else:
            values.add(int(float(item)))
    return sorted(values)


def percentile(values: List[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    if q <= 0:
        return vals[0]
    if q >= 100:
        return vals[-1]
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def build_display_ticks(k_values: List[int], tlm_max_k: int) -> List[int]:
    preferred = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1000]
    k_set = set(k_values)
    ticks = [value for value in preferred if value in k_set]
    if not ticks:
        ticks = sorted(k_set)
    if tlm_max_k in k_set and tlm_max_k not in ticks:
        ticks.append(tlm_max_k)
        ticks.sort()
    return ticks


def parse_networks(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_latency_ms(res) -> float:
    if getattr(res, "error_no", 0) != 0:
        return math.nan
    vals = []
    for item in getattr(res, "costs", []):
        try:
            vals.append(float(item.value) if hasattr(item, "value") else float(item))
        except Exception:
            pass
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals) * 1e3)


def load_tlm_sequences(path: Path) -> Dict[str, List[float]]:
    seqs: Dict[str, List[float]] = defaultdict(list)
    for inp, res in auto_scheduler.RecordReader(str(path)):
        wk = normalize_workload_key(inp.task.workload_key)
        seqs[wk].append(read_latency_ms(res))
    return dict(seqs)


def load_workload_meta(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], float]]:
    workload_meta: Dict[str, Dict[str, str]] = {}
    task_weights: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wk = normalize_workload_key(row["workload_key"])
            workload_meta[wk] = {
                "network_id": row["network_id"],
                "network_name": row["network_name"],
                "network_shape": row["network_shape"],
            }
            task_weights[(row["network_id"], wk)] = 1.0
    return workload_meta, task_weights


def resolve_ansor_log_path(summary_path: Path, raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)
    marker = "/ansor/"
    if marker in raw_path:
        suffix = raw_path.split(marker, 1)[1]
        alt = summary_path.parent / suffix
        if alt.exists():
            return str(alt)
    return raw_path


def load_ansor_log_paths(path: Path, target: str) -> Dict[Tuple[str, str], str]:
    best_log_by_task: Dict[Tuple[str, str], Tuple[int, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "task":
                continue
            if row.get("target") != target:
                continue
            wk = normalize_workload_key(row["workload_key"])
            net_id = row["network_id"]
            log_path = resolve_ansor_log_path(path, row.get("log_path", ""))
            if not log_path:
                continue
            try:
                budget = int(float(row.get("budget", "0") or 0))
            except Exception:
                continue
            try:
                weight = float(row.get("task_weight", "0") or 0.0)
            except Exception:
                weight = 0.0
            key = (net_id, wk)
            prev = best_log_by_task.get(key)
            if prev is None or budget > prev[0]:
                best_log_by_task[key] = (budget, log_path)
            if weight > 0:
                best_log_by_task.setdefault(key, (budget, log_path))
    return {key: log for key, (_, log) in best_log_by_task.items()}


def load_ansor_weights(path: Path, target: str) -> Dict[Tuple[str, str], float]:
    weights: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "task":
                continue
            if row.get("target") != target:
                continue
            wk = normalize_workload_key(row["workload_key"])
            net_id = row["network_id"]
            try:
                weight = float(row.get("task_weight", "0") or 0.0)
            except Exception:
                weight = 0.0
            key = (net_id, wk)
            if key not in weights or weight > weights[key]:
                weights[key] = weight
    return weights


def load_ansor_sequences(log_paths: Dict[Tuple[str, str], str]) -> Dict[Tuple[str, str], List[float]]:
    seqs: Dict[Tuple[str, str], List[float]] = {}
    for key, log_path in log_paths.items():
        vals: List[float] = []
        try:
            for _, res in auto_scheduler.RecordReader(str(log_path)):
                vals.append(read_latency_ms(res))
        except Exception:
            vals = []
        seqs[key] = vals
    return seqs


def prefix_best(seq: List[float], k: int) -> float:
    if not seq:
        return math.nan
    best = math.nan
    for v in seq[: min(k, len(seq))]:
        if math.isnan(v):
            continue
        if math.isnan(best) or v < best:
            best = v
    return best


def aggregate_rows(
    networks: List[str],
    k_values: List[int],
    workload_meta: Dict[str, Dict[str, str]],
    task_weights: Dict[Tuple[str, str], float],
    official_seqs: Dict[str, List[float]],
    lora_seqs: Dict[str, List[float]],
    ansor_seqs: Dict[Tuple[str, str], List[float]],
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []

    for net_id in networks:
        net_workloads = [wk for wk, meta in workload_meta.items() if meta["network_id"] == net_id]
        for k in k_values:
            off_prefix = {wk: prefix_best(official_seqs.get(wk, []), k) for wk in net_workloads}
            lora_prefix = {wk: prefix_best(lora_seqs.get(wk, []), k) for wk in net_workloads}
            ans_prefix = {
                wk: prefix_best(ansor_seqs.get((net_id, wk), []), k)
                for wk in net_workloads
            }

            common = [
                wk
                for wk in net_workloads
                if not math.isnan(off_prefix[wk])
                and not math.isnan(lora_prefix[wk])
                and not math.isnan(ans_prefix[wk])
            ]

            row = {
                "network_id": net_id,
                "k": k,
                "common_workloads": len(common),
                "official_available": sum(not math.isnan(off_prefix[wk]) for wk in net_workloads),
                "xavier_available": sum(not math.isnan(lora_prefix[wk]) for wk in net_workloads),
                "ansor_available": sum(not math.isnan(ans_prefix[wk]) for wk in net_workloads),
                "official_ms": math.nan,
                "xavier_lora_ms": math.nan,
                "ansor_ms": math.nan,
                "official_speedup_vs_ansor": math.nan,
                "xavier_speedup_vs_ansor": math.nan,
            }

            if common:
                off_ms = 0.0
                lora_ms = 0.0
                ans_ms = 0.0
                for wk in common:
                    weight = task_weights.get((net_id, wk), 1.0)
                    off_ms += off_prefix[wk] * weight
                    lora_ms += lora_prefix[wk] * weight
                    ans_ms += ans_prefix[wk] * weight
                row["official_ms"] = off_ms
                row["xavier_lora_ms"] = lora_ms
                row["ansor_ms"] = ans_ms
                if off_ms > 0:
                    row["official_speedup_vs_ansor"] = ans_ms / off_ms
                if lora_ms > 0:
                    row["xavier_speedup_vs_ansor"] = ans_ms / lora_ms

            out_rows.append(row)
    return out_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "network_id",
        "k",
        "common_workloads",
        "official_available",
        "xavier_available",
        "ansor_available",
        "official_ms",
        "xavier_lora_ms",
        "ansor_ms",
        "official_speedup_vs_ansor",
        "xavier_speedup_vs_ansor",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row[key]
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else f"{value:.6f}"
                else:
                    out[key] = value
            writer.writerow(out)


def draw_png(
    rows: List[Dict[str, object]],
    networks: List[str],
    out_path: Path,
    y_cap_percentile: float,
    tlm_max_k: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width = 2100
    height = 1200
    margin = 40
    cols = 2
    rows_n = max(1, (len(networks) + cols - 1) // cols)
    cell_w = (width - margin * (cols + 1)) // cols
    cell_h = (height - margin * (rows_n + 1)) // rows_n

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    colors = {
        "ansor_ms": (31, 119, 180),
        "official_ms": (255, 127, 14),
        "xavier_lora_ms": (44, 160, 44),
    }
    x_ticks = build_display_ticks(sorted({int(r["k"]) for r in rows}), tlm_max_k)

    def to_float(value: object) -> float:
        if value is None or value == "":
            return math.nan
        return float(value)

    for idx, net_id in enumerate(networks):
        row_idx = idx // cols
        col_idx = idx % cols
        x0 = margin + col_idx * (cell_w + margin)
        y0 = margin + row_idx * (cell_h + margin)
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        pad = 55
        px0, py0, px1, py1 = x0 + pad, y0 + pad, x1 - 20, y1 - 60
        draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200), width=1)

        sub = [r for r in rows if r["network_id"] == net_id]
        sub = sorted(sub, key=lambda r: int(r["k"]))
        series = {
            "ansor_ms": [(int(r["k"]), to_float(r["ansor_ms"])) for r in sub if not math.isnan(to_float(r["ansor_ms"]))],
            "official_ms": [
                (int(r["k"]), to_float(r["official_ms"]))
                for r in sub
                if int(r["k"]) <= tlm_max_k and not math.isnan(to_float(r["official_ms"]))
            ],
            "xavier_lora_ms": [
                (int(r["k"]), to_float(r["xavier_lora_ms"]))
                for r in sub
                if int(r["k"]) <= tlm_max_k and not math.isnan(to_float(r["xavier_lora_ms"]))
            ],
        }
        cov = max([int(r["common_workloads"]) for r in sub] + [0])
        draw.text((x0 + 8, y0 + 8), f"{net_id}  common_max={cov}", fill="black", font=font_small)
        if not series["ansor_ms"]:
            continue

        ansor_values = [value for _, value in series["ansor_ms"]]
        tlm_values = [value for _, value in series["official_ms"]] + [value for _, value in series["xavier_lora_ms"]]
        y_values = [v for v in ansor_values + tlm_values if not math.isnan(v)]
        if not y_values:
            continue
        y_min = min(y_values)
        y_cap_source = tlm_values if tlm_values else y_values
        y_cap = percentile(y_cap_source, y_cap_percentile)
        y_max = max(y_values)
        if not math.isnan(y_cap):
            y_max = min(y_max, y_cap)
        if y_max <= y_min:
            y_max = max(y_min + 1e-3, max(y_cap_source))
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        x_values = sorted({k for points in series.values() for k, _ in points})
        x_min = min(x_values)
        x_max = max(x_values)
        linear_end = min(tlm_max_k, x_max)
        left_ratio = 0.66 if x_max > tlm_max_k else 1.0
        split_x = px0 + int((px1 - px0) * left_ratio)

        def map_x(v: int) -> int:
            if x_max <= x_min:
                return px0
            if x_max <= tlm_max_k or linear_end <= x_min:
                t = (v - x_min) / max(1.0, (x_max - x_min))
                return int(px0 + t * (px1 - px0))
            if v <= linear_end:
                t = (v - x_min) / max(1.0, (linear_end - x_min))
                t = t ** 0.82
                return int(px0 + t * (split_x - px0))
            rhs_min = math.log10(max(linear_end, 1))
            rhs_max = math.log10(max(x_max, linear_end + 1))
            t = (math.log10(v) - rhs_min) / max(1e-6, (rhs_max - rhs_min))
            return int(split_x + t * (px1 - split_x))

        def map_y_raw(v: float) -> float:
            t = (v - y_min) / (y_max - y_min)
            return py1 - t * (py1 - py0)

        def map_y(v: float) -> int:
            v = min(v, y_max)
            t = (v - y_min) / (y_max - y_min)
            return int(py1 - t * (py1 - py0))

        def clip_segment_to_top(
            p1: Tuple[float, float], p2: Tuple[float, float]
        ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
            x1, y1 = p1
            x2, y2 = p2
            vis1 = y1 >= py0
            vis2 = y2 >= py0
            if vis1 and vis2:
                return ((x1, y1), (x2, y2))
            if not vis1 and not vis2:
                return None
            if y2 == y1:
                return None
            t = (py0 - y1) / (y2 - y1)
            xi = x1 + t * (x2 - x1)
            inter = (xi, float(py0))
            if vis1:
                return ((x1, y1), inter)
            return (inter, (x2, y2))

        draw.line([px0, py1, px1, py1], fill="black", width=1)
        draw.line([px0, py0, px0, py1], fill="black", width=1)

        for x_tick in x_ticks:
            if x_tick < x_min or x_tick > x_max:
                continue
            xx = map_x(x_tick)
            draw.line([xx, py1, xx, py1 + 4], fill="black", width=1)
            tick_label = str(x_tick)
            box = draw.textbbox((0, 0), tick_label, font=font_small)
            label_w = box[2] - box[0]
            draw.text((xx - label_w // 2, py1 + 8), tick_label, fill="black", font=font_small)

        if x_max > tlm_max_k and x_min < tlm_max_k:
            sep_x = map_x(tlm_max_k)
            y = py0
            while y < py1:
                draw.line([sep_x, y, sep_x, min(y + 8, py1)], fill=(180, 180, 180), width=1)
                y += 14
            draw.text((sep_x - 20, py0 - 20), "K=64", fill=(120, 120, 120), font=font_small)

        for frac in [0.0, 0.5, 1.0]:
            val = y_min + frac * (y_max - y_min)
            yy = map_y(val)
            draw.line([px0 - 4, yy, px0, yy], fill="black", width=1)
            draw.text((x0 + 4, yy - 7), f"{val:.1f}", fill="black", font=font_small)
            draw.line([px0, yy, px1, yy], fill=(235, 235, 235), width=1)

        for name, points_raw in series.items():
            raw_points = [(float(map_x(x)), float(map_y_raw(y))) for x, y in points_raw]
            for p1, p2 in zip(raw_points, raw_points[1:]):
                clipped = clip_segment_to_top(p1, p2)
                if clipped is None:
                    continue
                q1, q2 = clipped
                draw.line([q1, q2], fill=colors[name], width=3)

        draw.text((px0, y1 - 35), "K per workload (1-64 linear, >64 compressed)", fill="black", font=font_small)
        draw.text((x0 + 4, py0 - 25), f"Aggregated latency (ms), y_cap from TLM p{int(y_cap_percentile)}", fill="black", font=font_small)

    legend_items = [
        ("Ansor", colors["ansor_ms"]),
        ("Official-v100", colors["official_ms"]),
        ("Xavier-LoRA", colors["xavier_lora_ms"]),
    ]
    lx = margin
    ly = height - 28
    for name, color in legend_items:
        draw.line([lx, ly, lx + 30, ly], fill=color, width=4)
        draw.text((lx + 36, ly - 10), name, fill="black", font=font)
        lx += 260
    draw.text((lx + 10, ly - 10), "curves enter from top when clipped", fill="black", font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Orin aggregated latency vs per-workload candidate budget K.")
    parser.add_argument("--official-measured", required=True)
    parser.add_argument("--lora-measured", required=True)
    parser.add_argument("--ansor-summary", required=True)
    parser.add_argument("--workload-csv", required=True)
    parser.add_argument("--ansor-target", default="orin")
    parser.add_argument(
        "--networks",
        default="bert_base_1x128,resnet_50_1x3x224x224,mobilenet_v2_1x3x224x224,inception_v3_1x3x299x299",
    )
    parser.add_argument("--k-values", default="1-64,80,96,112,128,160,192,256,384,512,768,1000")
    parser.add_argument("--y-cap-percentile", type=float, default=90.0)
    parser.add_argument("--tlm-max-k", type=int, default=64)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", default="")
    args = parser.parse_args()

    networks = parse_networks(args.networks)
    k_values = parse_int_list(args.k_values)

    workload_meta, fallback_weights = load_workload_meta(Path(args.workload_csv))
    official_seqs = load_tlm_sequences(Path(args.official_measured))
    lora_seqs = load_tlm_sequences(Path(args.lora_measured))
    ansor_log_paths = load_ansor_log_paths(Path(args.ansor_summary), args.ansor_target)
    ansor_weights = load_ansor_weights(Path(args.ansor_summary), args.ansor_target)
    ansor_seqs = load_ansor_sequences(ansor_log_paths)

    task_weights = dict(fallback_weights)
    task_weights.update(ansor_weights)

    rows = aggregate_rows(
        networks=networks,
        k_values=k_values,
        workload_meta=workload_meta,
        task_weights=task_weights,
        official_seqs=official_seqs,
        lora_seqs=lora_seqs,
        ansor_seqs=ansor_seqs,
    )
    write_csv(Path(args.output_csv), rows)
    print(f"[CSV] {args.output_csv}")
    if args.output_png:
        draw_png(rows, networks, Path(args.output_png), args.y_cap_percentile, args.tlm_max_k)
        print(f"[PNG] {args.output_png}")


if __name__ == "__main__":
    main()
