#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


def parse_networks(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    if v is None or v == "":
        return math.nan
    return float(v)


def percentile(values: List[float], q: float) -> float:
    vals = sorted([v for v in values if not math.isnan(v)])
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def draw_with_pillow(rows: List[Dict[str, str]], networks: List[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width = 1800
    height = 1200
    margin = 40
    cols = 2
    rows_n = (len(networks) + cols - 1) // cols
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

    def collect_series(net_id: str) -> Tuple[List[int], List[float], List[float], List[float], int]:
        sub = [r for r in rows if r.get("network_id") == net_id]
        sub = sorted(sub, key=lambda x: int(float(x["budget"])))
        xs = [int(float(r["budget"])) for r in sub]
        ans = [to_float(r.get("ansor_ms", "")) for r in sub]
        off = [to_float(r.get("official_ms", "")) for r in sub]
        lora = [to_float(r.get("xavier_lora_ms", "")) for r in sub]
        cov = max([int(float(r.get("common_workloads", "0") or 0)) for r in sub] + [0])
        return xs, ans, off, lora, cov

    def build_x_mapper(xs: List[int], px0: int, px1: int):
        # 让 1~64 保留细粒度，64 之后再用 log 展开
        split_k = 64
        left_ratio = 0.82
        left_gamma = 1.0
        max_x = max(xs)
        right_has = max_x > split_k
        right_min = math.log10(split_k)
        right_max = math.log10(max_x) if max_x > 0 else right_min + 1.0

        def mapper(v: int) -> int:
            if v <= split_k or not right_has:
                t = 0.0 if split_k <= 1 else (v - 1) / (split_k - 1)
                t = max(0.0, min(1.0, t))
                t = pow(t, left_gamma)
                return int(px0 + t * (px1 - px0) * left_ratio)
            t = (math.log10(v) - right_min) / max(1e-6, right_max - right_min)
            t = max(0.0, min(1.0, t))
            return int(px0 + (left_ratio + t * (1.0 - left_ratio)) * (px1 - px0))

        return mapper

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

        xs, ys_ans, ys_off, ys_lora, cov = collect_series(net_id)
        draw.text((x0 + 8, y0 + 8), f"{net_id}  common={cov}", fill="black", font=font_small)
        if not xs:
            continue

        # official / xavier 只展示到 K<=64
        ys_off = [v if x <= 64 else math.nan for x, v in zip(xs, ys_off)]
        ys_lora = [v if x <= 64 else math.nan for x, v in zip(xs, ys_lora)]

        pairs_ans = [(x, y) for x, y in zip(xs, ys_ans) if not math.isnan(y)]
        pairs_off = [(x, y) for x, y in zip(xs, ys_off) if not math.isnan(y)]
        pairs_lora = [(x, y) for x, y in zip(xs, ys_lora) if not math.isnan(y)]
        y_values = [v for _, v in pairs_ans + pairs_off + pairs_lora]
        if not y_values:
            continue
        y_max = max(y_values)

        low_candidates = [y for x, y in pairs_ans if x >= 16] + [y for x, y in pairs_lora if x >= 16]
        if not low_candidates:
            low_candidates = [y for _, y in pairs_ans + pairs_lora]
        if not low_candidates:
            low_candidates = y_values
        y_min = min(low_candidates)
        y_low_max = percentile(low_candidates, 0.98)
        if math.isnan(y_low_max):
            y_low_max = max(low_candidates)

        high_candidates = [y for x, y in pairs_off if x <= 16] + [y for x, y in pairs_ans if x <= 16] + [y for x, y in pairs_lora if x <= 16]
        if not high_candidates:
            high_candidates = [y for _, y in pairs_off + pairs_ans + pairs_lora]
        y_high_min = percentile(high_candidates, 0.25)
        if math.isnan(y_high_min):
            y_high_min = percentile(y_values, 0.85)
        if math.isnan(y_high_min):
            y_high_min = y_low_max

        y_low_max = max(y_min + 1e-6, min(y_low_max, y_max - 1e-6))
        y_high_min = max(y_min + 1e-6, min(y_high_min, y_max - 1e-6))
        if y_high_min <= y_low_max * 1.10:
            y_high_min = percentile(y_values, 0.85)
            if math.isnan(y_high_min):
                y_high_min = y_low_max * 1.2
        y_high_min = max(y_low_max + 1e-6, min(y_high_min, y_max - 1e-6))

        y_span = max(1e-6, y_max - y_min)
        broken_axis = (y_high_min - y_low_max) > 0.05 * y_span and y_max > y_low_max * 1.3
        if not broken_axis:
            y_cap = percentile(y_values, 0.95)
            if math.isnan(y_cap):
                y_cap = y_max
            if y_cap <= y_min:
                y_cap = y_min + 1e-3
            y_pad = (y_cap - y_min) * 0.08
            y_min = max(0.0, y_min - y_pad)
            y_max = y_cap

        map_x = build_x_mapper(xs, px0, px1)
        top_ratio = 0.33
        break_gap = 14
        y_break_center = int(py0 + top_ratio * (py1 - py0))
        y_top_end = y_break_center - break_gap // 2
        y_bottom_start = y_break_center + break_gap // 2

        def map_y(v: float) -> int:
            if math.isnan(v):
                return py1
            if not broken_axis:
                v = min(v, y_max)
                t = (v - y_min) / (y_max - y_min)
                return int(py1 - t * (py1 - py0))
            if v <= y_low_max:
                t = (v - y_min) / max(1e-6, (y_low_max - y_min))
                return int(py1 - t * (py1 - y_bottom_start))
            v2 = max(v, y_high_min)
            v2 = min(v2, y_max)
            t = (v2 - y_high_min) / max(1e-6, (y_max - y_high_min))
            return int(y_top_end - t * (y_top_end - py0))

        draw.line([px0, py1, px1, py1], fill="black", width=1)
        draw.line([px0, py0, px0, py1], fill="black", width=1)

        major_ticks = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 512, 1000]
        major_ticks = [x for x in major_ticks if min(xs) <= x <= max(xs)]
        last_tick_x = -10_000
        for x_tick in major_ticks:
            xx = map_x(x_tick)
            if xx - last_tick_x < 22:
                continue
            last_tick_x = xx
            draw.line([xx, py1, xx, py1 + 4], fill="black", width=1)
            draw.text((xx - 10, py1 + 8), str(x_tick), fill="black", font=font_small)

        if broken_axis:
            low_ticks = [y_min, y_min + 0.5 * (y_low_max - y_min), y_low_max]
            high_ticks = [y_high_min, y_high_min + 0.5 * (y_max - y_high_min), y_max]
            for val in low_ticks + high_ticks:
                yy = map_y(val)
                draw.line([px0 - 4, yy, px0, yy], fill="black", width=1)
                draw.text((x0 + 4, yy - 7), f"{val:.1f}", fill="black", font=font_small)
                draw.line([px0, yy, px1, yy], fill=(235, 235, 235), width=1)
            # 轴断裂符号
            for dy in [-4, 4]:
                draw.line([px0 - 6, y_break_center + dy - 3, px0 + 6, y_break_center + dy + 3], fill="black", width=2)
        else:
            for frac in [0.0, 0.5, 1.0]:
                val = y_min + frac * (y_max - y_min)
                yy = map_y(val)
                draw.line([px0 - 4, yy, px0, yy], fill="black", width=1)
                draw.text((x0 + 4, yy - 7), f"{val:.1f}", fill="black", font=font_small)
                draw.line([px0, yy, px1, yy], fill=(235, 235, 235), width=1)

        series = {
            "ansor_ms": ys_ans,
            "official_ms": ys_off,
            "xavier_lora_ms": ys_lora,
        }
        for name, vals in series.items():
            points = [(map_x(x), map_y(y)) for x, y in zip(xs, vals) if not math.isnan(y)]
            if not points:
                continue
            if name == "official_ms":
                raw = [v for v in vals if not math.isnan(v)]
                if raw:
                    lo = min(raw)
                    hi = max(raw)
                    if hi > 0 and (hi - lo) / hi < 0.05:
                        # official 基本不降时，直接画水平线
                        y_line = map_y(sum(raw) / len(raw))
                        x_line0 = map_x(min([x for x, y in zip(xs, vals) if not math.isnan(y)]))
                        x_line1 = map_x(max([x for x, y in zip(xs, vals) if not math.isnan(y)]))
                        draw.line([x_line0, y_line, x_line1, y_line], fill=colors[name], width=3)
                        continue
            for p1, p2 in zip(points, points[1:]):
                draw.line([p1, p2], fill=colors[name], width=3)

        draw.text((px0, y1 - 35), "Budget (piecewise-x)", fill="black", font=font_small)
        draw.text((x0 + 4, py0 - 25), "Latency (ms, broken-y)", fill="black", font=font_small)
        if broken_axis:
            draw.text((px1 - 220, py0 - 25), f"break: {y_low_max:.1f} -> {y_high_min:.1f}", fill=(120, 0, 0), font=font_small)
        elif max(y_values) > y_max * 1.001:
            draw.text((px1 - 175, py0 - 25), f"clip>@{y_max:.1f}ms", fill=(120, 0, 0), font=font_small)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[PNG] {out_path} (pillow)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency-vs-budget from precomputed CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument(
        "--networks",
        default="bert_base_1x128,resnet_50_1x3x224x224,mobilenet_v2_1x3x224x224,inception_v3_1x3x299x299",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    networks = parse_networks(args.networks)
    out = Path(args.output_png)

    draw_with_pillow(rows, networks, out)


if __name__ == "__main__":
    main()
