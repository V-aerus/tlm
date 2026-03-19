#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_log(log_path: Path) -> Tuple[float, float, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    e2e = math.nan
    tune_min = math.nan
    mode = "empty" if text.strip() == "" else "unknown"

    mode_match = re.search(r"\[BUILD_MODE\]\s*(\S+)", text)
    if mode_match:
        mode = mode_match.group(1)

    total_match = re.search(r"\|\s*Total\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", text)
    if total_match:
        tune_min = float(total_match.group(1))

    mean_match = re.search(
        r"\n\s*([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*\n",
        text,
    )
    if mean_match:
        e2e = float(mean_match.group(1))
    return e2e, tune_min, mode


def best_so_far(values: List[float]) -> List[float]:
    out = []
    current = math.inf
    for value in values:
        if not math.isnan(value):
            current = min(current, value)
        out.append(current if current < math.inf else math.nan)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse tune_logs/*/k*.log and plot e2e-vs-K (raw + best-so-far)."
    )
    parser.add_argument("--tune-logs-root", required=True, help=".../tuning_time_eval/tune_logs")
    parser.add_argument("--method", default="official", help="official|xavier|ansor")
    parser.add_argument("--networks", default="", help="Optional CSV list of network ids.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-png", default="")
    parser.add_argument("--no-plot", action="store_true", help="Only export CSV + console summary.")
    parser.add_argument("--title", default="")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    method_dir = Path(args.tune_logs_root) / args.method
    if not method_dir.exists():
        raise FileNotFoundError(f"method log dir not found: {method_dir}")

    allow_nets = set(parse_csv_list(args.networks))
    network_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
    if allow_nets:
        network_dirs = [d for d in network_dirs if d.name in allow_nets]
    if not network_dirs:
        raise RuntimeError("no network directories found.")

    rows: List[Dict[str, object]] = []
    per_net: Dict[str, List[Tuple[int, float, float, str]]] = {}

    for net_dir in network_dirs:
        points = []
        for log_path in sorted(net_dir.glob("k*.log")):
            match = re.search(r"k(\d+)\.log$", log_path.name)
            if not match:
                continue
            k = int(match.group(1))
            e2e, tune_min, mode = parse_log(log_path)
            rows.append(
                {
                    "method": args.method,
                    "network_id": net_dir.name,
                    "k": k,
                    "e2e_ms": "" if math.isnan(e2e) else f"{e2e:.6f}",
                    "tuning_time_min": "" if math.isnan(tune_min) else f"{tune_min:.6f}",
                    "build_mode": mode,
                    "log_path": str(log_path),
                }
            )
            points.append((k, e2e, tune_min, mode))
        points.sort(key=lambda item: item[0])
        per_net[net_dir.name] = points

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "network_id",
                "k",
                "e2e_ms",
                "tuning_time_min",
                "build_mode",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_data: Dict[str, Tuple[List[int], List[float], List[float]]] = {}

    print("=== Tuning Summary ===")
    for net, points in per_net.items():
        if not points:
            continue
        ks = [item[0] for item in points]
        vals = [item[1] for item in points]
        modes = Counter(item[3] for item in points)
        valid = [(k, v) for k, v in zip(ks, vals) if not math.isnan(v)]

        if valid:
            k_first, first = valid[0]
            k_best, best = min(valid, key=lambda item: item[1])
            k_last, last = valid[-1]
            improve = (first - best) / first * 100.0 if first > 0 else 0.0
            print(
                f"- {net}: valid={len(valid)}/{len(points)} "
                f"first@k{k_first}={first:.4f} best@k{k_best}={best:.4f} "
                f"last@k{k_last}={last:.4f} improve={improve:.2f}% modes={dict(modes)}"
            )
        else:
            print(f"- {net}: valid=0/{len(points)} modes={dict(modes)}")

        raw_vals = vals
        best_vals = best_so_far(vals)
        plot_data[net] = (ks, raw_vals, best_vals)

    if not args.no_plot:
        if not args.out_png:
            raise ValueError("--out-png is required when --no-plot is not set.")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=args.dpi)
        ax_raw, ax_best = axes
        for net, (ks, raw_vals, best_vals) in plot_data.items():
            ax_raw.plot(ks, raw_vals, linewidth=1.3, label=net)
            ax_best.plot(ks, best_vals, linewidth=1.6, label=net)

        ax_raw.set_title(f"{args.method} e2e latency (raw)")
        ax_raw.set_xlabel("K")
        ax_raw.set_ylabel("E2E latency (ms)")
        ax_raw.grid(alpha=0.3)

        ax_best.set_title(f"{args.method} e2e latency (best-so-far)")
        ax_best.set_xlabel("K")
        ax_best.set_ylabel("E2E latency (ms)")
        ax_best.grid(alpha=0.3)

        ax_raw.legend(fontsize=8)
        ax_best.legend(fontsize=8)

        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout()

        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)
        print(f"[PNG] {out_png}")
    print(f"[CSV] {out_csv}")


if __name__ == "__main__":
    main()
