#!/usr/bin/env python3
"""Export per-subgraph latency stats from AutoScheduler measured records."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

from tvm import auto_scheduler


def normalize_workload_key(raw: str) -> str:
    """Normalize workload_key string for stable joins."""
    if raw is None:
        return ""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    try:
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        return raw


def _latency_seconds(res) -> Optional[float]:
    costs = []
    for item in res.costs:
        try:
            val = float(item.value) if hasattr(item, "value") else float(item)
        except (TypeError, ValueError):
            val = None
        if val is not None:
            costs.append(val)
    if not costs:
        return None
    return sum(costs) / len(costs)


def load_stats(path: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    total = 0
    ok = 0
    err = 0
    for inp, res in auto_scheduler.RecordReader(path):
        total += 1
        key = normalize_workload_key(inp.task.workload_key)
        entry = stats.setdefault(
            key,
            {
                "count": 0,
                "ok": 0,
                "err": 0,
                "sum": 0.0,
                "min": None,
            },
        )
        entry["count"] += 1
        if getattr(res, "error_no", 0) != 0:
            entry["err"] += 1
            err += 1
            continue
        lat = _latency_seconds(res)
        if lat is None:
            entry["err"] += 1
            err += 1
            continue
        entry["ok"] += 1
        ok += 1
        entry["sum"] += lat
        if entry["min"] is None or lat < entry["min"]:
            entry["min"] = lat
    return stats


def to_ms(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return val * 1e3


def parse_budgets(raw: str) -> List[int]:
    if not raw:
        return []
    budgets: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            budgets.append(int(float(item)))
        except ValueError:
            continue
    return budgets


def load_ansor_summary(path: str, target: Optional[str]) -> Tuple[Dict[Tuple[str, int], float], Dict[str, Dict[str, str]]]:
    import csv

    summary: Dict[Tuple[str, int], float] = {}
    meta: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "task":
                continue
            if target and row.get("target") != target:
                continue
            key_raw = row.get("workload_key")
            key = normalize_workload_key(key_raw) if key_raw else ""
            if not key:
                continue
            try:
                budget = int(float(row.get("budget", "0")))
            except ValueError:
                continue
            try:
                best_ms = float(row.get("best_latency_ms", "nan"))
            except ValueError:
                continue
            if key not in meta:
                meta[key] = {
                    "network_name": row.get("network_name", ""),
                    "network_shape": row.get("network_shape", ""),
                    "network_id": row.get("network_id", ""),
                }
            k = (key, budget)
            if k not in summary or best_ms < summary[k]:
                summary[k] = best_ms
    return summary, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-subgraph latency stats to CSV.")
    parser.add_argument("--measured-path", required=True, help="Measured JSONL file path.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: <measured>.csv).")
    parser.add_argument("--measured-name", default="measured", help="Label prefix for measured columns.")
    parser.add_argument("--compare-path", default=None, help="Optional second measured JSONL for ratio.")
    parser.add_argument("--compare-name", default="compare", help="Label prefix for compare columns.")
    parser.add_argument("--ansor-summary", default=None, help="Optional ansor_summary.csv for baseline columns.")
    parser.add_argument("--ansor-target", default=None, help="Optional target filter for ansor_summary (e.g., 4090).")
    parser.add_argument("--ansor-budgets", default="64,1000", help="Comma-separated ansor budgets to export.")
    args = parser.parse_args()

    if not os.path.exists(args.measured_path):
        raise FileNotFoundError(f"Measured file not found: {args.measured_path}")

    out_csv = args.output_csv or f"{args.measured_path}.csv"

    stats_a = load_stats(args.measured_path)
    stats_b = load_stats(args.compare_path) if args.compare_path else None
    ansor_summary = {}
    ansor_meta = {}
    budgets = parse_budgets(args.ansor_budgets)
    if args.ansor_summary:
        ansor_summary, ansor_meta = load_ansor_summary(args.ansor_summary, args.ansor_target)

    keys = sorted(set(stats_a.keys()) | (set(stats_b.keys()) if stats_b else set()))

    if stats_b:
        columns = [
            "workload_key",
            f"{args.measured_name}_best_ms",
            f"{args.measured_name}_mean_ms",
            f"{args.measured_name}_ok",
            f"{args.measured_name}_err",
            f"{args.measured_name}_count",
            f"{args.compare_name}_best_ms",
            f"{args.compare_name}_mean_ms",
            f"{args.compare_name}_ok",
            f"{args.compare_name}_err",
            f"{args.compare_name}_count",
            "ratio_best",
            "ratio_mean",
            "notes",
        ]
    else:
        columns = [
            "workload_key",
            "best_ms",
            "mean_ms",
            "ok",
            "err",
            "count",
        ]
    if ansor_summary:
        columns = ["network_name", "network_shape", "network_id"] + columns
        for budget in budgets:
            columns.extend(
                [
                    f"ansor_{budget}_best_ms",
                    f"speedup_ansor_{budget}_vs_measured_best",
                ]
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columns)
        writer.writeheader()
        for key in keys:
            a = stats_a.get(key)
            network_name = ""
            network_shape = ""
            network_id = ""
            if ansor_meta and key in ansor_meta:
                network_name = ansor_meta[key].get("network_name", "")
                network_shape = ansor_meta[key].get("network_shape", "")
                network_id = ansor_meta[key].get("network_id", "")

            if stats_b:
                b = stats_b.get(key)
                a_best = to_ms(a["min"]) if a and a["min"] is not None else None
                a_mean = to_ms(a["sum"] / a["ok"]) if a and a["ok"] > 0 else None
                b_best = to_ms(b["min"]) if b and b["min"] is not None else None
                b_mean = to_ms(b["sum"] / b["ok"]) if b and b["ok"] > 0 else None

                ratio_best = (a_best / b_best) if (a_best and b_best) else None
                ratio_mean = (a_mean / b_mean) if (a_mean and b_mean) else None
                notes = []
                if a is None:
                    notes.append("missing_measured")
                if b is None:
                    notes.append("missing_compare")

                row = {
                    "workload_key": key,
                    f"{args.measured_name}_best_ms": "" if a_best is None else f"{a_best:.6f}",
                    f"{args.measured_name}_mean_ms": "" if a_mean is None else f"{a_mean:.6f}",
                    f"{args.measured_name}_ok": 0 if a is None else a["ok"],
                    f"{args.measured_name}_err": 0 if a is None else a["err"],
                    f"{args.measured_name}_count": 0 if a is None else a["count"],
                    f"{args.compare_name}_best_ms": "" if b_best is None else f"{b_best:.6f}",
                    f"{args.compare_name}_mean_ms": "" if b_mean is None else f"{b_mean:.6f}",
                    f"{args.compare_name}_ok": 0 if b is None else b["ok"],
                    f"{args.compare_name}_err": 0 if b is None else b["err"],
                    f"{args.compare_name}_count": 0 if b is None else b["count"],
                    "ratio_best": "" if ratio_best is None else f"{ratio_best:.6f}",
                    "ratio_mean": "" if ratio_mean is None else f"{ratio_mean:.6f}",
                    "notes": ",".join(notes),
                }
                if ansor_summary:
                    row["network_name"] = network_name
                    row["network_shape"] = network_shape
                    row["network_id"] = network_id
                    for budget in budgets:
                        ans_key = (key, budget)
                        ans_best = ansor_summary.get(ans_key)
                        row[f"ansor_{budget}_best_ms"] = "" if ans_best is None else f"{ans_best:.6f}"
                        speed = (ans_best / a_best) if (ans_best and a_best) else None
                        row[f"speedup_ansor_{budget}_vs_measured_best"] = "" if speed is None else f"{speed:.6f}"
                writer.writerow(row)
            else:
                best = to_ms(a["min"]) if a and a["min"] is not None else None
                mean = to_ms(a["sum"] / a["ok"]) if a and a["ok"] > 0 else None
                row = {
                    "workload_key": key,
                    "best_ms": "" if best is None else f"{best:.6f}",
                    "mean_ms": "" if mean is None else f"{mean:.6f}",
                    "ok": 0 if a is None else a["ok"],
                    "err": 0 if a is None else a["err"],
                    "count": 0 if a is None else a["count"],
                }
                if ansor_summary:
                    row["network_name"] = network_name
                    row["network_shape"] = network_shape
                    row["network_id"] = network_id
                    for budget in budgets:
                        ans_key = (key, budget)
                        ans_best = ansor_summary.get(ans_key)
                        row[f"ansor_{budget}_best_ms"] = "" if ans_best is None else f"{ans_best:.6f}"
                        speed = (ans_best / best) if (ans_best and best) else None
                        row[f"speedup_ansor_{budget}_vs_measured_best"] = "" if speed is None else f"{speed:.6f}"
                writer.writerow(row)

    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
