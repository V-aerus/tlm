#!/usr/bin/env python3

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tvm import auto_scheduler


def normalize_workload_key(raw: str) -> str:
    if raw is None:
        return ""
    raw = raw.strip()
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        return raw


def parse_budgets(raw: str) -> List[int]:
    budgets: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budgets.append(int(float(item)))
    return budgets


def parse_networks(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_measured_best_ms(path: Path) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for inp, res in auto_scheduler.RecordReader(str(path)):
        if getattr(res, "error_no", 0) != 0:
            continue
        vals = []
        for item in res.costs:
            try:
                vals.append(float(item.value) if hasattr(item, "value") else float(item))
            except Exception:
                pass
        if not vals:
            continue
        lat_ms = float(sum(vals) / len(vals) * 1e3)
        wk = normalize_workload_key(inp.task.workload_key)
        if wk not in best or lat_ms < best[wk]:
            best[wk] = lat_ms
    return best


def read_ansor_task_rows(
    path: Path, target: str
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, str], float], Dict[str, str], List[int]]:
    ansor_best: Dict[Tuple[str, int, str], float] = {}
    task_weight: Dict[Tuple[str, str], float] = {}
    net_name: Dict[str, str] = {}
    budgets = set()

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("target") != target:
                continue
            row_type = row.get("row_type")
            if row_type != "task":
                continue
            wk_raw = row.get("workload_key")
            if not wk_raw:
                continue
            try:
                budget = int(float(row.get("budget", "0")))
            except Exception:
                continue
            wk = normalize_workload_key(wk_raw)
            net_id = row.get("network_id", "")
            if not net_id:
                continue
            try:
                weight = float(row.get("task_weight", "0") or 0.0)
            except Exception:
                weight = 0.0
            task_key = (net_id, wk)
            if task_key not in task_weight or weight > task_weight[task_key]:
                task_weight[task_key] = weight
            net_name[net_id] = row.get("network_name", net_id)

            best_raw = row.get("best_latency_ms", "")
            if not best_raw:
                budgets.add(budget)
                continue
            try:
                best_ms = float(best_raw)
            except Exception:
                budgets.add(budget)
                continue
            key = (net_id, budget, wk)
            if key not in ansor_best or best_ms < ansor_best[key]:
                ansor_best[key] = best_ms
            budgets.add(budget)

    return ansor_best, task_weight, net_name, sorted(budgets)


def aggregate_budget_rows(
    networks: List[str],
    budgets: List[int],
    ansor_best: Dict[Tuple[str, int, str], float],
    task_weight: Dict[Tuple[str, str], float],
    official_best: Dict[str, float],
    lora_best: Dict[str, float],
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []

    for net_id in networks:
        keys_for_net = [wk for (nid, wk) in task_weight.keys() if nid == net_id]
        keys_off = {wk for wk in keys_for_net if wk in official_best}
        keys_lora = {wk for wk in keys_for_net if wk in lora_best}
        keys_ans_any = {
            wk for wk in keys_for_net if any((net_id, b, wk) in ansor_best for b in budgets)
        }
        base_keys = keys_off & keys_lora & keys_ans_any

        for budget in budgets:
            ans_keys = {wk for wk in base_keys if (net_id, budget, wk) in ansor_best}
            if not ans_keys:
                out_rows.append(
                    {
                        "network_id": net_id,
                        "budget": budget,
                        "common_workloads": 0,
                        "ansor_ms": math.nan,
                        "official_ms": math.nan,
                        "xavier_lora_ms": math.nan,
                        "official_speedup_vs_ansor": math.nan,
                        "xavier_speedup_vs_ansor": math.nan,
                    }
                )
                continue

            ans_ms = 0.0
            off_ms = 0.0
            lora_ms = 0.0
            for wk in ans_keys:
                w = task_weight.get((net_id, wk), 1.0)
                ans_ms += ansor_best[(net_id, budget, wk)] * w
                off_ms += official_best[wk] * w
                lora_ms += lora_best[wk] * w

            out_rows.append(
                {
                    "network_id": net_id,
                    "budget": budget,
                    "common_workloads": len(ans_keys),
                    "ansor_ms": ans_ms,
                    "official_ms": off_ms,
                    "xavier_lora_ms": lora_ms,
                    "official_speedup_vs_ansor": (ans_ms / off_ms) if off_ms > 0 else math.nan,
                    "xavier_speedup_vs_ansor": (ans_ms / lora_ms) if lora_ms > 0 else math.nan,
                }
            )
    return out_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "network_id",
        "budget",
        "common_workloads",
        "ansor_ms",
        "official_ms",
        "xavier_lora_ms",
        "official_speedup_vs_ansor",
        "xavier_speedup_vs_ansor",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row[k]
                if isinstance(v, float):
                    out[k] = "" if math.isnan(v) else f"{v:.6f}"
                else:
                    out[k] = v
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Orin latency-vs-budget comparison.")
    parser.add_argument("--official-measured", required=True)
    parser.add_argument("--lora-measured", required=True)
    parser.add_argument("--ansor-summary", required=True)
    parser.add_argument("--ansor-target", default="orin")
    parser.add_argument(
        "--networks",
        default="bert_base_1x128,resnet_50_1x3x224x224,mobilenet_v2_1x3x224x224,inception_v3_1x3x299x299",
    )
    parser.add_argument("--budgets", default="1,32,64,1000")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", default="")
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    networks = parse_networks(args.networks)

    official_best = read_measured_best_ms(Path(args.official_measured))
    lora_best = read_measured_best_ms(Path(args.lora_measured))
    ansor_best, task_weight, net_name, ansor_budgets_all = read_ansor_task_rows(
        Path(args.ansor_summary), args.ansor_target
    )

    budgets = [b for b in budgets if b in ansor_budgets_all]
    rows = aggregate_budget_rows(networks, budgets, ansor_best, task_weight, official_best, lora_best)
    write_csv(Path(args.output_csv), rows)
    print(f"[CSV] {args.output_csv}")
    print(f"[INFO] official_keys={len(official_best)} lora_keys={len(lora_best)} budgets={budgets}")
    if args.output_png:
        plot_helper = Path(__file__).resolve().parent / "plot_orin_budget_latency_from_csv.py"
        if not plot_helper.exists():
            print(f"[WARN] 缺少绘图脚本，跳过 PNG: {plot_helper}")
            return
        cmd = [
            sys.executable,
            str(plot_helper),
            "--input-csv",
            args.output_csv,
            "--output-png",
            args.output_png,
            "--networks",
            ",".join(networks),
        ]
        try:
            subprocess.check_call(cmd)
        except Exception as exc:
            print(f"[WARN] 调用绘图脚本失败，跳过 PNG: {exc}")


if __name__ == "__main__":
    main()
