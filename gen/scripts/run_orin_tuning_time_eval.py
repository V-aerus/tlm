#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tvm import auto_scheduler

SIGNATURE_VERSION = "orin_tuning_time_eval_v2"


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
                raise ValueError(f"invalid step in k-values: {item}")
            for value in range(start, end + 1, step):
                values.add(value)
        else:
            values.add(int(float(item)))
    return sorted(values)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def file_fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def to_signature_str(payload: Dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def save_meta(path: Path, signature: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
    payload: Dict[str, object] = {"signature": signature}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_meta_signature(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sig = data.get("signature")
    if not isinstance(sig, dict):
        return None
    return sig


def normalize_workload_key(raw: str) -> str:
    if raw is None:
        return ""
    raw = raw.strip()
    try:
        return json.dumps(json.loads(raw), separators=(",", ":"))
    except Exception:
        return raw


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


def parse_network_shape(shape_raw: str) -> List[int]:
    shape = ast.literal_eval(shape_raw)
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"invalid network_shape: {shape_raw}")
    return [int(v) for v in shape]


def resolve_ansor_log_path(summary_path: Path, raw_path: str) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    marker = "/ansor/"
    if marker in raw_path:
        suffix = raw_path.split(marker, 1)[1]
        alt = summary_path.parent / suffix
        if alt.exists():
            return alt
    return None


def load_summary_meta(
    summary_csv: Path,
    ansor_target: str,
    selected_networks: List[str],
) -> Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, str],
    Dict[Tuple[str, str], Path],
]:
    selected_set = set(selected_networks)
    network_meta: Dict[str, Dict[str, object]] = {}
    workload_to_network: Dict[str, str] = {}
    best_log_by_task: Dict[Tuple[str, str], Tuple[int, Path]] = {}

    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "task":
                continue
            if row.get("target") != ansor_target:
                continue
            net_id = row.get("network_id", "")
            if not net_id or (selected_set and net_id not in selected_set):
                continue
            wk = normalize_workload_key(row.get("workload_key", ""))
            if not wk:
                continue

            network_meta.setdefault(
                net_id,
                {
                    "network_name": row.get("network_name", net_id),
                    "network_shape": parse_network_shape(row.get("network_shape", "[]")),
                },
            )
            workload_to_network[wk] = net_id

            log_path = resolve_ansor_log_path(summary_csv, row.get("log_path", ""))
            if log_path is None:
                continue
            try:
                budget = int(float(row.get("budget", "0") or 0))
            except Exception:
                budget = 0
            key = (net_id, wk)
            prev = best_log_by_task.get(key)
            if prev is None or budget > prev[0]:
                best_log_by_task[key] = (budget, log_path)

    ansor_task_logs = {key: path for key, (_, path) in best_log_by_task.items()}
    return network_meta, workload_to_network, ansor_task_logs


def load_method_sequences(
    measured_log: Path,
    workload_to_network: Dict[str, str],
    max_per_workload: Optional[int] = None,
    progress_label: str = "",
    progress_every: int = 200000,
) -> Dict[str, Dict[str, List[Tuple[object, object]]]]:
    out: Dict[str, Dict[str, List[Tuple[object, object]]]] = defaultdict(lambda: defaultdict(list))
    seen = 0
    kept = 0
    per_wk_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for inp, res in auto_scheduler.RecordReader(str(measured_log)):
        seen += 1
        wk = normalize_workload_key(inp.task.workload_key)
        net_id = workload_to_network.get(wk)
        if net_id is None:
            if progress_every > 0 and seen % progress_every == 0:
                print(f"[LOAD_SEQ] {progress_label} seen={seen} kept={kept}")
            continue
        count_key = (net_id, wk)
        if max_per_workload is not None and per_wk_counts[count_key] >= max_per_workload:
            if progress_every > 0 and seen % progress_every == 0:
                print(f"[LOAD_SEQ] {progress_label} seen={seen} kept={kept}")
            continue
        out[net_id][wk].append((inp, res))
        per_wk_counts[count_key] += 1
        kept += 1
        if progress_every > 0 and seen % progress_every == 0:
            print(f"[LOAD_SEQ] {progress_label} seen={seen} kept={kept}")
    print(f"[LOAD_SEQ_DONE] {progress_label} seen={seen} kept={kept}")
    return out


def load_ansor_sequences(
    ansor_task_logs: Dict[Tuple[str, str], Path],
    progress_every_tasks: int = 100,
) -> Dict[str, Dict[str, List[Tuple[object, object]]]]:
    out: Dict[str, Dict[str, List[Tuple[object, object]]]] = defaultdict(lambda: defaultdict(list))
    total_tasks = len(ansor_task_logs)
    for task_idx, ((net_id, wk), path) in enumerate(ansor_task_logs.items(), start=1):
        seq: List[Tuple[object, object]] = []
        try:
            for inp, res in auto_scheduler.RecordReader(str(path)):
                seq.append((inp, res))
        except Exception:
            seq = []
        out[net_id][wk] = seq
        if progress_every_tasks > 0 and task_idx % progress_every_tasks == 0:
            print(f"[LOAD_ANSOR] tasks={task_idx}/{total_tasks}")
    print(f"[LOAD_ANSOR_DONE] tasks={total_tasks}")
    return out


def build_prefix_records(
    network_sequences: Dict[str, List[Tuple[object, object]]],
    k: int,
) -> Tuple[List[object], List[object], int]:
    inputs: List[object] = []
    results: List[object] = []
    ok_count = 0
    for wk in sorted(network_sequences.keys()):
        take = network_sequences[wk][:k]
        for inp, res in take:
            inputs.append(inp)
            results.append(res)
            if getattr(res, "error_no", 0) == 0:
                ok_count += 1
    return inputs, results, ok_count


def write_prefix_log(path: Path, inputs: List[object], results: List[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not inputs:
        path.write_text("", encoding="utf-8")
        return
    auto_scheduler.save_records(str(path), inputs, results)


def parse_e2e_latency_ms(text: str) -> Optional[float]:
    patterns = [
        r"mean\s*\(ms\)\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        r"Mean\s*\(ms\)\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        r"mean[^0-9\n]*([0-9]*\.?[0-9]+)\s*ms",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def run_tune_relay_once(
    python_bin: str,
    tune_relay_path: Path,
    target: str,
    backend: str,
    use_auto_scheduler: bool,
    network_name: str,
    network_shape: List[int],
    log_path: Path,
    cuda_visible_devices: str,
    number: int,
    repeat: int,
    min_repeat_ms: int,
    timeout_sec: int,
) -> Tuple[int, float, Optional[float], str]:
    cmd = [
        python_bin,
        str(tune_relay_path),
        "--workload",
        network_name,
        "--input-shape",
        json.dumps(network_shape),
        "--target",
        target,
        "--backend",
        backend,
        "--use-auto-scheduler",
        "True" if use_auto_scheduler else "False",
        "--number",
        str(number),
        "--repeat",
        str(repeat),
        "--min-repeat-ms",
        str(min_repeat_ms),
    ]
    env = dict(os.environ)
    env["TLM_LOG_FILE"] = str(log_path)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    start = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    wall_time = time.time() - start
    e2e_ms = parse_e2e_latency_ms(proc.stdout or "")
    return proc.returncode, wall_time, e2e_ms, proc.stdout or ""


def save_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else f"{value:.6f}"
                else:
                    out[key] = value
            writer.writerow(out)


def load_prefix_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "method": str(row.get("method", "")),
                        "network_id": str(row.get("network_id", "")),
                        "k": int(float(row.get("k", "0") or 0)),
                        "records_total": int(float(row.get("records_total", "0") or 0)),
                        "records_ok": int(float(row.get("records_ok", "0") or 0)),
                        "log_path": str(row.get("log_path", "")),
                    }
                )
            except Exception:
                continue
    return rows


def load_existing_tune_rows(path: Path) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            net_id = row.get("network_id", "").strip()
            if not method or not net_id:
                continue
            try:
                k = int(float(row.get("k", "0") or 0))
            except Exception:
                continue
            key = (method, net_id, k)
            out[key] = dict(row)
    return out


def clone_tune_row_for_k(
    src: Dict[str, object],
    k: int,
    effective_k: int,
    mapped_from_k: str,
) -> Dict[str, object]:
    row = dict(src)
    row["k"] = k
    row["effective_k"] = effective_k
    row["mapped_from_k"] = mapped_from_k
    mode = str(row.get("mode", "") or "")
    if mapped_from_k:
        row["mode"] = f"{mode}|mapped" if mode else "mapped"
    return row


def parse_tune_log_resume(log_path: Path, records_total: int) -> Dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    mode = "skip_empty"
    status = "skip_empty"
    wall_time_s = 0.0
    e2e_ms = math.nan
    returncode = -1

    if not text.strip():
        if records_total > 0:
            status = "failed"
            mode = "empty_log"
            returncode = -1
        else:
            status = "skip_empty"
            mode = "skip_empty"
            returncode = 0
        return {
            "status": status,
            "mode": mode,
            "wall_time_s": wall_time_s,
            "e2e_latency_ms": e2e_ms,
            "returncode": returncode,
        }

    mode_match = re.search(r"\[BUILD_MODE\]\s*(\S+)", text)
    if mode_match:
        mode = mode_match.group(1)
    else:
        mode = "unknown"

    total_match = re.search(r"\|\s*Total\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", text)
    if total_match:
        wall_time_s = float(total_match.group(1)) * 60.0

    parsed_e2e = parse_e2e_latency_ms(text)
    if parsed_e2e is not None:
        e2e_ms = parsed_e2e
        if "fallback_topi" in mode:
            status = "fallback_topi"
        else:
            status = "ok"
        returncode = 0
    else:
        status = "failed"
        returncode = -1

    return {
        "status": status,
        "mode": mode,
        "wall_time_s": wall_time_s,
        "e2e_latency_ms": e2e_ms,
        "returncode": returncode,
    }


def bootstrap_existing_rows_from_logs(
    logs_dir: Path,
    prefix_rows: List[Dict[str, object]],
    tlm_max_k: int,
) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for item in prefix_rows:
        method = str(item["method"])
        net_id = str(item["network_id"])
        k = int(item["k"])
        records_total = int(item["records_total"])
        records_ok = int(item["records_ok"])
        prefix_log_path = str(item["log_path"])
        run_log_path = logs_dir / method / net_id / f"k{k}.log"
        if not run_log_path.exists():
            continue

        effective_k = k
        mapped_from_k = ""
        if method in ("official", "xavier") and tlm_max_k > 0 and k > tlm_max_k:
            effective_k = tlm_max_k
            mapped_from_k = str(tlm_max_k)

        parsed = parse_tune_log_resume(run_log_path, records_total=records_total)
        out[(method, net_id, k)] = {
            "method": method,
            "network_id": net_id,
            "network_name": "",
            "network_shape": "",
            "k": k,
            "effective_k": effective_k,
            "mapped_from_k": mapped_from_k,
            "records_total": records_total,
            "records_ok": records_ok,
            "mode": parsed["mode"],
            "wall_time_s": parsed["wall_time_s"],
            "auto_wall_time_s": parsed["wall_time_s"],
            "fallback_wall_time_s": 0.0,
            "e2e_latency_ms": parsed["e2e_latency_ms"],
            "returncode": parsed["returncode"],
            "auto_returncode": parsed["returncode"],
            "fallback_returncode": -1,
            "status": parsed["status"],
            "prefix_log_path": prefix_log_path,
            "run_log_path": str(run_log_path),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prefix-K logs and evaluate tuning time on Orin, aligned with latency_vs_k logic."
    )
    parser.add_argument("--official-measured", required=True)
    parser.add_argument("--lora-measured", required=True)
    parser.add_argument("--ansor-summary", required=True)
    parser.add_argument("--ansor-target", default="orin")
    parser.add_argument(
        "--networks",
        default="bert_base_1x128,resnet_50_1x3x224x224,mobilenet_v2_1x3x224x224,inception_v3_1x3x299x299",
    )
    parser.add_argument("--k-values", default="1-64,80,96,112,128,160,192,256,384,512,768,1000")
    parser.add_argument("--methods", default="official,xavier,ansor")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--build-prefix-only", action="store_true")
    parser.add_argument("--run-tune", action="store_true")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--tune-relay-path", default="")
    parser.add_argument("--target", default="cuda -keys=cuda,gpu -arch=sm_87 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32")
    parser.add_argument("--backend", default="graph")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--fallback-topi-on-fail", action="store_true")
    parser.add_argument(
        "--empty-topi-methods",
        default="official",
        help="Comma-separated methods that should run one TOPI-only tune when prefix log is empty (records_total=0), then reuse that result for larger K.",
    )
    parser.add_argument(
        "--tlm-max-k",
        type=int,
        default=64,
        help="For official/xavier, K values larger than this cap reuse capped-K tuning result.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    networks = parse_csv_list(args.networks)
    methods_requested = parse_csv_list(args.methods)
    k_values = parse_int_list(args.k_values)
    empty_topi_methods = set(parse_csv_list(args.empty_topi_methods))
    out_root = Path(args.out_root)
    prefix_dir = out_root / "prefix_logs"
    logs_dir = out_root / "tune_logs"
    prefix_csv = out_root / "prefix_logs_index.csv"
    prefix_meta = out_root / "prefix_meta.json"
    tune_meta = out_root / "tuning_meta.json"

    for need_path, desc in [
        (Path(args.official_measured), "official measured log"),
        (Path(args.lora_measured), "xavier measured log"),
        (Path(args.ansor_summary), "ansor summary csv"),
    ]:
        if not need_path.exists():
            raise FileNotFoundError(f"{desc} not found: {need_path}")

    if args.run_tune and not args.tune_relay_path:
        default_path = Path(__file__).resolve().parents[1] / "tune_relay.py"
        args.tune_relay_path = str(default_path)
    tune_relay_path = Path(args.tune_relay_path) if args.tune_relay_path else Path("")
    if args.run_tune and not tune_relay_path.exists():
        raise FileNotFoundError(f"tune_relay.py not found: {tune_relay_path}")

    network_meta, workload_to_network, ansor_task_logs = load_summary_meta(
        summary_csv=Path(args.ansor_summary),
        ansor_target=args.ansor_target,
        selected_networks=networks,
    )

    missing_meta = [net for net in networks if net not in network_meta]
    if missing_meta:
        raise RuntimeError(f"networks not found in ansor summary(target={args.ansor_target}): {missing_meta}")

    valid_methods = {"official", "xavier", "ansor"}
    methods = [m for m in methods_requested if m in valid_methods]
    invalid_methods = [m for m in methods_requested if m not in valid_methods]
    if invalid_methods:
        print(f"[WARN] Ignore unknown methods: {invalid_methods}")
    if not methods:
        raise RuntimeError("methods is empty after filtering valid method names.")

    max_k = max(k_values) if k_values else 0
    method_k_cap: Dict[str, int] = {}
    for method in methods:
        if method in ("official", "xavier") and args.tlm_max_k > 0:
            method_k_cap[method] = min(max_k, args.tlm_max_k)
        else:
            method_k_cap[method] = max_k

    expected_prefix_keys = {(method, net_id, k) for method in methods for net_id in networks for k in k_values}
    prefix_signature = {
        "version": SIGNATURE_VERSION,
        "phase": "prefix",
        "methods": methods,
        "networks": networks,
        "k_values": k_values,
        "ansor_target": args.ansor_target,
        "tlm_max_k": args.tlm_max_k,
        "source_files": {
            "official_measured": file_fingerprint(Path(args.official_measured)),
            "lora_measured": file_fingerprint(Path(args.lora_measured)),
            "ansor_summary": file_fingerprint(Path(args.ansor_summary)),
        },
    }

    prefix_rows: List[Dict[str, object]] = []
    can_resume_prefix = False
    if not args.force and prefix_csv.exists() and prefix_meta.exists():
        old_sig = load_meta_signature(prefix_meta)
        if old_sig is not None and to_signature_str(old_sig) == to_signature_str(prefix_signature):
            loaded_rows = load_prefix_rows(prefix_csv)
            loaded_keys = {(str(r["method"]), str(r["network_id"]), int(r["k"])) for r in loaded_rows}
            if loaded_keys == expected_prefix_keys:
                prefix_rows = loaded_rows
                can_resume_prefix = True
                print(f"[RESUME_PREFIX] reuse {prefix_csv} rows={len(prefix_rows)}")
            else:
                print(
                    f"[REBUILD_PREFIX] key mismatch loaded={len(loaded_keys)} expected={len(expected_prefix_keys)}"
                )
        else:
            print("[REBUILD_PREFIX] signature changed")

    if not can_resume_prefix:
        method_seq_map: Dict[str, Dict[str, Dict[str, List[Tuple[object, object]]]]] = {}
        for method in methods:
            load_start = time.time()
            if method == "official":
                method_seq_map[method] = load_method_sequences(
                    Path(args.official_measured),
                    workload_to_network,
                    max_per_workload=method_k_cap[method] if method_k_cap[method] > 0 else None,
                    progress_label=f"{method}",
                )
            elif method == "xavier":
                method_seq_map[method] = load_method_sequences(
                    Path(args.lora_measured),
                    workload_to_network,
                    max_per_workload=method_k_cap[method] if method_k_cap[method] > 0 else None,
                    progress_label=f"{method}",
                )
            else:
                method_seq_map[method] = load_ansor_sequences(ansor_task_logs)
            print(f"[LOAD_DONE] method={method} elapsed={time.time() - load_start:.2f}s")

        prefix_cache: Dict[Tuple[str, str, int], Tuple[int, int, str]] = {}
        for method in methods:
            per_method = method_seq_map[method]
            cap_k = method_k_cap.get(method, max_k)
            for net_idx, net_id in enumerate(networks, start=1):
                net_seq = per_method.get(net_id, {})
                print(f"[PREFIX_BUILD] method={method} net={net_id} ({net_idx}/{len(networks)})")
                for k in k_values:
                    source_k = k
                    if cap_k > 0 and method in ("official", "xavier") and k > cap_k:
                        source_k = cap_k
                    cache_key = (method, net_id, source_k)
                    if cache_key in prefix_cache:
                        total, ok, log_path_s = prefix_cache[cache_key]
                        prefix_rows.append(
                            {
                                "method": method,
                                "network_id": net_id,
                                "k": k,
                                "records_total": total,
                                "records_ok": ok,
                                "log_path": log_path_s,
                            }
                        )
                        continue

                    out_path = prefix_dir / method / net_id / f"k{source_k}.json"
                    if out_path.exists() and not args.force:
                        try:
                            total, ok = 0, 0
                            for _inp, res in auto_scheduler.RecordReader(str(out_path)):
                                total += 1
                                if getattr(res, "error_no", 0) == 0:
                                    ok += 1
                        except Exception:
                            total, ok = 0, 0
                    else:
                        inputs, results, ok = build_prefix_records(net_seq, source_k)
                        total = len(inputs)
                        write_prefix_log(out_path, inputs, results)
                    log_path_s = str(out_path)
                    prefix_cache[cache_key] = (total, ok, log_path_s)
                    prefix_rows.append(
                        {
                            "method": method,
                            "network_id": net_id,
                            "k": k,
                            "records_total": total,
                            "records_ok": ok,
                            "log_path": log_path_s,
                        }
                    )

        save_csv(
            prefix_csv,
            prefix_rows,
            ["method", "network_id", "k", "records_total", "records_ok", "log_path"],
        )
        save_meta(prefix_meta, prefix_signature, {"rows": len(prefix_rows)})
        print(f"[PREFIX] {prefix_csv}")
    else:
        print(f"[PREFIX] {prefix_csv} (reused)")

    if args.build_prefix_only or not args.run_tune:
        return

    tune_rows: List[Dict[str, object]] = []
    logs_dir.mkdir(parents=True, exist_ok=True)
    tune_csv = out_root / "tuning_time_results.csv"
    tune_fields = [
        "method",
        "network_id",
        "network_name",
        "network_shape",
        "k",
        "effective_k",
        "mapped_from_k",
        "records_total",
        "records_ok",
        "mode",
        "wall_time_s",
        "auto_wall_time_s",
        "fallback_wall_time_s",
        "e2e_latency_ms",
        "returncode",
        "auto_returncode",
        "fallback_returncode",
        "status",
        "prefix_log_path",
        "run_log_path",
    ]
    existing_tune_rows: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    tune_signature = {
        "version": SIGNATURE_VERSION,
        "phase": "tune",
        "prefix_signature": prefix_signature,
        "target": args.target,
        "backend": args.backend,
        "cuda_visible_devices": args.cuda_visible_devices,
        "number": args.number,
        "repeat": args.repeat,
        "min_repeat_ms": args.min_repeat_ms,
        "timeout_sec": args.timeout_sec,
        "fallback_topi_on_fail": bool(args.fallback_topi_on_fail),
        "empty_topi_methods": sorted(empty_topi_methods),
        "tlm_max_k": args.tlm_max_k,
        "tune_relay_path": str(tune_relay_path.resolve()) if tune_relay_path else "",
    }
    expected_tune_keys = expected_prefix_keys
    tune_signature_ok = False
    if not args.force:
        old_tune_sig = load_meta_signature(tune_meta)
        if old_tune_sig is not None and to_signature_str(old_tune_sig) == to_signature_str(tune_signature):
            tune_signature_ok = True
            existing_tune_rows = load_existing_tune_rows(tune_csv)
            print(f"[RESUME_TUNE_CSV] rows={len(existing_tune_rows)} csv={tune_csv}")
        else:
            if tune_csv.exists() or tune_meta.exists():
                print("[INVALIDATE_TUNE_CACHE] signature changed, ignore old tuning csv/log bootstrap")

    if not args.force and tune_signature_ok:
        have_keys = set(existing_tune_rows.keys())
        if not expected_tune_keys.issubset(have_keys):
            bootstrapped_rows = bootstrap_existing_rows_from_logs(
                logs_dir=logs_dir,
                prefix_rows=prefix_rows,
                tlm_max_k=args.tlm_max_k,
            )
            if bootstrapped_rows:
                before_cnt = len(existing_tune_rows)
                for key, row in bootstrapped_rows.items():
                    if key not in existing_tune_rows:
                        existing_tune_rows[key] = row
                added_cnt = len(existing_tune_rows) - before_cnt
                if added_cnt > 0:
                    print(
                        f"[BOOTSTRAP_TUNE] csv={before_cnt} + logs={len(bootstrapped_rows)} "
                        f"=> merged={len(existing_tune_rows)} (added {added_cnt})"
                    )
    if not args.force and tune_signature_ok:
        have_keys = set(existing_tune_rows.keys())
        if expected_tune_keys.issubset(have_keys):
            save_meta(tune_meta, tune_signature, {"rows": len(existing_tune_rows), "status": "complete"})
            print(f"[RESUME_TUNE] all keys already available, skip tune. csv={tune_csv}")
            return

    existing_empty_topi_rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    for (method, net_id, _k), row in existing_tune_rows.items():
        if method not in empty_topi_methods:
            continue
        total = int(float(row.get("records_total", "0") or 0))
        e2e = parse_float(row.get("e2e_latency_ms", ""))
        status = str(row.get("status", "") or "")
        if total != 0 or math.isnan(e2e):
            continue
        if status not in {"ok", "fallback_topi", "empty_topi", "empty_topi|mapped", "mapped"}:
            continue
        cache_key = (method, net_id)
        prev = existing_empty_topi_rows.get(cache_key)
        if prev is None:
            existing_empty_topi_rows[cache_key] = row
            continue
        prev_k = int(float(prev.get("k", "0") or 0))
        cur_k = int(float(row.get("k", "0") or 0))
        if cur_k < prev_k:
            existing_empty_topi_rows[cache_key] = row
    computed_tune_rows: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    empty_topi_cache_rows: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in prefix_rows:
        method = str(row["method"])
        net_id = str(row["network_id"])
        k = int(row["k"])
        total = int(row["records_total"])
        ok = int(row["records_ok"])
        log_path = Path(str(row["log_path"]))
        run_log_path = logs_dir / method / net_id / f"k{k}.log"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)

        network_name = str(network_meta[net_id]["network_name"])
        network_shape = list(network_meta[net_id]["network_shape"])

        effective_k = k
        mapped_from_k = ""
        if method in ("official", "xavier") and args.tlm_max_k > 0 and k > args.tlm_max_k:
            effective_k = args.tlm_max_k
            mapped_from_k = str(args.tlm_max_k)
        run_key = (method, net_id, k)
        source_key = (method, net_id, effective_k)

        if run_key in existing_tune_rows:
            resume_row = clone_tune_row_for_k(
                existing_tune_rows[run_key],
                k=k,
                effective_k=effective_k,
                mapped_from_k=mapped_from_k,
            )
            resume_row["network_name"] = network_name
            resume_row["network_shape"] = json.dumps(network_shape)
            tune_rows.append(resume_row)
            computed_tune_rows[run_key] = resume_row
            save_csv(tune_csv, tune_rows, tune_fields)
            print(
                f"[RESUME_TUNE] method={method} net={net_id} k={k} "
                f"status={resume_row.get('status', '')} mode={resume_row.get('mode', '')}"
            )
            continue

        if mapped_from_k:
            if source_key in computed_tune_rows:
                mapped_row = clone_tune_row_for_k(
                    computed_tune_rows[source_key],
                    k=k,
                    effective_k=effective_k,
                    mapped_from_k=mapped_from_k,
                )
                mapped_row["network_name"] = network_name
                mapped_row["network_shape"] = json.dumps(network_shape)
                tune_rows.append(mapped_row)
                computed_tune_rows[run_key] = mapped_row
                save_csv(tune_csv, tune_rows, tune_fields)
                print(f"[MAP_TUNE] method={method} net={net_id} k={k} -> k{effective_k}")
                continue
            if source_key in existing_tune_rows:
                mapped_row = clone_tune_row_for_k(
                    existing_tune_rows[source_key],
                    k=k,
                    effective_k=effective_k,
                    mapped_from_k=mapped_from_k,
                )
                mapped_row["network_name"] = network_name
                mapped_row["network_shape"] = json.dumps(network_shape)
                tune_rows.append(mapped_row)
                computed_tune_rows[run_key] = mapped_row
                save_csv(tune_csv, tune_rows, tune_fields)
                print(f"[MAP_TUNE] method={method} net={net_id} k={k} -> k{effective_k} (from existing csv)")
                continue

        status = "skip_empty"
        returncode = -1
        auto_returncode = -1
        fallback_returncode = -1
        wall_time = 0.0
        auto_wall_time = 0.0
        fallback_wall_time = 0.0
        e2e_ms = math.nan
        mode = "skip_empty"
        stdout = ""

        if total <= 0 and method in empty_topi_methods:
            cache_key = (method, net_id)
            cached_row = empty_topi_cache_rows.get(cache_key)
            if cached_row is None and cache_key in existing_empty_topi_rows:
                cached_row = existing_empty_topi_rows[cache_key]

            if cached_row is not None:
                mapped_k = int(float(cached_row.get("k", "0") or 0))
                mapped_row = clone_tune_row_for_k(
                    cached_row,
                    k=k,
                    effective_k=mapped_k if mapped_k > 0 else effective_k,
                    mapped_from_k=str(mapped_k) if mapped_k > 0 else "",
                )
                mapped_row["network_name"] = network_name
                mapped_row["network_shape"] = json.dumps(network_shape)
                mapped_row["records_total"] = total
                mapped_row["records_ok"] = ok
                mapped_row["prefix_log_path"] = str(log_path)
                mapped_row["run_log_path"] = str(run_log_path)
                tune_rows.append(mapped_row)
                computed_tune_rows[run_key] = mapped_row
                empty_topi_cache_rows[cache_key] = mapped_row
                save_csv(tune_csv, tune_rows, tune_fields)
                print(
                    f"[MAP_EMPTY_TOPI] method={method} net={net_id} k={k} "
                    f"-> k{mapped_k if mapped_k > 0 else 'cached'}"
                )
                continue

            fallback_returncode, fallback_wall_time, fallback_e2e, fallback_stdout = run_tune_relay_once(
                python_bin=args.python_bin,
                tune_relay_path=tune_relay_path,
                target=args.target,
                backend=args.backend,
                use_auto_scheduler=False,
                network_name=network_name,
                network_shape=network_shape,
                log_path=log_path,
                cuda_visible_devices=args.cuda_visible_devices,
                number=args.number,
                repeat=args.repeat,
                min_repeat_ms=args.min_repeat_ms,
                timeout_sec=args.timeout_sec,
            )
            wall_time = fallback_wall_time
            fallback_wall_time = fallback_wall_time
            stdout = fallback_stdout
            mode = "empty_topi"
            returncode = fallback_returncode
            if fallback_e2e is not None:
                e2e_ms = fallback_e2e
            status = "ok" if fallback_returncode == 0 else "failed"
        elif total > 0:
            auto_returncode, auto_wall_time, parsed_e2e, auto_stdout = run_tune_relay_once(
                python_bin=args.python_bin,
                tune_relay_path=tune_relay_path,
                target=args.target,
                backend=args.backend,
                use_auto_scheduler=True,
                network_name=network_name,
                network_shape=network_shape,
                log_path=log_path,
                cuda_visible_devices=args.cuda_visible_devices,
                number=args.number,
                repeat=args.repeat,
                min_repeat_ms=args.min_repeat_ms,
                timeout_sec=args.timeout_sec,
            )
            wall_time += auto_wall_time
            if parsed_e2e is not None:
                e2e_ms = parsed_e2e
            stdout = auto_stdout
            mode = "auto_scheduler"
            if auto_returncode == 0:
                status = "ok"
                returncode = auto_returncode
            elif args.fallback_topi_on_fail:
                fallback_returncode, fallback_wall_time, fallback_e2e, fallback_stdout = run_tune_relay_once(
                    python_bin=args.python_bin,
                    tune_relay_path=tune_relay_path,
                    target=args.target,
                    backend=args.backend,
                    use_auto_scheduler=False,
                    network_name=network_name,
                    network_shape=network_shape,
                    log_path=log_path,
                    cuda_visible_devices=args.cuda_visible_devices,
                    number=args.number,
                    repeat=args.repeat,
                    min_repeat_ms=args.min_repeat_ms,
                    timeout_sec=args.timeout_sec,
                )
                wall_time += fallback_wall_time
                stdout = (
                    "[AUTO_SCHEDULER]\n"
                    + auto_stdout
                    + "\n\n[FALLBACK_TOPI]\n"
                    + fallback_stdout
                )
                if fallback_e2e is not None:
                    e2e_ms = fallback_e2e
                if fallback_returncode == 0:
                    status = "fallback_topi"
                    mode = "fallback_topi"
                    returncode = 0
                else:
                    status = "failed"
                    mode = "failed_both"
                    returncode = fallback_returncode
            else:
                status = "failed"
                returncode = auto_returncode

        run_log_path.write_text(stdout, encoding="utf-8")
        tune_rows.append(
            {
                "method": method,
                "network_id": net_id,
                "network_name": network_name,
                "network_shape": json.dumps(network_shape),
                "k": k,
                "effective_k": effective_k,
                "mapped_from_k": mapped_from_k,
                "records_total": total,
                "records_ok": ok,
                "mode": mode,
                "wall_time_s": wall_time,
                "auto_wall_time_s": auto_wall_time,
                "fallback_wall_time_s": fallback_wall_time,
                "e2e_latency_ms": e2e_ms,
                "returncode": returncode,
                "auto_returncode": auto_returncode,
                "fallback_returncode": fallback_returncode,
                "status": status,
                "prefix_log_path": str(log_path),
                "run_log_path": str(run_log_path),
            }
        )
        computed_tune_rows[run_key] = tune_rows[-1]
        if total <= 0 and method in empty_topi_methods and status == "ok":
            empty_topi_cache_rows[(method, net_id)] = tune_rows[-1]
        save_csv(tune_csv, tune_rows, tune_fields)
        print(
            f"[TUNE] method={method} net={net_id} k={k} total={total} "
            f"status={status} mode={mode} wall={wall_time:.2f}s "
            f"e2e={'' if math.isnan(e2e_ms) else f'{e2e_ms:.4f}ms'}"
        )

    save_csv(tune_csv, tune_rows, tune_fields)
    save_meta(tune_meta, tune_signature, {"rows": len(tune_rows), "status": "complete"})
    print(f"[TUNE_CSV] {tune_csv}")


if __name__ == "__main__":
    main()
