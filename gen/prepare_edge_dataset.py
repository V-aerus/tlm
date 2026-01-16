#!/usr/bin/env python3
"""Prepare EdgeTLM SFT dataset with hardware embeddings and baseline latency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import DatasetDict
from transformers import AutoTokenizer


EMBEDDING_PATH_DEFAULT = Path("Embedding/hardware_embeddings_v4.json")
KNOWN_DEFAULTS = {
    "v100": ["nvidia/nvidia-v100"],
    "4090": ["nvidia/rtx-4090", "nvidia/nvidia-a40"],  # sm_86 (v4 uses rtx-4090)
    "xavier": ["nvidia/jetson-agx-xavier"],
    "xeon": ["aws/cpu/c5.18xlarge"],
}


def load_embeddings(embedding_path: Path) -> Dict[str, List[float]]:
    with embedding_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["hardware_name"]: entry["vector"] for entry in entries}


def detect_hardware(target_str: str) -> Tuple[str, str]:
    """Return (hardware_id, hardware_name_for_embedding)."""
    lower = target_str.lower()
    if "sm_70" in lower or "v100" in lower:
        return "v100", "nvidia/nvidia-v100"
    if "sm_86" in lower or "4090" in lower:
        return "4090", "nvidia/rtx-4090"
    if "jetson" in lower or "sm_72" in lower or "xavier" in lower:
        return "xavier", "nvidia/jetson-agx-xavier"
    if "llvm" in lower or "skylake" in lower or "xeon" in lower:
        return "xeon", "aws/cpu/c5.18xlarge"
    raise ValueError(f"Unsupported hardware target string: {target_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EdgeTLM training dataset with baseline latency and hardware embeddings.")
    parser.add_argument("--sft-dataset-path", default=None, help="Path to HuggingFace dataset folder (e.g., all_gen_best_multi).")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--embedding-json",
        default=str(EMBEDDING_PATH_DEFAULT),
        help="Hardware embedding json file (default: Embedding/hardware_embeddings_v4.json).",
    )
    parser.add_argument("--allow-missing-lora", action="store_true", help="If set, lat_lora_star will be None; otherwise defaults to lat_base_star.")
    parser.add_argument("--base-jsonl", default=None, help="Optional JSONL containing base measurements (produced by prepare_edge_dataset.py).")
    parser.add_argument("--lora-jsonl", default=None, help="Optional JSONL containing LoRA measurements (e.g., postprocess + make_dataset output).")
    parser.add_argument("--teacher-jsonl", default=None, help="Optional JSONL providing teacher text (e.g., all-mode best).")
    parser.add_argument(
        "--merge_mode",
        choices=("base_left", "lora_left"),
        default="base_left",
        help="JSONL merge mode: base_left keeps base entries and fills LoRA when available; lora_left keeps LoRA entries and fills base when available.",
    )
    parser.add_argument(
        "--merge_key",
        choices=("id", "repr"),
        default="repr",
        help="Merge key for base/lora JSONL: 'id' uses workload_id, 'repr' uses workload_repr (shape-aware).",
    )
    parser.add_argument(
        "--dedupe_mode",
        choices=("min", "keep_all"),
        default="keep_all",
        help="JSONL merge output mode: 'min' keeps one record per key (best latency). "
             "'keep_all' outputs all records but uses per-key min latency for *_star fields.",
    )
    parser.add_argument(
        "--hardware-id",
        default=None,
        help="Optional hardware id filter (e.g., v100). Use comma to select multiple; leave empty to export all.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used to decode samples when dataset does not include raw text.",
    )
    args = parser.parse_args()

    embeddings = load_embeddings(Path(args.embedding_json))

    hardware_filters: Optional[Iterable[str]] = None
    if args.hardware_id:
        hardware_filters = {item.strip() for item in args.hardware_id.split(",") if item.strip()}

    using_dataset = args.sft_dataset_path is not None
    using_jsonl_pair = args.base_jsonl is not None or args.lora_jsonl is not None
    using_teacher = args.teacher_jsonl is not None

    if using_jsonl_pair:
        if not (args.base_jsonl and args.lora_jsonl):
            raise ValueError("Both --base-jsonl and --lora-jsonl must be provided when using JSONL merge mode.")
        if args.sft_dataset_path is not None:
            print("[prepare_edge_dataset] Warning: --sft-dataset-path ignored because --base-jsonl/--lora-jsonl are provided.")
    elif not using_dataset:
        raise ValueError("Provide either --sft-dataset-path or both --base-jsonl/--lora-jsonl.")

    tokenizer = None
    dataset = None
    latency_field = None
    if using_dataset:
        sft_path = Path(args.sft_dataset_path)
        if not sft_path.exists():
            raise FileNotFoundError(f"SFT dataset path not found: {sft_path}")
        dataset = DatasetDict.load_from_disk(str(sft_path))["train"]
        requires_decode = "text" not in dataset.features
        if requires_decode:
            if not args.tokenizer_path:
                raise ValueError(
                    "Dataset does not contain 'text'. Provide --tokenizer-path to decode input_ids "
                    "(e.g., the tokenizer used in make_dataset.py --for_gen_best)."
                )
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        latency_field = "latency" if "latency" in dataset.features else None
    else:
        if args.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def parse_line(line_str: str) -> Tuple[str, str, Dict, str]:
        line_json = json.loads(line_str)
        if "i" not in line_json or not line_json["i"]:
            raise ValueError("Malformed line field without 'i'.")
        workload_repr = line_json["i"][0][0]
        target_str = line_json["i"][0][1]
        workload = json.loads(workload_repr)
        workload_id = workload[0]
        return workload_repr, target_str, line_json, workload_id

    def normalize_workload_repr(workload_repr: str) -> str:
        try:
            workload_obj = json.loads(workload_repr)
        except json.JSONDecodeError:
            return workload_repr
        return json.dumps(workload_obj, separators=(",", ":"), ensure_ascii=True)

    teacher_map = None
    if using_teacher:
        def build_teacher_key(line_str: str) -> Tuple[str, str]:
            workload_repr, target_str, _, workload_id = parse_line(line_str)
            if args.merge_key == "repr":
                return (normalize_workload_repr(workload_repr), target_str)
            return (workload_id, target_str)

        teacher_map = {}
        with open(args.teacher_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                line_str = rec.get("line")
                if not line_str:
                    continue
                key = build_teacher_key(line_str)
                if key not in teacher_map:
                    teacher_map[key] = rec

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    missing_lora = 0
    missing_base = 0
    paired_count = 0
    lora_only_count = 0

    def resolve_hw_info(line_str: str, base_record: Dict) -> Tuple[str, str, List[float]]:
        workload_repr, target_str, _, workload_id = parse_line(line_str)
        hw_id, hw_name = detect_hardware(target_str)

        # Prefer explicit hardware fields if present
        hw_id = base_record.get("hardware_id", hw_id)
        hw_name = base_record.get("hardware_name", hw_name)

        emb = base_record.get("hw_emb")
        if emb is None:
            lookup_name = hw_name
            if lookup_name not in embeddings and hw_id in KNOWN_DEFAULTS:
                for fallback in KNOWN_DEFAULTS[hw_id]:
                    if fallback in embeddings:
                        lookup_name = fallback
                        break
            if lookup_name not in embeddings:
                raise KeyError(f"Hardware embedding for '{hw_name}' not found in {args.embedding_json}")
            emb = embeddings[lookup_name]
        return hw_id, hw_name, emb

    with output_path.open("w", encoding="utf-8") as out_f:
        if using_dataset:
            for record in dataset:
                line_str = record.get("line")
                if not line_str:
                    continue
                _, _, line_json, _ = parse_line(line_str)
                hw_id, hw_name, hw_emb = resolve_hw_info(line_str, record)
                if hardware_filters and hw_id not in hardware_filters:
                    continue
                stats[hw_id] = stats.get(hw_id, 0) + 1

                if latency_field:
                    lat_base = float(record[latency_field])
                else:
                    raise KeyError("Dataset does not contain 'latency'. Provide dataset with baseline latency column.")
                lat_lora = None if args.allow_missing_lora else lat_base

                if "text" in record:
                    text = record["text"]
                else:
                    if tokenizer is None:
                        raise ValueError("Tokenizer required to decode samples without 'text'.")
                    input_ids = record["input_ids"]
                    text = tokenizer.decode(input_ids, skip_special_tokens=True)

                out_record = {
                    "text": text,
                    "hardware_id": hw_id,
                    "hardware_name": hw_name,
                    "hw_emb": hw_emb,
                    "lat_base_star": lat_base,
                    "lat_lora_star": lat_lora,
                    "line": record["line"],
                }
                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        else:
            collisions_by_id: Dict[Tuple[str, str], Dict[str, str]] = {}

            def record_collision(workload_id: str, target_str: str, workload_repr: str) -> None:
                try:
                    shapes = json.loads(workload_repr)[1:]
                    shape_repr = json.dumps(shapes, separators=(",", ":"), ensure_ascii=True)
                except Exception:
                    shape_repr = workload_repr
                bucket = collisions_by_id.setdefault((workload_id, target_str), {})
                if workload_repr not in bucket:
                    bucket[workload_repr] = shape_repr

            def build_merge_key(workload_repr: str, workload_id: str, target_str: str) -> Tuple[str, str]:
                if args.merge_key == "repr":
                    return (normalize_workload_repr(workload_repr), target_str)
                return (workload_id, target_str)

            def load_jsonl(path: str, is_lora: bool):
                best_records: Dict[Tuple[str, str], Dict] = {}
                best_latency: Dict[Tuple[str, str], float] = {}
                all_records: List[Tuple[Tuple[str, str], Dict]] = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        line_str = rec.get("line")
                        if not line_str:
                            continue
                        workload_repr, target_str, _, workload_id = parse_line(line_str)
                        norm_repr = normalize_workload_repr(workload_repr)
                        record_collision(workload_id, target_str, norm_repr)
                        key = build_merge_key(norm_repr, workload_id, target_str)
                        latency = rec.get("lat_lora_star" if is_lora else "lat_base_star", rec.get("latency"))
                        if latency is None:
                            continue
                        latency = float(latency)
                        all_records.append((key, rec))
                        prev_best = best_latency.get(key)
                        if prev_best is None or latency < prev_best:
                            best_latency[key] = latency
                            best_records[key] = rec
                return best_records, all_records, best_latency

            base_map, base_records, base_min = load_jsonl(args.base_jsonl, is_lora=False)
            lora_map, lora_records, lora_min = load_jsonl(args.lora_jsonl, is_lora=True)
            if args.merge_mode == "base_left" and not base_map:
                raise ValueError(f"No valid records found in base jsonl: {args.base_jsonl}")
            if args.merge_mode == "lora_left" and not lora_map:
                raise ValueError(f"No valid records found in lora jsonl: {args.lora_jsonl}")
            if args.merge_mode == "lora_left" and not base_map:
                print("[prepare_edge_dataset] Warning: base jsonl is empty; lat_base_star will be null for all outputs.")

            collisions = [
                (wid, target, variants) for (wid, target), variants in collisions_by_id.items() if len(variants) > 1
            ]
            if collisions:
                print(
                    f"[prepare_edge_dataset] Detected {len(collisions)} workload_id collisions (multiple shapes per id)."
                )
                collisions.sort(key=lambda x: len(x[2]), reverse=True)
                for (wid, target), variants in [((w, t), v) for w, t, v in collisions[:20]]:
                    shapes_list = list(variants.values())
                    print(f"  - workload_id={wid} target={target} variants={len(shapes_list)} shapes={shapes_list[:6]}")
                print(
                    f"[prepare_edge_dataset] merge_key='{args.merge_key}'; use --merge_key=repr to avoid shape mismatches."
                )
            if args.merge_mode == "base_left":
                base_iter = base_map.items() if args.dedupe_mode == "min" else base_records
                for key, base_rec in base_iter:
                    line_str = base_rec["line"]
                    workload_repr, target_str, line_json, _ = parse_line(line_str)
                    hw_id, hw_name, hw_emb = resolve_hw_info(line_str, base_rec)
                    if hardware_filters and hw_id not in hardware_filters:
                        continue

                    lat_base = base_min.get(key)
                    if lat_base is None:
                        raise KeyError(f"Base record missing latency information for workload: {key}")
                    lat_base = float(lat_base)

                    if key in lora_min:
                        lat_lora = float(lora_min[key])
                        paired_count += 1
                    else:
                        lat_lora = None if args.allow_missing_lora else lat_base
                        if lat_lora is None:
                            missing_lora += 1

                    text = None
                    if teacher_map and key in teacher_map:
                        text = teacher_map[key].get("text")
                    if text is None:
                        text = base_rec.get("text")
                    if text is None:
                        if tokenizer is None:
                            raise ValueError("Tokenizer required to decode samples without 'text'.")
                        input_ids = None
                        if teacher_map and key in teacher_map:
                            input_ids = teacher_map[key].get("input_ids")
                        if input_ids is None:
                            input_ids = base_rec.get("input_ids")
                        if input_ids is None:
                            raise ValueError("Base record missing both 'text' and 'input_ids'.")
                        text = tokenizer.decode(input_ids, skip_special_tokens=True)

                    stats[hw_id] = stats.get(hw_id, 0) + 1
                    out_rec = {
                        "text": text,
                        "hardware_id": hw_id,
                        "hardware_name": hw_name,
                        "hw_emb": hw_emb,
                        "lat_base_star": lat_base,
                        "lat_lora_star": lat_lora,
                        "line": line_str,
                    }
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            else:
                lora_iter = lora_map.items() if args.dedupe_mode == "min" else lora_records
                for key, lora_rec in lora_iter:
                    line_str = lora_rec["line"]
                    workload_repr, target_str, line_json, _ = parse_line(line_str)
                    hw_id, hw_name, hw_emb = resolve_hw_info(line_str, lora_rec)
                    if hardware_filters and hw_id not in hardware_filters:
                        continue

                    lat_lora = lora_min.get(key)
                    if lat_lora is None:
                        continue
                    lat_lora = float(lat_lora)

                    if key in base_min:
                        lat_base = float(base_min[key])
                        paired_count += 1
                    else:
                        lat_base = None
                        missing_base += 1
                        lora_only_count += 1

                    text = None
                    if teacher_map and key in teacher_map:
                        text = teacher_map[key].get("text")
                    if text is None:
                        text = lora_rec.get("text")
                    if text is None:
                        if tokenizer is None:
                            raise ValueError("Tokenizer required to decode samples without 'text'.")
                        input_ids = None
                        if teacher_map and key in teacher_map:
                            input_ids = teacher_map[key].get("input_ids")
                        if input_ids is None:
                            input_ids = lora_rec.get("input_ids")
                        if input_ids is None:
                            raise ValueError("LoRA record missing both 'text' and 'input_ids'.")
                        text = tokenizer.decode(input_ids, skip_special_tokens=True)

                    stats[hw_id] = stats.get(hw_id, 0) + 1
                    out_rec = {
                        "text": text,
                        "hardware_id": hw_id,
                        "hardware_name": hw_name,
                        "hw_emb": hw_emb,
                        "lat_base_star": lat_base,
                        "lat_lora_star": lat_lora,
                        "line": line_str,
                    }
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    total_written = sum(stats.values())
    print(f"Wrote {total_written} samples to {output_path}")
    for hw_id, count in stats.items():
        print(f"  - {hw_id}: {count}")
    if using_jsonl_pair:
        print(f"  paired_count={paired_count}")
        if args.merge_mode == "base_left":
            if args.allow_missing_lora and missing_lora:
                print(f"  Note: {missing_lora} samples are missing LoRA measurements (lat_lora_star=None).")
        else:
            print(f"  lora_only_count={lora_only_count}")
            if missing_base:
                print(f"  Note: {missing_base} samples are missing base measurements (lat_base_star=None).")


if __name__ == "__main__":
    main()
