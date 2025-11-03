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


EMBEDDING_PATH_DEFAULT = Path("Embedding/hardware_embeddings_v2.json")
KNOWN_DEFAULTS = {
    "v100": "nvidia/nvidia-v100",
    "4090": "nvidia/nvidia-a40",  # sm_86
    "xavier": "nvidia/jetson-agx-xavier",
    "xeon": "aws/cpu/c5.18xlarge",
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
        return "4090", "nvidia/nvidia-a40"
    if "jetson" in lower or "sm_72" in lower or "xavier" in lower:
        return "xavier", "nvidia/jetson-agx-xavier"
    if "llvm" in lower or "skylake" in lower or "xeon" in lower:
        return "xeon", "aws/cpu/c5.18xlarge"
    raise ValueError(f"Unsupported hardware target string: {target_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EdgeTLM training dataset with baseline latency and hardware embeddings.")
    parser.add_argument("--sft-dataset-path", required=True, help="Path to HuggingFace dataset folder (e.g., all_gen_best_multi).")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file path.")
    parser.add_argument("--embedding-json", default=str(EMBEDDING_PATH_DEFAULT), help="Hardware embedding json file.")
    parser.add_argument("--allow-missing-lora", action="store_true", help="If set, lat_lora_star will be None; otherwise defaults to lat_base_star.")
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

    sft_path = Path(args.sft_dataset_path)
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT dataset path not found: {sft_path}")

    embeddings = load_embeddings(Path(args.embedding_json))

    dataset = DatasetDict.load_from_disk(str(sft_path))["train"]

    tokenizer = None
    requires_decode = "text" not in dataset.features
    if requires_decode:
        if not args.tokenizer_path:
            raise ValueError("Dataset does not contain 'text'. Provide --tokenizer-path to decode input_ids.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    hardware_filters: Optional[Iterable[str]] = None
    if args.hardware_id:
        hardware_filters = {item.strip() for item in args.hardware_id.split(",") if item.strip()}

    latency_field = "latency" if "latency" in dataset.features else None

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in dataset:
            line_json = json.loads(record.get("line", "{}"))
            if "i" not in line_json:
                continue
            target_str = line_json["i"][0][1]
            hw_id, hw_name = detect_hardware(target_str)
            if hardware_filters and hw_id not in hardware_filters:
                continue
            stats[hw_id] = stats.get(hw_id, 0) + 1

            if hw_name not in embeddings:
                if hw_id in KNOWN_DEFAULTS and KNOWN_DEFAULTS[hw_id] in embeddings:
                    hw_name = KNOWN_DEFAULTS[hw_id]
                else:
                    raise KeyError(f"Hardware embedding for '{hw_name}' not found in {args.embedding_json}")

            if latency_field:
                lat_base = float(record[latency_field])
            else:
                raise KeyError("Dataset does not contain 'latency'. Provide dataset with baseline latency column.")
            lat_lora = None if args.allow_missing_lora else lat_base

            if "text" in record:
                text = record["text"]
            else:
                input_ids = record["input_ids"]
                text = tokenizer.decode(input_ids, skip_special_tokens=True)

            out_record = {
                "text": text,
                "hardware_id": hw_id,
                "hardware_name": hw_name,
                "hw_emb": embeddings[hw_name],
                "lat_base_star": lat_base,
                "lat_lora_star": lat_lora,
                "line": record["line"],
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Wrote {sum(stats.values())} samples to {output_path}")
    for hw_id, count in stats.items():
        print(f"  - {hw_id}: {count}")


if __name__ == "__main__":
    main()
