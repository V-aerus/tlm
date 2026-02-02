#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash gen/scripts/run_router_calibration.sh <dataset_jsonl> <output_root> <label=expert_dir> [label=expert_dir]..."
  exit 1
fi

TLM_ROOT="${TLM_ROOT:-/home/hehangshuai/workspace/tlm}"

DATASET_JSONL="$1"
OUTPUT_ROOT="$2"
shift 2

EXPERT_ARGS=()
for arg in "$@"; do
  EXPERT_ARGS+=(--expert "$arg")
done

python "$TLM_ROOT/gen/train_router_calibration.py" \
  --dataset-jsonl "$DATASET_JSONL" \
  --output-root "$OUTPUT_ROOT" \
  "${EXPERT_ARGS[@]}"
