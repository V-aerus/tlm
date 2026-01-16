#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_train_sketch.sh <idx> [hw]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  hw: 4090|v100|all (optional, default: 4090 or EDGE_HW)"
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX (expect non-negative integer)"
  exit 1
fi

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  echo "Set EDGE_PATHS_SH to your paths.sh location."
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

ITER=$(printf "iter%02d" "$IDX")
TEST_FILE_IDX="$IDX"
KEEP_CNT="${EDGE_KEEP_CNT:-48}"

HW_ARG="${2:-${EDGE_HW:-4090}}"

run_sketch() {
  local hw_id="$1"
  local target="$2"
  local dataset_path="$3"

  local out_dir="$RUN_ROOT/${hw_id}/${ITER}/sketch"
  mkdir -p "$out_dir"
  python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_train_sketch \
    --target "$target" \
    --dataset_path "$dataset_path" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$out_dir" \
    --keep_cnt="$KEEP_CNT" \
    --test_file_idx="$TEST_FILE_IDX"
}

if [[ "$HW_ARG" == "all" ]]; then
  run_sketch "4090" "$TARGET_4090" "$DATA_ROOT/dataset/to_measure_programs/4090"
  run_sketch "v100" "$TARGET_V100" "$DATA_ROOT/dataset/to_measure_programs/v100"
else
  case "$HW_ARG" in
    4090) run_sketch "4090" "$TARGET_4090" "$DATA_ROOT/dataset/to_measure_programs/4090" ;;
    v100) run_sketch "v100" "$TARGET_V100" "$DATA_ROOT/dataset/to_measure_programs/v100" ;;
    *) echo "Invalid hw: $HW_ARG (expect 4090|v100|all)"; exit 1 ;;
  esac
fi
