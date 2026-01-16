#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_measure.sh <idx> [hw] [mode]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  hw: 4090|v100 (optional, default: 4090 or EDGE_HW)"
  echo "  mode: base|kv_lora (optional, default: base)"
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX (expect non-negative integer)"
  exit 1
fi

HW_ARG="${2:-${EDGE_HW:-4090}}"
MODE="${3:-${EDGE_MEASURE_MODE:-base}}"
case "$MODE" in
  base|kv_lora) ;;
  *) echo "Invalid mode: $MODE (expect base|kv_lora)"; exit 1 ;;
esac

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  echo "Set EDGE_PATHS_SH to your paths.sh location."
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

ITER=$(printf "iter%02d" "$IDX")
CUDA_ID="${EDGE_CUDA:-0}"
BATCH_SIZE="${EDGE_MEASURE_BATCH_SIZE:-64}"

run_measure() {
  local hw_id="$1"
  local target="$2"
  local gen_dir="$RUN_ROOT/${hw_id}/${ITER}/gen"
  local measure_dir="$RUN_ROOT/${hw_id}/${ITER}/measure"
  local log_dir="$RUN_ROOT/${hw_id}/${ITER}/logs"
  local gen_file=""
  local out_file=""
  local log_file=""

  if [[ "$MODE" == "base" ]]; then
    gen_file="$gen_dir/base_bucketkv.json"
    out_file="$measure_dir/base_bucketkv.json"
    log_file="$log_dir/measure_base_bucketkv.log"
  else
    gen_file="$gen_dir/kv_lora.json"
    out_file="$measure_dir/kv_lora.json"
    log_file="$log_dir/measure_kv_lora.log"
  fi

  if [[ ! -f "$gen_file" ]]; then
    echo "Missing gen file: $gen_file"
    exit 1
  fi

  mkdir -p "$measure_dir" "$log_dir"
  if [[ -f "$out_file" && "${EDGE_FORCE:-0}" != "1" ]]; then
    echo "Measured file exists: $out_file (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/measure_programs.py" \
    --batch-size "$BATCH_SIZE" \
    --target "$target" \
    --to-measure-path "$gen_file" \
    --measured-path "$out_file" \
    | tee "$log_file"
}

case "$HW_ARG" in
  4090) run_measure "4090" "$TARGET_4090" ;;
  v100) run_measure "v100" "$TARGET_V100" ;;
  *) echo "Invalid hw: $HW_ARG (expect 4090|v100)"; exit 1 ;;
esac
