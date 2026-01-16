#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_gen_kv_lora.sh <idx> [expert_dir] [hw]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  expert_dir: LoRA expert dir with router.json (optional if EDGE_EXPERT_DIR is set)"
  echo "  hw: 4090|v100 (optional, default: 4090 or EDGE_HW)"
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX (expect non-negative integer)"
  exit 1
fi

EXPERT_DIR="${2:-${EDGE_EXPERT_DIR:-}}"
HW_ARG="${3:-${EDGE_HW:-4090}}"

if [[ -z "$EXPERT_DIR" ]]; then
  echo "Missing expert_dir. Provide as arg or set EDGE_EXPERT_DIR."
  exit 1
fi
if [[ ! -d "$EXPERT_DIR" ]]; then
  echo "expert_dir not found: $EXPERT_DIR"
  exit 1
fi
if [[ ! -f "$EXPERT_DIR/router.json" ]]; then
  echo "router.json not found in expert_dir: $EXPERT_DIR"
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
CUDA_ID="${EDGE_CUDA:-0}"
EDGE_EMB_PATH="${EDGE_LORA_EMB:-$HW_EMB_V4}"
KEEP_CNT="${EDGE_KEEP_CNT:-16}"

run_gen() {
  local hw_id="$1"
  local target="$2"
  local sketch="$RUN_ROOT/${hw_id}/${ITER}/sketch/0_merge.json"
  local out_dir="$RUN_ROOT/${hw_id}/${ITER}/gen"
  local out_json="$out_dir/kv_lora.json"
  local log_dir="$RUN_ROOT/${hw_id}/${ITER}/logs"
  local log_file="$log_dir/gen_kv_lora.log"

  if [[ ! -f "$sketch" ]]; then
    echo "Missing sketch: $sketch"
    echo "Run: bash gen/scripts/run_train_sketch.sh $IDX $hw_id"
    exit 1
  fi

  mkdir -p "$out_dir" "$log_dir"
  if [[ -f "$out_json" && "${EDGE_FORCE:-0}" != "1" ]]; then
    echo "Output exists: $out_json (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_kv_lora.py" \
    --model_path "$BASE_CKPT" \
    --tokenizer_path "$TOKENIZER" \
    --edge_expert_dirs "$EXPERT_DIR" \
    --edge_embedding_path "$EDGE_EMB_PATH" \
    --sketch_path "$sketch" \
    --save_path "$out_json" \
    --target "$target" \
    --target_hardware "$hw_id" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    --use_hw_kv --hw_kv_mode real \
    --hw_kv_aligner_path "$HW_KV_ALIGNER" \
    --hardware_embedding_path "$HW_EMB_V4" \
    --pos_compensate \
    | tee "$log_file"
}

case "$HW_ARG" in
  4090) run_gen "4090" "$TARGET_4090" ;;
  v100) run_gen "v100" "$TARGET_V100" ;;
  *) echo "Invalid hw: $HW_ARG (expect 4090|v100)"; exit 1 ;;
esac
