#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_train_v1_init.sh <hw>"
  echo "  hw: v100|4090 (required)"
  echo "  env: EDGE_ITER_MAX=0 (default 0)"
  exit 1
fi

HW="$1"
ITER_MAX="${EDGE_ITER_MAX:-0}"

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  echo "Set EDGE_PATHS_SH to your paths.sh location."
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT is empty. Check your paths.sh or export RUN_ROOT explicitly."
  exit 1
fi

if [[ "$HW" == "4090" ]]; then
  TARGET="$TARGET_4090"
elif [[ "$HW" == "v100" ]]; then
  TARGET="$TARGET_V100"
else
  echo "Invalid hw: $HW (expect v100|4090)"
  exit 1
fi

ITER="iter00"
BASE_RECORD_DIR="$DATA_ROOT/dataset/measure_records_base/${HW}"
ALL_RECORD_DIR="$DATA_ROOT/dataset/measure_records/${HW}"
BASE_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/base"
TEACHER_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/teacher"
EDGE_SFT_BASE="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}.jsonl"

echo "[STEP] postprocess (base, iter<=${ITER_MAX})"
python "$TLM_ROOT/gen/postprocess.py" \
  --target "$TARGET" \
  --record-mode base \
  --record-dir "$BASE_RECORD_DIR" \
  --iter-max "$ITER_MAX"

echo "[STEP] postprocess (all, iter<=${ITER_MAX})"
python "$TLM_ROOT/gen/postprocess.py" \
  --target "$TARGET" \
  --record-mode all \
  --record-dir "$ALL_RECORD_DIR" \
  --iter-max "$ITER_MAX"

echo "[STEP] make_dataset base"
mkdir -p "$BASE_SFT_DIR"
python "$TLM_ROOT/gen/make_dataset.py" \
  --for_type=for_gen_best \
  --target "$TARGET" \
  --dataset_path "$BASE_RECORD_DIR" \
  --tokenizer_path "$TOKENIZER" \
  --save_path "$BASE_SFT_DIR"

echo "[STEP] make_dataset teacher"
mkdir -p "$TEACHER_SFT_DIR"
python "$TLM_ROOT/gen/make_dataset.py" \
  --for_type=for_gen_best \
  --target "$TARGET" \
  --dataset_path "$ALL_RECORD_DIR" \
  --tokenizer_path "$TOKENIZER" \
  --save_path "$TEACHER_SFT_DIR"

echo "[STEP] prepare_edge_dataset base_only"
mkdir -p "$RUN_ROOT/${HW}/${ITER}/sft"
touch "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl"
python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
  --base-jsonl "$BASE_SFT_DIR/0_merge.json" \
  --lora-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl" \
  --teacher-jsonl "$TEACHER_SFT_DIR/0_merge.json" \
  --output-jsonl "$EDGE_SFT_BASE" \
  --merge_mode base_left \
  --merge_key repr \
  --dedupe_mode keep_all \
  --hardware-id "$HW" \
  --embedding-json "$HW_EMB_V4" \
  --allow-missing-lora

echo "[STEP] train v1_init"
export EDGE_DATASET_JSONL="$EDGE_SFT_BASE"
export EDGE_LORA_LAMBDA_GAIN=0
export EDGE_LORA_WARMUP=0
unset EDGE_INIT_EXPERT_DIR
export EDGE_RESUME_PREV=0
bash "$TLM_ROOT/gen/scripts/run_train_edge_expert.sh" 0 "$HW" v1_init
