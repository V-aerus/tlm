#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_GEN="${EDGE_SKIP_GEN:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    ARGS+=("$arg")
  fi
done
set -- "${ARGS[@]}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_iter_lora_only.sh <idx> [hw] [prev_tag] [stage] [--dry-run]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  hw: 4090|v100 (optional, default: EDGE_HW or 4090)"
  echo "  prev_tag: previous expert tag for KV+LoRA gen (default: EDGE_PREV_EXPERT_TAG or v1_init)"
  echo "  stage: numeric stage for gain training (default: EDGE_STAGE or 2)"
  echo "  --dry-run: only check missing artifacts, no commands executed"
  echo "Note: This script skips base generation/measurement steps."
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX (expect non-negative integer)"
  exit 1
fi

HW="${2:-${EDGE_HW:-4090}}"
PREV_TAG="${3:-${EDGE_PREV_EXPERT_TAG:-v1_init}}"
STAGE="${4:-${EDGE_STAGE:-2}}"

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  echo "Set EDGE_PATHS_SH to your paths.sh location."
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

ITER=$(printf "iter%02d" "$IDX")
GAIN_TAG="${EDGE_GAIN_TAG:-v${STAGE}_gain}"
BASE_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/base"
LORA_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/lora"
EDGE_SFT_BASE="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}.jsonl"
EDGE_SFT_LORA="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}_v${STAGE}.jsonl"
EDGE_SFT_MERGED="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}_v${STAGE}_lora_left.jsonl"
if [[ "$IDX" -gt 0 ]]; then
  PREV_ITER=$(printf "iter%02d" "$((IDX - 1))")
else
  PREV_ITER="iter00"
fi
PREV_EXPERT_DIR="${EDGE_PREV_EXPERT_DIR:-$RUN_ROOT/${HW}/${PREV_ITER}/experts/${PREV_TAG}}"

if [[ "$HW" == "4090" ]]; then
  TARGET="$TARGET_4090"
elif [[ "$HW" == "v100" ]]; then
  TARGET="$TARGET_V100"
else
  echo "Invalid hw: $HW (expect 4090|v100)"
  exit 1
fi

pick_measured() {
  local measure_dir="$1"
  local stem="$2"
  if [[ -s "${measure_dir}/${stem}.json" ]]; then
    echo "${measure_dir}/${stem}.json"
    return 0
  fi
  if [[ -s "${measure_dir}/${stem}_measured.json" ]]; then
    echo "${measure_dir}/${stem}_measured.json"
    return 0
  fi
  return 1
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
    return 0
  fi
  "$@"
}

echo "[STEP] Ensure sketch"
if [[ ! -s "$RUN_ROOT/${HW}/${ITER}/sketch/0_merge.json" ]]; then
  run_cmd bash "$TLM_ROOT/gen/scripts/run_train_sketch.sh" "$IDX" "$HW"
fi

BASE_RECORD_DIR="$DATA_ROOT/dataset/measure_records_base/${HW}"
LORA_RECORD_DIR="$DATA_ROOT/dataset/measure_records_kv_lora/${HW}"
ALL_RECORD_DIR="$DATA_ROOT/dataset/measure_records/${HW}"
ITER_MAX="${EDGE_ITER_MAX:-}"
ITER_LIST="${EDGE_ITER_LIST:-}"
POSTPROCESS_FILTER_ARGS=()
if [[ -n "$ITER_LIST" ]]; then
  POSTPROCESS_FILTER_ARGS+=(--iter-list "$ITER_LIST")
elif [[ -n "$ITER_MAX" ]]; then
  POSTPROCESS_FILTER_ARGS+=(--iter-max "$ITER_MAX")
fi
CLEAN_OUTPUT="${EDGE_CLEAN_OUTPUT:-0}"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  POSTPROCESS_FILTER_ARGS+=(--clean-output)
fi
FORCE_PREPARE="${EDGE_FORCE_PREPARE:-0}"

echo "[STEP] postprocess (base)"
run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode base --record-dir "$BASE_RECORD_DIR" "${POSTPROCESS_FILTER_ARGS[@]}"

echo "[STEP] Base SFT (for_gen_best -> edge_sft)"
if [[ ! -s "$BASE_SFT_DIR/0_merge.json" ]]; then
  run_cmd mkdir -p "$BASE_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$BASE_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$BASE_SFT_DIR"
fi

if [[ ! -s "$EDGE_SFT_BASE" ]]; then
  run_cmd mkdir -p "$RUN_ROOT/${HW}/${ITER}/sft"
  run_cmd touch "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl"
  # teacher/0_merge.json is created later after postprocess(all)
fi

echo "[STEP] KV+LoRA gen"
if [[ "$SKIP_GEN" == "1" ]]; then
  echo "Skip kv_lora gen (EDGE_SKIP_GEN=1)."
elif [[ ! -s "$RUN_ROOT/${HW}/${ITER}/gen/kv_lora.json" ]]; then
  if [[ ! -s "$PREV_EXPERT_DIR/router.json" ]]; then
    echo "Missing router.json in prev expert dir: $PREV_EXPERT_DIR"
    exit 1
  fi
  run_cmd bash "$TLM_ROOT/gen/scripts/run_gen_kv_lora.sh" "$IDX" "$PREV_EXPERT_DIR" "$HW"
fi

echo "[STEP] KV+LoRA measure"
if [[ "$SKIP_MEASURE" == "1" ]]; then
  echo "Skip kv_lora measure + add_measure_records (EDGE_SKIP_MEASURE=1)."
else
  LORA_MEASURE_DIR="$RUN_ROOT/${HW}/${ITER}/measure"
  if ! LORA_MEASURE_PATH="$(pick_measured "$LORA_MEASURE_DIR" "kv_lora")"; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY-RUN] Missing kv_lora measure: $LORA_MEASURE_DIR/kv_lora{,_measured}.json"
      LORA_MEASURE_PATH="$LORA_MEASURE_DIR/kv_lora.json"
    else
      bash "$TLM_ROOT/gen/scripts/run_measure.sh" "$IDX" "$HW" kv_lora
      LORA_MEASURE_PATH="$(pick_measured "$LORA_MEASURE_DIR" "kv_lora")"
    fi
  fi

  echo "[STEP] Add kv_lora measure to utils.json"
  run_cmd python "$TLM_ROOT/gen/scripts/add_measure_records.py" \
    --hardware "$HW" \
    --iter "$IDX" \
    --mode kv_lora \
    --measured-path "$LORA_MEASURE_PATH"
fi

echo "[STEP] postprocess (kv_lora)"
run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode kv_lora --record-dir "$LORA_RECORD_DIR" "${POSTPROCESS_FILTER_ARGS[@]}"
echo "[STEP] postprocess (all)"
run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode all --record-dir "$ALL_RECORD_DIR" "${POSTPROCESS_FILTER_ARGS[@]}"

echo "[STEP] LoRA SFT + merge"
if [[ ! -s "$LORA_SFT_DIR/0_merge.json" ]]; then
  run_cmd mkdir -p "$LORA_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$LORA_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$LORA_SFT_DIR"
fi

TEACHER_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/teacher"
if [[ ! -s "$TEACHER_SFT_DIR/0_merge.json" ]]; then
  run_cmd mkdir -p "$TEACHER_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$ALL_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$TEACHER_SFT_DIR"
fi

if [[ "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_BASE" ]]; then
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$BASE_SFT_DIR/0_merge.json" \
    --lora-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl" \
    --teacher-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/teacher/0_merge.json" \
    --output-jsonl "$EDGE_SFT_BASE" \
    --merge_mode base_left \
    --merge_key repr \
    --dedupe_mode keep_all \
    --hardware-id "$HW" \
    --embedding-json "$HW_EMB_V4" \
    --allow-missing-lora
fi

if [[ "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_LORA" ]]; then
  run_cmd touch "$RUN_ROOT/${HW}/${ITER}/sft/empty_base.jsonl"
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/empty_base.jsonl" \
    --lora-jsonl "$LORA_SFT_DIR/0_merge.json" \
    --teacher-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/teacher/0_merge.json" \
    --output-jsonl "$EDGE_SFT_LORA" \
    --merge_mode lora_left \
    --merge_key repr \
    --dedupe_mode keep_all \
    --hardware-id "$HW" \
    --embedding-json "$HW_EMB_V4"
fi

if [[ "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_MERGED" ]]; then
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$EDGE_SFT_BASE" \
    --lora-jsonl "$EDGE_SFT_LORA" \
    --teacher-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/teacher/0_merge.json" \
    --output-jsonl "$EDGE_SFT_MERGED" \
    --merge_key repr \
    --merge_mode lora_left \
    --dedupe_mode keep_all \
    --embedding-json "$HW_EMB_V4"
fi

echo "[STEP] Train gain expert"
GAIN_OUT="$RUN_ROOT/${HW}/${ITER}/experts/${GAIN_TAG}"
if [[ -s "$GAIN_OUT/adapter_model.safetensors" && "${EDGE_FORCE:-0}" != "1" ]]; then
  echo "Gain expert exists: $GAIN_OUT (set EDGE_FORCE=1 to retrain)"
  exit 0
fi

export EDGE_DATASET_JSONL="$EDGE_SFT_MERGED"
export EDGE_LORA_LAMBDA_GAIN="${EDGE_LORA_LAMBDA_GAIN:-0.1}"
INIT_EXPERT_DIR="${EDGE_INIT_EXPERT_DIR:-$PREV_EXPERT_DIR}"
run_cmd env EDGE_INIT_EXPERT_DIR="$INIT_EXPERT_DIR" bash "$TLM_ROOT/gen/scripts/run_train_edge_expert.sh" "$IDX" "$HW" "$GAIN_TAG"
