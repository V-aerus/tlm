#!/usr/bin/env bash
set -euo pipefail

# Xeon CPU iterative pipeline: LoRA-only stage (bucket-only, NO HwKVAligner KV injection).
# This mirrors gen/scripts/run_iter_lora_only.sh directory layout and artifact names.

DRY_RUN=0
SKIP_GEN="${EDGE_SKIP_GEN:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
SKIP_TRAIN="${EDGE_SKIP_TRAIN:-0}"
SKIP_SKETCH="${EDGE_SKIP_SKETCH:-0}"
MANUAL_MEASURE="${EDGE_MANUAL_MEASURE:-0}"
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
  echo "Usage: bash gen/scripts/run_iter_lora_only_xeon.sh <idx> [hw] [prev_tag] [stage] [--dry-run]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  hw: xeon (optional, default: EDGE_HW or xeon)"
  echo "  prev_tag: previous expert tag for bucket+LoRA gen (default: EDGE_PREV_EXPERT_TAG or v1_init)"
  echo "  stage: numeric stage for gain training (default: EDGE_STAGE or 2)"
  echo "  --dry-run: only check missing artifacts, no commands executed"
  echo "Note: This script skips base generation/measurement steps."
  echo "      For manual/offloaded measurement, set: EDGE_MANUAL_MEASURE=1"
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX (expect non-negative integer)"
  exit 1
fi

HW="${2:-${EDGE_HW:-xeon}}"
PREV_TAG="${3:-${EDGE_PREV_EXPERT_TAG:-v1_init}}"
STAGE="${4:-${EDGE_STAGE:-2}}"

if [[ "$HW" != "xeon" ]]; then
  echo "Invalid hw: $HW (expect xeon)"
  exit 1
fi

if [[ "$MANUAL_MEASURE" == "1" && "$SKIP_MEASURE" == "1" ]]; then
  echo "[WARN] EDGE_SKIP_MEASURE=1 is ignored when EDGE_MANUAL_MEASURE=1 (need measured files + add_measure_records)."
  SKIP_MEASURE=0
fi

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  echo "Set EDGE_PATHS_SH to your paths.sh location."
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

TARGET_XEON="${TARGET_XEON:-llvm -keys=cpu -mcpu=skylake-avx512 -model=xeon}"
TARGET="$TARGET_XEON"

# Prefer v5 embeddings when available; override with EDGE_EMB_JSON/EDGE_EMB_PATH.
if [[ -z "${EDGE_EMB_JSON:-}" ]]; then
  if [[ -n "${HW_EMB_V5:-}" && -f "${HW_EMB_V5}" ]]; then
    EDGE_EMB_JSON="$HW_EMB_V5"
  elif [[ -f "$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json" ]]; then
    EDGE_EMB_JSON="$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json"
  else
    EDGE_EMB_JSON="${HW_EMB_V4U:-$HW_EMB_V4}"
  fi
fi
if [[ -z "${EDGE_EMB_PATH:-}" ]]; then
  EDGE_EMB_PATH="$EDGE_EMB_JSON"
fi

ITER=$(printf "iter%02d" "$IDX")
TEST_FILE_IDX="$IDX"
SKETCH_KEEP_CNT="${EDGE_SKETCH_KEEP_CNT:-48}"
KEEP_CNT="${EDGE_KEEP_CNT:-16}"
CUDA_ID="${EDGE_CUDA:-3}"
BATCH_SIZE="${EDGE_MEASURE_BATCH_SIZE:-64}"

if [[ "$IDX" -gt 0 ]]; then
  PREV_ITER=$(printf "iter%02d" "$((IDX - 1))")
else
  PREV_ITER="iter00"
fi
PREV_EXPERT_DIR="${EDGE_PREV_EXPERT_DIR:-$RUN_ROOT/${HW}/${PREV_ITER}/experts/${PREV_TAG}}"

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

truncate_file() {
  local path="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] truncate -s 0 $path"
    return 0
  fi
  mkdir -p "$(dirname "$path")"
  : > "$path"
}

run_sketch() {
  local out_dir="$RUN_ROOT/${HW}/${ITER}/sketch"
  mkdir -p "$out_dir"
  python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_train_sketch \
    --target "$TARGET" \
    --dataset_path "$DATA_ROOT/dataset/to_measure_programs/${HW}" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$out_dir" \
    --keep_cnt="$SKETCH_KEEP_CNT" \
    --test_file_idx="$TEST_FILE_IDX"
}

run_gen_lora() {
  local expert_dir="$1"
  local sketch="$RUN_ROOT/${HW}/${ITER}/sketch/0_merge.json"
  local out_dir="$RUN_ROOT/${HW}/${ITER}/gen"
  local out_json="$out_dir/kv_lora.json"
  local log_dir="$RUN_ROOT/${HW}/${ITER}/logs"
  local log_file="$log_dir/gen_kv_lora.log"

  if [[ ! -s "$expert_dir/router.json" ]]; then
    echo "router.json not found in expert_dir: $expert_dir"
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
    --edge_expert_dirs "$expert_dir" \
    --edge_embedding_path "$EDGE_EMB_PATH" \
    --sketch_path "$sketch" \
    --save_path "$out_json" \
    --target "$TARGET" \
    --target_hardware "$HW" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    | tee "$log_file"
}

run_measure_lora() {
  local gen_dir="$RUN_ROOT/${HW}/${ITER}/gen"
  local measure_dir="$RUN_ROOT/${HW}/${ITER}/measure"
  local log_dir="$RUN_ROOT/${HW}/${ITER}/logs"
  local gen_file="$gen_dir/kv_lora.json"
  local out_file="$measure_dir/kv_lora.json"
  local log_file="$log_dir/measure_kv_lora.log"

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
    --target "$TARGET" \
    --to-measure-path "$gen_file" \
    --measured-path "$out_file" \
    | tee "$log_file"
}

echo "[STEP] Ensure sketch"
if [[ "$SKIP_SKETCH" == "1" ]]; then
  echo "Skip sketch (EDGE_SKIP_SKETCH=1)."
elif [[ ! -s "$RUN_ROOT/${HW}/${ITER}/sketch/0_merge.json" ]]; then
  run_cmd run_sketch
fi

echo "[STEP] Bucket+LoRA gen (no KV injection)"
if [[ "$SKIP_GEN" == "1" ]]; then
  echo "Skip lora gen (EDGE_SKIP_GEN=1)."
elif [[ ! -s "$RUN_ROOT/${HW}/${ITER}/gen/kv_lora.json" ]]; then
  run_cmd run_gen_lora "$PREV_EXPERT_DIR"
fi

if [[ "$MANUAL_MEASURE" == "1" ]]; then
  LORA_GEN_PATH="$RUN_ROOT/${HW}/${ITER}/gen/kv_lora.json"
  MEASURE_DIR="$RUN_ROOT/${HW}/${ITER}/measure"
  LORA_MEASURE_PATH="$(pick_measured "$MEASURE_DIR" "kv_lora" || true)"

  need=0
  if [[ ! -s "$LORA_GEN_PATH" ]]; then
    echo "[MANUAL-MEASURE] Missing lora gen: $LORA_GEN_PATH"
    need=1
  fi
  if [[ -z "$LORA_MEASURE_PATH" ]]; then
    echo "[MANUAL-MEASURE] Missing lora measured json under: $MEASURE_DIR/kv_lora{.json,_measured.json}"
    need=1
  fi
  if [[ "$need" == "1" ]]; then
    echo ""
    echo "[MANUAL-MEASURE] Offloaded workflow:"
    echo "  1) Copy gen json to remote worker:"
    echo "     - $LORA_GEN_PATH"
    echo "  2) Run measurement on target host (target: '$TARGET') and copy result back into:"
    echo "     - $MEASURE_DIR/kv_lora.json"
    echo ""
    echo "After copying measured files back, rerun this script (recommend: EDGE_SKIP_GEN=1 EDGE_SKIP_SKETCH=1)."
    if [[ "$DRY_RUN" == "1" ]]; then
      exit 0
    fi
    exit 2
  fi
fi

echo "[STEP] Bucket+LoRA measure"
if [[ "$SKIP_MEASURE" == "1" ]]; then
  echo "Skip lora measure + add_measure_records (EDGE_SKIP_MEASURE=1)."
else
  LORA_MEASURE_PATH="$(pick_measured "$RUN_ROOT/${HW}/${ITER}/measure" "kv_lora" || true)"
  if [[ -z "$LORA_MEASURE_PATH" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY-RUN] Missing lora measure: $RUN_ROOT/${HW}/${ITER}/measure/kv_lora{,_measured}.json"
      LORA_MEASURE_PATH="$RUN_ROOT/${HW}/${ITER}/measure/kv_lora.json"
    else
      run_measure_lora
      LORA_MEASURE_PATH="$(pick_measured "$RUN_ROOT/${HW}/${ITER}/measure" "kv_lora")"
    fi
  fi

  echo "[STEP] Add lora measure to utils.json"
  run_cmd python "$TLM_ROOT/gen/scripts/add_measure_records.py" \
    --hardware "$HW" \
    --iter "$IDX" \
    --mode kv_lora \
    --measured-path "$LORA_MEASURE_PATH"
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
POSTPROCESS_QUIET_ARGS=()
if [[ "${EDGE_QUIET:-0}" == "1" ]]; then
  POSTPROCESS_QUIET_ARGS+=(--quiet)
fi
CLEAN_OUTPUT="${EDGE_CLEAN_OUTPUT:-0}"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  POSTPROCESS_FILTER_ARGS+=(--clean-output)
fi
FORCE_PREPARE="${EDGE_FORCE_PREPARE:-0}"
FORCE_MAKE="${EDGE_FORCE_MAKE:-0}"

echo "[STEP] postprocess (base)"
run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode base --record-dir "$BASE_RECORD_DIR" "${POSTPROCESS_QUIET_ARGS[@]}" "${POSTPROCESS_FILTER_ARGS[@]}"

echo "[STEP] Base SFT (for_gen_best -> edge_sft)"
BASE_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/base"
if [[ "$FORCE_MAKE" == "1" || ! -s "$BASE_SFT_DIR/0_merge.json" ]]; then
  run_cmd mkdir -p "$BASE_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$BASE_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$BASE_SFT_DIR"
fi

KV_LORA_LIST_LEN="$(
  HW_KEY="$HW" DATA_ROOT="$DATA_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

data_root = os.environ.get("DATA_ROOT", "")
hw_key = os.environ.get("HW_KEY", "")
utils_path = Path(data_root) / "utils.json"
try:
    data = json.loads(utils_path.read_text(encoding="utf-8"))
except Exception:
    data = {}
lst = data.get(hw_key, {}).get("measure_records_kv_lora", [])
print(len(lst) if isinstance(lst, list) else 0)
PY
)"
HAS_KV_LORA=0
if [[ "$KV_LORA_LIST_LEN" -gt 0 ]]; then
  HAS_KV_LORA=1
fi

echo "[STEP] postprocess (kv_lora)"
if [[ "$HAS_KV_LORA" == "1" ]]; then
  run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode kv_lora --record-dir "$LORA_RECORD_DIR" "${POSTPROCESS_QUIET_ARGS[@]}" "${POSTPROCESS_FILTER_ARGS[@]}"
else
  echo "[WARN] No kv_lora measure records registered; skip kv_lora postprocess."
fi
echo "[STEP] postprocess (all)"
run_cmd python "$TLM_ROOT/gen/postprocess.py" --target "$TARGET" --record-mode all --record-dir "$ALL_RECORD_DIR" "${POSTPROCESS_QUIET_ARGS[@]}" "${POSTPROCESS_FILTER_ARGS[@]}"

count_json_files() {
  local dir="$1"
  shopt -s nullglob
  local files=("$dir"/*.json)
  shopt -u nullglob
  echo "${#files[@]}"
}
LORA_JSON_COUNT=0
if [[ "$HAS_KV_LORA" == "1" ]]; then
  LORA_JSON_COUNT="$(count_json_files "$LORA_RECORD_DIR")"
fi
if [[ "$LORA_JSON_COUNT" -eq 0 ]]; then
  echo "[WARN] No lora records found; will train using base-only dataset."
fi

echo "[STEP] LoRA SFT + merge"
LORA_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/lora"
if [[ "$LORA_JSON_COUNT" -gt 0 && ( "$FORCE_MAKE" == "1" || ! -s "$LORA_SFT_DIR/0_merge.json" ) ]]; then
  run_cmd mkdir -p "$LORA_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$LORA_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$LORA_SFT_DIR"
fi

TEACHER_SFT_DIR="$RUN_ROOT/${HW}/${ITER}/sft/teacher"
if [[ "$FORCE_MAKE" == "1" || ! -s "$TEACHER_SFT_DIR/0_merge.json" ]]; then
  run_cmd mkdir -p "$TEACHER_SFT_DIR"
  run_cmd python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$TARGET" \
    --dataset_path "$ALL_RECORD_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$TEACHER_SFT_DIR"
fi

EDGE_SFT_BASE="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}.jsonl"
EDGE_SFT_LORA="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}_v${STAGE}.jsonl"
EDGE_SFT_MERGED="$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}_v${STAGE}_lora_left.jsonl"

if [[ "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_BASE" ]]; then
  run_cmd mkdir -p "$RUN_ROOT/${HW}/${ITER}/sft"
  truncate_file "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl"
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$BASE_SFT_DIR/0_merge.json" \
    --lora-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/empty_lora.jsonl" \
    --teacher-jsonl "$TEACHER_SFT_DIR/0_merge.json" \
    --output-jsonl "$EDGE_SFT_BASE" \
    --merge_mode base_left \
    --merge_key repr \
    --dedupe_mode "${EDGE_DEDUPE_MODE:-min}" \
    --hardware-id "$HW" \
    --embedding-json "$EDGE_EMB_JSON" \
    --allow-missing-lora
fi

if [[ "$LORA_JSON_COUNT" -gt 0 && ( "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_LORA" ) ]]; then
  truncate_file "$RUN_ROOT/${HW}/${ITER}/sft/empty_base.jsonl"
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$RUN_ROOT/${HW}/${ITER}/sft/empty_base.jsonl" \
    --lora-jsonl "$LORA_SFT_DIR/0_merge.json" \
    --teacher-jsonl "$TEACHER_SFT_DIR/0_merge.json" \
    --output-jsonl "$EDGE_SFT_LORA" \
    --merge_mode lora_left \
    --merge_key repr \
    --dedupe_mode "${EDGE_DEDUPE_MODE:-min}" \
    --hardware-id "$HW" \
    --embedding-json "$EDGE_EMB_JSON"
fi

if [[ "$LORA_JSON_COUNT" -gt 0 && ( "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_MERGED" ) ]]; then
  run_cmd python "$TLM_ROOT/gen/prepare_edge_dataset.py" \
    --base-jsonl "$EDGE_SFT_BASE" \
    --lora-jsonl "$EDGE_SFT_LORA" \
    --teacher-jsonl "$TEACHER_SFT_DIR/0_merge.json" \
    --output-jsonl "$EDGE_SFT_MERGED" \
    --merge_key repr \
    --merge_mode lora_left \
    --dedupe_mode "${EDGE_DEDUPE_MODE:-min}" \
    --embedding-json "$EDGE_EMB_JSON"
fi
if [[ "$LORA_JSON_COUNT" -eq 0 ]]; then
  if [[ "$FORCE_PREPARE" == "1" || ! -s "$EDGE_SFT_MERGED" ]]; then
    run_cmd cp "$EDGE_SFT_BASE" "$EDGE_SFT_MERGED"
  fi
fi

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "Skip gain expert training (EDGE_SKIP_TRAIN=1)."
  exit 0
fi

GAIN_TAG="${EDGE_GAIN_TAG:-v${STAGE}_gain}"
echo "[STEP] Train gain expert: ${GAIN_TAG}"
GAIN_OUT="$RUN_ROOT/${HW}/${ITER}/experts/${GAIN_TAG}"
if [[ -s "$GAIN_OUT/adapter_model.safetensors" && "${EDGE_FORCE:-0}" != "1" ]]; then
  echo "Gain expert exists: $GAIN_OUT (set EDGE_FORCE=1 to retrain)"
  exit 0
fi

export EDGE_DATASET_JSONL="$EDGE_SFT_MERGED"
export EDGE_LORA_LAMBDA_GAIN="${EDGE_LORA_LAMBDA_GAIN:-0.1}"
INIT_EXPERT_DIR="${EDGE_INIT_EXPERT_DIR:-$PREV_EXPERT_DIR}"
run_cmd env EDGE_INIT_EXPERT_DIR="$INIT_EXPERT_DIR" bash "$TLM_ROOT/gen/scripts/run_train_edge_expert.sh" "$IDX" "$HW" "$GAIN_TAG"
