#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_SKETCH="${EDGE_SKIP_SKETCH:-0}"
SKIP_GEN="${EDGE_SKIP_GEN:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
SKIP_TRAIN="${EDGE_SKIP_TRAIN:-0}"
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
  echo "Usage: bash gen/scripts/run_iter_adapt_jetson.sh <idx> [hw] [src_tag] [stage] [adapt_tag] [--dry-run]"
  echo "  idx:        iteration index (e.g. 34)"
  echo "  hw:         xavier|4090|v100 (default: EDGE_HW or xavier)"
  echo "  src_tag:    source expert tag used for adaptation gen/train init"
  echo "              (default: EDGE_SRC_EXPERT_TAG or v1_init)"
  echo "  stage:      stage number for naming only (default: EDGE_STAGE or 2)"
  echo "  adapt_tag:  output adapted expert tag (default: EDGE_ADAPT_TAG or v<stage>_gain_adapt)"
  echo ""
  echo "Key env:"
  echo "  EDGE_ADAPT_FOR_TYPE=for_gen_evaltuning_sketch (default, official testtuning style)"
  echo "  EDGE_ADAPT_SRC_ITER=<idx> (default: same as idx)"
  echo "  EDGE_ADAPT_SRC_EXPERT_DIR=/abs/path/to/source/expert (overrides src_iter+src_tag)"
  echo "  EDGE_ADAPT_NAME=eval_adapt (subdir under iterXX/adapt/)"
  echo "  EDGE_ADAPT_SCHEDULE_FILE_PATH=/abs/path/task_sheduler_*.pkl (optional)"
  echo "  EDGE_FORCE=1 to overwrite existing artifacts"
  echo "  EDGE_MANUAL_MEASURE=1 for offloaded measurement"
  exit 1
fi

IDX="$1"
if [[ ! "$IDX" =~ ^[0-9]+$ ]]; then
  echo "Invalid idx: $IDX"
  exit 1
fi
HW="${2:-${EDGE_HW:-xavier}}"
SRC_TAG="${3:-${EDGE_SRC_EXPERT_TAG:-v1_init}}"
STAGE="${4:-${EDGE_STAGE:-2}}"
ADAPT_TAG="${5:-${EDGE_ADAPT_TAG:-v${STAGE}_gain_adapt}}"

if [[ "$HW" != "xavier" && "$HW" != "4090" && "$HW" != "v100" ]]; then
  echo "Invalid hw: $HW (expect xavier|4090|v100)"
  exit 1
fi

if [[ "$MANUAL_MEASURE" == "1" && "$SKIP_MEASURE" == "1" ]]; then
  echo "[WARN] EDGE_SKIP_MEASURE=1 is ignored when EDGE_MANUAL_MEASURE=1."
  SKIP_MEASURE=0
fi

PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

if [[ "$HW" == "xavier" ]]; then
  TARGET_XAVIER="${TARGET_XAVIER:-nvidia/jetson-agx-xavier}"
  TARGET="$TARGET_XAVIER"
elif [[ "$HW" == "4090" ]]; then
  TARGET="${TARGET_4090}"
else
  TARGET="${TARGET_V100}"
fi
ITER="$(printf "iter%02d" "$IDX")"
SRC_ITER="${EDGE_ADAPT_SRC_ITER:-$IDX}"
SRC_ITER_NAME="$(printf "iter%02d" "$SRC_ITER")"

if [[ -z "${EDGE_EMB_JSON:-}" ]]; then
  if [[ -n "${HW_EMB_V5:-}" && -f "${HW_EMB_V5}" ]]; then
    EDGE_EMB_JSON="$HW_EMB_V5"
  elif [[ -f "$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json" ]]; then
    EDGE_EMB_JSON="$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json"
  else
    EDGE_EMB_JSON="${HW_EMB_V4U:-$HW_EMB_V4}"
  fi
fi
EDGE_EMB_PATH="${EDGE_EMB_PATH:-$EDGE_EMB_JSON}"

SRC_EXPERT_DIR="${EDGE_ADAPT_SRC_EXPERT_DIR:-$RUN_ROOT/${HW}/${SRC_ITER_NAME}/experts/${SRC_TAG}}"
if [[ ! -s "$SRC_EXPERT_DIR/router.json" ]]; then
  echo "Source expert/router not found: $SRC_EXPERT_DIR/router.json"
  exit 1
fi

ADAPT_NAME="${EDGE_ADAPT_NAME:-eval_adapt}"
ADAPT_ROOT="$RUN_ROOT/${HW}/${ITER}/adapt/${ADAPT_NAME}"
SKETCH_DIR="$ADAPT_ROOT/sketch"
GEN_DIR="$ADAPT_ROOT/gen"
MEASURE_DIR="$ADAPT_ROOT/measure"
LOG_DIR="$ADAPT_ROOT/logs"
SKETCH_PATH="$SKETCH_DIR/0_merge.json"
GEN_PATH="$GEN_DIR/kv_lora.json"
MEASURE_PATH="$MEASURE_DIR/kv_lora.json"

mkdir -p "$SKETCH_DIR" "$GEN_DIR" "$MEASURE_DIR" "$LOG_DIR"

SKETCH_FOR_TYPE="${EDGE_ADAPT_FOR_TYPE:-for_gen_evaltuning_sketch}"
SKETCH_KEEP_CNT="${EDGE_ADAPT_SKETCH_KEEP_CNT:-64}"
SKETCH_SCHEDULE_FILE_PATH="${EDGE_ADAPT_SCHEDULE_FILE_PATH:-}"
GEN_KEEP_CNT="${EDGE_KEEP_CNT:-16}"
CUDA_ID="${EDGE_CUDA:-0}"
BATCH_SIZE="${EDGE_MEASURE_BATCH_SIZE:-64}"
FORCE="${EDGE_FORCE:-0}"
USE_HW_KV="${EDGE_USE_HW_KV:-0}"
EDGE_HW_KV_ALIGNER="${EDGE_HW_KV_ALIGNER:-${HW_KV_ALIGNER:-}}"
EDGE_HW_KV_PREPROCESS_JSON="${EDGE_HW_KV_PREPROCESS_JSON:-}"

if [[ "$USE_HW_KV" == "1" && -z "$EDGE_HW_KV_ALIGNER" ]]; then
  echo "EDGE_USE_HW_KV=1 but EDGE_HW_KV_ALIGNER is empty."
  exit 1
fi
if [[ "$USE_HW_KV" == "1" && ! -f "$EDGE_HW_KV_ALIGNER" ]]; then
  echo "HwKV aligner not found: $EDGE_HW_KV_ALIGNER"
  exit 1
fi

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
    return 0
  fi
  "$@"
}

echo "[ADAPT] iter=$ITER hw=$HW src_iter=$SRC_ITER_NAME src_tag=$SRC_TAG src_expert=$SRC_EXPERT_DIR"
echo "[ADAPT] for_type=$SKETCH_FOR_TYPE keep(sketch/gen)=${SKETCH_KEEP_CNT}/${GEN_KEEP_CNT} adapt_tag=$ADAPT_TAG"

echo "[STEP] make adaptation sketch"
if [[ "$SKIP_SKETCH" == "1" ]]; then
  echo "Skip sketch (EDGE_SKIP_SKETCH=1)."
elif [[ -s "$SKETCH_PATH" && "$FORCE" != "1" ]]; then
  echo "Sketch exists: $SKETCH_PATH (set EDGE_FORCE=1 to overwrite)"
else
  SKETCH_ARGS=(
    python "$TLM_ROOT/gen/make_dataset.py"
    --for_type "$SKETCH_FOR_TYPE"
    --target "$TARGET"
    --dataset_path "$DATA_ROOT/dataset/to_measure_programs/${HW}"
    --tokenizer_path "$TOKENIZER"
    --save_path "$SKETCH_DIR"
    --keep_cnt "$SKETCH_KEEP_CNT"
  )
  if [[ -n "${EDGE_ADAPT_TEST_FILE_IDX:-}" ]]; then
    SKETCH_ARGS+=(--test_file_idx "$EDGE_ADAPT_TEST_FILE_IDX")
  fi
  if [[ -n "$SKETCH_SCHEDULE_FILE_PATH" ]]; then
    SKETCH_ARGS+=(--schedule_file_path "$SKETCH_SCHEDULE_FILE_PATH")
  fi
  run_cmd "${SKETCH_ARGS[@]}"
fi

if [[ ! -s "$SKETCH_PATH" ]]; then
  echo "Sketch not found or empty: $SKETCH_PATH"
  exit 1
fi

echo "[STEP] gen kv_lora on adaptation sketch"
if [[ "$SKIP_GEN" == "1" ]]; then
  echo "Skip gen (EDGE_SKIP_GEN=1)."
elif [[ -s "$GEN_PATH" && "$FORCE" != "1" ]]; then
  echo "Gen exists: $GEN_PATH (set EDGE_FORCE=1 to overwrite)"
else
  GEN_ARGS=(
    python "$TLM_ROOT/gen/gen_state_kv_lora.py"
    --model_path "$BASE_CKPT"
    --tokenizer_path "$TOKENIZER"
    --edge_expert_dirs "$SRC_EXPERT_DIR"
    --edge_embedding_path "$EDGE_EMB_PATH"
    --hardware_embedding_path "$EDGE_EMB_PATH"
    --sketch_path "$SKETCH_PATH"
    --save_path "$GEN_PATH"
    --target "$TARGET"
    --target_hardware "$HW"
    --keep_cnt "$GEN_KEEP_CNT"
    --use_bucket
  )
  if [[ "$USE_HW_KV" == "1" ]]; then
    GEN_ARGS+=(--use_hw_kv --hw_kv_mode real --hw_kv_aligner_path "$EDGE_HW_KV_ALIGNER")
    if [[ -n "$EDGE_HW_KV_PREPROCESS_JSON" ]]; then
      GEN_ARGS+=(--hw_kv_preprocess_json "$EDGE_HW_KV_PREPROCESS_JSON")
    fi
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=$CUDA_ID ${GEN_ARGS[*]} | tee $LOG_DIR/gen_kv_lora.log"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_ID" "${GEN_ARGS[@]}" | tee "$LOG_DIR/gen_kv_lora.log"
  fi
fi

if [[ "$MANUAL_MEASURE" == "1" ]]; then
  need=0
  if [[ ! -s "$GEN_PATH" ]]; then
    echo "[MANUAL-MEASURE] Missing gen json: $GEN_PATH"
    need=1
  fi
  if [[ ! -s "$MEASURE_PATH" && ! -s "$MEASURE_DIR/kv_lora_measured.json" ]]; then
    echo "[MANUAL-MEASURE] Missing measured json under: $MEASURE_DIR/kv_lora{.json,_measured.json}"
    need=1
  fi
  if [[ "$need" == "1" ]]; then
    echo "Copy to Jetson and measure:"
    echo "  input : $GEN_PATH"
    echo "  output: $MEASURE_PATH"
    echo "Rerun this script with EDGE_SKIP_SKETCH=1 EDGE_SKIP_GEN=1 after copying measured json back."
    if [[ "$DRY_RUN" == "1" ]]; then
      exit 0
    fi
    exit 2
  fi
fi

echo "[STEP] measure kv_lora adaptation candidates"
if [[ "$SKIP_MEASURE" == "1" ]]; then
  echo "Skip measure (EDGE_SKIP_MEASURE=1)."
elif [[ -s "$MEASURE_PATH" && "$FORCE" != "1" ]]; then
  echo "Measured exists: $MEASURE_PATH (set EDGE_FORCE=1 to overwrite)"
else
  run_cmd bash -lc "CUDA_VISIBLE_DEVICES='$CUDA_ID' python '$TLM_ROOT/gen/measure_programs.py' \
    --batch-size '$BATCH_SIZE' \
    --target '$TARGET' \
    --to-measure-path '$GEN_PATH' \
    --measured-path '$MEASURE_PATH' | tee '$LOG_DIR/measure_kv_lora.log'"
fi

if [[ ! -s "$MEASURE_PATH" && -s "$MEASURE_DIR/kv_lora_measured.json" ]]; then
  MEASURE_PATH="$MEASURE_DIR/kv_lora_measured.json"
fi
if [[ ! -s "$MEASURE_PATH" ]]; then
  echo "Measured file not found: $MEASURE_PATH"
  exit 1
fi

echo "[STEP] register measured file into utils.json"
REGISTER_ARGS=(
  python "$TLM_ROOT/gen/scripts/add_measure_records.py"
  --hardware "$HW"
  --iter "$IDX"
  --mode kv_lora
  --measured-path "$MEASURE_PATH"
)
if [[ "${EDGE_ADAPT_REGISTER_TESTTUNING:-1}" == "1" ]]; then
  REGISTER_ARGS+=(--also-testtuning)
fi
run_cmd "${REGISTER_ARGS[@]}"

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "Skip train (EDGE_SKIP_TRAIN=1)."
  exit 0
fi

echo "[STEP] train adapted expert via existing lora-only pipeline"
if [[ "$HW" == "xavier" ]]; then
  TRAIN_SCRIPT="$TLM_ROOT/gen/scripts/run_iter_lora_only_jetson.sh"
else
  TRAIN_SCRIPT="$TLM_ROOT/gen/scripts/run_iter_lora_only.sh"
fi
run_cmd env \
  EDGE_SKIP_SKETCH=1 \
  EDGE_SKIP_GEN=1 \
  EDGE_SKIP_MEASURE=1 \
  EDGE_FORCE_PREPARE="${EDGE_FORCE_PREPARE:-1}" \
  EDGE_FORCE_MAKE="${EDGE_FORCE_MAKE:-1}" \
  EDGE_INIT_EXPERT_DIR="$SRC_EXPERT_DIR" \
  EDGE_GAIN_TAG="$ADAPT_TAG" \
  EDGE_EMB_JSON="$EDGE_EMB_JSON" \
  EDGE_EMB_PATH="$EDGE_EMB_PATH" \
  bash "$TRAIN_SCRIPT" "$IDX" "$HW" "$SRC_TAG" "$STAGE"

echo "[DONE] Xavier adaptation round finished."
echo "  adapt_root  : $ADAPT_ROOT"
echo "  measured    : $MEASURE_PATH"
echo "  adapted_tag : $ADAPT_TAG"
