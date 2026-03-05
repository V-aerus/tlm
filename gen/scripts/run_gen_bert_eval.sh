#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_gen_bert_eval.sh <expert_dir> [hw] [tag]"
  echo "  expert_dir: LoRA expert dir with router.json"
  echo "  hw: 4090|v100 (optional, default: EDGE_HW or 4090)"
  echo "  tag: output tag (optional, default: EDGE_BERT_TAG or basename(expert_dir))"
  echo "  env: EDGE_BERT_SKETCH=/path/to/0_merge.json"
  echo "       EDGE_BERT_EVAL_DIR=/path/to/output_dir"
  echo "       EDGE_BACKEND=graph (tune_relay backend)"
  echo "       EDGE_KEEP_CNT=16"
  echo "       EDGE_FORCE=1 (overwrite outputs)"
  echo "       EDGE_SKIP_TUNE=1 (skip tune_relay)"
  exit 1
fi

EXPERT_DIR="$1"
HW="${2:-${EDGE_HW:-4090}}"
TAG="${3:-${EDGE_BERT_TAG:-$(basename "$EXPERT_DIR")}}"

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

if [[ -z "${EDGE_EMB_PATH:-}" ]]; then
  if [[ -n "${HW_EMB_V5:-}" && -f "${HW_EMB_V5}" ]]; then
    EDGE_EMB_PATH="$HW_EMB_V5"
  elif [[ -f "$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json" ]]; then
    EDGE_EMB_PATH="$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json"
  else
    EDGE_EMB_PATH="${HW_EMB_V4U:-$HW_EMB_V4}"
  fi
fi
EDGE_HW_KV_ALIGNER="${EDGE_HW_KV_ALIGNER:-$HW_KV_ALIGNER}"

if [[ "$HW" == "4090" ]]; then
  TARGET="$TARGET_4090"
  TUNE_TARGET="4090"
  DEFAULT_SKETCH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json"
  DEFAULT_OUT_DIR="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form"
elif [[ "$HW" == "v100" ]]; then
  TARGET="$TARGET_V100"
  TUNE_TARGET="nvidia/nvidia-v100"
  DEFAULT_SKETCH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/v100_gen_eval_only_bert_new_form/0_merge.json"
  DEFAULT_OUT_DIR="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/v100_gen_eval_only_bert_new_form"
else
  echo "Invalid hw: $HW (expect 4090|v100)"
  exit 1
fi

SKETCH_PATH="${EDGE_BERT_SKETCH:-$DEFAULT_SKETCH}"
OUT_DIR="${EDGE_BERT_EVAL_DIR:-$DEFAULT_OUT_DIR}"
KEEP_CNT="${EDGE_KEEP_CNT:-64}"
CUDA_ID="${EDGE_CUDA:-2}"
BACKEND="${EDGE_BACKEND:-graph}"

if [[ ! -f "$SKETCH_PATH" ]]; then
  echo "Missing sketch: $SKETCH_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"
GEN_OUT="$OUT_DIR/kv_lora_${TAG}.json"
MEASURED_OUT="$OUT_DIR/kv_lora_${TAG}_measured.json"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$GEN_OUT" || "${EDGE_FORCE:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_kv_lora.py" \
    --model_path "$BASE_CKPT" \
    --tokenizer_path "$TOKENIZER" \
    --edge_expert_dirs "$EXPERT_DIR" \
    --edge_embedding_path "$EDGE_EMB_PATH" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$GEN_OUT" \
    --target "$TARGET" \
    --target_hardware "$HW" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    --use_hw_kv --hw_kv_mode real \
    --hw_kv_aligner_path "$EDGE_HW_KV_ALIGNER" \
    --hardware_embedding_path "$EDGE_EMB_PATH" \
    --pos_compensate \
    | tee "$LOG_DIR/gen_bert_${TAG}.log"
else
  echo "Gen output exists: $GEN_OUT (set EDGE_FORCE=1 to overwrite)"
fi

if [[ ! -f "$MEASURED_OUT" || "${EDGE_FORCE:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/measure_programs.py" \
    --batch-size 64 \
    --target "$TUNE_TARGET" \
    --to-measure-path "$GEN_OUT" \
    --measured-path "$MEASURED_OUT" \
    | tee "$LOG_DIR/measure_bert_${TAG}.log"
else
  echo "Measured output exists: $MEASURED_OUT (set EDGE_FORCE=1 to overwrite)"
fi

if [[ "${EDGE_SKIP_TUNE:-0}" != "1" ]]; then
  TLM_LOG_FILE="$MEASURED_OUT" CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/tune_relay.py" \
    --workload bert_base \
    --input-shape "[1,128]" \
    --target "$TUNE_TARGET" \
    --backend "$BACKEND" \
    | tee "$LOG_DIR/tune_bert_${TAG}.log"
else
  echo "Skip tune_relay (EDGE_SKIP_TUNE=1)."
fi
