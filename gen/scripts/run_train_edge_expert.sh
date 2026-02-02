#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_train_edge_expert.sh <idx> [hw] [tag]"
  echo "  idx: non-negative integer (0,1,2,...)"
  echo "  hw: 4090|v100 (optional, default: EDGE_HW or 4090)"
  echo "  tag: expert output tag (optional, default: EDGE_EXPERT_TAG or v1_init)"
  echo "  env: EDGE_INIT_EXPERT_DIR=/path/to/prev_expert (resume from previous LoRA)"
  echo "       EDGE_RESUME_PREV=0 (disable auto-resume from previous iter)"
  echo "       EDGE_PREV_EXPERT_TAG=v1_init (auto-resume tag when idx>0)"
  echo "       EDGE_LORA_GAIN_MARGIN=0.05 (override gain margin)"
  echo "       EDGE_ROUTER_PREPROCESS=zscore_mask (enable routing preprocess)"
  echo "       EDGE_ROUTER_PREPROCESS_EMB=/path/to/hardware_embeddings_v4_universe.json"
  echo "       EDGE_ROUTER_PREPROCESS_JSON=/path/to/preprocess_v4u_zscore_v1.json"
  echo "       EDGE_TRAIN_ROUTER_ONLY=1 (freeze LoRA, train router only)"
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

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT is empty. Check your paths.sh or export RUN_ROOT explicitly."
  exit 1
fi

ITER=$(printf "iter%02d" "$IDX")
HW="${2:-${EDGE_HW:-4090}}"
TAG="${3:-${EDGE_EXPERT_TAG:-v1_init}}"
CUDA_ID="${EDGE_CUDA:-0}"

DATASET="${EDGE_DATASET_JSONL:-$RUN_ROOT/${HW}/${ITER}/sft/edge_sft_${HW}.jsonl}"
OUT_DIR="${EDGE_EXPERT_OUT_DIR:-$RUN_ROOT/${HW}/${ITER}/experts/${TAG}}"
LOG_DIR="$RUN_ROOT/${HW}/${ITER}/logs"
LOG_FILE="$LOG_DIR/train_edge_expert_${TAG}.log"

echo "[CONFIG] dataset_jsonl=$DATASET"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET"
  echo "Expected after prepare_edge_dataset.py"
  exit 1
fi

if [[ -z "${BASE_CKPT:-}" || -z "${TOKENIZER:-}" ]]; then
  echo "BASE_CKPT or TOKENIZER not set. Check your paths.sh."
  exit 1
fi

EPOCHS="${EDGE_LORA_EPOCHS:-3}"
BATCH="${EDGE_LORA_BATCH:-4}"
LR="${EDGE_LORA_LR:-5e-5}"
ROUTER_LR="${EDGE_LORA_ROUTER_LR:-1e-4}"
WARMUP="${EDGE_LORA_WARMUP:-200}"
LAMBDA_GAIN="${EDGE_LORA_LAMBDA_GAIN:-0.0}"
GAIN_MARGIN="${EDGE_LORA_GAIN_MARGIN:-0.05}"
LAMBDA_ROUTER="${EDGE_LORA_LAMBDA_ROUTER:-1e-4}"
LAMBDA_ENTROPY="${EDGE_LORA_LAMBDA_ENTROPY:-1e-4}"
TARGET_MODULES="${EDGE_LORA_TARGET_MODULES:-attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}"
RANK="${EDGE_LORA_RANK:-16}"
ALPHA="${EDGE_LORA_ALPHA:-32}"
DROPOUT="${EDGE_LORA_DROPOUT:-0.05}"
MAXLEN="${EDGE_LORA_MAXLEN:-512}"
GRAD_CLIP="${EDGE_LORA_GRAD_CLIP:-1.0}"
ROUTER_PREPROCESS="${EDGE_ROUTER_PREPROCESS:-identity}"
ROUTER_PREPROCESS_EMB="${EDGE_ROUTER_PREPROCESS_EMB:-}"
ROUTER_PREPROCESS_JSON="${EDGE_ROUTER_PREPROCESS_JSON:-}"
if [[ -n "$ROUTER_PREPROCESS_JSON" ]]; then
  ROUTER_PREPROCESS_EMB=""
fi
ROUTER_PREPROCESS_STD_FLOOR="${EDGE_ROUTER_PREPROCESS_STD_FLOOR:-1e-3}"
ROUTER_PREPROCESS_CLIP="${EDGE_ROUTER_PREPROCESS_CLIP:-5.0}"
ROUTER_PREPROCESS_MASK_MODE="${EDGE_ROUTER_PREPROCESS_MASK_MODE:-zero}"
ROUTER_PREPROCESS_FALLBACK_DIM="${EDGE_ROUTER_PREPROCESS_FALLBACK_DIM:-4}"
TRAIN_ROUTER_ONLY="${EDGE_TRAIN_ROUTER_ONLY:-0}"
INIT_EXPERT_DIR="${EDGE_INIT_EXPERT_DIR:-}"
RESUME_PREV="${EDGE_RESUME_PREV:-1}"
if [[ -z "$INIT_EXPERT_DIR" && "$IDX" -gt 0 && "$RESUME_PREV" == "1" ]]; then
  PREV_ITER=$(printf "iter%02d" "$((IDX - 1))")
  if [[ -n "${EDGE_PREV_EXPERT_TAG:-}" ]]; then
    PREV_TAG="$EDGE_PREV_EXPERT_TAG"
  elif [[ "$TAG" =~ ^v([0-9]+)_gain$ ]]; then
    PREV_NUM="${BASH_REMATCH[1]}"
    if [[ "$PREV_NUM" -le 2 ]]; then
      PREV_TAG="v1_init"
    else
      PREV_TAG="v$((PREV_NUM - 1))_gain"
    fi
  else
    PREV_TAG="v1_init"
  fi
  INIT_EXPERT_DIR="$RUN_ROOT/${HW}/${PREV_ITER}/experts/${PREV_TAG}"
fi
INIT_EXPERT_ARGS=()
if [[ -n "$INIT_EXPERT_DIR" ]]; then
  if [[ ! -d "$INIT_EXPERT_DIR" ]]; then
    echo "init_expert_dir not found: $INIT_EXPERT_DIR"
    exit 1
  fi
  if [[ -f "$INIT_EXPERT_DIR/router.json" ]]; then
    INIT_HW_LIST="$(EDGE_INIT_ROUTER="$INIT_EXPERT_DIR/router.json" python - <<'PY'
import json
import os

p = os.environ.get("EDGE_INIT_ROUTER")
data = json.load(open(p, "r", encoding="utf-8"))
meta = data.get("meta", {})
hw = meta.get("hardware_ids") or meta.get("hardware_id") or meta.get("hardware_name")
if isinstance(hw, str):
    print(hw)
elif isinstance(hw, list):
    print(",".join(str(x) for x in hw))
PY
)"
    if [[ -n "$INIT_HW_LIST" ]]; then
      if [[ ",$INIT_HW_LIST," != *",$HW,"* ]]; then
        if [[ "${EDGE_ALLOW_CROSS_HW_INIT:-0}" != "1" ]]; then
          echo "init_expert_dir hardware mismatch: init_hw=[$INIT_HW_LIST] current_hw=[$HW]"
          echo "Set EDGE_ALLOW_CROSS_HW_INIT=1 to override."
          exit 1
        else
          echo "[WARN] init_expert_dir hardware mismatch: init_hw=[$INIT_HW_LIST] current_hw=[$HW] (override enabled)"
        fi
      fi
    fi
  fi
  INIT_EXPERT_ARGS=(--init-expert-dir "$INIT_EXPERT_DIR")
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "[CONFIG] init_expert_dir=${INIT_EXPERT_DIR:-<cold_start>}"
echo "[CONFIG] lambda_gain=$LAMBDA_GAIN gain_margin=$GAIN_MARGIN lambda_router=$LAMBDA_ROUTER lambda_entropy=$LAMBDA_ENTROPY"

CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/train_edge_expert.py" \
  --base-model-path "$BASE_CKPT" \
  --tokenizer-path "$TOKENIZER" \
  --dataset-jsonl "$DATASET" \
  --output-dir "$OUT_DIR" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --router-learning-rate "$ROUTER_LR" \
  --warmup-steps "$WARMUP" \
  --lambda-gain "$LAMBDA_GAIN" \
  --gain-margin "$GAIN_MARGIN" \
  --lambda-router "$LAMBDA_ROUTER" \
  --lambda-entropy "$LAMBDA_ENTROPY" \
  --target-modules "$TARGET_MODULES" \
  --lora-rank "$RANK" \
  --lora-alpha "$ALPHA" \
  --lora-dropout "$DROPOUT" \
  --max-length "$MAXLEN" \
  --gradient-clip "$GRAD_CLIP" \
  --router-preprocess "$ROUTER_PREPROCESS" \
  ${ROUTER_PREPROCESS_JSON:+--router-preprocess-json "$ROUTER_PREPROCESS_JSON"} \
  ${ROUTER_PREPROCESS_EMB:+--router-preprocess-embeddings "$ROUTER_PREPROCESS_EMB"} \
  --router-preprocess-std-floor "$ROUTER_PREPROCESS_STD_FLOOR" \
  --router-preprocess-clip "$ROUTER_PREPROCESS_CLIP" \
  --router-preprocess-mask-mode "$ROUTER_PREPROCESS_MASK_MODE" \
  --router-preprocess-fallback-dim "$ROUTER_PREPROCESS_FALLBACK_DIM" \
  $( [[ "$TRAIN_ROUTER_ONLY" == "1" ]] && echo "--train-router-only" ) \
  "${INIT_EXPERT_ARGS[@]}" \
  | tee "$LOG_FILE"
