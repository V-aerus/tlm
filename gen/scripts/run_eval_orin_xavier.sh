#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_eval_orin_xavier.sh <xavier_expert_dir>"
  echo "Env (optional):"
  echo "  TLM_ROOT=/home/orin/work/tlm"
  echo "  DATA_ROOT=\$TLM_ROOT/tlm_dataset/gen"
  echo "  RUN_ROOT=\$DATA_ROOT/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart"
  echo "  EDGE_BASE_CKPT=/path/to/base_ckpt"
  echo "  EDGE_TOKENIZER=/path/to/tokenizer"
  echo "  EDGE_EMB_PATH=/path/to/hardware_embeddings_v5_draft.json"
  echo "  EDGE_HW_KV_ALIGNER=/path/to/hw_kv_aligner.pt"
  echo "  EDGE_TARGET_ORIN='cuda -keys=... -arch=sm_87 ...'"
  echo "  EDGE_TARGET_HOST_ORIN='llvm -keys=arm_cpu,cpu -mcpu=cortex-a78 -mtriple=aarch64-linux-gnu -num-cores=12'"
  echo "  EDGE_TARGET_HARDWARE=nvidia/jetson-orin"
  echo "  EDGE_ORIN_MODEL=orin"
  echo "  EDGE_SKIP_DUMP=0   (1 to skip network_info/to_measure generation)"
  echo "  EDGE_SKIP_SKETCH=0 (1 to skip sketch generation)"
  echo "  EDGE_SKIP_GEN=0    (1 to skip gen_state)"
  echo "  EDGE_SKIP_MEASURE=0 (1 to skip measure)"
  echo "  EDGE_SKIP_OFFICIAL=0 (1 to skip official baseline)"
  echo "  EDGE_FORCE=0       (1 to overwrite existing outputs)"
  echo "  EDGE_CUDA=0"
  echo "  EDGE_KEEP_CNT=64"
  echo "  EDGE_MEASURE_BATCH=64"
  exit 1
fi

XAVIER_EXPERT_DIR="$1"
if [[ ! -d "$XAVIER_EXPERT_DIR" ]]; then
  echo "xavier expert dir not found: $XAVIER_EXPERT_DIR"
  exit 1
fi
if [[ ! -f "$XAVIER_EXPERT_DIR/router.json" ]]; then
  echo "router.json not found in expert dir: $XAVIER_EXPERT_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLM_ROOT="${TLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$TLM_ROOT/tlm_dataset/gen}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart}"

PATHS_SH="${EDGE_PATHS_SH:-$RUN_ROOT/_shared/paths.sh}"
if [[ -f "$PATHS_SH" ]]; then
  # shellcheck disable=SC1090
  source "$PATHS_SH"
fi

EDGE_TARGET_ORIN="${EDGE_TARGET_ORIN:-cuda -keys=cuda,gpu -arch=sm_87 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32}"
EDGE_TARGET_HOST_ORIN="${EDGE_TARGET_HOST_ORIN:-llvm -keys=arm_cpu,cpu -mcpu=cortex-a78 -mtriple=aarch64-linux-gnu -num-cores=12}"
EDGE_TARGET_HARDWARE="${EDGE_TARGET_HARDWARE:-nvidia/jetson-orin}"
EDGE_ORIN_MODEL="${EDGE_ORIN_MODEL:-orin}"

BASE_CKPT_PATH="${EDGE_BASE_CKPT:-${BASE_CKPT:-$DATA_ROOT/gen_data/clm_gen_best_v100}}"
TOKENIZER_PATH="${EDGE_TOKENIZER:-${TOKENIZER:-$BASE_CKPT_PATH}}"
OFFICIAL_CKPT="${EDGE_OFFICIAL_CKPT:-$BASE_CKPT_PATH}"
OFFICIAL_TOKENIZER="${EDGE_OFFICIAL_TOKENIZER:-$OFFICIAL_CKPT}"

EDGE_EMB_PATH="${EDGE_EMB_PATH:-${HW_EMB_V5:-$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json}}"
EDGE_HW_KV_ALIGNER="${EDGE_HW_KV_ALIGNER:-${HW_KV_ALIGNER:-}}"

CUDA_ID="${EDGE_CUDA:-0}"
KEEP_CNT="${EDGE_KEEP_CNT:-64}"
FORCE="${EDGE_FORCE:-0}"
USE_HW_KV="${EDGE_USE_HW_KV:-0}"
MEASURE_BATCH="${EDGE_MEASURE_BATCH:-64}"

SKIP_DUMP="${EDGE_SKIP_DUMP:-0}"
SKIP_SKETCH="${EDGE_SKIP_SKETCH:-0}"
SKIP_GEN="${EDGE_SKIP_GEN:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
SKIP_OFFICIAL="${EDGE_SKIP_OFFICIAL:-0}"

if [[ -n "${EDGE_OMP_THREADS:-}" ]]; then
  export OMP_NUM_THREADS="$EDGE_OMP_THREADS"
fi
if [[ -n "${EDGE_MKL_THREADS:-}" ]]; then
  export MKL_NUM_THREADS="$EDGE_MKL_THREADS"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NETWORK_INFO_DIR="${EDGE_ORIN_NETWORK_INFO_DIR:-$DATA_ROOT/dataset/network_info/$EDGE_ORIN_MODEL}"
TO_MEASURE_DIR="${EDGE_ORIN_TO_MEASURE_DIR:-$DATA_ROOT/dataset/to_measure_programs/$EDGE_ORIN_MODEL}"

EVAL_ROOT="${EDGE_ORIN_EVAL_ROOT:-$RUN_ROOT/orin/iter00/eval_xavier}"
SKETCH_DIR="${EDGE_ORIN_EVAL_SKETCH_DIR:-$EVAL_ROOT/sketch}"
GEN_DIR="${EDGE_ORIN_EVAL_GEN_DIR:-$EVAL_ROOT/gen}"
MEASURE_DIR="${EDGE_ORIN_EVAL_MEASURE_DIR:-$EVAL_ROOT/measure}"
LOG_DIR="${EDGE_ORIN_EVAL_LOG_DIR:-$EVAL_ROOT/logs}"
SKETCH_PATH="$SKETCH_DIR/0_merge.json"

mkdir -p "$NETWORK_INFO_DIR" "$TO_MEASURE_DIR" "$SKETCH_DIR" "$GEN_DIR" "$MEASURE_DIR" "$LOG_DIR"

if [[ ! -f "$EDGE_EMB_PATH" ]]; then
  echo "embedding file not found: $EDGE_EMB_PATH"
  exit 1
fi

if [[ "$USE_HW_KV" == "1" && -z "$EDGE_HW_KV_ALIGNER" ]]; then
  echo "[WARN] EDGE_USE_HW_KV=1 but EDGE_HW_KV_ALIGNER is empty; fallback to bucket-only (no hw_kv)."
  USE_HW_KV=0
fi
if [[ "$USE_HW_KV" == "1" && ! -f "$EDGE_HW_KV_ALIGNER" ]]; then
  echo "hw kv aligner not found: $EDGE_HW_KV_ALIGNER"
  exit 1
fi

if [[ "$SKIP_DUMP" != "1" ]]; then
  echo "[STEP] dump network_info + to_measure_programs (orin)"
  python "$TLM_ROOT/gen/tools/dump_network_info_and_programs_orin.py" \
    --cuda-target "$EDGE_TARGET_ORIN" \
    --host-target "$EDGE_TARGET_HOST_ORIN" \
    --model "$EDGE_ORIN_MODEL" \
    --data-root "$DATA_ROOT" \
    --network-info-dir "$NETWORK_INFO_DIR" \
    --to-measure-dir "$TO_MEASURE_DIR" \
    --dump_programs_size "${EDGE_DUMP_PROGRAMS_SIZE:-1000}" \
    --log_path "$LOG_DIR/dump_orin.log"
fi

if [[ "$SKIP_SKETCH" != "1" ]]; then
  if [[ -f "$SKETCH_PATH" && "$FORCE" != "1" ]]; then
    echo "Sketch exists: $SKETCH_PATH (set EDGE_FORCE=1 to overwrite)"
  else
    echo "[STEP] make eval sketch (bert/resnet/mobilenet/inception)"
    python "$TLM_ROOT/gen/make_dataset.py" \
      --for_type=for_gen_eval_sketch_ansor \
      --target "$EDGE_TARGET_ORIN" \
      --dataset_path "$TO_MEASURE_DIR" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --save_path "$SKETCH_DIR" \
      --keep_cnt "$KEEP_CNT"
  fi
fi

if [[ ! -s "$SKETCH_PATH" ]]; then
  echo "Sketch not found or empty: $SKETCH_PATH"
  exit 1
fi

run_gen_kv() {
  local tag="$1"
  local experts="$2"
  local out_json="$GEN_DIR/${tag}.json"
  local log_file="$LOG_DIR/gen_${tag}.log"
  if [[ -f "$out_json" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out_json (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  local cmd=(
    python "$TLM_ROOT/gen/gen_state_kv_lora.py"
    --model_path "$BASE_CKPT_PATH"
    --tokenizer_path "$TOKENIZER_PATH"
    --sketch_path "$SKETCH_PATH"
    --save_path "$out_json"
    --target "$EDGE_TARGET_ORIN"
    --target_hardware "$EDGE_TARGET_HARDWARE"
    --keep_cnt "$KEEP_CNT"
    --use_bucket
    --hardware_embedding_path "$EDGE_EMB_PATH"
    --pos_compensate
  )
  if [[ "$USE_HW_KV" == "1" ]]; then
    cmd+=(
      --use_hw_kv
      --hw_kv_mode real
      --hw_kv_aligner_path "$EDGE_HW_KV_ALIGNER"
    )
  fi
  if [[ -n "$experts" ]]; then
    cmd+=(
      --edge_expert_dirs "$experts"
      --edge_topk 1
      --edge_embedding_path "$EDGE_EMB_PATH"
      --edge_debug_topk
    )
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" "${cmd[@]}" | tee "$log_file"
}

run_gen_official() {
  local out_json="$GEN_DIR/official.json"
  local log_file="$LOG_DIR/gen_official.log"
  if [[ -f "$out_json" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out_json (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state.py" \
    --model_path "$OFFICIAL_CKPT" \
    --tokenizer_path "$OFFICIAL_TOKENIZER" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out_json" \
    --target "$EDGE_TARGET_ORIN" \
    --keep_cnt "$KEEP_CNT" \
    | tee "$log_file"
}

if [[ "$SKIP_GEN" != "1" ]]; then
  echo "[STEP] gen base_bucketkv"
  run_gen_kv "base_bucketkv" ""
  echo "[STEP] gen kv_lora_xavier"
  run_gen_kv "kv_lora_xavier" "$XAVIER_EXPERT_DIR"
  if [[ "$SKIP_OFFICIAL" != "1" ]]; then
    echo "[STEP] gen official"
    run_gen_official
  fi
fi

if [[ "$SKIP_MEASURE" != "1" ]]; then
  echo "[STEP] measure"
  JOB_ARGS=()
  if [[ -s "$GEN_DIR/base_bucketkv.json" ]]; then
    JOB_ARGS+=(--job "$GEN_DIR/base_bucketkv.json=$MEASURE_DIR/base_bucketkv_measured.json")
  fi
  if [[ -s "$GEN_DIR/kv_lora_xavier.json" ]]; then
    JOB_ARGS+=(--job "$GEN_DIR/kv_lora_xavier.json=$MEASURE_DIR/kv_lora_xavier_measured.json")
  fi
  if [[ -s "$GEN_DIR/official.json" ]]; then
    JOB_ARGS+=(--job "$GEN_DIR/official.json=$MEASURE_DIR/official_measured.json")
  fi
  if [[ ${#JOB_ARGS[@]} -eq 0 ]]; then
    echo "No gen outputs found for measurement under: $GEN_DIR"
    exit 1
  fi
  python "$TLM_ROOT/gen/scripts/measure_watchdog.py" \
    --repo-root "$TLM_ROOT" \
    --target "$EDGE_TARGET_ORIN" \
    --target-host "$EDGE_TARGET_HOST_ORIN" \
    --batch-size "$MEASURE_BATCH" \
    --log-path "$LOG_DIR/measure_watchdog.log" \
    "${JOB_ARGS[@]}"
fi

echo "[DONE] eval root: $EVAL_ROOT"
echo "  sketch : $SKETCH_PATH"
echo "  gen    : $GEN_DIR"
echo "  measure: $MEASURE_DIR"
