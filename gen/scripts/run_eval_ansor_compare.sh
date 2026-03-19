#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_eval_ansor_compare.sh <hw>"
  echo "  hw: 2080|2080ti|3090|4090|v100"
  echo "Env:"
  echo "  EDGE_KV_ALIGNER_OLD=/path/to/old/hw_kv_aligner.pt"
  echo "  EDGE_KV_ALIGNER_NEW=/path/to/new/hw_kv_aligner.pt"
  echo "  EDGE_EMB_OLD=/path/to/hardware_embeddings_v4.json (optional)"
  echo "  EDGE_EMB_NEW=/path/to/hardware_embeddings_v4_universe.json (optional)"
  echo "  EDGE_OFFICIAL_CKPT=/path/to/clm_gen_best_v100 (optional)"
  echo "  EDGE_OFFICIAL_TOKENIZER=/path/to/tokenizer (optional, defaults to ckpt)"
  echo "  EDGE_EVAL_SKETCH_DIR=/path/to/sketch_dir (optional)"
  echo "  EDGE_EVAL_OUT_DIR=/path/to/output_dir (optional)"
  echo "  EDGE_EVAL_MEASURE_DIR=/path/to/measure_dir (optional)"
  echo "  EDGE_EVAL_DATASET_DIR=/path/to/to_measure_programs/<hw> (optional)"
  echo "  EDGE_KEEP_CNT=64 (optional)"
  echo "  EDGE_CUDA=0 (optional)"
  echo "  EDGE_FORCE=1 (optional, overwrite outputs)"
  echo "  EDGE_SKIP_MEASURE=1 (optional)"
  echo "  EDGE_SKIP_COMPARE=1 (optional)"
  exit 1
fi

HW_RAW="$1"
PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

EDGE_EMB_OLD="${EDGE_EMB_OLD:-${HW_EMB_V4:-}}"
EDGE_EMB_NEW="${EDGE_EMB_NEW:-${HW_EMB_V4U:-${HW_EMB_V4:-}}}"
EDGE_KV_ALIGNER_OLD="${EDGE_KV_ALIGNER_OLD:-${HW_KV_ALIGNER:-}}"
EDGE_KV_ALIGNER_NEW="${EDGE_KV_ALIGNER_NEW:-}"

if [[ -z "$EDGE_KV_ALIGNER_NEW" ]]; then
  echo "Missing EDGE_KV_ALIGNER_NEW (path to new aligner checkpoint)."
  exit 1
fi

for f in "$EDGE_EMB_OLD" "$EDGE_EMB_NEW" "$EDGE_KV_ALIGNER_OLD" "$EDGE_KV_ALIGNER_NEW"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "Missing file: $f"
    exit 1
  fi
done

HW="${HW_RAW,,}"
TARGET_2080_DEFAULT="cuda -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32"

if [[ "$HW" == "3090" ]]; then
  TARGET="$TARGET_4090"
  HW_ID="3090"
  HW="3090"
elif [[ "$HW" == "4090" ]]; then
  TARGET="$TARGET_4090"
  HW_ID="4090"
  HW="4090"
elif [[ "$HW" == "v100" ]]; then
  TARGET="$TARGET_V100"
  HW_ID="v100"
  HW="v100"
elif [[ "$HW" == "2080" || "$HW" == "2080ti" ]]; then
  TARGET="${TARGET_2080:-$TARGET_2080_DEFAULT}"
  HW_ID="2080"
  HW="2080"
else
  echo "Invalid hw: $HW_RAW (expect 2080|2080ti|3090|4090|v100)"
  exit 1
fi

DATASET_DIR="${EDGE_EVAL_DATASET_DIR:-$DATA_ROOT/dataset/to_measure_programs/${HW}}"
if [[ ! -d "$DATASET_DIR" ]]; then
  if [[ "$HW" == "3090" ]]; then
    DATASET_DIR="$DATA_ROOT/dataset/to_measure_programs/4090"
    if [[ -d "$DATASET_DIR" ]]; then
      echo "[WARN] 3090 dataset missing; fallback to 4090 dataset: $DATASET_DIR"
    else
      echo "Dataset dir not found: $DATASET_DIR"
      exit 1
    fi
  else
    echo "Dataset dir not found: $DATASET_DIR"
    exit 1
  fi
fi

KEEP_CNT="${EDGE_KEEP_CNT:-64}"
CUDA_ID="${EDGE_CUDA:-0}"
FORCE="${EDGE_FORCE:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
SKIP_COMPARE="${EDGE_SKIP_COMPARE:-0}"
MEASURE_BATCH="${EDGE_MEASURE_BATCH:-64}"
MATCH_MODE="${EDGE_MATCH_MODE:-i}"

SKETCH_DIR="${EDGE_EVAL_SKETCH_DIR:-$RUN_ROOT/${HW}/iter00/eval_ansor_sketch}"
OUT_DIR="${EDGE_EVAL_OUT_DIR:-$RUN_ROOT/${HW}/iter00/eval_ansor_gen}"
MEASURE_DIR="${EDGE_EVAL_MEASURE_DIR:-$RUN_ROOT/${HW}/iter00/eval_ansor_measure}"
SKETCH_PATH="$SKETCH_DIR/0_merge.json"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$MEASURE_DIR" "$LOG_DIR"

if [[ ! -s "$SKETCH_PATH" ]]; then
  echo "[STEP] sketch"
  mkdir -p "$SKETCH_DIR"
  python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_eval_sketch_ansor \
    --target "$TARGET" \
    --dataset_path "$DATASET_DIR" \
    --tokenizer_path "$TOKENIZER" \
    --save_path "$SKETCH_DIR" \
    --keep_cnt "$KEEP_CNT"
  if [[ ! -s "$SKETCH_PATH" ]]; then
    echo "Sketch not generated or empty: $SKETCH_PATH"
    exit 1
  fi
fi

run_bucket_only() {
  local out="$OUT_DIR/bucket_only.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_debug_kv.py" \
    --model_path "$BASE_CKPT" \
    --tokenizer_path "$TOKENIZER" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out" \
    --target "$TARGET" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    | tee "$LOG_DIR/gen_bucket_only.log"
}

run_kv() {
  local tag="$1"
  local aligner="$2"
  local emb="$3"
  local out="$OUT_DIR/${tag}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_debug_kv.py" \
    --model_path "$BASE_CKPT" \
    --tokenizer_path "$TOKENIZER" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out" \
    --target "$TARGET" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    --use_hw_kv --hw_kv_mode real \
    --hw_kv_aligner_path "$aligner" \
    --hardware_embedding_path "$emb" \
    --hw_kv_num_slots 4 \
    --pos_compensate \
    | tee "$LOG_DIR/gen_${tag}.log"
}

OFFICIAL_CKPT="${EDGE_OFFICIAL_CKPT:-$GEN_DATA/clm_gen_best_v100}"
OFFICIAL_TOKENIZER="${EDGE_OFFICIAL_TOKENIZER:-$OFFICIAL_CKPT}"

run_official() {
  local out="$OUT_DIR/official.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state.py" \
    --model_path "$OFFICIAL_CKPT" \
    --tokenizer_path "$OFFICIAL_TOKENIZER" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out" \
    --target "$TARGET" \
    --keep_cnt "$KEEP_CNT" \
    | tee "$LOG_DIR/gen_official.log"
}

echo "[STEP] bucket-only"
run_bucket_only

echo "[STEP] kv-old"
run_kv "bucket_kv_old" "$EDGE_KV_ALIGNER_OLD" "$EDGE_EMB_OLD"

echo "[STEP] kv-new"
run_kv "bucket_kv_new" "$EDGE_KV_ALIGNER_NEW" "$EDGE_EMB_NEW"

echo "[STEP] official"
run_official

if [[ "$SKIP_MEASURE" == "1" ]]; then
  echo "[STEP] measure (skip)"
else
  echo "[STEP] measure"
  python "$TLM_ROOT/gen/scripts/measure_watchdog.py" \
    --repo-root "$TLM_ROOT" \
    --target "$TARGET" \
    --batch-size "$MEASURE_BATCH" \
    --match-mode "$MATCH_MODE" \
    --job "$OUT_DIR/bucket_only.json=$MEASURE_DIR/bucket_only_measured.json" \
    --job "$OUT_DIR/bucket_kv_old.json=$MEASURE_DIR/bucket_kv_old_measured.json" \
    --job "$OUT_DIR/bucket_kv_new.json=$MEASURE_DIR/bucket_kv_new_measured.json" \
    --job "$OUT_DIR/official.json=$MEASURE_DIR/official_measured.json"
fi

if [[ "$SKIP_COMPARE" == "1" ]]; then
  echo "[STEP] compare (skip)"
else
  echo "[STEP] compare"
  python "$TLM_ROOT/gen/scripts/compare_measured_versions.py" \
    --target "$HW_ID" \
    --compare-mode intersection \
    --measured bucket_only="$MEASURE_DIR/bucket_only_measured.json" \
    --measured kv_old="$MEASURE_DIR/bucket_kv_old_measured.json" \
    --measured kv_new="$MEASURE_DIR/bucket_kv_new_measured.json" \
    --measured official="$MEASURE_DIR/official_measured.json" \
    --baseline bucket_only \
    --output-network-csv "$MEASURE_DIR/compare_network.csv" \
    --output-workload-csv "$MEASURE_DIR/compare_workload.csv"
fi

echo "Outputs saved to: $OUT_DIR"
echo "Measured saved to: $MEASURE_DIR"
