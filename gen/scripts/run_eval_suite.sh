#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash gen/scripts/run_eval_suite.sh <hw>"
  echo "  hw: 2080|2080ti|3090|4090|v100"
  echo "Env:"
  echo "  EDGE_EXPERT_4090=/path/to/4090/expert"
  echo "  EDGE_EXPERT_V100=/path/to/v100/expert"
  echo "  EDGE_OFFICIAL_CKPT=/path/to/clm_gen_best_v100 (optional)"
  echo "  EDGE_OFFICIAL_TOKENIZER=/path/to/tokenizer (optional, defaults to ckpt)"
  echo "  EDGE_EVAL_SKETCH_DIR=/path/to/sketch_dir (optional)"
  echo "  EDGE_EVAL_OUT_DIR=/path/to/output_dir (optional)"
  echo "  EDGE_EVAL_DATASET_DIR=/path/to/to_measure_programs/<hw> (optional)"
  echo "  EDGE_KEEP_CNT=64 (optional)"
  echo "  EDGE_CUDA=0 (optional)"
  echo "  EDGE_FORCE=1 (optional, overwrite outputs)"
  exit 1
fi

: <<'EDGE_EVAL_SUITE_USAGE'
## 用法（你确认过的官方路径已内置默认值）

先设置专家与官方模型：
export EDGE_EXPERT_DEBUG_TOPK=1
export RUN_ROOT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart
source /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh
export EDGE_EXPERT_4090=$RUN_ROOT/4090/iterXX/experts/vX_gain
export EDGE_EXPERT_V100=$RUN_ROOT/v100/iterYY/experts/vY_gain
export EDGE_OFFICIAL_CKPT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100

生成 2080Ti / 3090 四种 baseline：

bash /home/hehangshuai/workspace/tlm/gen/scripts/run_eval_suite.sh 2080
bash /home/hehangshuai/workspace/tlm/gen/scripts/run_eval_suite.sh 3090

默认产物：

$RUN_ROOT/<hw>/iter00/eval_ansor_sketch/0_merge.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_mix.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_4090.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_v100.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/official.json

———

## 关键说明

- 3090 推理用 TARGET_4090（tvm 能识别），同时 --target_hardware 3090 让路由使用 3090 embedding。
- 2080Ti 推理默认使用 sm_75 target（脚本内置默认值）。
- 脚本会检测评测草图是否存在，不存在会自动生成（for_gen_eval_sketch_ansor）。
EDGE_EVAL_SUITE_USAGE

HW_RAW="$1"
PATHS_SH="${EDGE_PATHS_SH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  exit 1
fi

# shellcheck disable=SC1090
source "$PATHS_SH"

EXPERT_4090="${EDGE_EXPERT_4090:-}"
EXPERT_V100="${EDGE_EXPERT_V100:-}"
if [[ -z "$EXPERT_4090" || -z "$EXPERT_V100" ]]; then
  echo "Missing experts. Set EDGE_EXPERT_4090 and EDGE_EXPERT_V100."
  exit 1
fi
EXPERT_4090="${EXPERT_4090//$'\r'/}"
EXPERT_V100="${EXPERT_V100//$'\r'/}"
if [[ ! -f "$EXPERT_4090/router.json" ]]; then
  echo "router.json not found in EDGE_EXPERT_4090: $EXPERT_4090"
  exit 1
fi
if [[ ! -f "$EXPERT_V100/router.json" ]]; then
  echo "router.json not found in EDGE_EXPERT_V100: $EXPERT_V100"
  exit 1
fi

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
CUDA_ID="${EDGE_CUDA:-1}"
FORCE="${EDGE_FORCE:-0}"
TOPK_MIX="${EDGE_TOPK_MIX:-2}"
TOPK_SINGLE="${EDGE_TOPK_SINGLE:-1}"

SKETCH_DIR="${EDGE_EVAL_SKETCH_DIR:-$RUN_ROOT/${HW}/iter00/eval_ansor_sketch}"
OUT_DIR="${EDGE_EVAL_OUT_DIR:-$RUN_ROOT/${HW}/iter00/eval_ansor_gen}"
SKETCH_PATH="$SKETCH_DIR/0_merge.json"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

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

run_kv_lora() {
  local tag="$1"
  local experts="$2"
  local topk="$3"
  local out="$OUT_DIR/kv_lora_${tag}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  if [[ "$tag" == "mix" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_kv_lora.py" \
      --model_path "$BASE_CKPT" \
      --tokenizer_path "$TOKENIZER" \
      --edge_expert_dirs "$experts" \
      --edge_topk "$topk" \
      --edge_debug_topk \
      --edge_embedding_path "$HW_EMB_V4" \
      --hardware_embedding_path "$HW_EMB_V4" \
      --sketch_path "$SKETCH_PATH" \
      --save_path "$out" \
      --target "$TARGET" \
      --target_hardware "$HW_ID" \
      --keep_cnt "$KEEP_CNT" \
      --use_bucket \
      --use_hw_kv --hw_kv_mode real \
      --hw_kv_aligner_path "$HW_KV_ALIGNER" \
      --pos_compensate \
      | tee "$LOG_DIR/gen_${tag}.log"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state_kv_lora.py" \
    --model_path "$BASE_CKPT" \
    --tokenizer_path "$TOKENIZER" \
    --edge_expert_dirs "$experts" \
    --edge_topk "$topk" \
    --edge_embedding_path "$HW_EMB_V4" \
    --hardware_embedding_path "$HW_EMB_V4" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out" \
    --target "$TARGET" \
    --target_hardware "$HW_ID" \
    --keep_cnt "$KEEP_CNT" \
    --use_bucket \
    --use_hw_kv --hw_kv_mode real \
    --hw_kv_aligner_path "$HW_KV_ALIGNER" \
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

echo "[STEP] lora-mix"
run_kv_lora "mix" "${EXPERT_4090},${EXPERT_V100}" "$TOPK_MIX"

echo "[STEP] lora-4090"
run_kv_lora "4090" "${EXPERT_4090}" "$TOPK_SINGLE"

echo "[STEP] lora-v100"
run_kv_lora "v100" "${EXPERT_V100}" "$TOPK_SINGLE"

echo "[STEP] official"
run_official

echo "Outputs saved to: $OUT_DIR"
