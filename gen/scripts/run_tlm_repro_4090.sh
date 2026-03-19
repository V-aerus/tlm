#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 4090 上复刻 TLM（非 LoRA）迭代流水线（i -> i+2）
#
# 显卡分工（可按需改）：
#   EDGE_GPU_TRAIN=1   # SFT 训练卡
#   EDGE_GPU_GEN=2     # gen_state 生成卡
#   EDGE_GPU_MEASURE=3 # measure 卡
#
# 典型流程：
#   1) 先准备四个 batch sketch（B0..B3）：
#      bash gen/scripts/run_tlm_repro_4090.sh sketch 0
#      bash gen/scripts/run_tlm_repro_4090.sh sketch 1
#      bash gen/scripts/run_tlm_repro_4090.sh sketch 2
#      bash gen/scripts/run_tlm_repro_4090.sh sketch 3
#
#   2) 预热（base 生成并测量 B0/B1）：
#      bash gen/scripts/run_tlm_repro_4090.sh gen 0 "$EDGE_BASE_MODEL" base
#      bash gen/scripts/run_tlm_repro_4090.sh measure 0 base
#      bash gen/scripts/run_tlm_repro_4090.sh gen 1 "$EDGE_BASE_MODEL" base
#      bash gen/scripts/run_tlm_repro_4090.sh measure 1 base
#
#   3) 训练 TLM2（使用 records<=iter00），并生成 B2：
#      bash gen/scripts/run_tlm_repro_4090.sh train 0 tlm2
#      bash gen/scripts/run_tlm_repro_4090.sh gen 2 "$RUN_ROOT/4090/models/tlm2" tlm2
#      bash gen/scripts/run_tlm_repro_4090.sh measure 2 tlm2
#
#   4) 训练 TLM3（使用 records<=iter01），并生成 B3，以此循环。
#
# 说明：
#   - 每次 train 都“从 EDGE_BASE_MODEL 冷启动”，符合论文逻辑。
#   - measure 阶段默认自动写入 utils.json(4090.measure_records + measure_records_base)。
# -----------------------------------------------------------------------------

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash gen/scripts/run_tlm_repro_4090.sh <cmd> [args...]

Commands:
  sketch <iter_idx>
      生成该轮 sketch（test_file_idx = iter_idx % EDGE_NUM_BATCHES）

  gen <iter_idx> <model_path> [gen_tag]
      用指定模型生成候选（默认 gen_tag=candidates）

  measure <iter_idx> [gen_tag]
      测量对应 gen 输出，并自动追加到 utils.json

  train <iter_max> <model_tag>
      用 records<=iter_max 训练一个新 TLM（从 base 冷启动）

  status
      打印当前迭代目录的 sketch/gen/measure 简表

Environment (常用):
  EDGE_GPU_TRAIN=1
  EDGE_GPU_GEN=2
  EDGE_GPU_MEASURE=3

  TLM_ROOT=/home/hehangshuai/workspace/tlm
  DATA_ROOT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen
  RUN_ROOT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart

  EDGE_TARGET_4090='cuda -keys=cuda,gpu -arch=sm_86 ...'
  EDGE_BASE_MODEL=$DATA_ROOT/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000
  EDGE_TOKENIZER=$DATA_ROOT/gen_data/Model/gen_tokenizer_multi_v1_bucket
  EDGE_TO_MEASURE_DIR=$DATA_ROOT/dataset/to_measure_programs/4090

  EDGE_NUM_BATCHES=4
  EDGE_SKETCH_KEEP_CNT=48
  EDGE_GEN_KEEP_CNT=16
  EDGE_MEASURE_BATCH=64
  EDGE_FORCE=0

  EDGE_TRAIN_EPOCHS=3
  EDGE_TRAIN_BATCH=5
  EDGE_TRAIN_LR=5e-5
  EDGE_TRAIN_LOG_STEPS=100
  EDGE_TRAIN_SAVE_STEPS=2000
  EDGE_AUTO_REGISTER=1
EOF
  exit 0
fi

CMD="$1"; shift

TLM_ROOT="${TLM_ROOT:-/home/hehangshuai/workspace/tlm}"
DATA_ROOT="${DATA_ROOT:-$TLM_ROOT/tlm_dataset/gen}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart}"

EDGE_GPU_TRAIN="${EDGE_GPU_TRAIN:-2}"
EDGE_GPU_GEN="${EDGE_GPU_GEN:-2}"
EDGE_GPU_MEASURE="${EDGE_GPU_MEASURE:-1}"

EDGE_TARGET_4090="${EDGE_TARGET_4090:-cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32}"
EDGE_BASE_MODEL="${EDGE_BASE_MODEL:-$DATA_ROOT/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000}"
EDGE_TOKENIZER="${EDGE_TOKENIZER:-$DATA_ROOT/gen_data/Model/gen_tokenizer_multi_v1_bucket}"
EDGE_TO_MEASURE_DIR="${EDGE_TO_MEASURE_DIR:-$DATA_ROOT/dataset/to_measure_programs/4090}"

EDGE_NUM_BATCHES="${EDGE_NUM_BATCHES:-4}"
EDGE_SKETCH_KEEP_CNT="${EDGE_SKETCH_KEEP_CNT:-48}"
EDGE_GEN_KEEP_CNT="${EDGE_GEN_KEEP_CNT:-16}"
EDGE_MEASURE_BATCH="${EDGE_MEASURE_BATCH:-64}"
EDGE_FORCE="${EDGE_FORCE:-0}"
EDGE_AUTO_REGISTER="${EDGE_AUTO_REGISTER:-1}"

EDGE_TRAIN_EPOCHS="${EDGE_TRAIN_EPOCHS:-3}"
EDGE_TRAIN_BATCH="${EDGE_TRAIN_BATCH:-5}"
EDGE_TRAIN_LR="${EDGE_TRAIN_LR:-5e-5}"
EDGE_TRAIN_LOG_STEPS="${EDGE_TRAIN_LOG_STEPS:-100}"
EDGE_TRAIN_SAVE_STEPS="${EDGE_TRAIN_SAVE_STEPS:-2000}"

iter_name() {
  local idx="$1"
  printf "iter%02d" "$idx"
}

iter_dir() {
  local idx="$1"
  echo "$RUN_ROOT/4090/$(iter_name "$idx")"
}

ensure_common() {
  if [[ ! -d "$TLM_ROOT/gen" ]]; then
    echo "TLM_ROOT invalid: $TLM_ROOT"
    exit 1
  fi
  if [[ ! -d "$EDGE_TO_MEASURE_DIR" ]]; then
    echo "to_measure dir missing: $EDGE_TO_MEASURE_DIR"
    exit 1
  fi
  if [[ ! -d "$EDGE_BASE_MODEL" ]]; then
    echo "base model missing: $EDGE_BASE_MODEL"
    exit 1
  fi
  if [[ ! -d "$EDGE_TOKENIZER" ]]; then
    echo "tokenizer missing: $EDGE_TOKENIZER"
    exit 1
  fi
}

print_config() {
  echo "[CONFIG] TLM_ROOT=$TLM_ROOT"
  echo "[CONFIG] DATA_ROOT=$DATA_ROOT"
  echo "[CONFIG] RUN_ROOT=$RUN_ROOT"
  echo "[CONFIG] target=$EDGE_TARGET_4090"
  echo "[CONFIG] gpu(train/gen/measure)=${EDGE_GPU_TRAIN}/${EDGE_GPU_GEN}/${EDGE_GPU_MEASURE}"
}

cmd_sketch() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: ... sketch <iter_idx>"
    exit 1
  fi
  local idx="$1"
  local idir
  idir="$(iter_dir "$idx")"
  local sketch_dir="$idir/sketch"
  local sketch_path="$sketch_dir/0_merge.json"
  local file_idx=$((idx % EDGE_NUM_BATCHES))
  mkdir -p "$sketch_dir"
  if [[ -s "$sketch_path" && "$EDGE_FORCE" != "1" ]]; then
    echo "Sketch exists: $sketch_path (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi

  print_config
  echo "[STEP] sketch iter=$idx test_file_idx=$file_idx keep_cnt=$EDGE_SKETCH_KEEP_CNT"
  python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_train_sketch \
    --target "$EDGE_TARGET_4090" \
    --dataset_path "$EDGE_TO_MEASURE_DIR" \
    --tokenizer_path "$EDGE_TOKENIZER" \
    --save_path "$sketch_dir" \
    --keep_cnt "$EDGE_SKETCH_KEEP_CNT" \
    --test_file_idx "$file_idx"
}

cmd_gen() {
  if [[ $# -lt 2 ]]; then
    echo "Usage: ... gen <iter_idx> <model_path> [gen_tag]"
    exit 1
  fi
  local idx="$1"
  local model_path="$2"
  local gen_tag="${3:-candidates}"
  local idir
  idir="$(iter_dir "$idx")"
  local sketch_path="$idir/sketch/0_merge.json"
  local gen_dir="$idir/gen"
  local out_json="$gen_dir/${gen_tag}.json"
  local log_dir="$idir/logs"
  mkdir -p "$gen_dir" "$log_dir"

  if [[ ! -s "$sketch_path" ]]; then
    echo "Sketch missing: $sketch_path"
    exit 1
  fi
  if [[ ! -d "$model_path" ]]; then
    echo "Model path missing: $model_path"
    exit 1
  fi
  if [[ -s "$out_json" && "$EDGE_FORCE" != "1" ]]; then
    echo "Gen output exists: $out_json (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi

  print_config
  echo "[STEP] gen iter=$idx tag=$gen_tag model=$model_path keep_cnt=$EDGE_GEN_KEEP_CNT"
  CUDA_VISIBLE_DEVICES="$EDGE_GPU_GEN" python "$TLM_ROOT/gen/gen_state.py" \
    --model_path "$model_path" \
    --tokenizer_path "$EDGE_TOKENIZER" \
    --sketch_path "$sketch_path" \
    --save_path "$out_json" \
    --target "$EDGE_TARGET_4090" \
    --keep_cnt "$EDGE_GEN_KEEP_CNT" \
    --allow_repeat=True \
    | tee "$log_dir/gen_${gen_tag}.log"
}

cmd_measure() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: ... measure <iter_idx> [gen_tag]"
    exit 1
  fi
  local idx="$1"
  local gen_tag="${2:-candidates}"
  local idir
  idir="$(iter_dir "$idx")"
  local gen_json="$idir/gen/${gen_tag}.json"
  local measure_dir="$idir/measure"
  local measured_json="$measure_dir/${gen_tag}_measured.json"
  local log_dir="$idir/logs"
  mkdir -p "$measure_dir" "$log_dir"

  if [[ ! -s "$gen_json" ]]; then
    echo "Gen output missing: $gen_json"
    exit 1
  fi
  if [[ -s "$measured_json" && "$EDGE_FORCE" != "1" ]]; then
    echo "Measured exists, will resume/verify completeness via watchdog: $measured_json"
  fi

  print_config
  echo "[STEP] measure iter=$idx tag=$gen_tag batch=$EDGE_MEASURE_BATCH"
  echo "[ENV] CUDA_VISIBLE_DEVICES=$EDGE_GPU_MEASURE"
  CUDA_VISIBLE_DEVICES="$EDGE_GPU_MEASURE" python "$TLM_ROOT/gen/scripts/measure_watchdog.py" \
    --repo-root "$TLM_ROOT" \
    --target "$EDGE_TARGET_4090" \
    --batch-size "$EDGE_MEASURE_BATCH" \
    --log-path "$log_dir/measure_${gen_tag}.log" \
    --job "$gen_json=$measured_json"

  if [[ "$EDGE_AUTO_REGISTER" == "1" ]]; then
    echo "[STEP] add_measure_records iter=$idx"
    python "$TLM_ROOT/gen/scripts/add_measure_records.py" \
      --hardware 4090 \
      --iter "$idx" \
      --mode base \
      --measured-path "$measured_json"
  fi
}

cmd_train() {
  if [[ $# -lt 2 ]]; then
    echo "Usage: ... train <iter_max> <model_tag>"
    exit 1
  fi
  local iter_max="$1"
  local model_tag="$2"

  local record_dir="$DATA_ROOT/dataset/measure_records/4090"
  local sft_dir="$RUN_ROOT/4090/sft_upto_iter$(printf "%02d" "$iter_max")"
  local sft_json="$sft_dir/0_merge.json"
  local model_out="$RUN_ROOT/4090/models/$model_tag"
  local log_dir="$RUN_ROOT/4090/logs"
  mkdir -p "$sft_dir" "$model_out" "$log_dir"

  print_config
  echo "[STEP] postprocess all iter<=${iter_max}"
  python "$TLM_ROOT/gen/postprocess.py" \
    --target "$EDGE_TARGET_4090" \
    --record-mode all \
    --record-dir "$record_dir" \
    --iter-max "$iter_max"

  echo "[STEP] make_dataset for_gen_best"
  python "$TLM_ROOT/gen/make_dataset.py" \
    --for_type=for_gen_best \
    --target "$EDGE_TARGET_4090" \
    --dataset_path "$record_dir" \
    --tokenizer_path "$EDGE_TOKENIZER" \
    --save_path "$sft_dir"

  if [[ ! -s "$sft_json" ]]; then
    echo "SFT json missing: $sft_json"
    exit 1
  fi

  echo "[STEP] train model_tag=$model_tag (from base cold start)"
  CUDA_VISIBLE_DEVICES="$EDGE_GPU_TRAIN" python "$TLM_ROOT/gen/train_clm.py" \
    --do_train \
    --model_type gpt2 \
    --tokenizer_name "$EDGE_TOKENIZER" \
    --output_dir "$model_out" \
    --train_file "$sft_json" \
    --model_name_or_path "$EDGE_BASE_MODEL" \
    --per_device_train_batch_size "$EDGE_TRAIN_BATCH" \
    --num_train_epochs "$EDGE_TRAIN_EPOCHS" \
    --learning_rate "$EDGE_TRAIN_LR" \
    --lr_scheduler_type constant \
    --logging_steps "$EDGE_TRAIN_LOG_STEPS" \
    --save_steps "$EDGE_TRAIN_SAVE_STEPS" \
    | tee "$log_dir/train_${model_tag}.log"
}

cmd_status() {
  print_config
  echo "[STATUS] 4090 iter dirs under $RUN_ROOT/4090"
  for idir in "$RUN_ROOT"/4090/iter*; do
    [[ -d "$idir" ]] || continue
    iter_base
    iter_base="$(basename "$idir")"
    sketch="N"; gen_cnt=0; meas_cnt=0
    [[ -s "$idir/sketch/0_merge.json" ]] && sketch="Y"
    [[ -d "$idir/gen" ]] && gen_cnt=$(find "$idir/gen" -maxdepth 1 -type f -name "*.json" | wc -l | tr -d ' ')
    [[ -d "$idir/measure" ]] && meas_cnt=$(find "$idir/measure" -maxdepth 1 -type f -name "*_measured.json" | wc -l | tr -d ' ')
    echo "  - $iter_base sketch=$sketch gen_json=$gen_cnt measured_json=$meas_cnt"
  done
}

ensure_common
mkdir -p "$RUN_ROOT/4090"

case "$CMD" in
  sketch) cmd_sketch "$@" ;;
  gen) cmd_gen "$@" ;;
  measure) cmd_measure "$@" ;;
  train) cmd_train "$@" ;;
  status) cmd_status "$@" ;;
  *)
    echo "Unknown cmd: $CMD"
    exit 1
    ;;
esac
