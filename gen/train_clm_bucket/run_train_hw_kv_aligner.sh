#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000"
TOKENIZER_PATH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket"
TRAIN_JSON_PATHS="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Train_data/align_train_multi_merged_no_ints/0_merge_4090_bucket_kv_gpu_only.json"
HW_EMB_PATH="/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4_universe.json"
HW_EMB_PREPROCESS="${HW_EMB_PREPROCESS:-/home/hehangshuai/workspace/tlm/gen/Embedding/preprocess_v4u_zscore_v1.json}"
HW_REQUIRE_PREPROCESS="${HW_REQUIRE_PREPROCESS:-1}"
OUT_DIR="${HW_KV_OUT_DIR:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train}"

LEARNING_RATE="${LEARNING_RATE:-3e-5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
KV_SCALE_INIT="${KV_SCALE_INIT:-0.1}"
KV_SCALE_WARMUP="${KV_SCALE_WARMUP:-2000}"
KV_SCALE_REG="${KV_SCALE_REG:-1e-3}"
KD_WEIGHT="${KD_WEIGHT:-0.4}"
MONITOR_KV_LAYER="${MONITOR_KV_LAYER:-11}"

$PYTHON gen/train_hw_kv_aligner.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_json_paths "$TRAIN_JSON_PATHS" \
  --hardware_embeddings_path "$HW_EMB_PATH" \
  ${HW_EMB_PREPROCESS:+--preprocess_json "$HW_EMB_PREPROCESS"} \
  $( [[ "$HW_REQUIRE_PREPROCESS" == "1" ]] && echo "--require-preprocess" ) \
  --output_dir "$OUT_DIR" \
  --per_device_batch_size 8 \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type linear \
  --warmup_steps 500 \
  --max_steps 70000 \
  --logging_steps 100 \
  --save_steps 2000 \
  --l2_reg 1e-5 \
  --max_grad_norm 1.0 \
  --kd_weight "$KD_WEIGHT" \
  --kd_temperature "$KD_TEMPERATURE" \
  --kd_mismatch_threshold 0.2 \
  --kv_scale_init "$KV_SCALE_INIT" \
  --kv_scale_warmup_steps "$KV_SCALE_WARMUP" \
  --kv_scale_max 2.0 \
  --kv_scale_reg "$KV_SCALE_REG" \
  --monitor_kv_every 100 \
  --monitor_kv_layer "$MONITOR_KV_LAYER"
