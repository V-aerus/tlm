#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000"
TOKENIZER_PATH="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket"
TRAIN_JSON_PATHS="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Train_data/align_train_multi_merged_no_ints/0_merge_4090_bucket_kv_gpu_only.json"
HW_EMB_PATH="/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json"
OUT_DIR="/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train"

$PYTHON gen/train_hw_kv_aligner.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_json_paths "$TRAIN_JSON_PATHS" \
  --hardware_embeddings_path "$HW_EMB_PATH" \
  --output_dir "$OUT_DIR" \
  --per_device_batch_size 8 \
  --learning_rate 3e-5 \
  --lr_scheduler_type linear \
  --warmup_steps 500 \
  --max_steps 70000 \
  --logging_steps 100 \
  --save_steps 2000 \
  --l2_reg 1e-5 \
  --max_grad_norm 1.0 \
  --kd_weight 0.2 \
  --kd_temperature 2.0 \
  --kd_mismatch_threshold 0.2 \
  --kv_scale_init 0.01 \
  --kv_scale_warmup_steps 200 \
  --kv_scale_max 2.0 \
  --kv_scale_reg 0.0 \
  --monitor_kv_every 100 \
  --monitor_kv_layer 0
