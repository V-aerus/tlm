#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="${MODEL_PATH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket}"
TRAIN_JSON_PATHS="${TRAIN_JSON_PATHS:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Train_data/align_train_multi_merged_no_ints/0_merge_4090_bucket_kv_gpu_only.json}"
HW_EMB_PATH="${HW_EMB_PATH:-/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v5_draft.json}"
HW_EMB_PREPROCESS="${HW_EMB_PREPROCESS:-/home/hehangshuai/workspace/tlm/gen/Embedding/preprocess_v5_zscore_v1_nol2_aligner.json}"
HW_REQUIRE_PREPROCESS="${HW_REQUIRE_PREPROCESS:-1}"
OUT_DIR="${HW_KV_OUT_DIR:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train_v5_ab_nol2}"

LEARNING_RATE="${LEARNING_RATE:-3e-5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
KV_SCALE_INIT="${KV_SCALE_INIT:-0.1}"
KV_SCALE_WARMUP="${KV_SCALE_WARMUP:-2000}"
KV_SCALE_REG="${KV_SCALE_REG:-1e-3}"
MAX_STEPS="${MAX_STEPS:-70000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_KEEP_LAST="${SAVE_KEEP_LAST:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
L2_REG="${L2_REG:-1e-5}"
KD_WEIGHT="${KD_WEIGHT:-0.4}"
MONITOR_KV_LAYER="${MONITOR_KV_LAYER:-11}"
SWAP_KD_WEIGHT="${SWAP_KD_WEIGHT:-0.05}"
SWAP_KD_MARGIN="${SWAP_KD_MARGIN:-0.0}"
SWAP_REQUIRE_DIFF_HW="${SWAP_REQUIRE_DIFF_HW:-1}"
COUNTERFACTUAL_KD_WEIGHT="${COUNTERFACTUAL_KD_WEIGHT:-0.1}"
COUNTERFACTUAL_KD_WARMUP="${COUNTERFACTUAL_KD_WARMUP:-0}"
DELTA_CONSISTENCY_WEIGHT="${DELTA_CONSISTENCY_WEIGHT:-0.5}"
DELTA_CONSISTENCY_WARMUP="${DELTA_CONSISTENCY_WARMUP:-0}"
HW_PROBE_WEIGHT="${HW_PROBE_WEIGHT:-0.02}"
HW_PROBE_DETACH="${HW_PROBE_DETACH:-0}"
HW_KV_RESUME_CKPT="${HW_KV_RESUME_CKPT:-}"

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
  --warmup_steps "$WARMUP_STEPS" \
  --max_steps "$MAX_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_keep_last "$SAVE_KEEP_LAST" \
  --l2_reg "$L2_REG" \
  --max_grad_norm 1.0 \
  --kd_weight "$KD_WEIGHT" \
  --kd_temperature "$KD_TEMPERATURE" \
  --kd_mismatch_threshold 0.2 \
  --swap_kd_weight "$SWAP_KD_WEIGHT" \
  --swap_kd_margin "$SWAP_KD_MARGIN" \
  $( [[ "$SWAP_REQUIRE_DIFF_HW" == "1" ]] && echo "--swap_require_diff_hw" ) \
  --counterfactual_kd_weight "$COUNTERFACTUAL_KD_WEIGHT" \
  --counterfactual_kd_warmup_steps "$COUNTERFACTUAL_KD_WARMUP" \
  --delta_consistency_weight "$DELTA_CONSISTENCY_WEIGHT" \
  --delta_consistency_warmup_steps "$DELTA_CONSISTENCY_WARMUP" \
  --hw_probe_weight "$HW_PROBE_WEIGHT" \
  $( [[ "$HW_PROBE_DETACH" == "1" ]] && echo "--hw_probe_detach" ) \
  --kv_scale_init "$KV_SCALE_INIT" \
  --kv_scale_warmup_steps "$KV_SCALE_WARMUP" \
  --kv_scale_max 2.0 \
  --kv_scale_reg "$KV_SCALE_REG" \
  ${HW_KV_RESUME_CKPT:+--hw_kv_aligner_ckpt "$HW_KV_RESUME_CKPT"} \
  --monitor_kv_every 100 \
  --monitor_kv_layer "$MONITOR_KV_LAYER"
