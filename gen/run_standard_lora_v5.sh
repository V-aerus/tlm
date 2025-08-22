#!/bin/bash

# --- Standard LoRA Training Script for V5 ---
# 使用标准LoRA（不使用mixer）

BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5_standard_lora"
TRAIN_DATA_VERSION="5"

echo "Starting Standard LoRA Training: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"

# 使用独立的日志文件
LOG_FILE="run_standard_lora_v5.log"

time CUDA_VISIBLE_DEVICES=0 python run_train_clm_best_v100.py \
    --model_name_or_path /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --train_file /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --output_dir /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION} \
    --tokenizer_name /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --use_moslora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules c_attn,c_proj \
    2>&1 | tee ${LOG_FILE}
