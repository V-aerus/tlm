#!/bin/bash

# --- MoSLoRA Training Script for V5 (Direct) ---
# 直接运行，不使用tmux，避免会话冲突

BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5_moslora"
TRAIN_DATA_VERSION="5"

echo "Starting MoSLoRA Training: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"

# 使用独立的日志文件
LOG_FILE="run_moslora_v5.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_clm.py \
    --do_train \
    --model_type=gpt2 \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION} \
    --train_file=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=10 \
    --learning_rate=5e-05 \
    --model_name_or_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --lr_scheduler_type=constant \
    --use_lora=true \
    --use_mixer=true \
    --defuse_gpt2_attn=true \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --target_modules=q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj \
    2>&1 | tee ${LOG_FILE}
