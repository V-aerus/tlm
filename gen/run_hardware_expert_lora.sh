#!/bin/bash

# --- HardwareExpertLoRA训练脚本 ---
# 基于MTLoRA设计理念的硬件专家LoRA训练

BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5_hardware_expert_lora"
TRAIN_DATA_VERSION="5"

echo "Starting HardwareExpertLoRA Training: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"

# 使用独立的日志文件
LOG_FILE="run_hardware_expert_lora_v${NEW_MODEL_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_hardware_expert_lora.py \
    --do_train \
    --model_type=gpt2 \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION} \
    --train_file=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=10 \
    --learning_rate=5e-06 \
    --model_name_or_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --lr_scheduler_type=constant \
    --warmup_steps=100 \
    --block_size=512 \
    --save_steps=500 \
    --save_total_limit=2 \
    --evaluation_strategy=no \
    \
    --use_hardware_expert_lora \
    --defuse_gpt2_attn \
    \
    --ha_r=8 \
    --ha_alpha=16.0 \
    --ha_dropout=0.1 \
    \
    --hs_r=8 \
    --hs_alpha=16.0 \
    --hs_dropout=0.1 \
    \
    --hardware_types="gpu,cpu_hw,edge" \
    --target_modules="q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj" \
    --bias="none" \
    \
    --enable_hardware_routing \
    --default_hardware_type="gpu" \
    --trainable_ha \
    --trainable_hs \
    2>&1 | tee ${LOG_FILE}
