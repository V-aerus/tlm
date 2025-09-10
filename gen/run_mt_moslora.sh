#!/bin/bash

# --- MT-MoSLoRA Training Script ---
# 基于MTLoRA模式的MoSLoRA训练脚本，实现HA+HS双轨制架构

BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5_mt_moslora"
TRAIN_DATA_VERSION="5"

echo "Starting MT-MoSLoRA Training: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"
echo "Architecture: HA (Hardware-Agnostic) + HS (Hardware-Specific) dual-track"

# 使用独立的日志文件
LOG_FILE="run_mt_moslora_v${NEW_MODEL_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_mt_moslora.py \
    --do_train \
    --model_type=gpt2 \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION}\
    --train_file=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=10 \
    --learning_rate=5e-06 \
    --model_name_or_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --lr_scheduler_type=constant \
    --use_mt_moslora=true \
    --use_mixer=true \
    --defuse_gpt2_attn=true \
    --ha_lora_r=16 \
    --ha_lora_alpha=16 \
    --ha_lora_dropout=0.05 \
    --hs_lora_r=16 \
    --hs_lora_alpha=32 \
    --hs_lora_dropout=0.05 \
    --hardware_types=v100,xavier,i7 \
    --target_modules=q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj \
    2>&1 | tee ${LOG_FILE}
