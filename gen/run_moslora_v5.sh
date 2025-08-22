#!/bin/bash

# --- MoSLoRA Training Script for V5 ---
# 基于您的run_my_sft.sh脚本，添加MoSLoRA特定参数

BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5_moslora"
TRAIN_DATA_VERSION="5"

echo "Starting MoSLoRA Training: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"

# 使用独立的日志文件
LOG_FILE="run_moslora_v5.log"

time CUDA_VISIBLE_DEVICES=1 python run_train_clm_best_v100.py \
    --model_name_or_path /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --train_file /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --output_dir /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION} \
    --tokenizer_name /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --use_moslora \
    --defuse_gpt2_attn \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj \
    2>&1 | tee ${LOG_FILE}
