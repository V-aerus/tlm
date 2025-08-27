#!/bin/bash

# --- MoSLoRA迭代式监督微调脚本 ---
# 在固定的基础模型上，使用新的适配器继续训练到下一版本

BASE_MODEL_VERSION="4"
PREV_ADAPTER_VERSION="5_moslora"
NEW_ADAPTER_VERSION="6_moslora"
TRAIN_DATA_VERSION="6"

echo "Starting MoSLoRA Iterative Training: v${NEW_ADAPTER_VERSION} from v${PREV_ADAPTER_VERSION}"

# 使用独立的日志文件
LOG_FILE="run_moslora_iterative_v${TRAIN_DATA_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python train_lora_iterative.py \
    --base_model_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --lora_adapter_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${PREV_ADAPTER_VERSION} \
    --train_file=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_ADAPTER_VERSION} \
    --defuse_gpt2_attn=true \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --logging_steps=10 \
    --learning_rate=5e-06 \
    --lr_scheduler_type=constant \
    --warmup_steps=100 \
    --block_size=512 \
    --save_steps=500 \
    --save_total_limit=2 \
    --evaluation_strategy=no \
    --overwrite_output_dir=true \
    --do_train \
    2>&1 | tee ${LOG_FILE}
