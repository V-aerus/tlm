#!/bin/bash

# --- MT-MoSLoRA Multi-Hardware Training Script ---
# 基于混合多硬件数据的MT-MoSLoRA训练脚本，使用硬件分组策略

BASE_MODEL="clm_gen_multi_v1"
NEW_MODEL_VERSION="1_mt_moslora_grouped"

echo "Starting MT-MoSLoRA Multi-Hardware Training with Hardware Grouping"
echo "Base Model: ${BASE_MODEL}"
echo "New Model: ${NEW_MODEL_VERSION}"
echo "Architecture: HA (Hardware-Agnostic) + HS (Hardware-Specific) dual-track"
echo "Hardware Groups: high_perf_gpu(V100+4090), edge_gpu(Xavier), cpu(i7+Xeon)"
echo "Training Data: Mixed from V100, Xavier, RTX4090, and Xeon (2966 examples)"

# 使用独立的日志文件
LOG_FILE="run_mt_moslora_grouped_v${NEW_MODEL_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_mt_moslora.py \
    --do_train \
    --model_type=gpt2 \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_${NEW_MODEL_VERSION} \
    --train_file=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_iterative_v1_dataset/multi_sft_dataset_iter1/all_gen_best_multi/0_merge.json \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=20 \
    --learning_rate=5e-06 \
    --save_total_limit=2 \
    --disable_tqdm=false \
    --log_level=info \
    --model_name_or_path=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/${BASE_MODEL} \
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
    --hardware_types=high_perf_gpu,edge_gpu,cpu_group \
    --target_modules=q_proj,k_proj,v_proj,attn.c_proj,mlp.c_fc,mlp.c_proj \
    2>&1 | tee ${LOG_FILE}
