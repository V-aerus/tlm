#!/bin/bash

# MT-MoSLoRA 迭代训练 - v2版本训练
# 使用新的SFT数据集对MT-MoSLoRA进行迭代训练

echo "=== MT-MoSLoRA 迭代训练 - v2版本训练 ==="
echo "使用新SFT数据集训练MT-MoSLoRA v2..."

# 版本配置 - 可修改
ITERATION_VERSION="v2"  # 修改这里的版本号
MODEL_VERSION="2_mt_moslora_grouped"

# 基础路径和版本配置
BASE_MODEL_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1"
PREVIOUS_ADAPTER_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_1_mt_moslora_grouped"
ITERATIVE_DATASET_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"
TRAIN_DATA_PATH="$ITERATIVE_DATASET_PATH/all_gen_best_multi/0_merge.json"

echo "版本: $ITERATION_VERSION"
echo "模型版本: $MODEL_VERSION"
echo "基础模型: $BASE_MODEL_PATH"
echo "之前适配器: $PREVIOUS_ADAPTER_PATH"
echo "训练数据: $TRAIN_DATA_PATH"

# 使用独立的日志文件
LOG_FILE="run_mt_moslora_iterative_${MODEL_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_mt_moslora_iterative.py \
    --do_train \
    --model_type=gpt2 \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
    --output_dir=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_${MODEL_VERSION} \
    --train_file=${TRAIN_DATA_PATH} \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=10 \
    --learning_rate=5e-06 \
    --model_name_or_path=${BASE_MODEL_PATH} \
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
    --previous_adapter_dir=${PREVIOUS_ADAPTER_PATH} \
    2>&1 | tee ${LOG_FILE}

echo "✅ MT-MoSLoRA v2训练完成！"
echo "新适配器位置: /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_${MODEL_VERSION}"
