#!/bin/bash

# MT-MoSLoRA 迭代训练 - 生成张量程序阶段
# 使用MT-MoSLoRA v1为所有硬件分组生成新的张量程序

echo "=== MT-MoSLoRA 迭代训练 - 张量程序生成阶段 ==="
echo "使用MT-MoSLoRA v1为硬件分组生成张量程序..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号

# 基础路径
BASE_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen"
ITERATIVE_DATASET_PATH="$BASE_PATH/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"
MT_MODEL_PATH="$BASE_PATH/gen_data/clm_gen_multi_1_mt_moslora_grouped"
BASE_MODEL_PATH="$BASE_PATH/gen_data/clm_gen_multi_v1"

echo "📁 使用统一的数据集结构："
echo "  版本: $ITERATION_VERSION"
echo "  数据集文件夹: $ITERATIVE_DATASET_PATH"
echo "  基础模型: $BASE_MODEL_PATH"
echo "  v1适配器: $MT_MODEL_PATH"
echo ""

# 为每个硬件分组生成张量程序
echo "1. 为高性能GPU组 (V100) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target=nvidia/nvidia-v100 \
    --model_path=$BASE_MODEL_PATH \
    --multi_adapter_dir=$MT_MODEL_PATH \
    --target_hardware=high_perf_gpu \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "2. 为边缘GPU组 (Xavier) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target=nvidia/jetson-agx-xavier \
    --model_path=$BASE_MODEL_PATH \
    --multi_adapter_dir=$MT_MODEL_PATH \
    --target_hardware=edge_gpu \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "3. 为高性能GPU组 (RTX4090) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
    --model_path=$BASE_MODEL_PATH \
    --multi_adapter_dir=$MT_MODEL_PATH \
    --target_hardware=high_perf_gpu \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "4. 为CPU组 (Xeon) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target="llvm -mcpu=skylake-avx512 -model=xeon" \
    --model_path=$BASE_MODEL_PATH \
    --multi_adapter_dir=$MT_MODEL_PATH \
    --target_hardware=cpu_group \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "✅ 张量程序生成完成！"
echo "📁 生成的程序文件："
echo "  - V100: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json"
echo "  - Xavier: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json"
echo "  - RTX4090: $ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json"
echo "  - Xeon: $ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json"
