#!/bin/bash

# MT-MoSLoRA 迭代训练 - 使用基础模型生成张量程序
# 不使用任何adapter，直接用基础模型生成张量程序

echo "=== MT-MoSLoRA 迭代训练 - 基础模型张量程序生成阶段 ==="
echo "使用基础模型为硬件分组生成张量程序（不使用adapter）..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号

# 基础路径
BASE_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen"
ITERATIVE_DATASET_PATH="$BASE_PATH/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"
BASE_MODEL_PATH="$BASE_PATH/gen_data/clm_gen_multi_v1"

echo "📁 使用统一的数据集结构："
echo "  版本: $ITERATION_VERSION"
echo "  数据集文件夹: $ITERATIVE_DATASET_PATH"
echo "  基础模型: $BASE_MODEL_PATH"
echo "  使用模式: 基础模型（无adapter）"
echo ""

# 为每个硬件分组生成张量程序
echo "1. 为高性能GPU组 (V100) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target=nvidia/nvidia-v100 \
    --model_path=$BASE_MODEL_PATH \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "2. 为边缘GPU组 (Xavier) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target=nvidia/jetson-agx-xavier \
    --model_path=$BASE_MODEL_PATH \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "3. 为高性能GPU组 (RTX4090) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
    --model_path=$BASE_MODEL_PATH \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo "4. 为CPU组 (Xeon) 生成张量程序..."
CUDA_VISIBLE_DEVICES=1 python gen_state.py \
    --target="llvm -mcpu=skylake-avx512 -model=xeon" \
    --model_path=$BASE_MODEL_PATH \
    --sketch_path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/0_merge.json \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json \
    --allow_repeat=True \
    --keep_cnt=16

echo ""
echo "🔍 检查生成结果..."

# 检查每个硬件是否成功生成张量程序
success_count=0
total_count=4

check_file() {
    local file_path="$1"
    local hardware_name="$2"
    if [ -f "$file_path" ]; then
        echo "✅ $hardware_name: 张量程序生成成功"
        ((success_count++))
    else
        echo "❌ $hardware_name: 张量程序生成失败"
    fi
}

check_file "$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json" "V100"
check_file "$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json" "Xavier"
check_file "$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json" "RTX4090"
check_file "$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json" "Xeon"

echo ""
if [ $success_count -eq $total_count ]; then
    echo " 所有张量程序生成完成！ ($success_count/$total_count)"
    echo " 生成的程序文件："
    echo "  - V100: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json"
    echo "  - Xavier: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json"
    echo "  - RTX4090: $ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json"
    echo "  - Xeon: $ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json"
else
    echo "⚠️  部分张量程序生成失败！ ($success_count/$total_count)"
    echo "请检查错误信息并重新运行脚本"
    exit 1
fi
