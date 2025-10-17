#!/bin/bash

# MT-MoSLoRA 迭代训练 - 生成草图阶段
# 为所有硬件分组生成新的训练草图

echo "=== MT-MoSLoRA 迭代训练 - 草图生成阶段 ==="
echo "为硬件分组生成新的训练草图..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号

# 基础路径
BASE_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen"
TO_MEASURE_PATH="$BASE_PATH/dataset/to_measure_programs"
ITERATIVE_DATASET_PATH="$BASE_PATH/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"

# 创建迭代数据集目录结构
mkdir -p "$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}"
mkdir -p "$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}"
mkdir -p "$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}"
mkdir -p "$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}"

echo "📁 使用统一的数据集结构："
echo "  版本: $ITERATION_VERSION"
echo "  总文件夹: $ITERATIVE_DATASET_PATH"
echo "  各硬件子文件夹: multi_v100_iter${ITERATION_VERSION:1}/, multi_xavier_iter${ITERATION_VERSION:1}/, multi_4090_iter${ITERATION_VERSION:1}/, multi_xeon_iter${ITERATION_VERSION:1}/"
echo ""

# 为每个硬件分组生成草图
echo "1. 为高性能GPU组 (V100) 生成草图..."
python make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target=nvidia/nvidia-v100 \
    --dataset_path=$TO_MEASURE_PATH/v100 \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1} \
    --keep_cnt=48 \
    --test_file_idx=0

echo "2. 为边缘GPU组 (Xavier) 生成草图..."
python make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target=nvidia/jetson-agx-xavier \
    --dataset_path=$TO_MEASURE_PATH/xavier \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1} \
    --keep_cnt=48 \
    --test_file_idx=0

echo "3. 为高性能GPU组 (RTX4090) 生成草图..."
python make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
    --dataset_path=$TO_MEASURE_PATH/4090 \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1} \
    --keep_cnt=48 \
    --test_file_idx=0

echo "4. 为CPU组 (Xeon) 生成草图..."
python make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target="llvm -mcpu=skylake-avx512 -model=xeon" \
    --dataset_path=$TO_MEASURE_PATH/xeon \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1} \
    --keep_cnt=48 \
    --test_file_idx=0

echo ""
echo "🔍 检查生成结果..."

# 检查每个硬件是否成功生成草图
success_count=0
total_count=4

check_file() {
    local file_path="$1"
    local hardware_name="$2"
    if [ -f "$file_path" ]; then
        echo "✅ $hardware_name: 草图生成成功"
        ((success_count++))
    else
        echo "❌ $hardware_name: 草图生成失败"
    fi
}

check_file "$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/0_merge.json" "V100"
check_file "$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/0_merge.json" "Xavier"
check_file "$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/0_merge.json" "RTX4090"
check_file "$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/0_merge.json" "Xeon"

echo ""
if [ $success_count -eq $total_count ]; then
    echo "🎉 所有草图生成完成！ ($success_count/$total_count)"
    echo "📁 生成的草图文件："
    echo "  - V100: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/0_merge.json"
    echo "  - Xavier: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/0_merge.json"
    echo "  - RTX4090: $ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/0_merge.json"
    echo "  - Xeon: $ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/0_merge.json"
else
    echo "⚠️  部分草图生成失败！ ($success_count/$total_count)"
    echo "请检查错误信息并重新运行脚本"
    exit 1
fi
