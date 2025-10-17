#!/bin/bash

# MT-MoSLoRA 迭代训练 - 性能测量阶段
# 测量所有硬件分组生成的张量程序性能

echo "=== MT-MoSLoRA 迭代训练 - 性能测量阶段 ==="
echo "测量硬件分组生成的张量程序性能..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号

# 基础路径
BASE_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen"
ITERATIVE_DATASET_PATH="$BASE_PATH/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"

echo "📁 使用统一的数据集结构："
echo "  版本: $ITERATION_VERSION"
echo "  数据集文件夹: $ITERATIVE_DATASET_PATH"
echo ""

echo "1. 高性能GPU组 (V100) - 需要传输到其他服务器测量"
echo "   张量程序文件: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json"
echo "   目标硬件: nvidia/nvidia-v100"
echo "   ⚠️  请将此文件传输到V100服务器进行测量，并将结果保存为: measured_results.json"
echo ""

echo "2. 边缘GPU组 (Xavier) - 需要传输到其他服务器测量"
echo "   张量程序文件: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json"
echo "   目标硬件: nvidia/jetson-agx-xavier"
echo "   ⚠️  请将此文件传输到Xavier服务器进行测量，并将结果保存为: measured_results.json"
echo ""

echo "3. 测量高性能GPU组 (RTX4090) 性能..."
CUDA_VISIBLE_DEVICES=1 python measure_programs.py \
    --batch-size=64 \
    --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
    --to-measure-path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/gen_train.json \
    --measured-path=$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/measured_results.json

echo "4. 测量CPU组 (Xeon) 性能..."
CUDA_VISIBLE_DEVICES=1 python measure_programs.py \
    --batch-size=64 \
    --target="llvm -mcpu=skylake-avx512 -model=xeon" \
    --to-measure-path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/gen_train.json \
    --measured-path=$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/measured_results.json

echo ""
echo "🔍 检查本地测量结果..."

# 检查本地能测量的硬件是否成功完成测量
local_success_count=0
local_total_count=2

check_file() {
    local file_path="$1"
    local hardware_name="$2"
    if [ -f "$file_path" ]; then
        echo "✅ $hardware_name: 性能测量成功"
        ((local_success_count++))
    else
        echo "❌ $hardware_name: 性能测量失败"
    fi
}

echo "本地测量结果："
check_file "$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/measured_results.json" "RTX4090"
check_file "$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/measured_results.json" "Xeon"

echo ""
echo "远程测量结果检查："
remote_success_count=0
remote_total_count=2

check_file "$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/measured_results.json" "V100"
check_file "$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/measured_results.json" "Xavier"

echo ""
total_success_count=$((local_success_count + remote_success_count))
total_count=$((local_total_count + remote_total_count))

echo "📊 测量完成情况总结："
echo "  本地测量: $local_success_count/$local_total_count (RTX4090, Xeon)"
echo "  远程测量: $remote_success_count/$remote_total_count (V100, Xavier)"
echo "  总计: $total_success_count/$total_count"

if [ $local_success_count -eq $local_total_count ]; then
    echo "✅ 本地测量完成！"
    echo "📁 本地测量结果文件："
    echo "  - RTX4090: $ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/measured_results.json"
    echo "  - Xeon: $ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/measured_results.json"
else
    echo "❌ 本地测量失败！请检查错误信息并重新运行脚本"
    exit 1
fi

if [ $remote_success_count -eq $remote_total_count ]; then
    echo "✅ 远程测量也已完成！"
    echo "📁 远程测量结果文件："
    echo "  - V100: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/measured_results.json"
    echo "  - Xavier: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/measured_results.json"
else
    echo "⚠️  远程测量尚未完成 ($remote_success_count/$remote_total_count)"
    echo "请将以下文件传输到对应服务器进行测量："
    echo "  - V100文件: $ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/gen_train.json"
    echo "  - Xavier文件: $ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/gen_train.json"
    echo "测量完成后，请将结果文件放置到对应目录中"
fi

echo ""
echo "💡 提示：如果远程测量已完成，可以继续运行下一步脚本"
