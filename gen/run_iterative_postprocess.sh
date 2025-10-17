#!/bin/bash

# MT-MoSLoRA 迭代训练 - 数据整理阶段
# 整理测量数据并构建新的SFT数据集

echo "=== MT-MoSLoRA 迭代训练 - 数据整理阶段 ==="
echo "整理测量数据并构建新的SFT数据集..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号

# 基础路径
BASE_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen"
ITERATIVE_DATASET_PATH="$BASE_PATH/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"
PROCESS_PATH="$ITERATIVE_DATASET_PATH/process_iter${ITERATION_VERSION:1}"
SFT_DATASET_PATH="$ITERATIVE_DATASET_PATH/multi_sft_dataset_iter${ITERATION_VERSION:1}"

echo "📁 使用统一的数据集结构："
echo "  版本: $ITERATION_VERSION"
echo "  数据集文件夹: $ITERATIVE_DATASET_PATH"
echo "  处理中间文件夹: $PROCESS_PATH"
echo "  最终SFT数据集文件夹: $SFT_DATASET_PATH"
echo ""

# 创建必要的目录结构
mkdir -p "$PROCESS_PATH"
mkdir -p "$SFT_DATASET_PATH"

echo "1. 更新utils.json配置（仅追加 finetuning_files）..."
# 备份原始utils.json
cp "$BASE_PATH/utils.json" "$BASE_PATH/utils.json.backup"

# 只把新的迭代路径追加到 finetuning_files（累积、去重），不改动 measure_records
python3 -c "
import json
import os

with open('$BASE_PATH/utils.json', 'r') as f:
    utils_data = json.load(f)

new_paths = {
    'v100': '$ITERATIVE_DATASET_PATH/multi_v100_iter${ITERATION_VERSION:1}/measured_results.json',
    'xavier': '$ITERATIVE_DATASET_PATH/multi_xavier_iter${ITERATION_VERSION:1}/measured_results.json',
    '4090': '$ITERATIVE_DATASET_PATH/multi_4090_iter${ITERATION_VERSION:1}/measured_results.json',
    'xeon': '$ITERATIVE_DATASET_PATH/multi_xeon_iter${ITERATION_VERSION:1}/measured_results.json'
}

for hw, new_path in new_paths.items():
    if hw not in utils_data:
        utils_data[hw] = {
            'measure_records': [],
            'finetuning_files': [],
            'test_files': [],
            'testtuning_files': []
        }
    if 'finetuning_files' not in utils_data[hw] or not isinstance(utils_data[hw]['finetuning_files'], list):
        utils_data[hw]['finetuning_files'] = []
    if new_path not in utils_data[hw]['finetuning_files']:
        utils_data[hw]['finetuning_files'].append(new_path)
        print(f'✅ 添加到 {hw}.finetuning_files: {new_path}')
    else:
        print(f'⚠️  已存在于 {hw}.finetuning_files: {new_path}')

with open('$BASE_PATH/utils.json', 'w') as f:
    json.dump(utils_data, f, indent=2)

print('✅ utils.json 已更新：新增路径已追加到 finetuning_files（未改动 measure_records）')
"

echo "2. 整理V100测量数据..."
python postprocess.py --target=nvidia/nvidia-v100

echo "3. 整理Xavier测量数据..."
python postprocess.py --target=nvidia/jetson-agx-xavier

echo "4. 整理RTX4090测量数据..."
python postprocess.py --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32"

echo "5. 整理Xeon测量数据..."
python postprocess.py --target="llvm -mcpu=skylake-avx512 -model=xeon"

echo ""
echo "5. 构建新的多硬件SFT数据集..."

# 为每个硬件构建SFT数据集
echo "5.1 构建V100 SFT数据集..."
python make_dataset.py \
    --for_type=for_gen_best \
    --target=nvidia/nvidia-v100 \ 
    --dataset_path=$BASE_PATH/dataset/measure_records/v100 \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$SFT_DATASET_PATH/v100_gen_best_multi

echo "5.2 构建Xavier SFT数据集..."
python make_dataset.py \
    --for_type=for_gen_best \
    --target=nvidia/jetson-agx-xavier \
    --dataset_path=$BASE_PATH/dataset/measure_records/xavier \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$SFT_DATASET_PATH/xavier_gen_best_multi

echo "5.3 构建RTX4090 SFT数据集..."
python make_dataset.py \
    --for_type=for_gen_best \
    --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
    --dataset_path=$BASE_PATH/dataset/measure_records/4090 \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$SFT_DATASET_PATH/4090_gen_best_multi

echo "5.4 构建Xeon SFT数据集..."
python make_dataset.py \
    --for_type=for_gen_best \
    --target="llvm -mcpu=skylake-avx512 -model=xeon" \
    --dataset_path=$BASE_PATH/dataset/measure_records/xeon \
    --tokenizer_path=$BASE_PATH/gen_data/gen_tokenizer_multi_v1 \
    --save_path=$SFT_DATASET_PATH/xeon_gen_best_multi

echo ""
echo "6. 合并所有硬件的新SFT数据..."

# 创建合并数据目录
mkdir -p "$SFT_DATASET_PATH/all_gen_best_multi"

python3 -c "
import json
from datasets import DatasetDict, Dataset
from collections import Counter

print('📊 开始合并SFT数据集...')

# 加载所有硬件的新SFT数据
hardware_data = {}
hardware_paths = {
    'v100': '$SFT_DATASET_PATH/v100_gen_best_multi',
    'xavier': '$SFT_DATASET_PATH/xavier_gen_best_multi', 
    '4090': '$SFT_DATASET_PATH/4090_gen_best_multi',
    'xeon': '$SFT_DATASET_PATH/xeon_gen_best_multi'
}

for hw, path in hardware_paths.items():
    try:
        dataset = DatasetDict.load_from_disk(path)
        hardware_data[hw] = dataset['train']
        print(f'✅ 加载 {hw}: {len(dataset[\"train\"])} 样本')
    except Exception as e:
        print(f'❌ 加载 {hw} 失败: {e}')

# 合并所有数据
all_data = []
for hw, data in hardware_data.items():
    all_data.extend(data)

print(f'📈 合并后总样本数: {len(all_data)}')

# 统计硬件分布
hardware_counts = Counter()
for item in all_data:
    text = item['text'].lower()
    if 'sm_70' in text or 'v100' in text or '4090' in text or 'sm_86' in text:
        hardware_counts['high_perf_gpu'] += 1
    elif 'xavier' in text or 'jetson' in text or 'sm_72' in text:
        hardware_counts['edge_gpu'] += 1
    elif 'xeon' in text or 'skylake' in text or 'i7' in text or 'intel' in text:
        hardware_counts['cpu_group'] += 1

print('📊 新数据硬件分布:')
for hw, count in hardware_counts.most_common():
    percentage = count / len(all_data) * 100
    print(f'  {hw}: {count} 样本 ({percentage:.1f}%)')

# 保存合并数据
merged_dataset = DatasetDict({'train': Dataset.from_list(all_data)})
merged_dataset.save_to_disk('$SFT_DATASET_PATH/all_gen_best_multi')

print('✅ 新SFT数据集构建完成！')
"

echo ""
echo "🔍 检查生成结果..."

# 检查每个硬件是否成功生成SFT数据集
success_count=0
total_count=4

check_file() {
    local file_path="$1"
    local hardware_name="$2"
    if [ -d "$file_path" ]; then
        echo "✅ $hardware_name: SFT数据集生成成功"
        ((success_count++))
    else
        echo "❌ $hardware_name: SFT数据集生成失败"
    fi
}

check_file "$SFT_DATASET_PATH/v100_gen_best_multi" "V100"
check_file "$SFT_DATASET_PATH/xavier_gen_best_multi" "Xavier"
check_file "$SFT_DATASET_PATH/4090_gen_best_multi" "RTX4090"
check_file "$SFT_DATASET_PATH/xeon_gen_best_multi" "Xeon"

# 检查合并数据集
if [ -d "$SFT_DATASET_PATH/all_gen_best_multi" ]; then
    echo "✅ 合并数据集: 生成成功"
    ((success_count++))
else
    echo "❌ 合并数据集: 生成失败"
fi

echo ""
if [ $success_count -eq 5 ]; then
    echo "🎉 所有SFT数据集生成完成！ ($success_count/5)"
    echo "📁 生成的SFT数据集文件："
    echo "  - V100: $SFT_DATASET_PATH/v100_gen_best_multi"
    echo "  - Xavier: $SFT_DATASET_PATH/xavier_gen_best_multi"
    echo "  - RTX4090: $SFT_DATASET_PATH/4090_gen_best_multi"
    echo "  - Xeon: $SFT_DATASET_PATH/xeon_gen_best_multi"
    echo "  - 合并数据: $SFT_DATASET_PATH/all_gen_best_multi"
else
    echo "⚠️  部分SFT数据集生成失败！ ($success_count/5)"
    echo "请检查错误信息并重新运行脚本"
    exit 1
fi

echo ""
echo "7. 清理备份文件..."
# 删除备份文件，保留更新后的utils.json
rm "$BASE_PATH/utils.json.backup"
echo "✅ 已清理备份文件，utils.json保持更新状态"

echo ""
echo "💡 提示：数据整理完成，新迭代数据路径已添加到utils.json中"
echo "💡 可以继续运行下一步训练脚本"