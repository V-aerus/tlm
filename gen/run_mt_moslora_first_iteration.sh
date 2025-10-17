#!/bin/bash

# MT-MoSLoRA 第一次迭代训练 - 从基础模型开始
# 使用基础模型生成的SFT数据集训练第一个MT-MoSLoRA适配器

echo "=== MT-MoSLoRA 第一次迭代训练 ==="
echo "从基础模型开始训练第一个MT-MoSLoRA适配器..."

# 版本配置 - 可修改
ITERATION_VERSION="v1"  # 修改这里的版本号
MODEL_VERSION="1_mt_moslora_base_iteration"

# 基础路径和版本配置
BASE_MODEL_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1"
ITERATIVE_DATASET_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/multi_iterative_${ITERATION_VERSION}_dataset"
TRAIN_DATA_PATH="$ITERATIVE_DATASET_PATH/all_gen_best_multi/0_merge.json"

echo "版本: $ITERATION_VERSION"
echo "模型版本: $MODEL_VERSION"
echo "基础模型: $BASE_MODEL_PATH"
echo "训练数据: $TRAIN_DATA_PATH"
echo "使用模式: 第一次迭代（无之前适配器）"

# 检查训练数据是否存在
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA_PATH"
    echo "请先运行 postprocess 脚本生成合并的SFT数据集"
    exit 1
fi

# 使用独立的日志文件
LOG_FILE="run_mt_moslora_first_iteration_${MODEL_VERSION}.log"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_mt_moslora.py \
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
    2>&1 | tee ${LOG_FILE}

echo ""
echo "🔍 检查训练结果..."

# 检查模型是否成功训练
if [ -d "/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_${MODEL_VERSION}" ]; then
    echo "✅ MT-MoSLoRA 第一次迭代训练完成！"
    echo "新适配器位置: /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_${MODEL_VERSION}"
    echo "日志文件: ${LOG_FILE}"
else
    echo "❌ 训练失败！请检查日志文件: ${LOG_FILE}"
    exit 1
fi
