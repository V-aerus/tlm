#!/bin/bash

# --- MT-MoSLoRA Iterative Training Script ---
# 从已有的HA/HS适配器继续训练的迭代训练脚本

BASE_MODEL_VERSION="4"
PREVIOUS_ADAPTER_VERSION="5_mt_moslora"
NEW_MODEL_VERSION="6_mt_moslora"
TRAIN_DATA_VERSION="6"

echo "Starting MT-MoSLoRA Iterative Training: Training v${NEW_MODEL_VERSION} from v${PREVIOUS_ADAPTER_VERSION}"
echo "Architecture: HA (Hardware-Agnostic) + HS (Hardware-Specific) dual-track"

# 创建统一的日志文件夹
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# 使用统一的日志文件命名
LOG_FILE="${LOG_DIR}/mt_moslora_iterative_v${NEW_MODEL_VERSION}.log"

# 设置路径
BASE_MODEL_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION}"
PREVIOUS_ADAPTER_PATH="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${PREVIOUS_ADAPTER_VERSION}"
HA_ADAPTER_PATH="${PREVIOUS_ADAPTER_PATH}/ha_adapter.bin"
HS_ADAPTER_PATHS="${PREVIOUS_ADAPTER_PATH}/hs_v100_adapter.bin,${PREVIOUS_ADAPTER_PATH}/hs_xavier_adapter.bin,${PREVIOUS_ADAPTER_PATH}/hs_i7_adapter.bin"
ADAPTER_CONFIG_PATH="${PREVIOUS_ADAPTER_PATH}/adapter_config.json"
OUTPUT_DIR="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION}"
TRAIN_FILE="/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json"

# 检查文件是否存在
if [ ! -f "$HA_ADAPTER_PATH" ]; then
    echo "Error: HA adapter not found at $HA_ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$ADAPTER_CONFIG_PATH" ]; then
    echo "Error: Adapter config not found at $ADAPTER_CONFIG_PATH"
    exit 1
fi

echo "Base model: $BASE_MODEL_PATH"
echo "HA adapter: $HA_ADAPTER_PATH"
echo "HS adapters: $HS_ADAPTER_PATHS"
echo "Adapter config: $ADAPTER_CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Starting training at $(date)"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python train_mt_moslora_iterative.py \
    --do_train \
    --base_model_path="$BASE_MODEL_PATH" \
    --ha_adapter_path="$HA_ADAPTER_PATH" \
    --hs_adapter_paths="$HS_ADAPTER_PATHS" \
    --adapter_config_path="$ADAPTER_CONFIG_PATH" \
    --tokenizer_name=/home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --output_dir="$OUTPUT_DIR" \
    --train_file="$TRAIN_FILE" \
    --per_device_train_batch_size=5 \
    --num_train_epochs=3 \
    --overwrite_output_dir=true \
    --logging_steps=10 \
    --learning_rate=5e-06 \
    --lr_scheduler_type=constant \
    2>&1 | tee ${LOG_FILE}
