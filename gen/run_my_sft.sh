#!/bin/bash

#    --model_name_or_path /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_xavier_v${BASE_MODEL_VERSION} \
# --- Iteration 2: Training v2 from v1 ---
BASE_MODEL_VERSION="4"
NEW_MODEL_VERSION="5"
TRAIN_DATA_VERSION="5"

echo "Starting SFT for Xavier: Training v${NEW_MODEL_VERSION} from v${BASE_MODEL_VERSION}"

time CUDA_VISIBLE_DEVICES=0 python run_train_clm_best_v100.py \
    --model_name_or_path /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${BASE_MODEL_VERSION} \
    --train_file /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/sft_dataset_v100_v${TRAIN_DATA_VERSION}/0_merge.json \
    --output_dir /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100_v${NEW_MODEL_VERSION} \
    --tokenizer_name /home/hangshuaihe/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_v100 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5