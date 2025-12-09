#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3

python train_clm_bucket/train_clm.py \
  --model_name_or_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_init \
  --tokenizer_name /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --train_file /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Train_data/align_train_multi_merged_no_ints/0_merge_bucket.json \
  --output_dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --warmup_steps 200 \
  --sample_ratio 0.1 \
  --resume_from_checkpoint /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-8500 \
  --overwrite_output_dir \
  --do_train \
  --seed 42
