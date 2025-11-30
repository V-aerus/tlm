#!/bin/bash
# HwToken 注入 + LoRA 训练脚本（保留 8 个硬件整数版本）

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

ROOT=/home/hehangshuai/workspace/tlm
DATA_ROOT=$ROOT/tlm_dataset/gen/gen_data

MODEL_PATH=$DATA_ROOT/clm_gen_multi_v1
TOKENIZER_PATH=$DATA_ROOT/gen_tokenizer_multi_v1
# 使用保留 8 位硬件整数的合并数据
DATASET_PATH=$DATA_ROOT/align_train_multi_merged_with_number/0_merge.json
OUTPUT_DIR=$DATA_ROOT/hw_aligner_lora_proto_with_number

python gen/train_hw_injection_lora.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --num_epochs 1 \
  --max_length 512 \
  --sample_fraction 0.3 \
  --lr_lora 1e-5 \
  --lr_aligner 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1500 \
  --lr_scheduler_type cosine \
  --grad_accum_steps 1 \
  --hardware_embedding_path gen/Embedding/hardware_embeddings_v2.json \
  --prototype_names "nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge" \
  --hw_token "[MASK]" \
  --lm_window_size 0 \
  --lambda_cls 1.0 \
  --lambda_ortho 0.1 \
  --lambda_ctr 0.0 \
  --lambda_kd 0.1 \
  --kd_temp 2.0 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --target_modules "attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj" \
  --device cuda \
  --log_interval 50 \
  --save_steps 5000 \
  --save_total_limit 5
