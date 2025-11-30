#!/bin/bash
# HwToken 注入 + LoRA 训练脚本（4 连续 MASK + 不保留硬件整数，使用 v3 硬件向量，阶段1配置）

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

ROOT=/home/hehangshuai/workspace/tlm
DATA_ROOT=$ROOT/tlm_dataset/gen/gen_data

MODEL_PATH=$DATA_ROOT/clm_gen_multi_v1
TOKENIZER_PATH=$DATA_ROOT/gen_tokenizer_multi_v1
DATASET_PATH=$DATA_ROOT/align_train_multi_merged_no_ints/0_merge.json
OUTPUT_DIR=$DATA_ROOT/hw_aligner_lora_stage1

python gen/train_hw_injection_lora.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 16 \
  --num_epochs 1 \
  --max_length 512 \
  --sample_fraction 0.4 \
  --lr_lora 5e-6 \
  --lr_aligner 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 1500 \
  --lr_scheduler_type cosine \
  --grad_accum_steps 8 \
  --hardware_embedding_path gen/Embedding/hardware_embeddings_v3.json \
  --prototype_names "nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge" \
  --hw_token "[MASK] [MASK] [MASK] [MASK]" \
  --lm_window_size 0 \
  --lambda_cls 0.01 \
  --lambda_ortho 0.1 \
  --lambda_ctr 0.0 \
  --lambda_kd 0.0 \
  --kd_temp 3.0 \
  --lambda_geom 1.0 \
  --hw_target_path gen/Embedding/hw_token_targets_v1.pt \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules "mlp.c_fc,mlp.c_proj" \
  --last_lora_layers 4 \
  --device cuda \
  --log_interval 50 \
  --save_steps 5000 \
  --save_total_limit 5
