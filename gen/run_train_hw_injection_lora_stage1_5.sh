#!/bin/bash
# Stage1.5: 冻结 aligner，lambda_geom=0，只训 LoRA（后4层 MLP），使用无整数的 4-MASK 数据

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

ROOT=/home/hehangshuai/workspace/tlm
DATA_ROOT=$ROOT/tlm_dataset/gen/gen_data

MODEL_PATH=$DATA_ROOT/clm_gen_multi_v1
TOKENIZER_PATH=$DATA_ROOT/gen_tokenizer_multi_v1
DATASET_PATH=$DATA_ROOT/align_train_multi_merged_no_ints/0_merge.json
OUTPUT_DIR=$DATA_ROOT/hw_aligner_lora_stage1_5
RESUME_DIR=$DATA_ROOT/hw_aligner_lora_stage1

python gen/train_hw_injection_lora.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --num_epochs 1 \
  --max_length 512 \
  --sample_fraction 0.2 \
  --lr_lora 1e-5 \
  --lr_aligner 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --lr_scheduler_type cosine \
  --grad_accum_steps 1 \
  --hardware_embedding_path gen/Embedding/hardware_embeddings_v2.json \
  --prototype_names "nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge" \
  --hw_token "[MASK] [MASK] [MASK] [MASK]" \
  --lm_window_size 0 \
  --lambda_cls 0.01 \
  --lambda_ortho 0.1 \
  --lambda_ctr 0.0 \
  --lambda_kd 0.0 \
  --lambda_geom 0.0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules "mlp.c_fc,mlp.c_proj" \
  --last_lora_layers 4 \
  --device cuda \
  --log_interval 50 \
  --save_steps 5000 \
  --save_total_limit 5 \
  --freeze_aligner True \
  --resume_from_checkpoint "$RESUME_DIR"
