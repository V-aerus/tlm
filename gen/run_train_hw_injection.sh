#!/bin/bash
# ProtoMix 硬件对齐器训练脚本
# 训练硬件嵌入注入对齐器（ProtoMix Aligner）

# 设置 GPU
export CUDA_VISIBLE_DEVICES=3

# 设置 Python 无缓冲输出（实时显示日志）
export PYTHONUNBUFFERED=1

# 训练命令
  CUDA_VISIBLE_DEVICES=3 python gen/train_hw_injection.py \
    --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
    --dataset_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/align_train_multi_merged/0_merge.json \
    --output_dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/hw_aligner_proto_cosine_lowlr \
    --hardware_embedding_path gen/Embedding/hardware_embeddings_v2.json \
    --prototype_names "nvidia/nvidia-v100,nvidia/nvidia-a40,nvidia/jetson-agx-xavier,aws/cpu/c5.18xlarge" \
    --hw_token "[MASK]" \
    --batch_size 8 \
    --num_epochs 1 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --grad_accum_steps 1 \
    --max_length 512 \
    --temperature 1.0 \
    --trainable_temperature False \
    --warmup_steps 1500 \
    --lr_scheduler_type cosine \
    --sample_fraction 0.02 \
    --hw_noise_std 0.0 \
    --device cuda

