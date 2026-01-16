  iter01（4090）基础流转 + LoRA 流转（全链路）

  source /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh

  # 1) 生成 sketch
  bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_sketch.sh 1 4090

  # 2) base 生成
  bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_bucketkv.sh 1 4090

  # 3) base 测量
  bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 1 4090 base

  # 4) 写入 utils.json + postprocess
  python $TLM_ROOT/gen/scripts/add_measure_records.py \
    --hardware 4090 \
    --iter 1 \
    --mode base

  python $TLM_ROOT/gen/postprocess.py --target "$TARGET_4090"

  # 5) base SFT（iter01）
  export ITER=iter01
  mkdir -p $RUN_ROOT/4090/$ITER/sft/base
  python $TLM_ROOT/gen/make_dataset.py \
    --for_type=for_gen_best \
    --target "$TARGET_4090" \
    --dataset_path $DATA_ROOT/dataset/measure_records/4090 \
    --tokenizer_path "$TOKENIZER" \
    --save_path $RUN_ROOT/4090/$ITER/sft/base

  touch $RUN_ROOT/4090/$ITER/sft/empty_lora.jsonl
  python $TLM_ROOT/gen/prepare_edge_dataset.py \
    --base-jsonl $RUN_ROOT/4090/$ITER/sft/base/0_merge.json \
    --lora-jsonl $RUN_ROOT/4090/$ITER/sft/empty_lora.jsonl \
    --output-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090.jsonl \
    --merge_mode base_left \
    --merge_key repr \
    --dedupe_mode keep_all \
    --hardware-id 4090 \
    --embedding-json $HW_EMB_V4 \
    --allow-missing-lora

  # 6) 用 v1_init 做 kv+LoRA 生成
  bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_kv_lora.sh 1 \
    $RUN_ROOT/4090/iter00/experts/v1_init 4090

    bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_kv_lora.sh 8 \
    $RUN_ROOT/v100/iter07/experts/v8_gain v100


  # 7) 测量 kv_lora
  bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 1 4090 kv_lora

  # 8) 写入 utils + postprocess（LoRA）
    --mode kv_lora

  python $TLM_ROOT/gen/postprocess.py --target "$TARGET_4090"

  # 9) 生成 lora SFT 并合并
  mkdir -p $RUN_ROOT/4090/$ITER/sft/lora
  python $TLM_ROOT/gen/make_dataset.py \
    --for_type=for_gen_best \
    --target "$TARGET_4090" \
    --dataset_path $DATA_ROOT/dataset/measure_records/4090 \
    --tokenizer_path "$TOKENIZER" \
    --save_path $RUN_ROOT/4090/$ITER/sft/lora

  touch $RUN_ROOT/4090/$ITER/sft/empty_base.jsonl
  python $TLM_ROOT/gen/prepare_edge_dataset.py \
    --base-jsonl $RUN_ROOT/4090/$ITER/sft/empty_base.jsonl \
    --lora-jsonl $RUN_ROOT/4090/$ITER/sft/lora/0_merge.json \
    --output-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2.jsonl \
    --merge_mode lora_left \
    --merge_key repr \
    --dedupe_mode keep_all \
    --hardware-id 4090 \
    --embedding-json $HW_EMB_V4

  python $TLM_ROOT/gen/prepare_edge_dataset.py \
    --base-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090.jsonl \
    --lora-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2.jsonl \
    --output-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2_lora_left.jsonl \
    --merge_key repr \
    --merge_mode lora_left \
    --dedupe_mode keep_all

  # 10) 训练 v2（gain）
  EDGE_LORA_LAMBDA_GAIN=0.1 bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_edge_expert.sh 1 4090 v2_gain
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  我把它分为 3 个阶段：基础数据 → LoRA 专家 v1 → LoRA 专家 v2。
  ———

  # ✅ Phase 0：准备 canonical 4090 / v100 的 task & programs

  # 4090 canonical target
  TARGET_4090="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024
  -registers_per_block=65536 -thread_warp_size=32"
  TARGET_V100="nvidia/nvidia-v100"

  # 生成 network_info + to_measure_programs (含断点重试)
  python gen/tools/dump_network_info_and_programs.py --target "$TARGET_4090" --dump_programs_size 1000 --max_retries 20 --sleep_sec 30
  python gen/tools/dump_network_info_and_programs.py --target "$TARGET_V100" --dump_programs_size 1000 --max_retries 20 --sleep_sec 30

  ———

  # ✅ Phase 1：Base 生成 → 测量 → Base SFT

  ## 1. 生成 sketch（你问的 for_gen_train_sketch）

  # 4090
  python gen/make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target "$TARGET_4090" \
    --dataset_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/dataset/to_measure_programs/4090 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/4090/iter00 \
    --keep_cnt=48 \
    --test_file_idx=0


  # v100
  python gen/make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target "$TARGET_V100" \
    --dataset_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/dataset/to_measure_programs/v100 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/v100/iter00 \
    --keep_cnt=48 \
    --test_file_idx=0


  ## 2. Base 生成（不加 LoRA / KV）

  python gen/gen_state.py \
    --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/4090/iter00/0_merge.json \
    --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/base_iter00.json \
    --target "$TARGET_4090" \
    --keep_cnt 64 \
    --use_bucket

  （v100 同理，target 换成 nvidia/nvidia-v100，save_path 改成 v100）

  ## 3. Base 测量

  python gen/measure_programs.py \
    --batch-size 64 \
    --target "$TARGET_4090" \
    --to-measure-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/base_iter00.json \
    --measured-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_measure_data/4090/base_iter00.json

  （v100 同理）

  ## 4. postprocess（生成 measure_records/<hw>）

  先把测量文件路径加入 /home/hehangshuai/workspace/tlm/tlm_dataset/gen/utils.json 的 measure_records，然后：

  python gen/postprocess.py --target "$TARGET_4090"
  python gen/postprocess.py --target "$TARGET_V100"

  ## 5. 生成 base SFT jsonl

  python gen/make_dataset.py \
    --for_type=for_gen_best \
    --target "$TARGET_4090" \
    --dataset_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/dataset/measure_records/4090 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi

  然后：

  python gen/prepare_edge_dataset.py \
    --sft-dataset-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi \
    --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi/edge_sft_4090.jsonl \
    --hardware-id 4090

  ———

  # ✅ Phase 2：训练 LoRA v1（无 gain）

  CUDA_VISIBLE_DEVICES=3 python gen/train_edge_expert.py \
    --base-model-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
    --tokenizer-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --dataset-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi/edge_sft_4090.jsonl \
    --output-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_v1_init \
    --batch-size 4 \
    --num-epochs 3 \
    --lambda-gain 0.0 \
    --warmup-steps 200 \
    --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj

  ———

  # ✅ Phase 3：生成 LoRA 测量 → v2 merge → gain 训练

  ## 1. LoRA 生成（v1 expert）

  EDGE_EXPERT_DEBUG_TOPK=1 CUDA_VISIBLE_DEVICES=0 python gen/gen_state_kv_lora.py \
    --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
    --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --edge_expert_dirs /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_v1_init \
    --edge_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v2.json \
    --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/4090/iter00/0_merge.json \
    --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/lora_iter00.json \
    --target "$TARGET_4090" \
    --keep_cnt 64 \
    --use_bucket \
    --use_hw_kv --hw_kv_mode real \
    --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
    --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json

  ## 2. 测量 + postprocess

  python gen/measure_programs.py \
    --batch-size 64 \
    --target "$TARGET_4090" \
    --to-measure-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/lora_iter00.json \
    --measured-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_measure_data/4090/lora_iter00.json

  python gen/postprocess.py --target "$TARGET_4090"

  ## 3. 生成 v2 jsonl + merge

  python gen/prepare_edge_dataset.py \
    --base-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi/edge_sft_4090.jsonl \
    --lora-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_sft_collections/4090/iter00/edge_sft_4090_v2.jsonl \
    --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_sft_collections/4090/iter00/edge_sft_4090_v2_merged_repr.jsonl \
    --merge_key repr

  ## 4. 正式 gain 训练

  CUDA_VISIBLE_DEVICES=3 python gen/train_edge_expert.py \
    --base-model-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
    --tokenizer-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
    --dataset-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_sft_collections/4090/iter00/edge_sft_4090_v2_merged_repr.jsonl \
    --output-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_v2_fullrun \
    --batch-size 4 \
    --num-epochs 3 \
    --lambda-gain 0.1 \
    --gain-margin 0.05 \
    --warmup-steps 200 \
    --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj


    