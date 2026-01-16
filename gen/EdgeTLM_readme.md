# EdgeTLM 快速手册（2025-02 更新）

codex resume 019a48d6-27ec-7611-8c72-01163e73d45e
codex resume 019a5410-3c48-7cf3-8051-1ed95df16fdf -m gpt-5-codex
codex resume 019a95a5-8d92-7a21-935b-fce164b12d58

本文档汇总当前 EdgeTLM 管线的关键命令与目录规划，覆盖 V100 与 RTX4090 两个硬件场景。路径均以仓库根目录 `/home/hehangshuai/workspace/tlm/gen` 与数据根 `/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data` 为基准，请按照实际需求调整。

## 目录约定

- `edge_experts/<hw>/<tag>/`：LoRA 专家训练输出（`adapter_model.safetensors`、`router.json`、`metrics.json` 等）。
- `edge_sketches/<hw>/iterXX/`：草图（sketch）生成结果，每次迭代一个子目录。
- `edge_experts_gen/<hw>/...json`：`gen_state.py` 生成的候选张量程序。
- `edge_experts_measure_data/<hw>/...json`：真实硬件测量结果。
- SFT JSONL：`sft_dataset_mutil_v1/<hw>_gen_best_multi/edge_sft_<hw>[_v2].jsonl`。

## 1. 数据准备：`prepare_edge_dataset.py`

初次导出时尚未补齐 LoRA 延迟，可加 `--allow-missing-lora`。示例：

```bash
# V100
python prepare_edge_dataset.py \
  --sft-dataset-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/all_gen_best_multi \
  --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/v100_gen_best_multi/edge_sft_v100.jsonl \
  --hardware-id v100 \
  --allow-missing-lora

# RTX4090
python prepare_edge_dataset.py \
  --sft-dataset-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/all_gen_best_multi \
  --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi/edge_sft_4090.jsonl \
  --hardware-id 4090 \
  --allow-missing-lora
```

主要参数：
- `--hardware-id` 支持逗号分隔（如 `v100,4090`）。
- `--embedding-json` 默认 `Embedding/hardware_embeddings_v2.json`。
- 重新补入真实 LoRA 延迟时，只需再次执行并移除 `--allow-missing-lora`，输出到 `_v2.jsonl` 等新文件。

## 2. LoRA 专家训练：`train_edge_expert.py`

```bash
python train_edge_expert.py \
  --base-model-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --tokenizer-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/multi_hardware_tokenizer \
  --dataset-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/v100_gen_best_multi/edge_sft_v100.jsonl \
  --output-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/v100/stage02_20250212_fullrun \
  --num-epochs 3 \
  --batch-size 4 \
  --lambda-gain 0.0 \
  --warmup-steps 200 \
  --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj \
  | tee /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/v100/stage02_20250212_fullrun/train.log
```

4090 仿照该命令替换 `dataset-jsonl`、`output-dir`。首轮保持 `lambda_gain=0`；补齐真实 LoRA 延迟后可提升到 0.1~0.3。

产物路径示例：
- `edge_experts/v100/stage02_20250212_fullrun/`（V100 专家）
- `edge_experts/4090/stage02_20250212_fullrun/`（4090 专家）

## 3. 草图生成：`make_dataset.py`

```bash
# 4090 第 0 轮草图
python /home/hehangshuai/workspace/tlm/gen/make_dataset.py \
  --for_type=for_gen_train_sketch \
  --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
  --dataset_path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/dataset/to_measure_programs/4090 \
  --tokenizer_path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/gen_tokenizer_multi_v1 \
  --save_path=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/4090/iter00 \
  --keep_cnt=48 \
  --test_file_idx=0
```

V100 类似，目标与路径替换成对应硬件。下一轮迭代将 `test_file_idx` 改为 1、2 等，`save_path` 设为 `iter01` 等。

## 4. 推理生成：`gen_state.py`

并行运行时请在命令前设置不同的 `CUDA_VISIBLE_DEVICES`。脚本已改为使用唯一临时目录并新增 `--target_hardware`。

```bash
# V100（GPU 1）
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --edge_expert_dirs /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/v100/stage02_20250212_fullrun \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/v100/iter00/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/v100/stage02_20250212_gen_iter00.json \
  --target="nvidia/nvidia-v100" \
  --target_hardware v100 \
  --keep_cnt=32 \
  --allow_repeat True \
  --edge_topk 2 \
  | tee /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/v100/stage02_20250212_gen_iter00.log

# RTX4090（GPU 2）
CUDA_VISIBLE_DEVICES=2 python /home/hehangshuai/workspace/tlm/gen/gen_state.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
  --edge_expert_dirs /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_20250212_fullrun \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/4090/iter00/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/stage02_20250212_gen_iter00.json \
  --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
  --target_hardware 4090 \
  --keep_cnt=32 \
  --allow_repeat True \
  --edge_topk 2 \
  | tee /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/stage02_20250212_gen_iter00.log
```

## 4.5 HW-KV 调试推理：`gen_state_debug_kv.py`

背景：当跨硬件/未知硬件时，纯文本提示容易出现 OOV 语义漂移。HW-KV 注入通过 “hardware embedding -> KV past” 的方式，给模型提供硬件侧通道，帮助在相同 bucket/token 前提下仍能区分硬件差异。

三种模式（`--hw_kv_mode`）：
- `noop`：不注入 past（行为与基线一致），用于对照。
- `zero`：注入全 0 past（前缀位置存在但无信息），用于测试 prefix 影响。
- `real`：使用 HwKVAligner 输出的 KV past（真实注入）。

关键参数（相对 `gen_state.py` 新增）：
- `--use_hw_kv`：开启 KV 注入逻辑。
- `--hw_kv_mode {noop,zero,real}`：选择注入模式。
- `--hw_kv_num_slots`：KV 前缀 slot 数（默认 4）。
- `--pos_compensate`：启用位置补偿（prompt 的 position_ids 从 0 起，不包含 prefix 长度）。
- `--debug_kv_stats`：输出后 4 层 KV 强度统计（用于确认注入是否生效）。
- `--debug_forward_trace N`：打印前 N 次 forward 的 input_ids/position_ids/past_len。

常用命令模板：

```bash
# 4090（真实注入）
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state_debug_kv.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/bucket_with_kv/gen_bucket_hwkv_4090.json \
  --target 4090 \
  --keep_cnt 16 \
  --use_bucket \
  --use_hw_kv \
  --hw_kv_mode real \
  --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
  --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json \
  --hw_kv_num_slots 4 \
  --pos_compensate \
  --debug_kv_stats

# V100
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state_debug_kv.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/bucket_with_kv/gen_bucket_hwkv_v100.json \
  --target nvidia/nvidia-v100 \
  --keep_cnt 16 \
  --use_bucket \
  --use_hw_kv \
  --hw_kv_mode real \
  --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
  --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json \
  --hw_kv_num_slots 4 \
  --pos_compensate

# 3090（tag.cc 内置）
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state_debug_kv.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/bucket_with_kv/gen_bucket_hwkv_3090.json \
  --target nvidia/geforce-rtx-3090 \
  --keep_cnt 16 \
  --use_bucket \
  --use_hw_kv \
  --hw_kv_mode real \
  --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
  --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json \
  --hw_kv_num_slots 4 \
  --pos_compensate
```

输出说明：
- `--save_path` 写入合并后的 JSON（与 `gen_state.py` 同格式）。
- 同目录下会生成 `gen_state_debug_kv_*.log`，包含 valid 统计、KV 强度、debug 断言信息等。

## 4.6 KV + LoRA 路由推理：`gen_state_kv_lora.py`

该脚本在 `gen_state_debug_kv.py` 的基础上加入“多 LoRA 专家路由”，用于端到端验证：
bucket + KV + LoRA 路由。  
注意：**现有 LoRA 专家 `router.json` 的 `hardware_dim=29`（v2 embedding）**，所以若使用
`edge_experts/*/stage02_20250212_fullrun`，请显式传 `--edge_embedding_path Embedding/hardware_embeddings_v2.json`，
避免维度不匹配。

关键参数（相对 `gen_state_debug_kv.py` 新增）：
- `--edge_expert_dirs`：LoRA 专家目录，可多传（以空格分隔）。
- `--edge_embedding_path`：路由用硬件 embedding（默认 **v4**，若用旧专家请改成 v2）。
- `--edge_topk`：路由取前 K 个专家（默认 1）。
- `--sketch_hw_candidates`：手动限定可用草图硬件（逗号分隔）；若不填则自动扫描可用硬件并挑最相近者。
- `--debug_hw_similarity`：打印目标硬件与专家路由向量的相似度（方便调试）。

示例（4090）：
```bash
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state_kv_lora.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --edge_expert_dirs /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_20250212_fullrun \
  --edge_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v2.json \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/{hw}/iter00/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/gen_kv_lora.json \
  --target 4090 \
  --use_bucket \
  --use_hw_kv --hw_kv_mode real \
  --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
  --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json \
  --pos_compensate
```

示例（3090，自动选择最近硬件草图）：
```bash
CUDA_VISIBLE_DEVICES=1 python /home/hehangshuai/workspace/tlm/gen/gen_state_kv_lora.py \
  --model_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000 \
  --tokenizer_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket \
  --edge_expert_dirs /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/4090/stage02_20250212_fullrun \
  --edge_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v2.json \
  --sketch_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_sketches/{hw}/iter00/0_merge.json \
  --save_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/3090/gen_kv_lora.json \
  --target nvidia/geforce-rtx-3090 \
  --use_bucket \
  --use_hw_kv --hw_kv_mode real \
  --hw_kv_aligner_path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train/hw_kv_aligner.pt \
  --hardware_embedding_path /home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4.json \
  --pos_compensate \
  --debug_hw_similarity
```
- 控制台会打印 “成功处理 workload 数/生成记录数”，可作为 valid 率粗略指标。

## 5. 真机测量：`measure_programs.py`

```bash
CUDA_VISIBLE_DEVICES=2 python measure_programs.py \
  --batch-size 64 \
  --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32" \
  --to-measure-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_gen/4090/stage02_20250212_gen_iter00.json \
  --measured-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts_measure_data/4090/stage02_20250212_measure_iter00.json
```

V100 测量命令同理。测量后务必将路径追加到 `tlm_dataset/gen/utils.json` 对应硬件的 `measure_records` 列表。

## 6. 后处理：`postprocess.py`

更新 `utils.json` 后执行：

```bash
python postprocess.py \
  --target="cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -model=4090 -thread_warp_size=32"
```

产物会整理到 `dataset/measure_records/4090/`。再运行：

```bash
python prepare_edge_dataset.py \
  --sft-dataset-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/all_gen_best_multi \
  --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/4090_gen_best_multi/edge_sft_4090_v2.jsonl \
  --hardware-id 4090
```

此时 JSONL 已含真实 `lat_lora_star`，可用于 `lambda_gain>0` 的训练迭代。

## 7. 迭代与记录建议

1. 每轮训练与推理后，把关键命令、日志、`metrics.json` 指标更新到 `EdgeTLM_update_log.md` 顶端。
2. 并行执行 `gen_state.py` 时，不同 GPU 自带独立临时目录，运行前无需额外清理；若中途终止，可手动删除 `.gen_state_*`。
3. 草图/测量/专家目录建议以阶段号 + 日期命名，例如 `stage02_20250212_*`，保持与日志同步。
4. 在开启增益约束前确保 `lat_lora_star` 数据完整，避免 `gain_loss` 恒为零。

如需扩展到其他硬件，请参考 README 中的管线，先准备 `to_measure_programs/<hw>`、`network_info/<hw>` 等基础数据，再按本手册流程推进。***

---

## 附：多硬件记录语料的分批处理与 HwToken 导出提示

- 记录版数据位置与形态
  - `dataset/to_measure_programs/multi_hardware/`：按硬件前缀命名的 TVM 记录版 JSON（含 `"i"/"r"`），可用于 `make_dataset --for_gen` 重建 ComputeDAG + 张量句子。
  - `dataset/to_measure_programs/multi_hardware_simple/all_programs.txt`：已扁平化的纯文本总集，适合 tokenizer 训练，无法再恢复 DAG。

- 分批处理建议（避免跨后端冲突）
  - `make_dataset --for_gen` 一次只能注册一个后端的 `all_tasks.pkl`，不能同时处理 CUDA + LLVM 记录。
  - 建议用 `--file_filter` 按文件名前缀分两次跑：
    - CUDA 组：`--target="cuda ..." --file_filter "^(v100_|4090_|xavier_)"`
    - LLVM/CPU 组：`--target="llvm ..." --file_filter "^(xeon_|i7_|llvm_)"`
  - 分别生成两个 `0_merge.json`，再用 `cat` 合并为统一语料。

- HwToken（Student 文本）导出
  - 加 `--emit_hw_student True --hw_token_placeholder "[MASK]" --hardware_embedding_path=gen/Embedding/hardware_embeddings_v2.json`，可在 `0_merge.json` 里额外输出：
    - `text_student`（target→占位符）、`hw_emb`/`hw_id`/`hw_name`，用于对齐器/注入训练及推理。
  - 适用于记录版 JSON 输入；若源数据已是文本语料（如 `pretrain_data_multi_v1/0_merge.json`），无需再跑 `for_gen`。

- 记忆点
  - CUDA 设备间任务集一致，可任选 CUDA target 处理 CUDA 记录；CPU 记录需用 LLVM target 注册对应 `all_tasks.pkl`。
  - `target=multi` 的路径推断仅适配 `.../measure_records/<hw>`，不适用于 `to_measure_programs/multi_hardware`，请按上述分批方式处理。
