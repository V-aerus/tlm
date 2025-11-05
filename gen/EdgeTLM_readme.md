# EdgeTLM 快速手册（2025-02 更新）
-codex resume 019a48d6-27ec-7611-8c72-01163e73d45e
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
