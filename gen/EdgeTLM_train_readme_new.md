# EdgeTLM 端到端重跑（新目录规划版）

本文件基于 `gen/codex_todo.md` 的新规划，目标是把 **bucket+KV → LoRA 路由 → 测量 → SFT** 的全链路重新跑通，同时避免旧目录污染、便于复现与回溯。
# 一键脚本速查（先看这里）

> 前 4 轮（`iter00~iter03`）：建议使用**完整版**脚本，会包含 base 测量与更新。
>
> ```bash
> # 例如：iter01 用 v1_init 生成并训练 v2_gain
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_iter_full.sh 1 4090 v1_init 2
> ```
>
> 后续轮次（`iter04` 及以后）：若 base 已足够稳定，可用 **LoRA-only** 脚本（跳过 base 测量）。
>
> ```bash
> # 例如：iter04 基于 v4_gain 训练 v5_gain
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_iter_lora_only.sh 4 4090 v4_gain 5
> ```
>
> 注意：
> - Base SFT 会使用 **base-only** 测量记录；LoRA SFT 使用 **kv_lora-only** 记录。
> - `postprocess.py` 已支持 `--record-mode base|kv_lora|all`，脚本会自动选择。
> - 现在还会额外生成一份 **teacher(all)**：文本来自 all 记录，但 `lat_base_star/lat_lora_star` 仍来自 base/kv_lora。

## 泛化硬件评测（2080Ti / 3090，三网标准形状）

已新增 `for_gen_eval_sketch_ansor`，会选中 **bert_base / resnet_50 / mobilenet_v2** 的标准形状（与 Ansor baseline 对齐），用于泛化硬件评测。
source /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh
 export RUN_ROOT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart
一键生成（只生成，不测量）：

```bash
# 先设置两套专家 + 官方模型
export EDGE_EXPERT_4090=$RUN_ROOT/4090/iterXX/experts/vX_gain
export EDGE_EXPERT_V100=$RUN_ROOT/v100/iterYY/experts/vY_gain
export EDGE_OFFICIAL_CKPT=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_best_v100

# 生成 2080Ti 的四种 baseline（mix / official / 4090-only / v100-only）
bash /home/hehangshuai/workspace/tlm/gen/scripts/run_eval_suite.sh 2080

# 生成 3090 的四种 baseline
bash /home/hehangshuai/workspace/tlm/gen/scripts/run_eval_suite.sh 3090
```

产物默认输出：
```
$RUN_ROOT/<hw>/iter00/eval_ansor_sketch/0_merge.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_mix.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_4090.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/kv_lora_v100.json
$RUN_ROOT/<hw>/iter00/eval_ansor_gen/official.json
```

说明：
- 我们的方法默认 **bucket + KV 注入 + LoRA**（gen_state_kv_lora.py）。
- 官方 baseline 走 **gen_state.py**，不启用 bucket/KV，也不使用 LoRA（使用官方 tokenizer）。
- 生成后请手动搬运到测量服务器测量。

## 迭代重训必用环境变量（精简版）

**常用开关：**

- `EDGE_ITER_MAX=K`：只使用 `iter<=K` 的测量记录（严格迭代）
- `EDGE_ITER_LIST=0,1,2`：只使用指定迭代（优先级高于 `EDGE_ITER_MAX`）
- `EDGE_CLEAN_OUTPUT=1`：清空 postprocess 输出目录（避免历史残留）
- `EDGE_FORCE_PREPARE=1`：每次重建 `prepare_edge_dataset` 产物（不复用旧 `edge_sft_*.jsonl`）
- `EDGE_SKIP_GEN=1` / `EDGE_SKIP_MEASURE=1`：跳过生成/测量（使用已有记录重训）

**LoRA 训练相关：**

- `EDGE_LORA_LAMBDA_GAIN=0.1`（建议）
- `EDGE_LORA_GAIN_MARGIN=0.1`（建议）
- `EDGE_LORA_WARMUP=0`（需要时可设 0）
- `EDGE_LORA_LAMBDA_ENTROPY=1e-4`（按稳定性调整）

## 测量结果汇总导出（CSV）

用于从 measured json 导出 **每个 workload 的最短 latency**，并按网络汇总（总延迟、加权总延迟）。

**单个 measured 文件：**

```bash
python /home/hehangshuai/workspace/tlm/gen/scripts/export_measured_summary.py \
  --target nvidia/nvidia-v100 \
  --measured-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/v100/iter06/measure/kv_lora_measured.json \
  --output-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/v100/iter06/measure/kv_lora_summary.csv
```

**目录批量汇总：**

```bash
python /home/hehangshuai/workspace/tlm/gen/scripts/export_measured_summary.py \
  --target 4090 \
  --record-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/dataset/measure_records/4090
```

**多份 measured 合并到一个 CSV（推荐）：**

```bash
python /home/hehangshuai/workspace/tlm/gen/scripts/export_measured_summary.py \
  --target 3090 \
  --measured base_official=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/1.9/base_official_v100_measured.json \
  --measured kv_lora_4090=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/1.9/kv_lora_4090_v8_measured.json \
  --measured kv_lora_v100=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/1.9/kv_lora_v100_v7_measured.json \
  --measured kv_lora_mix=/root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/1.9/kv_lora_mix_v8v7_measured.json \
  --output-csv /root/tlm/tlm_dataset/gen/gen_data/to_measure_programs/1.9/summary_1.9.csv
```

**只看 BERT‑base (1×128)，避免 workload_key 冲突：**

```bash
python /home/hehangshuai/workspace/tlm/gen/scripts/export_measured_summary.py \
  --target 4090 \
  --measured-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/kv_lora_v8_gain_measured.json \
  --only-network bert_base \
  --only-shape 1,128 \
  --output-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/kv_lora_v8_gain_measured.csv
```

**多版本对比（同一份 BERT‑base 测量）**：

```bash
python /home/hehangshuai/workspace/tlm/gen/scripts/compare_measured_versions.py \
  --target 4090 \
  --only-network bert_base \
  --only-shape 1,128 \
  --measured base_official=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/base_official_v100_bert_measured.json \
  --measured v8_gain=/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/kv_lora_v8_gain_measured.json \
  --baseline base_official \
  --output-network-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/compare_network.csv \
  --output-workload-csv /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/gen/compare_workload.csv
```

输出列包含：`network_name`、`network_shape`、`workload_key`、`best_latency_ms`、`network_total_best_ms`、`network_total_weighted_ms`。

## 迭代重训示例（v100）

**iter01 → v2_gain（严格迭代 + 只重训）**

```bash
source $RUN_ROOT/_shared/paths.sh
export EDGE_SKIP_GEN=1
export EDGE_SKIP_MEASURE=1
export EDGE_CLEAN_OUTPUT=1
export EDGE_FORCE_PREPARE=1
export EDGE_ITER_MAX=1
bash /home/hehangshuai/workspace/tlm/gen/scripts/run_iter_full.sh 1 v100 v1_init 2
```

**iter04 → v5_gain（后续轮次建议 LoRA-only）**

```bash
export EDGE_ITER_MAX=4
bash /home/hehangshuai/workspace/tlm/gen/scripts/run_iter_lora_only.sh 4 v100 v4_gain 5
```
## 0. 目录规划（建议）

```
tlm_dataset/gen/gen_data/
  edge_runs/
    <RUN_TAG>/
      _shared/
        targets/
          4090.target.txt
          v100.target.txt
        paths.sh
        embeddings/      # 可选：软链接
        models/          # 可选：软链接
        hw_kv_aligner/   # 可选：软链接
      4090/
        iter00/
          sketch/0_merge.json
          gen/base_bucketkv.json
          gen/kv_lora.json
          measure/base_bucketkv.json
          measure/kv_lora.json
          post/
          sft/base/
          sft/edge_sft_4090.jsonl
          logs/
        experts/
          v1_init/
          v2_gain/
      v100/
        ...（同上）
```

**规则：**
1) 所有产物都落在 `edge_runs/<RUN_TAG>/<hw>/iterXX`，不覆盖旧目录。  
2) 每轮只改 `iterXX` 与 `test_file_idx`。  
3) 每个 iter 记录清楚 target / base_ckpt / tokenizer / embedding / hw_kv_aligner / expert_dirs。  

---

## 1. 一次性配置（建议写进 `paths.sh`）
source /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/_shared/paths.sh
> 注意：`TARGET_4090` 必须写成 **单行字符串**，不要换行。

```bash
export TLM_ROOT=/home/hehangshuai/workspace/tlm
export DATA_ROOT=$TLM_ROOT/tlm_dataset/gen
export GEN_DATA=$DATA_ROOT/gen_data

export RUN_TAG=2026-01-04_bucketkv_lora_restart
export RUN_ROOT=$GEN_DATA/edge_runs/$RUN_TAG

export TARGET_4090='cuda -keys=cuda,gpu -arch=sm_86 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32'
export TARGET_V100='nvidia/nvidia-v100'

export BASE_CKPT=$GEN_DATA/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000
export TOKENIZER=$GEN_DATA/Model/gen_tokenizer_multi_v1_bucket

export HW_KV_ALIGNER=$GEN_DATA/Model/hw_kv_aligner_train/hw_kv_aligner.pt
export HW_EMB_V4=$TLM_ROOT/gen/Embedding/hardware_embeddings_v4.json
export HW_EMB_V2=$TLM_ROOT/gen/Embedding/hardware_embeddings_v2.json

mkdir -p $RUN_ROOT/_shared/targets
printf "%s\n" "$TARGET_4090" > $RUN_ROOT/_shared/targets/4090.target.txt
printf "%s\n" "$TARGET_V100" > $RUN_ROOT/_shared/targets/v100.target.txt
```

---

## 2. Phase 0（可选）：重新 dump programs

```bash
python $TLM_ROOT/gen/tools/dump_network_info_and_programs.py \
  --target "$TARGET_4090" --dump_programs_size 1000 --max_retries 20 --sleep_sec 30

python $TLM_ROOT/gen/tools/dump_network_info_and_programs.py \
  --target "$TARGET_V100" --dump_programs_size 1000 --max_retries 20 --sleep_sec 30
```

---

## 3. Phase 1（iter00）：base bucket+kv → measure → postprocess → base SFT

### 3.1 生成 sketch（iter00 / test_file_idx=0）

> 推荐直接用脚本（只传一个数字即可）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_sketch.sh 0
> ```
>
> 也可指定硬件：
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_sketch.sh 0 v100
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_sketch.sh 0 all
> ```

```bash
export ITER=iter00
export TEST_FILE_IDX=0

mkdir -p $RUN_ROOT/4090/$ITER/sketch
python $TLM_ROOT/gen/make_dataset.py \
  --for_type=for_gen_train_sketch \
  --target "$TARGET_4090" \
  --dataset_path $DATA_ROOT/dataset/to_measure_programs/4090 \
  --tokenizer_path "$TOKENIZER" \
  --save_path $RUN_ROOT/4090/$ITER/sketch \
  --keep_cnt=48 \
  --test_file_idx=$TEST_FILE_IDX

mkdir -p $RUN_ROOT/v100/$ITER/sketch
python $TLM_ROOT/gen/make_dataset.py \
  --for_type=for_gen_train_sketch \
  --target "$TARGET_V100" \
  --dataset_path $DATA_ROOT/dataset/to_measure_programs/v100 \
  --tokenizer_path "$TOKENIZER" \
  --save_path $RUN_ROOT/v100/$ITER/sketch \
  --keep_cnt=48 \
  --test_file_idx=$TEST_FILE_IDX
```

### 3.2 base bucket+kv 生成（不带 LoRA）

> 推荐脚本（带检查：缺 sketch 会直接报错）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_bucketkv.sh 0
> ```
>
> 指定硬件：
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_bucketkv.sh 0 v100
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_bucketkv.sh 0 all
> ```

```bash
mkdir -p $RUN_ROOT/4090/$ITER/gen $RUN_ROOT/4090/$ITER/logs
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/gen_state_kv_lora.py \
  --model_path "$BASE_CKPT" \
  --tokenizer_path "$TOKENIZER" \
  --sketch_path $RUN_ROOT/4090/$ITER/sketch/0_merge.json \
  --save_path $RUN_ROOT/4090/$ITER/gen/base_bucketkv.json \
  --target "$TARGET_4090" \
  --use_bucket \
  --use_hw_kv --hw_kv_mode real \
  --hw_kv_aligner_path "$HW_KV_ALIGNER" \
  --hardware_embedding_path "$HW_EMB_V4" \
  --pos_compensate \
  | tee $RUN_ROOT/4090/$ITER/logs/gen_base_bucketkv.log

mkdir -p $RUN_ROOT/v100/$ITER/gen $RUN_ROOT/v100/$ITER/logs
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/gen_state_kv_lora.py \
  --model_path "$BASE_CKPT" \
  --tokenizer_path "$TOKENIZER" \
  --sketch_path $RUN_ROOT/v100/$ITER/sketch/0_merge.json \
  --save_path $RUN_ROOT/v100/$ITER/gen/base_bucketkv.json \
  --target "$TARGET_V100" \
  --use_bucket \
  --use_hw_kv --hw_kv_mode real \
  --hw_kv_aligner_path "$HW_KV_ALIGNER" \
  --hardware_embedding_path "$HW_EMB_V4" \
  --pos_compensate \
  | tee $RUN_ROOT/v100/$ITER/logs/gen_base_bucketkv.log
```

### 3.3 真机测量

> 推荐脚本（带检查：缺 gen 文件会直接报错）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 0
> ```
>
> 指定硬件：
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 0 v100
> ```
>
> 指定测量模式：
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 0 4090 base
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_measure.sh 0 4090 kv_lora
> ```

```bash
mkdir -p $RUN_ROOT/4090/$ITER/measure
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/measure_programs.py \
  --batch-size 64 \
  --target "$TARGET_4090" \
  --to-measure-path $RUN_ROOT/4090/$ITER/gen/base_bucketkv.json \
  --measured-path $RUN_ROOT/4090/$ITER/measure/base_bucketkv.json \
  | tee $RUN_ROOT/4090/$ITER/logs/measure_base_bucketkv.log

mkdir -p $RUN_ROOT/v100/$ITER/measure
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/measure_programs.py \
  --batch-size 64 \
  --target "$TARGET_V100" \
  --to-measure-path $RUN_ROOT/v100/$ITER/gen/base_bucketkv.json \
  --measured-path $RUN_ROOT/v100/$ITER/measure/base_bucketkv.json \
  | tee $RUN_ROOT/v100/$ITER/logs/measure_base_bucketkv.log
```

### 3.4 更新 utils.json 后 postprocess

> 注意：本仓库的 `utils.json` 结构是按硬件顶层键（`4090`/`v100`）存储，不是顶层 `measure_records`。
>
> `postprocess.py` 会根据 `--target` 设定的硬件（HARDWARE_PLATFORM）只处理该硬件的
> `measure_records/finetuning_files/testtuning_files`，不会把多个硬件混在一起。

```bash
python - << 'PY'
import json
from pathlib import Path

utils_path = Path("/home/hehangshuai/workspace/tlm/tlm_dataset/gen/utils.json")
utils = json.loads(utils_path.read_text())

def add(hw_key, p):
    utils.setdefault(hw_key, {}).setdefault("measure_records", [])
    if p not in utils[hw_key]["measure_records"]:
        utils[hw_key]["measure_records"].append(p)

add("4090", "/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/4090/iter00/measure/base_bucketkv.json")
add("v100", "/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/v100/iter00/measure/base_bucketkv.json")

utils_path.write_text(json.dumps(utils, indent=2))
print("Updated", utils_path)
PY

python $TLM_ROOT/gen/postprocess.py --target "$TARGET_4090"
python $TLM_ROOT/gen/postprocess.py --target "$TARGET_V100"
```

> 更安全的方式：用 CLI 脚本自动追加 measured 文件（带检查提示）：
>
> ```bash
> python $TLM_ROOT/gen/scripts/add_measure_records.py \
>   --hardware 4090 \
>   --iter 0 \
>   --mode base \
>   --run-tag $RUN_TAG
> ```
>
> 如果你已经有明确路径（可直接给目录或文件）：
>
> ```bash
> python $TLM_ROOT/gen/scripts/add_measure_records.py \
>   --hardware 4090 \
>   --iter 0 \
>   --mode base \
>   --measured-path $RUN_ROOT/4090/iter00/measure
> ```
>
> 注意：多行命令要加 `\`，否则下一行会被当成新命令。

### 3.5 生成 base SFT（for_gen_best → prepare_edge_dataset）

> 提醒：本段依赖 `$ITER`。如果是在新的 shell 中执行，请先 `export ITER=iter00`，
> 否则会写到 `$RUN_ROOT/4090/sft/...`，导致后续脚本找不到 `iter00` 下的文件。

```bash
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

mkdir -p $RUN_ROOT/v100/$ITER/sft/base
python $TLM_ROOT/gen/make_dataset.py \
  --for_type=for_gen_best \
  --target "$TARGET_V100" \
  --dataset_path $DATA_ROOT/dataset/measure_records/v100 \
  --tokenizer_path "$TOKENIZER" \
  --save_path $RUN_ROOT/v100/$ITER/sft/base

touch $RUN_ROOT/v100/$ITER/sft/empty_lora.jsonl
python $TLM_ROOT/gen/prepare_edge_dataset.py \
  --base-jsonl $RUN_ROOT/v100/$ITER/sft/base/0_merge.json \
  --lora-jsonl $RUN_ROOT/v100/$ITER/sft/empty_lora.jsonl \
  --output-jsonl $RUN_ROOT/v100/$ITER/sft/edge_sft_v100.jsonl \
  --merge_mode base_left \
  --merge_key repr \
  --dedupe_mode keep_all \
  --hardware-id v100 \
  --embedding-json $HW_EMB_V4 \
  --allow-missing-lora
```

> 说明：
> - `--merge_key repr` 实际使用 `(normalize(workload_repr), target_str)`；不会跨 target 混淆。
> - `--dedupe_mode keep_all` 会保留同一 workload 的多条 PPT 示例，同时用同 key 的最小 latency 填 `lat_*_star`。
> - 若想只保留每个 key 的单条最优记录，可改 `--dedupe_mode min`。

---

## 4. Phase 2：训练 LoRA v1（lambda_gain=0）

> 推荐脚本（支持迭代号/硬件/输出 tag，并可用环境变量覆盖超参）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_edge_expert.sh 0 4090 v1_init
> ```
>
> 也可只传迭代号（默认 4090 + v1_init）：
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_edge_expert.sh 0
> ```
>
> 若需要从上一轮 LoRA 继续训练（而不是每次冷启动），设置：
> ```bash
> export EDGE_INIT_EXPERT_DIR=$RUN_ROOT/4090/iter00/experts/v1_init
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_train_edge_expert.sh 1 4090 v2_gain
> ```
>
> 说明：
> - `run_train_edge_expert.sh` 在 `idx>0` 时会默认回溯上一轮：
>   - 若设置 `EDGE_PREV_EXPERT_TAG`，使用该 tag；
>   - 否则若当前 tag 形如 `vN_gain`，自动回溯到 `v(N-1)_gain`（`v2_gain` 的上一轮为 `v1_init`）；
>   - 否则默认 `v1_init`。
> - 可用 `EDGE_LORA_GAIN_MARGIN` 覆盖 `gain_margin`（默认 0.05），用于控制 gain 何时开始生效。
> - 若不希望自动续训，可设置 `EDGE_RESUME_PREV=0`。

```bash
mkdir -p $RUN_ROOT/4090/$ITER/experts/v1_init
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/train_edge_expert.py \
  --base-model-path "$BASE_CKPT" \
  --tokenizer-path "$TOKENIZER" \
  --dataset-jsonl $RUN_ROOT/4090/iter00/sft/edge_sft_4090.jsonl \
  --output-dir $RUN_ROOT/4090/$ITER/experts/v1_init \
  --num-epochs 3 \
  --batch-size 4 \
  --lambda-gain 0.0 \
  --warmup-steps 200 \
  --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj \
  | tee $RUN_ROOT/4090/iter00/logs/train_lora_v1_init.log

mkdir -p $RUN_ROOT/v100/$ITER/experts/v1_init
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/train_edge_expert.py \
  --base-model-path "$BASE_CKPT" \
  --tokenizer-path "$TOKENIZER" \
  --dataset-jsonl $RUN_ROOT/v100/iter00/sft/edge_sft_v100.jsonl \
  --output-dir $RUN_ROOT/v100/$ITER/experts/v1_init \
  --num-epochs 3 \
  --batch-size 4 \
  --lambda-gain 0.0 \
  --warmup-steps 200 \
  --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj \
  | tee $RUN_ROOT/v100/iter00/logs/train_lora_v1_init.log
```

---

## 5. Phase 3：LoRA 生成 → 测量 → v2 merge → gain 训练

> 一键脚本（会检测已有产物，缺什么补什么，支持指定迭代号与阶段）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_iter_full.sh 1 4090 v1_init 2
> ```
>
> 含义：`iter01` + `4090` + 以 `v1_init` 作为上一轮专家生成 LoRA，并训练 `v2_gain`。

### 5.1 LoRA 推理生成（KV + LoRA）

> 推荐脚本（带检查：缺 sketch 或 router.json 会直接报错）：
>
> ```bash
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_kv_lora.sh 0 /path/to/expert_dir 4090
> ```
>
> 或者先设置：
> ```bash
> export EDGE_EXPERT_DIR=/path/to/expert_dir
> bash /home/hehangshuai/workspace/tlm/gen/scripts/run_gen_kv_lora.sh 0  # 默认 4090
> ```

> 说明：路由 embedding 与 KV 注入统一使用 v4（默认 `--edge_embedding_path "$HW_EMB_V4"`）。

```bash
mkdir -p $RUN_ROOT/4090/$ITER/gen $RUN_ROOT/4090/$ITER/logs
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/gen_state_kv_lora.py \
  --model_path "$BASE_CKPT" \
  --tokenizer_path "$TOKENIZER" \
  --edge_expert_dirs $RUN_ROOT/4090/$ITER/experts/v1_init \
  --edge_embedding_path "$HW_EMB_V2" \
  --sketch_path $RUN_ROOT/4090/$ITER/sketch/0_merge.json \
  --save_path $RUN_ROOT/4090/$ITER/gen/kv_lora.json \
  --target "$TARGET_4090" \
  --use_bucket \
  --use_hw_kv --hw_kv_mode real \
  --hw_kv_aligner_path "$HW_KV_ALIGNER" \
  --hardware_embedding_path "$HW_EMB_V4" \
  --pos_compensate \
  | tee $RUN_ROOT/4090/$ITER/logs/gen_kv_lora.log
```

### 5.2 测量 + postprocess（LoRA 产物）

```bash
mkdir -p $RUN_ROOT/4090/$ITER/measure
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/measure_programs.py \
  --batch-size 64 \
  --target "$TARGET_4090" \
  --to-measure-path $RUN_ROOT/4090/$ITER/gen/kv_lora.json \
  --measured-path $RUN_ROOT/4090/$ITER/measure/kv_lora.json \
  | tee $RUN_ROOT/4090/$ITER/logs/measure_kv_lora.log

# 追加到 utils.json 后再 postprocess
# 参照 3.4 的脚本，把 kv_lora.json 加到 utils.json 的 4090.measure_records
python $TLM_ROOT/gen/postprocess.py --target "$TARGET_4090"
```

### 5.3 生成 v2 SFT + merge + gain 训练

```bash
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

# merge：优先用 lora_left（若 paired 数很小仍可训练）
python $TLM_ROOT/gen/prepare_edge_dataset.py \
  --base-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090.jsonl \
  --lora-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2.jsonl \
  --output-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2_lora_left.jsonl \
  --merge_key repr \
  --merge_mode lora_left \
  --dedupe_mode keep_all \
  --embedding-json $HW_EMB_V4

mkdir -p $RUN_ROOT/4090/experts/v2_gain
CUDA_VISIBLE_DEVICES=0 python $TLM_ROOT/gen/train_edge_expert.py \
  --base-model-path "$BASE_CKPT" \
  --tokenizer-path "$TOKENIZER" \
  --dataset-jsonl $RUN_ROOT/4090/$ITER/sft/edge_sft_4090_v2_lora_left.jsonl \
  --output-dir $RUN_ROOT/4090/experts/v2_gain \
  --num-epochs 3 \
  --batch-size 4 \
  --lambda-gain 0.1 \
  --gain-margin 0.05 \
  --warmup-steps 200 \
  --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj \
  | tee $RUN_ROOT/4090/$ITER/logs/train_lora_v2_gain.log
```

---

## 6. 迭代策略（iter01~03）

```
iter00: test_file_idx=0
iter01: test_file_idx=1
iter02: test_file_idx=2
iter03: test_file_idx=3
```

跑完 0~3 基本覆盖完整的 `to_measure_programs` 文件集（4-fold）。

---

## 7. 关键备注

1) `utils.json` 的结构是按硬件顶层键（`4090`/`v100`）存 `measure_records`，不要写成顶层 `measure_records`。  
2) `for_gen_train_sketch` 会按 `test_file_idx` 选 1/4 文件；只跑 idx=0 会导致 base 数据严重不完整。  
3) KV 使用 `hardware_embeddings_v4.json`；LoRA 路由默认还是 v2（29 维），需显式传 `--edge_embedding_path "$HW_EMB_V2"`。  
4) `prepare_edge_dataset.py --merge_mode lora_left` 已支持，`train_edge_expert.py` 会自动只对 paired 样本计算 gain loss。  
5) 如果 `paired_count` 为 0，说明 base/lora 不是同一批 repr，需检查是否共用同一份 sketch 与测量记录。  
