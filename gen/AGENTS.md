# Repository Guidelines

## Project Structure & Key Specs
- 代码根目录：`/home/hehangshuai/workspace/tlm`，主工作目录在 `gen/`。
- 当前主线架构：**Bucket 文本通道 + HW KV 注入 + LoRA 专家路由**；`TLM-BASE` 冻结，KV 对齐器提供硬件侧通道，LoRA 专家负责性能迁移。
- 数据根目录：`/home/hehangshuai/workspace/tlm/tlm_dataset/gen`，运行产物在 `gen_data/edge_runs/<RUN_TAG>/`。
- 统一环境入口：`gen_data/edge_runs/<RUN_TAG>/_shared/paths.sh`，集中配置 `RUN_ROOT / BASE_CKPT / TOKENIZER / HW_KV_ALIGNER / HW_EMB_V4`。
- 每轮迭代目录约定：
  - `iterXX/sketch/0_merge.json`：草图
  - `iterXX/gen/{base_bucketkv.json, kv_lora.json}`：生成结果
  - `iterXX/measure/{base_bucketkv.json, kv_lora.json|kv_lora_measured.json}`：真实测量
  - `iterXX/sft/`：SFT 数据与 merge 产物
  - `iterXX/experts/`：LoRA 专家（含 `router.json/metrics.json`）
  - `iterXX/logs/`：生成/测量/训练日志

## Current Run Status (2026-01-04_bucketkv_lora_restart)
- 4090：`iter07` 已完成 kv_lora 测量与专家 `v8_gain` 训练；`iter08` 仅完成生成与部分测量，测量行数未对齐。
- v100：`iter06` 已完成 kv_lora 测量与专家 `v7_gain` 训练；`iter07` 仅完成生成，测量与训练未推进。
- 3090/2080：仅有 eval 生成产物（用于跨硬件验证），未进入测量/训练链路。

## Runbook & Scripts
- 迭代主入口：
  - `gen/scripts/run_iter_full.sh`：全流程（含 base + kv_lora）
  - `gen/scripts/run_iter_lora_only.sh`：跳过 base 的增量流程
- 生成与测量：
  - `gen/scripts/run_gen_bucketkv.sh`：bucket+KV 基线生成
  - `gen/scripts/run_gen_kv_lora.sh`：KV+LoRA 生成（需专家目录）
  - `gen/scripts/run_measure.sh`：测量 base/kv_lora
  - `gen/scripts/measure_watchdog.py`：断点续测与进度监控
- 训练与评估：
  - `gen/scripts/run_train_edge_expert.sh`：训练 LoRA 专家
  - `gen/scripts/run_gen_bert_eval.sh`：BERT 端到端评测
  - `gen/scripts/run_pipeline_collect.py`：汇总覆盖率/延迟指标
- 常用环境变量：
  - `EDGE_PATHS_SH`：指向 `_shared/paths.sh`
  - `EDGE_HW`：`4090|v100`
  - `EDGE_CUDA`：GPU id
  - `EDGE_KEEP_CNT`：每个 workload 保留候选数
  - `EDGE_LORA_EMB`：路由 embedding（默认 v4；旧专家用 v2）

## Quality Gates & Sanity Checks
- **生成/测量行数对齐**：`gen/kv_lora.json` 与 `measure/kv_lora*.json` 行数应一致。
- **测量异常信号**：日志出现 `programs: 0/16` 需重测或用 watchdog 恢复。
- **生成异常信号**：`All states are invalid` 可少量出现，但不应造成大面积 workload 失败。
- **统计回归**：建议用 `run_pipeline_collect.py` 汇总 overall latency 与覆盖率趋势。

## Coding Style & Naming Conventions
- Python：4 空格缩进，`snake_case`；路由/门控类以 `Gated*` 命名，冻结基座以 `FrozenBase*` 命名。
- 专家产物固定为 `adapter_model.* / adapter_config.json / router.json / metrics.json`。
- `router.json.hardware_dim=24` 对应 `Embedding/hardware_embeddings_v4_universe.json`；旧专家若为 `hardware_dim=29` 必须切换到 v2。

## Testing Guidelines
- 本地变更优先跑 `python3 -m pytest -q`；需要真实硬件的测试可通过 `pytest -k` 过滤。
- 涉及测量的流程先用 `--dry-run` 或 `measure_watchdog` 验证路径与覆盖率。

## Documentation & Reporting
- 新方案或阶段性结论请更新 `EdgeTLM_update_log.md`，并标注 run tag/iter/硬件。
- 记录关键产物路径、专家 tag、测量覆盖率与异常信号，便于学生复现实验。
