## 2025-02-12 V100 单专家训练（阶段二）
- 数据集：`sft_dataset_mutil_v1/v100_gen_best_multi/edge_sft_v100.jsonl`（231 条，含 `hw_emb` + `lat_base_star`，`lat_lora_star=None`）。
- 训练命令：
  ```
  python train_edge_expert.py \
    --base-model-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/clm_gen_multi_v1 \
    --tokenizer-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/multi_hardware_tokenizer \
    --dataset-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/v100_gen_best_multi/edge_sft_v100.jsonl \
    --output-dir /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_experts/v100/stage02_20250212_fullrun \
    --num-epochs 3 \
    --batch-size 4 \
    --lambda-gain 0.0 \
    --warmup-steps 200 \
    --target-modules attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj
  ```
- 关键指标：`avg_loss=6.80 → final task_loss=5.09`，门控均值 `g_mean` 从 0.38 提升到 0.55，`gain_loss=0`（因暂未注入 `lat_lora_star`）。训练日志暂未持久化，后续补写 `train.log`。
- 产物目录：`edge_experts/v100/stage02_20250212_fullrun/`（含 `adapter_model.safetensors`、`router.json`、`metrics.json` 等，`router.meta.train_samples=231`）。
- 下一步：真机测得 LoRA 延迟并更新 JSONL 后，重新训练以开启 `lambda_gain>0`，同时在 `EdgeTLM_update_log.md` 持续记录阶段二→三进展。

## 统一改造计划（2025-XX-XX）
- **建模层重构**：在 `modeling/experts/` 内新增 `GatedLoRAExpert`，并引入 `base_plus_experts.py`，实现 `BasePlusExperts` 容器，确保单 LoRA 专家 + 单行路由的前向形式。
- **训练损失模块**：创建 `training/losses.py`，实现 `compute_task_loss`、`compute_gain_loss`、`entropy_reg`、`l2r_reg` 等函数，支持冷启动与退火。
- **训练脚本改造**：重写/新增 Edge 专家训练入口（暂定 `train_edge_expert.py`），使用 `FrozenBaseWrapper` + `ExpertRegistry` + 新专家，实现单专家训练循环；旧 HA/HS 流程暂保留，后续逐步移除。
- **推理与导出**：在完成训练改造后，调整推理脚本接入 `BasePlusExperts.forward_multi`，并统一导出 `{adapter_model.bin, adapter_config.json, router.json, metrics.json}`。
- **日志维护**：所有阶段进展继续追加至本文件顶端，方便学生按阶段阅读。

# EdgeTLM 更新记录（阶段一）

## 目标
- 冻结 TLM-BASE（作为稳定锚点），为“每硬件独立 LoRA 专家 + 单行路由向量”的新架构做准备。
- 不改变当前训练主体逻辑（保持无HA、无Mixer的LoRA路径），先完成基础设施改造。

## 本次新增/修改

### 新增模块
- `modeling/frozen_base.py`
  - 提供 `FrozenBaseWrapper`：
    - 冻结并包裹基础模型（`.eval()` + `requires_grad_(False)`）。
    - 透传属性访问，保留 `config`。
    - 提供可选 `router_feature(hidden_states, hw_emb)` 辅助。
- `modeling/__init__.py`
  - 暴露 `FrozenBaseWrapper`。

### 训练脚本更新
- `train_mt_moslora.py`
  - 引入顶层 `modeling` 包（添加路径与导入）。
  - 在应用 MT-MoSLoRA 前将基础模型包裹为 `FrozenBaseWrapper(model)`（若已包裹则跳过）。
  - 遍历/保存逻辑统一兼容包装层：使用 `target_model = getattr(model, 'base_model', model)`。

## 使用方式
1. 正常运行原有训练命令（无须新增参数）。
2. 训练流程中会自动冻结并包裹BASE，不影响现有的LoRA训练（默认为无HA、无Mixer）。
3. 若后续需要从BASE提取路由特征，可调用 `FrozenBaseWrapper.router_feature()`。

## 兼容性
- 不改变现有LoRA训练行为；仅引入了“冻结BASE + 兼容遍历/保存”的基础设施。
- 后续可在此基础上逐步接入“专家接口/注册表、单行路由向量与负偏置”等功能。

# EdgeTLM 更新记录（阶段二筹备）

## 本次更新（2025-XX-XX）
- 新增 `prepare_edge_dataset.py`，将 `all_gen_best_multi` 样本转化为包含 `hw_emb`、`lat_base_star` 的 JSONL，便于后续训练。
- 新增 `train_edge_expert.py`，串联 `FrozenBaseWrapper`、`BasePlusExperts`、`GatedLoRAExpert` 与损失模块，实现单专家训练并兼容缺失 `lat_lora_star` 的冷启动。
- `utils.py` 支持 `TLM_DATA_ROOT` 环境变量，适配当前仓库目录。
- `gen_state.py` 接入 `BasePlusExperts.forward_multi`，支持加载多专家目录并按 Top-K 门控生成。
- `train_edge_expert.py` 现导出 `{adapter_model.bin, adapter_config.json, router.json, metrics.json}`，`router.json.meta` 记录硬件信息与训练配置，以便专家上线与审计。
- 建模层新增 `modeling/experts/gated_lora.py` 与 `modeling/base_plus_experts.py`，实现单行路由 LoRA 专家及 `BasePlusExperts` 组合容器。
- 扩展 `modeling/__init__.py`、`modeling/experts/__init__.py`、`modeling/experts/interface.py`，统一导出接口并支持实例级注册/序列化。
- 新增训练损失模块 `training/losses.py`（含任务损失、增益惩罚、熵正则、路由 L2）及 `training/__init__.py` 导出入口。
- 更新 `EdgeTLM_update_log.md`，记录阶段性计划与日志规范。


## TODO
- [x] 落地 Edge 专家训练脚本（`train_edge_expert.py`）并迁移现有 SFT 流程。
- [x] 为 SFT 数据集补充 `hw_emb`、`lat_base_star`、`lat_lora_star` 等字段，或设计代理测量路径。确保 compute_gain_loss 可用。
- [x] 调整 `gen_state.py` 等推理脚本，引入 `BasePlusExperts.forward_multi` 与 Top-K 管理。
- [x] 统一导出 `{adapter_model.bin, adapter_config.json, router.json, metrics.json}`，实现专家序列化/反序列化。
- [ ] 编写/更新单元测试（冻结基座、专家注册与序列化、门控前向梯度）。
