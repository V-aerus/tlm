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
- 建模层新增 `modeling/experts/gated_lora.py` 与 `modeling/base_plus_experts.py`，实现单行路由 LoRA 专家及 `BasePlusExperts` 组合容器。
- 扩展 `modeling/__init__.py`、`modeling/experts/__init__.py`、`modeling/experts/interface.py`，统一导出接口并支持实例级注册/序列化。
- 新增训练损失模块 `training/losses.py`（含任务损失、增益惩罚、熵正则、路由 L2）及 `training/__init__.py` 导出入口。
- 更新 `EdgeTLM_update_log.md`，记录阶段性计划与日志规范。


## TODO
- [ ] 落地 Edge 专家训练脚本（`train_edge_expert.py`）并迁移现有 SFT 流程。基于新模块实现 train_edge_expert.py（或改造现有脚本），替换原 HA/HS 流程，串联 FrozenBaseWrapper → ExpertRegistry →GatedLoRAExpert → BasePlusExperts。
- [ ] 为 SFT 数据集补充 `hw_emb`、`lat_base_star`、`lat_lora_star` 等字段，或设计代理测量路径。确保 compute_gain_loss 可用。
- [ ] 调整 `gen_state.py` 等推理脚本，引入 `BasePlusExperts.forward_multi` 与 Top-K 管理。
- [ ] 统一导出 `{adapter_model.bin, adapter_config.json, router.json, metrics.json}`，实现专家序列化/反序列化。
- [ ] 编写/更新单元测试（冻结基座、专家注册与序列化、门控前向梯度）。
