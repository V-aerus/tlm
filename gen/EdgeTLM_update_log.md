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

123

