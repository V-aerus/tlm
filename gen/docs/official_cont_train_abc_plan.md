# Official-Continue 续训 A/B/C 方案（Xavier 数据）

## 1. 背景与核心问题
- 目标：验证 `clm_gen_best_v100` 在 Xavier 示范数据上继续训练后，是否能提升在边缘场景（hold-out 评测）的可行解与延迟表现。
- 已确认现状：
  - 现有 CLM 续训默认是**整句 token loss**（非 schedule-only）。
  - Xavier 数据中存在固定 OOV 片段（在 v100 tokenizer 下变成 `[UNK]`），可能稀释训练信号。
- 关键矛盾：
  - 若全句 loss：可能浪费容量在“复述前缀/硬件串”。
  - 若只看 schedule：担心忽略子图形状上下文。

## 2. 实验目标
- 在**不重建测量记录**前提下，做 A/B/C 三种续训方式对比。
- 统一评测口径，只比较“模型在测试集 hold-out 上的生成合法率与性能”。
- 输出可用于论文/报告的结论：哪种 loss 口径最稳、最有效。

## 3. 数据与防污染约束
- 训练数据：优先复用已有 Xavier SFT JSONL（避免再次生成引入混杂）。
- 评测数据：固定 hold-out sketch（与当前 eval 口径一致）。
- 本轮锁定训练集（未做测试集强化回灌）：
  - `/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/xavier/iter36/sft/`
  - 主要使用：`edge_sft_xavier_v37_lora_left.jsonl`
- 约束：
  - 不把本轮评测结果回灌到训练集（先做 zero-shot continuation 对比）。
  - 三组实验只改 loss/label 规则，其它训练超参数保持一致。

## 4. A/B/C 定义
- A（Full-Seq）：整句 CLM loss（当前默认方式，作为基线）。
- B（Prefix-Masked）：只屏蔽硬件 canonical 段（target+host 等）对应标签，不屏蔽 DAG/shape 与 schedule。
- C（Schedule-Focused）：仅对 `PPT` 之后的调度段计算 loss（前缀作为条件输入但不计监督）。

## 5. 统一训练设置（保持公平）
- 初始模型：`clm_gen_best_v100`（同一 checkpoint）。
- tokenizer：先保持 `clm_gen_best_v100` 自带 tokenizer（第一轮对比先不改词表）。
- epoch / lr / batch / max_len / seed：三组一致。
- 产物命名建议：
  - `official_cont_a_fullseq`
  - `official_cont_b_prefixmask`
  - `official_cont_c_schedule`

## 6. 统一评测设置
- 固定同一份 hold-out sketch、同一 keep_cnt。
- 每组评估：
  1) 生成覆盖：workload 级成功数、合法候选数。
  2) 测量覆盖：可测记录数、失败 workload 清单。
  3) 延迟统计：p50/p90/mean（与现有评测脚本保持一致）。
- 补充记录：
  - OOV/[UNK] 比例（训练集采样统计）
  - 训练 loss 曲线与最终 checkpoint 指标

## 7. 风险与解释框架
- 若 A > B/C：说明前缀监督仍提供有效正则，schedule-only 过窄。
- 若 B/C > A：说明全句监督被前缀/OOV 干扰，聚焦调度更有效。
- 若 B > C：说明保留 DAG/shape 监督有帮助。
- 若 C > B：说明生成任务本质更接近“条件续写调度”，应减少前缀学习负担。

## 8. To-Do（执行清单）
- [x] T1. 锁定一份 Xavier 复用训练集 JSONL（不重建，iter36/sft）。
- [ ] T2. 在训练入口增加可切换 label mask 模式（A/B/C）。
- [ ] T3. 跑 A/B/C 三组续训（同超参数、同 seed）。
- [ ] T4. 在固定 hold-out 上完成生成+测量。
- [ ] T5. 汇总合法率/延迟/失败 workload，形成对比表。
- [ ] T6. 给出推荐口径（后续是否替换默认续训方式）。

## 9. 下一步执行策略（本轮不下 CLI）
- 先实现最小改动的 A/B/C 开关（只动训练 label 构造，不动生成/测量链路）。
- 然后跑一轮小样本 sanity（每组少量 step）确认 loss 生效范围正确。
- 再跑完整 A/B/C 并汇总最终评估结果。

## 10. 代码改动与备份约束（先确认后开工）
- 先备份，再改代码（不直接覆盖稳定链路）：
  - 备份目录建议：`gen/docs/backups/official_cont_abc_<timestamp>/`
  - 备份目标（最小集合）：
    - `gen/train_clm.py`
    - `gen/run_train_clm_best_v100.py`（仅当需要统一入口参数时）
- 预期改动点（最小方案）：
  - 在 `gen/train_clm.py` 新增可选参数 `loss_mask_mode`，支持：
    - `full_seq`（A）
    - `prefix_hardware_mask`（B）
    - `post_ppt_only`（C）
  - 在 tokenization 阶段构造 `labels`（未监督 token 置 `-100`），避免影响现有推理/测量脚本。
  - `gen/run_train_clm_best_v100.py` 仅作透传参数（可选，不改也可手动 CLI 传参）。
