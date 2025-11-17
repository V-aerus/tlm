# EdgeTLM 背景知识（基于 TLM OSDI'24）

本文件汇总当前我们对 TLM 与 EdgeTLM 的共同理解，记录参考资料路径与关键结论，便于后续回顾与与他人/AI 协作讨论。

## 资料索引（仓库内路径）

- 论文 PDF：`gen/Zhai 等 - Enabling Tensor Language Model to Assist in Genera.pdf`
- 上游 README（TLM 主流程与命令）：`README.md`
- EdgeTLM 快速手册（命令与目录约定）：`gen/EdgeTLM_readme.md`
- EdgeTLM 架构演进与伪代码：`gen/EdgeTLM_update.md`
- EdgeTLM 迭代记录与进展：`gen/EdgeTLM_update_log.md`

## 上游 TLM（OSDI'24）概述

- 问题背景：高性能张量程序搜索空间大、探索困难。TLM 将“调度探索”转化为“语言模型生成”任务，借助“张量语言/张量句子”表示决策序列，通过采样与真实测量选择更优实现。
- 典型三阶段（以 V100 为例）：
  1) 拆分子图：`dump_network_info.py --target=<hw>`，将 BERT/ResNet 等模型划分为包含 MatMul/Relu/Norm 等算子的子图，存为 PKL。
  2) 生成张量程序语料：`dump_programs.py --target=<hw>`，利用改造后的 TVM 为子图生成“张量句子”（未标注、未测延迟）。
  3) 构建 tokenizer 与预训练：`make_dataset.py --for_type=for_gen_tokenizer ...` 用该硬件语料构建词表与 tokenizer；用未标注“张量句子”进行 CLM 预训练得到 TLM-base（学到语法与可行性）。
  4) 构建 SFT 数据：对候选程序做真机测量，`postprocess.py` 整理到 `dataset/measure_records/<hw>`，`make_dataset.py --for_type=for_gen_best` 生成用于 SFT 的监督语料。
  5) SFT 训练：在该硬件的测量数据上 `run_train_clm_best_<hw>.py`，得到“面向该硬件优化”的 TLM。
- 硬件耦合点：
  - `--target` 决定调度空间与出现的 token（如线程块、向量化宽度、架构名）；
  - tokenizer 通常“每硬件一套”（如 `gen_tokenizer_v100`）；
  - SFT 数据完全来源于该硬件的测量记录；
  - 因此“预训练 Base 虽不使用延迟标签，但仍带硬件分布偏好”。

## 我们的目标（EdgeTLM）

- 痛点：原链路通常需要“每个硬件重新构建 tokenizer、重训 Base、再做 SFT”，迁移成本高；面对陌生（未学习）硬件时冷启动慢。
- 目标：在不重新训练基座的前提下，让同一套系统在陌生硬件上也有较优表现，并能快速自举到更好性能。

## EdgeTLM 设计要点（与上游 TLM 的差异化增强）

- 冻结基座（Frozen TLM-BASE）：作为“稳定锚点”，评估态推理，参数 `requires_grad=False`；提供路由特征提取接口（如硬件 embedding 或隐状态均值）。见 `gen/EdgeTLM_update.md` 中 FrozenBaseWrapper 设计。
- LoRA 专家 + 单行路由：每个专家包含 LoRA 权重 + 一行路由向量 `r` + 负偏置 `b≤0` + 温度 `τ`。
  - 训练期：采用 Sigmoid 连续门控，学习“该开多大”；
  - 推理期：将 BASE 作为 0 分通道，专家分数与 BASE 一起做 Softmax Top-K 稀疏融合；
  - 负偏置保证“更挑活儿”，无强信号时不轻易开启专家。
- 收益驱动：利用真机测量的 `lat_base_star` 与 `lat_lora_star` 构造增益 `gain=lat_base_star-lat_lora_star`，以 `compute_gain_loss` 驱动门控学习；支持冷启动（前若干步屏蔽增益项）。
- 专家注册与聚合：`ExpertRegistry` 管理多个 `{LoRA, r, b, τ}` 专家；`BasePlusExperts` 负责 BASE 与专家的融合前向，支持 Top-K 与行级屏蔽治理；统一导出 `{adapter_model.*, router.json, metrics.json}`。

## “陌生硬件”策略

- 零样本可用：BASE 通道作为稳健兜底；路由负偏置避免错误迁移引发性能回退。
- 相似迁移：若有相近硬件的 `hw_emb`，Top-K 门控可能小幅启用相近专家作为“软迁移”。
- 快速自举：按“生成→测量→后处理”获得少量 `lat_lora_star` 后，在该硬件独立训练一个新 LoRA 专家并注册，无需改动 BASE 与其他专家。
- 持续演进：多组织/多人独立贡献专家，通过注册/屏蔽策略安全合并，保持扩展性与可审计。

## 与仓库现状的对应

- 快速手册给出了多硬件的命令与目录规范（`edge_experts/`、`edge_sketches/`、`edge_experts_gen/`、`edge_experts_measure_data/` 等），便于并行迭代与记录。见 `gen/EdgeTLM_readme.md`。
- 架构演进文档提供了 FrozenBaseWrapper、GatedLoRAExpert、BasePlusExperts 的伪代码与训练损失设计，以及推理合并/Top-K 路由与产物规范。见 `gen/EdgeTLM_update.md`。
- 迭代日志记录了 V100 单专家阶段二首轮训练（`lambda_gain=0` 冷启动），下一步是补入 `lat_lora_star` 并开启收益约束训练。见 `gen/EdgeTLM_update_log.md`。

## 关键结论（便于快速复用）

- 上游 TLM 的 base 预训练虽不含延迟标签，但会因语料/词表而携带硬件偏好；真正的“跨硬件泛化”需要在建模/门控层面补齐（EdgeTLM）。
- EdgeTLM 通过“冻结 BASE + 可插拔 LoRA 专家 + 收益驱动门控”把“是否开启、开多大”统一进可微分的门控学习；推理期用 BASE 0 分通道 + Softmax Top-K 稀疏融合，天然支持专家屏蔽与审计。
- 面向陌生硬件：先用 BASE 兜底与相邻专家软迁移，随后小样本真测自举新专家；不必重建 tokenizer 或重训 BASE。

## 术语对照

- 张量句子（tensor sentence）：以序列化“调度决策 token”表示的张量程序，用于 CLM 训练与推理生成。
- SFT：使用真机测量数据构建的监督语料，对齐“更快的程序”偏好。
- LoRA 专家：在冻结 BASE 上的低秩增量模块，配单行路由与负偏置；训练产物可独立分发与合并。

## 开放问题与备注

- tokenizer 统一化：多硬件共用一个 tokenizer（如 `gen_tokenizer_multi_v1`）的边界与兼容性；需权衡词表大小与生成质量。
- 代理收益与抽查真测：在训练环节如何更好地用代理模型加速、用真机抽查校准（`latency_cache.py` / `proxy_score.py` 预留）。
- 指标与审计：推理记录 Top-K 权重与命中专家清单，便于课堂/实验复现与诊断。

（本文件将随阶段性进展持续更新。如需教学用图示/流程图，可在本文件下方追加“附录”。）

