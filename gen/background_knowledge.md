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

## loss 设计（当前 Base 模型为解决 OOV 的尝试）

- 背景：  
  - 多硬件统一 tokenizer 后，新的/陌生硬件（如 3090）在 target 字符串与硬件参数上容易出现 OOV 或分词极不自然的问题；  
  - 我们希望让 Base 模型通过“注入一个 hw_embed 生成的向量”来替代原本冗长的硬件字段，从而在 prompt 层面对硬件语义进行对齐。
- 设计思路：  
  1. 数据构造：  
     - 使用 `make_dataset.py --for_gen --emit_hw_student`，对 TVM 记录构造两路文本：  
       - `text`：完整张量句子，包含 target 字符串与硬件参数数组 `[-1, 16, 64, ...]`；  
       - `text_student`：将 target 字符串与紧随其后的硬件参数整体替换为单个 `[MASK]`，并附上 `hw_id/hw_name/hw_emb`。  
  2. 模型结构：  
     - 冻结 `AutoModelForCausalLM` 作为 Base；  
     - 引入 `ProtoMixAligner(proto_keys, embed_dim)`，将连续硬件向量 `hw_emb` 映射到模型 embedding 空间的注入向量 `hw_embed`；  
     - 在 `text_student` 中找到 `[MASK]`，用 `hw_embed` 替换所有 `[MASK]` 位置的 embedding，再送入 Base 前向。  
  3. 损失组合（当前版本）：  
     - 局部 LM Loss：  
       - 只在 `[MASK]` 之后的一小段窗口（如 64 个 token）计算 LM 交叉熵，其余 label 置为 `-100`，避免整句无关 token 噪声主导梯度；  
     - 硬件分类 Loss：  
       - 使用 `cls_head(hw_embed)` 直接对 ProtoMix 输出进行 4 类硬件分类（V100/A40/Xavier/CPU），强制 `hw_embed` 携带硬件身份信息；  
     - 原型正交正则 Loss：  
       - 在 `ProtoMixAligner` 内对可学习原型向量做 Gram 矩阵正交正则（`||P_norm P_norm^T - I||^2`），防止所有原型塌缩到同一方向；  
     - 对比学习 Loss（探索中）：  
       - 尝试过 InfoNCE / SupCon 形式，用 `[MASK]` 处上下文表示 `z_prog` 与 `hw_embed` 做对比，期望同硬件样本对齐、异硬件样本拉远；  
       - 实验发现，在冻结 Base + LM 强约束 + 程序上下文极度多样的前提下，对比项很难优化（loss 长期停在 `log(batch_size)` 附近），目前在主线训练中默认关闭 `lambda_ctr=0`。
- 这样设计 loss 的原因：  
  - 局部 LM：保证“注入后局部语法/结构仍然正确”，避免 ProtoMix 为了分类等任务破坏 Base 已有的张量语言能力；  
  - 分类头接 `hw_embed`：防止分类器通过程序上下文作弊，把“区分硬件”的压力直接施加在 ProtoMix 输出上；  
  - 正交正则：在没有强几何约束时，原型向量容易全部学成“通用 GPU token”，导致不同硬件的 `hw_embed` 在几何上几乎重合；  
  - 对比学习（探索）：理论上可以把程序上下文表示与硬件向量几何对齐，但在当前设定下与 LM/CLS 目标冲突较大，优化困难，因此暂不作为主力损失。
- 遇到的难题：  
  - 程序上下文本身对硬件有强偏置：即便遮掉 target 与硬件参数，某些子图/shape/调度模式仍然“暗示”这是哪种硬件，导致分类任务容易靠上下文完成；  
  - LM Loss 对注入向量有强约束：在冻结 Base 情况下，大幅改变 `hw_embed` 会破坏原有预测分布，优化器更倾向于学出一个“统一安全的注入向量”；  
  - 对比 Learning 难以收敛：当同一硬件下的不同程序上下文差异很大时，把所有 `z_prog` 都拉向同一个硬件中心向量在几何上代价很高，在 LM/CLS 的压力下对比项往往停在随机 baseline。  
- 当前达成的效果（阶段性结论）：  
  - 在 V100/A40/Xavier/CPU 这 4 个硬件上，可以训练出一个稳定的 ProtoMix 对齐器，使局部 LM Loss 从 ~1.5 降到 ~0.4，CLS Loss 降到 ~0.6 左右，训练过程平稳；  
  - 但几何检查（mixing 权重与对齐后向量余弦相似度）显示，所有 GPU 类硬件的 mixing 仍然接近均匀，对齐后的 `hw_embed` 几乎重合，只有 CPU 略微偏离——  
    换言之，**当前 loss 设计在现有数据/约束下，只能勉强学出一个“通用 GPU 注入向量 + 略微区分 CPU”的解，还没有真正学到细粒度、可插值的硬件语义几何**。  
  - 这一结论提醒我们：在冻结 Base 的前提下，要想通过 prompt 注入彻底解决 OOV / 硬件语义问题，仅靠 LM+分类+简单几何正则仍然不够，后续需要结合更贴近端到端性能的目标（如延迟代理）或更强的结构性假设。

## 术语对照

- 张量句子（tensor sentence）：以序列化“调度决策 token”表示的张量程序，用于 CLM 训练与推理生成。
- SFT：使用真机测量数据构建的监督语料，对齐“更快的程序”偏好。
- LoRA 专家：在冻结 BASE 上的低秩增量模块，配单行路由与负偏置；训练产物可独立分发与合并。

## 开放问题与备注

- tokenizer 统一化：多硬件共用一个 tokenizer（如 `gen_tokenizer_multi_v1`）的边界与兼容性；需权衡词表大小与生成质量。
- 代理收益与抽查真测：在训练环节如何更好地用代理模型加速、用真机抽查校准（`latency_cache.py` / `proxy_score.py` 预留）。
- 指标与审计：推理记录 Top-K 权重与命中专家清单，便于课堂/实验复现与诊断。

（本文件将随阶段性进展持续更新。如需教学用图示/流程图，可在本文件下方追加“附录”。）

## HwToken 注入现状与问题小结

- 现状：  
  - 基于 `align_train_multi_merged/0_merge.json` 构造了 `text/text_student/hw_emb` 数据；  
  - 实现了 ProtoMix 注入 + HwToken 训练（`train_hw_injection.py`），以及 LoRA+ProtoMix 联合训练脚本（`train_hw_injection_lora.py`）；  
  - loss 曲线在多轮实验中表现良好：局部 LM CE 可从 ~1.5 降到 0.4 左右，CLS loss 降到 0.6~0.8，ORTH 稳定收敛；  
  - 但在 `gen_state.py + TVM` 端插入 HwToken（target→[MASK]、去除 8 个整数）后，对 4090 bert_base 的全部 workload 出现 “All states are invalid”，生成 0 条合法记录。
- 诊断结论（初步）：  
  - 对齐器几何检查（`debug_hw_aligner_mixture.py`）显示：三种 GPU 原型（V100/A40/Xavier）与 3090 的 mixing 几乎均匀，输出向量在 embedding 空间几乎完全重合，仅 CPU 略有偏离；  
  - 在冻结 Base、仅训练投影器（及其正则）的设置下，最容易的局部最优解是“学出一个通用 GPU 注入向量”，在训练数据上维持局部 LM 表现不坏，但在严格的 TVM 结构生成链上，缺乏足够硬件语义与结构先验，导致生成 schedule 大量不合法；  
  - 换言之，当前损失设计在“不动 Base”前提下，只能勉强学到“粗粒度 GPU vs CPU 区分”，无法支撑“完全用 HwToken 替代 target 字符串”这种强烈的格式变化。
- 与 Can-LLM 三阶段训练的对照启发：  
  - Can-LLM 中，Stage-2 并非只训练投影器（GraphToken），而是同时通过 LoRA/soft prompt 让 LLM 适配“带 GraphToken 的新 prompt 格式”；  
  - 我们最初的 Hw 注入尝试相当于仅做了“投影器训练 + 冻结 Base”，缺失了 “让 Base 在新格式下重新适配”的 LoRA 通路；  
  - 这解释了为什么在 loss 看起来不错的情况下，一旦真正贯穿 `gen_state.py → SketchPolicy → TVM` 全链路，注入版本仍然整体失败。
- 当前解决思路（阶段性规划）：  
  1. 在推理端严格对齐训练分布：`make_dataset.input_to_tokens` 中，当启用 HwToken 占位符时，同时用 `[MASK]` 替换 target 字符串并清空硬件参数数组，保证生成侧 prompt 与训练 `text_student` 的清洗方式一致。  
  2. 在训练端引入 LoRA+ProtoMix 联合训练：新增 `train_hw_injection_lora.py`，在冻结 TLM-Base 上挂一套 LoRA，并在注入格式的 prompt 上用 LM CE + 硬件分类 + 原型正交（CTR/KD 保留接口默认关闭）联合训练 LoRA 与投影器，使 Base 在新格式下有一条“可适配”的轻量通路。  
  3. 将蒸馏（KD）保留为选项：当需要更严格对齐“新格式 vs 老 Base”的输出分布时，可打开 KD，使用同一 student prompt 上的 Base logits 作为 soft target；目前默认关闭以简化优化难度。  
  4. 在短期内，将 Hw 注入主要作为“新/未见硬件的辅助路径或教学案例”，对已见硬件（如 4090）保留原 target 流；后续视 LoRA+ProtoMix 版本在 `gen_state+TVM` 的实际表现，再决定是否让 HwToken 在已见硬件上承担更大比重。
