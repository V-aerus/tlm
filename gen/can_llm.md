问题：做 DL 模型的性能预测（时延/内存/功耗等），传统 GNN 方法对新硬件/新网络适应慢、要大量标注重训。论文尝试把 GNN（结构表征）+ LLM（泛化与小样本适配） 结合起来，既保留图结构信息，又提升跨硬件/架构的迁移能力。

ICLR'25 CAN LLMS ENHANCE PERFORMANCE PREDICTION
FOR DEEP LEARNING MODELS?

核心做法：

用 GNN 编码 DL 计算图 得到一个图级 embedding；

通过 Projection 把这段 embedding 投入 LLM 的输入序列里，作为一个单一“GraphToken”向量，配合文本 prompt（用特殊标记 <graph> … </graph> 包围）；

训练上用 Soft Prompt + LoRA 对 LLM 做高效适配，并且让梯度从 LLM 回流到 GNN（在最终阶段）。

11825_Can_LLMs_Enhance_Perform

 

11825_Can_LLMs_Enhance_Perform

三阶段训练：
Stage-1：用 GraphMAE 式自监督预训练 GNN；
Stage-2：图-文本适配（冻结 GNN，只训投影层与 LLM 的 LoRA/soft prompt），让 LLM 学会“理解”图嵌入；
Stage-3：联合微调做最终性能预测（此时梯度贯通）。

11825_Can_LLMs_Enhance_Perform

结论数据：相对 JSON/高层文本描述，单一图嵌入 → LLM 的方式在准确率与训练效率上更优；相较 SoTA GNN，GNN+LLM 方案在多平台数据集上更准，且少样本适配新硬件时优势更明显（提升 30–70 个百分点/平台实验）。

11825_Can_LLMs_Enhance_Perform

 

11825_Can_LLMs_Enhance_Perform

 

11825_Can_LLMs_Enhance_Perform

关键启发：把复杂结构浓缩为连续向量投进 LLM，并通过专门的投影/软提示/LoRA教会 LLM “读懂这段连续条件”；同时保持梯度通路与分阶段训练来降低适配新域的成本。

11825_Can_LLMs_Enhance_Perform
