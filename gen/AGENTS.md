# Repository Guidelines

## Project Structure & Key Specs
`EdgeTLM_update.md` 描述的新架构是当前主线：冻结的 `TLM-BASE` 配合可插拔的 Gated LoRA 专家与单行路由。历史 HA+HS 双轨方案仍保存在 `MT_MoSLoRA_README.md` 供参考，可对比旧版接口与数据期望。核心 Python 源码集中在 tlm/gen`，未来阶段会新增 `modeling/frozen_base.py`、`modeling/experts/` 与 `training/` 等子模块，并在 `infer/` 内收敛推理管线。数据生成与采集脚本（`make_dataset.py`、`dump_programs.py` 等）位于仓库根目录；生成语料和测量记录则分别位于 `gen_data/` 与 `tlm_dataset/`。实验脚本和一次性流程存放在 `run_*.sh` 与 `scripts/` 下，测试文件保持根目录 `test_*.py` 命名。所有新增 Markdown 方案说明请放入 `EdgeTLM_update_log.md` 维护的迭代列表中。

## Build, Test, and Development Commands
新环境使用 `python3 -m venv .venv && source .venv/bin/activate` 后执行 `python3 -m pip install -r MosLora/requirements.txt`；如需 GPU，请同时安装匹配 CUDA 的 `torch` wheel。数据准备流程遵循：`python3 dump_network_info.py`（导出任务元数据）、`python3 dump_programs.py`（生成测量候选）、`python3 make_dataset.py --for_type FOR_GEN_BEST --target <llvm target> --dataset_path ...`（组装训练/验证集），必要时用 `--tokenizer_path` 指向最新分词模型。单专家训练与评估沿用现有脚本（例如 `bash run_moslora_iterative.sh` 或 `bash run_mt_moslora_iterative.sh`），迁移到 EdgeTLM 架构时请参考 `EdgeTLM_update.md` 中分阶段引入的 `train_single_expert.py`、`infer_multi_experts.py` 示例，并在日志中记录所处阶段编号。Edge 实验的实时日志统一写入 `logs/` 目录，便于复现实验。提交前运行 `python3 -m pytest -q`，必要时附加 `pytest -k <pattern>` 跳过需要真实硬件的重测；大型生成任务可用 `python3 run_iterative_gen_programs.py --dry-run` 先校验配置，完成自测后记得清理临时缓存目录。

## Coding Style & Naming Conventions
Python 统一采用四空格缩进、`snake_case` 函数名、常量大写。新增模块请保持显式导入并在文件头部列出公共常量；如果接口仍在探索，请以 `TODO(edge)` 形式标记后续收敛点。格式化使用 `black`（版本和 extras 固定在 `MosLora/requirements.txt`）并配合 `ruff` 做静态检查（见 `MosLora/peft/Makefile` 目标），必要时再补充 `isort` 保持导入顺序。LoRA/路由相关类命名以 `Gated*` 前缀区分，冻结基座封装统一为 `FrozenBase*`。序列化文件遵循 `adapter_model.bin`、`adapter_config.json`、`router.json`、`metrics.json` 约定，并将元信息写入 `router.json.meta` 字段。

## Testing Guidelines
单元测试建议放在根目录并以 `test_<feature>.py` 命名；新增 EdgeTLM 模块时务必补充针对冻结基座、路由门控和专家注册的单测。使用 `pytest` 运行，重点关注梯度流断言（确保 `FrozenBaseWrapper` 参数始终冻结）、门控数值范围以及序列化往返测试。涉及真实延迟或硬件抽查的集成测试应提供可替代的代理路径并在文档中注明；需要真实硬件的测试请以 `pytest -k` 过滤后执行，并在 PR 描述中告知 Reviewer 无法覆盖的场景。必要时生成 coverage 报告供导师复核，建议在阶段性合并前生成一次 `pytest --maxfail=1 --disable-warnings` 报告并附上覆盖截图，同时附带关键环境变量列表。

## Commit & Pull Request Guidelines
观察历史提交可见“英文/简体中文并存 + 模块前缀”的风格，例如 `make_dataset:`、`训练管线:`，建议沿用以便快速定位。主题句保持 60 字以内，并在正文说明数据格式或模型接口的兼容性风险；如变更路由或专家目录结构，请附差异清单。PR 需包含：目标概述、受影响的 Markdown 设计文档（如 `EdgeTLM_update.md`）、复现命令、关键日志或延迟指标，以及专家产物目录变动说明，必要时附样例 `router.json` 片段。若引入新专家，请明确导出路径、路由/偏置初始化策略与所处阶段编号，方便审查与回滚；合并前同步更新学生需要阅读的文档链接。

## Architecture Notes
EdgeTLM 模块化演进强调：冻结基座 (`FrozenBaseWrapper`)、注册式专家管理 (`ExpertRegistry`)、带负偏置的门控 (`GatedLoRAExpert`)、Top-K 稀疏推理 (`BasePlusExperts.forward_multi`) 以及延迟收益约束 (`compute_gain_loss`)。同时请预留 `latency_cache.py` 与 `proxy_score.py` 的插槽，以便后续接入真机/代理混合的收益评估。落地实现须与文档阶段划分（步骤 1-8）保持一致，并在 PR 中标注当前阶段与后续依赖，以确保团队和学生能按分工接力推进；若暂时跳过某阶段，请在 `EdgeTLM_update_log.md` 记录原因和补齐计划，请保持阶段序列不变。Document updates should reference `EdgeTLM_update_log.md` milestones explicitly.


11.18（正在解决的问题）
 一、当前状况梳理

  1. 目标与场景

  - 问题：base TLM 在新硬件上推理时会遇到 OOV / 硬件描述不准确。
  - 你的方案：
      - 用 make_dataset.py --emit_hw_student 生成带 text_student / hw_emb 的语料；
      - 训练一个 ProtoMixAligner，把 hw_emb 映射到模型 embedding 空间；
      - 在推理时，把 prompt 里的硬件字段替换成一个 [MASK]，用 ProtoMix 产出的向量注入，从而“用 hw_emb 取代原硬件字
        符串”。

  2. 数据来源与结构（已做的改动也一起梳理）

  - 生成：
      - make_dataset.py --for_type=for_gen --emit_hw_student True ...
      - 对 GPU / CPU 各跑一次，再用 cat 合并到
        tlm_dataset/gen/gen_data/align_train_multi_merged/0_merge.json。
  - 每条样本大致长这样（你给的例子）：
      - text：完整张量句子
          - 包含：ComputeDAG 哈希 +形状 + target 字符串 + 硬件参数数组 [-1, 16, 64, 49152, ...] + 调度指令序列。
      - text_student：
          - 原来：只把 target 字符串替换为 [MASK]，但 [-1, 16, 64, ...] 那 8 个硬件参数还在；
          - 现在：我们已经在
              - 生成阶段（make_dataset_utils.json_to_token）清空了 student_struct[1][0][2]，
              - 后处理脚本 gen/postprocess_align_hw_mask.py 里又删除了 text_student 中 [MASK] 后紧跟的一串整数。
          - 效果：当前用于训练的 text_student 已经 不再显式包含 target 字符串与那 8 个硬件参数数字。
      - 额外字段：
          - hw_id, hw_name, hw_emb（来自 hardware_embeddings_v2.json）。
  - 也就是说：现在训练集已经满足“硬件语义主要来自 hw_emb，而不是文本里泄露”的 desiderata。

  3. 训练脚本 gen/train_hw_injection.py 当前逻辑（关键路径）

  - 数据加载与 batch：
      - HwStudentDataset 只读 text_student 与 hw_emb；
      - HwCollator：
          - tokenizer(texts, padding=True, truncation=True, ...) 得到 input_ids / attention_mask；
          - labels = input_ids.clone()（标准 LM 设置）；
          - 同时返回 hw（将 hw_emb 转成 float32 tensor）。
  - 模型结构：
      - 冻结的 AutoModelForCausalLM.from_pretrained(model_path)；
      - ProtoMixAligner(proto_matrix, embed_dim) 只训练这一小块 +（可选）温度；
      - embed_layer = model.get_input_embeddings()。
  - 前向：

    embeds = embed_layer(input_ids)                # [B, L, d]
    hw_mask = (input_ids == hw_token_id)          # [B, L]，找 [MASK] 位置
    hw_embed = aligner(hw_vec)                    # [B, d]，由 hw_emb 生成
    embeds = torch.where(hw_mask.unsqueeze(-1),
                         hw_embed.unsqueeze(1),
                         embeds)                  # 把所有 [MASK] 位置替换为注入向量
    outputs = model(inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    labels=labels)
    loss = outputs.loss / grad_accum_steps
  - 损失：
      - 只有一个 LM 交叉熵 loss，对整条序列的全部 token 平均；
      - 没有额外的分类、对比或延迟头。
  - 现象：
      - loss 在 ~6–8 之间大幅抖动，移动平均稍有下降但不明显；
      - 调小 lr、加 warmup、加 noise 等都无法根治。

  4. 抽象问题

  - 监督目标：希望“hw_emb → 注入向量”真正承载硬件语义，尤其是对新硬件；
  - 实际损失：在强迫“注入后整句 LM 表现不要变差”的同时，并没有专门鼓励“硬件语义对齐”，导致：
      - 梯度主要来自与硬件无关的大量 token；
      - 小小的 ProtoMix 参数要为整句的所有 token 负责，loss 面极度嘈杂；
      - ProtoMix 倾向于学成“尽量别改变原模型”的方向，而不是“强化 hw_emb 语义”。

      ———

  二、逐步改进方案：分阶段、可观察、尽量不一下子改太多

  我建议按照“从最小侵入 → 加轻量对齐任务 → 再引入更强监督”的顺序来做，这样每一步都能单独观察效果，方便你和学生调试/
  对比。

  ———

  ### 阶段 1：局部化 LM CE（只在硬件区域附近算交叉熵）

  目标：

  - 先不引入新的头，只改变 loss 的“作用范围”；
  - 让 ProtoMix 主要为“硬件相关区域”负责，而不是整句。

  设计：

  - 利用当前代码里已有的信息：
      - input_ids、labels、hw_mask 和 hw_token_id。
  - 做法（在训练循环内改，不改 collator）：
      - 对每个 batch：
          - 对每条样本找出第一个 [MASK] 的位置 pos_mask;
          - 定义一个“监督窗口”：[pos_mask+1, pos_mask+K]，或直到句子结束/遇到某个分隔标记；
              - K 可以先取一个中值，比如 64 或 80；
              - 这段通常覆盖 target 区域后面的一些结构 token，与你想注入的硬件语义最相关。
          - 将窗口之外的 label 设为 -100（HuggingFace 的标准“忽略 label”约定）。
              - 这样 outputs.loss 只会对窗口内 token 计算 CE，其余 token 不贡献梯度。
  - 简单伪代码（在 for batch in dataloader 中）：

    # labels = batch["labels"].to(device)  # 原样
    hw_mask = (input_ids == hw_token_id)   # [B, L]
    # 找每行第一个 [MASK] 位置
    first_mask_pos = hw_mask.float().argmax(dim=1)  # 假设每行至少有一个 [MASK]
    window_size = 64
    B, L = labels.size()
    for i in range(B):
        start = int(first_mask_pos[i].item()) + 1
        end = min(start + window_size, L)
        # 把 [0:start) 和 [end:L) 的 label 置为 -100
        if start > 0:
            labels[i, :start] = -100
        if end < L:
            labels[i, end:] = -100
    # 然后喂给 model(labels=labels) 即可
  - 预期效果：
      - loss 数值会下降一些（因为只看一小段），但更关键的是：
          - 梯度主要来源于“硬件附近”的 token；
          - ProtoMix 调整的方向更与“这个区域的预测好坏”相关，而不是被整句牵着跑。

  观察指标：

  - 训练日志：
      - 不只是看平均 loss 变小，还看抖动是否变得更可控（比如 4–8 之间 → 3–6 之间）；
  - 推理侧：
      - 用你既有的 LOHO/插值验证流程，看看局部化 loss 后，hw_aligner 在插值实验上的行为是否更稳定、更可解释。

  ———

  ### 阶段 2：加入一个轻量的“硬件分类头”（辅助任务）

  目标：

  - 在“局部 LM CE”基础上，加一个专门的“硬件语义任务”；
  - 不依赖自然语言，只依赖已有的 hw_id / hw_name。

  设计：

  1. 额外参数：
      - 增加一个线性层：cls_head = nn.Linear(hidden_dim, num_hw_classes)；
      - 只训练 ProtoMixAligner + cls_head，BASE 仍然冻结。
  2. 前向：
      - 已经有了 embeds（注入后的 embedding）和 hw_mask；
      - 再加一步拿 hidden state：

        outputs = model(inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        labels=None,   # 先不让 HF 帮我们算 loss
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-1]      # [B, L, d]
        mask_pos = hw_mask.float().argmax(1)    # [B]
        h_mask = hidden[torch.arange(B), mask_pos]   # [B, d]
        logits_hw = cls_head(h_mask)            # [B, num_hw]
      - 构造硬件标签：
          - 从 dataset 中的 hw_id 建一个小词表：{"v100":0, "4090":1, "xavier":2, "xeon":3}；
          - 在 HwCollator 或 Dataset 中把 hw_id 转成 long 类型的 hw_label。
  3. 损失组合：
      - L_lm_local：阶段 1 的“局部 LM CE”（仍然可用 HF 自动 CE，或自己 CrossEntropyLoss）。
      - L_hw_cls：

        loss_fn = nn.CrossEntropyLoss()
        loss_cls = loss_fn(logits_hw, hw_label)
      - 总损失：
        [
        L = L_{\text{lm_local}} + \lambda_{\text{cls}} \cdot L_{\text{hw_cls}}
        ]
          - 初始可以取 lambda_cls = 0.5 或 1.0，先观察梯度规模。
  4. 预期效果：
      - ProtoMix 被迫学到：“给定 hw_emb，注入之后要让 [MASK] 位置的 hidden 足够区分不同硬件”；
      - 这在几何上会把 4 个硬件在 TLM 内部空间“拉开”，为未来的插值提供一个较清晰的骨架；
      - 新硬件的 hw_emb 会通过 ProtoMix 落在这几个 anchor 之间——这就是你希望的“可插值”语义空间的雏形。

  观察指标：

  - 训练日志：
      - 同时记录 loss_lm、loss_cls、loss_total；
      - 看 loss_cls 是否很快下降，说明 ProtoMix + cls_head 足够区分 4 个硬件；
  - 验证：
      - 在训练集上/hold-out 集上算一个硬件分类的 accuracy（辅助指标）；
      - 在插值实验中，观察“ProtoMix 输出 + [MASK] hidden”在不同硬件之间的几何结构（比如用 PCA 可视化 4 点）。

  ———

  ### 阶段 3：引入对比学习（program–hardware 对齐），进一步强化几何结构

  目标：

  - 不再只关心“离散分类”，而是对齐“程序上下文表示”和“hw_emb 表示”的距离结构；
  - 让空间更像“语义连续空间”，有利于对未见硬件做插值。

  设计（在阶段 2 的基础上追加）：

  1. 表示定义：
      - z_prog：仍然用 [MASK] 的 hidden 或其附近 token 的平均：

        z_prog = h_mask           # [B, d]
      - z_hw：可以直接用 ProtoMix 输出的 hw_embed，或再加一个线性变换：

        z_hw = hw_embed           # [B, d]
  2. InfoNCE 对比损失：
      - 对一个 batch：

        # 归一化
        z_p = F.normalize(z_prog, dim=-1)
        z_h = F.normalize(z_hw, dim=-1)
        logits = (z_p @ z_h.t()) / tau  # [B, B]
        labels = torch.arange(B, device=device)
        loss_contrast = F.cross_entropy(logits, labels)
      - 含义：
          - (i,i) 是正样本；
          - (i,j, i≠j) 是负样本；
          - 让“程序上下文”更接近配对的硬件 embedding，远离其他硬件。
  3. 总损失：
     [
     L = L_{\text{lm_local}} + \lambda_{\text{cls}} L_{\text{hw_cls}} + \lambda_{\text{ctr}} L_{\text{contrast}}
     ]
      - 初始：lambda_cls ≈ 0.5，lambda_ctr ≈ 0.1，先轻量引入对比项。
  4. 预期收益：
      - 让“hw_emb → ProtoMix → 注入向量”与“程序上下文”之间的关联更紧密；
      - 插值时，新硬件的 z_hw_new 不仅在硬件空间位置合理，而且与对应程序的 z_prog 也能形成合乎直觉的对齐。

  ———

  ### 阶段 4（可选，下游性能对齐）：加入延迟相关的辅助头

  这个阶段需要你这边稍微准备一下数据（或者你确认已有数据的字段）。总体思路是：

  - 用 (program, hw_emb, latency) 做一个额外的“性能回归/分类”任务：
      - 在 z_prog 或 concat(z_prog, z_hw) 上接一个小 MLP：
          - 预测 log_latency（回归）；或
          - 将 latency 分成几个 bucket（快/中/慢），做分类。
  - 损失再额外加一项 L_lat，权重可以从小到大试：
    [
    L = L_{\text{lm_local}} + \lambda_{\text{cls}} L_{\text{hw_cls}} +
    \lambda_{\text{ctr}} L_{\text{contrast}} + \lambda_{\text{lat}} L_{\text{latency}}
    ]
  - 好处：
      - 让 ProtoMix 学到的硬件语义不仅区分“硬件是谁”，还对“程序在该硬件上快/慢”有一定感知；
      - 相当于在 embedding 空间里把“性能几何”也 encode 进去，对新硬件插值更有帮助。

  这一步在你觉得前面三步已经有明显收益、并且方便抽取延迟数据时再做即可。

  ———

  三、建议的实施顺序与节奏

  1. 第一步（最小改动，主要改训练脚本）：
      - 在现有 train_hw_injection.py 上实现“局部 CE”（阶段 1）。
      - 跑一版和你刚才那套命令类似的实验，看 loss 曲线 + 插值验证性能。
  2. 第二步（轻量增加分类头）：
      - 加一个硬件分类头，改一下 Dataset/Collator 传 hw_label；
      - 先用 L = L_lm_local + λ_cls L_cls，不加对比损失；
      - 再对比：
          - loss 稳定性；
          - 分类准确率（辅助）；
          - 插值实验日志。
  3. 第三步（对比学习）：
      - 在前两步跑通且你觉得效果有改善之后，再加 InfoNCE 部分，观察是否进一步提升泛化/插值表现。
  4. 延迟相关的阶段，可在你觉得“hw 表示空间已经像样”之后再考虑接入。

   如果你认可这个分阶段方案，我建议先从阶段 1 + 阶段 2做起（局部 CE + 硬件分类头），这样改动相对简单，收益也最直观。
  你确认之后，我就按照这个规划开始改 train_hw_injection.py 和数据管道（尽量兼容你现有的训练脚本和日志格式），并在改完后
  告诉你：

  - 具体改了哪些函数 / 新增了哪些字段；
  - 新增的超参数（窗口大小 K、各个 λ）初始默认值；
  - 如何运行、如何看日志里的新指标。