# Repository Guidelines

## Project Structure & Key Specs
`EdgeTLM_update.md` 描述的新架构是当前主线：冻结的 `TLM-BASE` 配合可插拔的 Gated LoRA 专家与单行路由。历史 HA+HS 双轨方案仍保存在 `MT_MoSLoRA_README.md` 供参考，可对比旧版接口与数据期望。核心 Python 源码集中在 `src/python/tlm`，未来阶段会新增 `modeling/frozen_base.py`、`modeling/experts/` 与 `training/` 等子模块，并在 `infer/` 内收敛推理管线。数据生成与采集脚本（`make_dataset.py`、`dump_programs.py` 等）位于仓库根目录；生成语料和测量记录则分别位于 `gen_data/` 与 `tlm_dataset/`。实验脚本和一次性流程存放在 `run_*.sh` 与 `scripts/` 下，测试文件保持根目录 `test_*.py` 命名。所有新增 Markdown 方案说明请放入 `EdgeTLM_update_log.md` 维护的迭代列表中，方便学生同步，并把教学讲义或会议纪要链接到该文件尾部的开放问题列表。

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
