I am currently modifying the project for my Chinese students. Please answer questions with me in Simplified Chinese
        throughout the process. The main directory of this project runs under/home/hehangshuai/workspace/tlm/gen. Please only
        develop and submit on the branch features/edge expert set throughout the process, https://github.com/V-aerus/tlm/
        tree/feature/edge-expert-setup


        这是一个关于跨硬件张量优化的项目。请你阅读我们的项目readme文件，对我们的项目做一个了解。这个项目基于OSDI'24年的论文，Enabling Tensor
    Language Model to Assist in Generating {High-Performance} Tensor Programs for Deep Learning。这个项目发源于此，关于项目的readme在这：/home/
    hehangshuai/workspace/tlm/gen/EdgeTLM_readme.md，以及 EdgeTLM_update.md（你需要重点阅读，特别是 1000 行以后的，项目目前的最新进度）。你需要
    了解到的一点是，这个项目最主要的目标是解决这样一个问题：解决TLM仅仅 针 对单个硬件进行张量优化的问题，尝试通过我们的系统让TLM可以为陌生的
    （未学习的）的硬件也实现较优的优化效果。我们项目的 代 码就是刚才发给你的那个github分支。论文也附录了，请你结合阅读理解。/home/hehangshuai/
    workspace/tlm/gen/Zhai 等 - Enabling Tensor Language Model to Assist in Genera.pdf。我们的 github 项目在https://github.com/V-aerus/tlm/
    tree/feature/edge-expert-setup，你可以对照当前代码架构以理解我们的项目，具体任务我们稍后再谈。


     其实整个系统都已经打通了,目前正处在迭代测试环节（修正 bug，例如 emb 设计，aligner 训练等等）,请你阅读 /gen/scripts 目录下的这些脚本,还有我们的产物,/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart/,我想要你评估一下当前整体系统的情况,和我汇报一下.


很好，其实目前，两边的系统都已经打通。（我说的两边意思是，bucket+kv 为一边，lora routing 为一边）接下来，我想要完成整个端到端的系统。具体来说，就是 bucket+kv（已经完成）+ lora routing 以及 lora 的 sft 训练和推理时复用多个 lora（或者将多个 lora 合成一个新的 lora（究竟是多个 lora 还是合成一个新 lora，还不确定）（大体上已经完成）（lora 的 sft 训练还存在 lat_base 和 lat_lora没有对齐的问题），但是还没有串联起来，这个端到端的系统还没有稳定运行。我现在想要打通这整个系统，但我不确定从何做起，我想要你先帮我检查整个系统，这是我的项目https://github.com/V-aerus/tlm/tree/feature/edge-expert-setup，我不确定要给你提供哪些代码，你或许可以向我要求一下，我会将对应的 lora routing 这一部分的代码发送给你，然后请你帮我进行指挥，让 codex 进行具体工作。这是一部分的 lora 相关的代码，请你先阅读，然后帮我看看，我下一步该怎么做，该给你提供哪些代码。md 文件里是对我们项目的目标的回顾。除此以外，EdgeTLM_readme.md 中似乎也包含了一些关于这个阶段任务的描述。供你参考。（router.json包含的是 lora 内的内容）
