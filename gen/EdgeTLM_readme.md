# EdgeTLM 快速手册

## `prepare_edge_dataset.py` 使用示例

```bash
# 以 V100 硬件为例，将 all_gen_best_multi 子集转换为带硬件向量的 JSONL
python prepare_edge_dataset.py \
  --sft-dataset-path /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/all_gen_best_multi \
  --output-jsonl /home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/sft_dataset_mutil_v1/v100_gen_best_multi/edge_sft_v100.jsonl \
  --hardware-id v100 \
  --allow-missing-lora
```

执行后脚本会：
- 读取指定的 HuggingFace 数据集（默认为 `train` 分片）。
- 解析 `line` 字段推断硬件 ID，并加载 `Embedding/hardware_embeddings_v2.json` 中的对应向量。
- 写出一行一个样本的 JSONL；每行包含 `text`、`hw_emb`、`lat_base_star` 等字段，便于 `train_edge_expert.py` 直接读取。

若需要导出全部硬件，可以移除 `--hardware-id`，输出路径自行指定；如需同时导出多种硬件，使用逗号分隔（例如 `--hardware-id v100,4090`）。

## 主要参数说明

- `--sft-dataset-path`：指向 HuggingFace `Dataset.load_from_disk` 产物的目录，脚本会自动读取其中的 `train` 分片。
- `--output-jsonl`：输出文件路径（不存在的目录会自动创建）。
- `--hardware-id`：硬件过滤器，可省略（导出全部），或用逗号指定多个候选。
- `--embedding-json`：硬件向量表，默认 `Embedding/hardware_embeddings_v2.json`；需要扩展硬件时可自定义。
- `--allow-missing-lora`：允许 `lat_lora_star` 为空；适用于尚未完成 LoRA 真机测量的阶段，后续补齐延迟后可重新导出。
- `--tokenizer-path`：当数据集中缺少 `text` 字段时，用于解码 `input_ids`；当前 SFT 数据已经包含 `text`，通常无需设置。

脚本内部还提供了 `detect_hardware()` 助手，用于从编译目标字符串中自动识别硬件类型；如需新增硬件，优先在该函数与硬件 embedding 表中补充映射。

