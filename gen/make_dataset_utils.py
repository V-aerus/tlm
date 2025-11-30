from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Tuple
import copy
import os


def json_dfs_without_bracket(json, text_list: list):
    if isinstance(json, dict):
        json = dict(sorted(json.items()))
        for key, val in json.items():
            text_list.append(key)
            json_dfs_without_bracket(val, text_list)
    elif isinstance(json, (list, tuple)):
        for it in json:
            json_dfs_without_bracket(it, text_list)
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def json_dfs_with_bracket(json, text_list: list):
    if isinstance(json, dict):
        text_list.append("{")
        json = dict(sorted(json.items()))
        for idx, (key, val) in enumerate(json.items()):
            if idx != 0:
                text_list.append(",")
            text_list.append(key)
            json_dfs_with_bracket(val, text_list)
        text_list.append("}")
    elif isinstance(json, list):
        text_list.append("[")
        for idx, it in enumerate(json):
            if idx != 0:
                text_list.append(",")
            json_dfs_with_bracket(it, text_list)
        text_list.append("]")
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def detect_hardware_from_target(target_str: str) -> Tuple[str, str]:
    """Best-effort mapping from target string to (hw_id, hardware_name)."""
    lower = target_str.lower()
    if "sm_70" in lower or "v100" in lower:
        return "v100", "nvidia/nvidia-v100"
    if "sm_86" in lower or "4090" in lower or "a40" in lower:
        return "4090", "nvidia/nvidia-a40"
    if "xavier" in lower or "sm_72" in lower or "jetson" in lower:
        return "xavier", "nvidia/jetson-agx-xavier"
    if "llvm" in lower or "xeon" in lower or "skylake" in lower:
        return "xeon", "aws/cpu/c5.18xlarge"
    return "unknown", "unknown"


def json_to_token(
    json_lines: List[Dict[str, Any]],
    hw_token_placeholder: Optional[str] = None,
    hardware_embeddings: Optional[Dict[str, List[float]]] = None,
    emit_hw_student: bool = False,
):
    token_list = []
    should_emit_student = emit_hw_student and hw_token_placeholder and hardware_embeddings
    for json_line in json_lines:
        teacher_struct = copy.deepcopy(json_line["text"])
        text_list = []
        json_dfs_without_bracket(teacher_struct, text_list)
        json_line["text"] = " ".join(text_list)

        if should_emit_student:
            try:
                # task info 的第二个字段是 target 字符串
                target_str = teacher_struct[1][0][1]
            except Exception:
                target_str = ""

            student_struct = copy.deepcopy(teacher_struct)
            try:
                student_struct[1][0][1] = hw_token_placeholder
            except Exception:
                # 如果结构与预期不符，则跳过 student 文本
                student_struct = None

            if student_struct is not None and target_str:
                student_text_tokens = []
                json_dfs_without_bracket(student_struct, student_text_tokens)
                json_line["text_student"] = " ".join(student_text_tokens)

                hw_id, hw_name = detect_hardware_from_target(target_str)
                if hw_name in hardware_embeddings:
                    json_line["hw_id"] = hw_id
                    json_line["hw_name"] = hw_name
                    json_line["hw_emb"] = hardware_embeddings[hw_name]
        token_list.append(json_line)
    return token_list


def make_dataset(file, dataset_path, tokenizer_path, for_clm_or_mlm, valid_percentage=5):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )
    if valid_percentage > 0:
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{valid_percentage}%]",
            keep_in_memory=True
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{valid_percentage}%:]",
            keep_in_memory=True
        )
    else:
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train",
            keep_in_memory=True
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    column_names = list(raw_datasets["train"].features)
    if for_clm_or_mlm == "clm":
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            output["labels"] = output["input_ids"].copy()
            del output["token_type_ids"]
            return output
    elif for_clm_or_mlm == "mlm":
        column_names.remove("labels")
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            return output
    else:
        assert(False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)


def make_dataset_test(file, dataset_path, tokenizer_path, for_clm_or_mlm):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if for_clm_or_mlm == "clm":
        def tokenize_function(examples):
            output = tokenizer(examples["text"])
            input_ids = output["input_ids"]
            for inp in input_ids:
                del inp[-1]
            attention_mask = output["attention_mask"]
            for mask in attention_mask:
                del mask[-1]
            del output["token_type_ids"]
            return output
    elif for_clm_or_mlm == "mlm":
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            return output
    else:
        assert(False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)


def filter_files_with_regex(files: List[str], pattern: Optional[str]) -> List[str]:
    """Filter file list by regex on basename if pattern provided."""
    if not pattern:
        return files
    import re

    regex = re.compile(pattern)
    return [f for f in files if regex.search(os.path.basename(f))]
