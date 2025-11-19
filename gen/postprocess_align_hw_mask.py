#!/usr/bin/env python3
import json
import os
import re
from typing import Iterable


def remove_hw_params_from_text_student(
    text_student: str,
    hw_token: str = "[MASK]",
    max_params: int = 8,
    min_params: int = 4,
) -> str:
    tokens = text_student.split()
    if not tokens:
        return text_student

    try:
        idx = tokens.index(hw_token)
    except ValueError:
        return text_student

    int_pattern = re.compile(r"^-?\d+$")
    j = idx + 1
    removed = 0
    while j < len(tokens) and removed < max_params and int_pattern.match(tokens[j]):
        j += 1
        removed += 1

    if removed < min_params:
        return text_student

    new_tokens = tokens[: idx + 1] + tokens[idx + 1 + removed :]
    return " ".join(new_tokens)


def process_file(path: str, backup: bool = True) -> None:
    if backup:
        backup_path = path + ".backup"
        if not os.path.exists(backup_path):
            os.replace(path, backup_path)
            src_path = backup_path
        else:
            src_path = backup_path
    else:
        src_path = path

    tmp_path = path + ".tmp"
    with open(src_path, "r", encoding="utf-8") as fin, open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text_student = obj.get("text_student")
            if isinstance(text_student, str):
                obj["text_student"] = remove_hw_params_from_text_student(text_student)
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")

    os.replace(tmp_path, path)


def main() -> None:
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    default_path = os.path.join(
        base_dir,
        "tlm_dataset",
        "gen",
        "gen_data",
        "align_train_multi_merged",
        "0_merge.json",
    )
    target_path = os.environ.get("ALIGN_HW_MASK_PATH", default_path)
    process_file(target_path, backup=True)


if __name__ == "__main__":
    main()

