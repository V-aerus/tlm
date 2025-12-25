"""
用于将各个硬件的 eval 草图处理成 bucket 格式：
- 读入 auto_scheduler 记录版 JSONL（每行含 "i"/"r"/"v"）
- 按硬件字段替换 target 中的 -arch / -mcpu 为 bucket token
- 输出同样的 JSONL 结构，仅 target 字符串被替换
"""
import argparse
import json
from typing import Tuple

# bucket token 映射
BUCKET_TOKENS = {
    "hpc_gpu": "[HW_GPU_HPC]",
    "edge_gpu": "[HW_GPU_EDGE]",
    "cpu_x86": "[HW_CPU_X86]",
    "cpu_arm": "[HW_CPU_ARM]",
}


def bucketize_target(target: str) -> Tuple[str, bool]:
    """
    按规则替换 target 字符串里的硬件字段。
    支持同一行既有 cuda 又有 llvm（例如 Jetson host+device）。
    返回 (new_target, changed?)
    """
    new_target = target
    changed = False
    # GPU 部分
    if "cuda" in new_target:
        if "-arch=sm_72" in new_target:
            new_target = new_target.replace("-arch=sm_72", BUCKET_TOKENS["edge_gpu"])
            changed = True
        if "-arch=sm_70" in new_target or "-arch=sm_86" in new_target:
            if "-arch=sm_70" in new_target:
                new_target = new_target.replace("-arch=sm_70", BUCKET_TOKENS["hpc_gpu"])
                changed = True
            if "-arch=sm_86" in new_target:
                new_target = new_target.replace("-arch=sm_86", BUCKET_TOKENS["hpc_gpu"])
                changed = True
    # CPU/host 部分
    if "llvm" in new_target:
        if "-mcpu=skylake-avx512" in new_target:
            new_target = new_target.replace("-mcpu=skylake-avx512", BUCKET_TOKENS["cpu_x86"])
            changed = True
        if "-mcpu=carmel" in new_target:
            new_target = new_target.replace("-mcpu=carmel", BUCKET_TOKENS["cpu_arm"])
            changed = True
    return new_target, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="输入 eval 草图 JSONL（auto_scheduler 记录版）")
    parser.add_argument("--out_path", required=True, help="输出 bucket 化后的 JSONL")
    args = parser.parse_args()

    total = 0
    patched = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            try:
                target = obj["i"][0][1]
            except Exception:
                # 不符合预期结构，原样写出
                fout.write(line)
                continue
            new_target, changed = bucketize_target(target)
            obj["i"][0][1] = new_target
            if changed:
                patched += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total lines: {total}")
    print(f"Patched targets: {patched}")


if __name__ == "__main__":
    main()
