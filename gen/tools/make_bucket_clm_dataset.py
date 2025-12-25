import argparse
import json
import re

BUCKET_TOKENS = {
    "hpc_gpu": "[HW_GPU_HPC]",
    "edge_gpu": "[HW_GPU_EDGE]",
    "cpu_x86": "[HW_CPU_X86]",
    "cpu_arm": "[HW_CPU_ARM]",
}


def classify_and_replace(text: str) -> str:
    """
    根据硬件模式替换 -arch= / -mcpu= 字段为 bucket token。
    规则（与 bucket 初始化一致）：
      - HPC GPU: 包含 cuda 且 arch=sm_70 / sm_86
      - EDGE GPU: 包含 cuda 且 arch=sm_72
      - CPU X86:  包含 llvm 且 mcpu=skylake-avx512
      - CPU ARM:  包含 llvm 且 mcpu=carmel
    """
    new_text = text
    # 先处理 GPU 相关（可多次替换，不早退）
    if "cuda" in new_text:
        if "-arch=sm_72" in new_text:
            new_text = new_text.replace("-arch=sm_72", BUCKET_TOKENS["edge_gpu"])
        if "-arch=sm_70" in new_text or "-arch=sm_86" in new_text:
            new_text = new_text.replace("-arch=sm_70", BUCKET_TOKENS["hpc_gpu"])
            new_text = new_text.replace("-arch=sm_86", BUCKET_TOKENS["hpc_gpu"])
    # 再处理 CPU/host 相关
    if "llvm" in new_text:
        if "-mcpu=skylake-avx512" in new_text:
            new_text = new_text.replace("-mcpu=skylake-avx512", BUCKET_TOKENS["cpu_x86"])
        if "-mcpu=carmel" in new_text:
            new_text = new_text.replace("-mcpu=carmel", BUCKET_TOKENS["cpu_arm"])
    return new_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="输入 JSONL，比如 0_merge_4090_canonical.json")
    parser.add_argument("--out_path", required=True, help="输出 bucket 化 JSONL")
    args = parser.parse_args()

    total = 0
    patched = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(
        args.out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            text = obj.get("text", "")
            new_text = classify_and_replace(text)
            if new_text != text:
                patched += 1
            fout.write(json.dumps({"text": new_text}, ensure_ascii=False) + "\n")

    print(f"Total lines: {total}")
    print(f"Patched lines: {patched}")


if __name__ == "__main__":
    main()
