#!/usr/bin/env bash
# 监控 HwKVAligner 训练目录，一旦出现新的 checkpoint（文件大小稳定），
# 自动调用 gen_state 评测，并记录日志与生成记录数。
# 请根据需要修改下方配置。

set -euo pipefail

# ======== 配置区域 ========
CKPT_DIR="${CKPT_DIR:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_train}"
OUT_DIR="${OUT_DIR:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/hw_kv_aligner_eval}"
MODEL_PATH="${MODEL_PATH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/clm_gen_multi_v1_bucket_stage0/checkpoint-30000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/Model/gen_tokenizer_multi_v1_bucket}"
SKETCH_PATH="${SKETCH_PATH:-/home/hehangshuai/workspace/tlm/tlm_dataset/gen/gen_data/4090_gen_eval_only_bert_new_form/0_merge.json}"
TARGET="${TARGET:-4090}"
KEEP_CNT="${KEEP_CNT:-16}"
EMB_PATH="${EMB_PATH:-/home/hehangshuai/workspace/tlm/gen/Embedding/hardware_embeddings_v4_universe.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GEN_SCRIPT="${GEN_SCRIPT:-gen/gen_state_debug_kv.py}"  # 如需 debug 日志，可改成 gen/gen_state_debug_kv.py
USE_BUCKET="--use_bucket"
USE_HW_KV="--use_hw_kv"

POLL_INTERVAL=10  # 秒
SIZE_STABLE_WAIT=2  # 秒
SUMMARY_TSV="$OUT_DIR/summary.tsv"
PROCESSED_FILE="$OUT_DIR/processed_ckpts.txt"

mkdir -p "$OUT_DIR"

processed_list=()
touch "$SUMMARY_TSV"
touch "$PROCESSED_FILE"

is_processed() {
  local key="$1"
  if grep -Fxq "$key" "$PROCESSED_FILE"; then
    echo "[INFO] key $key already in $PROCESSED_FILE, skip"
    return 0
  fi
  for x in "${processed_list[@]}"; do
    if [[ "$x" == "$key" ]]; then
      echo "[INFO] key $key already processed in memory, skip"
      return 0
    fi
  done
  return 1
}

mark_processed() {
  processed_list+=("$1")
  echo "$1" >> "$PROCESSED_FILE"
  echo "[INFO] recorded processed key -> $PROCESSED_FILE"
}

count_records() {
  local f="$1"
  "$PYTHON_BIN" - "$f" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
count = 0
try:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                count = len(data)
        else:
            for line in f:
                if line.strip():
                    count += 1
except Exception as e:
    print(f"[WARN] count_records failed: {e}")
    count = 0
print(count)
PY
}

echo "Watching CKPT_DIR=$CKPT_DIR, output to $OUT_DIR"

while true; do
  shopt -s nullglob
  ckpts=("$CKPT_DIR"/hw_kv_aligner_step*.pt "$CKPT_DIR"/hw_kv_aligner.pt)
  shopt -u nullglob
  if ((${#ckpts[@]}==0)); then
    sleep "$POLL_INTERVAL"
    continue
  fi
  # 按修改时间排序，取最新
  latest=$(ls -1t "${ckpts[@]}" 2>/dev/null | head -n1 || true)
  if [[ -z "$latest" ]]; then
    sleep "$POLL_INTERVAL"
    continue
  fi
  # 用 mtime+size 作为唯一键，避免覆盖同名 pt 时漏评
  mtime=$(stat -c%Y "$latest" 2>/dev/null || echo 0)
  size=$(stat -c%s "$latest" 2>/dev/null || echo 0)
  key="${latest}:${mtime}:${size}"
  if is_processed "$key"; then
    sleep "$POLL_INTERVAL"
    continue
  fi
  # 检查文件大小是否稳定
  size1=$(stat -c%s "$latest" 2>/dev/null || echo 0)
  sleep "$SIZE_STABLE_WAIT"
  size2=$(stat -c%s "$latest" 2>/dev/null || echo 0)
  if [[ "$size1" != "$size2" || "$size1" == "0" ]]; then
    echo "[INFO] ckpt $latest size not stable ($size1->$size2), skip this round"
    sleep "$POLL_INTERVAL"
    continue
  fi

  ckpt_base=$(basename "$latest" .pt)
  save_path="$OUT_DIR/gen_${TARGET}_${ckpt_base}.json"
  log_file="$OUT_DIR/eval_${ckpt_base}.log"

  echo "[INFO] Evaluating ckpt $latest -> $save_path"
  set +e
  "$PYTHON_BIN" "$GEN_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$save_path" \
    --target "$TARGET" \
    --keep_cnt "$KEEP_CNT" \
    $USE_BUCKET \
    $USE_HW_KV \
    --hw_kv_aligner_path "$latest" \
    --hardware_embedding_path "$EMB_PATH" \
    >"$log_file" 2>&1
  exit_code=$?
  set -e

  count=$(count_records "$save_path")
  ts=$(date +"%Y-%m-%d %H:%M:%S")
  echo -e "${ts}\t${ckpt_base}\t${count}\t${exit_code}" >> "$SUMMARY_TSV"
  echo "[INFO] eval done ckpt=$ckpt_base count=$count exit=$exit_code (log: $log_file)"
  mark_processed "$key"

  sleep "$POLL_INTERVAL"
done
