#!/usr/bin/env bash
set -euo pipefail

# Orin-only Ansor baseline collection helper.
# Purpose: tune standard eval networks on Orin with explicit device+host target.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLM_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TLM_ROOT="${TLM_ROOT:-$TLM_ROOT_DEFAULT}"
DATA_ROOT="${DATA_ROOT:-$TLM_ROOT/tlm_dataset/gen}"

TARGET_ORIN="${TARGET_ORIN:-cuda -keys=cuda,gpu -arch=sm_87 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32}"
HOST_TARGET_ORIN="${HOST_TARGET_ORIN:-llvm -keys=arm_cpu,cpu -mcpu=cortex-a78 -mtriple=aarch64-linux-gnu -num-cores=12}"

NETWORKS="${EDGE_ANSOR_NETWORKS:-bert_base,resnet_50,mobilenet_v2,inception_v3}"
BUDGETS="${EDGE_ANSOR_BUDGETS:-1,32,64,1000}"
OUT_ROOT="${EDGE_ANSOR_OUT_ROOT:-$DATA_ROOT/gen_data/Ansor_baseline/ansor}"
SUMMARY_CSV="${EDGE_ANSOR_SUMMARY_CSV:-$DATA_ROOT/gen_data/Ansor_baseline/ansor_summary_orin.csv}"

MAX_TASKS="${EDGE_ANSOR_MAX_TASKS:-0}"
ONE_SHAPE="${EDGE_ANSOR_ONE_SHAPE:-1}"
NUMBER="${EDGE_ANSOR_NUMBER:-1}"
REPEAT="${EDGE_ANSOR_REPEAT:-10}"
MIN_REPEAT_MS="${EDGE_ANSOR_MIN_REPEAT_MS:-100}"
TIMEOUT="${EDGE_ANSOR_TIMEOUT:-10}"
VERBOSE="${EDGE_ANSOR_VERBOSE:-1}"
FORCE="${EDGE_ANSOR_FORCE:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash gen/scripts/run_ansor_baseline_orin.sh"
  echo "Env overrides:"
  echo "  TLM_ROOT / DATA_ROOT"
  echo "  TARGET_ORIN / HOST_TARGET_ORIN"
  echo "  EDGE_ANSOR_NETWORKS (default: bert_base,resnet_50,mobilenet_v2,inception_v3)"
  echo "  EDGE_ANSOR_BUDGETS (default: 1,32,64,1000)"
  echo "  EDGE_ANSOR_OUT_ROOT / EDGE_ANSOR_SUMMARY_CSV"
  echo "  EDGE_ANSOR_MAX_TASKS / EDGE_ANSOR_ONE_SHAPE / EDGE_ANSOR_FORCE"
  exit 0
fi

echo "[CONFIG] TLM_ROOT=$TLM_ROOT"
echo "[CONFIG] DATA_ROOT=$DATA_ROOT"
echo "[CONFIG] target(device)=$TARGET_ORIN"
echo "[CONFIG] target(host)=$HOST_TARGET_ORIN"
echo "[CONFIG] networks=$NETWORKS"
echo "[CONFIG] budgets=$BUDGETS"
echo "[CONFIG] out_root=$OUT_ROOT"
echo "[CONFIG] summary_csv=$SUMMARY_CSV"

CMD=(
  python "$TLM_ROOT/gen/run_ansor_baseline.py"
  --target "$TARGET_ORIN"
  --host_target "$HOST_TARGET_ORIN"
  --network_names "$NETWORKS"
  --budgets "$BUDGETS"
  --out_root "$OUT_ROOT"
  --summary_csv "$SUMMARY_CSV"
  --max_tasks "$MAX_TASKS"
  --number "$NUMBER"
  --repeat "$REPEAT"
  --min_repeat_ms "$MIN_REPEAT_MS"
  --timeout "$TIMEOUT"
  --verbose "$VERBOSE"
)

if [[ "$ONE_SHAPE" == "1" ]]; then
  CMD+=(--one_shape_per_network)
fi
if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi

"${CMD[@]}"
