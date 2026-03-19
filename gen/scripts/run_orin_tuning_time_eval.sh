#!/usr/bin/env bash
set -euo pipefail

# Build prefix-K logs and optionally run tune_relay timing on Orin.
#
# Modes:
#   build  : only generate prefix logs
#   tune   : generate prefix logs + run tune_relay
#   all    : same as tune
#
# Example:
#   bash gen/scripts/run_orin_tuning_time_eval.sh build
#   bash gen/scripts/run_orin_tuning_time_eval.sh tune

MODE="${1:-build}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLM_ROOT="${TLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$TLM_ROOT/tlm_dataset/gen}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart}"

EVAL_ROOT="${EDGE_ORIN_EVAL_ROOT:-$RUN_ROOT/orin/iter00/eval_xavier}"
MEASURE_DIR="${EDGE_ORIN_MEASURE_DIR:-$EVAL_ROOT/measure}"

OFFICIAL_MEASURED="${EDGE_OFFICIAL_MEASURED:-$MEASURE_DIR/official_measured.json}"
LORA_MEASURED="${EDGE_LORA_MEASURED:-$MEASURE_DIR/kv_lora_xavier_measured.json}"

ANSOR_SUMMARY=""
if [[ -n "${EDGE_ANSOR_SUMMARY_CSV:-}" ]]; then
  ANSOR_SUMMARY="$EDGE_ANSOR_SUMMARY_CSV"
else
  CAND_1="$MEASURE_DIR/ansor/ansor_summary_orin.csv"
  CAND_2="$DATA_ROOT/gen_data/Ansor_baseline/ansor_summary_orin.csv"
  if [[ -f "$CAND_1" ]]; then
    ANSOR_SUMMARY="$CAND_1"
  elif [[ -f "$CAND_2" ]]; then
    ANSOR_SUMMARY="$CAND_2"
  else
    # Keep a deterministic fallback path in logs even when file is not ready yet.
    ANSOR_SUMMARY="$CAND_2"
  fi
fi

OUT_ROOT="${EDGE_TUNING_TIME_OUT_ROOT:-$MEASURE_DIR/tuning_time_eval}"

NETWORKS="${EDGE_TUNING_TIME_NETWORKS:-bert_base_1x128,resnet_50_1x3x224x224,mobilenet_v2_1x3x224x224,inception_v3_1x3x299x299}"
K_VALUES="${EDGE_TUNING_TIME_K_VALUES:-1-64,80,96,112,128,160,192,256,384,512,768,1000}"
METHODS="${EDGE_TUNING_TIME_METHODS:-official,xavier,ansor}"

TARGET_ORIN="${EDGE_TARGET_ORIN:-cuda -keys=cuda,gpu -arch=sm_87 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32}"
BACKEND="${EDGE_BACKEND:-graph}"
CUDA_ID="${EDGE_CUDA:-0}"
PY_BIN="${EDGE_PYTHON_BIN:-python}"
FORCE="${EDGE_FORCE:-0}"

TIMEOUT_SEC="${EDGE_TUNING_TIME_TIMEOUT_SEC:-3600}"
NUMBER="${EDGE_TUNING_TIME_NUMBER:-1}"
REPEAT="${EDGE_TUNING_TIME_REPEAT:-10}"
MIN_REPEAT_MS="${EDGE_TUNING_TIME_MIN_REPEAT_MS:-100}"
FALLBACK_TOPI_ON_FAIL="${EDGE_TUNING_TIME_FALLBACK_TOPI_ON_FAIL:-0}"
TLM_MAX_K="${EDGE_TUNING_TIME_TLM_MAX_K:-64}"
EMPTY_TOPI_METHODS="${EDGE_TUNING_TIME_EMPTY_TOPI_METHODS:-official}"

CMD=(
  "$PY_BIN" "$TLM_ROOT/gen/scripts/run_orin_tuning_time_eval.py"
  --official-measured "$OFFICIAL_MEASURED"
  --lora-measured "$LORA_MEASURED"
  --ansor-summary "$ANSOR_SUMMARY"
  --ansor-target "${EDGE_ANSOR_TARGET:-orin}"
  --networks "$NETWORKS"
  --k-values "$K_VALUES"
  --methods "$METHODS"
  --out-root "$OUT_ROOT"
  --target "$TARGET_ORIN"
  --backend "$BACKEND"
  --cuda-visible-devices "$CUDA_ID"
  --python-bin "$PY_BIN"
  --number "$NUMBER"
  --repeat "$REPEAT"
  --min-repeat-ms "$MIN_REPEAT_MS"
  --timeout-sec "$TIMEOUT_SEC"
  --tlm-max-k "$TLM_MAX_K"
  --empty-topi-methods "$EMPTY_TOPI_METHODS"
)

if [[ ! -f "$OFFICIAL_MEASURED" ]]; then
  echo "official measured log not found: $OFFICIAL_MEASURED"
  exit 1
fi
if [[ ! -f "$LORA_MEASURED" ]]; then
  echo "xavier measured log not found: $LORA_MEASURED"
  exit 1
fi
if [[ ! -f "$ANSOR_SUMMARY" ]]; then
  echo "ansor summary not found: $ANSOR_SUMMARY"
  echo "Set EDGE_ANSOR_SUMMARY_CSV explicitly if your summary is elsewhere."
  exit 1
fi

if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi

case "$MODE" in
  build)
    CMD+=(--build-prefix-only)
    ;;
  tune|all)
    CMD+=(--run-tune)
    if [[ "$FALLBACK_TOPI_ON_FAIL" == "1" ]]; then
      CMD+=(--fallback-topi-on-fail)
    fi
    ;;
  *)
    echo "Unsupported mode: $MODE"
    echo "Usage: bash gen/scripts/run_orin_tuning_time_eval.sh [build|tune|all]"
    exit 1
    ;;
esac

"${CMD[@]}"

echo "[DONE] out_root=$OUT_ROOT"
echo "  - prefix index: $OUT_ROOT/prefix_logs_index.csv"
echo "  - tune result : $OUT_ROOT/tuning_time_results.csv (when mode=tune/all)"
