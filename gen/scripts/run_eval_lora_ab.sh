#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash gen/scripts/run_eval_lora_ab.sh <hw> <expert_a_dir> <expert_b_dir>"
  echo "  hw: 2080|2080ti|3090|4090|v100"
  echo "Env:"
  echo "  EDGE_LABEL_A=stacked_old   (default: basename of expert_a_dir)"
  echo "  EDGE_LABEL_B=fresh_new     (default: basename of expert_b_dir)"
  echo "  EDGE_EVAL_NETWORKS=bert_base,resnet_50,mobilenet_v2  (default)"
  echo "  EDGE_SKIP_SKETCH=0"
  echo "  EDGE_SKIP_GEN=0"
  echo "  EDGE_SKIP_MEASURE=0"
  echo "  EDGE_SKIP_COMPARE=0"
  echo "  EDGE_SKIP_OFFICIAL=1       (default: skip)"
  echo "  EDGE_BASELINE_MODE=base_kv (base_kv|bucket_only)"
  echo "  EDGE_SKIP_BASELINE=0       (1 to skip baseline gen/measure)"
  echo "  EDGE_USE_HW_KV=1           (default: on)"
  echo "  EDGE_FORCE=0               (default: no overwrite)"
  echo "  EDGE_CUDA=2                (single GPU id; optional)"
  echo "  EDGE_CUDA_CANDIDATES=1,2,3,0 (used when EDGE_CUDA is unset)"
  echo "  EDGE_KEEP_CNT=64"
  echo "  EDGE_MEASURE_BATCH=64"
  echo "  EDGE_MATCH_MODE=i"
  echo "  EDGE_BASELINE_LABEL=<label> (default: EDGE_LABEL_A)"
  echo "  EDGE_COMPARE_MODE=intersection"
  echo "  EDGE_EMB_PATH=/path/to/hardware_embeddings_v5_draft.json"
  echo "  EDGE_HW_KV_ALIGNER=/path/to/hw_kv_aligner.pt"
  echo "  EDGE_HW_KV_PREPROCESS_JSON=/path/to/preprocess.json (optional)"
  exit 1
fi

HW_RAW="$1"
EXPERT_A_DIR="$2"
EXPERT_B_DIR="$3"

for d in "$EXPERT_A_DIR" "$EXPERT_B_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "expert dir not found: $d"
    exit 1
  fi
  if [[ ! -f "$d/router.json" ]]; then
    echo "router.json not found in expert dir: $d"
    exit 1
  fi
done

sanitize_label() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s//,/__}"
  s="${s//[^a-zA-Z0-9._-]/_}"
  echo "$s"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLM_ROOT="${TLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$TLM_ROOT/tlm_dataset/gen}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/gen_data/edge_runs/2026-01-04_bucketkv_lora_restart}"

PATHS_SH="${EDGE_PATHS_SH:-$RUN_ROOT/_shared/paths.sh}"
if [[ ! -f "$PATHS_SH" ]]; then
  echo "paths.sh not found: $PATHS_SH"
  exit 1
fi
# shellcheck disable=SC1090
source "$PATHS_SH"

if [[ -z "${EDGE_EMB_PATH:-}" ]]; then
  if [[ -n "${HW_EMB_V5:-}" && -f "${HW_EMB_V5}" ]]; then
    EDGE_EMB_PATH="$HW_EMB_V5"
  elif [[ -f "$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json" ]]; then
    EDGE_EMB_PATH="$TLM_ROOT/gen/Embedding/hardware_embeddings_v5_draft.json"
  else
    EDGE_EMB_PATH="${HW_EMB_V4U:-$HW_EMB_V4}"
  fi
fi

EDGE_HW_KV_ALIGNER="${EDGE_HW_KV_ALIGNER:-${HW_KV_ALIGNER:-}}"
EDGE_HW_KV_PREPROCESS_JSON="${EDGE_HW_KV_PREPROCESS_JSON:-}"
USE_HW_KV="${EDGE_USE_HW_KV:-1}"
SKIP_SKETCH="${EDGE_SKIP_SKETCH:-0}"
SKIP_GEN="${EDGE_SKIP_GEN:-0}"
SKIP_MEASURE="${EDGE_SKIP_MEASURE:-0}"
SKIP_COMPARE="${EDGE_SKIP_COMPARE:-0}"
SKIP_OFFICIAL="${EDGE_SKIP_OFFICIAL:-1}"
SKIP_BASELINE="${EDGE_SKIP_BASELINE:-${EDGE_SKIP_BUCKET_ONLY:-0}}"
BASELINE_MODE="${EDGE_BASELINE_MODE:-base_kv}"
FORCE="${EDGE_FORCE:-0}"
KEEP_CNT="${EDGE_KEEP_CNT:-64}"
MEASURE_BATCH="${EDGE_MEASURE_BATCH:-64}"
MATCH_MODE="${EDGE_MATCH_MODE:-i}"
COMPARE_MODE="${EDGE_COMPARE_MODE:-intersection}"
NETWORKS="${EDGE_EVAL_NETWORKS:-bert_base,resnet_50,mobilenet_v2}"

pick_cuda_id() {
  if [[ -n "${EDGE_CUDA:-}" ]]; then
    echo "$EDGE_CUDA"
    return
  fi
  local candidates="${EDGE_CUDA_CANDIDATES:-1,2,3,0}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi
  local available
  available="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$available" ]]; then
    echo "0"
    return
  fi
  IFS=',' read -r -a cand_arr <<< "$candidates"
  for cand in "${cand_arr[@]}"; do
    cand="${cand// /}"
    if grep -qx "$cand" <<< "$available"; then
      echo "$cand"
      return
    fi
  done
  echo "$available" | head -n 1
}

CUDA_ID="$(pick_cuda_id)"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_ID (EDGE_CUDA=${EDGE_CUDA:-<auto>}, EDGE_CUDA_CANDIDATES=${EDGE_CUDA_CANDIDATES:-1,2,3,0})"

if [[ ! -f "$EDGE_EMB_PATH" ]]; then
  echo "embedding file not found: $EDGE_EMB_PATH"
  exit 1
fi
if [[ "$USE_HW_KV" == "1" ]]; then
  if [[ -z "$EDGE_HW_KV_ALIGNER" || ! -f "$EDGE_HW_KV_ALIGNER" ]]; then
    echo "EDGE_USE_HW_KV=1 but aligner not found: $EDGE_HW_KV_ALIGNER"
    exit 1
  fi
  if [[ -n "$EDGE_HW_KV_PREPROCESS_JSON" && ! -f "$EDGE_HW_KV_PREPROCESS_JSON" ]]; then
    echo "hw_kv preprocess json not found: $EDGE_HW_KV_PREPROCESS_JSON"
    exit 1
  fi
fi
if [[ "$BASELINE_MODE" == "base_kv" && "$USE_HW_KV" != "1" ]]; then
  echo "EDGE_BASELINE_MODE=base_kv requires EDGE_USE_HW_KV=1."
  exit 1
fi
if [[ "$BASELINE_MODE" != "base_kv" && "$BASELINE_MODE" != "bucket_only" ]]; then
  echo "Invalid EDGE_BASELINE_MODE: $BASELINE_MODE (expect base_kv|bucket_only)"
  exit 1
fi

HW="${HW_RAW,,}"
TARGET_2080_DEFAULT="cuda -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32"
if [[ "$HW" == "3090" ]]; then
  TARGET="$TARGET_4090"
  HW_ID="3090"
  HW="3090"
elif [[ "$HW" == "4090" ]]; then
  TARGET="$TARGET_4090"
  HW_ID="4090"
  HW="4090"
elif [[ "$HW" == "v100" ]]; then
  TARGET="$TARGET_V100"
  HW_ID="v100"
  HW="v100"
elif [[ "$HW" == "2080" || "$HW" == "2080ti" ]]; then
  TARGET="${TARGET_2080:-$TARGET_2080_DEFAULT}"
  HW_ID="2080"
  HW="2080"
else
  echo "Invalid hw: $HW_RAW (expect 2080|2080ti|3090|4090|v100)"
  exit 1
fi

DATASET_DIR="${EDGE_EVAL_DATASET_DIR:-$DATA_ROOT/dataset/to_measure_programs/${HW}}"
if [[ ! -d "$DATASET_DIR" ]]; then
  if [[ "$HW" == "3090" ]]; then
    DATASET_DIR="$DATA_ROOT/dataset/to_measure_programs/4090"
    if [[ -d "$DATASET_DIR" ]]; then
      echo "[WARN] 3090 dataset missing; fallback to 4090 dataset: $DATASET_DIR"
    else
      echo "Dataset dir not found: $DATASET_DIR"
      exit 1
    fi
  else
    echo "Dataset dir not found: $DATASET_DIR"
    exit 1
  fi
fi

LABEL_A_RAW="${EDGE_LABEL_A:-$(basename "$EXPERT_A_DIR")}"
LABEL_B_RAW="${EDGE_LABEL_B:-$(basename "$EXPERT_B_DIR")}"
LABEL_A="$(sanitize_label "$LABEL_A_RAW")"
LABEL_B="$(sanitize_label "$LABEL_B_RAW")"
BASELINE_LABEL="${EDGE_BASELINE_LABEL:-$LABEL_A}"

EVAL_TAG="${EDGE_EVAL_TAG:-${LABEL_A}_vs_${LABEL_B}}"
EVAL_TAG="$(sanitize_label "$EVAL_TAG")"
SKETCH_DIR="${EDGE_EVAL_SKETCH_DIR:-$RUN_ROOT/${HW}/iter00/eval_ab_sketch_${EVAL_TAG}}"
OUT_DIR="${EDGE_EVAL_OUT_DIR:-$RUN_ROOT/${HW}/iter00/eval_ab_gen_${EVAL_TAG}}"
MEASURE_DIR="${EDGE_EVAL_MEASURE_DIR:-$RUN_ROOT/${HW}/iter00/eval_ab_measure_${EVAL_TAG}}"
LOG_DIR="$OUT_DIR/logs"
SKETCH_PATH="$SKETCH_DIR/0_merge.json"

mkdir -p "$SKETCH_DIR" "$OUT_DIR" "$MEASURE_DIR" "$LOG_DIR"

FILTERED_DATASET_DIR="$DATASET_DIR"
if [[ "$NETWORKS" != "all" ]]; then
  FILTERED_DATASET_DIR="$SKETCH_DIR/_dataset_${EVAL_TAG}"
  export EDGE_FILTER_DATASET_SRC="$DATASET_DIR"
  export EDGE_FILTER_DATASET_DST="$FILTERED_DATASET_DIR"
  export EDGE_FILTER_TARGET="$TARGET"
  export EDGE_FILTER_NETWORKS="$NETWORKS"
  python - <<'PY'
import os
import pickle
import shutil
from pathlib import Path

import sys
tlm_root = Path(os.environ["TLM_ROOT"]).resolve()
sys.path.insert(0, str(tlm_root / "gen"))

from common import (  # noqa: E402
    register_data_path,
    yield_hold_out_five_files,
    get_task_info_filename,
    get_measure_record_filename,
)
import tvm  # noqa: E402

src = Path(os.environ["EDGE_FILTER_DATASET_SRC"]).resolve()
dst = Path(os.environ["EDGE_FILTER_DATASET_DST"]).resolve()
target = os.environ["EDGE_FILTER_TARGET"]
want = {s.strip() for s in os.environ["EDGE_FILTER_NETWORKS"].split(",") if s.strip()}

register_data_path(target)
target_tvm = tvm.target.Target(target.split(" -1 ")[0].strip())

if not src.is_dir():
    raise FileNotFoundError(f"dataset dir not found: {src}")
if not want:
    raise ValueError("EDGE_EVAL_NETWORKS is empty.")

selected = set()
for workload, task, record_file, _ in yield_hold_out_five_files(target_tvm, only_bert=False):
    if workload in want:
        selected.add(Path(record_file).name)

if "inception_v3" in want:
    task_file = get_task_info_filename(("inception_v3", [1, 3, 299, 299]), target_tvm)
    if Path(task_file).exists():
        tasks_part, task_weights = pickle.load(open(task_file, "rb"))
        for task, _w in zip(tasks_part, task_weights):
            selected.add(Path(get_measure_record_filename(task, target_tvm)).name)

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

kept = 0
for f in src.glob("*.json"):
    if f.name in selected:
        os.symlink(str(f), str(dst / f.name))
        kept += 1

print(f"[FILTER] requested={sorted(want)} selected_files={len(selected)} linked={kept} dst={dst}")
if kept == 0:
    raise RuntimeError(
        "No files matched requested networks. "
        f"requested={sorted(want)} src={src}"
    )
PY
fi

if [[ "$SKIP_SKETCH" != "1" ]]; then
  if [[ -f "$SKETCH_PATH" && "$FORCE" != "1" ]]; then
    echo "Sketch exists: $SKETCH_PATH (set EDGE_FORCE=1 to overwrite)"
  else
    echo "[STEP] sketch"
    MAKE_ARGS=(
      --for_type=for_gen_eval_sketch_ansor
      --target "$TARGET"
      --dataset_path "$FILTERED_DATASET_DIR"
      --tokenizer_path "$TOKENIZER"
      --save_path "$SKETCH_DIR"
      --keep_cnt "$KEEP_CNT"
    )
    python "$TLM_ROOT/gen/make_dataset.py" "${MAKE_ARGS[@]}"
  fi
fi

if [[ ! -s "$SKETCH_PATH" ]]; then
  echo "Sketch not found or empty: $SKETCH_PATH"
  exit 1
fi

BASELINE_NAME="$BASELINE_MODE"

run_gen_baseline() {
  local out="$OUT_DIR/${BASELINE_NAME}.json"
  local log="$LOG_DIR/gen_${BASELINE_NAME}.log"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CMD=(
    python "$TLM_ROOT/gen/gen_state_kv_lora.py"
    --model_path "$BASE_CKPT"
    --tokenizer_path "$TOKENIZER"
    --sketch_path "$SKETCH_PATH"
    --save_path "$out"
    --target "$TARGET"
    --target_hardware "$HW_ID"
    --keep_cnt "$KEEP_CNT"
    --use_bucket
    --hardware_embedding_path "$EDGE_EMB_PATH"
    --pos_compensate
  )
  if [[ "$BASELINE_MODE" == "base_kv" ]]; then
    CMD+=(
      --use_hw_kv
      --hw_kv_mode real
      --hw_kv_aligner_path "$EDGE_HW_KV_ALIGNER"
    )
    if [[ -n "$EDGE_HW_KV_PREPROCESS_JSON" ]]; then
      CMD+=(--hw_kv_preprocess_json "$EDGE_HW_KV_PREPROCESS_JSON")
    fi
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" "${CMD[@]}" | tee "$log"
}

run_gen_lora_single() {
  local label="$1"
  local expert_dir="$2"
  local out="$OUT_DIR/lora_${label}.json"
  local log="$LOG_DIR/gen_lora_${label}.log"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CMD=(
    python "$TLM_ROOT/gen/gen_state_kv_lora.py"
    --model_path "$BASE_CKPT"
    --tokenizer_path "$TOKENIZER"
    --sketch_path "$SKETCH_PATH"
    --save_path "$out"
    --target "$TARGET"
    --target_hardware "$HW_ID"
    --keep_cnt "$KEEP_CNT"
    --edge_expert_dirs "$expert_dir"
    --edge_topk 1
    --edge_embedding_path "$EDGE_EMB_PATH"
    --hardware_embedding_path "$EDGE_EMB_PATH"
    --use_bucket
    --pos_compensate
  )
  if [[ "$USE_HW_KV" == "1" ]]; then
    CMD+=(
      --use_hw_kv
      --hw_kv_mode real
      --hw_kv_aligner_path "$EDGE_HW_KV_ALIGNER"
    )
    if [[ -n "$EDGE_HW_KV_PREPROCESS_JSON" ]]; then
      CMD+=(--hw_kv_preprocess_json "$EDGE_HW_KV_PREPROCESS_JSON")
    fi
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" "${CMD[@]}" | tee "$log"
}

run_gen_official() {
  local out="$OUT_DIR/official.json"
  local log="$LOG_DIR/gen_official.log"
  local official_ckpt="${EDGE_OFFICIAL_CKPT:-$GEN_DATA/clm_gen_best_v100}"
  local official_tok="${EDGE_OFFICIAL_TOKENIZER:-$official_ckpt}"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    echo "Gen output exists: $out (set EDGE_FORCE=1 to overwrite)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_ID" python "$TLM_ROOT/gen/gen_state.py" \
    --model_path "$official_ckpt" \
    --tokenizer_path "$official_tok" \
    --sketch_path "$SKETCH_PATH" \
    --save_path "$out" \
    --target "$TARGET" \
    --keep_cnt "$KEEP_CNT" \
    | tee "$log"
}

if [[ "$SKIP_GEN" != "1" ]]; then
  if [[ "$SKIP_BASELINE" != "1" ]]; then
    echo "[STEP] gen ${BASELINE_NAME}"
    run_gen_baseline
  else
    echo "[STEP] gen baseline (skip)"
  fi
  echo "[STEP] gen lora_${LABEL_A}"
  run_gen_lora_single "$LABEL_A" "$EXPERT_A_DIR"
  echo "[STEP] gen lora_${LABEL_B}"
  run_gen_lora_single "$LABEL_B" "$EXPERT_B_DIR"
  if [[ "$SKIP_OFFICIAL" != "1" ]]; then
    echo "[STEP] gen official"
    run_gen_official
  fi
fi

if [[ "$SKIP_MEASURE" != "1" ]]; then
  echo "[STEP] measure"
  JOB_ARGS=()
  [[ -s "$OUT_DIR/${BASELINE_NAME}.json" ]] && JOB_ARGS+=(--job "$OUT_DIR/${BASELINE_NAME}.json=$MEASURE_DIR/${BASELINE_NAME}_measured.json")
  [[ -s "$OUT_DIR/lora_${LABEL_A}.json" ]] && JOB_ARGS+=(--job "$OUT_DIR/lora_${LABEL_A}.json=$MEASURE_DIR/lora_${LABEL_A}_measured.json")
  [[ -s "$OUT_DIR/lora_${LABEL_B}.json" ]] && JOB_ARGS+=(--job "$OUT_DIR/lora_${LABEL_B}.json=$MEASURE_DIR/lora_${LABEL_B}_measured.json")
  [[ -s "$OUT_DIR/official.json" ]] && JOB_ARGS+=(--job "$OUT_DIR/official.json=$MEASURE_DIR/official_measured.json")
  if [[ ${#JOB_ARGS[@]} -eq 0 ]]; then
    echo "No gen outputs found for measurement under: $OUT_DIR"
    exit 1
  fi
  python "$TLM_ROOT/gen/scripts/measure_watchdog.py" \
    --repo-root "$TLM_ROOT" \
    --target "$TARGET" \
    --batch-size "$MEASURE_BATCH" \
    --match-mode "$MATCH_MODE" \
    "${JOB_ARGS[@]}"
fi

if [[ "$SKIP_COMPARE" != "1" ]]; then
  echo "[STEP] compare"
  COMPARE_ARGS=(
    --target "$HW_ID"
    --compare-mode "$COMPARE_MODE"
  )
  [[ -s "$MEASURE_DIR/${BASELINE_NAME}_measured.json" ]] && COMPARE_ARGS+=(--measured "${BASELINE_NAME}=$MEASURE_DIR/${BASELINE_NAME}_measured.json")
  [[ -s "$MEASURE_DIR/lora_${LABEL_A}_measured.json" ]] && COMPARE_ARGS+=(--measured "$LABEL_A=$MEASURE_DIR/lora_${LABEL_A}_measured.json")
  [[ -s "$MEASURE_DIR/lora_${LABEL_B}_measured.json" ]] && COMPARE_ARGS+=(--measured "$LABEL_B=$MEASURE_DIR/lora_${LABEL_B}_measured.json")
  [[ -s "$MEASURE_DIR/official_measured.json" ]] && COMPARE_ARGS+=(--measured official="$MEASURE_DIR/official_measured.json")
  if [[ ${#COMPARE_ARGS[@]} -le 2 ]]; then
    echo "No measured files found under: $MEASURE_DIR"
    exit 1
  fi
  python "$TLM_ROOT/gen/scripts/compare_measured_versions.py" \
    "${COMPARE_ARGS[@]}" \
    --baseline "$BASELINE_LABEL" \
    --output-network-csv "$MEASURE_DIR/compare_network.csv" \
    --output-workload-csv "$MEASURE_DIR/compare_workload.csv"
fi

echo "[DONE] eval_ab=${EVAL_TAG}"
echo "  sketch : $SKETCH_PATH"
echo "  gen    : $OUT_DIR"
echo "  measure: $MEASURE_DIR"
echo "  compare: $MEASURE_DIR/compare_network.csv"
