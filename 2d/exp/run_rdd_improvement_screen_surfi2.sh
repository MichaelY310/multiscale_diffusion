#!/usr/bin/env bash
set -euo pipefail

# One-command 10K/50K screening entry point for an existing Unified RDD
# diffusion checkpoint.  It builds the parity JSONL and then samples/evaluates
# every row with the repository-local VAE, ADM reference, and evaluator.

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-rdd-eval/bin/python}"
GOOD_CKPT="${GOOD_CKPT:?Set GOOD_CKPT to a Unified RDD diffusion checkpoint}"
WEAK_CKPT="${WEAK_CKPT:-}"
MODEL="${MODEL:-DiT-XL/2}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SAMPLER="${SAMPLER:-dpm2}"
CFG_SCALE="${CFG_SCALE:-1.5}"
RUN_NAME="${RUN_NAME:-rdd_improvement_screen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_results/$RUN_NAME}"
CONFIG="${CONFIG:-$ROOT/eval_results/configs/$RUN_NAME.jsonl}"
LOG_DIR="${LOG_DIR:-$ROOT/eval_results/logs/$RUN_NAME}"

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: evaluation Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$GOOD_CKPT" ]] || { echo "ERROR: checkpoint missing: $GOOD_CKPT" >&2; exit 2; }
if [[ -n "$WEAK_CKPT" && ! -s "$WEAK_CKPT" ]]; then
  echo "ERROR: weak checkpoint missing: $WEAK_CKPT" >&2
  exit 2
fi

mkdir -p "$(dirname "$CONFIG")" "$OUTPUT_ROOT" "$LOG_DIR"

BUILD_ARGS=(
  --ckpt "$GOOD_CKPT"
  --model "$MODEL"
  --sampler "$SAMPLER"
  --cfg-scale "$CFG_SCALE"
  --suite full
  --output-root "$OUTPUT_ROOT"
  --config-out "$CONFIG"
)
if [[ -n "$WEAK_CKPT" ]]; then
  BUILD_ARGS+=(--guide-ckpt "$WEAK_CKPT" --guide-model "$MODEL")
fi

cd "$ROOT"
"$PYTHON_BIN" 2d/exp/build_rdd_fid_improvement_parity.py "${BUILD_ARGS[@]}"

CONFIG="$CONFIG" TOTAL_SAMPLES="$TOTAL_SAMPLES" \
  GPU0="$GPU0" GPU1="$GPU1" LOG_DIR="$LOG_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash 2d/exp/run_rdd_fid_improvement_parity_2gpu.sh

echo "RDD improvement screening complete."
echo "Config:  $CONFIG"
echo "Samples: $OUTPUT_ROOT"
echo "Logs:    $LOG_DIR"
