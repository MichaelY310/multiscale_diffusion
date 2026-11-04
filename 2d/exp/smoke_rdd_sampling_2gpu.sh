#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-rdd-eval/bin/python}"
GOOD_CKPT="${GOOD_CKPT:?Set GOOD_CKPT to a Unified RDD diffusion checkpoint}"
WEAK_CKPT="${WEAK_CKPT:-}"
MODEL="${MODEL:-DiT-XL/2}"
GPU0="${GPU0:-4}"
GPU1="${GPU1:-5}"
SMOKE_ROOT="${SMOKE_ROOT:-$ROOT/.tmp/rdd_improvement_sampling_smoke}"
CONFIG="$SMOKE_ROOT/config.jsonl"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: evaluation Python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -s "$GOOD_CKPT" ]]; then
  echo "ERROR: good checkpoint not found: $GOOD_CKPT" >&2
  exit 2
fi
if [[ -n "$WEAK_CKPT" && ! -s "$WEAK_CKPT" ]]; then
  echo "ERROR: weak checkpoint not found: $WEAK_CKPT" >&2
  exit 2
fi

mkdir -p "$SMOKE_ROOT"
BUILD_ARGS=(
  --ckpt "$GOOD_CKPT"
  --model "$MODEL"
  --suite smoke
  --output-root "$SMOKE_ROOT/samples"
  --config-out "$CONFIG"
)
if [[ -n "$WEAK_CKPT" ]]; then
  BUILD_ARGS+=(--guide-ckpt "$WEAK_CKPT" --guide-model "$MODEL")
fi

"$PYTHON_BIN" "$ROOT/2d/exp/build_rdd_fid_improvement_parity.py" "${BUILD_ARGS[@]}"

CONFIG="$CONFIG" TOTAL_SAMPLES="${TOTAL_SAMPLES:-4}" \
  BATCH_SIZE="${BATCH_SIZE:-2}" VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-2}" \
  GPU0="$GPU0" GPU1="$GPU1" NO_EVAL=1 PYTHON_BIN="$PYTHON_BIN" \
  bash "$ROOT/2d/exp/run_rdd_fid_improvement_parity_2gpu.sh"

echo "Sampling smoke passed. Outputs: $SMOKE_ROOT"
