#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
CONFIG="${CONFIG:-$ROOT/2d/exp/rdd_improvement_screen.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-rdd-eval/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: no Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 2
fi
TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NO_EVAL="${NO_EVAL:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/eval_results/logs}"
VAE_PATH="${VAE_PATH:-$ROOT/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
export RDD_REF_NPZ="${RDD_REF_NPZ:-$ROOT/eval_results/reference/VIRTUAL_imagenet256_labeled.npz}"
export RDD_FID_EVALUATOR="${RDD_FID_EVALUATOR:-$ROOT/guided-diffusion/evaluations/evaluator.py}"
export RDD_EVAL_LOG_DIR="${RDD_EVAL_LOG_DIR:-$ROOT/eval_results/fid_logs}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$VAE_PATH/config.json" ]]; then
  echo "ERROR: local SD VAE not found at $VAE_PATH" >&2
  exit 2
fi
if [[ "$NO_EVAL" != "1" && ! -s "$RDD_REF_NPZ" ]]; then
  echo "ERROR: FID reference not found: $RDD_REF_NPZ" >&2
  exit 2
fi

EVAL_ARGS=()
if [[ "$NO_EVAL" == "1" ]]; then
  EVAL_ARGS=(--no-eval)
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON_BIN" 2d/sample_fid_unified.py \
  --config "$CONFIG" --rank 0 --world-size 2 \
  --total-samples "$TOTAL_SAMPLES" --batch-size "$BATCH_SIZE" \
  --vae-batch-size "$VAE_BATCH_SIZE" --vae-path "$VAE_PATH" \
  "${EVAL_ARGS[@]}" \
  >"$LOG_DIR/rdd_improvement_rank0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON_BIN" 2d/sample_fid_unified.py \
  --config "$CONFIG" --rank 1 --world-size 2 \
  --total-samples "$TOTAL_SAMPLES" --batch-size "$BATCH_SIZE" \
  --vae-batch-size "$VAE_BATCH_SIZE" --vae-path "$VAE_PATH" --no-eval \
  >"$LOG_DIR/rdd_improvement_rank1.log" 2>&1 &
PID1=$!

trap 'kill "$PID0" "$PID1" 2>/dev/null || true' INT TERM
wait "$PID0"
wait "$PID1"

echo "Parity run complete. Logs: $LOG_DIR/rdd_improvement_rank{0,1}.log"
