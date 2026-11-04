#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-rdd-train/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: no Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 2
fi

if [[ -z "${DATA_PATH:-}" ]]; then
  for candidate in \
    "$ROOT/imagenet/ILSVRC/Data/CLS-LOC/train" \
    "$ROOT/imagenet/train" \
    "$ROOT/imagenet"; do
    if [[ -d "$candidate" ]]; then
      DATA_PATH="$candidate"
      break
    fi
  done
fi
if [[ -z "${DATA_PATH:-}" || ! -d "$DATA_PATH" ]]; then
  echo "ERROR: original ImageNet ImageFolder not found. Set DATA_PATH to its train directory." >&2
  exit 2
fi

VARIANT="${VARIANT:-minsnr}"
MAX_STEPS="${MAX_STEPS:-400000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
MODEL="${MODEL:-DiT-XL/2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
GPUS="${GPUS:-0,1}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT/checkpoints/rdd_improvement_parity}"
DINOV2_REPO="${DINOV2_REPO:-$ROOT/checkpoints/pretrained/dinov2}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-$ROOT/checkpoints/pretrained/dinov2_vitb14_pretrain.pth}"
VAE_PATH="${VAE_PATH:-$ROOT/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"

if [[ ! -f "$VAE_PATH/config.json" ]]; then
  echo "ERROR: local SD VAE not found at $VAE_PATH" >&2
  exit 2
fi

COMMON_ARGS=(
  --model "$MODEL"
  --data-source imagenet --data-path "$DATA_PATH"
  --vae-path "$VAE_PATH"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --ckpt-every "$CKPT_EVERY"
  --max-train-steps "$MAX_STEPS"
)

case "$VARIANT" in
  baseline) EXTRA_ARGS=() ;;
  minsnr) EXTRA_ARGS=(--loss-weighting min-snr --min-snr-gamma 5.0) ;;
  p2) EXTRA_ARGS=(--loss-weighting p2 --p2-gamma 1.0 --p2-k 1.0) ;;
  loss_second_moment) EXTRA_ARGS=(--timestep-sampler loss-second-moment) ;;
  repa_finest) EXTRA_ARGS=(--repa-weight 0.5 --repa-model dinov2_vitb14 --repa-depth 8 --repa-stages finest --repa-repo "$DINOV2_REPO" --repa-weights "$DINOV2_WEIGHTS") ;;
  repa) EXTRA_ARGS=(--repa-weight 0.5 --repa-model dinov2_vitb14 --repa-depth 8 --repa-repo "$DINOV2_REPO" --repa-weights "$DINOV2_WEIGHTS") ;;
  minsnr_repa) EXTRA_ARGS=(--loss-weighting min-snr --min-snr-gamma 5.0 --repa-weight 0.5 --repa-model dinov2_vitb14 --repa-depth 8 --repa-repo "$DINOV2_REPO" --repa-weights "$DINOV2_WEIGHTS") ;;
  fm_baseline)
    exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "$ROOT/2d/train_RDD_FM_imagenet256_unified.py" \
      --resolutions 32,16,8 --stage-lengths 400,300,300 --diffusion-steps 1000 \
      "${COMMON_ARGS[@]}" --time-sampler uniform \
      --results-dir "$RESULTS_BASE/$VARIANT"
    ;;
  fm_logit_normal)
    exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "$ROOT/2d/train_RDD_FM_imagenet256_unified.py" \
      --resolutions 32,16,8 --stage-lengths 400,300,300 --diffusion-steps 1000 \
      "${COMMON_ARGS[@]}" --time-sampler logit-normal \
      --results-dir "$RESULTS_BASE/$VARIANT"
    ;;
  fm_repa)
    exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "$ROOT/2d/train_RDD_FM_imagenet256_unified.py" \
      --resolutions 32,16,8 --stage-lengths 400,300,300 --diffusion-steps 1000 \
      "${COMMON_ARGS[@]}" --repa-weight 0.5 \
      --repa-repo "$DINOV2_REPO" --repa-weights "$DINOV2_WEIGHTS" \
      --results-dir "$RESULTS_BASE/$VARIANT"
    ;;
  fm_repa_finest)
    exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "$ROOT/2d/train_RDD_FM_imagenet256_unified.py" \
      --resolutions 32,16,8 --stage-lengths 400,300,300 --diffusion-steps 1000 \
      "${COMMON_ARGS[@]}" --repa-weight 0.5 --repa-stages finest \
      --repa-repo "$DINOV2_REPO" --repa-weights "$DINOV2_WEIGHTS" \
      --results-dir "$RESULTS_BASE/$VARIANT"
    ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$ROOT/2d/train_RDD_imagenet256_unified.py" \
  --resolutions 32,16,8 --stage-lengths 300,300,400 --diffusion-steps 1000 \
  "${COMMON_ARGS[@]}" --results-dir "$RESULTS_BASE/$VARIANT" \
  "${EXTRA_ARGS[@]}"
