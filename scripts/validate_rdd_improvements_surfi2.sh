#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$ROOT/.venv-rdd-train/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/.venv-rdd-eval/bin/python}"
GPUS="${GPUS:-4,5}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-0}"
SMOKE_MODEL="${SMOKE_MODEL:-DiT-B/2}"
SMOKE_ROOT="${SMOKE_ROOT:-$ROOT/.tmp/rdd_improvement_training_smoke}"
DATA_PATH="${DATA_PATH:-$ROOT/imagenet/ILSVRC/Data/CLS-LOC/train}"
VAE_PATH="${VAE_PATH:-$ROOT/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
DINOV2_REPO="${DINOV2_REPO:-$ROOT/checkpoints/pretrained/dinov2}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-$ROOT/checkpoints/pretrained/dinov2_vitb14_pretrain.pth}"

fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -x "$TRAIN_PYTHON" ]] || fail "training Python missing: $TRAIN_PYTHON"
[[ -x "$EVAL_PYTHON" ]] || fail "evaluation Python missing: $EVAL_PYTHON"
[[ -d "$DATA_PATH" ]] || fail "ImageNet train directory missing: $DATA_PATH"
[[ -f "$VAE_PATH/config.json" ]] || fail "local SD VAE missing: $VAE_PATH"
[[ -s "$ROOT/eval_results/reference/VIRTUAL_imagenet256_labeled.npz" ]] || \
  fail "ImageNet FID reference missing"

cd "$ROOT"
echo "[1/6] Dependency imports"
"$TRAIN_PYTHON" -c 'import torch, torchvision, diffusers, timm, numpy, PIL; print("train", torch.__version__, torch.version.cuda, torch.cuda.device_count())'
"$EVAL_PYTHON" -c 'import torch, diffusers, tensorflow, scipy, numpy, PIL; print("eval", torch.__version__, tensorflow.__version__)'

echo "[2/6] Python compilation"
"$TRAIN_PYTHON" -m py_compile \
  diffusion/rdd_improvements.py diffusion/repa.py diffusion/flow_matching_with_drift.py \
  diffusion/timestep_sampler.py DiT/models_multires.py \
  2d/train_RDD_imagenet256.py 2d/train_RDD_imagenet256_unified.py \
  2d/train_RDD_FM_imagenet256_unified.py 2d/sample_fid_unified.py \
  2d/exp/build_rdd_fid_improvement_parity.py scripts/compare_sample_dirs.py \
  scripts/download_rdd_sampling_smoke_ckpts.py \
  tests/test_rdd_improvements.py

echo "[3/6] Shell syntax"
bash -n \
  scripts/setup_rdd_improvements.sh \
  scripts/validate_rdd_improvements_surfi2.sh \
  2d/exp/train_rdd_improvement_parity_2gpu.sh \
  2d/exp/run_rdd_fid_improvement_parity_2gpu.sh \
  2d/exp/run_rdd_improvement_screen_surfi2.sh \
  2d/exp/smoke_rdd_sampling_2gpu.sh

echo "[4/6] CPU formula and compatibility tests"
"$TRAIN_PYTHON" tests/test_rdd_improvements.py

echo "[5/6] ImageNet and local resource probe"
"$TRAIN_PYTHON" - "$DATA_PATH" "$VAE_PATH" <<'PY'
import sys
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL

data_path, vae_path = sys.argv[1:]
dataset = ImageFolder(data_path)
assert len(dataset) == 1_281_167, len(dataset)
assert len(dataset.classes) == 1000, len(dataset.classes)
AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
print(f"ImageNet={len(dataset):,}, classes={len(dataset.classes)}, local VAE=OK")
PY

if [[ "$RUN_GPU_SMOKE" != "1" ]]; then
  echo "[6/6] GPU smoke skipped (set RUN_GPU_SMOKE=1). Preflight passed."
  exit 0
fi

[[ -d "$DINOV2_REPO" ]] || fail "DINOv2 repository missing; run scripts/setup_rdd_improvements.sh"
[[ -s "$DINOV2_WEIGHTS" ]] || fail "DINOv2 weights missing; run scripts/setup_rdd_improvements.sh"

echo "[6/6] Two-GPU one-update smoke matrix on visible GPUs $GPUS"
mkdir -p "$SMOKE_ROOT/logs"
variants=(
  baseline minsnr p2 loss_second_moment
  repa_finest repa minsnr_repa
  fm_baseline fm_logit_normal fm_repa_finest fm_repa
)
for variant in "${variants[@]}"; do
  echo "--- smoke: $variant ---"
  VARIANT="$variant" MAX_STEPS=1 MODEL="$SMOKE_MODEL" \
    GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=4 NUM_WORKERS=0 CKPT_EVERY=1 \
    GPUS="$GPUS" DATA_PATH="$DATA_PATH" VAE_PATH="$VAE_PATH" \
    DINOV2_REPO="$DINOV2_REPO" DINOV2_WEIGHTS="$DINOV2_WEIGHTS" \
    RESULTS_BASE="$SMOKE_ROOT/checkpoints" PYTHON_BIN="$TRAIN_PYTHON" \
    bash 2d/exp/train_rdd_improvement_parity_2gpu.sh \
    >"$SMOKE_ROOT/logs/$variant.log" 2>&1
  grep -q "Saved checkpoint at step 0000001" "$SMOKE_ROOT/logs/$variant.log" || \
    fail "$variant did not save its one-step checkpoint; inspect $SMOKE_ROOT/logs/$variant.log"
done

echo "All two-GPU training variants passed. Logs: $SMOKE_ROOT/logs"
