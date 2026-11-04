#!/usr/bin/env bash
# Run the existing step-doubling local-solver diagnostic only after the two
# higher-priority qualitative/teacher-forced GPU0 jobs have exited.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
cd "${REPO_ROOT}"
mkdir -p logs eval_results/diagnostics/local_solver_error

while tmux has-session -t teacher_forced_after_predx0 2>/dev/null; do
  echo "[$(date -Is)] waiting for teacher_forced_after_predx0"
  sleep 60
done
while nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; do
  echo "[$(date -Is)] physical GPU 0 still has a compute process; waiting"
  sleep 60
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

run_profile() {
  local label="$1"
  local lengths="$2"
  local ckpt_dir="$3"
  local ckpt_step="$4"
  local solver="$5"
  "${PYTHON}" -u 2d/exp/profile_rdd_local_error.py \
    --data-path imagenet/ILSVRC/Data/CLS-LOC/train \
    --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
    --ckpt-dir "${ckpt_dir}" \
    --ckpt-step "${ckpt_step}" \
    --resolutions 32,16,8 \
    --stage-lengths "${lengths}" \
    --architectures DiT-XL/2,DiT-L/2,DiT-B/2 \
    --solver "${solver}" \
    --num-images 256 \
    --image-batch-size 16 \
    --probe-bins 8 \
    --half-span-fraction 0.01 \
    --num-workers 8 \
    --device cuda \
    --seed 20260814 \
    --output "eval_results/diagnostics/local_solver_error/${label}_${solver}.json"
}

for solver in ddpm dpm2; do
  run_profile RDD_300_300_400 300,300,400 checkpoints/RDD-Multistage-300-300-400 0400000 "${solver}"
  run_profile RDD_100_300_600 100,300,600 checkpoints/RDD-Multistage-100-300-600 0400000 "${solver}"
  run_profile RDD_100_300_600_8x8continued600K 100,300,600 \
    eval_results/mixed_checkpoints/ms100300600_8x8continued600k_16x32orig400k_808090_ddpm \
    mixed "${solver}"
done

echo "[$(date -Is)] local solver diagnostics complete"
