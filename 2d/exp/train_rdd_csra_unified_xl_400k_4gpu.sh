#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${RDD_PYTHON:-${ROOT}/.venv-rdd-eval/bin/python}"
DATA="${RDD_IMAGENET:-${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train}"
CKPT="${RDD_CKPT:-${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt}"
VAE="${RDD_VAE:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUT="${RDD_CSRA_OUT:-${ROOT}/checkpoints/RDD-CSRA-Unified-XL-400K-r64-layers1_2_3_14_17_20}"

for path in "${PYTHON}" "${DATA}" "${CKPT}" "${VAE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# This server can report NCCL illegal memory access when a second communicator
# uses peer-to-peer transport on already-busy Blackwell GPUs.  CSRA gradients
# are only ~9 MiB, so the shared-memory path is both safe and sufficient.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cd "${ROOT}"
exec "${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${RDD_CSRA_NPROC:-4}" \
  2d/attention_analysis/train_rdd_csra_adapter.py \
  --rdd-ckpt "${CKPT}" \
  --teacher-family "${RDD_CSRA_TEACHER_FAMILY:-rdd}" \
  --model "${RDD_CSRA_MODEL:-DiT-XL/2}" \
  --vae-path "${VAE}" \
  --data-path "${DATA}" \
  --output-dir "${OUT}" \
  --layers "${RDD_CSRA_LAYERS:-1,2,3,14,17,20}" \
  --rank "${RDD_CSRA_RANK:-64}" \
  --adapter-architecture "${RDD_CSRA_ARCHITECTURE:-legacy_v1}" \
  --training-mode "${RDD_CSRA_TRAINING_MODE:-teacher_forced}" \
  --global-batch-size "${RDD_CSRA_GLOBAL_BATCH:-256}" \
  --micro-batch-size "${RDD_CSRA_MICRO_BATCH:-32}" \
  --max-train-steps "${RDD_CSRA_MAX_STEPS:-5000}" \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --num-workers 4 \
  --log-every 10 \
  --checkpoint-every "${RDD_CSRA_CHECKPOINT_EVERY:-1000}" \
  ${RDD_CSRA_EXTRA_ARGS:-}
