#!/usr/bin/env bash
set -euo pipefail

cd /home/tiger/multiscale_diffusion/guided-diffusion
source .venv/bin/activate
export PYTHONPATH="/home/tiger/multiscale_diffusion/guided-diffusion:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIST_BACKEND="${DIST_BACKEND:-gloo}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export RESUME_DIR="/home/tiger/multiscale_diffusion/guided-diffusion/guided_diffusion/results_lsun_church_gpu0/004"
export RESUME_STEP="45000"

export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:${LD_LIBRARY_PATH:-}

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
/home/tiger/multiscale_diffusion/guided-diffusion/.venv/bin/python - <<'PY'
import os
import sys
import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
elif os.environ.get("REQUIRE_CUDA", "1") == "1":
    raise SystemExit("CUDA is not available in this environment; aborting before CPU fallback.")
PY

cmd=(
  /home/tiger/multiscale_diffusion/guided-diffusion/.venv/bin/python
  scripts/image_train_with_drift_lsun_church.py
  --dataset_path "${DATASET_PATH:-/home/tiger/multiscale_diffusion/lsun/lsun_church_train}"
  --dataset_split "${DATASET_SPLIT:-train}"
  --batch_size "${BATCH_SIZE:-8}"
  --num_workers "${NUM_WORKERS:-8}"
  --total_steps "${TOTAL_STEPS:-1000000}"
  --log_every "${LOG_EVERY:-100}"
  --ckpt_every "${CKPT_EVERY:-5000}"
  --lr "${LR:-1e-4}"
  --results_dir "${RESULTS_DIR:-guided_diffusion/results_lsun_church_gpu0}"
  --num_channels_64 "${NUM_CHANNELS_64:-64}"
  --num_channels_128 "${NUM_CHANNELS_128:-128}"
  --num_channels_256 "${NUM_CHANNELS_256:-192}"
)

if [[ -n "${RESUME_DIR:-}" ]]; then
  cmd+=(--resume_dir "${RESUME_DIR}")
fi

if [[ -n "${RESUME_STEP:-}" ]]; then
  cmd+=(--resume_step "${RESUME_STEP}")
fi

"${cmd[@]}"


# DIST_BACKEND=gloo bash scripts/train_rdd_lsun_church_gpu0.sh
