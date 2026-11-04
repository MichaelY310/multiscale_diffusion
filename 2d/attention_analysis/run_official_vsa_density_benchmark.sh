#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
GPU="${GPU:-0}"
OUT="${OUT:-${ROOT}/eval_results/rdd_vsa_kernel_density_bench}"
LOG="${LOG:-${OUT}/benchmark.log}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-20}"
REQUIRED_IDLE_POLLS="${REQUIRED_IDLE_POLLS:-3}"

mkdir -p "${OUT}"

idle_count=0
while (( idle_count < REQUIRED_IDLE_POLLS )); do
  line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "${GPU}")"
  used="$(awk -F, '{gsub(/ /, "", $1); print $1}' <<<"${line}")"
  util="$(awk -F, '{gsub(/ /, "", $2); print $2}' <<<"${line}")"
  printf '[%s] physical GPU %s: used=%s MiB util=%s%% idle_count=%s/%s\n' \
    "$(date --iso-8601=seconds)" "${GPU}" "${used}" "${util}" "${idle_count}" "${REQUIRED_IDLE_POLLS}"
  if (( used <= MAX_USED_MIB && util <= MAX_UTIL )); then
    idle_count=$((idle_count + 1))
  else
    idle_count=0
  fi
  if (( idle_count < REQUIRED_IDLE_POLLS )); then
    sleep "${POLL_SECONDS}"
  fi
done

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python interpreter is not executable: ${PYTHON}" >&2
  exit 2
fi

{
  printf '[%s] starting official VSA density benchmark on physical GPU %s\n' "$(date --iso-8601=seconds)" "${GPU}"
  "${PYTHON}" 2d/attention_analysis/benchmark_official_vsa_density.py \
    --official-kernel third_party/FastVideo-kernel-ref/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py \
    --rdd-kernel 2d/attention_analysis/rdd_block_sparse_triton.py \
    --output-dir "${OUT}"
  printf '[%s] completed\n' "$(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
