#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-free, low-memory microbenchmark.  It intentionally permits a busy
# GPU because this is a preliminary shared-GPU run; it never signals or stops
# another process.  Re-run on an exclusive GPU before reporting paper latency.
ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
OUT="${OUT:-${ROOT}/eval_results/rdd_video_speed_benchmark_shared_gpu0}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"

mkdir -p "${OUT}"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | head -n 1 | tr -d ' ')"
if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: cannot read free memory for physical GPU ${GPU}" >&2
  exit 2
fi
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "ERROR: physical GPU ${GPU} has only ${FREE_MIB} MiB free; need ${MIN_FREE_MIB} MiB." >&2
  exit 3
fi

echo "Shared-GPU preliminary benchmark"
echo "physical GPU=${GPU}; free=${FREE_MIB} MiB; no checkpoint"
echo "existing processes will not be signaled or stopped"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_MODULE_LOADING=LAZY
cd "${ROOT}"

nice -n 19 "${PYTHON}" rdd_video/attention_analysis/benchmark_wan_video_methods.py \
  --official-kernel third_party/FastVideo-kernel-ref/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py \
  --fused-router-kernel third_party/FastVideo-kernel-ref/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py \
  --rdd-route-kernel rdd_video/attention_analysis/rdd_video_route_triton.py \
  --rdd-reuse-kernel rdd_video/attention_analysis/rdd_video_reuse_triton.py \
  --output-dir "${OUT}" \
  --device cuda:0 \
  --densities 0.125 0.20 0.25 \
  --reuse-every 3 \
  --warmup 2 \
  --repeats 10 \
  2>&1 | tee "${OUT}/benchmark.log"
