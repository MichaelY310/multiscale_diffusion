#!/usr/bin/env bash
# One fully specified FastWan/RDD timing + video worker.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907}"
GPU="${GPU:?set physical GPU index}"
NAME="${NAME:?set output stem}"
FACTORS="${FACTORS:?set spatial,temporal pairs}"
ATTENTION_MODE="${ATTENTION_MODE:-vsa}"
VSA_SPARSITY="${VSA_SPARSITY:-0.8}"
ROUTE_DENSITY="${ROUTE_DENSITY:-0.2}"
REUSE_EVERY="${REUSE_EVERY:-1}"
REUSE_SCOPE="${REUSE_SCOPE:-backend}"
SMALL_STAGE_EXECUTOR="${SMALL_STAGE_EXECUTOR:-native}"
SMALL_STAGE_MAX_PADDED_TOKENS="${SMALL_STAGE_MAX_PADDED_TOKENS:-0}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
TIMED_RUNS="${TIMED_RUNS:-5}"
SEED="${SEED:-20260905}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"

mkdir -p "${OUT}"
cd "${ROOT}"

MASTER_PORT="${MASTER_PORT:-$((29617 + GPU))}" \
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u rdd_video/sample_fastwan_native_vsa.py \
  --fastvideo-root third_party/FastVideo-ref \
  --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
  --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
  --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
  --prompt "${PROMPT}" --size 832x448 --frame-num 29 --seed "${SEED}" \
  --dmd-timesteps 1000,757,522 --stage-factors "${FACTORS}" \
  --transition-mode fresh --attention-mode "${ATTENTION_MODE}" \
  --vsa-sparsity "${VSA_SPARSITY}" --route-density "${ROUTE_DENSITY}" \
  --reuse-every "${REUSE_EVERY}" --reuse-scope "${REUSE_SCOPE}" \
  --small-stage-executor "${SMALL_STAGE_EXECUTOR}" \
  --small-stage-max-padded-tokens "${SMALL_STAGE_MAX_PADDED_TOKENS}" \
  --warmup-runs "${WARMUP_RUNS}" --timed-runs "${TIMED_RUNS}" \
  --output "${OUT}/${NAME}.mp4" \
  --metrics-json "${OUT}/${NAME}.metrics.json" \
  --latent-output "${OUT}/${NAME}.latent.pt" \
  2>&1 | tee "${OUT}/${NAME}.log"
