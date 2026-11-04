#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final RDD Unified XL/2 training-free attention benchmark.
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash 2d/attention_analysis/run_rdd_hierarchical_kernel_final.sh

PROJECT_ROOT="${PROJECT_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-rdd-train/bin/python}"
RDD_CKPT="${RDD_CKPT:-${PROJECT_ROOT}/checkpoints/RDD-Unified-XL/0400000.pt}"
VAE_DIR="${VAE_DIR:-${PROJECT_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_results/rdd_hierarchical_kernel_final}"
TIMING_REPEATS="${TIMING_REPEATS:-3}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/2d/attention_analysis/test_rdd_block_sparse_triton.py" \
  --dtype float32
"${PYTHON_BIN}" "${PROJECT_ROOT}/2d/attention_analysis/test_rdd_block_sparse_triton.py" \
  --dtype bfloat16

"${PYTHON_BIN}" "${PROJECT_ROOT}/2d/attention_analysis/sample_rdd_hierarchical_kernel.py" \
  --unified-ckpt "${RDD_CKPT}" \
  --vae-path "${VAE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --classes 207,282,323,817 \
  --cfg-scale 1.5 \
  --sample-steps 80,80,90 \
  --settings \
triton_dense_score3_all,triton_dense_score_adaptive_5_3_3,triton_no_dense_balanced_score3_all,triton_no_dense_cross5_score3_all \
  --timing-repeats "${TIMING_REPEATS}" \
  --device cuda:0 \
  --memory-fraction 0.45

echo "Metrics: ${OUTPUT_DIR}/metrics.json"
echo "Images:  ${OUTPUT_DIR}/triton_hierarchical_comparison.png"
