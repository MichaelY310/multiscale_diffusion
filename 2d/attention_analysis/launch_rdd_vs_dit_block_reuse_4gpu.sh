#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${OUT:-${ROOT}/eval_results/rdd_vs_dit_block_reuse_400k}"
PYTHON="${ROOT}/.venv-rdd-eval/bin/python"
SCRIPT="${ROOT}/2d/attention_analysis/probe_rdd_vs_dit_block_reuse.py"

mkdir -p "${OUT}/logs"
seeds=(1234 2345 3456 4567)
offsets=(0 113 227 349)
pids=()

for rank in 0 1 2 3; do
  log="${OUT}/logs/seed_${seeds[$rank]}.log"
  echo "launch GPU ${rank}: seed=${seeds[$rank]} offset=${offsets[$rank]} log=${log}"
  CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON}" -u "${SCRIPT}" \
    --rdd-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --dit-ckpt "${ROOT}/checkpoints/DiT-XL-2/0400000.pt" \
    --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --imagenet-train "${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train" \
    --output "${OUT}/seed_${seeds[$rank]}.json" \
    --device cuda:0 \
    --seed "${seeds[$rank]}" \
    --image-offset "${offsets[$rank]}" \
    --batch-size 4 \
    --interpolation nearest \
    --warmup 5 \
    --timing-rounds 25 \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more workers failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi

echo "All four true-model RDD-vs-DiT probes completed: ${OUT}"
