#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-eval/bin/python}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
ADAPTER_CKPT="${ADAPTER_CKPT:-${ROOT}/checkpoints/RDD-CSRA-OnPolicy-Unified-L-400K-r64-late6/0003000.pt}"
VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/eval_results/csra_onpolicy_late6_3k}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-2}"
ADAPTER_PRECISION="${ADAPTER_PRECISION:-legacy_autocast}"
SESSION_PREFIX="${SESSION_PREFIX:-rdd_csra_onpolicy_eval}"

[[ "${ADAPTER_PRECISION}" == fp32 || "${ADAPTER_PRECISION}" == legacy_autocast ]] || exit 2
[[ "${SESSION_PREFIX}" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2

seeds=(20260921 20260922 20260923)
gpus=(0 1 2)

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "missing base checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_CKPT}" ]]; then
  echo "missing adapter checkpoint: ${ADAPTER_CKPT}" >&2
  exit 1
fi

# Preflight all jobs before starting any, and never overwrite old evaluations.
for seed in "${seeds[@]}"; do
  session="${SESSION_PREFIX}_seed${seed}"
  if tmux has-session -t "=${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  if [[ -d "${OUT_ROOT}/seed_${seed}" ]] && [[ -n "$(ls -A "${OUT_ROOT}/seed_${seed}")" ]]; then
    echo "ERROR: evaluation output is not empty: ${OUT_ROOT}/seed_${seed}" >&2
    exit 1
  fi
done
mkdir -p "${OUT_ROOT}"
for index in 0 1 2; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  session="${SESSION_PREFIX}_seed${seed}"
  output="${OUT_ROOT}/seed_${seed}"
  mkdir -p "${output}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session}" \
    "set -o pipefail; cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u 2d/attention_analysis/sample_csra_adapter_preview.py --family rdd --base-ckpt '${BASE_CKPT}' --adapter-ckpt '${ADAPTER_CKPT}' --vae-path '${VAE_PATH}' --output-dir '${output}' --batch-size '${BATCH_SIZE}' --seed '${seed}' --rdd-steps 80,80,90 --benchmark-repeats '${BENCHMARK_REPEATS}' --adapter-precision '${ADAPTER_PRECISION}' --device cuda:0 2>&1 | tee '${output}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${output}/run.log"
done
