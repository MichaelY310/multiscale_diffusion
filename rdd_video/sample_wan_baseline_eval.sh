#!/usr/bin/env bash
# Generate the zero-finetuning Wan 2.1-1.3B baseline on held-out prompts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${PROMPTS_FILE:-${REPO_DIR}/rdd_video_data/prompts/pilot_eval_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results_rdd_video/wan_base_eval}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
SIZE="${SIZE:-832x448}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SEED="${SEED:-2026}"
KEEP_MODEL_RESIDENT="${KEEP_MODEL_RESIDENT:-1}"

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "ERROR: missing held-out prompts: ${PROMPTS_FILE}" >&2
  echo "Run: python rdd_video/build_prompt_dataset.py" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs"

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  extra=()
  if [[ "${KEEP_MODEL_RESIDENT}" == "1" ]]; then
    extra+=(--keep-model-resident)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/generate_training_data.py" \
    --pretrained-dir "${PRETRAINED_DIR}" \
    --prompts-file "${PROMPTS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --size "${SIZE}" --frame-num "${FRAME_NUM}" \
    --sampling-steps "${SAMPLING_STEPS}" --sample-solver unipc \
    --guide-scale "${GUIDE_SCALE}" --shift "${SHIFT}" \
    --seed "${SEED}" --num-gpus "${NUM_GPUS}" --gpu-rank "${gpu}" \
    --device 0 --no-latent "${extra[@]}" \
    >"${OUTPUT_DIR}/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

count="$(find "${OUTPUT_DIR}/pixel" -maxdepth 1 -name '*.mp4' | wc -l)"
if [[ "${count}" -lt "${NUM_SAMPLES}" ]]; then
  echo "ERROR: expected ${NUM_SAMPLES} videos, found ${count}" >&2
  exit 3
fi
echo "Wan baseline ready: ${OUTPUT_DIR}/pixel (${count} videos)"
