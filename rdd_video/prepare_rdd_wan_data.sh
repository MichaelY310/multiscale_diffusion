#!/usr/bin/env bash
# Generate conditional Wan teacher samples and cache all RDD stage latents.
# This is offline synthetic-data preparation, not step distillation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
TEACHER_MODEL="${TEACHER_MODEL:-t2v-1.3B}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/rdd_video_data}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-${DATA_ROOT}/synthetic_wan1p3b}"
PROMPTS_FILE="${PROMPTS_FILE:-}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
PROJECTION_MODE="${PROJECTION_MODE:-latent}"
MULTIRES_DIR="${MULTIRES_DIR:-${DATA_ROOT}/multires_latents_${PROJECTION_MODE}}"
SIZE="${SIZE:-832x448}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SEED="${SEED:-42}"
KEEP_MODEL_RESIDENT="${KEEP_MODEL_RESIDENT:-0}"

if [[ ! -d "${PRETRAINED_DIR}" ]]; then
    echo "ERROR: missing Wan checkpoint: ${PRETRAINED_DIR}" >&2
    exit 2
fi
if (( NUM_SAMPLES > 2000 )) && [[ -z "${PROMPTS_FILE}" ]]; then
    echo "ERROR: provide PROMPTS_FILE for a main run; the built-in 50 prompts are smoke-only." >&2
    exit 2
fi
if [[ -n "${PROMPTS_FILE}" && ! -f "${PROMPTS_FILE}" ]]; then
    echo "ERROR: prompts file does not exist: ${PROMPTS_FILE}" >&2
    exit 2
fi

mkdir -p "${SYNTHETIC_DIR}/logs" "${MULTIRES_DIR}"

echo "Generating ${NUM_SAMPLES} samples on ${NUM_GPUS} GPUs"
echo "Teacher: ${TEACHER_MODEL} at ${PRETRAINED_DIR}"
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    cmd=(python "${SCRIPT_DIR}/generate_training_data.py"
        --teacher-model "${TEACHER_MODEL}"
        --pretrained-dir "${PRETRAINED_DIR}"
        --output-dir "${SYNTHETIC_DIR}"
        --num-samples "${NUM_SAMPLES}"
        --size "${SIZE}"
        --frame-num "${FRAME_NUM}"
        --sampling-steps "${SAMPLING_STEPS}"
        --sample-solver "${SAMPLE_SOLVER}"
        --guide-scale "${GUIDE_SCALE}"
        --shift "${SHIFT}"
        --seed "${SEED}"
        --num-gpus "${NUM_GPUS}"
        --gpu-rank "${gpu}"
        --device 0)
    if [[ -n "${PROMPTS_FILE}" ]]; then
        cmd+=(--prompts-file "${PROMPTS_FILE}")
    fi
    if [[ "${PROJECTION_MODE}" == "latent" ]]; then
        cmd+=(--no-pixel)
    fi
    if [[ "${KEEP_MODEL_RESIDENT}" == "1" ]]; then
        cmd+=(--keep-model-resident)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" \
        >"${SYNTHETIC_DIR}/logs/generate_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "Caching ${PROJECTION_MODE}-projection stage latents"
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [[ "${PROJECTION_MODE}" == "pixel" ]]; then
        echo "WARNING: pixel re-encode caches are for a PyramidalWan baseline or future"
        echo "transform-specific inverse. Current RDD-Wan training intentionally rejects them."
        input_dir="${SYNTHETIC_DIR}/pixel"
        mode_args=(--mode pixel --vae-path "${PRETRAINED_DIR}/Wan2.1_VAE.pth" --device cuda)
    elif [[ "${PROJECTION_MODE}" == "latent" ]]; then
        input_dir="${SYNTHETIC_DIR}/latent"
        mode_args=(--mode latent --device cpu)
    else
        echo "ERROR: PROJECTION_MODE must be pixel or latent" >&2
        exit 2
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/precompute_latents.py" \
        --input-dir "${input_dir}" \
        --output-dir "${MULTIRES_DIR}" \
        "${mode_args[@]}" \
        --stage-spatial-factors 4,2,1 \
        --stage-temporal-factors 4,2,1 \
        --dtype bfloat16 \
        --num-shards "${NUM_GPUS}" \
        --shard-rank "${gpu}" \
        >"${SYNTHETIC_DIR}/logs/precompute_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

count="$(find "${MULTIRES_DIR}" -maxdepth 1 -name '*_multires.pt' | wc -l)"
if [[ "${count}" -lt "${NUM_SAMPLES}" ]]; then
    echo "ERROR: expected ${NUM_SAMPLES} cached records, found ${count}" >&2
    exit 3
fi
echo "Conditional multires data ready: ${MULTIRES_DIR} (${count} records)"
