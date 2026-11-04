#!/usr/bin/env bash
# Paper-scale data recipe, aligned with PyramidalWan where reproducible:
# 80K synthetic 81-frame 832x448 videos from Wan2.1-T2V-14B, conditioned on
# public Open-Sora-Plan HQ captions. Only latent records are retained.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEACHER_DIR="${TEACHER_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-14B}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/rdd_video_data_paper}"
PROMPTS_FILE="${PROMPTS_FILE:-${DATA_ROOT}/prompts/opensora_hq_train_80k.txt}"
NUM_GPUS="${NUM_GPUS:-8}"
MIN_FREE_GIB="${MIN_FREE_GIB:-750}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"

mkdir -p "${DATA_ROOT}"
free_bytes="$(df --output=avail -B1 "${DATA_ROOT}" | tail -n 1 | tr -d '[:space:]')"
required_bytes="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"
if (( free_bytes < required_bytes )) && [[ "${ALLOW_LOW_DISK}" != "1" ]]; then
  echo "ERROR: paper data path has less than ${MIN_FREE_GIB} GiB free." >&2
  echo "The 80K full + multires latent caches can require roughly 0.6--0.8 TiB." >&2
  echo "Choose a larger DATA_ROOT or explicitly set ALLOW_LOW_DISK=1." >&2
  exit 2
fi

if [[ ! -d "${TEACHER_DIR}" ]]; then
  echo "ERROR: missing Wan2.1-14B teacher: ${TEACHER_DIR}" >&2
  echo "Run: DOWNLOAD_TEACHER_14B=1 bash rdd_video/setup_wan21.sh" >&2
  exit 2
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  python "${SCRIPT_DIR}/build_opensora_prompt_bank.py" \
    --count 80000 \
    --seed 2026 \
    --output "${PROMPTS_FILE}"
fi

line_count="$(grep -cve '^[[:space:]]*$' "${PROMPTS_FILE}")"
if [[ "${line_count}" != "80000" ]]; then
  echo "ERROR: paper prompt bank must contain exactly 80000 non-empty prompts; found ${line_count}" >&2
  exit 2
fi

env \
  PRETRAINED_DIR="${TEACHER_DIR}" \
  TEACHER_MODEL="t2v-14B" \
  DATA_ROOT="${DATA_ROOT}" \
  SYNTHETIC_DIR="${DATA_ROOT}/synthetic_wan14b_80k" \
  MULTIRES_DIR="${DATA_ROOT}/multires_latents_latent_wan14b_80k" \
  PROMPTS_FILE="${PROMPTS_FILE}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_SAMPLES=80000 \
  PROJECTION_MODE=latent \
  SIZE=832x448 \
  FRAME_NUM=81 \
  SAMPLING_STEPS=50 \
  SAMPLE_SOLVER=unipc \
  GUIDE_SCALE=5.0 \
  SHIFT=5.0 \
  SEED=2026 \
  KEEP_MODEL_RESIDENT="${KEEP_MODEL_RESIDENT:-1}" \
  bash "${SCRIPT_DIR}/prepare_rdd_wan_data.sh"
