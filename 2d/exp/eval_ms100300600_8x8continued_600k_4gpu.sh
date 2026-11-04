#!/usr/bin/env bash
# Paired diagnostic for the weak 100/300/600 result.
# Keep original 16x16/32x32 at 400K and replace only 8x8 by continued 600K.
# Sampling remains DDPM250 80/80/90, noise filling, CFG 1.0 and 1.5.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_VENV_ACTIVATE="${ROOT_DIR}/.venv-rdd-eval/bin/activate"
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
elif [[ -f "${DEFAULT_VENV_ACTIVATE}" ]]; then
  source "${DEFAULT_VENV_ACTIVATE}"
else
  echo "ERROR: evaluation environment is missing: ${DEFAULT_VENV_ACTIVATE}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${#GPU_IDS[@]}"
if (( WORLD_SIZE != 4 )); then
  echo "ERROR: expected exactly four GPU ids; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
SEED="${SEED:-0}"

ORIGINAL_DIR="${ROOT_DIR}/checkpoints/RDD-Multistage-100-300-600"
CONTINUED_DIR="${ROOT_DIR}/checkpoints/RDD-Multistage-100-300-600-8x8-continued-from400k/imagenet256-multiscale/checkpoints_8x8"
CKPT_16="${ORIGINAL_DIR}/checkpoints_16x16/0400000.pt"
CKPT_32="${ORIGINAL_DIR}/checkpoints_32x32/0400000.pt"
CKPT_8="${CKPT_8:-${CONTINUED_DIR}/0600000.pt}"

if [[ ! -s "${CKPT_8}" ]]; then
  echo "ERROR: required continued 8x8 endpoint is missing: ${CKPT_8}" >&2
  echo "Available checkpoints:" >&2
  find "${CONTINUED_DIR}" -maxdepth 1 -type f -name '*.pt' -print 2>/dev/null | sort -V >&2 || true
  exit 1
fi
for checkpoint in "${CKPT_16}" "${CKPT_32}"; do
  [[ -s "${checkpoint}" ]] || { echo "ERROR: missing original checkpoint: ${checkpoint}" >&2; exit 1; }
done

VAE_DIR="${ROOT_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema"
EVAL_ROOT="${ROOT_DIR}/eval_results"
SAMPLE_ROOT="${EVAL_ROOT}/samples"
CONFIG_DIR="${EVAL_ROOT}/configs"
LOG_DIR="${EVAL_ROOT}/logs"
NPZ_DIR="${EVAL_ROOT}/npz"
SAMPLE_SCRIPT="${ROOT_DIR}/2d/sample_fid_unified.py"
PNG_TO_NPZ="${ROOT_DIR}/2d/png_to_npz.py"
EVALUATOR="${ROOT_DIR}/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="${REF_NPZ:-${EVAL_ROOT}/reference/VIRTUAL_imagenet256_labeled.npz}"
INCEPTION_MODEL="${EVAL_ROOT}/reference/classify_image_graph_def.pb"

for required in "${SAMPLE_SCRIPT}" "${PNG_TO_NPZ}" "${EVALUATOR}" "${REF_NPZ}" "${INCEPTION_MODEL}"; do
  [[ -s "${required}" ]] || { echo "ERROR: missing required file: ${required}" >&2; exit 1; }
done
[[ -d "${VAE_DIR}" ]] || { echo "ERROR: missing local VAE: ${VAE_DIR}" >&2; exit 1; }

TAG_BASE="ms100300600_8x8continued600k_16x32orig400k_808090_ddpm"
TAG_CFG10="${TAG_BASE}_cfg1_noisefilling"
TAG_CFG15="${TAG_BASE}_cfg15_noisefilling"
OUT_CFG10="${SAMPLE_ROOT}/${TAG_CFG10}"
OUT_CFG15="${SAMPLE_ROOT}/${TAG_CFG15}"
CONFIG="${CONFIG_DIR}/${TAG_BASE}_cfg1_cfg15_noisefilling.jsonl"
MIXED_DIR="${EVAL_ROOT}/mixed_checkpoints/${TAG_BASE}"

mkdir -p "${SAMPLE_ROOT}" "${CONFIG_DIR}" "${LOG_DIR}" "${NPZ_DIR}" \
  "${MIXED_DIR}/checkpoints_8x8" "${MIXED_DIR}/checkpoints_16x16" "${MIXED_DIR}/checkpoints_32x32"

# Symlink-only checkpoint view: no multi-GB copies.
ln -sfn "${CKPT_8}" "${MIXED_DIR}/checkpoints_8x8/mixed.pt"
ln -sfn "${CKPT_16}" "${MIXED_DIR}/checkpoints_16x16/mixed.pt"
ln -sfn "${CKPT_32}" "${MIXED_DIR}/checkpoints_32x32/mixed.pt"

cat > "${CONFIG}" <<EOF
{"model_type":"multistage","stage_lengths":"100,300,600","resolutions":"32,16,8","model_8":"DiT-B/2","model_16":"DiT-L/2","model_32":"DiT-XL/2","ckpt_dir":"${MIXED_DIR}","ckpt_step":"mixed","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"${OUT_CFG10}"}
{"model_type":"multistage","stage_lengths":"100,300,600","resolutions":"32,16,8","model_8":"DiT-B/2","model_16":"DiT-L/2","model_32":"DiT-XL/2","ckpt_dir":"${MIXED_DIR}","ckpt_step":"mixed","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.5,"upsample_mode":"fresh","output_dir":"${OUT_CFG15}"}
EOF

echo "============================================================"
echo "100/300/600 8x8-training diagnostic"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}; ranks: ${WORLD_SIZE}; batch/rank: ${BATCH_SIZE}"
echo "8x8 continued 600K: ${CKPT_8}"
echo "16x16 original 400K: ${CKPT_16}"
echo "32x32 original 400K: ${CKPT_32}"
echo "Sampling: DDPM250 80/80/90, noise filling, CFG 1.0 + 1.5"
echo "============================================================"

pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  gpu_id="${GPU_IDS[$rank]}"
  rank_log="${LOG_DIR}/sample_${TAG_BASE}_gpu${gpu_id}.log"
  echo "Starting rank ${rank} on physical GPU ${gpu_id}; ${rank_log}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -u "${SAMPLE_SCRIPT}" \
    --config "${CONFIG}" --rank "${rank}" --world-size "${WORLD_SIZE}" \
    --total-samples "${TOTAL_SAMPLES}" --batch-size "${BATCH_SIZE}" \
    --vae-batch-size "${VAE_BATCH_SIZE}" --seed "${SEED}" \
    --vae-path "${VAE_DIR}" --no-eval >"${rank_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "ERROR: sampling rank ${index} failed." >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

metrics_complete() {
  local result_file="$1" metric
  [[ -s "${result_file}" ]] || return 1
  for metric in "Inception Score" FID sFID Precision Recall; do
    grep -qE "^${metric}:" "${result_file}" || return 1
  done
}

compute_fid() {
  local tag="$1" sample_dir="$2"
  local npz_file="${NPZ_DIR}/${tag}_samples.npz"
  local result_file="${EVAL_ROOT}/${tag}.txt"
  local temp_result="${result_file}.tmp"
  local count
  if metrics_complete "${result_file}"; then
    echo "All five metrics already exist: ${result_file}"
    return 0
  fi
  count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if (( count < TOTAL_SAMPLES )); then
    echo "ERROR: ${tag} has ${count}/${TOTAL_SAMPLES} PNGs." >&2
    return 1
  fi
  python "${PNG_TO_NPZ}" "${sample_dir}" --num-samples "${TOTAL_SAMPLES}" --output "${npz_file}"
  (
    cd "$(dirname "${INCEPTION_MODEL}")"
    CUDA_VISIBLE_DEVICES="" python -u "${EVALUATOR}" "${REF_NPZ}" "${npz_file}"
  ) 2>&1 | tee "${temp_result}"
  if ! metrics_complete "${temp_result}"; then
    mv -f "${temp_result}" "${result_file}.failed.txt"
    echo "ERROR: evaluator did not produce IS/FID/sFID/Precision/Recall for ${tag}." >&2
    return 1
  fi
  mv -f "${temp_result}" "${result_file}"
  grep -E '^(Inception Score|FID|sFID|Precision|Recall):' "${result_file}"
  [[ "${KEEP_NPZ:-0}" == "1" ]] || rm -f "${npz_file}"
}

compute_fid "${TAG_CFG10}" "${OUT_CFG10}"
compute_fid "${TAG_CFG15}" "${OUT_CFG15}"
echo "Both paired diagnostic evaluations completed."
