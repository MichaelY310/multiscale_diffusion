#!/usr/bin/env bash
# Two evaluations only:
#   1) DDPM 250 steps (80/80/90 in coarse-to-fine sampling order), CFG 1.0
#   2) DDPM 250 steps (80/80/90 in coarse-to-fine sampling order), CFG 1.5
# Both use noise-filling transitions and GPUs 0,1 with batch size 512 per process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_VENV_ACTIVATE="${ROOT_DIR}/.venv-rdd-eval/bin/activate"
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # Set VENV_ACTIVATE when the server environment is not already active.
  # Example: VENV_ACTIVATE=/path/to/venv/bin/activate bash "$0"
  source "${VENV_ACTIVATE}"
elif [[ -f "${DEFAULT_VENV_ACTIVATE}" ]]; then
  source "${DEFAULT_VENV_ACTIVATE}"
else
  echo "ERROR: evaluation environment is missing: ${DEFAULT_VENV_ACTIVATE}" >&2
  echo "Run scripts/download_rdd_multistage_100_300_600_400k.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

GPU_IDS=(0 1)
WORLD_SIZE=2
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
SEED="${SEED:-0}"

CKPT_DIR="${ROOT_DIR}/checkpoints/RDD-Multistage-100-300-600"
CKPT_STEP="0400000"
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

TAG_CFG10="ms100300600_400k_808090_ddpm_cfg1_noisefilling"
TAG_CFG15="ms100300600_400k_808090_ddpm_cfg15_noisefilling"
OUT_CFG10="${SAMPLE_ROOT}/${TAG_CFG10}"
OUT_CFG15="${SAMPLE_ROOT}/${TAG_CFG15}"
CONFIG="${CONFIG_DIR}/ms100300600_400k_808090_ddpm_cfg1_cfg15_noisefilling.jsonl"

mkdir -p "${SAMPLE_ROOT}" "${CONFIG_DIR}" "${LOG_DIR}" "${NPZ_DIR}"

for resolution in 32x32 16x16 8x8; do
  checkpoint="${CKPT_DIR}/checkpoints_${resolution}/${CKPT_STEP}.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "ERROR: missing checkpoint: ${checkpoint}" >&2
    echo "Run scripts/download_rdd_multistage_100_300_600_400k.sh first." >&2
    exit 1
  fi
done

for required_file in "${SAMPLE_SCRIPT}" "${PNG_TO_NPZ}" "${EVALUATOR}" "${REF_NPZ}" "${INCEPTION_MODEL}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: missing required file: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${VAE_DIR}" ]]; then
  echo "ERROR: missing local VAE directory: ${VAE_DIR}" >&2
  exit 1
fi

cat > "${CONFIG}" <<EOF
{"model_type":"multistage","stage_lengths":"100,300,600","resolutions":"32,16,8","model_8":"DiT-B/2","model_16":"DiT-L/2","model_32":"DiT-XL/2","ckpt_dir":"${CKPT_DIR}","ckpt_step":"${CKPT_STEP}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"${OUT_CFG10}"}
{"model_type":"multistage","stage_lengths":"100,300,600","resolutions":"32,16,8","model_8":"DiT-B/2","model_16":"DiT-L/2","model_32":"DiT-XL/2","ckpt_dir":"${CKPT_DIR}","ckpt_step":"${CKPT_STEP}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.5,"upsample_mode":"fresh","output_dir":"${OUT_CFG15}"}
EOF

echo "============================================================"
echo "RDD-Multistage-100-300-600 @ 400K"
echo "GPUs: 0,1 | sampling batch/GPU: ${BATCH_SIZE} | VAE decode batch: ${VAE_BATCH_SIZE} | samples/experiment: ${TOTAL_SAMPLES}"
echo "Experiments: DDPM250 80/80/90 + noise filling, CFG 1.0 and 1.5"
echo "Checkpoints: ${CKPT_DIR}"
echo "Outputs: ${EVAL_ROOT}"
echo "============================================================"

pids=()
for local_rank in 0 1; do
  gpu_id="${GPU_IDS[$local_rank]}"
  rank_log="${LOG_DIR}/sample_ms100300600_400k_gpu${gpu_id}.log"
  echo "Starting sampling rank ${local_rank} on physical GPU ${gpu_id}; log: ${rank_log}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -u "${SAMPLE_SCRIPT}" \
    --config "${CONFIG}" \
    --rank "${local_rank}" \
    --world-size "${WORLD_SIZE}" \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --vae-batch-size "${VAE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --vae-path "${VAE_DIR}" \
    --no-eval \
    >"${rank_log}" 2>&1 &
  pids+=("$!")
done

sampling_failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "ERROR: sampling rank ${index} failed; inspect ${LOG_DIR}/sample_ms100300600_400k_gpu${GPU_IDS[$index]}.log" >&2
    sampling_failed=1
  fi
done
if [[ "${sampling_failed}" != "0" ]]; then
  exit 1
fi

metrics_complete() {
  local result_file="$1" metric
  [[ -s "${result_file}" ]] || return 1
  for metric in "Inception Score" FID sFID Precision Recall; do
    grep -qE "^${metric}:" "${result_file}" || return 1
  done
}

compute_fid() {
  local tag="$1"
  local sample_dir="$2"
  local npz_file="${NPZ_DIR}/${tag}_samples.npz"
  local result_file="${EVAL_ROOT}/${tag}.txt"
  local temp_result="${result_file}.tmp"
  local file_count

  if metrics_complete "${result_file}"; then
    echo "All five metrics already exist: ${result_file}"
    return 0
  fi

  file_count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "${file_count}" -lt "${TOTAL_SAMPLES}" ]]; then
    echo "ERROR: ${tag} has ${file_count}/${TOTAL_SAMPLES} PNG files." >&2
    return 1
  fi

  echo
  echo "Packing ${tag}: ${file_count} PNG files"
  python "${PNG_TO_NPZ}" "${sample_dir}" \
    --num-samples "${TOTAL_SAMPLES}" \
    --output "${npz_file}"

  echo "Computing FID for ${tag}"
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

  echo "Result: ${result_file}"
  grep -E '^(Inception Score|FID|sFID|Precision|Recall):' "${result_file}"

  if [[ "${KEEP_NPZ:-0}" != "1" ]]; then
    rm -f "${npz_file}"
  fi
}

compute_fid "${TAG_CFG10}" "${OUT_CFG10}"
compute_fid "${TAG_CFG15}" "${OUT_CFG15}"

echo
echo "All sampling and FID evaluations completed. Results are under ${EVAL_ROOT}."
