#!/usr/bin/env bash
# RDD-FM Unified XL/2 v2 @ 400K, original ImageNet.
#
# Experiments:
#   1) Euler 80/80/90, CFG 1.0, noise filling  (250 updates, 252 NFE)
#   2) Euler 80/80/90, CFG 1.5, noise filling  (250 updates, 252 NFE)
#   3) Midpoint RK2 40/40/45, CFG 1.5, noise filling
#      (125 updates, 252 NFE; compute-matched to Euler)
#
# DPM-Solver is not mathematically the right name for a generic FM velocity
# ODE. Midpoint RK2 is the meaningful second-order counterpart and mirrors the
# paper's diffusion comparison of DDPM-250 against DPM2-125 at matched NFE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_VENV_ACTIVATE="${ROOT_DIR}/.venv-rdd-eval/bin/activate"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
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

CKPT="${ROOT_DIR}/checkpoints/RDD-FM-Unified-XL-v2/0400000.pt"
VAE_DIR="${ROOT_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema"
EVAL_ROOT="${ROOT_DIR}/eval_results"
SAMPLE_ROOT="${EVAL_ROOT}/samples"
CONFIG_DIR="${EVAL_ROOT}/configs"
LOG_DIR="${EVAL_ROOT}/logs"
NPZ_DIR="${EVAL_ROOT}/npz"

SAMPLE_SCRIPT="${ROOT_DIR}/2d/sample_fid_unified.py"
CPU_TEST="${ROOT_DIR}/2d/test_rdd_fm_sampler_cpu.py"
PNG_TO_NPZ="${ROOT_DIR}/2d/png_to_npz.py"
EVALUATOR="${ROOT_DIR}/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="${REF_NPZ:-${EVAL_ROOT}/reference/VIRTUAL_imagenet256_labeled.npz}"
INCEPTION_MODEL="${EVAL_ROOT}/reference/classify_image_graph_def.pb"

TAG_EULER10="rddfm_unixl_v2_400k_808090_euler_cfg1_noisefilling"
TAG_EULER15="rddfm_unixl_v2_400k_808090_euler_cfg15_noisefilling"
TAG_RK2_15="rddfm_unixl_v2_400k_404045_midpoint_cfg15_noisefilling"
OUT_EULER10="${SAMPLE_ROOT}/${TAG_EULER10}"
OUT_EULER15="${SAMPLE_ROOT}/${TAG_EULER15}"
OUT_RK2_15="${SAMPLE_ROOT}/${TAG_RK2_15}"
CONFIG="${CONFIG_DIR}/rddfm_unixl_v2_400k_main.jsonl"

mkdir -p "${SAMPLE_ROOT}" "${CONFIG_DIR}" "${LOG_DIR}" "${NPZ_DIR}"

for required_file in \
  "${CKPT}" "${SAMPLE_SCRIPT}" "${CPU_TEST}" "${PNG_TO_NPZ}" \
  "${EVALUATOR}" "${REF_NPZ}" "${INCEPTION_MODEL}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: missing required file: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${VAE_DIR}" ]]; then
  echo "ERROR: missing local VAE directory: ${VAE_DIR}" >&2
  exit 1
fi

python "${CPU_TEST}"

cat > "${CONFIG}" <<EOF
{"model_type":"fm_unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"euler","cfg_scale":1.0,"upsample_mode":"noise_filling","output_dir":"${OUT_EULER10}"}
{"model_type":"fm_unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"euler","cfg_scale":1.5,"upsample_mode":"noise_filling","output_dir":"${OUT_EULER15}"}
{"model_type":"fm_unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"40,40,45","sampler":"midpoint","cfg_scale":1.5,"upsample_mode":"noise_filling","output_dir":"${OUT_RK2_15}"}
EOF

echo "============================================================"
echo "RDD-FM Unified XL/2 v2 @ 400K"
echo "GPUs: 0,1 | batch/GPU: ${BATCH_SIZE} | samples/experiment: ${TOTAL_SAMPLES}"
echo "Euler250 CFG1.0 + CFG1.5; midpoint-RK2 125 updates CFG1.5"
echo "All configurations use 252 actual velocity evaluations and noise filling."
echo "Checkpoint: ${CKPT}"
echo "Outputs: ${EVAL_ROOT}"
echo "============================================================"

pids=()
for local_rank in 0 1; do
  gpu_id="${GPU_IDS[$local_rank]}"
  rank_log="${LOG_DIR}/sample_rddfm_unixl_v2_400k_gpu${gpu_id}.log"
  echo "Starting rank ${local_rank} on GPU ${gpu_id}; log: ${rank_log}"
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
    echo "ERROR: rank ${index} failed; inspect ${LOG_DIR}/sample_rddfm_unixl_v2_400k_gpu${GPU_IDS[$index]}.log" >&2
    sampling_failed=1
  fi
done
[[ "${sampling_failed}" == "0" ]] || exit 1

compute_fid() {
  local tag="$1"
  local sample_dir="$2"
  local npz_file="${NPZ_DIR}/${tag}_samples.npz"
  local result_file="${EVAL_ROOT}/${tag}.txt"
  local temp_result="${result_file}.tmp.$$"
  local failed_result="${result_file}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
  local file_count

  metrics_complete() {
    local path="$1"
    local metric
    [[ -s "${path}" ]] || return 1
    for metric in "Inception Score" FID sFID Precision Recall; do
      grep -Eq "^${metric}:[[:space:]]*[-+0-9.eE]+[[:space:]]*$" "${path}" || return 1
    done
  }

  if metrics_complete "${result_file}"; then
    echo "All five metrics already exist: ${result_file}"
    grep -E '^(Inception Score|FID|sFID|Precision|Recall):' "${result_file}"
    return 0
  fi

  file_count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "${file_count}" -lt "${TOTAL_SAMPLES}" ]]; then
    echo "ERROR: ${tag} has ${file_count}/${TOTAL_SAMPLES} PNG files." >&2
    return 1
  fi

  python "${PNG_TO_NPZ}" "${sample_dir}" \
    --num-samples "${TOTAL_SAMPLES}" \
    --output "${npz_file}"
  rm -f "${temp_result}"
  if ! (
    cd "$(dirname "${INCEPTION_MODEL}")"
    CUDA_VISIBLE_DEVICES="" python -u "${EVALUATOR}" "${REF_NPZ}" "${npz_file}"
  ) 2>&1 | tee "${temp_result}"; then
    mv "${temp_result}" "${failed_result}"
    echo "ERROR: evaluator failed; incomplete output preserved at ${failed_result}" >&2
    return 1
  fi
  if ! metrics_complete "${temp_result}"; then
    mv "${temp_result}" "${failed_result}"
    echo "ERROR: evaluator did not produce IS/FID/sFID/Precision/Recall; output preserved at ${failed_result}" >&2
    return 1
  fi
  mv "${temp_result}" "${result_file}"
  echo "Result: ${result_file}"
  grep -E '^(Inception Score|FID|sFID|Precision|Recall):' "${result_file}"

  if [[ "${KEEP_NPZ:-0}" != "1" ]]; then
    rm -f "${npz_file}"
  fi
}

compute_fid "${TAG_EULER10}" "${OUT_EULER10}"
compute_fid "${TAG_EULER15}" "${OUT_EULER15}"
compute_fid "${TAG_RK2_15}" "${OUT_RK2_15}"

echo "All RDD-FM evaluations completed. Results are under ${EVAL_ROOT}."
