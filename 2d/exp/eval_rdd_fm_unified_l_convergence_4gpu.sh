#!/usr/bin/env bash
# Evaluate one corrected RDD-FM Unified-L allocation at every 50K checkpoint.
#
# The public allocation name is written in DDPM resolution order
# (fine/middle/coarse).  RDD-FM uses chronological data-time order
# (coarse/middle/fine), so the stage lengths are mirrored here exactly as in
# train_RDD_FM_unified_L_allocation_4gpu.sh.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
DDPM_ALLOCATION="${DDPM_ALLOCATION:?set DDPM_ALLOCATION to 300,300,400 or 400,300,300}"
GPU_IDS="${GPU_IDS:?set exactly four physical GPU ids, for example 0,1,2,3}"

case "${DDPM_ALLOCATION}" in
    300,300,400)
        RUN_TAG="300300400"
        RUN_SUFFIX="300-300-400"
        FM_STAGE_LENGTHS="400,300,300"
        ;;
    400,300,300)
        RUN_TAG="400300300"
        RUN_SUFFIX="400-300-300"
        FM_STAGE_LENGTHS="300,300,400"
        ;;
    *)
        echo "ERROR: unsupported DDPM_ALLOCATION=${DDPM_ALLOCATION}" >&2
        exit 1
        ;;
esac

CKPT_ROOT="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-${RUN_SUFFIX}/imagenet256-fm-unified/checkpoints"
EVAL_ROOT="${REPO_ROOT}/eval_results/rdd_fm_unified_l_corrected_${RUN_TAG}_convergence"
SAMPLE_ROOT="${EVAL_ROOT}/samples"
RESULT_ROOT="${EVAL_ROOT}/metrics"
NPZ_ROOT="${EVAL_ROOT}/npz_tmp"
LOG_ROOT="${EVAL_ROOT}/logs"
SUMMARY_PATH="${EVAL_ROOT}/metrics_summary.tsv"
SAMPLE_SCRIPT="${REPO_ROOT}/2d/sample_fid_unified.py"
PNG_TO_NPZ="${REPO_ROOT}/2d/png_to_npz.py"
EVALUATOR="${REPO_ROOT}/guided-diffusion/evaluations/evaluator.py"
PLOT_SCRIPT="${REPO_ROOT}/2d/plot_unified_l_fid_curve.py"
REF_NPZ="${RDD_REF_NPZ:-${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
KEEP_SAMPLES="${KEEP_SAMPLES:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
WORLD_SIZE=4
STEPS=(50000 100000 150000 200000 250000 300000 350000 400000)

IFS=',' read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"
if (( ${#PHYSICAL_GPUS[@]} != WORLD_SIZE )); then
    echo "ERROR: GPU_IDS must contain exactly four ids; got ${GPU_IDS}" >&2
    exit 1
fi

mkdir -p "${SAMPLE_ROOT}" "${RESULT_ROOT}" "${NPZ_ROOT}" "${LOG_ROOT}"

if [[ -f "${REPO_ROOT}/.venv-rdd-eval/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: evaluation environment not found below ${REPO_ROOT}" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT

for required in "${SAMPLE_SCRIPT}" "${PNG_TO_NPZ}" "${EVALUATOR}" "${PLOT_SCRIPT}" "${REF_NPZ}"; do
    [[ -s "${required}" ]] || { echo "ERROR: missing required file: ${required}" >&2; exit 1; }
done
[[ -d "${VAE_PATH}" ]] || { echo "ERROR: missing local VAE: ${VAE_PATH}" >&2; exit 1; }

if [[ -s "${REPO_ROOT}/eval_results/reference/classify_image_graph_def.pb" ]]; then
    INCEPTION_DIR="${REPO_ROOT}/eval_results/reference"
elif [[ -s "${REPO_ROOT}/classify_image_graph_def.pb" ]]; then
    INCEPTION_DIR="${REPO_ROOT}"
else
    echo "ERROR: classify_image_graph_def.pb is missing" >&2
    exit 1
fi

metrics_complete() {
    local path="$1" metric
    [[ -s "${path}" ]] || return 1
    for metric in "Inception Score" FID sFID Precision Recall; do
        grep -qE "^${metric}:[[:space:]]*[-+0-9.eE]+[[:space:]]*$" "${path}" || return 1
    done
}

tag_for() {
    local step="$1"
    printf 'rddfm_unifiedL_corrected_%s_ckpt%06dk_808090_euler_cfg1_noisefilling' \
        "${RUN_TAG}" "$((step / 1000))"
}

cleanup_intermediates() {
    local sample_dir="$1" npz_file="$2"
    rm -f -- "${npz_file}"
    if [[ "${KEEP_SAMPLES}" != "1" && "${sample_dir}" == "${SAMPLE_ROOT}/"* ]]; then
        find "${sample_dir}" -maxdepth 1 -type f -name '*.png' -delete 2>/dev/null || true
        rmdir -- "${sample_dir}" 2>/dev/null || true
    fi
}

available_gb() {
    df -Pk "${REPO_ROOT}" | awk 'NR==2 {printf "%d\n", $4 / 1024 / 1024}'
}

for step in "${STEPS[@]}"; do
    checkpoint="${CKPT_ROOT}/$(printf '%07d.pt' "${step}")"
    [[ -s "${checkpoint}" ]] || { echo "ERROR: missing checkpoint ${checkpoint}" >&2; exit 1; }
done

echo "============================================================"
echo "RDD-FM Unified-L ${DDPM_ALLOCATION} convergence evaluation"
echo "FM chronological stage lengths: ${FM_STAGE_LENGTHS}"
echo "Protocol: Euler 80/80/90 (250 updates, 252 NFE), CFG 1.0, noise filling"
echo "Checkpoints: 50K through 400K; samples/checkpoint: ${TOTAL_SAMPLES}"
echo "GPUs: ${GPU_IDS}; batch/GPU: ${BATCH_SIZE}; keep samples: ${KEEP_SAMPLES}"
echo "Results: ${EVAL_ROOT}"
echo "============================================================"

for step in "${STEPS[@]}"; do
    tag="$(tag_for "${step}")"
    checkpoint="${CKPT_ROOT}/$(printf '%07d.pt' "${step}")"
    sample_dir="${SAMPLE_ROOT}/${tag}"
    result_file="${RESULT_ROOT}/${tag}.txt"
    npz_file="${NPZ_ROOT}/${tag}_samples.npz"
    config_path="${EVAL_ROOT}/${tag}.jsonl"

    if metrics_complete "${result_file}"; then
        echo "SKIP complete metrics: ${tag}"
        cleanup_intermediates "${sample_dir}" "${npz_file}"
        continue
    fi

    free_gb="$(available_gb)"
    if (( free_gb < MIN_FREE_GB )); then
        echo "ERROR: only ${free_gb} GiB free; need at least ${MIN_FREE_GB} GiB" >&2
        exit 1
    fi

    printf '{"model_type":"fm_unified","stage_lengths":"%s","resolutions":"32,16,8","model":"DiT-L/2","ckpt":"%s","num_steps":"80,80,90","sampler":"euler","cfg_scale":1.0,"upsample_mode":"noise_filling","output_dir":"%s"}\n' \
        "${FM_STAGE_LENGTHS}" "${checkpoint}" "${sample_dir}" > "${config_path}"

    echo "[$(date -Is)] Sampling ${tag}; free disk=${free_gb} GiB"
    pids=()
    for rank in $(seq 0 $((WORLD_SIZE - 1))); do
        physical_gpu="${PHYSICAL_GPUS[$rank]}"
        rank_log="${LOG_ROOT}/${tag}_gpu${physical_gpu}_rank${rank}.log"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" python -u "${SAMPLE_SCRIPT}" \
            --config "${config_path}" \
            --rank "${rank}" --world-size "${WORLD_SIZE}" \
            --total-samples "${TOTAL_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --vae-batch-size "${VAE_BATCH_SIZE}" \
            --seed 0 --vae-path "${VAE_PATH}" --no-eval \
            > "${rank_log}" 2>&1 &
        pids+=("$!")
        echo "Started rank ${rank} on GPU ${physical_gpu}, PID ${pids[-1]}"
    done

    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "ERROR: sampling rank ${index} failed; inspect ${LOG_ROOT}" >&2
            failed=1
        fi
    done
    (( failed == 0 )) || exit 1

    file_count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [[ "${file_count}" -ne "${TOTAL_SAMPLES}" ]]; then
        echo "ERROR: ${tag} has ${file_count}/${TOTAL_SAMPLES} PNGs" >&2
        exit 1
    fi

    python "${PNG_TO_NPZ}" "${sample_dir}" \
        --num-samples "${TOTAL_SAMPLES}" --output "${npz_file}"

    temp_result="${result_file}.tmp.$$"
    set +e
    (
        cd "${INCEPTION_DIR}"
        CUDA_VISIBLE_DEVICES="" python -u "${EVALUATOR}" "${REF_NPZ}" "${npz_file}"
    ) 2>&1 | tee "${temp_result}"
    evaluator_status="${PIPESTATUS[0]}"
    set -e

    if [[ "${evaluator_status}" -ne 0 ]] || ! metrics_complete "${temp_result}"; then
        mv -f -- "${temp_result}" "${result_file}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
        echo "ERROR: evaluator failed or returned incomplete metrics for ${tag}" >&2
        exit 1
    fi
    mv -f -- "${temp_result}" "${result_file}"
    cleanup_intermediates "${sample_dir}" "${npz_file}"
    echo "[$(date -Is)] Completed ${tag}"
done

printf 'model\tstep\tinception_score\tfid\tsfid\tprecision\trecall\n' > "${SUMMARY_PATH}"
for step in "${STEPS[@]}"; do
    tag="$(tag_for "${step}")"
    result_file="${RESULT_ROOT}/${tag}.txt"
    metrics_complete "${result_file}" || { echo "ERROR: missing complete ${result_file}" >&2; exit 1; }
    inception="$(awk -F': ' '$1=="Inception Score"{print $2}' "${result_file}")"
    fid="$(awk -F': ' '$1=="FID"{print $2}' "${result_file}")"
    sfid="$(awk -F': ' '$1=="sFID"{print $2}' "${result_file}")"
    precision="$(awk -F': ' '$1=="Precision"{print $2}' "${result_file}")"
    recall="$(awk -F': ' '$1=="Recall"{print $2}' "${result_file}")"
    printf 'fm_%s\t%d\t%s\t%s\t%s\t%s\t%s\n' \
        "${RUN_TAG}" "${step}" "${inception}" "${fid}" "${sfid}" "${precision}" "${recall}" \
        >> "${SUMMARY_PATH}"
done

python "${PLOT_SCRIPT}" \
    --summary "${SUMMARY_PATH}" \
    --output-prefix "${EVAL_ROOT}/rdd_fm_unified_l_${RUN_TAG}_fid_curve" \
    --title "RDD-FM Unified-L ${DDPM_ALLOCATION} convergence"

echo "All eight checkpoints complete: ${SUMMARY_PATH}"

