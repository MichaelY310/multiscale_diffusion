#!/usr/bin/env bash
# Evaluate only the currently available 50K-multiple checkpoints from the
# Unified-L 400/300/300 training-allocation run.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
CKPT_ROOT="${REPO_ROOT}/checkpoints/RDD-Unified-L-400-300-300/imagenet256-unified/checkpoints"
EVAL_ROOT="${REPO_ROOT}/eval_results/unified_l_400300300_convergence"
SAMPLE_ROOT="${EVAL_ROOT}/samples"
RESULT_ROOT="${EVAL_ROOT}/metrics"
NPZ_ROOT="${EVAL_ROOT}/npz_tmp"
LOG_ROOT="${EVAL_ROOT}/logs"
CONFIG_PATH="${EVAL_ROOT}/available_at_launch.jsonl"
AVAILABILITY_PATH="${EVAL_ROOT}/checkpoint_availability.tsv"
SUMMARY_PATH="${EVAL_ROOT}/metrics_summary.tsv"
SAMPLE_SCRIPT="${REPO_ROOT}/2d/sample_fid_unified.py"
PNG_TO_NPZ="${REPO_ROOT}/2d/png_to_npz.py"
EVALUATOR="${REPO_ROOT}/guided-diffusion/evaluations/evaluator.py"
PLOT_SCRIPT="${REPO_ROOT}/2d/plot_unified_l_fid_curve.py"
COMPARISON_SCRIPT="${REPO_ROOT}/paper/experiments_scripts/plot_unified_l_stage_convergence_comparison.py"
REF_NPZ="${RDD_REF_NPZ:-${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORLD_SIZE="${WORLD_SIZE:-4}"

REQUESTED_STEPS_CSV="${REQUESTED_STEPS_CSV:-50000,100000,150000,200000,250000,300000,350000,400000}"
IFS=',' read -r -a REQUESTED_STEPS <<< "${REQUESTED_STEPS_CSV}"
IFS=',' read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"
declare -a AVAILABLE_STEPS=()

if (( ${#PHYSICAL_GPUS[@]} != WORLD_SIZE )); then
    echo "ERROR: GPU_IDS must contain WORLD_SIZE=${WORLD_SIZE} entries; got ${GPU_IDS}" >&2
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

for required in "${SAMPLE_SCRIPT}" "${PNG_TO_NPZ}" "${EVALUATOR}" "${REF_NPZ}"; do
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
    local result_file="$1" metric
    [[ -s "${result_file}" ]] || return 1
    for metric in "Inception Score" FID sFID Precision Recall; do
        grep -qE "^${metric}:" "${result_file}" || return 1
    done
}

tag_for() {
    local step="$1"
    printf 'unifiedL_400300300_ckpt%06dk_808090_ddpm_cfg1_noisefilling' "$((step / 1000))"
}

# Immutable launch-time snapshot: later checkpoints are intentionally ignored.
: > "${CONFIG_PATH}"
printf 'step\tcheckpoint\tstatus_at_start\n' > "${AVAILABILITY_PATH}"
pending_count=0
for step in "${REQUESTED_STEPS[@]}"; do
    checkpoint="${CKPT_ROOT}/$(printf '%07d.pt' "${step}")"
    if [[ ! -s "${checkpoint}" ]]; then
        printf '%d\t%s\tskipped_missing\n' "${step}" "${checkpoint}" >> "${AVAILABILITY_PATH}"
        continue
    fi
    AVAILABLE_STEPS+=("${step}")
    printf '%d\t%s\tavailable\n' "${step}" "${checkpoint}" >> "${AVAILABILITY_PATH}"
    tag="$(tag_for "${step}")"
    result_file="${RESULT_ROOT}/${tag}.txt"
    if metrics_complete "${result_file}"; then
        echo "SKIP complete metrics: ${tag}"
        continue
    fi
    output_dir="${SAMPLE_ROOT}/${tag}"
    printf '{"model_type":"unified","stage_lengths":"400,300,300","resolutions":"32,16,8","model":"DiT-L/2","ckpt":"%s","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"%s"}\n' \
        "${checkpoint}" "${output_dir}" >> "${CONFIG_PATH}"
    pending_count=$((pending_count + 1))
done

if (( ${#AVAILABLE_STEPS[@]} == 0 )); then
    echo "ERROR: no 50K-multiple checkpoints available in ${CKPT_ROOT}" >&2
    exit 1
fi

echo "============================================================"
echo "Unified-L 400/300/300 launch-time checkpoint evaluation"
echo "Available checkpoints: ${AVAILABLE_STEPS[*]}"
echo "Pending configs: ${pending_count}"
echo "Protocol: DDPM250, [8,16,32]=80/80/90, CFG1.0, noise filling"
echo "Samples/config: ${TOTAL_SAMPLES}; GPUs: ${GPU_IDS}; world size: ${WORLD_SIZE}"
echo "============================================================"

if (( pending_count > 0 )); then
    pids=()
    for rank in $(seq 0 $((WORLD_SIZE - 1))); do
        physical_gpu="${PHYSICAL_GPUS[$rank]}"
        rank_log="${LOG_ROOT}/sample_gpu${physical_gpu}_rank${rank}.log"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" python -u "${SAMPLE_SCRIPT}" \
            --config "${CONFIG_PATH}" \
            --rank "${rank}" --world-size "${WORLD_SIZE}" \
            --total-samples "${TOTAL_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --vae-batch-size "${VAE_BATCH_SIZE}" \
            --seed 0 --vae-path "${VAE_PATH}" --no-eval \
            > "${rank_log}" 2>&1 &
        pids+=("$!")
        echo "Started rank ${rank} on GPU ${physical_gpu}, PID ${pids[-1]} -> ${rank_log}"
    done
    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "ERROR: sampling rank ${index} failed; inspect ${LOG_ROOT}" >&2
            failed=1
        fi
    done
    (( failed == 0 )) || exit 1
fi

for step in "${AVAILABLE_STEPS[@]}"; do
    tag="$(tag_for "${step}")"
    sample_dir="${SAMPLE_ROOT}/${tag}"
    result_file="${RESULT_ROOT}/${tag}.txt"
    temp_result="${result_file}.tmp"
    npz_file="${NPZ_ROOT}/${tag}_samples.npz"
    metrics_complete "${result_file}" && continue

    file_count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [[ "${file_count}" -ne "${TOTAL_SAMPLES}" ]]; then
        echo "ERROR: ${tag} has ${file_count}/${TOTAL_SAMPLES} PNGs" >&2
        exit 1
    fi

    python "${PNG_TO_NPZ}" "${sample_dir}" \
        --num-samples "${TOTAL_SAMPLES}" --output "${npz_file}"
    set +e
    (
        cd "${INCEPTION_DIR}"
        CUDA_VISIBLE_DEVICES="" python -u "${EVALUATOR}" "${REF_NPZ}" "${npz_file}"
    ) 2>&1 | tee "${temp_result}"
    evaluator_status="${PIPESTATUS[0]}"
    set -e
    rm -f -- "${npz_file}"
    if [[ "${evaluator_status}" -ne 0 ]] || ! metrics_complete "${temp_result}"; then
        mv -f -- "${temp_result}" "${result_file}.failed.txt"
        echo "ERROR: incomplete evaluator output for ${tag}" >&2
        exit 1
    fi
    mv -f -- "${temp_result}" "${result_file}"
done

printf 'model\tstep\tinception_score\tfid\tsfid\tprecision\trecall\n' > "${SUMMARY_PATH}"
for step in "${AVAILABLE_STEPS[@]}"; do
    tag="$(tag_for "${step}")"
    result_file="${RESULT_ROOT}/${tag}.txt"
    metrics_complete "${result_file}" || { echo "ERROR: missing complete result ${result_file}" >&2; exit 1; }
    inception="$(awk -F': ' '$1=="Inception Score"{print $2}' "${result_file}")"
    fid="$(awk -F': ' '$1=="FID"{print $2}' "${result_file}")"
    sfid="$(awk -F': ' '$1=="sFID"{print $2}' "${result_file}")"
    precision="$(awk -F': ' '$1=="Precision"{print $2}' "${result_file}")"
    recall="$(awk -F': ' '$1=="Recall"{print $2}' "${result_file}")"
    printf '400300300\t%d\t%s\t%s\t%s\t%s\t%s\n' \
        "${step}" "${inception}" "${fid}" "${sfid}" "${precision}" "${recall}" >> "${SUMMARY_PATH}"
done

python "${PLOT_SCRIPT}" \
    --summary "${SUMMARY_PATH}" \
    --output-prefix "${EVAL_ROOT}/unified_l_400300300_fid_curve" \
    --title "RDD Unified-L 400/300/300 convergence"

THREE_STAGE="${REPO_ROOT}/eval_results/unified_l_300300400_convergence/metrics_summary.tsv"
BOUNDARY="${REPO_ROOT}/eval_results/unified_l_boundary_convergence/metrics_summary.tsv"
TWO_STAGE="${REPO_ROOT}/eval_results/unified_l_2stage_500500_convergence/metrics_summary.tsv"
DIRECT_TWO_STAGE="${REPO_ROOT}/eval_results/unified_l_2stage_8to32_500500_convergence/metrics_summary.tsv"
if [[ -s "${THREE_STAGE}" && -s "${BOUNDARY}" && -s "${TWO_STAGE}" && -s "${DIRECT_TWO_STAGE}" ]]; then
    python "${COMPARISON_SCRIPT}" \
        --three-stage "${THREE_STAGE}" \
        --boundary "${BOUNDARY}" \
        --two-stage "${TWO_STAGE}" \
        --direct-two-stage "${DIRECT_TWO_STAGE}" \
        --four-hundred-first "${SUMMARY_PATH}" \
        --output-dir "${EVAL_ROOT}/comparison_figures"
fi

echo "All launch-time-available evaluations are complete: ${SUMMARY_PATH}"
