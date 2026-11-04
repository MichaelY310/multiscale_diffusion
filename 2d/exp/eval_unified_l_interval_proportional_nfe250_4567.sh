#!/usr/bin/env bash
# Evaluate whether the fixed 80/80/90 sampler favors the 300/300/400
# Unified-L training boundary.  Every experiment keeps DDPM, CFG=1.0,
# noise-filling, 50K samples, and exactly 250 model evaluations.  Only the
# coarse->middle->fine allocation is changed to be proportional to the
# model's coarse->middle->fine diffusion-time interval lengths.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
EVAL_ROOT="${REPO_ROOT}/eval_results/unified_l_interval_proportional_nfe250"
SAMPLE_ROOT="${EVAL_ROOT}/samples"
RESULT_ROOT="${EVAL_ROOT}/metrics"
NPZ_ROOT="${EVAL_ROOT}/npz_tmp"
LOG_ROOT="${EVAL_ROOT}/logs"
CONFIG_PATH="${EVAL_ROOT}/configs.jsonl"
SUMMARY_PATH="${EVAL_ROOT}/metrics_summary.tsv"
SAMPLE_SCRIPT="${REPO_ROOT}/2d/sample_fid_unified.py"
PNG_TO_NPZ="${REPO_ROOT}/2d/png_to_npz.py"
EVALUATOR="${REPO_ROOT}/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="${RDD_REF_NPZ:-${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
WORLD_SIZE=4

IFS=',' read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"
if (( ${#PHYSICAL_GPUS[@]} != WORLD_SIZE )); then
    echo "ERROR: GPU_IDS must contain exactly four GPU ids; got ${GPU_IDS}" >&2
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

# Fields:
# label | fine,middle,coarse training lengths | coarse,middle,fine sample steps
# checkpoint | optimizer step
#
# All four schedules are compared at the common 250K training checkpoint.
# The two best-populated schedules are additionally compared at 400K.
declare -a SPECS=(
    "300300400|300,300,400|100,75,75|${REPO_ROOT}/checkpoints/RDD-Unified-L/0250000.pt|250000"
    "400300300|400,300,300|75,75,100|${REPO_ROOT}/checkpoints/RDD-Unified-L-400-300-300/imagenet256-unified/checkpoints/0250000.pt|250000"
    "100300600|100,300,600|150,75,25|${REPO_ROOT}/checkpoints/RDD-Unified-L-100-300-600/imagenet256-unified/checkpoints/0250000.pt|250000"
    "600300100|600,300,100|25,75,150|${REPO_ROOT}/checkpoints/RDD-Unified-L-600-300-100/imagenet256-unified/checkpoints/0250000.pt|250000"
    "300300400|300,300,400|100,75,75|${REPO_ROOT}/checkpoints/RDD-Unified-L/0400000.pt|400000"
    "400300300|400,300,300|75,75,100|${REPO_ROOT}/checkpoints/RDD-Unified-L-400-300-300/imagenet256-unified/checkpoints/0400000.pt|400000"
)

tag_for() {
    local label="$1" step="$2" sample_steps="$3"
    printf 'unifiedL_%s_ckpt%06dk_%s_ddpm_cfg1_noisefilling' \
        "${label}" "$((step / 1000))" "${sample_steps//,/}"
}

: > "${CONFIG_PATH}"
pending_count=0
for spec in "${SPECS[@]}"; do
    IFS='|' read -r label stage_lengths sample_steps checkpoint step <<< "${spec}"
    [[ -s "${checkpoint}" ]] || { echo "ERROR: missing checkpoint: ${checkpoint}" >&2; exit 1; }
    [[ "$(( $(tr ',' '+' <<< "${sample_steps}") ))" -eq 250 ]] || {
        echo "ERROR: ${label} sampling allocation does not sum to 250: ${sample_steps}" >&2
        exit 1
    }
    tag="$(tag_for "${label}" "${step}" "${sample_steps}")"
    result_file="${RESULT_ROOT}/${tag}.txt"
    if metrics_complete "${result_file}"; then
        echo "SKIP complete metrics: ${tag}"
        continue
    fi
    output_dir="${SAMPLE_ROOT}/${tag}"
    printf '{"model_type":"unified","stage_lengths":"%s","resolutions":"32,16,8","model":"DiT-L/2","ckpt":"%s","num_steps":"%s","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"%s"}\n' \
        "${stage_lengths}" "${checkpoint}" "${sample_steps}" "${output_dir}" >> "${CONFIG_PATH}"
    pending_count=$((pending_count + 1))
done

echo "============================================================"
echo "Unified-L interval-proportional sampler audit"
echo "Pending configs: ${pending_count}/${#SPECS[@]}"
echo "Protocol: DDPM, total NFE=250, CFG=1.0, noise filling"
echo "Samples/config: ${TOTAL_SAMPLES}; GPUs: ${GPU_IDS}; batch/GPU: ${BATCH_SIZE}"
echo "Results: ${RESULT_ROOT}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    python - "${CONFIG_PATH}" "${pending_count}" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
assert len(records) == expected, (len(records), expected)
for record in records:
    steps = [int(value) for value in record["num_steps"].split(",")]
    assert sum(steps) == 250, record
    assert record["sampler"] == "ddpm", record
    assert float(record["cfg_scale"]) == 1.0, record
    assert record["upsample_mode"] == "noise_filling", record
print(f"DRY RUN OK: validated {len(records)} pending configurations in {path}")
PY
    exit 0
fi

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

for spec in "${SPECS[@]}"; do
    IFS='|' read -r label stage_lengths sample_steps checkpoint step <<< "${spec}"
    tag="$(tag_for "${label}" "${step}" "${sample_steps}")"
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

printf 'model\tstep\ttraining_stage_lengths_fine_to_coarse\tsampling_steps_coarse_to_fine\tinception_score\tfid\tsfid\tprecision\trecall\n' > "${SUMMARY_PATH}"
for spec in "${SPECS[@]}"; do
    IFS='|' read -r label stage_lengths sample_steps checkpoint step <<< "${spec}"
    tag="$(tag_for "${label}" "${step}" "${sample_steps}")"
    result_file="${RESULT_ROOT}/${tag}.txt"
    metrics_complete "${result_file}" || { echo "ERROR: missing complete result ${result_file}" >&2; exit 1; }
    inception="$(awk -F': ' '$1=="Inception Score"{print $2}' "${result_file}")"
    fid="$(awk -F': ' '$1=="FID"{print $2}' "${result_file}")"
    sfid="$(awk -F': ' '$1=="sFID"{print $2}' "${result_file}")"
    precision="$(awk -F': ' '$1=="Precision"{print $2}' "${result_file}")"
    recall="$(awk -F': ' '$1=="Recall"{print $2}' "${result_file}")"
    printf '%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${step}" "${stage_lengths}" "${sample_steps}" \
        "${inception}" "${fid}" "${sfid}" "${precision}" "${recall}" >> "${SUMMARY_PATH}"
done

echo "All proportional-allocation evaluations are complete: ${SUMMARY_PATH}"
