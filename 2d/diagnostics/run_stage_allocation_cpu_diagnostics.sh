#!/usr/bin/env bash
# CPU-only diagnostics for the completed 100/300/600 50K sample archives.
# It waits for the standard full-metric backfill to finish so two TensorFlow
# evaluators do not fight for CPU/RAM.  No CUDA process is launched here.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/eval_results/diagnostics/stage_allocation}"
EVALUATOR="${REPO_ROOT}/guided-diffusion/evaluations/evaluator.py"

mkdir -p "${OUTPUT}/lowpass_metrics" "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

while tmux has-session -t eval_ms100_fullmetrics 2>/dev/null; do
  echo "[$(date -Is)] waiting for eval_ms100_fullmetrics to finish"
  sleep 60
done

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-16}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-4}"

pack_existing_pngs() {
  local tag="$1"
  local source_dir="$2"
  local output_npz="${REPO_ROOT}/eval_results/npz/${tag}_samples.npz"
  if [[ -s "${output_npz}" ]]; then
    echo "[pack] already present: ${output_npz}"
    return
  fi
  local count
  count="$(find "${source_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "${count}" != "50000" ]]; then
    echo "ERROR: ${source_dir} has ${count}/50000 PNGs" >&2
    return 1
  fi
  "${PYTHON}" -u 2d/png_to_npz.py "${source_dir}" \
    --num-samples 50000 --output "${output_npz}"
}

CONTINUED_TAG_CFG1="ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg1_noisefilling"
CONTINUED_TAG_CFG15="ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg15_noisefilling"
pack_existing_pngs "${CONTINUED_TAG_CFG1}" \
  "${REPO_ROOT}/eval_results/samples/${CONTINUED_TAG_CFG1}"
pack_existing_pngs "${CONTINUED_TAG_CFG15}" \
  "${REPO_ROOT}/eval_results/samples/${CONTINUED_TAG_CFG15}"

"${PYTHON}" -u 2d/diagnostics/analyze_image_frequency.py \
  --dataset "ImageNet_original=${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz" \
  --dataset "RDD_100_300_600_cfg1=${REPO_ROOT}/eval_results/npz/ms100300600_400k_808090_ddpm_cfg1_noisefilling_samples.npz" \
  --dataset "RDD_100_300_600_cfg1p5=${REPO_ROOT}/eval_results/npz/ms100300600_400k_808090_ddpm_cfg15_noisefilling_samples.npz" \
  --dataset "RDD_100_300_600_8x8continued600K_cfg1=${REPO_ROOT}/eval_results/npz/${CONTINUED_TAG_CFG1}_samples.npz" \
  --dataset "RDD_100_300_600_8x8continued600K_cfg1p5=${REPO_ROOT}/eval_results/npz/${CONTINUED_TAG_CFG15}_samples.npz" \
  --reference-label ImageNet_original \
  --output-dir "${OUTPUT}" \
  --max-images 10000 \
  --batch-size 32 \
  --seed 20260814 \
  --fft-bins 64 \
  --write-lowpass-npz \
  2>&1 | tee "${REPO_ROOT}/logs/stage_allocation_frequency.log"

for scale in 64 128; do
  "${PYTHON}" -u 2d/diagnostics/evaluate_many_npz.py \
    --repo-root "${REPO_ROOT}" \
    --reference "${OUTPUT}/lowpass_npz/ImageNet_original_lowpass${scale}_10000.npz" \
    --sample "RDD_100_300_600_cfg1=${OUTPUT}/lowpass_npz/RDD_100_300_600_cfg1_lowpass${scale}_10000.npz" \
    --sample "RDD_100_300_600_cfg1p5=${OUTPUT}/lowpass_npz/RDD_100_300_600_cfg1p5_lowpass${scale}_10000.npz" \
    --sample "RDD_100_300_600_8x8continued600K_cfg1=${OUTPUT}/lowpass_npz/RDD_100_300_600_8x8continued600K_cfg1_lowpass${scale}_10000.npz" \
    --sample "RDD_100_300_600_8x8continued600K_cfg1p5=${OUTPUT}/lowpass_npz/RDD_100_300_600_8x8continued600K_cfg1p5_lowpass${scale}_10000.npz" \
    --output-dir "${OUTPUT}/lowpass_metrics" \
    --name-suffix "lowpass${scale}_10k" \
    2>&1 | tee "${REPO_ROOT}/logs/stage_allocation_lowpass${scale}.log"
  for sample_label in \
      RDD_100_300_600_cfg1 \
      RDD_100_300_600_cfg1p5 \
      RDD_100_300_600_8x8continued600K_cfg1 \
      RDD_100_300_600_8x8continued600K_cfg1p5; do
    log="${OUTPUT}/lowpass_metrics/${sample_label}_lowpass${scale}_10k.txt"
    for metric in "Inception Score:" "FID:" "sFID:" "Precision:" "Recall:"; do
      grep -q "^${metric}" "${log}" || {
        echo "ERROR: ${log} is missing ${metric}" >&2
        exit 1
      }
    done
  done
done

echo "[$(date -Is)] CPU diagnostics complete: ${OUTPUT}"
