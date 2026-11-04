#!/usr/bin/env bash
# Evaluate a completed standard-suite sample folder with the official VBench.
# Static filtering is run before the temporal-flickering dimension as required
# by the benchmark.  All other 15 dimensions use the unfiltered folder.
#
# Usage:
#   VIDEOS_DIR=/path/to/vbench_rdd MODEL_NAME=rdd bash rdd_video/evaluate_vbench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VBENCH_DIR="${VBENCH_DIR:-${REPO_DIR}/third_party/VBench}"
VBENCH_ENV="${VBENCH_ENV:-${REPO_DIR}/third_party/vbench_env}"
VIDEOS_DIR="${VIDEOS_DIR:?Set VIDEOS_DIR to a complete standard VBench sample folder}"
MODEL_NAME="${MODEL_NAME:-$(basename "${VIDEOS_DIR}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results_rdd_video_paper/vbench_scores/${MODEL_NAME}}"
EVAL_GPUS="${EVAL_GPUS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${REPO_DIR}/pretrained_models/vbench_cache}"

PYTHON="${VBENCH_ENV}/bin/python"
if [[ ! -x "${PYTHON}" || ! -f "${VBENCH_DIR}/evaluate.py" ]]; then
  echo "ERROR: VBench is not set up; run bash rdd_video/setup_video_eval.sh" >&2
  exit 2
fi
if [[ ! -d "${VIDEOS_DIR}" ]]; then
  echo "ERROR: missing VIDEOS_DIR=${VIDEOS_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${VBENCH_CACHE_DIR}"

dimensions=(
  dynamic_degree subject_consistency background_consistency motion_smoothness
  aesthetic_quality imaging_quality object_class multiple_objects human_action
  color spatial_relationship scene temporal_style appearance_style
  overall_consistency
)

run_eval() {
  local videos="$1"
  local output="$2"
  shift 2
  if [[ "${EVAL_GPUS}" == "1" ]]; then
    "${PYTHON}" "${VBENCH_DIR}/evaluate.py" \
      --videos_path "${videos}" --output_path "${output}" \
      --full_json_dir "${VBENCH_DIR}/vbench/VBench_full_info.json" \
      --mode vbench_standard --dimension "$@"
  else
    "${PYTHON}" -m torch.distributed.run --standalone \
      --nproc_per_node "${EVAL_GPUS}" "${VBENCH_DIR}/evaluate.py" \
      --videos_path "${videos}" --output_path "${output}" \
      --full_json_dir "${VBENCH_DIR}/vbench/VBench_full_info.json" \
      --mode vbench_standard --dimension "$@"
  fi
}

# dynamic_degree is intentionally first: VBench downloads/caches RAFT here,
# and the same checkpoint is then available to the mandatory static filter.
run_eval "${VIDEOS_DIR}" "${OUTPUT_DIR}/standard_15" "${dimensions[@]}"

filter_root="${OUTPUT_DIR}/temporal_flickering_filter"
"${PYTHON}" "${VBENCH_DIR}/static_filter.py" \
  --videos_path "${VIDEOS_DIR}" \
  --result_path "${filter_root}" \
  --filter_scope temporal_flickering

filtered_dir="${filter_root}/filtered_videos"
if [[ ! -d "${filtered_dir}" ]]; then
  echo "ERROR: VBench static filter did not create ${filtered_dir}" >&2
  exit 3
fi
run_eval "${filtered_dir}" "${OUTPUT_DIR}/temporal_flickering" temporal_flickering

revision="$(git -C "${VBENCH_DIR}" rev-parse HEAD)"
printf '%s\n' "${revision}" > "${OUTPUT_DIR}/vbench_revision.txt"
printf '%s\n' "${VIDEOS_DIR}" > "${OUTPUT_DIR}/videos_dir.txt"
echo "VBench evaluation complete: ${OUTPUT_DIR}"

