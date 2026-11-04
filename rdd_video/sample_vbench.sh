#!/usr/bin/env bash
# Generate the official VBench suite with one long-lived process per GPU.
# Usage:
#   CHECKPOINT=/path/final.pt bash rdd_video/sample_vbench.sh rdd
#   bash rdd_video/sample_vbench.sh wan

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARM="${1:-}"
VBENCH_DIR="${VBENCH_DIR:-${REPO_DIR}/third_party/VBench}"
MANIFEST="${MANIFEST:-${REPO_DIR}/rdd_video_data/vbench/vbench_manifest.jsonl}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"

if [[ ! -f "${VBENCH_DIR}/vbench/VBench_full_info.json" ]]; then
  echo "ERROR: VBench is not set up at ${VBENCH_DIR}" >&2
  echo "Run: bash rdd_video/setup_video_eval.sh" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  python "${SCRIPT_DIR}/build_vbench_manifest.py" \
    --vbench-dir "${VBENCH_DIR}" \
    --output "${MANIFEST}" \
    --base-seed 2026
fi

case "${ARM}" in
  rdd)
    CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an RDD/boundary/fullres final.pt}"
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results_rdd_video_paper/vbench_rdd}"
    ;;
  wan)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results_rdd_video_paper/vbench_wan50}"
    ;;
  *)
    echo "Usage: bash ${BASH_SOURCE[0]} {rdd|wan}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}/logs"
cat >"${OUTPUT_DIR}/sampling_config.txt" <<EOF
arm=${ARM}
checkpoint=${CHECKPOINT:-none}
pretrained_dir=${PRETRAINED_DIR}
manifest=${MANIFEST}
num_gpus=${NUM_GPUS}
size=${SIZE:-832x448}
frame_num=${FRAME_NUM:-81}
solver=${SOLVER:-euler}
steps_per_stage=${STEPS_PER_STAGE:-20,20,10}
wan_solver=${SAMPLE_SOLVER:-unipc}
wan_steps=${SAMPLING_STEPS:-50}
guidance_scale=${GUIDANCE_SCALE:-5.0}
shift=${SHIFT:-5.0}
EOF

rdd_extra=()
if [[ "${KEEP_TEXT_ENCODER_RESIDENT:-1}" == "1" ]]; then
  rdd_extra+=(--keep-text-encoder-resident)
fi
if [[ "${OFFLOAD_MODEL_FOR_DECODE:-0}" == "1" ]]; then
  rdd_extra+=(--offload-model-for-decode)
fi
wan_extra=()
if [[ "${WAN_T5_CPU:-0}" == "1" ]]; then
  wan_extra+=(--t5-cpu)
fi
if [[ "${WAN_OFFLOAD_MODEL:-0}" == "1" ]]; then
  wan_extra+=(--offload-model)
fi

pids=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
  gpu=$((GPU_OFFSET + rank))
  if [[ "${ARM}" == "rdd" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/sample_rdd_wan_manifest.py" \
      --checkpoint "${CHECKPOINT}" \
      --pretrained-dir "${PRETRAINED_DIR}" \
      --manifest "${MANIFEST}" \
      --output-dir "${OUTPUT_DIR}" \
      --size "${SIZE:-832x448}" --frame-num "${FRAME_NUM:-81}" \
      --steps-per-stage "${STEPS_PER_STAGE:-20,20,10}" \
      --solver "${SOLVER:-euler}" \
      --guidance-scale "${GUIDANCE_SCALE:-5.0}" --shift "${SHIFT:-5.0}" \
      --num-shards "${NUM_GPUS}" --shard-rank "${rank}" --device 0 \
      "${rdd_extra[@]}" \
      >"${OUTPUT_DIR}/logs/gpu${gpu}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/sample_wan_manifest.py" \
      --pretrained-dir "${PRETRAINED_DIR}" \
      --manifest "${MANIFEST}" \
      --output-dir "${OUTPUT_DIR}" \
      --model t2v-1.3B \
      --size "${SIZE:-832x448}" --frame-num "${FRAME_NUM:-81}" \
      --sampling-steps "${SAMPLING_STEPS:-50}" --sample-solver "${SAMPLE_SOLVER:-unipc}" \
      --guidance-scale "${GUIDANCE_SCALE:-5.0}" --shift "${SHIFT:-5.0}" \
      --num-shards "${NUM_GPUS}" --shard-rank "${rank}" --device 0 \
      "${wan_extra[@]}" \
      >"${OUTPUT_DIR}/logs/gpu${gpu}.log" 2>&1 &
  fi
  pids+=("$!")
  echo "Started ${ARM} VBench shard ${rank}/${NUM_GPUS} on GPU ${gpu}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "ERROR: at least one VBench generation shard failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit 3
fi

expected="$(grep -cve '^[[:space:]]*$' "${MANIFEST}")"
actual="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "${actual}" -lt "${expected}" ]]; then
  echo "ERROR: expected ${expected} videos, found ${actual}" >&2
  exit 3
fi
echo "VBench generation complete: ${OUTPUT_DIR} (${actual} videos)"
