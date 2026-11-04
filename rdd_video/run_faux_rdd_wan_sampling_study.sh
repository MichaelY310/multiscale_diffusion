#!/usr/bin/env bash
# Qualitative, zero-finetuning sanity check for RDD-style sampling with Wan2.1.
# This does NOT turn the original Wan checkpoint into an RDD-trained model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${REPO_DIR}/.venv-rdd-video/bin/python}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
FASTVIDEO_KERNEL_ROOT="${FASTVIDEO_KERNEL_ROOT:-${REPO_DIR}/third_party/FastVideo-kernel-ref/fastvideo-kernel}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/eval_results/rdd_wan_faux_sampling/study}"
SIZE="${SIZE:-832x448}"
FRAME_NUM="${FRAME_NUM:-17}"
SEED="${SEED:-20260905}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SOLVER="${SOLVER:-ab2}"
SHIFT="${SHIFT:-1.0}"
FULL_STEPS="${FULL_STEPS:-15}"
RDD_STEPS="${RDD_STEPS:-6,6,3}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

run_one() {
    local name="$1"
    shift
    if [[ -s "${OUTPUT_DIR}/${name}.metrics.json" && -s "${OUTPUT_DIR}/${name}.mp4" ]]; then
        echo "SKIP ${name}: output already complete"
        return
    fi
    echo "============================================================"
    echo "RUN ${name}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" FASTVIDEO_VSA_TRITON=1 \
      "${PYTHON}" rdd_video/sample_rdd_wan.py \
        --use-pretrained-wan \
        --pretrained-dir "${PRETRAINED_DIR}" \
        --prompt "${PROMPT}" \
        --size "${SIZE}" --frame-num "${FRAME_NUM}" \
        --seed "${SEED}" --guidance-scale "${GUIDANCE_SCALE}" \
        --solver "${SOLVER}" --shift "${SHIFT}" \
        --output "${OUTPUT_DIR}/${name}.mp4" \
        --metrics-json "${OUTPUT_DIR}/${name}.metrics.json" \
        --latent-output "${OUTPUT_DIR}/${name}.latent.pt" \
        "$@" \
        2>&1 | tee "${OUTPUT_DIR}/${name}.log"
}

# Same implementation, same prompt and seed.  Only the trajectory/attention
# knobs below change.
run_one fullres_dense \
    --stage-configs "1,1" --stage-lengths "1000" --drift-scale 0 \
    --steps-per-stage "${FULL_STEPS}" --attention-mode dense

run_one boundary_only_dense \
    --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" --drift-scale 0 \
    --steps-per-stage "${RDD_STEPS}" --attention-mode dense

run_one faux_rdd_dense \
    --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" --drift-scale 1 \
    --steps-per-stage "${RDD_STEPS}" --attention-mode dense

run_one faux_rdd_route75 \
    --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" --drift-scale 1 \
    --steps-per-stage "${RDD_STEPS}" --attention-mode route --route-density 0.75 \
    --fastvideo-kernel-root "${FASTVIDEO_KERNEL_ROOT}"

run_one faux_rdd_route75_reuse2 \
    --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" --drift-scale 1 \
    --steps-per-stage "${RDD_STEPS}" --attention-mode route_reuse --route-density 0.75 \
    --reuse-every 2 --fastvideo-kernel-root "${FASTVIDEO_KERNEL_ROOT}"

run_one faux_rdd_route50_reuse2 \
    --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" --drift-scale 1 \
    --steps-per-stage "${RDD_STEPS}" --attention-mode route_reuse --route-density 0.50 \
    --reuse-every 2 --fastvideo-kernel-root "${FASTVIDEO_KERNEL_ROOT}"

"${PYTHON}" rdd_video/summarize_faux_rdd_wan_sampling.py \
    --input-dir "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/summary.json"

echo "Study complete: ${OUTPUT_DIR}"

