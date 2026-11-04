#!/usr/bin/env bash
# Generate a small matched-seed RDD-Wan solver/NFE sweep.
# VBench-scale evaluation should use the same matrix on every baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an RDD-Wan .pt file}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
PROMPTS_FILE="${PROMPTS_FILE:-${SCRIPT_DIR}/video_prompts_smoke.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results_rdd_video/solver_sweep}"
DEVICE="${DEVICE:-0}"
SIZE="${SIZE:-832x448}"
FRAME_NUM="${FRAME_NUM:-81}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
MAX_PROMPTS="${MAX_PROMPTS:-8}"

mkdir -p "${OUTPUT_DIR}/logs"
mapfile -t prompts < <(head -n "${MAX_PROMPTS}" "${PROMPTS_FILE}")

# AB2 and Euler are one NFE per update; Heun is two. Endpoint clean-prediction
# calls are printed by the sampler and must be included in reported true NFE.
if [[ -n "${SOLVER_CONFIGS:-}" ]]; then
    IFS=';' read -r -a configs <<<"${SOLVER_CONFIGS}"
else
    configs=(
        "euler:20,20,10"
        "ab2:20,20,10"
        "heun:10,10,5"
        "euler:10,10,5"
        "ab2:10,10,5"
    )
fi

for prompt_idx in "${!prompts[@]}"; do
    prompt="${prompts[$prompt_idx]}"
    seed=$((2026 + prompt_idx))
    for config in "${configs[@]}"; do
        solver="${config%%:*}"
        stages="${config#*:}"
        tag="p$(printf '%03d' "${prompt_idx}")_${solver}_${stages//,/-}"
        python "${SCRIPT_DIR}/sample_rdd_wan.py" \
            --checkpoint "${CHECKPOINT}" \
            --pretrained-dir "${PRETRAINED_DIR}" \
            --prompt "${prompt}" \
            --size "${SIZE}" --frame-num "${FRAME_NUM}" \
            --steps-per-stage "${stages}" \
            --solver "${solver}" --shift "${SHIFT}" \
            --guidance-scale "${GUIDANCE_SCALE}" \
            --seed "${seed}" --device "${DEVICE}" \
            --output "${OUTPUT_DIR}/${tag}.mp4" \
            2>&1 | tee "${OUTPUT_DIR}/logs/${tag}.log"
    done
done
