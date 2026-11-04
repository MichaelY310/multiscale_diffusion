#!/usr/bin/env bash
# Fast local contracts plus an optional one-update real-Wan GPU integration test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
python -m py_compile \
    rdd_video/flow_matching_video.py \
    rdd_video/train_rdd_wan.py \
    rdd_video/sample_rdd_wan.py \
    rdd_video/generate_training_data.py \
    rdd_video/precompute_latents.py \
    rdd_video/build_opensora_prompt_bank.py \
    rdd_video/build_prompt_dataset.py \
    rdd_video/build_vbench_manifest.py \
    rdd_video/sample_rdd_wan_manifest.py \
    rdd_video/sample_wan_manifest.py \
    rdd_video/aggregate_video_metrics.py
bash -n \
    rdd_video/setup_wan21.sh \
    rdd_video/setup_rdd_video_env.sh \
    rdd_video/setup_video_eval.sh \
    rdd_video/prepare_rdd_wan_data.sh \
    rdd_video/prepare_rdd_wan_paper_data.sh \
    rdd_video/train_rdd_wan.sh \
    rdd_video/train_paper_arm.sh \
    rdd_video/train_causal_matrix.sh \
    rdd_video/sample_vbench.sh \
    rdd_video/sample_solver_sweep.sh \
    rdd_video/sample_wan_baseline_eval.sh \
    rdd_video/evaluate_vbench.sh
python rdd_video/test_rdd_video_cpu.py

if [[ "${RUN_GPU_SMOKE:-0}" != "1" ]]; then
    echo "CPU contracts passed. Set RUN_GPU_SMOKE=1 to load the real Wan checkpoint."
    exit 0
fi

PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results_rdd_video}"
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    rdd_video/train_rdd_wan.py \
    --pretrained-dir "${PRETRAINED_DIR}" \
    --data-mode dummy \
    --stage-configs "4,4,1;2,2,1;1,1,1" \
    --stage-lengths "60,350,590" \
    --stage-sampling proportional \
    --latent-frames 4 --latent-height 16 --latent-width 16 \
    --global-batch-size "${NUM_GPUS}" \
    --max-steps 1 --warmup-steps 0 --ckpt-every 1 --log-every 1 \
    --num-workers 0 \
    --results-dir "${RESULTS_DIR}" \
    --exp-name smoke_real_wan

test -f "${RESULTS_DIR}/smoke_real_wan/checkpoints/0000001.pt"
echo "Real-Wan one-update smoke test passed."
