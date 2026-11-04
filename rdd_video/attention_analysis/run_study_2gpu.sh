#!/usr/bin/env bash
# Run the model probe on physical GPU 0 and the kernel prototype on GPU 1.
# It refuses to start if either GPU is already materially occupied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${RDD_ATTN_ENV_DIR:-${REPO_DIR}/.venv-rdd-attn}"
WAN_REPO_DIR="${WAN_REPO_DIR:-${REPO_DIR}/Wan2.1}"
MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B-transformer-only}"
RDD_CHECKPOINT="${RDD_WAN_CHECKPOINT:-}"
RDD_LATENT="${RDD_WAN_LATENT:-}"
RDD_CONTEXT="${RDD_WAN_CONTEXT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/eval_results/rdd_wan_attention_routing}"
GPU0="${PROBE_GPU:-0}"
GPU1="${BENCH_GPU:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-1024}"

source "${ENV_DIR}/bin/activate"
mkdir -p "${OUTPUT_DIR}/logs"

for gpu in "${GPU0}" "${GPU1}"; do
    used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used > MAX_USED_MIB )); then
        echo "ERROR: physical GPU ${gpu} already uses ${used} MiB; refusing to collide." >&2
        exit 3
    fi
done

checkpoint_args=()
if [[ -n "${RDD_CHECKPOINT}" ]]; then
    checkpoint_args=(--checkpoint "${RDD_CHECKPOINT}")
fi
input_args=()
if [[ -n "${RDD_LATENT}" ]]; then
    input_args+=(--latent-pt "${RDD_LATENT}")
fi
if [[ -n "${RDD_CONTEXT}" ]]; then
    input_args+=(--context-pt "${RDD_CONTEXT}")
fi

CUDA_VISIBLE_DEVICES="${GPU0}" python -u "${SCRIPT_DIR}/probe_rdd_wan_routing.py" \
    --wan-repo "${WAN_REPO_DIR}" \
    --pretrained-dir "${MODEL_DIR}" \
    "${checkpoint_args[@]}" \
    "${input_args[@]}" \
    --device cuda:0 \
    --output-dir "${OUTPUT_DIR}/model_probe" \
    2>&1 | tee "${OUTPUT_DIR}/logs/model_probe.log" &
probe_pid=$!

CUDA_VISIBLE_DEVICES="${GPU1}" python -u "${SCRIPT_DIR}/benchmark_harp_flex.py" \
    --device cuda:0 \
    --output "${OUTPUT_DIR}/kernel_benchmark/benchmark.json" \
    2>&1 | tee "${OUTPUT_DIR}/logs/kernel_benchmark.log" &
bench_pid=$!

set +e
wait "${probe_pid}"
probe_rc=$?
wait "${bench_pid}"
bench_rc=$?
set -e
printf '{"model_probe_exit":%d,"kernel_benchmark_exit":%d}\n' \
    "${probe_rc}" "${bench_rc}" > "${OUTPUT_DIR}/completion.json"
if (( probe_rc != 0 || bench_rc != 0 )); then
    echo "ERROR: model probe exit=${probe_rc}, kernel benchmark exit=${bench_rc}" >&2
    exit 4
fi
echo "RDD-Wan attention study complete: ${OUTPUT_DIR}"
