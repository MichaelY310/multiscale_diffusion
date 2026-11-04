#!/usr/bin/env bash
# Matched, sequential architecture experiment. Does not stop any existing job.
set -euo pipefail
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
export RDD_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RDD_CSRA_NPROC=4
export RDD_CSRA_MODEL=DiT-L/2
export RDD_CSRA_RANK=64
export RDD_CSRA_LAYERS=13,15,17,20
export RDD_CSRA_GLOBAL_BATCH=256
export RDD_CSRA_MICRO_BATCH=256
export RDD_CSRA_TRAINING_MODE=on_policy_layerwise
export RDD_CKPT="${ROOT}/checkpoints/RDD-Unified-L/0400000.pt"
export RDD_CSRA_MAX_STEPS="${RDD_CSRA_MAX_STEPS:-3000}"
export RDD_CSRA_CHECKPOINT_EVERY="${RDD_CSRA_CHECKPOINT_EVERY:-1000}"
INITIAL="${ROOT}/checkpoints/RDD-CSRA-OnPolicy-Unified-L-400K-r64-subset4-continue10k/0010000.pt"
RUN_TAG="${RUN_TAG:-20260912}"
[[ "${RUN_TAG}" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ -f "${INITIAL}" ]] || { echo "Missing ${INITIAL}" >&2; exit 2; }
cd "${ROOT}"
mkdir -p logs
exec 9>"logs/csra_context_pair_${RUN_TAG}.lock"
flock -n 9 || { echo "This pair is already running" >&2; exit 2; }

# A repeated launcher must not overwrite or silently resume an existing arm.
for arm in v2 v1control; do
  target="${ROOT}/checkpoints/RDD-CSRA-context-${arm}-L400K-r64-subset4-${RUN_TAG}"
  [[ ! -e "${target}" ]] || { echo "Output already exists: ${target}" >&2; exit 2; }
done

for arm in v2 v1control; do
  export RDD_CSRA_ARCHITECTURE=legacy_v1
  [[ "${arm}" != v2 ]] || export RDD_CSRA_ARCHITECTURE=fine_context_v2
  export RDD_CSRA_OUT="${ROOT}/checkpoints/RDD-CSRA-context-${arm}-L400K-r64-subset4-${RUN_TAG}"
  export RDD_CSRA_EXTRA_ARGS="--init-adapter ${INITIAL} --reseed-training-stream --seed 20260912"
  printf 'Starting %s, architecture=%s, 4 GPUs x 64; new optimizer; %s additional updates\n' \
    "${arm}" "${RDD_CSRA_ARCHITECTURE}" "${RDD_CSRA_MAX_STEPS}"
  bash 2d/exp/train_rdd_csra_unified_xl_400k_4gpu.sh \
    2>&1 | tee "logs/train_csra_context_${arm}_${RUN_TAG}.log"
  [[ -f "${RDD_CSRA_OUT}/latest.pt" ]] || exit 3
  printf 'Completed %s\n' "${arm}"
done
