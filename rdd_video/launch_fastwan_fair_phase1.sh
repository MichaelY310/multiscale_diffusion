#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"

launch() {
  local session="$1" gpu="$2" name="$3" scope="$4"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && env GPU='${gpu}' NAME='${name}' FACTORS='1,2;1,2;1,1' ATTENTION_MODE='vsa' VSA_SPARSITY='0.8' REUSE_EVERY='2' REUSE_SCOPE='${scope}' WARMUP_RUNS='1' TIMED_RUNS='5' bash '${WORKER}'"
  echo "started ${session} on GPU ${gpu}"
}

launch rdd_fair_reuse_backend 0 temporal_vsa20_reuse2_backend backend
launch rdd_fair_reuse_projected 1 temporal_vsa20_reuse2_projected projected

