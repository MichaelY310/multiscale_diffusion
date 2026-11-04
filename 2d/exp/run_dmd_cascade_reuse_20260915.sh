#!/usr/bin/env bash
# Two independent four-GPU jobs; never signals or stops an existing job.
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
SPEED_GATE=${SPEED_GATE:-$ROOT/speed_gate.json}
FAMILY=${1:?high_heavy or balanced}
MODE=${2:-train}
case "$FAMILY" in
  high_heavy) GPU_LIST=0,1,2,3; PORT=30301 ;;
  balanced|balanced_plus) GPU_LIST=4,5,6,7; PORT=30302 ;;
  *) exit 2 ;;
esac
case "$MODE" in smoke|train) ;; *) exit 2 ;; esac
OUT="$ROOT/${FAMILY}"
if [[ "$MODE" == smoke ]]; then OUT="$ROOT/smoke4_${FAMILY}"; fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/launcher.log") 2>&1
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'This exact job already has a launcher'; exit 3; }
mkdir -p "$ROOT/locks"
# Ascending physical-GPU lock order: prevents races with OUR benchmarks/smokes.
# These task-local locks do not touch or claim to control another user's job.
IFS=',' read -r -a physical_gpus <<< "$GPU_LIST"
for physical_gpu in "${physical_gpus[@]}"; do
  exec {gpu_lock_fd}>"$ROOT/locks/gpu${physical_gpu}.lock"
  flock "$gpu_lock_fd"
done
export CUDA_VISIBLE_DEVICES="$GPU_LIST" OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
PY=.venv-rdd-eval/bin/python
if [[ "$MODE" == train ]]; then
  "$PY" -c 'import json,sys; g=json.load(open(sys.argv[1])); assert g["complete"] and g["candidates"][sys.argv[2]]["passed"], "Video 20% gate failed"' "$SPEED_GATE" "$FAMILY"
fi
# Active GPUs are allowed, but the free-memory floor and allocator cap protect
# their existing jobs. If capacity changes, wait; do not evict anyone.
while true; do
  minimum=$(nvidia-smi -i "$GPU_LIST" --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n1)
  ((minimum>=20000)) && break
  echo "Waiting for >=20000MiB free on this job's GPUs; current minimum=$minimum"
  sleep 30
done
common=(--checkpoint "$ROOT/start/generator_start.pt" --output "$OUT" --family "$FAMILY" --global-batch 256)
if [[ "$MODE" == smoke ]]; then
  exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
    2d/attention_analysis/train_dmd_cascade_reuse.py "${common[@]}" --steps 2 --micro-batch 64 --smoke
fi
MICRO_BATCH=${MICRO_BATCH:-64}
exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
  2d/attention_analysis/train_dmd_cascade_reuse.py "${common[@]}" --steps 5000 --micro-batch "$MICRO_BATCH" \
  --speed-gate "$SPEED_GATE" --resume
