#!/usr/bin/env bash
# Never terminates any other job. Control and DMD each have four disjoint GPUs.
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915
PREVIOUS=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
OBJECTIVE=${1:?control or dmd_lpips}
MODE=${2:-train}
case "$OBJECTIVE" in
  control) GPU_LIST=0,1,2,3; PORT=30411; MEMORY_FRACTION=.22; FLOOR=19000 ;;
  dmd_lpips) GPU_LIST=4,5,6,7; PORT=30412; MEMORY_FRACTION=.29; FLOOR=32000 ;;
  *) exit 2 ;;
esac
case "$MODE" in smoke|train) ;; *) exit 2 ;; esac
OUT="$ROOT/$OBJECTIVE"
if [[ "$MODE" == smoke ]]; then OUT="$ROOT/smoke_$OBJECTIVE"; fi
mkdir -p "$OUT" "$ROOT/locks"
exec > >(tee -a "$OUT/launcher.log") 2>&1
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'This exact launcher is already active'; exit 3; }
IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
for gpu in "${gpu_ids[@]}"; do
  exec {lock_fd}>"$ROOT/locks/gpu$gpu.lock"
  flock "$lock_fd"
done
export CUDA_VISIBLE_DEVICES="$GPU_LIST" OMP_NUM_THREADS=2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv-rdd-eval/bin/python
while true; do
  available=$(nvidia-smi -i "$GPU_LIST" --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n1)
  ((available>=FLOOR)) && break
  echo "Waiting for $FLOOR MiB free (minimum=$available); existing jobs are untouched."
  sleep 30
done
common=(--initial "$ROOT/start/a5000_training_state.pt"
  --dmd-teacher "$PREVIOUS/start/generator_start.pt"
  --score-teacher checkpoints/RDD-Unified-L/0400000.pt
  --vae checkpoints/pretrained/stabilityai/sd-vae-ft-ema
  --speed-gate "$PREVIOUS/video_spatial_speed/speed_gate.json"
  --output "$OUT" --objective "$OBJECTIVE" --global-batch 256
  --batch-per-gpu "${BATCH_PER_GPU:-64}" --memory-fraction "$MEMORY_FRACTION"
  --perceptual-chunk "${PERCEPTUAL_CHUNK:-4}")
if [[ "$MODE" == smoke ]]; then
  exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
    2d/attention_analysis/train_dmd_reuse_objective.py "${common[@]}" --steps "${SMOKE_STEPS:-3}" --critic-warmup 3 --ramp-steps 1 --smoke --resume
fi
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["state"]=="smoke_passed"' "$ROOT/smoke_$OBJECTIVE/progress.json"
exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
  2d/attention_analysis/train_dmd_reuse_objective.py "${common[@]}" --steps 30000 --critic-warmup 200 --ramp-steps 1000 --resume
