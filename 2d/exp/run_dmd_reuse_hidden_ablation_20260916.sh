#!/usr/bin/env bash
# Independent matched A@5000 branches. Does not stop or change any existing job.
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
BASE=/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915
PREVIOUS=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
ROOT=/home/yuheng/rdd_experiments/dmd_reuse_hidden_20260916
VARIANT=${1:?baseline or hidden}
MODE=${2:-smoke}
case "$VARIANT" in
  baseline) HIDDEN_WEIGHT=0; PORT=30631 ;;
  hidden) HIDDEN_WEIGHT=.05; PORT=30632 ;;
  *) echo 'Expected baseline or hidden'; exit 2 ;;
esac
case "$MODE" in smoke|pilot|continuous) ;; *) echo 'Expected smoke, pilot or continuous'; exit 2 ;; esac
if [[ "$MODE" == continuous ]]; then
  [[ "$VARIANT" == hidden ]] || { echo 'Continuous continuation is authorized only for hidden'; exit 2; }
  OUT="$ROOT/pilot_hidden"
  [[ -f "$OUT/checkpoints/latest.pt" ]] || { echo 'Refusing fresh continuous training without own latest checkpoint'; exit 2; }
else
  OUT="$ROOT/${MODE}_${VARIANT}"
fi
mkdir -p "$OUT" "$ROOT/locks"
exec > >(tee -a "$OUT/launcher.log") 2>&1
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'This exact job is already active'; exit 3; }
# Both variants deliberately serialize on 0123; never touch the active 4567 run.
for gpu in 0 1 2 3; do
  exec {lock_fd}>"$ROOT/locks/gpu$gpu.lock"
  flock "$lock_fd"
done
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv-rdd-eval/bin/python
while true; do
  available=$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n1)
  ((available>=32000)) && break
  echo "Waiting for >=32000 MiB free on each GPU (minimum=$available); no other processes are signalled."
  sleep 30
done
common=(--initial "$BASE/start/a5000_training_state.pt"
  --dmd-teacher "$PREVIOUS/start/generator_start.pt"
  --score-teacher checkpoints/RDD-Unified-L/0400000.pt
  --vae checkpoints/pretrained/stabilityai/sd-vae-ft-ema
  --speed-gate "$PREVIOUS/video_spatial_speed/speed_gate.json"
  --output "$OUT" --objective dmd_lpips --global-batch 256 --batch-per-gpu 64
  --memory-fraction .29 --perceptual-chunk 4 --hidden-weight "$HIDDEN_WEIGHT")
if [[ "$MODE" == smoke ]]; then
  exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
    2d/attention_analysis/train_dmd_reuse_objective.py "${common[@]}" \
    --steps 3 --critic-warmup 3 --ramp-steps 1 --hidden-ramp-steps 1 --smoke --resume
fi
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["state"]=="smoke_passed"' \
  "$ROOT/smoke_hidden/progress.json"
# Initial model, optimizer, paired noise stream, critic warm-up and old objective
# are identical in the two branches. Only hidden supervision differs.
TRAIN_STEPS=2000
if [[ "$MODE" == continuous ]]; then
  TRAIN_STEPS=0
  echo 'Explicit unlimited continuation from pilot_hidden/latest.pt; no loss/mask/RNG reset. Ctrl-C or STOP_REQUESTED saves and exits.'
fi
exec "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
  2d/attention_analysis/train_dmd_reuse_objective.py "${common[@]}" \
  --steps "$TRAIN_STEPS" --critic-warmup 200 --ramp-steps 1000 --hidden-ramp-steps 500 \
  --eval-every 500 --holdout-start 500 --latest-only --resume
