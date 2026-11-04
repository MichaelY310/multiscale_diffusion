#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-260907}"
CKPT="${CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/eval_results/previews/fdm_rescaling_probe_unified_l_300300400_400k}"
CONFIG="${OUT_ROOT}/original_sp_vp_same_seed.jsonl"
LOG="${OUT_ROOT}/sample_original_sp_vp.log"

test -f "${CKPT}"
test -d "${VAE_PATH}"
mkdir -p "${OUT_ROOT}"

# All three rows use the same checkpoint, label stream, initial Gaussian, and
# boundary Gaussian draws.  SP/VP are deliberately inference-only probes: the
# existing checkpoint was not trained with f-DM's stage-wide rescaling.
printf '%s\n' \
  "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-L/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.5,\"upsample_mode\":\"fresh\",\"fdm_rescale_probe\":\"none\",\"output_dir\":\"${OUT_ROOT}/original\"}" \
  "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-L/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.5,\"upsample_mode\":\"fresh\",\"fdm_rescale_probe\":\"sp\",\"output_dir\":\"${OUT_ROOT}/sp_probe\"}" \
  "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-L/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.5,\"upsample_mode\":\"fresh\",\"fdm_rescale_probe\":\"vp\",\"output_dir\":\"${OUT_ROOT}/vp_probe\"}" \
  > "${CONFIG}"

echo "GPU=${GPU}; samples/mode=${TOTAL_SAMPLES}; batch=${BATCH_SIZE}; seed=${SEED}"
echo "checkpoint=${CKPT}"
echo "outputs=${OUT_ROOT}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${ROOT}/.venv-rdd-train/bin/python" -u 2d/sample_fid_unified.py \
    --config "${CONFIG}" \
    --rank 0 \
    --world-size 1 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --vae-batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --vae-path "${VAE_PATH}" \
    --no-eval \
  2>&1 | tee "${LOG}"

"${ROOT}/.venv-rdd-train/bin/python" 2d/make_fdm_rescaling_comparison_grid.py \
  --original "${OUT_ROOT}/original" \
  --sp "${OUT_ROOT}/sp_probe" \
  --vp "${OUT_ROOT}/vp_probe" \
  --output "${OUT_ROOT}/same_seed_original_sp_vp.png" \
  --max-rows "${TOTAL_SAMPLES}"

echo "Completed: ${OUT_ROOT}/same_seed_original_sp_vp.png"
