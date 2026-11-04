#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
SUITE_OVERRIDE="${ROOT}/rdd_video/experiments/fastwan_csbr_aggressive.json"
DECODE_OVERRIDE="spatial421_rdd_vsa20,csbr_uniform6_s075,csbr_uniform12_s100,csbr_uniform18_s075,csbr_uniform24_s075"
export SUITE_OVERRIDE DECODE_OVERRIDE

WARMUP=5 TIMED=20 bash "${ROOT}/rdd_video/launch_fastwan_csbr_verify.sh" \
  2 aggressive832 832x448 \
  "A red fox walks slowly through a snowy forest while snow falls, cinematic camera movement."

WARMUP=5 TIMED=20 bash "${ROOT}/rdd_video/launch_fastwan_csbr_verify.sh" \
  3 aggressive768 1280x768 \
  "A red fox walks slowly through a snowy forest while snow falls, cinematic camera movement."
