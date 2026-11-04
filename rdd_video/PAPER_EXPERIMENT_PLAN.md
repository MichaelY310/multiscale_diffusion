# RDD-Wan paper experiment protocol

This file is the frozen protocol for the first publishable RDD video result. It
separates the causal test of the residual drift from broad claims about generic
multiscale generation. Do not describe offline teacher-data generation as
few-step distillation: the current method learns ordinary conditional flow
matching from terminal clean latents.

## 1. Research question and minimum causal matrix

The core video question is whether the RDD residual path improves a pretrained
video flow model when the data, architecture, scale schedule, optimizer and
number of updates are fixed. The required arms are:

| Arm | Initialization | Training | Stages | Drift | Purpose |
|---|---|---:|---|---:|---|
| Wan2.1 baseline | official T2V-1.3B | 0 | full resolution | n/a | quality and latency anchor |
| full-resolution FM | same Wan2.1-1.3B | 5K | `1x` | 0 | benefit from teacher data and fine-tuning |
| boundary-only pyramid | same Wan2.1-1.3B | 5K | `4x -> 2x -> 1x` | 0 | isolates multiscale boundaries |
| RDD pyramid | same Wan2.1-1.3B | 5K | `4x -> 2x -> 1x` | 1 | complete method |

RDD versus boundary-only is the decisive novelty comparison. RDD versus
full-resolution FM measures effectiveness but is not exactly compute-matched;
report training GPU-hours and peak memory for all three fine-tuned arms.

## 2. Training data and model

- Student: official `Wan-AI/Wan2.1-T2V-1.3B`, all weights fine-tuned.
- Teacher: official `Wan-AI/Wan2.1-T2V-14B`.
- Teacher corpus: 80,000 synthetic clips at 832x448, 81 frames, generated with
  UniPC-50, CFG 5.0 and shift 5.0.
- Captions: deterministic 80K unique captions from the public Open-Sora-Plan
  v1.1 HQ annotation parts, excluding any supplied evaluation prompts.
- Store the clean Wan VAE latent and umT5 context. Build the lower-resolution
  targets by latent-space average-pool/repeat projections.
- Budget roughly 0.6--0.8 TiB while both the full-resolution source latents and
  multires records are retained. The preparation script refuses to begin below
  750 GiB free unless `ALLOW_LOW_DISK=1` is explicitly set.

PyramidalWan reports an 80K synthetic Wan2.1-14B corpus but does not release
the exact 350K source subset used to select its captions. Therefore this repo's
Open-Sora-Plan HQ prompt bank is a reproducible public proxy, not an exact
reproduction of their private subset.

## 3. Optimization and RDD schedule

- Hardware/budget: 2 H100 GPUs, 5,000 optimizer steps.
- Global batch: 12. Per-GPU micro-batch 1 with six-way gradient accumulation.
- Balanced accumulated batch: exactly 2 coarse, 2 middle and 2 fine examples
  per GPU per optimizer step.
- Precision: bfloat16; optimizer AdamW, betas `(0.9, 0.95)`, weight decay 0.
- Learning rate: `1e-5`, no warmup. EMA decay: `0.9999`.
- Data-time intervals: lengths `60/350/590` out of 1000. Wan uses
  `sigma=1-t`, so the noise boundaries are approximately `0.94` and `0.59`,
  close to PyramidalWan's natural boundary locations.
- Spatiotemporal latent factors: `(4,4)`, `(2,2)`, `(1,1)`.
- Main schedule: Euler `20/20/10` updates. RDD performs one endpoint velocity
  prediction per boundary, so with CFG this is 53 conditional predictions or
  106 conditional/unconditional model calls, not 50/100.
- Strict forward-matched control: Euler `18/19/10`; adding the three endpoint
  predictions gives exactly 50 conditional predictions or 100 model calls,
  matching Wan UniPC-50 with CFG.

The direct `20/20/10` result should be reported because it matches the common
pyramidal schedule. The `18/19/10` result prevents a reviewer from attributing
the gain to six percent more model evaluations.

## 4. Evaluation

Primary automatic evaluation is the official 16-dimension standard VBench
suite, not custom-input VBench. Generate five videos per standard prompt and 25
per temporal-flickering prompt, with identical seeds across all arms. Use
832x448, 81 frames, 16 fps, CFG 5.0 and shift 5.0. Run the official static-video
filter before temporal-flickering evaluation.

Report:

1. all 16 raw VBench dimensions plus official Quality, Semantic and Total;
2. scheduled updates and actual conditional/unconditional model forwards;
3. denoising/boundary time, end-to-end generation time and peak allocated CUDA
   memory after one warmup video per GPU;
4. training GPU-hours and maximum training memory;
5. a matched-seed blinded human preference study if the automatic result is
   strong (at least 100 prompts, three raters, RDD versus each key control).

VBench-2.0 is a useful secondary benchmark after the causal matrix succeeds,
but it is not a replacement for standard VBench and its environment is much
heavier. Do not delay the first decision on RDD to install all 18 VBench-2.0
dimensions.

## 5. Go/no-go sequence

1. Run CPU contracts and a real one-update GPU smoke test.
2. Generate a 2K Wan-1.3B teacher pilot and train RDD/boundary for 500 updates.
3. Sample 32--64 held-out prompts with identical seeds. Continue only if RDD is
   at least neutral on quality and has no boundary flicker regression.
4. Generate the 80K Wan-14B corpus, then train the three 5K arms.
5. Run a 256-prompt solver sweep, choose Euler as the main setting only if it
   is not worse than AB2/Heun at matched actual model-forward count.
6. Generate and evaluate the complete standard VBench matrix.

The smallest convincing paper result is: RDD beats the boundary-only pyramid
and full-resolution FM on VBench quality at lower latency/memory, retains the
advantage at strictly matched model-forward count, and does not degrade motion
smoothness or temporal flicker. If it only beats original Wan but not the two
fine-tuned controls, it is not evidence for the RDD mechanism.

## 6. Reproducible execution

```bash
# Isolated environment, models, Wan source and evaluation environment
DOWNLOAD_TEACHER_14B=1 bash rdd_video/setup_rdd_video_env.sh
source .venv-rdd-video/bin/activate
bash rdd_video/setup_video_eval.sh
RUN_GPU_SMOKE=1 NUM_GPUS=1 bash rdd_video/run_rdd_wan_smoke.sh

# Paper teacher corpus and latent projections (resume-safe)
NUM_GPUS=8 bash rdd_video/prepare_rdd_wan_paper_data.sh

# Run these in separate tmux sessions, one arm at a time on the same 2 GPUs
NUM_GPUS=2 bash rdd_video/train_paper_arm.sh rdd
NUM_GPUS=2 bash rdd_video/train_paper_arm.sh boundary
NUM_GPUS=2 bash rdd_video/train_paper_arm.sh fullres

# Standard VBench generation. Reuse the same manifest for every arm.
NUM_GPUS=8 CHECKPOINT=/path/to/rdd/final.pt \
  OUTPUT_DIR=results_rdd_video_paper/vbench_rdd_2010 \
  bash rdd_video/sample_vbench.sh rdd
NUM_GPUS=8 CHECKPOINT=/path/to/rdd/final.pt STEPS_PER_STAGE=18,19,10 \
  OUTPUT_DIR=results_rdd_video_paper/vbench_rdd_forward_matched \
  bash rdd_video/sample_vbench.sh rdd
NUM_GPUS=8 CHECKPOINT=/path/to/boundary/final.pt \
  OUTPUT_DIR=results_rdd_video_paper/vbench_boundary \
  bash rdd_video/sample_vbench.sh rdd
NUM_GPUS=8 CHECKPOINT=/path/to/fullres/final.pt STEPS_PER_STAGE=50 \
  OUTPUT_DIR=results_rdd_video_paper/vbench_fullres \
  bash rdd_video/sample_vbench.sh rdd
NUM_GPUS=8 OUTPUT_DIR=results_rdd_video_paper/vbench_wan50 \
  bash rdd_video/sample_vbench.sh wan

# Official VBench scores and sampler timing summaries
VIDEOS_DIR=results_rdd_video_paper/vbench_rdd_2010 MODEL_NAME=rdd_2010 \
  bash rdd_video/evaluate_vbench.sh
python rdd_video/aggregate_video_metrics.py \
  results_rdd_video_paper/vbench_rdd_2010/metrics
```

Primary references: [Wan2.1](https://github.com/Wan-Video/Wan2.1),
[PyramidalWan](https://openaccess.thecvf.com/content/CVPR2026/papers/Korzhenkov_PyramidalWan_On_Making_Pretrained_Video_Model_Pyramidal_for_Efficient_Inference_CVPR_2026_paper.pdf),
[Pyramidal Flow](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ab228c4703c4459b1a600ebadc5732c-Abstract-Conference.html),
[VBench](https://github.com/Vchitect/VBench), and
[Open-Sora-Plan](https://arxiv.org/abs/2412.00131).
