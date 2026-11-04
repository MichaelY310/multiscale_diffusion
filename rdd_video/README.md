# RDD-Wan experiment entry points

For the frozen paper-scale configuration, causal matrix, exact forward-count
controls and end-to-end commands, read
[PAPER_EXPERIMENT_PLAN.md](PAPER_EXPERIMENT_PLAN.md). The commands below are
the cheaper pilot path.

Current RDD-FM uses Wan noise time `sigma=1-t` and a latent-space idempotent
average-pool/repeat projection. Pixel downsampling followed by VAE re-encoding
is nonlinear and is not accepted by the RDD training entry point until a
transform-specific clean reconstruction method is implemented.

Run in this order:

```bash
# 1. Isolated video environment, official Wan source/dependencies and 1.3B.
#    Add DOWNLOAD_TEACHER_14B=1 for the paper-scale data recipe.
bash rdd_video/setup_rdd_video_env.sh
source .venv-rdd-video/bin/activate

# Official standard VBench code, isolated evaluation environment and manifest
bash rdd_video/setup_video_eval.sh

# 2. Static contracts (add RUN_GPU_SMOKE=1 for a real one-update Wan test)
bash rdd_video/run_rdd_wan_smoke.sh

# 3. Build a disjoint prompt bank, then generate offline conditional teacher
#    data and the multires latent cache.
python rdd_video/build_prompt_dataset.py
NUM_GPUS=8 NUM_SAMPLES=2000 \
PROMPTS_FILE=rdd_video_data/prompts/train_prompts.txt \
bash rdd_video/prepare_rdd_wan_data.sh

# 4. Minimal causal matrix. The pretrained Wan checkpoint is the fourth,
#    zero-finetuning baseline.
NUM_GPUS=2 bash rdd_video/train_causal_matrix.sh rdd
NUM_GPUS=2 bash rdd_video/train_causal_matrix.sh boundary
NUM_GPUS=2 bash rdd_video/train_causal_matrix.sh fullres

# Original pretrained Wan arm on held-out prompts (no fine-tuning)
NUM_GPUS=8 bash rdd_video/sample_wan_baseline_eval.sh

# 5. Matched-seed solver/NFE pilot
CHECKPOINT=/path/to/final.pt bash rdd_video/sample_solver_sweep.sh
```

For the primary RDD-versus-boundary comparison, sample the two checkpoints with
the same held-out prompts and a single identical configuration:

```bash
PROMPTS_FILE=rdd_video_data/prompts/pilot_eval_prompts.txt MAX_PROMPTS=256 \
SOLVER_CONFIGS='ab2:20,20,10' CHECKPOINT=/path/to/rdd/final.pt \
OUTPUT_DIR=results_rdd_video/eval_rdd bash rdd_video/sample_solver_sweep.sh

PROMPTS_FILE=rdd_video_data/prompts/pilot_eval_prompts.txt MAX_PROMPTS=256 \
SOLVER_CONFIGS='ab2:20,20,10' CHECKPOINT=/path/to/boundary/final.pt \
OUTPUT_DIR=results_rdd_video/eval_boundary bash rdd_video/sample_solver_sweep.sh

# The one-stage full-resolution control uses the same 50 scheduled updates.
PROMPTS_FILE=rdd_video_data/prompts/pilot_eval_prompts.txt MAX_PROMPTS=256 \
SOLVER_CONFIGS='ab2:50' CHECKPOINT=/path/to/fullres/final.pt \
OUTPUT_DIR=results_rdd_video/eval_fullres bash rdd_video/sample_solver_sweep.sh
```

For a main data run, set `PROMPTS_FILE`; the 50 built-in prompts are smoke-only.
All expensive paths and settings can be overridden with environment variables.
Sampling writes a `.metrics.json` beside every video with scheduled updates,
actual conditional/unconditional model forward calls, per-stage calls, latency,
and peak CUDA memory.

Paper-scale sampling uses `sample_vbench.sh`, which loads each model once per
GPU instead of once per video. `evaluate_vbench.sh` runs the official 15 regular
dimensions plus the required static-filtered temporal-flickering dimension.
Use `aggregate_video_metrics.py` on each output's `metrics/` directory. The
`20/20/10` RDD schedule has three endpoint predictions (53 predictions total);
also report `18/19/10`, which is exactly forward-matched to Wan UniPC-50.

`online_synthetic` (and its deprecated alias `distill`) only generates terminal
teacher samples and applies ordinary RDD-FM. It is not few-step distillation and
is deliberately not used by the main scripts.

The P0 experiment has four arms: original Wan, full-resolution FM fine-tuning,
the same three-stage pyramid with `drift_scale=0`, and RDD with
`drift_scale=1`. RDD versus the drift-free arm is the clean novelty test because
data, model, resolutions, stage sampling, update count, and optimizer are
identical. Full-resolution fine-tuning is an effectiveness/compute control and
must report GPU-hours rather than being described as compute-matched.

For evaluation, use identical held-out prompts, seeds, CFG, output size, frame
count, and actual model-forward budget. Report VBench dimensions, temporal
flicker/warping error, CLIP text alignment, wall-clock, peak memory, and a
blinded human preference study. First run 8--16 held-out prompts as a smoke
pilot; only launch the full VBench suite if RDD beats the drift-free arm.

See [RDD_NOVELTY_FM_VIDEO_AUDIT.md](../docs/research_docs/RDD_NOVELTY_FM_VIDEO_AUDIT.md)
for the derivation, limitations, baseline matrix, and acceptance-oriented plan.
