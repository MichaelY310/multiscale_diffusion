# Legacy image RDD

Read the root `AGENTS.md` and [operator guide](../docs/AGENT_GUIDE_ZH.md),
especially sections 2–3, before changing or launching these entry points.

- `train_RDD_imagenet256_unified.py`: shared-model DDPM trainer.
- `train_RDD_FM_imagenet256_unified.py`: shared-model FM trainer.
- `sample_fid_unified.py`: legacy JSONL sampling for shared/independent DDPM
  and shared FM. Its name does not mean it invokes `umdiff`.
- `../DiT/models_multires.py` and `../diffusion/`: shared native backends used
  by both legacy scripts and the new framework's RDD-native adapters. A change
  here can affect both sides of a parity test.

Training uses `torchrun` and CUDA; `--micro-batch-size` means the **global**
microbatch across ranks. It is not per-GPU batch size. With global batch 256,
global microbatch 32 and four ranks, each rank processes eight images per
forward and accumulates eight times. Verify divisibility and capacity before
launching. Both trainers accept `--vae-path` for an offline local VAE.

For `--resolutions 32,16,8`, DDPM `--stage-lengths 300,300,400` is
fine/middle/coarse. FM `--stage-lengths 300,300,400` is coarse/middle/fine,
giving `[0,.3,.6,1]`. The current sampler's `model_type=fm_unified` follows
this same FM order. Do not silently reverse clocks to match a filename.
Sampler `num_steps` is coarse-to-fine for both.

The legacy sampler defaults to eight ranks and large batches. Supply
`--rank 0 --world-size 1` for single-device work; multi-rank launches must
assign one visible GPU and an explicit rank to each process. Plain `torchrun`
does not fill these argparse options. Use `--no-eval` when you intend to run
the portable ADM setup separately; historical automatic evaluation assumes
old environment paths.

Write `upsample_mode` explicitly. Fresh/refresh and recovered noise_filling
are different boundaries. Historical DDPM noise_filling requires migration
to fresh, or an explicit `boundary_semantics: recovered_v2` for the new
residual-preserving variant. FM `dpm2` means midpoint, not DPM-Solver.

Legacy scripts and shell launchers include exploratory branches, old machine
paths and deprecated experiment names. Preserve historical records; add new
clearly labeled runs instead of relabeling old results. Do not infer a current
launch recipe or resource authorization solely from an old shell script.
