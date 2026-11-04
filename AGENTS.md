# RDD repository instructions

This Git repository contains both legacy RDD and the `unified_multires/`
framework. Work from the checked-out files; no previous chat, original server,
external workspace symlink, or unpublished handoff is required.

## Start here

Read [the operator guide](docs/AGENT_GUIDE_ZH.md) when setting up, training,
sampling, evaluating, or investigating a historical experiment. It routes to
the exact commands and evidence. Read only the relevant detailed documents.
For changes under `2d/` or `unified_multires/`, also read that directory's
`AGENTS.md`, even when your session started at the root.

| Request | Entry and authoritative documentation |
|---|---|
| Legacy image DDPM/FM | `2d/`, `DiT/`, `diffusion/`; operator guide sections 2–3 |
| Unified training / sampling / new components | [unified instructions](unified_multires/AGENTS.md), [usage](unified_multires/README.md) |
| Reproduce an old RDD checkpoint in unified | [native recipes](unified_multires/configs/rdd_reproduction/README_ZH.md), not a generic RDD demo |
| FM fresh sampling, then ImageNet PF training | [portable experiment bundle](unified_multires/experiments/imagenet256_fm_pf/README_ZH.md), including its evaluation guide |
| Common-model comparisons | `unified_multires/configs/comparison_imagenet256/manifest.json` and [comparison rules](unified_multires/README_ZH.md) |
| Author-native method support / limitations | [support matrix](unified_multires/SUPPORTED_REPRODUCTIONS_ZH.md) and the selected config |
| Historical metrics / old-new parity | `results/metrics.json`, [published evidence](docs/validation/20261104/README.md) |
| Video / 3D / theory | `rdd_video/`, `RDD_DiT_3D/`, `docs/theory/`; research code with separate dependencies and validation scope |

All these source trees live on `main`; do not switch historical branches merely
to access FM or unified. The framework is custom code, not FastVideo. Some
native adapters deliberately reuse the author's model or solver backend.
Legacy “Unified DiT” means one RDD model across resolutions; it is not the name
of the newer multi-paper framework.

## Preserve experiment identity

Keep three versions distinct: (1) author's original repo, (2) unified native
compatibility recipe, (3) controlled common-model transplant. A passing test
of (2) does not certify (3) as the paper's best-result reproduction.

- `fresh` = `refresh`: discard the old residual and add new IID noise around
  the upsampled predicted clean. `noise_filling`: recover residual from the old
  noisy state and predicted clean, preserve its normalized lift, then fill the
  new subspace. `legacy_initial_noise` is a separate historical variant.
- Old DDPM results labeled `noise_filling` actually used **fresh**. Old image
  FM `noise_filling` really preserved recovered residual. Use
  [the naming note](docs/BOUNDARY_NAMES_20261104_ZH.md); never transfer old FM
  metrics to a new fresh run by renaming them.
- Read the clock convention before copying numbers. Legacy resolutions are
  finest first; DDPM `stage_lengths` are finest first; current FM trainer and
  `fm_unified` JSONL `stage_lengths` are **coarsest first**. Sampling step lists
  are coarsest first. FM-XL v2 uses `[0,.3,.6,1]`; not every FM checkpoint does.
- Historical DDPM main sampling used coarse-to-fine `80/80/90`. The comparison
  config with `90/80/80` is another recipe. FM Euler `80/80/90` uses 252 actual
  NFE including boundary calls. FM `dpm2` is a legacy alias for midpoint RK2,
  not DDPM DPM-Solver. Record actual NFE, CFG and timing.
- Common ImageNet comparisons use the same DiT body, VAE, data preprocessing
  and RDD-style aggregation: same images across all stages, mean loss per
  stage, then sum with coefficient one. Retain each selected method's path,
  target, internal weighting/time sampling, position encoding, transition and
  solver. Any departure belongs in the config and result description.
- Common training uses ADM crop/flip, frozen SD-VAE-ft-ema posterior sampling
  on each visit, latent scale `0.18215`. Caching one latent, replacing the VAE,
  changing precision, microbatch/RNG, CFG or sampler is an experiment change.

## Execution and resources

Use an isolated environment. Run commands from the Git root; use relative
paths and explicit local assets. The operator guide has verified entry-point
syntax. Do not blindly execute historical `.sh` launchers: some contain old
machine paths, card assignments or timed monitors.

- Current unified public training/sampling and the FM/PF bundle are single
  process, single device. `world_size` must be 1. Do not wrap them in
  `torchrun` and claim DDP support. Legacy trainers support `torchrun`; the
  legacy sampler uses explicit rank/world-size sharding.
- The user's current task determines resource/download authorization. Prefer
  CPU synthetic checks for code verification. Do not infer authorization to
  start 400K training or large downloads from an example or an archived prompt.
  Conversely, do not ask again for work already authorized by the user.
- No repository-wide host, GPU ID or expired deadline is prescribed. Inspect
  the current machine and use only authorized resources. Do not disturb other
  jobs, shared environments or unrelated FastVideo checkouts.
- Fetch author source snapshots with `unified_multires/tools/fetch_sources.py`
  and `sources.lock.json`; fetch only needed sources. Model/data assets are
  separate. The FM/PF bundle pins its model/VAE assets in `assets.lock.json`.
- Keep runtime data in ignored run/asset directories. Clean your disposable
  tensors, temporary environments and generated test images after retaining
  compact reports. Preserve formal checkpoints, datasets, results and others'
  files. Never force-add weights, source caches, environments or credentials.

## Evidence and maintenance

For numerical parity, specify source/config, weights, input, RNG, dtype,
backend, tested layers/gradients and sampler. Distinguish exact elementwise
equality from tolerance-based agreement; measure errors. Shared backends are
reuse evidence, not independent reimplementations. Small random CPU tests do
not establish 400K-training quality, GPU/BF16 equivalence or a full video result.

Run checks relevant to the change, keep concise reproducible evidence, and
report untested scope plainly. Some archived raw evidence is local-only; cite
the tracked compact reports and say if the original payload is unavailable.
Do not invent missing FID results or treat old theoretical notes as proven
claims. `run_adm_evaluation.sh` in the FM/PF bundle consumes that bundle's
manifested run format, not arbitrary legacy PNG directories.

When an entry point, config schema, naming rule, asset source or supported
capability changes, update the corresponding guide and these routing rules in
the same change. Keep machine-specific live status in the run directory, not
in these persistent instructions. Before an authorized push, inspect the diff
and tracked files; preserve unrelated user edits and avoid rewriting history.
