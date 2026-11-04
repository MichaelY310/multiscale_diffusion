# Unified framework instructions

The repository root `AGENTS.md` applies. Read [README.md](README.md) for
commands and [README_ZH.md](README_ZH.md) for the comparison protocol.

## Configuration and implementation

- Raw method JSON goes to `python -m umdiff configure|train|sample|decode` as
  appropriate. Complete `format=umdiff_experiment_v1` JSON goes to
  `python -m umdiff.experiment plan|export|train|sample`.
- `configs/rdd_reproduction/` preserves legacy native RDD; `paper_recipes/`
  covers selected author-native settings; `comparison_imagenet256/` is an
  explicit common-model comparison. Copy a matching existing recipe. Do not
  replace native FM with generic `process=rdd` and claim historical parity.
- Model, path, projection, target, boundary and sampler are separate registered
  components, but methods do not all have the same config keys. Validate the
  actual schema; do not silently ignore unsupported options.

| Files in `umdiff/` | Responsibility |
|---|---|
| `cli.py`, `experiment.py` | Public commands and experiment-budget translation |
| `registry.py`, `engine.py` | Component construction and composition |
| `paths.py`, `multires_process.py`, `batches.py` | Probability paths and training examples |
| `grids.py`, `operators.py` | Spatial/temporal grids and projections |
| `targets.py`, `boundaries.py` | Prediction targets and stage transitions |
| `solvers.py`, `samplers.py`, `grid_sampling.py` | Integration and stage orchestration |
| `native_models.py`, `rdd_native.py`, `rdd_fm_native.py`, `rdd_multistage.py` | Native adapters, including parent-repo RDD backends |
| `training.py`, `input_transforms.py` | Updates/EMA/resume and preprocessing/local VAE |
| `locations.py` | Repository/source discovery; no hardcoded checkout directory name |

Images use BCHW, videos BCTHW. `wan_grid` is a common test backbone, not the
real pretrained Wan pipeline. Image VAE `decode` is not a video decoder.
See [custom component example](examples/custom_components.py) before extending
the registry. Preserve local/offline asset loading during model execution.

## Runtime semantics

- Public CLI is single-device. Experiment `world_size` is 1; internal
  distributed helpers do not establish public multi-GPU training or sampling.
- `umdiff train --steps` is the number of **additional** optimizer updates.
  `umdiff.experiment train` resumes to the configured total budget instead.
- Sampling uses EMA by default; `--raw-weights` explicitly changes it. Native
  sampler budgets are in the config/checkpoint; generic `sample --steps` is
  rejected for native samplers. Do not report a flag that was never applied.
- General `sample --count` is one batch, not a streaming 50K evaluator. Use
  the FM/PF bundle for its resumable batched image experiment. Its PF sampler
  currently supports CFG1 only.
- Legacy checkpoint importers create an inference envelope, not a resume of
  the old optimizer/data cursor. Keep strict key/shape/clock checks. Independent
  stage models require explicit resolution-to-checkpoint mapping.
- `activate_cpu.sh` hides GPUs and sets `UMDIFF_CPU_ONLY=1`; `activate.sh`
  configures imports/offline use without hiding GPUs. Do not carry CPU-only
  environment flags into an authorized GPU launch.

## Verification and files

For the FM/PF bundle, use the CPU test command in its
[README](experiments/imagenet256_fm_pf/README_ZH.md); the test uses random tiny
DiT/VAE weights and needs no pretrained download. For other changes, select
relevant tests from `tests/`; some author-native tests require optional
dependencies and snapshots from `tools/fetch_sources.py`. Missing assets are
not passing tests. Run conflicting author imports in separate processes.

Pinned sources and licences: [sources.lock.json](sources.lock.json),
[SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md). Numerical scope:
[SUPPORTED_REPRODUCTIONS_ZH.md](SUPPORTED_REPRODUCTIONS_ZH.md),
[published RDD evidence](../docs/validation/20261104/README.md),
[monorepo validation](evidence/monorepo_20261104/REPORT_ZH.md).
Historical large raw evidence, `sources/`, `_deps/`, `runs/`, models and local
environments are not all in Git. Do not force-add them or claim they came with
the clone. Keep compact provenance/error reports when cleaning test payloads.
