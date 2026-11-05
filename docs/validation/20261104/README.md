# RDD CPU conformance evidence

These reports were produced in the migrated research workspace on 2026-11-04,
before publication of this source snapshot. Source hashes, test scope and
temporary-payload cleanup are retained. The new `umdiff` framework is now in
`unified_multires/` inside this repository, alongside the old RDD implementation
and its archived comparison evidence. Integration did not rerun these historical
reports; see the separate monorepo validation report for migration checks.

| Report | Scope |
|---|---|
| [FM_SHARED.json](FM_SHARED.json) | Three shared-model training variants; 16 complete FM sampling trajectories; preprocessing/random VAE |
| [FM_MULTISTAGE.json](FM_MULTISTAGE.json) | Three independent DiTs, four training updates and eight complete sampling trajectories; checkpoint import |
| [DDPM_SAMPLING.json](DDPM_SAMPLING.json) | 40 DDPM/DPM sampling cases plus 12 video boundary cases |
| [DDPM_MULTISTAGE_TRAINING.json](DDPM_MULTISTAGE_TRAINING.json) | Independent-model training, gradients, Adam/EMA and resume |
| [DDPM_SHARED_CURRENT.json](DDPM_SHARED_CURRENT.json) | Current shared DDPM small-network training recheck |
| [FULL_XL_20261012.json](FULL_XL_20261012.json) | Previously executed complete XL CPU forward/backward test; not rerun on publication |
| [IMPORT_SHARED.json](IMPORT_SHARED.json) | Shared DDPM/FM, EMA/model import and actual sampling CLI |
| [IMPORT_MULTISTAGE.json](IMPORT_MULTISTAGE.json) | Independent DDPM import, square-string resolution and clock validation |
| [BOUNDARY_INTERFACES.json](BOUNDARY_INTERFACES.json) | Standalone fresh/filling components and invalid endpoint guards |
| [PUBLISH_VALIDATION.json](PUBLISH_VALIDATION.json) | Syntax, source-content and CPU core checks executed in this Git checkout |
| [FM_DDPM_FRESH_AUDIT_ZH.md](FM_DDPM_FRESH_AUDIT_ZH.md) | Subsequent CPU boundary-algebra audit: same fresh operation, different actual DDPM/FM noise clocks; no new quality run |
| [FM_CHECKPOINT_IDENTITY_AUDIT_ZH.md](FM_CHECKPOINT_IDENTITY_AUDIT_ZH.md) | Actual HF checkpoint metadata, historical XL-v2 identity, user-supplied remote EMA tensor audit, and current CPU sampler recheck; remote fresh/filling control reported; no local 50K rerun |

Native backbone and per-stage solver code is reused; stage orchestration and
training updates are compared separately. New trajectory tests use random
nonzero small networks with the actual latent grids and sampler budgets.
Module activations and gradients are checked, not every internal ATen operation.
There is no new 400K training run, 50K FID reproduction or full video-system
certificate implied by these reports.

Naming audit (November 4–5): [registry and guide](../../RDD_EXPERIMENT_NAMES_ZH.md),
[CPU grid/NFE trace](SAMPLING_STEP_ORDER_CALLS.json),
[archived original JSONL check](SAMPLING_STEP_ORDER_ARCHIVE.json),
[configuration/alias checks](NAMING_VALIDATION.json),
[website browser checks](NAMING_SITE_BROWSER.json), and
[HF rename/hash receipt](HF_NAMING_MIGRATION.json).
This cleanup does not rerun quality measurements or change training/sampling mathematics.
