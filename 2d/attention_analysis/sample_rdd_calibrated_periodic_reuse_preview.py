#!/usr/bin/env python3
"""Real RDD rollout for calibrated, periodically refreshed cross-scale reuse.

This is deliberately a small diagnostic, not an FID claim.  A frozen
per-channel affine map is fitted on a disjoint ImageNet calibration split.
During sampling the final 16x16 block residual is upsampled and aligned, then
used to replace selected 32x32 blocks either on every fine NFE or periodically.
The periodic variants execute exact fine blocks between reuse calls, limiting
the accumulation seen with stage-wide hard replacement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_hybrid_residual_predictor import (  # noqa: E402
    calibrate,
    low_prediction,
    make_diffusions as make_full_diffusions,
)
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    load_rdd,
    manual_forward,
    upsample_tokens,
)
from sample_rdd_stagewide_block_reuse_preview import (  # noqa: E402
    decode,
    make_diffusions,
    ManualBaselineModel,
    relative_l2,
    sample_once,
    save_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from diffusers.models import AutoencoderKL  # noqa: E402


class CalibratedPeriodicReuseModel:
    """Use calibrated coarse residuals on a controlled subset of fine NFEs."""

    def __init__(
        self,
        model: torch.nn.Module,
        coefficients: list[dict[str, torch.Tensor]],
        layers: list[int],
        period: int,
        phase: int = 0,
    ) -> None:
        self.model = model
        self.coefficients = coefficients
        self.layers = layers
        self.period = period
        self.phase = phase
        self.cached: dict[int, torch.Tensor] = {}
        self.fine_calls = 0
        self.reused_calls = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if resolution == 16 and int(t[0].item()) == 300:
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture=True
            )
            self.cached = {}
            for layer in self.layers:
                coarse = upsample_tokens(trace[layer]["residual"], 16, "nearest")
                self.cached[layer] = low_prediction(coarse, self.coefficients[layer])
            return output

        if resolution == 32 and self.cached:
            call_index = self.fine_calls
            self.fine_calls += 1
            should_reuse = (call_index - self.phase) % self.period == 0
            if should_reuse:
                self.reused_calls += 1
                output, _ = manual_forward(
                    self.model,
                    x,
                    t,
                    y,
                    multires=True,
                    replacements=self.cached,
                    replacement_mode="residual",
                )
                return output
        output, _ = manual_forward(self.model, x, t, y, multires=True)
        return output


class CalibratedStaggeredReuseModel:
    """Rotate disjoint coarse-residual layer groups across consecutive fine NFEs.

    Every fine NFE still executes most fine blocks.  A particular layer is
    refreshed exactly on the NFEs where its group is inactive, so no layer is
    permanently driven by a stale cross-scale update.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        coefficients: list[dict[str, torch.Tensor]],
        groups: list[list[int]],
    ) -> None:
        self.model = model
        self.coefficients = coefficients
        self.groups = groups
        self.layers = sorted({layer for group in groups for layer in group})
        self.cached: dict[int, torch.Tensor] = {}
        self.fine_calls = 0
        self.fine_blocks_reused = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if resolution == 16 and int(t[0].item()) == 300:
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture=True
            )
            self.cached = {}
            for layer in self.layers:
                coarse = upsample_tokens(trace[layer]["residual"], 16, "nearest")
                self.cached[layer] = low_prediction(coarse, self.coefficients[layer])
            return output

        if resolution == 32 and self.cached:
            group = self.groups[self.fine_calls % len(self.groups)]
            self.fine_calls += 1
            self.fine_blocks_reused += len(group)
            replacements = {layer: self.cached[layer] for layer in group}
            output, _ = manual_forward(
                self.model,
                x,
                t,
                y,
                multires=True,
                replacements=replacements,
                replacement_mode="residual",
            )
            return output
        output, _ = manual_forward(self.model, x, t, y, multires=True)
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", default="80,80,90")
    parser.add_argument("--calibration-seeds", default="101,202")
    parser.add_argument("--calibration-batch", type=int, default=8)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    num_steps = [int(value) for value in args.num_steps.split(",")]

    model = load_rdd(Path(args.rdd_ckpt), device, torch.float32)
    full_fine, full_middle = make_full_diffusions()
    calibration_start = time.perf_counter()
    coefficients = calibrate(
        model,
        full_fine,
        full_middle,
        Path(args.imagenet_train),
        Path(args.vae_path),
        device,
        [int(value) for value in args.calibration_seeds.split(",")],
        args.calibration_batch,
    )
    calibration_seconds = time.perf_counter() - calibration_start

    def build_wrappers() -> dict[str, torch.nn.Module]:
        return {
            "affine_stagewide3": CalibratedPeriodicReuseModel(
                model, coefficients, [14, 17, 20], 1
            ),
            "affine_stagger3_ranked": CalibratedStaggeredReuseModel(
                model, coefficients, [[14], [17], [20]]
            ),
            "affine_stagger6_ranked": CalibratedStaggeredReuseModel(
                model, coefficients, [[14, 11], [17, 18], [20, 15]]
            ),
            "affine_stagger9_ranked": CalibratedStaggeredReuseModel(
                model,
                coefficients,
                [[14, 11, 10], [17, 18, 0], [20, 15, 22]],
            ),
        }

    wrappers = build_wrappers()
    manual_baseline = ManualBaselineModel(model)
    settings = [("baseline", manual_baseline), *wrappers.items()]
    latents: dict[str, torch.Tensor] = {}
    timings: dict[str, float] = {}
    for name, wrapper in settings:
        sampler_diffusions = make_diffusions(num_steps)
        current_model = wrapper
        sample, elapsed = sample_once(
            current_model,
            sampler_diffusions,
            args.batch_size,
            num_steps,
            device,
            args.seed,
        )
        latents[name] = sample
        timings[name] = elapsed

    vae = AutoencoderKL.from_pretrained(str(args.vae_path), local_files_only=True)
    vae.to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, value) for name, value in latents.items()}
    save_grid(
        [(name, decoded[name]) for name, _ in settings],
        output_dir / "comparison_grid.png",
    )

    baseline_time = timings["baseline"]
    report = {
        "checkpoint": str(args.rdd_ckpt),
        "method": "frozen per-channel coarse-low affine + periodic exact fine refresh",
        "calibration_images": len(args.calibration_seeds.split(",")) * args.calibration_batch,
        "calibration_seconds": calibration_seconds,
        "sampler": "DDPM",
        "num_steps_coarse_middle_fine": num_steps,
        "upsample": "canonical noise_filling",
        "cfg": 1.0,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "settings": {},
    }
    for name, wrapper in settings:
        row = {
            "seconds": timings[name],
            "speedup_vs_baseline": baseline_time / timings[name],
            "latency_reduction_percent": 100.0 * (1.0 - timings[name] / baseline_time),
            "latent_relative_l2_vs_baseline": relative_l2(latents[name], latents["baseline"]),
        }
        if isinstance(wrapper, CalibratedPeriodicReuseModel):
            row.update({
                "schedule": "periodic",
                "layers": wrapper.layers,
                "period": wrapper.period,
                "fine_calls_reused": wrapper.reused_calls,
                "fine_calls_total": wrapper.fine_calls,
                "fine_blocks_reused": wrapper.reused_calls * len(wrapper.layers),
            })
        elif isinstance(wrapper, CalibratedStaggeredReuseModel):
            row.update({
                "schedule": "staggered",
                "groups": wrapper.groups,
                "fine_calls_total": wrapper.fine_calls,
                "fine_blocks_reused": wrapper.fine_blocks_reused,
            })
        report["settings"][name] = row

    if args.benchmark_repeats > 0:
        names = [name for name, _ in settings]
        raw_timings = {name: [] for name in names}
        schedule = []
        for repeat in range(args.benchmark_repeats):
            order = names.copy()
            random.Random(9100 + repeat).shuffle(order)
            schedule.append(order)
            for name in order:
                benchmark_wrappers = build_wrappers()
                current_model = manual_baseline if name == "baseline" else benchmark_wrappers[name]
                _, elapsed = sample_once(
                    current_model,
                    make_diffusions(num_steps),
                    args.batch_size,
                    num_steps,
                    device,
                    args.seed,
                )
                raw_timings[name].append(elapsed)
        medians = {
            name: float(torch.tensor(values, dtype=torch.float64).median())
            for name, values in raw_timings.items()
        }
        baseline_median = medians["baseline"]
        report["interleaved_benchmark"] = {
            "repeats": args.benchmark_repeats,
            "schedule": schedule,
            "raw_seconds": raw_timings,
            "median_seconds": medians,
            "speedup_vs_baseline": {
                name: baseline_median / value for name, value in medians.items()
            },
            "latency_reduction_percent": {
                name: 100.0 * (1.0 - value / baseline_median)
                for name, value in medians.items()
            },
        }

    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
