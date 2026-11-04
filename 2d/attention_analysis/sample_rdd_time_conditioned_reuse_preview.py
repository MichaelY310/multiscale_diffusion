#!/usr/bin/env python3
"""Time-conditioned cross-scale residual reuse on a real RDD checkpoint.

The final 16x16 block residual is a fixed coarse semantic proposal, but its
appropriate channel scale changes throughout the 32x32 denoising stage.  This
script fits a tiny frozen per-channel affine map at five fine timesteps on a
disjoint ImageNet calibration split and linearly interpolates the coefficients
during sampling.  It is a diagnostic adapter, not checkpoint fine-tuning.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_hybrid_residual_predictor import (  # noqa: E402
    accumulate,
    finish_coefficients,
    init_stats,
    make_diffusions as make_full_diffusions,
)
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    aligned_noise,
    discover_train_images,
    encode_images,
    load_rdd,
    low_high_tokens,
    manual_forward,
    stage_drift,
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


@torch.inference_mode()
def calibrate_time_grid(
    model: torch.nn.Module,
    imagenet_train: Path,
    vae_path: Path,
    device: torch.device,
    seeds: list[int],
    batch_size: int,
    knots: list[int],
) -> dict[int, list[dict[str, torch.Tensor]]]:
    fine, middle = make_full_diffusions()
    depth = len(model.blocks)
    channels = model.x_embedder.proj.out_channels
    stats = {knot: init_stats(depth, channels, device) for knot in knots}
    dtype = model.x_embedder.proj.weight.dtype

    for index, seed in enumerate(seeds):
        paths, labels = discover_train_images(imagenet_train, batch_size, index * 211)
        labels = labels.to(device)
        clean32 = encode_images(paths, vae_path, device)
        clean16 = torch.nn.functional.avg_pool2d(clean32, 2, 2)
        eps32, eps16 = aligned_noise(clean32, seed)
        source_local = torch.zeros(batch_size, device=device, dtype=torch.long)
        source = middle.q_sample(
            clean16, source_local, d=stage_drift(middle, clean16), noise=eps16
        )
        source_global = torch.full(
            (batch_size,), 300, device=device, dtype=torch.long
        )
        _, source_trace = manual_forward(
            model, source.to(dtype), source_global, labels, multires=True, capture=True
        )
        coarse = {
            layer: upsample_tokens(source_trace[layer]["residual"], 16, "nearest")
            for layer in range(depth)
        }

        for knot in knots:
            target_local = torch.full(
                (batch_size,), knot, device=device, dtype=torch.long
            )
            target = fine.q_sample(
                clean32, target_local, d=stage_drift(fine, clean32), noise=eps32
            )
            _, target_trace = manual_forward(
                model, target.to(dtype), target_local, labels,
                multires=True, capture=True,
            )
            for layer in range(depth):
                target_low, _ = low_high_tokens(target_trace[layer]["residual"])
                accumulate(stats[knot][layer], "low", coarse[layer], target_low)
            del target, target_trace

        del clean32, clean16, source, source_trace, coarse
        gc.collect()
        torch.cuda.empty_cache()

    return {knot: finish_coefficients(stats[knot]) for knot in knots}


class TimeConditionedReuseModel:
    def __init__(
        self,
        model: torch.nn.Module,
        coefficients: dict[int, list[dict[str, torch.Tensor]]],
        *,
        layers: list[int],
        period: int = 1,
        groups: list[list[int]] | None = None,
        time_conditioned: bool = True,
    ) -> None:
        self.model = model
        self.coefficients = coefficients
        self.knots = sorted(coefficients)
        self.layers = sorted(set(layers))
        self.period = period
        self.groups = groups
        self.time_conditioned = time_conditioned
        self.cache: dict[int, torch.Tensor] = {}
        self.fine_calls = 0
        self.blocks_reused = 0

    def coefficient(self, layer: int, timestep: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.time_conditioned:
            row = self.coefficients[self.knots[-1]][layer]
            return row["low_scale"], row["low_bias"]
        if timestep <= self.knots[0]:
            row = self.coefficients[self.knots[0]][layer]
            return row["low_scale"], row["low_bias"]
        if timestep >= self.knots[-1]:
            row = self.coefficients[self.knots[-1]][layer]
            return row["low_scale"], row["low_bias"]
        right_index = bisect.bisect_right(self.knots, timestep)
        left, right = self.knots[right_index - 1], self.knots[right_index]
        weight = float(timestep - left) / float(right - left)
        left_row, right_row = self.coefficients[left][layer], self.coefficients[right][layer]
        scale = left_row["low_scale"].lerp(right_row["low_scale"], weight)
        bias = left_row["low_bias"].lerp(right_row["low_bias"], weight)
        return scale, bias

    def selected_layers(self) -> list[int]:
        if self.groups is not None:
            return self.groups[self.fine_calls % len(self.groups)]
        if self.fine_calls % self.period == 0:
            return self.layers
        return []

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if resolution == 16 and int(t[0].item()) == 300:
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture=True
            )
            self.cache = {
                layer: upsample_tokens(trace[layer]["residual"], 16, "nearest")
                for layer in self.layers
            }
            return output
        if resolution == 32 and self.cache:
            timestep = int(t[0].item())
            active = self.selected_layers()
            self.fine_calls += 1
            if active:
                replacements = {}
                for layer in active:
                    scale, bias = self.coefficient(layer, timestep)
                    replacements[layer] = (
                        self.cache[layer] * scale.view(1, 1, -1)
                        + bias.view(1, 1, -1)
                    )
                self.blocks_reused += len(active)
                output, _ = manual_forward(
                    self.model, x, t, y, multires=True,
                    replacements=replacements, replacement_mode="residual",
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
    parser.add_argument("--knots", default="0,75,150,225,299")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_rdd(Path(args.rdd_ckpt), device, torch.float32)
    knots = [int(value) for value in args.knots.split(",")]
    calibration_start = time.perf_counter()
    coefficients = calibrate_time_grid(
        model,
        Path(args.imagenet_train),
        Path(args.vae_path),
        device,
        [int(value) for value in args.calibration_seeds.split(",")],
        args.calibration_batch,
        knots,
    )
    calibration_seconds = time.perf_counter() - calibration_start
    steps = [int(value) for value in args.num_steps.split(",")]

    def make_settings() -> dict[str, object]:
        return {
            "baseline": ManualBaselineModel(model),
            "static_stagewide3": TimeConditionedReuseModel(
                model, coefficients, layers=[14, 17, 20], time_conditioned=False
            ),
            "time_stagewide3": TimeConditionedReuseModel(
                model, coefficients, layers=[14, 17, 20], time_conditioned=True
            ),
            "time_period3_safe6": TimeConditionedReuseModel(
                model, coefficients, layers=[7, 9, 11, 13, 15, 16], period=3
            ),
            "time_stagger6_ranked": TimeConditionedReuseModel(
                model,
                coefficients,
                layers=[11, 14, 15, 17, 18, 20],
                groups=[[14, 11], [17, 18], [20, 15]],
            ),
        }

    settings_dict = make_settings()
    settings = list(settings_dict.items())
    latents: dict[str, torch.Tensor] = {}
    timings: dict[str, float] = {}
    for name, wrapper in settings:
        sample, elapsed = sample_once(
            wrapper,
            make_diffusions(steps),
            args.batch_size,
            steps,
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
        "method": "time-conditioned per-channel cross-scale residual alignment",
        "knots": knots,
        "calibration_images": len(args.calibration_seeds.split(",")) * args.calibration_batch,
        "calibration_seconds": calibration_seconds,
        "sampler": "DDPM",
        "num_steps_coarse_middle_fine": steps,
        "upsample": "canonical noise_filling",
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
        if isinstance(wrapper, TimeConditionedReuseModel):
            row.update({
                "layers": wrapper.layers,
                "period": wrapper.period,
                "groups": wrapper.groups,
                "time_conditioned": wrapper.time_conditioned,
                "fine_calls": wrapper.fine_calls,
                "blocks_reused": wrapper.blocks_reused,
            })
        report["settings"][name] = row
    if args.benchmark_repeats > 0:
        names = list(settings_dict)
        raw = {name: [] for name in names}
        schedule = []
        for repeat in range(args.benchmark_repeats):
            order = names[repeat % len(names):] + names[: repeat % len(names)]
            if repeat % 2:
                order = list(reversed(order))
            schedule.append(order)
            for name in order:
                wrapper = make_settings()[name]
                _, elapsed = sample_once(
                    wrapper,
                    make_diffusions(steps),
                    args.batch_size,
                    steps,
                    device,
                    args.seed + 1000 + repeat,
                )
                raw[name].append(elapsed)
        medians = {name: statistics.median(values) for name, values in raw.items()}
        baseline_median = medians["baseline"]
        report["interleaved_benchmark"] = {
            "repeats": args.benchmark_repeats,
            "schedule": schedule,
            "raw_seconds": raw,
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
