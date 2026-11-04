#!/usr/bin/env python3
"""RDD-specific temporal correction transport for real 2D sampling.

At an exact fine-stage refresh NFE, decompose each selected fine block update:

    fine_residual(t_r) = coarse_predictor(t_r) + correction(t_r)

For the following skipped NFEs, transport the cached sample-specific correction
on top of the *current* time-conditioned coarse prediction:

    predicted_fine_residual(t) = coarse_predictor(t) + correction(t_r)

The matched generic cache simply reuses fine_residual(t_r).  Therefore the two
methods skip exactly the same blocks; only RDD transport can use the already
available cross-scale trajectory to account for expected timestep evolution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
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
from sample_rdd_time_conditioned_reuse_preview import (  # noqa: E402
    calibrate_time_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from diffusers.models import AutoencoderKL  # noqa: E402


class PeriodicFineCacheModel:
    """Matched generic cache or RDD coarse-conditioned correction transport."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layers: list[int],
        period: int,
        mode: str,
        coefficients: dict[int, list[dict[str, torch.Tensor]]] | None = None,
        transport_max_t: int | None = None,
    ) -> None:
        if mode not in {"generic", "rdd_transport"}:
            raise ValueError(mode)
        self.model = model
        self.layers = layers
        self.layer_set = set(layers)
        self.period = period
        self.mode = mode
        self.coefficients = coefficients
        self.transport_max_t = transport_max_t
        self.knots = sorted(coefficients) if coefficients is not None else []
        self.coarse: dict[int, torch.Tensor] = {}
        self.cache: dict[int, torch.Tensor] = {}
        self.cache_coarse: dict[int, torch.Tensor] = {}
        self.fine_calls = 0
        self.blocks_reused = 0

    def affine(self, layer: int, timestep: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.coefficients is not None
        if timestep <= self.knots[0]:
            row = self.coefficients[self.knots[0]][layer]
            return row["low_scale"], row["low_bias"]
        if timestep >= self.knots[-1]:
            row = self.coefficients[self.knots[-1]][layer]
            return row["low_scale"], row["low_bias"]
        right_index = next(i for i, knot in enumerate(self.knots) if knot > timestep)
        left, right = self.knots[right_index - 1], self.knots[right_index]
        weight = float(timestep - left) / float(right - left)
        left_row = self.coefficients[left][layer]
        right_row = self.coefficients[right][layer]
        return (
            left_row["low_scale"].lerp(right_row["low_scale"], weight),
            left_row["low_bias"].lerp(right_row["low_bias"], weight),
        )

    def coarse_prediction(self, layer: int, timestep: int) -> torch.Tensor:
        scale, bias = self.affine(layer, timestep)
        dtype = self.coarse[layer].dtype
        scale = scale.to(dtype=dtype)
        bias = bias.to(dtype=dtype)
        return self.coarse[layer] * scale.view(1, 1, -1) + bias.view(1, 1, -1)

    def exact_forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        return manual_forward(
            self.model,
            x,
            t,
            y,
            multires=True,
            capture_layers=self.layer_set,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if (
            self.mode == "rdd_transport"
            and resolution == 16
            and int(t[0].item()) == 300
        ):
            output, trace = self.exact_forward(x, t, y)
            self.coarse = {
                layer: upsample_tokens(trace[layer]["residual"], 16, "nearest")
                for layer in self.layers
            }
            return output

        if resolution == 32:
            timestep = int(t[0].item())
            refresh = not self.cache or self.fine_calls % self.period == 0
            self.fine_calls += 1
            if refresh:
                output, trace = self.exact_forward(x, t, y)
                for layer in self.layers:
                    exact = trace[layer]["residual"]
                    if self.mode == "generic":
                        self.cache[layer] = exact
                    else:
                        coarse_now = self.coarse_prediction(layer, timestep)
                        self.cache[layer] = exact - coarse_now
                        self.cache_coarse[layer] = coarse_now
                return output

            replacements = {}
            for layer in self.layers:
                if self.mode == "generic":
                    replacements[layer] = self.cache[layer]
                else:
                    use_transport = (
                        self.transport_max_t is None
                        or timestep <= self.transport_max_t
                    )
                    coarse_base = (
                        self.coarse_prediction(layer, timestep)
                        if use_transport
                        else self.cache_coarse[layer]
                    )
                    replacements[layer] = coarse_base + self.cache[layer]
            self.blocks_reused += len(self.layers)
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
    parser.add_argument("--knots", default="0,75,150,225,299")
    parser.add_argument(
        "--sweep",
        choices=[
            "safe",
            "gap_layers",
            "trajectory_layers",
            "trajectory_late",
            "trajectory_top3",
        ],
        default="safe",
    )
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
    safe6 = [7, 9, 11, 13, 15, 16]
    ranked6 = [11, 14, 15, 17, 18, 20]
    if args.sweep == "trajectory_top3":
        top3 = [1, 2, 3]
        wrappers = {
            "generic_p5_top3": PeriodicFineCacheModel(
                model, layers=top3, period=5, mode="generic"
            ),
            "rdd_transport_p5_top3": PeriodicFineCacheModel(
                model, layers=top3, period=5, mode="rdd_transport",
                coefficients=coefficients,
            ),
            "generic_p8_top3": PeriodicFineCacheModel(
                model, layers=top3, period=8, mode="generic"
            ),
            "rdd_transport_p8_top3": PeriodicFineCacheModel(
                model, layers=top3, period=8, mode="rdd_transport",
                coefficients=coefficients,
            ),
        }
    elif args.sweep in {"gap_layers", "trajectory_layers", "trajectory_late"}:
        # ``gap_layers`` is retained for reproducibility of the earlier
        # q-sampled validation.  ``trajectory_layers`` comes from the denoiser's
        # own dense reverse trajectory and is the correct rollout hypothesis.
        gap6 = (
            [1, 2, 3, 24, 26, 27]
            if args.sweep == "gap_layers"
            else [1, 2, 3, 14, 17, 20]
        )
        wrappers = {
            "generic_p4_gap6": PeriodicFineCacheModel(
                model, layers=gap6, period=4, mode="generic"
            ),
            "rdd_transport_p4_gap6": PeriodicFineCacheModel(
                model, layers=gap6, period=4, mode="rdd_transport",
                coefficients=coefficients,
            ),
            "generic_p5_gap6": PeriodicFineCacheModel(
                model, layers=gap6, period=5, mode="generic"
            ),
            "rdd_transport_p5_gap6": PeriodicFineCacheModel(
                model, layers=gap6, period=5, mode="rdd_transport",
                coefficients=coefficients,
            ),
        }
        if args.sweep == "trajectory_late":
            wrappers = {
                "generic_p4_gap6": wrappers["generic_p4_gap6"],
                "rdd_transport_late180_p4_gap6": PeriodicFineCacheModel(
                    model, layers=gap6, period=4, mode="rdd_transport",
                    coefficients=coefficients, transport_max_t=180,
                ),
                "generic_p5_gap6": wrappers["generic_p5_gap6"],
                "rdd_transport_late180_p5_gap6": PeriodicFineCacheModel(
                    model, layers=gap6, period=5, mode="rdd_transport",
                    coefficients=coefficients, transport_max_t=180,
                ),
            }
    else:
        wrappers = {
            "generic_p3_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=3, mode="generic"
            ),
            "generic_p4_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=4, mode="generic"
            ),
            "generic_p5_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=5, mode="generic"
            ),
            "rdd_transport_p3_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=3, mode="rdd_transport",
                coefficients=coefficients,
            ),
            "rdd_transport_p4_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=4, mode="rdd_transport",
                coefficients=coefficients,
            ),
            "rdd_transport_p5_safe6": PeriodicFineCacheModel(
                model, layers=safe6, period=5, mode="rdd_transport",
                coefficients=coefficients,
            ),
            "rdd_transport_p3_ranked6": PeriodicFineCacheModel(
                model, layers=ranked6, period=3, mode="rdd_transport",
                coefficients=coefficients,
            ),
        }
    settings = [("baseline", ManualBaselineModel(model)), *wrappers.items()]
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
        "method": "RDD time-conditioned coarse base + periodically refreshed fine correction",
        "sweep": args.sweep,
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
        if isinstance(wrapper, PeriodicFineCacheModel):
            row.update({
                "mode": wrapper.mode,
                "layers": wrapper.layers,
                "period": wrapper.period,
                "transport_max_t": wrapper.transport_max_t,
                "fine_calls": wrapper.fine_calls,
                "blocks_reused": wrapper.blocks_reused,
            })
        report["settings"][name] = row
    if args.benchmark_repeats > 0:
        names = [name for name, _ in settings]
        prototypes = dict(settings)
        raw = {name: [] for name in names}
        schedule = []

        def fresh(name: str):
            if name == "baseline":
                return ManualBaselineModel(model)
            prototype = prototypes[name]
            return PeriodicFineCacheModel(
                model,
                layers=prototype.layers,
                period=prototype.period,
                mode=prototype.mode,
                coefficients=prototype.coefficients,
                transport_max_t=prototype.transport_max_t,
            )

        for repeat in range(args.benchmark_repeats):
            order = names[repeat % len(names):] + names[: repeat % len(names)]
            if repeat % 2:
                order = list(reversed(order))
            schedule.append(order)
            for name in order:
                _, elapsed = sample_once(
                    fresh(name),
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
