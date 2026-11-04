#!/usr/bin/env python3
"""Original DiT control for staggered previous-timestep block reuse.

This is the matched full-resolution control for the RDD cross-scale preview.
Selected layer groups rotate across denoising calls.  A skipped layer reuses
its most recently computed full-resolution residual; all inactive groups are
recomputed and refresh their cache on that call.
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
from probe_rdd_vs_dit_block_reuse import load_dit, manual_forward  # noqa: E402
from sample_rdd_stagewide_block_reuse_preview import (  # noqa: E402
    decode,
    relative_l2,
    save_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


class TemporalStaggerReuseModel:
    def __init__(self, model: torch.nn.Module, groups: list[list[int]]) -> None:
        self.model = model
        self.groups = groups
        self.layers = sorted({layer for group in groups for layer in group})
        self.cache: dict[int, torch.Tensor] = {}
        self.calls = 0
        self.blocks_reused = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The first NFE establishes a real full-resolution cache.
        if not self.cache:
            output, trace = manual_forward(
                self.model, x, t, y, multires=False, capture=True
            )
            self.cache = {layer: trace[layer]["residual"] for layer in self.layers}
            self.calls += 1
            return output

        group = self.groups[(self.calls - 1) % len(self.groups)]
        replacements = {layer: self.cache[layer] for layer in group}
        output, trace = manual_forward(
            self.model,
            x,
            t,
            y,
            multires=False,
            capture=True,
            replacements=replacements,
            replacement_mode="residual",
        )
        # Recomputed groups advance to the current timestep; the active group
        # intentionally keeps its previous exact value for one more call.
        for layer in self.layers:
            if layer not in group:
                self.cache[layer] = trace[layer]["residual"]
        self.calls += 1
        self.blocks_reused += len(group)
        return output


class ManualDiTBaselineModel:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output, _ = manual_forward(self.model, x, t, y, multires=False)
        return output


@torch.inference_mode()
def sample_once(model, batch_size: int, steps: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    diffusion = create_diffusion(str(steps))
    noise = torch.randn(batch_size, 4, 32, 32, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    latents = diffusion.p_sample_loop(
        model.forward,
        noise.shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs={"y": labels},
        progress=False,
        device=device,
    )
    torch.cuda.synchronize(device)
    return latents, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    base_model = load_dit(Path(args.dit_ckpt), device, torch.float32)
    def make_settings() -> dict[str, object]:
        return {
            "baseline": ManualDiTBaselineModel(base_model),
            "temporal_stagger9": TemporalStaggerReuseModel(
                base_model, [[8, 11, 14], [9, 12, 15], [10, 13, 16]]
            ),
            "temporal_stagger12": TemporalStaggerReuseModel(
                base_model,
                [[6, 9, 12, 15], [7, 10, 13, 16], [8, 11, 14, 17]],
            ),
        }

    settings_dict = make_settings()
    settings = list(settings_dict.items())
    latents: dict[str, torch.Tensor] = {}
    timings: dict[str, float] = {}
    for name, wrapper in settings:
        sample, elapsed = sample_once(
            wrapper, args.batch_size, args.steps, device, args.seed
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
        "checkpoint": str(args.dit_ckpt),
        "method": "full-resolution previous-timestep residual, staggered layer groups",
        "sampler": "DDPM",
        "steps": args.steps,
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
        if isinstance(wrapper, TemporalStaggerReuseModel):
            row.update({
                "groups": wrapper.groups,
                "calls": wrapper.calls,
                "blocks_reused": wrapper.blocks_reused,
            })
        report["settings"][name] = row
    if args.benchmark_repeats > 0:
        names = list(settings_dict)
        raw = {name: [] for name in names}
        schedule = []
        # Rotate the order so no method always pays the first/last-run effect.
        for repeat in range(args.benchmark_repeats):
            order = names[repeat % len(names):] + names[: repeat % len(names)]
            if repeat % 2:
                order = list(reversed(order))
            schedule.append(order)
            for name in order:
                wrapper = make_settings()[name]
                _, elapsed = sample_once(
                    wrapper,
                    args.batch_size,
                    args.steps,
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
