#!/usr/bin/env python3
"""Training-free, timestep-adaptive attention-probability reuse for RDD.

The older ``adaptive_5_3_3`` experiment was only layer-aware: its refresh
period was fixed in advance.  This experiment makes one decision per denoiser
call from the *current stage input* instead.  When the latent has changed only
slightly since the last refresh, all selected transformer layers reuse
``A_old @ V_current``; otherwise Q/K and the attention probability are
recomputed.  Comparing the stage input once per timestep avoids 28 host/device
synchronizations and keeps the decision overhead O(NC), rather than O(N^2).

This file is intentionally isolated from the historical sampler so old
checkpoints and old experiment scripts remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKL
from PIL import Image

from sample_rdd_cross_stage_sparse import (
    DEFAULT_CLASSES,
    STAGE_LENGTHS,
    create_sampling_diffusions,
    image_metrics,
    load_unified,
    save_comparison_grid,
    tensor_to_images,
)
from sample_rdd_hierarchical_kernel import KernelController, sample


@dataclass(frozen=True)
class AdaptiveCacheSetting:
    name: str
    stages: tuple[int, ...]
    layers: int = 28
    # Fixed controls.  Zero activates state-adaptive refresh instead.
    fixed_interval: int = 0
    fixed_interval8: int | None = None
    fixed_interval16: int | None = None
    fixed_interval32: int | None = None
    # Reuse while ||x-current - x-refresh||_RMS / ||x-refresh||_RMS is below
    # this threshold, but force a refresh at max_interval at the latest.
    relative_rms_threshold: float = 0.0
    max_interval: int = 5
    # The decision sketch keeps one pixel from each stride x stride window.
    sketch_stride: int = 2


SETTINGS = (
    # Isolate whether the cheap 8x8 stage can reuse scores at all.
    AdaptiveCacheSetting("fixed_8_k2", (8,), fixed_interval=2),
    AdaptiveCacheSetting("fixed_8_k3", (8,), fixed_interval=3),
    # Complete 8/16/32 controls.  These are small-2D controls; full A caches
    # are not proposed as the default long-video implementation.
    AdaptiveCacheSetting("fixed_all_k2", (8, 16, 32), fixed_interval=2),
    AdaptiveCacheSetting("fixed_all_k3", (8, 16, 32), fixed_interval=3),
    # Stage-specific and layer-count tuning.  These separate the cheap 8x8
    # stage from the more expensive 16/32 stages and locate the quality loss.
    AdaptiveCacheSetting("fixed_16_32_k2", (16, 32), fixed_interval=2),
    AdaptiveCacheSetting("fixed_16_32_k3", (16, 32), fixed_interval=3),
    AdaptiveCacheSetting(
        "fixed_16k2_32k3", (16, 32), fixed_interval16=2, fixed_interval32=3
    ),
    AdaptiveCacheSetting(
        "fixed_16k3_32k2", (16, 32), fixed_interval16=3, fixed_interval32=2
    ),
    AdaptiveCacheSetting("fixed_16_32_k3_first7", (16, 32), layers=7, fixed_interval=3),
    AdaptiveCacheSetting("fixed_16_32_k3_first14", (16, 32), layers=14, fixed_interval=3),
    AdaptiveCacheSetting("fixed_16_32_k3_first21", (16, 32), layers=21, fixed_interval=3),
    AdaptiveCacheSetting("fixed_16_32_k4_first7", (16, 32), layers=7, fixed_interval=4),
    # State-adaptive variants.  The max interval bounds accumulated error.
    AdaptiveCacheSetting("adaptive_16_32_tau02_max5", (16, 32), relative_rms_threshold=0.02),
    AdaptiveCacheSetting("adaptive_16_32_tau04_max5", (16, 32), relative_rms_threshold=0.04),
    AdaptiveCacheSetting("adaptive_16_32_tau06_max5", (16, 32), relative_rms_threshold=0.06),
    AdaptiveCacheSetting("adaptive_16_32_tau08_max5", (16, 32), relative_rms_threshold=0.08),
    AdaptiveCacheSetting("adaptive_all_tau02_max5", (8, 16, 32), relative_rms_threshold=0.02),
    AdaptiveCacheSetting("adaptive_all_tau04_max5", (8, 16, 32), relative_rms_threshold=0.04),
    AdaptiveCacheSetting("adaptive_all_tau06_max5", (8, 16, 32), relative_rms_threshold=0.06),
)


class AdaptiveProbabilityController(KernelController):
    """Dense attention refresh plus A_old @ V_current on adaptive cache hits."""

    def __init__(self, setting: AdaptiveCacheSetting, total_calls: dict[int, int]):
        # KernelController expects a hierarchical setting, but the methods used
        # below are fully overridden.  Keep its bookkeeping and boundary route.
        from sample_rdd_hierarchical_attention import HierarchicalSetting

        super().__init__(HierarchicalSetting(setting.name), total_calls)
        self.adaptive_setting = setting
        self.dense_probabilities = {8: {}, 16: {}, 32: {}}
        self.reference_inputs: dict[int, torch.Tensor] = {}
        self.last_refresh_call = {8: -1, 16: -1, 32: -1}
        self.reuse_this_call = {8: False, 16: False, 32: False}
        self.relative_rms: dict[int, list[float]] = {8: [], 16: [], 32: []}
        self.stats.update(
            {
                "adaptive_refresh_model_calls": {"8": 0, "16": 0, "32": 0},
                "adaptive_reuse_model_calls": {"8": 0, "16": 0, "32": 0},
                "adaptive_probability_hits": {"8": 0, "16": 0, "32": 0},
            }
        )

    def _sketch(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.adaptive_setting.sketch_stride
        return x.detach()[..., ::stride, ::stride].float().contiguous()

    def _fixed_interval(self, resolution: int) -> int:
        setting = self.adaptive_setting
        override = {
            8: setting.fixed_interval8,
            16: setting.fixed_interval16,
            32: setting.fixed_interval32,
        }[resolution]
        return setting.fixed_interval if override is None else override

    def begin_forward(self, x: torch.Tensor) -> None:
        super().begin_forward(x)
        assert self.resolution is not None
        resolution = self.resolution
        setting = self.adaptive_setting
        key = str(resolution)
        call = self.call_index[resolution]
        selected = resolution in setting.stages
        last = self._is_last_call(resolution)
        reuse = False

        if selected and call > 0 and not last and self.dense_probabilities[resolution]:
            since = call - self.last_refresh_call[resolution]
            fixed_interval = self._fixed_interval(resolution)
            if fixed_interval > 0:
                reuse = call % fixed_interval != 0
            elif since < setting.max_interval:
                current = self._sketch(x)
                reference = self.reference_inputs[resolution]
                delta_rms = (current - reference).square().mean().sqrt()
                reference_rms = reference.square().mean().sqrt().clamp_min(1e-8)
                relative = float((delta_rms / reference_rms).item())
                self.relative_rms[resolution].append(relative)
                reuse = relative <= setting.relative_rms_threshold

        self.reuse_this_call[resolution] = reuse
        if reuse:
            self.stats["adaptive_reuse_model_calls"][key] += 1
        else:
            self.stats["adaptive_refresh_model_calls"][key] += 1
            if selected:
                self.reference_inputs[resolution] = self._sketch(x)
                self.last_refresh_call[resolution] = call

    def use_v_only(self, layer: int) -> str | None:
        assert self.resolution is not None
        resolution = self.resolution
        if (
            resolution in self.adaptive_setting.stages
            and layer < self.adaptive_setting.layers
            and self.reuse_this_call[resolution]
            and layer in self.dense_probabilities[resolution]
        ):
            return "adaptive"
        return None

    def attention_v_only(self, layer: int, v: torch.Tensor, reason: str) -> torch.Tensor:
        assert reason == "adaptive"
        assert self.resolution is not None
        resolution = self.resolution
        self.stats["temporal_score_cache_hits"] += 1
        self.stats["v_only_projection_calls"] += 1
        self.stats["adaptive_probability_hits"][str(resolution)] += 1
        return self.dense_probabilities[resolution][layer].to(v.dtype) @ v

    def attention_qkv(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        assert self.resolution is not None
        resolution = self.resolution
        attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
        if resolution in self.adaptive_setting.stages and layer < self.adaptive_setting.layers:
            self.dense_probabilities[resolution][layer] = attention.detach()
        if resolution in (8, 16) and self._is_last_call(resolution):
            self.boundary_attention.setdefault(resolution, {})[layer] = attention.detach()
        self.stats["dense_attention_calls"] += 1
        if resolution != 8:
            self.stats["post8_dense_refresh_calls"] += 1
        return attention @ v

    def report(self) -> dict[str, Any]:
        distance_summary = {}
        for resolution, values in self.relative_rms.items():
            distance_summary[str(resolution)] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        return {
            "setting": asdict(self.adaptive_setting),
            **self.stats,
            "relative_rms_since_refresh": distance_summary,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--sample-steps", default="80,80,90")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    steps = tuple(int(x) for x in args.sample_steps.split(","))
    total_calls = dict(zip((8, 16, 32), steps))
    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(s for s in SETTINGS if requested is None or s.name in requested)
    unknown = set() if requested is None else requested - {s.name for s in SETTINGS}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    model, num_classes = load_unified(Path(args.unified_ckpt), device)
    diffusions = create_sampling_diffusions(steps)

    report: dict[str, Any] = {
        "scope": "training-free fixed and latent-state-adaptive dense probability reuse",
        "model": "RDD Unified XL/2 400K",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(steps),
        "variants": {},
    }
    all_decoded: dict[str, torch.Tensor] = {}
    all_latents: dict[str, torch.Tensor] = {}
    all_images: dict[str, list[Image.Image]] = {}

    # Warm the model and obtain a same-noise dense reference.
    sample(model, diffusions, vae, labels[:1], num_classes, args.cfg_scale, args.seed, None)
    dense_times = []
    for _ in range(args.timing_repeats):
        decoded, latent, stats = sample(
            model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, None
        )
        dense_times.append(stats["denoiser_seconds"])
    stats["denoiser_seconds_runs"] = dense_times
    stats["denoiser_seconds"] = statistics.median(dense_times)
    all_decoded["dense"], all_latents["dense"] = decoded.cpu(), latent.cpu()
    all_images["dense"] = tensor_to_images(all_decoded["dense"])
    report["variants"]["dense"] = stats
    del decoded, latent
    gc.collect()
    torch.cuda.empty_cache()

    for setting in settings:
        print("sampling", setting.name, flush=True)
        times = []
        for _ in range(args.timing_repeats):
            controller = AdaptiveProbabilityController(setting, total_calls)
            decoded, latent, stats = sample(
                model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, controller
            )
            times.append(stats["denoiser_seconds"])
        stats["denoiser_seconds_runs"] = times
        stats["denoiser_seconds"] = statistics.median(times)
        all_decoded[setting.name], all_latents[setting.name] = decoded.cpu(), latent.cpu()
        all_images[setting.name] = tensor_to_images(all_decoded[setting.name])
        variant_dir = outdir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(all_images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        stats["image_vs_dense"] = image_metrics(all_decoded["dense"], all_decoded[setting.name])
        stats["latent_vs_dense"] = image_metrics(all_latents["dense"], all_latents[setting.name])
        report["variants"][setting.name] = stats
        del decoded, latent
        gc.collect()
        torch.cuda.empty_cache()

    dense_dir = outdir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(all_images["dense"]):
        image.save(dense_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
    grid = outdir / "adaptive_probability_cache_comparison.png"
    save_comparison_grid(
        all_images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: fixed vs latent-state-adaptive attention probability reuse",
    )
    report["comparison_grid"] = str(grid)
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
