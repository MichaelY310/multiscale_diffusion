#!/usr/bin/env python3
"""True-kernel sweep for training-free RDD attention acceleration ideas.

The existing sampler intentionally remains unchanged.  This file imports its
stable Triton paths and adds experimental controllers for:

* timestep-phase score refresh schedules (as opposed to layer-only schedules),
* periodic current-fine pooled-QK route refresh, and
* sparse attention plus a periodically refreshed dense-output residual.

The residual method stores O_dense - O_sparse (O(ND) memory), not the complete
attention probability matrix (O(N^2) memory).  It is the video-scalable cache
candidate; complete probability reuse is retained only as a small-2D control.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKL
from PIL import Image

from rdd_block_sparse_triton import (
    block_sparse_attention_store,
    micro_block_sparse_attention_store,
)
from sample_rdd_cross_stage_sparse import (
    DEFAULT_CLASSES,
    STAGE_LENGTHS,
    create_sampling_diffusions,
    image_metrics,
    load_unified,
    save_comparison_grid,
    tensor_to_images,
)
from sample_rdd_hierarchical_attention import HierarchicalSetting, aggregate_attention
from sample_rdd_hierarchical_kernel import (
    KernelController,
    fast_block_indices_from_route,
    sample,
    warmup_kernels,
)


@dataclass(frozen=True)
class TrainingFreeKernelSetting(HierarchicalSetting):
    # (early, middle, late) denoiser-call intervals.  None preserves the
    # original layer-depth schedule.  Boundaries are 20% and 85% of a stage.
    time_score_intervals16: tuple[int, int, int] | None = None
    time_score_intervals32: tuple[int, int, int] | None = None
    # Rebuild block indices from current fine pooled Q/K every R calls.  Zero
    # means retain the previous-stage route for the whole stage.
    pooled_route_refresh16: int = 0
    pooled_route_refresh32: int = 0
    # Every R calls compute dense attention once, cache O_dense-O_sparse, and
    # add that O(ND) residual to subsequent sparse outputs.
    residual_refresh16: int = 0
    residual_refresh32: int = 0


SETTINGS = (
    # Timestep-phase tuning for the already successful dense probability cache.
    TrainingFreeKernelSetting(
        "dense_time_score_5_3_2",
        density16=1.0,
        density32=1.0,
        score_layers16=28,
        score_layers32=28,
        time_score_intervals16=(5, 3, 2),
        time_score_intervals32=(5, 3, 2),
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "dense_time_score_4_3_2",
        density16=1.0,
        density32=1.0,
        score_layers16=28,
        score_layers32=28,
        time_score_intervals16=(4, 3, 2),
        time_score_intervals32=(4, 3, 2),
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "dense_time_score_5_4_2",
        density16=1.0,
        density32=1.0,
        score_layers16=28,
        score_layers32=28,
        time_score_intervals16=(5, 4, 2),
        time_score_intervals32=(5, 4, 2),
        temporal_route_update=False,
    ),
    # Periodically refresh the route using a tiny current-fine pooled QK map.
    TrainingFreeKernelSetting(
        "route_refresh6_conservative_score3",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        pooled_route_refresh16=6,
        pooled_route_refresh32=6,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "route_refresh3_aggressive_score3",
        density16=0.875,
        density32=0.75,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        pooled_route_refresh16=3,
        pooled_route_refresh32=3,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "route_refresh6_aggressive_score3",
        density16=0.875,
        density32=0.75,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        pooled_route_refresh16=6,
        pooled_route_refresh32=6,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_route_refresh3_keep50_score3",
        density16=1.0,
        density32=0.50,
        score_interval32=3,
        score_layers32=28,
        pooled_route_refresh32=3,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_route_refresh6_keep50_score3",
        density16=1.0,
        density32=0.50,
        score_interval32=3,
        score_layers32=28,
        pooled_route_refresh32=6,
        temporal_route_update=False,
    ),
    # Dense-residual correction.  Stage 16 stays dense in the first group.
    TrainingFreeKernelSetting(
        "stage32_keep50_residual10",
        density16=1.0,
        density32=0.50,
        residual_refresh32=10,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_keep25_residual5",
        density16=1.0,
        density32=0.25,
        residual_refresh32=5,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_keep25_residual10",
        density16=1.0,
        density32=0.25,
        residual_refresh32=10,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_keep25_residual15",
        density16=1.0,
        density32=0.25,
        residual_refresh32=15,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_keep125_residual5",
        density16=1.0,
        density32=0.125,
        add_self_block=False,
        residual_refresh32=5,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "stage32_keep125_residual10",
        density16=1.0,
        density32=0.125,
        add_self_block=False,
        residual_refresh32=10,
        temporal_route_update=False,
    ),
    TrainingFreeKernelSetting(
        "cascade_keep25_residual10",
        density16=0.25,
        density32=0.25,
        block_side16=2,
        residual_refresh16=10,
        residual_refresh32=10,
        temporal_route_update=False,
    ),
)


def _block_mean(x: torch.Tensor, block_side: int) -> torch.Tensor:
    """Mean [B,H,N,D] tokens into row-major square spatial blocks."""
    batch, heads, tokens, width = x.shape
    grid = int(math.isqrt(tokens))
    if grid * grid != tokens or grid % block_side:
        raise ValueError(f"N={tokens} cannot use block_side={block_side}")
    axis = grid // block_side
    x = x.reshape(batch, heads, grid, grid, width)
    x = x.reshape(batch, heads, axis, block_side, axis, block_side, width)
    return x.mean((3, 5)).reshape(batch, heads, axis * axis, width)


class TrainingFreeKernelController(KernelController):
    setting: TrainingFreeKernelSetting

    def __init__(self, setting: TrainingFreeKernelSetting, total_calls: dict[int, int]):
        super().__init__(setting, total_calls)
        self.output_residuals: dict[int, dict[int, torch.Tensor]] = {16: {}, 32: {}}
        self.stats.update(
            {
                "pooled_route_refresh_calls": 0,
                "dense_residual_refresh_calls": 0,
                "residual_reuse_calls": 0,
            }
        )

    def _phase_intervals(self, resolution: int) -> tuple[int, int, int] | None:
        return (
            self.setting.time_score_intervals16
            if resolution == 16
            else self.setting.time_score_intervals32
        )

    def _interval(self, resolution: int, layer: int) -> int:
        intervals = self._phase_intervals(resolution)
        if intervals is None:
            return super()._interval(resolution, layer)
        denominator = max(self.total_calls[resolution] - 1, 1)
        fraction = self.call_index[resolution] / denominator
        if fraction < 0.20:
            return intervals[0]
        if fraction < 0.85:
            return intervals[1]
        return intervals[2]

    def _route_refresh_interval(self, resolution: int) -> int:
        return (
            self.setting.pooled_route_refresh16
            if resolution == 16
            else self.setting.pooled_route_refresh32
        )

    def _residual_interval(self, resolution: int) -> int:
        return (
            self.setting.residual_refresh16
            if resolution == 16
            else self.setting.residual_refresh32
        )

    def _pooled_route_due(self, resolution: int) -> bool:
        interval = self._route_refresh_interval(resolution)
        call = self.call_index[resolution]
        return interval > 0 and call > 0 and call % interval == 0

    def _residual_due(self, resolution: int) -> bool:
        interval = self._residual_interval(resolution)
        return interval > 0 and self.call_index[resolution] % interval == 0

    def use_v_only(self, layer: int) -> str | None:
        assert self.resolution is not None
        resolution = self.resolution
        if resolution != 8 and (
            self._pooled_route_due(resolution) or self._residual_due(resolution)
        ):
            return None
        return super().use_v_only(layer)

    def _set_indices_from_route(
        self, resolution: int, layer: int, route: torch.Tensor
    ) -> None:
        self.indices[resolution][layer] = fast_block_indices_from_route(
            route, self._density(resolution, layer), self.setting.add_self_block
        )
        # A probability cache has a packed layout tied to the old indices.
        self.probabilities[resolution].pop(layer, None)

    def _refresh_route_from_pooled_qk(
        self,
        resolution: int,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float,
    ) -> None:
        side = self._block_side(resolution)
        pooled_q = _block_mean(q.float(), side)
        pooled_k = _block_mean(k.float(), side)
        route = (pooled_q @ pooled_k.transpose(-2, -1) * scale).softmax(-1)
        self._set_indices_from_route(resolution, layer, route)
        self.stats["pooled_route_refresh_calls"] += 1

    def _sparse_output_and_store(
        self,
        resolution: int,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        indices = self._ensure_indices(resolution, layer, q.shape[-2])
        if self._block_side(resolution) == 2:
            output, probabilities = micro_block_sparse_attention_store(q, k, v, indices)
        else:
            output, probabilities = block_sparse_attention_store(q, k, v, indices)
        self.probabilities[resolution][layer] = probabilities.detach()
        self.stats["sparse_refresh_store_calls"] += 1
        return output

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
        if resolution == 8:
            return super().attention_qkv(layer, q, k, v, scale)

        density = self._density(resolution, layer)
        if density < 1.0 and self._pooled_route_due(resolution):
            self._refresh_route_from_pooled_qk(resolution, layer, q, k, scale)

        if density < 1.0 and self._residual_due(resolution):
            dense_attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            dense_output = dense_attention @ v

            # A dense refresh is also an exact opportunity to update the route.
            route = aggregate_attention(
                dense_attention, self._block_side(resolution), self._block_side(resolution)
            )
            self._set_indices_from_route(resolution, layer, route)
            sparse_output = self._sparse_output_and_store(resolution, layer, q, k, v)
            self.output_residuals[resolution][layer] = (dense_output - sparse_output).detach()
            self.stats["dense_residual_refresh_calls"] += 1
            self.stats["post8_dense_refresh_calls"] += 1
            if resolution == 16 and self._is_last_call(16):
                self.boundary_attention.setdefault(16, {})[layer] = dense_attention.detach()
            return dense_output

        output = super().attention_qkv(layer, q, k, v, scale)
        residual = self.output_residuals[resolution].get(layer)
        if residual is not None:
            self.stats["residual_reuse_calls"] += 1
            output = output + residual.to(output.dtype)
        return output

    def attention_v_only(self, layer: int, v: torch.Tensor, reason: str) -> torch.Tensor:
        output = super().attention_v_only(layer, v, reason)
        assert self.resolution is not None
        residual = self.output_residuals[self.resolution].get(layer)
        if residual is not None:
            self.stats["residual_reuse_calls"] += 1
            output = output + residual.to(output.dtype)
        return output


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
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    steps = tuple(int(x) for x in args.sample_steps.split(","))
    total_calls = dict(zip((8, 16, 32), steps))
    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(
        setting for setting in SETTINGS if requested is None or setting.name in requested
    )
    unknown = set() if requested is None else requested - {setting.name for setting in SETTINGS}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    model, num_classes = load_unified(Path(args.unified_ckpt), device)
    diffusions = create_sampling_diffusions(steps)
    warmup_kernels(device)

    images: dict[str, list[Image.Image]] = {}
    decoded: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "scope": "true Triton sparse calls plus experimental training-free refresh/correction",
        "model": "RDD Unified XL/2 400K",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(steps),
        "variants": {},
    }

    # Same-noise dense baseline, with an untimed warmup.
    sample(model, diffusions, vae, labels[:1], num_classes, args.cfg_scale, args.seed, None)
    dense_runs = []
    dense_stage_runs = []
    for _ in range(args.timing_repeats):
        out, latent, stats = sample(
            model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, None
        )
        dense_runs.append(stats["denoiser_seconds"])
        dense_stage_runs.append(stats["stage_seconds"])
    stats["denoiser_seconds"] = statistics.median(dense_runs)
    stats["denoiser_seconds_runs"] = dense_runs
    stats["stage_seconds"] = {
        resolution: statistics.median(run[resolution] for run in dense_stage_runs)
        for resolution in ("8", "16", "32")
    }
    decoded["dense"], latents["dense"] = out.cpu(), latent.cpu()
    images["dense"] = tensor_to_images(decoded["dense"])
    report["variants"]["dense"] = stats

    for setting in settings:
        print(f"sampling {setting.name}", flush=True)
        runs = []
        stage_runs = []
        for _ in range(args.timing_repeats):
            controller = TrainingFreeKernelController(setting, total_calls)
            out, latent, stats = sample(
                model,
                diffusions,
                vae,
                labels,
                num_classes,
                args.cfg_scale,
                args.seed,
                controller,
            )
            runs.append(stats["denoiser_seconds"])
            stage_runs.append(stats["stage_seconds"])
        stats["denoiser_seconds"] = statistics.median(runs)
        stats["denoiser_seconds_runs"] = runs
        stats["stage_seconds"] = {
            resolution: statistics.median(run[resolution] for run in stage_runs)
            for resolution in ("8", "16", "32")
        }
        stats["setting"] = asdict(setting)
        decoded[setting.name], latents[setting.name] = out.cpu(), latent.cpu()
        images[setting.name] = tensor_to_images(decoded[setting.name])
        stats["image_vs_dense"] = image_metrics(decoded["dense"], decoded[setting.name])
        stats["latent_vs_dense"] = image_metrics(latents["dense"], latents[setting.name])
        report["variants"][setting.name] = stats

        variant_dir = output_dir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        gc.collect()
        torch.cuda.empty_cache()

    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images["dense"]):
        image.save(dense_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")

    grid = output_dir / "training_free_kernel_comparison.png"
    save_comparison_grid(
        images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: training-free route refresh and residual correction",
    )
    report["comparison_grid"] = str(grid)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {grid}", flush=True)


if __name__ == "__main__":
    main()
