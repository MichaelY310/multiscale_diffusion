#!/usr/bin/env python3
"""Training-free RDD attention sweep with fine-stage centroid compensation.

This is a quality oracle: it materializes dense logits before applying a
structured mask.  It is intended to answer whether a configuration is worth a
real fused kernel, not to claim wall-clock acceleration.

Unlike the older ``hybrid`` experiment, the compensation branch summarizes
the *current fine-stage* K/V tokens inside each spatial block.  An omitted
block is represented by its mean K/V and a log(block_size) multiplicity term,
so selected blocks are computed exactly while skipped blocks still contribute
global context.  This is the simple, parameter-free compensation principle
used by recent sparse-attention work such as SVG-EAR.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from sample_rdd_hierarchical_attention import (
    HierarchicalController,
    HierarchicalSetting,
    block_mask_from_route,
    sample_variant,
)


@dataclass(frozen=True)
class CompensationSetting(HierarchicalSetting):
    centroid_compensation: bool = False
    # Reweight the inherited RDD route using current fine-stage K/V variance.
    # Zero means ordinary attention-mass Top-K routing.
    variance_route_beta: float = 0.0
    # Dense warmup is indexed over denoiser calls inside each stage.
    dense_warm_fraction: float = 0.0
    dense_warm_layers: int = 0
    # Instead of fixed Top-K, retain the smallest set of coarse-route blocks
    # whose cumulative route mass reaches this threshold.  Zero uses fixed K.
    route_mass_threshold: float = 0.0


SETTINGS = (
    CompensationSetting("dense"),
    # Drop-only references.
    CompensationSetting(
        "stage32_drop50",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_drop25",
        density16=1.0,
        density32=0.25,
        block_side32=4,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_drop125",
        density16=1.0,
        density32=0.125,
        block_side32=4,
        add_self_block=False,
        temporal_route_update=True,
    ),
    # Current-fine centroid compensation, stage 32 only.
    CompensationSetting(
        "stage32_centroid50",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid75",
        density16=1.0,
        density32=0.75,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid625",
        density16=1.0,
        density32=0.625,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid375",
        density16=1.0,
        density32=0.375,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid_mass85",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        route_mass_threshold=0.85,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid_mass90",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        route_mass_threshold=0.90,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid_mass95",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        route_mass_threshold=0.95,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid25",
        density16=1.0,
        density32=0.25,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid125",
        density16=1.0,
        density32=0.125,
        block_side32=4,
        add_self_block=False,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
    # Conservative dense warmup variants modeled after video sparse-attention
    # practice: first 20% calls and first transformer layer remain dense.
    CompensationSetting(
        "stage32_centroid25_warm20_layer1",
        density16=1.0,
        density32=0.25,
        block_side32=4,
        centroid_compensation=True,
        dense_warm_fraction=0.20,
        dense_warm_layers=1,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid50_warm10_layer1",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        dense_warm_fraction=0.10,
        dense_warm_layers=1,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid50_warm20_layer1",
        density16=1.0,
        density32=0.50,
        block_side32=4,
        centroid_compensation=True,
        dense_warm_fraction=0.20,
        dense_warm_layers=1,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid375_warm20_layer1",
        density16=1.0,
        density32=0.375,
        block_side32=4,
        centroid_compensation=True,
        dense_warm_fraction=0.20,
        dense_warm_layers=1,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid125_warm20_layer1",
        density16=1.0,
        density32=0.125,
        block_side32=4,
        add_self_block=False,
        centroid_compensation=True,
        dense_warm_fraction=0.20,
        dense_warm_layers=1,
        temporal_route_update=True,
    ),
    # Cheap error-aware proxies: blocks whose current K/V are heterogeneous are
    # harder to approximate by a centroid, so increase their routing priority.
    CompensationSetting(
        "stage32_centroid25_var05",
        density16=1.0,
        density32=0.25,
        block_side32=4,
        centroid_compensation=True,
        variance_route_beta=0.5,
        temporal_route_update=True,
    ),
    CompensationSetting(
        "stage32_centroid25_var10",
        density16=1.0,
        density32=0.25,
        block_side32=4,
        centroid_compensation=True,
        variance_route_beta=1.0,
        temporal_route_update=True,
    ),
    # Aggressive full cascade, included to see whether compensation also makes
    # the weaker 8->16 route usable.
    CompensationSetting(
        "cascade_centroid25",
        density16=0.25,
        density32=0.25,
        block_side16=2,
        block_side32=4,
        centroid_compensation=True,
        temporal_route_update=True,
    ),
)


def _block_ids(tokens: int, block_side: int, device: torch.device) -> torch.Tensor:
    grid = int(math.isqrt(tokens))
    if grid * grid != tokens or grid % block_side:
        raise ValueError(f"invalid token grid N={tokens}, block_side={block_side}")
    y, x = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    axis = grid // block_side
    return ((y // block_side) * axis + x // block_side).flatten().long()


def _block_mean(x: torch.Tensor, block_ids: torch.Tensor, blocks: int) -> torch.Tensor:
    """Mean [B,H,N,D] tokens into [B,H,M,D] spatial blocks."""
    batch, heads, _, width = x.shape
    out = x.new_zeros(batch, heads, blocks, width)
    index = block_ids.view(1, 1, -1, 1).expand(batch, heads, -1, width)
    out.scatter_add_(2, index, x)
    counts = torch.bincount(block_ids, minlength=blocks).to(x.dtype)
    return out / counts.view(1, 1, blocks, 1)


def _variance_adjusted_route(
    route: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    blocks: int,
    beta: float,
) -> torch.Tensor:
    if beta <= 0:
        return route
    k_mean = _block_mean(k.float(), block_ids, blocks)
    v_mean = _block_mean(v.float(), block_ids, blocks)
    k_error = (k.float() - k_mean[:, :, block_ids]).square().mean(-1)
    v_error = (v.float() - v_mean[:, :, block_ids]).square().mean(-1)
    error = _block_mean((k_error + v_error).unsqueeze(-1), block_ids, blocks).squeeze(-1)
    error = error / error.mean(-1, keepdim=True).clamp_min(1e-12)
    return route * (1.0 + beta * error[:, :, None, :])


def _block_mask_from_mass(
    route: torch.Tensor,
    block_ids: torch.Tensor,
    threshold: float,
    add_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive per-query Top-P block mask for the quality oracle."""
    blocks = route.shape[-1]
    normalized = route.clamp_min(0)
    normalized = normalized / normalized.sum(-1, keepdim=True).clamp_min(1e-12)
    values, order = normalized.sort(dim=-1, descending=True)
    # Keep the first block crossing the threshold as well.
    keep_sorted = values.cumsum(-1) - values < threshold
    selected = torch.zeros_like(keep_sorted)
    selected.scatter_(-1, order, keep_sorted)
    if add_self:
        diagonal = torch.eye(blocks, device=route.device, dtype=torch.bool)
        selected = selected | diagonal.view(1, 1, blocks, blocks)
    block_index_values = torch.arange(blocks, device=route.device).view(1, 1, 1, blocks)
    block_indices = torch.where(selected, block_index_values, -1)
    query_selected = selected[:, :, block_ids, :]
    token_mask = query_selected[..., block_ids]
    return token_mask, block_indices


class CompensationController(HierarchicalController):
    setting: CompensationSetting

    def __init__(self, setting: CompensationSetting, depth: int, total_calls: dict[int, int]):
        super().__init__(setting, depth)
        self.total_calls = total_calls
        self.stats.update(
            {
                "centroid_calls": {"16": 0, "32": 0},
                "variance_route_calls": {"16": 0, "32": 0},
                "warm_dense_calls": {"16": 0, "32": 0},
                "route_mass_calls": {"16": 0, "32": 0},
            }
        )

    def _warm_dense(self, resolution: int, layer: int) -> bool:
        if layer < self.setting.dense_warm_layers:
            return True
        calls = max(self.total_calls[resolution], 1)
        return self.call_index[resolution] < math.ceil(self.setting.dense_warm_fraction * calls)

    def attention(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        assert self.resolution is not None
        resolution = self.resolution
        key = str(resolution)
        self.stats["attention_calls"][key] += 1

        density = 1.0 if resolution == 8 else self._density(resolution, layer)
        if self.setting.dense or density >= 1.0 or self._warm_dense(resolution, layer):
            attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            self.current_attention[resolution][layer] = attention.detach()
            self.stats["dense_calls"][key] += 1
            if resolution != 8 and not self.setting.dense and self._warm_dense(resolution, layer):
                self.stats["warm_dense_calls"][key] += 1
            return attention @ v

        route = self._route(resolution, layer, q.shape[-2])
        side = self._block_side(resolution)
        block_ids = _block_ids(q.shape[-2], side, q.device)
        blocks = int(block_ids.max().item()) + 1
        route = _variance_adjusted_route(
            route, k, v, block_ids, blocks, self.setting.variance_route_beta
        )
        if self.setting.variance_route_beta > 0:
            self.stats["variance_route_calls"][key] += 1

        if self.setting.route_mass_threshold > 0:
            if self.setting.add_neighbor_blocks:
                raise NotImplementedError("Top-P oracle does not combine neighbor dilation")
            token_mask, block_indices = _block_mask_from_mass(
                route,
                block_ids,
                self.setting.route_mass_threshold,
                self.setting.add_self_block,
            )
            self.stats["route_mass_calls"][key] += 1
        else:
            token_mask, block_indices = block_mask_from_route(
                route,
                q.shape[-2],
                density,
                self.setting.add_self_block,
                self.setting.add_neighbor_blocks,
                target_block_side=side,
            )
        logits = q @ k.transpose(-2, -1) * scale

        if not self.setting.centroid_compensation:
            attention = logits.masked_fill(~token_mask, -torch.inf).softmax(-1)
            self.current_attention[resolution][layer] = attention.detach()
            output = attention @ v
        else:
            k_mean = _block_mean(k, block_ids, blocks)
            v_mean = _block_mean(v, block_ids, blocks)
            block_size = q.shape[-2] // blocks
            centroid_logits = q @ k_mean.transpose(-2, -1) * scale + math.log(block_size)

            # Sampling the first token in each key block converts the token mask
            # back to [B,H,N,M] selected-block flags.
            grid = int(math.isqrt(q.shape[-2]))
            axis = grid // side
            block_y, block_x = torch.meshgrid(
                torch.arange(axis, device=q.device),
                torch.arange(axis, device=q.device),
                indexing="ij",
            )
            first_token = ((block_y * side) * grid + block_x * side).flatten().long()
            selected_blocks = token_mask[..., first_token]

            exact_logits = logits.masked_fill(~token_mask, -torch.inf)
            centroid_logits = centroid_logits.masked_fill(selected_blocks, -torch.inf)
            weights = torch.cat((exact_logits, centroid_logits), dim=-1).softmax(-1)
            exact_weights = weights[..., : q.shape[-2]]
            centroid_weights = weights[..., q.shape[-2] :]
            output = exact_weights @ v + centroid_weights @ v_mean

            # Produce a full approximate A solely for the next-call/stage route.
            # Each centroid probability is distributed uniformly to its block.
            approximate_attention = exact_weights + centroid_weights[..., block_ids] / block_size
            self.current_attention[resolution][layer] = approximate_attention.detach()
            self.stats["centroid_calls"][key] += 1

        self.stats["sparse_qk_calls"][key] += 1
        effective_density = float(
            (block_indices >= 0).sum(-1).float().mean().item() / blocks
        )
        self.stats["selected_block_density_sum"][key] += effective_density
        self.stats["selected_block_density_count"][key] += 1
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
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    steps = tuple(int(x) for x in args.sample_steps.split(","))
    if len(steps) != 3:
        raise ValueError("--sample-steps must be coarse,middle,fine")
    total_calls = dict(zip((8, 16, 32), steps))

    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(
        setting
        for setting in SETTINGS
        if requested is None or setting.name in requested or setting.dense
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

    images: dict[str, list[Image.Image]] = {}
    decoded: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "scope": "training-free quality oracle; dense logits are materialized before masking",
        "model": "RDD Unified XL/2 400K",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(steps),
        "variants": {},
    }

    for setting in settings:
        print(f"sampling {setting.name}", flush=True)
        controller = None
        if not setting.dense:
            controller = CompensationController(setting, len(model.blocks), total_calls)
        out, latent, stats = sample_variant(
            model,
            diffusions,
            vae,
            labels,
            num_classes,
            args.cfg_scale,
            args.seed,
            controller,
        )
        decoded[setting.name] = out.cpu()
        latents[setting.name] = latent.cpu()
        images[setting.name] = tensor_to_images(decoded[setting.name])
        if setting.name != "dense":
            stats["image_vs_dense"] = image_metrics(decoded["dense"], decoded[setting.name])
            stats["latent_vs_dense"] = image_metrics(latents["dense"], latents[setting.name])
        stats["setting"] = asdict(setting)
        report["variants"][setting.name] = stats

        variant_dir = output_dir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        gc.collect()
        torch.cuda.empty_cache()

    grid = output_dir / "centroid_compensation_comparison.png"
    save_comparison_grid(
        images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: training-free current-fine centroid compensation",
    )
    report["comparison_grid"] = str(grid)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {grid}", flush=True)


if __name__ == "__main__":
    main()
