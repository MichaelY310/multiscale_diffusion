#!/usr/bin/env python3
"""RDD hierarchical-attention correctness oracle.

This file explores three independent inference approximations on the existing
Unified XL checkpoint:

1. cascade route: 8-stage dense attention routes 16-stage sparse attention;
   the resulting 16-stage sparse attention routes the 32 stage;
2. temporal score cache: reuse softmax(QK^T) for several denoiser calls while
   applying it to the current V;
3. cross-resolution score cache: lift the previous-stage probability map to
   child tokens and apply it to the current V in selected early layers.

The oracle deliberately materializes dense logits before masking.  It is for
quality search only.  The selected policy is executed by the separate Triton
backend without forming the dense matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKL
from PIL import Image

from sample_rdd_cross_stage_sparse import (
    DEFAULT_CLASSES,
    SAMPLE_STEPS_COARSE_TO_FINE,
    STAGE_LENGTHS,
    _qkv,
    cfg_model_call,
    create_sampling_diffusions,
    image_metrics,
    load_unified,
    save_comparison_grid,
    tensor_to_images,
    upsample_with_noise,
)


@dataclass(frozen=True)
class HierarchicalSetting:
    name: str
    density16: float = 1.0
    density32: float = 1.0
    middle_density16: float | None = None
    middle_density32: float | None = None
    block_side16: int = 4
    block_side32: int = 4
    # Recompute sparse QK/softmax every K calls.  Calls in between use
    # A_cached @ V_current.  One disables temporal score reuse.
    score_interval16: int = 1
    score_interval32: int = 1
    middle_score_interval16: int | None = None
    middle_score_interval32: int | None = None
    late_score_interval16: int | None = None
    late_score_interval32: int | None = None
    # Score reuse is only enabled for layers before these exclusive limits.
    score_layers16: int = 0
    score_layers32: int = 0
    # Directly lift the previous-resolution A in early layers/calls.
    cross_score_calls16: int = 0
    cross_score_calls32: int = 0
    cross_score_layers16: int = 0
    cross_score_layers32: int = 0
    add_self_block: bool = True
    add_neighbor_blocks: bool = False
    # Use the most recent current-stage sparse A as the route proposal.  When
    # false, the route remains the previous-stage boundary route.
    temporal_route_update: bool = True
    # "legacy" keeps the original one-spatial-block-per-loop Triton backend.
    # "packed64" groups several selected spatial blocks into a 64-token tile;
    # it changes only execution, never the selected route or softmax semantics.
    kernel_backend: str = "legacy"

    @property
    def dense(self) -> bool:
        return self.name == "dense"


SETTINGS = (
    HierarchicalSetting("dense"),
    HierarchicalSetting("cascade_r75_50", density16=0.75, density32=0.50),
    HierarchicalSetting(
        "cascade_fixed_r75_50",
        density16=0.75,
        density32=0.50,
        temporal_route_update=False,
    ),
    HierarchicalSetting("cascade_r75_60", density16=0.75, density32=0.60),
    HierarchicalSetting(
        "cascade_r75_50_nbr", density16=0.75, density32=0.50, add_neighbor_blocks=True
    ),
    HierarchicalSetting(
        "cascade_score5_early",
        density16=0.75,
        density32=0.50,
        score_interval16=5,
        score_interval32=5,
        score_layers16=7,
        score_layers32=7,
    ),
    HierarchicalSetting(
        "cascade_fixed_score5_early",
        density16=0.75,
        density32=0.50,
        score_interval16=5,
        score_interval32=5,
        score_layers16=7,
        score_layers32=7,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_micro16_r50_50",
        density16=0.50,
        density32=0.50,
        block_side16=2,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_micro16_r75_50",
        density16=0.75,
        density32=0.50,
        block_side16=2,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_micro16_r75_60_score10",
        density16=0.75,
        density32=0.60,
        block_side16=2,
        score_interval32=10,
        score_layers32=7,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "dense_route_score5_early",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        score_layers16=7,
        score_layers32=7,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "dense_route_score5_all",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "dense_route_score10_all",
        density16=1.0,
        density32=1.0,
        score_interval16=10,
        score_interval32=10,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "dense_route_score3_all",
        density16=1.0,
        density32=1.0,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "dense_route_score_adaptive_5_3_3",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        middle_score_interval16=3,
        middle_score_interval32=3,
        late_score_interval16=3,
        late_score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_safe_layeraware_score10",
        density16=0.875,
        density32=0.75,
        middle_density16=1.0,
        middle_density32=0.875,
        block_side16=2,
        score_interval32=10,
        score_layers32=7,
        cross_score_calls32=5,
        cross_score_layers32=4,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_safe_layeraware_score5_all",
        density16=0.875,
        density32=0.75,
        middle_density16=1.0,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=5,
        score_interval32=5,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_safe_layeraware_score3_all",
        density16=0.875,
        density32=0.75,
        middle_density16=1.0,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_safe_layeraware_adaptive_5_3_3",
        density16=0.875,
        density32=0.75,
        middle_density16=1.0,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=5,
        score_interval32=5,
        middle_score_interval16=3,
        middle_score_interval32=3,
        late_score_interval16=3,
        late_score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "stage32_layeraware_score3_all",
        density16=1.0,
        density32=0.75,
        middle_density32=0.875,
        score_interval32=3,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "stage32_layeraware_adaptive_5_3_3",
        density16=1.0,
        density32=0.75,
        middle_density32=0.875,
        score_interval32=5,
        middle_score_interval32=3,
        late_score_interval32=3,
        score_layers32=28,
        cross_score_calls32=5,
        cross_score_layers32=4,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "cascade_score10_early",
        density16=0.75,
        density32=0.50,
        score_interval16=10,
        score_interval32=10,
        score_layers16=7,
        score_layers32=7,
    ),
    HierarchicalSetting(
        "cascade_crossA5_early",
        density16=0.75,
        density32=0.50,
        cross_score_calls32=5,
        cross_score_layers32=4,
    ),
    HierarchicalSetting(
        "cascade_crossA10_score5",
        density16=0.75,
        density32=0.60,
        score_interval32=5,
        score_layers32=7,
        cross_score_calls32=10,
        cross_score_layers32=4,
        add_neighbor_blocks=True,
    ),
)


def _grid(tokens: int) -> int:
    side = int(math.isqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"attention token count is not square: {tokens}")
    return side


def _spatial_block_ids(grid: int, block_side: int, device: torch.device) -> torch.Tensor:
    if grid % block_side:
        raise ValueError(f"grid {grid} is not divisible by block side {block_side}")
    y, x = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    blocks_axis = grid // block_side
    return ((y // block_side) * blocks_axis + (x // block_side)).flatten().long()


def aggregate_attention(
    attention: torch.Tensor,
    query_block_side: int,
    key_block_side: int,
) -> torch.Tensor:
    """Aggregate token probabilities to block-to-block probability routes."""
    batch, heads, nq, nk = attention.shape
    gq, gk = _grid(nq), _grid(nk)
    q_ids = _spatial_block_ids(gq, query_block_side, attention.device)
    k_ids = _spatial_block_ids(gk, key_block_side, attention.device)
    q_blocks = int(q_ids.max()) + 1
    k_blocks = int(k_ids.max()) + 1

    # Sum probability over key tokens in a key block, then average query rows
    # in the query block.  Every output row remains a probability distribution.
    by_key = torch.zeros(
        batch, heads, nq, k_blocks, device=attention.device, dtype=attention.dtype
    )
    by_key.scatter_add_(
        -1,
        k_ids.view(1, 1, 1, nk).expand(batch, heads, nq, nk),
        attention,
    )
    route = torch.zeros(
        batch, heads, q_blocks, k_blocks, device=attention.device, dtype=attention.dtype
    )
    route.scatter_add_(
        -2,
        q_ids.view(1, 1, nq, 1).expand(batch, heads, nq, k_blocks),
        by_key,
    )
    counts = torch.bincount(q_ids, minlength=q_blocks).to(route.dtype)
    return route / counts.view(1, 1, q_blocks, 1)


def block_mask_from_route(
    route: torch.Tensor,
    target_tokens: int,
    density: float,
    add_self: bool,
    add_neighbors: bool,
    target_block_side: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a block route into a token mask and selected block indices."""
    grid = _grid(target_tokens)
    block_ids = _spatial_block_ids(grid, target_block_side, route.device)
    blocks = int(block_ids.max()) + 1
    if route.shape[-2:] != (blocks, blocks):
        raise ValueError(
            f"route {tuple(route.shape[-2:])} cannot address {blocks} target blocks"
        )
    keep = max(1, min(blocks, math.ceil(density * blocks)))
    selected = torch.zeros_like(route, dtype=torch.bool)
    selected.scatter_(-1, route.topk(keep, dim=-1).indices, True)
    if add_self:
        diag = torch.arange(blocks, device=route.device)
        selected[:, :, diag, diag] = True
    if add_neighbors:
        axis = int(math.isqrt(blocks))
        expanded = selected.clone()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for query in range(blocks):
                y, x = divmod(query, axis)
                yy, xx = y + dy, x + dx
                if 0 <= yy < axis and 0 <= xx < axis:
                    expanded[:, :, query, yy * axis + xx] = True
        selected = expanded
    token_mask = selected[:, :, block_ids, :][:, :, :, block_ids]
    # Fixed-width index representation used by the real sparse kernel.
    counts = selected.sum(-1)
    max_keep = int(counts.max().item())
    keys = torch.arange(blocks, device=selected.device)
    keys = keys.view(*([1] * (selected.ndim - 1)), blocks).expand_as(selected)
    packed = torch.where(selected, keys, torch.full_like(keys, blocks)).sort(-1).values
    indices = packed[..., :max_keep]
    indices = torch.where(indices < blocks, indices, torch.full_like(indices, -1)).to(torch.int32)
    return token_mask, indices


def expand_cross_resolution_attention(
    source: torch.Tensor,
    target_tokens: int,
) -> torch.Tensor:
    """Lift each coarse probability uniformly to its four child key tokens."""
    source_grid = _grid(source.shape[-1])
    target_grid = _grid(target_tokens)
    if target_grid != 2 * source_grid:
        raise ValueError(f"expected adjacent 2x resolution, got {source_grid}->{target_grid}")
    parent = _spatial_block_ids(target_grid, 2, source.device)
    expanded = source[:, :, parent, :][:, :, :, parent]
    return expanded / 4.0


class HierarchicalController:
    def __init__(self, setting: HierarchicalSetting, depth: int):
        self.setting = setting
        self.depth = depth
        self.resolution: int | None = None
        self.call_index = {8: -1, 16: -1, 32: -1}
        self.boundary_attention: dict[int, dict[int, torch.Tensor]] = {}
        self.current_attention: dict[int, dict[int, torch.Tensor]] = {}
        self.stats: dict[str, Any] = {
            "attention_calls": {"8": 0, "16": 0, "32": 0},
            "dense_calls": {"8": 0, "16": 0, "32": 0},
            "sparse_qk_calls": {"8": 0, "16": 0, "32": 0},
            "temporal_score_reuse_calls": {"8": 0, "16": 0, "32": 0},
            "cross_score_reuse_calls": {"8": 0, "16": 0, "32": 0},
            "selected_block_density_sum": {"16": 0.0, "32": 0.0},
            "selected_block_density_count": {"16": 0, "32": 0},
        }

    def begin_forward(self, x: torch.Tensor) -> None:
        self.resolution = int(x.shape[-1])
        self.call_index[self.resolution] += 1
        self.current_attention.setdefault(self.resolution, {})

    def end_forward(self) -> None:
        assert self.resolution is not None
        # The final denoiser call overwrites this at the end of each stage, so
        # the dict always contains the latest sparse/dense attention map.
        self.boundary_attention[self.resolution] = dict(
            self.current_attention[self.resolution]
        )
        self.resolution = None

    def _density(self, resolution: int, layer: int) -> float:
        fraction = layer / max(self.depth - 1, 1)
        in_middle = 0.15 <= fraction <= 0.85
        if resolution == 16:
            if in_middle and self.setting.middle_density16 is not None:
                return self.setting.middle_density16
            return self.setting.density16
        if in_middle and self.setting.middle_density32 is not None:
            return self.setting.middle_density32
        return self.setting.density32

    def _score_interval(self, resolution: int, layer: int) -> int:
        fraction = layer / max(self.depth - 1, 1)
        if resolution == 16:
            if fraction > 0.85 and self.setting.late_score_interval16 is not None:
                return self.setting.late_score_interval16
            if 0.15 <= fraction <= 0.85 and self.setting.middle_score_interval16 is not None:
                return self.setting.middle_score_interval16
            return self.setting.score_interval16
        if fraction > 0.85 and self.setting.late_score_interval32 is not None:
            return self.setting.late_score_interval32
        if 0.15 <= fraction <= 0.85 and self.setting.middle_score_interval32 is not None:
            return self.setting.middle_score_interval32
        return self.setting.score_interval32

    def _score_layers(self, resolution: int) -> int:
        return self.setting.score_layers16 if resolution == 16 else self.setting.score_layers32

    def _cross_calls(self, resolution: int) -> int:
        return (
            self.setting.cross_score_calls16
            if resolution == 16
            else self.setting.cross_score_calls32
        )

    def _cross_layers(self, resolution: int) -> int:
        return (
            self.setting.cross_score_layers16
            if resolution == 16
            else self.setting.cross_score_layers32
        )

    def _block_side(self, resolution: int) -> int:
        return self.setting.block_side16 if resolution == 16 else self.setting.block_side32

    def _source_attention(self, resolution: int, layer: int) -> torch.Tensor:
        source_resolution = resolution // 2
        try:
            return self.boundary_attention[source_resolution][layer]
        except KeyError as exc:
            raise RuntimeError(
                f"missing {source_resolution}-stage boundary attention for layer {layer}"
            ) from exc

    def _route(self, resolution: int, layer: int, q_tokens: int) -> torch.Tensor:
        call = self.call_index[resolution]
        if self.setting.temporal_route_update and call > 0 and layer in self.current_attention[resolution]:
            side = self._block_side(resolution)
            return aggregate_attention(
                self.current_attention[resolution][layer], side, side
            )
        source = self._source_attention(resolution, layer)
        source_side = self._block_side(resolution) // 2
        return aggregate_attention(source, source_side, source_side)

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

        if self.setting.dense or resolution == 8:
            attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            self.current_attention[resolution][layer] = attention.detach()
            self.stats["dense_calls"][key] += 1
            return attention @ v

        call = self.call_index[resolution]
        previous = self.current_attention[resolution].get(layer)

        # Cross-stage score reuse is intentionally restricted to early layers
        # and early calls; the diagnostics show it fails in late layers.
        if call < self._cross_calls(resolution) and layer < self._cross_layers(resolution):
            attention = expand_cross_resolution_attention(
                self._source_attention(resolution, layer), q.shape[-2]
            ).to(dtype=q.dtype)
            route = self._route(resolution, layer, q.shape[-2])
            token_mask, _ = block_mask_from_route(
                route,
                q.shape[-2],
                self._density(resolution, layer),
                self.setting.add_self_block,
                self.setting.add_neighbor_blocks,
                target_block_side=self._block_side(resolution),
            )
            attention = attention.masked_fill(~token_mask, 0.0)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
            self.current_attention[resolution][layer] = attention.detach()
            self.stats["cross_score_reuse_calls"][key] += 1
            return attention @ v

        interval = self._score_interval(resolution, layer)
        reuse_score = (
            interval > 1
            and layer < self._score_layers(resolution)
            and previous is not None
            and call % interval != 0
        )
        if reuse_score:
            self.stats["temporal_score_reuse_calls"][key] += 1
            return previous.to(dtype=v.dtype) @ v

        route = self._route(resolution, layer, q.shape[-2])
        token_mask, block_indices = block_mask_from_route(
            route,
            q.shape[-2],
            self._density(resolution, layer),
            self.setting.add_self_block,
            self.setting.add_neighbor_blocks,
            target_block_side=self._block_side(resolution),
        )
        logits = q @ k.transpose(-2, -1) * scale
        attention = logits.masked_fill(~token_mask, -torch.inf).softmax(-1)
        self.current_attention[resolution][layer] = attention.detach()
        self.stats["sparse_qk_calls"][key] += 1
        density = float(
            (block_indices >= 0).sum(-1).float().mean() / route.shape[-1]
        )
        self.stats["selected_block_density_sum"][key] += density
        self.stats["selected_block_density_count"][key] += 1
        return attention @ v

    def report(self) -> dict[str, Any]:
        result = {"setting": asdict(self.setting), **self.stats}
        result["mean_selected_block_density"] = {}
        for key in ("16", "32"):
            count = max(1, self.stats["selected_block_density_count"][key])
            result["mean_selected_block_density"][key] = (
                self.stats["selected_block_density_sum"][key] / count
            )
        return result


@contextmanager
def patched_hierarchical_attention(
    model: torch.nn.Module,
    controller: HierarchicalController,
):
    records: list[tuple[torch.nn.Module, Any]] = []
    for layer, block in enumerate(model.blocks):
        module = block.attn
        original = module.forward

        def replacement(self, *args, _layer=layer):
            q, k, v, scale = _qkv(self, args, unified=True)
            output = controller.attention(_layer, q, k, v, scale)
            x = output.transpose(1, 2).reshape(args[0].shape)
            return self.proj(x)

        records.append((module, original))
        module.forward = types.MethodType(replacement, module)
    try:
        yield
    finally:
        for module, original in records:
            module.forward = original


@torch.inference_mode()
def sample_variant(
    model: torch.nn.Module,
    diffusions: dict[int, Any],
    vae: AutoencoderKL,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float,
    seed: int,
    controller: HierarchicalController | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch = labels.shape[0]
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        y = torch.cat((labels, torch.full_like(labels, num_classes)), dim=0)
        batch_multiplier = 2
    else:
        y, batch_multiplier = labels, 1

    pred_xstart = None
    final_sample = None
    elapsed = 0.0
    context = (
        patched_hierarchical_attention(model, controller)
        if controller is not None
        else nullcontext()
    )
    with context:
        for stage_index, resolution in enumerate((8, 16, 32)):
            diffusion = diffusions[resolution]
            if stage_index == 0:
                state = torch.randn(
                    batch_multiplier * batch,
                    4,
                    resolution,
                    resolution,
                    device=labels.device,
                )
            else:
                assert pred_xstart is not None
                state = upsample_with_noise(pred_xstart, diffusion, resolution)
                if use_cfg:
                    state = torch.cat((state, state), 0)

            def model_forward(x, t, **kwargs):
                if controller is not None:
                    controller.begin_forward(x)
                try:
                    if use_cfg:
                        return cfg_model_call(model, x, t, kwargs["y"], cfg_scale)
                    return model(x, t, kwargs["y"])
                finally:
                    if controller is not None:
                        controller.end_forward()

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = diffusion.p_sample_loop(
                model_forward,
                state.shape,
                noise=state,
                clip_denoised=False,
                model_kwargs={"y": y},
                progress=False,
                device=labels.device,
            )
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            final_sample = output["sample"].float()
            pred_xstart = output["pred_xstart"].float()
            if use_cfg:
                final_sample = final_sample[:batch]
                pred_xstart = pred_xstart[:batch]

    assert final_sample is not None
    decoded = vae.decode(final_sample / 0.18215).sample.float()
    report: dict[str, Any] = {"oracle_denoiser_seconds": elapsed}
    if controller is not None:
        report.update(controller.report())
    return decoded, final_sample, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--sample-steps", default=",".join(map(str, SAMPLE_STEPS_COARSE_TO_FINE)))
    parser.add_argument("--settings", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    sample_steps = tuple(int(x) for x in args.sample_steps.split(","))
    if len(sample_steps) != 3:
        raise ValueError("--sample-steps must contain coarse,middle,fine values")
    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(s for s in SETTINGS if requested is None or s.name in requested or s.dense)
    if requested is not None:
        unknown = requested - {s.name for s in SETTINGS}
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    model, num_classes = load_unified(Path(args.unified_ckpt), device)
    diffusions = create_sampling_diffusions(sample_steps)

    images: dict[str, list[Image.Image]] = {}
    decoded: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "scope": "quality oracle; masked implementations still materialize dense logits",
        "model": "RDD Unified XL/2 400K",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(sample_steps),
        "variants": {},
    }

    for setting in settings:
        print(f"sampling {setting.name}", flush=True)
        controller = None if setting.dense else HierarchicalController(setting, len(model.blocks))
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
        decoded[setting.name], latents[setting.name] = out.cpu(), latent.cpu()
        images[setting.name] = tensor_to_images(decoded[setting.name])
        variant_dir = output_dir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        report["variants"][setting.name] = stats
        del out, latent
        gc.collect()
        torch.cuda.empty_cache()

    for name in images:
        if name == "dense":
            continue
        report["variants"][name]["image_vs_dense"] = image_metrics(decoded["dense"], decoded[name])
        report["variants"][name]["latent_vs_dense"] = image_metrics(latents["dense"], latents[name])

    grid = output_dir / "hierarchical_attention_comparison.png"
    save_comparison_grid(
        images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: cascaded routing and attention-score reuse",
    )
    report["comparison_grid"] = str(grid)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {grid}", flush=True)


if __name__ == "__main__":
    main()
