#!/usr/bin/env python3
"""Measure RDD self-attention evolution across timesteps and resolutions.

This is a diagnostic oracle, not an acceleration implementation.  It runs the
ordinary dense sampler and independently reconstructs attention probabilities
for a few representative DiT layers.  The recorded probabilities answer two
different reuse questions:

1. Within a fixed-resolution stage, how accurately can A[t-d] replace A[t]?
2. Across an RDD stage boundary, how accurately can the final coarse-stage
   attention replace the attention at successive fine-stage steps?

For score reuse we always apply the old probability matrix to the *current* V,
which isolates reuse of softmax(QK^T) from reuse of the entire attention output.
We also report output broadcasting (reuse of the old A@V) for comparison with
methods such as PAB/DiTFastAttn.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from sample_rdd_cross_stage_sparse import (  # noqa: E402
    _qkv,
    create_sampling_diffusions,
    load_multistage,
    load_unified,
    upsample_with_noise,
)


@dataclass
class AttentionSnapshot:
    step: int
    timestep: int
    resolution: int
    attention: torch.Tensor  # CPU fp16 [1, sampled_heads, N, N]
    value: torch.Tensor  # CPU fp16 [1, sampled_heads, N, D]
    output: torch.Tensor  # CPU fp16 [1, sampled_heads, N, D]
    route: torch.Tensor  # CPU fp16 [1, sampled_heads, R, R]


def root_mapping(grid: int, tile: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(grid, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    roots_axis = grid // tile
    return ((y // tile) * roots_axis + (x // tile)).flatten().long()


def aggregate_attention(
    attention: torch.Tensor, grid: int, tile: int
) -> torch.Tensor:
    """Aggregate token attention into root-to-root probability distributions."""
    mapping = root_mapping(grid, tile, attention.device)
    roots = int(mapping.max().item()) + 1
    membership = F.one_hot(mapping, roots).to(attention.dtype)
    key_mass = attention @ membership
    # index_add accumulates fine-query rows into their parent root.
    out = torch.zeros(
        attention.shape[0],
        attention.shape[1],
        roots,
        roots,
        device=attention.device,
        dtype=attention.dtype,
    )
    out.index_add_(2, mapping, key_mass)
    counts = torch.bincount(mapping, minlength=roots).to(attention.dtype)
    out = out / counts.view(1, 1, -1, 1).clamp_min(1)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


class DynamicsCapture:
    def __init__(
        self,
        models: dict[int, torch.nn.Module],
        layers: list[int],
        unified: bool,
        sampled_heads: int,
    ):
        self.models = models
        self.layers = layers
        self.unified = unified
        self.sampled_heads = sampled_heads
        target_depth = len(models[32].blocks)
        self.layer_map: dict[int, dict[int, int]] = {}
        for resolution, model in models.items():
            depth = len(model.blocks)
            self.layer_map[resolution] = {
                target_layer: round(
                    target_layer / max(target_depth - 1, 1) * max(depth - 1, 1)
                )
                for target_layer in layers
            }
        self.current_resolution: int | None = None
        self.current_step = -1
        self.current_timestep = -1
        self.stage_steps = {8: -1, 16: -1, 32: -1}
        self.snapshots: dict[int, dict[int, list[AttentionSnapshot]]] = {
            resolution: {layer: [] for layer in layers} for resolution in (8, 16, 32)
        }
        self.handles: list[Any] = []

        for resolution, model in models.items():
            for layer in layers:
                actual_layer = self.layer_map[resolution][layer]
                module = model.blocks[actual_layer].attn

                def hook(attn_module, args, *, layer_id=layer, expected_resolution=resolution):
                    if self.current_resolution != expected_resolution:
                        return
                    q, k, v, scale = _qkv(attn_module, args, self.unified)
                    q = q[:1, : self.sampled_heads].float()
                    k = k[:1, : self.sampled_heads].float()
                    v = v[:1, : self.sampled_heads].float()
                    attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
                    output = attention @ v
                    grid = int(math.isqrt(attention.shape[-1]))
                    tile = 2 if grid > 8 else 1
                    route = aggregate_attention(attention, grid, tile)
                    self.snapshots[expected_resolution][layer_id].append(
                        AttentionSnapshot(
                            step=self.current_step,
                            timestep=self.current_timestep,
                            resolution=expected_resolution,
                            attention=attention.detach().to("cpu", torch.float16),
                            value=v.detach().to("cpu", torch.float16),
                            output=output.detach().to("cpu", torch.float16),
                            route=route.detach().to("cpu", torch.float16),
                        )
                    )

                self.handles.append(module.register_forward_pre_hook(hook))

    def begin_forward(self, resolution: int, timestep: torch.Tensor) -> None:
        self.current_resolution = resolution
        self.stage_steps[resolution] += 1
        self.current_step = self.stage_steps[resolution]
        self.current_timestep = int(timestep.flatten()[0].item())

    def end_forward(self) -> None:
        self.current_resolution = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def mean_distribution_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float().clamp_min(0)
    b = b.float().clamp_min(0)
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    b = b / b.sum(-1, keepdim=True).clamp_min(1e-12)
    cosine = F.cosine_similarity(a, b, dim=-1)
    midpoint = 0.5 * (a + b)
    js = 0.5 * (
        (a * (a.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
        + (b * (b.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
    )
    result = {
        "cosine": float(cosine.mean()),
        "cosine_p10": float(torch.quantile(cosine.flatten(), 0.1)),
        "js": float(js.mean()),
    }
    roots = a.shape[-1]
    for density in (0.1, 0.2, 0.4):
        keep = max(1, math.ceil(density * roots))
        ia = a.topk(keep, dim=-1).indices
        ib = b.topk(keep, dim=-1).indices
        ma = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, ia, True)
        mb = torch.zeros_like(b, dtype=torch.bool).scatter_(-1, ib, True)
        result[f"top{int(density * 100)}_overlap"] = float(
            (ma & mb).sum(-1).float().mean() / keep
        )
        result[f"top{int(density * 100)}_old_mass_in_current"] = float(
            (b * ma).sum(-1).mean()
        )
    return result


def tensor_output_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    return {
        "cosine": float(F.cosine_similarity(candidate, reference, dim=-1).mean()),
        "relative_l2": float(
            (candidate - reference).norm() / reference.norm().clamp_min(1e-12)
        ),
    }


def compare_same_resolution(
    old: AttentionSnapshot,
    current: AttentionSnapshot,
    device: torch.device,
) -> dict[str, Any]:
    old_attention = old.attention.to(device=device, dtype=torch.float32)
    current_attention = current.attention.to(device=device, dtype=torch.float32)
    current_value = current.value.to(device=device, dtype=torch.float32)
    current_output = current.output.to(device=device, dtype=torch.float32)
    old_output = old.output.to(device=device, dtype=torch.float32)
    score_reuse_output = old_attention @ current_value
    return {
        "old_step": old.step,
        "current_step": current.step,
        "old_timestep": old.timestep,
        "current_timestep": current.timestep,
        "full_score": mean_distribution_metrics(old_attention, current_attention),
        "root_route": mean_distribution_metrics(old.route, current.route),
        "reuse_old_score_with_current_v": tensor_output_metrics(
            score_reuse_output, current_output
        ),
        "broadcast_old_attention_output": tensor_output_metrics(old_output, current_output),
    }


def expand_coarse_attention(
    coarse: torch.Tensor,
    fine_tokens: int,
    force_head_average: bool,
) -> torch.Tensor:
    coarse = coarse.float()
    fine_grid = int(math.isqrt(fine_tokens))
    coarse_grid = int(math.isqrt(coarse.shape[-1]))
    scale = fine_grid // coarse_grid
    mapping = root_mapping(fine_grid, scale, coarse.device)
    if force_head_average:
        coarse = coarse.mean(1, keepdim=True)
    expanded = coarse[:, :, mapping, :][:, :, :, mapping]
    return expanded / float(scale * scale)


def compare_cross_resolution(
    coarse: AttentionSnapshot,
    fine: AttentionSnapshot,
    device: torch.device,
    force_head_average: bool,
) -> dict[str, Any]:
    coarse_attention = coarse.attention.to(device=device, dtype=torch.float32)
    fine_attention = fine.attention.to(device=device, dtype=torch.float32)
    fine_value = fine.value.to(device=device, dtype=torch.float32)
    fine_output = fine.output.to(device=device, dtype=torch.float32)
    coarse_grid = int(math.isqrt(coarse_attention.shape[-1]))
    fine_grid = int(math.isqrt(fine_attention.shape[-1]))
    transition_scale = fine_grid // coarse_grid
    # The 16x16 stage participates in both 8->16 and 16->32 comparisons.
    # Aggregate the fine map to the source grid per pair instead of reusing one
    # fixed pre-aggregated representation.
    coarse_route = aggregate_attention(coarse_attention, coarse_grid, 1)
    fine_route = aggregate_attention(fine_attention, fine_grid, transition_scale)
    if force_head_average:
        coarse_route = coarse_route.mean(1, keepdim=True)
        fine_route = fine_route.mean(1, keepdim=True)
    else:
        heads = min(coarse_route.shape[1], fine_route.shape[1])
        coarse_route = coarse_route[:, :heads]
        fine_route = fine_route[:, :heads]
    expanded = expand_coarse_attention(
        coarse_attention,
        fine.attention.shape[-1],
        force_head_average=force_head_average,
    )
    if expanded.shape[1] == 1 and fine_value.shape[1] > 1:
        expanded = expanded.expand(-1, fine_value.shape[1], -1, -1)
    heads = min(expanded.shape[1], fine_value.shape[1])
    reused_output = expanded[:, :heads] @ fine_value[:, :heads]
    return {
        "coarse_step": coarse.step,
        "fine_step": fine.step,
        "coarse_timestep": coarse.timestep,
        "fine_timestep": fine.timestep,
        "root_route": mean_distribution_metrics(coarse_route, fine_route),
        "reuse_coarse_score_with_fine_v": tensor_output_metrics(
            reused_output, fine_output[:, :heads]
        ),
    }


@torch.inference_mode()
def run_dense_trajectory(
    models: dict[int, torch.nn.Module],
    diffusions: dict[int, Any],
    label: int,
    seed: int,
    capture: DynamicsCapture,
    device: torch.device,
) -> float:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    y = torch.tensor([label], device=device, dtype=torch.long)
    pred_xstart = None
    start = time.perf_counter()
    for stage_index, resolution in enumerate((8, 16, 32)):
        model = models[resolution]
        diffusion = diffusions[resolution]
        if stage_index == 0:
            state = torch.randn(1, 4, resolution, resolution, device=device)
        else:
            assert pred_xstart is not None
            state = upsample_with_noise(pred_xstart, diffusion, resolution)

        def model_forward(x, t, **kwargs):
            capture.begin_forward(resolution, t)
            try:
                return model(x, t, kwargs["y"])
            finally:
                capture.end_forward()

        output = diffusion.p_sample_loop(
            model_forward,
            state.shape,
            noise=state,
            clip_denoised=False,
            model_kwargs={"y": y},
            progress=False,
            device=device,
        )
        pred_xstart = output["pred_xstart"].float()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def analyze_capture(
    capture: DynamicsCapture,
    device: torch.device,
    unified: bool,
    lags: tuple[int, ...],
) -> dict[str, Any]:
    report: dict[str, Any] = {"within_resolution": {}, "cross_resolution": {}}
    for resolution in (8, 16, 32):
        report["within_resolution"][str(resolution)] = {}
        for layer in capture.layers:
            items = capture.snapshots[resolution][layer]
            by_step = {item.step: item for item in items}
            layer_report: dict[str, list[dict[str, Any]]] = {}
            for lag in lags:
                comparisons = []
                # Evaluate fixed ten-step anchors so lags are directly comparable.
                anchors = sorted(
                    step for step in by_step if step >= lag and (step % 10 == 0 or step == len(items) - 1)
                )
                for step in anchors:
                    if step - lag not in by_step:
                        continue
                    comparisons.append(
                        compare_same_resolution(by_step[step - lag], by_step[step], device)
                    )
                layer_report[f"lag_{lag}"] = comparisons
            report["within_resolution"][str(resolution)][str(layer)] = layer_report

    for coarse_resolution, fine_resolution in ((8, 16), (16, 32)):
        pair_key = f"{coarse_resolution}_to_{fine_resolution}"
        report["cross_resolution"][pair_key] = {}
        for layer in capture.layers:
            coarse_items = capture.snapshots[coarse_resolution][layer]
            fine_items = capture.snapshots[fine_resolution][layer]
            coarse = coarse_items[-1]
            selected_steps = {0, 1, 5, 10, 20, 30, 45, 60, 75, len(fine_items) - 1}
            comparisons = []
            for fine in fine_items:
                if fine.step not in selected_steps:
                    continue
                comparisons.append(
                    compare_cross_resolution(
                        coarse,
                        fine,
                        device,
                        force_head_average=not unified,
                    )
                )
            report["cross_resolution"][pair_key][str(layer)] = comparisons
    return report


def flatten_numeric(items: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for item in items:
        value: Any = item
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"within_resolution": {}, "cross_resolution": {}}
    for resolution, layers in report["within_resolution"].items():
        summary["within_resolution"][resolution] = {}
        for lag in (1, 5, 10):
            items = [
                item
                for layer in layers.values()
                for item in layer.get(f"lag_{lag}", [])
            ]
            if not items:
                continue
            summary["within_resolution"][resolution][f"lag_{lag}"] = {
                "comparisons": len(items),
                "score_cosine": sum(flatten_numeric(items, ("full_score", "cosine"))) / len(items),
                "route_cosine": sum(flatten_numeric(items, ("root_route", "cosine"))) / len(items),
                "route_top40_overlap": sum(
                    flatten_numeric(items, ("root_route", "top40_overlap"))
                )
                / len(items),
                "score_reuse_output_cosine": sum(
                    flatten_numeric(items, ("reuse_old_score_with_current_v", "cosine"))
                )
                / len(items),
                "score_reuse_output_relative_l2": sum(
                    flatten_numeric(items, ("reuse_old_score_with_current_v", "relative_l2"))
                )
                / len(items),
                "broadcast_output_cosine": sum(
                    flatten_numeric(items, ("broadcast_old_attention_output", "cosine"))
                )
                / len(items),
            }
    for pair, layers in report["cross_resolution"].items():
        items = [item for layer in layers.values() for item in layer]
        summary["cross_resolution"][pair] = {
            "comparisons": len(items),
            "route_cosine": sum(flatten_numeric(items, ("root_route", "cosine"))) / len(items),
            "route_top40_overlap": sum(
                flatten_numeric(items, ("root_route", "top40_overlap"))
            )
            / len(items),
            "direct_score_reuse_output_cosine": sum(
                flatten_numeric(items, ("reuse_coarse_score_with_fine_v", "cosine"))
            )
            / len(items),
            "direct_score_reuse_output_relative_l2": sum(
                flatten_numeric(items, ("reuse_coarse_score_with_fine_v", "relative_l2"))
            )
            / len(items),
        }
    return summary


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("unified", "multistage"), default="unified")
    parser.add_argument("--unified-ckpt")
    parser.add_argument("--multistage-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-steps", default="80,80,90")
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--sampled-heads", type=int, default=4)
    parser.add_argument("--lags", default="1,5,10")
    parser.add_argument("--label", type=int, default=207)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    layers = list(parse_int_tuple(args.layers))
    sample_steps = parse_int_tuple(args.sample_steps)
    if len(sample_steps) != 3:
        raise ValueError("--sample-steps must be coarse,middle,fine")

    if args.family == "unified":
        if not args.unified_ckpt:
            raise ValueError("--unified-ckpt is required")
        model, _ = load_unified(Path(args.unified_ckpt), device)
        models = {8: model, 16: model, 32: model}
        unified = True
    else:
        if not args.multistage_dir:
            raise ValueError("--multistage-dir is required")
        models, _ = load_multistage(Path(args.multistage_dir), device)
        unified = False

    target_depth = len(models[32].blocks)
    if any(layer < 0 or layer >= target_depth for layer in layers):
        raise ValueError(f"layers must be within target-model range [0, {target_depth - 1}]")
    diffusions = create_sampling_diffusions(sample_steps)
    capture = DynamicsCapture(models, layers, unified, args.sampled_heads)
    try:
        elapsed = run_dense_trajectory(
            models, diffusions, args.label, args.seed, capture, device
        )
        detailed = analyze_capture(capture, device, unified, parse_int_tuple(args.lags))
    finally:
        capture.close()

    result = {
        "scope": "dense diagnostic oracle; attention reconstructed for selected layers",
        "family": args.family,
        "sample_steps_coarse_to_fine": list(sample_steps),
        "layers": layers,
        "layers_by_resolution": capture.layer_map,
        "sampled_heads": args.sampled_heads,
        "label": args.label,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "reuse_definition": {
            "score_reuse": "A_old @ V_current",
            "output_broadcast": "A_old @ V_old",
            "cross_resolution": "coarse A expanded uniformly to child keys, then @ V_fine",
        },
        "summary": summarize(detailed),
        "detailed": detailed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"saved: {output}", flush=True)

    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
