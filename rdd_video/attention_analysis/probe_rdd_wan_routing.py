#!/usr/bin/env python3
"""Probe whether RDD-Wan lower-resolution attention can route fine attention.

This is a falsification experiment, not a quality evaluation.  It supports an
official Wan2.1 checkpoint today and an RDD-Wan checkpoint via --checkpoint
once one exists.  The default diagnostic shape is intentionally small enough
for exact attention-mass analysis while preserving the 4x/2x/1x hierarchy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdd_video.attention_analysis.routing_metrics import (  # noqa: E402
    compare_route_distributions,
    exact_root_attention_mass,
    make_root_layout,
    masked_attention_output_error,
    pooled_route_distribution,
    routed_attention_with_summary_tail_error,
)
from rdd_video.flow_matching_video import RDDVideoSchedule  # noqa: E402


def parse_ints(text: str, expected: int | None = None) -> tuple[int, ...]:
    values = tuple(int(x) for x in text.split(","))
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} integers, got {text!r}")
    return values


def load_clean_latent(path: str | None, shape: tuple[int, int, int, int], seed: int) -> torch.Tensor:
    channels, frames, height, width = shape
    if path:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(item, dict):
            for key in ("latent", "latents", "video_latent", "x"):
                if key in item:
                    item = item[key]
                    if key == "latents" and isinstance(item, (list, tuple)):
                        item = item[-1]
                    break
        if not torch.is_tensor(item):
            raise TypeError(f"could not find a latent tensor in {path}")
        if item.ndim == 5:
            item = item[0]
        if tuple(item.shape) != shape:
            raise ValueError(f"latent is {tuple(item.shape)}, expected {shape}")
        return item.float().unsqueeze(0)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    coarse_shape = (
        1,
        channels,
        max(1, math.ceil(frames / 4)),
        max(1, math.ceil(height / 4)),
        max(1, math.ceil(width / 4)),
    )
    coarse = torch.randn(coarse_shape, generator=generator)
    smooth = F.interpolate(coarse, size=(frames, height, width), mode="trilinear", align_corners=False)
    detail = torch.randn((1, channels, frames, height, width), generator=generator)
    detail = detail - F.avg_pool3d(detail, kernel_size=3, stride=1, padding=1)
    x = smooth + 0.20 * detail
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def load_context(path: str | None, device: torch.device) -> list[torch.Tensor]:
    if not path:
        return [torch.zeros(512, 4096, device=device, dtype=torch.bfloat16)]
    item = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(item, dict):
        for key in ("context", "text_emb", "text_embedding", "prompt_embeds"):
            if key in item:
                item = item[key]
                break
    if isinstance(item, (list, tuple)):
        if len(item) != 1:
            raise ValueError("diagnostic expects one prompt context")
        item = item[0]
    if not torch.is_tensor(item):
        raise TypeError(f"could not find a text context tensor in {path}")
    if item.ndim == 3:
        if item.shape[0] != 1:
            raise ValueError("diagnostic expects context batch size 1")
        item = item[0]
    if item.ndim != 2 or item.shape[-1] != 4096:
        raise ValueError(f"expected context [L,4096], got {tuple(item.shape)}")
    return [item.to(device=device, dtype=torch.bfloat16)]


def load_wan(args, device: torch.device):
    wan_root = Path(args.wan_repo).resolve()
    if not (wan_root / "wan" / "modules" / "model.py").exists():
        raise FileNotFoundError(f"Wan source not found at {wan_root}")
    sys.path.insert(0, str(wan_root))
    import wan.modules.model as wan_model_module
    from wan.modules.attention import attention as safe_attention
    from wan.modules.model import WanModel, rope_apply

    # Official WanSelfAttention imports the strict flash_attention function.
    # The generic wrapper safely chooses FlashAttention when installed and
    # otherwise PyTorch SDPA, which is sufficient for the reduced probe shape.
    wan_model_module.flash_attention = safe_attention

    model = WanModel.from_pretrained(
        args.pretrained_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval().requires_grad_(False)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model, safe_attention, rope_apply


def install_qkv_capture(model, layer_ids, safe_attention, rope_apply):
    bucket: dict[str, dict[int, dict[str, torch.Tensor | tuple[int, int, int]]]] = {}
    state = {"tag": None}

    for layer_id in layer_ids:
        module = model.blocks[layer_id].self_attn

        def captured_forward(x, seq_lens, grid_sizes, freqs, *, _m=module, _layer=layer_id):
            b, s = x.shape[:2]
            heads, dim = _m.num_heads, _m.head_dim
            q = _m.norm_q(_m.q(x)).view(b, s, heads, dim)
            k = _m.norm_k(_m.k(x)).view(b, s, heads, dim)
            v = _m.v(x).view(b, s, heads, dim)
            qr = rope_apply(q, grid_sizes, freqs)
            kr = rope_apply(k, grid_sizes, freqs)
            tag = state["tag"]
            if tag is not None:
                bucket.setdefault(tag, {})[_layer] = {
                    "q_pre_rope": q.detach().to("cpu", torch.float16),
                    "k_pre_rope": k.detach().to("cpu", torch.float16),
                    "q": qr.detach().to("cpu", torch.float16),
                    "k": kr.detach().to("cpu", torch.float16),
                    "v": v.detach().to("cpu", torch.float16),
                    "grid": tuple(int(z) for z in grid_sizes[0].tolist()),
                }
            out = safe_attention(
                qr,
                kr,
                v,
                k_lens=seq_lens,
                window_size=_m.window_size,
            )
            return _m.o(out.flatten(2))

        module.forward = captured_forward
    return bucket, state


def save_heatmap(matrix: torch.Tensor, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    image = matrix.float().mean(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(image, interpolation="nearest", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("key root tile")
    ax.set_ylabel("query root tile")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="optional RDD-Wan checkpoint")
    parser.add_argument("--latent-pt", default=None)
    parser.add_argument("--context-pt", default=None, help="optional precomputed Wan T5 context")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-shape", default="16,8,32,32", help="C,F,H,W")
    parser.add_argument("--layers", default="0,7,15,23,29")
    parser.add_argument("--stage-lengths", default="60,350,590")
    parser.add_argument("--densities", default="0.10,0.20,0.30,0.40")
    parser.add_argument("--boundary-eps", type=float, default=1e-3)
    parser.add_argument("--fine-reuse-delta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-traces", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the Wan attention probe requires CUDA")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latent_shape = parse_ints(args.latent_shape, 4)
    layer_ids = parse_ints(args.layers)
    densities = tuple(float(x) for x in args.densities.split(","))
    lengths = parse_ints(args.stage_lengths, 3)
    if sum(lengths) != 1000:
        raise ValueError("stage lengths must sum to 1000")

    torch.manual_seed(args.seed)
    model, safe_attention, rope_apply = load_wan(args, device)
    if max(layer_ids) >= len(model.blocks):
        raise ValueError(f"layer index exceeds model depth {len(model.blocks)}")
    traces, capture_state = install_qkv_capture(model, layer_ids, safe_attention, rope_apply)
    schedule = RDDVideoSchedule(
        [
            {"spatial_factor": 4, "temporal_factor": 4},
            {"spatial_factor": 2, "temporal_factor": 2},
            {"spatial_factor": 1, "temporal_factor": 1},
        ],
        list(lengths),
        downsample_mode="latent",
        drift_scale=1.0,
    )
    boundaries = schedule.s_boundaries
    clean = load_clean_latent(args.latent_pt, latent_shape, args.seed).to(device, torch.bfloat16)
    context = load_context(args.context_pt, device)

    def stage_input(stage: int, t_value: float, noise_seed: int) -> torch.Tensor:
        x1 = schedule.downsample_latent(clean, stage)
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
        matcher = schedule.flow_matchers[stage]
        drift = matcher.compute_drift(x1) if matcher.c is not None else None
        t = torch.tensor([t_value], device=device, dtype=torch.float32)
        return matcher.q_sample(x1, t, noise=noise, d=drift)

    calls = [
        ("coarse_end", 0, boundaries[1] - args.boundary_eps, args.seed + 10),
        ("middle_start", 1, boundaries[1] + args.boundary_eps, args.seed + 20),
        ("middle_end", 1, boundaries[2] - args.boundary_eps, args.seed + 20),
        ("fine_start", 2, boundaries[2] + args.boundary_eps, args.seed + 30),
        (
            "fine_next",
            2,
            min(boundaries[2] + args.fine_reuse_delta, 1.0 - args.boundary_eps),
            args.seed + 30,
        ),
    ]
    call_meta = {}
    with torch.inference_mode():
        for tag, stage, t_value, noise_seed in calls:
            x = stage_input(stage, t_value, noise_seed)
            seq_len = (x.shape[2] // model.patch_size[0]) * (x.shape[3] // model.patch_size[1]) * (
                x.shape[4] // model.patch_size[2]
            )
            capture_state["tag"] = tag
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model([x[0]], t=torch.tensor([(1.0 - t_value) * 1000], device=device), context=context, seq_len=seq_len)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            capture_state["tag"] = None
            call_meta[tag] = {
                "stage": stage,
                "data_time": t_value,
                "noise_time": 1.0 - t_value,
                "latent_shape": list(x.shape[1:]),
                "seq_len": seq_len,
                "forward_seconds": elapsed,
            }
            print(tag, call_meta[tag], flush=True)

    layouts = {}
    tile_by_tag = {
        "coarse_end": (1, 1, 1),
        "middle_start": (2, 2, 2),
        "middle_end": (2, 2, 2),
        "fine_start": (4, 4, 4),
        "fine_next": (4, 4, 4),
    }
    for tag in tile_by_tag:
        grid = traces[tag][layer_ids[0]]["grid"]
        layouts[tag] = make_root_layout(grid, tile_by_tag[tag], device)
    root_grids = {tag: layout.root_grid for tag, layout in layouts.items()}
    if len(set(root_grids.values())) != 1:
        raise RuntimeError(f"stage layouts are not nested: {root_grids}")
    root_grid = next(iter(root_grids.values()))

    comparisons = {
        "coarse_to_middle": ("coarse_end", "middle_start"),
        "middle_to_fine": ("middle_end", "fine_start"),
        "fine_timestep_reuse": ("fine_start", "fine_next"),
    }
    report = {
        "scope": "official Wan init" if not args.checkpoint else "RDD-Wan checkpoint",
        "checkpoint": args.checkpoint,
        "pretrained_dir": str(Path(args.pretrained_dir).resolve()),
        "latent_source": args.latent_pt or "deterministic structured synthetic latent",
        "context_source": args.context_pt or "zero text context",
        "stage_lengths": list(lengths),
        "boundaries": boundaries,
        "root_grid": root_grid,
        "calls": call_meta,
        "layers": {},
    }
    for layer_id in layer_ids:
        layer_report = {}
        exact = {}
        pooled = {}
        content_pooled = {}
        for tag in tile_by_tag:
            trace = traces[tag][layer_id]
            q = trace["q"].to(device)
            k = trace["k"].to(device)
            exact[tag] = exact_root_attention_mass(q, k, layouts[tag])
            pooled[tag] = pooled_route_distribution(q, k, layouts[tag])
            content_pooled[tag] = pooled_route_distribution(
                trace["q_pre_rope"].to(device),
                trace["k_pre_rope"].to(device),
                layouts[tag],
            )
        for comparison, (source_tag, target_tag) in comparisons.items():
            stats = compare_route_distributions(
                exact[source_tag],
                exact[target_tag],
                densities=densities,
                root_grid=root_grid,
                local_radius=1,
            )
            stats["pooled_router_vs_target"] = compare_route_distributions(
                pooled[source_tag],
                exact[target_tag],
                densities=densities,
                root_grid=root_grid,
                local_radius=1,
            )
            stats["content_only_router_vs_target"] = compare_route_distributions(
                content_pooled[source_tag],
                exact[target_tag],
                densities=densities,
                root_grid=root_grid,
                local_radius=1,
            )
            if comparison == "middle_to_fine":
                fine_trace = traces[target_tag][layer_id]
                stats["output_error_sweep"] = {}
                for output_density in (0.20, 0.30, 0.40):
                    key = f"direct_{int(output_density * 100)}pct"
                    stats["output_error_sweep"][key] = masked_attention_output_error(
                        fine_trace["q"].to(device),
                        fine_trace["k"].to(device),
                        fine_trace["v"].to(device),
                        layouts[target_tag],
                        exact[source_tag],
                        density=output_density,
                        local_radius=None,
                    )
                stats["output_error_sweep"]["source20pct_plus_local_r1"] = (
                    masked_attention_output_error(
                        fine_trace["q"].to(device),
                        fine_trace["k"].to(device),
                        fine_trace["v"].to(device),
                        layouts[target_tag],
                        exact[source_tag],
                        density=0.20,
                        local_radius=1,
                    )
                )
                middle_trace = traces[source_tag][layer_id]
                stats["summary_tail_at_20pct"] = {
                    "previous_middle_parent": routed_attention_with_summary_tail_error(
                        fine_trace["q"].to(device),
                        fine_trace["k"].to(device),
                        fine_trace["v"].to(device),
                        layouts[target_tag],
                        exact[source_tag],
                        density=0.20,
                        summary_k=middle_trace["k"].to(device),
                        summary_v=middle_trace["v"].to(device),
                        summary_layout=layouts[source_tag],
                    ),
                    "pooled_fine_upper_bound": routed_attention_with_summary_tail_error(
                        fine_trace["q"].to(device),
                        fine_trace["k"].to(device),
                        fine_trace["v"].to(device),
                        layouts[target_tag],
                        exact[source_tag],
                        density=0.20,
                        summary_k=fine_trace["k"].to(device),
                        summary_v=fine_trace["v"].to(device),
                        summary_layout=layouts[target_tag],
                    ),
                }
            layer_report[comparison] = stats
        report["layers"][str(layer_id)] = layer_report
        if layer_id in (layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]):
            save_heatmap(
                exact["middle_end"], output / f"layer{layer_id:02d}_middle_route.png", f"Layer {layer_id}: middle root attention"
            )
            save_heatmap(
                exact["fine_start"], output / f"layer{layer_id:02d}_fine_route.png", f"Layer {layer_id}: fine root attention"
            )

    # Compact aggregate used as a go/no-go gate for the proposed method.
    aggregate = {}
    for name in comparisons:
        per_layer = [report["layers"][str(i)][name] for i in layer_ids]
        aggregate[name] = {
            "cosine_mean": sum(x["cosine_mean"] for x in per_layer) / len(per_layer),
            "js_mean": sum(x["js_mean"] for x in per_layer) / len(per_layer),
            "coverage_at_20pct": sum(
                next(row for row in x["density_sweep"] if abs(row["requested_density"] - 0.20) < 1e-8)[
                    "target_mass_coverage"
                ]
                for x in per_layer
            )
            / len(per_layer),
        }
    report["aggregate"] = aggregate
    report["decision_rule"] = {
        "go": "middle_to_fine 20% route coverage >= 0.80 and output cosine >= 0.95 on most probed layers",
        "conditional": "coverage 0.65-0.80: use middle route only as candidate set, then run a cheap fine residual router",
        "stop": "coverage < 0.65 or strong head/layer instability: do not claim direct cross-stage routing",
    }
    with (output / "routing_probe.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    if args.save_traces:
        torch.save(traces, output / "qkv_traces.pt")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
