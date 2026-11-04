"""Small, reproducible diagnostics for the CSBR residual hypotheses.

This script intentionally does not train or alter a checkpoint.  It runs the
released FastWan model on a tiny paired protocol and measures:

1. cross-scale residual alignment in stage-aligned low/high subspaces;
2. output perturbation from RDD-CSBR vs same-resolution timestep caching;
3. residual stability across stages/timesteps;
4. per-layer error, frequency-band error and a scalar-adapter gradient proxy.

The released checkpoint is not RDD/CSBR trained, so the results are a
mechanistic pilot rather than a final quality claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sample_fastwan_native_vsa import (
    add_fastvideo_paths,
    parse_pairs,
    sample_three_call_dmd,
)


CSBR6 = [0, 5, 10, 15, 20, 25]
CSBR12 = [0, 3, 5, 8, 10, 13, 16, 18, 21, 23, 26, 29]


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0e-12)


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return safe_ratio(torch.dot(left, right), left.norm() * right.norm())


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def pair_metrics(student: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    student_f = student.float()
    target_f = target.float()
    difference = student_f - target_f
    target_rms = rms(target_f)
    student_power = student_f.square().mean()
    alpha = safe_ratio((student_f * target_f).mean(), student_power)
    fitted = alpha * student_f
    # This is the per-example gradient of MSE(s * student, target) at s=1.
    # It is only a scalar-adapter training proxy, not a full-model gradient.
    gradient_at_one = 2.0 * (difference * student_f).mean()
    return {
        "cosine": cosine(student_f, target_f),
        "relative_rmse": safe_ratio(rms(difference), target_rms),
        "optimal_scalar": alpha,
        "optimal_relative_rmse": safe_ratio(rms(fitted - target_f), target_rms),
        "scalar_gradient_at_one": gradient_at_one,
        "student_rms": rms(student_f),
        "target_rms": target_rms,
    }


def token_grid(model: Any, latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    patch = tuple(int(value) for value in model.patch_size)
    if any(size % stride for size, stride in zip(latent_shape, patch)):
        raise ValueError(f"latent shape {latent_shape} is not divisible by {patch}")
    return tuple(size // stride for size, stride in zip(latent_shape, patch))


def sequence_to_volume(
    sequence: torch.Tensor, grid: tuple[int, int, int]
) -> torch.Tensor:
    batch, tokens, channels = sequence.shape
    if tokens != math.prod(grid):
        raise ValueError(f"{tokens} tokens do not match grid {grid}")
    return sequence.view(batch, *grid, channels).permute(0, 4, 1, 2, 3)


def volume_to_sequence(volume: torch.Tensor) -> torch.Tensor:
    return volume.permute(0, 2, 3, 4, 1).reshape(volume.shape[0], -1, volume.shape[1])


def lift_sequence(
    sequence: torch.Tensor,
    source_grid: tuple[int, int, int],
    target_grid: tuple[int, int, int],
) -> torch.Tensor:
    if source_grid == target_grid:
        return sequence
    volume = sequence_to_volume(sequence, source_grid)
    lifted = F.interpolate(volume, size=target_grid, mode="nearest")
    return volume_to_sequence(lifted).contiguous()


def orthogonal_low_projection(
    sequence: torch.Tensor,
    fine_grid: tuple[int, int, int],
    coarse_grid: tuple[int, int, int],
) -> torch.Tensor:
    """Project onto tensors constant inside each nested coarse cell.

    Average pooling followed by nearest lifting is the Euclidean orthogonal
    projection when every grid ratio is an integer.  Therefore the high-band
    complement is exactly orthogonal up to floating-point roundoff.
    """

    if fine_grid == coarse_grid:
        return sequence
    if any(fine % coarse for fine, coarse in zip(fine_grid, coarse_grid)):
        raise ValueError(f"non-nested token grids: {coarse_grid} -> {fine_grid}")
    factors = tuple(fine // coarse for fine, coarse in zip(fine_grid, coarse_grid))
    volume = sequence_to_volume(sequence, fine_grid).float()
    pooled = F.avg_pool3d(volume, kernel_size=factors, stride=factors)
    lifted = F.interpolate(pooled, size=fine_grid, mode="nearest")
    return volume_to_sequence(lifted)


def half_resolution_projection(
    sequence: torch.Tensor, grid: tuple[int, int, int]
) -> torch.Tensor:
    coarse = (grid[0], max(1, grid[1] // 2), max(1, grid[2] // 2))
    if grid[1] % coarse[1] or grid[2] % coarse[2]:
        return sequence.float()
    return orthogonal_low_projection(sequence, grid, coarse)


def materialize(value: Any) -> Any:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [materialize(item) for item in value]
    return value


class ResidualRecorder:
    """Observe normal block calls without changing their outputs."""

    def __init__(self, controller: Any, run_name: str) -> None:
        self.controller = controller
        self.run_name = run_name
        self.cache: dict[
            tuple[int, int], tuple[tuple[int, int, int], torch.Tensor]
        ] = {}
        self.rows: list[dict[str, Any]] = []

    def observe(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        stage = int(self.controller.current_stage)
        shape = tuple(int(value) for value in self.controller.current_shape)
        source = hidden_states.squeeze(1) if hidden_states.dim() == 4 else hidden_states
        target = output.squeeze(1) if output.dim() == 4 else output
        residual = (target - source).detach()
        if stage > 0:
            previous_shape, previous = self.cache[(stage - 1, layer)]
            source_grid = token_grid(self.controller.model, previous_shape)
            target_grid = token_grid(self.controller.model, shape)
            lifted = lift_sequence(previous, source_grid, target_grid).float()
            current = residual.float()
            current_low = orthogonal_low_projection(current, target_grid, source_grid)
            current_high = current - current_low

            wrong_layer = (layer + 7) % len(self.controller.model.blocks)
            wrong_shape, wrong_previous = self.cache[(stage - 1, wrong_layer)]
            if wrong_shape != previous_shape:
                raise RuntimeError("wrong-layer cache has inconsistent shape")
            wrong_lifted = lift_sequence(
                wrong_previous, source_grid, target_grid
            ).float()

            lifted_low = orthogonal_low_projection(lifted, target_grid, source_grid)
            lifted_high = lifted - lifted_low
            error = lifted - current
            error_low = orthogonal_low_projection(error, target_grid, source_grid)
            error_high = error - error_low
            current_energy = current.square().mean()

            # A separate 2x split makes same-resolution timestep caching
            # comparable in low and high spatial bands.
            previous_half_low = half_resolution_projection(lifted, target_grid)
            current_half_low = half_resolution_projection(current, target_grid)
            previous_half_high = lifted - previous_half_low
            current_half_high = current - current_half_low

            multiscale_predictor = None
            if stage == 2 and (0, layer) in self.cache:
                coarse_shape, coarse_residual = self.cache[(0, layer)]
                coarse_grid = token_grid(self.controller.model, coarse_shape)
                coarse_lifted = lift_sequence(
                    coarse_residual, coarse_grid, target_grid
                ).float()
                trend = lifted - coarse_lifted
                hidden = source.detach().float()
                hidden_low = orthogonal_low_projection(
                    hidden, target_grid, source_grid
                )
                hidden_high = hidden - hidden_low

                # Sufficient statistics allow leakage-free fitting on some
                # cases and evaluation on held-out cases without saving huge
                # hidden-state tensors.  The low predictor is a*M+b*(M-C).
                # The orthogonal high predictor is c*highpass(h_fine).
                low_target = current_low
                high_target = current_high
                multiscale_predictor = {
                    "low_sufficient_statistics": {
                        "mm": (lifted * lifted).mean(),
                        "dd": (trend * trend).mean(),
                        "md": (lifted * trend).mean(),
                        "my": (lifted * low_target).mean(),
                        "dy": (trend * low_target).mean(),
                        "yy": (low_target * low_target).mean(),
                    },
                    "high_sufficient_statistics": {
                        "hh": (hidden_high * hidden_high).mean(),
                        "hy": (hidden_high * high_target).mean(),
                        "yy": (high_target * high_target).mean(),
                    },
                    "coarse_to_fine_low": pair_metrics(
                        coarse_lifted, low_target
                    ),
                    "trend_to_fine_low": pair_metrics(trend, low_target),
                    "hidden_high_to_fine_high": pair_metrics(
                        hidden_high, high_target
                    ),
                    "gamma_sweep": {
                        str(gamma): pair_metrics(
                            lifted + gamma * trend, low_target
                        )
                        for gamma in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
                    },
                }

            self.rows.append(
                {
                    "run": self.run_name,
                    "boundary": f"stage{stage - 1}_to_stage{stage}",
                    "source_stage": stage - 1,
                    "target_stage": stage,
                    "layer": layer,
                    "source_shape": previous_shape,
                    "target_shape": shape,
                    "source_grid": source_grid,
                    "target_grid": target_grid,
                    "matched_to_raw": pair_metrics(lifted, current),
                    "matched_to_stage_low": pair_metrics(lifted, current_low),
                    "matched_to_stage_high_cosine": cosine(lifted, current_high),
                    "wrong_layer_to_stage_low_cosine": cosine(
                        wrong_lifted, current_low
                    ),
                    "matched_minus_wrong_low_cosine": (
                        cosine(lifted, current_low)
                        - cosine(wrong_lifted, current_low)
                    ),
                    "stage_low_energy_fraction": safe_ratio(
                        current_low.square().mean(), current_energy
                    ),
                    "stage_high_energy_fraction": safe_ratio(
                        current_high.square().mean(), current_energy
                    ),
                    "lifted_high_energy_fraction": safe_ratio(
                        lifted_high.square().mean(), lifted.square().mean()
                    ),
                    "error_low_rms": rms(error_low),
                    "error_high_rms": rms(error_high),
                    "error_high_to_low_rms_ratio": safe_ratio(
                        rms(error_high), rms(error_low)
                    ),
                    "same_grid_half_low_cosine": cosine(
                        previous_half_low, current_half_low
                    ),
                    "same_grid_half_high_cosine": cosine(
                        previous_half_high, current_half_high
                    ),
                    "multiscale_predictor": multiscale_predictor,
                }
            )

        if stage < 2:
            # clone() avoids depending on buffers that FastVideo may reuse.
            self.cache[(stage, layer)] = (shape, residual.clone())

    def final_rows(self) -> list[dict[str, Any]]:
        result = materialize(self.rows)
        self.cache.clear()
        self.rows.clear()
        return result


def latent_metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    baseline = baseline.float()
    difference = candidate - baseline
    return materialize(
        {
            "cosine": cosine(candidate, baseline),
            "relative_rmse": safe_ratio(rms(difference), rms(baseline)),
            "mean_abs": difference.abs().mean(),
            "max_abs": difference.abs().max(),
        }
    )


def video_metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    baseline = baseline.float()
    difference = candidate - baseline
    mse = difference.square().mean()
    return materialize(
        {
            "cosine": cosine(candidate, baseline),
            "relative_rmse": safe_ratio(rms(difference), rms(baseline)),
            "psnr_assuming_range_2": 10.0 * torch.log10(4.0 / mse.clamp_min(1.0e-12)),
        }
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact per-run/per-boundary summaries for quick inspection."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["run"], row["boundary"]), []).append(row)
    output: dict[str, Any] = {}
    paths = {
        "raw_cosine": ("matched_to_raw", "cosine"),
        "raw_relative_rmse": ("matched_to_raw", "relative_rmse"),
        "raw_optimal_relative_rmse": ("matched_to_raw", "optimal_relative_rmse"),
        "low_cosine": ("matched_to_stage_low", "cosine"),
        "low_relative_rmse": ("matched_to_stage_low", "relative_rmse"),
        "low_optimal_relative_rmse": (
            "matched_to_stage_low",
            "optimal_relative_rmse",
        ),
        "low_optimal_scalar": ("matched_to_stage_low", "optimal_scalar"),
        "scalar_gradient_at_one": (
            "matched_to_stage_low",
            "scalar_gradient_at_one",
        ),
        "high_cosine_tautological": ("matched_to_stage_high_cosine",),
        "wrong_layer_low_cosine": ("wrong_layer_to_stage_low_cosine",),
        "matched_minus_wrong": ("matched_minus_wrong_low_cosine",),
        "low_energy_fraction": ("stage_low_energy_fraction",),
        "high_energy_fraction": ("stage_high_energy_fraction",),
        "error_high_to_low_ratio": ("error_high_to_low_rms_ratio",),
        "half_low_cosine": ("same_grid_half_low_cosine",),
        "half_high_cosine": ("same_grid_half_high_cosine",),
    }

    def read_path(row: dict[str, Any], path: tuple[str, ...]) -> float:
        value: Any = row
        for key in path:
            value = value[key]
        return float(value)

    for (run, boundary), group in groups.items():
        key = f"{run}/{boundary}"
        output[key] = {"count": len(group)}
        for name, path in paths.items():
            values = [read_path(row, path) for row in group]
            output[key][name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return output


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fastvideo_root = (root / args.fastvideo_root).resolve()
    kernel_root = (root / args.fastvideo_kernel_root).resolve()
    add_fastvideo_paths(fastvideo_root, kernel_root)

    from attention_analysis.fastwan_native_attention import (
        FastWanNativeAttentionController,
        install_projected_cache_class_patch,
    )
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from fastvideo.models.loader.component_loader import TransformerLoader
    from sample_rdd_wan import save_video
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    width, height = (int(value) for value in args.size.split("x"))
    latent_t = (args.frame_num - 1) // 4 + 1
    full_shape = (1, 16, latent_t, height // 8, width // 8)
    timesteps = [int(value) for value in args.dmd_timesteps.split(",")]
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    original = (root / args.original_wan_dir).resolve()
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.float32,
        device=torch.device("cpu"),
        checkpoint_path=str(original / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(original / "google/umt5-xxl"),
    )
    text_encoder.model.to(device)
    context_unpadded = text_encoder([args.prompt], device)[0]
    context = torch.zeros(
        (1, 512, context_unpadded.shape[-1]), device=device, dtype=torch.bfloat16
    )
    context[0, : context_unpadded.shape[0]] = context_unpadded.to(torch.bfloat16)
    text_encoder.model.cpu()
    del text_encoder
    torch.cuda.empty_cache()

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    transformer_path = str((root / args.fastwan_dir / "transformer").resolve())
    fastvideo_args = FastVideoArgs(
        model_path=transformer_path,
        num_gpus=1,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        pipeline_config=FastWan2_1_T2V_480P_Config(
            dit_config=WanVideoConfig(), dit_precision="bf16"
        ),
    )
    fastvideo_args.device = device
    print(f"Loading FastWan transformer from {transformer_path}", flush=True)
    model = TransformerLoader().load(transformer_path, fastvideo_args).to(
        dtype=torch.bfloat16
    ).eval()
    controller = FastWanNativeAttentionController(model)
    original_dispatch = controller._dispatch_block
    active_recorder: ResidualRecorder | None = None

    def observed_dispatch(this, layer, block, hidden_states, *call_args, **call_kwargs):
        output = original_dispatch(layer, block, hidden_states, *call_args, **call_kwargs)
        if active_recorder is not None:
            active_recorder.observe(layer, hidden_states, output)
        return output

    controller._dispatch_block = types.MethodType(observed_dispatch, controller)

    configs = [
        {
            "name": "rdd_nf_base",
            "factors": "4,1;2,1;1,1",
            "transition": "noise_filling",
            "capture": True,
        },
        {
            "name": "rdd_ip_base",
            "factors": "4,1;2,1;1,1",
            "transition": "innovation_preserving",
            "capture": True,
        },
        {
            "name": "full_ip_base",
            "factors": "1,1;1,1;1,1",
            "transition": "innovation_preserving",
            "capture": True,
        },
        {
            "name": "full_nf_base",
            "factors": "1,1;1,1;1,1",
            "transition": "noise_filling",
            "capture": True,
        },
        {
            "name": "rdd_nf_csbr6_s075",
            "factors": "4,1;2,1;1,1",
            "transition": "noise_filling",
            "layers": CSBR6,
            "scale": 0.75,
        },
        {
            "name": "rdd_nf_csbr12_s100",
            "factors": "4,1;2,1;1,1",
            "transition": "noise_filling",
            "layers": CSBR12,
            "scale": 1.0,
        },
        {
            "name": "rdd_ip_csbr6_s075",
            "factors": "4,1;2,1;1,1",
            "transition": "innovation_preserving",
            "layers": CSBR6,
            "scale": 0.75,
        },
        {
            "name": "rdd_ip_csbr12_s100",
            "factors": "4,1;2,1;1,1",
            "transition": "innovation_preserving",
            "layers": CSBR12,
            "scale": 1.0,
        },
        {
            "name": "full_ip_cache6_s075",
            "factors": "1,1;1,1;1,1",
            "transition": "innovation_preserving",
            "layers": CSBR6,
            "scale": 0.75,
        },
        {
            "name": "full_ip_cache12_s100",
            "factors": "1,1;1,1;1,1",
            "transition": "innovation_preserving",
            "layers": CSBR12,
            "scale": 1.0,
        },
        {
            "name": "full_nf_cache6_s075",
            "factors": "1,1;1,1;1,1",
            "transition": "noise_filling",
            "layers": CSBR6,
            "scale": 0.75,
        },
    ]
    if args.probe_only:
        keep = {"rdd_nf_base", "rdd_ip_base", "full_ip_base", "full_nf_base"}
        configs = [config for config in configs if config["name"] in keep]

    def configure(config: dict[str, Any]) -> None:
        controller.configure(
            attention_mode="vsa",
            route_density=0.2,
            reuse_every=1,
            reuse_scope="backend",
            cross_stage_reuse_layers=config.get("layers", []),
            cross_stage_reuse_stages=[2] if config.get("layers") else [],
            cross_stage_reuse_scale=float(config.get("scale", 1.0)),
            cross_stage_reuse_resize="nearest",
        )

    def sample(config: dict[str, Any], capture: bool) -> tuple[Any, list[dict[str, Any]]]:
        nonlocal active_recorder
        configure(config)
        recorder = ResidualRecorder(controller, config["name"]) if capture else None
        active_recorder = recorder
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent, shapes, stage_seconds, denoise_seconds = sample_three_call_dmd(
                model=model,
                context=context,
                full_shape=full_shape,
                stage_factors=parse_pairs(config["factors"]),
                timesteps=timesteps,
                seed=seed,
                vsa_sparsity=0.8,
                transition_mode=config["transition"],
                device=device,
                attention_controller=controller,
                boundary_executor="torch_nearest",
            )
        active_recorder = None
        rows = recorder.final_rows() if recorder is not None else []
        elapsed = time.perf_counter() - start
        record = {
            "config": config,
            "latent": latent.detach().cpu(),
            "stage_shapes": shapes,
            "stage_seconds": stage_seconds,
            "denoise_seconds": denoise_seconds,
            "wall_seconds": elapsed,
            "attention": controller.summary(),
        }
        del recorder
        torch.cuda.empty_cache()
        return record, rows

    # Compile/warm the two geometries without retaining diagnostics.
    for warm_name in ("rdd_nf_base", "full_ip_base"):
        warm_config = next(item for item in configs if item["name"] == warm_name)
        print(f"warmup={warm_name}", flush=True)
        sample(warm_config, capture=False)

    results: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    latents: dict[str, torch.Tensor] = {}
    for config in configs:
        print(f"run={config['name']}", flush=True)
        record, rows = sample(config, capture=bool(config.get("capture", False)))
        latents[config["name"]] = record.pop("latent")
        results[config["name"]] = materialize(record)
        all_rows.extend(rows)

    comparisons = {}
    comparison_specs = {
        "rdd_nf_csbr6_vs_rdd_nf_base": (
            "rdd_nf_csbr6_s075",
            "rdd_nf_base",
        ),
        "rdd_nf_csbr12_vs_rdd_nf_base": (
            "rdd_nf_csbr12_s100",
            "rdd_nf_base",
        ),
        "rdd_ip_csbr6_vs_rdd_ip_base": (
            "rdd_ip_csbr6_s075",
            "rdd_ip_base",
        ),
        "rdd_ip_csbr12_vs_rdd_ip_base": (
            "rdd_ip_csbr12_s100",
            "rdd_ip_base",
        ),
        "full_ip_cache6_vs_full_ip_base": (
            "full_ip_cache6_s075",
            "full_ip_base",
        ),
        "full_ip_cache12_vs_full_ip_base": (
            "full_ip_cache12_s100",
            "full_ip_base",
        ),
        "full_nf_cache6_vs_full_nf_base": (
            "full_nf_cache6_s075",
            "full_nf_base",
        ),
    }
    for comparison_name, (candidate_name, baseline_name) in comparison_specs.items():
        if candidate_name in latents and baseline_name in latents:
            comparisons[comparison_name] = latent_metrics(
                latents[candidate_name], latents[baseline_name]
            )

    videos: dict[str, Any] = {}
    if args.decode and not args.probe_only:
        decode_names = [
            "rdd_nf_base",
            "rdd_nf_csbr6_s075",
            "rdd_nf_csbr12_s100",
            "full_ip_base",
            "full_ip_cache6_s075",
            "full_ip_cache12_s100",
        ]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        vae = WanVAE(
            z_dim=16,
            vae_pth=str(original / "Wan2.1_VAE.pth"),
            dtype=torch.float32,
            device=device,
        )
        decoded: dict[str, torch.Tensor] = {}
        # Exact-shape warmup is excluded from per-video decode timing.
        vae.decode([latents[decode_names[0]][0].float().to(device)])
        torch.cuda.synchronize(device)
        for name in decode_names:
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            video = vae.decode([latents[name][0].float().to(device)])[0]
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - begin
            path = output_dir / f"{name}.mp4"
            save_video(video, str(path), fps=args.fps)
            decoded[name] = video.detach().cpu()
            videos[name] = {"path": str(path), "vae_seconds": seconds}
        videos["rdd_nf_csbr6_s075"]["vs_baseline"] = video_metrics(
            decoded["rdd_nf_csbr6_s075"], decoded["rdd_nf_base"]
        )
        videos["rdd_nf_csbr12_s100"]["vs_baseline"] = video_metrics(
            decoded["rdd_nf_csbr12_s100"], decoded["rdd_nf_base"]
        )
        videos["full_ip_cache6_s075"]["vs_baseline"] = video_metrics(
            decoded["full_ip_cache6_s075"], decoded["full_ip_base"]
        )
        videos["full_ip_cache12_s100"]["vs_baseline"] = video_metrics(
            decoded["full_ip_cache12_s100"], decoded["full_ip_base"]
        )
    else:
        del model

    payload = {
        "protocol": {
            "checkpoint": "FastWan2.1-T2V-1.3B-Diffusers (not RDD/CSBR finetuned)",
            "size": args.size,
            "frame_num": args.frame_num,
            "dmd_timesteps": timesteps,
            "vsa_keep_fraction": 0.2,
            "rdd_factors": "(4,1)->(2,1)->(1,1)",
            "full_factors": "(1,1)->(1,1)->(1,1)",
            "note": (
                "noise_filling is the deployment RDD protocol; innovation_preserving "
                "is included as a common-innovation control for timestep stability"
            ),
            "gradient_warning": (
                "scalar_gradient_at_one is a frozen-residual one-parameter adapter "
                "proxy, not full-network gradient variance"
            ),
            "high_cosine_warning": (
                "stage-aligned P_high is orthogonal to lifted coarse tensors by "
                "construction; near-zero cosine alone is not evidence"
            ),
        },
        "prompt": args.prompt,
        "seed": seed,
        "results": results,
        "latent_comparisons": comparisons,
        "residual_rows": all_rows,
        "residual_summary": aggregate_rows(all_rows),
        "videos": videos,
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fastvideo-root", default="third_party/FastVideo-ref")
    parser.add_argument(
        "--fastvideo-kernel-root",
        default="third_party/FastVideo-ref/fastvideo-kernel",
    )
    parser.add_argument(
        "--fastwan-dir", default="pretrained_models/FastWan2.1-T2V-1.3B-Diffusers"
    )
    parser.add_argument(
        "--original-wan-dir", default="pretrained_models/Wan2.1-T2V-1.3B"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=29)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dmd-timesteps", default="1000,757,522")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=30341)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run only the four residual-capture baselines, without reuse candidates.",
    )
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
