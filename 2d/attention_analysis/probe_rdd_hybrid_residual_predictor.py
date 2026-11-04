#!/usr/bin/env python3
"""Calibrate and validate a cheap coarse-low + fine-high RDD block predictor.

The predictor replaces an expensive fine DiT block update with:

    channel_affine(upsample(coarse block update))
      + channel_scale(highpass(current fine hidden state))

Coefficients are fitted from a small, disjoint ImageNet calibration split and
then frozen for independent images/noise seeds.  No checkpoint weights change.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    aligned_noise,
    benchmark_forward,
    discover_train_images,
    encode_images,
    load_rdd,
    low_high_tokens,
    manual_forward,
    pred_x0,
    stage_drift,
    tensor_metrics,
    upsample_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion_with_drift  # noqa: E402


def make_diffusions() -> tuple[Any, Any]:
    fine = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=0, T_c=0, T_c_plus_1=300
    )
    middle = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=1, T_c=300, T_c_plus_1=600
    )
    return fine, middle


@torch.inference_mode()
def make_pair(clean32: torch.Tensor, seed: int, fine: Any, middle: Any) -> dict[str, torch.Tensor]:
    batch = clean32.shape[0]
    clean16 = torch.nn.functional.avg_pool2d(clean32, 2, 2)
    eps32, eps16 = aligned_noise(clean32, seed)
    source_local = torch.zeros(batch, device=clean32.device, dtype=torch.long)
    target_local = torch.full((batch,), 299, device=clean32.device, dtype=torch.long)
    source = middle.q_sample(
        clean16, source_local, d=stage_drift(middle, clean16), noise=eps16
    )
    target = fine.q_sample(
        clean32, target_local, d=stage_drift(fine, clean32), noise=eps32
    )
    source_global = torch.full(
        (batch,), int(middle.timestep_map[0]), device=clean32.device, dtype=torch.long
    )
    target_global = torch.full(
        (batch,), int(fine.timestep_map[299]), device=clean32.device, dtype=torch.long
    )
    return {
        "source": source,
        "target": target,
        "source_global": source_global,
        "target_global": target_global,
        "target_local": target_local,
    }


def init_stats(depth: int, channels: int, device: torch.device) -> list[dict[str, torch.Tensor | float]]:
    zeros = lambda: torch.zeros(channels, device=device, dtype=torch.float64)
    return [
        {
            "n": 0.0,
            "low_sx": zeros(), "low_sy": zeros(), "low_sxx": zeros(), "low_sxy": zeros(),
            "high_sxx": zeros(), "high_sxy": zeros(),
        }
        for _ in range(depth)
    ]


def accumulate(stats: dict[str, torch.Tensor | float], prefix: str, x: torch.Tensor, y: torch.Tensor) -> None:
    x = x.detach().double().reshape(-1, x.shape[-1])
    y = y.detach().double().reshape(-1, y.shape[-1])
    if prefix == "low":
        stats["n"] = float(stats["n"]) + x.shape[0]
        stats["low_sx"] += x.sum(0)
        stats["low_sy"] += y.sum(0)
        stats["low_sxx"] += (x * x).sum(0)
        stats["low_sxy"] += (x * y).sum(0)
    else:
        stats["high_sxx"] += (x * x).sum(0)
        stats["high_sxy"] += (x * y).sum(0)


def finish_coefficients(stats: list[dict[str, torch.Tensor | float]]) -> list[dict[str, torch.Tensor]]:
    coefficients = []
    for row in stats:
        n = float(row["n"])
        mean_x = row["low_sx"] / n
        mean_y = row["low_sy"] / n
        variance_x = row["low_sxx"] / n - mean_x.square()
        covariance = row["low_sxy"] / n - mean_x * mean_y
        low_scale = (covariance / variance_x.clamp_min(1e-10)).clamp(-4.0, 4.0).float()
        low_bias = (mean_y - low_scale.double() * mean_x).float()
        high_scale = (row["high_sxy"] / row["high_sxx"].clamp_min(1e-10)).clamp(-4.0, 4.0).float()
        coefficients.append({
            "low_scale": low_scale,
            "low_bias": low_bias,
            "high_scale": high_scale,
        })
    return coefficients


@torch.inference_mode()
def calibrate(
    model: torch.nn.Module,
    fine: Any,
    middle: Any,
    imagenet_train: Path,
    vae_path: Path,
    device: torch.device,
    seeds: list[int],
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    depth = len(model.blocks)
    stats = init_stats(depth, model.x_embedder.proj.out_channels, device)
    dtype = model.x_embedder.proj.weight.dtype
    for index, seed in enumerate(seeds):
        paths, labels = discover_train_images(imagenet_train, batch_size, index * 211)
        clean32 = encode_images(paths, vae_path, device)
        pair = make_pair(clean32, seed, fine, middle)
        labels = labels.to(device)
        _, source_trace = manual_forward(
            model, pair["source"].to(dtype), pair["source_global"], labels,
            multires=True, capture=True,
        )
        _, target_trace = manual_forward(
            model, pair["target"].to(dtype), pair["target_global"], labels,
            multires=True, capture=True,
        )
        for layer in range(depth):
            coarse = upsample_tokens(source_trace[layer]["residual"], 16, "nearest")
            target_low, target_high = low_high_tokens(target_trace[layer]["residual"])
            _, input_high = low_high_tokens(target_trace[layer]["input"])
            accumulate(stats[layer], "low", coarse, target_low)
            accumulate(stats[layer], "high", input_high, target_high)
        del clean32, pair, source_trace, target_trace
        gc.collect()
        torch.cuda.empty_cache()
    return finish_coefficients(stats)


def low_prediction(coarse: torch.Tensor, coeff: dict[str, torch.Tensor]) -> torch.Tensor:
    return coarse * coeff["low_scale"].view(1, 1, -1) + coeff["low_bias"].view(1, 1, -1)


def hybrid_callable(coarse_low: torch.Tensor, coeff: dict[str, torch.Tensor]):
    def predict(current_hidden: torch.Tensor) -> torch.Tensor:
        _, high = low_high_tokens(current_hidden)
        return coarse_low + high * coeff["high_scale"].view(1, 1, -1)
    return predict


@torch.inference_mode()
def validate_one(
    model: torch.nn.Module,
    fine: Any,
    middle: Any,
    coefficients: list[dict[str, torch.Tensor]],
    clean32: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    warmup: int,
    rounds: int,
) -> dict[str, Any]:
    pair = make_pair(clean32, seed, fine, middle)
    dtype = model.x_embedder.proj.weight.dtype
    _, source_trace = manual_forward(
        model, pair["source"].to(dtype), pair["source_global"], labels,
        multires=True, capture=True,
    )
    exact_output, target_trace = manual_forward(
        model, pair["target"].to(dtype), pair["target_global"], labels,
        multires=True, capture=True,
    )
    exact_x0 = pred_x0(fine, exact_output, pair["target"], pair["target_local"])
    selected_layers = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 22]
    single = []
    coarse_by_layer = {
        layer: upsample_tokens(source_trace[layer]["residual"], 16, "nearest")
        for layer in range(len(model.blocks))
    }
    for layer in selected_layers:
        low = low_prediction(coarse_by_layer[layer], coefficients[layer])
        replacement = hybrid_callable(low, coefficients[layer])
        replaced_output, _ = manual_forward(
            model, pair["target"].to(dtype), pair["target_global"], labels,
            multires=True, replacements={layer: replacement}, replacement_mode="residual",
        )
        target_low, target_high = low_high_tokens(target_trace[layer]["residual"])
        _, input_high = low_high_tokens(target_trace[layer]["input"])
        predicted_residual = low + input_high * coefficients[layer]["high_scale"].view(1, 1, -1)
        single.append({
            "layer": layer,
            "low_predictor": tensor_metrics(low, target_low),
            "high_predictor": tensor_metrics(predicted_residual - low, target_high),
            "full_residual_predictor": tensor_metrics(predicted_residual, target_trace[layer]["residual"]),
            "pred_x0": tensor_metrics(
                pred_x0(fine, replaced_output, pair["target"], pair["target_local"]), exact_x0
            ),
        })

    layer_sets = {
        "peak2": [15, 16],
        "top4": [8, 13, 15, 16],
        "spaced4": [8, 11, 13, 16],
        "safe6": [7, 9, 11, 13, 15, 16],
    }
    combinations = {}
    methods: dict[str, dict[str, Any]] = {}
    for setting, layers in layer_sets.items():
        direct = {layer: coarse_by_layer[layer] for layer in layers}
        low_only = {
            layer: low_prediction(coarse_by_layer[layer], coefficients[layer])
            for layer in layers
        }
        hybrid = {
            layer: hybrid_callable(low_only[layer], coefficients[layer])
            for layer in layers
        }
        for method, replacements in (("direct", direct), ("low_affine", low_only), ("hybrid", hybrid)):
            output, _ = manual_forward(
                model, pair["target"].to(dtype), pair["target_global"], labels,
                multires=True, replacements=replacements, replacement_mode="residual",
            )
            combinations[f"{setting}_{method}"] = {
                "layers": layers,
                "pred_x0": tensor_metrics(
                    pred_x0(fine, output, pair["target"], pair["target_local"]), exact_x0
                ),
                "model_output": tensor_metrics(output[:, :4], exact_output[:, :4]),
            }
            methods[f"{setting}_{method}"] = replacements

    baseline_fn = lambda: manual_forward(
        model, pair["target"].to(dtype), pair["target_global"], labels, multires=True
    )
    latency = {"baseline": benchmark_forward(baseline_fn, pair["target"].device, warmup, rounds)}
    baseline_ms = latency["baseline"]["median_ms"]
    for key in (
        "top4_direct",
        "top4_low_affine",
        "top4_hybrid",
        "spaced4_low_affine",
        "spaced4_hybrid",
        "safe6_low_affine",
        "safe6_hybrid",
    ):
        function = lambda replacements=methods[key]: manual_forward(
            model, pair["target"].to(dtype), pair["target_global"], labels,
            multires=True, replacements=replacements, replacement_mode="residual",
        )
        timing = benchmark_forward(function, pair["target"].device, warmup, rounds)
        timing["speedup_vs_baseline"] = baseline_ms / timing["median_ms"]
        timing["latency_reduction_percent"] = 100.0 * (1.0 - timing["median_ms"] / baseline_ms)
        latency[key] = timing
    return {"seed": seed, "single_layer": single, "combinations": combinations, "latency": latency}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-seeds", default="101,202")
    parser.add_argument("--validation-seeds", default="1301,2302,3303")
    parser.add_argument("--calibration-batch", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-rounds", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_rdd(Path(args.rdd_ckpt), device, torch.bfloat16)
    fine, middle = make_diffusions()
    calibration_seeds = [int(value) for value in args.calibration_seeds.split(",")]
    validation_seeds = [int(value) for value in args.validation_seeds.split(",")]
    coefficients = calibrate(
        model, fine, middle, Path(args.imagenet_train), Path(args.vae_path), device,
        calibration_seeds, args.calibration_batch,
    )
    validations = []
    for index, seed in enumerate(validation_seeds):
        paths, labels = discover_train_images(
            Path(args.imagenet_train), args.validation_batch, 701 + index * 317
        )
        clean32 = encode_images(paths, Path(args.vae_path), device)
        validations.append(validate_one(
            model, fine, middle, coefficients, clean32, labels.to(device), seed,
            args.warmup, args.timing_rounds,
        ))
        del clean32
        gc.collect()
        torch.cuda.empty_cache()

    coefficient_summary = []
    for layer, coeff in enumerate(coefficients):
        coefficient_summary.append({
            "layer": layer,
            "low_scale_mean": float(coeff["low_scale"].mean()),
            "low_scale_std": float(coeff["low_scale"].std()),
            "high_scale_mean": float(coeff["high_scale"].mean()),
            "high_scale_std": float(coeff["high_scale"].std()),
        })
    report = {
        "method": "frozen channel-affine coarse-low + current-hidden high-pass predictor",
        "calibration_seeds": calibration_seeds,
        "validation_seeds": validation_seeds,
        "calibration_images": len(calibration_seeds) * args.calibration_batch,
        "validation_images": len(validation_seeds) * args.validation_batch,
        "coefficients": coefficient_summary,
        "validations": validations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
