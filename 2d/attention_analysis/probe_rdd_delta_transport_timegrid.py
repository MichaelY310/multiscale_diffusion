#!/usr/bin/env python3
"""Teacher-forced audit of RDD delta transport versus a generic time cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_hybrid_residual_predictor import make_diffusions  # noqa: E402
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    aligned_noise,
    discover_train_images,
    encode_images,
    load_rdd,
    low_high_tokens,
    manual_forward,
    stage_drift,
    tensor_metrics,
    upsample_tokens,
)
from sample_rdd_time_conditioned_reuse_preview import calibrate_time_grid  # noqa: E402


def interpolated_prediction(
    coefficients: dict[int, list[dict[str, torch.Tensor]]],
    coarse: torch.Tensor,
    layer: int,
    timestep: int,
) -> torch.Tensor:
    knots = sorted(coefficients)
    if timestep <= knots[0]:
        left = right = knots[0]
        weight = 0.0
    elif timestep >= knots[-1]:
        left = right = knots[-1]
        weight = 0.0
    else:
        right_index = next(index for index, knot in enumerate(knots) if knot > timestep)
        left, right = knots[right_index - 1], knots[right_index]
        weight = float(timestep - left) / float(right - left)
    left_row = coefficients[left][layer]
    right_row = coefficients[right][layer]
    scale = left_row["low_scale"].lerp(right_row["low_scale"], weight).to(coarse.dtype)
    bias = left_row["low_bias"].lerp(right_row["low_bias"], weight).to(coarse.dtype)
    return coarse * scale.view(1, 1, -1) + bias.view(1, 1, -1)


@torch.inference_mode()
def validation_seed(
    model: torch.nn.Module,
    coefficients: dict[int, list[dict[str, torch.Tensor]]],
    imagenet_train: Path,
    vae_path: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    image_offset: int,
    targets: list[int],
    gaps: list[int],
) -> dict:
    fine, middle = make_diffusions()
    paths, labels = discover_train_images(imagenet_train, batch_size, image_offset)
    labels = labels.to(device)
    clean32 = encode_images(paths, vae_path, device)
    clean16 = torch.nn.functional.avg_pool2d(clean32, 2, 2)
    eps32, eps16 = aligned_noise(clean32, seed)
    dtype = model.x_embedder.proj.weight.dtype

    source_local = torch.zeros(batch_size, device=device, dtype=torch.long)
    source = middle.q_sample(
        clean16, source_local, d=stage_drift(middle, clean16), noise=eps16
    )
    source_global = torch.full((batch_size,), 300, device=device, dtype=torch.long)
    _, source_trace = manual_forward(
        model, source.to(dtype), source_global, labels, multires=True, capture=True
    )
    coarse = {
        layer: upsample_tokens(source_trace[layer]["residual"], 16, "nearest")
        for layer in range(len(model.blocks))
    }

    needed_times = sorted(
        set(targets)
        | {target + gap for target in targets for gap in gaps if target + gap <= 299}
    )
    traces: dict[int, dict] = {}
    for timestep in needed_times:
        local = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        latent = fine.q_sample(
            clean32, local, d=stage_drift(fine, clean32), noise=eps32
        )
        _, traces[timestep] = manual_forward(
            model, latent.to(dtype), local, labels, multires=True, capture=True
        )

    comparisons = []
    for target in targets:
        for gap in gaps:
            previous = target + gap
            if previous > 299:
                continue
            layers = []
            for layer in range(len(model.blocks)):
                exact = traces[target][layer]["residual"]
                generic = traces[previous][layer]["residual"]
                coarse_target = interpolated_prediction(
                    coefficients, coarse[layer], layer, target
                )
                coarse_previous = interpolated_prediction(
                    coefficients, coarse[layer], layer, previous
                )
                transport = generic + coarse_target - coarse_previous
                exact_low, _ = low_high_tokens(exact)
                generic_low, _ = low_high_tokens(generic)
                transport_low, _ = low_high_tokens(transport)
                generic_metrics = tensor_metrics(generic, exact)
                transport_metrics = tensor_metrics(transport, exact)
                generic_low_metrics = tensor_metrics(generic_low, exact_low)
                transport_low_metrics = tensor_metrics(transport_low, exact_low)
                layers.append({
                    "layer": layer,
                    "generic": generic_metrics,
                    "transport": transport_metrics,
                    "generic_low": generic_low_metrics,
                    "transport_low": transport_low_metrics,
                    "full_relative_l2_improvement": (
                        generic_metrics["relative_l2"] - transport_metrics["relative_l2"]
                    ),
                    "low_relative_l2_improvement": (
                        generic_low_metrics["relative_l2"]
                        - transport_low_metrics["relative_l2"]
                    ),
                })
            comparisons.append({
                "target_t": target,
                "previous_t": previous,
                "gap": gap,
                "layers": layers,
            })
    return {
        "seed": seed,
        "images": [str(path) for path in paths],
        "comparisons": comparisons,
    }


def summarize(validations: list[dict], gaps: list[int]) -> dict:
    summary = {}
    for gap in gaps:
        rows = [
            layer
            for validation in validations
            for comparison in validation["comparisons"]
            if comparison["gap"] == gap
            for layer in comparison["layers"]
        ]
        per_layer = []
        for layer_index in range(28):
            selected = [row for row in rows if row["layer"] == layer_index]
            mean = lambda key: sum(row[key] for row in selected) / len(selected)
            per_layer.append({
                "layer": layer_index,
                "full_relative_l2_improvement": mean("full_relative_l2_improvement"),
                "low_relative_l2_improvement": mean("low_relative_l2_improvement"),
                "transport_wins_full_fraction": sum(
                    row["full_relative_l2_improvement"] > 0 for row in selected
                ) / len(selected),
                "transport_wins_low_fraction": sum(
                    row["low_relative_l2_improvement"] > 0 for row in selected
                ) / len(selected),
            })
        summary[str(gap)] = {
            "mean_full_improvement": sum(
                row["full_relative_l2_improvement"] for row in rows
            ) / len(rows),
            "mean_low_improvement": sum(
                row["low_relative_l2_improvement"] for row in rows
            ) / len(rows),
            "transport_wins_full_fraction": sum(
                row["full_relative_l2_improvement"] > 0 for row in rows
            ) / len(rows),
            "transport_wins_low_fraction": sum(
                row["low_relative_l2_improvement"] > 0 for row in rows
            ) / len(rows),
            "best_layers_by_full_improvement": sorted(
                per_layer, key=lambda row: row["full_relative_l2_improvement"], reverse=True
            )[:8],
            "best_layers_by_low_improvement": sorted(
                per_layer, key=lambda row: row["low_relative_l2_improvement"], reverse=True
            )[:8],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-seeds", default="101,202")
    parser.add_argument("--validation-seeds", default="8101,8209")
    parser.add_argument("--calibration-batch", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=4)
    parser.add_argument("--knots", default="0,75,150,225,299")
    parser.add_argument("--targets", default="0,75,150,220,280")
    parser.add_argument("--gaps", default="1,4,8,16")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_rdd(Path(args.rdd_ckpt), device, torch.float32)
    knots = [int(value) for value in args.knots.split(",")]
    gaps = [int(value) for value in args.gaps.split(",")]
    targets = [int(value) for value in args.targets.split(",")]
    calibration_seeds = [int(value) for value in args.calibration_seeds.split(",")]
    validation_seeds = [int(value) for value in args.validation_seeds.split(",")]
    coefficients = calibrate_time_grid(
        model,
        Path(args.imagenet_train),
        Path(args.vae_path),
        device,
        calibration_seeds,
        args.calibration_batch,
        knots,
    )
    validations = [
        validation_seed(
            model,
            coefficients,
            Path(args.imagenet_train),
            Path(args.vae_path),
            device,
            seed,
            args.validation_batch,
            1901 + index * 331,
            targets,
            gaps,
        )
        for index, seed in enumerate(validation_seeds)
    ]
    report = {
        "definition": (
            "generic=previous fine residual; rdd_transport=previous fine residual "
            "+ current coarse prediction - previous coarse prediction"
        ),
        "knots": knots,
        "targets": targets,
        "gaps": gaps,
        "calibration_images": len(calibration_seeds) * args.calibration_batch,
        "validation_images": len(validation_seeds) * args.validation_batch,
        "summary": summarize(validations, gaps),
        "validations": validations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
