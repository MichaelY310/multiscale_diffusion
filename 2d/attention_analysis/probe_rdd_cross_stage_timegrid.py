#!/usr/bin/env python3
"""Measure RDD block reuse structure across middle/fine stage timelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    aligned_noise,
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


def make_diffusion(c: int, start: int, end: int):
    return create_diffusion_with_drift(
        timestep_respacing="",
        diffusion_steps=1000,
        c=c,
        T_c=start,
        T_c_plus_1=end,
    )


@torch.inference_mode()
def capture_at(model, diffusion, clean, noise, local_timestep, labels, dtype):
    batch = clean.shape[0]
    local_t = torch.full((batch,), local_timestep, device=clean.device, dtype=torch.long)
    global_value = int(diffusion.timestep_map[local_timestep])
    global_t = torch.full((batch,), global_value, device=clean.device, dtype=torch.long)
    x_t = diffusion.q_sample(
        clean,
        local_t,
        d=stage_drift(diffusion, clean),
        noise=noise,
    )
    output, trace = manual_forward(
        model,
        x_t.to(dtype),
        global_t,
        labels,
        multires=True,
        capture=True,
    )
    return {
        "local_timestep": local_timestep,
        "global_timestep": global_value,
        "local_t": local_t,
        "global_t": global_t,
        "x": x_t,
        "output": output,
        "trace": trace,
    }


def cross_scale_metrics(source, target, depth):
    layers = []
    for layer in range(depth):
        source_residual = upsample_tokens(source["trace"][layer]["residual"], 16, "nearest")
        source_hidden = upsample_tokens(source["trace"][layer]["output"], 16, "nearest")
        target_residual = target["trace"][layer]["residual"]
        target_low, _ = low_high_tokens(target_residual)
        layers.append(
            {
                "layer": layer,
                "residual": tensor_metrics(source_residual, target_residual),
                "residual_to_target_low": tensor_metrics(source_residual, target_low),
                "whole_hidden": tensor_metrics(source_hidden, target["trace"][layer]["output"]),
            }
        )
    return {
        "source_global_timestep": source["global_timestep"],
        "target_global_timestep": target["global_timestep"],
        "mean_residual_cosine": sum(row["residual"]["cosine"] for row in layers) / depth,
        "mean_low_frequency_cosine": sum(row["residual_to_target_low"]["cosine"] for row in layers) / depth,
        "mean_hidden_cosine": sum(row["whole_hidden"]["cosine"] for row in layers) / depth,
        "layers": layers,
    }


def same_scale_metrics(source, target, depth):
    layers = []
    for layer in range(depth):
        layers.append(
            {
                "layer": layer,
                "residual": tensor_metrics(
                    source["trace"][layer]["residual"], target["trace"][layer]["residual"]
                ),
                "whole_hidden": tensor_metrics(
                    source["trace"][layer]["output"], target["trace"][layer]["output"]
                ),
            }
        )
    return {
        "source_global_timestep": source["global_timestep"],
        "target_global_timestep": target["global_timestep"],
        "mean_residual_cosine": sum(row["residual"]["cosine"] for row in layers) / depth,
        "mean_hidden_cosine": sum(row["whole_hidden"]["cosine"] for row in layers) / depth,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9191)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", default="299,225,150,75,0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_rdd(Path(args.rdd_ckpt), device, torch.bfloat16)
    dtype = model.x_embedder.proj.weight.dtype
    paths, labels = discover_train_images(Path(args.imagenet_train), args.batch_size, 991)
    clean32 = encode_images(paths, Path(args.vae_path), device)
    clean16 = F.avg_pool2d(clean32, 2, 2)
    eps32, eps16 = aligned_noise(clean32, args.seed)
    labels = labels.to(device)
    fine = make_diffusion(0, 0, 300)
    middle = make_diffusion(1, 300, 600)
    grid = [int(value) for value in args.timesteps.split(",")]
    fine_needed = sorted(set(grid + [min(299, t + 1) for t in grid] + [min(299, t + 4) for t in grid]))
    fine_calls = {
        t: capture_at(model, fine, clean32, eps32, t, labels, dtype)
        for t in fine_needed
    }
    middle_calls = {
        t: capture_at(model, middle, clean16, eps16, t, labels, dtype)
        for t in grid
    }
    depth = len(model.blocks)
    matrix = {
        str(target_t): {
            str(source_t): cross_scale_metrics(middle_calls[source_t], fine_calls[target_t], depth)
            for source_t in grid
        }
        for target_t in grid
    }
    normalized = {str(t): matrix[str(t)][str(t)] for t in grid}
    boundary = {str(t): matrix[str(t)]["0"] for t in grid}
    temporal = {}
    for target_t in grid:
        temporal[str(target_t)] = {}
        for gap in (1, 4):
            source_t = min(299, target_t + gap)
            temporal[str(target_t)][f"gap_{gap}"] = same_scale_metrics(
                fine_calls[source_t], fine_calls[target_t], depth
            )
    stagewide_direct = {}
    stable_layers = [14, 17, 20]
    boundary_call = middle_calls[0]
    replacements = {
        layer: upsample_tokens(
            boundary_call["trace"][layer]["residual"], 16, "nearest"
        )
        for layer in stable_layers
    }
    for target_t in grid:
        target_call = fine_calls[target_t]
        replaced_output, _ = manual_forward(
            model,
            target_call["x"].to(dtype),
            target_call["global_t"],
            labels,
            multires=True,
            replacements=replacements,
            replacement_mode="residual",
        )
        exact_x0 = pred_x0(
            fine, target_call["output"], target_call["x"], target_call["local_t"]
        )
        replaced_x0 = pred_x0(
            fine, replaced_output, target_call["x"], target_call["local_t"]
        )
        stagewide_direct[str(target_t)] = {
            "layers": stable_layers,
            "pred_x0": tensor_metrics(replaced_x0, exact_x0),
            "model_output": tensor_metrics(
                replaced_output[:, :4], target_call["output"][:, :4]
            ),
        }
    report = {
        "seed": args.seed,
        "images": [str(path) for path in paths],
        "local_timestep_grid": grid,
        "definitions": {
            "normalized_pair": "same local index in 16x16 and 32x32 stages; global times differ by 300",
            "boundary_source": "always use the final 16x16 call at global t=300",
            "fine_temporal": "same-resolution fine-stage cache from local t+1 or t+4",
        },
        "cross_scale_matrix": matrix,
        "normalized_pair": normalized,
        "boundary_source": boundary,
        "fine_temporal": temporal,
        "stagewide_direct_stable3": stagewide_direct,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
