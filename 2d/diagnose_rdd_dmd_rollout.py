#!/usr/bin/env python3
"""Compare paired teacher targets, teacher 3-NFE, and distilled 3-NFE rollouts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image

from train_rdd_phased_dmd3 import (
    build_diffusions,
    build_model,
    extract_teacher_weights,
    parse_csv_ints,
    predict_clean,
    top_timestep,
)
from diffusion.rdd_phased_dmd import fresh_transition, sampling_phases


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "abs_mean": float(value.abs().mean()),
        "finite_fraction": float(torch.isfinite(value).float().mean()),
        "channel_mean": [float(item) for item in value.mean((0, 2, 3))],
        "channel_std": [float(item) for item in value.std((0, 2, 3))],
    }


@torch.no_grad()
def rollout(model, diffusions, resolutions, pair, device):
    phases = sampling_phases(resolutions)
    boundary_noises = (pair["noise_16"], pair["noise_32"])
    state = pair["noise_8"].to(device=device, dtype=torch.float32)
    labels = pair["labels"].to(device=device, dtype=torch.long)
    outputs = []
    stats = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for phase in phases:
            diffusion = diffusions[phase.diffusion_index]
            timestep = top_timestep(diffusion, labels.shape[0], device)
            clean = predict_clean(diffusion, model, state, timestep, labels).float()
            stats.append(
                {
                    "phase": int(phase.phase_index),
                    "resolution": int(phase.resolution),
                    "global_timestep": int(diffusion.timestep_map[-1]),
                    "input": tensor_stats(state),
                    "predicted_clean": tensor_stats(clean),
                }
            )
            outputs.append(clean.cpu())
            if phase.phase_index + 1 < len(phases):
                next_phase = phases[phase.phase_index + 1]
                state = fresh_transition(
                    clean,
                    next_phase.resolution,
                    diffusions[next_phase.diffusion_index],
                    noise=boundary_noises[phase.phase_index].to(
                        device=device, dtype=clean.dtype
                    ),
                )
    return outputs, stats


@torch.no_grad()
def decode(vae, latents: torch.Tensor, device) -> list[Image.Image]:
    latents = latents.to(device=device, dtype=torch.float32)
    if latents.shape[-1] != 32:
        latents = F.interpolate(latents, size=(32, 32), mode="nearest")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = vae.decode(latents / 0.18215).sample
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    arrays = (
        (decoded * 255)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array) for array in arrays]


def save_row(images: list[Image.Image], path: Path) -> None:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    canvas.save(path)


def load_student_weights(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("ema", "student", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key], int(checkpoint.get("step", -1))
    raise KeyError(f"no student weights in {path}")


def main(args) -> None:
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolutions = parse_csv_ints(args.resolutions)
    stage_lengths = parse_csv_ints(args.stage_lengths)
    diffusions = build_diffusions(stage_lengths, resolutions, sum(stage_lengths))

    shard = sorted(Path(args.regression_data).glob("rank*_shard*.pt"))[0]
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    pair = {
        key: payload[key][: args.batch_size]
        for key in ("labels", "noise_8", "noise_16", "noise_32", "target_latent")
    }
    report = {
        "pair_shard": str(shard),
        "labels": [int(item) for item in pair["labels"]],
        "target_latent": tensor_stats(pair["target_latent"]),
    }

    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True
    ).to(device).eval()
    save_row(decode(vae, pair["target_latent"], device), output_dir / "paired_teacher_target.png")

    teacher_weights, teacher_metadata = extract_teacher_weights(args.teacher_checkpoint)
    teacher = build_model(args.model, resolutions, teacher_weights).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    teacher_outputs, teacher_stats = rollout(
        teacher, diffusions, resolutions, pair, device
    )
    report["teacher_checkpoint"] = teacher_metadata
    report["teacher_3nfe"] = teacher_stats
    for output, stage in zip(teacher_outputs, (8, 16, 32)):
        save_row(decode(vae, output, device), output_dir / f"teacher_3nfe_stage{stage}.png")
    del teacher, teacher_weights
    gc.collect()
    torch.cuda.empty_cache()

    student_weights, student_step = load_student_weights(args.student_checkpoint)
    student = build_model(args.model, resolutions, student_weights).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    student_outputs, student_stats = rollout(
        student, diffusions, resolutions, pair, device
    )
    report["student_step"] = student_step
    report["student_3nfe"] = student_stats
    for output, stage in zip(student_outputs, (8, 16, 32)):
        save_row(decode(vae, output, device), output_dir / f"student_3nfe_stage{stage}.png")

    student_final = student_outputs[-1].float()
    target = pair["target_latent"].float()
    report["student_vs_target"] = {
        "mse": float(F.mse_loss(student_final, target)),
        "mae": float(F.l1_loss(student_final, target)),
        "cosine": float(
            F.cosine_similarity(student_final.flatten(1), target.flatten(1)).mean()
        ),
    }
    teacher_final = teacher_outputs[-1].float()
    report["teacher_3nfe_vs_target"] = {
        "mse": float(F.mse_loss(teacher_final, target)),
        "mae": float(F.l1_loss(teacher_final, target)),
        "cosine": float(
            F.cosine_similarity(teacher_final.flatten(1), target.flatten(1)).mean()
        ),
    }
    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--regression-data", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="DiT-L/2")
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
