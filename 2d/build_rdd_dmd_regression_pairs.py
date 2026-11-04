#!/usr/bin/env python3
"""Build the fixed noise/teacher-output pairs required by original DMD Eq. 9.

Each pair stores all three noise tensors consumed by the 3-NFE RDD student and
the final latent produced by the frozen teacher with a deterministic 250-step
RDD marginal-DDIM sampler.  Storing boundary noise is necessary because RDD
noise filling is part of the generator input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from train_rdd_phased_dmd3 import (
    HistoricalCFGForward,
    build_model,
    extract_teacher_weights,
    infer_num_classes,
    parse_csv_ints,
)

from diffusion import create_diffusion_with_drift
from diffusion.rdd_phased_dmd import (
    fresh_transition,
    rdd_marginal_ddim_sample_loop,
    sampling_phases,
)


def build_sampling_diffusions(
    stage_lengths: tuple[int, ...],
    resolutions: tuple[int, ...],
    sampling_steps: tuple[int, ...],
):
    if len(stage_lengths) != len(resolutions) or len(sampling_steps) != len(resolutions):
        raise ValueError("stage lengths, resolutions, and sampling steps must align")
    boundaries = [0]
    for length in stage_lengths:
        boundaries.append(boundaries[-1] + length)
    steps_finest_first = tuple(reversed(sampling_steps))
    result = []
    for index, resolution in enumerate(resolutions):
        scale = resolution // resolutions[index + 1] if index + 1 < len(resolutions) else 2
        result.append(
            create_diffusion_with_drift(
                timestep_respacing=str(steps_finest_first[index]),
                diffusion_steps=sum(stage_lengths),
                c=index if index + 1 < len(resolutions) else None,
                T_c=boundaries[index],
                T_c_plus_1=boundaries[index + 1],
                sample_scale=scale,
            )
        )
    return result


@torch.no_grad()
def sample_teacher(
    teacher_forward,
    diffusions,
    resolutions: tuple[int, ...],
    labels: torch.Tensor,
    initial_noise: torch.Tensor,
    boundary_noise_16: torch.Tensor,
    boundary_noise_32: torch.Tensor,
) -> torch.Tensor:
    phases = sampling_phases(resolutions)
    boundary_noises = (boundary_noise_16, boundary_noise_32)
    state = initial_noise
    clean = None
    for phase in phases:
        diffusion = diffusions[phase.diffusion_index]
        output = rdd_marginal_ddim_sample_loop(
            diffusion,
            teacher_forward,
            state.shape,
            noise=state,
            clip_denoised=False,
            model_kwargs={"y": labels},
        )
        clean = output["pred_xstart"].float()
        if phase.phase_index + 1 < len(phases):
            next_phase = phases[phase.phase_index + 1]
            state = fresh_transition(
                clean,
                next_phase.resolution,
                diffusions[next_phase.diffusion_index],
                noise=boundary_noises[phase.phase_index],
            )
    if clean is None:
        raise RuntimeError("teacher sampler produced no output")
    return clean


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("pair generation requires CUDA")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolutions = parse_csv_ints(args.resolutions)
    stage_lengths = parse_csv_ints(args.stage_lengths)
    sampling_steps = parse_csv_ints(args.sampling_steps)
    if resolutions != (32, 16, 8):
        raise ValueError("the paper-faithful 3-NFE experiment expects 32,16,8")
    if sum(stage_lengths) != 1000:
        raise ValueError("stage lengths must sum to 1000")
    if sum(sampling_steps) != 250:
        raise ValueError("original DMD ImageNet uses a 256-step teacher; RDD parity uses 250")

    weights, metadata = extract_teacher_weights(args.teacher_checkpoint)
    if metadata["stage_lengths"] is not None and tuple(metadata["stage_lengths"]) != stage_lengths:
        raise ValueError("teacher checkpoint stage lengths do not match")
    num_classes = infer_num_classes(weights)
    teacher = build_model(args.model, resolutions, weights).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    del weights
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_forward = HistoricalCFGForward(teacher, num_classes, args.teacher_cfg)
    diffusions = build_sampling_diffusions(stage_lengths, resolutions, sampling_steps)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + rank * 1_000_003)
    local_indices = list(range(rank, int(args.total_pairs), world_size))
    manifest_path = output_dir / f"manifest_rank{rank:02d}.jsonl"
    manifest_temp = output_dir / f"manifest_rank{rank:02d}.jsonl.tmp-{os.getpid()}"
    with open(manifest_temp, "w", encoding="utf-8") as manifest:
        for shard_number, start in enumerate(range(0, len(local_indices), args.batch_per_gpu)):
            indices = local_indices[start : start + args.batch_per_gpu]
            batch = len(indices)
            labels = torch.randint(
                0, num_classes, (batch,), generator=generator, device=device
            )
            noise_8 = torch.randn(
                batch, 4, 8, 8, generator=generator, device=device
            )
            noise_16 = torch.randn(
                batch, 4, 16, 16, generator=generator, device=device
            )
            noise_32 = torch.randn(
                batch, 4, 32, 32, generator=generator, device=device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                target = sample_teacher(
                    teacher_forward,
                    diffusions,
                    resolutions,
                    labels,
                    noise_8,
                    noise_16,
                    noise_32,
                )
                if args.verify_determinism and shard_number == 0:
                    repeated_target = sample_teacher(
                        teacher_forward,
                        diffusions,
                        resolutions,
                        labels,
                        noise_8,
                        noise_16,
                        noise_32,
                    )
                    if not torch.equal(target, repeated_target):
                        max_error = float((target - repeated_target).abs().max())
                        raise RuntimeError(
                            "RDD marginal-DDIM pair generation is not bitwise "
                            f"deterministic; max_abs_error={max_error}"
                        )
            filename = f"rank{rank:02d}_shard{shard_number:05d}.pt"
            temporary = output_dir / f"{filename}.tmp-{os.getpid()}"
            torch.save(
                {
                    "indices": torch.tensor(indices, dtype=torch.int64),
                    "labels": labels.cpu(),
                    "noise_8": noise_8.to(torch.float16).cpu(),
                    "noise_16": noise_16.to(torch.float16).cpu(),
                    "noise_32": noise_32.to(torch.float16).cpu(),
                    "target_latent": target.to(torch.float16).cpu(),
                },
                temporary,
            )
            os.replace(temporary, output_dir / filename)
            manifest.write(json.dumps({"file": filename, "count": batch}) + "\n")
            manifest.flush()
            print(
                f"rank={rank} saved {filename}: {start + batch}/{len(local_indices)} local pairs",
                flush=True,
            )
    os.replace(manifest_temp, manifest_path)

    if distributed:
        dist.barrier()
    if rank == 0:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
                    "teacher_cfg": float(args.teacher_cfg),
                    "model": args.model,
                    "resolutions": resolutions,
                    "stage_lengths": stage_lengths,
                    "sampling_steps_coarse_to_fine": sampling_steps,
                    "sampler": "RDD marginal-DDIM (shared inferred noise)",
                    "sampler_version": 2,
                    "determinism_verified": bool(args.verify_determinism),
                    "total_pairs": int(args.total_pairs),
                    "seed": int(args.seed),
                    "world_size": world_size,
                },
                handle,
                indent=2,
            )
    if distributed:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="DiT-L/2")
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--sampling-steps", default="80,80,90")
    parser.add_argument("--teacher-cfg", type=float, default=1.5)
    parser.add_argument("--total-pairs", type=int, default=25000)
    parser.add_argument("--batch-per-gpu", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
