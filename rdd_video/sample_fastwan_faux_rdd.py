"""FastWan2.1 DMD baseline and zero-finetuning faux-RDD sanity samples.

FastVideo's published checkpoint supports different *fixed* output sizes but
was not trained to change resolution inside one trajectory.  This script uses
its three trained DMD timesteps and, for the faux-RDD mode, assigns one call to
each nested spatiotemporal resolution.  That experiment assumes the checkpoint
had been trained with RDD's matched boundary marginals; it does not claim that
the released FastWan weights actually contain an RDD drift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Wan2.1"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "attention_analysis"))

from sample_rdd_wan import save_video


def parse_pairs(text: str) -> list[tuple[int, int]]:
    result = []
    for item in text.split(";"):
        spatial, temporal = (int(value) for value in item.split(","))
        result.append((spatial, temporal))
    return result


def upsample_nested(value: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    target_t, target_h, target_w = target
    ratio_t = math.ceil(target_t / value.shape[2])
    ratio_h = math.ceil(target_h / value.shape[3])
    ratio_w = math.ceil(target_w / value.shape[4])
    value = value.repeat_interleave(ratio_t, 2)
    value = value.repeat_interleave(ratio_h, 3).repeat_interleave(ratio_w, 4)
    return value[:, :, :target_t, :target_h, :target_w]


@torch.inference_mode()
def predict_clean(model, state, context, timestep: int) -> torch.Tensor:
    time_input = torch.full(
        (state.shape[0],), float(timestep), device=state.device, dtype=torch.float32
    )
    prediction = model(
        hidden_states=state,
        timestep=time_input,
        encoder_hidden_states=context,
        return_dict=False,
    )[0]
    sigma = float(timestep) / 1000.0
    return state - sigma * prediction


@torch.inference_mode()
def sample_three_call_dmd(
    model,
    context,
    full_shape,
    stage_factors,
    timesteps,
    seed,
    controller=None,
):
    if len(stage_factors) != len(timesteps):
        raise ValueError("one stage factor is required for every DMD timestep")
    generator = torch.Generator(device=full_shape.device).manual_seed(seed)
    batch, channels, full_t, full_h, full_w = full_shape.shape
    clean = None
    stage_shapes = []
    start = time.perf_counter()
    for stage, ((spatial, temporal), timestep) in enumerate(zip(stage_factors, timesteps)):
        shape = (
            math.ceil(full_t / temporal),
            math.ceil(full_h / spatial),
            math.ceil(full_w / spatial),
        )
        stage_shapes.append(shape)
        if clean is None:
            state = torch.randn(
                batch,
                channels,
                *shape,
                generator=generator,
                device=full_shape.device,
                dtype=full_shape.dtype,
            )
        else:
            lowpass_clean = upsample_nested(clean, shape)
            noise = torch.randn(
                lowpass_clean.shape,
                generator=generator,
                device=lowpass_clean.device,
                dtype=lowpass_clean.dtype,
            )
            sigma = float(timestep) / 1000.0
            # Canonical 2-D-compatible RDD noise filling: upsample the clean
            # prediction and draw a fresh full-space i.i.d. Gaussian.
            # This historical diffusers path has always used this coupling.
            state = (1.0 - sigma) * lowpass_clean + sigma * noise
        if controller is not None:
            controller.begin_stage(stage, shape, capture=stage < len(timesteps) - 1)
        stage_start = time.perf_counter()
        clean = predict_clean(model, state, context, timestep)
        torch.cuda.synchronize(state.device)
        print(
            f"stage={stage} factor=({spatial},{temporal}) timestep={timestep} "
            f"shape={shape} seconds={time.perf_counter()-stage_start:.3f}"
        )
    return clean, stage_shapes, time.perf_counter() - start


def main(args) -> None:
    from diffusers import WanTransformer3DModel
    from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    device = torch.device(args.device)
    dtype = torch.bfloat16
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    width, height = (int(x) for x in args.size.split("x"))
    latent_t = (args.frame_num - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    if args.frame_num != (latent_t - 1) * 4 + 1:
        raise ValueError("frame_num must equal 4*k+1 for Wan VAE")
    stage_factors = parse_pairs(args.stage_factors)
    timesteps = [int(x) for x in args.dmd_timesteps.split(",")]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"Loading FastWan transformer from {args.fastwan_dir}")
    model = WanTransformer3DModel.from_pretrained(
        args.fastwan_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    controller = None
    if args.route_density < 1.0:
        from fastwan_diffusers_rdd_route import FastWanRDDRouteController

        controller = FastWanRDDRouteController(
            model,
            route_density=args.route_density,
            fastvideo_kernel_root=args.fastvideo_kernel_root,
        )

    original = Path(args.original_wan_dir)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=str(original / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(original / "google/umt5-xxl"),
    )
    text_encoder.model.to(device)
    context_list = text_encoder([args.prompt], device)
    context = context_list[0].unsqueeze(0).to(device=device, dtype=dtype)
    text_encoder.model.cpu()
    del text_encoder
    torch.cuda.empty_cache()

    full = torch.empty(
        1, 16, latent_t, latent_h, latent_w, device=device, dtype=dtype
    )
    sample_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        latent, stage_shapes, denoise_seconds = sample_three_call_dmd(
            model,
            context,
            full,
            stage_factors,
            timesteps,
            seed,
            controller,
        )
    total_before_decode = time.perf_counter() - sample_start

    latent_path = Path(args.latent_output or Path(args.output).with_suffix(".latent.pt"))
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent.detach().cpu(), latent_path)
    model.cpu()
    del model
    torch.cuda.empty_cache()

    vae = WanVAE(
        z_dim=16,
        vae_pth=str(original / "Wan2.1_VAE.pth"),
        dtype=torch.float32,
        device=device,
    )
    decode_start = time.perf_counter()
    video = vae.decode([latent[0].float()])[0]
    decode_seconds = time.perf_counter() - decode_start
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output), fps=args.fps)

    metrics = {
        "warning": "Released FastWan weights are DMD/VSA weights, not RDD weights.",
        "output": str(output.resolve()),
        "latent_output": str(latent_path.resolve()),
        "size": args.size,
        "frame_num": args.frame_num,
        "prompt": args.prompt,
        "seed": seed,
        "dmd_timesteps": timesteps,
        "stage_factors_spatial_temporal": stage_factors,
        "stage_latent_shapes": stage_shapes,
        "transition_mode": "fresh",
        "denoise_and_boundary_seconds": denoise_seconds,
        "sample_wrapper_seconds": total_before_decode,
        "vae_decode_seconds": decode_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "attention": controller.summary() if controller else {"mode": "dense"},
    }
    metrics_path = Path(args.metrics_json or output.with_suffix(".metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwan-dir", required=True)
    parser.add_argument("--original-wan-dir", required=True)
    parser.add_argument("--fastvideo-kernel-root", default="third_party/FastVideo-kernel-ref/fastvideo-kernel")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=29)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dmd-timesteps", default="1000,757,522")
    parser.add_argument("--stage-factors", default="4,4;2,2;1,1")
    parser.add_argument("--route-density", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-output")
    parser.add_argument("--metrics-json")
    main(parser.parse_args())
