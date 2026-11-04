"""
Pre-compute and cache VAE latents at all resolution stages for RDD-Wan training.

This script processes pixel-space videos (.mp4 or .pt) and saves pre-computed
VAE latents at ALL stage resolutions, so training doesn't need to call VAE online.

Two modes:
  - "pixel": downsample video in pixel space, then VAE encode each resolution (PyramidalWan style)
  - "latent": VAE encode full-res only, then avg-pool to get lower resolutions

Output format per sample:
  {
    "latents": [latent_stage0, latent_stage1, ..., latent_stageN],  # list of tensors
    "text_emb": tensor [L, D],  # T5 text embedding
    "prompt": str,              # original prompt text
  }

Usage:
    # From pixel videos (pixel mode, PyramidalWan-style):
    python precompute_latents.py \
        --input-dir /path/to/videos \
        --output-dir /path/to/cached_latents \
        --vae-path /path/to/Wan2.1-T2V-1.3B \
        --mode pixel \
        --stage-spatial-factors 4,2,1 \
        --stage-temporal-factors 4,2,1

    # From existing full-res latents (latent mode):
    python precompute_latents.py \
        --input-dir /path/to/full_res_latents \
        --output-dir /path/to/cached_latents \
        --mode latent \
        --stage-spatial-factors 4,2,1 \
        --stage-temporal-factors 4,2,1

    # Multi-GPU sharding:
    torchrun --nproc_per_node=8 precompute_latents.py ...
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def downsample_pixel_trilinear(video: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    """
    Downsample pixel-space video using trilinear interpolation.
    
    Args:
        video: [C, T, H, W] pixel video in [-1, 1]
        spatial_factor: spatial downsampling factor (e.g., 2 or 4)
        temporal_factor: temporal downsampling factor (e.g., 2 or 4)
    Returns:
        [C, T//tf, H//sf, W//sf] downsampled video
    """
    if spatial_factor == 1 and temporal_factor == 1:
        return video

    C, T, H, W = video.shape
    T_new = T // temporal_factor
    H_new = H // spatial_factor
    W_new = W // spatial_factor

    # Use F.interpolate with trilinear mode (requires 5D: [N, C, D, H, W])
    video_5d = video.unsqueeze(0)  # [1, C, T, H, W]
    video_down = F.interpolate(
        video_5d,
        size=(T_new, H_new, W_new),
        mode='trilinear',
        align_corners=False,
    )
    return video_down.squeeze(0)  # [C, T_new, H_new, W_new]


def downsample_latent_avgpool(latent: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    """
    Downsample latent via average pooling.
    
    Args:
        latent: [C, F, H, W] latent tensor
        spatial_factor: spatial downsampling factor
        temporal_factor: temporal downsampling factor
    Returns:
        [C, F//tf, H//sf, W//sf]
    """
    if spatial_factor == 1 and temporal_factor == 1:
        return latent

    C, F, H, W = latent.shape

    # Temporal avg pool
    if temporal_factor > 1 and F >= temporal_factor:
        pad_f = (temporal_factor - F % temporal_factor) % temporal_factor
        if pad_f > 0:
            latent = torch.cat([latent, latent[:, -1:].expand(-1, pad_f, -1, -1)], dim=1)
            F = F + pad_f
        latent = latent.view(C, F // temporal_factor, temporal_factor, H, W).mean(dim=2)
        F = F // temporal_factor

    # Spatial avg pool
    if spatial_factor > 1 and H >= spatial_factor and W >= spatial_factor:
        pad_h = (spatial_factor - H % spatial_factor) % spatial_factor
        pad_w = (spatial_factor - W % spatial_factor) % spatial_factor
        if pad_h > 0 or pad_w > 0:
            latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
        latent = latent.view(C, F, H // spatial_factor, spatial_factor, W // spatial_factor, spatial_factor)
        latent = latent.mean(dim=(3, 5))

    return latent


def load_vae(vae_path: str, device: torch.device):
    """Load Wan VAE model."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Wan2.1'))
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth=vae_path, device=device)
    return vae


@torch.no_grad()
def encode_video(vae, video: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encode a pixel-space video with VAE.
    
    Args:
        vae: Wan VAE model
        video: [C, T, H, W] in [-1, 1]
        device: target device
    Returns:
        [C_latent, F, H//8, W//8] latent tensor
    """
    video_on_device = video.to(device)
    latent_list = vae.encode([video_on_device])  # returns list of [z_dim, F_z, H_z, W_z]
    return latent_list[0].cpu()


def process_single_pixel_video(
    video: torch.Tensor,
    vae,
    device: torch.device,
    spatial_factors: list,
    temporal_factors: list,
) -> list:
    """
    Process one pixel video: downsample to each stage, VAE encode each.
    
    Args:
        video: [C, T, H, W] pixel video
        spatial_factors: list of spatial factors per stage (coarsest first)
        temporal_factors: list of temporal factors per stage (coarsest first)
    Returns:
        list of latent tensors, one per stage
    """
    latents = []
    for sf, tf in zip(spatial_factors, temporal_factors):
        video_down = downsample_pixel_trilinear(video, sf, tf)
        latent = encode_video(vae, video_down, device)
        latents.append(latent)
    return latents


def process_single_latent(
    full_latent: torch.Tensor,
    spatial_factors: list,
    temporal_factors: list,
) -> list:
    """
    Process one full-res latent: avg-pool to each stage.
    
    Args:
        full_latent: [C, F, H, W] full-resolution latent
        spatial_factors: list of spatial factors per stage
        temporal_factors: list of temporal factors per stage
    Returns:
        list of latent tensors, one per stage
    """
    latents = []
    for sf, tf in zip(spatial_factors, temporal_factors):
        latent_down = downsample_latent_avgpool(full_latent, sf, tf)
        latents.append(latent_down)
    return latents


def load_pixel_record(filepath: str):
    """Load a pixel video and recover matching prompt/T5 context when available."""
    path = Path(filepath)
    text_emb = None
    prompt = ""

    if path.suffix.lower() == ".mp4":
        video_thwc, _, _ = torchvision.io.read_video(
            str(path), pts_unit="sec", output_format="THWC"
        )
        video = video_thwc.permute(3, 0, 1, 2).float() / 127.5 - 1.0

        # generate_training_data.py stores the matching conditional context in
        # a sibling latent directory: <root>/pixel/sample_x.mp4 and
        # <root>/latent/sample_x.pt.
        companion = path.parent.parent / "latent" / f"{path.stem}.pt"
        if companion.exists():
            record = torch.load(companion, map_location="cpu", weights_only=True)
            if isinstance(record, dict):
                text_emb = record.get("text_emb")
                prompt = record.get("prompt", "")

        metadata = path.parent.parent / "metadata" / f"{path.stem}.json"
        if not prompt and metadata.exists():
            with metadata.open("r", encoding="utf-8") as handle:
                prompt = json.load(handle).get("prompt", "")
        return video, text_emb, prompt

    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        return data["video"].float(), data.get("text_emb"), data.get("prompt", "")
    return data.float(), None, ""


def main():
    parser = argparse.ArgumentParser(description="Pre-compute multi-resolution VAE latents")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with source data (.pt latents or .mp4 videos)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cached multi-res latents")
    parser.add_argument("--mode", type=str, choices=["pixel", "latent"], default="latent",
                        help="'pixel': downsample pixels then VAE encode; 'latent': VAE encode then avgpool")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to Wan VAE (required for pixel mode)")
    parser.add_argument("--stage-spatial-factors", type=str, default="4,2,1",
                        help="Comma-separated spatial downsample factors per stage (coarsest first)")
    parser.add_argument("--stage-temporal-factors", type=str, default="4,2,1",
                        help="Comma-separated temporal downsample factors per stage (coarsest first)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (only 1 supported for pixel mode with VAE)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Storage dtype for latents (float16 saves disk)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        help="Allow records without T5 context (unconditional smoke tests only)",
    )
    # Multi-GPU sharding
    parser.add_argument("--shard-rank", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")

    args = parser.parse_args()

    spatial_factors = [int(x) for x in args.stage_spatial_factors.split(",")]
    temporal_factors = [int(x) for x in args.stage_temporal_factors.split(",")]
    assert len(spatial_factors) == len(temporal_factors), "Spatial and temporal factors must have same length"

    num_stages = len(spatial_factors)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    save_dtype = dtype_map[args.dtype]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE if needed
    vae = None
    if args.mode == "pixel":
        assert args.vae_path is not None, "--vae-path required for pixel mode"
        print(f"Loading VAE from {args.vae_path}...")
        vae = load_vae(args.vae_path, device)
        print("VAE loaded.")

    # Find all input files
    input_files = []
    for ext in ["*.pt", "*.pth", "*.mp4"]:
        input_files.extend(sorted(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True)))

    if not input_files:
        print(f"ERROR: No .pt/.pth/.mp4 files found in {args.input_dir}")
        return

    # Shard for multi-GPU
    if args.num_shards > 1:
        input_files = input_files[args.shard_rank::args.num_shards]

    print(f"Mode: {args.mode}")
    print(f"Stages: {num_stages} (spatial: {spatial_factors}, temporal: {temporal_factors})")
    print(f"Input files: {len(input_files)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Save dtype: {args.dtype}")
    print(f"Shard {args.shard_rank}/{args.num_shards}")
    print()

    processed = 0
    skipped = 0
    start_time = time.time()

    for i, filepath in enumerate(input_files):
        # Output filename
        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(args.output_dir, f"{basename}_multires.pt")

        # Skip if already processed
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            # Load input data
            if args.mode == "pixel":
                video, text_emb, prompt = load_pixel_record(filepath)

                # Process: downsample in pixel space, encode each resolution
                latents = process_single_pixel_video(
                    video, vae, device, spatial_factors, temporal_factors
                )

            else:  # latent mode
                data = torch.load(filepath, map_location="cpu", weights_only=True)
                # Expect dict with "latent" [C, F, H, W] and optionally "text_emb"
                if isinstance(data, dict):
                    full_latent = data["latent"].float()
                    text_emb = data.get("text_emb", None)
                    prompt = data.get("prompt", "")
                else:
                    full_latent = data.float()
                    text_emb = None
                    prompt = ""

                # Process: avg-pool to each stage resolution
                latents = process_single_latent(full_latent, spatial_factors, temporal_factors)

            if text_emb is None and not args.allow_missing_text:
                raise RuntimeError(
                    f"Missing text_emb for {filepath}. Regenerate with the updated "
                    "generate_training_data.py, provide the sibling latent record, or "
                    "pass --allow-missing-text only for an unconditional smoke test."
                )

            # Save result
            save_data = {
                "latents": [l.to(save_dtype) for l in latents],
                "prompt": prompt,
            }
            if text_emb is not None:
                save_data["text_emb"] = text_emb.to(save_dtype)

            torch.save(save_data, out_path)
            processed += 1

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start_time
                speed = processed / max(elapsed, 1)
                remaining = (len(input_files) - i - 1) / max(speed, 0.01)
                print(f"[{i+1}/{len(input_files)}] processed={processed}, skipped={skipped}, "
                      f"speed={speed:.1f} it/s, ETA={remaining/60:.1f}min")

        except Exception as e:
            print(f"ERROR processing {filepath}: {e}")
            if "Missing text_emb" in str(e):
                raise
            continue

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {processed}, skipped {skipped} in {elapsed:.0f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
