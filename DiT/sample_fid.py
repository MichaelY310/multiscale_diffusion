"""
Distributed FID sampling for Original DiT.
Supports 8-GPU parallel sampling with rank-based sample allocation and skip logic.
Supports DDPM, DPM-Solver-1, DPM-Solver-2 sampling methods.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models


def main(args):
    import functools
    # Force all prints to flush immediately
    import builtins
    builtins.print = functools.partial(print, flush=True)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    rank = args.rank
    world_size = args.world_size
    if world_size > 1:
        device = "cuda:0"
        torch.cuda.set_device(0)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights.")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Loaded model weights (no EMA found).")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state dict.")
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    use_cfg = args.cfg_scale > 1.0
    total_samples = args.total_samples
    batch_size = args.batch_size

    print(f"device={device}  rank={rank}/{world_size}")
    print(f"total_samples={total_samples}  batch_size={batch_size}")
    print(f"cfg_scale={args.cfg_scale}  steps={args.num_sampling_steps}  method={args.sampling_method}")
    print(f"output_dir={args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Compute sample indices for this rank
    my_indices = list(range(rank, total_samples, world_size))
    # Skip logic: check existing files
    pending_indices = [i for i in my_indices if not os.path.exists(
        os.path.join(args.output_dir, f"{i:06d}.png"))]
    existing_count = len(my_indices) - len(pending_indices)
    if existing_count > 0:
        print(f"[Rank {rank}] Found {existing_count}/{len(my_indices)} existing, "
              f"{len(pending_indices)} remaining")

    total_to_generate = len(pending_indices)
    if total_to_generate == 0:
        print(f"[Rank {rank}] All samples already exist. Done.")
        return

    num_generated = 0
    io_executor = ThreadPoolExecutor(max_workers=4)
    io_futures = []

    print(f"\n=== [Rank {rank}] Generating {total_to_generate} samples ===")

    for batch_start in range(0, total_to_generate, batch_size):
        batch_indices = pending_indices[batch_start:batch_start + batch_size]
        current_batch_size = len(batch_indices)

        # Generate random class labels
        torch.manual_seed(args.seed + batch_indices[0])
        y = torch.randint(0, args.num_classes, (current_batch_size,), device=device)
        z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)

        if use_cfg:
            z_input = torch.cat([z, z], dim=0)
            y_null = torch.tensor([args.num_classes] * current_batch_size, device=device)
            y_input = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y_input, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            z_input = z
            y_input = y
            model_kwargs = dict(y=y_input)
            model_fn = model.forward

        # Sample
        t0 = time.time()
        if args.sampling_method == "dpm_solver_1":
            samples = diffusion.dpm_solver_sample_loop(
                model_fn, z_input.shape, noise=z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device, order=1,
            )
        elif args.sampling_method == "dpm_solver_2":
            samples = diffusion.dpm_solver_sample_loop(
                model_fn, z_input.shape, noise=z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device, order=2,
            )
        else:
            # Default: DDPM
            samples = diffusion.p_sample_loop(
                model_fn, z_input.shape, z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device,
            )

        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # VAE decode
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        elapsed = time.time() - t0

        # Save images asynchronously
        def save_batch(imgs, indices):
            for img, idx in zip(imgs, indices):
                path = os.path.join(args.output_dir, f"{idx:06d}.png")
                Image.fromarray(img).save(path)

        io_futures.append(io_executor.submit(save_batch, samples, batch_indices))

        num_generated += current_batch_size
        print(f"  [Rank {rank}] {num_generated}/{total_to_generate} "
              f"({elapsed:.1f}s, {elapsed/current_batch_size:.2f}s/img)")

    # Wait for all I/O to complete
    for f in io_futures:
        f.result()

    print(f"\n[Rank {rank}] Done. Generated {num_generated} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, default="ema")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampling-method", type=str, default="ddpm",
                        choices=["ddpm", "dpm_solver_1", "dpm_solver_2"])
    parser.add_argument("--total-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
