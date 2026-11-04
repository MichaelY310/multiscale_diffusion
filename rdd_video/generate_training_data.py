"""
Generate synthetic training data for RDD-Wan using Wan 2.1 T2V 1.3B inference.

This script generates videos from text prompts, saves both:
  - Pixel videos as .mp4 (for pixel-mode training)
  - VAE-encoded latents, the matching T5 context, and prompt in one .pt file
    (for conditional latent-mode training)

Follows PyramidalWan's approach of using synthetic data for training.

Usage:
    # Single GPU
    python generate_training_data.py \
        --pretrained-dir /path/to/Wan2.1-T2V-1.3B \
        --output-dir /path/to/output \
        --num-samples 1000

    # Multi-GPU (each GPU processes different prompts)
    CUDA_VISIBLE_DEVICES=0 python generate_training_data.py --device 0 --num-gpus 4 --gpu-rank 0 ...
    CUDA_VISIBLE_DEVICES=1 python generate_training_data.py --device 0 --num-gpus 4 --gpu-rank 1 ...
    CUDA_VISIBLE_DEVICES=2 python generate_training_data.py --device 0 --num-gpus 4 --gpu-rank 2 ...
    CUDA_VISIBLE_DEVICES=3 python generate_training_data.py --device 0 --num-gpus 4 --gpu-rank 3 ...
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import sys
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Wan2.1"))

# Default prompts for synthetic data generation (50 diverse prompts)
DEFAULT_PROMPTS = [
    "A golden retriever running through a field of wildflowers in slow motion, sunlight filtering through the petals.",
    "Timelapse of a bustling city intersection at night with streaks of car headlights and neon reflections.",
    "An astronaut floating in zero gravity inside a space station, reaching for a floating water droplet.",
    "Ocean waves crashing against rocky cliffs at sunset, with seagulls flying overhead.",
    "A dancer performing a graceful ballet routine on an empty stage with dramatic spotlight.",
    "Snow falling gently over a quiet Japanese garden with a stone lantern and koi pond.",
    "A chef preparing sushi in a traditional Japanese restaurant, carefully slicing fresh salmon.",
    "Aerial drone footage of a winding river cutting through autumn-colored forests.",
    "A mechanical clock being assembled piece by piece in stop-motion style.",
    "Lightning storm over a vast desert landscape at twilight with cactus silhouettes.",
    "A hummingbird hovering near a bright red flower, wings beating rapidly.",
    "Steam rising from a hot cup of coffee on a rainy window sill, droplets sliding down the glass.",
    "A group of children launching colorful kites on a windy beach.",
    "Northern lights dancing across a starry sky above a frozen lake in Iceland.",
    "An artist painting a large canvas with bold brushstrokes of vibrant colors.",
    "Underwater footage of a coral reef teeming with tropical fish and sea turtles.",
    "A vintage steam train chugging through a mountain pass with smoke billowing.",
    "Cherry blossom petals falling in slow motion along a peaceful walkway in Tokyo.",
    "A blacksmith forging a sword, sparks flying in a dimly lit workshop.",
    "Time-lapse of clouds rolling over mountain peaks at sunrise with golden light.",
    "A cat gracefully leaping from one rooftop to another in a European village.",
    "Fireworks exploding over a city skyline reflected in calm harbor waters.",
    "A surfer riding a massive wave in slow motion, water spraying dramatically.",
    "Raindrops creating ripples in a still pond surrounded by lily pads.",
    "A robot hand carefully picking up a delicate glass ornament.",
    "Hot air balloons rising over the valleys of Cappadocia at dawn.",
    "A pianist's hands moving rapidly over piano keys during a passionate concerto.",
    "Butterflies emerging from chrysalises in a greenhouse garden.",
    "A candle flame flickering in the dark, casting dancing shadows on the wall.",
    "Two wolves running through deep snow in a pine forest during winter.",
    "Lava flowing from a volcanic eruption meeting the ocean, creating steam clouds.",
    "A paper airplane being folded and launched from a skyscraper rooftop.",
    "Schools of fish forming mesmerizing swirling patterns in deep blue ocean.",
    "A woodpecker drumming on a tree trunk in a misty morning forest.",
    "Time-lapse of a flower blooming from bud to full bloom over several hours.",
    "A skateboarder performing tricks in an empty swimming pool at golden hour.",
    "Fog rolling through a dense redwood forest with shafts of sunlight breaking through.",
    "A street musician playing violin on a cobblestone street as people walk by.",
    "Ice crystals forming on a window pane in extreme cold, intricate patterns emerging.",
    "A falcon diving at high speed to catch prey, captured in slow motion.",
    "Ships passing through a canal lock system with water levels changing.",
    "A glass of red wine being poured in slow motion with light passing through.",
    "Wildfire spreading across dry grassland with thick smoke rising into the sky.",
    "A spider weaving an intricate web between branches, dew drops forming.",
    "Aerial view of rice terraces in Bali with farmers working in knee-deep water.",
    "A marble rolling through an elaborate Rube Goldberg machine.",
    "Northern sea otters floating on their backs, cracking shellfish with rocks.",
    "A time-lapse of an ice sculpture melting under warm studio lights.",
    "Bioluminescent plankton glowing blue in ocean waves crashing on a dark beach.",
    "A flock of starlings performing a murmuration at dusk, swirling across the sky.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using a Wan 2.1 T2V teacher"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        choices=["t2v-1.3B", "t2v-14B"],
        default="t2v-1.3B",
        help="Wan teacher architecture matching --pretrained-dir.",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        required=True,
        help="Path to the selected Wan 2.1 T2V checkpoint directory",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line). If not provided, uses built-in prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of videos to generate (default: 1000)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="832x448",
        help="Video resolution as WxH (default: 832x448, height divisible by 64 for PyramidalWan)",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=81,
        help="Number of frames per video (default: 81)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (Wan generates one at a time, default: 1)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device id (default: 0)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total number of GPUs for data-parallel generation (default: 1)",
    )
    parser.add_argument(
        "--gpu-rank",
        type=int,
        default=0,
        help="Rank of this GPU in data-parallel generation (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-pixel",
        action="store_true",
        default=True,
        help="Save pixel videos as .mp4 (default: True)",
    )
    parser.add_argument(
        "--save-latent",
        action="store_true",
        default=True,
        help="Save VAE-encoded latents as .pt (default: True)",
    )
    parser.add_argument(
        "--no-pixel",
        action="store_true",
        default=False,
        help="Skip saving pixel videos",
    )
    parser.add_argument(
        "--no-latent",
        action="store_true",
        default=False,
        help="Skip saving latents",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="FPS for saved .mp4 videos (default: 16)",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Teacher sampling steps (default: 50)",
    )
    parser.add_argument(
        "--sample-solver",
        type=str,
        choices=["unipc", "dpm++"],
        default="unipc",
        help="Official Wan teacher solver",
    )
    parser.add_argument(
        "--guide-scale",
        type=float,
        default=5.0,
        help="Teacher classifier-free guidance scale",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Wan inference sigma shift",
    )
    parser.add_argument(
        "--no-text-emb",
        action="store_true",
        help="Do not cache T5 context. Only use for unconditional debugging.",
    )
    parser.add_argument(
        "--keep-model-resident",
        action="store_true",
        help=(
            "Keep the Wan DiT on GPU between videos. Recommended on H100/96GB GPUs; "
            "otherwise the official pipeline offloads after every video."
        ),
    )
    return parser.parse_args()


def load_prompts(prompts_file, num_samples):
    """Load prompts from file or use default prompts, cycling to reach num_samples."""
    if prompts_file is not None:
        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} built-in default prompts")

    # Cycle prompts to reach num_samples
    if len(prompts) < num_samples:
        full_cycles = num_samples // len(prompts)
        remainder = num_samples % len(prompts)
        prompts = prompts * full_cycles + prompts[:remainder]
    else:
        prompts = prompts[:num_samples]

    return prompts


def save_video_mp4(video_tensor, output_path, fps=16):
    """
    Save video tensor as .mp4 file.

    Args:
        video_tensor: [C, T, H, W] in [-1, 1]
        output_path: path to save .mp4
        fps: frames per second
    """
    # Convert from [-1, 1] to [0, 255], shape [T, H, W, C]
    video = video_tensor.clone()
    video = (video + 1.0) / 2.0  # [-1,1] -> [0,1]
    video = video.clamp(0, 1)
    video = (video * 255).to(torch.uint8)
    # [C, T, H, W] -> [T, H, W, C]
    video = video.permute(1, 2, 3, 0).cpu()

    torchvision.io.write_video(output_path, video, fps=fps)


def get_completed_indices(output_dir, save_pixel, save_latent):
    """Check which samples have already been generated (for resume support)."""
    output_dir = Path(output_dir)

    def indices_for(directory, suffix):
        result = set()
        for path in directory.glob(f"sample_*.{suffix}"):
            try:
                result.add(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return result

    available = None
    if save_pixel:
        available = indices_for(output_dir / "pixel", "mp4")
    if save_latent:
        latent_indices = indices_for(output_dir / "latent", "pt")
        available = latent_indices if available is None else available & latent_indices
    return available or set()


@torch.no_grad()
def encode_prompt_context(pipe, prompt):
    """Encode exactly the conditional context expected by the official Wan DiT."""
    pipe.text_encoder.model.to(pipe.device)
    context = pipe.text_encoder([prompt], pipe.device)[0]
    context = context.detach().to(device="cpu", dtype=torch.bfloat16)
    pipe.text_encoder.model.cpu()
    return context


def main():
    args = parse_args()

    # Handle flags
    save_pixel = args.save_pixel and not args.no_pixel
    save_latent = args.save_latent and not args.no_latent

    if not save_pixel and not save_latent:
        raise ValueError("Must save at least one of pixel or latent (both disabled via flags)")

    # Parse size
    width, height = map(int, args.size.split("x"))

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    pixel_dir = output_dir / "pixel"
    latent_dir = output_dir / "latent"
    metadata_dir = output_dir / "metadata"

    if save_pixel:
        pixel_dir.mkdir(parents=True, exist_ok=True)
    if save_latent:
        latent_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    all_prompts = load_prompts(args.prompts_file, args.num_samples)

    # Multi-GPU: split prompts across GPUs
    indices = list(range(len(all_prompts)))
    if args.num_gpus > 1:
        indices = indices[args.gpu_rank :: args.num_gpus]
        if not indices:
            print(
                f"GPU {args.gpu_rank}/{args.num_gpus}: no assigned samples; exiting cleanly."
            )
            return
        print(
            f"GPU {args.gpu_rank}/{args.num_gpus}: processing {len(indices)} samples "
            f"(indices {indices[0]}..{indices[-1]})"
        )
    else:
        print(f"Single GPU: processing {len(indices)} samples")

    # Check for already completed samples (resume support)
    completed = get_completed_indices(str(output_dir), save_pixel, save_latent)
    remaining_indices = [i for i in indices if i not in completed]
    if len(completed) > 0:
        print(f"Resuming: {len(completed)} samples already completed, {len(remaining_indices)} remaining")

    if len(remaining_indices) == 0:
        print("All samples already generated. Nothing to do.")
        return

    # Load Wan 2.1 T2V pipeline
    print(
        f"Loading Wan 2.1 {args.teacher_model} from {args.pretrained_dir} "
        f"on device {args.device}..."
    )
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    config = WAN_CONFIGS[args.teacher_model]
    pipe = WanT2V(config, checkpoint_dir=args.pretrained_dir, device_id=args.device)
    print("Pipeline loaded successfully.")

    # Generate videos
    total = len(remaining_indices)
    start_time = time.time()

    for count, idx in enumerate(remaining_indices):
        prompt = all_prompts[idx]
        sample_name = f"sample_{idx:06d}"

        print(f"\n[{count + 1}/{total}] Generating: {sample_name}")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        # Generate video
        gen_start = time.time()
        video = pipe.generate(
            prompt,
            size=(width, height),
            frame_num=args.frame_num,
            shift=args.shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            seed=args.seed + idx,
            offload_model=not args.keep_model_resident,
        )
        # video: [C, T, H, W] in [-1, 1]
        gen_time = time.time() - gen_start
        print(f"  Generation time: {gen_time:.1f}s")

        # Save pixel video as .mp4
        if save_pixel:
            pixel_path = str(pixel_dir / f"{sample_name}.mp4")
            save_video_mp4(video, pixel_path, fps=args.fps)
            print(f"  Saved pixel: {pixel_path}")

        # Encode to latent and save
        if save_latent:
            with torch.no_grad():
                latent = pipe.vae.encode([video])  # returns list of [C_z, T_z, H_z, W_z]
                latent_tensor = latent[0].cpu()
                text_emb = None if args.no_text_emb else encode_prompt_context(pipe, prompt)
            latent_path = str(latent_dir / f"{sample_name}.pt")
            latent_record = {
                "latent": latent_tensor,
                "prompt": prompt,
                "teacher_seed": args.seed + idx,
            }
            if text_emb is not None:
                latent_record["text_emb"] = text_emb
            torch.save(latent_record, latent_path)
            print(f"  Saved latent: {latent_path} (shape: {tuple(latent_tensor.shape)})")

        # Save metadata
        meta = {
            "index": idx,
            "prompt": prompt,
            "teacher_model": args.teacher_model,
            "pretrained_dir": str(Path(args.pretrained_dir).resolve()),
            "size": [width, height],
            "frame_num": args.frame_num,
            "generation_time": gen_time,
            "teacher_seed": args.seed + idx,
            "sampling_steps": args.sampling_steps,
            "sample_solver": args.sample_solver,
            "guide_scale": args.guide_scale,
            "shift": args.shift,
        }
        if save_latent:
            meta["latent_shape"] = list(latent_tensor.shape)
        meta_path = str(metadata_dir / f"{sample_name}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Progress tracking
        elapsed = time.time() - start_time
        avg_time = elapsed / (count + 1)
        eta = avg_time * (total - count - 1)
        print(
            f"  Progress: {count + 1}/{total} | "
            f"Elapsed: {elapsed / 60:.1f}min | "
            f"ETA: {eta / 60:.1f}min | "
            f"Avg: {avg_time:.1f}s/sample"
        )

    # Save generation summary
    summary = {
        "total_samples": total,
        "output_dir": str(output_dir),
        "pretrained_dir": args.pretrained_dir,
        "teacher_model": args.teacher_model,
        "size": [width, height],
        "frame_num": args.frame_num,
        "fps": args.fps,
        "seed": args.seed,
        "gpu_rank": args.gpu_rank,
        "num_gpus": args.num_gpus,
        "save_pixel": save_pixel,
        "save_latent": save_latent,
        "save_text_emb": save_latent and not args.no_text_emb,
        "sampling_steps": args.sampling_steps,
        "sample_solver": args.sample_solver,
        "keep_model_resident": args.keep_model_resident,
        "guide_scale": args.guide_scale,
        "shift": args.shift,
        "total_time_seconds": time.time() - start_time,
    }
    summary_path = str(output_dir / f"generation_summary_gpu{args.gpu_rank}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Total samples: {total}")
    print(f"  Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Output directory: {output_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
