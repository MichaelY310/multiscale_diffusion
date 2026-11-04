"""Long-lived, sharded RDD-Wan sampler for VBench-scale manifests.

Unlike the single-video CLI, this loads the RDD DiT, T5 and VAE once per GPU.
Every output receives a metrics sidecar with true model-forward counts, latency,
peak memory, prompt and seed.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR / "Wan2.1"))

from flow_matching_video import RDDVideoSchedule
from sample_rdd_wan import (
    parse_stage_configs_from_checkpoint,
    parse_stage_lengths_from_checkpoint,
    sample_rdd_video,
    save_video,
)


def load_manifest(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("prompt", "seed", "relative_path"):
                if key not in record:
                    raise ValueError(f"Manifest line {line_number} is missing {key!r}")
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--steps-per-stage", default="20,20,10")
    parser.add_argument("--solver", choices=["euler", "ab2", "heun"], default="ab2")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument(
        "--keep-text-encoder-resident",
        action="store_true",
        help="Keep umT5-XXL on GPU between prompts (H100/96GB only).",
    )
    parser.add_argument(
        "--offload-model-for-decode",
        action="store_true",
        help="Move the RDD DiT to CPU for each VAE decode on smaller GPUs.",
    )
    args = parser.parse_args()

    if not (0 <= args.shard_rank < args.num_shards):
        parser.error("--shard-rank must be in [0, --num-shards)")
    if not torch.cuda.is_available():
        raise RuntimeError("VBench sampling requires CUDA")

    from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_manifest(args.manifest)
    records = all_records[args.shard_rank :: args.num_shards]
    print(
        f"Shard {args.shard_rank}/{args.num_shards}: {len(records)}/{len(all_records)} videos"
    )

    print(f"Loading RDD checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage_configs = parse_stage_configs_from_checkpoint(ckpt)
    stage_lengths = parse_stage_lengths_from_checkpoint(ckpt, len(stage_configs))
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)
    schedule = RDDVideoSchedule(
        stage_configs,
        stage_lengths=stage_lengths,
        diffusion_steps=1000,
        downsample_mode=ckpt_args.get("downsample_mode", "latent"),
        drift_scale=float(ckpt_args.get("drift_scale", 1.0)),
    )
    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        out_dim=16,
        dim=ckpt_args.get("dim", 1536),
        ffn_dim=ckpt_args.get("ffn_dim", 8960),
        freq_dim=256,
        text_dim=4096,
        num_heads=ckpt_args.get("num_heads", 12),
        num_layers=ckpt_args.get("num_layers", 30),
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )
    weights = ckpt.get("ema", ckpt.get("model"))
    if weights is None:
        raise ValueError("RDD checkpoint has neither EMA nor model weights")
    model.load_state_dict(weights)
    del ckpt, weights
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    pretrained_dir = Path(args.pretrained_dir)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=str(pretrained_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(pretrained_dir / "google" / "umt5-xxl"),
    )
    vae = WanVAE(
        z_dim=16,
        vae_pth=str(pretrained_dir / "Wan2.1_VAE.pth"),
        dtype=torch.float32,
        device=device,
    )

    negative_prompt = args.negative_prompt or wan_config.sample_neg_prompt
    text_encoder.model.to(device)
    context_uncond = [
        value.to(dtype=torch.bfloat16)
        for value in text_encoder([negative_prompt], device)
    ]
    if not args.keep_text_encoder_resident:
        text_encoder.model.cpu()
        torch.cuda.empty_cache()

    width, height = [int(value) for value in args.size.split("x")]
    latent_shape = (
        1,
        16,
        (args.frame_num - 1) // 4 + 1,
        height // 8,
        width // 8,
    )
    steps = tuple(int(value) for value in args.steps_per_stage.split(","))

    completed = 0
    for record_index, record in enumerate(records, 1):
        output_path = output_dir / record["relative_path"]
        metric_path = metrics_dir / (output_path.name + ".json")
        if output_path.is_file() and metric_path.is_file():
            completed += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = record["prompt"]
        seed = int(record["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)
        sample_start = time.perf_counter()

        if not args.keep_text_encoder_resident:
            text_encoder.model.to(device)
        encode_start = time.perf_counter()
        context = [
            value.to(dtype=torch.bfloat16)
            for value in text_encoder([prompt], device)
        ]
        text_seconds = time.perf_counter() - encode_start
        if not args.keep_text_encoder_resident:
            text_encoder.model.cpu()
            torch.cuda.empty_cache()

        if args.offload_model_for_decode:
            model.to(device)
        denoise_start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            video_latent, sample_stats = sample_rdd_video(
                model=model,
                schedule=schedule,
                context=context,
                context_uncond=context_uncond,
                full_shape=latent_shape,
                steps_per_stage=steps,
                guidance_scale=args.guidance_scale,
                shift=args.shift,
                solver=args.solver,
                seed=seed,
                device=device,
                dtype=torch.bfloat16,
                return_stats=True,
            )
        torch.cuda.synchronize(device)
        denoise_seconds = time.perf_counter() - denoise_start

        if args.offload_model_for_decode:
            model.cpu()
            torch.cuda.empty_cache()
        decode_start = time.perf_counter()
        with torch.no_grad():
            video = vae.decode([video_latent[0].float()])[0]
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_start
        save_video(video, str(output_path), fps=args.fps)
        total_seconds = time.perf_counter() - sample_start

        metrics = {
            **sample_stats,
            "shard_rank": args.shard_rank,
            "shard_record_index": record_index,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "pretrained_dir": str(Path(args.pretrained_dir).resolve()),
            "prompt": prompt,
            "seed": seed,
            "sample_index": record.get("sample_index"),
            "dimensions": record.get("dimensions", []),
            "solver": args.solver,
            "steps_per_stage": list(steps),
            "stage_lengths": list(stage_lengths),
            "stage_configs": stage_configs,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "size": args.size,
            "frame_num": args.frame_num,
            "text_encode_seconds": text_seconds,
            "denoise_and_boundary_seconds": denoise_seconds,
            "vae_decode_seconds": decode_seconds,
            "end_to_end_inference_seconds": total_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "output": str(output_path.resolve()),
        }
        metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        completed += 1
        print(
            f"[{record_index}/{len(records)}] {output_path.name} | "
            f"denoise={denoise_seconds:.2f}s total={total_seconds:.2f}s"
        )

    print(f"Shard complete: {completed}/{len(records)} videos available")


if __name__ == "__main__":
    main()
