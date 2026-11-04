#!/usr/bin/env python3
"""Create matched flat-FM and RDD-FM TRELLIS.2 fine-tune configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("rdd", "flat"), required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--ema-rate",
        type=float,
        default=0.9999,
        help="Fine-tune EMA decay; 0.999 is recommended for the 5K decision run.",
    )
    parser.add_argument("--min-aesthetic-score", type=float, default=-1.0)
    parser.add_argument(
        "--image-encoder",
        choices=("dinov3", "dinov2"),
        default="dinov3",
        help="Use dinov2 only for ungated pipeline smoke tests, not formal comparison.",
    )
    args = parser.parse_args()

    config = json.loads(args.upstream_config.read_text(encoding="utf-8"))
    fine_resolution = int(config["models"]["denoiser"]["args"]["resolution"])
    if fine_resolution < 4 or fine_resolution % 4:
        raise ValueError(
            f"denoiser resolution must support three dyadic stages, got {fine_resolution}"
        )
    rdd_resolutions = [fine_resolution // 4, fine_resolution // 2, fine_resolution]
    trainer = config["trainer"]
    trainer_args = trainer["args"]
    if args.mode == "rdd":
        trainer["name"] = "ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer"
        trainer_args.update(
            {
                "resolutions": rdd_resolutions,
                "time_fractions": [0.5, 0.3, 0.2],
                "stage_probabilities": [1 / 3, 1 / 3, 1 / 3],
                "normalize_global_velocity_loss": True,
            }
        )
    trainer_args.update(
        {
            "max_steps": args.max_steps,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "batch_split": 1,
            "finetune_ckpt": {"denoiser": str(args.checkpoint.resolve())},
            "i_log": 1 if args.max_steps <= 100 else 50,
            # Upstream snapshots use the flat sampler and are not valid for
            # the RDD path. Evaluation is performed by a paired evaluator.
            "i_sample": args.max_steps + 1,
            "i_save": max(args.max_steps // 2, 500),
            "ema_rate": [args.ema_rate],
        }
    )
    trainer_args["optimizer"]["args"]["lr"] = args.lr
    config["dataset"]["args"]["min_aesthetic_score"] = args.min_aesthetic_score
    if args.image_encoder == "dinov2":
        # DINOv2 ViT-L/14 has the same 1024 feature width as the official
        # DINOv3 ViT-L/16 condition encoder.  Its embedding distribution is not
        # equivalent, so this mode is deliberately limited to testing the data,
        # DDP, target, and optimizer path when the gated DINOv3 repo is unavailable.
        trainer_args["image_cond_model"] = {
            "name": "DinoV2FeatureExtractor",
            "args": {"model_name": "dinov2_vitl14"},
        }
        config["dataset"]["args"]["image_size"] = 518
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
