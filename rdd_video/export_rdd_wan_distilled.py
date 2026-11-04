"""Merge a distilled student LoRA into an ordinary RDD-Wan sampling checkpoint."""

from __future__ import annotations

import argparse
import os

import torch

from distill_rdd_wan import load_rdd_teacher
from lora_utils import inject_lora, load_lora_state_dict, merge_lora_


def main(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    distill_checkpoint = torch.load(args.distill_checkpoint, map_location="cpu", weights_only=False)
    config = dict(distill_checkpoint.get("config", {}))
    teacher_checkpoint = args.teacher_checkpoint or config.get("teacher_checkpoint")
    base_model_dir = args.base_model_dir or config.get("base_model_dir")
    if not base_model_dir or not teacher_checkpoint:
        raise ValueError("base model directory and original RDD teacher checkpoint are required")

    model = load_rdd_teacher(
        base_model_dir,
        teacher_checkpoint,
        device=device,
        dtype=torch.float32,
        allow_base_teacher=False,
    )
    inject_lora(
        model,
        rank=int(config.get("lora_rank", 128)),
        alpha=float(config.get("lora_alpha", 128.0)),
        dropout=0.0,
    )
    load_lora_state_dict(model, distill_checkpoint["student_lora"])
    merged = merge_lora_(model)
    stage_configs = config.get("stage_configs", "4,4,2;2,2,2;1,1,1")
    stage_lengths = config.get("stage_lengths", "60,350,590")
    payload = {
        "ema": {name: tensor.to(torch.bfloat16) for name, tensor in model.state_dict().items()},
        "args": {
            "stage_configs": stage_configs,
            "stage_lengths": stage_lengths,
            "downsample_mode": "latent",
            "drift_scale": float(config.get("drift_scale", 1.0)),
            "dim": int(model.dim),
            "ffn_dim": int(model.ffn_dim),
            "num_heads": int(model.num_heads),
            "num_layers": int(model.num_layers),
            "distilled_steps": config.get("student_steps", "2,2,1"),
            "distilled_boundary_clean": True,
            "teacher_cfg_scale": float(config.get("teacher_cfg_scale", 5.0)),
        },
        "distillation": {
            "source": os.path.abspath(args.distill_checkpoint),
            "step": int(distill_checkpoint.get("step", -1)),
            "format": distill_checkpoint.get("format"),
            "merged_lora_layers": merged,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    print(f"Exported merged RDD-Wan student: {args.output}")
    print("Sample with --steps-per-stage 2,2,1 --guidance-scale 1 --distilled-boundary-clean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-checkpoint", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
