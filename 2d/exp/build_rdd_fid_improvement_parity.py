#!/usr/bin/env python3
"""Build JSONL sampling parity configs for backward-compatible RDD improvements.

Example:
    python 2d/exp/build_rdd_fid_improvement_parity.py \
      --ckpt checkpoints/RDD-Unified-XL/0400000.pt \
      --guide-ckpt checkpoints/RDD-Unified-L/0100000.pt \
      --output-root eval_results/rdd_improvement_screen \
      --config-out 2d/exp/rdd_improvement_screen.jsonl

Use 10k samples for screening, then rebuild a small winner-only JSONL and run
50k samples for paper numbers.  Never compare FID values computed with
different sample counts as if they were directly interchangeable.
"""

import argparse
import json
from pathlib import Path


def entry(args, name, **overrides):
    config = {
        "model_type": "unified",
        "model": args.model,
        "ckpt": str(Path(args.ckpt).resolve()),
        "resolutions": args.resolutions,
        "stage_lengths": args.stage_lengths,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "cfg_scale": args.cfg_scale,
        "upsample_mode": "fresh",
        "output_dir": str((Path(args.output_root) / name).resolve()),
    }
    config.update(overrides)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--guide-ckpt", default=None)
    parser.add_argument("--model", default="DiT-XL/2")
    parser.add_argument("--guide-model", default=None)
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--num-steps", default="80,80,90")
    parser.add_argument("--sampler", choices=["ddpm", "dpm1", "dpm2"], default="dpm2")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--suite",
        choices=["full", "smoke"],
        default="full",
        help="Full paper screening matrix or a minimal legacy/LIG/Autoguidance smoke set.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-out", required=True)
    args = parser.parse_args()

    rows = [
        entry(args, "00_legacy_cfg"),
        entry(args, "01_lig_005_080", guidance_low=0.05, guidance_high=0.80),
        entry(args, "02_lig_010_070", guidance_low=0.10, guidance_high=0.70),
        entry(args, "03_lig_020_080", guidance_low=0.20, guidance_high=0.80),
        entry(
            args,
            "04_lig_cosine_010_080",
            guidance_low=0.10,
            guidance_high=0.80,
            guidance_schedule="cosine",
            guidance_min_scale=1.0,
        ),
    ]

    if args.guide_ckpt:
        guide_model = args.guide_model or args.model
        for scale in (1.5, 2.0, 2.5):
            rows.append(
                entry(
                    args,
                    f"autoguidance_{scale:.1f}".replace(".", "p"),
                    cfg_scale=1.0,
                    autoguidance_ckpt=str(Path(args.guide_ckpt).resolve()),
                    autoguidance_model=guide_model,
                    autoguidance_scale=scale,
                )
            )
        rows.append(
            entry(
                args,
                "autoguidance_2p0_interval_010_080",
                cfg_scale=1.0,
                autoguidance_ckpt=str(Path(args.guide_ckpt).resolve()),
                autoguidance_model=guide_model,
                autoguidance_scale=2.0,
                autoguidance_low=0.10,
                autoguidance_high=0.80,
            )
        )

    if args.suite == "smoke":
        wanted = {"00_legacy_cfg", "02_lig_010_070", "autoguidance_2p0"}
        rows = [row for row in rows if Path(row["output_dir"]).name in wanted]

    output = Path(args.config_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} parity experiments to {output}")


if __name__ == "__main__":
    main()
