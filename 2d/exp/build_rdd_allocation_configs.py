#!/usr/bin/env python3
"""Convert equal-FLOPs allocations into sample_fid_unified JSONL configs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-json", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", default="0400000")
    parser.add_argument("--model-type", choices=["multistage", "decouple"], default="multistage")
    parser.add_argument("--sampler", choices=["ddpm", "dpm1", "dpm2"], default=None)
    parser.add_argument("--cfg-scales", default="1.5")
    parser.add_argument("--upsample-mode", default="noise_filling")
    parser.add_argument(
        "--samples-per-config", type=int, default=5000,
        help="Recorded in JSONL for auditing; the launcher must use the same TOTAL_SAMPLES.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", default=None, help="Optional comma-separated candidate labels")
    args = parser.parse_args()

    plan = json.loads(args.allocation_json.read_text(encoding="utf-8"))
    sampler = args.sampler or plan["solver"]
    wanted = set(parse_csv(args.labels)) if args.labels else None
    cfg_scales = [float(value) for value in parse_csv(args.cfg_scales)]
    resolutions = plan["resolutions"]
    stage_lengths = plan["stage_lengths"]
    candidates = plan["candidates"]

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        if wanted is not None and candidate["label"] not in wanted:
            continue
        for cfg_scale in cfg_scales:
            cfg_tag = str(cfg_scale).replace(".", "p")
            name = safe_name(
                f"{args.ckpt_dir.name}_{candidate['label']}_{candidate['num_steps']}_"
                f"{sampler}_cfg{cfg_tag}_{args.upsample_mode}"
            )
            row = {
                "model_type": args.model_type,
                "ckpt_dir": str(args.ckpt_dir),
                "ckpt_step": args.ckpt_step,
                "resolutions": ",".join(str(value) for value in resolutions),
                "stage_lengths": ",".join(str(value) for value in stage_lengths),
                "num_steps": candidate["num_steps"],
                "sampler": sampler,
                "cfg_scale": cfg_scale,
                "upsample_mode": args.upsample_mode,
                "output_dir": str(args.output_root / name),
                "allocation_label": candidate["label"],
                "budget_tflops_per_image": candidate["tflops_per_image"],
                "planned_samples": args.samples_per_config,
                "planned_seed": args.seed,
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No configurations selected")
    args.config_out.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} experiments to {args.config_out}")
    print("Run, for example:")
    print(
        f"CONFIG={args.config_out} TOTAL_SAMPLES={args.samples_per_config} SEED={args.seed} "
        "GPUS=0,1,2,3 "
        "bash 2d/exp/run_rdd_allocation_search.sh"
    )


if __name__ == "__main__":
    main()
