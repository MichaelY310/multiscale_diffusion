#!/usr/bin/env python3
"""Run Pyramid Flow's released scheduler without loading a checkpoint.

This is provenance/configuration validation, not a comparable model benchmark:
the released miniFLUX/SD3 backbone differs from Wan2.1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scheduler_file = args.repo / "diffusion_schedulers" / "scheduling_flow_matching.py"
    spec = importlib.util.spec_from_file_location("pyramid_flow_official_scheduler", scheduler_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {scheduler_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.PyramidFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, stages=3, stage_range=[0, 1 / 3, 2 / 3, 1], gamma=1 / 3
    )
    stage_rows = []
    for stage in range(3):
        scheduler.set_timesteps(num_inference_steps=10, stage_index=stage, device="cpu")
        stage_rows.append({
            "stage": stage,
            "count": int(scheduler.timesteps.numel()),
            "first_timestep": float(scheduler.timesteps[0]),
            "last_timestep": float(scheduler.timesteps[-1]),
            "start_sigma": float(scheduler.start_sigmas[stage]),
            "end_sigma": float(scheduler.end_sigmas[stage]),
        })
    payload = {
        "source": str(scheduler_file),
        "class": "PyramidFlowMatchEulerDiscreteScheduler",
        "official_video_schedule": [10, 10, 10],
        "stage_range": [0, 1 / 3, 2 / 3, 1],
        "stages": stage_rows,
        "warning": "Scheduler-only no-checkpoint validation; not used as a raw latency comparison against Wan2.1.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

