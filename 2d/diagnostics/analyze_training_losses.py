#!/usr/bin/env python3
"""Parse RDD training logs and summarize per-stage convergence.

Absolute losses are not compared across allocations because each allocation
assigns a different global-timestep interval to a stage.  The useful signal is
the within-run trend: did a stage keep improving near the checkpoint, and did
the 8x8-only continuation move beyond the original 400K plateau?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\(step=(\d+)\)")
LOSS = re.compile(r"Loss\[(\d+)x\1\]=([-+0-9.eE]+)")


def dataset_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=/path/to/log")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing log: {path}")
    return label, path


def parse_log(label: str, path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI.sub("", raw_line)
            step_match = STEP.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            for resolution, loss in LOSS.findall(line):
                records.append(
                    {
                        "run": label,
                        "source_log": str(path),
                        "step": step,
                        "resolution": int(resolution),
                        "loss": float(loss),
                    }
                )
    if not records:
        raise ValueError(f"no loss records parsed from {path}")
    return records


def window_stats(rows: list[dict], window_steps: int) -> dict:
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    final_step = int(steps.max())
    current = losses[steps > final_step - window_steps]
    previous = losses[(steps > final_step - 2 * window_steps) & (steps <= final_step - window_steps)]
    recent_mask = steps > final_step - 2 * window_steps
    recent_steps = steps[recent_mask]
    recent_losses = losses[recent_mask]
    slope_per_100k = None
    if len(recent_steps) >= 2 and np.ptp(recent_steps) > 0:
        slope_per_100k = float(np.polyfit(recent_steps, recent_losses, 1)[0] * 100_000)
    return {
        "records": len(rows),
        "first_step": int(steps.min()),
        "final_step": final_step,
        "final_logged_loss": float(losses[steps.argmax()]),
        "last_window_steps": window_steps,
        "last_window_mean": float(current.mean()),
        "last_window_std": float(current.std(ddof=1)) if len(current) > 1 else 0.0,
        "previous_window_mean": float(previous.mean()) if len(previous) else None,
        "last_minus_previous": float(current.mean() - previous.mean()) if len(previous) else None,
        "relative_last_minus_previous": (
            float((current.mean() - previous.mean()) / previous.mean())
            if len(previous) and previous.mean() != 0
            else None
        ),
        "linear_slope_over_last_2windows_per_100k_steps": slope_per_100k,
    }


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=dataset_arg, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-steps", type=int, default=10_000)
    parser.add_argument("--rolling-records", type=int, default=25)
    args = parser.parse_args()
    if args.window_steps <= 0 or args.rolling_records <= 0:
        parser.error("window sizes must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, path in args.input:
        records.extend(parse_log(label, path))
    records.sort(key=lambda row: (row["run"], row["resolution"], row["step"]))

    with (output_dir / "training_loss_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in records:
        groups.setdefault((row["run"], row["resolution"]), []).append(row)
    summaries = []
    for (run, resolution), rows in groups.items():
        summaries.append(
            {
                "run": run,
                "resolution": resolution,
                **window_stats(rows, args.window_steps),
            }
        )
    payload = {
        "status": "training_loss_convergence_diagnostic_not_cross_schedule_loss_comparison",
        "warning": "absolute stage losses are not comparable across allocations because their global timestep intervals differ",
        "window_steps": args.window_steps,
        "rolling_records": args.rolling_records,
        "inputs": [{"label": label, "path": str(path)} for label, path in args.input],
        "summaries": summaries,
    }
    (output_dir / "training_loss_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    run_order = list(dict.fromkeys(label for label, _ in args.input))
    fig, axes = plt.subplots(len(run_order), 1, figsize=(11, 3.6 * len(run_order)), squeeze=False)
    colors = {8: "#0072B2", 16: "#D55E00", 32: "#009E73"}
    for axis, run in zip(axes[:, 0], run_order):
        for resolution in (8, 16, 32):
            rows = groups.get((run, resolution), [])
            if not rows:
                continue
            steps = np.asarray([row["step"] for row in rows])
            losses = np.asarray([row["loss"] for row in rows])
            width = min(args.rolling_records, len(rows))
            smooth = rolling_mean(losses, width)
            smooth_steps = steps[width - 1 :]
            axis.plot(smooth_steps / 1000.0, smooth, label=f"{resolution}x{resolution}", color=colors[resolution], linewidth=1.8)
        axis.set_title(run)
        axis.set_xlabel("training step (K)")
        axis.set_ylabel(f"loss ({args.rolling_records}-record moving mean)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("RDD training-loss convergence (compare trends, not raw values across schedules)", y=1.005)
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_convergence.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
