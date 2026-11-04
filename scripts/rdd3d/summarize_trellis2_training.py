#!/usr/bin/env python3
"""Summarize all TensorBoard scalars from one or more TRELLIS.2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(run: Path) -> dict:
    events = sorted((run / "tb_logs").glob("events.out.tfevents.*"))
    values: dict[str, list[tuple[int, float, float]]] = {}
    for event_file in events:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values.setdefault(tag, []).extend(
                (int(event.step), float(event.value), float(event.wall_time))
                for event in accumulator.Scalars(tag)
            )

    scalar_report = {}
    for tag, rows in sorted(values.items()):
        # A restarted run can contain the same step more than once.  Keep the
        # latest wall-time event for each step and make that choice auditable.
        by_step = {}
        for step, value, wall_time in rows:
            if step not in by_step or wall_time >= by_step[step][1]:
                by_step[step] = (value, wall_time)
        ordered = [(step, *by_step[step]) for step in sorted(by_step)]
        array = np.asarray([row[1] for row in ordered], dtype=np.float64)
        scalar_report[tag] = {
            "count": len(ordered),
            "first": {"step": ordered[0][0], "value": ordered[0][1]},
            "last": {"step": ordered[-1][0], "value": ordered[-1][1]},
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
            "tail": [{"step": step, "value": value} for step, value, _ in ordered[-10:]],
        }
    important = {}
    for tag in (
        "loss/loss",
        "loss/raw_global_velocity_mse",
        "loss/stage_16/weighted_mse",
        "loss/stage_32/weighted_mse",
        "loss/stage_64/weighted_mse",
        "loss/stage_16/tokens",
        "loss/stage_32/tokens",
        "loss/stage_64/tokens",
        "status/grad_norm",
        "time/step",
        "time/elapsed",
        "elastic/memory",
        "elastic/input_size",
    ):
        if tag in scalar_report:
            row = scalar_report[tag]
            important[tag] = {
                "first": row["first"],
                "last": row["last"],
                "mean": row["mean"],
                "min": row["min"],
                "max": row["max"],
            }
    return {
        "run": str(run),
        "event_files": [str(path) for path in events],
        "key_metrics": important,
        "scalars": scalar_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {run.name: summarize(run) for run in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: row["key_metrics"] for name, row in report.items()}, indent=2))


if __name__ == "__main__":
    main()
