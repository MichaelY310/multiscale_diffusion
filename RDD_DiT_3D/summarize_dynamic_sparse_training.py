#!/usr/bin/env python3
"""Summarize recent dynamic-sparse training health by stage and replay mode."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "loss",
    "velocity_loss",
    "occupancy_loss",
    "occupancy_recall_0.5",
    "occupancy_precision_0.5",
    "grad_norm",
)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--last-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for line_number, line in enumerate(
        args.log.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {args.log}:{line_number}") from exc
    if not rows:
        raise SystemExit(f"no rows in {args.log}")
    last_step = max(int(row["step"]) for row in rows)
    first_step = max(min(int(row["step"]) for row in rows), last_step - args.last_steps + 1)
    selected = [row for row in rows if int(row["step"]) >= first_step]

    nonfinite = []
    groups: dict[tuple[int, bool], list[dict]] = defaultdict(list)
    for row in selected:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                nonfinite.append({"step": row["step"], "metric": metric, "value": value})
        groups[(int(row["stage"]), bool(row["generated_parent"]))].append(row)

    summaries = []
    for (stage, generated_parent), group in sorted(groups.items()):
        summary = {
            "stage": stage,
            "generated_parent": generated_parent,
            "rows": len(group),
            "tokens_mean": float(np.mean([row["tokens"] for row in group])),
        }
        for metric in METRICS:
            summary[metric] = stats([float(row[metric]) for row in group])
        summaries.append(summary)
    report = {
        "log": str(args.log),
        "step_range": [first_step, last_step],
        "rows": len(selected),
        "all_finite": not nonfinite,
        "nonfinite": nonfinite,
        "groups": summaries,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
