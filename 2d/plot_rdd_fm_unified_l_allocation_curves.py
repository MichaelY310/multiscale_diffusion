#!/usr/bin/env python3
"""Plot the two corrected Unified-L RDD-FM allocation convergence curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_curve(path: Path) -> tuple[list[float], list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 checkpoints in {path}, found {len(rows)}")
    rows.sort(key=lambda row: int(row["step"]))
    return (
        [int(row["step"]) / 1000.0 for row in rows],
        [float(row["fid"]) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-300300400", required=True)
    parser.add_argument("--summary-400300300", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    curves = [
        (
            "RDD-FM Unified-L train 300/300/400",
            "#2468b4",
            *read_curve(Path(args.summary_300300400)),
        ),
        (
            "RDD-FM Unified-L train 400/300/300",
            "#d05a3a",
            *read_curve(Path(args.summary_400300300)),
        ),
    ]

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for label, color, steps, fids in curves:
        ax.plot(steps, fids, marker="o", linewidth=2.2, markersize=5.5, color=color, label=label)
        best = min(range(len(fids)), key=fids.__getitem__)
        ax.annotate(
            f"{fids[best]:.2f}",
            (steps[best], fids[best]),
            xytext=(5, -14),
            textcoords="offset points",
            color=color,
        )

    ax.set_title("Corrected RDD-FM Unified-L convergence")
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("FID-50K ↓")
    ax.set_xticks(curves[0][2])
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()

