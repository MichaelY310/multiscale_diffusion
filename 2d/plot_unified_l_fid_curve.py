#!/usr/bin/env python3
"""Plot the complete Unified-L convergence curve from the evaluator TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--title",
        default="RDD Unified-L 300/300/400 convergence",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with Path(args.summary).open("r", encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"no rows in {args.summary}")

    steps = [int(row["step"]) / 1000.0 for row in rows]
    fids = [float(row["fid"]) for row in rows]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(steps, fids, color="#2468b4", marker="o", linewidth=2.2, markersize=5.5)
    best = min(range(len(fids)), key=fids.__getitem__)
    ax.scatter([steps[best]], [fids[best]], color="#c43c39", zorder=3, s=52)
    ax.annotate(
        f"best {fids[best]:.2f}",
        (steps[best], fids[best]),
        xytext=(7, -15),
        textcoords="offset points",
        color="#9f2927",
    )
    ax.set_title(args.title)
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("FID-50K ↓")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.28, linewidth=0.8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
