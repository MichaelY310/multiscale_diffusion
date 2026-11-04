#!/usr/bin/env python3
"""Plot all valid original-ImageNet convergence runs that reached >=400K."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "RDD Unified-L/2 (300/300/400)": "#D97706",
    "RDD Unified-XL/2 (300/300/400)": "#2563EB",
    "DiT-XL/2": "#64748B",
}
MARKERS = {
    "RDD Unified-L/2 (300/300/400)": "s",
    "RDD Unified-XL/2 (300/300/400)": "o",
    "DiT-XL/2": "^",
}


def load_rows(path: Path) -> dict[str, list[tuple[int, float | None]]]:
    series: dict[str, list[tuple[int, float | None]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            fid = float(row["fid"]) if row["fid"].strip() else None
            series[row["model"]].append((int(row["step"]), fid))
    for values in series.values():
        values.sort()
    return dict(series)


def contiguous_segments(values: list[tuple[int, float | None]]) -> list[list[tuple[int, float]]]:
    """Break the curve at missing evaluations instead of visually interpolating them."""
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for step, fid in values:
        if fid is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append((step, fid))
    if current:
        segments.append(current)
    return segments


def draw_panel(ax: plt.Axes, series: dict[str, list[tuple[int, float | None]]], *, zoom: bool) -> None:
    for model, values in series.items():
        color = COLORS[model]
        marker = MARKERS[model]
        measured = [(step / 1000, fid) for step, fid in values if fid is not None]
        for segment_index, segment in enumerate(contiguous_segments(values)):
            ax.plot(
                [step / 1000 for step, _ in segment],
                [fid for _, fid in segment],
                color=color,
                linewidth=2.25,
                marker=marker,
                markersize=5.2,
                markeredgewidth=0.8,
                markeredgecolor="white",
                label=model if segment_index == 0 else None,
                zorder=3,
            )
        # An isolated post-gap point still needs to be visible.
        ax.scatter(
            [step for step, _ in measured],
            [fid for _, fid in measured],
            color=color,
            marker=marker,
            s=29,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    if zoom:
        ax.set_xlim(140, 508)
        ax.set_xticks(range(150, 501, 50))
        ax.set_ylim(5.5, 37.5)
        ax.set_title("Late-stage convergence (150K–500K)")
        ax.text(
            0.985,
            0.965,
            "Unified-XL 450K is missing; no interpolation",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="#64748B",
        )
    else:
        ax.set_xlim(42, 508)
        ax.set_xticks(range(50, 501, 50))
        ax.set_ylim(0, 82)
        ax.set_title("Full convergence")
    ax.set_xlabel("Training step (K)")
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.75)
    ax.grid(axis="x", color="#E2E8F0", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    series = load_rows(args.data)
    missing = set(COLORS) - set(series)
    if missing:
        raise RuntimeError(f"missing expected series: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), constrained_layout=True)
    draw_panel(axes[0], series, zoom=False)
    draw_panel(axes[1], series, zoom=True)
    axes[0].set_ylabel("FID-50K ↓")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.075),
    )
    fig.suptitle("ImageNet-256 convergence under the same DDPM-250 / CFG 1.0 protocol", y=1.15, fontsize=13.5)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_prefix.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
