#!/usr/bin/env python3
"""Create configuration-explicit FID figures for Unified-L and XL models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def save_all(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def plot_unified_l(args: argparse.Namespace) -> None:
    rows = read_tsv(args.l_baseline) + read_tsv(args.l_boundaries) + read_tsv(args.l_two_stage)
    specs = (
        (
            "300300400",
            "Unified-L, 3-stage | train [32,16,8]=300/300/400 | "
            "sample [8,16,32]=80/80/90 | DDPM, 250 NFE, CFG 1.0, noise filling",
            "#0072B2",
            "o",
        ),
        (
            "100300600",
            "Unified-L, 3-stage | train [32,16,8]=100/300/600 | "
            "sample [8,16,32]=80/80/90 | DDPM, 250 NFE, CFG 1.0, noise filling",
            "#D55E00",
            "s",
        ),
        (
            "600300100",
            "Unified-L, 3-stage | train [32,16,8]=600/300/100 | "
            "sample [8,16,32]=80/80/90 | DDPM, 250 NFE, CFG 1.0, noise filling",
            "#CC79A7",
            "^",
        ),
        (
            "500500",
            "Unified-L, 2-stage (16→32) | train [32,16]=500/500 | "
            "sample [16,32]=125/125 | DDPM, 250 NFE, CFG 1.0, noise filling",
            "#009E73",
            "D",
        ),
    )

    fig, ax = plt.subplots(figsize=(11.6, 7.3))
    for model, label, color, marker in specs:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: int(row["step"]),
        )
        x = [int(row["step"]) / 1000 for row in selected]
        y = [float(row["fid"]) for row in selected]
        ax.plot(x, y, label=label, color=color, marker=marker, linewidth=2.3, markersize=5.8)
        ax.annotate(
            f"{y[-1]:.2f}",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )

    ax.set_title("Unified DiT-L/2: training-allocation FID convergence", fontsize=15)
    ax.set_xlabel("Optimizer updates (K)")
    ax.set_ylabel("FID-50K ↓")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.1, labelspacing=0.8)
    ax.text(
        0.01,
        0.02,
        "Native ImageNet-256; 50K generated samples; ADM evaluator. "
        "The running 400/300/300 and 8→32 500/500 jobs have no FID points yet.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#4B5563",
    )
    fig.tight_layout()
    save_all(fig, args.output_dir / "unified_l_fid_convergence_all_configs")


def plot_xl(args: argparse.Namespace) -> None:
    rows = read_tsv(args.all_original)
    specs = (
        (
            "RDD Unified-XL/2 (300/300/400)",
            "RDD Unified-XL diffusion | train [32,16,8]=300/300/400 | "
            "sample [8,16,32]=80/80/90 | DDPM, 250 NFE, CFG 1.0, noise filling",
            "#0072B2",
            "o",
        ),
        (
            "DiT-XL/2",
            "Original DiT-XL/2 | train [32]=1000 | sample [32]=250 | "
            "DDPM, 250 NFE, CFG 1.0",
            "#4B5563",
            "v",
        ),
    )

    fig, ax = plt.subplots(figsize=(11.6, 7.3))
    for model, label, color, marker in specs:
        selected = sorted(
            (
                row
                for row in rows
                if row["model"] == model and row.get("fid", "").strip()
            ),
            key=lambda row: int(row["step"]),
        )
        x = [int(row["step"]) / 1000 for row in selected]
        y = [float(row["fid"]) for row in selected]
        ax.plot(x, y, label=label, color=color, marker=marker, linewidth=2.3, markersize=5.8)
        ax.annotate(
            f"{y[-1]:.2f}",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )

    fm_rows = read_tsv(args.fm_metrics)
    selected_fm = [
        row for row in fm_rows if row["solver"] == "euler" and float(row["cfg"]) == 1.0
    ]
    if len(selected_fm) != 1:
        raise RuntimeError("Expected exactly one legacy FM Euler CFG=1.0 result")
    fm = selected_fm[0]
    fm_x = int(fm["step"]) / 1000
    fm_y = float(fm["fid"])
    ax.scatter(
        [fm_x],
        [fm_y],
        color="#E69F00",
        marker="*",
        s=190,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label=(
            "RDD-FM Unified-XL v2 (legacy; 400K only) | train coarse→fine "
            "[8,16,32]=300/300/400 | sample=80/80/90 | Euler, 250 updates/252 NFE, "
            "CFG 1.0, noise filling"
        ),
    )
    ax.annotate(
        f"FM {fm_y:.2f}",
        (fm_x, fm_y),
        xytext=(7, 8),
        textcoords="offset points",
        color="#9A5B00",
        fontsize=9,
    )

    ax.set_title("XL/2 models: FID convergence and available FM result", fontsize=15)
    ax.set_xlabel("Optimizer updates (K)")
    ax.set_ylabel("FID-50K ↓")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0, labelspacing=0.9)
    ax.text(
        0.01,
        0.02,
        "Native ImageNet-256; 50K generated samples; ADM evaluator. "
        "FM has only one valid 400K point, so no FM curve is interpolated.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#4B5563",
    )
    fig.tight_layout()
    save_all(fig, args.output_dir / "xl_fid_convergence_all_configs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-baseline", type=Path, required=True)
    parser.add_argument("--l-boundaries", type=Path, required=True)
    parser.add_argument("--l-two-stage", type=Path, required=True)
    parser.add_argument("--all-original", type=Path, required=True)
    parser.add_argument("--fm-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_unified_l(args)
    plot_xl(args)
    print(f"Wrote Unified-L and XL figures to {args.output_dir}")


if __name__ == "__main__":
    main()
