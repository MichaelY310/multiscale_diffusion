#!/usr/bin/env python3
"""Plot paper-parity Unified-L stage-allocation convergence curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = (
    ("RDD 3-stage 300/300/400", "300300400", "#0072B2", "o"),
    ("RDD 3-stage 100/300/600", "100300600", "#D55E00", "s"),
    ("RDD 3-stage 600/300/100", "600300100", "#CC79A7", "^"),
    ("RDD 3-stage 400/300/300", "400300300", "#6A3D9A", "X"),
    ("RDD 2-stage 16→32 500/500", "500500", "#009E73", "D"),
    ("RDD 2-stage 8→32 500/500", "8to32_500500", "#E69F00", "P"),
)

METRICS = (
    ("fid", "FID ↓"),
    ("sfid", "sFID ↓"),
    ("inception_score", "Inception Score ↑"),
    ("precision", "Precision ↑"),
    ("recall", "Recall ↑"),
)

CONTEXT_SERIES = (
    (
        "RDD Unified-XL diffusion 300/300/400",
        "RDD Unified-XL/2 (300/300/400)",
        "#56B4E9",
        "o",
    ),
    ("Original DiT-XL/2", "DiT-XL/2", "#4B5563", "v"),
)


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "model": row["model"],
                    "step": float(row["step"]) / 1000.0,
                    "inception_score": float(row["inception_score"]),
                    "fid": float(row["fid"]),
                    "sfid": float(row["sfid"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                }
            )
    return rows


def read_original_context(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["model"] not in {item[1] for item in CONTEXT_SERIES}:
                continue
            if not row["fid"].strip():
                continue
            rows.append(
                {
                    "model": row["model"],
                    "step": float(row["step"]) / 1000.0,
                    "fid": float(row["fid"]),
                }
            )
    return rows


def read_fm_cfg1_point(path: Path) -> dict[str, float | str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [
        row
        for row in rows
        if row["solver"] == "euler" and float(row["cfg"]) == 1.0
    ]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one FM Euler CFG1.0 row in {path}")
    row = selected[0]
    return {
        "model": "rddfm_unixl_v2",
        "step": float(row["step"]) / 1000.0,
        "fid": float(row["fid"]),
        "nfe": float(row["actual_nfe"]),
    }


def plot_series(ax, rows, metric: str, annotate_last: bool = False) -> None:
    for label, model, color, marker in SERIES:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: row["step"],
        )
        if not selected:
            continue
        x = [float(row["step"]) for row in selected]
        y = [float(row[metric]) for row in selected]
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=2.1,
        )
        if annotate_last:
            ax.annotate(
                f"{y[-1]:.2f}",
                (x[-1], y[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=8.5,
                va="center",
            )
    ax.grid(True, alpha=0.24, linewidth=0.8)
    ax.set_xlabel("Training step (K)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-stage", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--two-stage", type=Path, required=True)
    parser.add_argument(
        "--direct-two-stage",
        type=Path,
        help="Optional direct 8x8-to-32x32 Unified-L 500/500 convergence TSV.",
    )
    parser.add_argument(
        "--four-hundred-first",
        type=Path,
        help="Optional Unified-L 400/300/300 convergence TSV.",
    )
    parser.add_argument(
        "--all-original",
        type=Path,
        help="Optional all-original convergence TSV containing DiT and Unified-XL.",
    )
    parser.add_argument(
        "--fm-metrics",
        type=Path,
        help="Optional RDD-FM 400K metrics TSV; only Euler CFG1.0 enters the curve figure.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required_paths = (args.three_stage, args.boundary, args.two_stage)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.direct_two_stage and not args.direct_two_stage.is_file():
        raise FileNotFoundError(args.direct_two_stage)
    if args.four_hundred_first and not args.four_hundred_first.is_file():
        raise FileNotFoundError(args.four_hundred_first)

    rows = (
        read_rows(args.three_stage)
        + read_rows(args.boundary)
        + read_rows(args.two_stage)
    )
    if args.direct_two_stage:
        rows += read_rows(args.direct_two_stage)
    if args.four_hundred_first:
        rows += read_rows(args.four_hundred_first)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 240,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    plot_series(ax, rows, "fid", annotate_last=True)
    ax.set_ylabel("FID ↓")
    ax.set_title("Unified DiT-L/2 on ImageNet-256: FID convergence")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "50K samples, ADM evaluator, DDPM 250 NFE, CFG 1.0, noise filling. "
        "Three-stage: 80/80/90; both two-stage samplers: 125/125 "
        "(equal NFE, unequal FLOPs).",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"unified_l_stage_allocation_fid_convergence.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    flat_axes = axes.ravel()
    for ax, (metric, ylabel) in zip(flat_axes, METRICS):
        plot_series(ax, rows, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace(" ↓", "").replace(" ↑", ""))
    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(handles, labels, frameon=False, loc="center", fontsize=10)
    fig.suptitle(
        "Unified DiT-L/2 stage-allocation convergence on ImageNet-256",
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.012,
        "50K samples, ADM evaluator, DDPM 250 NFE, CFG 1.0, noise filling. "
        "The 16→32 and direct 8→32 variants are labeled separately. "
        "Equal NFE does not imply equal FLOPs across different stage layouts.",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"unified_l_stage_allocation_all_metrics.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)

    print(f"Wrote comparison figures to {args.output_dir}")

    if bool(args.all_original) != bool(args.fm_metrics):
        raise ValueError("--all-original and --fm-metrics must be supplied together")
    if not args.all_original:
        return
    if not args.all_original.is_file() or not args.fm_metrics.is_file():
        raise FileNotFoundError((args.all_original, args.fm_metrics))

    context_rows = read_original_context(args.all_original)
    fm_point = read_fm_cfg1_point(args.fm_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.15), constrained_layout=True)
    plot_series(axes[0], rows, "fid", annotate_last=True)
    axes[0].set_ylabel("FID-50K ↓")
    axes[0].set_title("(a) Unified-L training-allocation ablation")
    axes[0].legend(frameon=False, fontsize=8.5)

    baseline_l = sorted(
        (row for row in rows if row["model"] == "300300400"),
        key=lambda row: row["step"],
    )
    axes[1].plot(
        [float(row["step"]) for row in baseline_l],
        [float(row["fid"]) for row in baseline_l],
        color="#0072B2",
        marker="s",
        linewidth=2.1,
        markersize=5.5,
        label="RDD Unified-L diffusion 300/300/400",
    )
    for label, model, color, marker in CONTEXT_SERIES:
        selected = sorted(
            (row for row in context_rows if row["model"] == model),
            key=lambda row: row["step"],
        )
        axes[1].plot(
            [float(row["step"]) for row in selected],
            [float(row["fid"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.5,
            label=label,
        )
    axes[1].scatter(
        [float(fm_point["step"])],
        [float(fm_point["fid"])],
        color="#E69F00",
        marker="*",
        s=175,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="RDD-FM Unified-XL v2 (400K only)",
    )
    axes[1].annotate(
        f"FM {float(fm_point['fid']):.2f}",
        (float(fm_point["step"]), float(fm_point["fid"])),
        xytext=(7, 8),
        textcoords="offset points",
        color="#A15C00",
        fontsize=8.7,
    )
    axes[1].set_xlabel("Training step (K)")
    axes[1].set_ylabel("FID-50K ↓")
    axes[1].set_title("(b) Architecture/formulation context")
    axes[1].grid(True, alpha=0.24, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8.3)

    fig.suptitle("ImageNet-256 convergence and the available RDD-FM operating point", fontsize=14)
    fig.text(
        0.5,
        -0.014,
        "Native ImageNet, 50K samples, CFG 1.0. Diffusion curves: DDPM 250 NFE. "
        "RDD-FM star: Euler 250 updates / 252 NFE; legacy coarse/middle/fine "
        "training intervals 300/300/400. FM has only a 400K evaluation, so no line is interpolated.",
        ha="center",
        va="top",
        fontsize=8.3,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"unified_l_stage_allocation_fid_with_dit_fm.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
