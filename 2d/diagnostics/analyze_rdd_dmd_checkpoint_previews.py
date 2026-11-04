#!/usr/bin/env python3
"""Quantify same-seed DMD preview sharpness and drift from teacher initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


TAGS = (
    "teacher_init_400k",
    "dmd_step005000_ema",
    "dmd_step050000_ema",
    "dmd_latest_ema",
    "dmd_latest_raw_student",
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def sharpness_metrics(image: np.ndarray) -> tuple[float, float, float]:
    gray = image @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(image.std()), float((dx + dy) / 2.0), float(laplacian.var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=None)
    args = parser.parse_args()

    teacher = [load_rgb(args.root / TAGS[0] / f"{index:06d}.png") for index in range(8)]
    print("tag\trgb_std\tmean_abs_gradient\tlaplacian_var\tpixel_mae_vs_teacher")
    for tag in TAGS:
        rows = []
        maes = []
        for index in range(8):
            image = load_rgb(args.root / tag / f"{index:06d}.png")
            rows.append(sharpness_metrics(image))
            maes.append(float(np.abs(image - teacher[index]).mean()))
        values = np.asarray(rows).mean(axis=0)
        print(
            f"{tag}\t{values[0]:.8f}\t{values[1]:.8f}\t{values[2]:.10f}"
            f"\t{np.mean(maes):.8f}"
        )
    if args.reference_dir is not None:
        rows = [
            sharpness_metrics(load_rgb(args.reference_dir / f"{index:06d}.png"))
            for index in range(8)
        ]
        values = np.asarray(rows).mean(axis=0)
        print(
            f"two_stage_reference\t{values[0]:.8f}\t{values[1]:.8f}"
            f"\t{values[2]:.10f}\tN/A"
        )


if __name__ == "__main__":
    main()
