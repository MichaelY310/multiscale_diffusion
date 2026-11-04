#!/usr/bin/env python3
"""Report the unintended scalar attenuation in the current exact RDD-DMD gradient."""

from __future__ import annotations

import numpy as np


def describe(values: np.ndarray) -> str:
    return (
        f"min={values.min():.6g}, median={np.median(values):.6g}, "
        f"mean={values.mean():.6g}, max={values.max():.6g}"
    )


def main() -> None:
    betas = np.linspace(1e-4, 2e-2, 1000, dtype=np.float64)
    global_alpha_bar = np.cumprod(1.0 - betas)
    stages = (
        ("32x32", 0, 300, True),
        ("16x16", 300, 600, True),
        ("8x8", 600, 1000, False),
    )
    print("Current exact code applies alpha_local^2 * M_t^2; alpha_local^2 is local alpha_bar.")
    for name, start, end, has_drift in stages:
        denominator = 1.0 if start == 0 else global_alpha_bar[start - 1]
        local_alpha_bar = global_alpha_bar[start:end] / denominator
        count = end - start
        low = max(0, min(count - 1, round(count * 0.02)))
        high = max(low + 1, min(count, round(count * 0.98)))
        selected = local_alpha_bar[low:high]
        print(f"{name} coarse/all-space attenuation: {describe(selected)}")
        if has_drift:
            global_t = np.arange(start + low, start + high, dtype=np.float64)
            lam = (global_t - start) / (end - start)
            residual = selected * (1.0 - lam) ** 2
            print(f"{name} new-detail attenuation:      {describe(residual)}")
        top = local_alpha_bar[-1]
        print(f"{name} one-call top SNR={top / (1.0 - top):.6g}")


if __name__ == "__main__":
    main()
