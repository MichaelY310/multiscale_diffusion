#!/usr/bin/env python3
"""CPU-only Laplacian-pyramid and FFT diagnostics for generated-image NPZs.

The ADM evaluator stores RGB uint8 images under ``arr_0``.  This script uses a
fixed subset from each NPZ, measures how much image variation lives in the
64->128->256 detail bands, and measures a radial Fourier spectrum.  It can also
write matched 10K low-pass NPZs for a diagnostic (not paper-standard) FID run.

Nothing in this file loads a diffusion model or touches a GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


METRIC_NAMES = (
    "low64_ac_energy",
    "mid_64_to_128_energy",
    "high_128_to_256_energy",
    "total_ac_energy",
    "low64_fraction",
    "mid_fraction",
    "high_fraction",
    "gradient_energy",
    "laplacian_filter_energy",
    "fft_low_fraction",
    "fft_mid_fraction",
    "fft_high_fraction",
)


def _display_label(label: str) -> str:
    aliases = {
        "ImageNet_original": "ImageNet real",
        "ImageNet_SD_VAE_mean_reconstruction": "ImageNet after SD-VAE",
        "RDD_100_300_600_cfg1": "100/300/600 orig · CFG 1.0",
        "RDD_100_300_600_cfg1p5": "100/300/600 orig · CFG 1.5",
        "RDD_100_300_600_8x8continued600K_cfg1": "8×8 continued · CFG 1.0",
        "RDD_100_300_600_8x8continued600K_cfg1p5": "8×8 continued · CFG 1.5",
    }
    return aliases.get(label, label)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_")
    return value or "dataset"


def _parse_dataset(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be LABEL=/path/to/file.npz")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip():
        raise argparse.ArgumentTypeError("dataset label cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"NPZ does not exist: {path}")
    return DatasetSpec(label.strip(), path)


def _block_project(images: np.ndarray, target: int) -> np.ndarray:
    """Average-pool to ``target`` then nearest-neighbor project back."""
    b, h, w, c = images.shape
    if h != w or h % target:
        raise ValueError(f"expected square resolution divisible by {target}, got {images.shape}")
    factor = h // target
    low = images.reshape(b, target, factor, target, factor, c).mean(axis=(2, 4))
    return np.repeat(np.repeat(low, factor, axis=1), factor, axis=2)


def _to_uint8(images: np.ndarray) -> np.ndarray:
    return np.clip(np.rint((images + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _visible_residual(residual: np.ndarray, gain: float = 4.0) -> np.ndarray:
    return np.clip(np.rint((0.5 + gain * residual / 2.0) * 255.0), 0, 255).astype(np.uint8)


def _make_montage(
    originals: np.ndarray,
    p64: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    output_path: Path,
) -> None:
    count = min(len(originals), 4)
    thumb = 192
    header = 32
    labels = ("original", "64px projection", "64->128 detail (x4)", "128->256 detail (x4)")
    canvas = Image.new("RGB", (4 * thumb, count * (thumb + header)), "white")
    draw = ImageDraw.Draw(canvas)
    columns = (
        _to_uint8(originals[:count]),
        _to_uint8(p64[:count]),
        _visible_residual(mid[:count]),
        _visible_residual(high[:count]),
    )
    for row in range(count):
        for col, (label, images) in enumerate(zip(labels, columns)):
            x = col * thumb
            y = row * (thumb + header)
            draw.text((x + 5, y + 8), label, fill="black")
            tile = Image.fromarray(images[row]).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x, y + header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _radial_fft_setup(height: int, width: int, bins: int):
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    edges = np.linspace(0.0, math.sqrt(2.0), bins + 1)
    bin_ids = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    order = np.argsort(bin_ids)
    sorted_ids = bin_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    occupied = sorted_ids[starts]
    counts = np.bincount(bin_ids, minlength=bins).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return radius, order, starts, occupied, counts, centers


def _summarize(values: np.ndarray) -> Mapping[str, float]:
    n = int(values.size)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(max(n, 1)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _select_indices(total: int, requested: int, seed: int) -> np.ndarray:
    n = min(total, requested)
    if n == total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # Sorted indices make disk access much less random while retaining an
    # unbiased deterministic subset.
    return np.sort(rng.choice(total, size=n, replace=False)).astype(np.int64)


def analyze_dataset(
    spec: DatasetSpec,
    output_dir: Path,
    max_images: int,
    batch_size: int,
    seed: int,
    fft_bins: int,
    write_lowpass_npz: bool,
) -> Tuple[dict, np.ndarray, Dict[str, np.ndarray]]:
    print(f"[load] {spec.label}: {spec.path}", flush=True)
    archive = np.load(spec.path, allow_pickle=False)
    if "arr_0" not in archive.files:
        raise KeyError(f"{spec.path} has keys {archive.files}, expected arr_0")
    images = archive["arr_0"]
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 NHWC RGB, got {images.shape} {images.dtype}")
    indices = _select_indices(len(images), max_images, seed)
    n = len(indices)
    height, width = int(images.shape[1]), int(images.shape[2])
    if (height, width) != (256, 256):
        raise ValueError(f"this RDD diagnostic expects 256x256 decoded images, got {height}x{width}")

    radius, fft_order, fft_starts, fft_occupied, fft_counts, fft_centers = _radial_fft_setup(
        height, width, fft_bins
    )
    radial_sum = np.zeros(fft_bins, dtype=np.float64)
    all_metrics: Dict[str, List[np.ndarray]] = {name: [] for name in METRIC_NAMES}
    lowpass64 = np.empty((n, height, width, 3), dtype=np.uint8) if write_lowpass_npz else None
    lowpass128 = np.empty((n, height, width, 3), dtype=np.uint8) if write_lowpass_npz else None
    montage_payload = None

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        raw = images[indices[start:end]]
        x = raw.astype(np.float32) / 127.5 - 1.0
        p128 = _block_project(x, 128)
        p64 = _block_project(x, 64)
        high = x - p128
        mid = p128 - p64
        channel_mean = x.mean(axis=(1, 2), keepdims=True)
        low_ac = p64 - channel_mean

        if montage_payload is None:
            montage_payload = (x[:4].copy(), p64[:4].copy(), mid[:4].copy(), high[:4].copy())
        if write_lowpass_npz:
            lowpass64[start:end] = _to_uint8(p64)
            lowpass128[start:end] = _to_uint8(p128)

        low_e = np.mean(low_ac * low_ac, axis=(1, 2, 3), dtype=np.float64)
        mid_e = np.mean(mid * mid, axis=(1, 2, 3), dtype=np.float64)
        high_e = np.mean(high * high, axis=(1, 2, 3), dtype=np.float64)
        total_e = low_e + mid_e + high_e
        denom = np.maximum(total_e, 1e-20)
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, 1:, :, :] - x[:, :-1, :, :]
        grad_e = 0.5 * (
            np.mean(dx * dx, axis=(1, 2, 3), dtype=np.float64)
            + np.mean(dy * dy, axis=(1, 2, 3), dtype=np.float64)
        )
        center = x[:, 1:-1, 1:-1, :]
        lap = (
            -4.0 * center
            + x[:, :-2, 1:-1, :]
            + x[:, 2:, 1:-1, :]
            + x[:, 1:-1, :-2, :]
            + x[:, 1:-1, 2:, :]
        )
        lap_e = np.mean(lap * lap, axis=(1, 2, 3), dtype=np.float64)

        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        gray -= gray.mean(axis=(1, 2), keepdims=True)
        fourier = np.fft.fft2(gray, axes=(1, 2))
        power = (fourier.real * fourier.real + fourier.imag * fourier.imag) / float((height * width) ** 2)
        flat_sorted = power.reshape(len(x), -1)[:, fft_order]
        occupied_sums = np.add.reduceat(flat_sorted, fft_starts, axis=1)
        binned = np.zeros((len(x), fft_bins), dtype=np.float64)
        binned[:, fft_occupied] = occupied_sums
        total_power = np.maximum(binned.sum(axis=1, keepdims=True), 1e-30)
        fractions = binned / total_power
        radial_sum += fractions.sum(axis=0)
        low_fft = fractions[:, fft_centers < 0.25].sum(axis=1)
        mid_fft = fractions[:, (fft_centers >= 0.25) & (fft_centers < 0.5)].sum(axis=1)
        high_fft = fractions[:, fft_centers >= 0.5].sum(axis=1)

        batch_values = {
            "low64_ac_energy": low_e,
            "mid_64_to_128_energy": mid_e,
            "high_128_to_256_energy": high_e,
            "total_ac_energy": total_e,
            "low64_fraction": low_e / denom,
            "mid_fraction": mid_e / denom,
            "high_fraction": high_e / denom,
            "gradient_energy": grad_e,
            "laplacian_filter_energy": lap_e,
            "fft_low_fraction": low_fft,
            "fft_mid_fraction": mid_fft,
            "fft_high_fraction": high_fft,
        }
        for name, values in batch_values.items():
            all_metrics[name].append(np.asarray(values, dtype=np.float64))
        print(f"  {end:>6}/{n}", flush=True)

    metric_arrays = {name: np.concatenate(parts) for name, parts in all_metrics.items()}
    summary = {
        "label": spec.label,
        "source_npz": str(spec.path),
        "source_images": int(len(images)),
        "analyzed_images": n,
        "selection_seed": seed,
        "selection": "all" if n == len(images) else "sorted_uniform_without_replacement",
        "pixel_scale": "[-1,1]",
        "metrics": {name: _summarize(values) for name, values in metric_arrays.items()},
    }
    radial_profile = radial_sum / float(n)

    slug = _slug(spec.label)
    with (output_dir / f"{slug}_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", *METRIC_NAMES])
        for row in range(n):
            writer.writerow([int(indices[row]), *[float(metric_arrays[name][row]) for name in METRIC_NAMES]])
    assert montage_payload is not None
    _make_montage(*montage_payload, output_dir / f"{slug}_laplacian_examples.png")

    lowpass_paths: Dict[str, np.ndarray] = {}
    if write_lowpass_npz:
        lowpass_dir = output_dir / "lowpass_npz"
        lowpass_dir.mkdir(parents=True, exist_ok=True)
        p64_path = lowpass_dir / f"{slug}_lowpass64_{n}.npz"
        p128_path = lowpass_dir / f"{slug}_lowpass128_{n}.npz"
        np.savez(p64_path, arr_0=lowpass64)
        np.savez(p128_path, arr_0=lowpass128)
        summary["lowpass_npz"] = {"64": str(p64_path), "128": str(p128_path)}
        del lowpass64, lowpass128

    del images, archive
    gc.collect()
    return summary, radial_profile, metric_arrays


def _write_plots(
    summaries: Sequence[dict],
    profiles: Mapping[str, np.ndarray],
    fft_centers: np.ndarray,
    output_dir: Path,
) -> None:
    # Imported lazily so the numerical CSV/JSON analysis remains usable in a
    # minimal environment.  The server setup installs matplotlib once.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["label"] for item in summaries]
    display_labels = [_display_label(label) for label in labels]
    x = np.arange(len(labels))
    width = 0.8 / 3.0
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 5.2))
    for offset, (key, name) in enumerate(
        (("low64_fraction", "base <=64px"), ("mid_fraction", "64->128 detail"), ("high_fraction", "128->256 detail"))
    ):
        values = [item["metrics"][key]["mean"] for item in summaries]
        ax.bar(x + (offset - 1) * width, values, width, label=name)
    ax.set_xticks(x, display_labels, rotation=10, ha="right")
    ax.set_ylabel("fraction of spatial AC energy")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "laplacian_band_fractions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label in labels:
        ax.plot(
            fft_centers,
            np.maximum(profiles[label], 1e-12),
            label=_display_label(label),
            linewidth=2,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, math.sqrt(2.0))
    ax.set_xlabel("radial spatial frequency / axis Nyquist")
    ax.set_ylabel("mean fraction of Fourier power per radial bin")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fft_radial_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 5.2))
    keys = ("fft_low_fraction", "fft_mid_fraction", "fft_high_fraction")
    names = ("low <0.25", "middle 0.25-0.5", "high >=0.5")
    for offset, (key, name) in enumerate(zip(keys, names)):
        values = [item["metrics"][key]["mean"] for item in summaries]
        ax.bar(x + (offset - 1) * width, values, width, label=name)
    ax.set_xticks(x, display_labels, rotation=10, ha="right")
    ax.set_ylabel("fraction of Fourier AC power")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fft_band_fractions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--fft-bins", type=int, default=64)
    parser.add_argument("--write-lowpass-npz", action="store_true")
    args = parser.parse_args()
    if args.max_images <= 0 or args.batch_size <= 0 or args.fft_bins < 8:
        parser.error("max-images/batch-size must be positive and fft-bins >= 8")
    labels = [spec.label for spec in args.dataset]
    if len(set(labels)) != len(labels):
        parser.error("dataset labels must be unique")
    if args.reference_label not in labels:
        parser.error("reference-label must match one --dataset label")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    profiles = {}
    for spec in args.dataset:
        summary, profile, _ = analyze_dataset(
            spec=spec,
            output_dir=output_dir,
            max_images=args.max_images,
            batch_size=args.batch_size,
            seed=args.seed,
            fft_bins=args.fft_bins,
            write_lowpass_npz=args.write_lowpass_npz,
        )
        summaries.append(summary)
        profiles[spec.label] = profile
        with (output_dir / "summary.partial.json").open("w", encoding="utf-8") as handle:
            json.dump({"datasets": summaries}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    _, _, _, _, _, fft_centers = _radial_fft_setup(256, 256, args.fft_bins)
    with (output_dir / "fft_radial_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency", *labels])
        for i, center in enumerate(fft_centers):
            writer.writerow([float(center), *[float(profiles[label][i]) for label in labels]])

    reference = next(item for item in summaries if item["label"] == args.reference_label)
    comparisons = []
    for item in summaries:
        if item is reference:
            continue
        for metric in METRIC_NAMES:
            ref_value = reference["metrics"][metric]["mean"]
            value = item["metrics"][metric]["mean"]
            comparisons.append(
                {
                    "dataset": item["label"],
                    "metric": metric,
                    "value": value,
                    "reference": ref_value,
                    "delta": value - ref_value,
                    "ratio": value / ref_value if abs(ref_value) > 1e-30 else None,
                }
            )
    with (output_dir / "comparison_vs_reference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "metric", "value", "reference", "delta", "ratio"))
        writer.writeheader()
        writer.writerows(comparisons)

    result = {
        "method": {
            "laplacian_pyramid": "P64, P128; bands P64-mean, P128-P64, image-P128",
            "fft": "per-image mean-removed luminance FFT; radial power normalized to sum one",
            "fft_band_edges_axis_nyquist": [0.25, 0.5],
            "important_note": "frequency diagnostics explain failure modes; they do not by themselves prove causal optimality of stage boundaries",
        },
        "reference_label": args.reference_label,
        "datasets": summaries,
        "comparisons": comparisons,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _write_plots(summaries, profiles, fft_centers, output_dir)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
