#!/usr/bin/env python3
"""Analyze RDD Unified stage-boundary mechanics and matched FID curves.

This diagnostic deliberately uses only the Python standard library.  It mirrors
the exact linear beta schedule and the conditional per-stage alpha products
created by ``SpacedDiffusionWithDrift``.  The output separates three effects
that are easy to conflate:

1. resolution handoff while a residual band is still recoverable;
2. drift contribution to the epsilon-training target;
3. coarsest-stage terminal mismatch when sampling starts from pure Gaussian.

The default residual energies come from the fixed ImageNet validation latent
projection used by the earlier teacher-forced diagnostic.  They are properties
of the data/VAE projection, not of the multistage model, so they also apply to
the Unified-L comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCHEDULES = {
    "100300600": (100, 300, 600),
    "300300400": (300, 300, 400),
    "600300100": (600, 300, 100),
}
RESOLUTIONS = (32, 16, 8)


def linear_beta_cumprod(steps: int = 1000) -> list[float]:
    beta_start = 0.0001 * 1000 / steps
    beta_end = 0.02 * 1000 / steps
    product = 1.0
    values = []
    for index in range(steps):
        fraction = index / (steps - 1)
        beta = beta_start + fraction * (beta_end - beta_start)
        product *= 1.0 - beta
        values.append(product)
    return values


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def profile_schedule(
    name: str,
    lengths: tuple[int, int, int],
    global_alpha_bar: list[float],
    high_residual_mse: float,
    middle_residual_mse: float,
    optimizer_steps: int,
    global_batch_size: int,
) -> list[dict]:
    rows = []
    start = 0
    residual_mse = (high_residual_mse, middle_residual_mse, None)
    for stage_index, (resolution, length) in enumerate(zip(RESOLUTIONS, lengths)):
        previous_alpha_bar = 1.0 if start == 0 else global_alpha_bar[start - 1]
        local_alpha_bar = [
            global_alpha_bar[start + local_t] / previous_alpha_bar
            for local_t in range(length)
        ]
        local_snr = [value / (1.0 - value) for value in local_alpha_bar]
        if stage_index < 2:
            target_multipliers = [
                (local_t / length) * math.sqrt(value / (1.0 - value))
                for local_t, value in enumerate(local_alpha_bar)
            ]
            multiplier_rms = math.sqrt(
                sum(value * value for value in target_multipliers) / length
            )
            target_energy_ratio = multiplier_rms**2 * residual_mse[stage_index]
            terminal_band_snr = local_snr[-1] * residual_mse[stage_index]
        else:
            target_multipliers = [0.0] * length
            multiplier_rms = 0.0
            target_energy_ratio = 0.0
            terminal_band_snr = None
        rows.append(
            {
                "model": name,
                "resolution": resolution,
                "global_interval": [start, start + length],
                "length": length,
                "expected_examples_per_local_t_at_checkpoint": (
                    optimizer_steps * global_batch_size / length
                ),
                "drift_lambda_slope_per_global_t": 0.0 if stage_index == 2 else 1.0 / length,
                "local_alpha_bar_start": local_alpha_bar[0],
                "local_alpha_bar_end": local_alpha_bar[-1],
                "local_snr_start": local_snr[0],
                "local_snr_end": local_snr[-1],
                "drift_target_multiplier_mean": sum(target_multipliers) / length,
                "drift_target_multiplier_rms": multiplier_rms,
                "drift_target_multiplier_p90": quantile(target_multipliers, 0.9),
                "drift_target_energy_over_gaussian": target_energy_ratio,
                "terminal_residual_band_snr": terminal_band_snr,
            }
        )
        start += length
    return rows


def read_metrics(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parsed = {"model": row["model"], "step": int(row["step"])}
            for key in ("inception_score", "fid", "sfid", "precision", "recall"):
                parsed[key] = float(row[key])
            grouped.setdefault(parsed["model"], []).append(parsed)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["step"])
    return grouped


def curve_summary(grouped: dict[str, list[dict]]) -> dict:
    common_steps = sorted(
        set.intersection(*({row["step"] for row in rows} for rows in grouped.values()))
    )
    results = {}
    for model, rows in grouped.items():
        selected = [row for row in rows if row["step"] in common_steps]
        area = 0.0
        for left, right in zip(selected, selected[1:]):
            area += (right["step"] - left["step"]) * (left["fid"] + right["fid"]) / 2
        width = selected[-1]["step"] - selected[0]["step"]
        results[model] = {
            "first": selected[0],
            "last": selected[-1],
            "mean_fid_over_common_interval": area / width,
            "fid_drop": selected[0]["fid"] - selected[-1]["fid"],
            "rows": selected,
        }
    baseline = results["300300400"]["last"]["fid"]
    for model, result in results.items():
        final_fid = result["last"]["fid"]
        result["final_fid_gap_vs_300300400"] = final_fid - baseline
        result["final_fid_relative_worse_percent"] = (final_fid / baseline - 1.0) * 100.0
    return {"common_steps": common_steps, "models": results}


def render_svg(grouped: dict[str, list[dict]], path: Path) -> None:
    width, height = 900, 560
    left, right, top, bottom = 88, 28, 54, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    all_rows = [row for rows in grouped.values() for row in rows]
    x_min, x_max = min(row["step"] for row in all_rows), max(row["step"] for row in all_rows)
    y_min = 0.0
    y_max = math.ceil(max(row["fid"] for row in all_rows) / 10) * 10
    colors = {"300300400": "#2563eb", "100300600": "#dc2626", "600300100": "#d97706"}

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{font-size:14px}.small{font-size:13px}.title{font-size:21px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}.frame{stroke:#374151;stroke-width:1.2;fill:none}</style>',
        '<text class="title" x="88" y="31">Unified DiT-L/2: stage-allocation FID convergence</text>',
        '<text class="small" x="88" y="49">DDPM-250 · steps [8,16,32]=80/80/90 · CFG 1.0 · noise filling · 50K samples</text>',
    ]
    for tick in range(0, int(y_max) + 1, 10):
        yy = y(tick)
        chunks.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{yy:.2f}" y2="{yy:.2f}"/>')
        chunks.append(f'<text class="axis" x="{left-12}" y="{yy+5:.2f}" text-anchor="end">{tick}</text>')
    for tick in sorted({row["step"] for row in all_rows}):
        xx = x(tick)
        chunks.append(f'<text class="axis" x="{xx:.2f}" y="{height-bottom+27}" text-anchor="middle">{tick//1000}K</text>')
    chunks.append(f'<rect class="frame" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>')
    chunks.append(f'<text class="axis" x="{left+plot_w/2}" y="{height-22}" text-anchor="middle">Training updates</text>')
    chunks.append(f'<text class="axis" transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle">FID (lower is better)</text>')
    legend_x = left + 420
    for index, model in enumerate(("300300400", "100300600", "600300100")):
        lx = legend_x + index * 140
        chunks.append(f'<line x1="{lx}" x2="{lx+25}" y1="29" y2="29" stroke="{colors[model]}" stroke-width="3"/>')
        chunks.append(f'<text class="small" x="{lx+31}" y="34">{model[:3]}/{model[3:6]}/{model[6:]}</text>')
        rows = grouped[model]
        points = " ".join(f'{x(row["step"]):.2f},{y(row["fid"]):.2f}' for row in rows)
        chunks.append(f'<polyline points="{points}" fill="none" stroke="{colors[model]}" stroke-width="3"/>')
        for row in rows:
            chunks.append(f'<circle cx="{x(row["step"]):.2f}" cy="{y(row["fid"]):.2f}" r="4.5" fill="white" stroke="{colors[model]}" stroke-width="2.5"/>')
        last = rows[-1]
        chunks.append(f'<text class="small" x="{x(last["step"])+8:.2f}" y="{y(last["fid"])-8:.2f}">{last["fid"]:.2f}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("eval_results/unified_l_boundary_analysis/metrics_summary.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/unified_l_boundary_analysis"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("paper/figures/unified_l_stage_allocation_fid.svg"),
    )
    parser.add_argument("--optimizer-steps", type=int, default=250_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--high-residual-mse", type=float, default=0.24137)
    parser.add_argument("--middle-residual-mse", type=float, default=0.33787 - 0.24137)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_alpha_bar = linear_beta_cumprod()
    mechanics = []
    for name, lengths in SCHEDULES.items():
        mechanics.extend(
            profile_schedule(
                name,
                lengths,
                global_alpha_bar,
                args.high_residual_mse,
                args.middle_residual_mse,
                args.optimizer_steps,
                args.global_batch_size,
            )
        )
    grouped = read_metrics(args.metrics)
    curves = curve_summary(grouped)
    payload = {
        "protocol": {
            "architecture": "Unified DiT-L/2",
            "sampler": "DDPM",
            "sampling_steps_coarse_middle_fine": [80, 80, 90],
            "cfg": 1.0,
            "noise_filling": True,
            "samples": 50_000,
            "optimizer_steps_for_mechanics": args.optimizer_steps,
            "global_batch_size": args.global_batch_size,
        },
        "residual_energy_source": {
            "high_16_to_32_mse": args.high_residual_mse,
            "middle_8_to_16_mse": args.middle_residual_mse,
            "note": "ImageNet validation SD-VAE latent block-projection residuals from the prior teacher-forced diagnostic",
        },
        "mechanics": mechanics,
        "curves": curves,
    }
    (args.output_dir / "unified_stage_allocation_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    render_svg(grouped, args.svg)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
