#!/usr/bin/env python3
"""Quantify and plot cross-stage block-reuse sensitivity by layer depth."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator


def rank(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0] * len(values)
    for position, index in enumerate(order):
        result[index] = position
    return result


def case_curves(root: Path, name_template: str) -> list[list[float]]:
    curves = []
    for path in sorted(root.glob("case_*/paired_suite_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = payload["baseline_name"]
        curves.append(
            [
                float(
                    payload["latent_comparisons"][
                        f"{name_template.format(layer=layer)}_vs_{baseline}"
                    ]["relative_rmse"]
                )
                for layer in range(30)
            ]
        )
    if not curves:
        raise RuntimeError(f"no case metrics below {root}")
    return curves


def summarize(curves: list[list[float]]) -> dict:
    mean = [statistics.mean(case[layer] for case in curves) for layer in range(30)]
    pairwise_spearman = []
    pairwise_top8_overlap = []
    for first, second in itertools.combinations(curves, 2):
        pairwise_spearman.append(pearson(rank(first), rank(second)))
        top_first = set(sorted(range(30), key=first.__getitem__)[:8])
        top_second = set(sorted(range(30), key=second.__getitem__)[:8])
        pairwise_top8_overlap.append(len(top_first & top_second) / 8)
    return {
        "cases": len(curves),
        "rrmse_by_layer": mean,
        "pearson_layer_vs_rrmse": pearson(list(range(30)), mean),
        "pairwise_spearman_min": min(pairwise_spearman),
        "pairwise_spearman_mean": statistics.mean(pairwise_spearman),
        "pairwise_top8_overlap_min": min(pairwise_top8_overlap),
        "pairwise_top8_overlap_mean": statistics.mean(pairwise_top8_overlap),
        "safe8": sorted(range(30), key=mean.__getitem__)[:8],
    }


def svg_chart(fine: dict, middle: dict) -> str:
    width, height = 980, 560
    left, right, top, bottom = 82, 34, 52, 78
    chart_w, chart_h = width - left - right, height - top - bottom
    y_max = 0.9

    def x_pos(layer: int) -> float:
        return left + chart_w * layer / 29

    def y_pos(value: float) -> float:
        return top + chart_h * (1 - value / y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{x_pos(21.5):.1f}" y="{top}" width="{x_pos(29)-x_pos(21.5):.1f}" height="{chart_h}" fill="#e8f5e9"/>',
        '<text x="700" y="72" font-size="15" fill="#2e7d32">稳定尾层区（22–29）</text>',
        '<text x="490" y="28" text-anchor="middle" font-size="22" font-weight="700">CSBR 单层替换的最终 latent 误差</text>',
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = y_pos(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="14">{tick:.1f}</text>')
    for tick in [0, 5, 10, 15, 20, 25, 29]:
        x = x_pos(tick)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+chart_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-size="14">{tick}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>',
            f'<text x="{left+chart_w/2:.1f}" y="{height-24}" text-anchor="middle" font-size="16">Transformer block 层号</text>',
            f'<text x="22" y="{top+chart_h/2:.1f}" transform="rotate(-90 22 {top+chart_h/2:.1f})" text-anchor="middle" font-size="16">最终 latent relative RMSE（越低越好）</text>',
        ]
    )
    for data, color, label in [
        (fine, "#1565c0", "middle→fine"),
        (middle, "#ef6c00", "coarse→middle"),
    ]:
        points = " ".join(
            f"{x_pos(layer):.1f},{y_pos(value):.1f}"
            for layer, value in enumerate(data["rrmse_by_layer"])
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for layer, value in enumerate(data["rrmse_by_layer"]):
            lines.append(f'<circle cx="{x_pos(layer):.1f}" cy="{y_pos(value):.1f}" r="3" fill="{color}"/>')
        legend_y = 102 if label == "middle→fine" else 130
        lines.append(f'<line x1="105" y1="{legend_y}" x2="145" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="155" y="{legend_y+5}" font-size="15">{label}，r={data["pearson_layer_vs_rrmse"]:.3f}</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-root", type=Path, required=True)
    parser.add_argument("--middle-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    fine = summarize(case_curves(args.fine_root, "single_l{layer:02d}_multiscale"))
    middle = summarize(
        case_curves(args.middle_root, "single_s1_l{layer:02d}_scalar075")
    )
    payload = {"fine_stage": fine, "middle_stage": middle}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_svg.write_text(svg_chart(fine, middle), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
