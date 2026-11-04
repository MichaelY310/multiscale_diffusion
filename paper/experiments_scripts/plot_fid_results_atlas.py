#!/usr/bin/env python3
"""Build an organized, dependency-free SVG atlas of all paper-valid FID results.

The script intentionally keeps old-HF-data results in a separate, grey,
hypothesis-only panel.  Training stage lengths are displayed fine->coarse
(32,16,8), while sampling allocations are normalized to the same order before
plotting, even though sample_fid_unified.py stores them coarse->fine.
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_results" / "fid_results_atlas"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "pink": "#db2777",
    "slate": "#475569",
    "gray": "#9ca3af",
    "black": "#111827",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f(value: str) -> float:
    return float(value)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str, subtitle: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.10"/></filter>',
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="#f1f5f9"/>',
            f'<text x="60" y="62" font-family="Segoe UI,Arial,sans-serif" font-size="34" font-weight="700" fill="#0f172a">{esc(title)}</text>',
            f'<text x="60" y="94" font-family="Segoe UI,Arial,sans-serif" font-size="17" fill="#475569">{esc(subtitle)}</text>',
        ]

    def add(self, text: str) -> None:
        self.parts.append(text)

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def panel_frame(svg: SVG, x: float, y: float, w: float, h: float, title: str, subtitle: str = "") -> tuple[float, float, float, float]:
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="white" filter="url(#shadow)"/>')
    svg.add(f'<text x="{x + 24}" y="{y + 36}" font-family="Segoe UI,Arial,sans-serif" font-size="22" font-weight="700" fill="#0f172a">{esc(title)}</text>')
    if subtitle:
        svg.add(f'<text x="{x + 24}" y="{y + 61}" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#64748b">{esc(subtitle)}</text>')
    top = y + (86 if subtitle else 66)
    return x + 78, top, w - 104, h - (top - y) - 68


def scale_fn(lo: float, hi: float, out_lo: float, out_hi: float, log: bool = False):
    if log:
        lo_t, hi_t = math.log(lo), math.log(hi)
        return lambda value: out_lo + (math.log(value) - lo_t) / (hi_t - lo_t) * (out_hi - out_lo)
    return lambda value: out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def line_panel(
    svg: SVG,
    box: tuple[float, float, float, float],
    title: str,
    subtitle: str,
    series: list[dict],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: Iterable[float],
    yticks: Iterable[float],
    xlabel: str,
    ylabel: str = "FID (lower is better)",
    xlog: bool = False,
    ylog: bool = False,
    xfmt=lambda value: str(value),
    yfmt=lambda value: str(value),
    legend_cols: int = 1,
):
    x, y, w, h = box
    px, py, pw, ph = panel_frame(svg, x, y, w, h, title, subtitle)
    sx = scale_fn(xlim[0], xlim[1], px, px + pw, xlog)
    sy = scale_fn(ylim[0], ylim[1], py + ph, py, ylog)

    for tick in yticks:
        yy = sy(tick)
        svg.add(f'<line x1="{px}" y1="{yy:.2f}" x2="{px + pw}" y2="{yy:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.add(f'<text x="{px - 12}" y="{yy + 5:.2f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#64748b">{esc(yfmt(tick))}</text>')
    for tick in xticks:
        xx = sx(tick)
        svg.add(f'<line x1="{xx:.2f}" y1="{py}" x2="{xx:.2f}" y2="{py + ph}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.add(f'<text x="{xx:.2f}" y="{py + ph + 24}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#64748b">{esc(xfmt(tick))}</text>')
    svg.add(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#94a3b8" stroke-width="1.5"/>')
    svg.add(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#94a3b8" stroke-width="1.5"/>')

    for item in series:
        valid = [(xx, yy) for xx, yy in item["points"] if xx is not None and yy is not None]
        points = " ".join(f"{sx(xx):.2f},{sy(yy):.2f}" for xx, yy in valid)
        dash = f' stroke-dasharray="{item.get("dash")}"' if item.get("dash") else ""
        svg.add(f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        for xx, yy in valid:
            svg.add(f'<circle cx="{sx(xx):.2f}" cy="{sy(yy):.2f}" r="5.2" fill="white" stroke="{item["color"]}" stroke-width="3"/>')

    legend_x = px + 10
    legend_y = py + 12
    rows = math.ceil(len(series) / legend_cols)
    col_w = min(260, pw / legend_cols)
    legend_h = rows * 24 + 16
    svg.add(f'<rect x="{legend_x - 8}" y="{legend_y - 14}" width="{min(pw - 8, col_w * legend_cols)}" height="{legend_h}" rx="8" fill="white" fill-opacity="0.91" stroke="#e2e8f0"/>')
    for index, item in enumerate(series):
        col = index // rows
        row = index % rows
        lx = legend_x + col * col_w
        ly = legend_y + row * 24
        dash = f' stroke-dasharray="{item.get("dash")}"' if item.get("dash") else ""
        svg.add(f'<line x1="{lx}" y1="{ly}" x2="{lx + 25}" y2="{ly}" stroke="{item["color"]}" stroke-width="3"{dash}/>')
        svg.add(f'<text x="{lx + 34}" y="{ly + 5}" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#334155">{esc(item["label"])}</text>')

    svg.add(f'<text x="{px + pw / 2}" y="{py + ph + 52}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="600" fill="#475569">{esc(xlabel)}</text>')
    svg.add(f'<text x="{px - 57}" y="{py + ph / 2}" text-anchor="middle" transform="rotate(-90 {px - 57} {py + ph / 2})" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="600" fill="#475569">{esc(ylabel)}</text>')


def grouped_bar_panel(
    svg: SVG,
    box: tuple[float, float, float, float],
    title: str,
    subtitle: str,
    categories: list[str],
    groups: list[dict],
    ylim: tuple[float, float],
    yticks: Iterable[float],
    ylabel: str = "FID (lower is better)",
):
    x, y, w, h = box
    px, py, pw, ph = panel_frame(svg, x, y, w, h, title, subtitle)
    sy = scale_fn(ylim[0], ylim[1], py + ph, py)
    for tick in yticks:
        yy = sy(tick)
        svg.add(f'<line x1="{px}" y1="{yy:.2f}" x2="{px + pw}" y2="{yy:.2f}" stroke="#e2e8f0"/>')
        svg.add(f'<text x="{px - 12}" y="{yy + 5:.2f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#64748b">{esc(tick)}</text>')
    svg.add(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#94a3b8" stroke-width="1.5"/>')
    svg.add(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#94a3b8" stroke-width="1.5"/>')
    cat_w = pw / len(categories)
    group_gap = 5
    bar_w = min(42, (cat_w - 24 - (len(groups) - 1) * group_gap) / len(groups))
    total_w = len(groups) * bar_w + (len(groups) - 1) * group_gap
    for ci, category in enumerate(categories):
        center = px + (ci + 0.5) * cat_w
        label_lines = category.split("\n")
        tspans = []
        for line_index, line in enumerate(label_lines):
            dy = "0" if line_index == 0 else "16"
            tspans.append(f'<tspan x="{center:.2f}" dy="{dy}">{esc(line)}</tspan>')
        svg.add(f'<text x="{center:.2f}" y="{py + ph + 25}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#475569">{"".join(tspans)}</text>')
        for gi, group in enumerate(groups):
            value = group["values"][ci]
            if value is None:
                continue
            bx = center - total_w / 2 + gi * (bar_w + group_gap)
            by = sy(value)
            bh = sy(ylim[0]) - by
            svg.add(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" rx="3" fill="{group["color"]}"/>')
            svg.add(f'<text x="{bx + bar_w / 2:.2f}" y="{by - 7:.2f}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="600" fill="{group["color"]}">{value:.2f}</text>')
    legend_x, legend_y = px + 8, py + 12
    for gi, group in enumerate(groups):
        lx = legend_x + gi * 210
        svg.add(f'<rect x="{lx}" y="{legend_y - 10}" width="16" height="16" rx="3" fill="{group["color"]}"/>')
        svg.add(f'<text x="{lx + 24}" y="{legend_y + 3}" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#334155">{esc(group["label"])}</text>')
    svg.add(f'<text x="{px - 57}" y="{py + ph / 2}" text-anchor="middle" transform="rotate(-90 {px - 57} {py + ph / 2})" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="600" fill="#475569">{esc(ylabel)}</text>')


def curve(rows: list[dict[str, str]], model: str | None = None) -> list[tuple[float, float]]:
    selected = rows if model is None else [row for row in rows if row["model"] == model]
    return sorted((f(row["step"]) / 1000.0, f(row["fid"])) for row in selected if row.get("fid"))


convergence = read_tsv(ROOT / "eval_results/convergence_all_original/fid_convergence_all_original.tsv")
boundary = read_tsv(ROOT / "eval_results/unified_l_boundary_analysis/metrics_summary.tsv")
alloc400 = read_tsv(ROOT / "eval_results/unified_l_400300300_convergence/metrics_summary.tsv")
two16 = read_tsv(ROOT / "eval_results/unified_l_2stage_500500_convergence/metrics_summary.tsv")
two8 = read_tsv(ROOT / "eval_results/unified_l_2stage_8to32_500500_convergence/metrics_summary.tsv")
proportional = read_tsv(ROOT / "eval_results/unified_l_interval_proportional_nfe250/metrics_summary.tsv")

conv_series = [
    {"label": "RDD Unified-XL 300/300/400", "color": COLORS["blue"], "points": curve(convergence, "RDD Unified-XL/2 (300/300/400)")},
    {"label": "RDD Unified-L 300/300/400", "color": COLORS["cyan"], "points": curve(convergence, "RDD Unified-L/2 (300/300/400)")},
    {"label": "Original DiT-XL", "color": COLORS["slate"], "points": curve(convergence, "DiT-XL/2"), "dash": "9 6"},
]

l300 = curve(convergence, "RDD Unified-L/2 (300/300/400)")
l_schedule_series = [
    {"label": "3-stage 300/300/400", "color": COLORS["blue"], "points": l300},
    {"label": "3-stage 400/300/300", "color": COLORS["green"], "points": curve(alloc400)},
    {"label": "3-stage 100/300/600", "color": COLORS["red"], "points": curve(boundary, "100300600")},
    {"label": "3-stage 600/300/100", "color": COLORS["orange"], "points": curve(boundary, "600300100")},
    {"label": "2-stage 16->32, 500/500", "color": COLORS["purple"], "points": curve(two16), "dash": "8 5"},
    {"label": "2-stage 8->32, 500/500", "color": COLORS["pink"], "points": curve(two8), "dash": "8 5"},
]

xl_allocation = [(60, 3.701), (70, 3.624), (90, 3.537), (130, 3.500), (150, 3.404)]
ms_allocation = [(60, 4.013), (70, 3.905), (90, 3.814852), (130, 3.793), (150, 3.700)]
solver_series = [
    {"label": "DDPM", "color": COLORS["blue"], "points": [(35, 5.973), (65, 4.429), (125, 3.888), (250, 3.537)]},
    {"label": "DPM2 (actual NFE)", "color": COLORS["orange"], "points": [(64, 11.761), (124, 3.325), (244, 2.626), (494, 2.617)]},
]
cfg_series = [{"label": "Unified-XL DDPM250", "color": COLORS["purple"], "points": [(1.0, 7.061), (1.5, 3.537), (2.0, 8.335), (2.5, 13.184), (4.0, 20.594)]}]

fixed_250 = {
    "300/300/400": 11.222006288234184,
    "400/300/300": 12.914493964228143,
    "100/300/600": 15.777307312451285,
    "600/300/100": 20.12467344143346,
}
prop_250 = {row["model"][:3] + "/" + row["model"][3:6] + "/" + row["model"][6:]: f(row["fid"]) for row in proportional if row["step"] == "250000"}

# Main six-panel atlas.
atlas = SVG(2400, 1780, "RDD ImageNet-256 FID Results Atlas", "Paper-valid original-ImageNet results; every panel isolates one comparison axis. FID lower is better.")
margin, gap = 45, 28
panel_w = (2400 - 2 * margin - gap) / 2
panel_h = 510
boxes = [
    (margin, 125, panel_w, panel_h),
    (margin + panel_w + gap, 125, panel_w, panel_h),
    (margin, 125 + panel_h + gap, panel_w, panel_h),
    (margin + panel_w + gap, 125 + panel_h + gap, panel_w, panel_h),
    (margin, 125 + 2 * (panel_h + gap), panel_w, panel_h),
    (margin + panel_w + gap, 125 + 2 * (panel_h + gap), panel_w, panel_h),
]
line_panel(atlas, boxes[0], "A. Training convergence", "Same DDPM250, CFG1.0 and 50K evaluation protocol", conv_series, (50, 500), (5, 90), range(50, 501, 50), [5, 10, 20, 40, 80], "Training checkpoint", ylog=True, xfmt=lambda x: f"{int(x)}K")
line_panel(atlas, boxes[1], "B. Unified-L schedule/topology", "Same model and sampler; stage lengths shown fine->coarse", l_schedule_series, (50, 500), (7, 80), range(50, 501, 50), [8, 10, 15, 20, 30, 50, 80], "Training checkpoint", ylog=True, xfmt=lambda x: f"{int(x)}K", legend_cols=2)
line_panel(atlas, boxes[2], "C. Where should 250 DDPM steps go?", "Same 400K checkpoint, CFG1.5, noise filling; x = final 32x32 steps", [
    {"label": "Unified-XL", "color": COLORS["blue"], "points": xl_allocation},
    {"label": "Multistage XL+L+B", "color": COLORS["green"], "points": ms_allocation},
], (55, 155), (3.25, 4.15), [60, 70, 90, 110, 130, 150], [3.4, 3.6, 3.8, 4.0], "Steps assigned to final 32x32 stage", yfmt=lambda y: f"{y:.1f}")
line_panel(atlas, boxes[3], "D. Solver vs actual compute", "Unified-XL 400K, CFG1.5; DPM2 scheduled count is not its NFE", solver_series, (32, 520), (2.3, 12.3), [35, 64, 124, 250, 494], [2.5, 4, 6, 8, 10, 12], "Actual model evaluations (NFE)", xlog=True)
line_panel(atlas, boxes[4], "E. CFG has a sharp optimum", "Unified-XL 400K, DDPM250, noise filling", cfg_series, (0.9, 4.1), (2.5, 22), [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], [4, 8, 12, 16, 20], "CFG scale", xfmt=lambda x: f"{x:.1f}")
categories = ["300/300/400", "400/300/300", "100/300/600", "600/300/100"]
grouped_bar_panel(atlas, boxes[5], "F. Fixed vs interval-proportional steps", "Unified-L at 250K; same total NFE=250 and CFG1.0", categories, [
    {"label": "Fixed [32,16,8]=90/80/80", "color": COLORS["blue"], "values": [fixed_250[c] for c in categories]},
    {"label": "Interval proportional", "color": COLORS["orange"], "values": [prop_250[c] for c in categories]},
], (0, 22), [0, 5, 10, 15, 20])
atlas.save(OUT / "fid_results_atlas_overview.svg")

# Detailed convergence figure.
conv = SVG(2200, 850, "FID Convergence: model size, schedule and stage topology", "All lines use original ImageNet, DDPM250, CFG1.0 and 50K samples. Missing points are not interpolated.")
line_panel(conv, (45, 125, 1038, 670), "Model scale and Original DiT", "Unified-XL 450K is intentionally missing", conv_series, (50, 500), (5, 90), range(50, 501, 50), [5, 10, 20, 40, 80], "Training checkpoint", ylog=True, xfmt=lambda x: f"{int(x)}K")
line_panel(conv, (1117, 125, 1038, 670), "Unified-L boundary and topology", "The ordering is stable across training, not a single-checkpoint accident", l_schedule_series, (50, 500), (7, 80), range(50, 501, 50), [8, 10, 15, 20, 30, 50, 80], "Training checkpoint", ylog=True, xfmt=lambda x: f"{int(x)}K", legend_cols=2)
conv.save(OUT / "fid_convergence_organized.svg")

# Detailed sampling-policy figure.
sampling = SVG(2200, 1450, "FID Sampling Policy Diagnostics", "Correct original-ImageNet results except the explicitly grey legacy panel.")
small_boxes = [(45, 125, 1038, 610), (1117, 125, 1038, 610), (45, 770, 1038, 610), (1117, 770, 1038, 610)]
line_panel(sampling, small_boxes[0], "Same-setting stage allocation", "Both model families improve as more of 250 steps reach final 32x32", [
    {"label": "Unified-XL", "color": COLORS["blue"], "points": xl_allocation},
    {"label": "Multistage XL+L+B", "color": COLORS["green"], "points": ms_allocation},
], (55, 155), (3.25, 4.15), [60, 70, 90, 110, 130, 150], [3.4, 3.6, 3.8, 4.0], "Steps at final 32x32 stage", yfmt=lambda y: f"{y:.1f}")
line_panel(sampling, small_boxes[1], "Sampler budget and solver", "DPM2 wins near 124-244 NFE but collapses near 64 NFE", solver_series, (32, 520), (2.3, 12.3), [35, 64, 124, 250, 494], [2.5, 4, 6, 8, 10, 12], "Actual NFE", xlog=True)
grouped_bar_panel(sampling, small_boxes[2], "Does fixed 90/80/80 bias schedule ranking?", "No: proportional sampling preserves exactly the same training-schedule ranking", categories, [
    {"label": "Fixed 90/80/80", "color": COLORS["blue"], "values": [fixed_250[c] for c in categories]},
    {"label": "Interval proportional", "color": COLORS["orange"], "values": [prop_250[c] for c in categories]},
], (0, 22), [0, 5, 10, 15, 20])
line_panel(sampling, small_boxes[3], "Legacy HF-data trend (hypothesis only)", "Invalid absolute FID; same-setting trend peaks before starving coarse stages", [
    {"label": "Old Multistage, invalid HF training data", "color": COLORS["gray"], "points": [(50, 20.14699), (90, 19.78899), (130, 19.59756), (150, 19.42996), (170, 19.29334), (200, 19.65978)], "dash": "8 6"},
], (45, 205), (19.1, 20.3), [50, 90, 130, 150, 170, 200], [19.2, 19.5, 19.8, 20.1], "Steps at final 32x32 stage", yfmt=lambda y: f"{y:.1f}")
sampling.save(OUT / "fid_sampling_policy_detailed.svg")

# Controls and operating points.
controls = SVG(2200, 850, "FID Controls: boundary matching, transition and operating point", "These panels explain which knobs materially change FID and which comparisons must stay separated.")
control_boxes = [(45, 125, 680, 670), (760, 125, 680, 670), (1475, 125, 680, 670)]
grouped_bar_panel(controls, control_boxes[0], "Inference boundary must match training", "Unified-XL 400K, DDPM250, CFG1.5", ["matched\n300/300/400", "250/250/500", "400/300/300", "200/300/500"], [
    {"label": "FID", "color": COLORS["red"], "values": [3.537, 9.183, 25.023, 40.282]},
], (0, 44), [0, 10, 20, 30, 40])
grouped_bar_panel(controls, control_boxes[1], "Transition initialization", "Unified-XL 400K, DDPM250, CFG1.0", ["Noise filling", "Bilinear"], [
    {"label": "FID", "color": COLORS["green"], "values": [7.061, 9.543]},
], (0, 11), [0, 2, 4, 6, 8, 10])
grouped_bar_panel(controls, control_boxes[2], "400K operating points", "CFG1.0 values are comparable; CFG1.5 is a different quality/diversity point", ["RDD-XL\nDiff CFG1", "RDD-L\nDiff CFG1", "FM-XL\nEuler CFG1", "DiT-XL\nCFG1", "RDD-XL\nDiff CFG1.5", "FM-XL\nRK2 CFG1.5"], [
    {"label": "FID", "color": COLORS["purple"], "values": [7.061, 8.461, 14.617, 19.781, 3.537, 3.291]},
], (0, 22), [0, 5, 10, 15, 20])
controls.save(OUT / "fid_controls_and_operating_points.svg")

# Machine-readable manifest of every curated point used above.
records: list[dict[str, object]] = []
for item in convergence:
    if item.get("fid"):
        records.append({"family": "training_convergence", "model": item["model"], "train_step": item["step"], "setting": item["eval_protocol"], "x": item["step"], "fid": item["fid"], "validity": "paper-valid"})
for label, rows in [("Unified-L 400/300/300", alloc400), ("Unified-L boundary", boundary), ("Unified-L 16->32", two16), ("Unified-L 8->32", two8)]:
    for row in rows:
        records.append({"family": "unified_l_schedule", "model": label + ":" + row["model"], "train_step": row["step"], "setting": "DDPM250 CFG1.0 noise filling", "x": row["step"], "fid": row["fid"], "validity": "paper-valid"})
for model, points in [("Unified-XL", xl_allocation), ("Multistage XL+L+B", ms_allocation)]:
    for xval, fid in points:
        records.append({"family": "sampling_stage_allocation", "model": model, "train_step": 400000, "setting": "DDPM total250 CFG1.5 noise filling", "x": xval, "fid": fid, "validity": "paper-valid"})
for item in solver_series:
    for xval, fid in item["points"]:
        records.append({"family": "solver_nfe", "model": item["label"], "train_step": 400000, "setting": "Unified-XL CFG1.5 noise filling", "x": xval, "fid": fid, "validity": "paper-valid"})
for xval, fid in cfg_series[0]["points"]:
    records.append({"family": "cfg_sweep", "model": "Unified-XL", "train_step": 400000, "setting": "DDPM250 noise filling", "x": xval, "fid": fid, "validity": "paper-valid"})
for category in categories:
    records.append({"family": "fixed_vs_proportional", "model": category, "train_step": 250000, "setting": "fixed fine/mid/coarse=90/80/80", "x": category, "fid": fixed_250[category], "validity": "paper-valid"})
    records.append({"family": "fixed_vs_proportional", "model": category, "train_step": 250000, "setting": "interval-proportional NFE250", "x": category, "fid": prop_250[category], "validity": "paper-valid"})
with (OUT / "fid_results_atlas_data.tsv").open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["family", "model", "train_step", "setting", "x", "fid", "validity"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(records)

print(f"Wrote FID atlas to {OUT}")
