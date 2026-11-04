"""Generate the frozen RDD-3D final-test quality/compute figure.

Sources:
  eval_results/rdd3d_dynamic_sparse/dynamic_sparse_final_test_64_seed20260812/
    final_summary.json
    runtime_benchmark.json

The learned table is a one-shot 64-asset final test with two sampling seeds.
The runtime panel uses the selected endpoint-head model for both sparse and
force-keep policies, drops one warm-up asset per process, and averages the next
15 assets on each of two GPUs/processes (30 measured samples total).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Rect, String
from reportlab.lib import colors


METHODS = [
    ("19K zero-shot", 5.18, 0.157, 0.376),
    ("29K on-policy", 4.38, 0.175, 0.406),
    ("+ endpoint head", 4.53, 0.191, 0.509),
]
RUNTIME = [("Sparse", 0.5056), ("Force-dense", 1.9904)]
MEMORY = [("Sparse", 1542), ("Force-dense", 2852)]


def label(drawing, x, y, text, *, size=8, anchor="middle", color=None, bold=False):
    drawing.add(
        String(
            x,
            y,
            text,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            fillColor=color or colors.HexColor("#202124"),
            textAnchor=anchor,
        )
    )


def axes(drawing, left, bottom, width, height, x_ticks, y_ticks, x_map, y_map):
    ink = colors.HexColor("#202124")
    grid = colors.HexColor("#DADCE0")
    drawing.add(Rect(left, bottom, width, height, fillColor=None, strokeColor=ink, strokeWidth=0.8))
    for value, text in y_ticks:
        y = y_map(value)
        if y > bottom and y < bottom + height:
            drawing.add(Line(left, y, left + width, y, strokeColor=grid, strokeWidth=0.45))
        drawing.add(Line(left - 3, y, left, y, strokeColor=ink, strokeWidth=0.7))
        label(drawing, left - 6, y - 2.5, text, size=7, anchor="end")
    for value, text in x_ticks:
        x = x_map(value)
        drawing.add(Line(x, bottom, x, bottom - 3, strokeColor=ink, strokeWidth=0.7))
        label(drawing, x, bottom - 13, text, size=7)


def bar_panel(drawing, *, left, bottom, width, height, values, ymax, ticks, title, unit):
    ink = colors.HexColor("#202124")
    grid = colors.HexColor("#DADCE0")
    series = colors.HexColor("#0072B2")
    frame = colors.HexColor("#8A8D91")
    y = lambda value: bottom + value / ymax * height
    drawing.add(Rect(left, bottom, width, height, fillColor=None, strokeColor=frame, strokeWidth=0.8))
    for tick in ticks:
        yy = y(tick)
        if tick:
            drawing.add(Line(left, yy, left + width, yy, strokeColor=grid, strokeWidth=0.45))
        label(drawing, left - 5, yy - 2.5, str(tick), size=7, anchor="end")
    centers = [left + width * 0.29, left + width * 0.71]
    bar_w = width * 0.27
    for center, (name, value) in zip(centers, values):
        top = y(value)
        drawing.add(
            Rect(
                center - bar_w / 2,
                bottom,
                bar_w,
                top - bottom,
                fillColor=series if name == "Sparse" else colors.white,
                strokeColor=series,
                strokeWidth=1.2,
            )
        )
        label(drawing, center, top + 5, f"{value:g}", size=7.5, bold=True, color=ink)
        label(drawing, center, bottom - 13, name, size=7)
    label(drawing, left + width / 2, bottom + height + 12, title, size=9, bold=True)
    label(drawing, left + width / 2, bottom - 27, unit, size=7.5)


def build_figure() -> Drawing:
    width, height = 720, 270
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=None))
    ink = colors.HexColor("#202124")
    blue = colors.HexColor("#0072B2")
    orange = colors.HexColor("#D55E00")
    gray = colors.HexColor("#6B6F73")

    # (a) Quality-compute Pareto on the frozen 64-asset test.
    left, bottom, plot_w, plot_h = 48, 54, 310, 174
    x_min, x_max = 4.1, 5.4
    y_min, y_max = 0.15, 0.198
    x_map = lambda value: left + (value - x_min) / (x_max - x_min) * plot_w
    y_map = lambda value: bottom + (value - y_min) / (y_max - y_min) * plot_h
    axes(
        drawing,
        left,
        bottom,
        plot_w,
        plot_h,
        [(4.2, "4.2"), (4.6, "4.6"), (5.0, "5.0"), (5.4, "5.4")],
        [(0.15, ".15"), (0.16, ".16"), (0.17, ".17"), (0.18, ".18"), (0.19, ".19")],
        x_map,
        y_map,
    )
    label(drawing, left + plot_w / 2, bottom - 29, "Matched-dense token-NFE reduction (higher is better)", size=8.5)
    y_label = Group()
    y_label.add(
        String(
            0,
            0,
            "Support IoU",
            fontName="Helvetica",
            fontSize=8.5,
            fillColor=ink,
            textAnchor="middle",
        )
    )
    y_label.transform = (0, 1, -1, 0, 13, bottom + plot_h / 2)
    drawing.add(y_label)
    label(drawing, left, bottom + plot_h + 14, "(a) Final-test quality–compute Pareto", size=9, anchor="start", bold=True)

    for name, reduction, iou, recall in METHODS:
        x, y = x_map(reduction), y_map(iou)
        selected = name == "+ endpoint head"
        drawing.add(
            Circle(
                x,
                y,
                4.3 if selected else 3.6,
                fillColor=orange if selected else colors.white,
                strokeColor=orange if selected else blue,
                strokeWidth=1.5,
            )
        )
        if name == "19K zero-shot":
            label(drawing, x - 7, y - 14, f"{name}  R={recall:.3f}", size=7.3, anchor="end", color=gray)
        elif name == "29K on-policy":
            label(drawing, x + 8, y + 7, f"{name}  R={recall:.3f}", size=7.3, anchor="start", color=gray)
        else:
            label(drawing, x + 8, y + 7, f"{name}  R={recall:.3f}", size=7.5, anchor="start", color=ink, bold=True)

    # (b,c) Same-checkpoint actual controls.
    bar_panel(
        drawing,
        left=414,
        bottom=54,
        width=126,
        height=174,
        values=RUNTIME,
        ymax=2.2,
        ticks=(0, 0.5, 1.0, 1.5, 2.0),
        title="(b) Sampler runtime",
        unit="seconds / asset (3.94x)",
    )
    bar_panel(
        drawing,
        left=582,
        bottom=54,
        width=112,
        height=174,
        values=MEMORY,
        ymax=3200,
        ticks=(0, 1000, 2000, 3000),
        title="(c) Peak memory",
        unit="allocated MiB (-46%)",
    )
    return drawing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drawing = build_figure()
    pdf = args.output_dir / "rdd3d_final_quality_compute.pdf"
    svg = args.output_dir / "rdd3d_final_quality_compute.svg"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    print(f"Saved {pdf}")
    print(f"Saved {svg}")


if __name__ == "__main__":
    main()
