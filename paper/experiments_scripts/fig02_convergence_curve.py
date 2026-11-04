"""Generate the paper's FID-vs-training-steps convergence figure.

The values come from ``eval_results_exp04`` except for the corrected DiT 400K
point, which comes from ``eval_results_unified/dit400k_250step_ddpm_cfg0.txt``.
All evaluations use native ImageNet training data, 50K generated samples, the
ADM ImageNet-256 reference statistics, CFG=1.0, and 250-step DDPM sampling.

RDD is the unified DiT-XL/2 model with the 300/300/400 training schedule,
noise-filling transitions, and 90/80/80 fine/mid/coarse sampling steps.  The
RDD 450K measurement is unavailable and is intentionally not interpolated.

Outputs (by default under ``paper/figures``):
  - fid_convergence.pdf  (vector, included by main.tex)
  - fid_convergence.svg  (vector preview/source)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, PolyLine, Rect, String
from reportlab.lib import colors


STEPS_K = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

DIT_FID = [
    77.1497018563,
    47.3273668078,
    35.7419844835,
    29.6619752498,
    26.0552149420,
    23.1251302074,
    20.9134082497,
    19.7812880842,  # corrected 400K evaluation
    18.5994947022,
    17.4635682410,
]

RDD_FID = [
    47.4113469811,
    20.4880171878,
    13.7545927726,
    10.6639062567,
    9.2500474415,
    8.1492405078,
    7.5101247906,
    7.0650797606,
    None,  # 450K result is missing; keep the curve discontinuous.
    6.4608067325,
]


def _add_centered_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: colors.Color,
    font: str = "Helvetica",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName=font,
            fontSize=size,
            fillColor=color,
            textAnchor="middle",
        )
    )


def build_figure() -> Drawing:
    width, height = 475, 280
    left, right, bottom, top = 48, 14, 42, 16
    plot_w = width - left - right
    plot_h = height - bottom - top

    x_min, x_max = 50.0, 500.0
    y_min, y_max = 0.0, 80.0

    def x_pos(step_k: float) -> float:
        return left + (step_k - x_min) / (x_max - x_min) * plot_w

    def y_pos(fid: float) -> float:
        return bottom + (fid - y_min) / (y_max - y_min) * plot_h

    ink = colors.HexColor("#202124")
    grid = colors.HexColor("#DADCE0")
    dit_color = colors.HexColor("#0072B2")
    rdd_color = colors.HexColor("#D55E00")

    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=None))
    drawing.add(Rect(left, bottom, plot_w, plot_h, fillColor=None, strokeColor=ink, strokeWidth=0.8))

    for tick in range(0, 81, 10):
        y = y_pos(tick)
        if tick not in (0, 80):
            drawing.add(Line(left, y, left + plot_w, y, strokeColor=grid, strokeWidth=0.45))
        drawing.add(Line(left - 3, y, left, y, strokeColor=ink, strokeWidth=0.7))
        drawing.add(
            String(
                left - 7,
                y - 2.6,
                str(tick),
                fontName="Helvetica",
                fontSize=7.5,
                fillColor=ink,
                textAnchor="end",
            )
        )

    for tick in STEPS_K:
        x = x_pos(tick)
        drawing.add(Line(x, bottom, x, bottom - 3, strokeColor=ink, strokeWidth=0.7))
        _add_centered_text(drawing, x, bottom - 13, str(tick), size=7.5, color=ink)

    _add_centered_text(
        drawing,
        left + plot_w / 2,
        8,
        "Training steps (thousands)",
        size=9.5,
        color=ink,
    )
    y_label = Group()
    y_label.add(
        String(
            0,
            0,
            "FID-50K (lower is better)",
            fontName="Helvetica",
            fontSize=9.5,
            fillColor=ink,
            textAnchor="middle",
        )
    )
    y_label.transform = (0, 1, -1, 0, 12, bottom + plot_h / 2)
    drawing.add(y_label)

    dit_points = [(x_pos(step), y_pos(fid)) for step, fid in zip(STEPS_K, DIT_FID)]
    drawing.add(PolyLine([coord for point in dit_points for coord in point], strokeColor=dit_color, strokeWidth=1.8))
    for x, y in dit_points:
        drawing.add(Rect(x - 2.4, y - 2.4, 4.8, 4.8, fillColor=colors.white, strokeColor=dit_color, strokeWidth=1.2))

    # Draw only contiguous observed RDD segments.  This deliberately leaves a
    # gap from 400K to 500K because the 450K evaluation is missing.
    rdd_runs: list[list[tuple[float, float]]] = []
    current_run: list[tuple[float, float]] = []
    for step, fid in zip(STEPS_K, RDD_FID):
        if fid is None:
            if current_run:
                rdd_runs.append(current_run)
                current_run = []
            continue
        current_run.append((x_pos(step), y_pos(fid)))
    if current_run:
        rdd_runs.append(current_run)

    for run in rdd_runs:
        if len(run) > 1:
            drawing.add(PolyLine([coord for point in run for coord in point], strokeColor=rdd_color, strokeWidth=2.0))
        for x, y in run:
            drawing.add(Circle(x, y, 2.7, fillColor=colors.white, strokeColor=rdd_color, strokeWidth=1.4))

    # Directly mark the controlled 400K comparison.
    x400 = x_pos(400)
    drawing.add(Line(x400, bottom, x400, bottom + plot_h, strokeColor=grid, strokeWidth=0.7, strokeDashArray=[2, 2]))
    drawing.add(String(x400 + 5, y_pos(DIT_FID[7]) + 5, "19.78", fontName="Helvetica-Bold", fontSize=7.5, fillColor=dit_color))
    drawing.add(String(x400 + 5, y_pos(RDD_FID[7]) + 5, "7.07", fontName="Helvetica-Bold", fontSize=7.5, fillColor=rdd_color))

    # Compact legend inside the otherwise empty upper-right region.
    legend_x = left + plot_w - 188
    legend_y = bottom + plot_h - 13
    drawing.add(Line(legend_x, legend_y, legend_x + 20, legend_y, strokeColor=rdd_color, strokeWidth=2.0))
    drawing.add(Circle(legend_x + 10, legend_y, 2.7, fillColor=colors.white, strokeColor=rdd_color, strokeWidth=1.4))
    drawing.add(String(legend_x + 26, legend_y - 2.6, "RDD Unified XL", fontName="Helvetica", fontSize=8.5, fillColor=ink))

    drawing.add(Line(legend_x, legend_y - 15, legend_x + 20, legend_y - 15, strokeColor=dit_color, strokeWidth=1.8))
    drawing.add(Rect(legend_x + 7.6, legend_y - 17.4, 4.8, 4.8, fillColor=colors.white, strokeColor=dit_color, strokeWidth=1.2))
    drawing.add(String(legend_x + 26, legend_y - 17.6, "DiT-XL/2", fontName="Helvetica", fontSize=8.5, fillColor=ink))

    return drawing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the FID convergence figure")
    default_dir = Path(__file__).resolve().parents[1] / "figures"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    drawing = build_figure()

    pdf_path = args.output_dir / "fid_convergence.pdf"
    svg_path = args.output_dir / "fid_convergence.svg"
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))

    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
