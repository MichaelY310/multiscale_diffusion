#!/usr/bin/env python3
"""Summarize and plot outputs from probe_rdd_attention_dynamics.py."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "eval_results" / "rdd_attention_dynamics"
UNIFIED_FILES = (
    RESULT / "unified_xl_400k_seed20260819.json",
    RESULT / "unified_xl_400k_seed20260820.json",
)
MULTISTAGE_FILE = RESULT / "multistage_300300400_400k_seed20260819.json"


def get(item: dict[str, Any], *path: str) -> float:
    value: Any = item
    for key in path:
        value = value[key]
    return float(value)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_within(
    reports: list[dict[str, Any]], resolution: str, lag: int
) -> list[dict[str, Any]]:
    return [
        item
        for report in reports
        for layer in report["detailed"]["within_resolution"][resolution].values()
        for item in layer[f"lag_{lag}"]
    ]


def collect_cross(
    reports: list[dict[str, Any]], pair: str
) -> list[dict[str, Any]]:
    return [
        item
        for report in reports
        for layer in report["detailed"]["cross_resolution"][pair].values()
        for item in layer
    ]


def by_step(items: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[int(item[key])].append(item)
    return dict(sorted(grouped.items()))


def avg(items: list[dict[str, Any]], *path: str) -> float:
    return mean(get(item, *path) for item in items)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")


def plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    y_range: tuple[float, float],
    y_label: str,
) -> None:
    left, top, right, bottom = box
    pad_l, pad_r, pad_t, pad_b = 65, 18, 48, 50
    x0, y0, x1, y1 = left + pad_l, top + pad_t, right - pad_r, bottom - pad_b
    draw.rectangle(box, fill="white", outline="#d1d5db", width=1)
    draw.text((left + 14, top + 12), title, fill="#111827", font=font(20, True))
    all_x = [x for _, points in series for x, _ in points]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = y_range
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, py, x1, py), fill="#e5e7eb", width=1)
        draw.text((left + 8, py - 8), f"{value:.2f}", fill="#4b5563", font=font(13))
    for tick in range(6):
        value = xmin + (xmax - xmin) * tick / 5
        px = x0 + (value - xmin) / max(xmax - xmin, 1e-9) * (x1 - x0)
        draw.text((px - 12, y1 + 8), f"{value:.0f}", fill="#4b5563", font=font(13))
    draw.text((left + 8, top + 32), y_label, fill="#6b7280", font=font(12))
    for index, (name, points) in enumerate(series):
        color = COLORS[index]
        pixels = []
        for x, y in points:
            px = x0 + (x - xmin) / max(xmax - xmin, 1e-9) * (x1 - x0)
            py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
            pixels.append((px, py))
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=3)
        for px, py in pixels:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        legend_x = x0 + index * 180
        draw.line((legend_x, bottom - 18, legend_x + 24, bottom - 18), fill=color, width=3)
        draw.text((legend_x + 30, bottom - 27), name, fill="#111827", font=font(13))


def main() -> None:
    unified = [load(path) for path in UNIFIED_FILES]
    multistage = [load(MULTISTAGE_FILE)]

    within_rows = []
    for family, reports in (("Unified XL", unified), ("Multistage", multistage)):
        for resolution in (8, 16, 32):
            for lag in (1, 5, 10):
                items = collect_within(reports, str(resolution), lag)
                within_rows.append(
                    {
                        "family": family,
                        "resolution": resolution,
                        "lag": lag,
                        "score_cosine": avg(items, "full_score", "cosine"),
                        "route_cosine": avg(items, "root_route", "cosine"),
                        "route_top40_overlap": avg(items, "root_route", "top40_overlap"),
                        "score_reuse_output_cosine": avg(
                            items, "reuse_old_score_with_current_v", "cosine"
                        ),
                        "score_reuse_output_relative_l2": avg(
                            items, "reuse_old_score_with_current_v", "relative_l2"
                        ),
                        "broadcast_output_cosine": avg(
                            items, "broadcast_old_attention_output", "cosine"
                        ),
                    }
                )
    write_csv(RESULT / "within_resolution_summary.csv", within_rows)

    within_step_rows = []
    for family, reports in (("Unified XL", unified), ("Multistage", multistage)):
        for step, items in by_step(collect_within(reports, "32", 10), "current_step").items():
            within_step_rows.append(
                {
                    "family": family,
                    "current_step": step,
                    "score_cosine": avg(items, "full_score", "cosine"),
                    "route_cosine": avg(items, "root_route", "cosine"),
                    "route_top40_overlap": avg(items, "root_route", "top40_overlap"),
                    "score_reuse_output_cosine": avg(
                        items, "reuse_old_score_with_current_v", "cosine"
                    ),
                    "score_reuse_output_relative_l2": avg(
                        items, "reuse_old_score_with_current_v", "relative_l2"
                    ),
                    "broadcast_output_cosine": avg(
                        items, "broadcast_old_attention_output", "cosine"
                    ),
                }
            )
    write_csv(RESULT / "within_32_lag10_by_step.csv", within_step_rows)

    cross_rows = []
    for family, reports in (("Unified XL", unified), ("Multistage", multistage)):
        for pair in ("8_to_16", "16_to_32"):
            for step, items in by_step(collect_cross(reports, pair), "fine_step").items():
                cross_rows.append(
                    {
                        "family": family,
                        "pair": pair,
                        "fine_step": step,
                        "route_cosine": avg(items, "root_route", "cosine"),
                        "route_top40_overlap": avg(items, "root_route", "top40_overlap"),
                        "direct_score_reuse_output_cosine": avg(
                            items, "reuse_coarse_score_with_fine_v", "cosine"
                        ),
                        "direct_score_reuse_output_relative_l2": avg(
                            items, "reuse_coarse_score_with_fine_v", "relative_l2"
                        ),
                    }
                )
    write_csv(RESULT / "cross_resolution_by_step.csv", cross_rows)

    within32 = by_step(collect_within(unified, "32", 10), "current_step")
    cross16 = by_step(collect_cross(unified, "16_to_32"), "fine_step")
    cross8 = by_step(collect_cross(unified, "8_to_16"), "fine_step")
    canvas = Image.new("RGB", (1600, 1000), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (36, 18),
        "RDD Attention Dynamics — Unified XL/2 400K (2 seeds, layers 0/14/27, 4 heads)",
        fill="#111827",
        font=font(28, True),
    )
    draw.text(
        (36, 54),
        "Score reuse = old softmax(QK^T) applied to current V; route = attention pooled to coarse blocks",
        fill="#4b5563",
        font=font(16),
    )
    plot_panel(
        draw,
        (30, 90, 785, 525),
        "32x32: ten-step similarity",
        [
            ("full score", [(s, avg(x, "full_score", "cosine")) for s, x in within32.items()]),
            ("block route", [(s, avg(x, "root_route", "cosine")) for s, x in within32.items()]),
        ],
        (0.75, 1.0),
        "cosine",
    )
    plot_panel(
        draw,
        (815, 90, 1570, 525),
        "32x32: ten-step reuse quality",
        [
            (
                "route top40",
                [(s, avg(x, "root_route", "top40_overlap")) for s, x in within32.items()],
            ),
            (
                "old score @ new V",
                [
                    (s, avg(x, "reuse_old_score_with_current_v", "cosine"))
                    for s, x in within32.items()
                ],
            ),
            (
                "old output",
                [
                    (s, avg(x, "broadcast_old_attention_output", "cosine"))
                    for s, x in within32.items()
                ],
            ),
        ],
        (0.55, 1.0),
        "overlap / cosine",
    )
    plot_panel(
        draw,
        (30, 550, 785, 970),
        "16x16 final score -> 32x32 stage",
        [
            ("route cosine", [(s, avg(x, "root_route", "cosine")) for s, x in cross16.items()]),
            (
                "route top40",
                [(s, avg(x, "root_route", "top40_overlap")) for s, x in cross16.items()],
            ),
            (
                "direct score output",
                [
                    (s, avg(x, "reuse_coarse_score_with_fine_v", "cosine"))
                    for s, x in cross16.items()
                ],
            ),
        ],
        (0.45, 1.0),
        "similarity",
    )
    plot_panel(
        draw,
        (815, 550, 1570, 970),
        "Cross-resolution direct score reuse error",
        [
            (
                "8 -> 16",
                [
                    (s, avg(x, "reuse_coarse_score_with_fine_v", "relative_l2"))
                    for s, x in cross8.items()
                ],
            ),
            (
                "16 -> 32",
                [
                    (s, avg(x, "reuse_coarse_score_with_fine_v", "relative_l2"))
                    for s, x in cross16.items()
                ],
            ),
        ],
        (0.0, 1.1),
        "relative L2 (lower is better)",
    )
    canvas.save(RESULT / "rdd_attention_dynamics.png")

    def row(family: str, resolution: int, lag: int) -> dict[str, Any]:
        return next(
            item
            for item in within_rows
            if item["family"] == family
            and item["resolution"] == resolution
            and item["lag"] == lag
        )

    u32 = row("Unified XL", 32, 10)
    m32 = row("Multistage", 32, 10)
    report = f"""# RDD attention 随 timestep / resolution 的变化

## 结论

1. 同一 stage 内，attention 的**粗粒度路由**明显比完整 softmax 权重稳定。
2. 每10步刷新一次 route 很有希望；每10步直接复用完整 attention score 风险较大，需要实际采样/FID验证。
3. 跨 resolution 可以复用拓扑来选 block，但不能把 coarse attention probability 原样当作 fine probability。
4. `A_old @ V_current` 明显优于直接广播旧的 `A_old @ V_old`，说明 V 随 timestep 的变化不可忽略。

## 32×32 stage，相隔10步

| model | full-score cosine | block-route cosine | Top-40% overlap | `A_old @ V_new` cosine | relative L2 | old-output cosine |
|---|---:|---:|---:|---:|---:|---:|
| Unified XL（2 seeds） | {u32['score_cosine']:.4f} | {u32['route_cosine']:.4f} | {u32['route_top40_overlap']:.4f} | {u32['score_reuse_output_cosine']:.4f} | {u32['score_reuse_output_relative_l2']:.4f} | {u32['broadcast_output_cosine']:.4f} |
| Multistage B/L/XL | {m32['score_cosine']:.4f} | {m32['route_cosine']:.4f} | {m32['route_top40_overlap']:.4f} | {m32['score_reuse_output_cosine']:.4f} | {m32['score_reuse_output_relative_l2']:.4f} | {m32['broadcast_output_cosine']:.4f} |

## 实验定义

- Unified：400K checkpoint，两个 class/seed。
- Multistage：300/300/400 的400K checkpoints，一个 class/seed。
- 采样：80/80/90，dense DDPM trajectory，CFG 1.0。
- 层：按相对深度对齐的 early/middle/late 三层；每层前4个 heads。
- `score reuse`：不复用旧 V，而是计算 `A_old @ V_current`。
- `route`：把 fine attention 聚合到 coarse blocks 后比较。

详细逐 step 数据见 `within_resolution_summary.csv` 和 `cross_resolution_by_step.csv`。
"""
    (RESULT / "RDD_ATTENTION_DYNAMICS_REPORT_ZH.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
