#!/usr/bin/env python3
"""Build dependency-free SVG figures for the real-checkpoint RDD report."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research_docs/assets/rdd_real_checkpoint_validation"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def svg_text(x, y, value, size=14, anchor="start", fill="#182033", weight=400):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">'
        f'{html.escape(str(value))}</text>'
    )


def pareto() -> None:
    rdd = load("eval_results/rdd_time_conditioned_stable_400k/metrics.json")
    dit = load("eval_results/dit_fair_temporal_stagger_stable_400k/metrics.json")
    top3 = load("eval_results/rdd_delta_top3_stable_400k/metrics.json")
    rb = rdd["interleaved_benchmark"]["latency_reduction_percent"]
    db = dit["interleaved_benchmark"]["latency_reduction_percent"]
    tb = top3["interleaved_benchmark"]["latency_reduction_percent"]
    points = [
        (rb["static_stagewide3"], rdd["settings"]["static_stagewide3"]["latent_relative_l2_vs_baseline"], "RDD static 3", "#dc2626"),
        (rb["time_stagewide3"], rdd["settings"]["time_stagewide3"]["latent_relative_l2_vs_baseline"], "RDD time 3", "#f97316"),
        (rb["time_period3_safe6"], rdd["settings"]["time_period3_safe6"]["latent_relative_l2_vs_baseline"], "RDD period-3/6", "#7c3aed"),
        (rb["time_stagger6_ranked"], rdd["settings"]["time_stagger6_ranked"]["latent_relative_l2_vs_baseline"], "RDD stagger-6", "#2563eb"),
        (tb["rdd_transport_p5_top3"], top3["settings"]["rdd_transport_p5_top3"]["latent_relative_l2_vs_baseline"], "RDD transport p5/top-3", "#0f766e"),
        (tb["rdd_transport_p8_top3"], top3["settings"]["rdd_transport_p8_top3"]["latent_relative_l2_vs_baseline"], "RDD transport p8/top-3", "#0891b2"),
        (db["temporal_stagger9"], dit["settings"]["temporal_stagger9"]["latent_relative_l2_vs_baseline"], "DiT temporal-9", "#059669"),
        (db["temporal_stagger12"], dit["settings"]["temporal_stagger12"]["latent_relative_l2_vs_baseline"], "DiT temporal-12", "#047857"),
    ]
    w, h = 940, 560
    left, right, top, bottom = 90, 35, 60, 80
    xmax, ymax = 10.0, 0.45
    px = lambda x: left + x / xmax * (w - left - right)
    py = lambda y: top + (1 - y / ymax) * (h - top - bottom)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']
    body.append(svg_text(w / 2, 30, "真实 400K checkpoint：速度—扰动 Pareto", 20, "middle", weight=700))
    for tick in range(0, 11, 2):
        x = px(tick)
        body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{h-bottom}" stroke="#e5e7eb"/>')
        body.append(svg_text(x, h - bottom + 24, tick, 12, "middle"))
    for tick in [0, .1, .2, .3, .4]:
        y = py(tick)
        body.append(f'<line x1="{left}" y1="{y}" x2="{w-right}" y2="{y}" stroke="#e5e7eb"/>')
        body.append(svg_text(left - 12, y + 4, f"{tick:.1f}", 12, "end"))
    body.append(f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#111827"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#111827"/>')
    body.append(svg_text((left + w-right)/2, h-25, "稳定 denoiser 延迟减少（%）→ 越右越快", 14, "middle", weight=600))
    body.append(f'<g transform="translate(25 {(top+h-bottom)/2}) rotate(-90)">{svg_text(0,0,"最终 latent 相对 L2 → 越低越好",14,"middle",weight=600)}</g>')
    offsets = [(10,-10),(10,18),(10,-10),(10,18),(10,-12),(10,20),(10,-10),(-10,20)]
    for (x, y, label, color), (dx, dy) in zip(points, offsets):
        cx, cy = px(x), py(y)
        body.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
        body.append(svg_text(cx + dx, cy + dy, label, 12, "end" if dx < 0 else "start", color, 600))
    body.append(svg_text(w-right, h-8, "同一模型内相对基线；RDD 与 DiT 只作强基线比较，不比较绝对秒数", 11, "end", "#64748b"))
    body.append('</svg>')
    (OUT / "quality_speed_pareto.svg").write_text("\n".join(body), encoding="utf-8")


def transport_gap() -> None:
    reports = [
        load("eval_results/rdd_generation_trajectory_seed_91001/metrics.json"),
        load("eval_results/rdd_generation_trajectory_seed_92003/metrics.json"),
    ]
    gaps = [1, 2, 3, 4, 5, 8, 16]
    series = {}
    for band, key in [
        ("完整 residual", "mean_full_relative_l2_improvement"),
        ("低频 residual", "mean_low_relative_l2_improvement"),
        ("高频 residual", "mean_high_relative_l2_improvement"),
    ]:
        series[band] = [
            sum(report["summary"][str(gap)][key] for report in reports) / len(reports)
            for gap in gaps
        ]
    w, h = 940, 540
    left, right, top, bottom = 90, 40, 65, 80
    ymin, ymax = -0.001, 0.018
    px = lambda i: left + i / (len(gaps)-1) * (w-left-right)
    py = lambda y: top + (ymax-y)/(ymax-ymin)*(h-top-bottom)
    colors = {"完整 residual":"#2563eb", "低频 residual":"#059669", "高频 residual":"#dc2626"}
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']
    body.append(svg_text(w/2, 30, "真实 dense reverse trajectory：RDD transport 相对 stale cache 的改善", 19, "middle", weight=700))
    for tick in [-.001, 0, .004, .008, .012, .016]:
        y = py(tick)
        body.append(f'<line x1="{left}" y1="{y}" x2="{w-right}" y2="{y}" stroke="{("#94a3b8" if tick==0 else "#e5e7eb")}"/>')
        body.append(svg_text(left-10,y+4,f"{tick:.3f}",12,"end"))
    for i, gap in enumerate(gaps):
        x=px(i)
        body.append(svg_text(x,h-bottom+24,gap,12,"middle"))
    for label, values in series.items():
        pts=" ".join(f"{px(i)},{py(v)}" for i,v in enumerate(values))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{colors[label]}" stroke-width="3"/>')
        for i,v in enumerate(values): body.append(f'<circle cx="{px(i)}" cy="{py(v)}" r="5" fill="{colors[label]}"/>')
    body.append(svg_text((left+w-right)/2,h-25,"缓存间隔（NFE 数）",14,"middle",weight=600))
    body.append(f'<g transform="translate(25 {(top+h-bottom)/2}) rotate(-90)">{svg_text(0,0,"relative-L2 改善（>0 表示 transport 更好）",14,"middle",weight=600)}</g>')
    for i,(label,color) in enumerate(colors.items()):
        x=left+20+i*235
        body.append(f'<line x1="{x}" y1="{top-18}" x2="{x+28}" y2="{top-18}" stroke="{color}" stroke-width="4"/>')
        body.append(svg_text(x+36,top-13,label,12,"start",color,600))
    body.append('</svg>')
    (OUT / "transport_improvement_by_gap.svg").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pareto()
    transport_gap()
    print(OUT)


if __name__ == "__main__":
    main()
