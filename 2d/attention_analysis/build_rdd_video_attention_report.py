#!/usr/bin/env python3
"""Build a self-contained Chinese HTML/Markdown report from RDD attention sweeps."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


def f(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def svg_scatter(rows: list[dict[str, Any]], output: Path) -> None:
    points = [
        row
        for row in rows
        if row["family"] not in {"centroid_oracle", "centroid_tuning16"}
        and row.get("speedup_median") is not None
        and row.get("image_cosine_mean") is not None
    ]
    width, height = 1280, 760
    left, right, top, bottom = 95, 35, 55, 85
    xmin, xmax = 0.93, max(1.25, max(p["speedup_median"] for p in points) + 0.02)
    ymin, ymax = 0.94, 1.001
    colors = {
        "true_kernel": "#2563eb",
        "cfg_probability_share": "#dc2626",
        "adaptive_probability": "#059669",
        "probability_tuning16": "#7c3aed",
    }

    def xmap(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def ymap(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif}.tick{font-size:14px;fill:#334155}.label{font-size:10px;fill:#111827}.title{font-size:22px;font-weight:700}</style>',
        '<text x="95" y="30" class="title">RDD training-free attention: measured denoiser speed vs. same-noise image cosine</text>',
    ]
    for i in range(7):
        y = ymin + i * (ymax - ymin) / 6
        py = ymap(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-10}" y="{py+5:.1f}" text-anchor="end" class="tick">{y:.3f}</text>')
    for i in range(7):
        x = xmin + i * (xmax - xmin) / 6
        px = xmap(x)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+25}" text-anchor="middle" class="tick">{x:.2f}×</text>')
    parts.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-25}" text-anchor="middle" class="tick">denoiser speedup vs dense RDD (higher is better)</text>')
    parts.append(f'<text x="25" y="{(top+height-bottom)/2:.1f}" transform="rotate(-90 25 {(top+height-bottom)/2:.1f})" text-anchor="middle" class="tick">image cosine (higher is better)</text>')

    for point in points:
        px, py = xmap(point["speedup_median"]), ymap(point["image_cosine_mean"])
        color = colors.get(point["family"], "#64748b")
        label = html.escape(point["name"].replace("stage32_", "32_").replace("dense_time_score_", "time_"))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.85"><title>{html.escape(point["family"])} / {label}</title></circle>')
        # Label only Pareto points to keep the plot readable.
        if point.get("pareto"):
            parts.append(f'<text x="{px+7:.1f}" y="{py-7:.1f}" class="label">{label}</text>')
    legend_x = width - 315
    for index, (family, color) in enumerate(colors.items()):
        y = 72 + index * 22
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+12}" y="{y+4}" class="tick">{html.escape(family)}</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | setting | seeds | speedup | cosine mean | cosine min | PSNR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | `{row['name']}` | {row['runs']} | "
            f"{f(row.get('speedup_median'))}× | {f(row.get('image_cosine_mean'), 6)} | "
            f"{f(row.get('image_cosine_min'), 6)} | {f(row.get('image_psnr_db_mean'), 2)} |"
        )
    return "\n".join(lines)


def pct(value: Any) -> str:
    return f"{100.0 * float(value):g}%"


CASCADE_RUNS = (
    ("explore-2cls", Path("eval_results/rdd_packed_cascade_search/metrics.json"), False),
    ("layout-4cls", Path("eval_results/rdd_packed_layout_confirm/seed_20260826/metrics.json"), True),
    ("hybrid-4cls", Path("eval_results/rdd_hybrid32_confirm/seed_20260827/metrics.json"), True),
    ("final-4cls", Path("eval_results/rdd_packed_final_confirm/seed_20260828/metrics.json"), True),
    ("nocache-4cls", Path("eval_results/rdd_nocache_isolation/seed_20260828/metrics.json"), True),
    ("density-4cls", Path("eval_results/rdd_strict_cascade_density_sweep/seed_20260828/metrics.json"), True),
)


def load_cascade_rows() -> list[dict[str, Any]]:
    """Load the focused native 8->16->32 routing/kernel experiments when present."""
    rows: list[dict[str, Any]] = []
    for run_name, metrics_path, confirm in CASCADE_RUNS:
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        dense_time = float(data["variants"]["dense"]["denoiser_seconds"])
        for name, item in data["variants"].items():
            if name == "dense":
                continue
            setting = item.get("setting", {})
            measured = item.get("mean_selected_block_density", {})
            target16 = float(setting.get("density16", 1.0))
            target32 = float(setting.get("density32", 1.0))
            rows.append(
                {
                    "run": run_name,
                    "confirm": confirm,
                    "seed": data.get("seed"),
                    "classes": len(data.get("classes", [])),
                    "name": name,
                    "backend": setting.get("kernel_backend", "legacy"),
                    "target16": target16,
                    "target32": target32,
                    "actual16": 1.0 if target16 == 1.0 else measured.get("16"),
                    "actual32": 1.0 if target32 == 1.0 else measured.get("32"),
                    "k16": setting.get("score_interval16", 1),
                    "k32": setting.get("score_interval32", 1),
                    "speedup": dense_time / float(item["denoiser_seconds"]),
                    "cosine": item.get("image_vs_dense", {}).get("pixel_cosine"),
                    "psnr": item.get("image_vs_dense", {}).get("psnr_db"),
                }
            )
    return rows


def cascade_detail_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | setting | backend | 实际保留block 16/32 | K16/K32 | denoiser加速 | cosine | PSNR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        density = (
            f"{100.0 * float(row['actual16']):.1f}%/"
            f"{100.0 * float(row['actual32']):.1f}%"
        )
        lines.append(
            f"| {row['run']} | `{row['name']}` | {row['backend']} | {density} | "
            f"{row['k16']}/{row['k32']} | {row['speedup']:.3f}× | "
            f"{f(row['cosine'], 6)} | {f(row['psnr'], 2)} |"
        )
    return "\n".join(lines)


def load_official_vsa_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def official_vsa_table(rows: list[dict[str, Any]]) -> str:
    dense = {
        int(row["sequence_length"]): float(row["median_ms"])
        for row in rows
        if row["family"] == "torch_sdpa_dense"
    }
    sparse: dict[tuple[int, float], dict[str, Any]] = {}
    for row in rows:
        if row["family"] != "official_vsa_triton64":
            continue
        key = (int(row["sequence_length"]), round(float(row["actual_density"]), 6))
        # N=256 quantizes requested 12.5% to actual 25%, and requested
        # 87.5% to actual 100%. Keep one row per executable density.
        sparse.setdefault(key, row)
    lines = [
        "| N tokens | dense SDPA ms | 实际保留率 | VSA ms | VSA/dense 加速 | 判断 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for (n, density), row in sorted(sparse.items()):
        speedup = float(row["speedup_vs_dense"])
        verdict = "更快" if speedup > 1.02 else ("近似持平" if speedup >= 0.98 else "更慢")
        lines.append(
            f"| {n:,} | {dense[n]:.4f} | {100.0 * density:.1f}% | "
            f"{float(row['median_ms']):.4f} | {speedup:.3f}× | {verdict} |"
        )
    return "\n".join(lines)


def svg_vsa_density(rows: list[dict[str, Any]], output: Path) -> None:
    points = [row for row in rows if row["family"] == "official_vsa_triton64"]
    if not points:
        return
    unique: dict[tuple[int, float], dict[str, Any]] = {}
    for row in points:
        key = (int(row["sequence_length"]), round(float(row["actual_density"]), 6))
        unique.setdefault(key, row)
    width, height = 1050, 650
    left, right, top, bottom = 85, 35, 45, 75
    xmin, xmax, ymin, ymax = 0.1, 1.0, 0.4, 9.0
    colors = ["#dc2626", "#ea580c", "#ca8a04", "#059669", "#2563eb"]
    n_values = sorted({key[0] for key in unique})

    def xmap(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def ymap(y: float) -> float:
        lo, hi = math.log(ymin), math.log(ymax)
        return top + (hi - math.log(max(ymin, y))) / (hi - lo) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif}.tick{font-size:13px;fill:#334155}.title{font-size:21px;font-weight:700}</style>',
        '<text x="85" y="28" class="title">Official FastVideo VSA Triton64: speedup vs retained KV blocks</text>',
    ]
    for y in (0.5, 0.75, 1.0, 2.0, 4.0, 8.0):
        py = ymap(y)
        stroke = "#dc2626" if y == 1.0 else "#e2e8f0"
        width_line = 2 if y == 1.0 else 1
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="{stroke}" stroke-width="{width_line}"/>')
        parts.append(f'<text x="{left-9}" y="{py+4:.1f}" text-anchor="end" class="tick">{y:g}×</text>')
    for density in (0.125, 0.25, 0.5, 0.75, 0.875, 1.0):
        px = xmap(density)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+23}" text-anchor="middle" class="tick">{100*density:g}%</text>')
    for index, n in enumerate(n_values):
        color = colors[index % len(colors)]
        series = sorted(
            (row for (seq, _), row in unique.items() if seq == n),
            key=lambda row: float(row["actual_density"]),
        )
        coords = [
            (xmap(float(row["actual_density"])), ymap(float(row["speedup_vs_dense"])))
            for row in series
        ]
        parts.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
            color, " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        ))
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = 62 + index * 23
        parts.append(f'<line x1="{width-175}" y1="{legend_y}" x2="{width-150}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-143}" y="{legend_y+4}" class="tick">N={n:,}</text>')
    parts.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-20}" text-anchor="middle" class="tick">retained KV blocks (actual density)</text>')
    parts.append(f'<text x="22" y="{(top+height-bottom)/2:.1f}" transform="rotate(-90 22 {(top+height-bottom)/2:.1f})" text-anchor="middle" class="tick">kernel speedup vs dense SDPA (log scale)</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def cascade_aggregate_table(rows: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["confirm"]:
            groups.setdefault(row["name"], []).append(row)
    lines = [
        "| setting | 确认runs | speedup中位数 [min,max] | cosine均值/min | PSNR均值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, items in sorted(groups.items()):
        speed = [float(item["speedup"]) for item in items]
        cosine = [float(item["cosine"]) for item in items if item["cosine"] is not None]
        psnr = [float(item["psnr"]) for item in items if item["psnr"] is not None]
        lines.append(
            f"| `{name}` | {len(items)} | {statistics.median(speed):.3f}× "
            f"[{min(speed):.3f},{max(speed):.3f}] | "
            f"{statistics.fmean(cosine):.6f}/{min(cosine):.6f} | "
            f"{statistics.fmean(psnr):.2f} |"
        )
    return "\n".join(lines)


def cascade_report_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    detail_order = {name: index for index, (name, _, _) in enumerate(CASCADE_RUNS)}
    rows = sorted(rows, key=lambda row: (detail_order[row["run"]], row["name"]))
    return [
        "## 新增重点实验：8→16→32 原生级联 route 与 FastVideo 风格 packed kernel",
        "",
        "### 这轮到底测试了什么",
        "",
        "- `triton_dense_score2_all`：16/32 都不做空间删除，只做同 stage 的 probability K=2 复用，即刷新步计算 A=softmax(QK)，下一步计算 A_old V_current。它是判断空间稀疏是否真的额外有益的关键 control。",
        "- `legacy_cascade_safe_k2`：8-stage 最后得到的 route 决定 16-stage block；16-stage 的稀疏结果再决定 32-stage block。stage 内不做 dense route refresh；16/32 都每2步刷新一次已选 block 内的 A。目标密度为93.75%/87.5%，加 self block 后实测约93.9%/87.6%。",
        "- `packed_cascade_safe_k2`：数学与 legacy-safe 完全相同，只把选中 block 打包成较大的 tile，用 FastVideo/VSA kernel 同类的 online-softmax Triton 计算，`kv_tile=64`。",
        "- `*_cascade_safe_nocache`：与 safe 级联保留完全相同的空间 route，但每一步都重新计算当前 QK/softmax（K=1）。它专门用来隔离困难 seed 中的误差究竟来自空间删 block，还是 K=2 probability cache。",
        "- `*_cascade_keep125/1875/50_nocache`：16和32两个stage都严格使用上一stage给出的固定route，只保留12.5%、18.75%、50%；不强制self block、不复用A、不做current-fine route refresh、不加centroid/residual。当前每stage有16个空间route blocks，所以严格20%无法表示，最接近的是3/16=18.75%。",
        "- `hybrid32_cascade_safe_k2`：16-stage 保留适合小 query-block 的 legacy micro-kernel，32-stage 改用 `kv_tile=32` 的 packed kernel。`balanced` 把目标密度进一步降到87.5%/75%。",
        "- 所有图片均使用 Unified XL、DDPM 80/80/90、CFG 1.5、同 class/同初始噪声/同 transition noise。表中 cosine/PSNR 是相对 dense 的逐像素指标，不等于 FID。",
        "",
        "### 每一轮的原始结果",
        "",
        cascade_detail_table(rows),
        "",
        "`explore-2cls` 只有2类且计时重复1次，只用于筛参数；下表聚合只使用4-class确认轮。由于同机 wall-clock 波动明显，应同时报告区间，不能只挑最快一次。",
        "",
        "### 4-class 确认轮聚合",
        "",
        cascade_aggregate_table(rows),
        "",
        "### 客观结论",
        "",
        "- safe 级联在 seed 20260826/20260827 上保持主体与布局，但在困难 seed 20260828 上发生构图分叉：legacy/packed K=2 cosine 约0.943。它不是一个跨 seed 稳健的 training-free 配置。balanced 更不稳定。",
        "- 困难 seed 的无-cache隔离仍只有 cosine 约0.9415–0.9419，说明主因是固定上一 stage route 删错关键 block，而不是 K=2 probability cache。作为对照，同 seed 的纯 dense-K2 为0.962，证明概率复用也有误差，但不是级联失真的主要来源。",
        "- safe 实际只删除约6.1%的16-stage block和12.4%的32-stage block，空间稀疏很弱；即使如此也会在某些 seed 触发明显轨迹分叉。当前不能把固定边界 route 直接作为无需训练的默认推理方法。",
        "- 低密度完整级联给出了更明确的负结果：12.5% 的 legacy/packed cosine约0.792，18.75%约0.807，50%约0.886；12.5/18.75%基本破坏主体，50%虽保留语义但纹理和构图明显损坏。三档端到端速度都约等于或慢于dense，说明当前2D短序列里索引/调度开销大于稀疏QK/AV节省。",
        "- 三轮确认中，dense-K2、legacy-safe-K2、packed-safe-K2 的 wall-clock 区间高度重叠，且排名会反转。结合 high-density 的小算术节省，当前不能声称 coarse-route sparsity 在2D短序列上带来稳定的额外净加速。",
        "- packed64 在算子级 microbenchmark 能减少 legacy kernel 的索引/softmax开销，但 16-stage 每个 route block 只有4个 query tokens，填充到较大计算 tile 会浪费算力；packed32 hybrid 也没有稳定胜过 legacy。短序列 N=64/256、高保留率下，Flash/SDPA 或小型 micro-kernel 已经很强。",
        "- 因而这轮真正的硬件结论是：代码路径已实现且数值正确，但**当前2D尺度没有足够长的序列，也没有足够高且跨 seed 安全的稀疏率，让 FastVideo 风格 packed kernel 稳定形成额外端到端优势**。下一步应先解决 route 可靠性（current-fine轻量刷新、teacher route蒸馏或sparse finetune），再把同一kernel移到视频长序列；不应继续对4-token query block做过拟合优化。",
        "",
        "### 对比图片",
        "",
        "![Packed cascade exploratory sweep](../../eval_results/rdd_packed_cascade_search/triton_hierarchical_comparison.png)",
        "",
        "![Packed layout confirmation](../../eval_results/rdd_packed_layout_confirm/seed_20260826/triton_hierarchical_comparison.png)",
        "",
        "![Hybrid32 confirmation](../../eval_results/rdd_hybrid32_confirm/seed_20260827/triton_hierarchical_comparison.png)",
        "",
        "![Final packed-vs-legacy confirmation](../../eval_results/rdd_packed_final_confirm/seed_20260828/triton_hierarchical_comparison.png)",
        "",
        "![No-cache isolation on the difficult seed](../../eval_results/rdd_nocache_isolation/seed_20260828/triton_hierarchical_comparison.png)",
        "",
        "![Strict cascade 12.5, 18.75 and 50 percent](../../eval_results/rdd_strict_cascade_density_sweep/seed_20260828/triton_hierarchical_comparison.png)",
        "",
    ]


def method_combo(row: dict[str, Any]) -> str:
    """Return the explicitly defined method primitives used by one setting."""
    family = row["family"]
    name = row["name"]
    setting = row.get("setting") or {}
    if family in {"adaptive_probability", "probability_tuning16"}:
        return "P 概率复用" + (" + A 自适应刷新" if name.startswith("adaptive_") else "")
    if family == "cfg_probability_share":
        interval = int(setting.get("interval", 1))
        return "G CFG共享" + (" + P 跨步复用" if interval > 1 else "")
    if family == "true_kernel":
        if name.startswith("dense_time_score_"):
            return "P 概率复用 + H 时间分段周期"
        if "route_refresh" in name:
            return "R 跨stage路由 + S 块稀疏 + P 概率复用 + F fine路由刷新"
        if "residual" in name:
            return "R 跨stage路由 + S 块稀疏 + E 输出残差"
    if family in {"centroid_oracle", "centroid_tuning16"}:
        methods = ["R 跨stage路由"]
        methods.append("C centroid补偿" if "centroid" in name else "S 仅删除(oracle)")
        if setting.get("temporal_route_update"):
            methods.append("U 本stage路由更新")
        if setting.get("dense_warm_fraction", 0) or setting.get("dense_warm_layers", 0):
            methods.append("W dense预热")
        if setting.get("route_mass_threshold", 0):
            methods.append("M Top-P route")
        if setting.get("variance_route_beta", 0):
            methods.append("V 方差加权route")
        return " + ".join(methods)
    return "未分类"


def setting_parameters(row: dict[str, Any]) -> str:
    """Human-readable, exact parameter decode for one experiment row."""
    family = row["family"]
    name = row["name"]
    s = row.get("setting") or {}
    if family in {"adaptive_probability", "probability_tuning16"}:
        stages = "/".join(str(x) for x in s.get("stages", []))
        layers = int(s.get("layers", 28))
        if name.startswith("adaptive_"):
            return (
                f"stage={stages}; 前{layers}层; tau={f(s.get('relative_rms_threshold'), 2)}; "
                f"最多{s.get('max_interval')}步不刷新; sketch_stride={s.get('sketch_stride')}"
            )
        base = int(s.get("fixed_interval", 0))
        periods = []
        for resolution in s.get("stages", []):
            override = s.get(f"fixed_interval{resolution}")
            periods.append(f"K{resolution}={base if override is None else int(override)}")
        return f"stage={stages}; 前{layers}层; " + ", ".join(periods) + "; 最后一步强制刷新"
    if family == "cfg_probability_share":
        stages = "/".join(str(x) for x in s.get("stages", []))
        return f"stage={stages}; 前{s.get('layers')}层; timestep K={s.get('interval')}; cond的A供uncond使用"
    if family == "true_kernel":
        if name.startswith("dense_time_score_"):
            intervals = s.get("time_score_intervals16") or s.get("time_score_intervals32")
            return (
                f"16/32均100% dense; 前28层; stage时间前20%/中65%/后15%的A刷新周期="
                f"{tuple(intervals)}"
            )
        parts = [
            f"density16={pct(s.get('density16', 1.0))}",
            f"density32={pct(s.get('density32', 1.0))}",
        ]
        if s.get("middle_density16") is not None:
            parts.append(f"中间层density16={pct(s['middle_density16'])}")
        if s.get("middle_density32") is not None:
            parts.append(f"中间层density32={pct(s['middle_density32'])}")
        parts.append(f"block16={s.get('block_side16', 4)}x{s.get('block_side16', 4)}")
        parts.append(f"block32={s.get('block_side32', 4)}x{s.get('block_side32', 4)}")
        if "route_refresh" in name:
            parts.append(f"A刷新K16={s.get('score_interval16', 1)}, K32={s.get('score_interval32', 1)}")
            parts.append(
                f"pooled-QK route刷新R16={s.get('pooled_route_refresh16', 0)}, "
                f"R32={s.get('pooled_route_refresh32', 0)}"
            )
        if "residual" in name:
            parts.append(
                f"dense residual刷新R16={s.get('residual_refresh16', 0)}, "
                f"R32={s.get('residual_refresh32', 0)}"
            )
        parts.append("强制self block" if s.get("add_self_block", True) else "不强制self block")
        return "; ".join(parts)
    if family in {"centroid_oracle", "centroid_tuning16"}:
        cascade = s.get("density16", 1.0) < 1.0
        parts = ["16和32级联" if cascade else "仅32 stage"]
        threshold = float(s.get("route_mass_threshold", 0.0))
        if threshold > 0:
            parts.append(f"每个query保留累计route mass达到{pct(threshold)}的最小block集合")
        else:
            if cascade:
                parts.append(f"density16={pct(s.get('density16', 1.0))}")
            parts.append(f"density32={pct(s.get('density32', 1.0))}")
        if cascade:
            parts.append(f"block16={s.get('block_side16', 4)}x{s.get('block_side16', 4)}")
        parts.append(f"block32={s.get('block_side32', 4)}x{s.get('block_side32', 4)}")
        if s.get("centroid_compensation"):
            parts.append("未选block用当前fine K/V均值补偿")
        else:
            parts.append("未选block直接删除")
        warm_fraction = float(s.get("dense_warm_fraction", 0.0))
        warm_layers = int(s.get("dense_warm_layers", 0))
        if warm_fraction:
            parts.append(f"stage前{pct(warm_fraction)} calls全层dense")
        if warm_layers:
            parts.append(f"前{warm_layers}个Transformer block始终dense")
        beta = float(s.get("variance_route_beta", 0.0))
        if beta:
            parts.append(f"K/V块内误差route权重beta={beta:g}")
        parts.append("逐步用本stage的A更新route" if s.get("temporal_route_update") else "整stage保持边界route")
        parts.append("强制self block" if s.get("add_self_block", True) else "不强制self block")
        return "; ".join(parts)
    return "—"


def detailed_table(rows: list[dict[str, Any]], measured: bool = True) -> str:
    speed_header = "| speedup" if measured else ""
    speed_rule = "|---:" if measured else ""
    lines = [
        f"| 实验批次(family) | setting | 方法组合 | 精确参数 {speed_header} | cosine mean | cosine min | PSNR |",
        f"|---|---|---|---{speed_rule}|---:|---:|---:|",
    ]
    for row in rows:
        speed = f"| {f(row.get('speedup_median'))}× " if measured else ""
        lines.append(
            f"| {row['family']} | `{row['name']}` | {method_combo(row)} | {setting_parameters(row)} {speed}| "
            f"{f(row.get('image_cosine_mean'), 6)} | {f(row.get('image_cosine_min'), 6)} | "
            f"{f(row.get('image_psnr_db_mean'), 2)} |"
        )
    return "\n".join(lines)


def md_to_html(md: str) -> str:
    """Small renderer for this generated report; no JavaScript or MathJax needed."""
    lines = md.splitlines()
    out: list[str] = []
    in_table = False
    in_code = False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                out.append("</code></pre>")
            else:
                out.append("<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not in_table:
                out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(c).replace('`','')}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            out.append(f"<p class=bullet>• {inline_html(line[2:])}</p>")
        elif line.startswith("![") and "](" in line:
            alt = line[2 : line.index("](")]
            src = line[line.index("](") + 2 : -1]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline_html(line)}</p>")
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


def inline_html(text: str) -> str:
    escaped = html.escape(text)
    # Generated text uses only simple Markdown links and bold spans.
    import re

    escaped = re.sub(
        r"\[([^]]+)\]\(((?:https?://|\.\.?/)[^)]+)\)",
        r'<a href="\2">\1</a>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("eval_results/rdd_training_free_attention_summary.json"))
    parser.add_argument("--cost-model", type=Path, default=Path("eval_results/rdd_video_attention_cost_model.json"))
    parser.add_argument(
        "--vsa-benchmark",
        type=Path,
        default=Path("eval_results/rdd_vsa_kernel_density_bench/metrics.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research_docs"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    costs = json.loads(args.cost_model.read_text(encoding="utf-8"))
    rows = summary["variants"]
    cascade_rows = load_cascade_rows()
    vsa_rows = load_official_vsa_benchmark(args.vsa_benchmark)
    svg = Path("eval_results/rdd_training_free_attention_pareto.svg")
    svg_scatter(rows, svg)
    vsa_svg = Path("eval_results/rdd_vsa_kernel_density_bench/vsa_density_speedup.svg")
    svg_vsa_density(vsa_rows, vsa_svg)

    measured = sorted(
        [r for r in rows if r["family"] not in {"centroid_oracle", "centroid_tuning16"}],
        key=lambda r: (-(r.get("image_cosine_mean") or 0), -(r.get("speedup_median") or 0)),
    )
    centroid = sorted(
        [r for r in rows if r["family"] in {"centroid_oracle", "centroid_tuning16"}],
        key=lambda r: -(r.get("image_cosine_mean") or 0),
    )
    fixed_probability = sorted(
        [
            r
            for r in measured
            if r["family"] in {"adaptive_probability", "probability_tuning16"}
            and r["name"].startswith("fixed_")
        ],
        key=lambda r: (r["family"], r["name"]),
    )
    adaptive_probability = sorted(
        [
            r
            for r in measured
            if r["family"] == "adaptive_probability" and r["name"].startswith("adaptive_")
        ],
        key=lambda r: r["name"],
    )
    time_phase_probability = sorted(
        [r for r in measured if r["family"] == "true_kernel" and r["name"].startswith("dense_time_score_")],
        key=lambda r: r["name"],
    )
    sparse_route = sorted(
        [r for r in measured if r["family"] == "true_kernel" and "route_refresh" in r["name"]],
        key=lambda r: r["name"],
    )
    sparse_residual = sorted(
        [r for r in measured if r["family"] == "true_kernel" and "residual" in r["name"]],
        key=lambda r: r["name"],
    )
    cfg_share = sorted(
        [r for r in measured if r["family"] == "cfg_probability_share"],
        key=lambda r: r["name"],
    )
    grouped = (
        fixed_probability
        + adaptive_probability
        + time_phase_probability
        + sparse_route
        + sparse_residual
        + cfg_share
    )
    if {id(row) for row in grouped} != {id(row) for row in measured}:
        missing = [row["name"] for row in measured if id(row) not in {id(item) for item in grouped}]
        raise RuntimeError(f"unclassified measured settings: {missing}")
    bounds = costs["conditional_speed_bounds"]
    flops = costs["example_multiresolution_pair_flops"]
    counts = summary["complete_seed_counts"]

    sections = [
        "# RDD 视频 attention 加速：确定性边界、VSA 对比与 training-free 实验",
        "",
        "## 先给确切答案",
        "",
        "- **RDD 用到视频上，在固定网络、固定总采样步数和固定 stage token schedule 时，一定减少算术量；但在同等视频质量下是否一定减少端到端时间，不能在视频实验前声称为定理。** 质量可能要求更多训练或更多采样步；数据搬运、VAE、通信和 kernel 利用率也会影响真实时间。",
        "- **若比较的是 RDD+VSA 与 RDD 原生上一 stage route，在相同 fine sparsity、相同输出质量、相同 block-sparse kernel、且原生 route 不需要额外频繁刷新时，RDD 原生 route 严格更省。** 上一 stage attention 本来就是 RDD 生成的一部分；VSA 则每个 fine layer、每个 timestep 都要对当前 fine Q/K/V 再做 pooling、coarse attention、Top-K 和 coarse output。",
        f"- 但这个增量优势没有想象中大。VSA 论文报告：87.5% sparsity 时 coarse router 约占 attention runtime 的 14%。理想地完全删掉它，attention 子系统相对 VSA 最多约 **{bounds['rdd_native_vs_vsa_attention_speedup_if_same_sparse_kernel']:.3f}×**；VSA 优化后 attention 只占端到端约 20%，因此额外端到端上界约 **{bounds['rdd_native_vs_vsa_e2e_speedup_if_attention_is_given_fraction']:.3f}×**。这是“完全不再算 current-fine coarse branch”的上界；若质量补偿需要每层重新计算当前 fine centroid QK/V，则会加回 VSA coarse branch 的大部分工作。RDD 真正可能带来大收益的是多分辨率生成本身，而不是只省掉 VSA router。",
        "- **同 stage 复用 attention probability 可以继续提速，但 2D 中有效的完整 A-cache 不能直接当成视频方案。** 长视频的 A 是 O(H·rho·N²·layers) 显存；应优先复用 O(H·M·K) 的 block route，或只在少数层/短窗口缓存 probability。",
        "",
        "## 为什么 RDD 的多分辨率本身有确定 FLOPs 优势",
        "",
        "固定采样步分配时，attention pair 计算量近似：",
        "```",
        "C_RDD = Σ_s L_s · N_s²",
        "C_all_fine = (Σ_s L_s) · N_fine²",
        "```",
        f"示例 token 数 {flops['stage_tokens']}、步数 {flops['stage_steps']} 时，RDD 的 pairwise-attention FLOPs 仅为全程 fine-resolution 的 **{flops['rdd_over_all_fine_pair_flops']:.3f}**。这是算术量结论，不等于同质量端到端 speedup。",
        "",
        "## RDD 原生 route 和 VSA 的真实区别",
        "",
        "- VSA 对**当前 fine hidden state**做 cube pooling，当前 timestep 重新计算 coarse Qc/Kc/Vc、Ac、Top-K，并把 Oc 和 fine sparse output 经过 learned gate 融合；这不是 training-free。",
        "- RDD 原生方案把**生成过程中本来已经算过的上一 resolution attention**变成下一 stage 的候选 block route；fine stage 只在选中的 block 内算当前 fine QK/AV。",
        "- 因而在前述强假设下，RDD-native = VSA 减去 online coarse router；但若跨 stage route 漂移严重，需要 current-fine refresh 或保留更多 block，优势会缩小甚至消失。我们的 2D 结果正说明了这个限制。",
        "- VSA 的 Oc 还承担被删区域的全局信息。若 RDD 用当前 fine K/V centroid 做同样的 global tail，就不再是“完全免费 coarse”；若直接复用上一 stage 的 coarse summary，计算最省，但过去 2D probe 中 parent-summary 明显弱于 current-fine summary。因此‘稳定快于 VSA’必须以同质量实验为条件。",
        "",
        "## 为什么 FastVideo/VSA 能明显加速，而当前 RDD 2D 稀疏几乎不能",
        "",
        "- 序列长度完全不是一个量级。RDD Unified XL 的8/16/32 latent经过patch=2后只有16/64/256 tokens；FastVideo官方VSA kernel benchmark使用 `q_seq_lens=49152`。后者比RDD最细stage长192倍。",
        "- QKV/MLP等线性层成本随N增长，attention QK/AV随N²增长。因此N=256时QK/AV只占很小部分，删掉attention pairs并不会删掉QKV projection、output projection和MLP；N约5万时二次attention才成为主要瓶颈。",
        "- VSA不是training-free硬删dense模型的route。它把coarse router、fine sparse attention和coarse/fine gate一起训练或微调，让模型适应高稀疏率；当前RDD实验直接复用原模型，固定上一stage route在部分seed会漏掉决定构图的block。",
        "- VSA论文应区分两个数字：单独VSA把Wan-2.1 attention约加速6×，端到端从31秒降到18秒，约1.72×；FastVideo文档中的几十倍denoising加速还叠加了3-step DMD/sparse-distill，不能全部归因于稀疏attention。",
        "- 原始依据：[VSA论文](https://arxiv.org/abs/2505.13389)、[FastVideo VSA文档](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md)、[49,152-token kernel benchmark](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/README.md)、[DMD sparse-distill配置](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md)。",
        "",
        "### 当前RDD kernel和官方VSA kernel是不是同一个",
        "",
        "- 不是同一份代码。当前RDD的`packed64`是VSA风格实现：同样接收每个query block的selected key-block indices、用online softmax只计算选中QK/AV，但它支持2x2或4x4空间micro-block，并把若干小block打包到64-token compute tile。",
        "- 官方FastVideo Triton VSA固定以64-token Q/KV blocks执行，输入是`q2k_idx`、`q2k_num`和`variable_block_sizes`，Q/KV长度必须是64的倍数。当前RDD 16-stage只有64 tokens，因此官方VSA只看到1个block，无法稀疏；32-stage有256 tokens，也只有4个block，最小非零保留率是25%。这是2D实验必须使用更小自定义block的原因。",
        "- route来源与执行kernel可以解耦。到RDD video时，把上一resolution attention lift到fine的64-token时空cube网格，转换成VSA的三组metadata后，可以直接调用官方`block_sparse_attn_from_indices`；执行kernel不在乎route来自VSA current-fine coarse-router还是RDD上一stage。",
        "- 但替换kernel只解决速度，不解决质量。低密度实验中legacy与packed输出cosine几乎完全相同，却一起崩坏，说明误差来自route遗漏和模型未适应稀疏attention，而不是Triton数值实现。视频方案仍需要coarse/global输出补偿、route refresh或sparse finetune。",
        "",
        "### 官方VSA kernel会不会在不够稀疏时反而变慢：实测答案是会",
        "",
        "以下是同一张RTX PRO 6000 Blackwell上直接加载FastVideo官方、未经修改的Triton64 forward kernel，与PyTorch dense SDPA的CUDA-event中位数。route metadata已预先构造，不计router成本；测试时该GPU还在运行训练，因此绝对毫秒数只能视作共享GPU初测，但跨N一致的交叉趋势非常明确。1.0×以下表示VSA比dense更慢。",
        "",
        "![官方VSA密度—速度交叉曲线](../../eval_results/rdd_vsa_kernel_density_bench/vsa_density_speedup.svg)",
        "",
        official_vsa_table(vsa_rows) if vsa_rows else "尚未找到官方VSA benchmark结果。",
        "",
        "- N>=1,024时，12.5%和25%保留率稳定更快；50%仍约1.06–1.30×；75%开始全面跌到0.70–0.88×。因此官方VSA也存在清晰的稀疏率交叉点，不是‘只要调用稀疏kernel就不会慢’。",
        "- 100%保留时，官方VSA只有dense SDPA约0.51–0.67×；原因是它做了全部QK/AV后仍支付block-index读取、动态稀疏循环和更弱的连续访存/调度效率。",
        "- N=256时官方64-token block只有4个KV blocks，最低非零密度25%；25%约1.54×、50%约1.23×、75%约1.04×、100%约0.90×。它并没有提供RDD 2D想要的12.5%细粒度route。",
        "- 当前RDD 16-token packed kernel在N=256的25%/50%/75%初测为2.03×/1.29×/1.08×，与官方VSA的交叉形状一致；这不支持‘RDD kernel根本写坏了’。4-token kernel在密度升高后更差，主要是query tile利用率太低。共享GPU造成个别非单调离群点，正式绝对数字需独占复测。",
        "- 因而现在没有理由把官方64-token kernel生硬魔改回RDD 2D：它会失去route粒度，且无法解决高密度本来就不划算的问题。到长视频N很大时，应直接复用官方64/128-token执行kernel，只替换route来源；若继续优化2D micro-kernel，目标应是16-token block的低密度路径，而不是追求75–100%密度胜过FlashAttention。",
        "- 原始数据：[metrics.json](../../eval_results/rdd_vsa_kernel_density_bench/metrics.json)；官方实现：[FastVideo block-sparse API](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py)、[Triton64 kernel](https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py)。",
        "",
        "## 方法原语：先明确每个字母到底做什么",
        "",
        "下面的字母是本报告内部的短名称，不是新的模型名。一个 setting 可以组合多个原语；例如 `R + S + P + F` 就是“上一 stage 给初始 route、按 block 稀疏计算、同 stage 复用 A、并周期性用 current-fine pooled QK 刷新 route”。",
        "",
        "| 编号 | 方法 | 实际做法 | 主要参数 |",
        "|---|---|---|---|",
        "| R | 跨 stage route | 用 8-stage 末尾的 attention 决定 16-stage 候选 block，再用 16-stage 末尾 attention 决定 32-stage 候选 block | density、block_side、self block、route是否在stage内更新 |",
        "| S | block-sparse attention | 只对 route 选中的 query-key block 运行 QK/softmax/AV | density16/32、block_side16/32 |",
        "| P | 同 stage probability 复用 | 缓存 A=softmax(QK)，后续跳过Q/K并计算 A_old V_current | stage、layers、刷新周期K |",
        "| F | current-fine route 刷新 | 对当前 fine Q/K 做 block pooling，每R步重新计算一次候选 route | pooled_route_refresh16/32 |",
        "| C | centroid/global-tail 补偿 | 选中block逐token精算；未选block用当前fine K/V均值表示 | density或route_mass、block_side |",
        "| E | 输出残差补偿 | 每R步算一次 O_dense-O_sparse，随后加到 sparse 输出 | residual_refresh16/32 |",
        "| G | CFG probability 共享 | conditional 和 unconditional 分支共用同一个 A | stage、layers、跨步周期K |",
        "| A | 自适应刷新判据 | latent相对RMS变化小于tau时命中P-cache，否则刷新 | tau、max_interval、sketch_stride |",
        "| H | 时间分段周期 | 在stage前20%/中65%/后15%采用不同的A刷新周期 | time_score_intervals=(early,middle,late) |",
        "| U | 本 stage route 更新 | 第一次由上一stage给route，后续用本stage上一步的attention更新route | temporal_route_update |",
        "| W | dense warm-up | stage开头一段calls或模型最前几层保持dense | dense_warm_fraction、dense_warm_layers |",
        "| M | Top-P route | 每个query保留累计route mass达到阈值的最小block集合 | route_mass_threshold |",
        "| V | 方差加权route | K/V块内越不均匀，越提高该block被精算的优先级 | variance_route_beta |",
        "",
        "### family 只是实验批次，不等于一种方法",
        "",
        "| family | 包含的方法 | 数据规模 | 是否可用真实时间 |",
        "|---|---|---|---|",
        "| adaptive_probability | P固定周期，或P+A自适应周期 | 第一轮4 classes | 是；完整A-cache只适合作为2D control |",
        "| probability_tuning16 | 与adaptive_probability相同，但专门细调16/32 stage的K和层数 | 第二轮16 classes | 是；tuning16指16个类别，不是16x16分辨率 |",
        "| true_kernel | P+H，或R+S+P+F，或R+S+E | 第一轮4 classes | 是；空间稀疏配置走实际Triton kernel |",
        "| cfg_probability_share | G，部分配置再叠加跨timestep的P | 第一轮4 classes | 是 |",
        "| centroid_oracle | R+C及其W/M/V变体 | 第一轮4 classes | 否；先构造dense logits，只看质量上限 |",
        "| centroid_tuning16 | 与centroid_oracle相同的扩大测试 | 第二轮16 classes | 否；tuning16仍表示16个类别 |",
        "",
        "### 通用参数中最容易混淆的地方",
        "",
        "- `density32=0.5`：目标是每个 query block 保留约50%的 key blocks；强制 self block 后实际密度可能略高。",
        "- `block_side32=4`：一个空间 block 是4x4=16个tokens，不是总共4个block。",
        "- `score_interval=3`：每3个 denoiser calls 重算一次 A；中间使用 A_old V_current。",
        "- `score_layers=7`：只在前7个 Transformer blocks 复用A；不是前7个采样steps。",
        "- `middle_density` 和 `middle_score_interval`：这里的 middle 指模型深度15%到85%的 Transformer blocks。",
        "- `time_score_intervals=(5,3,2)`：这里才是采样时间轴，表示stage前20%每5步、中65%每3步、后15%每2步刷新。",
        "- `pooled_route_refresh=6` 更新的是“算哪些block”；`score_interval=3` 更新的是“已选block里的attention probability”。",
        "- `temporal_route_update=false`：除显式的F刷新外，route保持上一stage边界给出的版本；true表示后续用本stage最近一次A更新。",
        "",
        "## 全部 training-free 实验：按方法组合重新分组",
        "",
        f"完整 seed 数：`{counts}`。speedup 是 denoiser wall-clock；cosine/PSNR 是同 class、同初始噪声、同 transition noise 相对 dense RDD 的逐像素相似度，不是 FID。centroid 分支先构造 dense logits，是质量 oracle，不能用其时间声称加速。",
        "",
        "![速度—质量 Pareto](../../eval_results/rdd_training_free_attention_pareto.svg)",
        "",
        "### P：固定周期复用 attention probability",
        "",
        "这些 setting 不做空间稀疏。它们缓存完整A，跳过部分timestep的Q/K和softmax；适合验证时间冗余，但完整A的O(N²)显存不适合作为长视频最终实现。",
        "",
        detailed_table(fixed_probability),
        "",
        "### P + A：用 latent 变化自动决定 cache hit/miss",
        "",
        detailed_table(adaptive_probability),
        "",
        "### P + H：按采样早/中/晚阶段改变复用周期",
        "",
        detailed_table(time_phase_probability),
        "",
        "### R + S + P + F：跨stage route、真实块稀疏、概率复用、current-fine route刷新",
        "",
        detailed_table(sparse_route),
        "",
        "### R + S + E：跨stage route、真实块稀疏、dense-output residual补偿",
        "",
        detailed_table(sparse_residual),
        "",
        "### G（部分再加P）：CFG cond/uncond probability共享",
        "",
        detailed_table(cfg_share),
        "",
        "### R + C：Centroid/global-tail质量oracle及其U/W/M/V组合",
        "",
        "Centroid表中的speedup故意为空：这些实验先计算dense logits再模拟稀疏，只能回答质量是否值得继续实现，不能回答速度。",
        "",
        detailed_table(centroid, measured=False),
        "",
        "## 已经能下的实验结论",
        "",
        "- 固定短周期 `A_old @ V_current` 在早期测试中最有希望：4-class round 1 的全 stage K=2 为1.146× / cosine 0.9951；16-class tuning 为1.311× / cosine mean 0.9920 / min 0.9880。但新增 seed 20260828 的 dense-K2 control 只有 cosine 0.9621，出现明显姿态/构图分叉。因此它是有收益的近似，不是跨 seed 稳健的 training-free 默认项；正式论文需要50K样本FID/precision/recall或至少更大seed集。",
        "- 全 stage K=3 更激进：16-class tuning 只有1.124× / cosine 0.9832，困难 seed 的 min 0.9734；`16-stage K=3, 32-stage K=2` 同样较差，说明 16×16 stage 的误差会被32×32继续放大。若做 training-free control，K=2仍优于K=3，但现在不能把K=2表述为无损。",
        "- 直接把 route 压到 50% 再定期用 current-fine pooled QK 刷新，只得到约 1.07×，同时 cosine 约 0.983；route refresh 没有自动解决遗漏的全局信息。",
        "- centroid compensation 显著优于直接丢弃，但 training-free 下仍需较高密度：32-stage 保留 75% 时 cosine 0.9968，62.5% 时 0.9933，50% 时 0.9896。给第一层加入 20% dense warm-up 可把 50% 提到 0.9930，但这些数值来自先构造 dense logits 的质量 oracle，不能声称真实加速。",
        "- Top-P attention-mass 路由没有胜过固定密度：mass 85/90/95 平均实际保留约 40.4%/47.7%/58.9% block，cosine 为 0.9844/0.9881/0.9928；与相近固定密度相比没有质量优势。当前不值得为 Top-P 增加动态排序和不规则 kernel 复杂度。",
        "- dense-output residual 可恢复质量，却因定期 dense refresh 和额外 kernel 调度而慢于 dense；它更像 distillation target 或误差分析工具。",
        "- CFG cond/uncond probability 共享会造成明显分布误差。只共享前 7 层接近保真，却没有净提速。",
        "- 以 latent 相对 RMS 决定 cache hit 的自适应原型命中率低且包含同步开销；固定周期优于这个判据。",
        "",
        *cascade_report_section(cascade_rows),
        "## 视频上应该怎么组合",
        "",
        "推荐的可扩展顺序是：",
        "",
        "1. RDD 多分辨率 video denoising，先拿到基础 FLOPs 和 wall-clock 收益。",
        "2. 8→16、16→32 只传 block route，不传完整 A；进入 fine stage 后保留一个短 dense/current-fine warm-up，随后周期性刷新 route。",
        "3. 对遗漏 block 加 parameter-free centroid/global-tail；若质量仍不足，再做 sparse finetune 或 teacher distillation。",
        "4. 同 stage 首先复用 route；只在 token 较少的 coarse stage、少量层或短时间窗口复用完整 probability。",
        "5. 写融合 kernel：route lifting + Top-K/index packing + sparse QK/AV；不要把 Python gather 和 host-side decision 算进最终方法。",
        "",
        "## Probability cache 的视频显存警告",
        "",
        "以 fp16、12 heads、30 layers、12.5% density 为例：N=23,296 时全层 probability cache 约 45.5 GiB；coarse route cache 约 0.089 GiB；attention-output residual 约 2.0 GiB。N=61,440 时 probability cache 约 316 GiB。因此 2D 的完整 A 复用不能直接宣称为长视频方案。",
        "",
        "## 主要原始文献",
        "",
        "- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)",
        "- [DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance](https://arxiv.org/abs/2505.14708)",
        "- [DiTFastAttn: Attention Compression for Diffusion Transformer Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/0267925e3c276e79189251585b4100bf-Paper-Conference.pdf)",
        "- [SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing](https://arxiv.org/abs/2603.08982)",
        "- [FastVideo 官方实现](https://github.com/hao-ai-lab/fastvideo)",
        "",
        "## Round-1 对比图片",
        "",
        "![Centroid compensation](../../eval_results/rdd_training_free_centroid_sweep/seed_20260819/centroid_compensation_comparison.png)",
        "",
        "![True Triton kernel](../../eval_results/rdd_training_free_kernel_sweep/seed_20260819/training_free_kernel_comparison.png)",
        "",
        "![Adaptive probability cache](../../eval_results/rdd_adaptive_probability_sweep/seed_20260819/adaptive_probability_cache_comparison.png)",
        "",
        "![CFG probability sharing](../../eval_results/rdd_cfg_probability_share_sweep/seed_20260819/cfg_probability_share_comparison.png)",
        "",
        "## 16-class tuning 对比图片",
        "",
        "下图为 seed 20260822 的轻量预览；完整分辨率 PNG 保留在 surfi2 对应结果目录。其余两个 seed 的预览也已下载到本地结果目录。",
        "",
        "![Centroid tuning 16 classes](../../eval_results/rdd_centroid_tuning16/seed_20260822/centroid_compensation_comparison_preview.webp)",
        "",
        "![Probability reuse tuning 16 classes](../../eval_results/rdd_probability_tuning16/seed_20260822/adaptive_probability_cache_comparison_preview.webp)",
    ]
    md = "\n".join(sections) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "RDD_VIDEO_ATTENTION_ACCELERATION_ANALYSIS_ZH.md"
    html_path = args.output_dir / "RDD_VIDEO_ATTENTION_ACCELERATION_ANALYSIS_ZH.html"
    md_path.write_text(md, encoding="utf-8")
    body = md_to_html(md)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDD 视频 attention 加速分析</title>
<style>
body{{font-family:Segoe UI,Arial,'Microsoft YaHei',sans-serif;max-width:1180px;margin:32px auto;padding:0 24px;color:#172033;line-height:1.68;background:#f8fafc}}
h1,h2,h3{{color:#0f172a;line-height:1.25}} h1{{font-size:34px}} h2{{margin-top:42px;border-bottom:1px solid #cbd5e1;padding-bottom:8px}}
p,table,pre,figure{{background:white}} p{{padding:2px 12px;margin:7px 0}} p.bullet{{padding-left:22px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0;display:block;overflow-x:auto}} th,td{{border:1px solid #cbd5e1;padding:6px 8px;white-space:nowrap}} th{{background:#e2e8f0}}
pre{{padding:14px;border:1px solid #cbd5e1;border-radius:8px;overflow-x:auto}} code{{background:#eef2ff;padding:1px 4px;border-radius:4px}}
figure{{margin:20px 0;padding:12px;border:1px solid #dbe4ef;border-radius:10px}} img{{max-width:100%;height:auto;display:block;margin:auto}} figcaption{{text-align:center;color:#475569;margin-top:8px}}
a{{color:#1d4ed8}} strong{{color:#7c2d12}} hr{{border:0;border-top:1px solid #cbd5e1;margin:30px 0}}
</style></head><body>{body}</body></html>"""
    html_path.write_text(page, encoding="utf-8")
    print(md_path)
    print(html_path)
    print(svg)


if __name__ == "__main__":
    main()
