#!/usr/bin/env python3
"""Aggregate true 2-D RDD vs DiT reuse probes into JSON, SVG and Markdown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_rows(reports: list[dict[str, Any]], arm: str) -> list[dict[str, float]]:
    rows = []
    depth = reports[0][arm]["depth"]
    for layer in range(depth):
        items = [report[arm]["single_layer"][layer] for report in reports]
        rows.append({
            "layer": layer,
            "pred_x0_relative_l2_mean": mean([x["pred_x0"]["relative_l2"] for x in items]),
            "pred_x0_relative_l2_std": stdev([x["pred_x0"]["relative_l2"] for x in items]),
            "residual_cosine_mean": mean([x["source_vs_target_residual"]["cosine"] for x in items]),
            "low_frequency_cosine_mean": mean([x["source_vs_target_low_frequency"]["cosine"] for x in items]),
        })
    return rows


def aggregate_combinations(reports: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    result = {}
    for setting in reports[0][arm]["combinations"]:
        quality = [r[arm]["combinations"][setting]["pred_x0"]["relative_l2"] for r in reports]
        latency = [r[arm]["latency"][setting]["latency_reduction_percent"] for r in reports if setting in r[arm]["latency"]]
        result[setting] = {
            "layers": reports[0][arm]["combinations"][setting]["layers"],
            "pred_x0_relative_l2_mean": mean(quality),
            "pred_x0_relative_l2_std": stdev(quality),
            "latency_reduction_percent_mean": mean(latency) if latency else None,
            "latency_reduction_percent_std": stdev(latency) if latency else None,
        }
    return result


def aggregate_hidden(reports: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    result = {}
    layers = [
        row["layer"] for row in reports[0][arm]["single_layer"]
        if "whole_hidden_replacement" in row
    ]
    for layer in layers:
        values = [
            r[arm]["single_layer"][layer]["whole_hidden_replacement"]["pred_x0"]["relative_l2"]
            for r in reports
        ]
        result[str(layer)] = {"mean": mean(values), "std": stdev(values)}
    return result


def points(values: list[float], x0: float, y0: float, width: float, height: float, ymin: float, ymax: float) -> str:
    if len(values) == 1:
        xs = [x0]
    else:
        xs = [x0 + width * i / (len(values) - 1) for i in range(len(values))]
    ys = [y0 + height * (1.0 - (value - ymin) / max(ymax - ymin, 1e-12)) for value in values]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def build_svg(summary: dict[str, Any], output: Path) -> None:
    rdd = summary["rdd"]["layers"]
    dit = summary["dit"]["layers"]
    width, height = 1200, 430
    left, top, plot_w, plot_h = 70, 55, 500, 300
    panels = [
        ("Block residual cosine", [x["residual_cosine_mean"] for x in rdd], [x["residual_cosine_mean"] for x in dit], 0.0, 1.02),
        ("Final predicted-x0 relative L2 (log10)", [math.log10(max(x["pred_x0_relative_l2_mean"], 1e-5)) for x in rdd], [math.log10(max(x["pred_x0_relative_l2_mean"], 1e-5)) for x in dit], -3.0, 0.1),
    ]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for panel, (title, rdd_values, dit_values, ymin, ymax) in enumerate(panels):
        x0 = left + panel * 600
        svg += [
            f'<text x="{x0}" y="25" font-family="sans-serif" font-size="18" font-weight="bold">{title}</text>',
            f'<line x1="{x0}" y1="{top+plot_h}" x2="{x0+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
            f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top+plot_h}" stroke="#333"/>',
        ]
        for tick in range(6):
            value = ymin + (ymax - ymin) * tick / 5
            y = top + plot_h * (1 - tick / 5)
            svg += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#ddd"/>', f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>']
        svg += [
            f'<polyline fill="none" stroke="#d1495b" stroke-width="3" points="{points(rdd_values,x0,top,plot_w,plot_h,ymin,ymax)}"/>',
            f'<polyline fill="none" stroke="#00798c" stroke-width="3" points="{points(dit_values,x0,top,plot_w,plot_h,ymin,ymax)}"/>',
            f'<text x="{x0+plot_w/2}" y="{top+plot_h+35}" text-anchor="middle" font-family="sans-serif" font-size="13">DiT block index</text>',
        ]
        for tick in (0, 7, 14, 21, 27):
            x = x0 + plot_w * tick / 27
            svg.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick}</text>')
    svg += [
        '<rect x="850" y="390" width="16" height="4" fill="#d1495b"/><text x="872" y="397" font-family="sans-serif" font-size="13">RDD 16→32 cross-scale</text>',
        '<rect x="1030" y="390" width="16" height="4" fill="#00798c"/><text x="1052" y="397" font-family="sans-serif" font-size="13">DiT t300→299</text>',
        '</svg>',
    ]
    output.write_text("\n".join(svg), encoding="utf-8")


def build_markdown(summary: dict[str, Any], output: Path, figure: Path) -> None:
    rdd_best = sorted(summary["rdd"]["layers"], key=lambda x: x["pred_x0_relative_l2_mean"])[:8]
    dit_best = sorted(summary["dit"]["layers"], key=lambda x: x["pred_x0_relative_l2_mean"])[:8]
    lines = [
        "# 真实 ImageNet RDD 与原始 DiT：跨尺度 hidden/residual 复用诊断",
        "",
        "## 结论先行",
        "",
        "这组严格控制的 400K XL/2 实验**不支持直接把低分辨率整段 hidden 搬到高分辨率**。它也不支持不加适配地连续替换许多 fine block。RDD 的 coarse residual 确实携带 fine residual 的低频方向，但幅值、通道语义和缺失的 cell 内高频使直接替换误差明显。",
        "",
        "相反，原始 DiT 的相邻 timestep full-resolution residual 几乎不变。因此，若论文要主张 RDD 独有优势，方法必须是 `coarse low-frequency update + fine high-frequency correction`，并和普通 timestep cache 做公平比较；不能把 naive CSBR 当成已成立结果。",
        "",
        "## 公平口径",
        "",
        "- 模型：RDD-Unified-XL/2 400K 对原始 DiT-XL/2 400K；hidden size、depth、head、patch size 和训练步数匹配。为支持变分辨率，RDD 使用 RoPE，而原始 DiT 使用固定 2D sin-cos position embedding，因此这是最佳可用的 matched-capacity 对比，不应写成逐算子完全相同。",
        "- 数据：同一批 ImageNet-256 VAE latent、类别和高斯噪声。",
        "- 时间：两边都是全局 `t=300 → 299`。",
        "- RDD：16×16 residual 最近邻放大后替换 32×32 同编号 block。",
        "- DiT：32×32 的前一 timestep residual 替换当前 timestep 同编号 block。",
        "- 计算：跳过相同数量的 XL/2 block。",
        "",
        f"![逐层结果]({figure.name})",
        "",
        "## 最安全的单层（按最终 predicted-x0 相对 L2）",
        "",
        "| RDD layer | error | residual cosine | low-frequency cosine | DiT layer | error | residual cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rdd_row, dit_row in zip(rdd_best, dit_best):
        lines.append(
            f"| {rdd_row['layer']} | {rdd_row['pred_x0_relative_l2_mean']:.4f} | {rdd_row['residual_cosine_mean']:.3f} | {rdd_row['low_frequency_cosine_mean']:.3f} | {dit_row['layer']} | {dit_row['pred_x0_relative_l2_mean']:.5f} | {dit_row['residual_cosine_mean']:.5f} |"
        )
    lines += ["", "## 多层替换", "", "| arm | setting | layers | predicted-x0 relative L2 | latency reduction |", "|---|---|---|---:|---:|"]
    for arm in ("rdd", "dit"):
        for setting, row in summary[arm]["combinations"].items():
            reduction = row["latency_reduction_percent_mean"]
            reduction_text = "—" if reduction is None else f"{reduction:.1f}%"
            lines.append(f"| {arm.upper()} | {setting} | `{row['layers']}` | {row['pred_x0_relative_l2_mean']:.4f} | {reduction_text} |")
    lines += [
        "",
        "## 方法判断",
        "",
        "1. **整段 hidden replacement 失败**：它覆盖了 fine stage 已积累的全部细节，而不是只替代当前 block 的工作。",
        "2. **coarse residual 主要是低频先验**：它与 fine low-pass residual 的 cosine 系统性高于与完整 residual 的 cosine；与构造出的 high-pass 分量近似正交是投影结构本身导致，不能被误写成学到的性质。",
        "3. **RDD 的安全层在中段，不是简单的尾层规则**：这和 FastWan 伪 RDD 的层规律不同，证明真实 RDD checkpoint 必须单独校准。",
        "4. **原始 DiT timestep cache 是强基线**：相邻一步 residual cosine 约等于 1；RDD 版本若只做到相同画质/速度，并不能证明跨尺度优势。",
        "5. 下一步只值得验证两种版本：固定、跨数据可泛化的轻量 channel adapter；以及只让 coarse 提供低频、保留少量 fine blocks/高频分支。",
        "",
        f"原始聚合数据：`{Path(summary['_summary_json']).name}`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).glob("seed_*.json"))
    if not files:
        raise SystemExit("no seed_*.json files")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    figure_path = output_dir / "rdd_vs_dit_block_reuse.svg"
    report_path = output_dir / "RDD_VS_DIT_BLOCK_REUSE_DIAGNOSTIC_ZH.md"
    summary = {
        "num_seeds": len(reports),
        "num_images": sum(len(r["images"]) for r in reports),
        "source_files": [str(path) for path in files],
        "rdd": {
            "layers": aggregate_rows(reports, "rdd"),
            "combinations": aggregate_combinations(reports, "rdd"),
            "whole_hidden": aggregate_hidden(reports, "rdd"),
        },
        "dit": {
            "layers": aggregate_rows(reports, "dit"),
            "combinations": aggregate_combinations(reports, "dit"),
            "whole_hidden": aggregate_hidden(reports, "dit"),
        },
        "_summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_svg(summary, figure_path)
    build_markdown(summary, report_path, figure_path)
    print(report_path)


if __name__ == "__main__":
    main()
