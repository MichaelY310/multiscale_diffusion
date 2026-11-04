#!/usr/bin/env python3
"""Turn one focused packed-cascade metrics.json into a readable comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    variants = data["variants"]
    dense = variants["dense"]
    dense_time = float(dense["denoiser_seconds"])
    dense_stages = dense.get("stage_seconds", {})

    rows = []
    for name, item in variants.items():
        if name == "dense":
            continue
        setting = item["setting"]
        image = item.get("image_vs_dense", {})
        stage_times = item.get("stage_seconds", {})
        measured_density = item.get("mean_selected_block_density", {})
        rows.append(
            {
                "name": name,
                "backend": setting.get("kernel_backend", "legacy"),
                "density16": setting.get("density16"),
                "density32": setting.get("density32"),
                "middle16": setting.get("middle_density16"),
                "middle32": setting.get("middle_density32"),
                "k16": setting.get("score_interval16"),
                "k32": setting.get("score_interval32"),
                "actual16": measured_density.get("16"),
                "actual32": measured_density.get("32"),
                "speedup": dense_time / float(item["denoiser_seconds"]),
                "speedup16": (
                    float(dense_stages["16"]) / float(stage_times["16"])
                    if dense_stages and stage_times
                    else None
                ),
                "speedup32": (
                    float(dense_stages["32"]) / float(stage_times["32"])
                    if dense_stages and stage_times
                    else None
                ),
                "cosine": image.get("pixel_cosine"),
                "psnr": image.get("psnr_db"),
            }
        )

    rows.sort(key=lambda x: (-(x["cosine"] or 0), -(x["speedup"] or 0)))
    lines = [
        "# RDD 8→16→32 packed cascade 小规模搜索",
        "",
        f"- classes: `{data.get('classes')}`",
        f"- seed: `{data.get('seed')}`",
        f"- dense denoiser: `{dense_time:.4f}s`",
        "- 本表速度若与其他训练共享 GPU，只能用于初筛；最终数字必须在空闲 GPU 上重跑。",
        "- `实际16/32` 包含强制 self block 后的真实平均 block density。",
        "",
        "| setting | backend | 目标16/32 | 中层16/32 | K16/K32 | 实际16/32 | 总加速 | 16-stage | 32-stage | cosine | PSNR |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['backend']} | "
            f"{fmt(row['density16'])}/{fmt(row['density32'])} | "
            f"{fmt(row['middle16'])}/{fmt(row['middle32'])} | "
            f"{row['k16']}/{row['k32']} | "
            f"{fmt(row['actual16'])}/{fmt(row['actual32'])} | "
            f"{fmt(row['speedup'])}× | {fmt(row['speedup16'])}× | "
            f"{fmt(row['speedup32'])}× | {fmt(row['cosine'], 6)} | "
            f"{fmt(row['psnr'], 2)} |"
        )

    output = args.output or args.metrics.with_name("PACKED_CASCADE_SUMMARY_ZH.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
