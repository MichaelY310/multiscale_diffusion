#!/usr/bin/env python3
"""Merge paired FastWan/RDD audit runs into paper/report-friendly JSON + CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def family(name: str) -> str:
    if name == "fixed_vsa20":
        return "baseline"
    if name.startswith("spatial"):
        return "spatial"
    if name.startswith("st"):
        return "spatiotemporal"
    return "other"


def scene(prompt: str) -> str:
    lowered = prompt.lower()
    if "fox" in lowered:
        return "fox"
    if "raincoat" in lowered or "city" in lowered:
        return "city"
    if "vintage car" in lowered or "coastal" in lowered:
        return "coast"
    return "other"


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    rows = []
    for name, result in payload["results"].items():
        timing = result["denoise_and_boundary_seconds"]
        config = result["config"]
        row = {
            "run": run_dir.name,
            "size": payload["size"],
            "frame_num": payload["frame_num"],
            "prompt": payload["prompt"],
            "scene": scene(payload["prompt"]),
            "seed": payload["seed"],
            "name": name,
            "family": family(name),
            "stage_factors": config["stage_factors"],
            "attention_mode": config["attention_mode"],
            "route_density_requested": config["route_density"],
            "reuse_every": config["reuse_every"],
            "reuse_scope": config["reuse_scope"],
            "small_stage_executor": config["small_stage_executor"],
            "small_stage_max_padded_tokens": config[
                "small_stage_max_padded_tokens"
            ],
            "boundary_executor": config.get("boundary_executor", "native"),
            "resident_dit": payload.get("protocol", {}).get("resident_dit"),
            "dit_layerwise_offload": payload.get("protocol", {}).get(
                "dit_layerwise_offload"
            ),
            "denoise_seconds_median": timing["median"],
            "denoise_seconds_mad": timing["mad"],
            "denoise_seconds_p25": timing["p25"],
            "denoise_seconds_p75": timing["p75"],
            "denoise_spread_fraction": timing["spread_fraction"],
            "denoise_speedup_vs_baseline": result[
                "denoise_speedup_vs_baseline"
            ],
            "denoise_reduction_fraction": 1.0
            - 1.0 / result["denoise_speedup_vs_baseline"],
            "stage_seconds_median": [
                item["median"] for item in result["stage_forward_seconds"]
            ],
            "stage_latent_shapes": result["stage_latent_shapes"],
            "effective_c2f_density": result["attention"].get(
                "mean_c2f_density"
            ),
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "video_file": None,
            "poster_file": None,
            "vae_decode_seconds": result.get("vae_decode_seconds"),
            "sample_plus_vae_seconds": result.get("sample_plus_vae_seconds"),
            "sample_plus_vae_speedup_vs_baseline": result.get(
                "sample_plus_vae_speedup_vs_baseline"
            ),
            "raw_records": result["records"],
            "stage_breakdown": result.get("stage_breakdown", {}),
        }
        video = result.get("video")
        if video:
            video_name = Path(video).name
            row["video_file"] = video_name
            poster = run_dir / f"{Path(video_name).stem}_mid.jpg"
            if poster.exists():
                row["poster_file"] = poster.name
        rows.append(row)
    return {
        "path": str(path),
        "protocol": payload["protocol"],
        "size": payload["size"],
        "frame_num": payload["frame_num"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
        "rows": rows,
    }


def comparison(
    rows: list[dict[str, Any]], run: str, names: list[str]
) -> dict[str, Any] | None:
    by_name = {row["name"]: row for row in rows if row["run"] == run}
    if any(name not in by_name for name in names):
        return None
    values = [by_name[name]["denoise_seconds_median"] for name in names]
    return {
        "run": run,
        "size": next(iter(by_name.values()))["size"],
        "scene": next(iter(by_name.values()))["scene"],
        "names": names,
        "denoise_seconds": values,
        "relative_to_first": [values[0] / value for value in values],
        "stage_seconds": [by_name[name]["stage_seconds_median"] for name in names],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--small-stage-microbench", type=Path)
    args = parser.parse_args()

    metric_files = sorted(args.audit_root.glob("*/paired_suite_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"no paired_suite_metrics.json under {args.audit_root}")
    runs = [load_run(path) for path in metric_files]
    rows = [row for run in runs for row in run["rows"]]
    output_json = args.output_json or args.audit_root / "efficiency_audit_summary.json"
    output_csv = args.output_csv or args.audit_root / "efficiency_audit_rows.csv"

    comparisons = []
    for run in sorted({row["run"] for row in rows}):
        for names in (
            ["spatial221_c2f20", "spatial421_c2f20", "spatial441_c2f20"],
            ["st221_c2f20", "st421_c2f20", "st441_c2f20"],
        ):
            item = comparison(rows, run, names)
            if item:
                comparisons.append(item)

    payload: dict[str, Any] = {
        "audit_root": str(args.audit_root.resolve()),
        "runs": runs,
        "rows": rows,
        "path_comparisons": comparisons,
    }
    if args.small_stage_microbench and args.small_stage_microbench.exists():
        payload["small_stage_microbench"] = json.loads(
            args.small_stage_microbench.read_text(encoding="utf-8")
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    scalar_fields = [
        key
        for key in rows[0]
        if key not in {"raw_records", "stage_seconds_median", "stage_latent_shapes"}
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=scalar_fields + ["stage_seconds_median", "stage_latent_shapes"],
        )
        writer.writeheader()
        for row in rows:
            serialized = {key: row.get(key) for key in scalar_fields}
            serialized["stage_seconds_median"] = json.dumps(row["stage_seconds_median"])
            serialized["stage_latent_shapes"] = json.dumps(row["stage_latent_shapes"])
            writer.writerow(serialized)
    print(output_json)
    print(output_csv)


if __name__ == "__main__":
    main()
