#!/usr/bin/env python3
"""Summarize the paired FastWan RDD/C2F/reuse experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def median_record(data: dict, key: str) -> float:
    return statistics.median(
        float(row[key]) for row in data["steady_state_benchmark"]["records"]
    )


def row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    attention = data.get("attention", {})
    denoise = median_record(data, "denoise_and_boundary_seconds")
    sample = median_record(data, "sample_wrapper_seconds")
    vae = float(data.get("vae_decode_seconds", 0.0))
    video_file = Path(data["output"]).name
    poster_file = path.parent / f"{Path(video_file).stem}_mid.jpg"
    return {
        "name": path.name.removesuffix(".metrics.json"),
        "metrics_file": path.name,
        "video_file": video_file,
        "poster_file": poster_file.name if poster_file.exists() else None,
        "prompt": data.get("prompt"),
        "seed": data.get("seed"),
        "size": data.get("size"),
        "frame_num": data.get("frame_num"),
        "dmd_timesteps": data.get("dmd_timesteps"),
        "stage_factors": data.get("stage_factors_spatial_temporal"),
        "stage_latent_shapes": data.get("stage_latent_shapes"),
        "transition_mode": data.get("transition_mode"),
        "keep_nominal": data.get("attention_pair_keep_fraction_nominal"),
        "attention_mode": attention.get("attention_mode"),
        "route_density_requested": attention.get("route_density_requested"),
        "effective_c2f_density": attention.get("mean_c2f_density"),
        "reuse_every": attention.get("reuse_every"),
        "reuse_scope": attention.get("reuse_scope"),
        "online_vsa_calls": attention.get("online_vsa_calls"),
        "c2f_sparse_calls": attention.get("c2f_sparse_calls"),
        "c2f_seed_vsa_calls": attention.get("c2f_seed_vsa_calls"),
        "projected_reuse_hits": attention.get("projected_reuse_hits"),
        "denoise_seconds_median": denoise,
        "sample_seconds_median": sample,
        "vae_seconds": vae,
        "sample_plus_vae_seconds": sample + vae,
        "peak_cuda_memory_bytes": data.get("peak_cuda_memory_before_decode_bytes"),
    }


def average_baseline(a: dict, b: dict) -> dict:
    return {
        "denoise": (a["denoise_seconds_median"] + b["denoise_seconds_median"]) / 2,
        "sample": (a["sample_seconds_median"] + b["sample_seconds_median"]) / 2,
        "vae": (a["vae_seconds"] + b["vae_seconds"]) / 2,
        "sample_plus_vae": (
            a["sample_plus_vae_seconds"] + b["sample_plus_vae_seconds"]
        )
        / 2,
        "drift_fraction": abs(
            a["denoise_seconds_median"] - b["denoise_seconds_median"]
        )
        / ((a["denoise_seconds_median"] + b["denoise_seconds_median"]) / 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    rows = {
        item["name"]: item
        for item in (row(path) for path in sorted(root.glob("*.metrics.json")))
    }

    final_pre = rows["final_fixed_vsa20_pre"]
    final_post = rows["final_fixed_vsa20_post"]
    final_baseline = average_baseline(final_pre, final_post)
    primary_names = [
        "final_temporal_vsa20",
        "final_temporal_c2f20",
        "final_temporal_c2f20_reuse2",
    ]
    primary = []
    for name in primary_names:
        item = dict(rows[name])
        item["denoise_speedup_vs_fixed_vsa20"] = (
            final_baseline["denoise"] / item["denoise_seconds_median"]
        )
        item["sample_plus_vae_speedup_vs_fixed_vsa20"] = (
            final_baseline["sample_plus_vae"] / item["sample_plus_vae_seconds"]
        )
        primary.append(item)

    density_rows = []
    for tag, density in (("50", 0.50), ("25", 0.25), ("20", 0.20), ("12p5", 0.125)):
        prefix = f"pair_d{tag}_"
        pre = rows[prefix + "fixed_vsa_pre"]
        post = rows[prefix + "fixed_vsa_post"]
        baseline = average_baseline(pre, post)
        methods = []
        for suffix in ("temporal_vsa", "temporal_c2f_only", "temporal_c2f_reuse2"):
            item = dict(rows[prefix + suffix])
            item["denoise_speedup_vs_matched_fixed_vsa"] = (
                baseline["denoise"] / item["denoise_seconds_median"]
            )
            item["sample_plus_vae_speedup_vs_matched_fixed_vsa"] = (
                baseline["sample_plus_vae"] / item["sample_plus_vae_seconds"]
            )
            methods.append(item)
        density_rows.append(
            {"density": density, "fixed_vsa_baseline": baseline, "methods": methods}
        )

    longwarm = None
    longwarm_names = {
        "pre": "longwarm_fixed_vsa20_pre",
        "method": "longwarm_temporal_c2f20_reuse2",
        "post": "longwarm_fixed_vsa20_post",
    }
    if all(name in rows for name in longwarm_names.values()):
        longwarm_baseline = average_baseline(
            rows[longwarm_names["pre"]], rows[longwarm_names["post"]]
        )
        longwarm_method = dict(rows[longwarm_names["method"]])
        longwarm_method["denoise_speedup_vs_fixed_vsa20"] = (
            longwarm_baseline["denoise"]
            / longwarm_method["denoise_seconds_median"]
        )
        longwarm_method["sample_plus_vae_speedup_vs_fixed_vsa20"] = (
            longwarm_baseline["sample_plus_vae"]
            / longwarm_method["sample_plus_vae_seconds"]
        )
        longwarm = {
            "warmup_runs": 20,
            "timed_runs": 50,
            "fixed_vsa_baseline": longwarm_baseline,
            "method": longwarm_method,
        }

    grouped_metrics_path = root / "grouped_same_seed_pixel_metrics.json"
    grouped_metrics = (
        json.loads(grouped_metrics_path.read_text(encoding="utf-8"))
        if grouped_metrics_path.exists()
        else None
    )

    # Five stage-grid paths requested for the report.  The two newly measured
    # 4->2->1 paths ran while another training process shared the GPU, so each
    # one is normalized by fixed-resolution VSA measurements immediately
    # before and after it on that same GPU.  The three older paths were
    # measured in the clean single-GPU sweep and use the primary clean
    # baseline.  Keep both the raw latency and the matched baseline in the
    # summary so the report cannot accidentally mix the two timing contexts.
    st421_recheck = "grid_spatiotemporal_421_c2f20_recheck" in rows
    requested_grid_specs = [
        {
            "name": "grid_temporal_4x2x1_c2f20",
            "label": "Temporal 4->2->1",
            "pre": "grid_temporal_4x2x1_c2f20_fixed_pre",
            "post": "grid_temporal_4x2x1_c2f20_fixed_post",
            "timing_context": "shared_gpu_paired",
        },
        {
            "name": "spatial_c2f_only20",
            "label": "Spatial 2->2->1",
            "timing_context": "clean_single_gpu",
        },
        {
            "name": "grid_spatial_4x2x1_c2f20",
            "label": "Spatial 4->2->1",
            "pre": "grid_spatial_4x2x1_c2f20_fixed_pre",
            "post": "grid_spatial_4x2x1_c2f20_fixed_post",
            "timing_context": "shared_gpu_paired",
        },
        {
            "name": "spatiotemporal_c2f_only20",
            "label": "Spatiotemporal 2->2->1",
            "timing_context": "clean_single_gpu",
        },
        {
            "name": (
                "grid_spatiotemporal_421_c2f20_recheck"
                if st421_recheck
                else "spatiotemporal_3stage_c2f_only20_triton"
            ),
            "label": "Spatiotemporal 4->2->1",
            "pre": "grid_spatiotemporal_421_c2f20_recheck_fixed_pre",
            "post": "grid_spatiotemporal_421_c2f20_recheck_fixed_post",
            "timing_context": (
                "shared_gpu_paired" if st421_recheck else "clean_single_gpu"
            ),
        },
    ]
    requested_stage_grids = []
    for spec in requested_grid_specs:
        if spec["name"] not in rows:
            continue
        if "pre" in spec and spec["pre"] in rows and spec["post"] in rows:
            matched_baseline = average_baseline(rows[spec["pre"]], rows[spec["post"]])
        else:
            matched_baseline = dict(final_baseline)
        item = dict(rows[spec["name"]])
        item.update(
            {
                "display_label": spec["label"],
                "timing_context": spec["timing_context"],
                "matched_fixed_vsa_baseline": matched_baseline,
                "denoise_speedup_vs_matched_fixed_vsa": (
                    matched_baseline["denoise"] / item["denoise_seconds_median"]
                ),
                "sample_plus_vae_speedup_vs_matched_fixed_vsa": (
                    matched_baseline["sample_plus_vae"]
                    / item["sample_plus_vae_seconds"]
                ),
            }
        )
        requested_stage_grids.append(item)

    # The extra city/coast stage-path videos were generated while training
    # shared the GPUs.  Normalize each scene by the fixed-resolution VSA run
    # immediately after that scene's queue instead of comparing raw seconds
    # with the older idle-GPU quality videos.
    quality_stagepath_suffixes = (
        "temporal421_c2f20",
        "spatial221_c2f20",
        "spatial421_c2f20",
        "spatiotemporal221_c2f20",
        "spatiotemporal421_c2f20",
    )
    for scene in ("city", "coast"):
        baseline_name = f"quality_{scene}_stagepaths_fixed_post"
        if baseline_name not in rows:
            continue
        quality_baseline = rows[baseline_name]
        for suffix in quality_stagepath_suffixes:
            name = f"quality_{scene}_{suffix}"
            if name not in rows:
                continue
            rows[name]["denoise_speedup_vs_quality_fixed_vsa"] = (
                quality_baseline["denoise_seconds_median"]
                / rows[name]["denoise_seconds_median"]
            )
            rows[name]["sample_plus_vae_speedup_vs_quality_fixed_vsa"] = (
                quality_baseline["sample_plus_vae_seconds"]
                / rows[name]["sample_plus_vae_seconds"]
            )
            rows[name]["quality_fixed_vsa_baseline_name"] = baseline_name

    output = {
        "scope": {
            "model": "FastWan2.1-T2V-1.3B-Diffusers",
            "resolution": "832x448",
            "frames": 29,
            "fps": 16,
            "nfe": 3,
            "timesteps": [1000, 757, 522],
            "transition": "noise_filling",
            "hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "warning": "FastWan checkpoint is VSA/DMD-trained, not RDD/C2F-trained.",
        },
        "primary_final_baseline": final_baseline,
        "primary_final": primary,
        "density_sweep": density_rows,
        "requested_stage_grids": requested_stage_grids,
        "longwarm_stability": longwarm,
        "same_seed_pixel_perturbation": grouped_metrics,
        "quality_runs": [
            item for name, item in sorted(rows.items()) if name.startswith("quality_")
        ],
        "all_rows": list(rows.values()),
        "route_lift_benchmark": json.loads(
            (root / "c2f_route_lift_benchmark.json").read_text(encoding="utf-8")
        ),
        "route_comparison_benchmark": json.loads(
            (root / "c2f_vs_online_route_benchmark.json").read_text(encoding="utf-8")
        ),
    }
    json_path = root / "fair_acceleration_summary.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = root / "fair_acceleration_all_metrics.csv"
    flat_rows = list(rows.values())
    fieldnames = [
        "name",
        "size",
        "frame_num",
        "keep_nominal",
        "attention_mode",
        "route_density_requested",
        "effective_c2f_density",
        "reuse_every",
        "reuse_scope",
        "online_vsa_calls",
        "c2f_sparse_calls",
        "c2f_seed_vsa_calls",
        "projected_reuse_hits",
        "denoise_seconds_median",
        "sample_seconds_median",
        "vae_seconds",
        "sample_plus_vae_seconds",
        "seed",
        "video_file",
        "metrics_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
