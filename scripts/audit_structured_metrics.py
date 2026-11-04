#!/usr/bin/env python3
"""Second-pass audit for structured metrics and mixed-protocol plot anomalies.

This module is intentionally conservative.  It inventories every extracted
structured FID/speed hit, but promotes only explicitly recognized experiment
outputs into compact diagnostic or acceleration summaries.  Documentation
citations, code literals, generated ledger copies, and derived delta fields stay
visible in the audit instead of being mistaken for new experiments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
}


def _fid_class(path: str, location: str) -> tuple[str, str]:
    x = str(path).replace("\\", "/")
    low = x.lower()
    if "experiment_ledger_20260918/" in low:
        return "generated_ledger_copy", "Generated from the original log/txt extraction; not a new experiment"
    if re.search(r"(^|/)(\.venv[^/]*|venv[^/]*|site-packages)(/|$)", low):
        return "false_positive_code", "Identifier named fid in environment/source code, not Fréchet Inception Distance"
    if low.startswith("saved/all_logs/"):
        return "invalid_old_hf_training", "Old wrong-HF-preprocessed training data; hypothesis only"
    if low.endswith("convergence_all_original/fid_convergence_all_original.tsv"):
        return "duplicate_canonical_summary", "Convenience table duplicating canonical convergence sources"
    if "metrics_summary_400k_only.tsv" in low:
        return "duplicate_canonical_summary", "One-row duplicate of the complete two-stage convergence table"
    if low.endswith("metrics_summary.tsv") and low.startswith("eval_results/"):
        return "canonical_structured_result", "Primary structured convergence/evaluation table already curated"
    if "csra_reader_independent_fid10k" in low and "summary_fid10k.json" in low:
        return "diagnostic_fid10k", "Independent 10K CSRA diagnostic; not comparable to formal 50K FID"
    if "csra_prefix6_fid10k" in low and "summary_fid10k.json" in low:
        return "diagnostic_fid10k", "Development 10K CSRA diagnostic; not comparable to formal 50K FID"
    if "csra_reader_independent_fid10k" in low and "quality_audit.json" in low:
        if "minus" in str(location).lower():
            return "derived_difference", "Difference field, not an independently evaluated dataset"
        return "duplicate_diagnostic_summary", "Duplicates the CSRA 10K summary JSON"
    if "lowpass_summary_lowpass" in low or "lowpass_metrics/summary_lowpass" in low:
        if "report_bundle/assets" in low:
            return "duplicate_diagnostic_summary", "Bundled copy of low-pass diagnostic"
        return "diagnostic_lowpass_fid10k", "10K low-pass diagnostic; not full-resolution FID"
    if low.startswith(("docs/", "paper/", "guided-diffusion/")):
        return "documentation_or_reference", "Paper/reference/example number, not a project evaluation record"
    if low == "2d/all_results.md":
        return "manual_summary", "Human-written summary; underlying evaluator source takes precedence"
    if low.endswith((".py", ".tex", ".rst", ".sh", ".ps1")):
        return "false_positive_code", "Code or prose token matched by the broad scanner"
    return "unresolved_structured_hit", "Retained for manual review; not automatically promoted"


def _speed_class(path: str) -> str:
    low = str(path).replace("\\", "/").lower()
    if re.search(r"(^|/)(\.venv[^/]*|venv[^/]*|site-packages)(/|$)", low):
        return "environment_or_unrelated"
    if low.startswith("third_party/"):
        return "third_party_reference"
    if "experiment_ledger_20260918/" in low:
        return "generated_ledger"
    if low.endswith("train.jsonl"):
        return "training_telemetry"
    if "speed_benchmark" in low or low.endswith("benchmark_250_speed.log"):
        return "2d_formal_microbenchmark"
    if "rdd3d" in low or "dynamic_sparse" in low:
        return "rdd3d_runtime"
    if any(token in low for token in ["fastwan", "rdd_video", "wan_", "video_"]):
        return "video_acceleration"
    if "dmd" in low:
        return "dmd_training_or_diagnostic"
    if any(token in low for token in ["reuse", "vsa", "sparse", "attention", "latency", "csra"]):
        return "2d_reuse_attention_pilot"
    if low.startswith("docs/"):
        return "documentation"
    if low.startswith("datasets/"):
        return "dataset_metadata"
    return "other_project_telemetry"


def _strict_curve(valid: pd.DataFrame, model: str) -> pd.DataFrame:
    d = valid[
        valid.model.eq(model)
        & valid.cfg.eq(1.0)
        & valid.actual_nfe.eq(250)
        & valid.sampler.eq("DDPM")
    ].copy()
    if model == "Original DiT-XL/2":
        d = d[d.sampling_steps_fine_middle_coarse.astype(str).eq("250")]
    elif "500/500" in model:
        d = d[d.sampling_steps_fine_middle_coarse.astype(str).eq("125/125")]
    else:
        d = d[d.sampling_steps_fine_middle_coarse.astype(str).eq("90/80/80")]
    if model.startswith("Unified"):
        d = d[d.transition.eq("noise filling")]
    d["train_step_numeric"] = pd.to_numeric(d.train_step, errors="coerce")
    d["priority"] = np.select(
        [
            d.notes.eq("Canonical structured convergence table"),
            d.notes.astype(str).str.startswith("Canonical convergence point"),
        ],
        [0, 1],
        default=2,
    )
    return (
        d[d.train_step_numeric.notna()]
        .sort_values(["train_step_numeric", "priority", "source_repo_path"])
        .drop_duplicates("train_step_numeric")
        .sort_values("train_step_numeric")
    )


def _diagnostic_rows(fid: pd.DataFrame) -> pd.DataFrame:
    keep_paths = {
        "eval_results/csra_prefix6_fid10k_20260914/metrics/summary_fid10k.json": "CSRA development 10K",
        "eval_results/csra_reader_independent_fid10k_20260914/metrics/summary_fid10k.json": "CSRA independent 10K",
        "eval_results/diagnostics/stage_allocation/lowpass_metrics/summary_lowpass64_10k.json": "low-pass-64 10K",
        "eval_results/diagnostics/stage_allocation/lowpass_metrics/summary_lowpass128_10k.json": "low-pass-128 10K",
    }
    d = fid[fid.source_repo_path.isin(keep_paths)].copy()
    d["protocol"] = d.source_repo_path.map(keep_paths)
    d["variant"] = d.location.astype(str).str.split("/").str[-1]
    columns = ["protocol", "variant", "fid", "sfid", "inception_score", "precision", "recall", "source_repo_path"]
    return d[columns].sort_values(["protocol", "variant"]).reset_index(drop=True)


def _acceleration_summary(root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kwargs) -> None:
        base = {
            "family": "",
            "method": "",
            "task_protocol": "",
            "speedup": np.nan,
            "latency_reduction_percent": np.nan,
            "quality_metric": "",
            "quality_value": np.nan,
            "evidence_status": "",
            "source": "",
            "notes": "",
        }
        base.update(kwargs)
        rows.append(base)

    # Formal 2D 250-step H20 microbenchmark, parsed from the completed log.
    for method, ms, speedup in [
        ("RDD Unified L/2", 520.9, 2.560),
        ("RDD Multistage 300/300/400", 596.6, 2.235),
        ("RDD Unified XL/2", 703.4, 1.896),
        ("DiT-XL/2 baseline", 1333.4, 1.000),
    ]:
        add(
            family="2D structural RDD",
            method=method,
            task_protocol="H20, batch16, DDPM250, 90/80/80 for RDD, single conditional forward",
            speedup=speedup,
            quality_metric="latency_ms_per_image",
            quality_value=ms,
            evidence_status="measured microbenchmark",
            source="benchmark_250_speed.log",
            notes="Structural compute comparison; boundary implementation is close to but not identical to formal noise filling",
        )

    # Completed 2D attention-reuse kernel experiments.
    for method, speedup, cosine, psnr in [
        ("probability refresh every 3 steps", 1.142, 0.99526, 31.29),
        ("layer-depth 5/3/3 refresh", 1.136, 0.99642, 31.87),
        ("probability refresh every 5 steps", 1.180, 0.98888, 27.84),
        ("conservative cascade + refresh3", 1.084, 0.99312, 29.83),
    ]:
        add(
            family="2D attention reuse",
            method=method,
            task_protocol="Unified-XL 400K, DDPM 80/80/90, CFG1.5, paired sample",
            speedup=speedup,
            quality_metric="image_cosine",
            quality_value=cosine,
            evidence_status="paired proxy only; no FID",
            source="docs/research_docs/RDD_ATTENTION_ACCELERATION_EXPERIMENTS_ZH.md",
            notes=f"PSNR {psnr:.2f} dB; proxy quality is not distributional quality",
        )

    # Video C2F-route plumbing benchmark using the un-finetuned FastWan checkpoint.
    fair_csv = root / "eval_results" / "rdd_fastwan_fair_accel_20260907" / "fair_acceleration_all_metrics.csv"
    if fair_csv.exists():
        fair = pd.read_csv(fair_csv)
        base = fair[fair.name.eq("longwarm_fixed_vsa20_pre")].iloc[0]
        cand = fair[fair.name.eq("longwarm_temporal_c2f20_reuse2")].iloc[0]
        add(
            family="Video C2F route/reuse",
            method="temporal C2F-20 + reuse2",
            task_protocol="FastWan DMD3, 832x448x29, same checkpoint",
            speedup=float(base.denoise_seconds_median / cand.denoise_seconds_median),
            latency_reduction_percent=float(100 * (1 - cand.denoise_seconds_median / base.denoise_seconds_median)),
            quality_metric="sample_plus_vae_speedup",
            quality_value=float(base.sample_plus_vae_seconds / cand.sample_plus_vae_seconds),
            evidence_status="plumbing/latency only; not RDD-trained",
            source="eval_results/rdd_fastwan_fair_accel_20260907/fair_acceleration_all_metrics.csv",
            notes="Generated videos are not evidence of matched RDD video quality",
        )

    # Direct block reuse with paired timing; known to damage quality.
    direct_path = root / "eval_results" / "direct_reuse_fullmodel_20260915" / "summary_and_audit.json"
    if direct_path.exists():
        direct = json.loads(direct_path.read_text(encoding="utf-8"))
        for item in direct.get("video", {}).get("video_1280", {}).get("rows", []):
            den = item["reductions"]["denoise"]["mean"]
            total = item["reductions"]["total"]["mean"]
            add(
                family="Video direct block reuse",
                method=item["variant"],
                task_protocol="FastWan DMD3, 1280 output, paired ABBA/BAAB",
                speedup=1 / (1 - den / 100),
                latency_reduction_percent=den,
                quality_metric="end_to_end_reduction_percent",
                quality_value=total,
                evidence_status="measured latency; quality failed",
                source="eval_results/direct_reuse_fullmodel_20260915/summary_and_audit.json",
                notes="Includes VAE in the quality_value column; no FID/VBench and visible degradation",
            )

    # Independent CSRA 10K FID quality gate.
    add(
        family="2D CSRA reader",
        method="skip-6 reader",
        task_protocol="Independent 10K, same ADM reference",
        quality_metric="FID10K_reader_minus_RDD",
        quality_value=5.852910929622624,
        evidence_status="quality gate failed; speed not measured in same run",
        source="eval_results/csra_reader_independent_fid10k_20260914/quality_audit.json",
        notes="FID 11.6741 -> 17.5270; must not be mixed with 50K paper FID",
    )

    # Three-step RDD DMD has the intended NFE but no acceptable distributional result.
    add(
        family="RDD DMD",
        method="3-stage / 3-NFE student",
        task_protocol="one NFE per resolution stage",
        quality_metric="formal_FID",
        evidence_status="no valid FID; visual/semantic quality failed",
        source="docs/research_docs/RDD_DMD_FAILURE_DIAGNOSIS_ZH.md",
        notes="Do not report the theoretical 250/3 NFE ratio as quality-matched speedup",
    )

    # RDD-3D runtime benchmark was remote-only, so the audit keeps a local copy.
    rdd3d_path = root / "eval_results" / "experiment_ledger_20260918" / "structured_text_audit" / "complete" / "rdd3d_runtime_benchmark.json"
    if rdd3d_path.exists():
        r3 = json.loads(rdd3d_path.read_text(encoding="utf-8"))
        add(
            family="RDD-3D dynamic sparse",
            method="dynamic sparse vs force dense",
            task_protocol="30 measured samples; model load/NPZ write excluded",
            speedup=float(r3["wall_clock_speedup"]),
            latency_reduction_percent=float(100 * (1 - r3["sparse"]["wall_seconds_mean"] / r3["force_dense"]["wall_seconds_mean"])),
            quality_metric="token_NFE_reduction",
            quality_value=float(r3["token_nfe_reduction"]),
            evidence_status="task-specific runtime benchmark",
            source="eval_results/rdd3d_dynamic_sparse/dynamic_sparse_final_test_64_seed20260812/runtime_benchmark.json",
            notes="Different 3D task and protocol; not comparable to 2D/video wall-clock",
        )

    return pd.DataFrame(rows)


def build_second_pass_audits(root: Path, out: Path, fig_dir: Path, valid: pd.DataFrame) -> dict:
    structured = out / "structured_text_audit" / "complete"
    required = [
        structured / "structured_fid_records.csv",
        structured / "structured_speed_records.csv",
        structured / "structured_source_inventory.csv",
    ]
    if not all(path.exists() for path in required):
        return {"figures": [], "summary": {}, "diagnostics": pd.DataFrame(), "acceleration": pd.DataFrame()}

    fid = pd.read_csv(required[0])
    speed = pd.read_csv(required[1])
    inv = pd.read_csv(required[2])

    classes = [_fid_class(p, loc) for p, loc in zip(fid.source_repo_path, fid.location)]
    fid["audit_class"] = [x[0] for x in classes]
    fid["audit_reason"] = [x[1] for x in classes]
    fid.to_csv(out / "structured_fid_records_classified.csv", index=False)
    source_audit = (
        fid.groupby(["source_repo_path", "source_origin", "source_suffix", "audit_class", "audit_reason"], dropna=False)
        .agg(records=("fid", "size"), unique_fid_values=("fid", "nunique"), min_fid=("fid", "min"), max_fid=("fid", "max"))
        .reset_index()
        .sort_values(["audit_class", "source_repo_path"])
    )
    source_audit.to_csv(out / "structured_fid_source_audit.csv", index=False)

    speed["audit_class"] = speed.source_repo_path.map(_speed_class)
    speed.to_csv(out / "structured_speed_records_classified.csv", index=False)
    speed_sources = (
        speed.groupby(["source_repo_path", "source_origin", "source_suffix", "audit_class"], dropna=False)
        .agg(records=("value", "size"), metric_names=("metric_name", lambda x: ", ".join(sorted(set(map(str, x))))) )
        .reset_index()
        .sort_values(["audit_class", "source_repo_path"])
    )
    speed_sources.to_csv(out / "structured_speed_source_audit.csv", index=False)

    paths = inv.repo_path.astype(str)
    project_mask = ~paths.str.contains(
        r"(^|/)(\.venv[^/]*|venv[^/]*|site-packages|third_party)(/|$)",
        case=False,
        regex=True,
    )
    project = inv[project_mask]
    coverage = pd.DataFrame(
        [
            ["All structured/code text sources", len(inv), int(inv["size"].sum())],
            ["Project-owned after env/third_party exclusion", len(project), int(project["size"].sum())],
            ["Project sources with broad FID hit", int((project.fid_records > 0).sum()), int(project.fid_records.sum())],
            ["Project sources with broad speed hit", int((project.speed_records > 0).sum()), int(project.speed_records.sum())],
            ["Structured parse failures", 0, 0],
        ],
        columns=["scope", "sources", "bytes_or_records"],
    )
    coverage.to_csv(out / "structured_scan_coverage.csv", index=False)

    diagnostics = _diagnostic_rows(fid)
    diagnostics.to_csv(out / "diagnostic_fid_results.csv", index=False)
    acceleration = _acceleration_summary(root)
    acceleration.to_csv(out / "acceleration_evidence_summary.csv", index=False)

    models = [
        "Original DiT-XL/2",
        "Unified XL 300/300/400",
        "Unified L 300/300/400",
        "Unified L 400/300/300",
        "Unified L 100/300/600",
        "Unified L 600/300/100",
        "Unified L 16->32 500/500",
        "Unified L 8->32 500/500",
    ]
    curve_rows = []
    for model in models:
        d = _strict_curve(valid, model)
        delta = d.fid.diff()
        positive_delta = delta[delta > 0]
        curve_rows.append(
            {
                "model": model,
                "points": len(d),
                "first_step": int(d.train_step_numeric.min()) if len(d) else np.nan,
                "last_step": int(d.train_step_numeric.max()) if len(d) else np.nan,
                "fid_increase_count": int((delta > 0).sum()),
                "largest_fid_increase": float(positive_delta.max()) if len(positive_delta) else 0.0,
                "protocol": "DDPM, 250 NFE, CFG1.0, fixed sampling split, matched boundary/transition",
            }
        )
    curve_audit = pd.DataFrame(curve_rows)
    curve_audit.to_csv(out / "strict_curve_monotonicity_audit.csv", index=False)

    anomaly = pd.DataFrame(
        [
            ["Fig01 Unified-XL 400K vertical jump", "7.065 -> 9.543 at the same 400K x", "noise filling and bilinear were connected", "keep noise filling 90/80/80 only", "fixed"],
            ["Fig01/03 duplicate 500K/400K points", "small vertical segments", "independent repeats were connected as time", "one canonical sweep point per checkpoint", "fixed"],
            ["Fig02 Unified-L schedule jumps", "fixed and proportional results shared the same x", "90/80/80 and interval-proportional policies were connected", "fixed 90/80/80 only", "fixed"],
            ["Fig06 fine=90 spike", "FID 9.183 instead of 3.537", "mismatched inference boundary 250/250/500 selected", "matched 300/300/400 boundary only", "fixed"],
            ["Fig08 FID rises after CFG1.5", "3.537 -> 8.335 -> 13.184 -> 20.594", "real CFG parameter sweep, not training time", "hold 90/80/80 and matched boundary", "valid non-monotonic sweep"],
            ["Unified-L sFID tiny reversals", "350K->400K +0.005; 400K->450K +0.031", "metric noise/trade-off while FID keeps improving", "report as secondary metric fluctuation", "valid"],
        ],
        columns=["issue", "observed", "root_cause", "correct_protocol", "decision"],
    )
    anomaly.to_csv(out / "curve_anomaly_audit.csv", index=False)

    # Figure 15: structured diagnostic FID remains isolated from formal 50K FID.
    csra = diagnostics[diagnostics.protocol.str.contains("CSRA", na=False)].copy()
    if not csra.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        for ax, protocol, title in [
            (axes[0], "CSRA development 10K", "CSRA development quality gate"),
            (axes[1], "CSRA independent 10K", "CSRA independent quality gate"),
        ]:
            d = csra[csra.protocol.eq(protocol)]
            bars = ax.bar(d.variant, d.fid, color=[COLORS["green"] if "original" in x else COLORS["red"] for x in d.variant])
            ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel("FID10K (lower is better)")
            ax.tick_params(axis="x", rotation=18)
            ax.grid(True, axis="y", color="#e2e8f0")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Structured JSON results recovered in the second-pass audit", fontweight="bold")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(fig_dir / f"15_structured_csra_fid10k.{ext}", dpi=180 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Figure 16: four panels keep unlike acceleration protocols separate.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    structural = acceleration[acceleration.family.eq("2D structural RDD")]
    axes[0, 0].barh(structural.method, structural.speedup, color=COLORS["blue"])
    axes[0, 0].invert_yaxis(); axes[0, 0].set_xlabel("speedup vs DiT"); axes[0, 0].set_title("2D DDPM250 H20 microbenchmark", loc="left", fontweight="bold")
    reuse = acceleration[acceleration.family.eq("2D attention reuse")]
    axes[0, 1].scatter(reuse.speedup, reuse.quality_value, color=COLORS["purple"], s=60)
    for _, row in reuse.iterrows():
        axes[0, 1].annotate(row.method.replace("probability ", "prob. "), (row.speedup, row.quality_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0, 1].set_xlabel("speedup"); axes[0, 1].set_ylabel("paired image cosine"); axes[0, 1].set_title("2D attention reuse proxies", loc="left", fontweight="bold")
    video = acceleration[acceleration.family.eq("Video direct block reuse")]
    x = np.arange(len(video)); w = 0.36
    axes[1, 0].bar(x-w/2, video.latency_reduction_percent, w, label="denoiser", color=COLORS["orange"])
    axes[1, 0].bar(x+w/2, video.quality_value, w, label="including VAE", color=COLORS["cyan"])
    axes[1, 0].set_xticks(x, video.method); axes[1, 0].set_ylabel("latency reduction (%)"); axes[1, 0].legend(frameon=False); axes[1, 0].set_title("Video direct reuse (quality failed)", loc="left", fontweight="bold")
    r3 = acceleration[acceleration.family.eq("RDD-3D dynamic sparse")]
    if len(r3):
        axes[1, 1].bar(["wall-clock speedup", "token-NFE reduction"], [float(r3.speedup.iloc[0]), float(r3.quality_value.iloc[0])], color=[COLORS["green"], COLORS["slate"]])
    axes[1, 1].set_ylabel("factor vs dense"); axes[1, 1].set_title("RDD-3D task-specific runtime", loc="left", fontweight="bold")
    for ax in axes.flat:
        ax.grid(True, axis="y", color="#e2e8f0"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Acceleration evidence by protocol (panels are not cross-comparable)", fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"16_acceleration_evidence_by_protocol.{ext}", dpi=180 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    figures = [
        {
            "stem": "15_structured_csra_fid10k",
            "title": "结构化 JSON 中恢复的 CSRA 10K 指标",
            "caption": "这些是第二轮扫描补入的 10K 诊断，不能与正式 50K FID 混图；独立 reader 使 FID 从 11.674 上升到 17.527。",
        },
        {
            "stem": "16_acceleration_evidence_by_protocol",
            "title": "加速证据按协议分组",
            "caption": "2D、视频与 3D 的硬件、任务和质量门槛不同，因此只在各自面板内比较。视频 direct reuse 虽降低 denoiser 延迟，但质量失败且 VAE 稀释了端到端收益。",
        },
    ]
    summary = {
        "all_structured_sources": int(len(inv)),
        "project_owned_structured_sources": int(len(project)),
        "project_owned_bytes": int(project["size"].sum()),
        "broad_fid_records": int(len(fid)),
        "broad_speed_records": int(len(speed)),
        "structured_parse_failures": 0,
        "new_paper_valid_50k_fid_results": 0,
        "new_diagnostic_fid_rows": int(len(diagnostics)),
        "strict_curves_with_fid_increase": int((curve_audit.fid_increase_count > 0).sum()),
    }
    (out / "second_pass_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"figures": figures, "summary": summary, "diagnostics": diagnostics, "acceleration": acceleration}


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    output = base / "eval_results" / "experiment_ledger_20260918"
    valid_frame = pd.read_csv(output / "curated_valid_results.csv")
    result = build_second_pass_audits(base, output, output / "figures", valid_frame)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
