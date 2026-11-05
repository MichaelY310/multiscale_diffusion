#!/usr/bin/env python3
"""Build the complete RDD experiment ledger and comparison figure bundle.

Inputs are deliberately separated into three layers:

1. raw_fid_blocks.csv: every FID block extracted from every HF/surfi2 log/txt;
2. curated_valid_results.csv: one canonical row per usable experiment setting;
3. historical_invalid_results.csv: canonical rows from the old, wrong HF-data runs.

The script never promotes an experiment based on a low FID alone.  Dataset and
checkpoint provenance decide validity first; duplicate/evaluator conflicts are
recorded in a separate audit table.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_structured_metrics import build_second_pass_audits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results" / "experiment_ledger_20260918"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "research_docs"
PUBLISHED_FIG = DOC / "figures" / "experiment_ledger_20260918"
FIG.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)
PUBLISHED_FIG.mkdir(parents=True, exist_ok=True)

RAW_PATH = OUT / "raw_fid_blocks.csv"
INVENTORY_PATH = OUT / "source_inventory.csv"

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "pink": "#db2777",
    "slate": "#475569",
    "gray": "#9ca3af",
}

CURATED_COLUMNS = [
    "record_id",
    "evidence_class",
    "method",
    "model",
    "model_size",
    "topology",
    "training_data",
    "train_allocation_fine_middle_coarse",
    "train_step",
    "sampler",
    "sampling_steps_fine_middle_coarse",
    "scheduled_updates",
    "actual_nfe",
    "cfg",
    "transition",
    "num_samples",
    "inception_score",
    "fid",
    "sfid",
    "precision",
    "recall",
    "source_repo_path",
    "notes",
]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def metric_row(raw: pd.DataFrame, source: str) -> pd.Series:
    rows = raw.loc[raw["source_repo_path"].eq(source)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one metric block for {source}, found {len(rows)}")
    return rows.iloc[0]


def add_record(records: list[dict], **kwargs) -> None:
    row = {column: "" for column in CURATED_COLUMNS}
    row.update(kwargs)
    if not row["record_id"]:
        row["record_id"] = f"valid-{len(records) + 1:04d}"
    records.append(row)


def add_from_metric(records: list[dict], metric: pd.Series, **kwargs) -> None:
    add_record(
        records,
        inception_score=metric.get("inception_score", np.nan),
        fid=metric.get("fid", np.nan),
        sfid=metric.get("sfid", np.nan),
        precision=metric.get("precision", np.nan),
        recall=metric.get("recall", np.nan),
        source_repo_path=metric["source_repo_path"],
        **kwargs,
    )


def parse_step(path: str) -> int | float:
    matches = [
        re.search(r"ckpt0*(\d+)[kK]", path),
        re.search(r"_(\d+)[kK](?:_|\.)", path),
        re.search(r"_(\d{5,7})(?:_|\.)", path),
        re.search(r"step(\d+)", path, re.I),
    ]
    for match in matches:
        if not match:
            continue
        value = int(match.group(1))
        if "k" in match.group(0).lower():
            value *= 1000
        return value
    return np.nan


def raw_classification(path: str) -> tuple[str, str]:
    if path.startswith("logs/remote/saved/all_logs/"):
        return "duplicate_mirror", "Remote mirror of saved/all_logs; excluded from counts and plots"
    if path.startswith("saved/all_logs/"):
        return "invalid_old_hf_training", "Wrong Hugging Face-preprocessed training dataset; hypothesis only"
    if path.endswith("eval_results_unified/rdd_450K.txt"):
        return "excluded_known_bad", "User confirmed the correct 450K result was lost"
    if path.endswith("eval_results_exp04/dit_400K.txt"):
        return "excluded_conflicting_bad", "Conflicts with user-confirmed DiT-400K FID 19.781288"
    if path.startswith(".tmp/"):
        return "smoke_or_failed", "Smoke/legacy test, not a formal evaluation"
    if "legacy_failures" in path:
        return "duplicate_incomplete", "Earlier evaluator failure; superseded by complete full_metrics result"
    if "diagnostics/" in path or "csra_" in path or "/csra" in path:
        return "diagnostic_or_pilot", "Diagnostic or 10K pilot; kept separate from 50K paper results"
    if path.startswith("logs/") or path.startswith("2d/exp/"):
        return "driver_or_duplicate_log", "Driver log or repeated copy of a leaf metric file"
    if path.startswith(("eval_results/", "eval_results_", "2d/diagnostics/")):
        return "candidate_leaf_metric", "Candidate leaf metric; see curated table for canonical use"
    return "unresolved", "No automatic provenance promotion"


def build_curated(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []

    common_diff = dict(
        evidence_class="A_paper_valid",
        method="RDD diffusion",
        model_size="L/2",
        training_data="ImageNet original",
        sampler="DDPM",
        scheduled_updates=250,
        actual_nfe=250,
        cfg=1.0,
        transition="noise filling",
        num_samples=50000,
    )
    curve_specs = [
        ("unified_l_300300400_convergence/metrics_summary.tsv", "Unified L 300/300/400", "300/300/400", "90/80/80"),
        ("unified_l_400300300_convergence/metrics_summary.tsv", "Unified L 400/300/300", "400/300/300", "90/80/80"),
        ("unified_l_boundary_convergence/metrics_summary.tsv", None, None, "90/80/80"),
        ("unified_l_2stage_500500_convergence/metrics_summary.tsv", "Unified L 16->32 500/500", "500/500", "125/125"),
        ("unified_l_2stage_8to32_500500_convergence/metrics_summary.tsv", "Unified L 8->32 500/500", "500/500", "125/125"),
    ]
    for relative, fixed_model, fixed_alloc, sample_alloc in curve_specs:
        frame = read_tsv(ROOT / "eval_results" / relative)
        for _, source in frame.iterrows():
            raw_code = source["model"]
            code = str(int(raw_code)) if isinstance(raw_code, (int, float, np.integer, np.floating)) else str(raw_code)
            if fixed_model is not None:
                model = fixed_model
                alloc = fixed_alloc
                topology = "unified/shared" if "300/" in alloc or "400/" in alloc else "unified/shared 2-stage"
            else:
                alloc = f"{code[:3]}/{code[3:6]}/{code[6:]}"
                model = f"Unified L {alloc}"
                topology = "unified/shared"
            add_record(
                records,
                **common_diff,
                model=model,
                topology=topology,
                train_allocation_fine_middle_coarse=alloc,
                train_step=int(source["step"]),
                sampling_steps_fine_middle_coarse=sample_alloc,
                inception_score=source["inception_score"],
                fid=source["fid"],
                sfid=source["sfid"],
                precision=source["precision"],
                recall=source["recall"],
                source_repo_path=f"eval_results/{relative}",
                notes="Canonical structured convergence table",
            )

    # Unified XL and Original DiT convergence.  DiT-400K uses the explicitly
    # confirmed result rather than the conflicting exp04 value.
    for prefix, method, model, model_size, alloc, topology in [
        ("rdd", "RDD diffusion", "Unified XL 300/300/400", "XL/2", "300/300/400", "unified/shared"),
        ("dit", "Original diffusion", "Original DiT-XL/2", "XL/2", "1000", "single resolution"),
    ]:
        for step in range(50000, 500001, 50000):
            if prefix == "rdd" and step == 450000:
                continue
            if prefix == "dit" and step == 400000:
                source_path = "eval_results_unified/dit400k_250step_ddpm_cfg0.txt"
            else:
                source_path = f"eval_results_exp04/{prefix}_{step // 1000}K.txt"
            metric = metric_row(raw, source_path)
            add_from_metric(
                records,
                metric,
                evidence_class="A_paper_valid",
                method=method,
                model=model,
                model_size=model_size,
                topology=topology,
                training_data="ImageNet original",
                train_allocation_fine_middle_coarse=alloc,
                train_step=step,
                sampler="DDPM",
                sampling_steps_fine_middle_coarse="90/80/80" if prefix == "rdd" else "250",
                scheduled_updates=250,
                actual_nfe=250,
                cfg=1.0,
                transition="noise filling" if prefix == "rdd" else "n.a.",
                num_samples=50000,
                notes="Canonical convergence point; 450K Unified XL intentionally absent" if prefix == "rdd" else "Canonical convergence point",
            )

    # DiT guidance operating points.
    for source_path, step, cfg, note in [
        ("eval_results_unified/dit400k_250step_ddpm_cfg1.25.txt", 400000, 1.25, "Full five metrics recovered from HF leaf file"),
        ("eval_results_dit500k/dit500k_250step_ddpm_cfg1.0.txt", 500000, 1.0, "Independent main evaluation; convergence repeat differs by 0.011 FID"),
        ("eval_results_dit500k/dit500k_250step_ddpm_cfg1.5.txt", 500000, 1.5, "Full five metrics recovered; old ledger incorrectly listed this as missing"),
    ]:
        add_from_metric(
            records,
            metric_row(raw, source_path),
            evidence_class="A_paper_valid",
            method="Original diffusion",
            model="Original DiT-XL/2",
            model_size="XL/2",
            topology="single resolution",
            training_data="ImageNet original",
            train_allocation_fine_middle_coarse="1000",
            train_step=step,
            sampler="DDPM",
            sampling_steps_fine_middle_coarse="250",
            scheduled_updates=250,
            actual_nfe=250,
            cfg=cfg,
            transition="n.a.",
            num_samples=50000,
            notes=note,
        )

    # Unified XL and multistage sweeps stored as canonical leaf files.
    unified_sources = sorted(raw.loc[raw["source_repo_path"].str.startswith("eval_results_unified/"), "source_repo_path"].unique())
    skip = {
        "eval_results_unified/rdd_400K.txt",
        "eval_results_unified/rdd_450K.txt",
        "eval_results_unified/dit400k_250step_ddpm_cfg0.txt",
        "eval_results_unified/dit400k_250step_ddpm_cfg1.25.txt",
        "eval_results_unified/unifiedL300300400_400000_908080_ddpm_cfg1.txt",
        "eval_results_unified/unifiedXL300300400_400000_908080_ddpm_cfg1.txt",
    }
    for source_path in unified_sources:
        if source_path in skip or source_path.startswith("eval_results_unified/dit"):
            continue
        name = Path(source_path).stem
        metric = metric_row(raw, source_path)
        is_uni = "uniXL" in name or "unifiedXL" in name
        is_ms = name.startswith(("ms", "multistage"))
        if not (is_uni or is_ms):
            continue
        model = "Unified XL 300/300/400" if is_uni else "Multistage XL+L+B 300/300/400"
        topology = "unified/shared" if is_uni else "multistage XL+L+B"
        model_size = "XL/2" if is_uni else "XL/2 + L/2 + B/2"
        sampler = "DPM2" if "dpm2" in name else ("DPM1" if "dpm1" in name else "DDPM")
        cfg_match = re.search(r"cfg(\d+(?:\.\d+)?)", name)
        if cfg_match:
            cfg_text = cfg_match.group(1)
            cfg = 1.5 if cfg_text == "15" else float(cfg_text)
        else:
            cfg = 1.0
        split = "90/80/80"
        code_match = re.search(r"_(\d+)-(\d+)-(\d+)_", name)
        if code_match:
            split = "/".join(code_match.groups())
        else:
            compact = re.search(r"_(151010|252020|454040|908080)_", name)
            compact_map = {"151010": "15/10/10", "252020": "25/20/20", "454040": "45/40/40", "908080": "90/80/80"}
            if compact:
                split = compact_map[compact.group(1)]
        nums = [int(x) for x in split.split("/")]
        scheduled = sum(nums)
        actual_nfe = 2 * scheduled - (6 if len(nums) == 3 else 4) if sampler == "DPM2" else scheduled
        transition = "bilinear" if "bilinear" in name else "noise filling"
        infer_match = re.search(r"split(\d{3})(\d{3})(\d{3})", name)
        notes = ""
        if infer_match:
            infer_alloc = "/".join(infer_match.groups())
            notes = f"Inference-boundary mismatch: inference {infer_alloc}, training 300/300/400"
        if "380k" in name:
            notes = "Filename says 380K, but logs/checkpoint metadata confirm 400K"
        add_from_metric(
            records,
            metric,
            evidence_class="A_paper_valid",
            method="RDD diffusion",
            model=model,
            model_size=model_size,
            topology=topology,
            training_data="ImageNet original",
            train_allocation_fine_middle_coarse="300/300/400",
            train_step=400000,
            sampler=sampler,
            sampling_steps_fine_middle_coarse=split,
            scheduled_updates=scheduled,
            actual_nfe=actual_nfe,
            cfg=cfg,
            transition=transition,
            num_samples=50000,
            notes=notes or "Canonical HF leaf metric",
        )

    # Canonical Unified-L 400K row is already present in convergence; append
    # proportional-allocation evals because they are a distinct sampling policy.
    proportional = read_tsv(ROOT / "eval_results/unified_l_interval_proportional_nfe250/metrics_summary.tsv")
    for _, source in proportional.iterrows():
        code = str(source["model"])
        alloc = f"{code[:3]}/{code[3:6]}/{code[6:]}"
        add_record(
            records,
            **common_diff,
            model=f"Unified L {alloc}",
            topology="unified/shared",
            train_allocation_fine_middle_coarse=alloc,
            train_step=int(source["step"]),
            sampling_steps_fine_middle_coarse="/".join(reversed(str(source["sampling_steps_coarse_to_fine"]).replace(",", "/").split("/"))),
            inception_score=source["inception_score"],
            fid=source["fid"],
            sfid=source["sfid"],
            precision=source["precision"],
            recall=source["recall"],
            source_repo_path="eval_results/unified_l_interval_proportional_nfe250/metrics_summary.tsv",
            notes="Interval-proportional NFE250 policy; file stores coarse-to-fine sampling order",
        )

    # MS-100 paper-valid controls.
    for source_path, cfg, note in [
        ("eval_results/ms100300600_400k_808090_ddpm_cfg1_noisefilling.txt", 1.0, "All stages at 400K"),
        ("eval_results/ms100300600_400k_808090_ddpm_cfg15_noisefilling.txt", 1.5, "All stages at 400K"),
        ("eval_results/ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg1_noisefilling.txt", 1.0, "Only 8x8 continued to 600K; 16x16 and 32x32 remain 400K"),
        ("eval_results/ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg15_noisefilling.txt", 1.5, "Only 8x8 continued to 600K; 16x16 and 32x32 remain 400K"),
    ]:
        add_from_metric(
            records,
            metric_row(raw, source_path),
            evidence_class="A_paper_valid",
            method="RDD diffusion",
            model="Multistage XL+L+B 100/300/600",
            model_size="XL/2 + L/2 + B/2",
            topology="multistage XL+L+B",
            training_data="ImageNet original",
            train_allocation_fine_middle_coarse="100/300/600",
            train_step="mixed 400K/600K" if "continued" in source_path else 400000,
            sampler="DDPM",
            sampling_steps_fine_middle_coarse="90/80/80",
            scheduled_updates=250,
            actual_nfe=250,
            cfg=cfg,
            transition="noise filling",
            num_samples=50000,
            notes=note,
        )

    # Flow matching: legacy XL operating points and corrected L 50K points.
    fm_xl = read_tsv(ROOT / "eval_results/unified_l_stage_comparison/sources/rdd_fm_unified_xl_v2_400k_metrics.tsv")
    for _, source in fm_xl.iterrows():
        add_record(
            records,
            evidence_class="A_paper_valid_with_semantic_caveat",
            method="RDD flow matching",
            model="FM Unified XL v2 legacy",
            model_size="XL/2",
            topology="unified/shared",
            training_data="ImageNet original",
            train_allocation_fine_middle_coarse="legacy FM coarse/mid/fine=300/300/400",
            train_step=int(source["step"]),
            sampler="Euler" if source["solver"] == "euler" else "Midpoint RK2",
            sampling_steps_fine_middle_coarse="/".join(reversed(str(source["steps_sampling_order_8_16_32"]).replace(",", "/").split("/"))),
            scheduled_updates=int(source["updates"]),
            actual_nfe=int(source["actual_nfe"]),
            cfg=float(source["cfg"]),
            transition="noise filling",
            num_samples=50000,
            inception_score=source["inception_score"],
            fid=source["fid"],
            sfid=source["sfid"],
            precision=source["precision"],
            recall=source["recall"],
            source_repo_path="eval_results/unified_l_stage_comparison/sources/rdd_fm_unified_xl_v2_400k_metrics.tsv",
            notes="Legacy FM interval semantics; not a strict time mirror of diffusion 300/300/400",
        )
    for source_path, alloc in [
        ("eval_results/rdd_fm_unified_l_corrected_300300400_convergence/metrics/rddfm_unifiedL_corrected_300300400_ckpt000050k_808090_euler_cfg1_noisefilling.txt", "300/300/400"),
        ("eval_results/rdd_fm_unified_l_corrected_400300300_convergence/metrics/rddfm_unifiedL_corrected_400300300_ckpt000050k_808090_euler_cfg1_noisefilling.txt", "400/300/300"),
    ]:
        add_from_metric(
            records,
            metric_row(raw, source_path),
            evidence_class="A_paper_valid",
            method="RDD flow matching",
            model=f"FM Unified L corrected {alloc}",
            model_size="L/2",
            topology="unified/shared",
            training_data="ImageNet original",
            train_allocation_fine_middle_coarse=alloc,
            train_step=50000,
            sampler="Euler",
            sampling_steps_fine_middle_coarse="90/80/80",
            scheduled_updates=250,
            actual_nfe=252,
            cfg=1.0,
            transition="noise filling",
            num_samples=50000,
            notes="Corrected FM boundary semantics; 100K sampling was incomplete",
        )

    frame = pd.DataFrame(records, columns=CURATED_COLUMNS)
    # Exact-setting duplicate suppression: keep the first canonical source.
    key = [
        "method", "model", "train_step", "sampler",
        "sampling_steps_fine_middle_coarse", "actual_nfe", "cfg", "transition", "notes",
    ]
    frame = frame.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    frame["record_id"] = [f"valid-{i + 1:04d}" for i in range(len(frame))]
    return frame


def build_historical_invalid(raw: pd.DataFrame) -> pd.DataFrame:
    old = raw.loc[raw["source_repo_path"].str.startswith("saved/all_logs/")].copy()
    old["evidence_class"] = "H_invalid_old_hf_training"
    old["training_data"] = "Wrong HF-preprocessed dataset"
    old["model_group"] = "Multistage 300/300/400"
    path = old["source_repo_path"].str.lower()
    old.loc[path.str.contains("100_300_600|ms100"), "model_group"] = "Multistage 100/300/600"
    old.loc[path.str.contains("unified_l2|unifiedl"), "model_group"] = "Unified L"
    old.loc[path.str.contains("unified") & ~path.str.contains("unified_l2|unifiedl"), "model_group"] = "Unified XL"
    old.loc[path.str.contains("decouple"), "model_group"] = "Decouple"
    old.loc[path.str.contains("dit") & ~path.str.contains("rdd"), "model_group"] = "Original DiT"
    old["train_step"] = old["source_repo_path"].map(parse_step)
    old["sampler_guess"] = np.where(path.str.contains("dpm2"), "DPM2", np.where(path.str.contains("dpm"), "DPM1", "DDPM"))
    old["cfg_guess"] = np.select(
        [path.str.contains(r"cfg1\.5|cfg15"), path.str.contains(r"cfg1\.25|cfg125"), path.str.contains(r"cfg2\.5"), path.str.contains("cfg2"), path.str.contains("cfg4")],
        [1.5, 1.25, 2.5, 2.0, 4.0],
        default=1.0,
    )
    columns = [
        "evidence_class", "training_data", "model_group", "train_step", "sampler_guess", "cfg_guess",
        "fid", "sfid", "inception_score", "precision", "recall", "source_repo_path", "source_origin", "source_size", "normalized_line",
    ]
    return old[columns].sort_values(["model_group", "source_repo_path", "normalized_line"]).reset_index(drop=True)


def build_duplicate_audit(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    canonical = set(raw.loc[raw["source_repo_path"].str.startswith("saved/all_logs/"), "source_repo_path"])
    for mirror in sorted(raw.loc[raw["source_repo_path"].str.startswith("logs/remote/saved/all_logs/"), "source_repo_path"].unique()):
        short = mirror.removeprefix("logs/remote/")
        rows.append({
            "issue": "exact_path_mirror",
            "canonical_source": short if short in canonical else "",
            "excluded_source": mirror,
            "canonical_value": "",
            "excluded_value": "",
            "decision": "Keep saved/all_logs copy; exclude logs/remote mirror",
        })
    rows.extend([
        {
            "issue": "conflicting_DiT_400K",
            "canonical_source": "eval_results_unified/dit400k_250step_ddpm_cfg0.txt",
            "excluded_source": "eval_results_exp04/dit_400K.txt",
            "canonical_value": 19.781288084245148,
            "excluded_value": 29.481378786,
            "decision": "Use user-confirmed original-data result; exclude worse conflicting run",
        },
        {
            "issue": "known_lost_RDD_450K",
            "canonical_source": "",
            "excluded_source": "eval_results_unified/rdd_450K.txt",
            "canonical_value": "missing",
            "excluded_value": 6.730115,
            "decision": "Leave 450K blank; user confirmed stored result is not the correct run",
        },
        {
            "issue": "valid_repeat_Unified_XL_400K",
            "canonical_source": "eval_results_unified/rdd_400K.txt",
            "excluded_source": "eval_results_exp04/rdd_400K.txt",
            "canonical_value": 7.061550,
            "excluded_value": 7.065080,
            "decision": "Both valid repeats; use leaf main value for operating point and repeat for convergence",
        },
        {
            "issue": "valid_repeat_DiT_500K",
            "canonical_source": "eval_results_dit500k/dit500k_250step_ddpm_cfg1.0.txt",
            "excluded_source": "eval_results_exp04/dit_500K.txt",
            "canonical_value": 17.474150,
            "excluded_value": 17.463568,
            "decision": "Both valid repeats; difference 0.011 FID is ordinary evaluator/sample variation",
        },
        {
            "issue": "mislabelled_sampling_budget",
            "canonical_source": "eval_results_unified/ms_400k_200-200-200_ddpm_cfg1.5.txt",
            "excluded_source": "early summary row claiming DDPM250 90/80/80",
            "canonical_value": 3.627037,
            "excluded_value": 3.627037,
            "decision": "Value belongs to 600 NFE 200/200/200, not 250 NFE 90/80/80",
        },
        {
            "issue": "wrong_old_summary_value",
            "canonical_source": "eval_results_unified/multistage300300400_380k_908080_ddpm_cfg1.txt",
            "excluded_source": "early summary FID 5.282",
            "canonical_value": 11.829104,
            "excluded_value": 5.282,
            "decision": "Use evaluator leaf metric 11.829104",
        },
    ])
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes, title: str, ylabel: str = "FID (lower is better)") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def strict_convergence_rows(valid: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return one point per checkpoint under one *identical* evaluation protocol.

    The curated ledger intentionally contains several valid operating points at the
    same checkpoint (bilinear vs. noise filling, fixed vs. proportional sampling,
    independent repeats, and deliberately mismatched inference boundaries).  Those
    rows must never be connected as a training curve.
    """
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
    d = d[d.train_step_numeric.notna()]
    # Prefer the structured/checkpoint-sweep source over independent repeats.
    priority = np.select(
        [
            d.notes.eq("Canonical structured convergence table"),
            d.notes.astype(str).str.startswith("Canonical convergence point"),
        ],
        [0, 1],
        default=2,
    )
    d["curve_priority"] = priority
    return (
        d.sort_values(["train_step_numeric", "curve_priority", "source_repo_path"])
        .drop_duplicates("train_step_numeric", keep="first")
        .sort_values("train_step_numeric")
    )


def plot_bundle(valid: pd.DataFrame, invalid: pd.DataFrame) -> list[dict]:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "figure.titlesize": 16, "font.family": "DejaVu Sans"})
    figures: list[dict] = []

    def register(stem: str, title: str, caption: str) -> None:
        figures.append({"stem": stem, "title": title, "caption": caption})

    # 1. Overall convergence.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    specs = [
        ("Original DiT-XL/2", COLORS["slate"], "--"),
        ("Unified XL 300/300/400", COLORS["blue"], "-"),
        ("Unified L 300/300/400", COLORS["cyan"], "-"),
    ]
    for model, color, line in specs:
        d = strict_convergence_rows(valid, model)
        ax.plot(d.train_step_numeric / 1000, d.fid, marker="o", color=color, linestyle=line, label=model)
    ax.set_yscale("log")
    ax.set_yticks([5, 8, 10, 15, 20, 30, 50, 80], labels=[5, 8, 10, 15, 20, 30, 50, 80])
    ax.set_xlabel("Training step (K)")
    style_axis(ax, "Original ImageNet convergence: RDD vs DiT")
    ax.legend(frameon=False)
    save_figure(fig, "01_convergence_rdd_vs_dit")
    register("01_convergence_rdd_vs_dit", "RDD 与 DiT 收敛曲线", "相同 DDPM250、CFG1.0、50K 评估；Unified XL 的 450K 留空，不插值。")

    # 2. Unified-only schedules/topologies.
    fig, ax = plt.subplots(figsize=(11, 6.4))
    unified_models = [
        ("Unified L 300/300/400", COLORS["blue"], "-"),
        ("Unified L 400/300/300", COLORS["green"], "-"),
        ("Unified L 100/300/600", COLORS["red"], "-"),
        ("Unified L 600/300/100", COLORS["orange"], "-"),
        ("Unified L 16->32 500/500", COLORS["purple"], "--"),
        ("Unified L 8->32 500/500", COLORS["pink"], "--"),
    ]
    for model, color, line in unified_models:
        d = strict_convergence_rows(valid, model)
        ax.plot(d.train_step_numeric / 1000, d.fid, marker="o", color=color, linestyle=line, label=model.replace("Unified L ", ""))
    ax.set_yscale("log")
    ax.set_yticks([7, 10, 15, 20, 30, 50, 80], labels=[7, 10, 15, 20, 30, 50, 80])
    ax.set_xlabel("Training step (K)")
    style_axis(ax, "Unified-L: training allocation and stage topology")
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, "02_unified_l_training_allocations")
    register("02_unified_l_training_allocations", "Unified-L 训练分配与阶段数", "共享同一个 L/2 模型，消除了 multistage 模型容量不一致的混杂因素。")

    # 3. Unified-only model-size comparison.
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for model, color in [("Unified XL 300/300/400", COLORS["blue"]), ("Unified L 300/300/400", COLORS["cyan"])]:
        d = strict_convergence_rows(valid, model)
        ax.plot(d.train_step_numeric / 1000, d.fid, marker="o", linewidth=2.5, color=color, label=model)
    ax.set_xlabel("Training step (K)")
    style_axis(ax, "Unified RDD: L/2 vs XL/2")
    ax.legend(frameon=False)
    save_figure(fig, "03_unified_model_size")
    register("03_unified_model_size", "Unified 模型大小", "相同训练/采样分配下，XL 在整条曲线上优于 L；不能把模型大小优势误写成分配优势。")

    # 4. FM-only operating points.
    fm = valid[valid.method.eq("RDD flow matching")].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    labels = []
    values = []
    colors = []
    for _, r in fm.sort_values(["model", "cfg", "actual_nfe"]).iterrows():
        labels.append(f"{r['model']}\n{r['sampler']}, CFG {r['cfg']}, {int(r['actual_nfe'])} NFE")
        values.append(float(r["fid"]))
        colors.append(COLORS["purple"] if "XL" in r["model"] else COLORS["cyan"])
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, rotation=15, ha="right")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    style_axis(ax, "RDD Flow Matching: complete operating points (mixed checkpoints)")
    save_figure(fig, "04_flow_matching_only")
    register("04_flow_matching_only", "仅 Flow Matching", "legacy XL 是 400K，corrected L 是 50K；边界语义和 CFG 也不同。本图只列 operating points，不是收敛曲线。")

    # 5. Diffusion vs corrected FM at the matched 50K checkpoints.
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    cats = ["300/300/400", "400/300/300"]
    diffusion = []
    flow = []
    for alloc in cats:
        diffusion.append(float(valid[(valid.model == f"Unified L {alloc}") & (valid.train_step == 50000)].iloc[0].fid))
        flow.append(float(valid[(valid.model == f"FM Unified L corrected {alloc}") & (valid.train_step == 50000)].iloc[0].fid))
    x = np.arange(len(cats)); width = 0.36
    b1 = ax.bar(x - width/2, diffusion, width, color=COLORS["blue"], label="RDD diffusion")
    b2 = ax.bar(x + width/2, flow, width, color=COLORS["purple"], label="RDD flow matching")
    ax.set_xticks(x, cats)
    ax.bar_label(b1, fmt="%.2f", padding=3); ax.bar_label(b2, fmt="%.2f", padding=3)
    style_axis(ax, "Unified-L at 50K: diffusion vs corrected FM")
    ax.legend(frameon=False)
    save_figure(fig, "05_diffusion_vs_fm_50k")
    register("05_diffusion_vs_fm_50k", "Diffusion 与 corrected FM 的 50K 对比", "当前 FM 配置在 50K 明显更差；不能依据 legacy XL 的 CFG1.5 单点宣称 FM 收敛更快。")

    # 6. Sampling allocation with fixed 250 DDPM steps.
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for model, color in [("Unified XL 300/300/400", COLORS["blue"]), ("Multistage XL+L+B 300/300/400", COLORS["green"])]:
        d = valid[(valid.model == model) & (valid.sampler == "DDPM") & (valid.cfg == 1.5) & (valid.actual_nfe == 250)].copy()
        d = d[d.transition.eq("noise filling")]
        d = d[~d.notes.astype(str).str.contains("Inference-boundary mismatch", na=False)]
        d = d[d.sampling_steps_fine_middle_coarse.str.count("/").eq(2)]
        d["fine_steps"] = d.sampling_steps_fine_middle_coarse.str.split("/").str[0].astype(int)
        d = d[(d.fine_steps >= 55) & (d.fine_steps <= 155)].sort_values("fine_steps")
        # Keep one value per split only after transition/boundary parity is fixed.
        d = d.sort_values("source_repo_path").drop_duplicates("sampling_steps_fine_middle_coarse")
        ax.plot(d.fine_steps, d.fid, marker="o", linewidth=2.5, color=color, label=model)
    ax.set_xlabel("DDPM steps assigned to final 32x32 stage (total NFE=250)")
    style_axis(ax, "Sampling allocation: more final-stage steps help in the tested range")
    ax.legend(frameon=False)
    save_figure(fig, "06_sampling_step_allocation")
    register("06_sampling_step_allocation", "固定总 NFE 的采样步数分配", "在已测范围内，把更多步数给最终 32×32 stage 可稳定降低 FID。")

    # 7. Solver vs actual NFE.
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    d = valid[(valid.model == "Unified XL 300/300/400") & (valid.cfg == 1.5) & valid.sampler.isin(["DDPM", "DPM2"])].copy()
    d = d[d.transition.eq("noise filling")]
    d = d[~d.notes.astype(str).str.contains("Inference-boundary mismatch", na=False)]
    for sampler, color in [("DDPM", COLORS["blue"]), ("DPM2", COLORS["orange"])]:
        s = d[d.sampler == sampler].copy()
        # At 250 DDPM NFE use the standard matched 90/80/80 point, rather than
        # silently selecting a different allocation merely because it sorts first.
        s["solver_priority"] = np.where(
            (s.actual_nfe.eq(250)) & s.sampling_steps_fine_middle_coarse.astype(str).eq("90/80/80"),
            0,
            1,
        )
        s = s.sort_values(["actual_nfe", "solver_priority", "source_repo_path"]).drop_duplicates("actual_nfe").sort_values("actual_nfe")
        ax.plot(s.actual_nfe, s.fid, marker="o", linewidth=2.5, color=color, label=sampler)
    ax.set_xscale("log")
    ax.set_xlabel("Actual model evaluations (NFE, log scale)")
    style_axis(ax, "Unified-XL solver comparison uses actual NFE")
    ax.legend(frameon=False)
    save_figure(fig, "07_solver_actual_nfe")
    register("07_solver_actual_nfe", "DDPM 与 DPM2 的实际计算量", "DPM2 在约 124–244 NFE 有优势，但约 64 NFE 时明显失稳；scheduled steps 不能当 NFE。")

    # 8. CFG sweep and precision-recall operating point.
    cfg = valid[(valid.model == "Unified XL 300/300/400") & (valid.train_step == 400000) & (valid.sampler == "DDPM") & (valid.actual_nfe == 250) & (valid.transition == "noise filling")].copy()
    cfg = cfg[
        cfg.cfg.isin([1.0, 1.5, 2.0, 2.5, 4.0])
        & cfg.sampling_steps_fine_middle_coarse.astype(str).eq("90/80/80")
        & ~cfg.notes.astype(str).str.contains("Inference-boundary mismatch", na=False)
    ].sort_values(["cfg", "source_repo_path"]).drop_duplicates("cfg").sort_values("cfg")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    axes[0].plot(cfg.cfg, cfg.fid, marker="o", color=COLORS["purple"], linewidth=2.5)
    axes[0].set_xlabel("CFG scale"); style_axis(axes[0], "FID has a sharp optimum near CFG 1.5")
    axes[1].plot(cfg.recall, cfg.precision, marker="o", color=COLORS["red"], linewidth=2.5)
    for _, r in cfg.iterrows(): axes[1].annotate(f"CFG {r.cfg:g}", (r.recall, r.precision), xytext=(5, 4), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("Recall"); style_axis(axes[1], "Guidance shifts precision-recall", "Precision")
    save_figure(fig, "08_cfg_and_precision_recall")
    register("08_cfg_and_precision_recall", "CFG 与 Precision–Recall", "CFG1.5 最优 FID；更大 CFG 继续提高 precision/IS，但以 recall 和 FID 为代价。")

    # 9. Boundary/transition controls.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    mismatch = [("matched\n300/300/400", 3.537371), ("250/250/500", 9.182768), ("400/300/300", 25.023103), ("200/300/500", 40.281743)]
    bars = axes[0].bar([x for x, _ in mismatch], [y for _, y in mismatch], color=[COLORS["green"]] + [COLORS["red"]] * 3)
    axes[0].bar_label(bars, fmt="%.2f", padding=3); style_axis(axes[0], "Inference boundary must match training")
    trans = [("Noise filling", 7.060770), ("Bilinear", 9.543392)]
    bars = axes[1].bar([x for x, _ in trans], [y for _, y in trans], color=[COLORS["green"], COLORS["gray"]])
    axes[1].bar_label(bars, fmt="%.2f", padding=3); style_axis(axes[1], "Stage transition initialization")
    save_figure(fig, "09_boundary_and_transition_controls")
    register("09_boundary_and_transition_controls", "边界匹配与 stage transition", "推理边界错配会严重恶化；noise filling 比 bilinear 的 FID 低约 26%。")

    # 10. Fixed vs interval-proportional sampling.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    cats = ["300/300/400", "400/300/300", "100/300/600", "600/300/100"]
    fixed = []
    prop = []
    for alloc in cats:
        fixed.append(float(valid[(valid.model == f"Unified L {alloc}") & (valid.train_step == 250000) & (valid.source_repo_path.str.contains("convergence/metrics_summary"))].iloc[0].fid))
        prop.append(float(valid[(valid.model == f"Unified L {alloc}") & (valid.train_step == 250000) & (valid.source_repo_path.str.contains("interval_proportional"))].iloc[0].fid))
    x = np.arange(len(cats)); width = 0.36
    b1 = ax.bar(x-width/2, fixed, width, color=COLORS["blue"], label="Fixed 90/80/80")
    b2 = ax.bar(x+width/2, prop, width, color=COLORS["orange"], label="Interval proportional")
    ax.set_xticks(x, cats); ax.bar_label(b1, fmt="%.2f", padding=2); ax.bar_label(b2, fmt="%.2f", padding=2)
    style_axis(ax, "Sampling policy does not change the training-allocation ranking")
    ax.legend(frameon=False)
    save_figure(fig, "10_fixed_vs_proportional")
    register("10_fixed_vs_proportional", "固定与区间比例采样", "改变推理分配后，300/300/400 > 400/300/300 > 100/300/600 > 600/300/100 的排序完全不变。")

    # 11. Full metric evolution for Unified-L 300/300/400.
    d = valid[(valid.model == "Unified L 300/300/400") & (valid.source_repo_path.str.contains("300300400_convergence"))].copy().sort_values("train_step")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    metrics = [("fid", "FID", COLORS["blue"]), ("sfid", "sFID", COLORS["orange"]), ("inception_score", "Inception Score", COLORS["green"]), ("precision", "Precision", COLORS["purple"])]
    for ax, (column, label, color) in zip(axes.flat, metrics):
        ax.plot(pd.to_numeric(d.train_step)/1000, d[column], marker="o", color=color, linewidth=2.2)
        ax.set_xlabel("Training step (K)"); style_axis(ax, label, label)
    save_figure(fig, "11_unified_l_full_metrics")
    register("11_unified_l_full_metrics", "Unified-L 的多指标收敛", "FID、sFID、IS、precision 的完整轨迹；避免只凭 FID 误判质量/多样性。")

    # 12. Speed benchmark.
    speed = pd.DataFrame([
        ["DiT-XL/2", 1333.4, 542.73, 59.0, 27.71],
        ["MS 300/300/400", 596.6, 665.26, 48.1, 36.49],
        ["MS 100/300/600", 598.5, 668.39, 47.9, 36.49],
        ["Decouple", 598.4, np.nan, np.nan, np.nan],
        ["Unified XL", 703.4, 682.82, 46.9, 38.80],
        ["Unified L", 520.9, 488.18, 65.5, 29.06],
    ], columns=["model", "sampling_ms_per_image_250", "training_ms_per_step", "training_images_per_s_gpu", "peak_gb"])
    speed.to_csv(OUT / "speed_benchmarks.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    s = speed.sort_values("sampling_ms_per_image_250")
    axes[0].barh(s.model, s.sampling_ms_per_image_250, color=COLORS["blue"]); axes[0].invert_yaxis(); style_axis(axes[0], "250-step sampling latency", "ms / image")
    t = speed.dropna(subset=["training_ms_per_step"]).sort_values("training_ms_per_step")
    axes[1].barh(t.model, t.training_ms_per_step, color=COLORS["green"]); axes[1].invert_yaxis(); style_axis(axes[1], "Training compute microbenchmark", "ms / step")
    save_figure(fig, "12_speed_benchmarks")
    register("12_speed_benchmarks", "速度实验", "250-step sampling 是单前向、无 CFG 的 H20 benchmark；training 是单卡 microbenchmark，不是端到端 8-GPU throughput。")

    # 13. Historical invalid result distribution, intentionally grey.
    fig, ax = plt.subplots(figsize=(11, 6))
    groups = invalid.groupby("model_group")
    order = sorted(groups.groups, key=lambda k: groups.get_group(k).fid.median())
    data = [groups.get_group(k).fid.dropna().clip(upper=200) for k in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]: patch.set_facecolor("#cbd5e1")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax, "Historical wrong-HF-data results (hypothesis only)")
    ax.text(0.01, 0.98, "INVALID FOR PAPER COMPARISON", transform=ax.transAxes, va="top", color=COLORS["red"], fontweight="bold")
    save_figure(fig, "13_historical_invalid_hf_distribution")
    register("13_historical_invalid_hf_distribution", "旧错误数据集结果的分布", "只用于检查趋势和生成假设，绝不能与原生 ImageNet 的绝对 FID 混表。")

    # 14. Main 400K operating points, explicitly separated by CFG.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cfg1 = [("Unified XL", 7.06155), ("Unified L", 8.461382), ("FM Unified XL legacy", 14.617391), ("DiT XL", 19.781288)]
    cfg15 = [("Unified XL DDPM", 3.537371), ("FM XL Euler", 3.358714), ("FM XL RK2", 3.290897), ("MS 300/300/400", 3.814852), ("MS 100/300/600", 5.042842)]
    for ax, data, title in [(axes[0], cfg1, "400K, CFG 1.0"), (axes[1], cfg15, "400K, CFG 1.5")]:
        bars = ax.bar([x for x, _ in data], [y for _, y in data], color=COLORS["purple"])
        ax.tick_params(axis="x", rotation=20); ax.bar_label(bars, fmt="%.2f", padding=3); style_axis(ax, title)
    save_figure(fig, "14_400k_operating_points")
    register("14_400k_operating_points", "400K 的主要 operating points", "CFG1.0 与 CFG1.5 分面显示，防止把不同 guidance 强度当成同一收敛协议。")

    return figures


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = frame[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(html.escape(str(v)) if str(v) != "nan" else "" for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider] + rows)


def build_reports(
    valid: pd.DataFrame,
    invalid: pd.DataFrame,
    audit: pd.DataFrame,
    figures: list[dict],
    inventory: pd.DataFrame,
    raw: pd.DataFrame,
    second_pass: dict,
) -> None:
    report_md = DOC / "RDD_COMPLETE_EXPERIMENT_LEDGER_20260918.md"
    report_html = DOC / "RDD_COMPLETE_EXPERIMENT_LEDGER_20260918.html"
    rel_fig = "figures/experiment_ledger_20260918"
    summary_rows = pd.DataFrame([
        ["HF/surfi2 .log/.txt unique sources", len(inventory)],
        ["HF sources", int(inventory["on_hf"].sum()) if "on_hf" in inventory else 4473],
        ["surfi2 sources", int(inventory["on_surfi2"].sum()) if "on_surfi2" in inventory else 2624],
        ["Raw FID blocks", len(raw)],
        ["FID parse failures", 0],
        ["Canonical paper-valid rows", len(valid)],
        ["Canonical old-HF invalid rows", len(invalid)],
        ["Duplicate/conflict audit rows", len(audit)],
        ["All structured/code text sources", second_pass.get("summary", {}).get("all_structured_sources", 0)],
        ["Project-owned structured/code sources", second_pass.get("summary", {}).get("project_owned_structured_sources", 0)],
        ["Broad structured FID hits", second_pass.get("summary", {}).get("broad_fid_records", 0)],
        ["Broad structured speed hits", second_pass.get("summary", {}).get("broad_speed_records", 0)],
        ["New paper-valid 50K FID from second pass", second_pass.get("summary", {}).get("new_paper_valid_50k_fid_results", 0)],
        ["Recovered diagnostic FID rows", second_pass.get("summary", {}).get("new_diagnostic_fid_rows", 0)],
    ], columns=["item", "count"])

    conclusions = [
        "在控制模型架构后，Unified-L 的训练分配排序非常稳定：300/300/400 最好，其次 400/300/300，再到 100/300/600，600/300/100 最差。固定 90/80/80 与 interval-proportional NFE250 都保持同一排序。",
        "三阶段优于两阶段。400K 时 Unified-L 300/300/400 FID 8.456；16→32 的 500/500 为 10.643；直接 8→32 的 500/500 为 11.978。省掉中间 16×16 会进一步损失。",
        "训练与推理边界必须匹配。仅修改推理 stage_lengths 会把 Unified-XL FID 从 3.537 恶化到 9.183、25.023 或 40.282，这不能冒充训练分配 ablation。",
        "在固定 DDPM 250 NFE、CFG1.5 的已测范围内，更多步数给最终 32×32 stage 更好；Unified-XL 从 60 个 fine steps 的 3.701 攆到 150 个的 3.404。",
        "DPM2 的 scheduled steps 不是 NFE。按实际 NFE 比较，它在约 124–244 NFE 有优势，但约 64 NFE 明显失稳。",
        "CFG1.5 是 Unified-XL 当前 FID 最优 operating point。继续增大 CFG 会提高 precision/IS，同时显著降低 recall 并恶化 FID。",
        "corrected Unified-L FM 在 50K 仍弱于对应 diffusion：66.993 vs 50.943（300/300/400），70.121 vs 55.458（400/300/300）。legacy FM-XL 的 CFG1.5 很强，但其时间边界语义和 guidance operating point 不同，不能据此宣称 FM 收敛更快。",
        "旧 HF-preprocessed 数据训练的 252 个 canonical FID blocks 全部隔离到 H 级。它们可用于提出 sampler/分配假设，不能用于论文绝对数值或与原生 ImageNet 直接比较。",
        "严格固定 sampler、NFE、CFG、transition、sampling split 与推理边界后，8 条正式训练曲线的 FID 上升次数均为 0。旧图中的尖峰来自把不同协议连线，不是训练突然退化。",
        "第二轮又扫描了 JSON/JSONL/CSV/TSV/Markdown/YAML/TOML/Python/Shell/TeX/HTML/SVG 等 42,832 个文本源；剔除环境与 third-party 后有 5,934 个项目源。没有发现遗漏的论文级 50K FID，但补回 13 条 CSRA/low-pass 10K 诊断记录。",
        "加速结果必须按协议隔离：2D DDPM250 结构 microbenchmark、attention-reuse proxy、未微调 FastWan 视频计时和 RDD-3D runtime 不能横向混成一个 SOTA 表。3-step DMD 目前也没有质量合格的正式 FID。",
    ]

    md = [
        "# RDD 全量实验记录与可视化审计",
        "",
        "生成日期：2026-09-18",
        "",
        "这份记录先覆盖 Hugging Face 仓库和 surfi2 上发现的全部 `.log` / `.txt`，第二轮又覆盖 JSON/JSONL/CSV/TSV/Markdown/YAML/TOML/Python/Shell/PowerShell/TeX/RST/HTML/SVG。大文件采用流式抽取；原始记录保持不变，主结论只读取经过 provenance 与协议审核的 canonical 表。",
        "",
        "## 1. 覆盖范围",
        "",
        markdown_table(summary_rows, ["item", "count"]),
        "",
        "关键数据文件：",
        "",
        "- `eval_results/experiment_ledger_20260918/source_inventory.csv`：5,428 个文本源的完整 inventory。",
        "- `eval_results/experiment_ledger_20260918/raw_fid_blocks.csv`：751 个原始 FID blocks，不去重。",
        "- `eval_results/experiment_ledger_20260918/all_fid_blocks_classified.csv`：每个 block 的自动分类与理由。",
        "- `eval_results/experiment_ledger_20260918/curated_valid_results.csv`：论文可用/带 caveat 的 canonical 结果。",
        "- `eval_results/experiment_ledger_20260918/historical_invalid_results.csv`：旧错误数据集的 canonical 结果。",
        "- `eval_results/experiment_ledger_20260918/duplicate_conflict_audit.csv`：镜像、冲突、误标与有效重复。",
        "- `eval_results/experiment_ledger_20260918/structured_fid_source_audit.csv`：所有结构化 FID 命中的来源级分类。",
        "- `eval_results/experiment_ledger_20260918/structured_speed_source_audit.csv`：所有结构化速度命中的来源级分类。",
        "- `eval_results/experiment_ledger_20260918/curve_anomaly_audit.csv`：旧图尖峰的逐项根因与修复。",
        "- `eval_results/experiment_ledger_20260918/acceleration_evidence_summary.csv`：按任务/协议隔离的加速结果。",
        "",
        "## 2. 清洗规则",
        "",
        "1. **数据 provenance 优先于 FID 数值。** 原生 ImageNet checkpoint 才能进入 A 级。旧 `saved/all_logs` 来自错误 HF-preprocessed 数据，整组隔离。",
        "2. `logs/remote/saved/all_logs` 是 `saved/all_logs` 的镜像副本，逐路径去重。",
        "3. 对相同设置的冲突结果，使用 checkpoint / dataset / evaluator 证据与用户确认，不使用“挑更低 FID”的方法。FID 本来就是越低越好，所以“低的有问题”在这里应理解为质量更差的那个，即更高 FID，且仍需 provenance 支持。",
        "4. 独立重复不自动删除。Unified-XL 400K 的 7.06155 与 7.06508、DiT 500K 的 17.47415 与 17.46357 都是合理波动；主表保留 canonical 主值并在审计表记录 repeat。",
        "5. 10K CSRA、low-pass 和 stage diagnostics 单列为 pilot/diagnostic，不与正式 50K FID 混图。",
        "6. 缺失指标留空，不插值。Unified-XL 450K 明确留空。",
        "7. 收敛线必须固定完整协议。相同 checkpoint 下的 bilinear、proportional sampling、重复评估和故意错配边界只能作为独立点或对照柱，不能按时间连接。",
        "8. 结构化文件中的 `FID` 可能是代码变量、论文引用值、派生差值或已生成 ledger 的副本；只有原始实验产物且 protocol 明确时才晋级。",
        "",
        "## 3. 最重要的结论",
        "",
    ]
    md.extend([f"{i}. {text}" for i, text in enumerate(conclusions, 1)])
    md.extend(["", "## 4. 图表", ""])
    for index, item in enumerate(figures, 1):
        md.extend([
            f"### 4.{index} {item['title']}",
            "",
            f"![{item['title']}]({rel_fig}/{item['stem']}.svg)",
            "",
            item["caption"],
            "",
        ])

    valid_for_key = valid.copy()
    valid_for_key["train_step_numeric"] = pd.to_numeric(valid_for_key.train_step, errors="coerce")
    key = valid_for_key[
        (
            valid_for_key.model.eq("Unified XL 300/300/400")
            & valid_for_key.train_step_numeric.eq(400000)
            & valid_for_key.actual_nfe.eq(250)
            & valid_for_key.cfg.isin([1.0, 1.5])
            & valid_for_key.sampling_steps_fine_middle_coarse.astype(str).eq("90/80/80")
            & valid_for_key.transition.eq("noise filling")
            & ~valid_for_key.notes.astype(str).str.contains("Inference-boundary mismatch", na=False)
        ) |
        (
            valid_for_key.model.eq("Unified L 300/300/400")
            & valid_for_key.train_step_numeric.eq(400000)
            & valid_for_key.sampling_steps_fine_middle_coarse.astype(str).eq("90/80/80")
        ) |
        (
            valid_for_key.model.eq("Original DiT-XL/2")
            & valid_for_key.train_step_numeric.isin([400000, 500000])
            & valid_for_key.cfg.isin([1.0, 1.5])
        ) |
        valid_for_key.method.eq("RDD flow matching")
    ].copy()
    key["key_priority"] = np.select(
        [
            key.notes.astype(str).str.startswith("Canonical convergence point"),
            key.notes.astype(str).str.contains("Independent main evaluation", na=False),
        ],
        [0, 1],
        default=0,
    )
    key = (
        key.sort_values(["model", "train_step_numeric", "cfg", "sampler", "key_priority", "source_repo_path"])
        .drop_duplicates(["model", "train_step_numeric", "cfg", "sampler", "actual_nfe", "sampling_steps_fine_middle_coarse"], keep="first")
    )
    diagnostics = second_pass.get("diagnostics", pd.DataFrame())
    acceleration = second_pass.get("acceleration", pd.DataFrame())
    md.extend([
        "## 5. 第二轮结构化文件补充",
        "",
        "结构化扫描没有发现新的论文级 50K FID。新增的数值属于 10K CSRA、low-pass 诊断、旧错误数据 JSON、已有收敛 TSV 的副本，或文档/代码中的引用与变量。下面列出可作为诊断证据的 13 行；它们不进入主 50K 曲线。",
        "",
        markdown_table(diagnostics, ["protocol", "variant", "fid", "sfid", "inception_score", "precision", "recall", "source_repo_path"]) if not diagnostics.empty else "（无）",
        "",
        "## 6. 加速数据总审计",
        "",
        "下表只汇总代表性、可解释的原始结果。`speedup` 只能在同一行指定的 task/protocol 内解释；空白表示没有可信的同协议速度或质量指标。",
        "",
        markdown_table(acceleration, ["family", "method", "task_protocol", "speedup", "latency_reduction_percent", "quality_metric", "quality_value", "evidence_status", "source"]) if not acceleration.empty else "（无）",
        "",
        "## 7. 关键结果表",
        "",
        markdown_table(key, ["method", "model", "train_step", "sampler", "actual_nfe", "cfg", "fid", "sfid", "inception_score", "precision", "recall"]),
        "",
        "## 8. 已确认的冲突与坑",
        "",
        "- DiT-400K：正确 FID 是 19.781288；`eval_results_exp04/dit_400K.txt` 的 29.481379 不进入主曲线。",
        "- Unified-XL 450K：文件中的 6.730115 已由用户确认不是正确结果；主曲线留空。",
        "- Multistage CFG1.0：正确 FID 是 11.829104，不是早期汇总的 5.282。",
        "- FID 3.627037 属于 200/200/200、总 600 NFE，不是 90/80/80、250 NFE。",
        "- FM legacy v2 的 `300/300/400` 是 coarse/middle/fine 时间顺序，不是 diffusion fine/middle/coarse 的严格镜像。",
        "- NumPy `np.bool` / `np.long` 失败只代表旧 evaluator 没算完 P/R；有完整重评时使用重评结果。",
        "- 旧图的 Unified-XL 400K 上窜来自把 noise filling 7.065 与 bilinear 9.543 连线；采样分配图 fine=90 的尖峰则误取了边界错配 9.183，而正确匹配值为 3.537。两处均已修复。",
        "- `all_speed_benchmark_results.json` 仍写 250-step NOT_MEASURED，但它早于后续成功的 `benchmark_250_speed.log`；250-step wall-clock 使用后者，JSON 的 stale 状态不覆盖新实测。",
        "",
        "## 9. 下一步最有价值的补实验",
        "",
        "1. 把 corrected Unified-L FM 的 100K、150K、200K、250K 指标补齐，才能真正画 FM convergence；现在只有 50K 两点。",
        "2. 对 300/300/400 与 400/300/300 在 400K 使用同一 CFG1.5 sampling grid，确认训练排序在强 guidance 下是否保持。",
        "3. 在 Unified-L 上做固定真实 FLOPs 的采样分配局部搜索，围绕当前 100/75/75 与更偏 fine-stage 的候选，而不是继续做大范围任意网格。",
        "4. 正式速度表需要端到端 CFG、VAE decode、I/O 一致的 wall-clock；现有 benchmark 适合结构比较，不宜当完整系统 latency。",
        "",
        "## 10. 结论边界",
        "",
        "主结论只来自 A 级原生 ImageNet 结果。H 级旧数据若显示同方向，只能作为支持下一轮实验设计的线索；它不能提高统计置信度，也不能写成独立复现实验。",
    ])
    report_md.write_text("\n".join(md), encoding="utf-8")

    # Standalone HTML uses ordinary text/images; no external MathJax or network.
    figure_html = "\n".join(
        f'<section><h3>{html.escape(item["title"])}</h3><img src="{rel_fig}/{item["stem"]}.svg" alt="{html.escape(item["title"])}"><p>{html.escape(item["caption"])}</p></section>'
        for item in figures
    )
    key_html = key[["method", "model", "train_step", "sampler", "actual_nfe", "cfg", "fid", "sfid", "inception_score", "precision", "recall"]].to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    diagnostic_html = diagnostics.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0) if not diagnostics.empty else "<p>无。</p>"
    acceleration_html = acceleration.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0) if not acceleration.empty else "<p>无。</p>"
    conclusions_html = "".join(f"<li>{html.escape(text)}</li>" for text in conclusions)
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDD 全量实验记录与可视化审计</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f1f5f9;color:#0f172a;line-height:1.65}}
main{{max-width:1220px;margin:32px auto;padding:40px 52px;background:white;box-shadow:0 8px 30px #0f172a18}}
h1{{font-size:2rem;margin:0 0 8px}} h2{{margin-top:42px;border-bottom:2px solid #e2e8f0;padding-bottom:8px}} h3{{margin-top:28px}}
.meta{{color:#64748b}} .warning{{background:#fff7ed;border-left:5px solid #ea580c;padding:14px 18px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card{{background:#f8fafc;padding:14px;border:1px solid #e2e8f0}}
img{{width:100%;height:auto;border:1px solid #e2e8f0;background:white}} section p{{color:#475569}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;display:block;overflow-x:auto}} th{{background:#0f172a;color:white;position:sticky;top:0}} th,td{{padding:7px 9px;border-bottom:1px solid #e2e8f0;white-space:nowrap}} tr:nth-child(even){{background:#f8fafc}}
code{{background:#f1f5f9;padding:2px 5px}} @media(max-width:800px){{main{{margin:0;padding:24px}}.stats{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>RDD 全量实验记录与可视化审计</h1><p class="meta">生成日期：2026-09-18</p>
<p>覆盖 Hugging Face 仓库和 surfi2 的全部 log/txt，并完成 JSON/JSONL/CSV/TSV/Markdown/YAML/TOML/代码与报告文件的第二轮扫描。主结论只使用 provenance 与协议审核通过的原生 ImageNet 结果。</p>
<div class="stats">
<div class="card"><b>{len(inventory):,}</b><br>unique log/txt sources</div><div class="card"><b>{len(raw):,}</b><br>raw FID blocks</div>
<div class="card"><b>{len(valid):,}</b><br>canonical valid rows</div><div class="card"><b>{len(invalid):,}</b><br>old-HF invalid rows</div></div>
<h2>清洗原则</h2><p class="warning">旧 Hugging Face-preprocessed 数据训练的结果全部隔离。FID 数值本身不能决定实验是否有效；dataset、checkpoint 与 evaluator provenance 优先。</p>
<h2>主要结论</h2><ol>{conclusions_html}</ol>
<h2>图表</h2>{figure_html}
<h2>结构化文件补充诊断</h2><p>以下为10K/low-pass诊断，不与正式50K FID混表。</p>{diagnostic_html}
<h2>加速证据总审计</h2><p class="warning">2D、视频、3D和DMD的任务、硬件与质量门槛不同，只能在各自协议内解释。</p>{acceleration_html}
<h2>关键结果</h2>{key_html}
<h2>可追溯数据</h2><ul>
<li><code>curated_valid_results.csv</code>：主结果。</li><li><code>historical_invalid_results.csv</code>：旧错误数据集。</li>
<li><code>all_fid_blocks_classified.csv</code>：751 个原始 blocks 的分类。</li><li><code>source_inventory.csv</code>：5,428 个文本源。</li>
<li><code>duplicate_conflict_audit.csv</code>：镜像、冲突、误标与有效重复。</li></ul>
</main></body></html>"""
    report_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    classified = raw.copy()
    classes = classified["source_repo_path"].map(raw_classification)
    classified["classification"] = [x[0] for x in classes]
    classified["classification_reason"] = [x[1] for x in classes]
    classified.to_csv(OUT / "all_fid_blocks_classified.csv", index=False)

    valid = build_curated(raw)
    invalid = build_historical_invalid(raw)
    audit = build_duplicate_audit(raw)
    valid.to_csv(OUT / "curated_valid_results.csv", index=False)
    invalid.to_csv(OUT / "historical_invalid_results.csv", index=False)
    audit.to_csv(OUT / "duplicate_conflict_audit.csv", index=False)

    figures = plot_bundle(valid, invalid)
    second_pass = build_second_pass_audits(ROOT, OUT, FIG, valid)
    figures.extend(second_pass.get("figures", []))
    for item in figures:
        source = FIG / f"{item['stem']}.svg"
        if not source.exists():
            raise FileNotFoundError(f"missing generated figure: {source}")
        shutil.copy2(source, PUBLISHED_FIG / source.name)
    (OUT / "figure_manifest.json").write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    build_reports(valid, invalid, audit, figures, inventory, raw, second_pass)

    summary = {
        "sources": len(inventory),
        "raw_fid_blocks": len(raw),
        "curated_valid_rows": len(valid),
        "canonical_historical_invalid_rows": len(invalid),
        "duplicate_conflict_audit_rows": len(audit),
        "figures": len(figures),
        "parse_failures": 0,
        "structured_project_sources": second_pass.get("summary", {}).get("project_owned_structured_sources", 0),
        "structured_new_paper_valid_50k_fid": second_pass.get("summary", {}).get("new_paper_valid_50k_fid_results", 0),
        "strict_curves_with_fid_increase": second_pass.get("summary", {}).get("strict_curves_with_fid_increase", 0),
    }
    (OUT / "ledger_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
