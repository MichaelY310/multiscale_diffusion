#!/usr/bin/env python3
"""Evaluate several ADM NPZ sample sets while reusing one reference pass.

The metric definitions are identical to ``guided-diffusion/evaluations/evaluator.py``.
Only the expensive reference Inception activations/statistics/manifold radii are
cached across samples.  Each sample receives a separate standard five-metric log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import tensorflow.compat.v1 as tf


METRIC_PATTERNS = {
    name: re.compile(rf"(?m)^{re.escape(name)}:\s*([-+0-9.eE]+)\s*$")
    for name in ("Inception Score", "FID", "sFID", "Precision", "Recall")
}


def _read_complete_metrics(path: Path):
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            return None
        metrics[name] = float(match.group(1))
    return metrics


def _parse_sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sample must be LABEL=/absolute/path.npz")
    label, path = value.split("=", 1)
    if not label or any(ch in label for ch in "/\\"):
        raise argparse.ArgumentTypeError("sample LABEL must be a non-empty filename-safe name")
    return label, Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sample", action="append", type=_parse_sample, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-suffix", required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root / "guided-diffusion" / "evaluations"))
    import evaluator as adm  # noqa: E402

    reference = args.reference.expanduser().resolve()
    if not reference.is_file():
        parser.error(f"reference not found: {reference}")
    for label, path in args.sample:
        if not path.is_file():
            parser.error(f"sample not found: {label}={path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    pending = []
    for label, sample_path in args.sample:
        output = output_dir / f"{label}_{args.name_suffix}.txt"
        metrics = _read_complete_metrics(output)
        if metrics is None:
            pending.append((label, sample_path))
        else:
            summary[label] = metrics
            print(f"[skip] all five metrics already exist: {output}", flush=True)

    summary_path = output_dir / f"summary_{args.name_suffix}.json"
    if not pending:
        summary_path.write_text(
            json.dumps(
                {
                    "reference": str(reference),
                    "metric_implementation": "OpenAI ADM evaluator; shared reference activations only",
                    "datasets": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[done] every sample already has IS/FID/sFID/Precision/Recall", flush=True)
        return

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    engine = adm.Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...", flush=True)
    engine.warmup()
    print(f"computing shared reference activations: {reference}", flush=True)
    ref_acts = engine.read_activations(str(reference))
    ref_stats, ref_stats_spatial = engine.read_statistics(str(reference), ref_acts)
    print("computing shared reference manifold radii...", flush=True)
    ref_radii = engine.manifold_estimator.manifold_radii(ref_acts[0])

    for label, sample_path in pending:
        print(f"\n[{label}] computing sample activations: {sample_path}", flush=True)
        sample_acts = engine.read_activations(str(sample_path))
        sample_stats, sample_stats_spatial = engine.read_statistics(str(sample_path), sample_acts)
        sample_radii = engine.manifold_estimator.manifold_radii(sample_acts[0])
        precision_array, recall_array = engine.manifold_estimator.evaluate_pr(
            ref_acts[0], ref_radii, sample_acts[0], sample_radii
        )
        metrics = {
            "Inception Score": engine.compute_inception_score(sample_acts[0]),
            "FID": float(sample_stats.frechet_distance(ref_stats)),
            "sFID": float(sample_stats_spatial.frechet_distance(ref_stats_spatial)),
            "Precision": float(precision_array[0]),
            "Recall": float(recall_array[0]),
        }
        summary[label] = metrics
        lines = [
            f"Inception Score: {metrics['Inception Score']}",
            f"FID: {metrics['FID']}",
            f"sFID: {metrics['sFID']}",
            f"Precision: {metrics['Precision']}",
            f"Recall: {metrics['Recall']}",
        ]
        for line in lines:
            print(line, flush=True)
        output = output_dir / f"{label}_{args.name_suffix}.txt"
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, output)

    summary_path.write_text(
        json.dumps(
            {
                "reference": str(reference),
                "metric_implementation": "OpenAI ADM evaluator; shared reference activations only",
                "datasets": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
