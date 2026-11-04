#!/usr/bin/env python3
"""Paired asset bootstrap for the frozen dynamic-sparse final-test methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_asset_means(
    eval_root: Path, method: str, seeds: tuple[int, ...], metric: str
) -> dict[str, float]:
    by_asset: dict[str, list[float]] = {}
    for seed in seeds:
        path = eval_root / f"{method}_seed{seed}_heldout_support.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["samples"]:
            by_asset.setdefault(row["asset"], []).append(float(row[metric]))
    missing = [asset for asset, values in by_asset.items() if len(values) != len(seeds)]
    if missing:
        raise ValueError(f"{method}/{metric}: assets missing seeds: {missing[:3]}")
    return {asset: float(np.mean(values)) for asset, values in by_asset.items()}


def paired_bootstrap(
    selected: dict[str, float],
    baseline: dict[str, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    assets = sorted(set(selected) & set(baseline))
    if len(assets) != len(selected) or len(assets) != len(baseline):
        raise ValueError("paired methods do not contain the same assets")
    delta = np.asarray([selected[asset] - baseline[asset] for asset in assets])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "assets": len(assets),
        "selected_macro_mean": float(np.mean([selected[a] for a in assets])),
        "baseline_macro_mean": float(np.mean([baseline[a] for a in assets])),
        "mean_paired_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "asset_win_rate": float(np.mean(delta > 0)),
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(314159, 271828))
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    metrics = ("support_iou", "support_recall", "support_precision")
    report = {
        "protocol": {
            "selected": args.selected,
            "baselines": args.baselines,
            "seeds": list(seeds),
            "bootstrap_draws": args.draws,
            "bootstrap_seed": args.seed,
            "estimand": "macro per-asset metric after averaging two sampling seeds",
        },
        "comparisons": [],
    }
    for baseline_index, baseline in enumerate(args.baselines):
        comparison = {"baseline": baseline, "metrics": {}}
        for metric_index, metric in enumerate(metrics):
            selected_values = load_asset_means(
                args.eval_root, args.selected, seeds, metric
            )
            baseline_values = load_asset_means(args.eval_root, baseline, seeds, metric)
            comparison["metrics"][metric] = paired_bootstrap(
                selected_values,
                baseline_values,
                draws=args.draws,
                seed=args.seed + 100 * baseline_index + metric_index,
            )
        report["comparisons"].append(comparison)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "| Baseline | Metric | Paired delta | 95% bootstrap CI | Asset win rate | P(delta>0) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "support_iou": "IoU",
        "support_recall": "Recall",
        "support_precision": "Precision",
    }
    for comparison in report["comparisons"]:
        for metric, values in comparison["metrics"].items():
            lines.append(
                "| {baseline} | {metric} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] | {wins:.1%} | {prob:.1%} |".format(
                    baseline=comparison["baseline"],
                    metric=labels[metric],
                    delta=values["mean_paired_delta"],
                    low=values["ci95_low"],
                    high=values["ci95_high"],
                    wins=values["asset_win_rate"],
                    prob=values["bootstrap_probability_positive"],
                )
            )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
