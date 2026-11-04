#!/usr/bin/env python3
"""Leakage-free analysis of the CSBR multi-scale residual predictor probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EPS = 1.0e-12


def clamp_mse(value: float) -> float:
    return max(float(value), 0.0)


def prediction_metrics(
    mm: float,
    dd: float,
    md: float,
    my: float,
    dy: float,
    yy: float,
    a: float,
    b: float,
) -> dict[str, float]:
    power = a * a * mm + b * b * dd + 2.0 * a * b * md
    dot = a * my + b * dy
    mse = clamp_mse(power - 2.0 * dot + yy)
    return {
        "cosine": dot / math.sqrt(max(power * yy, EPS)),
        "relative_rmse": math.sqrt(mse / max(yy, EPS)),
        "mse": mse,
        "prediction_power": power,
        "target_power": yy,
        "dot_target": dot,
    }


def high_metrics(hh: float, hy: float, yy: float, beta: float) -> dict[str, float]:
    power = beta * beta * hh
    dot = beta * hy
    mse = clamp_mse(power - 2.0 * dot + yy)
    return {
        "cosine": dot / math.sqrt(max(power * yy, EPS)),
        "relative_rmse": math.sqrt(mse / max(yy, EPS)),
        "mse": mse,
        "prediction_power": power,
        "target_power": yy,
        "dot_target": dot,
    }


def combine_bands(low: dict[str, float], high: dict[str, float]) -> dict[str, float]:
    power = low["prediction_power"] + high["prediction_power"]
    target = low["target_power"] + high["target_power"]
    dot = low["dot_target"] + high["dot_target"]
    mse = low["mse"] + high["mse"]
    return {
        "cosine": dot / math.sqrt(max(power * target, EPS)),
        "relative_rmse": math.sqrt(mse / max(target, EPS)),
        "mse": mse,
        "prediction_power": power,
        "target_power": target,
        "dot_target": dot,
    }


def add_stats(rows: list[dict[str, float]], names: tuple[str, ...]) -> dict[str, float]:
    return {name: sum(float(row[name]) for row in rows) for name in names}


def fit_scalar(stats: dict[str, float]) -> float:
    return stats["my"] / max(stats["mm"], EPS)


def fit_constrained_gamma(stats: dict[str, float]) -> float:
    return (stats["dy"] - stats["md"]) / max(stats["dd"], EPS)


def fit_two_term(stats: dict[str, float]) -> tuple[float, float]:
    ridge = 1.0e-6 * max(stats["mm"] + stats["dd"], EPS)
    mm = stats["mm"] + ridge
    dd = stats["dd"] + ridge
    md = stats["md"]
    det = mm * dd - md * md
    if abs(det) < EPS:
        return fit_scalar(stats), 0.0
    a = (stats["my"] * dd - stats["dy"] * md) / det
    b = (stats["dy"] * mm - stats["my"] * md) / det
    return a, b


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / max(len(rows), 1)


def load_cases(root: Path, run_name: str) -> list[dict[int, dict[str, Any]]]:
    cases: list[dict[int, dict[str, Any]]] = []
    for path in sorted(root.glob("case_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected: dict[int, dict[str, Any]] = {}
        for row in payload["residual_rows"]:
            if (
                row["run"] == run_name
                and row["boundary"] == "stage1_to_stage2"
                and row.get("multiscale_predictor") is not None
            ):
                selected[int(row["layer"])] = row["multiscale_predictor"]
        if not selected:
            raise RuntimeError(f"no predictor rows in {path} for {run_name}")
        cases.append(selected)
    if len(cases) < 2:
        raise RuntimeError(f"need at least two completed cases under {root}")
    return cases


def cross_validate(cases: list[dict[int, dict[str, Any]]]) -> list[dict[str, float]]:
    layer_ids = sorted(set.intersection(*(set(case) for case in cases)))
    output: list[dict[str, float]] = []
    low_names = ("mm", "dd", "md", "my", "dy", "yy")
    high_names = ("hh", "hy", "yy")
    for heldout, case in enumerate(cases):
        training = [item for index, item in enumerate(cases) if index != heldout]
        for layer in layer_ids:
            train_low = add_stats(
                [item[layer]["low_sufficient_statistics"] for item in training],
                low_names,
            )
            test_low = {
                key: float(value)
                for key, value in case[layer]["low_sufficient_statistics"].items()
            }
            train_high = add_stats(
                [item[layer]["high_sufficient_statistics"] for item in training],
                high_names,
            )
            test_high = {
                key: float(value)
                for key, value in case[layer]["high_sufficient_statistics"].items()
            }
            scalar = fit_scalar(train_low)
            gamma = fit_constrained_gamma(train_low)
            a, b = fit_two_term(train_low)
            beta = train_high["hy"] / max(train_high["hh"], EPS)
            zero_high = high_metrics(
                test_high["hh"], test_high["hy"], test_high["yy"], 0.0
            )
            fitted_high = high_metrics(
                test_high["hh"], test_high["hy"], test_high["yy"], beta
            )
            methods = {
                "raw_middle": (1.0, 0.0, zero_high),
                "scalar_middle": (scalar, 0.0, zero_high),
                "middle_plus_trend": (1.0, gamma, zero_high),
                "fitted_middle_trend": (a, b, zero_high),
                "fitted_middle_trend_high": (a, b, fitted_high),
            }
            for method, (coef_a, coef_b, high) in methods.items():
                low = prediction_metrics(**test_low, a=coef_a, b=coef_b)
                total = combine_bands(low, high)
                output.append(
                    {
                        "heldout": heldout,
                        "layer": layer,
                        "method": method,
                        "a": coef_a,
                        "b": coef_b,
                        "beta": beta if method.endswith("_high") else 0.0,
                        "low_cosine": low["cosine"],
                        "low_relative_rmse": low["relative_rmse"],
                        "full_cosine": total["cosine"],
                        "full_relative_rmse": total["relative_rmse"],
                    }
                )
    return output


def fit_all(cases: list[dict[int, dict[str, Any]]]) -> list[dict[str, float]]:
    layer_ids = sorted(set.intersection(*(set(case) for case in cases)))
    low_names = ("mm", "dd", "md", "my", "dy", "yy")
    high_names = ("hh", "hy", "yy")
    output = []
    for layer in layer_ids:
        low = add_stats(
            [case[layer]["low_sufficient_statistics"] for case in cases], low_names
        )
        high = add_stats(
            [case[layer]["high_sufficient_statistics"] for case in cases], high_names
        )
        a, b = fit_two_term(low)
        output.append(
            {
                "layer": layer,
                "scalar_middle": fit_scalar(low),
                "constrained_gamma": fit_constrained_gamma(low),
                "a": a,
                "b": b,
                "beta": high["hy"] / max(high["hh"], EPS),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", default="rdd_nf_base")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_root
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.input_root, args.run)
    rows = cross_validate(cases)
    fitted = fit_all(cases)

    method_summary: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        method_summary[method] = {
            "low_cosine": mean_metric(subset, "low_cosine"),
            "low_relative_rmse": mean_metric(subset, "low_relative_rmse"),
            "full_cosine": mean_metric(subset, "full_cosine"),
            "full_relative_rmse": mean_metric(subset, "full_relative_rmse"),
        }

    baseline_by_key = {
        (row["heldout"], row["layer"]): row
        for row in rows
        if row["method"] == "scalar_middle"
    }
    per_layer: list[dict[str, float]] = []
    for layer in sorted({int(row["layer"]) for row in rows}):
        trend = [
            row
            for row in rows
            if row["layer"] == layer and row["method"] == "fitted_middle_trend"
        ]
        trend_high = [
            row
            for row in rows
            if row["layer"] == layer
            and row["method"] == "fitted_middle_trend_high"
        ]
        baseline = [
            baseline_by_key[(row["heldout"], layer)]
            for row in trend
        ]
        per_layer.append(
            {
                "layer": layer,
                "scalar_low_rrmse": mean_metric(baseline, "low_relative_rmse"),
                "trend_low_rrmse": mean_metric(trend, "low_relative_rmse"),
                "low_rrmse_gain": (
                    mean_metric(baseline, "low_relative_rmse")
                    - mean_metric(trend, "low_relative_rmse")
                ),
                "trend_full_rrmse": mean_metric(trend, "full_relative_rmse"),
                "trend_high_full_rrmse": mean_metric(
                    trend_high, "full_relative_rmse"
                ),
                "highpass_full_rrmse_gain": (
                    mean_metric(trend, "full_relative_rmse")
                    - mean_metric(trend_high, "full_relative_rmse")
                ),
            }
        )

    payload = {
        "protocol": {
            "run": args.run,
            "cases": len(cases),
            "validation": "leave-one-case-out; coefficients fit on other cases",
            "low_predictors": {
                "raw_middle": "M",
                "scalar_middle": "a*M",
                "middle_plus_trend": "M+gamma*(M-C)",
                "fitted_middle_trend": "a*M+b*(M-C)",
                "fitted_middle_trend_high": "a*M+b*(M-C)+beta*highpass(H_fine)",
            },
        },
        "method_summary": method_summary,
        "per_layer": per_layer,
        "fit_all_coefficients": fitted,
    }
    (output_dir / f"{args.run}_predictor_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{args.run}_cross_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["method_summary"], indent=2))
    print("best layers by trend low-RRMSE gain:")
    for row in sorted(per_layer, key=lambda item: item["low_rrmse_gain"], reverse=True)[:10]:
        print(row)


if __name__ == "__main__":
    main()
