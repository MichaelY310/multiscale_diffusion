#!/usr/bin/env python3
"""Audit RDD delta transport on an actual dense DDPM generation trajectory.

This differs from ``probe_rdd_delta_transport_timegrid.py``: the reference
features are produced by the model's own dense reverse trajectory rather than
by independently noising ImageNet validation latents.  The model output is
never changed; this script only measures whether a cross-scale transport rule
predicts a later fine-stage block residual better than a matched stale
full-resolution residual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    load_rdd,
    low_high_tokens,
    manual_forward,
    tensor_metrics,
    upsample_tokens,
)
from sample_rdd_delta_transport_preview import sample_once  # noqa: E402
from sample_rdd_stagewide_block_reuse_preview import make_diffusions  # noqa: E402
from sample_rdd_time_conditioned_reuse_preview import calibrate_time_grid  # noqa: E402


def interpolate(
    coefficients: dict[int, list[dict[str, torch.Tensor]]],
    coarse: torch.Tensor,
    layer: int,
    timestep: int,
) -> torch.Tensor:
    knots = sorted(coefficients)
    if timestep <= knots[0]:
        left = right = knots[0]
        weight = 0.0
    elif timestep >= knots[-1]:
        left = right = knots[-1]
        weight = 0.0
    else:
        right_index = next(i for i, knot in enumerate(knots) if knot > timestep)
        left, right = knots[right_index - 1], knots[right_index]
        weight = float(timestep - left) / float(right - left)
    left_row = coefficients[left][layer]
    right_row = coefficients[right][layer]
    scale = left_row["low_scale"].lerp(right_row["low_scale"], weight).to(coarse.dtype)
    bias = left_row["low_bias"].lerp(right_row["low_bias"], weight).to(coarse.dtype)
    return coarse * scale.view(1, 1, -1) + bias.view(1, 1, -1)


class DenseTrajectoryAudit:
    def __init__(self, model, coefficients, gaps: list[int]):
        self.model = model
        self.coefficients = coefficients
        self.gaps = gaps
        self.max_gap = max(gaps)
        self.coarse: dict[int, torch.Tensor] = {}
        self.history: list[dict] = []
        self.rows: list[dict] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        timestep = int(t[0].item())
        if resolution == 16 and timestep == 300:
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture=True
            )
            self.coarse = {
                layer: upsample_tokens(trace[layer]["residual"], 16, "nearest").detach()
                for layer in range(len(self.model.blocks))
            }
            return output

        if resolution != 32:
            output, _ = manual_forward(self.model, x, t, y, multires=True)
            return output

        output, trace = manual_forward(
            self.model, x, t, y, multires=True, capture=True
        )
        current = {
            layer: trace[layer]["residual"].detach()
            for layer in range(len(self.model.blocks))
        }
        current_coarse = {
            layer: interpolate(self.coefficients, self.coarse[layer], layer, timestep)
            for layer in range(len(self.model.blocks))
        }
        for gap in self.gaps:
            if len(self.history) < gap:
                continue
            previous = self.history[-gap]
            for layer in range(len(self.model.blocks)):
                exact = current[layer]
                generic = previous["residuals"][layer]
                transport = generic + current_coarse[layer] - previous["coarse"][layer]
                exact_low, exact_high = low_high_tokens(exact)
                generic_low, generic_high = low_high_tokens(generic)
                transport_low, transport_high = low_high_tokens(transport)
                generic_full = tensor_metrics(generic, exact)
                transport_full = tensor_metrics(transport, exact)
                generic_low_m = tensor_metrics(generic_low, exact_low)
                transport_low_m = tensor_metrics(transport_low, exact_low)
                generic_high_m = tensor_metrics(generic_high, exact_high)
                transport_high_m = tensor_metrics(transport_high, exact_high)
                self.rows.append({
                    "target_t": timestep,
                    "previous_t": previous["timestep"],
                    "gap_nfe": gap,
                    "layer": layer,
                    "generic": generic_full,
                    "transport": transport_full,
                    "generic_low": generic_low_m,
                    "transport_low": transport_low_m,
                    "generic_high": generic_high_m,
                    "transport_high": transport_high_m,
                })
        self.history.append({
            "timestep": timestep,
            "residuals": current,
            "coarse": current_coarse,
        })
        if len(self.history) > self.max_gap:
            self.history.pop(0)
        return output


def improvement(row: dict, band: str) -> float:
    suffix = "" if band == "full" else f"_{band}"
    return row[f"generic{suffix}"]["relative_l2"] - row[f"transport{suffix}"]["relative_l2"]


def summarize(rows: list[dict], gaps: list[int]) -> dict:
    result = {}
    for gap in gaps:
        selected = [row for row in rows if row["gap_nfe"] == gap]
        layer_rows = []
        for layer in range(28):
            group = [row for row in selected if row["layer"] == layer]
            layer_rows.append({
                "layer": layer,
                **{
                    f"mean_{band}_relative_l2_improvement": sum(improvement(row, band) for row in group) / len(group)
                    for band in ("full", "low", "high")
                },
                "transport_wins_full_fraction": sum(improvement(row, "full") > 0 for row in group) / len(group),
            })
        result[str(gap)] = {
            "comparisons": len(selected),
            **{
                f"mean_{band}_relative_l2_improvement": sum(improvement(row, band) for row in selected) / len(selected)
                for band in ("full", "low", "high")
            },
            "transport_wins_full_fraction": sum(improvement(row, "full") > 0 for row in selected) / len(selected),
            "best_layers": sorted(
                layer_rows,
                key=lambda row: row["mean_full_relative_l2_improvement"],
                reverse=True,
            )[:10],
            "worst_layers": sorted(
                layer_rows,
                key=lambda row: row["mean_full_relative_l2_improvement"],
            )[:5],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", default="80,80,90")
    parser.add_argument("--calibration-seeds", default="101,202")
    parser.add_argument("--calibration-batch", type=int, default=8)
    parser.add_argument("--knots", default="0,75,150,225,299")
    parser.add_argument("--gaps", default="1,2,3,4,5,8,16")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = load_rdd(Path(args.rdd_ckpt), device, torch.float32)
    knots = [int(value) for value in args.knots.split(",")]
    coefficients = calibrate_time_grid(
        model,
        Path(args.imagenet_train),
        Path(args.vae_path),
        device,
        [int(value) for value in args.calibration_seeds.split(",")],
        args.calibration_batch,
        knots,
    )
    gaps = [int(value) for value in args.gaps.split(",")]
    wrapper = DenseTrajectoryAudit(model, coefficients, gaps)
    steps = [int(value) for value in args.num_steps.split(",")]
    _, elapsed = sample_once(
        wrapper, make_diffusions(steps), args.batch_size, steps, device, args.seed
    )
    report = {
        "checkpoint": args.rdd_ckpt,
        "definition": "dense generated trajectory; generic=stale exact fine residual; transport=generic + current coarse prediction - stale coarse prediction",
        "sampler": "DDPM",
        "num_steps_coarse_middle_fine": steps,
        "upsample": "noise_filling",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "knots": knots,
        "gaps_nfe": gaps,
        "dense_sample_seconds": elapsed,
        "summary": summarize(wrapper.rows, gaps),
        "rows": wrapper.rows,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
