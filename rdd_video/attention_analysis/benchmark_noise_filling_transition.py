"""Validate and time native vs fused RDD noise-filling stage boundaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from rdd_noise_filling_triton import fused_noise_filling_transition


def upsample_native(value: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    import math

    ratios = [math.ceil(target[index] / value.shape[index + 2]) for index in range(3)]
    value = value.repeat_interleave(ratios[0], 2)
    value = value.repeat_interleave(ratios[1], 3).repeat_interleave(ratios[2], 4)
    return value[:, :, : target[0], : target[1], : target[2]]


def native(
    clean: torch.Tensor,
    target: tuple[int, int, int],
    sigma: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    projected = upsample_native(clean, target)
    noise = torch.randn(
        projected.shape, generator=generator, device=clean.device, dtype=clean.dtype
    )
    return (1.0 - sigma) * projected + sigma * noise


def torch_nearest(
    clean: torch.Tensor,
    target: tuple[int, int, int],
    sigma: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    projected = F.interpolate(clean, size=target, mode="nearest")
    noise = torch.randn(
        projected.shape, generator=generator, device=clean.device, dtype=clean.dtype
    )
    return (1.0 - sigma) * projected + sigma * noise


def bench(function, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split("x"))
    if len(values) != 3:
        raise ValueError(f"expected TxHxW, got {text}")
    return values


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sigma = torch.tensor(args.sigma, device=device, dtype=torch.bfloat16)
    cases = []
    for specification in args.case:
        source_text, target_text = specification.split("->")
        source = parse_shape(source_text)
        target = parse_shape(target_text)
        # Match sample_three_call_dmd: x0 is produced in BTCHW and permuted to
        # BCTHW, so the transition must honor non-contiguous source strides.
        clean = torch.randn(
            (args.batch, source[0], args.channels, source[1], source[2]),
            device=device,
            dtype=torch.bfloat16,
        ).permute(0, 2, 1, 3, 4)
        native_generator = torch.Generator(device=device).manual_seed(args.seed)
        fused_generator = torch.Generator(device=device).manual_seed(args.seed)
        expected = native(clean, target, sigma, native_generator)
        actual = fused_noise_filling_transition(clean, target, sigma, fused_generator)
        difference = (expected.float() - actual.float()).abs()
        nearest_generator = torch.Generator(device=device).manual_seed(args.seed)
        nearest_actual = torch_nearest(
            clean, target, sigma, nearest_generator
        )
        nearest_difference = (expected.float() - nearest_actual.float()).abs()

        native_bench_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        fused_bench_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        nearest_bench_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        native_values = bench(
            lambda: native(clean, target, sigma, native_bench_generator),
            args.warmup,
            args.repeats,
        )
        fused_values = bench(
            lambda: fused_noise_filling_transition(
                clean, target, sigma, fused_bench_generator
            ),
            args.warmup,
            args.repeats,
        )
        nearest_values = bench(
            lambda: torch_nearest(
                clean, target, sigma, nearest_bench_generator
            ),
            args.warmup,
            args.repeats,
        )
        native_median = statistics.median(native_values)
        fused_median = statistics.median(fused_values)
        nearest_median = statistics.median(nearest_values)
        cases.append(
            {
                "source": source,
                "target": target,
                "elements": actual.numel(),
                "native_ms_median": native_median,
                "fused_ms_median": fused_median,
                "speedup": native_median / fused_median,
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "rmse": float(difference.square().mean().sqrt().item()),
                "torch_nearest_ms_median": nearest_median,
                "torch_nearest_speedup": native_median / nearest_median,
                "torch_nearest_max_abs": float(nearest_difference.max().item()),
                "torch_nearest_mean_abs": float(nearest_difference.mean().item()),
                "torch_nearest_rmse": float(
                    nearest_difference.square().mean().sqrt().item()
                ),
            }
        )
    payload = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sigma": args.sigma,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="TxHxW->TxHxW; may be repeated",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args()
    if not parsed.case:
        parsed.case = [
            "8x14x26->8x28x52",
            "8x28x52->8x56x104",
            "2x14x26->4x28x52",
            "4x28x52->8x56x104",
        ]
    main(parsed)
