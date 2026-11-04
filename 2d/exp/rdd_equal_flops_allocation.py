#!/usr/bin/env python3
"""Generate integer RDD sampling allocations under an equal-FLOPs budget.

Conventions match ``sample_fid_unified.py``:
  * resolutions / stage_lengths / architectures: finest -> coarsest
  * baseline_steps and emitted sampling_steps: coarsest -> finest

The FLOPs model deliberately counts actual model forwards.  DDPM/DPM1 uses
``n`` forwards per stage.  In this repository DPM2 has an order-1 first update,
two forwards for each interior update, and a one-forward final ``t=0 -> 0``
trivial update: ``max(1, 2*n-2)`` forwards.
CFG > 1 doubles the effective batch and therefore doubles these FLOPs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


ARCHITECTURES = {
    "DiT-XL/2": {"hidden": 1152, "depth": 28, "patch": 2},
    "DiT-L/2": {"hidden": 1024, "depth": 24, "patch": 2},
    "DiT-B/2": {"hidden": 768, "depth": 12, "patch": 2},
}


def dit_forward_flops(architecture: str, resolution: int) -> tuple[float, int]:
    """Count dominant dense ops for the repository's DiT forward.

    One multiply-add is reported as two FLOPs.  The count includes patch/time
    embedding, QKV/projection, attention matmuls, MLP, per-block AdaLN and the
    final AdaLN/output projection.  Elementwise activations, normalization,
    positional addition and class embedding lookup are intentionally omitted.
    """
    spec = ARCHITECTURES[architecture]
    hidden, depth, patch = spec["hidden"], spec["depth"], spec["patch"]
    tokens = (resolution // patch) ** 2
    in_channels, out_channels = 4, 8
    patch_macs = tokens * (patch**2) * in_channels * hidden
    time_embed_macs = 256 * hidden + hidden * hidden
    # Per block: QKV+out = 4Nd²; attention = 2N²d; MLP = 8Nd²;
    # AdaLN modulation = 6d².
    block_macs = depth * (
        12 * tokens * hidden * hidden
        + 2 * tokens * tokens * hidden
        + 6 * hidden * hidden
    )
    final_macs = 2 * hidden * hidden + tokens * hidden * (patch**2) * out_channels
    return 2.0 * (patch_macs + time_embed_macs + block_macs + final_macs), tokens


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def model_calls(steps: float, solver: str) -> float:
    if solver in ("ddpm", "dpm1"):
        return steps
    if solver == "dpm2":
        return max(1.0, 2.0 * steps - 2.0)
    raise ValueError(f"unsupported solver: {solver}")


def total_flops(
    steps: list[float], costs: list[float], solver: str, cfg_multiplier: int
) -> float:
    return cfg_multiplier * sum(
        cost * model_calls(step, solver) for step, cost in zip(steps, costs)
    )


def grid_values(minimum: int, maximum: int, grid: int) -> list[int]:
    first = max(minimum, math.ceil(minimum / grid) * grid)
    values = list(range(first, maximum + 1, grid))
    if maximum >= minimum and maximum not in values:
        values.append(maximum)
    return sorted(set(values))


def inverse_last_stage_calls(remaining: float, cost: float, solver: str) -> float:
    if solver in ("ddpm", "dpm1"):
        return remaining / cost
    return 0.5 * (remaining / cost + 2.0)


def enumerate_near_budget(
    maxima: list[int],
    costs: list[float],
    solver: str,
    cfg_multiplier: int,
    budget: float,
    minimum: int,
    grid: int,
    tolerance: float,
) -> tuple[list[tuple[int, ...]], float]:
    """Enumerate the first K-1 stages and solve the costly final stage."""
    value_lists = [grid_values(minimum, maximum, grid) for maximum in maxima]
    if any(not values for values in value_lists):
        raise ValueError(f"minimum={minimum} exceeds at least one stage limit {maxima}")

    candidates: dict[tuple[int, ...], float] = {}
    prefix_lists = value_lists[:-1]
    final_values = set(value_lists[-1])
    for prefix in itertools.product(*prefix_lists) if prefix_lists else [()]:
        prefix_cost = total_flops(
            list(prefix), costs[: len(prefix)], solver, cfg_multiplier
        )
        remaining_without_cfg = (budget - prefix_cost) / cfg_multiplier
        ideal = inverse_last_stage_calls(remaining_without_cfg, costs[-1], solver)
        rounded = round(ideal / grid) * grid
        probes = {
            rounded + offset * grid for offset in range(-3, 4)
        } | {minimum, maxima[-1]}
        for final in probes:
            if final not in final_values:
                continue
            allocation = tuple(prefix) + (int(final),)
            relative_error = abs(
                total_flops(list(allocation), costs, solver, cfg_multiplier) - budget
            ) / budget
            previous = candidates.get(allocation)
            if previous is None or relative_error < previous:
                candidates[allocation] = relative_error

    if not candidates:
        raise RuntimeError("no legal allocations were generated")
    closest_error = min(candidates.values())
    accepted_tolerance = max(tolerance, closest_error + 1e-12)
    accepted = [
        allocation
        for allocation, relative_error in candidates.items()
        if relative_error <= accepted_tolerance
    ]
    return accepted, accepted_tolerance


def bounded_continuous_target(
    weights: list[float],
    maxima: list[int],
    costs: list[float],
    solver: str,
    cfg_multiplier: int,
    budget: float,
    minimum: int,
) -> list[float]:
    """Scale positive weights to the budget while respecting stage limits."""
    if any(weight <= 0 for weight in weights):
        raise ValueError("all difficulty-derived weights must be positive")

    def allocation(scale: float) -> list[float]:
        return [
            min(float(maximum), max(float(minimum), scale * weight))
            for weight, maximum in zip(weights, maxima)
        ]

    minimum_cost = total_flops(
        [float(minimum)] * len(weights), costs, solver, cfg_multiplier
    )
    maximum_cost = total_flops(
        [float(value) for value in maxima], costs, solver, cfg_multiplier
    )
    if not minimum_cost <= budget <= maximum_cost:
        raise ValueError(
            f"budget {budget/1e12:.6f}T is outside legal range "
            f"[{minimum_cost/1e12:.6f}T, {maximum_cost/1e12:.6f}T]"
        )

    low, high = 0.0, 1.0
    while total_flops(allocation(high), costs, solver, cfg_multiplier) < budget:
        high *= 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if total_flops(allocation(middle), costs, solver, cfg_multiplier) < budget:
            low = middle
        else:
            high = middle
    return allocation(0.5 * (low + high))


def nearest_candidate(
    candidates: list[tuple[int, ...]], target: list[float], used: set[tuple[int, ...]]
) -> tuple[int, ...] | None:
    pool = [candidate for candidate in candidates if candidate not in used]
    if not pool:
        return None

    def distance(candidate: tuple[int, ...]) -> float:
        return sum(
            ((actual - desired) / max(5.0, desired)) ** 2
            for actual, desired in zip(candidate, target)
        )

    return min(pool, key=distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument(
        "--architectures", default="DiT-XL/2,DiT-L/2,DiT-B/2"
    )
    parser.add_argument(
        "--baseline-steps", default="80,80,90",
        help="Reference allocation in coarse-to-fine sampling order.",
    )
    parser.add_argument(
        "--difficulty", default=None,
        help=(
            "Positive local-error densities in coarse-to-fine order. The default "
            "1,1,... is a zero-profile prior; replace it with measured values."
        ),
    )
    parser.add_argument("--solver", choices=["ddpm", "dpm1", "dpm2"], default="ddpm")
    parser.add_argument(
        "--budget-solver", choices=["ddpm", "dpm1", "dpm2"], default="ddpm",
        help="Solver used to turn --baseline-steps into the reference budget.",
    )
    parser.add_argument("--budget-tflops", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--solver-order", type=int, choices=[1, 2], default=None)
    parser.add_argument("--minimum-steps", type=int, default=1)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--max-candidates", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resolutions = csv_ints(args.resolutions)
    lengths_finest = csv_ints(args.stage_lengths)
    architectures_finest = csv_strings(args.architectures)
    baseline = csv_ints(args.baseline_steps)
    count = len(resolutions)
    if not (
        count == len(lengths_finest) == len(architectures_finest) == len(baseline)
    ):
        raise ValueError("resolutions, lengths, architectures and baseline must align")
    for finer, coarser in zip(resolutions, resolutions[1:]):
        if finer <= coarser or finer % coarser:
            raise ValueError("resolutions must be finest-to-coarsest integer multiples")
    if any(step <= 0 for step in baseline):
        raise ValueError("baseline steps must be positive")

    cost_token_pairs = [
        dit_forward_flops(architecture, resolution)
        for architecture, resolution in zip(architectures_finest, resolutions)
    ]
    costs_finest = [pair[0] for pair in cost_token_pairs]
    tokens_finest = [pair[1] for pair in cost_token_pairs]
    sampling_resolutions = list(reversed(resolutions))
    maxima = list(reversed(lengths_finest))
    costs = list(reversed(costs_finest))
    architectures = list(reversed(architectures_finest))
    if any(step > maximum for step, maximum in zip(baseline, maxima)):
        raise ValueError(f"baseline {baseline} exceeds stage limits {maxima}")

    cfg_multiplier = 2 if args.cfg_scale > 1.0 else 1
    if args.budget_tflops is None:
        budget = total_flops(
            [float(value) for value in baseline], costs,
            args.budget_solver, cfg_multiplier,
        )
    else:
        budget = args.budget_tflops * 1e12

    difficulty = (
        csv_floats(args.difficulty) if args.difficulty else [1.0] * count
    )
    if len(difficulty) != count or any(value <= 0 for value in difficulty):
        raise ValueError("difficulty must contain one positive value per stage")
    order = args.solver_order or (2 if args.solver == "dpm2" else 1)

    candidates, used_tolerance = enumerate_near_budget(
        maxima=maxima,
        costs=costs,
        solver=args.solver,
        cfg_multiplier=cfg_multiplier,
        budget=budget,
        minimum=args.minimum_steps,
        grid=args.grid,
        tolerance=args.tolerance,
    )

    def target_for(profile: list[float]) -> list[float]:
        weights = [
            length * (error / cost) ** (1.0 / (order + 1.0))
            for length, error, cost in zip(maxima, profile, costs)
        ]
        return bounded_continuous_target(
            weights, maxima, costs, args.solver, cfg_multiplier,
            budget, args.minimum_steps,
        )

    targets: list[tuple[str, list[float]]] = []
    targets.append(("predicted", target_for(difficulty)))
    targets.append(("reference_transformed", [float(value) for value in baseline]))
    equal_flops_weights = [1.0 / cost for cost in costs]
    targets.append(("equal_stage_flops", bounded_continuous_target(
        equal_flops_weights, maxima, costs, args.solver, cfg_multiplier,
        budget, args.minimum_steps,
    )))
    for stage_index, resolution in enumerate(sampling_resolutions):
        if count == 2 or args.max_candidates >= 6:
            profile = list(difficulty)
            # A large, explicit perturbation makes these useful stress tests
            # even when a cheap stage otherwise saturates at its interval cap.
            profile[stage_index] *= 64.0
            targets.append((f"heavy_{resolution}x{resolution}", target_for(profile)))

    selected: list[dict[str, object]] = []
    used: set[tuple[int, ...]] = set()
    for label, target in targets:
        if len(selected) >= args.max_candidates:
            break
        candidate = nearest_candidate(candidates, target, used)
        if candidate is None:
            continue
        used.add(candidate)
        selected.append({"label": label, "allocation": candidate, "target": target})

    # Two nearest unused integer neighbours make the local sensitivity check
    # explicit instead of relying only on hand-designed heavy-stage controls.
    predicted_target = targets[0][1]
    while len(selected) < args.max_candidates:
        candidate = nearest_candidate(candidates, predicted_target, used)
        if candidate is None:
            break
        used.add(candidate)
        selected.append({
            "label": f"predicted_neighbor_{len(selected) + 1}",
            "allocation": candidate,
            "target": predicted_target,
        })

    records = []
    for item in selected:
        allocation = list(item["allocation"])
        flops = total_flops(allocation, costs, args.solver, cfg_multiplier)
        records.append({
            "label": item["label"],
            "sampling_steps": allocation,
            "num_steps": ",".join(str(value) for value in allocation),
            "model_calls": [model_calls(value, args.solver) for value in allocation],
            "tflops_per_image": flops / 1e12,
            "relative_budget_error": (flops - budget) / budget,
            "continuous_target": [round(value, 4) for value in item["target"]],
        })

    payload = {
        "conventions": {
            "resolutions_and_stage_lengths": "finest_to_coarsest",
            "sampling_steps": "coarsest_to_finest",
        },
        "resolutions": resolutions,
        "sampling_resolutions": sampling_resolutions,
        "stage_lengths": lengths_finest,
        "stage_limits_sampling_order": maxima,
        "architectures_sampling_order": architectures,
        "flops_counting": (
            "architecture-aware dense DiT count; multiply-add=2 FLOPs; excludes "
            "elementwise ops, normalization, embedding lookup and VAE"
        ),
        "tokens_sampling_order": list(reversed(tokens_finest)),
        "forward_flops_sampling_order": costs,
        "forward_cost_ratio_to_coarsest": [cost / costs[0] for cost in costs],
        "solver": args.solver,
        "solver_order": order,
        "cfg_scale": args.cfg_scale,
        "cfg_multiplier": cfg_multiplier,
        "budget_tflops_per_image": budget / 1e12,
        "budget_source": (
            f"explicit:{args.budget_tflops}" if args.budget_tflops is not None
            else f"{args.budget_solver}:{args.baseline_steps}"
        ),
        "difficulty_sampling_order": difficulty,
        "integer_grid": args.grid,
        "requested_tolerance": args.tolerance,
        "used_tolerance": used_tolerance,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Budget: {budget/1e12:.6f} TFLOPs/image (CFG multiplier {cfg_multiplier})")
    print("Sampling order:", " -> ".join(f"{r}x{r}" for r in sampling_resolutions))
    print("Per-forward cost ratio:", ", ".join(f"{v:.2f}" for v in payload["forward_cost_ratio_to_coarsest"]))
    print(f"Accepted relative budget tolerance: {used_tolerance:.4%}")
    print("label\tsteps\tTFLOPs/image\tdelta")
    for record in records:
        print(
            f"{record['label']}\t{record['num_steps']}\t"
            f"{record['tflops_per_image']:.6f}\t"
            f"{record['relative_budget_error']:+.4%}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
