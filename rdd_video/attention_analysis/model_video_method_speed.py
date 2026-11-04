#!/usr/bin/env python3
"""Analytic Wan2.1-1.3B speed model for RDD/video acceleration methods.

This intentionally needs no checkpoint and no GPU.  It compares algorithms on
one common Wan2.1-T2V-1.3B backbone and the paper's 81x448x832 output grid.  The
numbers are operation counts, not quality claims and not measured latency.

FLOP convention: one multiply-add is two FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WanConfig:
    dim: int = 1536
    ffn_dim: int = 8960
    heads: int = 12
    layers: int = 30
    text_tokens: int = 512
    head_dim: int = 128
    fine_grid: tuple[int, int, int] = (21, 28, 52)
    vsa_tile: tuple[int, int, int] = (4, 4, 4)


@dataclass(frozen=True)
class Stage:
    name: str
    factor: int
    grid: tuple[int, int, int]
    padded_grid: tuple[int, int, int]
    tokens: int
    padded_tokens: int


@dataclass
class Row:
    regime: str
    rank: int
    method: str
    status: str
    stage_calls: str
    total_calls: int
    retained_attention: str
    relative_flops: float
    ideal_speedup: float
    probability_cache_gib: float
    notes: str


def prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def stage_for_factor(cfg: WanConfig, name: str, factor: int) -> Stage:
    grid = tuple(ceil_div(x, factor) for x in cfg.fine_grid)
    padded_grid = tuple(
        ceil_div(x, tile) * tile for x, tile in zip(grid, cfg.vsa_tile)
    )
    return Stage(name, factor, grid, padded_grid, prod(grid), prod(padded_grid))


def stages(cfg: WanConfig) -> dict[str, Stage]:
    return {
        "coarse": stage_for_factor(cfg, "coarse", 4),
        "middle": stage_for_factor(cfg, "middle", 2),
        "fine": stage_for_factor(cfg, "fine", 1),
    }


def common_flops(cfg: WanConfig, n: int) -> float:
    """Everything in one Wan block except self-attention's QKV/O and QK/AV.

    Includes the FFN and a conservative T2V cross-attention model.  Text K/V
    projection is included per block/call because the reference Wan forward
    does not expose a cross-step KV cache.
    """
    d, f, l = cfg.dim, cfg.ffn_dim, cfg.text_tokens
    ffn = 4.0 * n * d * f
    cross_video_qo = 4.0 * n * d * d
    cross_text_kv = 4.0 * l * d * d
    cross_qk_av = 4.0 * n * l * d
    return ffn + cross_video_qo + cross_text_kv + cross_qk_av


def self_projection_flops(cfg: WanConfig, n: int) -> float:
    # Q, K, V, and output projections.
    return 8.0 * n * cfg.dim * cfg.dim


def self_pair_flops(cfg: WanConfig, n_attention: int, density: float = 1.0) -> float:
    # QK^T and A@V.
    return 4.0 * density * n_attention * n_attention * cfg.dim


def vsa_router_flops(cfg: WanConfig, stage: Stage) -> float:
    """Optimistic VSA coarse-branch arithmetic (metadata/top-k excluded)."""
    blocks = stage.padded_tokens // prod(cfg.vsa_tile)
    compress_qkv = 3.0 * stage.padded_tokens * cfg.dim
    coarse_qk_av = 4.0 * blocks * blocks * cfg.dim
    coarse_broadcast = stage.padded_tokens * cfg.dim
    return compress_qkv + coarse_qk_av + coarse_broadcast


def block_flops(
    cfg: WanConfig,
    stage: Stage,
    *,
    density: float = 1.0,
    sparse: bool = False,
    vsa_router: bool = False,
    probability_hit: bool = False,
    output_hit: bool = False,
) -> float:
    result = common_flops(cfg, stage.tokens)
    if output_hit:
        return result

    if probability_hit:
        # Current V and output projections remain; old sparse A is multiplied by
        # current V.  Q/K projection and QK/softmax are skipped.
        result += 4.0 * stage.tokens * cfg.dim * cfg.dim
        result += 2.0 * density * stage.padded_tokens**2 * cfg.dim
        return result

    result += self_projection_flops(cfg, stage.tokens)
    attention_tokens = stage.padded_tokens if sparse else stage.tokens
    result += self_pair_flops(cfg, attention_tokens, density)
    if vsa_router:
        result += vsa_router_flops(cfg, stage)
    return result


def probability_cache_gib(cfg: WanConfig, stage: Stage, density: float, bytes_per: float = 2.0) -> float:
    elements = cfg.layers * cfg.heads * stage.padded_tokens**2 * density
    return elements * bytes_per / 1024**3


def method_cost(
    cfg: WanConfig,
    stage_map: dict[str, Stage],
    calls: dict[str, int],
    *,
    density: float = 1.0,
    sparse: bool = False,
    vsa_router: bool = False,
    reuse_every: int = 1,
    reuse_kind: str = "none",
    sparse_stages: tuple[str, ...] | None = None,
) -> float:
    total = 0.0
    for stage_name, count in calls.items():
        stage = stage_map[stage_name]
        stage_sparse = sparse and (sparse_stages is None or stage_name in sparse_stages)
        stage_density = density if stage_sparse else 1.0
        miss = block_flops(
            cfg,
            stage,
            density=stage_density,
            sparse=stage_sparse,
            vsa_router=vsa_router and stage_sparse,
        )
        if reuse_every <= 1 or reuse_kind == "none":
            per_block = count * miss
        else:
            misses = ceil_div(count, reuse_every)
            hits = count - misses
            hit = block_flops(
                cfg,
                stage,
                density=stage_density,
                sparse=stage_sparse,
                probability_hit=reuse_kind == "probability",
                output_hit=reuse_kind == "output",
            )
            per_block = misses * miss + hits * hit
        total += cfg.layers * per_block
    return total


def temporal_only_stages(cfg: WanConfig) -> dict[str, Stage]:
    result = {}
    for name, factor in (("coarse", 4), ("middle", 2), ("fine", 1)):
        t, h, w = cfg.fine_grid
        grid = (ceil_div(t, factor), h, w)
        padded = tuple(ceil_div(x, tile) * tile for x, tile in zip(grid, cfg.vsa_tile))
        result[name] = Stage(name, factor, grid, padded, prod(grid), prod(padded))
    return result


def add_method(
    target: list[dict],
    *,
    regime: str,
    method: str,
    status: str,
    calls: dict[str, int],
    cost: float,
    density: float,
    cache_gib: float = 0.0,
    notes: str,
) -> None:
    target.append(
        {
            "regime": regime,
            "method": method,
            "status": status,
            "calls": calls,
            "cost": cost,
            "density": density,
            "cache_gib": cache_gib,
            "notes": notes,
        }
    )


def make_methods(cfg: WanConfig, density: float, reuse_every: int) -> tuple[list[dict], float]:
    ss = stages(cfg)
    fine_only = {"fine": 50}
    pyramid_50 = {"coarse": 20, "middle": 20, "fine": 10}
    rdd_forward_matched = {"coarse": 19, "middle": 20, "fine": 11}
    few_stage = {"coarse": 2, "middle": 2, "fine": 1}
    methods: list[dict] = []
    baseline = method_cost(cfg, ss, fine_only)

    regime = "matched_50_model_calls"
    add_method(methods, regime=regime, method="Wan2.1 dense full-resolution", status="released baseline",
               calls=fine_only, cost=baseline, density=1.0,
               notes="50 full-resolution Wan calls; common comparison anchor.")
    add_method(methods, regime=regime, method="FastVideo VSA only", status="released method",
               calls=fine_only,
               cost=method_cost(cfg, ss, fine_only, density=density, sparse=True, vsa_router=True),
               density=density,
               notes="No DMD step reduction; current-fine compression/router/coarse output every layer and call.")
    add_method(methods, regime=regime, method="TPDiff-style temporal-only pyramid", status="matched Wan proxy",
               calls=pyramid_50, cost=method_cost(cfg, temporal_only_stages(cfg), pyramid_50), density=1.0,
               notes="Only temporal token count changes. This is a Wan-shape proxy, not the authors' different backbone.")
    add_method(methods, regime=regime, method="PyramidalWan / boundary-only pyramid", status="released path / matched proxy",
               calls=pyramid_50, cost=method_cost(cfg, ss, pyramid_50), density=1.0,
               notes="At fixed backbone and stage grid, drift/transition elementwise work is negligible.")
    add_method(methods, regime=regime, method="RDD dense, strict 50-forward", status="our planned method",
               calls=rdd_forward_matched, cost=method_cost(cfg, ss, rdd_forward_matched), density=1.0,
               notes="18/19/10 scheduled updates plus three endpoint calls, represented as 19/20/11 calls.")
    add_method(methods, regime=regime, method="RDD + previous-stage route", status="our proposed method",
               calls=rdd_forward_matched,
               cost=method_cost(cfg, ss, rdd_forward_matched, density=density, sparse=True,
                                sparse_stages=("middle", "fine")), density=density,
               notes="Coarse stage stays dense; middle/fine use the same FastVideo sparse executor with cached external routes.")
    cache = (probability_cache_gib(cfg, ss["coarse"], 1.0)
             + sum(probability_cache_gib(cfg, ss[s], density) for s in ("middle", "fine")))
    add_method(methods, regime=regime, method=f"RDD + route + probability reuse k={reuse_every}",
               status="research prototype; memory-limited",
               calls=rdd_forward_matched,
               cost=method_cost(cfg, ss, rdd_forward_matched, density=density, sparse=True,
                                reuse_every=reuse_every, reuse_kind="probability",
                                sparse_stages=("middle", "fine")),
               density=density, cache_gib=cache,
               notes="Cache sparse A; hits compute A_old@V_current. Full 30-layer BF16 cache is usually impractical.")
    add_method(methods, regime=regime, method=f"RDD + route + attention-output reuse k={reuse_every}",
               status="deployable proxy; quality unverified",
               calls=rdd_forward_matched,
               cost=method_cost(cfg, ss, rdd_forward_matched, density=density, sparse=True,
                                reuse_every=reuse_every, reuse_kind="output",
                                sparse_stages=("middle", "fine")),
               density=density,
               notes="PAB-style cache hit skips self-attention branch. It is not exact probability reuse.")

    regime = "paper_default_or_hypothetical_few_step"
    add_method(methods, regime=regime, method="FastVideo DMD3 + VSA", status="released method",
               calls={"fine": 3},
               cost=method_cost(cfg, ss, {"fine": 3}, density=density, sparse=True, vsa_router=True),
               density=density,
               notes="Three full-resolution calls; published system combines step distillation and VSA.")
    add_method(methods, regime=regime, method="PyramidalWan DMD 2/2/1", status="released method",
               calls=few_stage, cost=method_cost(cfg, ss, few_stage), density=1.0,
               notes="Five calls, but only one is at full resolution.")
    add_method(methods, regime=regime, method="RDD distilled 2/2/1 dense", status="hypothetical until trained",
               calls=few_stage, cost=method_cost(cfg, ss, few_stage), density=1.0,
               notes="Speed-only projection for a future RDD-distilled student.")
    add_method(methods, regime=regime, method="RDD distilled 2/2/1 + previous-stage route", status="hypothetical until trained",
               calls=few_stage, cost=method_cost(cfg, ss, few_stage, density=density, sparse=True,
                                                 sparse_stages=("middle", "fine")),
               density=density,
               notes="Potential hardware target; no quality claim without sparse/distillation finetuning.")

    # Pyramid Flow's official video inference default is 10/10/10.  Its released
    # backbone and autoregressive workflow differ from Wan, so this row applies
    # only that public schedule to the common Wan token/FLOP model.
    pyramid_flow_calls = {"coarse": 10, "middle": 10, "fine": 10}
    add_method(methods, regime=regime, method="Pyramid Flow 10/10/10 schedule on Wan proxy",
               status="official schedule / matched Wan proxy",
               calls=pyramid_flow_calls, cost=method_cost(cfg, ss, pyramid_flow_calls), density=1.0,
               notes="Not a raw Pyramid Flow runtime claim; backbone, text encoder and autoregressive chunking differ.")

    # Explicit allocation sweep.  Each allocation is interpreted as scheduled
    # denoising updates plus one endpoint evaluation per stage, matching RDD's
    # current sampler accounting.  The strict 18/19/10 schedule therefore has
    # exactly 50 forward calls after endpoints are included.
    allocation_regime = "rdd_step_allocation_sweep"
    allocations = {
        "strict 18/19/10 (+ endpoints)": {"coarse": 19, "middle": 20, "fine": 11},
        "balanced 20/20/10 (+ endpoints)": {"coarse": 21, "middle": 21, "fine": 11},
        "fine-heavy 10/15/25 (+ endpoints)": {"coarse": 11, "middle": 16, "fine": 26},
        "coarse-heavy 30/15/5 (+ endpoints)": {"coarse": 31, "middle": 16, "fine": 6},
    }
    for allocation_name, allocation_calls in allocations.items():
        add_method(methods, regime=allocation_regime,
                   method=f"RDD dense — {allocation_name}", status="speed-only allocation proxy",
                   calls=allocation_calls, cost=method_cost(cfg, ss, allocation_calls), density=1.0,
                   notes="Step allocation changes cost; equal quality is not assumed.")
        add_method(methods, regime=allocation_regime,
                   method=f"RDD + previous-stage route — {allocation_name}", status="speed-only allocation proxy",
                   calls=allocation_calls,
                   cost=method_cost(cfg, ss, allocation_calls, density=density, sparse=True,
                                    sparse_stages=("middle", "fine")), density=density,
                   notes="Uses the same retained-pair ratio at all routed stages; quality requires later checkpoints.")
    return methods, baseline


def render_svg(rows: list[Row], output: Path, regime: str) -> None:
    selected = [r for r in rows if r.regime == regime]
    width = 1120
    row_h = 42
    height = 90 + row_h * len(selected)
    left = 410
    usable = width - left - 150
    max_speed = max(r.ideal_speedup for r in selected)
    colors = {"released": "#2563eb", "our": "#059669", "hypothetical": "#d97706", "proxy": "#7c3aed"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:21px;font-weight:700}.label{font-size:13px}.value{font-size:13px;font-weight:700}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" class="title">Wan2.1-1.3B theoretical speed — {regime}</text>',
        '<text x="24" y="58" class="label">Same 81×448×832 output grid; operation-count upper bounds, not measured latency or quality.</text>',
    ]
    for i, row in enumerate(selected):
        y = 82 + i * row_h
        key = "hypothetical" if "hypothetical" in row.status else "our" if "our" in row.status else "proxy" if "proxy" in row.status else "released"
        bar = usable * row.ideal_speedup / max_speed
        parts.append(f'<text x="24" y="{y + 18}" class="label">{row.rank}. {row.method}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="24" rx="4" fill="{colors[key]}" opacity="0.86"/>')
        parts.append(f'<text x="{left + bar + 8:.1f}" y="{y + 18}" class="value">{row.ideal_speedup:.2f}×</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=0.20, help="Retained fine KV-pair fraction")
    parser.add_argument("--reuse-every", type=int, default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("eval_results/rdd_video_speed_theory"))
    args = parser.parse_args()
    if not 0.0 < args.density <= 1.0:
        raise ValueError("--density must be in (0,1]")
    if args.reuse_every < 1:
        raise ValueError("--reuse-every must be >= 1")

    cfg = WanConfig()
    ss = stages(cfg)
    raw_methods, baseline = make_methods(cfg, args.density, args.reuse_every)
    rows: list[Row] = []
    for regime in sorted({m["regime"] for m in raw_methods}):
        group = sorted((m for m in raw_methods if m["regime"] == regime), key=lambda x: x["cost"])
        for rank, method in enumerate(group, 1):
            rows.append(Row(
                regime=regime,
                rank=rank,
                method=method["method"],
                status=method["status"],
                stage_calls="/".join(str(method["calls"].get(s, 0)) for s in ("coarse", "middle", "fine")),
                total_calls=sum(method["calls"].values()),
                retained_attention=f'{100.0 * method["density"]:.1f}%',
                relative_flops=method["cost"] / baseline,
                ideal_speedup=baseline / method["cost"],
                probability_cache_gib=method["cache_gib"],
                notes=method["notes"],
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "warning": "FLOP upper bounds only; random-tensor kernel measurements and quality-matched checkpoints are separate requirements.",
        "config": asdict(cfg),
        "density": args.density,
        "reuse_every": args.reuse_every,
        "stages": {name: asdict(stage) for name, stage in ss.items()},
        "rows": [asdict(row) for row in rows],
    }
    (args.output_dir / "theory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "theory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    md = [
        "# RDD video methods: theoretical Wan2.1-1.3B speed ranking",
        "",
        "> These are operation-count upper bounds. They do not establish equal quality, and they are not wall-clock measurements.",
        "",
        "## Token grids",
        "",
        "| stage | valid grid | valid tokens | VSA-padded grid | padded tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ss.values():
        md.append(f"| {stage.name} | {'×'.join(map(str, stage.grid))} | {stage.tokens:,} | "
                  f"{'×'.join(map(str, stage.padded_grid))} | {stage.padded_tokens:,} |")
    for regime in ("matched_50_model_calls", "paper_default_or_hypothetical_few_step",
                   "rdd_step_allocation_sweep"):
        md.extend(["", f"## {regime}", "",
                   "| rank | method | calls C/M/F | retain | ideal speedup | BF16 probability cache | status |",
                   "|---:|---|---:|---:|---:|---:|---|"])
        for row in (r for r in rows if r.regime == regime):
            md.append(f"| {row.rank} | {row.method} | {row.stage_calls} | {row.retained_attention} | "
                      f"{row.ideal_speedup:.3f}× | {row.probability_cache_gib:.1f} GiB | {row.status} |")
        md.extend(["", "Notes:"])
        for row in (r for r in rows if r.regime == regime):
            md.append(f"- **{row.method}:** {row.notes}")
        render_svg(rows, args.output_dir / f"{regime}.svg", regime)
    (args.output_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
