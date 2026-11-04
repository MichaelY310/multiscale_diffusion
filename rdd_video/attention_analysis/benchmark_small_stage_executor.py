"""Benchmark exact small-stage VSA execution strategies.

This isolates the block-sparse executor from the Wan projections/MLP.  It
compares FastVideo's native 64-token Triton kernel with an exact PyTorch SDPA
fallback and RDD's exact 16/32-query-token Triton executors.  The latter keep
the same 64-token VSA route/key blocks but expose more independent GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


BLOCK = 64


def make_route(
    *, batch: int, heads: int, blocks: int, density: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = max(1, min(blocks, math.ceil(density * blocks)))
    # Deterministic circular routes include self and avoid duplicate keys.
    q = torch.arange(blocks, device=device, dtype=torch.int32)
    delta = torch.arange(keep, device=device, dtype=torch.int32)
    row = (q[:, None] + delta[None, :]) % blocks
    indices = row.view(1, 1, blocks, keep).expand(batch, heads, -1, -1).contiguous()
    counts = torch.full(
        (batch, heads, blocks), keep, device=device, dtype=torch.int32
    )
    return indices, counts


def block_route_to_token_mask(
    indices: torch.Tensor,
    counts: torch.Tensor,
    variable_sizes: torch.Tensor,
) -> torch.Tensor:
    batch, heads, q_blocks, slots = indices.shape
    k_blocks = int(variable_sizes.numel())
    slot = torch.arange(slots, device=indices.device).view(1, 1, 1, -1)
    valid_slot = slot < counts.unsqueeze(-1)
    block_mask = torch.zeros(
        (batch, heads, q_blocks, k_blocks),
        device=indices.device,
        dtype=torch.bool,
    )
    block_mask.scatter_(-1, indices.to(torch.long), valid_slot)
    token_mask = block_mask.repeat_interleave(BLOCK, dim=-2).repeat_interleave(
        BLOCK, dim=-1
    )
    valid_key = (
        torch.arange(BLOCK, device=indices.device)[None, :]
        < variable_sizes[:, None]
    ).reshape(-1)
    return token_mask & valid_key.view(1, 1, 1, -1)


def bench(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def main(args: argparse.Namespace) -> None:
    kernel_python = Path(args.kernel_root).resolve() / "python"
    import sys

    sys.path.insert(0, str(kernel_python))
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices
    from rdd_sparse_attn_smallq_triton import smallq_block_sparse_attention

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    results = []
    for blocks in args.blocks:
        sequence = blocks * BLOCK
        q = torch.randn(
            args.batch,
            args.heads,
            sequence,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        indices, counts = make_route(
            batch=args.batch,
            heads=args.heads,
            blocks=blocks,
            density=args.density,
            device=device,
        )
        # Reproduce edge padding while keeping at least half a block valid.
        variable_sizes = torch.full(
            (blocks,), BLOCK, device=device, dtype=torch.int32
        )
        if args.partial_last:
            variable_sizes[-1] = BLOCK // 2
        mask = block_route_to_token_mask(indices, counts, variable_sizes)

        def native():
            return block_sparse_attn_from_indices(
                q, k, v, indices, counts, variable_sizes
            )[0]

        def masked_sdpa():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        def triton_q16():
            return smallq_block_sparse_attention(
                q, k, v, indices, counts, variable_sizes, query_tile=16
            )

        def triton_q32():
            return smallq_block_sparse_attention(
                q, k, v, indices, counts, variable_sizes, query_tile=32
            )

        # Dense is not equivalent to the sparse route.  It is measured only as
        # a hardware lower-overhead reference for tiny sequences.
        def dense_sdpa():
            return F.scaled_dot_product_attention(q, k, v)

        native_out = native()
        masked_out = masked_sdpa()
        q16_out = triton_q16()
        q32_out = triton_q32()
        valid_query = (
            torch.arange(BLOCK, device=device)[None, :]
            < variable_sizes[:, None]
        ).reshape(-1)
        diff = (native_out[:, :, valid_query] - masked_out[:, :, valid_query]).float()
        native_ms = bench(native, args.warmup, args.repeats)
        masked_ms = bench(masked_sdpa, args.warmup, args.repeats)
        q16_ms = bench(triton_q16, args.warmup, args.repeats)
        q32_ms = bench(triton_q32, args.warmup, args.repeats)
        dense_ms = bench(dense_sdpa, args.warmup, args.repeats)
        q16_diff = (native_out[:, :, valid_query] - q16_out[:, :, valid_query]).float()
        q32_diff = (native_out[:, :, valid_query] - q32_out[:, :, valid_query]).float()
        results.append(
            {
                "blocks": blocks,
                "padded_tokens": sequence,
                "keep_blocks": int(counts[0, 0, 0].item()),
                "actual_density": float(counts[0, 0, 0].item() / blocks),
                "native_ms_median": statistics.median(native_ms),
                "masked_sdpa_ms_median": statistics.median(masked_ms),
                "triton_q16_ms_median": statistics.median(q16_ms),
                "triton_q32_ms_median": statistics.median(q32_ms),
                "dense_sdpa_ms_median": statistics.median(dense_ms),
                "masked_speedup_vs_native": statistics.median(native_ms)
                / statistics.median(masked_ms),
                "dense_speedup_vs_native": statistics.median(native_ms)
                / statistics.median(dense_ms),
                "triton_q16_speedup_vs_native": statistics.median(native_ms)
                / statistics.median(q16_ms),
                "triton_q32_speedup_vs_native": statistics.median(native_ms)
                / statistics.median(q32_ms),
                "max_abs_exact_valid_queries": float(diff.abs().max().item()),
                "mean_abs_exact_valid_queries": float(diff.abs().mean().item()),
                "triton_q16_max_abs_valid_queries": float(q16_diff.abs().max().item()),
                "triton_q16_mean_abs_valid_queries": float(q16_diff.abs().mean().item()),
                "triton_q32_max_abs_valid_queries": float(q32_diff.abs().max().item()),
                "triton_q32_mean_abs_valid_queries": float(q32_diff.abs().mean().item()),
            }
        )
        del q, k, v, indices, counts, variable_sizes, mask
        torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "density_requested": args.density,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 16, 32, 56, 128, 182])
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--partial-last", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
