#!/usr/bin/env python3
"""Microbenchmark RDD sparse kernels at the actual Unified XL shapes."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from rdd_block_sparse_triton import (
    BLOCK,
    MICRO_BLOCK,
    block_sparse_attention,
    block_sparse_attention_store,
    cached_probability_av,
    micro_block_sparse_attention,
    micro_block_sparse_attention_store,
    micro_cached_probability_av,
    packed_block_sparse_attention,
    packed_block_sparse_attention_store,
    packed_cached_probability_av,
)


def ms(fn, warmup=25, rep=100):
    return float(do_bench(fn, warmup=warmup, rep=rep))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=72)
    parser.add_argument("--tokens", type=int, nargs="+", default=(64, 256))
    parser.add_argument("--densities", type=float, nargs="+", default=(0.5, 0.75))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(123)
    print("device", torch.cuda.get_device_name(0), "dtype", dtype)
    print("batch", args.batch, "heads", args.heads, "head_dim", args.dim)

    for tokens in args.tokens:
        q = torch.randn(args.batch, args.heads, tokens, args.dim, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        blocks = tokens // BLOCK
        router_scores = torch.randn(args.batch, args.heads, blocks, blocks, device="cuda")
        dense_explicit = lambda: (q @ k.transpose(-2, -1) / math.sqrt(args.dim)).softmax(-1) @ v
        dense_sdpa = lambda: F.scaled_dot_product_attention(q, k, v)
        explicit_ms = ms(dense_explicit)
        sdpa_ms = ms(dense_sdpa)
        print(f"\nN={tokens}: explicit_dense={explicit_ms:.4f} ms sdpa={sdpa_ms:.4f} ms")
        for density in args.densities:
            keep = max(1, math.ceil(density * blocks))
            indices = router_scores.topk(keep, dim=-1).indices.to(torch.int32)
            # Warm once to obtain a valid probability cache.
            _, probabilities = block_sparse_attention_store(q, k, v, indices)
            sparse_ms = ms(lambda: block_sparse_attention(q, k, v, indices))
            store_ms = ms(lambda: block_sparse_attention_store(q, k, v, indices))
            reuse_ms = ms(lambda: cached_probability_av(v, indices, probabilities))
            _, packed_probabilities = packed_block_sparse_attention_store(
                q, k, v, indices, block_side=4
            )
            packed_sparse_ms = ms(
                lambda: packed_block_sparse_attention(q, k, v, indices, block_side=4)
            )
            packed_store_ms = ms(
                lambda: packed_block_sparse_attention_store(q, k, v, indices, block_side=4)
            )
            packed_reuse_ms = ms(
                lambda: packed_cached_probability_av(
                    v, indices, packed_probabilities, block_side=4
                )
            )
            print(
                f" density={density:.2f} keep={keep}/{blocks}"
                f" sparse={sparse_ms:.4f} ms ({explicit_ms/sparse_ms:.2f}x explicit, {sdpa_ms/sparse_ms:.2f}x sdpa)"
                f" refresh+store={store_ms:.4f} ms"
                f" cached_AV={reuse_ms:.4f} ms ({explicit_ms/reuse_ms:.2f}x explicit, {sdpa_ms/reuse_ms:.2f}x sdpa)"
            )
            print(
                f"   packed64 sparse={packed_sparse_ms:.4f} ms"
                f" refresh+store={packed_store_ms:.4f} ms"
                f" cached_AV={packed_reuse_ms:.4f} ms"
                f" legacy/packed={sparse_ms/packed_sparse_ms:.2f}x/"
                f"{store_ms/packed_store_ms:.2f}x/{reuse_ms/packed_reuse_ms:.2f}x"
            )

    # Actual stage-16 path: N=64 and fine 2x2 spatial routing blocks.  This is
    # separate because the generic benchmark above uses 4x4 blocks.
    tokens = 64
    q = torch.randn(args.batch, args.heads, tokens, args.dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    blocks = tokens // MICRO_BLOCK
    router_scores = torch.randn(args.batch, args.heads, blocks, blocks, device="cuda")
    print("\nN=64, spatial block=2x2 (stage-16 route)")
    for density in args.densities:
        keep = max(1, math.ceil(density * blocks))
        indices = router_scores.topk(keep, dim=-1).indices.to(torch.int32)
        _, legacy_p = micro_block_sparse_attention_store(q, k, v, indices)
        _, packed_p = packed_block_sparse_attention_store(q, k, v, indices, block_side=2)
        legacy_sparse_ms = ms(lambda: micro_block_sparse_attention(q, k, v, indices))
        legacy_store_ms = ms(lambda: micro_block_sparse_attention_store(q, k, v, indices))
        legacy_reuse_ms = ms(lambda: micro_cached_probability_av(v, indices, legacy_p))
        packed_sparse_ms = ms(
            lambda: packed_block_sparse_attention(q, k, v, indices, block_side=2)
        )
        packed_store_ms = ms(
            lambda: packed_block_sparse_attention_store(q, k, v, indices, block_side=2)
        )
        packed_reuse_ms = ms(
            lambda: packed_cached_probability_av(v, indices, packed_p, block_side=2)
        )
        print(
            f" density={density:.4f} keep={keep}/{blocks}"
            f" legacy={legacy_sparse_ms:.4f}/{legacy_store_ms:.4f}/{legacy_reuse_ms:.4f} ms"
            f" packed={packed_sparse_ms:.4f}/{packed_store_ms:.4f}/{packed_reuse_ms:.4f} ms"
            f" speedup={legacy_sparse_ms/packed_sparse_ms:.2f}x/"
            f"{legacy_store_ms/packed_store_ms:.2f}x/{legacy_reuse_ms/packed_reuse_ms:.2f}x"
        )

    # Projection benchmark: the current module has a fused [Q,K,V] Linear.
    # On cache hits an optimized path slices and computes only the V projection.
    tokens = max(args.tokens)
    width = args.heads * args.dim
    x = torch.randn(args.batch, tokens, width, device="cuda", dtype=dtype)
    weight = torch.randn(3 * width, width, device="cuda", dtype=dtype)
    bias = torch.randn(3 * width, device="cuda", dtype=dtype)
    qkv_ms = ms(lambda: F.linear(x, weight, bias))
    v_ms = ms(lambda: F.linear(x, weight[2 * width :], bias[2 * width :]))
    print(
        f"\nprojection N={tokens}: fused_QKV={qkv_ms:.4f} ms V_only={v_ms:.4f} ms "
        f"speedup={qkv_ms/v_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
