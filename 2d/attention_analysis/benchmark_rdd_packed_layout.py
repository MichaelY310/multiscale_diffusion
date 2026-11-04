#!/usr/bin/env python3
"""Benchmark packed RDD kernels with the actual Q/K/V view layout."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from rdd_block_sparse_triton import (
    packed_block_sparse_attention,
    packed_block_sparse_attention_store,
    packed_cached_probability_av,
)


def ms(fn):
    return float(do_bench(fn, warmup=50, rep=200))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=72)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20260825)

    for tokens, block_side, densities in (
        (64, 2, (0.75, 0.875, 0.9375)),
        (256, 4, (0.625, 0.75, 0.875)),
    ):
        # Fused projection layout is [B,N,3,H,D].  Rotary creates contiguous Q/K
        # in the real model, while V remains a BNHD-strided view.
        fused = torch.randn(
            args.batch, tokens, 3, args.heads, args.dim,
            device="cuda", dtype=dtype,
        )
        q, k, v = fused.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = q.contiguous(), k.contiguous()
        assert not v.is_contiguous()
        blocks = tokens // (block_side * block_side)
        scores = torch.randn(args.batch, args.heads, blocks, blocks, device="cuda")
        dense_ms = ms(lambda: F.scaled_dot_product_attention(q, k, v))
        copy_ms = ms(lambda: v.contiguous())
        print(
            f"N={tokens} side={block_side} dense_sdpa={dense_ms:.4f}ms "
            f"V_copy={copy_ms:.4f}ms strides={v.stride()}"
        )
        for density in densities:
            keep = max(1, round(density * blocks))
            indices = scores.topk(keep, dim=-1).indices.to(torch.int32)
            _, p_contig = packed_block_sparse_attention_store(
                q, k, v, indices, block_side, contiguous_inputs=True
            )
            _, p_strided = packed_block_sparse_attention_store(
                q, k, v, indices, block_side, contiguous_inputs=False
            )
            values = {}
            for label, contiguous in (("copy", True), ("strided", False)):
                p = p_contig if contiguous else p_strided
                values[label] = (
                    ms(lambda: packed_block_sparse_attention(
                        q, k, v, indices, block_side, contiguous_inputs=contiguous
                    )),
                    ms(lambda: packed_block_sparse_attention_store(
                        q, k, v, indices, block_side, contiguous_inputs=contiguous
                    )),
                    ms(lambda: packed_cached_probability_av(
                        v, indices, p, block_side, contiguous_inputs=contiguous
                    )),
                )
            print(
                f" density={density:.4f} keep={keep}/{blocks} "
                f"copy={values['copy'][0]:.4f}/{values['copy'][1]:.4f}/{values['copy'][2]:.4f}ms "
                f"strided={values['strided'][0]:.4f}/{values['strided'][1]:.4f}/{values['strided'][2]:.4f}ms "
                f"copy/strided={values['copy'][0]/values['strided'][0]:.2f}x/"
                f"{values['copy'][1]/values['strided'][1]:.2f}x/"
                f"{values['copy'][2]/values['strided'][2]:.2f}x"
            )


if __name__ == "__main__":
    main()
