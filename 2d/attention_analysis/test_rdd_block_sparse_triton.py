#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import torch

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


def dense_reference(q, k, v, indices, block_side=4):
    batch, heads, tokens, _ = q.shape
    block_size = block_side * block_side
    q_blocks = tokens // block_size
    grid = int(math.isqrt(tokens))
    y, x = torch.meshgrid(
        torch.arange(grid, device=q.device),
        torch.arange(grid, device=q.device),
        indexing="ij",
    )
    block_ids = ((y // block_side) * (grid // block_side) + x // block_side).flatten()
    selected = torch.zeros(batch, heads, q_blocks, q_blocks, dtype=torch.bool, device=q.device)
    selected.scatter_(-1, indices.long(), True)
    mask = selected[:, :, block_ids, :][:, :, :, block_ids]
    attention = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).masked_fill(
        ~mask, -torch.inf
    ).softmax(-1)
    return attention @ v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(123)
    batch, heads, tokens, dim = 2, 4, 256, 72
    packed = torch.randn(batch, tokens, 3, heads, dim, device="cuda", dtype=dtype)
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    assert not q.is_contiguous(), "test must cover the fused-QKV strided view"
    q_blocks = tokens // BLOCK
    scores = torch.randn(batch, heads, q_blocks, q_blocks, device="cuda")
    indices = scores.topk(q_blocks // 2, dim=-1).indices.to(torch.int32)
    reference = dense_reference(q, k, v, indices)
    sparse = block_sparse_attention(q, k, v, indices)
    stored, probabilities = block_sparse_attention_store(q, k, v, indices)
    reused = cached_probability_av(v, indices, probabilities)
    packed_sparse = packed_block_sparse_attention(q, k, v, indices, block_side=4)
    packed_stored, packed_probabilities = packed_block_sparse_attention_store(
        q, k, v, indices, block_side=4
    )
    packed_reused = packed_cached_probability_av(
        v, indices, packed_probabilities, block_side=4
    )
    packed_strided = packed_block_sparse_attention(
        q, k, v, indices, block_side=4, contiguous_inputs=False
    )
    packed_strided_stored, packed_strided_probabilities = packed_block_sparse_attention_store(
        q, k, v, indices, block_side=4, contiguous_inputs=False
    )
    packed_strided_reused = packed_cached_probability_av(
        v,
        indices,
        packed_strided_probabilities,
        block_side=4,
        contiguous_inputs=False,
    )
    packed32 = packed_block_sparse_attention(
        q, k, v, indices, block_side=4, kv_tile=32
    )
    packed32_stored, packed32_probabilities = packed_block_sparse_attention_store(
        q, k, v, indices, block_side=4, kv_tile=32
    )
    packed32_reused = packed_cached_probability_av(
        v, indices, packed32_probabilities, block_side=4, kv_tile=32
    )
    for name, candidate in (
        ("sparse", sparse),
        ("stored", stored),
        ("reused", reused),
        ("packed_sparse", packed_sparse),
        ("packed_stored", packed_stored),
        ("packed_reused", packed_reused),
        ("packed_strided", packed_strided),
        ("packed_strided_stored", packed_strided_stored),
        ("packed_strided_reused", packed_strided_reused),
        ("packed32", packed32),
        ("packed32_stored", packed32_stored),
        ("packed32_reused", packed32_reused),
    ):
        max_abs = float((candidate.float() - reference.float()).abs().max())
        rel_l2 = float((candidate.float() - reference.float()).norm() / reference.float().norm())
        cosine = float(torch.nn.functional.cosine_similarity(
            candidate.float().flatten(2), reference.float().flatten(2), dim=-1
        ).mean())
        print(name, "max_abs", max_abs, "relative_l2", rel_l2, "cosine", cosine)
        # Triton uses TensorFloat-32 for the FP32 tl.dot path on Blackwell,
        # matching the project's allow_tf32 inference configuration.
        tolerance = 3e-3 if dtype == torch.float32 else 2e-2
        if rel_l2 > tolerance:
            raise AssertionError(f"{name} relative L2 {rel_l2} > {tolerance}")

    # Stage-16 micro-block path: spatial 2x2 blocks, N=64, head_dim=72.
    tokens = 64
    packed = torch.randn(batch, tokens, 3, heads, dim, device="cuda", dtype=dtype)
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    micro_blocks = tokens // MICRO_BLOCK
    scores = torch.randn(batch, heads, micro_blocks, micro_blocks, device="cuda")
    indices = scores.topk(14, dim=-1).indices.to(torch.int32)
    reference = dense_reference(q, k, v, indices, block_side=2)
    sparse = micro_block_sparse_attention(q, k, v, indices)
    stored, probabilities = micro_block_sparse_attention_store(q, k, v, indices)
    reused = micro_cached_probability_av(v, indices, probabilities)
    packed_sparse = packed_block_sparse_attention(q, k, v, indices, block_side=2)
    packed_stored, packed_probabilities = packed_block_sparse_attention_store(
        q, k, v, indices, block_side=2
    )
    packed_reused = packed_cached_probability_av(
        v, indices, packed_probabilities, block_side=2
    )
    packed_strided = packed_block_sparse_attention(
        q, k, v, indices, block_side=2, contiguous_inputs=False
    )
    packed_strided_stored, packed_strided_probabilities = packed_block_sparse_attention_store(
        q, k, v, indices, block_side=2, contiguous_inputs=False
    )
    packed_strided_reused = packed_cached_probability_av(
        v,
        indices,
        packed_strided_probabilities,
        block_side=2,
        contiguous_inputs=False,
    )
    for name, candidate in (
        ("micro_sparse", sparse),
        ("micro_stored", stored),
        ("micro_reused", reused),
        ("packed_micro_sparse", packed_sparse),
        ("packed_micro_stored", packed_stored),
        ("packed_micro_reused", packed_reused),
        ("packed_strided_micro_sparse", packed_strided),
        ("packed_strided_micro_stored", packed_strided_stored),
        ("packed_strided_micro_reused", packed_strided_reused),
    ):
        rel_l2 = float((candidate.float() - reference.float()).norm() / reference.float().norm())
        cosine = float(torch.nn.functional.cosine_similarity(
            candidate.float().flatten(2), reference.float().flatten(2), dim=-1
        ).mean())
        print(name, "relative_l2", rel_l2, "cosine", cosine)
        # The packed micro path pads the 4-row spatial block to a 16-row tensor
        # core tile, so FP32 follows the same TF32 numerical path as the 4x4
        # packed kernel.  The legacy scalar micro kernel remains nearly exact.
        tolerance = (
            3e-3
            if dtype == torch.float32 and name.startswith("packed_")
            else (2e-5 if dtype == torch.float32 else 2e-2)
        )
        if rel_l2 > tolerance:
            raise AssertionError(f"{name} relative L2 {rel_l2} > {tolerance}")


if __name__ == "__main__":
    main()
