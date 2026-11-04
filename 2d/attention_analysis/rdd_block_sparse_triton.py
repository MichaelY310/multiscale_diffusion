"""Inference-only Triton kernels for RDD hierarchical sparse attention.

The online-softmax structure follows the Apache-2.0 FastVideo VSA Triton
kernel, but this implementation is specialized for RDD image DiTs:

* 16-token (4x4 spatial) blocks instead of 64-token video cubes;
* arbitrary head dimensions, including DiT-XL/2's head_dim=72;
* an optional sparse-probability cache for A_old @ V_current reuse;
* forward-only, because the first target is training-free sampling.

No dense N x N attention tensor is formed by these kernels.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK = 16
MICRO_BLOCK = 4


@triton.jit
def _rdd_sparse_attention_fwd(
    Q,
    K,
    V,
    BlockIndices,
    Out,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ib,
    stride_ih,
    stride_iq,
    stride_ik,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    head = bh % H

    blocks_axis = GRID_SIZE // 4
    q_block_y = q_block // blocks_axis
    q_block_x = q_block % blocks_axis
    local_rows = tl.arange(0, BLOCK_SIZE)
    rows = (
        (q_block_y * 4 + local_rows // 4) * GRID_SIZE
        + q_block_x * 4
        + local_rows % 4
    )
    dims = tl.arange(0, BLOCK_D)
    row_mask = rows < N
    dim_mask = dims < D
    q_ptrs = (
        Q
        + batch * stride_qb
        + head * stride_qh
        + rows[:, None] * stride_qn
        + dims[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)

    m = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_SIZE,), tl.float32)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)

    index_base = (
        BlockIndices
        + batch * stride_ib
        + head * stride_ih
        + q_block * stride_iq
    )
    cols_local = tl.arange(0, BLOCK_SIZE)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        valid_block = kv_block >= 0
        kv_block_y = kv_block // blocks_axis
        kv_block_x = kv_block % blocks_axis
        cols = (
            (kv_block_y * 4 + cols_local // 4) * GRID_SIZE
            + kv_block_x * 4
            + cols_local % 4
        )
        col_mask = valid_block & (cols < N)

        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + cols[None, :] * stride_kn
            + dims[:, None] * stride_kd
        )
        k_t = tl.load(k_ptrs, mask=dim_mask[:, None] & col_mask[None, :], other=0.0)
        scores = tl.dot(q, k_t) * sm_scale
        scores = tl.where(row_mask[:, None] & col_mask[None, :], scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m - m_new)
        p = tl.where(row_mask[:, None] & col_mask[None, :], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + cols[:, None] * stride_vn
            + dims[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=col_mask[:, None] & dim_mask[None, :], other=0.0)
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        m = m_new

    acc = acc / l[:, None]
    out_ptrs = (
        Out
        + batch * stride_ob
        + head * stride_oh
        + rows[:, None] * stride_on
        + dims[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & dim_mask[None, :])


@triton.jit
def _rdd_sparse_attention_store_fwd(
    Q,
    K,
    V,
    BlockIndices,
    Probabilities,
    Out,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ib,
    stride_ih,
    stride_iq,
    stride_ik,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_ps,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Sparse attention plus materialized sparse probabilities.

    A two-pass kernel is intentional: refresh calls pay an extra sparse QK pass
    so the following calls can skip Q/K projections and QK/softmax entirely.
    """
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    head = bh % H
    blocks_axis = GRID_SIZE // 4
    q_block_y = q_block // blocks_axis
    q_block_x = q_block % blocks_axis
    local_rows = tl.arange(0, BLOCK_SIZE)
    rows = (
        (q_block_y * 4 + local_rows // 4) * GRID_SIZE
        + q_block_x * 4
        + local_rows % 4
    )
    dims = tl.arange(0, BLOCK_D)
    row_mask = rows < N
    dim_mask = dims < D
    q_ptrs = (
        Q
        + batch * stride_qb
        + head * stride_qh
        + rows[:, None] * stride_qn
        + dims[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)
    index_base = BlockIndices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    local = tl.arange(0, BLOCK_SIZE)

    m = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_SIZE,), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        kv_block_y = kv_block // blocks_axis
        kv_block_x = kv_block % blocks_axis
        cols = (
            (kv_block_y * 4 + local // 4) * GRID_SIZE
            + kv_block_x * 4
            + local % 4
        )
        col_mask = (kv_block >= 0) & (cols < N)
        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + cols[None, :] * stride_kn
            + dims[:, None] * stride_kd
        )
        k_t = tl.load(k_ptrs, mask=dim_mask[:, None] & col_mask[None, :], other=0.0)
        scores = tl.dot(q, k_t) * sm_scale
        scores = tl.where(row_mask[:, None] & col_mask[None, :], scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(row_mask[:, None] & col_mask[None, :], p, 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

    acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        kv_block_y = kv_block // blocks_axis
        kv_block_x = kv_block % blocks_axis
        cols = (
            (kv_block_y * 4 + local // 4) * GRID_SIZE
            + kv_block_x * 4
            + local % 4
        )
        col_mask = (kv_block >= 0) & (cols < N)
        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + cols[None, :] * stride_kn
            + dims[:, None] * stride_kd
        )
        k_t = tl.load(k_ptrs, mask=dim_mask[:, None] & col_mask[None, :], other=0.0)
        scores = tl.dot(q, k_t) * sm_scale
        p = tl.exp(scores - m[:, None]) / l[:, None]
        p = tl.where(row_mask[:, None] & col_mask[None, :], p, 0.0)
        probability_cols = slot * BLOCK_SIZE + local
        p_ptrs = (
            Probabilities
            + batch * stride_pb
            + head * stride_ph
            + rows[:, None] * stride_pn
            + probability_cols[None, :] * stride_ps
        )
        tl.store(p_ptrs, p, mask=row_mask[:, None])

        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + cols[:, None] * stride_vn
            + dims[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=col_mask[:, None] & dim_mask[None, :], other=0.0)
        acc += tl.dot(p.to(V.dtype.element_ty), v)

    out_ptrs = (
        Out
        + batch * stride_ob
        + head * stride_oh
        + rows[:, None] * stride_on
        + dims[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & dim_mask[None, :])


@triton.jit
def _rdd_cached_probability_av_fwd(
    V,
    BlockIndices,
    Probabilities,
    Out,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ib,
    stride_ih,
    stride_iq,
    stride_ik,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_ps,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    N: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    head = bh % H
    blocks_axis = GRID_SIZE // 4
    q_block_y = q_block // blocks_axis
    q_block_x = q_block % blocks_axis
    local_rows = tl.arange(0, BLOCK_SIZE)
    rows = (
        (q_block_y * 4 + local_rows // 4) * GRID_SIZE
        + q_block_x * 4
        + local_rows % 4
    )
    dims = tl.arange(0, BLOCK_D)
    row_mask = rows < N
    dim_mask = dims < D
    local = tl.arange(0, BLOCK_SIZE)
    index_base = BlockIndices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)

    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        kv_block_y = kv_block // blocks_axis
        kv_block_x = kv_block % blocks_axis
        cols = (
            (kv_block_y * 4 + local // 4) * GRID_SIZE
            + kv_block_x * 4
            + local % 4
        )
        col_mask = (kv_block >= 0) & (cols < N)
        probability_cols = slot * BLOCK_SIZE + local
        p_ptrs = (
            Probabilities
            + batch * stride_pb
            + head * stride_ph
            + rows[:, None] * stride_pn
            + probability_cols[None, :] * stride_ps
        )
        p = tl.load(p_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + cols[:, None] * stride_vn
            + dims[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=col_mask[:, None] & dim_mask[None, :], other=0.0)
        acc += tl.dot(p.to(V.dtype.element_ty), v)

    out_ptrs = (
        Out
        + batch * stride_ob
        + head * stride_oh
        + rows[:, None] * stride_on
        + dims[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & dim_mask[None, :])


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda and indices.is_cuda):
        raise ValueError("RDD Triton sparse attention requires CUDA tensors")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, v must have identical [B,H,N,D] shapes")
    if q.shape[-2] % BLOCK:
        raise ValueError(f"token count must be divisible by {BLOCK}")
    expected = (q.shape[0], q.shape[1], q.shape[2] // BLOCK)
    if tuple(indices.shape[:3]) != expected:
        raise ValueError(f"indices prefix must be {expected}, got {tuple(indices.shape)}")
    if indices.dtype != torch.int32:
        raise ValueError("indices must be int32")


def _launch(kernel, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor,
            probabilities: torch.Tensor | None = None) -> torch.Tensor:
    _validate(q, k, v, indices)
    # Contiguous BHND copies are faster than strided token loads on the target
    # Blackwell GPU, despite the copy cost (measured end-to-end, not assumed).
    q, k, v, indices = q.contiguous(), k.contiguous(), v.contiguous(), indices.contiguous()
    batch, heads, tokens, head_dim = q.shape
    grid_size = math.isqrt(tokens)
    if grid_size * grid_size != tokens or grid_size % 4:
        raise ValueError("tokens must form a square grid divisible by 4")
    max_blocks = indices.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    output = torch.empty_like(q)
    grid = (tokens // BLOCK, batch * heads)
    common = (
        v,
        indices,
    )
    if probabilities is None:
        kernel[grid](
            q, k, v, indices, output, 1.0 / math.sqrt(head_dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *output.stride(),
            N=tokens, GRID_SIZE=grid_size, H=heads, D=head_dim, MAX_BLOCKS=max_blocks,
            BLOCK_SIZE=BLOCK, BLOCK_D=block_d,
            num_warps=4, num_stages=3,
        )
    else:
        kernel[grid](
            q, k, v, indices, probabilities, output, 1.0 / math.sqrt(head_dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(),
            *probabilities.stride(), *output.stride(),
            N=tokens, GRID_SIZE=grid_size, H=heads, D=head_dim, MAX_BLOCKS=max_blocks,
            BLOCK_SIZE=BLOCK, BLOCK_D=block_d,
            num_warps=4, num_stages=3,
        )
    return output


@torch.inference_mode()
def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
) -> torch.Tensor:
    """Exact softmax over selected 16-token K/V blocks."""
    return _launch(_rdd_sparse_attention_fwd, q, k, v, block_indices)


@torch.inference_mode()
def block_sparse_attention_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention and its compact post-softmax probability cache."""
    _validate(q, k, v, block_indices)
    probabilities = torch.empty(
        q.shape[0], q.shape[1], q.shape[2], block_indices.shape[-1] * BLOCK,
        device=q.device, dtype=q.dtype,
    )
    output = _launch(
        _rdd_sparse_attention_store_fwd,
        q,
        k,
        v,
        block_indices,
        probabilities,
    )
    return output, probabilities


@torch.inference_mode()
def cached_probability_av(
    v: torch.Tensor,
    block_indices: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Compute A_cached @ V_current without Q/K or a dense matrix."""
    dummy = v
    _validate(dummy, dummy, v, block_indices)
    expected = (*v.shape[:3], block_indices.shape[-1] * BLOCK)
    if tuple(probabilities.shape) != expected:
        raise ValueError(f"probability cache must have shape {expected}")
    v, block_indices, probabilities = (
        v.contiguous(), block_indices.contiguous(), probabilities.contiguous()
    )
    batch, heads, tokens, head_dim = v.shape
    grid_size = math.isqrt(tokens)
    output = torch.empty_like(v)
    grid = (tokens // BLOCK, batch * heads)
    _rdd_cached_probability_av_fwd[grid](
        v, block_indices, probabilities, output,
        *v.stride(), *block_indices.stride(), *probabilities.stride(), *output.stride(),
        N=tokens, GRID_SIZE=grid_size, H=heads, D=head_dim,
        MAX_BLOCKS=block_indices.shape[-1],
        BLOCK_SIZE=BLOCK, BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4, num_stages=3,
    )
    return output


@triton.jit
def _rdd_micro_sparse_fwd(
    Q, K, V, Indices, Out, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, MICRO_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    query = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < D
    q = tl.load(
        Q + batch * stride_qb + head * stride_qh + query * stride_qn + dims * stride_qd,
        mask=dim_mask,
        other=0.0,
    )
    qy, qx = query // GRID_SIZE, query % GRID_SIZE
    blocks_axis = GRID_SIZE // 2
    q_block = (qy // 2) * blocks_axis + qx // 2
    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    local = tl.arange(0, MICRO_SIZE)
    m = tl.full((), -float("inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        by, bx = kv_block // blocks_axis, kv_block % blocks_axis
        keys = (by * 2 + local // 2) * GRID_SIZE + bx * 2 + local % 2
        valid = (kv_block >= 0) & (keys < N)
        k = tl.load(
            K + batch * stride_kb + head * stride_kh
            + keys[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        acc *= alpha
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + keys[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(p[:, None] * value, axis=0)
        m = m_new
    acc /= l
    tl.store(
        Out + batch * stride_ob + head * stride_oh + query * stride_on + dims * stride_od,
        acc,
        mask=dim_mask,
    )


@triton.jit
def _rdd_micro_sparse_store_fwd(
    Q, K, V, Indices, Probabilities, Out, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_pb, stride_ph, stride_pn, stride_ps,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, MICRO_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    query = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < D
    q = tl.load(
        Q + batch * stride_qb + head * stride_qh + query * stride_qn + dims * stride_qd,
        mask=dim_mask,
        other=0.0,
    )
    qy, qx = query // GRID_SIZE, query % GRID_SIZE
    blocks_axis = GRID_SIZE // 2
    q_block = (qy // 2) * blocks_axis + qx // 2
    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    local = tl.arange(0, MICRO_SIZE)
    m = tl.full((), -float("inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        by, bx = kv_block // blocks_axis, kv_block % blocks_axis
        keys = (by * 2 + local // 2) * GRID_SIZE + bx * 2 + local % 2
        valid = (kv_block >= 0) & (keys < N)
        k = tl.load(
            K + batch * stride_kb + head * stride_kh
            + keys[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dim_mask[None, :], other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        by, bx = kv_block // blocks_axis, kv_block % blocks_axis
        keys = (by * 2 + local // 2) * GRID_SIZE + bx * 2 + local % 2
        valid = (kv_block >= 0) & (keys < N)
        k = tl.load(
            K + batch * stride_kb + head * stride_kh
            + keys[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dim_mask[None, :], other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        p = tl.exp(scores - m) / l
        p = tl.where(valid, p, 0.0)
        tl.store(
            Probabilities + batch * stride_pb + head * stride_ph + query * stride_pn
            + (slot * MICRO_SIZE + local) * stride_ps,
            p,
        )
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + keys[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dim_mask[None, :], other=0.0,
        )
        acc += tl.sum(p[:, None] * value, axis=0)
    tl.store(
        Out + batch * stride_ob + head * stride_oh + query * stride_on + dims * stride_od,
        acc,
        mask=dim_mask,
    )


@triton.jit
def _rdd_micro_cached_av_fwd(
    V, Indices, Probabilities, Out,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_pb, stride_ph, stride_pn, stride_ps,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, MICRO_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    query = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < D
    qy, qx = query // GRID_SIZE, query % GRID_SIZE
    blocks_axis = GRID_SIZE // 2
    q_block = (qy // 2) * blocks_axis + qx // 2
    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    local = tl.arange(0, MICRO_SIZE)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for slot in range(MAX_BLOCKS):
        kv_block = tl.load(index_base + slot * stride_ik).to(tl.int32)
        by, bx = kv_block // blocks_axis, kv_block % blocks_axis
        keys = (by * 2 + local // 2) * GRID_SIZE + bx * 2 + local % 2
        valid = (kv_block >= 0) & (keys < N)
        p = tl.load(
            Probabilities + batch * stride_pb + head * stride_ph + query * stride_pn
            + (slot * MICRO_SIZE + local) * stride_ps,
            mask=valid,
            other=0.0,
        )
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + keys[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dim_mask[None, :], other=0.0,
        )
        acc += tl.sum(p[:, None] * value, axis=0)
    tl.store(
        Out + batch * stride_ob + head * stride_oh + query * stride_on + dims * stride_od,
        acc,
        mask=dim_mask,
    )


def _micro_validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    indices: torch.Tensor) -> tuple[int, int, int, int]:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("micro q/k/v must share [B,H,N,D]")
    batch, heads, tokens, dim = q.shape
    grid = math.isqrt(tokens)
    if grid * grid != tokens or grid % 2:
        raise ValueError("micro tokens must form an even square grid")
    expected = (batch, heads, tokens // MICRO_BLOCK)
    if tuple(indices.shape[:3]) != expected or indices.dtype != torch.int32:
        raise ValueError(f"micro indices prefix must be {expected} and int32")
    return batch, heads, tokens, dim


def _micro_launch(kernel, q, k, v, indices, probabilities=None):
    batch, heads, tokens, dim = _micro_validate(q, k, v, indices)
    q, k, v, indices = q.contiguous(), k.contiguous(), v.contiguous(), indices.contiguous()
    output = torch.empty_like(q)
    grid_size = math.isqrt(tokens)
    launch_grid = (tokens, batch * heads)
    constexprs = dict(
        N=tokens, GRID_SIZE=grid_size, H=heads, D=dim,
        MAX_BLOCKS=indices.shape[-1], MICRO_SIZE=MICRO_BLOCK,
        BLOCK_D=triton.next_power_of_2(dim), num_warps=4, num_stages=2,
    )
    if probabilities is None:
        kernel[launch_grid](
            q, k, v, indices, output, 1.0 / math.sqrt(dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *output.stride(),
            **constexprs,
        )
    else:
        kernel[launch_grid](
            q, k, v, indices, probabilities, output, 1.0 / math.sqrt(dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(),
            *probabilities.stride(), *output.stride(), **constexprs,
        )
    return output


@torch.inference_mode()
def micro_block_sparse_attention(q, k, v, block_indices):
    return _micro_launch(_rdd_micro_sparse_fwd, q, k, v, block_indices)


@torch.inference_mode()
def micro_block_sparse_attention_store(q, k, v, block_indices):
    _micro_validate(q, k, v, block_indices)
    probabilities = torch.empty(
        q.shape[0], q.shape[1], q.shape[2], block_indices.shape[-1] * MICRO_BLOCK,
        device=q.device, dtype=q.dtype,
    )
    output = _micro_launch(
        _rdd_micro_sparse_store_fwd, q, k, v, block_indices, probabilities
    )
    return output, probabilities


@torch.inference_mode()
def micro_cached_probability_av(v, block_indices, probabilities):
    batch, heads, tokens, dim = _micro_validate(v, v, v, block_indices)
    expected = (*v.shape[:3], block_indices.shape[-1] * MICRO_BLOCK)
    if tuple(probabilities.shape) != expected:
        raise ValueError(f"micro probability cache must have shape {expected}")
    v, block_indices, probabilities = v.contiguous(), block_indices.contiguous(), probabilities.contiguous()
    output = torch.empty_like(v)
    grid_size = math.isqrt(tokens)
    _rdd_micro_cached_av_fwd[(tokens, batch * heads)](
        v, block_indices, probabilities, output,
        *v.stride(), *block_indices.stride(), *probabilities.stride(), *output.stride(),
        N=tokens, GRID_SIZE=grid_size, H=heads, D=dim,
        MAX_BLOCKS=block_indices.shape[-1], MICRO_SIZE=MICRO_BLOCK,
        BLOCK_D=triton.next_power_of_2(dim), num_warps=4, num_stages=2,
    )
    return output


# ---------------------------------------------------------------------------
# Short-sequence packed kernels
# ---------------------------------------------------------------------------
# FastVideo's VSA kernel obtains good utilization from 64-token Q/K tiles.  RDD
# image stages are much shorter (N=64 and N=256), while their spatial routing
# blocks contain only 4 or 16 tokens.  The legacy kernels above therefore issue
# one tiny dot/reduction per selected spatial block.  The kernels below preserve
# exactly the same selected-token softmax, but pack consecutive *route slots*
# into a 64-token compute tile.  A 16-token spatial route processes four selected
# blocks per online-softmax iteration; a 4-token route processes sixteen.

PACK_TILE = 64


@triton.jit
def _rdd_packed_sparse_fwd(
    Q, K, V, Indices, Out, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, BLOCK_SIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, Q_TILE: tl.constexpr,
    KV_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H

    blocks_axis = GRID_SIZE // BLOCK_SIDE
    q_block_y, q_block_x = q_block // blocks_axis, q_block % blocks_axis
    q_local = tl.arange(0, Q_TILE)
    q_valid = q_local < BLOCK_SIZE
    rows = (
        (q_block_y * BLOCK_SIDE + q_local // BLOCK_SIDE) * GRID_SIZE
        + q_block_x * BLOCK_SIDE
        + q_local % BLOCK_SIDE
    )
    q_valid = q_valid & (rows < N)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < D
    q = tl.load(
        Q + batch * stride_qb + head * stride_qh
        + rows[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=q_valid[:, None] & dim_valid[None, :], other=0.0,
    )

    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    kv_local = tl.arange(0, KV_TILE)
    m = tl.full((Q_TILE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE,), tl.float32)
    acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

    for packed_start in range(0, MAX_BLOCKS * BLOCK_SIZE, KV_TILE):
        packed = packed_start + kv_local
        packed_valid = packed < MAX_BLOCKS * BLOCK_SIZE
        slot = packed // BLOCK_SIZE
        local = packed % BLOCK_SIZE
        kv_block = tl.load(
            index_base + slot * stride_ik,
            mask=packed_valid,
            other=-1,
        ).to(tl.int32)
        valid_block = packed_valid & (kv_block >= 0)
        safe_block = tl.maximum(kv_block, 0)
        key_y, key_x = safe_block // blocks_axis, safe_block % blocks_axis
        cols = (
            (key_y * BLOCK_SIDE + local // BLOCK_SIDE) * GRID_SIZE
            + key_x * BLOCK_SIDE
            + local % BLOCK_SIDE
        )
        key_valid = valid_block & (cols < N)
        k_t = tl.load(
            K + batch * stride_kb + head * stride_kh
            + dims[:, None] * stride_kd + cols[None, :] * stride_kn,
            mask=dim_valid[:, None] & key_valid[None, :], other=0.0,
        )
        scores = tl.dot(q, k_t) * sm_scale
        scores = tl.where(q_valid[:, None] & key_valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(q_valid[:, None] & key_valid[None, :], p, 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + cols[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=key_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        acc += tl.dot(p.to(V.dtype.element_ty), value)
        m = m_new

    acc /= l[:, None]
    tl.store(
        Out + batch * stride_ob + head * stride_oh
        + rows[:, None] * stride_on + dims[None, :] * stride_od,
        acc,
        mask=q_valid[:, None] & dim_valid[None, :],
    )


@triton.jit
def _rdd_packed_sparse_store_fwd(
    Q, K, V, Indices, Probabilities, Out, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_pb, stride_ph, stride_pn, stride_ps,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, BLOCK_SIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, Q_TILE: tl.constexpr,
    KV_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Packed sparse attention that also materializes normalized selected A.

    The first pass computes global online-softmax statistics across packed
    64-token tiles.  The second pass writes normalized probabilities and AV.
    It is mathematically identical to the legacy store kernel, but performs up
    to 4x/16x fewer loop iterations for 4x4/2x2 spatial blocks respectively.
    """
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    blocks_axis = GRID_SIZE // BLOCK_SIDE
    q_block_y, q_block_x = q_block // blocks_axis, q_block % blocks_axis
    q_local = tl.arange(0, Q_TILE)
    q_valid = q_local < BLOCK_SIZE
    rows = (
        (q_block_y * BLOCK_SIDE + q_local // BLOCK_SIDE) * GRID_SIZE
        + q_block_x * BLOCK_SIDE
        + q_local % BLOCK_SIDE
    )
    q_valid = q_valid & (rows < N)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < D
    q = tl.load(
        Q + batch * stride_qb + head * stride_qh
        + rows[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=q_valid[:, None] & dim_valid[None, :], other=0.0,
    )
    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    kv_local = tl.arange(0, KV_TILE)
    m = tl.full((Q_TILE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE,), tl.float32)

    for packed_start in range(0, MAX_BLOCKS * BLOCK_SIZE, KV_TILE):
        packed = packed_start + kv_local
        packed_valid = packed < MAX_BLOCKS * BLOCK_SIZE
        slot, local = packed // BLOCK_SIZE, packed % BLOCK_SIZE
        kv_block = tl.load(index_base + slot * stride_ik, mask=packed_valid, other=-1).to(tl.int32)
        key_valid = packed_valid & (kv_block >= 0)
        safe_block = tl.maximum(kv_block, 0)
        key_y, key_x = safe_block // blocks_axis, safe_block % blocks_axis
        cols = (
            (key_y * BLOCK_SIDE + local // BLOCK_SIDE) * GRID_SIZE
            + key_x * BLOCK_SIDE + local % BLOCK_SIDE
        )
        key_valid = key_valid & (cols < N)
        k_t = tl.load(
            K + batch * stride_kb + head * stride_kh
            + dims[:, None] * stride_kd + cols[None, :] * stride_kn,
            mask=dim_valid[:, None] & key_valid[None, :], other=0.0,
        )
        scores = tl.dot(q, k_t) * sm_scale
        scores = tl.where(q_valid[:, None] & key_valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(q_valid[:, None] & key_valid[None, :], p, 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

    acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
    for packed_start in range(0, MAX_BLOCKS * BLOCK_SIZE, KV_TILE):
        packed = packed_start + kv_local
        packed_valid = packed < MAX_BLOCKS * BLOCK_SIZE
        slot, local = packed // BLOCK_SIZE, packed % BLOCK_SIZE
        kv_block = tl.load(index_base + slot * stride_ik, mask=packed_valid, other=-1).to(tl.int32)
        key_valid = packed_valid & (kv_block >= 0)
        safe_block = tl.maximum(kv_block, 0)
        key_y, key_x = safe_block // blocks_axis, safe_block % blocks_axis
        cols = (
            (key_y * BLOCK_SIDE + local // BLOCK_SIDE) * GRID_SIZE
            + key_x * BLOCK_SIDE + local % BLOCK_SIDE
        )
        key_valid = key_valid & (cols < N)
        k_t = tl.load(
            K + batch * stride_kb + head * stride_kh
            + dims[:, None] * stride_kd + cols[None, :] * stride_kn,
            mask=dim_valid[:, None] & key_valid[None, :], other=0.0,
        )
        scores = tl.dot(q, k_t) * sm_scale
        p = tl.exp(scores - m[:, None]) / l[:, None]
        p = tl.where(q_valid[:, None] & key_valid[None, :], p, 0.0)
        tl.store(
            Probabilities + batch * stride_pb + head * stride_ph
            + rows[:, None] * stride_pn + packed[None, :] * stride_ps,
            p,
            mask=q_valid[:, None] & packed_valid[None, :],
        )
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + cols[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=key_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        acc += tl.dot(p.to(V.dtype.element_ty), value)

    tl.store(
        Out + batch * stride_ob + head * stride_oh
        + rows[:, None] * stride_on + dims[None, :] * stride_od,
        acc,
        mask=q_valid[:, None] & dim_valid[None, :],
    )


@triton.jit
def _rdd_packed_cached_av_fwd(
    V, Indices, Probabilities, Out,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_pb, stride_ph, stride_pn, stride_ps,
    stride_ob, stride_oh, stride_on, stride_od,
    N: tl.constexpr, GRID_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, BLOCK_SIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, Q_TILE: tl.constexpr,
    KV_TILE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    blocks_axis = GRID_SIZE // BLOCK_SIDE
    q_block_y, q_block_x = q_block // blocks_axis, q_block % blocks_axis
    q_local = tl.arange(0, Q_TILE)
    q_valid = q_local < BLOCK_SIZE
    rows = (
        (q_block_y * BLOCK_SIDE + q_local // BLOCK_SIDE) * GRID_SIZE
        + q_block_x * BLOCK_SIDE + q_local % BLOCK_SIDE
    )
    q_valid = q_valid & (rows < N)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < D
    index_base = Indices + batch * stride_ib + head * stride_ih + q_block * stride_iq
    kv_local = tl.arange(0, KV_TILE)
    acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

    for packed_start in range(0, MAX_BLOCKS * BLOCK_SIZE, KV_TILE):
        packed = packed_start + kv_local
        packed_valid = packed < MAX_BLOCKS * BLOCK_SIZE
        slot, local = packed // BLOCK_SIZE, packed % BLOCK_SIZE
        kv_block = tl.load(index_base + slot * stride_ik, mask=packed_valid, other=-1).to(tl.int32)
        key_valid = packed_valid & (kv_block >= 0)
        safe_block = tl.maximum(kv_block, 0)
        key_y, key_x = safe_block // blocks_axis, safe_block % blocks_axis
        cols = (
            (key_y * BLOCK_SIDE + local // BLOCK_SIDE) * GRID_SIZE
            + key_x * BLOCK_SIDE + local % BLOCK_SIDE
        )
        key_valid = key_valid & (cols < N)
        p = tl.load(
            Probabilities + batch * stride_pb + head * stride_ph
            + rows[:, None] * stride_pn + packed[None, :] * stride_ps,
            mask=q_valid[:, None] & key_valid[None, :], other=0.0,
        )
        value = tl.load(
            V + batch * stride_vb + head * stride_vh
            + cols[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=key_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        acc += tl.dot(p.to(V.dtype.element_ty), value)

    tl.store(
        Out + batch * stride_ob + head * stride_oh
        + rows[:, None] * stride_on + dims[None, :] * stride_od,
        acc,
        mask=q_valid[:, None] & dim_valid[None, :],
    )


def _packed_validate(q, k, v, indices, block_side: int):
    if not (q.is_cuda and k.is_cuda and v.is_cuda and indices.is_cuda):
        raise ValueError("RDD packed sparse attention requires CUDA tensors")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("packed q/k/v must share [B,H,N,D]")
    if block_side not in (2, 4):
        raise ValueError(f"packed kernel supports block_side 2 or 4, got {block_side}")
    batch, heads, tokens, dim = q.shape
    grid = math.isqrt(tokens)
    block_size = block_side * block_side
    if grid * grid != tokens or grid % block_side:
        raise ValueError("tokens must form a square grid divisible by block_side")
    expected = (batch, heads, tokens // block_size)
    if tuple(indices.shape[:3]) != expected or indices.dtype != torch.int32:
        raise ValueError(f"packed indices prefix must be {expected} and int32")
    return batch, heads, tokens, dim, grid, block_size


def _packed_launch(
    kernel,
    q,
    k,
    v,
    indices,
    block_side: int,
    probabilities=None,
    contiguous_inputs: bool = True,
    kv_tile: int = PACK_TILE,
):
    batch, heads, tokens, dim, grid_size, block_size = _packed_validate(
        q, k, v, indices, block_side
    )
    if contiguous_inputs:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    indices = indices.contiguous()
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    launch_grid = (tokens // block_size, batch * heads)
    constexprs = dict(
        N=tokens, GRID_SIZE=grid_size, H=heads, D=dim,
        MAX_BLOCKS=indices.shape[-1], BLOCK_SIDE=block_side,
        BLOCK_SIZE=block_size, Q_TILE=16, KV_TILE=kv_tile,
        BLOCK_D=triton.next_power_of_2(dim), num_warps=4, num_stages=3,
    )
    if probabilities is None:
        kernel[launch_grid](
            q, k, v, indices, output, 1.0 / math.sqrt(dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(), *output.stride(),
            **constexprs,
        )
    else:
        kernel[launch_grid](
            q, k, v, indices, probabilities, output, 1.0 / math.sqrt(dim),
            *q.stride(), *k.stride(), *v.stride(), *indices.stride(),
            *probabilities.stride(), *output.stride(), **constexprs,
        )
    return output


@torch.inference_mode()
def packed_block_sparse_attention(
    q,
    k,
    v,
    block_indices,
    block_side: int = 4,
    contiguous_inputs: bool = True,
    kv_tile: int = PACK_TILE,
):
    """Exact selected-token softmax using 64-token packed route tiles."""
    return _packed_launch(
        _rdd_packed_sparse_fwd,
        q,
        k,
        v,
        block_indices,
        block_side,
        contiguous_inputs=contiguous_inputs,
        kv_tile=kv_tile,
    )


@torch.inference_mode()
def packed_block_sparse_attention_store(
    q,
    k,
    v,
    block_indices,
    block_side: int = 4,
    contiguous_inputs: bool = True,
    kv_tile: int = PACK_TILE,
):
    """Packed sparse attention plus compact normalized probability cache."""
    _, _, _, _, _, block_size = _packed_validate(q, k, v, block_indices, block_side)
    probabilities = torch.empty(
        q.shape[0], q.shape[1], q.shape[2], block_indices.shape[-1] * block_size,
        device=q.device, dtype=q.dtype,
    )
    output = _packed_launch(
        _rdd_packed_sparse_store_fwd,
        q,
        k,
        v,
        block_indices,
        block_side,
        probabilities,
        contiguous_inputs=contiguous_inputs,
        kv_tile=kv_tile,
    )
    return output, probabilities


@torch.inference_mode()
def packed_cached_probability_av(
    v,
    block_indices,
    probabilities,
    block_side: int = 4,
    contiguous_inputs: bool = True,
    kv_tile: int = PACK_TILE,
):
    """A_cached @ V_current for the packed sparse representation."""
    batch, heads, tokens, dim, grid_size, block_size = _packed_validate(
        v, v, v, block_indices, block_side
    )
    expected = (*v.shape[:3], block_indices.shape[-1] * block_size)
    if tuple(probabilities.shape) != expected:
        raise ValueError(f"packed probability cache must have shape {expected}")
    if contiguous_inputs:
        v = v.contiguous()
    block_indices, probabilities = block_indices.contiguous(), probabilities.contiguous()
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    launch_grid = (tokens // block_size, batch * heads)
    _rdd_packed_cached_av_fwd[launch_grid](
        v, block_indices, probabilities, output,
        *v.stride(), *block_indices.stride(), *probabilities.stride(), *output.stride(),
        N=tokens, GRID_SIZE=grid_size, H=heads, D=dim,
        MAX_BLOCKS=block_indices.shape[-1], BLOCK_SIDE=block_side,
        BLOCK_SIZE=block_size, Q_TILE=16, KV_TILE=kv_tile,
        BLOCK_D=triton.next_power_of_2(dim), num_warps=4, num_stages=3,
    )
    return output
