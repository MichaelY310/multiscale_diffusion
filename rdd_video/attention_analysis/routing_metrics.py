"""Utilities for measuring cross-resolution attention routing.

The important unit is a *root tile*.  For the paper Wan setting, a fine-grid
root is 4x4x4 DiT tokens, a middle-grid root is 2x2x2 tokens, and a coarse-grid
root is one token.  After end padding all three therefore describe the same
physical spatiotemporal cells.

This module deliberately computes exact dense attention only for diagnostic
shapes.  It is not the production sparse kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RootLayout:
    grid: tuple[int, int, int]
    tile: tuple[int, int, int]
    root_grid: tuple[int, int, int]
    token_to_root: torch.Tensor
    root_counts: torch.Tensor

    @property
    def num_roots(self) -> int:
        return math.prod(self.root_grid)


def make_root_layout(
    grid: Sequence[int], tile: Sequence[int], device: torch.device | str
) -> RootLayout:
    grid = tuple(int(x) for x in grid)
    tile = tuple(int(x) for x in tile)
    if len(grid) != 3 or len(tile) != 3 or any(x <= 0 for x in grid + tile):
        raise ValueError(f"invalid grid/tile: {grid=}, {tile=}")
    root_grid = tuple(math.ceil(g / b) for g, b in zip(grid, tile))
    t, h, w = torch.meshgrid(
        torch.arange(grid[0], device=device),
        torch.arange(grid[1], device=device),
        torch.arange(grid[2], device=device),
        indexing="ij",
    )
    token_to_root = (
        (t // tile[0]) * (root_grid[1] * root_grid[2])
        + (h // tile[1]) * root_grid[2]
        + (w // tile[2])
    ).reshape(-1)
    counts = torch.bincount(token_to_root, minlength=math.prod(root_grid))
    return RootLayout(grid, tile, root_grid, token_to_root, counts)


def _as_shd(x: torch.Tensor) -> torch.Tensor:
    """Accept [B,S,H,D] or [S,H,D], return [S,H,D]."""
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError("the diagnostic currently expects batch size 1")
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"expected [S,H,D], got {tuple(x.shape)}")
    return x


@torch.no_grad()
def pooled_route_distribution(
    q: torch.Tensor, k: torch.Tensor, layout: RootLayout
) -> torch.Tensor:
    """Cheap VSA-like root routing distribution, shape [heads,R,R]."""
    q, k = _as_shd(q).float(), _as_shd(k).float()
    if q.shape != k.shape or q.shape[0] != layout.token_to_root.numel():
        raise ValueError("Q/K shape and root layout disagree")
    roots, heads, dim = layout.num_roots, q.shape[1], q.shape[2]
    q_root = torch.zeros(roots, heads, dim, device=q.device, dtype=q.dtype)
    k_root = torch.zeros_like(q_root)
    q_root.index_add_(0, layout.token_to_root, q)
    k_root.index_add_(0, layout.token_to_root, k)
    denom = layout.root_counts.clamp_min(1).to(q.dtype)[:, None, None]
    q_root, k_root = q_root / denom, k_root / denom
    score = torch.einsum("rhd,shd->hrs", q_root, k_root) / math.sqrt(dim)
    return score.softmax(dim=-1)


@torch.no_grad()
def exact_root_attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    layout: RootLayout,
    query_chunk: int = 128,
) -> torch.Tensor:
    """Aggregate exact token attention into [heads, query-root, key-root]."""
    q, k = _as_shd(q).float(), _as_shd(k).float()
    seq, heads, dim = q.shape
    if k.shape != q.shape or seq != layout.token_to_root.numel():
        raise ValueError("Q/K shape and root layout disagree")
    key_membership = F.one_hot(
        layout.token_to_root, num_classes=layout.num_roots
    ).to(q.dtype)
    out = torch.zeros(
        heads, layout.num_roots, layout.num_roots, device=q.device, dtype=q.dtype
    )
    for head in range(heads):
        kh = k[:, head]
        for start in range(0, seq, query_chunk):
            stop = min(start + query_chunk, seq)
            logits = q[start:stop, head] @ kh.T / math.sqrt(dim)
            root_mass = logits.softmax(dim=-1) @ key_membership
            out[head].index_add_(
                0, layout.token_to_root[start:stop], root_mass
            )
    counts = layout.root_counts.clamp_min(1).to(out.dtype)
    return out / counts[None, :, None]


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0) / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = _normalize_rows(p), _normalize_rows(q)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
    kl_qm = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def local_root_mask(
    root_grid: Sequence[int], radius: int, device: torch.device | str
) -> torch.Tensor:
    """Chebyshev 3-D neighborhood mask [R,R]."""
    root_grid = tuple(int(x) for x in root_grid)
    coords = torch.stack(
        torch.meshgrid(
            *(torch.arange(x, device=device) for x in root_grid), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 3)
    return (coords[:, None] - coords[None]).abs().amax(dim=-1) <= radius


@torch.no_grad()
def compare_route_distributions(
    source: torch.Tensor,
    target: torch.Tensor,
    densities: Iterable[float] = (0.1, 0.2, 0.3),
    root_grid: Sequence[int] | None = None,
    local_radius: int = 1,
) -> dict:
    """Compare a reused source route against target fine-stage attention."""
    source, target = _normalize_rows(source.float()), _normalize_rows(target.float())
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(f"route shapes differ: {source.shape} vs {target.shape}")
    heads, roots, _ = source.shape
    cos = F.cosine_similarity(source, target, dim=-1)
    result: dict[str, object] = {
        "heads": heads,
        "roots": roots,
        "cosine_mean": float(cos.mean()),
        "cosine_p10": float(torch.quantile(cos.flatten(), 0.10)),
        "js_mean": float(_js_divergence(source, target).mean()),
    }
    local = None
    if root_grid is not None:
        local = local_root_mask(root_grid, local_radius, source.device)[None]
    density_rows = []
    for density in densities:
        keep = max(1, min(roots, int(math.ceil(float(density) * roots))))
        src_idx = source.topk(keep, dim=-1).indices
        tgt_idx = target.topk(keep, dim=-1).indices
        src_mask = torch.zeros_like(source, dtype=torch.bool).scatter_(-1, src_idx, True)
        tgt_mask = torch.zeros_like(target, dtype=torch.bool).scatter_(-1, tgt_idx, True)
        coverage = (target * src_mask).sum(dim=-1)
        oracle = (target * tgt_mask).sum(dim=-1)
        overlap = (src_mask & tgt_mask).sum(dim=-1).float() / keep
        row = {
            "requested_density": float(density),
            "topk": keep,
            "route_density": keep / roots,
            "target_mass_coverage": float(coverage.mean()),
            "oracle_mass_coverage": float(oracle.mean()),
            "topk_recall": float(overlap.mean()),
        }
        if local is not None:
            hybrid = src_mask | local
            row["hybrid_density"] = float(hybrid.float().mean())
            row["hybrid_target_mass_coverage"] = float((target * hybrid).sum(-1).mean())
        density_rows.append(row)
    result["density_sweep"] = density_rows
    return result


@torch.no_grad()
def masked_attention_output_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: RootLayout,
    source_route: torch.Tensor,
    density: float,
    local_radius: int | None = 1,
    max_heads: int = 4,
    max_query_tokens: int = 256,
) -> dict[str, float]:
    """Sampled dense-vs-routed attention output error on the fine Q/K/V."""
    q, k, v = _as_shd(q).float(), _as_shd(k).float(), _as_shd(v).float()
    seq, heads, dim = q.shape
    used_heads = min(heads, max_heads)
    roots = layout.num_roots
    keep = max(1, min(roots, int(math.ceil(float(density) * roots))))
    route = _normalize_rows(source_route.float())[:used_heads]
    idx = route.topk(keep, dim=-1).indices
    allowed_root = torch.zeros_like(route, dtype=torch.bool).scatter_(-1, idx, True)
    if local_radius is not None:
        allowed_root |= local_root_mask(layout.root_grid, local_radius, q.device)[None]
    query_ids = torch.linspace(
        0, seq - 1, steps=min(seq, max_query_tokens), device=q.device
    ).round().long().unique()
    rel_l2, cosines = [], []
    for head in range(used_heads):
        logits = q[query_ids, head] @ k[:, head].T / math.sqrt(dim)
        dense = logits.softmax(-1) @ v[:, head]
        q_roots = layout.token_to_root[query_ids]
        token_allowed = allowed_root[head, q_roots][:, layout.token_to_root]
        sparse_logits = logits.masked_fill(~token_allowed, -torch.inf)
        sparse = sparse_logits.softmax(-1) @ v[:, head]
        rel_l2.append((sparse - dense).norm() / dense.norm().clamp_min(1e-12))
        cosines.append(F.cosine_similarity(sparse, dense, dim=-1).mean())
    return {
        "sampled_heads": used_heads,
        "sampled_query_tokens": int(query_ids.numel()),
        "effective_root_density": float(allowed_root.float().mean()),
        "relative_l2": float(torch.stack(rel_l2).mean()),
        "output_cosine": float(torch.stack(cosines).mean()),
    }


def _pool_tokens_to_roots(x: torch.Tensor, layout: RootLayout) -> torch.Tensor:
    """Average [S,H,D] values into [H,R,D] root summaries."""
    x = _as_shd(x).float()
    roots, heads, dim = layout.num_roots, x.shape[1], x.shape[2]
    out = torch.zeros(roots, heads, dim, device=x.device, dtype=x.dtype)
    out.index_add_(0, layout.token_to_root, x)
    out /= layout.root_counts.clamp_min(1).to(x.dtype)[:, None, None]
    return out.permute(1, 0, 2).contiguous()


@torch.no_grad()
def routed_attention_with_summary_tail_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: RootLayout,
    source_route: torch.Tensor,
    density: float,
    summary_k: torch.Tensor,
    summary_v: torch.Tensor,
    summary_layout: RootLayout,
    max_heads: int = 4,
    max_query_tokens: int = 256,
) -> dict[str, float]:
    """Exact selected fine blocks plus one K/V summary for every skipped root.

    ``summary_k/v`` may come from the previous RDD stage (the proposed parent
    memory) or from pooled target-scale K/V (an optimistic upper bound).  The
    summary logit includes log(number of represented fine keys), which is the
    exact log-sum-exp correction when all keys in that root equal the summary.
    """
    q, k, v = _as_shd(q).float(), _as_shd(k).float(), _as_shd(v).float()
    if layout.root_grid != summary_layout.root_grid:
        raise ValueError("fine and summary layouts must share the same root grid")
    seq, heads, dim = q.shape
    used_heads = min(heads, max_heads)
    roots = layout.num_roots
    keep = max(1, min(roots, int(math.ceil(float(density) * roots))))
    route = _normalize_rows(source_route.float())[:used_heads]
    idx = route.topk(keep, dim=-1).indices
    selected_roots = torch.zeros_like(route, dtype=torch.bool).scatter_(-1, idx, True)
    sk = _pool_tokens_to_roots(summary_k, summary_layout)[:used_heads]
    sv = _pool_tokens_to_roots(summary_v, summary_layout)[:used_heads]
    query_ids = torch.linspace(
        0, seq - 1, steps=min(seq, max_query_tokens), device=q.device
    ).round().long().unique()
    represented = layout.root_counts.clamp_min(1).to(q.dtype).log()
    rel_l2, cosines = [], []
    for head in range(used_heads):
        logits = q[query_ids, head] @ k[:, head].T / math.sqrt(dim)
        dense = logits.softmax(-1) @ v[:, head]
        q_roots = layout.token_to_root[query_ids]
        root_allowed = selected_roots[head, q_roots]
        token_allowed = root_allowed[:, layout.token_to_root]
        exact_logits = logits.masked_fill(~token_allowed, -torch.inf)
        tail_logits = q[query_ids, head] @ sk[head].T / math.sqrt(dim)
        tail_logits = tail_logits + represented[None]
        tail_logits = tail_logits.masked_fill(root_allowed, -torch.inf)
        combined = torch.cat((exact_logits, tail_logits), dim=-1).softmax(-1)
        sparse = combined[:, :seq] @ v[:, head] + combined[:, seq:] @ sv[head]
        rel_l2.append((sparse - dense).norm() / dense.norm().clamp_min(1e-12))
        cosines.append(F.cosine_similarity(sparse, dense, dim=-1).mean())
    return {
        "sampled_heads": used_heads,
        "sampled_query_tokens": int(query_ids.numel()),
        "exact_root_density": float(selected_roots.float().mean()),
        "relative_l2": float(torch.stack(rel_l2).mean()),
        "output_cosine": float(torch.stack(cosines).mean()),
    }
