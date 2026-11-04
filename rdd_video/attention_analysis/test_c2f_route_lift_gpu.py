"""GPU parity/range test for the actual FastWan three-stage VSA grids."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rdd_video"))

from attention_analysis.fastwan_native_attention import (  # noqa: E402
    BlockRoute,
    FastWanNativeAttentionController,
)


def legacy_rows(source: BlockRoute, target: tuple[int, int, int]):
    controller = FastWanNativeAttentionController.__new__(
        FastWanNativeAttentionController
    )
    parents, children = controller._parent_children(source.block_grid, target)
    source_indices = source.indices.cpu()
    source_counts = source.counts.cpu()
    result = []
    for query, parent_query in enumerate(parents):
        selected = source_indices[
            0, 0, parent_query, : int(source_counts[0, 0, parent_query])
        ].tolist()
        values = []
        for parent_key in selected:
            values.extend(children[int(parent_key)])
        if query not in values:
            values.append(query)
        result.append(list(dict.fromkeys(values)))
    return result


def make_route(grid: tuple[int, int, int], keep: int) -> BlockRoute:
    blocks = math.prod(grid)
    rows = [
        [query] + [(query + offset + 1) % blocks for offset in range(keep - 1)]
        for query in range(blocks)
    ]
    indices = torch.tensor(rows, dtype=torch.int32, device="cuda").view(
        1, 1, blocks, keep
    )
    counts = torch.full((1, 1, blocks), keep, dtype=torch.int32, device="cuda")
    return BlockRoute(grid, indices, counts)


def check(source: BlockRoute, target: tuple[int, int, int]) -> BlockRoute:
    controller = FastWanNativeAttentionController.__new__(
        FastWanNativeAttentionController
    )
    controller.current_stage = 1
    controller.lifted_routes = {}
    controller._lift_geometry_cache = {}
    expected = legacy_rows(source, target)
    actual = controller._lift_route(0, source, target)
    torch.cuda.synchronize()
    indices = actual.indices.cpu()
    counts = actual.counts.cpu()
    blocks = math.prod(target)
    assert int(counts.min()) >= 1
    assert int(counts.max()) <= indices.shape[-1]
    for query, row in enumerate(expected):
        count = int(counts[0, 0, query])
        values = indices[0, 0, query, :count].tolist()
        assert min(values) >= 0 and max(values) < blocks, (query, values, blocks)
        assert set(values) == set(row), (query, values, row)
    print(
        "PASS",
        source.block_grid,
        "->",
        target,
        "shape=",
        tuple(actual.indices.shape),
        "count_range=",
        (int(counts.min()), int(counts.max())),
    )
    return actual


def main() -> None:
    coarse = make_route((1, 2, 4), 2)
    middle = check(coarse, (1, 4, 7))
    check(middle, (2, 7, 13))


if __name__ == "__main__":
    main()

