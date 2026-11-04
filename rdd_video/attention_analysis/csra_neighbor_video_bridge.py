"""Untrained VIDEO cost analogue of a front1 bridge with fine neighbors.

Does not replace/remove the existing coarse global reader or native VSA gates.
"""
from attention_analysis.csra_fine_neighbor import FineSpatialNeighbor, add_neighbor_after_bridge
from attention_analysis.csra_prefix_video_bridge import VideoConditionalPrefixBridge


class VideoNeighborPrefixBridge(VideoConditionalPrefixBridge):
    def __init__(self, hidden, heads, rank=128):
        super().__init__(hidden, heads)
        self.neighbor = FineSpatialNeighbor(hidden, rank)
        self.neighbor_enabled = True

    def forward(self, h0, c, memory, source_grid, target_grid):
        hidden = super().forward(h0, c, memory, source_grid, target_grid)
        if self.neighbor_enabled:
            hidden = add_neighbor_after_bridge(hidden, self.neighbor, target_grid)
        return hidden
