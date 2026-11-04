"""UNTRAINED video cost analogue; original native VSA/gates remain unchanged."""
from attention_analysis.csra_memory_refiner import CoarseMemoryRefiner
from attention_analysis.csra_prefix_video_bridge import VideoConditionalPrefixBridge


class VideoRefinedPrefixBridge(VideoConditionalPrefixBridge):
    def __init__(self, hidden, heads, depth=2):
        super().__init__(hidden, heads)
        self.memory_refiner = CoarseMemoryRefiner(hidden, heads, depth)
        self.refiner_enabled = True

    def forward(self, h0, c, memory, source_grid, target_grid):
        if self.refiner_enabled:
            memory = self.memory_refiner(memory, h0, c, source_grid, target_grid)
        return super().forward(h0, c, memory, source_grid, target_grid)
