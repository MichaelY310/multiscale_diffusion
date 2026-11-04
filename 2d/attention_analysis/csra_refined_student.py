"""Separate opt-in candidate; old student formats/checkpoints are not changed."""
import math
from csra_neighbor_student import NeighborFrontStudent
from attention_analysis.csra_memory_refiner import CoarseMemoryRefiner

FORMAT = 'csra_front1_skip6_current_coarse_memory_v1'


class RefinedFrontStudent(NeighborFrontStudent):
    def __init__(self, *args, refine_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        if self.neighbor_enabled or self.coarse_source != 'aligned':
            raise ValueError('This pilot keeps neighbors off and the original real coarse source')
        self.memory_refiner = CoarseMemoryRefiner(self.proj.out_features, self.heads, refine_depth)

    def initialize_from_neighbor_parent(self, state):
        expected = self.state_dict()
        old_keys = {k for k in expected if not k.startswith('memory_refiner.')}
        if set(state) != old_keys or any(state[k].shape != expected[k].shape for k in old_keys):
            raise ValueError('Parent keys/shapes must match exactly; not a resume loader')
        merged = {k: v for k, v in expected.items() if k.startswith('memory_refiner.')}
        merged.update(state)
        self.load_state_dict(merged, strict=True)

    def lift(self, h0, c, source):
        fine, coarse = math.isqrt(h0.shape[1]), math.isqrt(source.shape[1])
        if fine ** 2 != h0.shape[1] or coarse ** 2 != source.shape[1] or fine != 2 * coarse:
            raise ValueError('2D pilot requires square, factor2 nested token grids')
        memory = self.memory_refiner(source, h0, c, (1, coarse, coarse), (1, fine, fine))
        return super().lift(h0, c, memory)
