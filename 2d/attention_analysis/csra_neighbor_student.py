"""Separate opt-in architecture: old front1 reader + fine spatial correction.

Not an automatic old-checkpoint upgrade. Parent loading is explicit and strict;
new checkpoints must use FORMAT and record neighbor_enabled/rank and protocol.
The two existing coarse/fine backbones and skipped block range do not change.
"""
import sys
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

from csra_front_keep_student import FrontKeepStudent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rdd_video'))
from attention_analysis.csra_fine_neighbor import FineSpatialNeighbor

FORMAT = 'csra_front1_skip6_spatial_neighbor_v1'


class NeighborFrontStudent(FrontKeepStudent):
    def __init__(self, *args, neighbor_enabled=True, neighbor_rank=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbor_enabled, self.neighbor_rank = neighbor_enabled, neighbor_rank
        self.neighbor = FineSpatialNeighbor(self.proj.out_features, neighbor_rank)
        self.neighbor.requires_grad_(neighbor_enabled)

    def initialize_from_front_parent(self, old_state):
        """Explicit parent migration only; do not use this to resume new runs."""
        expected = self.state_dict()
        parent_keys = {k for k in expected if not k.startswith('neighbor.')}
        if set(old_state) != parent_keys:
            raise ValueError('Parent must contain exactly the original front1 student keys')
        if any(old_state[k].shape != expected[k].shape for k in parent_keys):
            raise ValueError('Parent architecture shape mismatch')
        merged = {k: v for k, v in expected.items() if k.startswith('neighbor.')}
        merged.update(old_state)
        self.load_state_dict(merged, strict=True)

    def forward(self, batch, return_hidden=False):
        x, t, y = batch['x'], batch['t'], batch['labels']
        m = self.backbone
        gh, gw = x.shape[-2] // m.patch_size, x.shape[-1] // m.patch_size
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            h = m.x_embedder(x)
            c = m.t_embedder(t) + m.y_embedder(y, False)
            cos, sin = m.get_rope(gh, gw, x.device)
            def execute(block, value):
                if self.checkpoint_blocks and torch.is_grad_enabled():
                    return checkpoint(block, value, c, cos, sin, use_reentrant=False)
                return block(value, c, cos, sin)
            for block in m.blocks[:self.front_keep]:
                h = execute(block, h)
            source = None if self.coarse_source == 'zero' else batch['coarse'][self.source_layer]
            h = self.lift(h, c, source).to(h.dtype)
            if self.neighbor_enabled:
                h = self.neighbor(h, (1, gh, gw))
            bridge_hidden = h
            for block in m.blocks[self.front_keep + self.prefix_depth:]:
                h = execute(block, h)
            out = m.unpatchify(m.final_layer(h, c), gh, gw)
            return (out, bridge_hidden) if return_hidden else out
