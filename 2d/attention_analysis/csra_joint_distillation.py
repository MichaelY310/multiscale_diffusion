"""Opt-in joint training; no changes to old frozen-backbone behavior.

Current student makes its OWN coarse residuals with gradients. Both source
inputs and fine inputs remain from teacher trajectories (not online rollouts).
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from csra_runtime import predict_adapter
from train_rdd_csra_adapter import upsample_tokens


class JointCSRAStudent(nn.Module):
    def __init__(self, backbone, adapter, train_backbone=True, checkpoint_blocks=True):
        super().__init__()
        self.backbone = backbone.float().eval().requires_grad_(train_backbone)
        self.adapter = adapter.float().requires_grad_(True)
        self.checkpoint_blocks = checkpoint_blocks

    def stage(self, x, t, labels, cache=None, capture=False):
        m = self.backbone
        gh, gw = x.shape[-2] // m.patch_size, x.shape[-1] // m.patch_size
        if gh != gw:
            raise ValueError("Image pilot requires square token grids")
        if cache is not None and set(cache) != set(self.adapter.layers):
            raise ValueError("Cache mask differs from adapter mask")
        captured = {}
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            h = m.x_embedder(x)
            cos, sin = m.get_rope(gh, gw, x.device)
            c = m.t_embedder(t) + m.y_embedder(labels, False)
            for l, block in enumerate(m.blocks):
                before = h
                if cache is not None and l in cache:
                    native = getattr(self.adapter.adapters[str(l)], "requires_native_coarse", False)
                    if getattr(self, "coarse_source", "aligned") == "zero":
                        # Trainable no-memory control. Do not read source values.
                        source = h.new_zeros(h.shape[0], (gh//2)**2, h.shape[-1]) if native else torch.zeros_like(h)
                    else:
                        source = cache[l] if native or cache[l].shape[1] == gh*gw else upsample_tokens(cache[l], gh)
                    delta = predict_adapter(self.adapter.adapters[str(l)], source, h, t,
                                            "fp32", condition=c)
                    h = h + delta.to(h.dtype)
                elif self.checkpoint_blocks and torch.is_grad_enabled() and h.requires_grad:
                    h = checkpoint(block, h, c, cos, sin, use_reentrant=False)
                else:
                    h = block(h, c, cos, sin)
                if capture and l in self.adapter.layers:
                    # NO detach: fine loss can train the coarse branch too.
                    captured[l] = h - before
            output = m.unpatchify(m.final_layer(h, c), gh, gw)
        return output, captured

    def forward(self, batch):
        middle, cache = self.stage(batch["source16_x"], batch["source16_t"],
                                   batch["labels"], capture=True)
        fine, _ = self.stage(batch["x"], batch["t"], batch["labels"], cache=cache)
        coarse, _ = self.stage(batch["source8_x"], batch["source8_t"], batch["labels"])
        return fine, middle, coarse


class PreservedCoarseStudent(JointCSRAStudent):
    """Train the executed fine backbone; ORIGINAL coarse weights stay separate.

    Training consumes exact original-teacher caches, never caches produced by
    the updated fine backbone. Inference MUST use the unchanged original
    teacher for 8/16 stages. This is a two-weight-set stage-specialized model,
    not a newly trained single unified model. Keep both weights resident when
    measuring latency, and account separately for the extra memory.
    """
    def __init__(self, backbone, adapter, train_backbone=True, checkpoint_blocks=True, coarse_source="aligned"):
        if coarse_source not in ("aligned", "zero"):
            raise ValueError("Unsupported preserved-coarse source")
        super().__init__(backbone, adapter, train_backbone, checkpoint_blocks)
        self.coarse_source = coarse_source
        # These fine blocks are always replaced. Do not claim to train unused
        # weights or give DDP parameters whose gradients cannot exist.
        for layer in self.adapter.layers:
            self.backbone.blocks[layer].requires_grad_(False)

    def forward(self, batch):
        return self.stage(batch["x"], batch["t"], batch["labels"],
                          cache=batch["coarse"])[0]
