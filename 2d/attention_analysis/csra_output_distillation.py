"""Differentiable final-output training path; old inference remains unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from csra_runtime import predict_adapter
from train_rdd_csra_adapter import upsample_tokens


class CSRAOutputStudent(nn.Module):
    def __init__(self, teacher, adapter, checkpoint_blocks=True):
        super().__init__()
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("The backbone must be frozen")
        # Deliberately not a registered child: DDP synchronizes adapters only.
        # The caller loads the same frozen checkpoint on every rank.
        object.__setattr__(self, "teacher", teacher.eval())
        self.adapter = adapter
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, x, t, labels, coarse):
        model = self.teacher
        if set(coarse) != set(self.adapter.layers):
            raise ValueError("Cache mask must match the adapter exactly")
        gh, gw = x.shape[-2] // model.patch_size, x.shape[-1] // model.patch_size
        if gh != gw:
            raise ValueError("This ImageNet prototype expects square token grids")
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            hidden = model.x_embedder(x)
            cos, sin = model.get_rope(gh, gw, x.device)
            condition = model.t_embedder(t) + model.y_embedder(labels, False)
            for layer, block in enumerate(model.blocks):
                if layer in coarse:
                    source = coarse[layer]
                    if source.shape[1] != gh * gw:
                        source = upsample_tokens(source, gh)
                    delta = predict_adapter(self.adapter.adapters[str(layer)],
                                            source, hidden, t, "fp32", condition=condition)
                    hidden = hidden + delta.to(hidden.dtype)
                elif self.checkpoint_blocks and torch.is_grad_enabled() and hidden.requires_grad:
                    # Do NOT detach hidden here. Earlier adapters need the
                    # Jacobian of every remaining frozen block and output head.
                    hidden = checkpoint(block, hidden, condition, cos, sin,
                                        use_reentrant=False)
                else:
                    hidden = block(hidden, condition, cos, sin)
            return model.unpatchify(model.final_layer(hidden, condition), gh, gw)


def prediction_loss(prediction, target, variance_weight=0.05):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 8:
        raise ValueError("Expected matching [B,8,H,W] main+learned-variance outputs")
    if variance_weight < 0:
        raise ValueError("Negative variance weight")
    prediction, target = prediction.float(), target.float()
    main = (prediction[:, :4] - target[:, :4]).square().mean()
    variance = (prediction[:, 4:] - target[:, 4:]).square().mean()
    return main + variance_weight * variance, {"main_mse": main.detach(),
                                              "variance_mse": variance.detach()}
