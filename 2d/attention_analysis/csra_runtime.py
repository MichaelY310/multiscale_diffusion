"""Explicit precision policy for a CSRA adapter, independent of backbone AMP."""

from contextlib import nullcontext

import torch


def predict_adapter(adapter, coarse, hidden, timestep, precision="legacy_autocast", condition=None):
    """Keep the historical path by default; FP32 matches adapter training.

    Disabling autocast covers the adapter only. The caller still converts its
    residual to the backbone dtype before addition; the frozen DiT is unchanged.
    """
    if precision not in {"legacy_autocast", "fp32"}:
        raise ValueError(f"Unknown adapter precision: {precision}")
    context = (
        torch.autocast(device_type=hidden.device.type, enabled=False)
        if precision == "fp32" else nullcontext()
    )
    with context:
        if getattr(adapter, "requires_condition", False):
            return adapter(coarse, hidden, timestep, condition=condition)
        return adapter(coarse, hidden, timestep)
