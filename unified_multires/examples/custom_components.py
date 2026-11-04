"""Example: register a new band path without modifying the trainer or solver."""
import torch
from umdiff.registry import register
from umdiff.paths import BandPath

@register('path','smooth_detail')
class SmoothDetail(BandPath):
    def coefficients(self,t):
        # Low signal follows t; detail follows smoothstep. Noise follows 1-t.
        high=t*t*(3-2*t);dhigh=6*t*(1-t)
        one=torch.ones_like(t)
        return t,high,1-t,one,dhigh,-one
