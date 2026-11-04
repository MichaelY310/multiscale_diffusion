"""Aggressive image masks; the shared residual-reuse implementation is unchanged.

Keep original400K and prior200-step experiments intact. No new learned modules.
Both masks keep first/last blocks live and replace only fine-stage blocks.
Video additionally tests coarse->middle->fine cascade; do NOT pretend that
the image dataset has middle-stage trajectories needed to train that variant.
"""
from direct_reuse_fullmodel import DirectReuseStudent, DirectReuseRollout, lift


def spread(n, depth=24):
    values=tuple(round(1+i*(depth-3)/(n-1)) for i in range(n))
    if len(set(values))!=n or values[0]!=1 or values[-1]!=depth-2:
        raise ValueError('Invalid evenly-spaced reuse mask')
    return values


MASKS={'dense':(), 'reuse12':spread(12), 'reuse18':spread(18)}
