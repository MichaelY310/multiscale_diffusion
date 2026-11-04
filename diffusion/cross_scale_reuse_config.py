"""Shared, depth-normalized IMAGE/VIDEO ablations. No checkpoint modification."""
from fractions import Fraction


FAMILIES = {
    'high_heavy': (Fraction(1, 3), Fraction(5, 6)),
    'balanced': (Fraction(2, 3), Fraction(2, 3)),
    'balanced_plus': (Fraction(2, 3), Fraction(3, 4)),
}


def layer_masks(family, depth):
    """Zero-based; keep first/last TWO blocks live at every resolution.

    Middle/fine masks have different offsets, but may overlap at high density.
    Thus a fine cache entry can itself be a reused middle update. Never describe
    these aggressive configurations as disjoint or fresh-middle-only.
    """
    if family == 'dense':
        return {0: (), 1: (), 2: ()}
    ratios = FAMILIES[family]
    pool = list(range(2, depth - 2))
    masks = {0: ()}
    for stage, ratio in enumerate(ratios, 1):
        count = int(depth * ratio + Fraction(1, 2))
        if not 1 < count <= len(pool):
            raise ValueError((family, depth, count))
        positions = [round(i * (len(pool) - 1) / (count - 1)) for i in range(count)]
        shift = 0 if stage == 1 else 2
        chosen = tuple(sorted(pool[(p + shift) % len(pool)] for p in positions))
        assert len(set(chosen)) == count
        masks[stage] = chosen
    return masks


def description(family, depth):
    masks = layer_masks(family, depth)
    return dict(family=family, depth=depth, masks_zero_based=masks,
                middle_reuse=len(masks[1]), fine_reuse=len(masks[2]),
                overlapping_layers=sorted(set(masks[1]) & set(masks[2])),
                first_last_two_live=True, source='previous-stage block delta',
                interpolation='nearest', scale=1., added_parameters=0)
