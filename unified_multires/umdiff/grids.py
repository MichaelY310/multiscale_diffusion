"""Explicit image/video grids and orthogonal block projections.

No video is flattened into independent training examples.  A spatial projection
leaves T intact; a spatiotemporal projection can preserve the first frame.
"""
import math
import torch
import torch.nn.functional as F


def as_video(x):
    if x.ndim not in (4, 5):
        raise ValueError('Expected BCHW or BCTHW')
    return x.unsqueeze(2) if x.ndim == 4 else x


def grid_shape(x):
    return tuple(x.shape[2:])


def block_down(x, factors):
    factors = tuple(factors)
    if len(factors) != x.ndim - 2 or any(f < 1 for f in factors):
        raise ValueError('One positive block factor per grid axis is required')
    shape = list(x.shape[:2])
    for n, f in zip(x.shape[2:], factors):
        if n % f:
            raise ValueError('Grid axes must divide exactly; padding is not implicit')
        shape.extend((n // f, f))
    return x.reshape(shape).mean(tuple(range(3, len(shape), 2)))


def block_up(x, factors):
    for axis, factor in enumerate(factors, 2):
        x = x.repeat_interleave(factor, axis)
    return x


class GridMap:
    """Average/repeat pair D,U with DU=I and UD an orthogonal projector."""
    def __init__(self, fine, coarse, first_frame=False):
        self.fine, self.coarse = tuple(fine), tuple(coarse)
        if len(self.fine) not in (2, 3) or len(self.fine) != len(self.coarse):
            raise ValueError('A grid has H,W or T,H,W axes')
        if any(n < 1 for n in (*self.fine, *self.coarse)):
            raise ValueError('Grid sizes must be positive')
        self.first_frame = bool(first_frame and len(fine) == 3 and fine[0] != coarse[0])
        a, b = list(fine), list(coarse)
        if self.first_frame:
            a[0] -= 1
            b[0] -= 1
            if b[0] < 1:
                raise ValueError('Preserved-first-frame pooling requires a nonempty tail')
        if any(x % y or x < y for x, y in zip(a, b)):
            raise ValueError(f'Nonintegral grid map {fine} -> {coarse}')
        self.factors = tuple(x // y for x, y in zip(a, b))

    def down(self, x):
        if grid_shape(x) != self.fine:
            raise ValueError(f'Expected fine grid {self.fine}, got {grid_shape(x)}')
        if self.first_frame:
            return torch.cat((block_down(x[:, :, :1], (1, *self.factors[1:])),
                              block_down(x[:, :, 1:], self.factors)), 2)
        return block_down(x, self.factors)

    def up(self, x):
        if grid_shape(x) != self.coarse:
            raise ValueError(f'Expected coarse grid {self.coarse}, got {grid_shape(x)}')
        if self.first_frame:
            return torch.cat((block_up(x[:, :, :1], (1, *self.factors[1:])),
                              block_up(x[:, :, 1:], self.factors)), 2)
        return block_up(x, self.factors)

    def project(self, x):
        return self.up(self.down(x))

    def resize_signal(self, x, mode='linear'):
        if mode=='nearest':return self.up(x)
        if grid_shape(x)!=self.coarse:raise ValueError('Signal resize expects the coarse grid')
        if self.first_frame:
            return torch.cat((resize_grid(x[:,:,:1],(1,*self.fine[1:]),mode),
                              resize_grid(x[:,:,1:],(self.fine[0]-1,*self.fine[1:]),mode)),2)
        return resize_grid(x,self.fine,mode)

    __call__ = project
    orthogonal = True

    def omega(self, x):
        """sqrt(block volume), with the separate first-frame multiplier."""
        scale = x.new_tensor(math.sqrt(math.prod(self.factors)))
        if not self.first_frame:
            return scale
        values = x.new_full((1, 1, x.shape[2], 1, 1), scale.item())
        values[:, :, 0] = math.sqrt(math.prod(self.factors[1:]))
        return values

    def down_noise(self, x):
        value = self.down(x)
        return value * self.omega(value)

    def lift_noise(self, coarse, innovation):
        """Unit covariance; normalized coarse observation is preserved exactly."""
        return self.up(coarse) / self.omega(innovation) + innovation - self.project(innovation)


def resize_grid(x, size, mode='nearest'):
    size = tuple(size)
    if len(size) != x.ndim - 2:
        raise ValueError('Resize target rank differs from input grid')
    if mode == 'linear':
        mode = 'bilinear' if x.ndim == 4 else 'trilinear'
    kwargs = {'align_corners': False} if mode in ('bilinear', 'trilinear', 'bicubic') else {}
    return F.interpolate(x, size=size, mode=mode, **kwargs)


def haar_split(x, axes=None):
    """Orthonormal separable Haar; first leaf is LL, remaining leaves details."""
    axes = tuple(range(2, x.ndim)) if axes is None else tuple(axes)
    leaves = [x]
    for axis in axes:
        out = []
        for leaf in leaves:
            if leaf.shape[axis] % 2:
                raise ValueError('Haar axes must be even')
            a = leaf.index_select(axis, torch.arange(0, leaf.shape[axis], 2, device=x.device))
            b = leaf.index_select(axis, torch.arange(1, leaf.shape[axis], 2, device=x.device))
            out.extend(((a+b)/math.sqrt(2), (a-b)/math.sqrt(2)))
        leaves = out
    return leaves


def haar_merge(leaves, axes):
    leaves = list(leaves)
    if len(leaves) != 2**len(axes):
        raise ValueError('Haar detail count differs from axes')
    for axis in reversed(axes):
        out = []
        for a, b in zip(leaves[::2], leaves[1::2]):
            shape = list(a.shape)
            shape[axis] *= 2
            x = a.new_empty(shape)
            even = [slice(None)] * x.ndim
            odd = even.copy()
            even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
            x[tuple(even)] = (a+b)/math.sqrt(2)
            x[tuple(odd)] = (a-b)/math.sqrt(2)
            out.append(x)
        leaves = out
    return leaves[0]
