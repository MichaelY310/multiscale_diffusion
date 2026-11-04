"""Numerical helpers for a read-only cross-scale information probe, not a sampler."""
import torch


MODES = ('fine_only', 'coarse_aligned', 'coarse_roll', 'coarse_wrong', 'coarse_mean')


def select_records(manifest):
    train = [r for r in manifest['files'] if r['split'] == 'train'][:8]
    val = [r for r in manifest['files'] if r['split'] == 'validation'][:4]
    if len(train) != 8 or len(val) != 4:
        raise ValueError('Need eight training and four validation batches per shard')
    records = [dict(r, role='fit' if i < 6 else 'calibration') for i, r in enumerate(train)]
    records += [dict(r, role='validation') for r in val]
    if len({r['seed'] for r in records}) != len(records):
        raise ValueError('Duplicate seed across probe splits')
    return records


def coarse_lift(source, mode):
    b, n, h = source.shape
    g = int(n ** .5)
    if g * g != n or b < 1 or (mode == 'coarse_wrong' and b < 2):
        raise ValueError('Need square grid; wrong-image control needs at least two images')
    grid = source.reshape(b, g, g, h)
    if mode == 'coarse_roll':
        grid = torch.roll(grid, (1, 1), (1, 2))
    elif mode == 'coarse_wrong':
        grid = torch.roll(grid, 1, 0)
    elif mode == 'coarse_mean':
        grid = grid.mean((1, 2), keepdim=True).expand_as(grid)
    elif mode != 'coarse_aligned':
        raise ValueError(mode)
    return grid.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(b, 4*n, h)


def features(h0, condition, source, mode):
    parts = [h0, condition[:, None].expand(-1, h0.shape[1], -1)]
    if mode != 'fine_only':
        parts.append(coarse_lift(source, mode))
    return torch.cat(parts, dim=-1).flatten(0, 1).float()


def split_error(error):
    b, n, h = error.shape
    g = int(n ** .5)
    if g*g != n or g % 2:
        raise ValueError('Need an even square fine grid')
    grouped = error.reshape(b, g//2, 2, g//2, 2, h)
    low = grouped.mean((2, 4), keepdim=True).expand_as(grouped).reshape_as(error)
    return low, error-low


class Moments:
    """Centered FP32 products + FP64 Chan merges avoid subtractive cancellation."""
    def __init__(self, dimension, output_dimension, device):
        self.n = 0
        self.mx = torch.zeros(dimension, device=device, dtype=torch.float64)
        self.my = torch.zeros(output_dimension, device=device, dtype=torch.float64)
        self.cxx = torch.zeros(dimension, dimension, device=device, dtype=torch.float64)
        self.cxy = torch.zeros(dimension, output_dimension, device=device, dtype=torch.float64)

    def add(self, x, y):
        xm, ym = x.double().mean(0), y.double().mean(0)
        xc, yc = (x.double()-xm).float(), (y.double()-ym).float()
        total = self.n+len(x)
        dx, dy = xm-self.mx, ym-self.my
        weight = self.n*len(x)/total
        self.cxx += (xc.T@xc).double() + weight*dx[:,None]*dx[None]
        self.cxy += (xc.T@yc).double() + weight*dx[:,None]*dy[None]
        self.mx += dx*(len(x)/total)
        self.my += dy*(len(x)/total)
        self.n = total

    def cpu(self):
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(self).items()}


def fit_ridge(stats, ridge):
    """Fit centered, standardized ridge in FP64; intercept not penalized.

    ridge is applied to E[X_std.T X_std], not the unnormalized sample sum.
    Hyperparameters must be selected on calibration, never reported validation.
    """
    n = stats['n']
    mx, my = stats['mx'], stats['my']
    cov = stats['cxx']/n
    cov = (cov + cov.T) * .5
    scale = cov.diag().clamp_min(1e-8).sqrt()
    a = cov / scale[:, None] / scale[None]
    a.diagonal().add_(ridge)
    rhs = (stats['cxy']/n) / scale[:, None]
    factor = torch.linalg.cholesky(a)
    weight = torch.cholesky_solve(rhs, factor)
    residual = float((a@weight-rhs).norm()/rhs.norm().clamp_min(1e-12))
    if not torch.isfinite(weight).all() or residual > 1e-7:
        raise RuntimeError(f'Unstable ridge solution: {residual}')
    return {'mean': mx, 'scale': scale, 'output_mean': my, 'weight': weight,
            'ridge': ridge, 'solve_relative_residual': residual}


def predict(x, fit):
    return ((x-fit['mean'])/fit['scale']) @ fit['weight'] + fit['output_mean']
