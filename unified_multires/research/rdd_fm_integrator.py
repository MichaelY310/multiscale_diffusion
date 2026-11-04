"""Research-only RDD-FM quadrature in an orthonormal two-band basis.

No production registration or checkpoint changes. Time runs noise -> data.
The caller must reset multistep history at each resolution/stochastic boundary.
These are specializations of integrating-factor quadrature, not novelty claims.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Stage:
    start: float
    end: float
    drift: bool = True

    def __post_init__(self):
        if not 0 <= self.start < self.end <= 1:
            raise ValueError('Invalid stage interval')
        if self.drift and self.start == 0:
            raise ValueError('Drifting high band is unidentifiable at t=0')

    def coefficients(self, t):
        if not self.start - 1e-12 <= t <= self.end + 1e-12:
            raise ValueError('Time outside this stage')
        if self.drift:
            length = self.end - self.start
            a = np.array([t, t * (t - self.start) / length])
            da = np.array([1., (2 * t - self.start) / length])
        else:
            a, da = np.full(2, t), np.ones(2)
        sigma = 1. - t
        return a, da, sigma, a + sigma * da

    def clean(self, x, velocity, t):
        _, _, sigma, identifiable = self.coefficients(t)
        if np.min(np.abs(identifiable)) < 1e-12:
            raise ValueError('Clean inverse is singular')
        return (x + sigma * velocity) / identifiable

    def weights(self, t, s):
        """Exact zeroth/first time moments of sigma(s) B(u)/sigma(u)^2.

        The s=1 expressions are continuous limits. Interior convergence order
        does not by itself establish a uniform endpoint error bound.
        """
        if not self.start <= t < s <= self.end + 1e-12:
            raise ValueError('A step must remain inside one stage')
        at, _, q, _ = self.coefficients(t)
        ass, _, r, _ = self.coefficients(s)
        h = s - t
        ratio = r / q
        w0 = ass - ratio * at
        jlow = h if r == 0 else h + r * math.log1p(-h / q)
        jhigh = (((1. - self.start) * jlow - r * h * h / 2)
                 / (self.end - self.start)) if self.drift else jlow
        return ratio, w0, np.array([jlow, jhigh])

    def step(self, x, velocity, t, s, previous=None):
        """One call per step; previous=(previous_time, previous_clean).

        Without history: first order generally, exact for constant clean
        posterior along the step. With smooth history: second order inside
        a stage, under the usual bounded step-ratio/stability assumptions.
        """
        clean = self.clean(x, velocity, t)
        ratio, w0, w1 = self.weights(t, s)
        result = ratio * x + w0 * clean
        if previous is not None:
            tp, mp = previous
            if not self.start <= tp < t:
                raise ValueError('History must be earlier in this same stage')
            result = result + w1 * (clean - mp) / (t - tp)
        return result, clean
