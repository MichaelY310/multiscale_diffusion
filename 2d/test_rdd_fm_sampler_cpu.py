"""CPU contracts for legacy RDD-FM-v2 inference math."""

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from diffusion.flow_matching_with_drift import (  # noqa: E402
    FlowMatchingWithDrift,
    noise_filling_transition_fm,
)


def assert_close(actual, expected, name, atol=2e-5, rtol=2e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        error = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max error {error}")


def test_clean_inverse():
    torch.manual_seed(0)
    for c, start, end in [(None, 0.0, 0.3), (1, 0.3, 0.6), (0, 0.6, 1.0)]:
        fm = FlowMatchingWithDrift(c=c, sample_scale=2, s_k=start, s_k_plus_1=end)
        x1 = torch.randn(4, 4, 8, 8)
        eps = torch.randn_like(x1)
        t = torch.linspace(start + 0.03, end - 0.03, x1.shape[0])
        drift = None if c is None else fm.compute_drift(x1)
        xt = fm.q_sample(x1, t, noise=eps, d=drift)
        velocity = fm.compute_velocity_target(x1, eps, t, d=drift)
        recovered = fm.predict_x1_from_velocity(xt, velocity, t)
        assert_close(recovered, x1, f"clean inverse stage {start}-{end}")


def test_noise_filling_projection():
    torch.manual_seed(1)
    boundary = 0.3
    x1 = torch.randn(3, 4, 8, 8)
    eps = torch.randn_like(x1)
    state = boundary * x1 + (1.0 - boundary) * eps
    lifted = noise_filling_transition_fm(state, x1, boundary, target_size=16)

    signal = x1.repeat_interleave(2, -2).repeat_interleave(2, -1)
    combined_eps = (lifted - boundary * signal) / (1.0 - boundary)
    projected = combined_eps.view(3, 4, 8, 2, 8, 2).mean(dim=(3, 5))
    # The low-frequency innovation is an isometric (1/scale) lift.
    assert_close(projected, eps / 2.0, "noise-filling projected innovation")


class OracleVelocity(torch.nn.Module):
    def __init__(self, fm, x1, eps):
        super().__init__()
        self.fm = fm
        self.register_buffer("x1", x1)
        self.register_buffer("eps", eps)
        drift = torch.zeros_like(x1) if fm.c is None else fm.compute_drift(x1)
        self.register_buffer("drift", drift)

    def forward(self, x, t, y=None):
        time = t / 1000.0
        drift = None if self.fm.c is None else self.drift[: x.shape[0]]
        return self.fm.compute_velocity_target(
            self.x1[: x.shape[0]], self.eps[: x.shape[0]], time, d=drift
        )


def test_midpoint_solver_and_nfe():
    torch.manual_seed(2)
    for c, start, end in [(None, 0.0, 0.3), (1, 0.3, 0.6), (0, 0.6, 1.0)]:
        fm = FlowMatchingWithDrift(c=c, sample_scale=2, s_k=start, s_k_plus_1=end)
        x1 = torch.randn(2, 4, 8, 8)
        eps = torch.randn_like(x1)
        drift = None if c is None else fm.compute_drift(x1)
        t_start = torch.full((2,), start)
        initial = fm.q_sample(x1, t_start, noise=eps, d=drift)
        model = OracleVelocity(fm, x1, eps)
        out = fm.sample_ode(
            model,
            initial,
            num_steps=5,
            model_kwargs={"y": torch.zeros(2, dtype=torch.long)},
            cfg_scale=1.0,
            solver="midpoint",
        )
        t_end = torch.full((2,), end)
        expected = fm.q_sample(x1, t_end, noise=eps, d=drift)
        assert_close(out["sample"], expected, f"midpoint endpoint {start}-{end}")
        assert_close(out["pred_x1"], x1, f"midpoint clean {start}-{end}")
        expected_nfe = 10 if end == 1.0 else 11
        if out["nfe"] != expected_nfe:
            raise AssertionError(f"NFE {out['nfe']} != {expected_nfe}")


if __name__ == "__main__":
    test_clean_inverse()
    test_noise_filling_projection()
    test_midpoint_solver_and_nfe()
    print("RDD-FM sampler CPU tests passed (3/3).")
