"""Quick CPU test for RDD Flow Matching module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from diffusion import FlowMatchingWithDrift


class DummyDiT(nn.Module):
    """Minimal model that mimics DiT output shape."""
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        self.net = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, y=None):
        # t is [B] float, ignore it for shape test
        return self.net(x)


def test_flow_matching_no_drift():
    """Test standard FM (coarsest stage, c=None)."""
    print("Test 1: Standard FM (no drift, c=None)...")
    fm = FlowMatchingWithDrift(c=None)
    model = DummyDiT(in_channels=4, out_channels=8)

    x_1 = torch.randn(2, 4, 8, 8)
    model_kwargs = {"y": torch.tensor([3, 7])}

    result = fm.training_losses(model, x_1, model_kwargs=model_kwargs)
    assert "loss" in result
    assert result["loss"].shape == (2,)
    assert result["loss"].mean() > 0
    print(f"  Loss shape: {result['loss'].shape}, value: {result['loss'].mean().item():.4f}")
    print("  PASSED ✓")


def test_flow_matching_with_drift():
    """Test FM with drift (fine stage, c=0)."""
    print("\nTest 2: FM with drift (c=0, scale=2)...")
    fm = FlowMatchingWithDrift(c=0, sample_scale=2)
    model = DummyDiT(in_channels=4, out_channels=8)

    x_1 = torch.randn(2, 4, 16, 16)
    model_kwargs = {"y": torch.tensor([3, 7])}

    result = fm.training_losses(model, x_1, model_kwargs=model_kwargs)
    assert "loss" in result
    assert result["loss"].shape == (2,)
    print(f"  Loss shape: {result['loss'].shape}, value: {result['loss'].mean().item():.4f}")
    print("  PASSED ✓")


def test_q_sample_boundaries():
    """Test that q_sample respects boundary conditions."""
    print("\nTest 3: q_sample boundary conditions...")
    fm = FlowMatchingWithDrift(c=0, sample_scale=2)

    x_1 = torch.randn(4, 4, 16, 16)
    noise = torch.randn_like(x_1)
    d = fm.compute_drift(x_1)

    # t=0 should give noise
    t_zero = torch.zeros(4)
    x_t0 = fm.q_sample(x_1, t_zero, noise=noise, d=d)
    assert torch.allclose(x_t0, noise, atol=1e-6), "At t=0, x_t should be noise"
    print("  t=0: x_t ≈ ε ✓")

    # t=1 should give x_1
    t_one = torch.ones(4)
    x_t1 = fm.q_sample(x_1, t_one, noise=noise, d=d)
    assert torch.allclose(x_t1, x_1, atol=1e-6), "At t=1, x_t should be x_1"
    print("  t=1: x_t ≈ x_1 ✓")

    # t=0.5: drift should be maximal
    t_half = torch.full((4,), 0.5)
    x_t05 = fm.q_sample(x_1, t_half, noise=noise, d=d)
    expected = 0.5 * x_1 + 0.5 * noise + 0.25 * d
    assert torch.allclose(x_t05, expected, atol=1e-6), "At t=0.5, drift should be 0.25*d"
    print("  t=0.5: x_t = 0.5·x_1 + 0.5·ε + 0.25·d ✓")
    print("  PASSED ✓")


def test_velocity_target():
    """Test velocity target formula."""
    print("\nTest 4: Velocity target formula...")
    fm = FlowMatchingWithDrift(c=0, sample_scale=2)

    x_1 = torch.randn(4, 4, 16, 16)
    noise = torch.randn_like(x_1)
    d = fm.compute_drift(x_1)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])

    u_t = fm.compute_velocity_target(x_1, noise, t, d=d)

    # At t=0: u_t = (x_1 - ε) + d
    t_expanded = t.view(-1, 1, 1, 1)
    expected = (x_1 - noise) + (1 - 2 * t_expanded) * d
    assert torch.allclose(u_t, expected, atol=1e-6)
    print("  u_t = (x_1 - ε) + (1-2t)·d ✓")

    # Without drift
    u_t_nodrift = fm.compute_velocity_target(x_1, noise, t, d=None)
    assert torch.allclose(u_t_nodrift, x_1 - noise, atol=1e-6)
    print("  u_t (no drift) = x_1 - ε ✓")
    print("  PASSED ✓")


def test_drift_computation():
    """Test drift d = UP(DOWN(x_1)) - x_1."""
    print("\nTest 5: Drift computation...")
    fm = FlowMatchingWithDrift(c=0, sample_scale=2)

    # Constant image: drift should be zero
    x_const = torch.ones(1, 4, 8, 8) * 3.14
    d_const = fm.compute_drift(x_const)
    assert torch.allclose(d_const, torch.zeros_like(d_const), atol=1e-6)
    print("  Constant image → drift ≈ 0 ✓")

    # Random image: drift should be non-zero (high-freq component)
    x_rand = torch.randn(1, 4, 8, 8)
    d_rand = fm.compute_drift(x_rand)
    assert d_rand.abs().mean() > 0.01
    print(f"  Random image → drift magnitude: {d_rand.abs().mean().item():.4f} ✓")

    # UP(DOWN(x)) - x should remove high-freq info
    low_pass = fm.upsample(fm.downsample(x_rand, 2), 2)
    assert torch.allclose(low_pass, x_rand + d_rand, atol=1e-6)
    print("  UP(DOWN(x_1)) = x_1 + d ✓")
    print("  PASSED ✓")


def test_sampling():
    """Test ODE sampling runs without error."""
    print("\nTest 6: ODE sampling (Euler, 10 steps)...")
    fm = FlowMatchingWithDrift(c=None)
    model = DummyDiT(in_channels=4, out_channels=8)

    noise = torch.randn(2, 4, 8, 8)
    model_kwargs = {"y": torch.tensor([3, 7])}

    samples = fm.sample_ode(model, noise, num_steps=10, model_kwargs=model_kwargs)
    assert samples.shape == noise.shape
    print(f"  Output shape: {samples.shape} ✓")
    print("  PASSED ✓")


def test_sampling_midpoint():
    """Test midpoint ODE sampling."""
    print("\nTest 7: ODE sampling (Midpoint, 5 steps)...")
    fm = FlowMatchingWithDrift(c=None)
    model = DummyDiT(in_channels=4, out_channels=8)

    noise = torch.randn(2, 4, 8, 8)
    model_kwargs = {"y": torch.tensor([3, 7])}

    samples = fm.sample_ode_midpoint(model, noise, num_steps=5, model_kwargs=model_kwargs)
    assert samples.shape == noise.shape
    print(f"  Output shape: {samples.shape} ✓")
    print("  PASSED ✓")


def test_cfg_sampling():
    """Test CFG sampling."""
    print("\nTest 8: CFG sampling (scale=4.0)...")
    fm = FlowMatchingWithDrift(c=None)
    model = DummyDiT(in_channels=4, out_channels=8)

    noise = torch.randn(2, 4, 8, 8)
    model_kwargs = {"y": torch.tensor([3, 7])}

    samples = fm.sample_ode(model, noise, num_steps=5, model_kwargs=model_kwargs, cfg_scale=4.0)
    assert samples.shape == noise.shape
    print(f"  Output shape: {samples.shape} ✓")
    print("  PASSED ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("RDD Flow Matching - CPU Unit Tests")
    print("=" * 60)

    test_flow_matching_no_drift()
    test_flow_matching_with_drift()
    test_q_sample_boundaries()
    test_velocity_target()
    test_drift_computation()
    test_sampling()
    test_sampling_midpoint()
    test_cfg_sampling()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
