"""
CPU unit tests for RDD-Video Flow Matching module.
Verifies mathematical correctness against the 2D FM version.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from flow_matching_video import (
    SpatioTemporalDownsample,
    SpatioTemporalUpsample,
    VideoFlowMatchingWithDrift,
    RDDVideoSchedule,
)
from sample_rdd_wan import get_sigmas_shifted, euler_sample, ab2_sample, heun_sample


def test_downsample_upsample_roundtrip():
    """Test that Up(Down(x)) produces a valid low-frequency approximation."""
    print("Test 1: Downsample/Upsample roundtrip...")
    B, C, F, H, W = 2, 16, 4, 8, 8
    x = torch.randn(B, C, F, H, W)

    down = SpatioTemporalDownsample(2, 2)
    up = SpatioTemporalUpsample(2, 2)

    x_low = down(x)
    x_recon = up(x_low)

    assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} vs {x.shape}"

    # Drift should be pure high-frequency (zero mean per block)
    drift = x_recon - x
    drift_low = down(drift)
    assert drift_low.abs().max() < 1e-5, f"Drift is not pure high-freq: {drift_low.abs().max()}"
    print("  PASSED ✓")


def test_drift_boundary_conditions():
    """Test f(s_{k+1}) = 0 and f(s_k) = s_k (drift envelope boundaries)."""
    print("Test 2: Drift boundary conditions...")
    # Stage with s_k=0.3, s_{k+1}=0.6, L_k=0.3
    fm = VideoFlowMatchingWithDrift(c=1, spatial_factor=2, temporal_factor=2,
                                     s_k=0.3, s_k_plus_1=0.6)

    B, C, F, H, W = 2, 4, 4, 4, 4
    x_1 = torch.randn(B, C, F, H, W)
    noise = torch.randn(B, C, F, H, W)
    d = fm.compute_drift(x_1)

    # At t = s_{k+1} = 0.6: f(t) should be 0, so x_t = t*x_1 + (1-t)*noise
    t_end = torch.full((B,), 0.6)
    x_at_end = fm.q_sample(x_1, t_end, noise=noise, d=d)
    x_expected = 0.6 * x_1 + 0.4 * noise
    assert torch.allclose(x_at_end, x_expected, atol=1e-5), \
        f"At t=s_k+1, drift should be 0. Max error: {(x_at_end - x_expected).abs().max()}"

    # At t = s_k = 0.3: f(t) = t*(s_{k+1}-t)/L_k = 0.3*(0.6-0.3)/0.3 = 0.3
    t_start = torch.full((B,), 0.3)
    x_at_start = fm.q_sample(x_1, t_start, noise=noise, d=d)
    x_expected_start = 0.3 * x_1 + 0.7 * noise + 0.3 * d
    assert torch.allclose(x_at_start, x_expected_start, atol=1e-5), \
        f"At t=s_k, drift coeff should be s_k. Max error: {(x_at_start - x_expected_start).abs().max()}"

    print("  PASSED ✓")


def test_velocity_target_matches_2d():
    """Test Wan target w=dx/dsigma=-dx/dt."""
    print("Test 3: Velocity target correctness...")
    fm = VideoFlowMatchingWithDrift(c=1, spatial_factor=2, temporal_factor=2,
                                     s_k=0.4, s_k_plus_1=0.7)

    B, C, F, H, W = 2, 4, 4, 4, 4
    x_1 = torch.randn(B, C, F, H, W)
    noise = torch.randn(B, C, F, H, W)
    t = torch.tensor([0.45, 0.65])
    d = fm.compute_drift(x_1)

    v = fm.compute_velocity_target(x_1, noise, t, d)

    # Manual calculation
    t_exp = t.view(-1, 1, 1, 1, 1)
    L_k = 0.3
    drift_coeff = (0.7 - 2 * t_exp) / L_k
    v_expected = (noise - x_1) - drift_coeff * d
    assert torch.allclose(v, v_expected, atol=1e-6), "Velocity target mismatch"
    print("  PASSED ✓")


def test_time_sampling_within_stage():
    """Test that t is sampled only within [s_k, s_{k+1}]."""
    print("Test 4: Time sampling within stage bounds...")
    fm = VideoFlowMatchingWithDrift(c=None, spatial_factor=2, temporal_factor=2,
                                     s_k=0.0, s_k_plus_1=0.4)

    B, C, F, H, W = 100, 4, 4, 4, 4
    x_1 = torch.randn(B, C, F, H, W)

    def dummy_model(x_list, t_input):
        return [torch.zeros_like(x_list[i]) for i in range(len(x_list))]

    result = fm.training_losses(dummy_model, x_1)
    t_sampled = result["t"]

    assert t_sampled.min() >= 0.0, f"t below s_k: {t_sampled.min()}"
    assert t_sampled.max() <= 0.4, f"t above s_k+1: {t_sampled.max()}"
    print("  PASSED ✓")


def test_coarsest_stage_no_drift():
    """Test that coarsest stage (c=None) has no drift contribution."""
    print("Test 5: Coarsest stage no drift...")
    fm = VideoFlowMatchingWithDrift(c=None, spatial_factor=2, temporal_factor=2,
                                     s_k=0.0, s_k_plus_1=0.4)

    B, C, F, H, W = 2, 4, 4, 4, 4
    x_1 = torch.randn(B, C, F, H, W)
    noise = torch.randn(B, C, F, H, W)
    t = torch.tensor([0.1, 0.3])

    # q_sample without drift
    x_t = fm.q_sample(x_1, t, noise=noise)
    t_exp = t.view(-1, 1, 1, 1, 1)
    x_expected = t_exp * x_1 + (1 - t_exp) * noise
    assert torch.allclose(x_t, x_expected, atol=1e-6), "Coarsest should be standard FM"

    # Velocity without drift
    v = fm.compute_velocity_target(x_1, noise, t)
    v_expected = noise - x_1
    assert torch.allclose(v, v_expected, atol=1e-6), "Wan velocity should be noise - x_1"
    print("  PASSED ✓")


def test_schedule_time_boundaries():
    """Test RDDVideoSchedule computes correct time boundaries."""
    print("Test 6: Schedule time boundaries...")
    # PyramidalWan-style: each stage is ×2 in both space and time
    # Full res: 81×448×832 → latent 21×56×104
    # Stage 0 (coarsest): spatial÷4, temporal÷4 → ~6×14×26
    # Stage 1 (middle):   spatial÷2, temporal÷2 → ~11×28×52
    # Stage 2 (finest):   spatial÷1, temporal÷1 → 21×56×104
    stage_configs = [
        {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 20},
        {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 20},
        {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 10},
    ]
    # PyramidalWan sigma boundaries: [0, 0.06, 0.41, 1.0]
    # Mapped to our t convention (t=0 noise, t=1 data):
    # Coarsest at high noise (low t): stage_lengths ~ [60, 350, 590]
    schedule = RDDVideoSchedule(stage_configs, stage_lengths=[60, 350, 590], diffusion_steps=1000)

    expected_boundaries = [0.0, 0.06, 0.41, 1.0]
    for i, (actual, expected) in enumerate(zip(schedule.s_boundaries, expected_boundaries)):
        assert abs(actual - expected) < 1e-10, f"Boundary {i}: {actual} != {expected}"

    # Check flow matchers have correct intervals
    assert schedule.flow_matchers[0].s_k == 0.0
    assert abs(schedule.flow_matchers[0].s_k_plus_1 - 0.06) < 1e-10
    assert schedule.flow_matchers[0].c is None  # coarsest, no drift

    assert abs(schedule.flow_matchers[1].s_k - 0.06) < 1e-10
    assert abs(schedule.flow_matchers[1].s_k_plus_1 - 0.41) < 1e-10
    assert schedule.flow_matchers[1].c == 1  # has drift

    assert abs(schedule.flow_matchers[2].s_k - 0.41) < 1e-10
    assert schedule.flow_matchers[2].s_k_plus_1 == 1.0
    assert schedule.flow_matchers[2].c == 2  # has drift

    print("  PASSED ✓")


def test_upsample_with_noise():
    """Test stage transition: x_new = t_boundary * UP(x1) + (1-t_boundary) * noise."""
    print("Test 7: Upsample with noise (stage transition)...")
    # PyramidalWan-style: spatial and temporal both ×2 per stage
    stage_configs = [
        {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 20},
        {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 20},
        {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 10},
    ]
    schedule = RDDVideoSchedule(stage_configs, stage_lengths=[60, 350, 590], diffusion_steps=1000)

    B, C = 1, 16
    # Full res latent: F=16, H=16, W=16
    # Stage 0 (sf=4, tf=4): F/4=4, H/4=4, W/4=4
    pred_x1_stage0 = torch.randn(B, C, 4, 4, 4)

    # Transition to stage 1: boundary = s_boundaries[1] = 0.06
    # From stage 0 (sf=4, tf=4) to stage 1 (sf=2, tf=2):
    #   s_ratio = 4//2 = 2, tau_ratio = 4//2 = 2
    # Upsampled shape: (B, C, 4*2, 4*2, 4*2) = (B, C, 8, 8, 8)
    x_new = schedule.upsample_with_noise(pred_x1_stage0, from_stage=0, to_stage=1)

    expected_shape = (B, C, 8, 8, 8)
    assert x_new.shape == expected_shape, f"Shape {x_new.shape} != {expected_shape}"

    # Verify it's mostly noise (t_boundary=0.06, so 94% noise)
    assert x_new.abs().mean() > 0.1, "Output should not be zero"

    # Wan's 81 pixel frames become 21 latent frames: stages are 6 -> 11 -> 21.
    # A raw x2 repeat would incorrectly produce 12 frames at the middle stage.
    odd_stage0 = torch.randn(B, C, 6, 14, 26)
    odd_stage1 = schedule.upsample_with_noise(
        odd_stage0,
        from_stage=0,
        to_stage=1,
        target_shape=(11, 28, 52),
    )
    assert odd_stage1.shape == (B, C, 11, 28, 52), odd_stage1.shape
    print("  PASSED ✓")


def test_training_losses_all_stages():
    """Test training loop: compute loss for all stages (unified mode)."""
    print("Test 8: Training losses all stages...")
    # PyramidalWan-style config
    stage_configs = [
        {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 20},
        {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 20},
        {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 10},
    ]
    schedule = RDDVideoSchedule(stage_configs, stage_lengths=[60, 350, 590], diffusion_steps=1000)

    B, C, F, H, W = 2, 4, 16, 16, 16  # full res (divisible by 4)
    x_1_full = torch.randn(B, C, F, H, W)

    def dummy_model(x_list, t_input):
        return [torch.zeros_like(x_list[i]) for i in range(len(x_list))]

    total_loss = 0.0
    for i in range(schedule.num_stages):
        x_stage = schedule.downsample_latent(x_1_full, i)
        result = schedule.flow_matchers[i].training_losses(dummy_model, x_stage)
        loss = result["loss"].mean()
        total_loss += loss.item()

        # Verify t is within this stage's bounds
        t = result["t"]
        assert t.min() >= schedule.flow_matchers[i].s_k, f"Stage {i}: t below s_k"
        assert t.max() <= schedule.flow_matchers[i].s_k_plus_1, f"Stage {i}: t above s_k+1"

    assert total_loss > 0, "Total loss should be positive"
    print("  PASSED ✓")


def test_ode_integration_endpoint():
    """Test that ODE integration from s_k to s_{k+1} with perfect model gives x_1."""
    print("Test 9: ODE integration endpoint convergence...")
    # If model predicts the true velocity, ODE should recover x_1 at t=s_{k+1}
    # For coarsest stage (no drift): v_true = x_1 - noise
    # Test: start at x_0 = s_k*x_1 + (1-s_k)*noise, integrate to s_{k+1}

    fm = VideoFlowMatchingWithDrift(c=None, spatial_factor=2, temporal_factor=2,
                                     s_k=0.0, s_k_plus_1=0.4)

    B, C, F, H, W = 1, 4, 4, 4, 4
    x_1_true = torch.randn(B, C, F, H, W)
    noise_true = torch.randn(B, C, F, H, W)

    # Initial state at t=0: pure noise
    x_init = noise_true.clone()  # at t=0: x = 0*x_1 + 1*noise = noise

    # Wan predicts dx/dsigma = noise - x_1.
    def perfect_model(x_list, t_input):
        v = noise_true - x_1_true
        return [v[i] for i in range(len(x_list))]

    out = fm.sample_ode(perfect_model, x_init, num_steps=1000)
    # At t=0.4: x should be 0.4*x_1 + 0.6*noise = x_init + 0.4*(x_1-noise)
    x_expected = 0.4 * x_1_true + 0.6 * noise_true
    error = (out["sample"] - x_expected).abs().max()
    assert error < 0.01, f"ODE endpoint error too large: {error}"
    print("  PASSED ✓")


def test_rdd_clean_reconstruction():
    """The projection-residual inverse must recover x_1 from a perfect velocity."""
    print("Test 10: RDD-FM clean reconstruction...")
    fm = VideoFlowMatchingWithDrift(
        c=1, spatial_factor=2, temporal_factor=2, s_k=0.3, s_k_plus_1=0.7
    )
    x_1 = torch.randn(3, 4, 4, 4, 4)
    noise = torch.randn_like(x_1)
    t = torch.tensor([0.35, 0.50, 0.68])
    d = fm.compute_drift(x_1)
    x_t = fm.q_sample(x_1, t, noise=noise, d=d)
    w_sigma = fm.compute_velocity_target(x_1, noise, t, d=d)
    recovered = fm.predict_x1_from_velocity(x_t, w_sigma, 1.0 - t)
    error = (recovered - x_1).abs().max()
    assert error < 2e-5, f"Clean reconstruction error: {error}"

    # The projection and inverse must remain exact for the padded/cropped odd
    # temporal sizes used by 81-frame Wan videos (e.g. 11 latent frames).
    x_1_odd = torch.randn(1, 4, 11, 28, 52)
    noise_odd = torch.randn_like(x_1_odd)
    t_odd = torch.tensor([0.5])
    d_odd = fm.compute_drift(x_1_odd)
    x_t_odd = fm.q_sample(x_1_odd, t_odd, noise=noise_odd, d=d_odd)
    w_odd = fm.compute_velocity_target(x_1_odd, noise_odd, t_odd, d=d_odd)
    recovered_odd = fm.predict_x1_from_velocity(x_t_odd, w_odd, 1.0 - t_odd)
    odd_error = (recovered_odd - x_1_odd).abs().max()
    assert odd_error < 2e-5, f"Odd-size clean reconstruction error: {odd_error}"
    print("  PASSED ✓")


def test_wan_time_change_of_variables():
    """Analytic Wan velocity must equal -d x_t / d t by finite differences."""
    print("Test 11: Wan sigma/data-time change of variables...")
    fm = VideoFlowMatchingWithDrift(
        c=1, spatial_factor=2, temporal_factor=2, s_k=0.2, s_k_plus_1=0.8
    )
    x_1 = torch.randn(1, 2, 4, 4, 4)
    noise = torch.randn_like(x_1)
    d = fm.compute_drift(x_1)
    t = torch.tensor([0.47])
    delta = 1e-4
    dx_dt = (
        fm.q_sample(x_1, t + delta, noise=noise, d=d)
        - fm.q_sample(x_1, t - delta, noise=noise, d=d)
    ) / (2 * delta)
    w_sigma = fm.compute_velocity_target(x_1, noise, t, d=d)
    assert torch.allclose(w_sigma, -dx_dt, atol=2e-3, rtol=2e-3)
    print("  PASSED ✓")


def test_stage_sigma_schedule_and_solvers():
    """Stage endpoints are preserved and all generic solvers follow Wan direction."""
    print("Test 12: Stage sigma schedule and ODE solvers...")
    sigmas = get_sigmas_shifted(8, shift=5.0, sigma_max=0.94, sigma_min=0.59)
    assert abs(sigmas[0].item() - 0.94) < 1e-6
    assert abs(sigmas[-1].item() - 0.59) < 1e-6
    assert torch.all(sigmas[:-1] > sigmas[1:])

    x_1 = torch.randn(1, 2, 2, 2, 2)
    noise = torch.randn_like(x_1)
    x_start = (1 - sigmas[0]) * x_1 + sigmas[0] * noise
    constant_w = noise - x_1

    def perfect_model(x_list, t_input):
        return [constant_w[i] for i in range(len(x_list))]

    expected = (1 - sigmas[-1]) * x_1 + sigmas[-1] * noise
    for solver in (euler_sample, ab2_sample, heun_sample):
        out, _ = solver(perfect_model, x_start.clone(), sigmas, show_progress=False)
        assert torch.allclose(out, expected, atol=2e-5), solver.__name__
    print("  PASSED ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("RDD-Video CPU Unit Tests (Correct FM Version)")
    print("=" * 60)
    print()

    test_downsample_upsample_roundtrip()
    test_drift_boundary_conditions()
    test_velocity_target_matches_2d()
    test_time_sampling_within_stage()
    test_coarsest_stage_no_drift()
    test_schedule_time_boundaries()
    test_upsample_with_noise()
    test_training_losses_all_stages()
    test_ode_integration_endpoint()
    test_rdd_clean_reconstruction()
    test_wan_time_change_of_variables()
    test_stage_sigma_schedule_and_solvers()

    print()
    print("=" * 60)
    print("ALL 12 TESTS PASSED ✓")
    print("=" * 60)
