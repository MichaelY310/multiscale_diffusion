#!/usr/bin/env python3
"""Twelve independent CPU checks for the paper-faithful RDD-DMD derivation."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusion.rdd_phased_dmd import (  # noqa: E402
    apply_resolution_drift,
    dmd_surrogate_loss,
    downsample_clean,
    drift_fraction,
    fresh_transition,
    projection,
    rdd_dmd_gradient,
    rdd_marginal_ddim_update,
    rdd_renoise,
    sampling_phases,
)


def close(actual: torch.Tensor, expected: torch.Tensor, message: str) -> None:
    if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-5):
        error = float((actual - expected).abs().max())
        raise AssertionError(f"{message}; max_abs_error={error}")


def passed(number: int, message: str) -> None:
    print(f"CHECK {number:02d} PASS: {message}")


def make_drift_diffusion():
    return create_diffusion_with_drift(
        timestep_respacing="",
        diffusion_steps=1000,
        c=1,
        T_c=300,
        T_c_plus_1=600,
        sample_scale=2,
    )


def make_plain_diffusion():
    return create_diffusion_with_drift(
        timestep_respacing="",
        diffusion_steps=1000,
        c=None,
        T_c=600,
        T_c_plus_1=1000,
        sample_scale=2,
    )


def extract(values, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(values, dtype=like.dtype, device=like.device)
    result = table[timesteps]
    return result.reshape(result.shape[0], *([1] * (like.ndim - 1)))


def main() -> None:
    torch.manual_seed(7)
    drift_diffusion = make_drift_diffusion()
    plain_diffusion = make_plain_diffusion()
    t = torch.tensor([0, drift_diffusion.num_timesteps - 1], dtype=torch.long)
    x = torch.randn(2, 4, 16, 16)
    noise = torch.randn_like(x)

    phases = sampling_phases((32, 16, 8))
    assert [item.resolution for item in phases] == [8, 16, 32]
    assert [item.diffusion_index for item in phases] == [2, 1, 0]
    passed(1, "sampling order is exactly 8 -> 16 -> 32")

    projected = projection(x, 2)
    close(projection(projected, 2), projected, "down-up must be idempotent")
    passed(2, "average-pool then nearest-repeat is idempotent")

    y = torch.randn_like(x)
    close(
        (projection(x, 2) * y).sum(),
        (x * projection(y, 2)).sum(),
        "down-up must equal its transpose",
    )
    passed(3, "down-up is self-adjoint, so the chain-rule transpose is exact")

    fraction = drift_fraction(drift_diffusion, t, x)
    global_t = torch.as_tensor(drift_diffusion.timestep_map)[t].to(x.dtype)
    expected_fraction = ((global_t - 300.0) / 300.0).view(2, 1, 1, 1)
    close(fraction, expected_fraction, "drift fraction must match q_sample source")
    passed(4, "drift fraction exactly matches the repository q_sample expression")

    signal_scale = extract(drift_diffusion.sqrt_alphas_cumprod, t, x)
    noise_scale = extract(drift_diffusion.sqrt_one_minus_alphas_cumprod, t, x)
    expected_xt = signal_scale * apply_resolution_drift(
        drift_diffusion, x, t
    ) + noise_scale * noise
    actual_xt = rdd_renoise(drift_diffusion, x, t, noise=noise)
    close(actual_xt, expected_xt, "RDD re-noising must equal q_sample algebra")
    passed(5, "RDD re-noising reproduces q_sample exactly")

    plain_xt = rdd_renoise(plain_diffusion, x, t, noise=noise)
    expected_plain_xt = (
        extract(plain_diffusion.sqrt_alphas_cumprod, t, x) * x
        + extract(plain_diffusion.sqrt_one_minus_alphas_cumprod, t, x) * noise
    )
    close(plain_xt, expected_plain_xt, "no-drift q_sample must be standard VP")
    passed(6, "disabling resolution drift recovers ordinary VP diffusion")

    x_for_grad = x.clone().requires_grad_(True)
    probe = torch.randn_like(x_for_grad)
    xt_for_grad = rdd_renoise(
        drift_diffusion, x_for_grad, t, noise=torch.zeros_like(x_for_grad)
    )
    jacobian_transpose_probe = torch.autograd.grad(
        (xt_for_grad * probe).sum(), x_for_grad
    )[0]
    expected_jacobian_transpose_probe = signal_scale * apply_resolution_drift(
        drift_diffusion, probe, t
    )
    close(
        jacobian_transpose_probe,
        expected_jacobian_transpose_probe,
        "closed-form q_sample transpose-Jacobian is wrong",
    )
    passed(7, "closed-form q_sample Jacobian agrees with PyTorch autograd")

    teacher_clean = torch.randn_like(x)
    fake_clean = torch.randn_like(x)
    noisy = torch.randn_like(x)
    variance = 1.0 - extract(drift_diffusion.alphas_cumprod, t, x)
    teacher_forward_mean = signal_scale * apply_resolution_drift(
        drift_diffusion, teacher_clean, t
    )
    fake_forward_mean = signal_scale * apply_resolution_drift(
        drift_diffusion, fake_clean, t
    )
    teacher_score = -(noisy - teacher_forward_mean) / variance
    fake_score = -(noisy - fake_forward_mean) / variance
    expected_score_difference = (
        signal_scale
        * apply_resolution_drift(
            drift_diffusion, fake_clean - teacher_clean, t
        )
        / variance
    )
    close(
        fake_score - teacher_score,
        expected_score_difference,
        "RDD fake-real score identity is wrong",
    )
    passed(8, "fake-real RDD score difference follows the exact Gaussian score")

    generated = torch.randn_like(x)
    normalizer = (generated - teacher_clean).abs().mean(
        dim=(1, 2, 3), keepdim=True
    )
    paper_weight = variance / signal_scale / normalizer
    pulled_back_score = signal_scale * apply_resolution_drift(
        drift_diffusion, fake_score - teacher_score, t
    )
    gradient_from_paper_equations = paper_weight * pulled_back_score
    gradient_from_implementation = rdd_dmd_gradient(
        drift_diffusion, generated, teacher_clean, fake_clean, t
    )
    close(
        gradient_from_implementation,
        gradient_from_paper_equations,
        "implementation must equal DMD Eq.7 plus Eq.8",
    )
    passed(9, "implemented RDD-DMD gradient equals DMD Eq.7 and Eq.8")

    plain_signal_scale = extract(plain_diffusion.sqrt_alphas_cumprod, t, x)
    expected_plain_gradient = (
        plain_signal_scale * (fake_clean - teacher_clean) / normalizer
    )
    actual_plain_gradient = rdd_dmd_gradient(
        plain_diffusion, generated, teacher_clean, fake_clean, t
    )
    close(
        actual_plain_gradient,
        expected_plain_gradient,
        "no-drift reduction must match VP DMD Eq.7/8",
    )
    passed(10, "no-drift limit is exactly the original VP-DMD gradient")

    edm_like = SimpleNamespace(
        c=None,
        sqrt_alphas_cumprod=torch.ones(2).numpy(),
    )
    edm_t = torch.tensor([0, 1], dtype=torch.long)
    expected_algorithm2 = (fake_clean - teacher_clean) / normalizer
    actual_algorithm2 = rdd_dmd_gradient(
        edm_like, generated, teacher_clean, fake_clean, edm_t
    )
    close(
        actual_algorithm2,
        expected_algorithm2,
        "unit signal scale must reduce to DMD Algorithm 2",
    )
    passed(11, "unit signal-scale limit is exactly DMD Algorithm 2")

    generated_for_loss = generated.clone().requires_grad_(True)
    loss = dmd_surrogate_loss(generated_for_loss, gradient_from_implementation)
    loss.backward()
    close(
        generated_for_loss.grad,
        gradient_from_implementation / generated_for_loss.numel(),
        "surrogate loss derivative is wrong",
    )
    passed(12, "surrogate MSE backpropagates exactly the derived gradient")

    boundary_noise = torch.randn(2, 4, 16, 16)
    coarse = torch.randn(2, 4, 8, 8)
    first = fresh_transition(
        coarse, 16, drift_diffusion, noise=boundary_noise
    )
    second = fresh_transition(
        coarse, 16, drift_diffusion, noise=boundary_noise
    )
    close(first, second, "explicit boundary noise must make transition reproducible")
    print("EXTRA CHECK PASS: stored boundary noise gives deterministic DMD pairs")

    jump_t = torch.tensor(
        [drift_diffusion.num_timesteps - 1, drift_diffusion.num_timesteps // 2],
        dtype=torch.long,
    )
    jump_s = torch.tensor([0, drift_diffusion.num_timesteps // 4], dtype=torch.long)
    exact_clean = torch.randn_like(x)
    shared_noise = torch.randn_like(x)
    jump_start = rdd_renoise(
        drift_diffusion, exact_clean, jump_t, noise=shared_noise
    )
    jump_expected = rdd_renoise(
        drift_diffusion, exact_clean, jump_s, noise=shared_noise
    )
    jump_actual = rdd_marginal_ddim_update(
        drift_diffusion, jump_start, exact_clean, jump_t, jump_s
    )
    close(
        jump_actual,
        jump_expected,
        "RDD marginal-DDIM must preserve the same forward noise exactly",
    )
    passed(13, "RDD marginal-DDIM exactly couples two drifted forward marginals")

    plain_start = rdd_renoise(
        plain_diffusion, exact_clean, jump_t, noise=shared_noise
    )
    plain_actual = rdd_marginal_ddim_update(
        plain_diffusion, plain_start, exact_clean, jump_t, jump_s
    )
    alpha_s = extract(plain_diffusion.sqrt_alphas_cumprod, jump_s, x)
    sigma_s = extract(plain_diffusion.sqrt_one_minus_alphas_cumprod, jump_s, x)
    plain_expected = alpha_s * exact_clean + sigma_s * shared_noise
    close(plain_actual, plain_expected, "no-drift DDIM reduction is wrong")
    passed(14, "no-drift marginal coupling reduces to ordinary eta=0 DDIM")

    repeated = rdd_marginal_ddim_update(
        drift_diffusion, jump_start, exact_clean, jump_t, jump_s
    )
    close(jump_actual, repeated, "marginal-DDIM update must be deterministic")
    passed(15, "fixed input gives a deterministic RDD teacher mapping")

    fine = torch.randn(2, 4, 32, 32, requires_grad=True)
    coarse_probe = torch.randn(2, 4, 8, 8)
    coarse_from_final = downsample_clean(fine, 8)
    pullback = torch.autograd.grad((coarse_from_final * coarse_probe).sum(), fine)[0]
    expected_pullback = (
        coarse_probe.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
        / 16.0
    )
    close(
        pullback,
        expected_pullback,
        "final-output coarse score must pull back through average pooling",
    )
    passed(16, "low-resolution final-output DMD score has the exact pool transpose")

    stage_specs = ((0, 300, 32), (300, 600, 16), (600, 1000, 8))
    global_indices = []
    counts = []
    for stage_index, (start, end, _) in enumerate(stage_specs):
        stage_diffusion = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=1000,
            c=stage_index if stage_index < 2 else None,
            T_c=start,
            T_c_plus_1=end,
            sample_scale=2,
        )
        mapping = torch.as_tensor(stage_diffusion.timestep_map)
        selected = mapping[(mapping >= 20) & (mapping < 980)]
        global_indices.extend(selected.tolist())
        counts.append(int(selected.numel()))
    assert sorted(global_indices) == list(range(20, 980))
    assert counts == [280, 300, 380]
    passed(17, "score sampling partitions global [20,980) with 280/300/380 weights")

    print("ALL 17 REQUIRED CHECKS PASSED")


if __name__ == "__main__":
    main()
