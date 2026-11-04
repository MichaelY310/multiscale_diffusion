"""CPU contract tests for the RDD distillation mathematics."""

from __future__ import annotations

import torch

from flow_matching_video import RDDVideoSchedule
from rdd_distillation import (
    RDDDistillationSchedule,
    apply_rdd_signal_operator,
    exact_dmd_path_correction,
    integrate_teacher_interval,
    rdd_renoise,
    recover_clean_from_velocity,
    trajectory_distillation_loss,
)


def make_schedule() -> RDDVideoSchedule:
    return RDDVideoSchedule(
        stage_configs=[
            {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 2},
            {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 2},
            {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 1},
        ],
        stage_lengths=[60, 350, 590],
        diffusion_steps=1000,
        drift_scale=1.0,
    )


def test_calls_do_not_cross_boundaries() -> None:
    schedule = make_schedule()
    distilled = RDDDistillationSchedule(schedule, (2, 2, 1))
    assert len(distilled.calls) == 5
    assert [len(distilled.calls_for_stage(i)) for i in range(3)] == [2, 2, 1]
    assert abs(distilled.calls[0].t_start - 0.0) < 1e-8
    assert abs(distilled.calls[-1].t_end - 1.0) < 1e-8
    for call in distilled.calls:
        assert schedule.s_boundaries[call.stage_index] <= call.t_start
        assert call.t_end <= schedule.s_boundaries[call.stage_index + 1]


def test_renoise_is_existing_q_sample() -> None:
    torch.manual_seed(0)
    schedule = make_schedule()
    for stage_index, fm in enumerate(schedule.flow_matchers):
        shape = [(2, 3, 3, 4, 4), (2, 3, 5, 8, 8), (2, 3, 9, 16, 16)][stage_index]
        x1 = torch.randn(shape)
        noise = torch.randn_like(x1)
        left, right = schedule.s_boundaries[stage_index : stage_index + 2]
        t = torch.tensor([left + 0.35 * (right - left), left + 0.7 * (right - left)])
        d = None if fm.c is None else fm.drift_scale * fm.compute_drift(x1)
        expected = fm.q_sample(x1, t, noise=noise, d=d)
        actual = rdd_renoise(fm, x1, t, noise)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_clean_inverse_on_all_stages() -> None:
    torch.manual_seed(1)
    schedule = make_schedule()
    for stage_index, fm in enumerate(schedule.flow_matchers):
        shape = [(2, 3, 3, 4, 4), (2, 3, 5, 8, 8), (2, 3, 9, 16, 16)][stage_index]
        x1 = torch.randn(shape)
        noise = torch.randn_like(x1)
        left, right = schedule.s_boundaries[stage_index : stage_index + 2]
        t = torch.tensor([left + 0.23 * (right - left), left + 0.61 * (right - left)])
        d = None if fm.c is None else fm.drift_scale * fm.compute_drift(x1)
        x_t = fm.q_sample(x1, t, noise=noise, d=d)
        velocity = fm.compute_velocity_target(x1, noise, t, d=d)
        recovered = recover_clean_from_velocity(fm, x_t, velocity, t)
        torch.testing.assert_close(recovered, x1, rtol=2e-4, atol=2e-5)


def test_path_correction_matches_operator_squared() -> None:
    torch.manual_seed(2)
    fm = make_schedule().flow_matchers[2]
    delta = torch.randn(2, 3, 9, 16, 16)
    t = torch.tensor([0.55, 0.83])
    expected = apply_rdd_signal_operator(
        fm, apply_rdd_signal_operator(fm, delta, t), t
    )
    actual = exact_dmd_path_correction(fm, delta, t)
    torch.testing.assert_close(actual, expected)


def test_teacher_integrator_and_zero_trajectory_loss() -> None:
    x0 = torch.randn(2, 3, 2, 4, 4)
    constant_w = torch.full_like(x0, 0.25)

    def velocity(x, t):
        del t
        return constant_w.expand_as(x)

    end = integrate_teacher_interval(velocity, x0, 0.2, 0.7, substeps=5, solver="heun")
    torch.testing.assert_close(end, x0 - 0.5 * constant_w)

    call = RDDDistillationSchedule(make_schedule(), (2, 2, 1)).calls[3]
    loss, student_end, teacher_end = trajectory_distillation_loss(
        velocity, velocity, x0, call, teacher_substeps=7, teacher_solver="heun"
    )
    assert loss.item() < 1e-12
    torch.testing.assert_close(student_end, teacher_end)


def main() -> None:
    tests = [
        test_calls_do_not_cross_boundaries,
        test_renoise_is_existing_q_sample,
        test_clean_inverse_on_all_stages,
        test_path_correction_matches_operator_squared,
        test_teacher_integrator_and_zero_trajectory_loss,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"RDD distillation CPU tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
