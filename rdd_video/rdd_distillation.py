"""Core mathematics for stage-aware RDD video distillation.

This module intentionally has no Wan/FastVideo dependency.  It defines the
RDD path used by both trajectory distillation and DMD2, and makes the most
important contract explicit: a distilled step is never allowed to cross a
resolution boundary.

Conventions match :mod:`flow_matching_video`:

* data time ``t`` runs from 0 (noise) to 1 (data);
* Wan receives noise time ``sigma = 1 - t`` and predicts ``dx/dsigma``;
* an RDD stage follows
  ``x_t = t*x1 + (1-t)*eps + f(t)*(P*x1-x1)``;
* ``P`` is latent average-pool followed by repeat upsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from flow_matching_video import RDDVideoSchedule, VideoFlowMatchingWithDrift


Tensor = torch.Tensor
VelocityFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class DistillationCall:
    """One student solver call wholly contained in one RDD stage."""

    global_index: int
    stage_index: int
    index_in_stage: int
    t_start: float
    t_end: float

    @property
    def dt(self) -> float:
        return self.t_end - self.t_start


class RDDDistillationSchedule:
    """Fixed few-step schedule aligned exactly to RDD stage boundaries."""

    def __init__(
        self,
        rdd_schedule: RDDVideoSchedule,
        student_steps_per_stage: Sequence[int] = (2, 2, 1),
    ) -> None:
        if len(student_steps_per_stage) != rdd_schedule.num_stages:
            raise ValueError(
                "student_steps_per_stage must contain one positive value per "
                f"RDD stage ({len(student_steps_per_stage)} vs "
                f"{rdd_schedule.num_stages})"
            )
        if any(int(n) <= 0 for n in student_steps_per_stage):
            raise ValueError("every stage must receive at least one student call")

        self.rdd_schedule = rdd_schedule
        self.student_steps_per_stage = tuple(int(n) for n in student_steps_per_stage)
        calls: List[DistillationCall] = []
        global_index = 0
        for stage_index, num_steps in enumerate(self.student_steps_per_stage):
            left = float(rdd_schedule.s_boundaries[stage_index])
            right = float(rdd_schedule.s_boundaries[stage_index + 1])
            dt = (right - left) / num_steps
            for index_in_stage in range(num_steps):
                t_start = left + index_in_stage * dt
                t_end = left + (index_in_stage + 1) * dt
                calls.append(
                    DistillationCall(
                        global_index=global_index,
                        stage_index=stage_index,
                        index_in_stage=index_in_stage,
                        t_start=t_start,
                        t_end=t_end,
                    )
                )
                global_index += 1
        self.calls = tuple(calls)
        self._validate()

    def _validate(self) -> None:
        tol = 1e-7
        for call in self.calls:
            left = self.rdd_schedule.s_boundaries[call.stage_index]
            right = self.rdd_schedule.s_boundaries[call.stage_index + 1]
            if call.t_start < left - tol or call.t_end > right + tol:
                raise AssertionError(f"distillation call crosses an RDD boundary: {call}")
            if call.dt <= 0:
                raise AssertionError(f"non-positive distillation interval: {call}")

        for stage_index in range(self.rdd_schedule.num_stages):
            stage_calls = self.calls_for_stage(stage_index)
            left = self.rdd_schedule.s_boundaries[stage_index]
            right = self.rdd_schedule.s_boundaries[stage_index + 1]
            if abs(stage_calls[0].t_start - left) > tol:
                raise AssertionError("student schedule does not start at the stage boundary")
            if abs(stage_calls[-1].t_end - right) > tol:
                raise AssertionError("student schedule does not end at the stage boundary")
            for first, second in zip(stage_calls[:-1], stage_calls[1:]):
                if abs(first.t_end - second.t_start) > tol:
                    raise AssertionError("student schedule contains a gap or overlap")

    def calls_for_stage(self, stage_index: int) -> List[DistillationCall]:
        return [call for call in self.calls if call.stage_index == stage_index]

    def call(self, global_index: int) -> DistillationCall:
        return self.calls[int(global_index)]

    def round_robin(self, optimizer_step: int, rank: int = 0) -> DistillationCall:
        """Deterministic balanced call assignment across ranks and iterations."""

        return self.calls[(int(optimizer_step) + int(rank)) % len(self.calls)]


def _as_batch_time(value: float | Tensor, x: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        value = value.to(device=x.device, dtype=x.dtype)
        if value.ndim == 0:
            value = value.expand(x.shape[0])
        if value.shape != (x.shape[0],):
            raise ValueError(f"time must have shape [B], got {tuple(value.shape)}")
        return value
    return torch.full((x.shape[0],), float(value), device=x.device, dtype=x.dtype)


def _expand_time(value: Tensor, x: Tensor) -> Tensor:
    return value.view(value.shape[0], *([1] * (x.ndim - 1)))


def projection(fm: VideoFlowMatchingWithDrift, x: Tensor) -> Tensor:
    """Apply the stage projection P; the coarsest stage uses the identity."""

    if fm.c is None:
        return x
    return x + fm.compute_drift(x)


def drift_envelope(fm: VideoFlowMatchingWithDrift, t: Tensor) -> Tensor:
    """Return ``f(t)=scale*t*(stage_end-t)/stage_length``."""

    if fm.c is None:
        return torch.zeros_like(t)
    return (
        fm.drift_scale
        * t
        * (float(fm.s_k_plus_1) - t)
        / float(fm.stage_length)
    )


def apply_rdd_signal_operator(
    fm: VideoFlowMatchingWithDrift,
    x1: Tensor,
    t: float | Tensor,
) -> Tensor:
    """Apply ``A_t x1 = t*x1 + f(t)*(P-I)x1``."""

    t_batch = _as_batch_time(t, x1)
    t_expanded = _expand_time(t_batch, x1)
    result = t_expanded * x1
    if fm.c is not None:
        f_expanded = _expand_time(drift_envelope(fm, t_batch), x1)
        result = result + f_expanded * (projection(fm, x1) - x1)
    return result


def rdd_renoise(
    fm: VideoFlowMatchingWithDrift,
    x1: Tensor,
    t: float | Tensor,
    noise: Optional[Tensor] = None,
) -> Tensor:
    """Put a clean sample on the exact RDD path at data time ``t``."""

    t_batch = _as_batch_time(t, x1)
    if noise is None:
        noise = torch.randn_like(x1)
    if noise.shape != x1.shape:
        raise ValueError("noise and x1 must have identical shapes")
    sigma = 1.0 - t_batch
    return apply_rdd_signal_operator(fm, x1, t_batch) + _expand_time(sigma, x1) * noise


def recover_clean_from_velocity(
    fm: VideoFlowMatchingWithDrift,
    x_t: Tensor,
    velocity_sigma: Tensor,
    t: float | Tensor,
) -> Tensor:
    """Use the existing analytic RDD inverse to recover x1."""

    t_batch = _as_batch_time(t, x_t)
    sigma = 1.0 - t_batch
    return fm.predict_x1_from_velocity(x_t, velocity_sigma, sigma)


def exact_dmd_path_correction(
    fm: VideoFlowMatchingWithDrift,
    clean_prediction_delta: Tensor,
    t: float | Tensor,
) -> Tensor:
    """Map a clean-prediction difference to the RDD generator gradient.

    For ``x_t=A_t x1+sigma*eps``, the score difference induced by two clean
    estimators is proportional to ``A_t delta``.  Back-propagating through the
    re-noising path contributes another ``A_t^T``.  The latent avg/repeat P is
    a symmetric idempotent projection, hence this is ``A_t(A_t(delta))``.
    The global ``1/sigma^2`` scalar is deliberately omitted because DMD/DMD2
    normalize the gradient magnitude per batch.
    """

    first = apply_rdd_signal_operator(fm, clean_prediction_delta, t)
    return apply_rdd_signal_operator(fm, first, t)


def normalized_dmd_surrogate(
    generated_x1: Tensor,
    teacher_x1: Tensor,
    fake_x1: Tensor,
    *,
    fm: VideoFlowMatchingWithDrift,
    score_t: float | Tensor,
    path_correction: str = "exact",
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """DMD2 stop-gradient surrogate and the normalized target gradient.

    ``fake_x1-teacher_x1`` follows the official DMD2 clean-prediction form.
    ``path_correction='exact'`` adds the RDD signal-operator correction;
    ``'simple'`` reproduces the empirically common uncorrected approximation.
    """

    delta = fake_x1 - teacher_x1
    if path_correction == "exact":
        delta = exact_dmd_path_correction(fm, delta, score_t)
    elif path_correction != "simple":
        raise ValueError("path_correction must be 'exact' or 'simple'")
    denom = (generated_x1 - teacher_x1).abs().mean().clamp_min(float(eps))
    grad = torch.nan_to_num(delta / denom)
    target = (generated_x1 - grad).detach()
    loss = 0.5 * F.mse_loss(generated_x1.float(), target.float())
    return loss, grad.detach()


@torch.no_grad()
def integrate_teacher_interval(
    velocity_fn: VelocityFn,
    x_start: Tensor,
    t_start: float,
    t_end: float,
    *,
    substeps: int = 8,
    solver: str = "heun",
) -> Tensor:
    """Integrate a teacher within one stage; never performs a transition."""

    if substeps <= 0:
        raise ValueError("substeps must be positive")
    if not t_end > t_start:
        raise ValueError("teacher interval must have positive length")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")

    x = x_start
    dt = (float(t_end) - float(t_start)) / int(substeps)
    for index in range(int(substeps)):
        t0 = float(t_start) + index * dt
        t0_batch = _as_batch_time(t0, x)
        # Wan velocity is dx/dsigma, while data time increases: dx/dt=-w.
        k1 = -velocity_fn(x, t0_batch)
        if solver == "euler":
            x = x + dt * k1
        else:
            proposed = x + dt * k1
            t1_batch = _as_batch_time(t0 + dt, proposed)
            k2 = -velocity_fn(proposed, t1_batch)
            x = x + 0.5 * dt * (k1 + k2)
    return x


def student_euler_step(
    velocity_fn: VelocityFn,
    x_start: Tensor,
    call: DistillationCall,
) -> tuple[Tensor, Tensor]:
    """One differentiable student Euler call and its velocity prediction."""

    t_batch = _as_batch_time(call.t_start, x_start)
    velocity = velocity_fn(x_start, t_batch)
    return x_start - call.dt * velocity, velocity


def trajectory_distillation_loss(
    student_velocity_fn: VelocityFn,
    teacher_velocity_fn: VelocityFn,
    x_start: Tensor,
    call: DistillationCall,
    *,
    teacher_substeps: int = 8,
    teacher_solver: str = "heun",
) -> tuple[Tensor, Tensor, Tensor]:
    """Match one student call to a many-step teacher endpoint."""

    with torch.no_grad():
        teacher_end = integrate_teacher_interval(
            teacher_velocity_fn,
            x_start,
            call.t_start,
            call.t_end,
            substeps=teacher_substeps,
            solver=teacher_solver,
        )
    student_end, _ = student_euler_step(student_velocity_fn, x_start, call)
    loss = F.mse_loss(student_end.float(), teacher_end.float())
    return loss, student_end, teacher_end


def parse_positive_ints(value: str | Iterable[int], expected: Optional[int] = None) -> tuple[int, ...]:
    """Parse CLI values such as ``'2,2,1'`` with strict validation."""

    if isinstance(value, str):
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result or any(item <= 0 for item in result):
        raise ValueError("step allocation must contain positive integers")
    if expected is not None and len(result) != int(expected):
        raise ValueError(f"expected {expected} step counts, got {len(result)}")
    return result

