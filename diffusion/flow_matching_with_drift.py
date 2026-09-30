"""
RDD Flow Matching: Resolution Drift Diffusion with Flow Matching framework.

This implementation MIRRORS the DDPM version's time-axis partitioning:
- The time axis t∈[0,1] is split into K stages: [s_0, s_1), [s_1, s_2), ..., [s_{K-1}, s_K=1]
- Each stage operates at a different spatial resolution
- At stage boundaries, we upsample and add noise (analogous to DDPM's upsample_with_noise)

Mathematical formulation (per stage k with drift):
  Time range: t ∈ [s_k, s_{k+1})
  Drift coefficient: f(t) = t·(s_{k+1} - t) / L_k  where L_k = s_{k+1} - s_k
  
  Forward interpolation (with drift):
      x_t = t·x_1 + (1-t)·ε + f(t)·d
    where d = UP(DOWN(x_1)) - x_1 (low-pass residual)

  Key properties of f(t):
    - f(s_k)     = s_k · 1 = s_k  (max drift at noise end)
    - f(s_{k+1}) = s_{k+1} · 0 = 0  (no drift at data end → x_1 is clean)
    - f'(t) = (s_{k+1} - 2t) / L_k  (negative for t > s_{k+1}/2: drift shrinks)

  Velocity field target:
      u_t = (x_1 - ε) + (s_{k+1} - 2t)/L_k · d

  Training loss:
      L = E_t[ ||v_θ(x_t, t) - u_t||² ]

  Sampling ODE:
      dx/dt = v_θ(x_t, t),  integrate from t=s_k to t=s_{k+1}

For coarsest stage (no drift, d=0): standard Flow Matching:
    x_t = t·x_1 + (1-t)·ε,  u_t = x_1 - ε

Stage transition (at boundary s_{k+1}):
    1. ODE integration reaches t=s_{k+1}; at this point f(s_{k+1})=0
    2. Use FM Tweedie: pred_x1 = x + (1-t)·v_θ (exact when drift=0 at boundary)
    3. Upsample pred_x1 to next resolution (nearest-neighbor)
    4. Add noise: x_{new} = s_{k+1}·UP(pred_x1) + (1-s_{k+1})·ε_new
    5. Continue ODE in next stage

Comparison with DDPM version:
    DDPM: κ(T)=(T-T_c)/(T_{c+1}-T_c), maximal at noisy end (large T), zero at clean end (T=0)
    FM:   τ_rev=(s_{k+1}-t)/L_k, maximal at noisy end (small t=s_k), zero at clean end (t=s_{k+1})
    Both achieve: drift is strongest when the signal is most contaminated by noise,
    and vanishes completely when the model outputs clean data.

References:
  - SiT (ECCV 2024): Flow Matching with DiT backbone
  - RDD: Resolution Drift Diffusion (this project)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class FlowMatchingWithDrift:
    """
    RDD Flow Matching module with time-axis partitioning.

    Mirrors the DDPM version: the full time axis is split into stages,
    each stage operates at a different resolution.

    Args:
        c: stage index (None for coarsest = no drift, 0,1,... for finer stages)
        sample_scale: downsampling factor for computing drift
        s_k: start of this stage's time interval (e.g. 0.0, 0.3, 0.6)
        s_k_plus_1: end of this stage's time interval (e.g. 0.3, 0.6, 1.0)
        train_eps: small epsilon to avoid t=0/1 singularities
    """

    def __init__(
        self,
        *,
        c: Optional[int] = None,
        sample_scale: int = 2,
        s_k: float = 0.0,
        s_k_plus_1: float = 1.0,
        train_eps: float = 1e-5,
    ):
        self.c = c
        self.sample_scale = sample_scale
        self.s_k = s_k
        self.s_k_plus_1 = s_k_plus_1
        self.stage_length = s_k_plus_1 - s_k
        self.train_eps = train_eps

    def downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Average pooling downsample by factor `scale`."""
        B, C, H, W = x.shape
        assert H % scale == 0 and W % scale == 0
        return x.view(B, C, H // scale, scale, W // scale, scale).mean(dim=(3, 5))

    def upsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Nearest-neighbor upsample by factor `scale`."""
        return x.repeat_interleave(scale, dim=-1).repeat_interleave(scale, dim=-2)

    def compute_drift(self, x_1: torch.Tensor) -> torch.Tensor:
        """Compute drift d = UP(DOWN(x_1)) - x_1."""
        return self.upsample(self.downsample(x_1, self.sample_scale), self.sample_scale) - x_1

    def q_sample(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute x_t from the forward interpolation path.

        x_t = t·x_1 + (1-t)·ε + t·(s_{k+1} - t)/L_k · d

        where:
          - d = UP(DOWN(x_1)) - x_1 (low-pass residual, i.e., negative high-freq)
          - L_k = s_{k+1} - s_k (stage length)
          - f(t) = t·(s_{k+1} - t)/L_k is the drift coefficient

        Key boundary conditions:
          - At t = s_{k+1} (data end): f = 0, so x_t = x_1 (clean data, no drift) ✓
          - At t = s_k (noise end): f = s_k·1 = s_k, so signal = x_1 + s_k·d
            which biases toward UP(DOWN(x_1)) (low-pass), matching the coarser
            stage's output.

        This mirrors the DDPM version where:
            x_t = √ᾱ_t·(x_0 + κ(t)·d) + √(1-ᾱ_t)·ε
        with κ(T) = (T-T_c)/(T_{c+1}-T_c) ∈ [0,1), and κ=0 at the clean-data
        end (T=0). In DDPM, drift is maximal at the noisy end and vanishes at
        the clean end — our reversed τ achieves the same behavior in FM.

        Args:
            x_1: data [B, C, H, W]
            t: time [B] in [s_k, s_{k+1}]
            noise: optional pre-sampled noise ε
            d: optional pre-computed drift
        Returns:
            x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_1)

        t_expanded = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # Standard FM interpolation
        x_t = t_expanded * x_1 + (1 - t_expanded) * noise

        # Add drift: f(t) = t·(s_{k+1} - t)/L_k
        # At t=s_{k+1}: f=0 (no drift at data end) → x_t = x_1
        # At t=s_k: f=s_k (max drift at noise end)
        if d is not None:
            tau_rev = (self.s_k_plus_1 - t) / self.stage_length  # reversed τ ∈ (0, 1]
            tau_rev_expanded = tau_rev.view(-1, 1, 1, 1)
            x_t = x_t + t_expanded * tau_rev_expanded * d

        return x_t

    def compute_velocity_target(
        self,
        x_1: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the velocity field target u_t = dx_t/dt.

        Given x_t = t·x_1 + (1-t)·ε + f(t)·d
        where f(t) = t·(s_{k+1} - t)/L_k

        u_t = dx_t/dt = (x_1 - ε) + f'(t)·d

        f'(t) = d/dt[ (t·s_{k+1} - t²) / L_k ] = (s_{k+1} - 2t) / L_k

        So: u_t = (x_1 - ε) + (s_{k+1} - 2t)/L_k · d

        Note: the drift contribution to velocity is NEGATIVE for t > s_{k+1}/2,
        meaning the ODE naturally "removes" the drift as t increases toward s_{k+1}.
        This mirrors DDPM where drift vanishes as we denoise toward clean data.

        Args:
            x_1: data [B, C, H, W]
            noise: sampled noise ε [B, C, H, W]
            t: time [B] in [s_k, s_{k+1}]
            d: drift [B, C, H, W] or None
        Returns:
            u_t [B, C, H, W]
        """
        u_t = x_1 - noise

        if d is not None:
            t_expanded = t.view(-1, 1, 1, 1)
            # f'(t) = (s_{k+1} - 2t) / L_k
            drift_coeff = (self.s_k_plus_1 - 2 * t_expanded) / self.stage_length
            u_t = u_t + drift_coeff * d

        return u_t

    def predict_x1_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover clean ``x_1`` from the RDD-FM state and velocity.

        For ``P = UP(DOWN(.))`` and

        ``x_t = t*x_1 + (1-t)*eps + b(t)*(P-I)x_1``,

        let ``y = x_t + (1-t)*velocity`` and
        ``c = b(t) + (1-t)b'(t)``.  Because average-pool/repeat is an
        idempotent projection, ``P y = P x_1`` and
        ``(I-P)y = (1-c)(I-P)x_1``.  This is the drift-aware analogue of the
        standard FM identity ``x_1 = x_t + (1-t)v``.

        The method is checkpoint-compatible: it changes only inference and
        uses exactly the probability path on which the legacy v2 checkpoint
        was trained.
        """
        t_expanded = t.view(-1, 1, 1, 1).to(dtype=x_t.dtype)
        y = x_t + (1.0 - t_expanded) * velocity
        if self.c is None:
            return y

        b = t_expanded * (self.s_k_plus_1 - t_expanded) / self.stage_length
        b_prime = (self.s_k_plus_1 - 2.0 * t_expanded) / self.stage_length
        c = b + (1.0 - t_expanded) * b_prime
        denom = 1.0 - c
        if torch.any(denom.abs() < 1e-6):
            raise RuntimeError("RDD-FM clean inverse is singular at the requested time")

        p_y = self.upsample(self.downsample(y, self.sample_scale), self.sample_scale)
        return p_y + (y - p_y) / denom

    def _model_velocity(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Evaluate the legacy velocity model with all latent channels guided."""
        channels = x.shape[1]
        if cfg_scale != 1.0 and "y" in model_kwargs:
            y_cond = model_kwargs["y"]
            y_uncond = torch.full_like(y_cond, 1000)
            extra = {k: v for k, v in model_kwargs.items() if k != "y"}
            # A single batched call has the same semantics as two CFG calls and
            # matches the memory layout used by the existing diffusion sampler.
            out = model(
                torch.cat([x, x], dim=0),
                torch.cat([t, t], dim=0) * 1000.0,
                y=torch.cat([y_cond, y_uncond], dim=0),
                **extra,
            )
            if out.shape[1] == channels * 2:
                out = out[:, :channels]
            v_cond, v_uncond = out.chunk(2, dim=0)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        out = model(x, t * 1000.0, **model_kwargs)
        if out.shape[1] == channels * 2:
            out = out[:, :channels]
        return out

    def training_losses(
        self,
        model: nn.Module,
        x_1: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        t: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute Flow Matching training losses for this stage.

        Samples t uniformly within this stage's time interval [s_k, s_{k+1}],
        mirroring how the DDPM version samples t within [T_c, T_{c+1}).

        Args:
            model: velocity prediction network v_θ(x_t, t, ...)
            x_1: data batch [B, C, H, W]
            model_kwargs: extra kwargs (e.g. class labels)
        Returns:
            dict with "loss" containing [B] per-sample losses
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x_1.shape[0]
        device = x_1.device

        # Sample t ~ Uniform[s_k + eps, s_{k+1} - eps] by default.  Supplying
        # t enables optional stage-local importance distributions without
        # changing the legacy RNG path.
        t_min = self.s_k + self.train_eps
        t_max = self.s_k_plus_1 - self.train_eps
        if t is None:
            t = torch.rand(B, device=device) * (t_max - t_min) + t_min
        else:
            if t.shape != (B,):
                raise ValueError(f"Expected t shape {(B,)}, got {tuple(t.shape)}")
            t = t.to(device=device, dtype=x_1.dtype)
            if bool(((t < t_min) | (t > t_max)).any()):
                raise ValueError(
                    f"Provided FM times must lie in [{t_min}, {t_max}] for this stage"
                )

        # Sample noise
        noise = torch.randn_like(x_1)

        # Compute drift if this is a drift stage (not coarsest)
        d = None
        if self.c is not None:
            d = self.compute_drift(x_1)

        # Forward interpolation: x_t
        x_t = self.q_sample(x_1, t, noise=noise, d=d)

        # Velocity target: u_t
        u_t = self.compute_velocity_target(x_1, noise, t, d=d)

        # Model prediction
        # DiT expects integer timesteps; we convert t∈[0,1] to [0, 1000]
        t_input = t * 1000.0
        model_output = model(x_t, t_input, **model_kwargs)

        # If model outputs extra channels (learned sigma), take only first half
        if model_output.shape[1] == x_1.shape[1] * 2:
            model_output, _ = model_output.chunk(2, dim=1)

        # MSE loss
        loss = mean_flat((model_output - u_t) ** 2)

        return {"loss": loss}

    @torch.no_grad()
    def sample_ode(
        self,
        model: nn.Module,
        x_init: torch.Tensor,
        num_steps: int = 250,
        model_kwargs: Optional[dict] = None,
        cfg_scale: float = 1.0,
        solver: str = "euler",
    ) -> dict:
        """
        ODE sampling within THIS stage's time interval ``[s_k, s_{k+1}]``.

        Integrates from t=s_k to t=s_{k+1}.

        Args:
            model: velocity prediction network
            x_init: initial state at t=s_k (either pure noise for coarsest stage,
                    or upsampled+noised from previous stage)
            num_steps: number of Euler steps for this stage
            model_kwargs: extra kwargs for model
            cfg_scale: classifier-free guidance scale
            solver: ``euler`` (one evaluation/update) or ``midpoint``/``rk2``
                (two evaluations/update). ``dpm2`` is accepted only as a
                backwards-compatible alias for midpoint; it must be reported
                as midpoint RK2 rather than DPM-Solver in Flow Matching tables.
        Returns:
            dict with "sample" (x at t=s_{k+1}) and "pred_x1" (predicted clean data)
        """
        if model_kwargs is None:
            model_kwargs = {}

        device = x_init.device
        B = x_init.shape[0]

        solver = solver.lower()
        if solver == "dpm2":
            solver = "midpoint"
        if solver not in {"euler", "midpoint", "rk2"}:
            raise ValueError(f"Unknown FM ODE solver: {solver}")

        dt = self.stage_length / num_steps
        x = x_init.clone()
        nfe = 0

        for i in range(num_steps):
            t_val = self.s_k + i * dt
            t = torch.full((B,), t_val, device=device, dtype=x.dtype)
            v = self._model_velocity(model, x, t, model_kwargs, cfg_scale)
            nfe += 1

            if solver == "euler":
                x = x + v * dt
            else:
                t_mid = t + 0.5 * dt
                x_mid = x + 0.5 * dt * v
                v_mid = self._model_velocity(model, x_mid, t_mid, model_kwargs, cfg_scale)
                nfe += 1
                x = x + dt * v_mid

        # The final stage ends at t=1 and therefore directly returns x_1.  At
        # intermediate boundaries we make one endpoint evaluation and use the
        # exact drift-aware clean inverse above.  Boundary evaluations are
        # reported separately in nfe so matched-compute tables remain honest.
        t_end = self.s_k_plus_1
        if t_end >= 1.0 - 1e-5:
            pred_x1 = x
        else:
            t_final = torch.full((B,), t_end, device=device, dtype=x.dtype)
            v_final = self._model_velocity(model, x, t_final, model_kwargs, cfg_scale)
            nfe += 1
            pred_x1 = self.predict_x1_from_velocity(x, v_final, t_final)

        return {"sample": x, "pred_x1": pred_x1, "nfe": nfe}


def upsample_with_noise_fm(pred_x1: torch.Tensor, t_next: float,
                           target_size: int, mode: str = "nearest") -> torch.Tensor:
    """
    Upsample pred_x1 and add noise at level t_next.
    
    Mirrors the DDPM version's upsample_with_noise:
        x_{t_next} = t_next·UP(pred_x1) + (1-t_next)·ε_new

    This is the FM equivalent of:
        x_t = √ᾱ_t·UP(x_0) + √(1-ᾱ_t)·ε

    Args:
        pred_x1: predicted clean image from previous stage [B, C, H, W]
        t_next: time value at the start of next stage (e.g. 0.3)
        target_size: spatial size for next stage
        mode: 'nearest' (default, matches DDPM's repeat_interleave) or 'bilinear'
    Returns:
        x at time t_next in the new resolution [B, C, target_size, target_size]
    """
    current_size = pred_x1.shape[-1]
    scale = target_size // current_size

    if mode == "nearest":
        upsampled = pred_x1.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    else:
        upsampled = F.interpolate(pred_x1, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)

    noise = torch.randn_like(upsampled)
    # FM version of adding noise at time t_next:
    # x_{t_next} = t_next · x_1 + (1 - t_next) · ε
    return t_next * upsampled + (1 - t_next) * noise


def noise_filling_transition_fm(
    previous_state: torch.Tensor,
    previous_x1: torch.Tensor,
    t_boundary: float,
    target_size: int,
) -> torch.Tensor:
    """Lift an FM boundary state while filling only the new null space.

    Average-pool/repeat is an orthogonal projection ``P``.  A standard
    high-resolution Gaussian can be decomposed as an orthonormal lift of the
    coarse Gaussian plus an independent ``(I-P)`` component.  Reusing a
    normalized lift of the recovered coarse innovation carries its direction
    into the finer stage while giving that stage the identity covariance used
    during training.  Exact equality after downsampling is neither expected
    nor possible here because training used unit-variance noise independently
    at every resolution.

    ``previous_state`` is at the previous stage endpoint, where its drift
    coefficient is zero: ``x_s = s*x_1 + (1-s)*eps``.
    """
    if not (0.0 < t_boundary < 1.0):
        raise ValueError(f"FM stage boundary must lie in (0, 1), got {t_boundary}")
    current_size = previous_x1.shape[-1]
    if target_size % current_size != 0:
        raise ValueError(f"Cannot lift {current_size} to {target_size}")
    scale = target_size // current_size

    signal_up = previous_x1.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
    eps_coarse = (previous_state - t_boundary * previous_x1) / (1.0 - t_boundary)

    # U/sqrt(scale^2) is an isometric embedding for a 2-D repeat lift.
    eps_low = eps_coarse.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
    eps_low = eps_low / float(scale)

    eps_high = torch.randn_like(signal_up)
    pooled = eps_high.view(
        eps_high.shape[0], eps_high.shape[1], current_size, scale, current_size, scale
    ).mean(dim=(3, 5))
    projected = pooled.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
    eps_null = eps_high - projected

    return t_boundary * signal_up + (1.0 - t_boundary) * (eps_low + eps_null)


def sample_rdd_fm_multistage(
    model: nn.Module,
    flow_matchers: List[FlowMatchingWithDrift],
    resolutions: List[int],
    num_steps_per_stage: List[int],
    batch_size: int,
    device: torch.device,
    model_kwargs: Optional[dict] = None,
    cfg_scale: float = 1.0,
    upsample_mode: str = "nearest",
    solver: str = "euler",
    return_metrics: bool = False,
) -> torch.Tensor:
    """
    Full multi-stage RDD-FM sampling pipeline.

    Mirrors the DDPM version's coarse→fine sampling:
    1. Start with noise at coarsest resolution, t=0
    2. ODE integrate within each stage
    3. At stage boundary: upsample + add noise, continue in next stage

    Args:
        model: unified DiTMultiRes model
        flow_matchers: list of FlowMatchingWithDrift (one per stage, coarsest first in sampling order)
        resolutions: list of resolutions in sampling order (coarsest first, e.g. [8, 16, 32])
        num_steps_per_stage: Euler steps per stage
        batch_size: number of samples
        device: torch device
        model_kwargs: extra kwargs for model
        cfg_scale: classifier-free guidance scale
        upsample_mode: ``noise_filling`` (recommended), ``nearest`` or
            ``bilinear``. The latter two reconstruct clean signal and redraw
            all noise, and are kept as compatibility baselines.
        solver: ``euler`` or ``midpoint``/``rk2``.
    Returns:
        Final generated samples [B, C, finest_res, finest_res]
    """
    if model_kwargs is None:
        model_kwargs = {}

    num_stages = len(resolutions)
    pred_x1 = None
    previous_state = None
    total_nfe = 0
    stage_nfe = []

    for stage_idx in range(num_stages):
        fm = flow_matchers[stage_idx]
        res = resolutions[stage_idx]
        n_steps = num_steps_per_stage[stage_idx]

        if stage_idx == 0:
            # First (coarsest) stage: start from pure noise at t=s_k=0
            x_init = torch.randn(batch_size, 4, res, res, device=device)
        else:
            # Lift the previous endpoint to the finer stage.  The preferred
            # path preserves the coarse innovation and fills only new degrees
            # of freedom; nearest/bilinear retain the old redraw-all-noise
            # behavior for ablation and checkpoint archaeology.
            if upsample_mode == "noise_filling":
                x_init = noise_filling_transition_fm(
                    previous_state, pred_x1, fm.s_k, res
                )
            elif upsample_mode in {"nearest", "bilinear"}:
                x_init = upsample_with_noise_fm(
                    pred_x1, fm.s_k, res, mode=upsample_mode
                )
            else:
                raise ValueError(f"Unknown FM upsample mode: {upsample_mode}")

        # ODE integrate within this stage
        out = fm.sample_ode(
            model, x_init, num_steps=n_steps,
            model_kwargs=model_kwargs, cfg_scale=cfg_scale,
            solver=solver,
        )

        previous_state = out["sample"]
        pred_x1 = out["pred_x1"]
        total_nfe += out["nfe"]
        stage_nfe.append(out["nfe"])

    if return_metrics:
        return pred_x1, {"nfe": total_nfe, "stage_nfe": stage_nfe}
    return pred_x1  # At the final stage (t→1), pred_x1 = generated sample
