"""
RDD-Video Flow Matching with Spatiotemporal Drift for Wan 2.1.

The paper is written in data time t (t=0 noise, t=1 data), while a pretrained
Wan checkpoint is parameterized by noise time sigma=1-t (sigma=1 noise,
sigma=0 data).  This module keeps the paper's t for constructing the path but
trains the network to predict dx/dsigma and feeds sigma*1000 to Wan.  Keeping
that change of variables explicit is required for pretrained-Wan fine-tuning.

RDD path in data time:
- The time axis t∈[0,1] is split into K stages (analogous to DDPM's [0,1000])
- Each stage operates at a different spatiotemporal resolution
- Drift coefficient f(t) = t*(s_{k+1} - t)/L_k is STAGE-LOCAL with proper boundaries:
    - f(s_{k+1}) = 0  → no drift at data end (clean output)
    - f(s_k) = s_k    → max drift at noise end (biases toward low-freq)

Wan 2.1 Convention:
  - sigma=1-t; sigma=1 is noise and sigma=0 is data
  - x_sigma = (1-sigma)*x_1 + sigma*epsilon
  - Model predicts w = dx/dsigma = epsilon - x_1
  - Standard clean recovery is x_1 = x_sigma - sigma*w

RDD Extension (per-stage drift):
  - Forward: x_t = t*x_1 + (1-t)*ε + f(t)*d
    where f(t) = t*(s_{k+1} - t)/L_k, L_k = s_{k+1} - s_k
  - Paper velocity: v_t = dx/dt = (x_1-epsilon) + f'(t)*d
  - Wan target: w_sigma = dx/dsigma = -v_t
  - Key property: f(s_{k+1})=0 guarantees ODE endpoint = x_1 (no drift residual)

Stage transition:
  At boundary t=s_{k+1}, f=0, so ODE output ≈ x_1.
  Then: x_new = s_{k+1}*UP(pred_x1) + (1-s_{k+1})*ε_new
  Continue ODE in next stage from t=s_{k+1}.

The exact clean inverse implemented here assumes the "latent" operator:
average-pool followed by nearest-neighbor repetition. Pixel downsampling followed
by VAE re-encoding is nonlinear and requires an x-prediction head or another
transform-specific inverse; the training entry point therefore rejects it.

Designed for 5D video latents: [B, C, F, H, W] where
  C=16 (Wan-VAE channels), F=frames_latent, H=height/8, W=width/8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ─── Downsampling/Upsampling in Latent Space ─────────────────────────────────

class SpatioTemporalDownsample:
    """Joint spatial-temporal downsampling via average pooling on 5D latent tensors."""

    def __init__(self, spatial_factor: int = 2, temporal_factor: int = 2):
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video latent
        Returns:
            [B, C, F//tau, H//s, W//s]
        """
        B, C, F, H, W = x.shape
        s = self.spatial_factor
        tau = self.temporal_factor

        if tau > 1 and F >= tau:
            pad_f = (tau - F % tau) % tau
            if pad_f > 0:
                x = F_pad_temporal(x, pad_f)
                F = F + pad_f
            x = x.view(B, C, F // tau, tau, H, W).mean(dim=3)
            F = F // tau

        if s > 1 and H >= s and W >= s:
            pad_h = (s - H % s) % s
            pad_w = (s - W % s) % s
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
                H = H + pad_h
                W = W + pad_w
            x = x.view(B, C, x.shape[2], H // s, s, W // s, s).mean(dim=(4, 6))

        return x


def F_pad_temporal(x, pad_f):
    """Replicate-pad temporal dimension."""
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_f), mode='replicate')


class SpatioTemporalUpsample:
    """Joint spatial-temporal upsampling via nearest-neighbor repetition on 5D tensors."""

    def __init__(self, spatial_factor: int = 2, temporal_factor: int = 2):
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] downsampled video latent
        Returns:
            [B, C, F*tau, H*s, W*s]
        """
        tau = self.temporal_factor
        s = self.spatial_factor

        if tau > 1:
            x = x.repeat_interleave(tau, dim=2)
        if s > 1:
            x = x.repeat_interleave(s, dim=3).repeat_interleave(s, dim=4)

        return x


# ─── Core Flow Matching Module (per-stage, mirrors 2D version) ───────────────

class VideoFlowMatchingWithDrift:
    """
    RDD-Video Flow Matching module for a single resolution stage.

    Mirrors the 2D FlowMatchingWithDrift exactly:
    - Time range: t ∈ [s_k, s_{k+1})
    - Drift coefficient: f(t) = t*(s_{k+1} - t)/L_k
    - Forward: x_t = t*x_1 + (1-t)*ε + f(t)*d
    - Wan velocity target: w_sigma = (epsilon-x_1) - f'(t)*d
    - Coarsest stage (c=None): no drift, standard FM

    Args:
        c: stage index (None for coarsest = no drift)
        spatial_factor: downsampling factor for drift computation
        temporal_factor: temporal downsampling factor for drift computation
        s_k: start of this stage's time interval
        s_k_plus_1: end of this stage's time interval
        train_eps: small epsilon to avoid boundary singularities
    """

    def __init__(
        self,
        *,
        c: Optional[int] = None,
        spatial_factor: int = 2,
        temporal_factor: int = 2,
        s_k: float = 0.0,
        s_k_plus_1: float = 1.0,
        train_eps: float = 1e-5,
        drift_scale: float = 1.0,
    ):
        self.c = c
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        self.s_k = s_k
        self.s_k_plus_1 = s_k_plus_1
        self.stage_length = s_k_plus_1 - s_k
        self.train_eps = train_eps
        self.drift_scale = float(drift_scale)

        if c is not None:
            self.down = SpatioTemporalDownsample(spatial_factor, temporal_factor)
            self.up = SpatioTemporalUpsample(spatial_factor, temporal_factor)

    def compute_drift(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        d = STUp(STDown(x_1)) - x_1
        This is the (negative) high-frequency residual removed by downsampling.
        """
        down = self.down(x_1)
        up = self.up(down)
        # Ensure shape matches (handle padding edge cases)
        if up.shape != x_1.shape:
            up = up[:, :, :x_1.shape[2], :x_1.shape[3], :x_1.shape[4]]
        return up - x_1

    def q_sample(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward interpolation: x_t = t*x_1 + (1-t)*ε + f(t)*d

        where f(t) = t*(s_{k+1} - t)/L_k

        Boundary conditions:
          - At t=s_{k+1} (data end): f=0, x_t = s_{k+1}*x_1 + (1-s_{k+1})*ε → approaches x_1
          - At t=s_k (noise end): f = s_k*(s_{k+1}-s_k)/L_k = s_k
            → signal biased toward UP(DOWN(x_1)) (low-pass)
        """
        if noise is None:
            noise = torch.randn_like(x_1)

        t_expanded = t.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

        # Standard FM interpolation
        x_t = t_expanded * x_1 + (1 - t_expanded) * noise

        # Add drift: f(t) = t*(s_{k+1} - t)/L_k
        if d is not None:
            tau_rev = (self.s_k_plus_1 - t) / self.stage_length  # ∈ (0, 1]
            tau_rev_expanded = tau_rev.view(-1, 1, 1, 1, 1)
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
        Wan velocity target w_sigma = dx/dsigma with sigma=1-t.

        Since dx/dsigma = -dx/dt,

            w_sigma = (epsilon - x_1) - f'(t) d,
            f'(t) = (s_{k+1} - 2t)/L_k.

        The drift contribution is:
          - Positive for t < s_{k+1}/2: drift increasing
          - Negative for t > s_{k+1}/2: drift decreasing (being removed)
          - Zero at t = s_{k+1}/2: inflection point
        """
        u_t = noise - x_1

        if d is not None:
            t_expanded = t.view(-1, 1, 1, 1, 1)
            # -f'(t) = (2t - s_{k+1}) / L_k
            drift_coeff = (self.s_k_plus_1 - 2 * t_expanded) / self.stage_length
            u_t = u_t - drift_coeff * d

        return u_t

    def training_losses(
        self,
        model_fn,
        x_1: torch.Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute FM training loss for this stage.

        Samples t uniformly within [s_k + eps, s_{k+1} - eps],
        mirroring how the DDPM version samples within its stage interval.

        Args:
            model_fn: callable(x_list, t, **kwargs) -> List[Tensor] or Tensor
                For Wan: x_list is List[Tensor[C,F,H,W]], time is sigma*1000.
            x_1: [B, C, F, H, W] clean video latent at this stage's resolution
            model_kwargs: extra kwargs (context, seq_len, etc.)
        Returns:
            dict with "loss" [B] and "t" [B]
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x_1.shape[0]
        device = x_1.device

        # Sample t ~ Uniform[s_k + eps, s_{k+1} - eps]
        t_min = self.s_k + self.train_eps
        t_max = self.s_k_plus_1 - self.train_eps
        t = torch.rand(B, device=device) * (t_max - t_min) + t_min

        noise = torch.randn_like(x_1)

        # Compute drift if this is a drift stage (not coarsest)
        d = None
        if self.c is not None:
            d = self.drift_scale * self.compute_drift(x_1)

        # Forward interpolation: x_t
        x_t = self.q_sample(x_1, t, noise=noise, d=d)

        # Velocity target: u_t
        u_t = self.compute_velocity_target(x_1, noise, t, d=d)

        # Wan is pretrained in noise time sigma=1-t, not the paper's data time.
        sigma = 1.0 - t
        t_input = sigma * 1000.0

        # Wan model input: list of [C,F,H,W], output: list of [C,F,H,W]
        x_list = [x_t[i] for i in range(B)]
        v_pred_list = model_fn(x_list, t_input, **model_kwargs)

        # Stack predictions back to batch tensor
        if isinstance(v_pred_list, list):
            v_pred = torch.stack(v_pred_list)
        else:
            v_pred = v_pred_list

        # MSE loss per sample
        loss = mean_flat((v_pred - u_t) ** 2)

        return {"loss": loss, "t": t, "sigma": sigma}

    def predict_x1_from_velocity(
        self,
        x_sigma: torch.Tensor,
        velocity_sigma: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean sample for the current residualized FM path.

        For standard rectified flow, ``y=x_sigma-sigma*w=x_1``.  With RDD,

            y = [I + c(t)(P-I)] x_1,
            c(t) = f(t) + sigma f'(t),  t=1-sigma.

        Average-pool followed by nearest-neighbor upsampling is an idempotent
        projection P.  Therefore the inverse is analytic:

            x_1 = P y + (I-P)y / (1-c).

        A nonlinear or non-idempotent transform needs its own inverse/solver or
        an x-prediction head; it must not silently use the standard FM formula.
        """
        sigma_expanded = sigma.view(-1, 1, 1, 1, 1).to(
            device=x_sigma.device, dtype=x_sigma.dtype
        )
        y = x_sigma - sigma_expanded * velocity_sigma
        if self.c is None:
            return y

        t = 1.0 - sigma_expanded
        f_t = t * (self.s_k_plus_1 - t) / self.stage_length
        f_prime_t = (self.s_k_plus_1 - 2.0 * t) / self.stage_length
        coeff = self.drift_scale * (f_t + sigma_expanded * f_prime_t)
        denom = 1.0 - coeff
        if torch.any(denom.abs() < 1e-4):
            raise RuntimeError(
                "RDD-FM clean reconstruction is singular because |1-c(t)| is too small. "
                "Change the stage boundaries/envelope or use an x-prediction head."
            )

        py = self.compute_drift(y) + y
        return py + (y - py) / denom

    @torch.no_grad()
    def sample_ode(
        self,
        model_fn,
        x_init: torch.Tensor,
        num_steps: int = 20,
        model_kwargs: Optional[dict] = None,
        cfg_scale: float = 1.0,
    ) -> dict:
        """
        ODE sampling (Euler) within this stage's data-time interval.

        Integrates from t=s_k to t=s_{k+1}.
        At t=s_k: x is noisy (from noise or from previous stage transition)
        At t=s_{k+1}: x should be close to clean data x_1

        Args:
            model_fn: callable(x_list, t_input, **kwargs) -> predictions
                If cfg_scale > 1, model_fn should accept additional 'uncond' kwarg
                OR caller should handle CFG externally.
            x_init: initial state at t=s_k
            num_steps: Euler steps within this stage
            model_kwargs: extra kwargs
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
        Returns:
            dict with "sample" (x at t=s_{k+1}) and "pred_x1" (predicted clean data)
        """
        if model_kwargs is None:
            model_kwargs = {}

        device = x_init.device
        B = x_init.shape[0]

        dt = self.stage_length / num_steps
        x = x_init.clone()

        for i in range(num_steps):
            t_val = self.s_k + i * dt
            t = torch.full((B,), t_val, device=device)
            sigma = 1.0 - t
            t_input = sigma * 1000.0

            x_list = [x[j] for j in range(B)]
            v = model_fn(x_list, t_input, **model_kwargs)
            if isinstance(v, list):
                v = torch.stack(v)

            # Wan predicts dx/dsigma and dsigma=-dt.
            x = x - v * dt

        # Recover x_1 using the RDD path inverse, not the standard FM shortcut.
        t_end = self.s_k_plus_1
        if t_end >= 1.0 - 1e-5:
            pred_x1 = x
        else:
            t_final = torch.full((B,), t_end, device=device)
            sigma_final = 1.0 - t_final
            t_input_final = sigma_final * 1000.0
            x_list = [x[j] for j in range(B)]
            v_final = model_fn(x_list, t_input_final, **model_kwargs)
            if isinstance(v_final, list):
                v_final = torch.stack(v_final)
            sigma_batch = sigma_final.to(dtype=x.dtype)
            pred_x1 = self.predict_x1_from_velocity(x, v_final, sigma_batch)

        return {"sample": x, "pred_x1": pred_x1}


# ─── Multi-Stage Schedule ────────────────────────────────────────────────────

class RDDVideoSchedule:
    """
    Multi-stage spatiotemporal resolution schedule for RDD-Video.

    Mirrors the 2D version's stage_lengths-based time partitioning:
      stage_lengths = [60, 350, 590] with diffusion_steps=1000
      → boundaries = [0.0, 0.06, 0.41, 1.0]
      → coarsest (÷4) at t∈[0.0, 0.06), middle (÷2) at t∈[0.06, 0.41), finest (÷1) at t∈[0.41, 1.0)
      
    This matches PyramidalWan's sigma distribution where:
      - Coarsest stage occupies a tiny time interval (high noise → needs few but cheap steps)
      - Finest stage occupies the largest interval (low noise → needs more steps but is most expensive)

    Convention (same as 2D):
      - stages are ordered coarsest-first for sampling
      - coarsest stage has NO drift (standard FM)
      - finer stages have drift relative to one-level-coarser resolution
      - sampling goes: coarsest (t: 0→s1) → upsample+noise → middle (t: s1→s2) → ... → finest (t: s_{K-1}→1)

    The analytic path currently supports latent avg-pool/repeat projection.
    ``downsample_mode`` is recorded for checkpoint provenance; the distributed
    training entry point rejects pixel/VAE re-encoding because it is nonlinear.
    """

    def __init__(
        self,
        stage_configs: List[dict],
        stage_lengths: List[int],
        diffusion_steps: int = 1000,
        downsample_mode: str = "latent",
        drift_scale: float = 1.0,
    ):
        """
        Args:
            stage_configs: list of dicts, coarsest first. Each has:
                - "spatial_factor": spatial downsampling factor from full resolution
                - "temporal_factor": temporal downsampling factor from full resolution
                - "num_steps": Euler steps for sampling this stage
                - "drift_spatial": spatial factor for drift computation (default 2)
                - "drift_temporal": temporal factor for drift computation (default 2)
            stage_lengths: list of ints summing to diffusion_steps.
                First entry = coarsest stage length (maps to lowest t interval).
                Example: [400, 300, 300] with diffusion_steps=1000
                → boundaries [0, 0.4, 0.7, 1.0]
            diffusion_steps: total steps (used to normalize stage_lengths to [0,1])
            downsample_mode: "latent" or "pixel"
        """
        assert len(stage_configs) == len(stage_lengths)
        assert sum(stage_lengths) == diffusion_steps

        self.stage_configs = stage_configs
        self.stage_lengths = list(stage_lengths)
        self.num_stages = len(stage_configs)
        self.downsample_mode = downsample_mode
        self.drift_scale = float(drift_scale)
        self.diffusion_steps = diffusion_steps

        # Compute time boundaries [0, s1, s2, ..., 1.0]
        T_boundaries = [0]
        for length in stage_lengths:
            T_boundaries.append(T_boundaries[-1] + length)
        self.s_boundaries = [T / diffusion_steps for T in T_boundaries]

        # Create flow matchers for each stage
        self.flow_matchers: List[VideoFlowMatchingWithDrift] = []
        for i, cfg in enumerate(stage_configs):
            s_k = self.s_boundaries[i]
            s_k_plus_1 = self.s_boundaries[i + 1]

            # Coarsest stage (first) has no drift
            if i == 0:
                c = None
            else:
                c = i

            fm = VideoFlowMatchingWithDrift(
                c=c,
                spatial_factor=cfg.get("drift_spatial", 2),
                temporal_factor=cfg.get("drift_temporal", 2),
                s_k=s_k,
                s_k_plus_1=s_k_plus_1,
                drift_scale=self.drift_scale,
            )
            self.flow_matchers.append(fm)

    def downsample_latent(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Downsample full-res latent to stage resolution (latent mode)."""
        cfg = self.stage_configs[stage_idx]
        s = cfg["spatial_factor"]
        tau = cfg["temporal_factor"]

        if s == 1 and tau == 1:
            return x

        down = SpatioTemporalDownsample(s, tau)
        return down(x)

    def upsample_between_stages(
        self, x: torch.Tensor, from_stage: int, to_stage: int
    ) -> torch.Tensor:
        """Upsample from one stage's resolution to the next (nearest-neighbor)."""
        from_cfg = self.stage_configs[from_stage]
        to_cfg = self.stage_configs[to_stage]

        s_ratio = from_cfg["spatial_factor"] // to_cfg["spatial_factor"]
        tau_ratio = from_cfg["temporal_factor"] // to_cfg["temporal_factor"]

        if s_ratio == 1 and tau_ratio == 1:
            return x

        up = SpatioTemporalUpsample(s_ratio, tau_ratio)
        return up(x)

    def upsample_with_noise(
        self,
        pred_x1: torch.Tensor,
        from_stage: int,
        to_stage: int,
        target_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Canonical noise-filling stage transition (mirrors the established 2-D
        RDD evaluator): upsample clean, then draw fresh full-space Gaussian.

        Formula:
            x_new = s_{k+1} * UP(pred_x1) + (1 - s_{k+1}) * ε_new

        where s_{k+1} is the time boundary between stages.
        This places x at the correct FM interpolation point for the next stage to start.
        """
        t_boundary = self.s_boundaries[to_stage]  # = s_{k+1} of from_stage

        upsampled = self.upsample_between_stages(pred_x1, from_stage, to_stage)
        if target_shape is not None:
            target_f, target_h, target_w = target_shape
            pad_f = max(target_f - upsampled.shape[2], 0)
            pad_h = max(target_h - upsampled.shape[3], 0)
            pad_w = max(target_w - upsampled.shape[4], 0)
            if pad_f or pad_h or pad_w:
                upsampled = torch.nn.functional.pad(
                    upsampled, (0, pad_w, 0, pad_h, 0, pad_f), mode="replicate"
                )
            upsampled = upsampled[:, :, :target_f, :target_h, :target_w]
        noise = torch.randn_like(upsampled)

        # FM: x_t = t*x_1 + (1-t)*ε at t = t_boundary
        return t_boundary * upsampled + (1 - t_boundary) * noise
