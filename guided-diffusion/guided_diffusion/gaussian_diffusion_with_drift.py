"""
Gaussian diffusion with multi-scale drift.

Extends the standard GaussianDiffusion class with a drift term derived from
the difference between a coarser-scale reconstruction and the original image:

    d = upsample(downsample(x_0, scale), scale) - x_0

The modified forward process becomes:
    q(x_t | x_0) = N(sposterior_upsample_between_scalesqrt(ā_t) * (x_0 + (t-T_c)/(T_c+1-T_c) * d),
                      (1 - ā_t) * I)

The noise target in training is shifted accordingly:
    ε̃ = ε + sqrt(ā_t / (1-ā_t)) * (t-T_c)/(T_c+1-T_c) * d

When c is None (coarsest scale) the class behaves identically to the
standard GaussianDiffusion.
"""

import numpy as np
import torch as th
from torch.distributions import MultivariateNormal

from .gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    _extract_into_tensor,
    get_named_beta_schedule,
    betas_for_alpha_bar,
)
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


class GaussianDiffusionWithDrift(GaussianDiffusion):
    """
    Gaussian diffusion augmented with a multi-scale drift term.

    :param c: scale index (0, 1, …).  When None no drift is applied and the
              class is identical to GaussianDiffusion.
    :param T_c: global timestep at which scale-c's sub-interval starts.
    :param T_c_plus_1: global timestep at which scale-c's sub-interval ends.
    :param sample_scale: spatial downsampling factor used to compute d
                         (default 2, i.e. 2× average-pool then 2× repeat).
    :param timestep_map: list of global timestep indices that map the local
                         indices [0, num_timesteps) to global values.
                         Set automatically by SpacedDiffusionWithDrift.
    :param rescale_timesteps: kept for API compatibility; always False here
                              because timestep remapping is done by the wrapper.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        c=None,
        T_c=None,
        T_c_plus_1=None,
        sample_scale=2,
        timestep_map=None,
    ):
        # GaussianDiffusion.__init__ sets all the standard schedules.
        super().__init__(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=False,   # handled by wrapper
        )

        self.c = c
        self.T_c = T_c
        self.T_c_plus_1 = T_c_plus_1
        self.sample_scale = sample_scale
        self.timestep_map = timestep_map  # set by SpacedDiffusionWithDrift

        # Pre-compute the drift coefficient for the posterior mean.
        # posterior_mean_coef_drift[t] is the coefficient of d in
        #   q(x_{t-1} | x_t, x_0)  when drift is active.
        if self.c is not None and T_c is not None and T_c_plus_1 is not None:
            alphas = 1.0 - betas
            t_idx = np.arange(0, self.num_timesteps)
            # If timestep_map is already available use it; otherwise assume
            # identity (will be overwritten by SpacedDiffusionWithDrift).
            if timestep_map is not None:
                t_global = np.array(timestep_map)[t_idx]
            else:
                t_global = t_idx

            self.posterior_mean_coef_drift = (
                np.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
                * ((t_global - T_c) - alphas * (t_global - T_c) - alphas + self.alphas_cumprod)
                * 1.0 / (T_c_plus_1 - T_c)
            )
        else:
            self.posterior_mean_coef_drift = None

    # ------------------------------------------------------------------
    # Helpers for spatial down/up-sampling
    # ------------------------------------------------------------------

    def downsample(self, x, scale):
        """Average-pool x by *scale* in both spatial dims."""
        if scale == 1:
            return x
        B, C, H, W = x.shape
        assert H % scale == 0 and W % scale == 0
        x = x.view(B, C, H // scale, scale, W // scale, scale)
        return x.mean(dim=(3, 5))

    def upsample(self, x, scale):
        """Nearest-neighbour upsample x by *scale* in both spatial dims."""
        if scale == 1:
            return x
        x = x.repeat_interleave(scale, dim=2)
        x = x.repeat_interleave(scale, dim=3)
        return x

    def _drift(self, x_start):
        """Compute d = upsample(downsample(x_0)) - x_0."""
        return self.upsample(self.downsample(x_start, self.sample_scale), self.sample_scale) - x_start

    def _global_t(self, t):
        """Map local timestep indices to global values via self.timestep_map."""
        assert self.timestep_map is not None, \
            "timestep_map must be set before using drift (done by SpacedDiffusionWithDrift)"
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        return map_tensor[t]

    # ------------------------------------------------------------------
    # Modified forward-process methods
    # ------------------------------------------------------------------

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0) including drift when c is not None.
        Returns (mean, variance, log_variance).
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        if self.c is not None:
            d = self._drift(x_start)
            t_global = self._global_t(t).view(-1, 1, 1, 1)
            kappa = (t_global - self.T_c) / (self.T_c_plus_1 - self.T_c)
            mean = mean + _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * kappa * d
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0) with drift when c is not None.

        x_t = sqrt(ā_t) * (x_0 + κ_t * d) + sqrt(1-ā_t) * ε
        where κ_t = (global_t - T_c) / (T_c+1 - T_c).
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        sqrt_alphas = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        if self.c is not None:
            d = self._drift(x_start)
            t_global = self._global_t(t).view(-1, 1, 1, 1)
            kappa = (t_global - self.T_c) / (self.T_c_plus_1 - self.T_c)
            return sqrt_alphas * (x_start + kappa * d) + sqrt_one_minus * noise
        else:
            return sqrt_alphas * x_start + sqrt_one_minus * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of q(x_{t-1} | x_t, x_0) with drift.
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if self.c is not None:
            d = self._drift(x_start)
            posterior_mean = posterior_mean + _extract_into_tensor(
                self.posterior_mean_coef_drift, t, x_t.shape
            ) * d

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses.

        When drift is active the noise target is shifted:
            ε̃ = ε + sqrt(ā_t / (1-ā_t)) * κ_t * d
        so the model learns to predict ε̃ instead of plain Gaussian noise.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        if self.c is not None:
            d = self._drift(x_start)
            x_t = self.q_sample(x_start, t, noise=noise)
            t_global = self._global_t(t)
            kappa = ((t_global - self.T_c) / (self.T_c_plus_1 - self.T_c)).reshape(-1, 1, 1, 1)
            sqrt_ratio = th.sqrt(
                _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
                / (1.0 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape))
            )
            noise_target = noise + sqrt_ratio * kappa * d
        else:
            x_t = self.q_sample(x_start, t, noise=noise)
            noise_target = noise

        terms = {}

        if self.loss_type in (LossType.KL, LossType.RESCALED_KL):
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise_target,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    # ------------------------------------------------------------------
    # Sampling helpers (drift-aware)
    # ------------------------------------------------------------------

    def _predict_pure_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Given predicted x_0, recover the *pure* Gaussian noise ε (without
        the drift component ε̃ = ε + drift_shift).

        Used in drift-aware DDIM sampling to decouple ε from d.
        """
        d_approx = self._drift(pred_xstart)
        t_expanded = t.int().view(-1, 1, 1, 1)
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        kappa = (map_tensor[t_expanded] - self.T_c) / (self.T_c_plus_1 - self.T_c)
        epsilon_tilde_minus_epsilon = (
            kappa * d_approx
            / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) - epsilon_tilde_minus_epsilon

    def _ddim_prev_kappa(self, t, x_shape):
        """
        Drift coefficient κ for the *previous retained step* in DDIM sampling.

        For local timestep index t, DDIM uses alpha_bar_prev, i.e. the previous
        retained point on the respaced schedule. The drift term must use the
        same previous global timestep rather than the DDPM posterior coefficient.
        """
        assert self.c is not None
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        prev_global = th.where(t > 0, map_tensor[t - 1], th.full_like(t, self.T_c))
        kappa_prev = (prev_global - self.T_c) / (self.T_c_plus_1 - self.T_c)
        return kappa_prev.view(-1, *([1] * (len(x_shape) - 1)))

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        DDIM sampling step, drift-aware when c is not None.

        When drift is active the pure noise ε is decoupled from d so that
        the DDIM trajectory follows the same drifted forward process:
            x_s = sqrt(ā_s) * (x_0 + κ_s d) + sqrt(1-ā_s) * ε
        rather than reusing the DDPM posterior drift coefficient.
        """
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        if self.c is None:
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        else:
            eps = self._predict_pure_eps_from_xstart(x, t, out["pred_xstart"])
            d_approx = self._drift(out["pred_xstart"])

        alpha_bar      = _extract_into_tensor(self.alphas_cumprod,      t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        if self.c is not None:
            mean_pred = mean_pred + th.sqrt(alpha_bar_prev) * self._ddim_prev_kappa(
                t, out["pred_xstart"].shape
            ) * d_approx

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_and_upsample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        p_sample_output = self.p_sample(model, x, t, clip_denoised, denoised_fn, cond_fn, model_kwargs)
        pred_xstart = p_sample_output["pred_xstart"]
        sample = p_sample_output["sample"]
        sample = self.posterior_upsample_between_scales(pred_xstart, sample, t)
        return sample

    def posterior_upsample_between_scales(self, pred_xstart, sample, t):
        mu, Sigma, sample = self.compute_posterior_x1(
            x0_c=pred_xstart,
            x1_c=sample,
            a=_extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape),
            b=_extract_into_tensor(1.0 - self.alphas_cumprod, t, sample.shape),
            sample=True
        )
        return {"sample": sample}

    def compute_posterior_x1(
        self,
        x0_c,                # (B, C, H, W)
        x1_c,                # (B, C, H, W) - current low-res sample at stage c
        a,                   # tensor: sqrt(alpha_bar) at time t
        b,                   # tensor: (1 - alpha_bar) at time t
        sample: bool = True
    ):
        B, C, H, W = b.shape
        H = H * self.sample_scale
        W = W * self.sample_scale
        D = self._D(H, W, self.sample_scale).to(b.device, dtype=b.dtype)
        U = self._U(H//self.sample_scale, W//self.sample_scale, self.sample_scale).to(b.device, dtype=b.dtype)
        H_prime, W_prime = H // self.sample_scale, W // self.sample_scale
        x0_c_vector = self.image_to_vector(x0_c, scale=1)
        x1_c_vector = self.image_to_vector(x1_c, scale=1)
        b = b[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        a = a[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        B = b * (th.eye(H_prime * W_prime, device=b.device, dtype=b.dtype) - D @ D.T)
        B_inv = th.linalg.inv(B)
        Sigma = b * th.linalg.inv(
            th.eye(H * W, device=b.device, dtype=b.dtype)
            + D.T @ th.linalg.inv(th.eye(H_prime * W_prime, device=b.device, dtype=b.dtype) - D @ D.T) @ D
        ).unsqueeze(0).unsqueeze(0)
        Sigma_inv = (
            th.eye(H * W, device=b.device, dtype=b.dtype)
            + D.T @ th.linalg.inv(th.eye(H_prime * W_prime, device=b.device, dtype=b.dtype) - D @ D.T) @ D
        ).unsqueeze(0).unsqueeze(0) / b
        DT_Binv = (B_inv.permute(0, 1, 3, 2) @ D).permute(0, 1, 3, 2)
        mu1 = th.einsum('n c i j, n c j -> n c i', DT_Binv, x1_c_vector)
        mu2 = (a / b).squeeze(-1) * th.einsum('i j, n c j -> n c i', U, x0_c_vector)
        Mu = th.einsum('n c i j, n c j -> n c i', Sigma, (mu1 + mu2))
        mvn = MultivariateNormal(loc=Mu, covariance_matrix=Sigma)
        sample = mvn.sample() if sample else Mu

        Mu = self.vector_to_image(Mu, H, W, 2)
        sample = self.vector_to_image(sample, H, W, 2)
        return Mu, Sigma, sample

    def vector_to_image(self, v, H, W, scale):
        bh, bw = scale, scale
        B, N, L = v.shape
        return (
            v
            .view(B, N, H // bh, W // bw, bh, bw)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(B, N, H, W)
        )

    def image_to_vector(self, M, scale):
        bh, bw = scale, scale
        B, N, H, W = M.shape
        return (
            M
            .view(B, N, H // bh, bh, W // bw, bw)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(B, N, H * W)
        )

    def _U(self, H, W, scale): # (H*W*scale*scale) x (H*W)
        # H, W: input resolution
        return scale**2 * self._D(H*scale, W*scale, scale).T

    def _D(self, H, W, scale):  # (H*W/scale/scale) x (H*W)
        # H, W: input resolution
        return 1 / scale**2 * th.eye(H * W // scale // scale).repeat_interleave(scale * scale, dim=1)
