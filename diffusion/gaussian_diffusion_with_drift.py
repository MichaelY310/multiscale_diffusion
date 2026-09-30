# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import torch
import enum
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        c=None,
        T_c=None,
        T_c_plus_1=None,
        sample_scale=2,
        **kwargs,
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        self.c = c
        self.T_c = T_c
        self.T_c_plus_1 = T_c_plus_1
        self.sample_scale = sample_scale
        self.drift_scale_multiplier = 1.0  # external multiplier for ablation (default: no change)
        
        # Only compute drift coefficient if drift parameters are provided
        if self.c is not None:
            t = np.arange(0, self.num_timesteps)
            if 'timestep_map' in kwargs:
                map_tensor = np.array(kwargs['timestep_map'])
                t = map_tensor[t]

            self.posterior_mean_coef_drift = (
                np.sqrt(self.alphas_cumprod_prev) / (1-self.alphas_cumprod) * 
                ((t-self.T_c) - alphas*(t-self.T_c) - alphas + self.alphas_cumprod) * 
                1/(T_c_plus_1 - T_c)
            )
        else:
            self.posterior_mean_coef_drift = None

    def q_mean_variance(self, x_start, t, d = None):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        if d is not None:
            map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            mean = mean + _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (map_tensor[t]-self.T_c) / (self.T_c_plus_1 - self.T_c) * d
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, d=None, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param d: drift tensor (same shape as x_start), or None for no drift.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if d is not None:
            # Compute drift coefficient: (t - T_c) / (T_c_plus_1 - T_c)
            # Need to expand t to match x_start shape for broadcasting
            t_expanded = t.int().view(-1, 1, 1, 1) if len(t.shape) == 1 else t
            map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self.drift_scale_multiplier * _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (map_tensor[t_expanded]-self.T_c) / (self.T_c_plus_1 - self.T_c) * d
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
        else:
                return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if self.c is not None:
            d_approx = self.upsample(self.downsample(x_start, self.sample_scale), self.sample_scale) - x_start   # bar(x_t)_{c+1} - bar(x_t)_c
            posterior_mean += self.drift_scale_multiplier * _extract_into_tensor(self.posterior_mean_coef_drift, t, x_t.shape) * d_approx
        
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

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        # Decouple mode: model outputs (eps [+ var] | d_pred), with d_pred
        # occupying the last C channels.  Strip d_pred before further parsing.
        #   - learned variance, no decouple : shape (B, 2C, H, W)
        #   - learned variance, decouple    : shape (B, 3C, H, W)
        #   - fixed variance,   no decouple : shape (B,  C, H, W)
        #   - fixed variance,   decouple    : shape (B, 2C, H, W)
        d_pred = None
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if model_output.shape[1] == C * 3:          # eps + var + d
                d_pred       = model_output[:, -C:]
                model_output = model_output[:, :-C]
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            if model_output.shape[1] == C * 2:          # eps + d
                d_pred       = model_output[:, -C:]
                model_output = model_output[:, :-C]
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            # NOTE: For decouple models, the eps head already predicts ε̃ (noise
            # with drift baked in), so we must NOT apply drift correction again.
            # d_pred is kept for potential use in upsampling but not passed here.
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output, d=None)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps, d=None):
        assert x_t.shape == eps.shape

        if d != None:
            map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            t_expanded = t.int().view(-1, 1, 1, 1) if len(t.shape) == 1 else t
            eps = eps + d * (map_tensor[t_expanded] - self.T_c) / (self.T_c_plus_1 - self.T_c) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_pure_eps_from_xstart(self, x_t, t, pred_xstart):
        d_approx = self.upsample(self.downsample(pred_xstart, self.sample_scale), self.sample_scale) - pred_xstart   # bar(x_t)_{c+1} - bar(x_t)_c
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        t_expanded = t.int().view(-1, 1, 1, 1) if len(t.shape) == 1 else t
        # print("======sqrt_recipm1_alphas_cumprod======")
        # print(_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
        # print("=================")
        epsilon_tilde_minus_epsilon = (map_tensor[t_expanded] - self.T_c) / (self.T_c_plus_1 - self.T_c) * d_approx / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        # print("======epsilon_tilde_minus_epsilon=======")
        # print(epsilon_tilde_minus_epsilon)
        # print("=================")
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) - epsilon_tilde_minus_epsilon

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(out["mean"])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
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

    # def p_sample_and_upsample(
    #     self,
    #     model,
    #     x,
    #     t,
    #     clip_denoised=True,
    #     denoised_fn=None,
    #     cond_fn=None,
    #     model_kwargs=None,
    # ):

    #     # Experiment:
    #     #     Setting 1: upsample mean and log_variance by constant (nearest) interpolation
    #     #     Setting 2: upsample mean and log_variance by bilinear interpolation
    #     #     Setting 3: use computed posterior mean and covariance
    #     out = self.p_mean_variance(
    #         model,
    #         x,
    #         t,
    #         clip_denoised=clip_denoised,
    #         denoised_fn=denoised_fn,
    #         model_kwargs=model_kwargs,
    #     )
    #     # out["mean"] = self.upsample_interpolate(out["mean"], self.sample_scale)
    #     # out["log_variance"] = self.upsample_interpolate(out["log_variance"], self.sample_scale)
    #     noise = th.randn_like(out["mean"])
    #     nonzero_mask = (
    #         (t != 0).float().view(-1, *([1] * (len(out["mean"].shape) - 1)))
    #     )  # no noise when t == 0
    #     if cond_fn is not None:
    #         out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
    #     sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
    #     mu, Sigma, sample = self.compute_posterior_x1(
    #         x0_c=out["pred_xstart"],
    #         x1_c=sample,
    #         a=_extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape),
    #         b=_extract_into_tensor(1.0 - self.alphas_cumprod, t, sample.shape),
    #         sample=True
    #     )
    #     return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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
        D = self._D(H, W, self.sample_scale).to(b.device)
        U = self._U(H//self.sample_scale, W//self.sample_scale, self.sample_scale).to(b.device)
        H_prime, W_prime = H // self.sample_scale, W // self.sample_scale
        x0_c_vector = self.image_to_vector(x0_c, scale=1)
        x1_c_vector = self.image_to_vector(x1_c, scale=1)
        b = b[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        a = a[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        print(D.device)
        print(b.device)
        B = b * (th.eye(H_prime * W_prime).to(b.device) - D @ D.T)
        B_inv = th.linalg.inv(B)
        Sigma = b * th.linalg.inv(th.eye(H * W).to(b.device) + D.T @ th.linalg.inv(th.eye(H_prime * W_prime).to(b.device) - D @ D.T) @ D).unsqueeze(0).unsqueeze(0)
        Sigma_inv = (th.eye(H * W).to(b.device) + D.T @ th.linalg.inv(th.eye(H_prime * W_prime).to(b.device) - D @ D.T) @ D).unsqueeze(0).unsqueeze(0)  / b
        DT_Binv = (B_inv.permute(0, 1, 3, 2) @ D).permute(0, 1, 3, 2)
        mu1 = th.einsum('n c i j, n c j -> n c i', DT_Binv, x1_c_vector)
        mu2 = (a / b).squeeze(-1) * th.einsum('i j, n c j -> n c i', U, x0_c_vector)
        Mu = th.einsum('n c i j, n c j -> n c i', Sigma, (mu1 + mu2))        
        mvn = MultivariateNormal(loc=Mu, covariance_matrix=Sigma)
        sample = mvn.sample()

        Mu = self.vector_to_image(Mu, H, W, 2)
        sample = self.vector_to_image(sample, H, W, 2)
        return Mu, Sigma, sample

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final  # dict: {"sample": x_{T_c}, "pred_xstart": x_0_pred}

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

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
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # decouple epsilon and d:
        if self.c is None:
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        else:
            eps = self._predict_pure_eps_from_xstart(x, t, out["pred_xstart"])
            d_approx = self.upsample(self.downsample(out["pred_xstart"], self.sample_scale), self.sample_scale) - out["pred_xstart"]   # bar(x_t)_{c+1} - bar(x_t)_c
            

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        t_expanded = t.int().view(-1, 1, 1, 1) if len(t.shape) == 1 else t
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            # + (map_tensor[t_expanded] - self.T_c) / (self.T_c_plus_1 - self.T_c) * _extract_into_tensor(self.sqrt_alphas_cumprod, t, out["pred_xstart"].shape) * d_approx
        )
        if self.c is not None:
            mean_pred += self.drift_scale_multiplier * _extract_into_tensor(self.posterior_mean_coef_drift, t, out["pred_xstart"].shape) * d_approx
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        if torch.isnan(mean_pred).any().item() or torch.isinf(mean_pred).any().item() or torch.isnan(out["pred_xstart"]).any().item() or torch.isinf(out["pred_xstart"]).any().item():
            print("===============================")
            print("t")
            print(t)
            print("mean_pred")
            print(mean_pred)
            print("pred_xstart")
            print(out["pred_xstart"])
            print("===============================")
            exit(1)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # def ddim_sample(
    #     self,
    #     model,
    #     x,
    #     t,
    #     clip_denoised=True,
    #     denoised_fn=None,
    #     cond_fn=None,
    #     model_kwargs=None,
    #     eta=0.0,
    # ):
    #     """
    #     Sample x_{t-1} from the model using DDIM.
    #     Same usage as p_sample().
    #     """
    #     out = self.p_mean_variance(
    #         model,
    #         x,
    #         t,
    #         clip_denoised=clip_denoised,
    #         denoised_fn=denoised_fn,
    #         model_kwargs=model_kwargs,
    #     )
    #     if cond_fn is not None:
    #         out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

    #     # Usually our model outputs epsilon, but we re-derive it
    #     # in case we used x_start or x_prev prediction.
    #     # decouple epsilon and d:
    #     eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])


    #     alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    #     sigma = (
    #         eta
    #         * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
    #         * th.sqrt(1 - alpha_bar / alpha_bar_prev)
    #     )
    #     # Equation 12.
    #     noise = th.randn_like(x)
    #     mean_pred = (
    #         out["pred_xstart"] * th.sqrt(alpha_bar_prev)
    #         + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    #     )
    #     nonzero_mask = (
    #         (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    #     )  # no noise when t == 0
    #     sample = mean_pred + nonzero_mask * sigma * noise
    #     return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def ddim_reverse_sample(
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
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps        
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        Returns a dict with keys 'sample' and 'pred_xstart'.
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final  # Return the full dict with 'sample' and 'pred_xstart'

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        if self.c is not None:
            d = self.upsample(self.downsample(x_start, self.sample_scale), self.sample_scale) - x_start
            x_t = self.q_sample(x_start, t, noise=noise, d=d)
            map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)

            t_factor = ((map_tensor[t] - self.T_c) / (self.T_c_plus_1 - self.T_c)).reshape(-1, 1, 1, 1)
            noise = noise + self.drift_scale_multiplier * th.sqrt(_extract_into_tensor(self.alphas_cumprod, t, x_start.shape) / (1 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape))) * t_factor * d
        else:
            x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
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
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
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

    def training_losses_decouple(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Decoupled training losses for drift stages.

        Requires the model to output (out_channels + in_channels) total channels:
          - first out_channels channels: identical to the original training_losses output
            (eps_pred when learn_sigma=False; eps_pred + learned_variance when learn_sigma=True)
            → receives the *same* combined-noise target ε̃ = ε + kappa·sqrt(ᾱ/(1−ᾱ))·d
          - last in_channels (= C) channels: d prediction
            → receives target d = upsample(downsample(x_0)) − x_0

        The original MSE + VB losses are fully preserved; MSE(d_pred, d) is added on top.

        For stages without drift (c=None), falls back to standard training_losses().
        """
        if self.c is None:
            return self.training_losses(model, x_start, t, model_kwargs, noise)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Wrap model for timestep remapping when using SpacedDiffusionWithDrift
        if hasattr(self, '_wrap_model'):
            model = self._wrap_model(model)

        # Drift term and drifted forward sample (same as training_losses)
        d = self.upsample(self.downsample(x_start, self.sample_scale), self.sample_scale) - x_start
        x_t = self.q_sample(x_start, t, noise=noise, d=d)

        # Combined-noise target used by the main head (same adjustment as training_losses)
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        t_factor = ((map_tensor[t] - self.T_c) / (self.T_c_plus_1 - self.T_c)).reshape(-1, 1, 1, 1)
        noise_tilde = noise + self.drift_scale_multiplier * th.sqrt(
            _extract_into_tensor(self.alphas_cumprod, t, x_start.shape) /
            (1 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape))
        ) * t_factor * d

        # Full model output: (main_out | d_pred)
        full_output = model(x_t, t, **model_kwargs)
        C = x_start.shape[1]
        # main_out has out_channels (C when learn_sigma=False, 2C when learn_sigma=True)
        # d_pred always has C channels
        assert full_output.shape[1] > C, (
            f"Decoupled mode requires more than C={C} output channels, got {full_output.shape[1]}. "
            f"Create the model with d_prediction=True."
        )
        main_out = full_output[:, :-C]   # (eps [+ var])  — out_channels channels
        d_pred   = full_output[:, -C:]   # d prediction   — C channels

        terms = {}

        # ── Reproduce the original MSE (+ optional VB) loss on the main head ──
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            raise NotImplementedError("KL loss not supported in training_losses_decouple")

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert main_out.shape[1] == C * 2, (
                f"Expected 2C={2*C} for main head with LEARNED variance, got {main_out.shape[1]}"
            )
            eps_pred, var_values = th.split(main_out, C, dim=1)
            frozen_out = th.cat([eps_pred.detach(), var_values], dim=1)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            if self.loss_type == LossType.RESCALED_MSE:
                terms["vb"] *= self.num_timesteps / 1000.0
            eps_out = eps_pred
        else:
            eps_out = main_out

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise_tilde,
        }[self.model_mean_type]
        assert eps_out.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - eps_out) ** 2)

        # ── Extra d-prediction loss ──────────────────────────────────────────
        terms["mse_d"] = mean_flat((d_pred - d) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"] + terms["mse_d"]
        else:
            terms["loss"] = terms["mse"] + terms["mse_d"]

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

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
    
    def downsample_D(self, x, scale):
        N, C, H, W = x.shape
        if scale == 1:
            return x

        v = self.image_to_vector(x, scale)
        return self.vector_to_image(v @ self._D(H, W, scale).T, H//2, W//2, 1)

    def upsample_U(self, x, scale):
        N, C, H, W = x.shape
        if scale == 1:
            return x

        v = self.image_to_vector(x, 1)
        return self.vector_to_image(v @ self._U(H, W, scale).T, H*2, W*2, scale)
        
    def downsample(self, x, scale):
        if scale == 1:
            return x
        B, C_dim, H, W = x.shape
        assert H % scale == 0 and W % scale == 0, f"H={H} and W={W} must be divisible by c={scale}"
        
        # Reshape to (B, C_dim, H/c, c, W/c, c) and take mean over the c×c blocks
        x = x.view(B, C_dim, H // scale, scale, W // scale, scale)
        x = x.mean(dim=(3, 5))  # Average over the c×c spatial dimensions
        return x
    
    def upsample(self, x, scale):
        """
        Upsample by repeating each pixel scale×scale times (nearest neighbor).
        :param x: input tensor of shape (B, C, H, W)
        :param scale: upsampling factor
        :return: upsampled tensor of shape (B, C, H*scale, W*scale)
        """
        if scale == 1:
            return x
        # Repeat each pixel scale times in both height and width dimensions
        x = x.repeat_interleave(scale, dim=2)  # Repeat along height
        x = x.repeat_interleave(scale, dim=3)  # Repeat along width
        return x

    def upsample_interpolate(self, x, scale, mode='bilinear'):
        """
        Upsample by repeating each pixel scale×scale times (nearest neighbor).
        :param x: input tensor of shape (B, C, H, W)
        :param scale: upsampling factor
        :return: upsampled tensor of shape (B, C, H*scale, W*scale)
        """
        if scale == 1:
            return x
        x = F.interpolate(x, size=(x.shape[-2]*scale, x.shape[-1]*scale), mode=mode, align_corners=False)
        return x


        # -------------------------------------------------------------------------
    # DPM-Solver
    # -------------------------------------------------------------------------

    def _lambda_to_discrete_t(self, lam, device):
        """Map continuous lambda values to the nearest discrete internal timestep indices.

        :param lam: (B,) float tensor of lambda values.
        :param device: target device.
        :return: (B,) long tensor of internal indices in [0, num_timesteps-1].
        """
        alphas = th.from_numpy(self.alphas_cumprod).float().to(device)
        lambdas_all = 0.5 * th.log(alphas / (1.0 - alphas))   # (num_timesteps,)
        diffs = (lambdas_all.unsqueeze(0) - lam.unsqueeze(1)).abs()  # (B, num_timesteps)
        return diffs.argmin(dim=1).long()  # (B,)

    def dpm_solver_sample(
        self,
        model,
        x,
        t,
        t_next,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        order=2,
        x_prev=None,
        eps_prev=None,
    ):
        """Single DPM-Solver step from internal timestep *t* to *t_next* (t > t_next).

        For drift diffusion (self.c is not None) the extra correction term
            (real_t_next - real_t) / (T_c_plus_1 - T_c) * d
        is added to the standard DPM-Solver update, where
            real_t = self.timestep_map[t]
        and
            d = upsample(downsample(x_0_pred)) - x_0_pred.

        For the order-2 case the same correction is applied to the intermediate
        half-step u as well (using real_t_s instead of real_t_next).
        """
        if model_kwargs is None:
            model_kwargs = {}

        def call_model(x_in, t_in):
            out_i = self.p_mean_variance(
                model, x_in, t_in,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if self.c is not None:
                eps_i = self._predict_pure_eps_from_xstart(x_in, t_in, out_i["pred_xstart"])
            else:
                eps_i = self._predict_eps_from_xstart(x_in, t_in, out_i["pred_xstart"])
            return eps_i, out_i["pred_xstart"]

        def drift_from_pred(pred_x0):
            if self.c is None:
                return None
            return self.upsample(
                self.downsample(pred_x0, self.sample_scale), self.sample_scale
            ) - pred_x0

        def drift_step(alpha_bar_target, real_target, real_source, d_term):
            return th.sqrt(alpha_bar_target) * (real_target - real_source) * drift_scale * d_term

        alpha_bar_t      = _extract_into_tensor(self.alphas_cumprod, t,      x.shape)
        alpha_bar_t_next = _extract_into_tensor(self.alphas_cumprod, t_next, x.shape)
        sigma_t_next     = th.sqrt(1.0 - alpha_bar_t_next)
        sqrt_alpha_bar_t      = _extract_into_tensor(self.sqrt_alphas_cumprod, t,      x.shape)

        lambda_t      = 0.5 * th.log(alpha_bar_t      / (1.0 - alpha_bar_t))
        lambda_t_next = 0.5 * th.log(alpha_bar_t_next / (1.0 - alpha_bar_t_next))
        h = lambda_t_next - lambda_t  # > 0 in the denoising direction

        # trivial step: t == t_next
        if h.abs().max() < 1e-8:
            out0 = self.p_mean_variance(
                model, x, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            return {"sample": x, "pred_xstart": out0["pred_xstart"]}

        eps_t, pred_xstart = call_model(x, t)

        # Drift quantities (only when drift is active)
        d_approx    = None
        map_tensor  = None
        real_t      = None
        real_t_next = None
        drift_scale = None
        d_t         = None
        if self.c is not None:
            d_t         = drift_from_pred(pred_xstart)
            map_tensor  = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            t_exp       = t.int().view(-1, 1, 1, 1)
            t_next_exp  = t_next.int().view(-1, 1, 1, 1)
            real_t      = map_tensor[t_exp].float()
            real_t_next = map_tensor[t_next_exp].float()
            drift_scale = self.drift_scale_multiplier / (self.T_c_plus_1 - self.T_c)

        def alpha_sigma_from_lambda(lam):
            ab = th.sigmoid(2.0 * lam)
            return ab, th.sqrt(1.0 - ab)

        def lam_to_t(lam_4d):
            return self._lambda_to_discrete_t(
                lam_4d.reshape(x.shape[0], -1)[:, 0], x.device
            )

        phi_1 = th.expm1(h)  # e^h - 1  (numerically stable)

        if order == 1:
            x_next = (
                th.sqrt(alpha_bar_t_next / alpha_bar_t) * x
                - sigma_t_next * phi_1 * eps_t
            )
            if self.c is not None:
                x_next = x_next + drift_step(alpha_bar_t_next, real_t_next, real_t, d_t)

        elif order == 2:
            r1       = 0.5
            lambda_s = lambda_t + r1 * h
            alpha_bar_s, sigma_s = alpha_sigma_from_lambda(lambda_s)
            t_s      = lam_to_t(lambda_s)   # (B,) internal index

            phi_11 = th.expm1(r1 * h)

            # Half-step to intermediate point u
            u = (
                th.sqrt(alpha_bar_s / alpha_bar_t) * x
                - sigma_s * phi_11 * eps_t
            )
            if self.c is not None:
                t_s_exp  = t_s.view(-1, 1, 1, 1)
                real_t_s = map_tensor[t_s_exp].float()
                u = u + drift_step(alpha_bar_s, real_t_s, real_t, d_t)

            eps_s, pred_xstart_s = call_model(u, t_s)

            # Full step using eps at midpoint
            x_next = (
                th.sqrt(alpha_bar_t_next / alpha_bar_t) * x
                - sigma_t_next * phi_1 * eps_s
            )
            if self.c is not None:
                d_s = drift_from_pred(pred_xstart_s)
                x_next = x_next + drift_step(alpha_bar_t_next, real_t_next, real_t, d_s)

        else:
            raise NotImplementedError(f"DPM-Solver order {order} not implemented for drift diffusion")

        return {"sample": x_next, "pred_xstart": pred_xstart, "eps": eps_t}

    def dpm_solver_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        order=2,
    ):
        """Generate samples using DPM-Solver and yield intermediate outputs.

        Arguments are the same as p_sample_loop_progressive().
        Each yielded dict has keys 'sample', 'pred_xstart', and 'eps'.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices_iter = tqdm(list(enumerate(indices)), total=len(indices),
                                desc=f"DPM-Solver (order {order})")
        else:
            indices_iter = enumerate(indices)

        x_prev   = None
        eps_prev = None

        for i, idx in indices_iter:
            t      = th.tensor([idx] * shape[0], device=device)
            t_next = (
                th.tensor([indices[i + 1]] * shape[0], device=device)
                if i < len(indices) - 1
                else th.zeros_like(t)
            )
            with th.no_grad():
                out = self.dpm_solver_sample(
                    model, img, t, t_next,
                    order=order if i > 0 else 1,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    x_prev=x_prev,
                    eps_prev=eps_prev,
                )
                yield out
                x_prev   = img
                eps_prev = out.get("eps", None)
                img      = out["sample"]

    def dpm_solver_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        order=2,
    ):
        """Generate samples using DPM-Solver and return the final result dict.

        Returns a dict with keys 'sample' and 'pred_xstart' (same interface as
        ddim_sample_loop and p_sample_loop).
        """
        final = None
        for sample in self.dpm_solver_sample_loop_progressive(
            model, shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            order=order,
        ):
            final = sample
        return final


    # ------------------------------------------------------------------ #
    #  Jump-step sampler: multi-step Bayesian posterior                    #
    # ------------------------------------------------------------------ #

    def jump_step_sample(
        self,
        model,
        x,
        t,
        t_next,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        deterministic=True,
    ):
        """Single jump-step from local index *t* to local index *t_next* via the
        exact multi-step Bayesian posterior  q(x_s | x_t, x_0).

        Forward-process identities used (s = t_next in denoising time,
        bar_alpha_s > bar_alpha_t; global offsets from T_c denoted s, t-s):

            q(x_t | x_s) = N(x_t | A*x_s + B*d,  sigma1^2 I)
            q(x_s | x_0) = N(x_s | C*x_0 + D*d,  sigma2^2 I)

        Coefficients (bar_alpha = self.alphas_cumprod at local index):
            A      = sqrt(ab_t / ab_s)
            B      = sqrt(ab_t) * (global_t - global_s) / DeltaT
            C      = sqrt(ab_s)
            D      = sqrt(ab_s) * (global_s - T_c)  / DeltaT
            sigma1^2 = 1 - ab_t / ab_s
            sigma2^2 = 1 - ab_s

        Posterior:
            Sigma     = (ab_s - ab_t)(1 - ab_s) / (ab_s * (1 - ab_t))
            coeff_xt  = (1 - ab_s) / (1 - ab_t) * sqrt(ab_t / ab_s)
            coeff_x0  = (ab_s - ab_t)            / (sqrt(ab_s) * (1 - ab_t))
            coeff_d   = Sigma * (D/sigma2^2 - A*B/sigma1^2)

        When *deterministic* is True (default), the noise term is dropped,
        giving a deterministic trajectory analogous to DDIM.
        When *deterministic* is False, stochastic noise ~ N(0, Sigma*I) is added.

        For a single-step jump (global t → global t-1) this reduces exactly to
        the standard DDPM posterior.
        """
        if model_kwargs is None:
            model_kwargs = {}

        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]

        ab_t = _extract_into_tensor(self.alphas_cumprod, t,      x.shape)  # bar_alpha_t
        ab_s = _extract_into_tensor(self.alphas_cumprod, t_next, x.shape)  # bar_alpha_s

        # Trivial step: no change
        if (ab_s - ab_t).abs().max() < 1e-8:
            return {"sample": x, "pred_xstart": pred_xstart}

        # ── core posterior coefficients ────────────────────────────────
        Sigma    = (ab_s - ab_t) * (1.0 - ab_s) / (ab_s * (1.0 - ab_t))
        coeff_xt = (1.0 - ab_s) / (1.0 - ab_t) * th.sqrt(ab_t / ab_s)
        coeff_x0 = (ab_s - ab_t) / (th.sqrt(ab_s) * (1.0 - ab_t))

        mu = coeff_xt * x + coeff_x0 * pred_xstart

        # ── drift coefficient (only when drift is active) ──────────────
        if self.c is not None:
            map_tensor  = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            t_exp       = t.int().view(-1, 1, 1, 1)
            t_next_exp  = t_next.int().view(-1, 1, 1, 1)
            real_t      = map_tensor[t_exp].float()       # global timestep
            real_t_next = map_tensor[t_next_exp].float()  # global timestep of target
            DT          = float(self.T_c_plus_1 - self.T_c)

            s_offset  = real_t_next - self.T_c  # user's "s":   global_s - T_c
            ts_diff   = real_t - real_t_next    # user's "t-s": step in global time

            sigma1_sq = 1.0 - ab_t / ab_s                             # 1 - ab_t/ab_s
            sigma2_sq = 1.0 - ab_s                                    # 1 - ab_s

            A_coef = th.sqrt(ab_t / ab_s)
            B_coef = th.sqrt(ab_t) * ts_diff  / DT
            D_coef = th.sqrt(ab_s) * s_offset / DT

            d = self.upsample(
                    self.downsample(pred_xstart, self.sample_scale),
                    self.sample_scale) - pred_xstart

            coeff_d = Sigma * (D_coef / sigma2_sq - A_coef * B_coef / sigma1_sq)
            mu = mu + self.drift_scale_multiplier * coeff_d * d

        # ── output ────────────────────────────────────────────────────
        if deterministic:
            return {"sample": mu, "pred_xstart": pred_xstart}
        else:
            nonzero = (t_next != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            noise   = th.randn_like(x)
            sample  = mu + nonzero * th.sqrt(Sigma.clamp(min=0)) * noise
            return {"sample": sample, "pred_xstart": pred_xstart}

    def jump_step_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        deterministic=True,
    ):
        """Yield successive samples produced by the jump-step sampler."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm as _tqdm
            indices_iter = _tqdm(indices)
        else:
            indices_iter = indices

        for i, idx in enumerate(indices_iter):
            t      = th.tensor([idx] * shape[0], device=device)
            nxt    = indices[i + 1] if i < len(indices) - 1 else 0
            t_next = th.tensor([nxt]  * shape[0], device=device)
            with th.no_grad():
                out = self.jump_step_sample(
                    model, img, t, t_next,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    deterministic=deterministic,
                )
                yield out
                img = out["sample"]

    def jump_step_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        deterministic=True,
    ):
        """Run the full jump-step sampling loop and return the final dict."""
        final = None
        for sample in self.jump_step_sample_loop_progressive(
            model, shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            deterministic=deterministic,
        ):
            final = sample
        return final

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
