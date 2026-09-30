# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

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
        loss_type
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

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
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

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
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
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
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

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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
        return final["sample"]

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
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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
        return final["sample"]

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

    def _get_score_from_model(self, model, x, t, model_kwargs=None):
        """
        Convert model output to score (gradient of log probability).
        Assumes model predicts epsilon (noise) by default.
        
        For VP-SDE: score = -epsilon / sigma
        where sigma^2 = 1 - alpha_bar
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Get model output
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        
        # Handle learned variance (model outputs both epsilon and variance)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            B, C = model_output.shape[:2]
            assert C % 2 == 0, f"Expected even number of channels, got {C}"
            model_output, _ = th.split(model_output, C // 2, dim=1)
        
        # Convert epsilon to score
        # For VP-SDE: score = -epsilon / sqrt(1 - alpha_bar)
        if self.model_mean_type == ModelMeanType.EPSILON:
            sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            score = -model_output / sigma
        elif self.model_mean_type == ModelMeanType.START_X:
            # If model predicts x_0, convert to score
            # score = (x_0 - sqrt(alpha_bar) * x_t) / (1 - alpha_bar)
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            sigma_sq = 1.0 - alpha_bar
            score = (model_output - th.sqrt(alpha_bar) * x) / sigma_sq
        else:
            raise NotImplementedError(self.model_mean_type)
        
        return score

    def _euler_maruyama_predictor(self, model, x, t, model_kwargs=None):
        """
        Euler-Maruyama predictor for VP-SDE.
        Based on sde_sampling.py EulerMaruyamaPredictor
        
        Uses numerical integration of the reverse-time SDE:
        dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw
        
        For VP-SDE forward SDE:
        - f(x,t) = -0.5 * beta(t) * x  (forward drift)
        - g(t) = sqrt(beta(t))          (diffusion coefficient)
        
        For reverse-time SDE:
        - drift = f - g^2 * score = -0.5 * beta * x - beta * score
        - diffusion = g = sqrt(beta)
        
        Args:
            model: score model
            x: current state x_t
            t: current timestep (0 to num_timesteps-1)
            model_kwargs: additional model arguments
            
        Returns:
            x: next state x_{t-1} with noise
            x_mean: next state x_{t-1} without noise
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Get discrete beta for this timestep
        beta = _extract_into_tensor(self.betas, t, x.shape)
        
        # Get score from model
        score = self._get_score_from_model(model, x, t, model_kwargs)
        
        # Euler-Maruyama update (from sde_sampling.py line 182-186)
        # dt = -1/N (negative because we go backward in time)
        dt = -1.0 / self.num_timesteps
        
        # Forward SDE terms
        forward_drift = -0.5 * beta * x
        diffusion = th.sqrt(beta)
        
        # Reverse SDE (following reverse() method in sde_lib):
        # reverse_drift = forward_drift - diffusion^2 * score
        #               = -0.5 * beta * x - beta * score
        reverse_drift = forward_drift - (diffusion ** 2) * score
        
        # Euler-Maruyama step: x_{t+dt} = x_t + drift * dt + diffusion * sqrt(|dt|) * noise
        # Since dt < 0 (backward), sqrt(|dt|) = sqrt(-dt)
        x_mean = x + reverse_drift * dt
        
        # Add noise (no noise when t == 0)
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        # Use np.sqrt for scalar dt
        x = x_mean + nonzero_mask * diffusion * np.sqrt(-dt) * noise
        
        return x, x_mean

    def _vpsde_predictor(self, model, x, t, model_kwargs=None):
        """
        Ancestral sampling predictor for VP-SDE.
        Based on sde_sampling.py AncestralSamplingPredictor.vpsde_update_fn
        
        Args:
            model: score model
            x: current state x_t
            t: current timestep (0 to num_timesteps-1)
            model_kwargs: additional model arguments
            
        Returns:
            x: next state x_{t-1} with noise
            x_mean: next state x_{t-1} without noise
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Get discrete beta for this timestep
        beta = _extract_into_tensor(self.betas, t, x.shape)
        
        # Get score from model
        score = self._get_score_from_model(model, x, t, model_kwargs)
        
        # Ancestral sampling update (from sde_sampling.py line 230-233):
        # x_mean = (x + beta * score) / sqrt(1 - beta)
        # x = x_mean + sqrt(beta) * noise
        x_mean = (x + beta * score) / th.sqrt(1.0 - beta)
        
        # Add noise (no noise when t == 0)
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x = x_mean + nonzero_mask * th.sqrt(beta) * noise
        
        return x, x_mean

    def _langevin_corrector(self, model, x, t, model_kwargs=None, snr=0.16, n_steps=1):
        """
        Langevin dynamics corrector for VP-SDE.
        Based on sde_sampling.py LangevinCorrector
        
        Args:
            model: score model
            x: current state
            t: current timestep
            model_kwargs: additional model arguments
            snr: signal-to-noise ratio
            n_steps: number of Langevin steps
            
        Returns:
            x: corrected state with noise
            x_mean: corrected state without noise (last step)
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Get alpha_t (NOT alpha_bar) for current timestep (from sde_sampling.py line 269)
        # alpha_t = 1 - beta_t
        alphas = 1.0 - self.betas
        alpha = _extract_into_tensor(alphas, t, x.shape)
        
        x_mean = x
        for i in range(n_steps):
            # Skip correction at t=0
            if (t == 0).all():
                break
            
            # Get score from model
            score = self._get_score_from_model(model, x, t, model_kwargs)
            
            # Compute step size based on SNR (from sde_sampling.py line 276-278)
            noise = th.randn_like(x)
            grad_norm = th.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
            noise_norm = th.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-6)) ** 2 * 2 * alpha
            
            # Langevin update (from sde_sampling.py line 279-280)
            x_mean = x + step_size * score
            x = x_mean + th.sqrt(step_size * 2) * noise
        
        return x, x_mean

    def pc_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        snr=0.16,
        n_corrector_steps=1,
        predictor="euler_maruyama",
    ):
        """
        Sample x_{t-1} from the model using Predictor-Corrector sampling.
        Based on VP-SDE (Variance Preserving SDE) from Score-Based Generative Modeling.
        
        This implementation follows sde_sampling.py:
        - Predictor: euler_maruyama (default), ancestral, or reverse_diffusion
        - Corrector: Langevin dynamics (optional, set n_corrector_steps=0 to disable)
        
        Args:
            model: the model to sample from (assumes it predicts score/epsilon).
            x: the current tensor at x_t.
            t: the value of t, starting at 0 for the first diffusion step.
            clip_denoised: if True, clip the x_start prediction to [-1, 1].
            denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: if not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            snr: signal-to-noise ratio for Langevin corrector steps.
            n_corrector_steps: number of Langevin corrector steps (0 to disable corrector).
            predictor: predictor type - "euler_maruyama", "ancestral", or "reverse_diffusion"
        
        Returns:
            a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0 (can be None).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # ====================
        # Corrector step (from sde_sampling.py line 406)
        # ====================
        if n_corrector_steps > 0:
            x, x_mean = self._langevin_corrector(
                model, x, t, model_kwargs=model_kwargs, 
                snr=snr, n_steps=n_corrector_steps
            )

        # ====================
        # Predictor step (from sde_sampling.py line 407)
        # ====================
        if predictor == "euler_maruyama":
            x, x_mean = self._euler_maruyama_predictor(
                model, x, t, model_kwargs=model_kwargs
            )
        elif predictor == "ancestral":
            x, x_mean = self._vpsde_predictor(
                model, x, t, model_kwargs=model_kwargs
            )
        else:
            raise ValueError(f"Unknown predictor: {predictor}")
        
        return {"sample": x, "pred_xstart": None}

    def pc_sample_loop(
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
        snr=0.16,
        n_corrector_steps=0,
        predictor="euler_maruyama",
    ):
        """
        Generate samples from the model using Predictor-Corrector sampling.
        Based on VP-SDE (Variance Preserving SDE).
        
        Args:
            model: the model to sample from.
            shape: the shape of the samples, (N, C, H, W).
            noise: if specified, the noise from the encoder to sample.
                   Should be of the same shape as `shape`.
            clip_denoised: if True, clip x_start predictions to [-1, 1].
            denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample.
            cond_fn: if not None, this is a gradient function that acts
                similarly to the model.
            model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            device: if specified, the device to create the samples on.
                    If not specified, use a model parameter's device.
            progress: if True, show a tqdm progress bar.
            snr: signal-to-noise ratio for Langevin corrector steps.
            n_corrector_steps: number of Langevin corrector steps (0 to disable).
            predictor: predictor type - "euler_maruyama" (default) or "ancestral"
        
        Returns:
            a non-differentiable batch of samples.
        """
        final = None
        for sample in self.pc_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            snr=snr,
            n_corrector_steps=n_corrector_steps,
            predictor=predictor,
        ):
            final = sample
        return final["sample"]

    def pc_sample_loop_progressive(
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
        snr=0.16,
        n_corrector_steps=0,
        predictor="euler_maruyama",
    ):
        """
        Generate samples from the model using Predictor-Corrector sampling and yield intermediate samples.
        Based on VP-SDE (Variance Preserving SDE).
        
        Same arguments as pc_sample_loop().
        
        Yields:
            dicts with the same keys as pc_sample().
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

            indices = tqdm(indices, desc="PC Sampling")

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.pc_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    snr=snr,
                    n_corrector_steps=n_corrector_steps,
                    predictor=predictor,
                )
                yield out
                img = out["sample"]


    def inverse_time_ode_solver_sample(
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
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        score = - 1 / torch.sqrt(1 - _extract_into_tensor(self.alphas_cumprod, t, x.shape)) * eps

        noise = th.randn_like(x)
        mean_pred = (
            (2 - torch.sqrt(1 - _extract_into_tensor(self.betas, t, x.shape))) * x 
            + 1/2 * _extract_into_tensor(self.betas, t, x.shape) * score
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * torch.sqrt(1 - _extract_into_tensor(self.betas, t, x.shape)) * noise * eta
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def inverse_time_ode_solver_sample_loop(
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
        final = None
        for sample in self.inverse_time_ode_solver_sample(
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
        return final["sample"]


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

    # ═══════════════════════════════════════════════════════════════════════════
    # DDPM JumpStep Sampling (exact multi-step posterior without drift)
    # ═══════════════════════════════════════════════════════════════════════════

    def ddpm_jump_step_sample(
        self, model, x, t, t_next, clip_denoised=True, denoised_fn=None, model_kwargs=None, stochastic=True
    ):
        """
        Sample x_{t_next} from x_t using the exact multi-step posterior q(x_s | x_t, x_0).

        For standard diffusion q(x_t | x_0) = N(sqrt(ab_t)*x_0, (1-ab_t)*I):
            Sigma    = (ab_s - ab_t)(1 - ab_s) / (ab_s * (1 - ab_t))
            coeff_xt = (1 - ab_s) / (1 - ab_t) * sqrt(ab_t / ab_s)
            coeff_x0 = (ab_s - ab_t) / (sqrt(ab_s) * (1 - ab_t))
            mu       = coeff_xt * x_t + coeff_x0 * pred_x0

        where ab_s = alphas_cumprod[t_next], ab_t = alphas_cumprod[t].
        Note: ab_s > ab_t since t_next < t (less noise).

        When stochastic=True:  x_s = mu + sqrt(Sigma) * noise  (DDPM-like)
        When stochastic=False: x_s = mu                         (deterministic, DDIM-like)
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

        ab_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        ab_s = _extract_into_tensor(self.alphas_cumprod, t_next, x.shape)

        # Posterior variance
        Sigma = (ab_s - ab_t) * (1.0 - ab_s) / (ab_s * (1.0 - ab_t))

        # Posterior mean coefficients
        coeff_xt = (1.0 - ab_s) / (1.0 - ab_t) * th.sqrt(ab_t / ab_s)
        coeff_x0 = (ab_s - ab_t) / (th.sqrt(ab_s) * (1.0 - ab_t))

        mu = coeff_xt * x + coeff_x0 * pred_xstart

        if stochastic:
            nonzero = (t_next != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            noise = th.randn_like(x)
            sample = mu + nonzero * th.sqrt(Sigma.clamp(min=0)) * noise
        else:
            sample = mu

        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddpm_jump_step_sample_loop(
        self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
        model_kwargs=None, device=None, progress=False, stochastic=True,
    ):
        """
        Generate samples using DDPM JumpStep (exact multi-step posterior).
        Same interface as p_sample_loop.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(indices))
        else:
            pbar = None

        for i_pos in range(len(indices)):
            i = indices[i_pos]
            t = th.tensor([i] * shape[0], device=device)
            nxt = indices[i_pos + 1] if i_pos < len(indices) - 1 else 0
            t_next = th.tensor([nxt] * shape[0], device=device)
            with th.no_grad():
                out = self.ddpm_jump_step_sample(
                    model, img, t, t_next,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    stochastic=stochastic,
                )
                img = out["sample"]
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        return img

    # ═══════════════════════════════════════════════════════════════════════════
    # DPM-Solver++ Sampling (order 1 and 2, multistep)
    # ═══════════════════════════════════════════════════════════════════════════

    def dpm_solver_sample_loop(
        self, model, shape, noise=None, clip_denoised=False, denoised_fn=None,
        model_kwargs=None, device=None, progress=False, order=2,
    ):
        """
        Generate samples using DPM-Solver++ (multistep).

        Uses the dpmsolver++ algorithm which works in the data-prediction (x0) space.
        For order=1: equivalent to DDIM (one model eval per step).
        For order=2: multistep DPM-Solver-2 (one model eval per step after init).

        The total NFE equals self.num_timesteps (e.g., 250).

        Mathematical formulation (DPM-Solver++):
            lambda_t = log(alpha_t / sigma_t)  [half-logSNR]

            Order-1 update from s to t (t less noisy, lambda_t > lambda_s):
                h = lambda_t - lambda_s
                phi_1 = expm1(-h)
                x_t = (sigma_t/sigma_s) * x_s - alpha_t * phi_1 * x0_pred(x_s, s)

            Order-2 multistep update (using cached model values):
                h_0 = lambda_s - lambda_{s_prev}  [previous step size]
                h   = lambda_t - lambda_s         [current step size]
                r0  = h_0 / h
                D1  = (1/r0) * (x0_pred_s - x0_pred_s_prev)
                phi_1 = expm1(-h)
                x_t = (sigma_t/sigma_s)*x_s - alpha_t*phi_1*x0_pred_s - 0.5*alpha_t*phi_1*D1
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        assert order in [1, 2], "DPM-Solver order must be 1 or 2"

        img = noise if noise is not None else th.randn(*shape, device=device)
        N = self.num_timesteps  # number of steps (NFE)

        # Precompute log(alpha) and log(sigma) for all local indices
        alphas_cumprod_th = th.from_numpy(self.alphas_cumprod).float().to(device)
        log_alpha_all = 0.5 * th.log(alphas_cumprod_th)
        log_sigma_all = 0.5 * th.log(1.0 - alphas_cumprod_th)

        def get_model_x0(x, idx_int):
            """Get x0 prediction at integer local index."""
            t_tensor = th.tensor([idx_int] * shape[0], device=device)
            out = self.p_mean_variance(
                model, x, t_tensor,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            return out["pred_xstart"]

        # Use all N local indices from N-1 down to 0 (no duplicates)
        # N steps: evaluate model at indices[0..N-1], target index after each step
        steps_int = list(range(N - 1, -1, -1))  # [N-1, N-2, ..., 1, 0]

        if order == 1:
            # Pure DPM-Solver-1 (DDIM equivalent)
            # N grid points define N-1 steps, each with 1 model eval → NFE = N-1
            num_steps = N - 1
            if progress:
                from tqdm.auto import tqdm
                step_range = tqdm(range(num_steps))
            else:
                step_range = range(num_steps)

            for step in step_range:
                s_idx = steps_int[step]
                t_idx = steps_int[step + 1]

                log_alpha_s = log_alpha_all[s_idx]
                log_sigma_s = log_sigma_all[s_idx]

                log_alpha_t = log_alpha_all[t_idx]
                log_sigma_t = log_sigma_all[t_idx]

                lambda_s = log_alpha_s - log_sigma_s
                lambda_t = log_alpha_t - log_sigma_t

                h = lambda_t - lambda_s
                phi_1 = th.expm1(-h)

                sigma_s = th.exp(log_sigma_s)
                sigma_t = th.exp(log_sigma_t)
                alpha_t = th.exp(log_alpha_t)

                with th.no_grad():
                    x0_pred = get_model_x0(img, s_idx)
                    img = (sigma_t / sigma_s) * img - alpha_t * phi_1 * x0_pred

        elif order == 2:
            # Multistep DPM-Solver-2
            # N grid points → N-1 total steps. First step uses order-1 (2 evals),
            # then N-2 order-2 steps (1 eval each, skip last) → NFE = 2 + (N-3) = N-1
            num_steps = N - 1  # total transitions

            # First step: use order 1
            s_idx = steps_int[0]
            t_idx = steps_int[1]

            log_alpha_s = log_alpha_all[s_idx]
            log_sigma_s = log_sigma_all[s_idx]
            lambda_s = log_alpha_s - log_sigma_s

            log_alpha_t = log_alpha_all[t_idx]
            log_sigma_t = log_sigma_all[t_idx]
            lambda_t = log_alpha_t - log_sigma_t

            h = lambda_t - lambda_s
            phi_1 = th.expm1(-h)
            sigma_s = th.exp(log_sigma_s)
            sigma_t = th.exp(log_sigma_t)
            alpha_t = th.exp(log_alpha_t)

            with th.no_grad():
                model_prev_0 = get_model_x0(img, s_idx)
                img = (sigma_t / sigma_s) * img - alpha_t * phi_1 * model_prev_0

            # Cache for multistep: previous lambda and x0 predictions
            lambda_prev_prev = lambda_s
            model_prev_1 = model_prev_0

            with th.no_grad():
                model_prev_0 = get_model_x0(img, t_idx)

            if progress:
                from tqdm.auto import tqdm
                step_range = tqdm(range(1, num_steps), initial=1, total=num_steps)
            else:
                step_range = range(1, num_steps)

            for step in step_range:
                s_idx_cur = steps_int[step]
                t_idx_cur = steps_int[step + 1]

                log_alpha_s_cur = log_alpha_all[s_idx_cur]
                log_sigma_s_cur = log_sigma_all[s_idx_cur]
                lambda_s_cur = log_alpha_s_cur - log_sigma_s_cur

                log_alpha_t_cur = log_alpha_all[t_idx_cur]
                log_sigma_t_cur = log_sigma_all[t_idx_cur]
                lambda_t_cur = log_alpha_t_cur - log_sigma_t_cur

                h_cur = lambda_t_cur - lambda_s_cur
                h_prev = lambda_s_cur - lambda_prev_prev
                r0 = h_prev / h_cur

                phi_1_cur = th.expm1(-h_cur)
                sigma_s_cur = th.exp(log_sigma_s_cur)
                sigma_t_cur = th.exp(log_sigma_t_cur)
                alpha_t_cur = th.exp(log_alpha_t_cur)

                # D1 = (1/r0) * (model_prev_0 - model_prev_1)
                D1 = (1.0 / r0) * (model_prev_0 - model_prev_1)

                with th.no_grad():
                    img = (sigma_t_cur / sigma_s_cur) * img - alpha_t_cur * phi_1_cur * model_prev_0 - 0.5 * alpha_t_cur * phi_1_cur * D1

                # Update cache
                lambda_prev_prev = lambda_s_cur
                model_prev_1 = model_prev_0
                # Evaluate model at new point (skip at last step)
                if step < num_steps - 1:
                    with th.no_grad():
                        model_prev_0 = get_model_x0(img, t_idx_cur)

        return img


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
