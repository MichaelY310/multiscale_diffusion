# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import gaussian_diffusion_with_drift as gd_with_drift
from .respace import SpacedDiffusion, SpacedDiffusionWithDrift, space_timesteps
from .flow_matching_with_drift import FlowMatchingWithDrift


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    variance_scale=1.0
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        variance_scale=variance_scale
        # rescale_timesteps=rescale_timesteps,
    )


def create_diffusion_with_drift(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    c=None,
    T_c=None,
    T_c_plus_1=None,
    sample_scale=2,
):
    """
    Create a diffusion process with drift support.
    
    :param timestep_respacing: respacing string for sampling (e.g., "" for all steps, "ddim50" for 50 DDIM steps)
    :param noise_schedule: noise schedule type ("linear" or "squaredcos_cap_v2")
    :param use_kl: whether to use KL loss
    :param sigma_small: whether to use small fixed variance
    :param predict_xstart: whether to predict x_start directly (vs epsilon)
    :param learn_sigma: whether to learn variance
    :param rescale_learned_sigmas: whether to rescale learned sigmas
    :param diffusion_steps: number of diffusion timesteps
    :param c: stage identifier; ``None`` disables drift for the coarsest stage
    :param T_c: timestep where drift starts
    :param T_c_plus_1: timestep where drift ends (typically T_c + 1)
    :param sample_scale: spatial projection factor (2 for adjacent stages, 4 for 32->8)
    :return: SpacedDiffusionWithDrift object
    """

    # TODO: 
    # T_c_plus_1 = T_c_plus_1 + 1

    betas = gd_with_drift.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_with_drift.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_with_drift.LossType.RESCALED_MSE
    else:
        loss_type = gd_with_drift.LossType.MSE
    if T_c is not None and T_c_plus_1 is not None:
        interval_diffusion_steps = T_c_plus_1 - T_c
    else:
        interval_diffusion_steps = diffusion_steps
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [interval_diffusion_steps]
    return SpacedDiffusionWithDrift(
        use_timesteps=space_timesteps(interval_diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_with_drift.ModelMeanType.EPSILON if not predict_xstart else gd_with_drift.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_with_drift.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_with_drift.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_with_drift.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        c=c,
        T_c=T_c,
        T_c_plus_1=T_c_plus_1,
        sample_scale=sample_scale,
    )
