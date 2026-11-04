"""
Classifier-guided multi-stage sampling for RDD / diffusion-with-drift.

This sampler uses one classifier per RDD stage and intentionally avoids the
exact pixel-space posterior upsample operator, which is prohibitively expensive
at 64->128 and 128->256. Instead it uses a practical boundary reseeding step:
nearest-neighbor upsample followed by stage-consistent q_sample() noise.

The classifier guidance itself is standard DDPM guidance and is intended for
DDPM sampling. DDIM guidance with drift is left unsupported here.
"""

import argparse
import math
import os

import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image

from guided_diffusion.script_util import (
    create_classifier,
    create_gaussian_diffusion_with_drift,
    create_model_for_drift,
)

T_0, T_1, T_2, T_3 = 0, 300, 600, 1000


STAGES = {
    "64": dict(image_size=64, c=None, T_c=T_2, T_c_plus_1=T_3),
    "128": dict(image_size=128, c=1, T_c=T_1, T_c_plus_1=T_2),
    "256": dict(image_size=256, c=0, T_c=T_0, T_c_plus_1=T_1),
}


def load_model(ckpt_path, device, **model_kwargs):
    ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
    weights = ckpt.get("ema", ckpt.get("model", ckpt))
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model = create_model_for_drift(**model_kwargs).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model


def load_classifier(ckpt_path, device, **classifier_kwargs):
    weights = th.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    classifier = create_classifier(**classifier_kwargs).to(device)
    classifier.load_state_dict(weights)
    classifier.eval()
    return classifier


def cond_fn_from_classifier(classifier, scale):
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = log_probs[th.arange(logits.shape[0], device=logits.device), y]
            grad = th.autograd.grad(chosen.sum(), x_in)[0]
            return grad * scale

    return cond_fn


def p_sample_loop_with_pred(
    diffusion,
    model,
    shape,
    noise,
    model_kwargs,
    cond_fn,
    device,
    clip_denoised,
):
    final = None
    for out in diffusion.p_sample_loop_progressive(
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=True,
    ):
        final = out
    return final["sample"], final["pred_xstart"]


def upsample_and_reseed(sample, diffusion_next):
    up = F.interpolate(sample, scale_factor=2, mode="nearest")
    t = th.full((up.shape[0],), diffusion_next.num_timesteps - 1, device=up.device, dtype=th.long)
    noise = th.randn_like(up)
    return diffusion_next.q_sample(up, t, noise=noise)


def main(args):
    th.manual_seed(args.seed)
    th.set_grad_enabled(False)
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    if args.use_ddim:
        raise ValueError("This script only supports DDPM classifier guidance for drift stages.")

    shared = dict(
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=True,
        use_checkpoint=False,
        num_heads=args.num_heads,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=0.0,
        use_fp16=args.use_fp16,
    )

    diffusion_64 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        c=None,
        T_c=T_2,
        T_c_plus_1=T_3,
        sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_64),
    )
    diffusion_128 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        c=1,
        T_c=T_1,
        T_c_plus_1=T_2,
        sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_128),
    )
    diffusion_256 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        c=0,
        T_c=T_0,
        T_c_plus_1=T_1,
        sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_256),
    )

    model_64 = load_model(
        args.ckpt_64,
        device,
        image_size=64,
        num_channels=args.num_channels_64,
        attention_resolutions=args.attention_resolutions_64,
        **shared,
    )
    model_128 = load_model(
        args.ckpt_128,
        device,
        image_size=128,
        num_channels=args.num_channels_128,
        attention_resolutions=args.attention_resolutions_128,
        **shared,
    )
    model_256 = load_model(
        args.ckpt_256,
        device,
        image_size=256,
        num_channels=args.num_channels_256,
        attention_resolutions=args.attention_resolutions_256,
        **shared,
    )

    classifier_64 = load_classifier(
        args.classifier_64,
        device,
        image_size=64,
        classifier_use_fp16=args.classifier_use_fp16,
        classifier_width=args.classifier_width,
        classifier_depth=args.classifier_depth,
        classifier_attention_resolutions=args.classifier_attention_resolutions_64,
        classifier_use_scale_shift_norm=args.classifier_use_scale_shift_norm,
        classifier_resblock_updown=args.classifier_resblock_updown,
        classifier_pool=args.classifier_pool,
    )
    classifier_128 = load_classifier(
        args.classifier_128,
        device,
        image_size=128,
        classifier_use_fp16=args.classifier_use_fp16,
        classifier_width=args.classifier_width,
        classifier_depth=args.classifier_depth,
        classifier_attention_resolutions=args.classifier_attention_resolutions_128,
        classifier_use_scale_shift_norm=args.classifier_use_scale_shift_norm,
        classifier_resblock_updown=args.classifier_resblock_updown,
        classifier_pool=args.classifier_pool,
    )
    classifier_256 = load_classifier(
        args.classifier_256,
        device,
        image_size=256,
        classifier_use_fp16=args.classifier_use_fp16,
        classifier_width=args.classifier_width,
        classifier_depth=args.classifier_depth,
        classifier_attention_resolutions=args.classifier_attention_resolutions_256,
        classifier_use_scale_shift_norm=args.classifier_use_scale_shift_norm,
        classifier_resblock_updown=args.classifier_resblock_updown,
        classifier_pool=args.classifier_pool,
    )

    n = args.num_samples
    y = th.full((n,), args.class_label, dtype=th.long, device=device)
    model_kwargs = {"y": y}

    cond_64 = cond_fn_from_classifier(classifier_64, args.classifier_scale_64)
    cond_128 = cond_fn_from_classifier(classifier_128, args.classifier_scale_128)
    cond_256 = cond_fn_from_classifier(classifier_256, args.classifier_scale_256)

    noise_64 = th.randn(n, 3, 64, 64, device=device)
    sample_64, _ = p_sample_loop_with_pred(
        diffusion_64,
        model_64,
        noise_64.shape,
        noise_64,
        model_kwargs,
        cond_64,
        device,
        args.clip_denoised,
    )

    init_128 = upsample_and_reseed(sample_64, diffusion_128)
    sample_128, _ = p_sample_loop_with_pred(
        diffusion_128,
        model_128,
        init_128.shape,
        init_128,
        model_kwargs,
        cond_128,
        device,
        args.clip_denoised,
    )

    init_256 = upsample_and_reseed(sample_128, diffusion_256)
    sample_256, _ = p_sample_loop_with_pred(
        diffusion_256,
        model_256,
        init_256.shape,
        init_256,
        model_kwargs,
        cond_256,
        device,
        args.clip_denoised,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    nrow = math.isqrt(n)
    save_image(sample_256, args.output, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"Saved {n} samples to {args.output}")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_64", type=str, required=True)
    parser.add_argument("--ckpt_128", type=str, required=True)
    parser.add_argument("--ckpt_256", type=str, required=True)
    parser.add_argument("--classifier_64", type=str, required=True)
    parser.add_argument("--classifier_128", type=str, required=True)
    parser.add_argument("--classifier_256", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--class_label", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="samples_rdd_classifier_guided.png")
    parser.add_argument("--clip_denoised", action="store_true", default=True)
    parser.add_argument("--use_ddim", action="store_true")

    parser.add_argument("--num_steps_64", type=int, default=400)
    parser.add_argument("--num_steps_128", type=int, default=300)
    parser.add_argument("--num_steps_256", type=int, default=300)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--learn_sigma", action="store_true", default=False)
    parser.add_argument("--sample_scale", type=int, default=2)

    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--num_channels_64", type=int, default=64)
    parser.add_argument("--num_channels_128", type=int, default=128)
    parser.add_argument("--num_channels_256", type=int, default=192)
    parser.add_argument("--attention_resolutions_64", type=str, default="16,8")
    parser.add_argument("--attention_resolutions_128", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_256", type=str, default="32,16,8")

    parser.add_argument("--classifier_use_fp16", action="store_true", default=False)
    parser.add_argument("--classifier_width", type=int, default=128)
    parser.add_argument("--classifier_depth", type=int, default=2)
    parser.add_argument("--classifier_pool", type=str, default="attention")
    parser.add_argument("--classifier_resblock_updown", action="store_true", default=True)
    parser.add_argument("--classifier_use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--classifier_attention_resolutions_64", type=str, default="32,16,8")
    parser.add_argument("--classifier_attention_resolutions_128", type=str, default="32,16,8")
    parser.add_argument("--classifier_attention_resolutions_256", type=str, default="32,16,8")
    parser.add_argument("--classifier_scale_64", type=float, default=1.0)
    parser.add_argument("--classifier_scale_128", type=float, default=1.0)
    parser.add_argument("--classifier_scale_256", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
