"""
Multi-scale sampling for UNet models trained with drift on ImageNet-1k-256×256.

Sampling proceeds coarse→fine through three stages:

  Stage 1 —  64×64   (c=None, t: T_3→T_2 = 1000→600)  pure noise start
  Stage 2 — 128×128  (c=1,    t: T_2→T_1 =  600→300)  seeded from stage-1 output
  Stage 3 — 256×256  (c=0,    t: T_1→T_0 =  300→0  )  seeded from stage-2 output

Between stages the next scale is initialised with the RDD posterior upsample
operator, using the previous stage's final sample and clean prediction.

Output is a PNG grid of pixel-space images (no VAE).
"""

import argparse
import math
import os

import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image

from guided_diffusion.script_util import (
    create_model_for_drift,
    create_gaussian_diffusion_with_drift,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

T_0, T_1, T_2, T_3 = 0, 300, 600, 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device, **model_kwargs):
    """Load a UNet from a drift checkpoint (uses EMA weights if present)."""
    ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
    weights = ckpt.get("ema", ckpt.get("model"))
    model = create_model_for_drift(**model_kwargs).to(device)
    # Strip DDP 'module.' prefix if present
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    return model


def p_sample_loop_with_pred(diffusion, model, shape, noise, model_kwargs, device,
                             clip_denoised, progress, use_ddim, ddim_eta):
    """
    Run p_sample_loop_progressive and return the final sample AND pred_xstart
    from the last step.
    """
    loop_fn = (diffusion.ddim_sample_loop_progressive if use_ddim
               else diffusion.p_sample_loop_progressive)
    kwargs = dict(noise=noise, clip_denoised=clip_denoised,
                  model_kwargs=model_kwargs, device=device, progress=progress)
    if use_ddim:
        kwargs["eta"] = ddim_eta

    final = None
    for out in loop_fn(model, shape, **kwargs):
        final = out
    return final["sample"], final["pred_xstart"]


def upsample_and_reseed(pred_xstart, sample, diffusion_next, num_steps_next, n, device,
                        use_cheap_upsample=False, cheap_upsample_mode="nearest"):
    """
    Initialise the next stage using the posterior upsample operator copied from
    the original RDD diffusion implementation.
    """
    t_boundary = th.full((n,), num_steps_next - 1, dtype=th.long, device=device)
    if use_cheap_upsample:
        up = F.interpolate(sample, scale_factor=2, mode=cheap_upsample_mode)
        noise = th.randn_like(up)
        return diffusion_next.q_sample(up, t_boundary, noise=noise)
    return diffusion_next.posterior_upsample_between_scales(
        pred_xstart, sample, t_boundary
    )["sample"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    th.manual_seed(args.seed)
    th.set_grad_enabled(False)
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sampler = "DDIM" if args.use_ddim else "DDPM"
    print(f"Sampler: {sampler}  |  num_samples: {args.num_samples}")
    print(f"Steps: 64×64={args.num_steps_64}  128×128={args.num_steps_128}  "
          f"256×256={args.num_steps_256}")
    upsample_method = (
        f"cheap-{args.cheap_upsample_mode}" if args.use_cheap_upsample else "posterior"
    )
    print(f"Inter-stage upsample: {upsample_method}")
    if args.class_label is not None:
        print(f"Class label: {args.class_label}")

    # ---- Shared model kwargs ----
    shared = dict(
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_label is not None,
        use_checkpoint=False,
        num_heads=args.num_heads,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=0.0,
        use_fp16=args.use_fp16,
    )

    # ---- Load models ----
    diffusion_64 = create_gaussian_diffusion_with_drift(
        steps=T_3, noise_schedule=args.noise_schedule, learn_sigma=args.learn_sigma,
        c=None, T_c=T_2, T_c_plus_1=T_3, sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_64),
    )
    diffusion_128 = create_gaussian_diffusion_with_drift(
        steps=T_3, noise_schedule=args.noise_schedule, learn_sigma=args.learn_sigma,
        c=1, T_c=T_1, T_c_plus_1=T_2, sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_128),
    )
    diffusion_256 = create_gaussian_diffusion_with_drift(
        steps=T_3, noise_schedule=args.noise_schedule, learn_sigma=args.learn_sigma,
        c=0, T_c=T_0, T_c_plus_1=T_1, sample_scale=args.sample_scale,
        timestep_respacing=str(args.num_steps_256),
    )
    if args.use_single_model:
        print("\nLoading shared adapter model …")
        model_shared = load_model(
            args.ckpt_shared, device,
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            attention_resolutions=args.attention_resolutions_256,
            use_single_model=True,
            sample_scale=args.sample_scale,
            **shared,
        )
        model_64 = model_128 = model_256 = model_shared
        print(f"  ✓  shared model loaded")
    else:
        print("\nLoading 64×64 model …")
        model_64 = load_model(
            args.ckpt_64, device,
            image_size=args.image_size // 4,
            num_channels=args.num_channels_64,
            attention_resolutions=args.attention_resolutions_64,
            **shared,
        )
        print(f"  ✓  (interval t={T_2}→{T_3}, {diffusion_64.num_timesteps} steps)")

        print("Loading 128×128 model …")
        model_128 = load_model(
            args.ckpt_128, device,
            image_size=args.image_size // 2,
            num_channels=args.num_channels_128,
            attention_resolutions=args.attention_resolutions_128,
            **shared,
        )
        print(f"  ✓  (interval t={T_1}→{T_2}, {diffusion_128.num_timesteps} steps)")

        print("Loading 256×256 model …")
        model_256 = load_model(
            args.ckpt_256, device,
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            attention_resolutions=args.attention_resolutions_256,
            **shared,
        )
        print(f"  ✓  (interval t={T_0}→{T_1}, {diffusion_256.num_timesteps} steps)")

    # ---- model_kwargs (class label) ----
    n = args.num_samples
    if args.class_label is not None:
        y = th.full((n,), args.class_label, dtype=th.long, device=device)
        base_model_kwargs = dict(y=y)
    else:
        base_model_kwargs = {}

    # ---- Stage 1: 64×64 ----
    size_64 = args.image_size // 4
    print(f"\n[Stage 1/3] Sampling 64×64 ({sampler}, t={T_3}→{T_2}) …")
    noise_64 = th.randn(n, 3, size_64, size_64, device=device)
    x_64, pred_x0_64 = p_sample_loop_with_pred(
        diffusion_64, model_64, noise_64.shape, noise=noise_64,
        model_kwargs={**base_model_kwargs, **({"rdd_stage": "coarse"} if args.use_single_model else {})}, device=device,
        clip_denoised=args.clip_denoised, progress=True,
        use_ddim=args.use_ddim, ddim_eta=args.ddim_eta,
    )
    print(f"  ✓  shape={x_64.shape}")

    # ---- Upsample 64→128 ----
    print(f"\nUpsampling 64×64 → 128×128 ({upsample_method}) …")
    x_128_init = upsample_and_reseed(pred_x0_64, x_64, diffusion_128, args.num_steps_128,
                                     n, device, args.use_cheap_upsample, args.cheap_upsample_mode)
    print(f"  ✓  shape={x_128_init.shape}")

    # ---- Stage 2: 128×128 ----
    print(f"\n[Stage 2/3] Sampling 128×128 ({sampler}, t={T_2}→{T_1}) …")
    x_128, pred_x0_128 = p_sample_loop_with_pred(
        diffusion_128, model_128, x_128_init.shape, noise=x_128_init,
        model_kwargs={**base_model_kwargs, **({"rdd_stage": "mid"} if args.use_single_model else {})}, device=device,
        clip_denoised=args.clip_denoised, progress=True,
        use_ddim=args.use_ddim, ddim_eta=args.ddim_eta,
    )
    print(f"  ✓  shape={x_128.shape}")

    # ---- Upsample 128→256 ----
    print(f"\nUpsampling 128×128 → 256×256 ({upsample_method}) …")
    x_256_init = upsample_and_reseed(pred_x0_128, x_128, diffusion_256, args.num_steps_256,
                                     n, device, args.use_cheap_upsample, args.cheap_upsample_mode)
    print(f"  ✓  shape={x_256_init.shape}")

    # ---- Stage 3: 256×256 ----
    print(f"\n[Stage 3/3] Sampling 256×256 ({sampler}, t={T_1}→{T_0}) …")
    x_256, _ = p_sample_loop_with_pred(
        diffusion_256, model_256, x_256_init.shape, noise=x_256_init,
        model_kwargs={**base_model_kwargs, **({"rdd_stage": "fine"} if args.use_single_model else {})}, device=device,
        clip_denoised=args.clip_denoised, progress=True,
        use_ddim=args.use_ddim, ddim_eta=args.ddim_eta,
    )
    print(f"  ✓  shape={x_256.shape}")

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    nrow = math.isqrt(n)
    save_image(x_256, args.output, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"\n✓ Saved {n} samples → {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Multi-scale drift sampling: 64×64 → 128×128 → 256×256"
    )

    # Checkpoints — provide either all three explicit paths, or --ckpt_dir + --ckpt_step
    parser.add_argument("--use_single_model", action="store_true", default=False)
    parser.add_argument("--ckpt_shared", type=str, default=None,
                        help="Path to shared adapter checkpoint .pt")
    parser.add_argument("--ckpt_64",  type=str, default=None,
                        help="Path to 64×64 model checkpoint .pt")
    parser.add_argument("--ckpt_128", type=str, default=None,
                        help="Path to 128×128 model checkpoint .pt")
    parser.add_argument("--ckpt_256", type=str, default=None,
                        help="Path to 256×256 model checkpoint .pt")
    parser.add_argument("--ckpt_dir",  type=str, default=None,
                        help="Experiment directory (contains checkpoints_*/)")
    parser.add_argument("--ckpt_step", type=str, default=None,
                        help="7-digit step string, e.g. 0035000")

    # Sampling
    parser.add_argument("--num_samples",   type=int,   default=16)
    parser.add_argument("--num_steps_64",  type=int,   default=400,
                        help="DDPM/DDIM steps for the 64×64 stage")
    parser.add_argument("--num_steps_128", type=int,   default=300,
                        help="DDPM/DDIM steps for the 128×128 stage")
    parser.add_argument("--num_steps_256", type=int,   default=300,
                        help="DDPM/DDIM steps for the 256×256 stage")
    parser.add_argument("--use_ddim",      action="store_true")
    parser.add_argument("--ddim_eta",      type=float, default=0.0)
    parser.add_argument("--clip_denoised", action="store_true", default=True)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--output",        type=str,   default="samples_drift_imagenet256.png")
    parser.add_argument("--use_cheap_upsample", action="store_true", default=False,
                        help="Use interpolate + q_sample between stages instead of exact posterior upsample.")
    parser.add_argument("--cheap_upsample_mode", type=str, default="nearest",
                        choices=["nearest", "bilinear"],
                        help="Interpolation mode when --use_cheap_upsample is enabled.")

    # Conditioning
    parser.add_argument("--class_label", type=int, default=None,
                        help="ImageNet class label [0, 999]. Omit for unconditional.")

    # Diffusion
    parser.add_argument("--noise_schedule", type=str,  default="linear")
    parser.add_argument("--learn_sigma",    action="store_true", default=False)
    parser.add_argument("--sample_scale",   type=int,  default=2)

    # Architecture (must match training)
    parser.add_argument("--image_size",    type=int, default=256)
    parser.add_argument("--num_res_blocks",type=int, default=2)
    parser.add_argument("--num_heads",     type=int, default=4)
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--use_fp16",      action="store_true", default=False)
    parser.add_argument("--num_channels_256",          type=int, default=192)
    parser.add_argument("--num_channels_128",          type=int, default=128)
    parser.add_argument("--num_channels_64",           type=int, default=64)
    parser.add_argument("--attention_resolutions_256", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_128", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_64",  type=str, default="16,8")

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()

    # Auto-construct checkpoint paths from --ckpt_dir + --ckpt_step
    if args.ckpt_dir and args.ckpt_step:
        if args.use_single_model:
            if not args.ckpt_shared:
                args.ckpt_shared = f"{args.ckpt_dir}/checkpoints_shared/{args.ckpt_step}.pt"
        else:
            if not args.ckpt_64:
                args.ckpt_64  = f"{args.ckpt_dir}/checkpoints_64/{args.ckpt_step}.pt"
            if not args.ckpt_128:
                args.ckpt_128 = f"{args.ckpt_dir}/checkpoints_128/{args.ckpt_step}.pt"
            if not args.ckpt_256:
                args.ckpt_256 = f"{args.ckpt_dir}/checkpoints_256/{args.ckpt_step}.pt"

    if args.use_single_model:
        if not args.ckpt_shared:
            raise ValueError(
                "Provide either --ckpt_shared or (--ckpt_dir + --ckpt_step) with --use_single_model"
            )
    else:
        if not (args.ckpt_64 and args.ckpt_128 and args.ckpt_256):
            raise ValueError(
                "Provide either (--ckpt_64 / --ckpt_128 / --ckpt_256) "
                "or (--ckpt_dir + --ckpt_step)"
            )

    main(args)
