"""
Sampling from a standard (no-drift) class-conditional 256×256 UNet
trained on ImageNet-1k-256×256.

Usage:
    python scripts/image_sample_imagenet256.py \
        --ckpt path/to/checkpoints/0075000.pt \
        --class_label 388 \
        --num_samples 16 \
        --output samples.png
"""

import argparse
import math
import os

import torch as th
from torchvision.utils import save_image

from guided_diffusion.script_util import create_model_and_diffusion

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True


def load_model(ckpt_path, device, **model_kwargs):
    ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model = model.to(device)

    ema = ckpt.get("ema")
    if ema is not None and isinstance(ema, list):
        for p, e in zip(model.parameters(), ema):
            p.data.copy_(e)
    else:
        state = ckpt.get("model", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

    model.eval()
    return model, diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image_size",            type=int,   default=256)
    parser.add_argument("--num_channels",          type=int,   default=192)
    parser.add_argument("--num_res_blocks",        type=int,   default=2)
    parser.add_argument("--num_heads",             type=int,   default=4)
    parser.add_argument("--attention_resolutions", default="32,16,8")
    parser.add_argument("--noise_schedule",        default="linear")
    parser.add_argument("--diffusion_steps",       type=int,   default=1000)
    parser.add_argument("--num_samples",           type=int,   default=16)
    parser.add_argument("--num_steps",             type=int,   default=250)
    parser.add_argument("--use_ddpm",              action="store_true")
    parser.add_argument("--ddim_eta",              type=float, default=0.0)
    parser.add_argument("--class_label",           type=int,   default=None)
    parser.add_argument("--clip_denoised",         action="store_true", default=True)
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--output",                default="samples_imagenet256.png")
    parser.add_argument("--gpu",                   type=int,   default=0)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    timestep_respacing = "" if args.use_ddpm else str(args.num_steps)

    print(f"loading model from {args.ckpt} ...")
    model, diffusion = load_model(
        args.ckpt, device,
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions=args.attention_resolutions,
        channel_mult="",
        dropout=0.0,
        class_cond=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing=timestep_respacing,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if args.class_label is not None:
        labels = th.full((args.num_samples,), args.class_label, dtype=th.long, device=device)
    else:
        labels = th.randint(0, 1000, (args.num_samples,), device=device)
    print(f"class labels: {labels.tolist()}")

    shape = (args.num_samples, 3, args.image_size, args.image_size)
    with th.no_grad():
        if args.use_ddpm:
            samples = diffusion.p_sample_loop(
                model, shape, clip_denoised=args.clip_denoised,
                model_kwargs={"y": labels}, device=device, progress=True,
            )
        else:
            samples = diffusion.ddim_sample_loop(
                model, shape, clip_denoised=args.clip_denoised,
                model_kwargs={"y": labels}, eta=args.ddim_eta,
                device=device, progress=True,
            )

    samples = (samples.clamp(-1, 1) + 1) / 2
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(samples, args.output, nrow=math.isqrt(args.num_samples))
    print(f"saved {args.num_samples} samples → {args.output}")


if __name__ == "__main__":
    main()
