"""
Visualize the RDD forward process with noise fixed to zero.

For each RDD stage, this script:
1. loads one ImageNet-256 training image,
2. builds the stage diffusion via create_gaussian_diffusion_with_drift(),
3. calls q_sample(..., noise=0) at several timesteps,
4. saves figures where every diagnostic image is derived from q_sample.

The only post-processing on images is:
- whole-image scaling by a scalar, and
- downsample/upsample of the q_sample output itself to expose block structure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as th
from datasets import load_dataset
from torchvision import transforms

from guided_diffusion.script_util import create_gaussian_diffusion_with_drift

T_0, T_1, T_2, T_3 = 0, 300, 600, 1000


def downsample_image(x, target_size):
    b, c, h, w = x.shape
    assert h == w and h % target_size == 0
    factor = h // target_size
    x = x.view(b, c, target_size, factor, target_size, factor)
    return x.mean(dim=(3, 5))


def to_display(x):
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return x[0].permute(1, 2, 0).numpy()


def block_std(x, block_size=2):
    b, c, h, w = x.shape
    assert h % block_size == 0 and w % block_size == 0
    blocks = x.view(b, c, h // block_size, block_size, w // block_size, block_size)
    return blocks.std(dim=(3, 5), unbiased=False).mean().item()


def load_one_training_sample(hf_repo, split, image_size, sample_index):
    ds = load_dataset(hf_repo, split=split)
    item = ds[sample_index]
    image = item["image"]
    label = int(item["label"])
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    x = transform(image).unsqueeze(0)
    return x, label


def build_stage_diffusion(stage):
    if stage == "256":
        return create_gaussian_diffusion_with_drift(
            steps=T_3,
            noise_schedule="linear",
            learn_sigma=False,
            timestep_respacing="",
            c=0,
            T_c=T_0,
            T_c_plus_1=T_1,
            sample_scale=2,
        )
    if stage == "128":
        return create_gaussian_diffusion_with_drift(
            steps=T_3,
            noise_schedule="linear",
            learn_sigma=False,
            timestep_respacing="",
            c=1,
            T_c=T_1,
            T_c_plus_1=T_2,
            sample_scale=2,
        )
    if stage == "64":
        return create_gaussian_diffusion_with_drift(
            steps=T_3,
            noise_schedule="linear",
            learn_sigma=False,
            timestep_respacing="",
            c=None,
            T_c=T_2,
            T_c_plus_1=T_3,
            sample_scale=2,
        )
    raise ValueError(stage)


def stage_config(stage):
    if stage == "256":
        return dict(image_size=256, T_c=T_0, T_c_plus_1=T_1, global_ts=[0, 75, 150, 225, 299])
    if stage == "128":
        return dict(image_size=128, T_c=T_1, T_c_plus_1=T_2, global_ts=[300, 375, 450, 525, 599])
    if stage == "64":
        return dict(image_size=64, T_c=T_2, T_c_plus_1=T_3, global_ts=[600, 700, 800, 900, 999])
    raise ValueError(stage)


def visualize_stage(x_256, label, sample_index, stage, output_dir):
    cfg = stage_config(stage)
    diffusion = build_stage_diffusion(stage)

    if cfg["image_size"] == 256:
        x_stage = x_256
    else:
        x_stage = downsample_image(x_256, cfg["image_size"])

    x_stage_up = x_stage
    if cfg["image_size"] != 256:
        x_stage_up = th.nn.functional.interpolate(x_stage, size=(256, 256), mode="nearest")

    fig, axes = plt.subplots(5, len(cfg["global_ts"]) + 1, figsize=(3.3 * (len(cfg["global_ts"]) + 1), 13.5))

    axes[0, 0].imshow(to_display(x_stage_up))
    axes[0, 0].set_title(f"stage input\n{cfg['image_size']}x{cfg['image_size']}")
    axes[1, 0].imshow(to_display(x_stage_up))
    axes[1, 0].set_title("q_sample, noise=0")
    axes[2, 0].imshow(to_display(x_stage_up))
    axes[2, 0].set_title("q_sample / sqrt(a_bar)")
    axes[3, 0].imshow(to_display(x_stage_up))
    axes[3, 0].set_title("down_up(q_sample / sqrt(a_bar))")
    axes[4, 0].imshow(to_display(x_stage_up))
    axes[4, 0].set_title("residual to block mean")

    for row in axes:
        for ax in row:
            ax.axis("off")

    for col, global_t in enumerate(cfg["global_ts"], start=1):
        local_t = global_t - cfg["T_c"]
        t = th.tensor([local_t], dtype=th.long)
        zero_noise = th.zeros_like(x_stage)
        x_t = diffusion.q_sample(x_stage, t, noise=zero_noise)

        sqrt_alpha = diffusion.sqrt_alphas_cumprod[local_t]
        renorm = x_t / sqrt_alpha
        renorm_down_up = diffusion.upsample(
            diffusion.downsample(renorm, diffusion.sample_scale),
            diffusion.sample_scale,
        )
        residual = renorm - renorm_down_up

        x_t_up = x_t
        renorm_up = renorm
        renorm_down_up_up = renorm_down_up
        residual_up = residual
        if cfg["image_size"] != 256:
            x_t_up = th.nn.functional.interpolate(x_t, size=(256, 256), mode="nearest")
            renorm_up = th.nn.functional.interpolate(renorm, size=(256, 256), mode="nearest")
            renorm_down_up_up = th.nn.functional.interpolate(renorm_down_up, size=(256, 256), mode="nearest")
            residual_up = th.nn.functional.interpolate(residual, size=(256, 256), mode="nearest")

        renorm_block_std = block_std(renorm)
        residual_scale = max(float(residual_up.abs().max().item()), 1e-8)
        residual_vis = residual_up / residual_scale

        axes[0, col].imshow(to_display(x_stage_up))
        axes[0, col].set_title(f"reference\n(global t={global_t})")
        axes[1, col].imshow(to_display(x_t_up))
        axes[1, col].set_title(f"q_sample\nsqrt(a)={float(sqrt_alpha):.4f}")
        axes[2, col].imshow(to_display(renorm_up))
        axes[2, col].set_title(f"x_t / sqrt(a_bar)\nblock_std={renorm_block_std:.5f}")
        axes[3, col].imshow(to_display(renorm_down_up_up))
        axes[3, col].set_title("block mean upsample")
        axes[4, col].imshow(to_display(residual_vis))
        axes[4, col].set_title(f"residual\nmax={float(residual.abs().max()):.5f}")

    title = (
        f"RDD q_sample-only visualization | stage {stage} | "
        f"sample_index={sample_index} | label={label}"
    )
    if diffusion.c is None:
        title += " | coarse stage: drift disabled"
    fig.suptitle(title)
    fig.tight_layout()

    out_path = output_dir / f"rdd_drift_only_stage_{stage}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/rdd_forward_drift_only_visualization",
    )
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_256, label = load_one_training_sample(
        hf_repo=args.hf_repo,
        split=args.hf_split,
        image_size=256,
        sample_index=args.sample_index,
    )

    saved = []
    for stage in ["256", "128", "64"]:
        saved.append(visualize_stage(x_256, label, args.sample_index, stage, output_dir))

    print("Saved files:")
    for path in saved:
        print(path)
