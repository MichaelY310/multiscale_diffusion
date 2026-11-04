"""
exp09_psd_analysis.py
=====================
分析 drift d^c = UP(DOWN(z)) - z 的频率内容（Power Spectral Density）。

步骤：
  1. 加载 VAE + 100 张 ImageNet 验证集图片
  2. Encode 到 latent space (32×32×4)
  3. 计算 d^c = UP(DOWN(z)) - z（使用 avg_pool2d + F.interpolate）
  4. 对 d^c 做 2D FFT，计算 radial PSD
  5. 对比：原图 z 的 PSD、低频部分 UP(DOWN(z)) 的 PSD
  6. 计算高频能量占比（freq > Nyquist/2）
  7. 保存 JSON 结果 + radial PSD 曲线图

输出：
  - psd_analysis_results.json
  - psd_radial_curve.pdf

用法：
  source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
  python exp09_psd_analysis.py [--num-images 100] [--output-dir ./]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Path setup
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from diffusers.models import AutoencoderKL

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
HF_HOME = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"
DATA_PATH = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
VAE_MODEL = "stabilityai/sd-vae-ft-ema"
IMAGE_SIZE = 256
LATENT_SIZE = 32  # 256 / 8
DOWNSAMPLE_FACTOR = 2  # 32 -> 16


def center_crop_arr(pil_image, image_size):
    """ADM-style center crop."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def compute_radial_psd(fft_mag_squared, H, W):
    """
    Compute radial (azimuthally averaged) PSD from 2D |FFT|^2.
    fft_mag_squared: [H, W] numpy array of |F(u,v)|^2.
    Returns (freqs, psd) where freqs are radial frequency bins.
    """
    cy, cx = H // 2, W // 2
    Y, X = np.mgrid[0:H, 0:W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_radius = min(cy, cx)
    radial_psd = np.zeros(max_radius)
    counts = np.zeros(max_radius)
    for r in range(max_radius):
        mask = R == r
        radial_psd[r] = fft_mag_squared[mask].sum()
        counts[r] = mask.sum()
    # Average over annulus
    valid = counts > 0
    radial_psd[valid] /= counts[valid]
    freqs = np.arange(max_radius) / max_radius  # normalized [0, 1)
    return freqs, radial_psd


def compute_2d_psd(tensor_batch):
    """
    Compute average 2D PSD over a batch.
    tensor_batch: [N, C, H, W] tensor
    Returns: [H, W] numpy array (averaged over N and C)
    """
    # FFT over spatial dims
    fft2 = torch.fft.fft2(tensor_batch, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
    mag_sq = (fft_shifted.real ** 2 + fft_shifted.imag ** 2)
    # Average over batch and channels
    avg_psd = mag_sq.mean(dim=(0, 1)).cpu().numpy()
    return avg_psd


def main():
    parser = argparse.ArgumentParser(description="PSD analysis of drift d^c")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of images to analyze")
    parser.add_argument("--output-dir", type=str, default="./",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--downsample-factor", type=int, default=2,
                        help="Downsample factor for computing drift")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set HF cache
    os.environ["HF_HOME"] = HF_HOME

    print("[1/7] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(device).eval()

    print("[2/7] Loading ImageNet images...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(DATA_PATH, transform=transform)

    # Take first N images
    num_images = min(args.num_images, len(dataset))
    images = []
    for i in range(num_images):
        img, _ = dataset[i]
        images.append(img)
    images = torch.stack(images).to(device)
    print(f"  Loaded {num_images} images, shape: {images.shape}")

    print("[3/7] Encoding to latent space...")
    batch_size = 10
    latents = []
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch = images[i:i + batch_size]
            posterior = vae.encode(batch).latent_dist
            z = posterior.sample() * 0.18215  # VAE scaling factor
            latents.append(z)
    z_all = torch.cat(latents, dim=0)  # [N, 4, 32, 32]
    print(f"  Latent shape: {z_all.shape}")

    print("[4/7] Computing drift d^c = UP(DOWN(z)) - z ...")
    factor = args.downsample_factor
    target_size = LATENT_SIZE // factor  # 16
    # DOWN: average pooling
    z_down = F.avg_pool2d(z_all, kernel_size=factor, stride=factor)  # [N, 4, 16, 16]
    # UP: bilinear interpolation
    z_up = F.interpolate(z_down, size=(LATENT_SIZE, LATENT_SIZE), mode='bilinear',
                         align_corners=False)  # [N, 4, 32, 32]
    drift = z_up - z_all  # d^c
    print(f"  drift shape: {drift.shape}, mean abs: {drift.abs().mean().item():.6f}")

    print("[5/7] Computing 2D FFT and radial PSD...")
    psd_z = compute_2d_psd(z_all)
    psd_low = compute_2d_psd(z_up)
    psd_drift = compute_2d_psd(drift)

    H, W = LATENT_SIZE, LATENT_SIZE
    freqs_z, rpsd_z = compute_radial_psd(psd_z, H, W)
    freqs_low, rpsd_low = compute_radial_psd(psd_low, H, W)
    freqs_drift, rpsd_drift = compute_radial_psd(psd_drift, H, W)

    print("[6/7] Computing high-frequency energy ratio...")
    nyquist_half = len(freqs_z) // 2  # freq > Nyquist/2
    total_energy_z = rpsd_z.sum()
    hf_energy_z = rpsd_z[nyquist_half:].sum()
    total_energy_drift = rpsd_drift.sum()
    hf_energy_drift = rpsd_drift[nyquist_half:].sum()
    total_energy_low = rpsd_low.sum()
    hf_energy_low = rpsd_low[nyquist_half:].sum()

    hf_ratio_z = float(hf_energy_z / total_energy_z) if total_energy_z > 0 else 0.0
    hf_ratio_drift = float(hf_energy_drift / total_energy_drift) if total_energy_drift > 0 else 0.0
    hf_ratio_low = float(hf_energy_low / total_energy_low) if total_energy_low > 0 else 0.0

    print(f"  High-freq energy ratio (z):     {hf_ratio_z:.4f}")
    print(f"  High-freq energy ratio (drift): {hf_ratio_drift:.4f}")
    print(f"  High-freq energy ratio (low):   {hf_ratio_low:.4f}")

    # Compute ||d^c|| / sqrt(n) for DNR normalization
    n = z_all.shape[1] * z_all.shape[2] * z_all.shape[3]  # 4*32*32
    drift_norm = drift.view(drift.shape[0], -1).norm(dim=1).mean().item()
    drift_norm_per_dim = drift_norm / np.sqrt(n)

    print(f"  ||d^c||: {drift_norm:.4f}")
    print(f"  ||d^c||/sqrt(n): {drift_norm_per_dim:.6f}")

    print("[7/7] Saving results...")
    results = {
        "num_images": num_images,
        "latent_size": LATENT_SIZE,
        "downsample_factor": factor,
        "high_freq_energy_ratio": {
            "z_original": hf_ratio_z,
            "z_drift_dc": hf_ratio_drift,
            "z_low_freq": hf_ratio_low,
        },
        "drift_norm_mean": drift_norm,
        "drift_norm_per_dim": drift_norm_per_dim,
        "radial_psd": {
            "freqs": freqs_z.tolist(),
            "psd_z": rpsd_z.tolist(),
            "psd_low": rpsd_low.tolist(),
            "psd_drift": rpsd_drift.tolist(),
        },
    }

    json_path = os.path.join(args.output_dir, "psd_analysis_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(freqs_z, rpsd_z, label="Original z", color="blue", linewidth=1.5)
    ax.semilogy(freqs_low, rpsd_low, label="Low-freq UP(DOWN(z))", color="green", linewidth=1.5)
    ax.semilogy(freqs_drift, rpsd_drift, label="Drift d^c = UP(DOWN(z)) - z",
                color="red", linewidth=1.5)

    # Mark Nyquist/2 boundary
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Nyquist/2")
    ax.set_xlabel("Normalized Radial Frequency", fontsize=12)
    ax.set_ylabel("Power Spectral Density (log scale)", fontsize=12)
    ax.set_title("Radial PSD: Original vs Low-freq vs Drift", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    pdf_path = os.path.join(args.output_dir, "psd_radial_curve.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print("Done.")


if __name__ == "__main__":
    main()
