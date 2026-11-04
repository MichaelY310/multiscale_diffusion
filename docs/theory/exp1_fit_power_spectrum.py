"""
Experiment 1: Fit Power Spectrum of VAE Latents
================================================
Loads ImageNet images, encodes through SD VAE, computes radial power spectrum,
fits a power-law model P(ω) = A * |ω|^(-β), and computes SPEED optimal stage lengths.
"""

import os
import random
import numpy as np
import torch
from math import sqrt, pi

# Set environment variables before importing HF libraries
os.environ["HF_HOME"] = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"

from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_random_images(data_root, num_images=100, seed=42):
    """Load random images from ImageNet train folder."""
    random.seed(seed)
    
    # Get class folders
    class_folders = sorted(os.listdir(data_root))
    # Pick a subset of class folders
    selected_classes = random.sample(class_folders, min(20, len(class_folders)))
    
    # Collect all image paths from selected classes
    all_image_paths = []
    for cls_folder in selected_classes:
        cls_path = os.path.join(data_root, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG'))]
        all_image_paths.extend(files)
    
    # Random sample
    selected_paths = random.sample(all_image_paths, min(num_images, len(all_image_paths)))
    print(f"Selected {len(selected_paths)} images from {len(selected_classes)} classes")
    return selected_paths


def load_and_preprocess_images(image_paths, image_size=256):
    """Load and preprocess images to [-1, 1] normalized tensors."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue
    
    return torch.stack(images)  # [B, 3, 256, 256]


def encode_with_vae(vae, images, batch_size=2):
    """Encode images through VAE to get latents."""
    vae.eval()
    all_latents = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # VAE encode returns a distribution, sample from it
            posterior = vae.encode(batch).latent_dist
            latents = posterior.mean  # Use mean for deterministic encoding
            # Scale by the VAE scaling factor
            latents = latents * vae.config.scaling_factor
            all_latents.append(latents)
            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  Encoded batch {batch_idx}/{num_batches}")
    
    return torch.cat(all_latents, dim=0)  # [B, 4, 32, 32]


def compute_radial_power_spectrum(latents):
    """
    Compute 2D radial power spectrum averaged over batch and channels.
    latents: [B, C, H, W]
    Returns: frequencies (1D), power spectrum values (1D)
    """
    B, C, H, W = latents.shape
    
    # Compute 2D FFT for each sample and channel
    # Use real FFT shifted so DC is at center
    power_sum = np.zeros((H, W))
    
    for b in range(B):
        for c in range(C):
            x = latents[b, c].numpy()
            # 2D FFT
            fft2d = np.fft.fft2(x)
            fft2d_shifted = np.fft.fftshift(fft2d)
            # Power spectrum (magnitude squared)
            power = np.abs(fft2d_shifted) ** 2
            power_sum += power
    
    # Average over batch and channels
    power_avg = power_sum / (B * C)
    
    # Compute radial average
    cy, cx = H // 2, W // 2
    max_radius = min(cy, cx)
    
    # Create radius map
    y_coords = np.arange(H) - cy
    x_coords = np.arange(W) - cx
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    radius_map = np.sqrt(xx**2 + yy**2)
    
    # Bin by radius
    num_bins = max_radius
    radial_freq = np.zeros(num_bins)
    radial_power = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for r_bin in range(num_bins):
        r_low = r_bin
        r_high = r_bin + 1
        mask = (radius_map >= r_low) & (radius_map < r_high)
        if np.any(mask):
            radial_power[r_bin] = np.mean(power_avg[mask])
            radial_freq[r_bin] = (r_low + r_high) / 2.0
            counts[r_bin] = np.sum(mask)
    
    # Skip DC component (bin 0)
    valid = counts > 0
    valid[0] = False  # Skip DC
    
    return radial_freq[valid], radial_power[valid]


def fit_power_law(frequencies, power_spectrum):
    """
    Fit P(ω) = A * |ω|^(-β) in log-log space using least squares.
    Returns: A, beta
    """
    log_freq = np.log(frequencies)
    log_power = np.log(power_spectrum)
    
    # Linear fit in log-log space: log(P) = log(A) - β * log(ω)
    # y = a + b*x  where y=log(P), x=log(ω), a=log(A), b=-β
    A_matrix = np.vstack([np.ones_like(log_freq), log_freq]).T
    result = np.linalg.lstsq(A_matrix, log_power, rcond=None)
    coeffs = result[0]
    
    log_A = coeffs[0]
    neg_beta = coeffs[1]
    
    A = np.exp(log_A)
    beta = -neg_beta
    
    return A, beta


def plot_power_spectrum(frequencies, power_spectrum, A, beta, save_path):
    """Plot measured vs fitted power spectrum."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Measured
    ax.loglog(frequencies, power_spectrum, 'b.', markersize=8, label='Measured', alpha=0.7)
    
    # Fitted
    freq_fit = np.linspace(frequencies.min(), frequencies.max(), 200)
    power_fit = A * freq_fit ** (-beta)
    ax.loglog(freq_fit, power_fit, 'r-', linewidth=2, 
              label=f'Fit: P(ω) = {A:.2f} × |ω|^(-{beta:.3f})')
    
    ax.set_xlabel('Radial Frequency ω', fontsize=12)
    ax.set_ylabel('Power Spectrum P(ω)', fontsize=12)
    ax.set_title('Radial Power Spectrum of VAE Latents (ImageNet 256×256)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def activation_time(P_omega, delta):
    """
    SPEED optimal activation time for a frequency component.
    τ(ω) = 1 / (1 + sqrt(δ / (P(ω) * (1 + P(ω) - δ))))
    """
    denom_inner = P_omega * (1.0 + P_omega - delta)
    if denom_inner <= 0:
        return 1.0  # fully active
    ratio = delta / denom_inner
    return 1.0 / (1.0 + sqrt(ratio))


def delta_optimal_transitions(scales, delta=0.01, A=1.0, beta=2.0, H=32, W=32):
    """
    Compute SPEED optimal transition points for given scale config.
    
    scales: list of resolution fractions [0.25, 0.5, 1.0]
    Returns: list of transition times between stages
    """
    # For each scale, compute the max frequency it can represent
    # scale s means we use s*H x s*W resolution, max freq = s * max_freq_full
    max_freq_full = min(H, W) / 2.0  # Nyquist for full resolution
    
    # Compute activation times for frequencies at scale boundaries
    # Transition from scale_i to scale_{i+1} happens when frequencies
    # above scale_i's Nyquist become "needed"
    transitions = []
    
    for i in range(len(scales) - 1):
        # The critical frequency is at the boundary of scale_i
        critical_freq = scales[i] * max_freq_full
        
        # Power at this frequency
        P_omega = A * critical_freq ** (-beta)
        
        # Activation time for this frequency
        tau = activation_time(P_omega, delta)
        transitions.append(tau)
    
    return transitions


def transitions_to_stage_lengths(transitions, T_total):
    """Convert transition times to stage lengths."""
    # transitions are times where we switch to next stage
    # Stage 1: [0, t1], Stage 2: [t1, t2], ..., Stage N: [t_{N-1}, T]
    boundaries = [0.0] + sorted(transitions) + [1.0]
    stage_lengths = []
    for i in range(len(boundaries) - 1):
        length = (boundaries[i+1] - boundaries[i]) * T_total
        stage_lengths.append(length)
    return stage_lengths


def main():
    print("=" * 60)
    print("Experiment 1: Fit Power Spectrum of VAE Latents")
    print("=" * 60)
    
    # Paths
    data_root = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
    save_dir = "/home/tiger/multiscale_diffusion/research_experiments"
    plot_path = os.path.join(save_dir, "power_spectrum_fit.png")
    
    # Step 1: Load VAE
    print("\n[1/6] Loading VAE (stabilityai/sd-vae-ft-ema)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        torch_dtype=torch.float32,
    )
    vae.eval()
    print(f"  VAE loaded. Scaling factor: {vae.config.scaling_factor}")
    
    # Step 2: Load images
    print("\n[2/6] Loading images from ImageNet...")
    image_paths = load_random_images(data_root, num_images=100)
    images = load_and_preprocess_images(image_paths, image_size=256)
    print(f"  Loaded {images.shape[0]} images, shape: {images.shape}")
    
    # Step 3: Encode through VAE
    print("\n[3/6] Encoding images through VAE...")
    latents = encode_with_vae(vae, images, batch_size=2)
    print(f"  Latents shape: {latents.shape}")  # Expected: [B, 4, 32, 32]
    
    # Step 4: Compute radial power spectrum
    print("\n[4/6] Computing radial power spectrum...")
    frequencies, power_spectrum = compute_radial_power_spectrum(latents)
    print(f"  Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}]")
    print(f"  Power range: [{power_spectrum.min():.4f}, {power_spectrum.max():.4f}]")
    
    # Step 5: Fit power law
    print("\n[5/6] Fitting power-law model P(ω) = A * |ω|^(-β)...")
    A, beta = fit_power_law(frequencies, power_spectrum)
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ Fitted Parameters:               │")
    print(f"  │   A    = {A:.6f}              │")
    print(f"  │   β    = {beta:.6f}               │")
    print(f"  └─────────────────────────────────┘")
    
    # Step 6: Plot
    print("\n[6/6] Generating power spectrum plot...")
    plot_power_spectrum(frequencies, power_spectrum, A, beta, plot_path)
    
    # Step 7: Compute SPEED optimal stage lengths
    print("\n" + "=" * 60)
    print("SPEED Optimal Stage Lengths")
    print("=" * 60)
    
    scales = [0.25, 0.5, 1.0]
    delta = 0.01
    
    print(f"\n  Scale config: {scales}")
    print(f"  Delta: {delta}")
    print(f"  Fitted A={A:.6f}, β={beta:.6f}")
    print(f"  Latent size: 32×32")
    
    transitions = delta_optimal_transitions(
        scales=scales, delta=delta, A=A, beta=beta, H=32, W=32
    )
    print(f"\n  Transition times (normalized [0,1]): {[f'{t:.4f}' for t in transitions]}")
    
    # DDPM (T=1000 steps)
    stage_lengths_ddpm = transitions_to_stage_lengths(transitions, T_total=1000)
    print(f"\n  DDPM (T=1000 total steps):")
    for i, (s, l) in enumerate(zip(scales, stage_lengths_ddpm)):
        print(f"    Stage {i+1} (scale={s}): {l:.1f} steps")
    
    # Flow Matching (T=1.0)
    stage_lengths_fm = transitions_to_stage_lengths(transitions, T_total=1.0)
    print(f"\n  Flow Matching (T=1.0):")
    for i, (s, l) in enumerate(zip(scales, stage_lengths_fm)):
        print(f"    Stage {i+1} (scale={s}): {l:.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
