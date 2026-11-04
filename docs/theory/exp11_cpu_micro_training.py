"""
Experiment 11: CPU Micro Training - RDD-FM vs Standard FM vs DDPM
=================================================================
Proof-of-concept experiment verifying:
1. RDD-FM trains correctly and loss decreases
2. FM converges faster than DDPM in the RDD framework
3. Generated samples show coarse-to-fine structure

All runs on CPU with tiny MLP models and synthetic 8x8 data.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration - optimized for CPU speed
# ============================================================
SEED = 42
IMG_SIZE = 8
CHANNELS = 1
IMG_DIM = IMG_SIZE * IMG_SIZE * CHANNELS  # 64
BATCH_SIZE = 64
LR = 1e-3
HIDDEN = 128
N_TRAIN = 500
N_VAL = 100
N_STEPS = 5000
LOG_EVERY = 50
SAMPLE_AT = [100, 500, 1000, 5000]  # Steps at which to generate samples
N_GEN = 10  # Samples to generate (small for speed)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Synthetic Dataset
# ============================================================
def generate_data(n, seed=42):
    """Generate Gaussian blob images (low-frequency, natural-like)."""
    torch.manual_seed(seed)
    imgs = torch.zeros(n, 1, IMG_SIZE, IMG_SIZE)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, IMG_SIZE),
        torch.linspace(-1, 1, IMG_SIZE), indexing='ij')

    for i in range(n):
        for _ in range(torch.randint(2, 5, (1,)).item()):
            cx, cy = torch.randn(2).tolist()
            cx *= 0.4; cy *= 0.4
            sig = 0.2 + torch.rand(1).item() * 0.4
            amp = 0.5 + torch.rand(1).item() * 0.5
            imgs[i, 0] += amp * torch.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sig**2))

    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-8)
    return imgs.clamp(-1, 1)


# ============================================================
# Model: 2-layer MLP (fast on CPU)
# ============================================================
class TinyMLP(nn.Module):
    def __init__(self, dim=IMG_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        # x: [B, 1, 8, 8], t: [B]
        B = x.shape[0]
        inp = torch.cat([x.view(B, -1), t.unsqueeze(-1)], dim=-1)
        return self.net(inp).view(B, 1, IMG_SIZE, IMG_SIZE)


# ============================================================
# Training functions (all unified for speed)
# ============================================================
def train_all_methods(train_loader, val_data, n_steps):
    """Train all 5 methods and collect losses + samples."""

    methods = {
        'Standard FM': {'family': 'fm', 'drift': 'none'},
        'RDD-FM t(1-t)': {'family': 'fm', 'drift': 't1t'},
        'RDD-FM t²(1-t)²': {'family': 'fm', 'drift': 't2'},
        'Standard DDPM': {'family': 'ddpm', 'drift': 'none'},
        'RDD-DDPM': {'family': 'ddpm', 'drift': 't1t'},
    }

    # DDPM schedule (precompute once)
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    results = {}
    for name, cfg in methods.items():
        print(f"  {name} ...", end=' ', flush=True)
        torch.manual_seed(SEED)
        model = TinyMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        losses = []
        samples_dict = {}
        t0 = time.time()
        step = 0

        while step < n_steps:
            for (x_data,) in train_loader:
                if step >= n_steps:
                    break

                B = x_data.shape[0]

                if cfg['family'] == 'fm':
                    # Flow matching
                    x0 = torch.randn_like(x_data)
                    x1 = x_data
                    t = torch.rand(B)
                    te = t[:, None, None, None]
                    xt = (1 - te) * x0 + te * x1

                    # Base velocity
                    v_target = x1 - x0

                    # Add drift
                    if cfg['drift'] == 't1t':
                        kappa = 2.0 * te * (1 - te)
                        v_target = v_target + kappa * (x1 - xt)
                    elif cfg['drift'] == 't2':
                        kappa = 2.0 * (te**2) * ((1 - te)**2)
                        v_target = v_target + kappa * (x1 - xt)

                    pred = model(xt, t)
                    loss = F.mse_loss(pred, v_target)

                else:
                    # DDPM
                    x0_real = x_data
                    noise = torch.randn_like(x0_real)
                    ts = torch.randint(0, T, (B,))
                    t_norm = ts.float() / T
                    ab = alpha_bar[ts][:, None, None, None]
                    xt = torch.sqrt(ab) * x0_real + torch.sqrt(1 - ab) * noise

                    if cfg['drift'] == 't1t':
                        te = t_norm[:, None, None, None]
                        kappa = 2.0 * te * (1 - te)
                        drift_c = kappa * (x0_real - xt) / (torch.sqrt(1 - ab) + 1e-8)
                        noise_target = noise - drift_c
                    else:
                        noise_target = noise

                    pred = model(xt, t_norm)
                    loss = F.mse_loss(pred, noise_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                if step % LOG_EVERY == 0:
                    losses.append((step, loss.item()))

                # Generate samples at checkpoints
                if step in SAMPLE_AT:
                    with torch.no_grad():
                        model.eval()
                        if cfg['family'] == 'fm':
                            s = _sample_fm(model, N_GEN)
                        else:
                            s = _sample_ddpm(model, N_GEN, alpha_bar, alphas, betas)
                        model.train()
                    samples_dict[step] = s

        elapsed = time.time() - t0
        fl = losses[-1][1] if losses else float('inf')
        print(f"{elapsed:.0f}s, loss={fl:.4f}")
        results[name] = {'losses': losses, 'samples': samples_dict, 'model': model}

    return results


def _sample_fm(model, n, steps=50):
    x = torch.randn(n, 1, IMG_SIZE, IMG_SIZE)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + model(x, t) * dt
    return x.clamp(-1, 1)


def _sample_ddpm(model, n, alpha_bar, alphas, betas, steps=50):
    T = len(betas)
    step_size = max(1, T // steps)
    ts = list(range(T - 1, -1, -step_size))
    x = torch.randn(n, 1, IMG_SIZE, IMG_SIZE)
    for t_idx in ts:
        t_n = torch.full((n,), t_idx / T)
        eps = model(x, t_n)
        beta = betas[t_idx]
        alpha = alphas[t_idx]
        ab = alpha_bar[t_idx]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ab)) * eps)
        if t_idx > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)


# ============================================================
# Plotting
# ============================================================
def make_plots(results, train_data):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    names = list(results.keys())

    # 1. Training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, name in enumerate(names):
        losses = results[name]['losses']
        steps = [l[0] for l in losses]
        vals = [l[1] for l in losses]
        # Smooth
        w = 5
        if len(vals) > w:
            smoothed = np.convolve(vals, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], smoothed, label=name, color=colors[idx], linewidth=1.5)
        else:
            ax.plot(steps, vals, label=name, color=colors[idx], linewidth=1.5)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss vs Step (5 Methods)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: training_curves.png")

    # 2-4. Generated samples at each checkpoint
    for step in SAMPLE_AT:
        fig, axes = plt.subplots(len(names), N_GEN, figsize=(N_GEN * 1.2, len(names) * 1.5))
        fig.suptitle(f'Generated Samples at Step {step}', fontsize=12)
        for row, name in enumerate(names):
            samps = results[name]['samples'].get(step)
            for col in range(N_GEN):
                ax = axes[row, col]
                if samps is not None and col < samps.shape[0]:
                    ax.imshow(samps[col, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(name, fontsize=6, rotation=0, labelpad=50, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'generated_samples_step{step}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: generated_samples_step{step}.png")

    # 5. Convergence comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    thresholds = [2.0, 1.5, 1.0, 0.8, 0.6, 0.5]
    x_pos = np.arange(len(thresholds))
    width = 0.15
    for idx, name in enumerate(names):
        losses = results[name]['losses']
        steps_needed = []
        for thresh in thresholds:
            found = N_STEPS
            for s, l in losses:
                if l <= thresh:
                    found = s
                    break
            steps_needed.append(found)
        offset = (idx - 2) * width
        ax.bar(x_pos + offset, steps_needed, width, label=name, color=colors[idx], alpha=0.8)

    ax.set_xlabel('Loss Threshold')
    ax.set_ylabel('Steps to Reach')
    ax.set_title('Convergence Speed Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'convergence_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: convergence_comparison.png")

    # 6. Frequency analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    real_spec = _power_spectrum(train_data[:20])
    freqs = np.arange(len(real_spec))
    ax.plot(freqs, real_spec.numpy(), 'k-', linewidth=2.5, label='Real Data')

    for idx, name in enumerate(names):
        samps = results[name]['samples'].get(SAMPLE_AT[-1])
        if samps is not None:
            spec = _power_spectrum(samps)
            ax.plot(freqs, spec.numpy(), color=colors[idx], linewidth=1.5,
                    label=name, linestyle='--')

    ax.set_xlabel('Frequency (radial bin)')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Spectrum: Generated vs Real (Step {SAMPLE_AT[-1]})')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'frequency_analysis_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: frequency_analysis_samples.png")


def _power_spectrum(images):
    """Radial power spectrum."""
    imgs = images.mean(dim=1)
    fft = torch.fft.fft2(imgs)
    power = torch.abs(fft) ** 2
    power = torch.fft.fftshift(power, dim=(-2, -1)).mean(dim=0)
    H, W = power.shape
    cx, cy = H // 2, W // 2
    max_r = min(H, W) // 2
    freq_bins = torch.zeros(max_r)
    counts = torch.zeros(max_r)
    for i in range(H):
        for j in range(W):
            r = int(math.sqrt((i-cx)**2 + (j-cy)**2))
            if r < max_r:
                freq_bins[r] += power[i, j]
                counts[r] += 1
    mask = counts > 0
    freq_bins[mask] /= counts[mask]
    return freq_bins


# ============================================================
# Main
# ============================================================
def main():
    total_start = time.time()
    print("=" * 60)
    print("Experiment 11: CPU Micro Training - RDD Verification")
    print("=" * 60)

    # Data
    print("\n[1/4] Generating data...")
    train_data = generate_data(N_TRAIN, seed=SEED)
    val_data = generate_data(N_VAL, seed=SEED + 1)
    loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)
    print(f"  Train: {train_data.shape}, range=[{train_data.min():.2f}, {train_data.max():.2f}]")

    # Train
    print(f"\n[2/4] Training 5 methods x {N_STEPS} steps...")
    results = train_all_methods(loader, val_data, N_STEPS)

    # Plot
    print(f"\n[3/4] Generating plots...")
    make_plots(results, train_data)

    # Report
    print(f"\n[4/4] Results Summary")
    print("=" * 60)
    print("\n--- Final Losses (step 5000) ---")
    final = {}
    for name, r in results.items():
        l = r['losses'][-1][1] if r['losses'] else float('inf')
        final[name] = l
        print(f"  {name:20s}: {l:.4f}")

    print("\n--- Loss at step 500 ---")
    for name, r in results.items():
        for s, l in r['losses']:
            if s >= 500:
                print(f"  {name:20s}: {l:.4f}")
                break

    print("\n--- Loss at step 1000 ---")
    for name, r in results.items():
        for s, l in r['losses']:
            if s >= 1000:
                print(f"  {name:20s}: {l:.4f}")
                break

    print("\n--- Key Comparisons ---")
    fm = final['Standard FM']
    rdd1 = final['RDD-FM t(1-t)']
    rdd2 = final['RDD-FM t²(1-t)²']
    ddpm = final['Standard DDPM']
    rddpm = final['RDD-DDPM']

    print(f"\n  Q1: Does RDD-FM converge faster than Standard FM?")
    if rdd1 < fm:
        print(f"      YES - RDD-FM t(1-t) loss {rdd1:.4f} < Standard FM {fm:.4f} ({(1-rdd1/fm)*100:.1f}% better)")
    else:
        print(f"      Standard FM ({fm:.4f}) ≤ RDD-FM ({rdd1:.4f})")
        print(f"      Note: with tiny model, the added drift complexity may not help at final convergence")

    print(f"\n  Q2: Does FM converge faster than DDPM in RDD?")
    if rdd1 < rddpm:
        print(f"      YES - RDD-FM {rdd1:.4f} < RDD-DDPM {rddpm:.4f}")
    else:
        print(f"      RDD-DDPM ({rddpm:.4f}) ≤ RDD-FM ({rdd1:.4f})")

    print(f"\n  Q3: t(1-t) vs t²(1-t)² envelope?")
    if abs(rdd1 - rdd2) / max(rdd1, rdd2) < 0.05:
        print(f"      SIMILAR - t(1-t)={rdd1:.4f}, t²(1-t)²={rdd2:.4f}")
    elif rdd2 < rdd1:
        print(f"      t²(1-t)² BETTER: {rdd2:.4f} vs {rdd1:.4f}")
    else:
        print(f"      t(1-t) BETTER: {rdd1:.4f} vs {rdd2:.4f}")

    # NN-MSE quality
    print("\n--- Sample Quality (NN-MSE) ---")
    for name, r in results.items():
        samps = r['samples'].get(SAMPLE_AT[-1])
        if samps is not None:
            gen_flat = samps.view(samps.shape[0], -1)
            train_flat = train_data[:200].view(200, -1)
            d = torch.cdist(gen_flat, train_flat)
            score = (d.min(dim=1).values ** 2).mean().item() / gen_flat.shape[1]
            print(f"  {name:20s}: {score:.6f}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Plots saved to: {SAVE_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
