#!/usr/bin/env python3
"""
Visualize frequency-domain SNR comparison between RDD and standard DDPM.

Generates four figures:
  1. SNR vs timestep for different frequencies
  2. SNR vs frequency at fixed timesteps
  3. SNR ratio (RDD/DDPM) heatmap
  4. Comparison with DCTdiff
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def linear_beta_schedule(beta_min, beta_max, T):
    """Linear beta schedule from beta_min to beta_max over T timesteps."""
    return np.linspace(beta_min, beta_max, T)


def compute_alpha_bar(betas):
    """Compute cumulative product of (1 - beta_t)."""
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def compute_kappa(t, stage_boundaries):
    """
    Compute kappa(t) which linearly goes from 0 to 1 within each stage.
    t can be a scalar or array of timestep indices (0-based).
    stage_boundaries: list of boundaries e.g. [0, 300, 600, 1000]
    """
    t = np.asarray(t, dtype=np.float64)
    kappa = np.zeros_like(t)
    for i in range(len(stage_boundaries) - 1):
        start = stage_boundaries[i]
        end = stage_boundaries[i + 1]
        mask = (t >= start) & (t < end)
        length = end - start
        kappa[mask] = (t[mask] - start) / length
    return kappa


def butterworth_lp(f, f_c, order=4):
    """
    Butterworth low-pass filter transfer function.
    LP(f) = 1 / (1 + (f/f_c)^(2*order))
    """
    f = np.asarray(f, dtype=np.float64)
    ratio = np.where(f_c > 0, f / f_c, np.inf)
    return 1.0 / (1.0 + ratio ** (2 * order))


def get_cutoff_freq(stage_idx, resolutions, base_resolution):
    """
    Get the low-pass cutoff frequency for a given stage.
    The cutoff is the Nyquist frequency of the target (downsampled) resolution
    relative to the base resolution.
    """
    if stage_idx >= len(resolutions) - 1:
        # Last stage: maximum downsampling
        target_res = resolutions[-1]
    else:
        target_res = resolutions[stage_idx + 1]
    # Nyquist of target within base: (target_res/2) / base_resolution = target_res / (2*base)
    # In normalized freq (0 to 0.5 for base), cutoff = 0.5 * (target_res / base_resolution)
    f_c = 0.5 * target_res / base_resolution
    return f_c


def compute_snr_ddpm(alpha_bar):
    """SNR_DDPM = alpha_bar / (1 - alpha_bar), independent of frequency."""
    return alpha_bar / (1.0 - alpha_bar)


def compute_snr_rdd(f, t_indices, alpha_bar, stage_boundaries, resolutions, base_resolution, order=4):
    """
    Compute RDD SNR for given frequencies and timesteps.

    SNR_RDD(f,t) = alpha_bar_t * ((1 - kappa(t)) + kappa(t) * LP(f))^2 / (1 - alpha_bar_t)

    The LP filter cutoff depends on which stage t is in.
    """
    f = np.asarray(f, dtype=np.float64)
    t_indices = np.asarray(t_indices, dtype=np.float64)

    # Result shape: (len(t_indices), len(f)) if both are arrays
    if f.ndim == 0:
        f = f.reshape(1)
    if t_indices.ndim == 0:
        t_indices = t_indices.reshape(1)

    snr = np.zeros((len(t_indices), len(f)))

    for i, t in enumerate(t_indices):
        ti = int(t)
        ab = alpha_bar[ti]

        # Determine which stage this timestep belongs to
        stage_idx = 0
        for s in range(len(stage_boundaries) - 1):
            if t >= stage_boundaries[s] and t < stage_boundaries[s + 1]:
                stage_idx = s
                break

        kappa = (t - stage_boundaries[stage_idx]) / (stage_boundaries[stage_idx + 1] - stage_boundaries[stage_idx])
        f_c = get_cutoff_freq(stage_idx, resolutions, base_resolution)
        lp = butterworth_lp(f, f_c, order=order)

        signal_factor = ((1.0 - kappa) + kappa * lp) ** 2
        snr[i, :] = ab * signal_factor / (1.0 - ab)

    return snr


def compute_snr_dctdiff(alpha_bar, S=4.0):
    """
    DCTdiff SNR: SNR_DCT(f,t) = S * alpha_bar / (1 - alpha_bar)
    S is a global scale factor, same for all frequencies.
    """
    return S * alpha_bar / (1.0 - alpha_bar)


def compute_optimal_stage_lengths_speed(A, beta_param, delta=0.01, T=1000):
    """
    Compute optimal stage lengths based on power spectrum activation times.

    Given power spectrum P(omega) = A * |omega|^{-beta}, compute when the
    SNR at the Nyquist frequency of each scale drops below delta.

    Parameters:
        A: amplitude of power spectrum
        beta_param: spectral exponent (P(omega) = A * |omega|^{-beta})
        delta: SNR threshold for activation
        T: total number of timesteps

    Returns:
        stage_lengths: list of optimal stage lengths
    """
    # Standard linear beta schedule
    betas = linear_beta_schedule(0.0001, 0.02, T)
    alpha_bar = compute_alpha_bar(betas)

    # Resolutions and their Nyquist frequencies (normalized)
    resolutions = [32, 16, 8]
    base_resolution = resolutions[0]
    nyquist_freqs = [0.5 * r / base_resolution for r in resolutions]

    # Compute signal power at each Nyquist frequency from the power spectrum
    # P(f) = A * |f|^{-beta} (in normalized frequency)
    activation_times = []
    for f_nyq in nyquist_freqs:
        if f_nyq <= 0:
            activation_times.append(0)
            continue
        # Signal power at this frequency
        signal_power = A * (f_nyq ** (-beta_param))
        # Find t where SNR(f) = signal_power * alpha_bar_t / (1 - alpha_bar_t) drops below delta
        snr_values = signal_power * alpha_bar / (1.0 - alpha_bar)
        # Activation time: when SNR first drops below delta
        below_delta = np.where(snr_values < delta)[0]
        if len(below_delta) > 0:
            activation_times.append(below_delta[0])
        else:
            activation_times.append(T)

    # Convert activation times to stage lengths
    activation_times = sorted(activation_times)
    stage_lengths = []
    prev = 0
    for at in activation_times:
        stage_lengths.append(max(at - prev, 1))
        prev = at
    # Add remaining timesteps to last stage
    if prev < T:
        stage_lengths[-1] += (T - prev)

    return stage_lengths, activation_times


def plot_snr_vs_t(alpha_bar, stage_boundaries, resolutions, base_resolution, output_dir, dpi):
    """Figure 1: SNR vs timestep for different frequencies."""
    print("Figure 1: SNR vs timestep for different frequencies (snr_vs_t.png)")
    print("  Shows how SNR decays over time. DDPM is frequency-independent,")
    print("  while RDD attenuates high frequencies more aggressively.\n")

    T = len(alpha_bar)
    t_indices = np.arange(T)
    freqs = [0.1, 0.3, 0.5]
    freq_labels = ['f=0.1 (low)', 'f=0.3 (mid)', 'f=0.5 (Nyquist)']
    colors = ['#2196F3', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(8, 5))

    # DDPM SNR (frequency-independent)
    snr_ddpm = compute_snr_ddpm(alpha_bar)
    ax.plot(t_indices, np.log10(snr_ddpm), 'k--', linewidth=2, label='DDPM (all freq)')

    # RDD SNR at different frequencies
    for idx, (f, label, color) in enumerate(zip(freqs, freq_labels, colors)):
        snr_rdd = compute_snr_rdd(
            np.array([f]), t_indices, alpha_bar,
            stage_boundaries, resolutions, base_resolution
        )
        ax.plot(t_indices, np.log10(snr_rdd[:, 0]), color=color, linewidth=1.5, label=f'RDD {label}')

    # Stage boundaries
    for sb in stage_boundaries[1:-1]:
        ax.axvline(x=sb, color='gray', linestyle=':', alpha=0.7, linewidth=1)

    ax.set_xlabel('Timestep t', fontsize=12)
    ax.set_ylabel('log₁₀(SNR)', fontsize=12)
    ax.set_title('SNR vs Timestep for Different Frequencies', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, T)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_vs_t.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_snr_vs_f(alpha_bar, stage_boundaries, resolutions, base_resolution, output_dir, dpi):
    """Figure 2: SNR vs frequency at fixed timesteps."""
    print("Figure 2: SNR vs frequency at fixed timesteps (snr_vs_f.png)")
    print("  Shows frequency-dependent SNR at selected timesteps.")
    print("  DDPM is flat; RDD rolls off at high frequencies.\n")

    timesteps = [100, 300, 500, 700, 900]
    freqs = np.linspace(0.01, 0.5, 200)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(timesteps)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, t in enumerate(timesteps):
        # DDPM (horizontal line)
        snr_ddpm = compute_snr_ddpm(alpha_bar[t])
        ax.axhline(y=np.log10(snr_ddpm), color=colors[idx], linestyle='--', alpha=0.5, linewidth=1)

        # RDD
        snr_rdd = compute_snr_rdd(
            freqs, np.array([t]), alpha_bar,
            stage_boundaries, resolutions, base_resolution
        )
        ax.plot(freqs, np.log10(snr_rdd[0, :]), color=colors[idx], linewidth=2,
                label=f't={t}')

    ax.set_xlabel('Normalized Frequency f', fontsize=12)
    ax.set_ylabel('log₁₀(SNR)', fontsize=12)
    ax.set_title('SNR vs Frequency at Fixed Timesteps', fontsize=13)
    ax.legend(fontsize=10, title='Solid=RDD, Dashed=DDPM')
    ax.set_xlim(0, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_vs_f.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_snr_ratio_heatmap(alpha_bar, stage_boundaries, resolutions, base_resolution, output_dir, dpi):
    """Figure 3: SNR ratio (RDD/DDPM) heatmap."""
    print("Figure 3: SNR ratio heatmap (snr_ratio_heatmap.png)")
    print("  Shows log10(SNR_RDD / SNR_DDPM) across frequency and time.")
    print("  Blue = RDD attenuates, White = same as DDPM.\n")

    T = len(alpha_bar)
    freqs = np.linspace(0.01, 0.5, 100)
    t_indices = np.arange(0, T, 2)  # subsample for efficiency

    # Compute RDD SNR
    snr_rdd = compute_snr_rdd(
        freqs, t_indices, alpha_bar,
        stage_boundaries, resolutions, base_resolution
    )

    # Compute DDPM SNR (broadcast over frequencies)
    snr_ddpm = compute_snr_ddpm(alpha_bar[t_indices.astype(int)])
    snr_ddpm_2d = snr_ddpm[:, np.newaxis] * np.ones((1, len(freqs)))

    # Ratio
    ratio = np.log10(snr_rdd / snr_ddpm_2d)

    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = max(abs(ratio.min()), abs(ratio.max()))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.imshow(
        ratio, aspect='auto', origin='lower',
        extent=[freqs[0], freqs[-1], t_indices[0], t_indices[-1]],
        cmap='RdBu_r', norm=norm
    )

    # Stage boundaries
    for sb in stage_boundaries[1:-1]:
        ax.axhline(y=sb, color='black', linestyle='--', alpha=0.5, linewidth=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(SNR_RDD / SNR_DDPM)', fontsize=11)

    ax.set_xlabel('Normalized Frequency', fontsize=12)
    ax.set_ylabel('Timestep', fontsize=12)
    ax.set_title('SNR Ratio: RDD vs DDPM', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_ratio_heatmap.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_rdd_vs_dctdiff(alpha_bar, stage_boundaries, resolutions, base_resolution, output_dir, dpi):
    """Figure 4: Comparison with DCTdiff."""
    print("Figure 4: RDD vs DCTdiff comparison (snr_rdd_vs_dctdiff.png)")
    print("  Shows that DCTdiff applies uniform SNR scaling across all frequencies,")
    print("  while RDD is frequency-adaptive: protects low freq, attenuates high freq.\n")

    t_rep = 500  # representative timestep
    freqs = np.linspace(0.01, 0.5, 200)
    S = 4.0  # DCTdiff global scale factor

    # DDPM baseline
    snr_ddpm = compute_snr_ddpm(alpha_bar[t_rep])
    snr_ddpm_line = np.full_like(freqs, np.log10(snr_ddpm))

    # DCTdiff
    snr_dct = compute_snr_dctdiff(alpha_bar[t_rep], S=S)
    snr_dct_line = np.full_like(freqs, np.log10(snr_dct))

    # RDD
    snr_rdd = compute_snr_rdd(
        freqs, np.array([t_rep]), alpha_bar,
        stage_boundaries, resolutions, base_resolution
    )
    snr_rdd_line = np.log10(snr_rdd[0, :])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(freqs, snr_ddpm_line, 'k--', linewidth=2, label='DDPM (baseline)')
    ax.plot(freqs, snr_dct_line, color='#9C27B0', linestyle='-.', linewidth=2,
            label=f'DCTdiff (uniform S={S})')
    ax.plot(freqs, snr_rdd_line, color='#E91E63', linewidth=2.5, label='RDD (adaptive)')

    # Annotations
    ax.annotate('Low-freq preserved', xy=(0.05, snr_rdd_line[5]),
                xytext=(0.08, snr_rdd_line[5] + 0.3),
                fontsize=9, color='#E91E63',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1))
    ax.annotate('High-freq attenuated', xy=(0.45, snr_rdd_line[-10]),
                xytext=(0.32, snr_rdd_line[-10] - 0.4),
                fontsize=9, color='#E91E63',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1))

    ax.set_xlabel('Normalized Frequency f', fontsize=12)
    ax.set_ylabel('log₁₀(SNR)', fontsize=12)
    ax.set_title(f'SNR Comparison at t={t_rep}: RDD vs DCTdiff', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_rdd_vs_dctdiff.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize frequency-domain SNR comparison: RDD vs DDPM vs DCTdiff'
    )
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output figures (default: current directory)')
    parser.add_argument('--stage-lengths', type=str, default='300,300,400',
                        help='Comma-separated stage lengths from finest to coarsest (default: 300,300,400)')
    parser.add_argument('--beta-min', type=float, default=0.0001,
                        help='Minimum beta for linear schedule (default: 0.0001)')
    parser.add_argument('--beta-max', type=float, default=0.02,
                        help='Maximum beta for linear schedule (default: 0.02)')
    parser.add_argument('--T', type=int, default=1000,
                        help='Total number of timesteps (default: 1000)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure resolution in DPI (default: 300)')
    parser.add_argument('--compute-speed-optimal', action='store_true',
                        help='Compute and print optimal stage lengths from power spectrum')

    args = parser.parse_args()

    # Parse stage lengths
    stage_lengths = [int(x) for x in args.stage_lengths.split(',')]
    assert sum(stage_lengths) == args.T, \
        f"Stage lengths must sum to T={args.T}, got {sum(stage_lengths)}"

    # Stage configuration
    stage_boundaries = [0]
    for sl in stage_lengths:
        stage_boundaries.append(stage_boundaries[-1] + sl)

    # Resolutions: from finest to coarsest (matching stages)
    # Default: 32 -> 16 -> 8
    resolutions = [32, 16, 8]
    base_resolution = resolutions[0]

    # Noise schedule
    betas = linear_beta_schedule(args.beta_min, args.beta_max, args.T)
    alpha_bar = compute_alpha_bar(betas)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Frequency-Domain SNR Visualization: RDD vs DDPM")
    print("=" * 60)
    print(f"  Stage lengths: {stage_lengths}")
    print(f"  Stage boundaries: {stage_boundaries}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Beta schedule: linear [{args.beta_min}, {args.beta_max}], T={args.T}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60 + "\n")

    # Generate all figures
    plot_snr_vs_t(alpha_bar, stage_boundaries, resolutions, base_resolution, args.output_dir, args.dpi)
    plot_snr_vs_f(alpha_bar, stage_boundaries, resolutions, base_resolution, args.output_dir, args.dpi)
    plot_snr_ratio_heatmap(alpha_bar, stage_boundaries, resolutions, base_resolution, args.output_dir, args.dpi)
    plot_rdd_vs_dctdiff(alpha_bar, stage_boundaries, resolutions, base_resolution, args.output_dir, args.dpi)

    print("All figures saved successfully.")

    # Optionally compute optimal stage lengths
    if args.compute_speed_optimal:
        print("\n" + "=" * 60)
        print("  Computing Optimal Stage Lengths from Power Spectrum")
        print("=" * 60)
        # Typical natural image: P(omega) = A * |omega|^{-beta}
        # beta ~ 2 for natural images (1/f^2 spectrum)
        A = 1.0
        beta_param = 2.0
        optimal_lengths, activation_times = compute_optimal_stage_lengths_speed(
            A, beta_param, delta=0.01, T=args.T
        )
        print(f"  Power spectrum: P(ω) = {A} * |ω|^{{-{beta_param}}}")
        print(f"  SNR threshold (delta): 0.01")
        print(f"  Activation times: {activation_times}")
        print(f"  Optimal stage lengths: {optimal_lengths}")
        print("=" * 60)


if __name__ == '__main__':
    main()
