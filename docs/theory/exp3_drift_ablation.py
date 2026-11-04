"""
Experiment 3: Drift Design Ablation Study
==========================================
Analyzes how different drift envelope designs affect training signal
in both DDPM and Flow Matching frameworks.

Key questions:
- Which κ(t) gives the most uniform training signal?
- For FM, which envelope satisfies ∫g'(t)dt=0 AND gives most uniform velocity?
- What is the "natural" strength of drift relative to the main signal?
- Is there evidence that λ≠1 might be better?
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Output directory
OUT_DIR = Path(__file__).parent
os.makedirs(OUT_DIR, exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# =============================================================================
# Section 1: Kappa envelope functions and their derivatives
# =============================================================================

def kappa_linear(s):
    """Linear: κ(s) = s where s = (t - T_c) / (T_{c+1} - T_c) ∈ [0,1]"""
    return s

def kappa_cosine(s):
    """Cosine: κ(s) = 0.5 * (1 - cos(π*s))"""
    return 0.5 * (1 - np.cos(np.pi * s))

def kappa_quadratic(s):
    """Quadratic: κ(s) = s²"""
    return s ** 2

def kappa_sqrt(s):
    """Sqrt: κ(s) = √s"""
    return np.sqrt(s)

def kappa_linear_deriv(s):
    return np.ones_like(s)

def kappa_cosine_deriv(s):
    return 0.5 * np.pi * np.sin(np.pi * s)

def kappa_quadratic_deriv(s):
    return 2 * s

def kappa_sqrt_deriv(s):
    return 0.5 / np.sqrt(np.maximum(s, 1e-8))

KAPPA_VARIANTS = {
    'Linear': (kappa_linear, kappa_linear_deriv),
    'Cosine': (kappa_cosine, kappa_cosine_deriv),
    'Quadratic': (kappa_quadratic, kappa_quadratic_deriv),
    'Sqrt': (kappa_sqrt, kappa_sqrt_deriv),
}


# =============================================================================
# Section 2: Generate synthetic data
# =============================================================================

def generate_synthetic_data(batch=64, channels=4, size=32):
    """Generate x_1 and compute drift d = upsample(downsample(x_1)) - x_1"""
    x_1 = torch.randn(batch, channels, size, size)

    # Downsample by factor 2 using avg_pool2d
    x_down = F.avg_pool2d(x_1, kernel_size=2, stride=2)

    # Upsample back using repeat_interleave (nearest neighbor)
    x_up = x_down.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    # Drift: difference between low-pass filtered and original
    d = x_up - x_1

    return x_1, d


# =============================================================================
# Section 3: DDPM alpha_bar schedule (cosine schedule)
# =============================================================================

def cosine_alpha_bar(t, s=0.008):
    """Cosine schedule for alpha_bar as in Improved DDPM"""
    f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return f_t / f_0


# =============================================================================
# Section 4: Flow Matching envelopes
# =============================================================================

def fm_envelope_t1t(t):
    """g(t) = t(1-t), g'(t) = 1-2t, ∫₀¹ g'(t)dt = 0"""
    return t * (1 - t)

def fm_envelope_t1t_deriv(t):
    return 1 - 2 * t

def fm_envelope_sin(t):
    """g(t) = sin(πt)/π, g'(t) = cos(πt), ∫₀¹ g'(t)dt = 0"""
    return np.sin(np.pi * t) / np.pi

def fm_envelope_sin_deriv(t):
    return np.cos(np.pi * t)

def fm_envelope_t2_1mt2(t):
    """g(t) = t²(1-t)², g'(t) = 2t(1-t)(1-2t), ∫₀¹ g'(t)dt = 0"""
    return (t ** 2) * ((1 - t) ** 2)

def fm_envelope_t2_1mt2_deriv(t):
    return 2 * t * (1 - t) * (1 - 2 * t)

def fm_envelope_linear(t):
    """g(t) = t, g'(t) = 1, ∫₀¹ g'(t)dt = 1 (does NOT satisfy endpoint condition!)"""
    return t

def fm_envelope_linear_deriv(t):
    return np.ones_like(t)

FM_ENVELOPES = {
    't(1-t)': (fm_envelope_t1t, fm_envelope_t1t_deriv),
    'sin(πt)/π': (fm_envelope_sin, fm_envelope_sin_deriv),
    't²(1-t)²': (fm_envelope_t2_1mt2, fm_envelope_t2_1mt2_deriv),
    't (BAD)': (fm_envelope_linear, fm_envelope_linear_deriv),
}


# =============================================================================
# Plot 1: Drift kappa shapes
# =============================================================================

def plot_kappa_shapes():
    print("\n[Plot 1] Generating drift_kappa_shapes.png...")
    s = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # κ(s) values
    ax = axes[0]
    for name, (kfunc, _) in KAPPA_VARIANTS.items():
        ax.plot(s, kfunc(s), linewidth=2, label=name)
    ax.set_xlabel('s = (t - T_c) / (T_{c+1} - T_c)')
    ax.set_ylabel('κ(s)')
    ax.set_title('Drift Envelope Functions κ(s)')
    ax.legend()

    # κ'(s) derivatives
    ax = axes[1]
    s_inner = np.linspace(0.01, 0.99, 200)  # avoid endpoints for sqrt
    for name, (_, kderiv) in KAPPA_VARIANTS.items():
        ax.plot(s_inner, kderiv(s_inner), linewidth=2, label=name)
    ax.set_xlabel('s = (t - T_c) / (T_{c+1} - T_c)')
    ax.set_ylabel("κ'(s)")
    ax.set_title("Drift Envelope Derivatives κ'(s)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'drift_kappa_shapes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved drift_kappa_shapes.png")


# =============================================================================
# Plot 2: DDPM target inflation
# =============================================================================

def plot_ddpm_target_inflation():
    print("\n[Plot 2] Generating ddpm_target_inflation.png...")
    x_1, d = generate_synthetic_data()

    # Compute norms
    eps = torch.randn_like(x_1)
    eps_norm_sq = (eps ** 2).sum(dim=(1, 2, 3)).mean().item()  # E[||ε||²]
    d_norm_sq = (d ** 2).sum(dim=(1, 2, 3)).mean().item()  # E[||d||²]

    # Cross term E[ε·d]
    cross = (eps * d).sum(dim=(1, 2, 3)).mean().item()

    print(f"  E[||ε||²] = {eps_norm_sq:.2f}")
    print(f"  E[||d||²] = {d_norm_sq:.2f}")
    print(f"  E[ε·d] = {cross:.4f} (should be ~0)")

    t_values = np.linspace(0.01, 0.99, 200)
    alpha_bar = cosine_alpha_bar(t_values)

    # For DDPM: ε̃ = ε + √(ᾱ_t/(1-ᾱ_t)) · κ(t) · d
    # E[||ε̃||²] = E[||ε||²] + (ᾱ_t/(1-ᾱ_t)) · κ(t)² · E[||d||²] + 2·√(ᾱ_t/(1-ᾱ_t))·κ(t)·E[ε·d]
    # Ratio = 1 + (ᾱ_t/(1-ᾱ_t)) · κ(t)² · (E[||d||²]/E[||ε||²])

    snr = alpha_bar / (1 - alpha_bar)
    d_to_eps_ratio = d_norm_sq / eps_norm_sq

    fig, ax = plt.subplots(figsize=(10, 6))

    max_ratios = {}
    for name, (kfunc, _) in KAPPA_VARIANTS.items():
        kappa_sq = kfunc(t_values) ** 2
        inflation = 1 + snr * kappa_sq * d_to_eps_ratio
        ax.plot(t_values, inflation, linewidth=2, label=name)
        max_ratios[name] = np.max(inflation)

    ax.set_xlabel('t')
    ax.set_ylabel('||ε̃||² / ||ε||²')
    ax.set_title('DDPM Target Inflation Ratio vs t\n(How much drift inflates the noise prediction target)')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ddpm_target_inflation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Max inflation ratios:")
    for name, ratio in sorted(max_ratios.items(), key=lambda x: x[1]):
        print(f"    {name}: {ratio:.4f}")
    print("  Saved ddpm_target_inflation.png")
    return max_ratios


# =============================================================================
# Plot 3: FM velocity magnitude
# =============================================================================

def plot_fm_velocity_magnitude():
    print("\n[Plot 3] Generating fm_velocity_magnitude.png...")
    x_1, d = generate_synthetic_data()
    eps = torch.randn_like(x_1)

    # Standard FM: u_t = x_1 - ε (constant)
    u_standard = x_1 - eps
    u_std_norm_sq = (u_standard ** 2).sum(dim=(1, 2, 3)).mean().item()

    # d norm
    d_norm_sq = (d ** 2).sum(dim=(1, 2, 3)).mean().item()
    # cross term (x_1 - eps) · d
    cross = ((x_1 - eps) * d).sum(dim=(1, 2, 3)).mean().item()

    t_values = np.linspace(0.01, 0.99, 300)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Standard FM baseline
    ax.axhline(y=u_std_norm_sq, color='black', linestyle='--', linewidth=1.5, label='Standard FM (constant)')

    velocity_variances = {}
    integral_checks = {}

    for name, (gfunc, gderiv) in FM_ENVELOPES.items():
        # u_t = (x_1 - ε) + g'(t) · d
        # ||u_t||² = ||x_1 - ε||² + g'(t)² · ||d||² + 2·g'(t)·(x_1-ε)·d
        g_prime = gderiv(t_values)
        velocity_norm_sq = u_std_norm_sq + g_prime ** 2 * d_norm_sq + 2 * g_prime * cross

        ax.plot(t_values, velocity_norm_sq, linewidth=2, label=name)
        velocity_variances[name] = np.var(velocity_norm_sq)

        # Check integral condition: ∫₀¹ g'(t)dt = g(1) - g(0)
        integral = gfunc(1.0) - gfunc(0.0)
        integral_checks[name] = integral

    ax.set_xlabel('t')
    ax.set_ylabel('||u_t||²')
    ax.set_title('Flow Matching Velocity Magnitude vs t\n(For different drift envelopes)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fm_velocity_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Velocity variance across t (lower = more uniform):")
    for name, var in sorted(velocity_variances.items(), key=lambda x: x[1]):
        integral = integral_checks[name]
        status = "✓" if abs(integral) < 1e-6 else f"✗ (∫g'dt={integral:.4f})"
        print(f"    {name}: var={var:.2f}, endpoint: {status}")
    print("  Saved fm_velocity_magnitude.png")
    return velocity_variances, integral_checks


# =============================================================================
# Plot 4: FM velocity uniformity bar chart
# =============================================================================

def plot_fm_velocity_uniformity(velocity_variances, integral_checks):
    print("\n[Plot 4] Generating fm_velocity_uniformity.png...")

    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(velocity_variances.keys())
    variances = [velocity_variances[n] for n in names]
    colors = ['green' if abs(integral_checks[n]) < 1e-6 else 'red' for n in names]

    bars = ax.bar(names, variances, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Variance of ||u_t||² across t')
    ax.set_title('Velocity Uniformity Comparison\n(Green = satisfies ∫g\'(t)dt=0, Red = does not)')
    ax.set_xlabel('Envelope function')

    # Add value labels
    for bar, val in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fm_velocity_uniformity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fm_velocity_uniformity.png")


# =============================================================================
# Plot 5: Drift strength sweep
# =============================================================================

def plot_drift_strength_sweep():
    print("\n[Plot 5] Generating drift_strength_sweep.png...")
    x_1, d = generate_synthetic_data()
    eps = torch.randn_like(x_1)

    eps_norm_sq = (eps ** 2).sum(dim=(1, 2, 3)).mean().item()
    d_norm_sq = (d ** 2).sum(dim=(1, 2, 3)).mean().item()

    lambdas = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    t_values = np.linspace(0.01, 0.99, 200)
    alpha_bar = cosine_alpha_bar(t_values)
    snr = alpha_bar / (1 - alpha_bar)

    # Use linear kappa for this analysis
    kappa = kappa_linear(t_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot a: Max inflation factor vs lambda
    ax = axes[0]
    max_inflations = []
    mean_inflations = []
    for lam in lambdas:
        inflation = 1 + snr * kappa ** 2 * lam ** 2 * (d_norm_sq / eps_norm_sq)
        max_inflations.append(np.max(inflation))
        mean_inflations.append(np.mean(inflation))

    ax.plot(lambdas, max_inflations, 'o-', linewidth=2, markersize=8, label='Max inflation')
    ax.plot(lambdas, mean_inflations, 's--', linewidth=2, markersize=8, label='Mean inflation')
    ax.set_xlabel('λ (drift strength multiplier)')
    ax.set_ylabel('Loss amplification factor')
    ax.set_title('DDPM Loss Amplification vs Drift Strength λ')
    ax.legend()
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)

    # Plot b: Effective SNR modification at different frequencies
    ax = axes[1]
    # d primarily captures high-frequency content removed by downsampling
    # Decompose into frequency bands
    freq_bands = ['Low freq\n(kept by pool)', 'High freq\n(removed by pool)']

    # Estimate: d has mostly high-freq content, x_1 has both
    # Ratio of high-freq energy in d vs x_1
    x_1_fft = torch.fft.fft2(x_1)
    d_fft = torch.fft.fft2(d)

    # Split into low and high frequency
    h, w = x_1.shape[-2], x_1.shape[-1]
    mask_low = torch.zeros(h, w)
    mask_low[:h // 4, :w // 4] = 1
    mask_low[:h // 4, -w // 4:] = 1
    mask_low[-h // 4:, :w // 4] = 1
    mask_low[-h // 4:, -w // 4:] = 1

    mask_high = 1 - mask_low

    x1_low = (torch.abs(x_1_fft) ** 2 * mask_low).sum(dim=(1, 2, 3)).mean().item()
    x1_high = (torch.abs(x_1_fft) ** 2 * mask_high).sum(dim=(1, 2, 3)).mean().item()
    d_low = (torch.abs(d_fft) ** 2 * mask_low).sum(dim=(1, 2, 3)).mean().item()
    d_high = (torch.abs(d_fft) ** 2 * mask_high).sum(dim=(1, 2, 3)).mean().item()

    print(f"  Frequency analysis:")
    print(f"    x_1 low-freq energy: {x1_low:.2f}, high-freq: {x1_high:.2f}")
    print(f"    d low-freq energy: {d_low:.2f}, high-freq: {d_high:.2f}")
    print(f"    d high-freq / d total: {d_high / (d_low + d_high):.4f}")

    # SNR modification by lambda at t=0.5 (midpoint)
    t_mid = 0.5
    ab_mid = cosine_alpha_bar(t_mid)
    snr_mid = ab_mid / (1 - ab_mid)

    snr_mod_low = []
    snr_mod_high = []
    for lam in lambdas:
        # Effective SNR modification: SNR_eff = SNR_orig * (1 + λ²·κ²·d_freq/eps_freq)
        mod_l = 1 + lam ** 2 * 0.25 * (d_low / (eps_norm_sq * x1_low / (x1_low + x1_high)))
        mod_h = 1 + lam ** 2 * 0.25 * (d_high / (eps_norm_sq * x1_high / (x1_low + x1_high)))
        snr_mod_low.append(mod_l)
        snr_mod_high.append(mod_h)

    ax.plot(lambdas, snr_mod_low, 'o-', linewidth=2, markersize=8, label='Low-freq SNR mod')
    ax.plot(lambdas, snr_mod_high, 's-', linewidth=2, markersize=8, label='High-freq SNR mod')
    ax.set_xlabel('λ (drift strength multiplier)')
    ax.set_ylabel('Effective SNR modification factor')
    ax.set_title('SNR Modification at Different Frequencies (t=0.5)')
    ax.legend()
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'drift_strength_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  λ=1 max inflation: {max_inflations[lambdas.index(1.0)]:.4f}")
    print(f"  λ=0.5 max inflation: {max_inflations[lambdas.index(0.5)]:.4f}")
    print("  Saved drift_strength_sweep.png")
    return lambdas, max_inflations, mean_inflations


# =============================================================================
# Plot 6: Drift statistics
# =============================================================================

def plot_drift_statistics():
    print("\n[Plot 6] Generating drift_statistics.png...")
    x_1, d = generate_synthetic_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Histogram of d values
    ax = axes[0]
    d_flat = d.flatten().numpy()
    x_flat = x_1.flatten().numpy()
    ax.hist(d_flat, bins=100, density=True, alpha=0.7, color='blue', label='d values')
    ax.hist(x_flat, bins=100, density=True, alpha=0.5, color='orange', label='x_1 values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of drift d vs x_1')
    ax.legend()

    d_mean = d_flat.mean()
    d_std = d_flat.std()
    x_std = x_flat.std()
    print(f"  d: mean={d_mean:.6f}, std={d_std:.4f}")
    print(f"  x_1: std={x_std:.4f}")
    print(f"  Ratio std(d)/std(x_1): {d_std / x_std:.4f}")

    # (b) Scatter: d vs x_1 (subsample for visibility)
    ax = axes[1]
    n_points = 5000
    idx = np.random.choice(len(d_flat), n_points, replace=False)
    ax.scatter(x_flat[idx], d_flat[idx], alpha=0.1, s=1, color='blue')
    ax.set_xlabel('x_1')
    ax.set_ylabel('d')
    ax.set_title(f'd vs x_1 (corr={np.corrcoef(x_flat[:10000], d_flat[:10000])[0,1]:.4f})')

    # (c) Frequency spectrum of d
    ax = axes[2]
    # Average power spectrum
    d_fft = torch.fft.fft2(d)
    power = (torch.abs(d_fft) ** 2).mean(dim=(0, 1))  # Average over batch and channels
    # Radial average
    h, w = power.shape
    cy, cx = h // 2, w // 2
    power_shifted = torch.fft.fftshift(power)

    max_radius = min(cy, cx)
    radii = torch.zeros(max_radius)
    counts = torch.zeros(max_radius)
    for y in range(h):
        for x in range(w):
            r = int(np.sqrt((y - cy) ** 2 + (x - cx) ** 2))
            if r < max_radius:
                radii[r] += power_shifted[y, x]
                counts[r] += 1

    radial_power = (radii / counts.clamp(min=1)).numpy()
    freqs = np.arange(max_radius) / max_radius

    # Also compute for x_1
    x1_fft = torch.fft.fft2(x_1)
    x1_power = (torch.abs(x1_fft) ** 2).mean(dim=(0, 1))
    x1_power_shifted = torch.fft.fftshift(x1_power)
    x1_radii = torch.zeros(max_radius)
    x1_counts = torch.zeros(max_radius)
    for y in range(h):
        for x in range(w):
            r = int(np.sqrt((y - cy) ** 2 + (x - cx) ** 2))
            if r < max_radius:
                x1_radii[r] += x1_power_shifted[y, x]
                x1_counts[r] += 1
    x1_radial_power = (x1_radii / x1_counts.clamp(min=1)).numpy()

    ax.semilogy(freqs, radial_power, linewidth=2, label='d (drift)')
    ax.semilogy(freqs, x1_radial_power, linewidth=2, label='x_1 (original)')
    ax.set_xlabel('Normalized frequency')
    ax.set_ylabel('Power spectral density')
    ax.set_title('Radial Power Spectrum')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'drift_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved drift_statistics.png")

    return d_std, x_std


# =============================================================================
# Main execution and findings report
# =============================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 3: DRIFT DESIGN ABLATION STUDY")
    print("=" * 70)

    # Generate all plots
    plot_kappa_shapes()
    ddpm_max_ratios = plot_ddpm_target_inflation()
    fm_variances, fm_integrals = plot_fm_velocity_magnitude()
    plot_fm_velocity_uniformity(fm_variances, fm_integrals)
    lambdas, max_inflations, mean_inflations = plot_drift_strength_sweep()
    d_std, x_std = plot_drift_statistics()

    # =================================================================
    # Key Findings Report
    # =================================================================
    print("\n")
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Finding 1: Which κ gives most uniform training signal
    print("\n1. MOST UNIFORM TRAINING SIGNAL (DDPM):")
    print("   " + "-" * 50)
    sorted_ratios = sorted(ddpm_max_ratios.items(), key=lambda x: x[1])
    best_kappa = sorted_ratios[0][0]
    worst_kappa = sorted_ratios[-1][0]
    print(f"   Best κ(t):  {best_kappa} (max inflation = {sorted_ratios[0][1]:.4f})")
    print(f"   Worst κ(t): {worst_kappa} (max inflation = {sorted_ratios[-1][1]:.4f})")
    print(f"   Improvement: {sorted_ratios[-1][1] / sorted_ratios[0][1]:.2f}x less inflation")
    print(f"   → {best_kappa} κ is recommended for DDPM as it minimizes peak inflation.")

    # Finding 2: FM envelope analysis
    print("\n2. BEST FLOW MATCHING ENVELOPE:")
    print("   " + "-" * 50)
    valid_envelopes = {k: v for k, v in fm_variances.items() if abs(fm_integrals[k]) < 1e-6}
    if valid_envelopes:
        best_fm = min(valid_envelopes.items(), key=lambda x: x[1])
        print(f"   Envelopes satisfying ∫g'(t)dt=0: {list(valid_envelopes.keys())}")
        print(f"   Best (lowest variance): {best_fm[0]} (var = {best_fm[1]:.2f})")
        worst_valid = max(valid_envelopes.items(), key=lambda x: x[1])
        print(f"   Worst valid: {worst_valid[0]} (var = {worst_valid[1]:.2f})")
    else:
        print("   No valid envelopes found!")

    invalid_envelopes = {k: v for k, v in fm_variances.items() if abs(fm_integrals[k]) >= 1e-6}
    if invalid_envelopes:
        print(f"   Invalid envelopes (∫g'dt≠0): {list(invalid_envelopes.keys())}")
        print(f"     → These break the ODE endpoint condition!")

    # Finding 3: Natural drift strength
    print("\n3. NATURAL DRIFT STRENGTH:")
    print("   " + "-" * 50)
    print(f"   std(d) / std(x_1) = {d_std / x_std:.4f}")
    print(f"   This means drift is {d_std / x_std * 100:.1f}% the magnitude of the original signal.")
    print(f"   ||d||² / ||x_1||² ≈ {(d_std / x_std) ** 2:.4f}")
    print(f"   The drift captures high-frequency details lost in downsampling.")

    # Finding 4: Lambda analysis
    print("\n4. OPTIMAL DRIFT STRENGTH λ:")
    print("   " + "-" * 50)
    idx_1 = lambdas.index(1.0)
    print(f"   λ=1.0: max inflation = {max_inflations[idx_1]:.4f}, mean = {mean_inflations[idx_1]:.4f}")
    idx_05 = lambdas.index(0.5)
    print(f"   λ=0.5: max inflation = {max_inflations[idx_05]:.4f}, mean = {mean_inflations[idx_05]:.4f}")
    idx_075 = lambdas.index(0.75)
    print(f"   λ=0.75: max inflation = {max_inflations[idx_075]:.4f}, mean = {mean_inflations[idx_075]:.4f}")

    # Compute the "sweet spot" - where inflation is manageable but drift is meaningful
    # Inflation grows as λ², so halving λ quarters the max inflation
    print(f"\n   Analysis: Inflation ∝ λ². Reducing λ from 1→0.5 cuts max inflation by 4x.")
    print(f"   However, the drift guidance effect also scales with λ.")
    print(f"   Trade-off: λ=0.75 gives 56% of the inflation of λ=1 while retaining 75% of guidance.")

    # Determine if λ≠1 is recommended
    if max_inflations[idx_1] > 5.0:
        print(f"\n   ⚠ WARNING: At λ=1, max inflation ({max_inflations[idx_1]:.1f}x) is very high!")
        print(f"   RECOMMENDATION: Consider λ=0.5-0.75 for more stable training.")
        print(f"   Alternatively, use Cosine/Quadratic κ to reduce peak inflation at t≈0.")
    else:
        print(f"\n   λ=1 appears safe (max inflation = {max_inflations[idx_1]:.2f}x).")
        print(f"   If training is unstable, try λ=0.75 as first intervention.")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<40} {'Value':<30}")
    print("-" * 70)
    print(f"{'Best DDPM κ(t)':<40} {best_kappa:<30}")
    print(f"{'Best FM envelope (valid)':<40} {best_fm[0] if valid_envelopes else 'N/A':<30}")
    print(f"{'Drift/Signal ratio (std)':<40} {d_std/x_std:.4f}")
    print(f"{'Recommended λ':<40} {'0.75-1.0 (context dependent)':<30}")
    print(f"{'Max inflation at λ=1 (linear κ)':<40} {max_inflations[idx_1]:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
