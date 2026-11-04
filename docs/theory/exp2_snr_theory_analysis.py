"""
Experiment 2: Theoretical Analysis of RDD's Frequency-Domain SNR Properties

This script performs theoretical analysis comparing SNR schedules between
standard DDPM and RDD (Resolution-based Drift Diffusion) across different
stage_lengths configurations.

Key analyses:
1. SNR schedules for different stage_lengths configurations
2. Effective SNR computation for RDD with drift
3. Visualization of signal amplitude by stage and frequency
4. Optimal transition point analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Core Diffusion Schedule Functions
# ============================================================================

def linear_beta_schedule(T=1000, beta_min=0.0001, beta_max=0.02):
    """Linear beta schedule from beta_min to beta_max over T timesteps."""
    return np.linspace(beta_min, beta_max, T)


def compute_alpha_bar(betas):
    """Compute cumulative product of (1 - beta_t) = alpha_bar_t."""
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def compute_snr_standard(alpha_bar):
    """Standard DDPM SNR: alpha_bar_t / (1 - alpha_bar_t)."""
    return alpha_bar / (1.0 - alpha_bar)


# ============================================================================
# RDD Drift Model
# ============================================================================

def get_stage_boundaries(stage_lengths):
    """Convert stage_lengths to cumulative boundaries [0, T1, T2, T3]."""
    boundaries = [0]
    for sl in stage_lengths:
        boundaries.append(boundaries[-1] + sl)
    return boundaries


def compute_drift_kappa(t, stage_boundaries):
    """
    Compute drift strength kappa(t) for each timestep.
    Within each stage c (from T_c to T_{c+1}), kappa linearly increases from 0 to 1.
    kappa(t) = (t - T_c) / (T_{c+1} - T_c)
    """
    T = stage_boundaries[-1]
    kappa = np.zeros(T)
    for c in range(len(stage_boundaries) - 1):
        T_c = stage_boundaries[c]
        T_c1 = stage_boundaries[c + 1]
        mask = (np.arange(T) >= T_c) & (np.arange(T) < T_c1)
        stage_t = np.arange(T)[mask]
        kappa[mask] = (stage_t - T_c) / (T_c1 - T_c)
    return kappa


def get_nyquist_freq(stage_idx, resolutions=[32, 16, 8]):
    """Get Nyquist frequency for a given stage based on resolution."""
    return resolutions[stage_idx] // 2


def compute_effective_signal_rdd(t, freq, alpha_bar, stage_boundaries, resolutions=[32, 16, 8]):
    """
    Compute effective signal amplitude for RDD at frequency f and timestep t.
    
    For RDD with drift:
    - At stage c, frequencies f > Nyquist_c are suppressed by the drift
    - The drift d = UP(DOWN(x)) - x removes frequencies above Nyquist
    - Effective signal at frequency f > Nyquist_c: original * (1 - kappa(t))
    - Effective signal at frequency f <= Nyquist_c: original (no suppression)
    
    Returns: effective signal power (not SNR)
    """
    T = stage_boundaries[-1]
    signal = np.sqrt(alpha_bar)  # signal amplitude = sqrt(alpha_bar)
    
    effective_signal = np.copy(signal)
    
    for c in range(len(stage_boundaries) - 1):
        T_c = stage_boundaries[c]
        T_c1 = stage_boundaries[c + 1]
        nyquist_c = get_nyquist_freq(c, resolutions)
        
        if freq > nyquist_c:
            # This frequency is above Nyquist for this stage -> drift suppresses it
            mask = (np.arange(T) >= T_c) & (np.arange(T) < T_c1)
            stage_t = np.arange(T)[mask]
            kappa = (stage_t - T_c) / (T_c1 - T_c)
            effective_signal[mask] *= (1.0 - kappa)
    
    return effective_signal


def compute_snr_rdd(freq, alpha_bar, stage_boundaries, resolutions=[32, 16, 8]):
    """
    Compute effective SNR for RDD at a specific frequency.
    
    SNR_RDD(f, t) = effective_signal(f,t)^2 / noise_variance(t)
    where noise_variance(t) = 1 - alpha_bar_t (same as standard DDPM)
    """
    effective_signal = compute_effective_signal_rdd(
        np.arange(len(alpha_bar)), freq, alpha_bar, stage_boundaries, resolutions
    )
    noise_var = 1.0 - alpha_bar
    # SNR = signal_power / noise_power
    snr = (effective_signal ** 2) / noise_var
    return snr


def compute_snr_standard_freq(freq, alpha_bar, signal_power_spectrum=None):
    """
    Standard DDPM SNR at frequency f.
    SNR(f, t) = alpha_bar_t * P(f) / (1 - alpha_bar_t)
    
    If signal_power_spectrum is None, assume P(f) = 1/f^2 (natural images).
    """
    if signal_power_spectrum is None:
        # Natural image power spectrum: P(f) ~ 1/f^2
        P_f = 1.0 / (freq ** 2) if freq > 0 else 1.0
    else:
        P_f = signal_power_spectrum
    
    return alpha_bar * P_f / (1.0 - alpha_bar)


# ============================================================================
# Analysis Functions
# ============================================================================

def find_snr_crossing(snr_values, threshold=1.0):
    """
    Find the timestep where SNR crosses the threshold (from above).
    In reverse diffusion (denoising), we go from t=T to t=0.
    We want to find when SNR first exceeds threshold going from high t to low t.
    
    Returns the timestep (going forward) where SNR = threshold.
    """
    # Going from t=T-1 to t=0, find where SNR first exceeds threshold
    for t in range(len(snr_values) - 1, -1, -1):
        if snr_values[t] >= threshold:
            return t
    return 0


def compute_activation_time(snr_values, threshold=1.0):
    """
    SPEED's activation time: the timestep during denoising where a frequency
    band becomes "active" (SNR crosses 1.0).
    
    Convention: t=0 is clean, t=T is noise.
    Activation happens at the first t (counting from T down to 0) where SNR > 1.
    We return T - t_activation to represent "steps from noise".
    """
    T = len(snr_values)
    for t in range(T - 1, -1, -1):
        if snr_values[t] >= threshold:
            return T - t  # steps from noise end
    return T


# ============================================================================
# Plot 1: SNR vs timestep for different stage_lengths configurations
# ============================================================================

def plot_snr_stage_lengths_comparison():
    """Generate subplot comparison of SNR schedules across configurations."""
    T = 1000
    betas = linear_beta_schedule(T)
    alpha_bar = compute_alpha_bar(betas)
    
    configs = {
        '[300, 300, 400] (default)': [300, 300, 400],
        '[200, 300, 500] (longer final)': [200, 300, 500],
        '[400, 300, 300] (longer first)': [400, 300, 300],
        '[333, 333, 334] (uniform)': [333, 333, 334],
    }
    
    frequencies = [1, 4, 8, 12, 16]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(frequencies)))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (config_name, stage_lengths) in enumerate(configs.items()):
        ax = axes[idx]
        boundaries = get_stage_boundaries(stage_lengths)
        
        for fi, freq in enumerate(frequencies):
            snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries)
            snr_rdd_db = 10 * np.log10(np.clip(snr_rdd, 1e-10, None))
            ax.plot(np.arange(T), snr_rdd_db, color=colors[fi], 
                    label=f'f={freq}', linewidth=1.5)
        
        # Mark stage boundaries
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Mark SNR = 0 dB (SNR = 1)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('SNR (dB)')
        ax.set_title(f'RDD SNR: stage_lengths = {config_name}')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, T)
        ax.set_ylim(-40, 40)
    
    plt.suptitle('RDD Effective SNR vs Timestep for Different Stage Configurations\n'
                 '(Red dashed = stage boundaries, Gray dotted = SNR=1)', 
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUTPUT_DIR / 'snr_stage_lengths_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot 1] Saved: {save_path}")


# ============================================================================
# Plot 2: Effective signal amplitude by stage
# ============================================================================

def plot_effective_signal_by_stage():
    """Show how effective signal amplitude varies across stages for frequency bands."""
    T = 1000
    betas = linear_beta_schedule(T)
    alpha_bar = compute_alpha_bar(betas)
    
    stage_lengths = [300, 300, 400]  # default config
    boundaries = get_stage_boundaries(stage_lengths)
    resolutions = [32, 16, 8]
    
    # Frequency bands
    freq_bands = {
        'DC (f=0)': 0.5,  # avoid division by zero
        'Low (f=2)': 2,
        'Mid-low (f=4)': 4,
        'Mid (f=8)': 8,
        'Mid-high (f=12)': 12,
        'High (f=16)': 16,
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Effective signal amplitude
    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freq_bands)))
    
    for i, (band_name, freq) in enumerate(freq_bands.items()):
        eff_signal = compute_effective_signal_rdd(
            np.arange(T), freq, alpha_bar, boundaries, resolutions
        )
        ax.plot(np.arange(T), eff_signal, color=colors[i], 
                label=band_name, linewidth=1.8)
    
    # Also plot standard signal (sqrt(alpha_bar)) for reference
    ax.plot(np.arange(T), np.sqrt(alpha_bar), 'k--', alpha=0.5, 
            label='Standard (all freq)', linewidth=1.2)
    
    # Mark stage boundaries with shading
    stage_colors = ['#FFE0B2', '#C8E6C9', '#BBDEFB']
    stage_labels = [f'Stage 0 (res={resolutions[0]})', 
                    f'Stage 1 (res={resolutions[1]})',
                    f'Stage 2 (res={resolutions[2]})']
    for c in range(len(boundaries) - 1):
        ax.axvspan(boundaries[c], boundaries[c+1], alpha=0.15, 
                   color=stage_colors[c], label=stage_labels[c])
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Effective Signal Amplitude')
    ax.set_title('RDD Effective Signal Amplitude by Frequency Band\n'
                 '(stage_lengths=[300,300,400], resolutions=[32,16,8])')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    
    # Bottom: Signal reduction ratio (RDD / standard)
    ax = axes[1]
    for i, (band_name, freq) in enumerate(freq_bands.items()):
        eff_signal = compute_effective_signal_rdd(
            np.arange(T), freq, alpha_bar, boundaries, resolutions
        )
        standard_signal = np.sqrt(alpha_bar)
        ratio = eff_signal / np.clip(standard_signal, 1e-10, None)
        ax.plot(np.arange(T), ratio, color=colors[i], 
                label=band_name, linewidth=1.8)
    
    for c in range(len(boundaries) - 1):
        ax.axvspan(boundaries[c], boundaries[c+1], alpha=0.15, color=stage_colors[c])
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Signal Retention Ratio (RDD / Standard)')
    ax.set_title('Signal Retention: How much RDD drift suppresses each frequency band')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'effective_signal_by_stage.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot 2] Saved: {save_path}")


# ============================================================================
# Plot 3: Optimal transition heatmap
# ============================================================================

def plot_optimal_transition_heatmap():
    """Heatmap of SNR at transition points for different configs and frequencies."""
    T = 1000
    betas = linear_beta_schedule(T)
    alpha_bar = compute_alpha_bar(betas)
    
    configs = {
        '[300,300,400]': [300, 300, 400],
        '[200,300,500]': [200, 300, 500],
        '[400,300,300]': [400, 300, 300],
        '[333,333,334]': [333, 333, 334],
        '[250,250,500]': [250, 250, 500],
        '[500,250,250]': [500, 250, 250],
    }
    
    frequencies = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    
    # For each config, compute SNR at each transition point for each frequency
    # We'll use the first transition point (boundary between stage 0 and 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for trans_idx, trans_name in enumerate(['Stage 0→1 boundary', 'Stage 1→2 boundary']):
        ax = axes[trans_idx]
        heatmap_data = np.zeros((len(configs), len(frequencies)))
        
        for ci, (config_name, stage_lengths) in enumerate(configs.items()):
            boundaries = get_stage_boundaries(stage_lengths)
            transition_t = boundaries[trans_idx + 1]  # first or second transition
            
            for fi, freq in enumerate(frequencies):
                snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries)
                snr_at_transition = snr_rdd[min(transition_t, T-1)]
                heatmap_data[ci, fi] = 10 * np.log10(max(snr_at_transition, 1e-10))
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu',
                       vmin=-30, vmax=30)
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f'f={f}' for f in frequencies], fontsize=8)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(list(configs.keys()), fontsize=9)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Stage Configuration')
        ax.set_title(f'SNR (dB) at {trans_name}')
        
        # Add text annotations
        for ci in range(len(configs)):
            for fi in range(len(frequencies)):
                val = heatmap_data[ci, fi]
                color = 'white' if abs(val) > 15 else 'black'
                ax.text(fi, ci, f'{val:.1f}', ha='center', va='center', 
                        fontsize=7, color=color)
        
        plt.colorbar(im, ax=ax, label='SNR (dB)')
    
    plt.suptitle('SNR at Stage Transition Points\n'
                 '(Positive = signal dominant, Negative = noise dominant)', 
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = OUTPUT_DIR / 'optimal_transition_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot 3] Saved: {save_path}")


# ============================================================================
# Plot 4: RDD vs Standard DDPM SNR gap
# ============================================================================

def plot_rdd_vs_standard_snr_gap():
    """Show the gap between RDD's effective SNR and standard DDPM's SNR."""
    T = 1000
    betas = linear_beta_schedule(T)
    alpha_bar = compute_alpha_bar(betas)
    
    stage_lengths = [300, 300, 400]
    boundaries = get_stage_boundaries(stage_lengths)
    resolutions = [32, 16, 8]
    
    frequencies = [1, 4, 8, 12, 16]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(frequencies)))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: SNR comparison (both RDD and standard on same plot)
    ax = axes[0]
    for fi, freq in enumerate(frequencies):
        # Standard DDPM SNR (using 1/f^2 power spectrum)
        snr_std = compute_snr_standard_freq(freq, alpha_bar)
        snr_std_db = 10 * np.log10(np.clip(snr_std, 1e-10, None))
        
        # RDD SNR
        snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
        snr_rdd_db = 10 * np.log10(np.clip(snr_rdd, 1e-10, None))
        
        ax.plot(np.arange(T), snr_std_db, color=colors[fi], linestyle='--', 
                alpha=0.6, linewidth=1.2)
        ax.plot(np.arange(T), snr_rdd_db, color=colors[fi], linestyle='-', 
                linewidth=2.0, label=f'f={freq}')
    
    for b in boundaries[1:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Comparison: RDD (solid) vs Standard DDPM (dashed)\n'
                 'stage_lengths=[300,300,400], P(f)=1/f²')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, T)
    ax.set_ylim(-50, 50)
    
    # Bottom: SNR gap (RDD - Standard) in dB
    ax = axes[1]
    for fi, freq in enumerate(frequencies):
        snr_std = compute_snr_standard_freq(freq, alpha_bar)
        snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
        
        # Gap in dB
        snr_gap_db = 10 * np.log10(np.clip(snr_rdd, 1e-10, None)) - \
                     10 * np.log10(np.clip(snr_std, 1e-10, None))
        
        ax.plot(np.arange(T), snr_gap_db, color=colors[fi], 
                linewidth=2.0, label=f'f={freq}')
    
    for b in boundaries[1:-1]:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Shade negative regions
    ax.fill_between(np.arange(T), -60, 0, alpha=0.05, color='red')
    ax.text(T*0.5, -15, 'RDD suppresses signal\n(delays high-freq activation)', 
            ha='center', fontsize=10, color='darkred', alpha=0.7)
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('SNR Gap (dB): RDD − Standard')
    ax.set_title('High-Frequency Delay: How RDD suppresses high-freq signals relative to DDPM')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, T)
    ax.set_ylim(-60, 5)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'rdd_vs_standard_snr_gap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot 4] Saved: {save_path}")


# ============================================================================
# Theoretical Insights Computation
# ============================================================================

def compute_theoretical_insights():
    """Compute and print key theoretical insights."""
    T = 1000
    betas = linear_beta_schedule(T)
    alpha_bar = compute_alpha_bar(betas)
    
    configs = {
        '[300, 300, 400] (default)': [300, 300, 400],
        '[200, 300, 500] (longer final)': [200, 300, 500],
        '[400, 300, 300] (longer first)': [400, 300, 300],
        '[333, 333, 334] (uniform)': [333, 333, 334],
    }
    
    resolutions = [32, 16, 8]
    frequencies = [1, 4, 8, 12, 16]
    
    print("\n" + "=" * 80)
    print("THEORETICAL INSIGHTS: RDD Frequency-Domain SNR Analysis")
    print("=" * 80)
    
    # ---- Insight 1: SNR=1 crossing points ----
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 1. SNR = 1.0 Crossing Points (timestep where signal = noise)       │")
    print("│    Convention: t=0 is clean image, t=999 is pure noise             │")
    print("│    Lower t means frequency 'activates' earlier during denoising    │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Standard DDPM crossings
    print("\n  Standard DDPM (no drift, P(f)=1/f²):")
    std_crossings = {}
    for freq in frequencies:
        snr_std = compute_snr_standard_freq(freq, alpha_bar)
        crossing = find_snr_crossing(snr_std, threshold=1.0)
        std_crossings[freq] = crossing
        print(f"    f={freq:2d}: SNR=1 at t={crossing:4d}")
    
    # RDD crossings for each config
    print("\n  RDD Effective SNR crossings:")
    rdd_crossings = {}
    for config_name, stage_lengths in configs.items():
        boundaries = get_stage_boundaries(stage_lengths)
        print(f"\n    Config: {config_name}")
        print(f"    Stage boundaries: {boundaries}")
        rdd_crossings[config_name] = {}
        
        for freq in frequencies:
            snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
            crossing = find_snr_crossing(snr_rdd, threshold=1.0)
            rdd_crossings[config_name][freq] = crossing
            print(f"      f={freq:2d}: SNR=1 at t={crossing:4d}")
    
    # ---- Insight 2: SPEED activation time comparison ----
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 2. SPEED Activation Time Comparison                                │")
    print("│    (Steps from noise end until frequency becomes active)            │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print(f"\n  {'Config':<30s} | ", end='')
    for freq in frequencies:
        print(f"f={freq:2d} ", end='')
    print()
    print("  " + "-" * 75)
    
    # Standard
    print(f"  {'Standard DDPM':<30s} | ", end='')
    for freq in frequencies:
        snr_std = compute_snr_standard_freq(freq, alpha_bar)
        act_time = compute_activation_time(snr_std)
        print(f"{act_time:4d} ", end='')
    print()
    
    # RDD configs
    for config_name, stage_lengths in configs.items():
        boundaries = get_stage_boundaries(stage_lengths)
        short_name = config_name[:30]
        print(f"  {short_name:<30s} | ", end='')
        for freq in frequencies:
            snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
            act_time = compute_activation_time(snr_rdd)
            print(f"{act_time:4d} ", end='')
        print()
    
    # ---- Insight 3: High-frequency delay metric ----
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 3. High-Frequency Delay Metric                                     │")
    print("│    Extra steps for f=16 to activate in RDD vs standard DDPM        │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    snr_std_f16 = compute_snr_standard_freq(16, alpha_bar)
    std_activation_f16 = compute_activation_time(snr_std_f16)
    
    print(f"\n  Standard DDPM f=16 activation: {std_activation_f16} steps from noise")
    print()
    
    for config_name, stage_lengths in configs.items():
        boundaries = get_stage_boundaries(stage_lengths)
        snr_rdd_f16 = compute_snr_rdd(16, alpha_bar, boundaries, resolutions)
        rdd_activation_f16 = compute_activation_time(snr_rdd_f16)
        delay = rdd_activation_f16 - std_activation_f16
        
        print(f"  {config_name}:")
        print(f"    RDD f=16 activation: {rdd_activation_f16} steps from noise")
        print(f"    Delay vs standard:   +{delay} steps")
        print(f"    Relative delay:      {delay/std_activation_f16*100:.1f}% longer")
        print()
    
    # ---- Insight 4: Frequency ordering consistency ----
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 4. Frequency Ordering & Coarse-to-Fine Consistency                 │")
    print("│    Does RDD maintain strict low→high freq ordering?                │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    for config_name, stage_lengths in configs.items():
        boundaries = get_stage_boundaries(stage_lengths)
        print(f"\n  Config: {config_name}")
        
        activation_times = []
        for freq in frequencies:
            snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
            act_time = compute_activation_time(snr_rdd)
            activation_times.append(act_time)
        
        # Check if activation times are monotonically increasing
        is_ordered = all(activation_times[i] <= activation_times[i+1] 
                        for i in range(len(activation_times)-1))
        print(f"    Activation times: {activation_times}")
        print(f"    Strictly ordered (low→high): {'✓ YES' if is_ordered else '✗ NO'}")
    
    # ---- Insight 5: Stage utilization efficiency ----
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 5. Stage Utilization Efficiency                                    │")
    print("│    How much SNR range is covered within each stage?                │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    config_name = '[300, 300, 400] (default)'
    stage_lengths = configs[config_name]
    boundaries = get_stage_boundaries(stage_lengths)
    
    print(f"\n  Config: {config_name}")
    print(f"  Resolution per stage: {resolutions}")
    print()
    
    for freq in frequencies:
        snr_rdd = compute_snr_rdd(freq, alpha_bar, boundaries, resolutions)
        snr_rdd_db = 10 * np.log10(np.clip(snr_rdd, 1e-10, None))
        
        print(f"  f={freq:2d}:")
        for c in range(len(boundaries) - 1):
            T_c = boundaries[c]
            T_c1 = boundaries[c + 1] - 1
            snr_start = snr_rdd_db[T_c]
            snr_end = snr_rdd_db[T_c1]
            print(f"    Stage {c} (t={T_c:3d}→{T_c1:3d}): "
                  f"SNR {snr_start:6.1f}dB → {snr_end:6.1f}dB "
                  f"(Δ={snr_end-snr_start:6.1f}dB)")
        print()
    
    print("=" * 80)
    print("END OF THEORETICAL ANALYSIS")
    print("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Experiment 2: RDD Frequency-Domain SNR Theoretical Analysis")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Parameters: T=1000, beta_min=0.0001, beta_max=0.02 (linear schedule)")
    print(f"Resolutions: [32, 16, 8], Nyquist frequencies: [16, 8, 4]")
    print()
    
    # Generate all plots
    print("Generating plots...")
    plot_snr_stage_lengths_comparison()
    plot_effective_signal_by_stage()
    plot_optimal_transition_heatmap()
    plot_rdd_vs_standard_snr_gap()
    
    # Compute and print theoretical insights
    compute_theoretical_insights()
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
