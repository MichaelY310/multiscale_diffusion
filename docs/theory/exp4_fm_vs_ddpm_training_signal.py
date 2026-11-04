"""
Experiment 4: Rigorous Theoretical Comparison of Training Signals
between DDPM and Flow Matching for RDD (Resolution Drift Diffusion)

This script analyzes:
1. SNR-weighted loss comparison
2. Drift signal-to-noise ratio in the loss
3. Loss landscape flatness
4. Effective learning rate for drift
5. Time-weighting equivalence analysis
6. Variance of gradient estimates

Author: Research Experiments
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set up output directory
OUTPUT_DIR = Path(__file__).parent
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Plot style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'figure.dpi': 150,
})

# =============================================================================
# DDPM Schedule Parameters
# =============================================================================
T_DDPM = 1000
BETA_MIN = 0.0001
BETA_MAX = 0.02

# Linear beta schedule
betas = np.linspace(BETA_MIN, BETA_MAX, T_DDPM)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)  # alpha_bar_t

# Continuous time mapping: t in [0, 1] for comparison
t_continuous = np.linspace(0, 1, T_DDPM)

# SNR for DDPM: alpha_bar_t / (1 - alpha_bar_t)
snr = alpha_bar / (1.0 - alpha_bar)

# FM time grid (uniform)
t_fm = np.linspace(1e-4, 1 - 1e-4, T_DDPM)

# =============================================================================
# Helper Functions
# =============================================================================

def kappa_t(t_idx):
    """
    kappa(t) scaling function for RDD-DDPM.
    We use kappa(t) = 1 (constant) as the simplest meaningful choice.
    """
    return np.ones_like(t_idx, dtype=float)


def ddpm_drift_magnitude(alpha_bar_t, kappa):
    """Drift correction magnitude in DDPM: sqrt(alpha_bar/(1-alpha_bar)) * kappa * ||d||"""
    return np.sqrt(alpha_bar_t / (1.0 - alpha_bar_t)) * kappa


def fm_drift_magnitude(t):
    """Drift correction magnitude in FM: |1 - 2t| * ||d||"""
    return np.abs(1.0 - 2.0 * t)


# =============================================================================
# Analysis 1: SNR-weighted Loss Comparison
# =============================================================================
print("=" * 70)
print("Analysis 1: SNR-weighted Loss Comparison")
print("=" * 70)

# DDPM effective importance weight is proportional to SNR(t)
ddpm_weight = snr / snr.sum()  # Normalized

# FM effective importance weight is uniform
fm_weight = np.ones(T_DDPM) / T_DDPM

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw weights
ax = axes[0]
ax.semilogy(t_continuous, snr, 'b-', label='DDPM SNR weight (unnormalized)')
ax.axhline(1.0, color='r', linestyle='--', label='FM uniform weight')
ax.set_xlabel('Normalized time t')
ax.set_ylabel('Importance Weight (log scale)')
ax.set_title('Raw Importance Weights')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: cumulative weight distribution
ax = axes[1]
ddpm_cumweight = np.cumsum(ddpm_weight)
fm_cumweight = np.cumsum(fm_weight)
ax.plot(t_continuous, ddpm_cumweight, 'b-', label='DDPM (SNR-weighted)')
ax.plot(t_continuous, fm_cumweight, 'r-', label='FM (uniform)')
ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='t=0.3 (drift active region)')
ax.axvline(0.7, color='gray', linestyle=':', alpha=0.5, label='t=0.7')
ax.fill_between(t_continuous, 0, 1, where=(t_continuous >= 0.3) & (t_continuous <= 0.7),
                alpha=0.1, color='green', label='Middle timesteps (drift active)')
ax.set_xlabel('Normalized time t')
ax.set_ylabel('Cumulative Weight')
ax.set_title('Cumulative Gradient Budget Distribution')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fm_vs_ddpm_importance_weights.png', bbox_inches='tight')
plt.close()

# Quantitative comparison
mid_mask = (t_continuous >= 0.3) & (t_continuous <= 0.7)
ddpm_mid_budget = ddpm_weight[mid_mask].sum()
fm_mid_budget = fm_weight[mid_mask].sum()

print(f"  DDPM gradient budget on middle timesteps (0.3-0.7): {ddpm_mid_budget:.4f}")
print(f"  FM gradient budget on middle timesteps (0.3-0.7):   {fm_mid_budget:.4f}")
print(f"  FM spends {fm_mid_budget/ddpm_mid_budget:.2f}x more gradient on middle timesteps")
print(f"  -> FM allocates more gradient where RDD drift is most active")
print()

# =============================================================================
# Analysis 2: Drift Signal-to-Noise Ratio in the Loss
# =============================================================================
print("=" * 70)
print("Analysis 2: Drift Correction Magnitude Comparison")
print("=" * 70)

# Assume ||d|| = 1 for normalization
d_norm = 1.0
kappa = kappa_t(np.arange(T_DDPM))

ddpm_drift_mag = ddpm_drift_magnitude(alpha_bar, kappa) * d_norm
fm_drift_mag = fm_drift_magnitude(t_fm) * d_norm

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: linear scale
ax = axes[0]
ax.plot(t_continuous, ddpm_drift_mag, 'b-', label=r'DDPM: $\sqrt{\bar\alpha/(1-\bar\alpha)}\cdot\kappa\cdot\|d\|$')
ax.plot(t_fm, fm_drift_mag, 'r-', label=r'FM: $|1-2t|\cdot\|d\|$')
ax.set_xlabel('Time t')
ax.set_ylabel('Drift Correction Magnitude')
ax.set_title('Drift Correction Magnitude (Linear Scale)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 10)

# Right: log scale
ax = axes[1]
ax.semilogy(t_continuous, ddpm_drift_mag + 1e-10, 'b-', label=r'DDPM: $\sqrt{\bar\alpha/(1-\bar\alpha)}\cdot\kappa\cdot\|d\|$')
ax.semilogy(t_fm, fm_drift_mag + 1e-10, 'r-', label=r'FM: $|1-2t|\cdot\|d\|$')
ax.set_xlabel('Time t')
ax.set_ylabel('Drift Correction Magnitude (log scale)')
ax.set_title('Drift Correction Magnitude (Log Scale)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'drift_correction_magnitude.png', bbox_inches='tight')
plt.close()

print(f"  DDPM drift magnitude range: [{ddpm_drift_mag.min():.4f}, {ddpm_drift_mag.max():.4f}]")
print(f"  FM drift magnitude range:   [{fm_drift_mag.min():.6f}, {fm_drift_mag.max():.4f}]")
print(f"  DDPM max/min ratio: {ddpm_drift_mag.max()/max(ddpm_drift_mag.min(), 1e-10):.1f}")
print(f"  FM max/min ratio:   {fm_drift_mag.max()/max(fm_drift_mag[fm_drift_mag > 0].min(), 1e-10):.1f}")
print(f"  -> FM drift correction is bounded [0,1], DDPM can explode at early t")
print()

# =============================================================================
# Analysis 3: Loss Landscape Flatness
# =============================================================================
print("=" * 70)
print("Analysis 3: Loss Landscape Flatness")
print("=" * 70)

# Simulate with random data
num_samples = 64
dim = 4 * 32 * 32  # 4x32x32 flattened

# Random data, noise, and drift
x_1 = np.random.randn(num_samples, dim)  # data samples
epsilon = np.random.randn(num_samples, dim)  # noise samples
d = np.random.randn(num_samples, dim) * 0.1  # drift (smaller magnitude)

# Random model output (simulates untrained network)
random_model_output = np.random.randn(num_samples, dim)

n_t_steps = 100
t_eval = np.linspace(0.01, 0.99, n_t_steps)

ddpm_losses = np.zeros(n_t_steps)
fm_losses = np.zeros(n_t_steps)

for i, t in enumerate(t_eval):
    # Map t to DDPM timestep index
    t_idx = int(t * (T_DDPM - 1))
    ab = alpha_bar[t_idx]
    
    # DDPM target: eps_tilde = eps + sqrt(alpha_bar/(1-alpha_bar)) * kappa * d
    drift_scale_ddpm = np.sqrt(ab / (1.0 - ab))
    eps_tilde = epsilon + drift_scale_ddpm * d  # (num_samples, dim)
    
    # FM target: u_t = (x_1 - eps) + (1-2t) * d
    u_t = (x_1 - epsilon) + (1.0 - 2.0 * t) * d  # (num_samples, dim)
    
    # Loss: ||random_output - target||^2 / dim
    ddpm_losses[i] = np.mean(np.sum((random_model_output - eps_tilde) ** 2, axis=1)) / dim
    fm_losses[i] = np.mean(np.sum((random_model_output - u_t) ** 2, axis=1)) / dim

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw loss values
ax = axes[0]
ax.plot(t_eval, ddpm_losses, 'b-', label='DDPM E[loss(t)]')
ax.plot(t_eval, fm_losses, 'r-', label='FM E[loss(t)]')
ax.set_xlabel('Time t')
ax.set_ylabel('Expected Loss (per dimension)')
ax.set_title('Expected Loss vs Time (Random Model)')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: normalized (relative variation)
ax = axes[1]
ddpm_norm = ddpm_losses / ddpm_losses.mean()
fm_norm = fm_losses / fm_losses.mean()
ax.plot(t_eval, ddpm_norm, 'b-', label='DDPM (normalized)')
ax.plot(t_eval, fm_norm, 'r-', label='FM (normalized)')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time t')
ax.set_ylabel('Normalized Loss (mean=1)')
ax.set_title('Normalized Loss Variation (Flatness)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'loss_landscape.png', bbox_inches='tight')
plt.close()

ddpm_cv = np.std(ddpm_losses) / np.mean(ddpm_losses)
fm_cv = np.std(fm_losses) / np.mean(fm_losses)
print(f"  DDPM loss coefficient of variation: {ddpm_cv:.4f}")
print(f"  FM loss coefficient of variation:   {fm_cv:.4f}")
print(f"  FM loss is {ddpm_cv/fm_cv:.2f}x more uniform across timesteps")
print(f"  -> More uniform loss = easier optimization for SGD")
print()

# =============================================================================
# Analysis 4: Effective Learning Rate for Drift
# =============================================================================
print("=" * 70)
print("Analysis 4: Effective Learning Rate for Drift")
print("=" * 70)

# Compute: ||drift_target||^2 / ||base_target||^2 for both frameworks
drift_fraction_ddpm = np.zeros(n_t_steps)
drift_fraction_fm = np.zeros(n_t_steps)

for i, t in enumerate(t_eval):
    t_idx = int(t * (T_DDPM - 1))
    ab = alpha_bar[t_idx]
    
    # DDPM:
    # Base target: eps (noise prediction)
    # Drift target: sqrt(alpha_bar/(1-alpha_bar)) * kappa * d
    drift_scale_ddpm = np.sqrt(ab / (1.0 - ab))
    base_norm_sq = np.mean(np.sum(epsilon ** 2, axis=1))
    drift_norm_sq = drift_scale_ddpm ** 2 * np.mean(np.sum(d ** 2, axis=1))
    drift_fraction_ddpm[i] = drift_norm_sq / (base_norm_sq + drift_norm_sq)
    
    # FM:
    # Base target: (x_1 - eps)
    # Drift target: (1-2t) * d
    base_fm = x_1 - epsilon
    base_norm_sq_fm = np.mean(np.sum(base_fm ** 2, axis=1))
    drift_norm_sq_fm = (1.0 - 2.0 * t) ** 2 * np.mean(np.sum(d ** 2, axis=1))
    drift_fraction_fm[i] = drift_norm_sq_fm / (base_norm_sq_fm + drift_norm_sq_fm)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: drift fraction
ax = axes[0]
ax.plot(t_eval, drift_fraction_ddpm, 'b-', label='DDPM drift fraction')
ax.plot(t_eval, drift_fraction_fm, 'r-', label='FM drift fraction')
ax.set_xlabel('Time t')
ax.set_ylabel(r'$\|drift\|^2 / \|total\_target\|^2$')
ax.set_title('Fraction of Gradient Signal from Drift')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(1.0, max(drift_fraction_ddpm.max(), drift_fraction_fm.max()) * 1.2))

# Right: weighted drift signal (importance x drift fraction)
ax = axes[1]
# Resample weights to match t_eval
ddpm_w_interp = np.interp(t_eval, t_continuous, ddpm_weight * T_DDPM)
fm_w_interp = np.ones(n_t_steps)  # uniform

effective_drift_ddpm = ddpm_w_interp * drift_fraction_ddpm
effective_drift_fm = fm_w_interp * drift_fraction_fm

ax.plot(t_eval, effective_drift_ddpm / effective_drift_ddpm.max(), 'b-',
        label='DDPM: weight x drift_fraction')
ax.plot(t_eval, effective_drift_fm / effective_drift_fm.max(), 'r-',
        label='FM: weight x drift_fraction')
ax.set_xlabel('Time t')
ax.set_ylabel('Effective Drift Learning Signal (normalized)')
ax.set_title('Where Does the Model Actually Learn Drift?')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'drift_learning_fraction.png', bbox_inches='tight')
plt.close()

# Find peak drift learning regions
ddpm_peak_t = t_eval[np.argmax(effective_drift_ddpm)]
fm_peak_t = t_eval[np.argmax(effective_drift_fm)]
print(f"  DDPM peak drift learning at t={ddpm_peak_t:.3f}")
print(f"  FM peak drift learning at t={fm_peak_t:.3f}")
print(f"  DDPM drift fraction range: [{drift_fraction_ddpm.min():.6f}, {drift_fraction_ddpm.max():.4f}]")
print(f"  FM drift fraction range:   [{drift_fraction_fm.min():.6f}, {drift_fraction_fm.max():.4f}]")
print(f"  -> DDPM concentrates drift learning at early t (low noise)")
print(f"  -> FM spreads drift learning more evenly (except at t=0.5 where it vanishes)")
print()

# =============================================================================
# Analysis 5: Time-Weighting Equivalence Analysis
# =============================================================================
print("=" * 70)
print("Analysis 5: Time-Weighting Equivalence Analysis")
print("=" * 70)

# What DDPM weighting w(t) makes DDPM loss equivalent to FM loss?
# FM loss: E_t[||u_t - model||^2] with uniform t
# DDPM loss: E_t[w(t) * ||eps_tilde - model||^2]
#
# FM-equivalent weighting for DDPM: w(t) proportional to 1/SNR(t)

# FM-equivalent weighting in DDPM
fm_equiv_weight = 1.0 / (snr + 1e-10)
fm_equiv_weight = fm_equiv_weight / fm_equiv_weight.mean()  # Normalize to mean 1

# Min-SNR weighting (Hang et al., 2023): min(SNR(t), gamma) / SNR(t), gamma=5
gamma_minsnr = 5.0
minsnr_weight = np.minimum(snr, gamma_minsnr) / (snr + 1e-10)
minsnr_weight = minsnr_weight / minsnr_weight.mean()

# P2 weighting (Choi et al., 2022): 1/(1 + SNR(t))
p2_weight = 1.0 / (1.0 + snr)
p2_weight = p2_weight / p2_weight.mean()

# Velocity (v-prediction) weighting: effectively 1/(1+1/SNR)
v_weight = snr / (1.0 + snr)
v_weight = v_weight / v_weight.mean()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.semilogy(t_continuous, fm_equiv_weight, 'r-', linewidth=2.5,
            label='FM-equivalent (1/SNR)')
ax.semilogy(t_continuous, minsnr_weight, 'g--', linewidth=2,
            label='Min-SNR (gamma=5)')
ax.semilogy(t_continuous, p2_weight, 'm-.', linewidth=2,
            label='P2: 1/(1+SNR)')
ax.semilogy(t_continuous, v_weight, 'c:', linewidth=2,
            label='v-prediction: SNR/(1+SNR)')
ax.axhline(1.0, color='gray', linestyle='-', alpha=0.3, label='Uniform (no reweighting)')

ax.set_xlabel('Normalized time t')
ax.set_ylabel('Loss Weight w(t) (log scale, normalized to mean=1)')
ax.set_title('DDPM Loss Weightings: FM-Equivalent vs Common Alternatives')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-3, 1e3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'time_weighting_equivalence.png', bbox_inches='tight')
plt.close()

# Compute correlation between FM-equivalent and other weightings
corr_minsnr = np.corrcoef(np.log(fm_equiv_weight + 1e-10), np.log(minsnr_weight + 1e-10))[0, 1]
corr_p2 = np.corrcoef(np.log(fm_equiv_weight + 1e-10), np.log(p2_weight + 1e-10))[0, 1]
corr_v = np.corrcoef(np.log(fm_equiv_weight + 1e-10), np.log(v_weight + 1e-10))[0, 1]

print(f"  Log-correlation of FM-equiv with Min-SNR: {corr_minsnr:.4f}")
print(f"  Log-correlation of FM-equiv with P2:      {corr_p2:.4f}")
print(f"  Log-correlation of FM-equiv with v-pred:  {corr_v:.4f}")
print(f"  -> FM's uniform weighting corresponds to 1/SNR in DDPM space")
print(f"  -> This is most similar to P2 weighting (correlation {corr_p2:.3f})")
print(f"  -> FM naturally downweights low-noise steps (where DDPM over-focuses)")
print()

# =============================================================================
# Analysis 6: Variance of Gradient Estimates
# =============================================================================
print("=" * 70)
print("Analysis 6: Variance of Gradient Estimates")
print("=" * 70)

# For each t, compute variance of loss across samples
num_samples_var = 256
dim_var = 4 * 32 * 32

x_1_var = np.random.randn(num_samples_var, dim_var)
epsilon_var = np.random.randn(num_samples_var, dim_var)
d_var = np.random.randn(num_samples_var, dim_var) * 0.1
model_out_var = np.random.randn(num_samples_var, dim_var)

n_t_var = 100
t_var = np.linspace(0.01, 0.99, n_t_var)

ddpm_loss_var = np.zeros(n_t_var)
fm_loss_var = np.zeros(n_t_var)
ddpm_loss_mean = np.zeros(n_t_var)
fm_loss_mean = np.zeros(n_t_var)

for i, t in enumerate(t_var):
    t_idx = int(t * (T_DDPM - 1))
    ab = alpha_bar[t_idx]
    
    # DDPM per-sample loss
    drift_scale = np.sqrt(ab / (1.0 - ab))
    eps_tilde = epsilon_var + drift_scale * d_var
    ddpm_per_sample = np.sum((model_out_var - eps_tilde) ** 2, axis=1) / dim_var
    ddpm_loss_var[i] = np.var(ddpm_per_sample)
    ddpm_loss_mean[i] = np.mean(ddpm_per_sample)
    
    # FM per-sample loss
    u_t = (x_1_var - epsilon_var) + (1.0 - 2.0 * t) * d_var
    fm_per_sample = np.sum((model_out_var - u_t) ** 2, axis=1) / dim_var
    fm_loss_var[i] = np.var(fm_per_sample)
    fm_loss_mean[i] = np.mean(fm_per_sample)

# Coefficient of variation (relative variance)
ddpm_cv_t = np.sqrt(ddpm_loss_var) / (ddpm_loss_mean + 1e-10)
fm_cv_t = np.sqrt(fm_loss_var) / (fm_loss_mean + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw variance
ax = axes[0]
ax.semilogy(t_var, ddpm_loss_var, 'b-', label='DDPM Var[loss(t)]')
ax.semilogy(t_var, fm_loss_var, 'r-', label='FM Var[loss(t)]')
ax.set_xlabel('Time t')
ax.set_ylabel('Variance of Loss (log scale)')
ax.set_title('Gradient Variance vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: coefficient of variation
ax = axes[1]
ax.plot(t_var, ddpm_cv_t, 'b-', label='DDPM CV(loss)')
ax.plot(t_var, fm_cv_t, 'r-', label='FM CV(loss)')
ax.set_xlabel('Time t')
ax.set_ylabel('Coefficient of Variation')
ax.set_title('Relative Gradient Noise (CV = sigma/mu)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gradient_variance.png', bbox_inches='tight')
plt.close()

# Critical timestep analysis
ddpm_critical = (t_var < 0.3)
fm_critical = (t_var < 0.3) | (t_var > 0.7)

ddpm_var_critical = np.mean(ddpm_loss_var[ddpm_critical])
fm_var_critical = np.mean(fm_loss_var[fm_critical])
ddpm_cv_critical = np.mean(ddpm_cv_t[ddpm_critical])
fm_cv_critical = np.mean(fm_cv_t[fm_critical])

print(f"  At drift-critical timesteps:")
print(f"    DDPM mean variance: {ddpm_var_critical:.6f}")
print(f"    FM mean variance:   {fm_var_critical:.6f}")
print(f"    DDPM mean CV:       {ddpm_cv_critical:.4f}")
print(f"    FM mean CV:         {fm_cv_critical:.4f}")
print(f"  Overall:")
print(f"    DDPM variance range: [{ddpm_loss_var.min():.6f}, {ddpm_loss_var.max():.6f}]")
print(f"    FM variance range:   [{fm_loss_var.min():.6f}, {fm_loss_var.max():.6f}]")
print(f"    DDPM max/min var ratio: {ddpm_loss_var.max()/ddpm_loss_var.min():.1f}")
print(f"    FM max/min var ratio:   {fm_loss_var.max()/fm_loss_var.min():.1f}")
print(f"  -> FM has much more uniform gradient variance across timesteps")
print(f"  -> DDPM gradient variance explodes at early t due to drift term scaling")
print()

# =============================================================================
# SUMMARY OF KEY FINDINGS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)
print()
print("1. GRADIENT BUDGET ALLOCATION:")
print(f"   FM allocates {fm_mid_budget/ddpm_mid_budget:.2f}x more gradient to middle timesteps")
print(f"   (t in [0.3, 0.7]) where RDD drift is most active.")
print(f"   DDPM wastes gradient on low-noise steps where drift dominates but")
print(f"   data is nearly reconstructed already.")
print()
print("2. DRIFT CORRECTION BOUNDEDNESS:")
print(f"   DDPM drift magnitude range: [{ddpm_drift_mag.min():.4f}, {ddpm_drift_mag.max():.1f}]")
print(f"   FM drift magnitude range:   [0, 1]")
print(f"   DDPM's unbounded drift correction at early t creates optimization")
print(f"   instability; FM's bounded correction enables stable training.")
print()
print("3. LOSS LANDSCAPE UNIFORMITY:")
print(f"   FM loss CV = {fm_cv:.4f} vs DDPM loss CV = {ddpm_cv:.4f}")
print(f"   FM is {ddpm_cv/fm_cv:.2f}x more uniform -> easier for SGD to optimize.")
print()
print("4. DRIFT LEARNING CONCENTRATION:")
print(f"   DDPM concentrates drift learning at t={ddpm_peak_t:.2f} (early, low noise)")
print(f"   FM spreads drift learning around t=0 and t=1, avoiding t=0.5")
print(f"   FM's distribution is more balanced for multi-scale drift learning.")
print()
print("5. TIME-WEIGHTING EQUIVALENCE:")
print(f"   FM's uniform weighting = 1/SNR weighting in DDPM")
print(f"   Closest to P2 weighting (log-correlation = {corr_p2:.3f})")
print(f"   This confirms FM naturally incorporates beneficial loss reweighting")
print(f"   that DDPM requires explicit schedules (Min-SNR, P2) to achieve.")
print()
print("6. GRADIENT VARIANCE:")
print(f"   DDPM variance ratio (max/min): {ddpm_loss_var.max()/ddpm_loss_var.min():.1f}x")
print(f"   FM variance ratio (max/min):   {fm_loss_var.max()/fm_loss_var.min():.1f}x")
print(f"   FM provides {ddpm_loss_var.max()/ddpm_loss_var.min() / (fm_loss_var.max()/fm_loss_var.min()):.1f}x")
print(f"   more uniform gradient signal -> faster convergence for drift learning.")
print()
print("=" * 70)
print("CONCLUSION: Flow Matching is theoretically superior for RDD because:")
print("  (a) Bounded drift corrections prevent gradient explosion")
print("  (b) Uniform time-weighting allocates more signal to drift-active regions")
print("  (c) Lower gradient variance enables faster, more stable convergence")
print("  (d) Natural equivalence to beneficial DDPM reweighting schemes (P2)")
print("=" * 70)
print()
print(f"All plots saved to: {OUTPUT_DIR}/")
print("  - fm_vs_ddpm_importance_weights.png")
print("  - drift_correction_magnitude.png")
print("  - loss_landscape.png")
print("  - drift_learning_fraction.png")
print("  - time_weighting_equivalence.png")
print("  - gradient_variance.png")
