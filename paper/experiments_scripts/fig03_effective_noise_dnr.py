"""
fig03_effective_noise_dnr.py
============================
绘制 Drift-to-Noise Ratio (DNR) vs timestep 的曲线。

使用 noise schedule 参数直接计算（不需要实验数据）：
  - Linear schedule: beta_1=1e-4, beta_T=0.02, T=1000
  - Stage 0 (t∈[0,300]):   DNR(t) = sqrt(alpha_bar_t/(1-alpha_bar_t)) * (t-0)/300
  - Stage 1 (t∈[300,600]): DNR(t) = sqrt(alpha_bar_t/(1-alpha_bar_t)) * (t-300)/300
  - Stage 2 (t∈[600,1000]):DNR(t) = sqrt(alpha_bar_t/(1-alpha_bar_t)) * (t-600)/400
  - DNR 还需乘以 ||d^c||/sqrt(n)，这里用归一化值=1

输出：
  - fig_effective_noise_dnr.pdf

用法：
  source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
  python fig03_effective_noise_dnr.py [--output-dir ./] [--drift-norm 1.0]
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_noise_schedule(T=1000, beta_1=1e-4, beta_T=0.02):
    """Compute linear noise schedule and derived quantities."""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def compute_dnr(alpha_bar, stage_bounds, drift_norm=1.0):
    """
    Compute DNR for each timestep.
    
    DNR(t) = (||d^c|| / sqrt(n)) * sqrt(alpha_bar_t / (1 - alpha_bar_t)) * progress(t)
    
    where progress(t) is the fractional position within the current stage.
    drift_norm represents ||d^c||/sqrt(n), normalized to 1 by default.
    """
    T = len(alpha_bar)
    dnr = np.zeros(T)
    
    num_stages = len(stage_bounds) - 1
    for stage_idx in range(num_stages):
        t_start = stage_bounds[stage_idx]
        t_end = stage_bounds[stage_idx + 1]
        stage_length = t_end - t_start
        
        for t in range(t_start, min(t_end, T)):
            snr_t = alpha_bar[t] / (1 - alpha_bar[t])
            progress = (t - t_start) / stage_length
            dnr[t] = drift_norm * np.sqrt(snr_t) * progress
    
    return dnr


def main():
    parser = argparse.ArgumentParser(description="DNR vs timestep figure")
    parser.add_argument("--output-dir", type=str, default="./")
    parser.add_argument("--drift-norm", type=float, default=1.0,
                        help="||d^c||/sqrt(n) normalization factor (default: 1.0)")
    parser.add_argument("--T", type=int, default=1000, help="Total diffusion steps")
    parser.add_argument("--beta-1", type=float, default=1e-4, help="beta_1")
    parser.add_argument("--beta-T", type=float, default=0.02, help="beta_T")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute schedule
    # ─────────────────────────────────────────────────────────────────────────
    T = args.T
    betas, alphas, alpha_bar = compute_noise_schedule(T, args.beta_1, args.beta_T)

    # Stage boundaries: fine [0,300), mid [300,600), coarse [600,1000)
    stage_bounds = [0, 300, 600, 1000]
    stage_names = ["Fine (32×32)", "Mid (16×16)", "Coarse (8×8)"]
    stage_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # Compute DNR
    dnr = compute_dnr(alpha_bar, stage_bounds, drift_norm=args.drift_norm)

    # Also compute SNR for reference
    snr = alpha_bar / (1 - alpha_bar)
    log_snr = np.log10(snr + 1e-10)

    t_axis = np.arange(T)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    # ─── Top panel: log₁₀(SNR) ───
    ax1.plot(t_axis, log_snr, color="navy", linewidth=2)
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("log₁₀(SNR)", fontsize=12)
    ax1.set_title("Noise Schedule & Drift-to-Noise Ratio (DNR)", 
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    # Color stages in background
    for i in range(3):
        ax1.axvspan(stage_bounds[i], stage_bounds[i + 1],
                   alpha=0.1, color=stage_colors[i], label=stage_names[i])
    ax1.legend(fontsize=9, loc="upper right")

    # ─── Bottom panel: DNR ───
    for i in range(3):
        t_start = stage_bounds[i]
        t_end = stage_bounds[i + 1]
        mask = (t_axis >= t_start) & (t_axis < t_end)
        ax2.plot(t_axis[mask], dnr[mask], 
                color=stage_colors[i], linewidth=2.5, label=stage_names[i])
        # Background
        ax2.axvspan(t_start, t_end, alpha=0.05, color=stage_colors[i])

    # DNR = 1 reference line
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
               label="DNR = 1 (boundary)")

    # Annotate regions
    # Find where DNR > 1 and DNR < 1
    ax2.fill_between(t_axis, 1.0, dnr, where=(dnr > 1.0),
                    alpha=0.1, color="red")
    ax2.fill_between(t_axis, 0, np.minimum(dnr, 1.0),
                    alpha=0.05, color="blue")

    # Text annotations
    ax2.text(100, max(dnr) * 0.7, "Drift-dominated\n(d^c > noise)",
            fontsize=10, color="red", ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax2.text(800, 0.3, "Noise-dominated\n(noise > d^c)",
            fontsize=10, color="blue", ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Stage boundary vertical lines
    for b in stage_bounds[1:-1]:
        ax2.axvline(x=b, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(b, max(dnr) * 1.05, f"t={b}", ha="center", fontsize=9, color="gray")

    ax2.set_xlabel("Diffusion timestep t", fontsize=12)
    ax2.set_ylabel("DNR (Drift-to-Noise Ratio)", fontsize=12)
    ax2.set_xlim(0, T)
    ax2.set_ylim(bottom=0, top=max(dnr) * 1.2)
    ax2.legend(fontsize=9, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.2)

    # ─── Save ───
    pdf_path = os.path.join(args.output_dir, "fig_effective_noise_dnr.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")

    # Print some statistics
    print(f"\nSchedule statistics:")
    print(f"  T={T}, beta_1={args.beta_1}, beta_T={args.beta_T}")
    print(f"  alpha_bar[0]={alpha_bar[0]:.6f}, alpha_bar[{T-1}]={alpha_bar[T-1]:.6f}")
    print(f"  SNR[0]={snr[0]:.2f}, SNR[{T-1}]={snr[T-1]:.6f}")
    print(f"  Max DNR={dnr.max():.4f} at t={dnr.argmax()}")
    print(f"  DNR > 1 for t in [0, {np.where(dnr < 1)[0][0] if np.any(dnr < 1) else T})")
    print(f"  drift_norm (||d^c||/sqrt(n)): {args.drift_norm}")


if __name__ == "__main__":
    main()
