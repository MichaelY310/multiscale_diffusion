"""
fig01_method_overview.py
========================
创建 RDD 方法概览图：
  - 上半部分：时间轴 t=0→1000，分为3段（颜色区分）
  - 每段标注分辨率(32×32, 16×16, 8×8)和 token 数
  - 中间：示意箭头展示 drift 方向（高频→低频）
  - 下半部分：对应的 SNR 曲线和 drift-to-noise ratio 曲线

输出：
  - fig_method_overview.pdf

用法：
  source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
  python fig01_method_overview.py [--output-dir ./]
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def compute_alpha_bar(T=1000, beta_1=1e-4, beta_T=0.02):
    """Linear noise schedule."""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar


def main():
    parser = argparse.ArgumentParser(description="Method overview figure")
    parser.add_argument("--output-dir", type=str, default="./")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Parameters
    # ─────────────────────────────────────────────────────────────────────────
    T = 1000
    # Stage boundaries (reversed diffusion order: generation goes T→0)
    # Fine: t ∈ [0, 300), Mid: t ∈ [300, 600), Coarse: t ∈ [600, 1000)
    stage_bounds = [0, 300, 600, 1000]
    stage_colors = ["#2196F3", "#FF9800", "#4CAF50"]  # blue, orange, green
    stage_labels = ["Fine Stage\n32×32 (1024 tokens)", 
                    "Mid Stage\n16×16 (256 tokens)",
                    "Coarse Stage\n8×8 (64 tokens)"]
    resolutions = ["32×32", "16×16", "8×8"]

    alpha_bar = compute_alpha_bar(T)
    t_axis = np.arange(T)
    snr = alpha_bar / (1 - alpha_bar)
    log_snr = np.log10(snr + 1e-10)

    # DNR (simplified with normalized drift)
    dnr = np.zeros(T)
    # Stage 0: t ∈ [0, 300)
    for t in range(0, 300):
        dnr[t] = np.sqrt(snr[t]) * (t / 300.0)
    # Stage 1: t ∈ [300, 600)
    for t in range(300, 600):
        dnr[t] = np.sqrt(snr[t]) * ((t - 300) / 300.0)
    # Stage 2: t ∈ [600, 1000)
    for t in range(600, 1000):
        dnr[t] = np.sqrt(snr[t]) * ((t - 600) / 400.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Create figure
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.5, 1.5], hspace=0.3)

    # ─── Top: Timeline ───
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(-20, 1020)
    ax_top.set_ylim(-0.5, 1.5)
    ax_top.axis("off")
    ax_top.set_title("RDD: Resolution-Dependent Diffusion — Method Overview", 
                     fontsize=14, fontweight="bold", pad=15)

    # Draw stage blocks
    for i in range(3):
        t_start = stage_bounds[i]
        t_end = stage_bounds[i + 1]
        width = t_end - t_start
        rect = mpatches.FancyBboxPatch(
            (t_start, 0.2), width, 0.8,
            boxstyle="round,pad=5",
            facecolor=stage_colors[i], alpha=0.3,
            edgecolor=stage_colors[i], linewidth=2
        )
        ax_top.add_patch(rect)
        # Label
        ax_top.text((t_start + t_end) / 2, 0.6, stage_labels[i],
                   ha="center", va="center", fontsize=11, fontweight="bold",
                   color=stage_colors[i])

    # Timeline arrow
    ax_top.annotate("", xy=(1010, 0.1), xytext=(-10, 0.1),
                   arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax_top.text(500, -0.2, "Diffusion timestep t →", ha="center", fontsize=11)
    ax_top.text(-10, -0.2, "t=0\n(clean)", ha="center", fontsize=9, color="gray")
    ax_top.text(1010, -0.2, "t=1000\n(noise)", ha="center", fontsize=9, color="gray")

    # Stage boundary markers
    for b in stage_bounds[1:-1]:
        ax_top.axvline(x=b, ymin=0.1, ymax=0.9, color="gray", 
                      linestyle="--", linewidth=1.5, alpha=0.7)
        ax_top.text(b, 1.1, f"T={b}", ha="center", fontsize=9, color="gray")

    # ─── Middle: Drift arrows ───
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_xlim(-20, 1020)
    ax_mid.set_ylim(-1, 1)
    ax_mid.axis("off")

    # Drift direction arrows at stage boundaries
    for i, b in enumerate(stage_bounds[1:-1]):
        ax_mid.annotate("",
                       xy=(b - 40, 0), xytext=(b + 40, 0),
                       arrowprops=dict(arrowstyle="->", lw=2.5,
                                      color="red", connectionstyle="arc3,rad=0.2"))
        ax_mid.text(b, -0.6, f"drift d^c\n(high→low freq)", 
                   ha="center", fontsize=9, color="red", style="italic")

    ax_mid.text(500, 0.7, "← Generation direction (T→0): progressively refine resolution →",
               ha="center", fontsize=10, color="purple", fontweight="bold")

    # ─── Bottom: SNR and DNR curves ───
    ax_bot = fig.add_subplot(gs[2])
    ax_bot2 = ax_bot.twinx()

    # SNR curve
    ln1 = ax_bot.plot(t_axis, log_snr, color="navy", linewidth=2, label="log₁₀(SNR)")
    ax_bot.set_xlabel("Diffusion timestep t", fontsize=12)
    ax_bot.set_ylabel("log₁₀(SNR)", fontsize=12, color="navy")
    ax_bot.tick_params(axis="y", labelcolor="navy")

    # DNR curve
    ln2 = ax_bot2.plot(t_axis, dnr, color="crimson", linewidth=2, 
                       linestyle="--", label="DNR (normalized)")
    ax_bot2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax_bot2.set_ylabel("Drift-to-Noise Ratio", fontsize=12, color="crimson")
    ax_bot2.tick_params(axis="y", labelcolor="crimson")

    # Color background by stage
    for i in range(3):
        ax_bot.axvspan(stage_bounds[i], stage_bounds[i + 1], 
                      alpha=0.08, color=stage_colors[i])

    # Legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_bot.legend(lns, labs, loc="upper right", fontsize=10)
    ax_bot.set_xlim(0, 1000)
    ax_bot.grid(True, alpha=0.2)

    # ─── Save ───
    pdf_path = os.path.join(args.output_dir, "fig_method_overview.pdf")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
