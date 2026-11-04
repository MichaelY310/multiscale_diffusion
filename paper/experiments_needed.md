# Experiments Needed for RDD Paper Revision

## Priority 1: Critical (Required for Acceptance)

### 1. Training Convergence Curves (§6.8)
**Goal:** Demonstrate faster convergence of RDD vs DiT baseline — this is the central empirical claim.

**Checkpoints needed:**
- DiT-XL/2 baseline: evaluate at 50K, 100K, 150K, 200K, 250K, 300K, 350K, 400K, 450K, 500K steps
- RDD Unified XL: evaluate at 50K, 100K, 150K, 200K, 250K, 300K, 350K, 400K steps (already have 400K)

**Evaluation per checkpoint:**
```bash
# For each checkpoint at step N:
python sample_fid_unified.py --ckpt checkpoints/dit_baseline_${N}.pt --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_${N}.pt --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000
```

**Output:** Plot FID (y-axis) vs training steps (x-axis) for both methods at CFG=1.0.

**Hardware:** 8×A100, ~2 hours per checkpoint evaluation × 18 checkpoints = ~36 GPU-hours.

---

### 2. Wall-Clock Timing Benchmark (§6.12)
**Goal:** Verify theoretical 2.7× speedup with actual measurements.

**Measurements needed:**
```bash
# RDD timing (250 steps, batch size 64)
python benchmark_throughput.py --model rdd_unified_xl --steps 250 --batch 64 --resolution_schedule 300-300-400

# DiT baseline timing (250 steps, batch size 64, full 32×32 resolution throughout)
python benchmark_throughput.py --model dit_xl --steps 250 --batch 64

# Also measure at different NFEs: 125, 65, 35
```

**Report:** images/sec on single A100, latency per image, breakdown by stage.

**Hardware:** 1×A100, ~1 hour.

---

### 3. Published Methods Comparison Table (§6.13)
**Goal:** Fill in MDM and Relay Diffusion numbers from their papers.

**Action items:**
- Look up MDM (Gu et al., 2024) ImageNet 256×256 FID from their Table/Figure (ICLR 2024 paper)
- Look up Relay Diffusion (Teng et al., 2024) ImageNet 256×256 FID from their Table (ICLR 2024 paper)
- Note training steps/compute for each for fair framing
- If available, also get f-DM numbers

**No compute needed** — literature lookup only.

---

## Priority 2: Important (Strongly Recommended)

### 4. Flow Matching Validation (§6.10)
**Goal:** Validate the FM extension experimentally.

**Status:** FM Unified XL v2 currently at 205K steps, target 400K.

**Once training completes:**
```bash
# Evaluate FM model
python sample_fid_unified.py --ckpt checkpoints/fm_unified_xl_400K.pt --cfg 1.0 --steps 250 --sampler euler --num_samples 50000 --mode fm
python sample_fid_unified.py --ckpt checkpoints/fm_unified_xl_400K.pt --cfg 1.5 --steps 250 --sampler euler --num_samples 50000 --mode fm
```

**Hardware:** Training: already running. Evaluation: 8×A100, ~4 hours.

---

### 5. CFG=1.0 Complete Ablation (§6.7)
**Goal:** Provide all ablations at CFG=1.0 to eliminate guidance confound.

```bash
bash 2d/exp/ablation_cfg1.sh
# Covers: DPM-Solver-2 and DDPM at 250/125/65/35 steps, stage split sensitivity
```

**Status:** Script ready, waiting for GPU allocation.

**Hardware:** 8×A100, ~2 days.

---

### 6. Extended Training for Scaling Analysis (§6.9)
**Goal:** Show XL eventually surpasses L with more training.

```bash
# Continue training both models to 750K steps
python train_unified.py --model L --resume checkpoints/rdd_unified_l_400K.pt --total_steps 750000
python train_unified.py --model XL --resume checkpoints/rdd_unified_xl_400K.pt --total_steps 750000

# Evaluate at 50K intervals
for step in 450000 500000 550000 600000 650000 700000 750000; do
  python sample_fid_unified.py --ckpt checkpoints/rdd_unified_l_${step}.pt --cfg 1.0 --steps 250
  python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_${step}.pt --cfg 1.0 --steps 250
done
```

**Hardware:** 8×A100, ~5 days per model.

---

## Priority 3: Nice-to-Have (Strengthens Paper)

### 7. Alternative Stage Split (§6.11)
**Status:** Multistage 100-300-600 at 240K steps.

**Once complete (400K):** Evaluate and compare with 300-300-400.

---

### 8. Posterior Consistency Verification (Reviewer Major Issue 5)
**Goal:** Numerically verify that the derived posterior satisfies the consistency condition.

```python
# Verify: ∫ q(x_s|x_t, x_0) q(x_t|x_0) dx_t = q(x_s|x_0)
# Monte Carlo verification with 100K samples at various (s, t) pairs
python verify_posterior_consistency.py --num_samples 100000 --pairs "[(100,500), (200,800), (50,950)]"
```

**Hardware:** CPU only, ~1 hour.

---

### 9. Drift Coefficient Numerical Stability Check (Reviewer Major Issue 5)
**Goal:** Verify γ_{s,t}^c is numerically stable for large jumps.

```python
# Check γ for extreme (s,t) pairs
python check_drift_stability.py --pairs "[(1,999), (1,500), (100,999)]" --schedule linear
```

**Hardware:** CPU only, ~10 minutes.

---

### 10. Downsampling Operator Ablation (Reviewer Q7)
**Goal:** Compare average pooling vs alternatives.

```bash
# Train with different downsampling operators (short runs, 100K steps)
python train_unified.py --downsample avg_pool --steps 100000
python train_unified.py --downsample gaussian_blur --steps 100000  
python train_unified.py --downsample strided_conv --steps 100000
```

**Hardware:** 8×A100, ~3 days total.

---

## Summary of Compute Requirements

| Experiment | Priority | GPU-Hours (est.) | Status |
|-----------|----------|-----------------|--------|
| Convergence curves | P1 | 36 | Need to run |
| Wall-clock timing | P1 | 1 | Need to run |
| Published comparison | P1 | 0 (literature) | Need to look up |
| FM validation | P2 | 4 (eval only) | Training in progress (205K/400K) |
| CFG=1.0 ablation | P2 | ~48 | Script ready |
| Extended training | P2 | ~240 | Need to start |
| Alt stage split | P3 | 4 (eval only) | Training in progress (240K/400K) |
| Posterior verification | P3 | 1 | Need to run |
| Drift stability | P3 | 0.1 | Need to run |
| Downsample ablation | P3 | ~72 | Need to run |

**Total estimated:** ~400 GPU-hours for all experiments.
**Critical path:** Convergence curves (P1) — most impactful, can start immediately.


---

## NEW EXPERIMENTS FROM ROUND 2 REVIEW

### Priority 0: CRITICAL (Reviewer's Core Concerns)

### 11. σ_h Sweep at Inference Time (M1, Q1)
**Goal:** Empirically validate that σ_h = 0 is optimal by sweeping the continuum at inference time (NO retraining needed).

**Rationale:** Reviewer argues the σ_h = 0 argument is tautological. This experiment provides direct empirical evidence by testing intermediate values.

**Protocol:**
```bash
# For each sigma_h value, modify the noise filling initialization:
# x_{T_c} = sqrt(alpha_bar) * UP(x0_coarse) + sqrt(1 - alpha_bar + alpha_bar * sigma_h^2) * eps_null + sqrt(1 - alpha_bar) * eps_row
# where eps_null is noise in null-space directions, eps_row in row-space

for sigma_h in 0.0 0.05 0.1 0.2 0.5 1.0; do
    python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt         --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000         --sigma_h ${sigma_h}
done
```

**Expected result:** Monotonically increasing FID with σ_h > 0.
- σ_h = 0: FID 7.06 (known)
- σ_h = 0.05: FID ~7.2 (slight degradation)
- σ_h = 0.1: FID ~7.5-8.0
- σ_h = 0.5: FID ~12-15
- σ_h = 1.0: FID ~20+ (severe, 33× excess null-space variance)

**Hardware:** 8×A100, ~12 hours (6 evaluations).

---

### 12. Cascade Ablation Without Drift (M7, Q3)
**Goal:** Isolate drift's contribution from the cascade/easier-problem effect.

**Rationale:** Reviewer argues "faster convergence" may simply be because the coarsest stage (8×8, 192 dim) is a much easier problem than full 32×32 (3072 dim). We need to disentangle: (a) benefit from solving easier problems first (cascade effect), vs (b) benefit from the drift term itself.

**Protocol:**
```bash
# Ablation A: Standard cascade WITHOUT drift (d^c = 0 during training)
# Train a multi-resolution model that processes different resolutions but with NO drift.
# At stage boundaries, simply noise-fill from coarse to fine.
python train_unified.py --model XL --no_drift --resolution_schedule 300-300-400     --total_steps 400000 --name "cascade_no_drift"

# Ablation B: Single-resolution at each level (no cascade, to isolate problem-difficulty effect)
# Train 3 separate models: one for 8×8, one for 16×16, one for 32×32
python train_single_resolution.py --model XL --resolution 8 --steps 133000
python train_single_resolution.py --model XL --resolution 16 --steps 133000
python train_single_resolution.py --model XL --resolution 32 --steps 133000

# Compare:
# 1. RDD full (cascade + drift) = FID 7.06 at CFG=1.0
# 2. Cascade without drift (ablation A)
# 3. Single-resolution models (ablation B)
# 4. Standard DiT at 32×32 = FID 19.78
```

**Expected result:** 
- Full RDD > Cascade-no-drift > Single-resolution ≈ Standard DiT
- This shows drift provides benefit BEYOND just solving easier problems

**Hardware:** 8×A100, ~5 days (multiple 400K-step trainings).

---

### 13. Convergence Curve with Asymptote Analysis (M4, Q4)
**Goal:** Address the "400K is misleading" concern by showing convergence behavior AND estimating asymptotic FID.

**Protocol:**
```bash
# Extended evaluation - push to 1M steps for both models
# Evaluate at: 50K, 100K, 150K, 200K, 300K, 400K, 500K, 600K, 750K, 1M

# Fit exponential decay: FID(t) = FID_inf + A * exp(-t/tau)
# Report: estimated FID_infinity for both methods
# If RDD asymptote < DiT asymptote: RDD is fundamentally better
# If RDD asymptote ≈ DiT asymptote: RDD just converges faster (still valuable)
```

**Hardware:** 8×A100, ~7 days (extended training + evaluation).

---

## Priority 1 (updated): Critical

### 14. Training with σ_h > 0 (M2, Q2)
**Goal:** Test whether an alternative training scheme with σ_h > 0 at initialization could be better.

**Protocol:**
```bash
# Modify forward process: at stage boundary during TRAINING, inject high-frequency:
# x_{T_c} = sqrt(alpha_bar) * UP(x0_coarse) + sqrt(alpha_bar) * sigma_h_train * (I-P)x_0 + sqrt(1-alpha_bar) * eps
for sigma_h_train in 0.1 0.3 0.5; do
    python train_unified.py --model XL --sigma_h_train ${sigma_h_train}         --total_steps 200000 --name "sigma_h_train_${sigma_h_train}"
done
# Then evaluate each with matching sigma_h at inference
```

**Note:** This requires retraining and tests the reviewer's deeper question about whether the training design itself is optimal.

**Hardware:** 8×A100, ~4 days per run.

---

### 15. Drift Magnitude Statistics Across ImageNet (Q6)
**Goal:** Report ‖d^c‖ statistics to characterize how much resolution change the drift actually performs.

**Protocol:**
```python
# Compute drift magnitudes for 50K ImageNet images at each stage
import torch
from torchvision import datasets, transforms
# For each image: d^c = UP(DOWN(x)) - x; report mean, std, percentiles
# Also report: ||d^c|| / ||x|| (relative magnitude)
# And: frequency spectrum of d^c (verify it's purely high-frequency)
```

**Hardware:** CPU, ~1 hour.

---

### 16. Error Accumulation with More Stages (Q7)
**Goal:** Test whether adding more stages degrades quality due to error accumulation.

**Protocol:**
```bash
# Compare: 2 stages, 3 stages (default), 4 stages, 5 stages
# Each trained for 200K steps
python train_unified.py --model XL --num_stages 2 --schedule "500-500" --steps 200000
python train_unified.py --model XL --num_stages 3 --schedule "300-300-400" --steps 200000
python train_unified.py --model XL --num_stages 4 --schedule "200-200-300-300" --steps 200000
python train_unified.py --model XL --num_stages 5 --schedule "150-150-200-250-250" --steps 200000
```

**Hardware:** 8×A100, ~3 days per model.

---

### 17. SNR at Stage Boundaries (Q8)
**Goal:** Report signal-to-noise ratio at each stage boundary to characterize the operating point.

**Protocol:**
```python
# For default 300-300-400 schedule:
# At t=300: SNR = alpha_bar_300 / (1 - alpha_bar_300)
# At t=600: SNR = alpha_bar_600 / (1 - alpha_bar_600)
# Also decompose into row-space SNR and null-space SNR
# Report in dB scale
```

**Hardware:** CPU, trivial computation.

---

### 18. Sign Verification of γ_{s,t}^c (C1)
**Goal:** Verify the drift coefficient has correct sign throughout the reverse process.

**Protocol:**
```python
# For all valid (s,t) pairs within each stage, verify:
# 1. γ_{s,t}^c has the correct sign for the reverse (denoising) direction
# 2. The drift correction moves the sample in the correct direction (toward higher resolution)
# Plot γ_{s,t}^c as heatmap over (s,t) space
```

**Hardware:** CPU, ~10 minutes.

---

### 19. FM Velocity Continuity at Stage Boundaries (C3)
**Goal:** Measure velocity discontinuity at stage boundaries in the FM formulation.

**Protocol:**
```python
# For the trained FM model, compute:
# v_theta(x, s_{k+1} - epsilon) vs v_theta(x, s_{k+1} + epsilon) 
# for epsilon -> 0
# Report magnitude of discontinuity ||v+ - v-|| / ||v||
# If small: velocity is approximately continuous (learned to be smooth)
# If large: this is a genuine concern requiring architectural fix
```

**Hardware:** 1×A100, ~2 hours.

---

### 20. Orthogonal Projection Verification (C4)
**Goal:** Verify P = UP∘DOWN is truly an orthogonal projection with proper normalization.

**Protocol:**
```python
# Verify: P^2 = P (idempotent)
# Verify: P^T = P (symmetric, hence orthogonal)
# Verify: (I-P)P = 0 (orthogonality of subspaces)
# Note: This depends on normalization convention for UP/DOWN
# For average-pool DOWN and repeat-UP: P is orthogonal projection
# But for unnormalized: may need 1/s^2 factor
```

**Hardware:** CPU, trivial.

---

## Updated Summary Table

| Experiment | Priority | GPU-Hours (est.) | Status | Round |
|-----------|----------|-----------------|--------|-------|
| Convergence curves | P1 | 36 | Need to run | R1 |
| Wall-clock timing | P1 | 1 | Need to run | R1 |
| Published comparison | P1 | 0 (literature) | Need to look up | R1 |
| σ_h sweep (inference) | P0 | 12 | **NEW - R2** | R2 |
| Cascade ablation (no drift) | P0 | 120 | **NEW - R2** | R2 |
| Convergence + asymptote | P0 | 168 | **NEW - R2** | R2 |
| Training with σ_h > 0 | P1 | 96 | **NEW - R2** | R2 |
| Drift magnitude stats | P1 | 1 | **NEW - R2** | R2 |
| Error accumulation (stages) | P1 | 72 | **NEW - R2** | R2 |
| SNR at boundaries | P1 | 0.01 | **NEW - R2** | R2 |
| γ sign verification | P1 | 0.1 | **NEW - R2** | R2 |
| FM velocity continuity | P2 | 2 | **NEW - R2** | R2 |
| Projection verification | P3 | 0.01 | **NEW - R2** | R2 |
| FM validation | P2 | 4 (eval only) | Training in progress | R1 |
| CFG=1.0 ablation | P2 | ~48 | Script ready | R1 |
| Extended training (L vs XL) | P2 | ~240 | Need to start | R1 |
| Alt stage split | P3 | 4 | Training in progress | R1 |
| Posterior consistency | P3 | 1 | Need to run | R1 |
| Drift stability | P3 | 0.1 | Need to run | R1 |
| Downsample ablation | P3 | ~72 | Need to run | R1 |

**Total estimated (all):** ~875 GPU-hours
**Critical path (R2 new):** σ_h sweep (12h, immediate) → Cascade ablation (120h) → Convergence asymptote (168h)
**Quick wins:** σ_h sweep, drift stats, SNR computation, projection verification (all < 1 day)


---

## NEW EXPERIMENTS FROM ROUND 3 REVIEW

### Priority 0: IMMEDIATE (Zero/Low Cost — Must Complete Within 48 Hours)

### 21. γ=0 Ablation: No Drift Correction at Inference (M4)
**Goal:** Isolate the contribution of drift correction vs. noise filling by setting γ_{s,t}^c = 0 at inference time while keeping noise-filling initialization intact.

**Rationale:** If FID barely changes, drift correction at inference is redundant (the model's effective noise prediction already accounts for drift). If FID significantly degrades, drift correction is a critical inference-time component.

**Protocol:**
```bash
# Ablation A: γ=0 (no drift correction, keep noise filling)
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --gamma_zero

# Also test with DPM-Solver-2 (where drift correction matters more)
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000 \
    --gamma_zero

# Ablation B: Random initialization (drift but no noise filling)
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --random_init

# Ablation C: Neither drift correction nor noise filling
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --gamma_zero --random_init
```

**Implementation notes:**
- `--gamma_zero`: In the posterior mean computation (Eq. 11), set γ_{s,t}^c = 0, removing the drift correction term. The model still predicts effective noise ε̃ and we still compute x̂₀ from it—we just don't apply the explicit γ correction.
- `--random_init`: At stage boundaries, instead of noise-filling from the coarse estimate, initialize with x_{T_c} = √ᾱ_{T_c} · 0 + √(1-ᾱ_{T_c}) · ε (pure noise, no coarse conditioning).

**Expected results:**
| Ablation | FID (predicted) | Interpretation |
|----------|----------------|----------------|
| Full RDD (DDPM) | 3.54 (known) | Baseline |
| Full RDD (DPM-2) | 2.62 (known) | Baseline |
| γ=0 (DDPM) | ~3.7–4.0 | Modest degradation; ε̃ prediction absorbs most drift |
| γ=0 (DPM-2) | ~3.0–3.5 | Larger degradation; DPM-2 relies on explicit drift |
| Random init | ~5–8 | Severe; noise filling is critical |
| Neither | ~8–15 | Worst case |

**Hardware:** 8×A100, ~8 hours total (4 evaluations × 2h each).

---

### 22. Drift-to-Noise Ratio ||ε̃ - ε||² vs Timestep Plot (M1)
**Goal:** Quantify the drift contribution to the effective noise target across timesteps, providing the plot requested by the reviewer.

**Protocol:**
```python
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

# Load 10K ImageNet validation images, compute d^c for each
# For each timestep t in [0, 1000]:
#   drift_contribution = (sqrt(alpha_bar_t) / sqrt(1 - alpha_bar_t)) * (t - T_c) / Delta_T * d^c
#   noise = eps ~ N(0, I)
#   ratio = ||drift_contribution||^2 / ||noise||^2  (expected = ||drift||^2 / n)
#   Also compute: ratio_per_pixel = SNR(t) * ((t-T_c)/DeltaT)^2 * mean(d^c_pixel^2)

# Plot:
# x-axis: timestep t (0 to 1000)
# y-axis 1: ||ε̃ - ε||² / ||ε||² (drift-to-noise squared ratio)
# y-axis 2: fraction of loss attributable to drift vs noise
# Separate curves for each stage (c=0, c=1, c=2)

python compute_drift_noise_ratio.py \
    --data_dir data/imagenet/val \
    --num_images 10000 \
    --schedule linear \
    --beta_start 0.0001 --beta_end 0.02 \
    --stage_boundaries "0,300,600,1000" \
    --output drift_noise_ratio_plot.pdf
```

**Key quantities to report:**
- Mean ||d^c|| for each stage across ImageNet (expected: ~0.8-1.2 for stage 0, ~0.5-0.8 for stage 1)
- Drift-to-noise ratio as function of t (expected: peaks at stage starts, decays with t)
- Crossover point where drift contribution = noise contribution

**Hardware:** CPU only, ~30 minutes for 10K images.

---

### 23. Per-Stage Training Loss Decomposition (M5, Q3)
**Goal:** Report training loss separately for each resolution stage, showing whether coarse stages converge to near-zero loss (confirming overparameterization concern).

**Protocol:**
```bash
# Extract per-stage loss from training logs
# If not logged separately, re-evaluate on validation set with loss decomposition:
python eval_per_stage_loss.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --data_dir data/imagenet/val \
    --num_samples 10000 \
    --stage_boundaries "0,300,600,1000"

# Output: Loss_c0 (32×32), Loss_c1 (16×16), Loss_c2 (8×8)
# Also: loss per timestep within each stage (heatmap)
```

**Expected results:**
- Stage c=2 (8×8, 16 tokens): Loss ≈ 0.001–0.01 (near-zero, confirming overparameterization)
- Stage c=1 (16×16, 64 tokens): Loss ≈ 0.05–0.1
- Stage c=0 (32×32, 256 tokens): Loss ≈ 0.2–0.4 (highest, most complex task)

**Hardware:** 1×A100, ~1 hour.

---

### 24. Monte Carlo Posterior Verification — Single-Step Limit (M3)
**Goal:** Numerically verify that γ_{s,t}^c produces the correct posterior for t=s+1 (single-step) AND for multi-step jumps.

**Protocol:**
```python
import numpy as np

def verify_posterior_consistency(s, t, T_c, Delta_T, alpha_bar, d_c, num_samples=100000):
    """
    Verify: samples from q(x_s|x_t, x_0) with x_t ~ q(x_t|x_0) 
    should match samples from q(x_s|x_0) in mean and variance.
    """
    n = len(d_c)  # dimension
    x_0 = np.random.randn(n)  # fixed clean image
    
    # Sample x_t from q(x_t|x_0)
    mean_t = np.sqrt(alpha_bar[t]) * (x_0 + (t - T_c)/Delta_T * d_c)
    x_t_samples = mean_t + np.sqrt(1 - alpha_bar[t]) * np.random.randn(num_samples, n)
    
    # For each x_t sample, sample x_s from q(x_s|x_t, x_0)
    # Using our derived posterior with gamma
    gamma = compute_gamma(s, t, T_c, Delta_T, alpha_bar)
    # ... [posterior sampling code]
    
    # Compare marginal statistics with q(x_s|x_0)
    mean_s_expected = np.sqrt(alpha_bar[s]) * (x_0 + (s - T_c)/Delta_T * d_c)
    var_s_expected = 1 - alpha_bar[s]
    
    # Report relative error
    mean_error = np.linalg.norm(x_s_samples.mean(0) - mean_s_expected) / np.linalg.norm(mean_s_expected)
    var_error = abs(x_s_samples.var() - var_s_expected) / var_s_expected
    return mean_error, var_error

# Test cases:
# Single-step: (s, t) = (99, 100), (299, 300), (599, 600)
# Multi-step: (s, t) = (0, 100), (0, 300), (100, 500), (300, 900)
# Cross-boundary (should NOT be used, but verify it fails): (250, 350) spanning T_1=300

python verify_posterior.py --num_samples 100000 \
    --test_pairs "[(99,100),(299,300),(0,100),(0,300),(100,500)]" \
    --schedule linear --stage_boundaries "0,300,600,1000"
```

**Acceptance criterion:** Relative error < 0.01 (within Monte Carlo sampling noise for 100K samples).

**Hardware:** CPU only, ~10 minutes.

---

### 25. Random-Init Ablation: Drift Without Noise Filling (M4)
**Goal:** Test whether the drift mechanism alone (without noise-filling initialization) provides value.

**Protocol:** (Same as Ablation B in experiment #21 above)

**Hardware:** 8×A100, ~2 hours.

---

### 26. Per-Stage Loss Curves Over Training (M5, Q7)
**Goal:** Plot per-stage training loss as a function of training step to show convergence rates by stage.

**Protocol:**
```bash
# If training logs contain per-stage loss breakdown:
python plot_per_stage_loss_curves.py --log_dir logs/rdd_unified_xl/ \
    --output per_stage_loss_curves.pdf

# If not, re-evaluate at saved checkpoints:
for step in 50000 100000 150000 200000 250000 300000 350000 400000; do
    python eval_per_stage_loss.py --ckpt checkpoints/rdd_unified_xl_${step}.pt \
        --data_dir data/imagenet/val --num_samples 5000 \
        --stage_boundaries "0,300,600,1000"
done
```

**Expected result:** 
- Stage c=2 (8×8): converges to near-zero by 50K steps
- Stage c=1 (16×16): converges by ~150K steps  
- Stage c=0 (32×32): still decreasing at 400K steps (bottleneck)

**Hardware:** 1×A100, ~8 hours (evaluating 8 checkpoints).

---

## Updated Summary Table (All Rounds)

| # | Experiment | Priority | GPU-Hours | Status | Round |
|---|-----------|----------|-----------|--------|-------|
| 1 | Convergence curves | P1 | 36 | Need to run | R1 |
| 2 | Wall-clock timing | P1 | 1 | Need to run | R1 |
| 3 | Published comparison | P1 | 0 | Need to look up | R1 |
| 4 | FM validation | P2 | 4 | Training (305K/400K) | R1 |
| 5 | CFG=1.0 ablation | P2 | 48 | Script ready | R1 |
| 6 | Extended training | P2 | 240 | Need to start | R1 |
| 7 | Alt stage split | P3 | 4 | Training (340K/400K) | R1 |
| 8 | Posterior consistency | P3→P0 | 0.1 | **OVERDUE** | R1 |
| 9 | Drift stability | P3 | 0.1 | Need to run | R1 |
| 10 | Downsample ablation | P3 | 72 | Need to run | R1 |
| 11 | σ_h sweep (inference) | P0 | 12 | Need to run | R2 |
| 12 | Cascade no-drift ablation | P0 | 120 | Need to run | R2 |
| 13 | Convergence + asymptote | P0 | 168 | Need to run | R2 |
| 14 | Training with σ_h > 0 | P1 | 96 | Need to run | R2 |
| 15 | Drift magnitude stats | P1 | 1 | Need to run | R2 |
| 16 | Error accumulation | P1 | 72 | Need to run | R2 |
| 17 | SNR at boundaries | P1 | 0.01 | Need to run | R2 |
| 18 | γ sign verification | P1 | 0.1 | Need to run | R2 |
| 19 | FM velocity continuity | P2 | 2 | Need to run | R2 |
| 20 | Projection verification | P3 | 0.01 | Need to run | R2 |
| **21** | **γ=0 ablation (inference)** | **P0** | **8** | **NEW - R3** | **R3** |
| **22** | **Drift-to-noise ratio plot** | **P0** | **0.5** | **NEW - R3** | **R3** |
| **23** | **Per-stage loss decomposition** | **P0** | **1** | **NEW - R3** | **R3** |
| **24** | **Monte Carlo posterior (single-step)** | **P0** | **0.01** | **NEW - R3** | **R3** |
| **25** | **Random-init ablation** | **P0** | **2** | **NEW - R3** | **R3** |
| **26** | **Per-stage loss curves over training** | **P0** | **8** | **NEW - R3** | **R3** |

**Total estimated (all rounds):** ~895 GPU-hours
**Critical path (R3 new — 48h deadline):** Experiments 22, 23, 24, 26 (all CPU/log analysis, <1h total) → Experiment 21 (8h GPU) → Experiment 25 (2h GPU)
**Absolute minimum for next revision:** #21, #22, #23, #24 (these answer the reviewer's most pointed questions with near-zero cost)


---

## NEW EXPERIMENTS FROM ROUND 4 REVIEW

### Priority 0: CRITICAL (Directly Addresses Round 4 Major Issues)

### 27. Per-Class FID Analysis (M3: Data-Dependent Drift / OOD Classes)
**Goal:** Test whether classes with unusual frequency content (text-heavy, regular patterns, fine textures) have systematically worse FID, indicating drift prediction bias for rare/unusual classes.

**Rationale:** Reviewer argues that classes with atypical frequency content (e.g., "honeycomb," "chain-link fence," "newspaper") will have atypical drift d^c, and the model's drift prediction may be biased for these rare classes.

**Protocol:**
```bash
# Generate 50 samples per class (1000 classes × 50 = 50K total, matches our standard eval)
python sample_per_class.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --samples_per_class 50

# Compute per-class FID (or per-class IS since per-class FID requires many samples)
python eval_per_class_fid.py --generated_dir samples/per_class/ \
    --reference_dir data/imagenet/val/ \
    --output per_class_fid_analysis.csv

# Identify outlier classes (>2σ worse than mean)
# Cross-reference with drift magnitude statistics per class
python analyze_per_class_drift.py --data_dir data/imagenet/val/ \
    --fid_results per_class_fid_analysis.csv \
    --output drift_fid_correlation.pdf
```

**Key analysis:**
- Pearson correlation between per-class ||d^c|| and per-class FID
- Identify 10 worst-performing classes and check if they have unusual frequency content
- Compare RDD per-class FID distribution vs DiT baseline per-class FID distribution

**Hardware:** 8×A100, ~4 hours (generation) + CPU analysis.

---

### 28. Stage Split Training Ablation (M4: 300-300-400 Unmotivated)
**Goal:** Determine whether the 300-300-400 split is optimal or if alternative allocations improve quality.

**Rationale:** Reviewer correctly notes that the split was never justified from first principles and no TRAINING ablation exists (Table 4 only shows inference mismatch, not different training splits).

**Protocol:**
```bash
# Train 4 models with different stage splits (200K steps each for efficiency)
python train_unified.py --model XL --schedule "200-300-500" --total_steps 200000 --name split_200_300_500
python train_unified.py --model XL --schedule "300-300-400" --total_steps 200000 --name split_300_300_400
python train_unified.py --model XL --schedule "400-300-300" --total_steps 200000 --name split_400_300_300
python train_unified.py --model XL --schedule "250-250-500" --total_steps 200000 --name split_250_250_500
python train_unified.py --model XL --schedule "150-350-500" --total_steps 200000 --name split_150_350_500

# Evaluate each at matched inference splits
for split in "200-300-500" "300-300-400" "400-300-300" "250-250-500" "150-350-500"; do
    python sample_fid_unified.py --ckpt checkpoints/split_${split//"-"/"_"}_200K.pt \
        --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000 \
        --schedule ${split}
done
```

**Expected results:**
- 300-300-400 should be competitive but not necessarily optimal
- Allocating more to coarse (high-noise) stages may help (e.g., 200-300-500)
- Allocating less to coarse (e.g., 400-300-300) likely hurts (insufficient budget for low-SNR regime)

**Hardware:** 8×A100, ~5 days (5 × 200K-step trainings).

---

### 29. Adaptive Patch Size Baseline (M6: Simpler Alternative)
**Goal:** Implement and compare adaptive patch size (larger patches at high noise, smaller at low noise) as a baseline that achieves token count reduction without drift.

**Rationale:** Reviewer suggests that variable patch size (e.g., patch=4 at high noise → patch=2 at low noise) achieves similar compute reduction without the mathematical machinery of RDD. This baseline was never compared.

**Protocol:**
```bash
# Adaptive patch baseline: patch_size=4 for t∈[600,1000], patch_size=3 for t∈[300,600), patch_size=2 for t∈[0,300)
# This gives token counts: 64 (high noise), ~113 (mid), 256 (low noise) — similar compute profile to RDD
python train_adaptive_patch.py --model XL --total_steps 400000 \
    --patch_schedule "2:0-300,3:300-600,4:600-1000" \
    --name adaptive_patch_baseline

# Evaluate
python sample_fid_adaptive.py --ckpt checkpoints/adaptive_patch_baseline_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000
```

**Key implementation challenges (to document):**
1. Different patch sizes produce different sequence lengths → need separate positional embeddings per regime (architectural modification)
2. Patch-size transition mid-inference creates discontinuity (analogous to RDD's stage boundary)
3. Information loss: patch_size=4 downsamples 4× (losing information), unlike RDD's drift which smoothly transitions

**Expected result:** Adaptive patch achieves moderate improvement over baseline DiT (from compute savings) but worse than RDD (lacks the drift's frequency-aware training signal).

**Hardware:** 8×A100, ~5 days.

---

### 30. Honest Wall-Clock at 125 NFE (M7: Speedup Dominated by 32×32)
**Goal:** Measure actual wall-clock time at the practical 125 NFE operating point and at 250 NFE, breaking down per-stage contribution.

**Protocol:**
```bash
# Measure per-stage timing with warm-up (discard first 10 batches)
python benchmark_per_stage.py --model rdd_unified_xl \
    --batch_size 16 --num_warmup 10 --num_measure 100 \
    --nfe 250 --step_alloc "90/80/80" \
    --output timing_250nfe.json

python benchmark_per_stage.py --model rdd_unified_xl \
    --batch_size 16 --num_warmup 10 --num_measure 100 \
    --nfe 125 --step_alloc "45/40/40" \
    --output timing_125nfe.json

# Baseline DiT at both NFEs
python benchmark_per_stage.py --model dit_xl \
    --batch_size 16 --num_warmup 10 --num_measure 100 \
    --nfe 250 --output timing_dit_250nfe.json

python benchmark_per_stage.py --model dit_xl \
    --batch_size 16 --num_warmup 10 --num_measure 100 \
    --nfe 125 --output timing_dit_125nfe.json
```

**Report format:**
| Setting | DiT (ms) | RDD (ms) | Speedup | 32×32 % |
|---------|----------|----------|---------|---------|
| 250 NFE | ? | ? | ~1.8-2.0× | ~78% |
| 125 NFE | ? | ? | ~1.8-2.0× | ~78% |

**Hardware:** 1×A100, ~2 hours (dedicated, uncontended).

---

### 31. Reduced Noise Filling Experiment (M5: Noise Filling "Resets" Progress)
**Goal:** Empirically validate that noise filling is essential by testing reduced-noise variants that partially preserve coarse-stage denoising progress.

**Rationale:** Reviewer argues noise filling "resets" denoising. We predict reduced noise variants will degrade quality because the fine model was trained to expect full isotropic noise at T_c.

**Protocol:**
```bash
# Variant A: Full noise filling (standard, baseline)
# x_{T_c} = sqrt(alpha_bar) * UP(x0_coarse) + sqrt(1-alpha_bar) * eps

# Variant B: 50% reduced noise (attempt to "preserve" coarse work)
# x_{T_c} = sqrt(alpha_bar) * UP(x0_coarse) + 0.5 * sqrt(1-alpha_bar) * eps

# Variant C: 0% noise (pure deterministic upsample, no noise at all)
# x_{T_c} = sqrt(alpha_bar) * UP(x0_coarse)

# Variant D: Anisotropic — full noise in null-space, reduced in row-space
# This tests whether row-space noise is the "redundant" part

for noise_scale in 1.0 0.5 0.0; do
    python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
        --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
        --noise_fill_scale ${noise_scale}
done

# Anisotropic variant
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --noise_fill_anisotropic --null_scale 1.0 --row_scale 0.0
```

**Expected results:**
- Full noise (scale=1.0): FID 3.54 (known baseline)
- Reduced noise (scale=0.5): FID ~5-8 (out-of-distribution for fine model)
- Zero noise (scale=0.0): FID ~15-25 (model receives deterministic input it never trained on; cannot generate diverse textures)
- Anisotropic (full null, zero row): FID ~4-5 (row-space noise is partially redundant since mean already encodes coarse info)

**Hardware:** 8×A100, ~8 hours.

---

### Priority 1: Important (Strengthens Response)

### 32. DPM-Solver Drift Correction Overhead Measurement (m8)
**Goal:** Quantify the overhead of drift correction (UP/DOWN operations) per DPM-Solver step.

**Protocol:**
```python
# Profile the drift correction operations:
# hat_d^c = UP(DOWN(hat_x0)) - hat_x0
# This requires 1 DOWN (avg pool) + 1 UP (nearest neighbor) per solver step
# For DPM-Solver-2: 2× these operations per step (initial + midpoint)

# Measure:
# Time for DOWN (avg_pool2d, 32×32 → 16×16): expected ~0.01ms
# Time for UP (nearest upsample, 16×16 → 32×32): expected ~0.01ms
# vs total step time (~25ms): overhead < 0.1%
```

**Hardware:** 1×A100, ~30 minutes.

---

### 33. Cosine Schedule Compatibility Analysis (m1)
**Goal:** Verify that RDD's drift formulation works with cosine noise schedule (not just linear).

**Protocol:**
```bash
# Train a short run (100K steps) with cosine schedule
python train_unified.py --model XL --schedule_type cosine \
    --total_steps 100000 --name rdd_cosine_schedule

# Compare convergence with linear schedule at 100K
python sample_fid_unified.py --ckpt checkpoints/rdd_cosine_schedule_100K.pt \
    --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000
```

**Analysis:** The drift coefficient γ_{s,t}^c depends on the alpha_bar schedule. For cosine schedule, alpha_bar changes more smoothly, which may improve or worsen drift estimation. The key question is whether the marginal (Eq. 5) still holds with cosine schedule — it does, since the derivation only requires the per-step structure, not a specific schedule form.

**Hardware:** 8×A100, ~1.5 days.

---

### 34. Precision/Recall Metrics for All Main Results (m5)
**Goal:** Report Precision and Recall for all entries in Table 2 (main results).

**Protocol:**
```bash
# Re-evaluate main results with P/R metrics
python eval_precision_recall.py --generated_dir samples/rdd_unified_xl_dpm2_cfg1.5/ \
    --reference_dir data/imagenet/ref_batch/ \
    --output precision_recall_main.json

# Also for baseline
python eval_precision_recall.py --generated_dir samples/dit_baseline_cfg1.25/ \
    --reference_dir data/imagenet/ref_batch/ \
    --output precision_recall_baseline.json
```

**Hardware:** 1×A100, ~2 hours.

---

### 35. Multi-Seed Variance Estimation (m7)
**Goal:** Run 3 seeds for the main result (DPM-Solver-2, CFG=1.5) to report confidence intervals.

**Protocol:**
```bash
for seed in 42 123 456; do
    python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
        --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000 \
        --seed ${seed} --output_dir samples/seed_${seed}/
done
# Report: FID mean ± std across 3 seeds
```

**Expected:** FID variance across seeds should be small (±0.05-0.1) for 50K samples.

**Hardware:** 8×A100, ~6 hours (3 evaluations).

---

## Updated Summary Table (All Rounds Including R4)

| # | Experiment | Priority | GPU-Hours | Status | Round |
|---|-----------|----------|-----------|--------|-------|
| 1 | Convergence curves | P1 | 36 | Need to run | R1 |
| 2 | Wall-clock timing | P1 | 1 | Need to run | R1 |
| 3 | Published comparison | P1 | 0 | Need to look up | R1 |
| 4 | FM validation | P2 | 4 | Training (305K/400K) | R1 |
| 5 | CFG=1.0 ablation | P2 | 48 | Script ready | R1 |
| 6 | Extended training | P2 | 240 | Need to start | R1 |
| 7 | Alt stage split | P3 | 4 | Training (340K/400K) | R1 |
| 8 | Posterior consistency | P0 | 0.1 | OVERDUE | R1 |
| 9 | Drift stability | P3 | 0.1 | Need to run | R1 |
| 10 | Downsample ablation | P3 | 72 | Need to run | R1 |
| 11 | σ_h sweep (inference) | P0 | 12 | Need to run | R2 |
| 12 | Cascade no-drift ablation | P0 | 120 | Need to run | R2 |
| 13 | Convergence + asymptote | P0 | 168 | Need to run | R2 |
| 14 | Training with σ_h > 0 | P1 | 96 | Need to run | R2 |
| 15 | Drift magnitude stats | P1 | 1 | Need to run | R2 |
| 16 | Error accumulation | P1 | 72 | Need to run | R2 |
| 17 | SNR at boundaries | P1 | 0.01 | Need to run | R2 |
| 18 | γ sign verification | P1 | 0.1 | Need to run | R2 |
| 19 | FM velocity continuity | P2 | 2 | Need to run | R2 |
| 20 | Projection verification | P3 | 0.01 | Need to run | R2 |
| 21 | γ=0 ablation (inference) | P0 | 8 | Need to run | R3 |
| 22 | Drift-to-noise ratio plot | P0 | 0.5 | Need to run | R3 |
| 23 | Per-stage loss decomposition | P0 | 1 | Need to run | R3 |
| 24 | Monte Carlo posterior (single-step) | P0 | 0.01 | Need to run | R3 |
| 25 | Random-init ablation | P0 | 2 | Need to run | R3 |
| 26 | Per-stage loss curves over training | P0 | 8 | Need to run | R3 |
| **27** | **Per-class FID analysis** | **P0** | **4** | **NEW - R4** | **R4** |
| **28** | **Stage split TRAINING ablation** | **P0** | **120** | **NEW - R4** | **R4** |
| **29** | **Adaptive patch size baseline** | **P0** | **120** | **NEW - R4** | **R4** |
| **30** | **Honest wall-clock at 125/250 NFE** | **P0** | **2** | **NEW - R4** | **R4** |
| **31** | **Reduced noise filling experiment** | **P0** | **8** | **NEW - R4** | **R4** |
| **32** | **DPM-Solver drift overhead** | **P1** | **0.5** | **NEW - R4** | **R4** |
| **33** | **Cosine schedule compatibility** | **P1** | **36** | **NEW - R4** | **R4** |
| **34** | **Precision/Recall for main results** | **P1** | **2** | **NEW - R4** | **R4** |
| **35** | **Multi-seed variance (3 seeds)** | **P1** | **6** | **NEW - R4** | **R4** |

**Total estimated (all rounds R1-R4):** ~1,190 GPU-hours
**Critical path (R4 new):** Experiments 30, 31 (inference-only, ~10h) → Experiment 27 (4h) → Experiments 28, 29 (training, ~5 days each)
**Quick wins (R4):** #30 (wall-clock, 2h), #31 (noise fill variants, 8h), #32 (overhead measurement, 30min), #34 (P/R, 2h), #35 (multi-seed, 6h)


---

## NEW EXPERIMENTS FROM ROUND 5 REVIEW

### Priority 0: CRITICAL (Directly Addresses Round 5 Major Issues)

### 36. Progressive Training Baseline (M2: Curriculum Learning Confound)
**Goal:** Isolate curriculum learning benefit from RDD's drift contribution by training a standard DiT with progressive resolution curriculum (no drift, no multi-resolution forward process).

**Rationale:** Reviewer argues that RDD's improvement may come from the implicit curriculum (coarse converges first) rather than the drift mathematics. A progressive DiT baseline that trains on 8×8 → 16×16 → 32×32 sequentially would capture the curriculum effect without any drift.

**Protocol:**
```bash
# Progressive DiT: train at each resolution sequentially
python train_dit_progressive.py --model XL \
    --phase1_resolution 8 --phase1_steps 133000 \
    --phase2_resolution 16 --phase2_steps 133000 \
    --phase3_resolution 32 --phase3_steps 134000 \
    --total_steps 400000 --name progressive_dit

# Evaluate at CFG=1.0 and CFG=1.5
python sample_fid.py --ckpt checkpoints/progressive_dit_400K.pt \
    --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000
python sample_fid.py --ckpt checkpoints/progressive_dit_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000
```

**Key challenges:**
- Progressive training requires changing input size mid-training (pos-embed interpolation between phases)
- Model must handle all resolutions at inference despite being fine-tuned on 32×32 last
- Catastrophic forgetting of coarse behavior during fine-tuning phase is expected

**Expected results:**
- Progressive DiT (CFG=1.0): FID ~12–15 (better than baseline 19.78 due to curriculum, worse than RDD 7.06)
- Progressive DiT (CFG=1.5): FID ~6–8 (better than baseline 10.26, worse than RDD 2.62)
- If progressive DiT matches RDD → drift provides no value beyond curriculum (bad for paper)
- If progressive DiT is significantly worse → drift provides value beyond curriculum (good for paper)

**Hardware:** 8×A100, ~5 days (400K-step training).

---

### 37. DDIM η Sweep (M3: Stochastic vs Deterministic Gap Explanation)
**Goal:** Confirm that the 0.92 FID gap between DDPM (stochastic) and DPM-Solver-2 (deterministic) is explained by how well the sampler captures the smooth drift structure—by showing a smooth interpolation with DDIM's η parameter.

**Rationale:** DDIM with η=0 is fully deterministic, η=1 is equivalent to DDPM. If the gap is explained by drift-structure exploitation, FID should interpolate smoothly between 2.62 (η=0) and 3.54 (η=1).

**Protocol:**
```bash
for eta in 0.0 0.25 0.5 0.75 1.0; do
    python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
        --cfg 1.5 --steps 250 --sampler ddim --eta ${eta} \
        --num_samples 50000 --output_dir samples/ddim_eta_${eta}/
done
```

**Expected results:**
| η | Expected FID | Interpretation |
|---|-------------|----------------|
| 0.0 | ~2.62 | Full deterministic, matches DPM-Solver-2 |
| 0.25 | ~2.85 | Mostly deterministic, slight stochastic noise |
| 0.5 | ~3.10 | Balanced |
| 0.75 | ~3.35 | Mostly stochastic |
| 1.0 | ~3.54 | Full stochastic, matches DDPM |

Smooth monotonic increase confirms the drift-structure exploitation hypothesis.

**Hardware:** 8×A100, ~10 hours (5 evaluations × 2h each).

---

### 38. Precision/Recall at DPM-Solver-2 vs DDPM (M4: Mode-Seeking Bias Test)
**Goal:** Verify that DPM-Solver-2's stronger drift correction does NOT reduce sample diversity (Recall), confirming the SDS analogy does not apply.

**Rationale:** If drift correction were mode-seeking (like SDS), DPM-Solver-2 (which relies more heavily on drift) would have lower Recall than DDPM. If Recall is comparable, mode-seeking is not occurring.

**Protocol:**
```bash
# Compute P/R for DPM-Solver-2 samples
python eval_precision_recall.py \
    --generated_dir samples/rdd_unified_xl_dpm2_cfg1.5/ \
    --reference_dir data/imagenet/ref_batch/ \
    --output pr_dpm2.json

# Compute P/R for DDPM samples
python eval_precision_recall.py \
    --generated_dir samples/rdd_unified_xl_ddpm_cfg1.5/ \
    --reference_dir data/imagenet/ref_batch/ \
    --output pr_ddpm.json

# Also compute for DiT baseline
python eval_precision_recall.py \
    --generated_dir samples/dit_baseline_cfg1.25/ \
    --reference_dir data/imagenet/ref_batch/ \
    --output pr_dit_baseline.json
```

**Expected results:**
| Model | Precision | Recall | FID |
|-------|-----------|--------|-----|
| RDD DPM-2 (CFG=1.5) | ~0.83 | ~0.56 | 2.62 |
| RDD DDPM (CFG=1.5) | ~0.80 | ~0.58 | 3.54 |
| DiT baseline (CFG=1.25) | ~0.72 | ~0.62 | 10.26 |

Key prediction: DPM-Solver-2's Recall ≈ DDPM's Recall (mode-seeking NOT confirmed).

**Hardware:** 8×A100, ~4 hours.

---

### 39. Truncated Sampling — Skip Coarsest Stage (m5: Practical Deployment)
**Goal:** Test RDD performance when skipping the coarsest stage (starting from t=600 with pure noise at 16×16 resolution), simulating a practical truncated deployment.

**Protocol:**
```bash
# Standard 3-stage RDD (baseline)
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000

# Truncated: skip coarsest stage, start from t=600 at 16×16
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 170 --sampler dpm2 --num_samples 50000 \
    --truncate_start 600 --start_resolution 16
```

**Expected results:**
- Full 3-stage: FID 2.62 (known)
- Truncated 2-stage: FID ~3.5–5.0 (moderate degradation from skipping coarse planning stage)

**Hardware:** 8×A100, ~2 hours.

---

### 40. FLOP-Normalized Convergence Curve (M1: Fair Comparison)
**Goal:** Replot the convergence curve (Experiment #1) with x-axis = total FLOPs (not training steps) to provide the fairest possible comparison accounting for RDD's 13% overhead.

**Protocol:**
```python
# No new computation needed — replot existing convergence data
# RDD: multiply step count by 1.133 (13.3% overhead) for FLOP-normalized x-axis
# DiT: step count = FLOP count (normalized to 1.0 per step)
# This shows that even on a FLOP-matched basis, RDD converges dramatically faster

python plot_convergence_flop_normalized.py \
    --rdd_results convergence_rdd.csv \
    --dit_results convergence_dit.csv \
    --rdd_flop_multiplier 1.133 \
    --output convergence_flop_normalized.pdf
```

**Hardware:** CPU, trivial (replotting).

---

## NEW EXPERIMENTS FROM ROUND 6 REVIEW

### Priority 0: CRITICAL (Directly Addresses Round 6 Major Issues)

### 41. Reference Implementation + Numerical Test Cases (M1: Reproducibility)
**Goal:** Provide a minimal self-contained reference implementation (~100 lines) of RDD's forward process, drift computation, and posterior sampling, along with exact numerical test cases for external verification.

**Rationale:** Reviewer correctly identifies that bugs in drift computation would be "silent" (model adapts). Providing exact numerical test cases allows any implementer to verify correctness without training a model.

**Protocol:**
```python
# Create file: rdd_reference_implementation.py (~100 lines)
# Contains:
# 1. forward_process(x0, t, d_c, alpha_bar) -> x_t (Eq. 5)
# 2. compute_drift(x0, down_op, up_op) -> d_c
# 3. posterior_mean(x_t, x0, t, s, gamma, d_c) -> mu_s|t
# 4. posterior_variance(alpha_bar, t, s) -> sigma_s|t^2
# 5. noise_fill(x_coarse, alpha_bar_Tc) -> x_Tc_fine

# Numerical test cases (to 6 decimal places):
# Test 1: x0 = [1.0, -0.5, 0.3, 0.8] (4-dim toy), t=500, s=499
#   Expected posterior mean: [...]
#   Expected posterior variance: [...]
# Test 2: x0 = randn(192) with seed=42, stage c=2 (8×8), t=800, s=750
#   Expected: [first 10 values of posterior mean]
# Test 3: Monte Carlo verification (100K samples) confirming marginal consistency

python create_reference_implementation.py
python run_numerical_tests.py --output numerical_test_results.json
```

**Deliverable:** Self-contained Python file + JSON with exact expected outputs.

**Hardware:** CPU only, ~30 minutes.

---

### 42. Per-Stage Drift Magnitude Analysis (M2: Degeneracy at Coarse Stage)
**Goal:** Quantify ||d^c|| separately for each stage to determine if the coarsest stage (8→4) is degenerate (d^c ≈ 0).

**Rationale:** If natural images at 8×8 are already nearly piecewise-constant (each 2×2 block has low variance), then d^{c=2} ≈ 0 and the coarsest stage provides no drift benefit—it's effectively standard DDPM.

**Protocol:**
```bash
python compute_per_stage_drift.py \
    --data_dir data/imagenet/val \
    --num_images 50000 \
    --resolutions "32:16,16:8,8:4" \
    --output per_stage_drift_stats.json

# Output format:
# {
#   "stage_0_32to16": {"mean": ..., "std": ..., "p5": ..., "p95": ...},
#   "stage_1_16to8": {"mean": ..., "std": ..., "p5": ..., "p95": ...},
#   "stage_2_8to4": {"mean": ..., "std": ..., "p5": ..., "p95": ...}
# }
```

**Expected results:**
- ||d^{c=0}|| (32→16): ~0.8–1.2 (substantial high-frequency content lost)
- ||d^{c=1}|| (16→8): ~0.4–0.7 (moderate)
- ||d^{c=2}|| (8→4): ~0.1–0.3 (small but likely non-zero)

If ||d^{c=2}|| < 0.2, this confirms the reviewer's concern about coarse-stage degeneracy.

**Hardware:** CPU, ~1 hour.

---

### 43. 2-Stage Model Comparison (M2: Is 3rd Stage Necessary?)
**Goal:** Train and evaluate a 2-stage RDD model (16×16 → 32×32 only, no 8×8 stage) to test whether the degenerate coarsest stage contributes meaningfully.

**Protocol:**
```bash
# 2-stage model: only 16×16 and 32×32
python train_unified.py --model XL --num_stages 2 \
    --schedule "500-500" --resolutions "16,32" \
    --total_steps 400000 --name rdd_2stage

# Evaluate
python sample_fid_unified.py --ckpt checkpoints/rdd_2stage_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 --num_samples 50000
```

**Expected results:**
- 3-stage RDD: FID 2.62
- 2-stage RDD: FID ~3.0–3.5 (if coarsest stage contributes little)
- If 2-stage ≈ 3-stage: confirms coarse stage is degenerate, simplify to 2 stages
- If 2-stage significantly worse: coarse stage provides value beyond drift (coarse planning benefit)

**Hardware:** 8×A100, ~5 days.

---

### 44. Sequential Contribution Decomposition Ablation (M5: Which Component Matters?)
**Goal:** Decompose the total FID improvement into contributions from each of RDD's 6 components via sequential ablation.

**Rationale:** The paper attributes 4× improvement to "RDD" but never identifies which components contribute most. Without this, the theoretical contribution (drift) may be overstated if most benefit comes from simpler mechanisms.

**Protocol:**
```bash
# Start from full RDD (FID 2.62 DPM-2, 3.54 DDPM) and ablate one component at a time:

# A. Remove DPM-Solver-2 drift correction (use DDPM): Already known → FID 3.54 (+0.92)
# B. Remove stage conditioning c (set c=0 always):
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --force_stage_cond 0

# C. Remove drift correction γ=0 (keep noise filling): Experiment #21
# D. Remove noise filling (random init): Experiment #25
# E. Remove multi-resolution (single resolution, all at 32×32 with drift):
#    Requires retraining → Experiment #12 (cascade no-drift already covers partial)

# Full decomposition table:
# | Configuration | FID | Δ from full RDD | Component isolated |
# |---------------|-----|-----------------|-------------------|
# | Full RDD (DPM-2) | 2.62 | — | — |
# | Full RDD (DDPM) | 3.54 | +0.92 | DPM-Solver-2 value |
# | DDPM + γ=0 | ~3.7-4.0 | +0.2-0.5 | Inference drift correction |
# | DDPM + γ=0 + random init | ~8-15 | +4-11 | Noise filling value |
# | DDPM + no stage cond | ~4-5 | +0.5-1.5 | Stage conditioning value |
# | DiT baseline | 10.26 | +7.64 | Everything together |
```

**Key insight:** This is partially achievable with EXISTING experiments (#21, #25) plus one new run (remove stage conditioning). The full factorial requires retraining (cascade ablations from #12).

**Implementation:** Combine results from experiments #21, #25, #12 + new "no stage conditioning" ablation.

**Hardware:** 8×A100, ~4 hours (inference-only for the new ablation; training ablations already committed).

---

### 45. Full Precision/Recall at CFG=1.0 and CFG=1.5 (M6: Diversity Bottleneck)
**Goal:** Provide complete Precision/Recall decomposition comparing RDD vs DiT at both guidance scales to assess whether the 8×8 bottleneck hurts diversity.

**Protocol:**
```bash
# RDD at CFG=1.0
python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.0 --steps 250 --sampler dpm2 --num_samples 50000 \
    --output_dir samples/rdd_cfg1.0/
python eval_precision_recall.py --generated_dir samples/rdd_cfg1.0/ \
    --reference_dir data/imagenet/ref_batch/ --output pr_rdd_cfg1.0.json

# RDD at CFG=1.5
python eval_precision_recall.py --generated_dir samples/rdd_cfg1.5/ \
    --reference_dir data/imagenet/ref_batch/ --output pr_rdd_cfg1.5.json

# DiT baseline at CFG=1.0
python sample_fid.py --ckpt checkpoints/dit_baseline_400K.pt \
    --cfg 1.0 --steps 250 --sampler ddpm --num_samples 50000 \
    --output_dir samples/dit_cfg1.0/
python eval_precision_recall.py --generated_dir samples/dit_cfg1.0/ \
    --reference_dir data/imagenet/ref_batch/ --output pr_dit_cfg1.0.json

# DiT baseline at CFG=1.5
python sample_fid.py --ckpt checkpoints/dit_baseline_400K.pt \
    --cfg 1.5 --steps 250 --sampler ddpm --num_samples 50000 \
    --output_dir samples/dit_cfg1.5/
python eval_precision_recall.py --generated_dir samples/dit_cfg1.5/ \
    --reference_dir data/imagenet/ref_batch/ --output pr_dit_cfg1.5.json
```

**Expected results:**
| Model | CFG | Precision | Recall | FID |
|-------|-----|-----------|--------|-----|
| RDD DPM-2 | 1.0 | ~0.70 | ~0.65 | 7.06 |
| RDD DPM-2 | 1.5 | ~0.83 | ~0.56 | 2.62 |
| DiT baseline | 1.0 | ~0.55 | ~0.68 | 19.78 |
| DiT baseline | 1.5 | ~0.70 | ~0.60 | 10.26 |

**Key prediction:** RDD's Recall should be comparable to or slightly lower than DiT's (due to CFG, not bottleneck). If RDD Recall at CFG=1.0 is comparable to DiT at CFG=1.0, the bottleneck is NOT hurting diversity.

**Hardware:** 8×A100, ~8 hours (4 sample generations + 4 P/R evaluations).

---

### 46. Stage-Dependent Unconditional Rate Ablation (M4: CFG Optimization)
**Goal:** Test whether stage-dependent unconditional dropping rates improve CFG quality over the uniform 10% baseline.

**Protocol:**
```bash
# Train with stage-dependent rates (200K steps for quick comparison)
# Variant A: 5% at 8×8, 10% at 16×16, 15% at 32×32
python train_unified.py --model XL --total_steps 200000 \
    --uncond_rates "0.05,0.10,0.15" --name uncond_stage_dependent

# Variant B: 0% at 8×8, 10% at 16×16, 20% at 32×32 (extreme)
python train_unified.py --model XL --total_steps 200000 \
    --uncond_rates "0.00,0.10,0.20" --name uncond_stage_extreme

# Baseline: uniform 10% (existing model at 200K)
# Compare CFG sweep for each at CFG = {1.0, 1.25, 1.5, 1.75, 2.0}
```

**Expected results:**
- Stage-dependent rates should improve optimal CFG quality slightly (~0.1-0.3 FID at best CFG)
- The improvement comes from better calibrated conditional/unconditional gap at the fine stage
- 0% at 8×8 likely safe since class signal is overwhelmed by noise there anyway

**Hardware:** 8×A100, ~3 days (2 × 200K-step trainings).

---

### 47. Score Field Consistency with Linear Drift (M3: Fighting the Assumption)
**Goal:** Analyze whether the trained model's predicted score/velocity is consistent with the linear drift assumption or "fighting" against it.

**Rationale:** If the linear drift assumption constrains expressivity, the model's predictions should show systematic deviations from what the linear assumption predicts—indicating the model is trying to express non-linear dynamics within the linear framework.

**Protocol:**
```python
# For a batch of images, compare:
# 1. The model's predicted effective noise eps_theta(x_t, t)
# 2. The "ideal" effective noise if drift were exactly linear: eps + SNR*(t-Tc)/DeltaT * d^c
# The residual (model prediction - linear prediction) reveals non-linear corrections

python analyze_score_consistency.py \
    --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --data_dir data/imagenet/val \
    --num_images 1000 \
    --timesteps "10,50,100,150,200,250,300" \
    --output score_consistency_analysis.json

# Key metrics:
# - ||eps_theta - eps_linear|| / ||eps_linear|| at each timestep
# - Spatial distribution of residuals (uniform vs structured)
# - Whether residuals correlate with image complexity
```

**Expected results:**
- Small residuals at most timesteps (model is consistent with linear assumption)
- Possibly larger residuals near stage boundaries (where the assumption is most constraining)
- If residuals are large and structured: the linear assumption IS constraining expressivity

**Hardware:** 1×A100, ~2 hours.

---

### 48. Failure Case Analysis (m4: Worst Samples)
**Goal:** Identify and analyze the worst-quality RDD samples to check for systematic failure modes (cross-resolution inconsistency, boundary artifacts, etc.).

**Protocol:**
```bash
# Generate 10K samples with per-sample quality scores
python generate_with_quality_scores.py \
    --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --cfg 1.5 --steps 250 --sampler dpm2 \
    --num_samples 10000 --output_dir samples/quality_analysis/

# Rank by LPIPS/CLIP score, examine bottom 100
python analyze_worst_samples.py \
    --samples_dir samples/quality_analysis/ \
    --num_worst 100 \
    --output worst_samples_analysis.json
```

**Key questions:**
- Do worst samples show artifacts at resolution boundaries (t=300, t=600)?
- Is there coarse/fine inconsistency (fine details contradict coarse structure)?
- Are worst samples concentrated in specific classes?

**Hardware:** 8×A100, ~2 hours + manual inspection.

---

## Priority 1: Important (Strengthens Paper)

### 49. Reference FID Statistics Specification (m3: Which Inception Stats?)
**Goal:** Clarify and document which pre-computed Inception statistics are used for FID computation.

**Protocol:**
```bash
# Verify: are we using ADM's reference stats or custom?
python verify_fid_reference.py --reference_file data/imagenet/VIRTUAL_imagenet256_labeled.npz
# This should be the standard ADM/guided-diffusion reference (Dhariwal & Nichol, 2021)
# containing mu and sigma computed from 50K real ImageNet 256×256 validation images
# using the standard Inception-v3 feature extractor (pool3 layer, 2048-dim)
```

**Deliverable:** Explicit statement in paper: "FID computed against ADM reference statistics (Dhariwal & Nichol, 2021) using 50K generated samples and the standard Inception-v3 pool3 features."

**Hardware:** CPU, trivial.

---

## Updated Summary Table (All Rounds Including R5 and R6)

| # | Experiment | Priority | GPU-Hours | Status | Round |
|---|-----------|----------|-----------|--------|-------|
| 1 | Convergence curves | P1 | 36 | Need to run | R1 |
| 2 | Wall-clock timing | P1 | 1 | Need to run | R1 |
| 3 | Published comparison | P1 | 0 | Need to look up | R1 |
| 4 | FM validation | P2 | 4 | Training (305K/400K) | R1 |
| 5 | CFG=1.0 ablation | P2 | 48 | Script ready | R1 |
| 6 | Extended training | P2 | 240 | Need to start | R1 |
| 7 | Alt stage split | P3 | 4 | Training (340K/400K) | R1 |
| 8 | Posterior consistency | P0 | 0.1 | OVERDUE | R1 |
| 9 | Drift stability | P3 | 0.1 | Need to run | R1 |
| 10 | Downsample ablation | P3 | 72 | Need to run | R1 |
| 11 | σ_h sweep (inference) | P0 | 12 | Need to run | R2 |
| 12 | Cascade no-drift ablation | P0 | 120 | Need to run | R2 |
| 13 | Convergence + asymptote | P0 | 168 | Need to run | R2 |
| 14 | Training with σ_h > 0 | P1 | 96 | Need to run | R2 |
| 15 | Drift magnitude stats | P1 | 1 | Need to run | R2 |
| 16 | Error accumulation | P1 | 72 | Need to run | R2 |
| 17 | SNR at boundaries | P1 | 0.01 | Need to run | R2 |
| 18 | γ sign verification | P1 | 0.1 | Need to run | R2 |
| 19 | FM velocity continuity | P2 | 2 | Need to run | R2 |
| 20 | Projection verification | P3 | 0.01 | Need to run | R2 |
| 21 | γ=0 ablation (inference) | P0 | 8 | Need to run | R3 |
| 22 | Drift-to-noise ratio plot | P0 | 0.5 | Need to run | R3 |
| 23 | Per-stage loss decomposition | P0 | 1 | Need to run | R3 |
| 24 | Monte Carlo posterior (single-step) | P0 | 0.01 | Need to run | R3 |
| 25 | Random-init ablation | P0 | 2 | Need to run | R3 |
| 26 | Per-stage loss curves over training | P0 | 8 | Need to run | R3 |
| 27 | Per-class FID analysis | P0 | 4 | Need to run | R4 |
| 28 | Stage split TRAINING ablation | P0 | 120 | Need to run | R4 |
| 29 | Adaptive patch size baseline | P0 | 120 | Need to run | R4 |
| 30 | Honest wall-clock at 125/250 NFE | P0 | 2 | Need to run | R4 |
| 31 | Reduced noise filling experiment | P0 | 8 | Need to run | R4 |
| 32 | DPM-Solver drift overhead | P1 | 0.5 | Need to run | R4 |
| 33 | Cosine schedule compatibility | P1 | 36 | Need to run | R4 |
| 34 | Precision/Recall for main results | P1 | 2 | Need to run | R4 |
| 35 | Multi-seed variance (3 seeds) | P1 | 6 | Need to run | R4 |
| **36** | **Progressive training baseline** | **P0** | **120** | **NEW - R5** | **R5** |
| **37** | **DDIM η sweep** | **P0** | **10** | **NEW - R5** | **R5** |
| **38** | **P/R at DPM-Solver-2 vs DDPM** | **P0** | **4** | **NEW - R5** | **R5** |
| **39** | **Truncated sampling (skip coarsest)** | **P1** | **2** | **NEW - R5** | **R5** |
| **40** | **FLOP-normalized convergence curve** | **P0** | **0** | **NEW - R5** | **R5** |
| **41** | **Reference implementation + test cases** | **P0** | **0.5** | **NEW - R6** | **R6** |
| **42** | **Per-stage drift magnitude** | **P0** | **1** | **NEW - R6** | **R6** |
| **43** | **2-stage model comparison** | **P0** | **120** | **NEW - R6** | **R6** |
| **44** | **Sequential contribution decomposition** | **P0** | **4** | **NEW - R6** | **R6** |
| **45** | **Full P/R at CFG=1.0 and 1.5** | **P0** | **8** | **NEW - R6** | **R6** |
| **46** | **Stage-dependent uncond rate ablation** | **P1** | **72** | **NEW - R6** | **R6** |
| **47** | **Score field linear drift consistency** | **P1** | **2** | **NEW - R6** | **R6** |
| **48** | **Failure case analysis** | **P1** | **2** | **NEW - R6** | **R6** |
| **49** | **Reference FID statistics spec** | **P1** | **0** | **NEW - R6** | **R6** |

**Total estimated (all rounds R1-R6):** ~1,545 GPU-hours
**Critical path (R5+R6 new):** #40 (replot, immediate) → #41, #42 (CPU, 1h) → #37, #38, #44, #45 (inference, ~26h) → #36, #43 (training, 5 days each)
**Quick wins (R5+R6):** #40 (0h), #41 (0.5h CPU), #42 (1h CPU), #49 (0h), #44 (4h inference), #38 (4h inference)

## NEW EXPERIMENTS FROM ROUND 7 REVIEW

### Priority 1: Addresses Core Review Concerns

### 50. Complexity Comparison Benchmark (M1: RDD vs Cascade Complexity)
**Goal:** Provide an honest accounting of implementation complexity: lines of code, number of hyperparameters, training time overhead, inference latency overhead for RDD vs. a naive 3-model cascade.

**Protocol:**
```bash
# Count lines of code modified vs standard DiT training
diff --stat train_standard_dit.py train_rdd_unified.py
diff --stat sample_standard_dit.py sample_rdd_unified.py

# Measure training overhead: time per training step with/without drift computation
python benchmark_training_step.py --model XL --with_drift --num_iters 100
python benchmark_training_step.py --model XL --no_drift --num_iters 100

# Document all hyperparameters unique to RDD:
# T_1, T_2, s_1, s_2, step_allocation (derived), gamma (derived)
# vs cascade: per-stage noise level, conditioning mechanism, per-stage duration, per-stage LR
```

**Output:** Table comparing RDD vs cascade across: LoC modified, hyperparameters, checkpoint count, training overhead, inference overhead.

**Hardware:** CPU analysis + 1h A100 for timing.

---

### 51. γ Correction Magnitude vs Step Size Plot (M2: Self-Referential Drift)
**Goal:** Empirically confirm that γ correction matters more for larger steps (DPM-Solver) than smaller steps (DDPM), validating the "inter-timestep drift change" explanation.

**Protocol:**
```python
import numpy as np
import matplotlib.pyplot as plt

# For each step size Delta_t in [1, 5, 10, 25, 50, 100, 125, 250]:
#   Compute |gamma_{s,t}^c| for representative (s,t) pairs within each stage
#   Plot |gamma| vs Delta_t
#   Also plot: FID degradation from gamma=0 vs step size (requires inference runs)

# Quick version (CPU only): compute gamma analytically for all step sizes
python plot_gamma_vs_stepsize.py \
    --schedule linear --beta_start 0.0001 --beta_end 0.02 \
    --stage_boundaries "0,300,600,1000" \
    --step_sizes "1,5,10,25,50,100,125,250" \
    --output gamma_vs_stepsize.pdf

# Extended version (GPU): measure FID at different NFEs with/without gamma
for nfe in 250 125 65 35; do
    python sample_fid_unified.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
        --cfg 1.5 --steps ${nfe} --sampler dpm2 --num_samples 50000 --gamma_zero
done
```

**Expected result:** |γ| grows approximately linearly with step size. FID degradation from γ=0 is minimal at 250 NFE but severe at 35 NFE.

**Hardware:** CPU for plot (10min); 8×A100, ~16h for FID measurements at multiple NFEs.

---

### 52. d^c ≈ 0 Class Analysis (M4: Which Classes Get No Benefit?)
**Goal:** Identify ImageNet classes where ||d^c|| ≈ 0 (piecewise-constant images) and verify RDD degenerates to standard DDPM quality for these.

**Protocol:**
```python
# Compute per-class mean ||d^c|| over validation set
python analyze_drift_per_class.py --data_dir data/imagenet/val/ \
    --num_images_per_class 50 \
    --output per_class_drift_magnitude.csv

# Identify bottom 50 classes (lowest drift) and top 50 (highest drift)
# Generate samples for these 100 classes with both RDD and baseline DiT
# Compare per-class FID/IS

python sample_per_class_subset.py --ckpt checkpoints/rdd_unified_xl_400K.pt \
    --classes_file low_drift_classes.txt --samples_per_class 100
python sample_per_class_subset.py --ckpt checkpoints/dit_baseline_400K.pt \
    --classes_file low_drift_classes.txt --samples_per_class 100
```

**Expected result:** Low-drift classes (solid colors, smooth gradients) show similar FID for RDD and DiT. High-drift classes (textures, patterns, text) show significant RDD improvement.

**Hardware:** CPU for analysis (1h); 8×A100, ~4h for targeted generation.

---

### 53. Continuous-Blur Variant Feasibility (M4: Continuous Resolution)
**Goal:** Test whether a Gaussian-blur-based "soft resolution" variant can work with numerical ODE integration, as a relaxation of the discrete-stage limitation.

**Protocol:**
```bash
# Implement continuous blur variant:
# d(t) = GaussianBlur(x_0, sigma=sigma(t)) - x_0
# where sigma(t) increases linearly from 0 to sigma_max over [0,T]
# Use numerical ODE solver (RK45) for sampling since no closed-form posterior

python train_continuous_blur.py --model XL --sigma_max 4.0 \
    --total_steps 100000 --sampler ode_rk45 --name continuous_blur_feasibility

python sample_fid_continuous_blur.py --ckpt checkpoints/continuous_blur_100K.pt \
    --cfg 1.0 --ode_steps 250 --num_samples 50000
```

**Expected result:** This variant loses the exact posterior advantage but may still converge faster than standard DDPM due to the frequency curriculum effect. FID likely worse than discrete RDD but better than baseline.

**Hardware:** 8×A100, ~2 days (short training for feasibility).

---

### Priority 2: Nice-to-Have (Strengthens Framework Understanding)

### 54 (renumbered from R8). "Minimum Viable" Experiment Set
**Goal:** Complete the absolute minimum experiments for submission readiness.
**Components:** Convergence curves (#1) + Wall-clock timing (#2/#30) + Adaptive patch (#29) + Table 8 (#3)
**This is a meta-experiment aggregating existing items. See individual entries for details.**
**Total cost:** ~160 GPU-hours. Deadline: Aug 20, 2026.

---

## NEW EXPERIMENTS FROM ROUND 8 REVIEW

### Priority 0: SUBMISSION-BLOCKING

### 55. Per-Class d^c=0 Analysis (M4/m4: Improvement Concentrated on Textured Images)
**Goal:** Validate that RDD's improvement over baseline is concentrated on classes with high-frequency content (high ||d^c||), while classes with low ||d^c|| see no benefit.

**Protocol:**
```python
# 1. Compute per-class drift statistics (reuse #38 results)
# 2. Bin classes into quartiles by ||d^c||
# 3. For each quartile: compute mean per-class FID for RDD and DiT baseline
# 4. Plot: per-quartile FID improvement (RDD vs DiT) as function of drift magnitude

python per_class_drift_improvement.py \
    --rdd_samples_dir samples/rdd_per_class/ \
    --dit_samples_dir samples/dit_per_class/ \
    --drift_stats per_class_drift_magnitude.csv \
    --output drift_quartile_improvement.pdf
```

**Expected result:**
- Q1 (lowest drift): RDD improvement ≈ 0 (same as standard DDPM)
- Q4 (highest drift): RDD improvement is largest (frequency-decomposition curriculum helps most)
- This validates the mechanism and identifies the regime where RDD provides value.

**Hardware:** CPU analysis (2h) + reuses generation from #52.

---

### 56. Latent-Space Feasibility Test (M3: Impact Assessment)
**Goal:** Quick feasibility check: can RDD operate in the latent space of a pre-trained VAE?

**Protocol:**
```bash
# Use Stable Diffusion's VAE encoder to get latent representations
# Apply RDD framework in latent space (32×32 latent → 16×16 → 8×8)
# Train for 100K steps only (feasibility, not SOTA)

python train_rdd_latent.py --model DiT-L/2 \
    --vae_ckpt stabilityai/sd-vae-ft-mse \
    --latent_size 32 --stages "32,16,8" \
    --total_steps 100000 --name rdd_latent_feasibility

python sample_fid_latent.py --ckpt checkpoints/rdd_latent_feasibility_100K.pt \
    --vae_ckpt stabilityai/sd-vae-ft-mse \
    --cfg 1.5 --steps 250 --num_samples 50000
```

**Expected result:** The framework should work mechanically (latent spatial dimensions support DOWN/UP). Quality at 100K steps will be poor but should show the same convergence advantage pattern.

**Note:** This is P3 priority — nice for a future version but NOT required for NeurIPS 2026 submission.

**Hardware:** 8×A100, ~1.5 days.

---

## FINAL SUMMARY TABLE (All Rounds R1–R8)

| # | Experiment | Priority | GPU-Hours | Status | Round |
|---|-----------|----------|-----------|--------|-------|
| 1 | Convergence curves | P0 | 36 | **SUBMISSION-BLOCKING** | R1 |
| 2 | Wall-clock timing | P0 | 1 | **SUBMISSION-BLOCKING** | R1 |
| 3 | Published comparison (Table 8) | P0 | 0 | **SUBMISSION-BLOCKING** | R1 |
| 4 | FM validation | REMOVED | — | FM removed from contributions | R1 |
| 5 | CFG=1.0 ablation | P2 | 48 | Script ready | R1 |
| 6 | Extended training | P2 | 240 | Need to start | R1 |
| 7 | Alt stage split | P3 | 4 | Training (340K/400K) | R1 |
| 8 | Posterior consistency | P0 | 0.1 | **DONE** ✅ | R1 |
| 9 | Drift stability | P3 | 0.1 | Need to run | R1 |
| 10 | Downsample ablation | P3 | 72 | Need to run | R1 |
| 11 | σ_h sweep (inference) | P1 | 12 | Need to run | R2 |
| 12 | Cascade no-drift ablation | P1 | 120 | Need to run | R2 |
| 13 | Convergence + asymptote | P0 | 168 | Subsumes #1 | R2 |
| 14 | Training with σ_h > 0 | P1 | 96 | Need to run | R2 |
| 15 | Drift magnitude stats | P1 | 1 | **DONE (partial)** ✅ | R2 |
| 16 | Error accumulation | P2 | 72 | Need to run | R2 |
| 17 | SNR at boundaries | P1 | 0.01 | **DONE** ✅ | R2 |
| 18 | γ sign verification | P1 | 0.1 | **DONE** ✅ | R2 |
| 19 | FM velocity continuity | REMOVED | — | FM removed from contributions | R2 |
| 20 | Projection verification | P3 | 0.01 | **DONE** ✅ | R2 |
| 21 | γ=0 ablation (inference) | P0 | 8 | **DONE** ✅ | R3 |
| 22 | Drift-to-noise ratio plot | P1 | 0.5 | Need to run | R3 |
| 23 | Per-stage loss decomposition | P1 | 1 | Need to run | R3 |
| 24 | Monte Carlo posterior (single-step) | P0 | 0.01 | **DONE** ✅ (with #8) | R3 |
| 25 | Random-init ablation | P1 | 2 | Need to run | R3 |
| 26 | Per-stage loss curves over training | P1 | 8 | Need to run | R3 |
| 27 | Per-class FID analysis | P1 | 4 | Need to run | R4 |
| 28 | Stage split TRAINING ablation | P1 | 120 | Need to run | R4 |
| 29 | Adaptive patch size baseline | P0 | 120 | **SUBMISSION-BLOCKING** | R4 |
| 30 | Honest wall-clock at 125/250 NFE | P0 | 2 | **SUBMISSION-BLOCKING** (= #2) | R4 |
| 31 | Reduced noise filling experiment | P1 | 8 | Need to run | R4 |
| 32 | DPM-Solver drift overhead | P2 | 0.5 | Need to run | R4 |
| 33 | Cosine schedule compatibility | P2 | 36 | Need to run | R4 |
| 34 | Precision/Recall for main results | P1 | 2 | Need to run | R4 |
| 35 | Multi-seed variance (3 seeds) | P1 | 6 | Need to run | R4 |
| 36 | Progressive training baseline | P0 | 120 | Need to run | R5 |
| 37 | DDIM η sweep | P0 | 10 | Need to run | R5 |
| 38 | P/R at DPM-Solver-2 vs DDPM | P0 | 4 | Need to run | R5 |
| 39 | Truncated sampling (skip coarsest) | P1 | 2 | Need to run | R5 |
| 40 | FLOP-normalized convergence curve | P0 | 0 | Need to run | R5 |
| 41 | Reference implementation + test cases | P0 | 0.5 | Need to run | R6 |
| 42 | Per-stage drift magnitude | P0 | 1 | Need to run | R6 |
| 43 | 2-stage model comparison | P0 | 120 | Need to run | R6 |
| 44 | Sequential contribution decomposition | P0 | 4 | Need to run | R6 |
| 45 | Full P/R at CFG=1.0 and 1.5 | P0 | 8 | Need to run | R6 |
| 46 | Stage-dependent uncond rate ablation | P1 | 72 | Need to run | R6 |
| 47 | Score field linear drift consistency | P1 | 2 | Need to run | R6 |
| 48 | Failure case analysis | P1 | 2 | Need to run | R6 |
| 49 | Reference FID statistics spec | P1 | 0 | Need to run | R6 |
| 50 | Complexity comparison benchmark | P2 | 1 | Need to run | R7 |
| 51 | γ magnitude vs step size plot | P1 | 16 | Need to run | R7 |
| 52 | d^c≈0 class analysis | P2 | 5 | Need to run | R7 |
| 53 | Continuous-blur variant feasibility | P3 | 48 | Need to run | R7 |
| 54 | "Minimum viable" meta-experiment | P0 | ~160 | Need to run | R8 |
| 55 | Per-class d^c=0 improvement analysis | P1 | 2 | Need to run | R8 |
| 56 | Latent-space feasibility test | P3 | 36 | Need to run | R8 |

---

## SUBMISSION-BLOCKING EXPERIMENTS (Must Complete Before NeurIPS 2026)

| # | Experiment | Deadline | Cost | Dependencies |
|---|-----------|----------|------|--------------|
| 1/13 | Convergence curves + asymptote | Aug 13, 2026 | 36-168h | Checkpoints exist |
| 2/30 | Wall-clock timing | Aug 8, 2026 | 2h | Code ready |
| 3 | Table 8 literature lookup | Aug 7, 2026 | 0 | None |
| 29 | Adaptive patch baseline | Aug 20, 2026 | 120h | Need implementation |
| — | Remove FM from contributions | Aug 7, 2026 | 0 | Decision only |
| — | Remove all TODO sections | Aug 14, 2026 | 0 | Writing only |

**Total submission-blocking compute:** ~160-290 GPU-hours (depending on convergence curve depth)
**Critical path:** Table 8 (Aug 7) → Timing (Aug 8) → Convergence (Aug 13) → TODOs removed (Aug 14) → Adaptive patch (Aug 20)
**If any deadline missed:** Target shifts to ICLR 2027.

---

## Completed Experiments Summary

| # | Experiment | Result | Round Completed |
|---|-----------|--------|-----------------|
| 8/24 | Posterior consistency (Monte Carlo) | Mean error < 0.005, variance error < 0.01 ✅ | R7 |
| 15 | Drift magnitude stats (partial, 1K imgs) | Mean ||d^c|| ≈ 0.95, std ≈ 0.25 | R7 |
| 17 | SNR at boundaries | t=300: SNR=32 (15dB), t=600: SNR=2.3 (3.6dB) | R7 |
| 18 | γ sign verification | γ < 0 in reverse direction ✅ | R7 |
| 20 | Projection verification | P²=P to machine precision ✅ | R7 |
| 21 | γ=0 ablation | DDPM: 3.54→3.82; DPM-2: 2.62→3.15 ✅ | R7 |

**Total completed: 7/56 = 12.5%**
**Submission-blocking completed: 0/5 = 0%**

---

**Grand total estimated compute (all rounds R1-R8):** ~1,750 GPU-hours
**Submission-blocking compute:** ~160 GPU-hours (achievable in 2 weeks with dedicated 8×A100)
**Realistic submission target:** NeurIPS 2026 (if deadlines met) or ICLR 2027 (fallback)
