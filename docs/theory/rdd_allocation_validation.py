#!/usr/bin/env python3
"""
Numerical Validation of RDD Optimal Allocation Theory
=====================================================

This script validates the key theoretical predictions for RDD-FM (Rectified Denoising Diffusion
Flow Matching) ODE solvers:

Theory Summary:
--------------
RDD-FM velocity field: v(x, t) = (x₁ - ε) + (1-2t)·d
- This is LINEAR in t, so Heun (2nd-order) should be EXACT (one step suffices)
- Euler has local truncation error O(h²), global error O(h) = O(1/N)
- For imperfect models with nonlinear residual, Heun error is O(h³), Euler is O(h²)

Optimal allocation predicts that steps should be distributed across stages
proportionally to stage error contributions, outperforming uniform allocation.

Exact ODE path for single-stage RDD-FM:
  x(t) = t·x₁ + (1-t)·ε + t(1-t)·d
  x(0) = ε, x(1) = x₁
  dx/dt = (x₁ - ε) + (1-2t)·d  ✓

For multistage (K stages), each stage k has its own drift d_k and operates
on a frequency band. The overall trajectory concatenates stage solutions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============================================================
# Core ODE Solvers
# ============================================================

def euler_step(v_fn, x, t, h):
    """Single Euler step: x_{n+1} = x_n + h * v(x_n, t_n)"""
    return x + h * v_fn(x, t)


def heun_step(v_fn, x, t, h):
    """Single Heun step (explicit trapezoidal):
    k1 = v(x_n, t_n)
    x_pred = x_n + h * k1
    k2 = v(x_pred, t_{n+1})
    x_{n+1} = x_n + h/2 * (k1 + k2)
    
    For LINEAR velocity fields v(x,t) = a(t)*x + b(t), Heun is exact
    when v is linear in t (since it integrates quadratics exactly in time).
    """
    k1 = v_fn(x, t)
    x_pred = x + h * k1
    k2 = v_fn(x_pred, t + h)
    return x + (h / 2.0) * (k1 + k2)


def solve_ode(v_fn, x0, t_start, t_end, n_steps, method='euler'):
    """Solve ODE dx/dt = v(x,t) from t_start to t_end with n_steps."""
    h = (t_end - t_start) / n_steps
    x = x0.copy()
    t = t_start
    step_fn = euler_step if method == 'euler' else heun_step
    for _ in range(n_steps):
        x = step_fn(v_fn, x, t, h)
        t += h
    return x


# ============================================================
# RDD-FM Velocity Field (Perfect Model)
# ============================================================

def make_rdd_velocity(x1, eps, d):
    """Create the exact RDD-FM velocity field.
    
    v(x, t) = (x₁ - ε) + (1 - 2t)·d
    
    Note: This is INDEPENDENT of x (linear ODE with no x-dependence),
    and LINEAR in t. Heun integrates this exactly.
    """
    drift = x1 - eps  # constant part
    def v_fn(x, t):
        return drift + (1.0 - 2.0 * t) * d
    return v_fn


def exact_solution(x1, eps, d, t):
    """Exact ODE path: x(t) = t·x₁ + (1-t)·ε + t(1-t)·d"""
    return t * x1 + (1.0 - t) * eps + t * (1.0 - t) * d


# ============================================================
# Imperfect Model (with nonlinear perturbation)
# ============================================================

def make_imperfect_velocity(x1, eps, d, alpha=0.1):
    """Create an imperfect velocity field with nonlinear-in-t perturbation.
    
    v_θ(x, t) = (x₁ - ε) + (1-2t)·d + α·(t² - t)·d + α·(t³ - t²)·d₂
    
    The polynomial terms (t², t³) are genuinely nonlinear in t, NOT periodic,
    so neither Euler nor Heun can integrate them exactly:
    - Euler local error is O(h²) → global O(h) = O(1/N)
    - Heun local error is O(h³) → global O(h²) = O(1/N²)
    
    We use (t²-t) so the integral over [0,1] is nonzero = -1/6,
    and (t³-t²) with integral = -1/12.
    """
    drift = x1 - eps
    # Use a second random direction for richer structure
    rng = np.random.default_rng(123)
    d2 = rng.standard_normal(d.shape) * np.linalg.norm(d) / np.sqrt(len(d))
    
    def v_fn(x, t):
        # Base linear velocity + nonlinear-in-t perturbation
        return (drift + (1.0 - 2.0 * t) * d 
                + alpha * (t**2 - t) * d 
                + alpha * (t**3 - t**2) * d2)
    return v_fn, d2


def compute_imperfect_reference(x1, eps, d, d2, alpha=0.1):
    """Compute reference endpoint by analytical integration.
    
    x(1) = ε + ∫₀¹ v(x,t) dt
         = ε + (x₁-ε)·1 + d·∫₀¹(1-2t)dt + α·d·∫₀¹(t²-t)dt + α·d₂·∫₀¹(t³-t²)dt
    
    Integrals:
    - ∫₀¹(1-2t)dt = [t-t²]₀¹ = 0
    - ∫₀¹(t²-t)dt = [t³/3-t²/2]₀¹ = 1/3-1/2 = -1/6
    - ∫₀¹(t³-t²)dt = [t⁴/4-t³/3]₀¹ = 1/4-1/3 = -1/12
    
    x(1) = x₁ + α·(-1/6)·d + α·(-1/12)·d₂
    """
    return x1 + alpha * (-1.0/6.0) * d + alpha * (-1.0/12.0) * d2


# ============================================================
# Multistage RDD Setup
# ============================================================

def make_multistage_problem(dim=100, K=3, seed=42):
    """Create a K-stage RDD problem with different drift magnitudes per stage.
    
    Stage k operates on frequency band, with drift d_k having different magnitudes.
    Higher-frequency stages typically have larger drift (more curvature).
    
    Returns:
        x1: target (dim,)
        eps: noise (dim,)
        drifts: list of K drift vectors (dim,)
        stage_difficulties: relative error contribution of each stage
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(dim)
    eps = rng.standard_normal(dim)
    
    # Drift magnitudes increase with stage (higher freq = more curvature)
    drift_scales = [1.0, 3.0, 8.0]  # stage difficulties
    drifts = [scale * rng.standard_normal(dim) for scale in drift_scales]
    
    # Stage difficulties are proportional to ||d_k||²
    # (Euler error for stage k with n_k steps is ~ ||d_k||² / n_k)
    norms_sq = np.array([np.sum(d**2) for d in drifts])
    stage_difficulties = norms_sq / norms_sq.sum()
    
    return x1, eps, drifts, stage_difficulties


def solve_multistage(x1, eps, drifts, allocations, method='euler'):
    """Solve K-stage RDD ODE with given step allocations per stage.
    
    Each stage k: solve from t=0 to t=1 with velocity v_k(x,t).
    The output of stage k becomes input of stage k+1.
    
    For the linear case, each stage independently adds:
      x₁ - ε + 0 (drift integrates to zero over [0,1])
    But we track cumulative state.
    """
    K = len(drifts)
    x = eps.copy()
    
    for k in range(K):
        # Each stage velocity: v_k(x,t) = (x1 - eps)/K + (1-2t)*d_k
        # This ensures total drift sums correctly
        stage_drift = (x1 - eps) / K
        d_k = drifts[k]
        
        def v_fn(x, t, _drift=stage_drift, _d=d_k):
            return _drift + (1.0 - 2.0 * t) * _d
        
        n_k = max(1, int(allocations[k]))
        x = solve_ode(v_fn, x, 0.0, 1.0, n_k, method=method)
    
    return x


def compute_optimal_allocation(stage_difficulties, total_nfe, method='euler'):
    """Compute optimal step allocation across stages.
    
    For Euler (error ~ ||d_k||² / n_k):
        Optimal: n_k ∝ ||d_k|| (square root of difficulty)
        This minimizes total error = Σ ||d_k||² / n_k subject to Σ n_k = N
        
    By Lagrange multipliers: n_k* = N · ||d_k|| / Σ_j ||d_j||
    """
    # stage_difficulties ∝ ||d_k||², so ||d_k|| ∝ sqrt(stage_difficulties)
    weights = np.sqrt(stage_difficulties)
    weights /= weights.sum()
    allocation = weights * total_nfe
    # Ensure at least 1 step per stage
    allocation = np.maximum(allocation, 1.0)
    # Renormalize to total budget
    allocation = allocation * total_nfe / allocation.sum()
    return allocation


# ============================================================
# Validation Tests
# ============================================================

def test_heun_exactness(dim=100, seed=42):
    """Test 1: Heun should be exact for linear velocity fields.
    
    Since v(x,t) = (x₁-ε) + (1-2t)·d is linear in t and independent of x,
    a single Heun step should give the exact answer (up to floating point).
    """
    print("=" * 60)
    print("TEST 1: Heun Exactness for Linear Velocity")
    print("=" * 60)
    
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(dim)
    eps = rng.standard_normal(dim)
    d = rng.standard_normal(dim) * 5.0  # large drift to stress-test
    
    v_fn = make_rdd_velocity(x1, eps, d)
    true_endpoint = x1.copy()  # x(1) = x₁
    
    nfe_values = [1, 2, 5, 10, 20, 50]
    errors_heun = []
    errors_euler = []
    
    for n in nfe_values:
        # Heun
        x_heun = solve_ode(v_fn, eps.copy(), 0.0, 1.0, n, method='heun')
        err_heun = np.linalg.norm(x_heun - true_endpoint)
        errors_heun.append(err_heun)
        
        # Euler for comparison
        x_euler = solve_ode(v_fn, eps.copy(), 0.0, 1.0, n, method='euler')
        err_euler = np.linalg.norm(x_euler - true_endpoint)
        errors_euler.append(err_euler)
        
        print(f"  NFE={n:3d}: Heun error = {err_heun:.2e}, Euler error = {err_euler:.2e}")
    
    # Verify Heun is at machine precision (allow accumulation over many steps)
    assert max(errors_heun) < 1e-13, f"Heun not exact! Max error: {max(errors_heun):.2e}"
    print(f"\n  ✓ Heun achieves machine precision (max error: {max(errors_heun):.2e})")
    print(f"  ✓ Euler has O(1/N) error as expected")
    
    return nfe_values, errors_heun, errors_euler


def test_euler_error_scaling(dim=100, seed=42):
    """Test 2: Euler error should follow O(1/N) scaling.
    
    For the RDD velocity (linear in t), Euler local truncation error is O(h²),
    giving global error O(h) = O(1/N).
    
    On log-log plot: log(error) = -1 * log(N) + const → slope = -1
    """
    print("\n" + "=" * 60)
    print("TEST 2: Euler Error Scaling O(1/N)")
    print("=" * 60)
    
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(dim)
    eps = rng.standard_normal(dim)
    d = rng.standard_normal(dim) * 5.0
    
    v_fn = make_rdd_velocity(x1, eps, d)
    true_endpoint = x1.copy()
    
    nfe_values = [5, 10, 20, 50, 100, 200, 500]
    errors = []
    
    for n in nfe_values:
        x_euler = solve_ode(v_fn, eps.copy(), 0.0, 1.0, n, method='euler')
        err = np.linalg.norm(x_euler - true_endpoint)
        errors.append(err)
        print(f"  N={n:4d}: error = {err:.6e}")
    
    # Fit log-log slope
    log_n = np.log(np.array(nfe_values, dtype=float))
    log_err = np.log(np.array(errors))
    slope, intercept = np.polyfit(log_n, log_err, 1)
    
    print(f"\n  Log-log slope: {slope:.3f} (expected: -1.0)")
    assert abs(slope + 1.0) < 0.1, f"Slope {slope:.3f} deviates from -1.0"
    print(f"  ✓ Euler error follows O(1/N) scaling (slope = {slope:.3f})")
    
    return nfe_values, errors, slope


def test_optimal_vs_uniform(dim=100, seed=42):
    """Test 3: Optimal allocation should outperform uniform allocation.
    
    For K stages with varying difficulty (||d_k|| varies), the optimal allocation
    n_k ∝ ||d_k|| minimizes total error compared to uniform n_k = N/K.
    
    The theoretical improvement ratio is:
        Error_uniform / Error_optimal = (Σ||d_k||)² / (K · Σ||d_k||²)  ≥ 1
    """
    print("\n" + "=" * 60)
    print("TEST 3: Optimal vs Uniform Allocation")
    print("=" * 60)
    
    x1, eps, drifts, stage_difficulties = make_multistage_problem(dim=dim, K=3, seed=seed)
    true_endpoint = x1.copy()
    
    print(f"  Stage difficulties: {stage_difficulties}")
    print(f"  (Higher = more steps needed)")
    
    total_nfe_values = [12, 24, 48, 96, 192]
    errors_uniform = []
    errors_optimal = []
    
    K = len(drifts)
    
    for total_nfe in total_nfe_values:
        # Uniform allocation
        uniform_alloc = np.array([total_nfe / K] * K)
        x_uniform = solve_multistage(x1, eps, drifts, uniform_alloc, method='euler')
        err_uniform = np.linalg.norm(x_uniform - true_endpoint)
        errors_uniform.append(err_uniform)
        
        # Optimal allocation
        opt_alloc = compute_optimal_allocation(stage_difficulties, total_nfe)
        x_optimal = solve_multistage(x1, eps, drifts, opt_alloc, method='euler')
        err_optimal = np.linalg.norm(x_optimal - true_endpoint)
        errors_optimal.append(err_optimal)
        
        ratio = err_uniform / max(err_optimal, 1e-16)
        print(f"  NFE={total_nfe:4d}: uniform={err_uniform:.4e}, optimal={err_optimal:.4e}, "
              f"ratio={ratio:.2f}x")
    
    # Verify optimal is better
    for eu, eo in zip(errors_uniform, errors_optimal):
        assert eo <= eu * 1.01, "Optimal allocation should not be worse than uniform!"
    
    print(f"\n  ✓ Optimal allocation consistently outperforms uniform allocation")
    
    return total_nfe_values, errors_uniform, errors_optimal


def test_imperfect_model(dim=100, seed=42):
    """Test 4: With nonlinear perturbation, verify convergence orders.
    
    Imperfect velocity: v_θ(x,t) = (x₁-ε) + (1-2t)·d + α·(t²-t)·d + α·(t³-t²)·d₂
    
    The polynomial terms (t², t³) are genuinely nonlinear in t:
    - Euler: local O(h²), global O(h) = O(1/N)
    - Heun:  local O(h³), global O(h²) = O(1/N²)
    
    So on log-log: Euler slope = -1, Heun slope = -2
    """
    print("\n" + "=" * 60)
    print("TEST 4: Imperfect Model - Convergence Orders")
    print("=" * 60)
    
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(dim)
    eps = rng.standard_normal(dim)
    d = rng.standard_normal(dim) * 2.0
    alpha = 0.3  # perturbation strength
    
    v_fn, d2 = make_imperfect_velocity(x1, eps, d, alpha=alpha)
    
    # Compute analytical reference solution
    x_ref = compute_imperfect_reference(x1, eps, d, d2, alpha=alpha)
    
    nfe_values = [5, 10, 20, 50, 100, 200, 500]
    errors_euler = []
    errors_heun = []
    
    for n in nfe_values:
        x_e = solve_ode(v_fn, eps.copy(), 0.0, 1.0, n, method='euler')
        x_h = solve_ode(v_fn, eps.copy(), 0.0, 1.0, n, method='heun')
        
        err_e = np.linalg.norm(x_e - x_ref)
        err_h = np.linalg.norm(x_h - x_ref)
        errors_euler.append(err_e)
        errors_heun.append(err_h)
        
        print(f"  Steps={n:4d}: Euler err={err_e:.4e}, Heun err={err_h:.4e}")
    
    # Fit slopes
    log_n = np.log(np.array(nfe_values[1:], dtype=float))  # skip first for better fit
    
    log_err_e = np.log(np.array(errors_euler[1:]))
    slope_e, _ = np.polyfit(log_n, log_err_e, 1)
    
    log_err_h = np.log(np.array(errors_heun[1:]))
    slope_h, _ = np.polyfit(log_n, log_err_h, 1)
    
    print(f"\n  Euler slope: {slope_e:.2f} (expected: -1.0)")
    print(f"  Heun slope:  {slope_h:.2f} (expected: -2.0)")
    
    assert abs(slope_e + 1.0) < 0.2, f"Euler slope {slope_e:.2f} not ~-1"
    assert abs(slope_h + 2.0) < 0.4, f"Heun slope {slope_h:.2f} not ~-2"
    
    print(f"  ✓ Euler: O(1/N) convergence confirmed")
    print(f"  ✓ Heun: O(1/N²) convergence confirmed")
    
    return nfe_values, errors_euler, errors_heun, slope_e, slope_h


# ============================================================
# Plotting
# ============================================================

def plot_heun_exactness(nfe_values, errors_heun, errors_euler, save_dir):
    """Figure 1: Heun exactness for linear velocity field."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.semilogy(nfe_values, errors_euler, 'rs-', markersize=8, linewidth=2,
                label='Euler (global O(1/N))')
    ax.semilogy(nfe_values, [max(e, 1e-16) for e in errors_heun], 'bo-', 
                markersize=8, linewidth=2, label='Heun (exact for linear v)')
    
    # Machine precision line
    ax.axhline(y=1e-15, color='gray', linestyle='--', alpha=0.7, 
               label='Machine precision (~1e-15)')
    
    ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=12)
    ax.set_ylabel('L2 Error ||x_computed - x_true||', fontsize=12)
    ax.set_title('Heun Exactness for Linear RDD-FM Velocity Field\n'
                 r'$v(x,t) = (x_1 - \epsilon) + (1-2t) \cdot d$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-17, None)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_fig1_heun_exactness.png', dpi=150, 
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: validation_fig1_heun_exactness.png")


def plot_error_scaling(nfe_values, errors_euler, slope, 
                       nfe_heun, errors_heun, slope_h, save_dir):
    """Figure 2: Log-log error scaling for both methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    nfe_arr = np.array(nfe_values, dtype=float)
    
    # Euler data and fit
    ax.loglog(nfe_arr, errors_euler, 'rs-', markersize=8, linewidth=2,
              label=f'Euler (slope={slope:.2f}, theory=-1.0)')
    
    # Reference O(1/N) line
    ref_euler = errors_euler[2] * (nfe_arr[2] / nfe_arr)
    ax.loglog(nfe_arr, ref_euler, 'r--', alpha=0.5, linewidth=1.5,
              label=r'Reference $O(1/N)$')
    
    # Heun data (for imperfect model)
    if errors_heun is not None:
        nfe_h_arr = np.array(nfe_heun, dtype=float)
        ax.loglog(nfe_h_arr, errors_heun, 'bo-', markersize=8, linewidth=2,
                  label=f'Heun (slope={slope_h:.2f}, theory=-2.0)')
        ref_heun = errors_heun[2] * (nfe_h_arr[2] / nfe_h_arr)**2
        ax.loglog(nfe_h_arr, ref_heun, 'b--', alpha=0.5, linewidth=1.5,
                  label=r'Reference $O(1/N^2)$')
    
    ax.set_xlabel('Number of Steps (N)', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Error Scaling: Euler O(1/N) vs Heun O(1/N²)\n'
                 '(Imperfect model with nonlinear perturbation)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_fig2_error_scaling.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: validation_fig2_error_scaling.png")


def plot_optimal_vs_uniform(nfe_values, errors_uniform, errors_optimal, save_dir):
    """Figure 3: Optimal vs uniform allocation comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    nfe_arr = np.array(nfe_values, dtype=float)
    
    # Left: Error comparison
    ax1.loglog(nfe_arr, errors_uniform, 'rs-', markersize=8, linewidth=2,
               label='Uniform allocation (n_k = N/K)')
    ax1.loglog(nfe_arr, errors_optimal, 'bo-', markersize=8, linewidth=2,
               label=r'Optimal allocation (n_k $\propto$ ||d_k||)')
    
    ax1.set_xlabel('Total NFE Budget', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Multistage Error: Optimal vs Uniform', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right: Improvement ratio
    ratios = np.array(errors_uniform) / np.array(errors_optimal)
    ax2.bar(range(len(nfe_values)), ratios, color='steelblue', alpha=0.8)
    ax2.set_xticks(range(len(nfe_values)))
    ax2.set_xticklabels([str(n) for n in nfe_values])
    ax2.set_xlabel('Total NFE Budget', fontsize=12)
    ax2.set_ylabel('Error Ratio (Uniform / Optimal)', fontsize=12)
    ax2.set_title('Improvement Factor of Optimal Allocation', fontsize=13)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_fig3_optimal_vs_uniform.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: validation_fig3_optimal_vs_uniform.png")


def plot_imperfect_model(nfe_values, errors_euler, errors_heun, 
                         slope_e, slope_h, save_dir):
    """Figure 4: Imperfect model convergence comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    nfe_arr = np.array(nfe_values, dtype=float)
    # For Heun, NFE = 2 * steps (two function evaluations per step)
    nfe_heun_actual = 2.0 * nfe_arr
    
    # Left: Error vs number of steps
    ax1.loglog(nfe_arr, errors_euler, 'rs-', markersize=8, linewidth=2,
               label=f'Euler (slope={slope_e:.2f})')
    ax1.loglog(nfe_arr, errors_heun, 'bo-', markersize=8, linewidth=2,
               label=f'Heun (slope={slope_h:.2f})')
    
    # Reference lines
    ref_e = errors_euler[3] * (nfe_arr[3] / nfe_arr)**1
    ref_h = errors_heun[3] * (nfe_arr[3] / nfe_arr)**2
    ax1.loglog(nfe_arr, ref_e, 'r--', alpha=0.4, label=r'$O(N^{-1})$ reference')
    ax1.loglog(nfe_arr, ref_h, 'b--', alpha=0.4, label=r'$O(N^{-2})$ reference')
    
    ax1.set_xlabel('Number of Steps', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Convergence Order (Steps)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right: Error vs actual NFE (fair comparison)
    ax2.loglog(nfe_arr, errors_euler, 'rs-', markersize=8, linewidth=2,
               label='Euler (1 eval/step)')
    ax2.loglog(nfe_heun_actual, errors_heun, 'bo-', markersize=8, linewidth=2,
               label='Heun (2 evals/step)')
    
    ax2.set_xlabel('Actual NFE (Function Evaluations)', fontsize=12)
    ax2.set_ylabel('L2 Error', fontsize=12)
    ax2.set_title('Fair Comparison: Error vs Actual NFE\n'
                  '(Heun still wins due to quadratic convergence)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_fig4_imperfect_model.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: validation_fig4_imperfect_model.png")


# ============================================================
# Main
# ============================================================

def main():
    save_dir = '/home/tiger/multiscale_diffusion/research_experiments'
    
    print("=" * 64)
    print("  RDD Optimal Allocation Theory - Numerical Validation")
    print("=" * 64)
    print("  Validating:")
    print("  1. Heun exactness for linear velocity fields")
    print("  2. Euler O(1/N) error scaling")
    print("  3. Optimal > uniform allocation")
    print("  4. Heun O(1/N^2) vs Euler O(1/N) for nonlinear model")
    print("=" * 64 + "\n")
    
    # --- Test 1: Heun Exactness ---
    nfe1, err_heun1, err_euler1 = test_heun_exactness()
    
    # --- Test 2: Euler Error Scaling ---
    nfe2, err_euler2, slope2 = test_euler_error_scaling()
    
    # --- Test 3: Optimal vs Uniform ---
    nfe3, err_uniform3, err_optimal3 = test_optimal_vs_uniform()
    
    # --- Test 4: Imperfect Model ---
    nfe4, err_euler4, err_heun4, slope_e4, slope_h4 = test_imperfect_model()
    
    # --- Generate Figures ---
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    plot_heun_exactness(nfe1, err_heun1, err_euler1, save_dir)
    plot_error_scaling(nfe4, err_euler4, slope_e4, nfe4, err_heun4, slope_h4, save_dir)
    plot_optimal_vs_uniform(nfe3, err_uniform3, err_optimal3, save_dir)
    plot_imperfect_model(nfe4, err_euler4, err_heun4, slope_e4, slope_h4, save_dir)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)
    print(f"""
Summary of Validated Predictions:
---------------------------------
1. Heun is EXACT for linear RDD-FM velocity (error < 1e-12)
2. Euler error scales as O(1/N) with slope = {slope2:.3f}
3. Optimal allocation outperforms uniform by ~{np.mean(np.array(err_uniform3)/np.array(err_optimal3)):.1f}x
4. For imperfect models:
   - Euler convergence: O(N^({slope_e4:.2f})) ~ O(1/N)
   - Heun convergence:  O(N^({slope_h4:.2f})) ~ O(1/N^2)

Figures saved to: {save_dir}/
  - validation_fig1_heun_exactness.png
  - validation_fig2_error_scaling.png
  - validation_fig3_optimal_vs_uniform.png
  - validation_fig4_imperfect_model.png
""")


if __name__ == '__main__':
    main()
