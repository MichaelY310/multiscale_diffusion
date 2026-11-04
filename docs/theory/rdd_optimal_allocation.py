"""
RDD Optimal Stage Allocation: Theoretical Framework and Implementation
======================================================================

This script derives and validates the theoretically optimal allocation of:
  (A) Training stage lengths (timestep range per stage)
  (B) Inference sampling steps per stage (given a total NFE budget)

for RDD-FM (Resolution Drift Diffusion with Flow Matching).

Key insight vs SPEED:
  SPEED uses activation_time t*(ω) = 1/(1+√(δ/(P(ω)·(1+P(ω)-δ))))
  which fails for DiT VAE latents (A=13910 too large → all steps go to stage 1).
  
  Our approach instead derives allocation from:
  - Training: minimize total learning difficulty → stage_length_k ∝ √C_k
  - Inference: minimize discretization error → n_k ∝ L_k·√R_k (Euler)
              or exploit linear velocity theorem → 2K NFE exact (Heun)

Mathematical Foundation:
  RDD-FM forward: x_t = t·x₁ + (1-t)·ε + t(1-t)·d
  Velocity field:  u_t = (x₁ - ε) + (1-2t)·d  [linear in t!]
  Key property:   g'(t) = 1-2t is linear → Heun exact per stage

Author: Research Agent
Date: 2026-08-01
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os

OUTPUT_DIR = "/home/tiger/multiscale_diffusion/research_experiments"


# =============================================================================
# Part I: Power Spectrum Model and Band Energy Computation
# =============================================================================

def power_spectrum(omega: np.ndarray, A: float, beta: float) -> np.ndarray:
    """Radial power-law spectrum P(ω) = A·|ω|^(-β)."""
    omega = np.asarray(omega, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = A * np.abs(omega) ** (-beta)
    P = np.where(np.isfinite(P), P, 0.0)
    return P


def band_energy(A: float, beta: float, omega_min: float, omega_max: float) -> float:
    """
    Compute energy in frequency band [ω_min, ω_max]:
      E = ∫_{ω_min}^{ω_max} P(ω) dω = ∫ A·ω^(-β) dω
    
    For β ≠ 1:
      E = A/(1-β) · [ω_max^(1-β) - ω_min^(1-β)]
    For β = 1:
      E = A · [ln(ω_max) - ln(ω_min)]
    
    Note: ω_min must be > 0 for the integral to converge.
    """
    if omega_min <= 0:
        omega_min = 0.5  # Regularize: DC component excluded, start from 0.5
    
    if abs(beta - 1.0) < 1e-10:
        return A * (np.log(omega_max) - np.log(omega_min))
    else:
        exp = 1.0 - beta
        return A / exp * (omega_max**exp - omega_min**exp)


def band_energy_2d(A: float, beta: float, omega_min: float, omega_max: float) -> float:
    """
    2D radial band energy (integrating over annulus in 2D frequency space):
      E_2d = ∫_{ω_min}^{ω_max} 2π·ω · P(ω) dω = 2πA ∫ ω^(1-β) dω
    
    For β ≠ 2:
      E_2d = 2πA/(2-β) · [ω_max^(2-β) - ω_min^(2-β)]
    """
    if omega_min <= 0:
        omega_min = 0.5
    
    if abs(beta - 2.0) < 1e-10:
        return 2 * np.pi * A * (np.log(omega_max) - np.log(omega_min))
    else:
        exp = 2.0 - beta
        return 2 * np.pi * A / exp * (omega_max**exp - omega_min**exp)


# =============================================================================
# Part II: Training Stage Length Allocation Theory
# =============================================================================

def compute_stage_complexity(
    A: float, beta: float, 
    frequency_bands: List[Tuple[float, float]],
    use_2d: bool = True,
    drift_weights: Optional[np.ndarray] = None,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Compute complexity C_k for each stage.
    
    Theory:
    -------
    The training difficulty of stage k is determined by:
      1. Band energy E_k: how much signal power the model must learn
      2. Band width W_k: how many frequency components to resolve
      3. Drift magnitude ||d_k||: affects velocity field amplitude
    
    C_k = E_k × W_k × (1 + α·||d_k||²/||d_max||²)
    
    The √(band_energy × bandwidth) formulation comes from:
    - Variance of the velocity target: Var[u_t] ∝ E_k (energy)
    - Effective dimensionality: dim ∝ W_k (bandwidth)  
    - Total learning load: L_k ∝ √(E_k × W_k) (standard statistical learning theory)
    
    The optimal allocation minimizes:
      Σ_k L_k² / stage_length_k   subject to  Σ_k stage_length_k = T_total
    
    Solution (Cauchy-Schwarz):
      stage_length_k ∝ L_k = √C_k
    
    Args:
        A: Power-law amplitude
        beta: Power-law exponent
        frequency_bands: [(ω_min, ω_max)] for each stage
        use_2d: If True, use 2D radial integration (correct for images)
        drift_weights: Optional ||d_k||² for each stage (normalized)
        alpha: Weight for drift contribution
    
    Returns:
        C_k array (unnormalized complexity per stage)
    """
    K = len(frequency_bands)
    C = np.zeros(K)
    
    energy_fn = band_energy_2d if use_2d else band_energy
    
    for k, (w_min, w_max) in enumerate(frequency_bands):
        E_k = energy_fn(A, beta, w_min, w_max)
        W_k = w_max - w_min  # bandwidth
        C[k] = E_k * W_k
    
    # Drift-aware refinement
    if drift_weights is not None:
        drift_weights = np.asarray(drift_weights)
        d_max_sq = np.max(drift_weights)
        if d_max_sq > 0:
            C *= (1.0 + alpha * drift_weights / d_max_sq)
    
    return C


def optimal_training_allocation(
    A: float, beta: float,
    frequency_bands: List[Tuple[float, float]],
    T_total: int = 1000,
    use_2d: bool = True,
    drift_weights: Optional[np.ndarray] = None,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Compute optimal training stage lengths.
    
    Result: stage_length_k = T_total × √C_k / Σ_j √C_j
    
    Returns:
        Array of stage lengths (summing to T_total)
    """
    C = compute_stage_complexity(A, beta, frequency_bands, use_2d, drift_weights, alpha)
    sqrt_C = np.sqrt(C)
    stage_lengths = T_total * sqrt_C / np.sum(sqrt_C)
    return stage_lengths


# =============================================================================
# Part III: Inference Step Allocation Theory
# =============================================================================

def euler_optimal_allocation(
    stage_lengths_normalized: np.ndarray,
    drift_norms: np.ndarray,
    total_nfe: int,
    min_steps_per_stage: int = 1
) -> np.ndarray:
    """
    Optimal inference step allocation for Euler method.
    
    Theory:
    -------
    For Euler discretization of ODE dx/dt = v_θ(x,t):
      - Step size in stage k: h_k = L_k / n_k
      - Local truncation error: LTE_k = h_k²/2 · |v''(t)|
      - For RDD-FM: v(t) is linear in t (exactly)
        BUT the learned v_θ(t) has residual curvature R_k
      - Proxy for R_k: R_k ∝ ||d_k|| (drift magnitude drives model imperfection)
    
    Total error (sum of squared local errors):
      E = Σ_k (L_k²·R_k / n_k)
    
    Minimize E subject to Σ_k n_k = N:
      Lagrangian: ∂/∂n_k [Σ L_j²·R_j/n_j + λ(Σn_j - N)] = 0
      → n_k² = L_k²·R_k / λ
      → n_k ∝ L_k · √R_k
    
    With R_k ∝ ||d_k||:
      n_k ∝ L_k · ||d_k||^(1/2)
    
    For the coarsest stage (no drift, d=0): use R_k = small constant.
    
    Args:
        stage_lengths_normalized: L_k in [0,1] (fraction of total timestep range)
        drift_norms: ||d_k|| for each stage (can be 0 for coarsest)
        total_nfe: Total number of function evaluations
        min_steps_per_stage: Minimum steps per stage
    
    Returns:
        Array of integer steps per stage
    """
    K = len(stage_lengths_normalized)
    L = stage_lengths_normalized.copy()
    
    # Regularize drift norms: coarsest stage has no drift but still needs steps
    R = drift_norms.copy()
    R[R == 0] = np.min(R[R > 0]) * 0.1 if np.any(R > 0) else 1.0
    
    # Continuous optimal: n_k ∝ L_k · √R_k
    weights = L * np.sqrt(R)
    n_continuous = total_nfe * weights / np.sum(weights)
    
    # Round to integers with minimum constraint
    n_steps = np.maximum(np.round(n_continuous).astype(int), min_steps_per_stage)
    
    # Adjust to meet total NFE budget
    diff = total_nfe - np.sum(n_steps)
    if diff > 0:
        # Add extra steps to stages with highest weight
        indices = np.argsort(-weights)
        for i in range(int(diff)):
            n_steps[indices[i % K]] += 1
    elif diff < 0:
        # Remove steps from stages with lowest weight (respecting minimum)
        indices = np.argsort(weights)
        for i in range(int(-diff)):
            idx = indices[i % K]
            if n_steps[idx] > min_steps_per_stage:
                n_steps[idx] -= 1
    
    return n_steps


def heun_optimal_allocation(
    stage_lengths_normalized: np.ndarray,
    total_nfe: int,
    min_steps_per_stage: int = 1
) -> np.ndarray:
    """
    Optimal inference step allocation for Heun method.
    
    Theory:
    -------
    KEY THEOREM (Linear Velocity Theorem for RDD-FM):
      Within each stage, the true velocity field u_t = (x₁-ε) + (1-2t)·d
      is LINEAR in t. Heun's method (trapezoidal rule) is exact for linear ODEs.
      
      Therefore, with a perfect model v_θ = u_t:
        Each stage needs exactly 1 Heun step (= 2 NFE) → Total = 2K NFE
    
    For an imperfect model v_θ ≈ u_t + δv(t):
      Heun error is O(h³) per step (3rd order method)
      Total error in stage k: E_k ∝ L_k³/n_k² (since n_k steps of size h_k=L_k/n_k)
      → n_k ∝ L_k^(3/2) for Heun with imperfect model
    
    Each Heun step = 2 NFE, so actual NFE = 2·Σn_k.
    Given total_nfe budget: n_k_heun = total_nfe/2 allocated as n_k ∝ L_k^(3/2)
    
    Args:
        stage_lengths_normalized: L_k in [0,1]
        total_nfe: Total NFE budget (will use total_nfe/2 Heun steps)
        min_steps_per_stage: Minimum Heun steps per stage
    
    Returns:
        Array of Heun steps per stage (actual NFE = 2 × returned values)
    """
    K = len(stage_lengths_normalized)
    L = stage_lengths_normalized.copy()
    
    # Total Heun steps available
    total_heun_steps = total_nfe // 2
    
    # Weights: n_k ∝ L_k^(3/2)
    weights = L ** 1.5
    n_continuous = total_heun_steps * weights / np.sum(weights)
    
    # Round to integers
    n_steps = np.maximum(np.round(n_continuous).astype(int), min_steps_per_stage)
    
    # Adjust
    diff = total_heun_steps - np.sum(n_steps)
    if diff > 0:
        indices = np.argsort(-weights)
        for i in range(int(diff)):
            n_steps[indices[i % K]] += 1
    elif diff < 0:
        indices = np.argsort(weights)
        for i in range(int(-diff)):
            idx = indices[i % K]
            if n_steps[idx] > min_steps_per_stage:
                n_steps[idx] -= 1
    
    return n_steps


def quality_speed_curve(
    stage_lengths_normalized: np.ndarray,
    drift_norms: np.ndarray,
    nfe_range: np.ndarray,
    method: str = "euler"
) -> np.ndarray:
    """
    Compute theoretical error as a function of total NFE.
    
    Theory:
    -------
    Euler total error:  E(N) ∝ Σ_k L_k²·R_k/n_k where Σn_k=N
                        With optimal allocation: E*(N) = (Σ_k √(L_k²·R_k))² / N
    
    Heun total error:   E(N) ∝ Σ_k L_k³/n_k² where Σ2n_k=N
                        With optimal allocation: E*(N) ∝ (Σ_k (L_k³)^(1/3))³ / (N/2)²
    
    Returns:
        Array of theoretical error values for each NFE in nfe_range
    """
    L = stage_lengths_normalized
    R = drift_norms.copy()
    R[R == 0] = np.min(R[R > 0]) * 0.1 if np.any(R > 0) else 1.0
    
    errors = np.zeros_like(nfe_range, dtype=np.float64)
    
    if method == "euler":
        # E*(N) = (Σ √(L_k²·R_k))² / N = (Σ L_k·√R_k)² / N
        numerator = np.sum(L * np.sqrt(R)) ** 2
        errors = numerator / nfe_range.astype(np.float64)
    elif method == "heun":
        # E*(N) = (Σ L_k^(3/2))³ / (N/2)²
        numerator = np.sum(L ** 1.5) ** 3
        errors = numerator / (nfe_range.astype(np.float64) / 2) ** 2
    
    return errors


def minimum_nfe_for_tolerance(
    stage_lengths_normalized: np.ndarray,
    drift_norms: np.ndarray,
    epsilon_max: float,
    method: str = "euler"
) -> int:
    """
    Compute minimum NFE needed to achieve error tolerance ε_max.
    
    Euler: N_min = (Σ L_k·√R_k)² / ε_max
    Heun:  N_min = 2 · √((Σ L_k^(3/2))³ / ε_max)
    """
    L = stage_lengths_normalized
    R = drift_norms.copy()
    R[R == 0] = np.min(R[R > 0]) * 0.1 if np.any(R > 0) else 1.0
    
    if method == "euler":
        N = np.sum(L * np.sqrt(R)) ** 2 / epsilon_max
    elif method == "heun":
        N = 2 * np.sqrt(np.sum(L ** 1.5) ** 3 / epsilon_max)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return int(np.ceil(N))


# =============================================================================
# Part IV: RDDScheduleOptimizer Class
# =============================================================================

@dataclass
class RDDScheduleOptimizer:
    """
    RDD Schedule Optimizer: computes theoretically optimal stage allocations
    for both training and inference.
    
    Theory Summary:
    ==============
    
    TRAINING ALLOCATION:
      stage_length_k ∝ √C_k where C_k = E_k(2D band energy) × W_k(bandwidth)
      Derivation: Minimize Σ_k Var[loss_k]/stage_length_k via Cauchy-Schwarz
      
    INFERENCE ALLOCATION (Euler):
      n_k ∝ L_k · √R_k  (L_k=stage length fraction, R_k=curvature proxy)
      Derivation: Lagrange multiplier on Σ L_k²·R_k/n_k subject to Σn_k=N
      
    INFERENCE ALLOCATION (Heun - exact for linear v(t)):
      Perfect model: n_k = 1 per stage → total = 2K NFE (exact!)
      Imperfect model: n_k ∝ L_k^(3/2)
      
    QUALITY-SPEED TRADEOFF:
      Euler: E*(N) = (Σ L_k·√R_k)² / N  → E ∝ 1/N
      Heun:  E*(N) ∝ 1/N²  → quadratic convergence (much better!)
    """
    
    # Configuration
    K: int = 3                        # Number of stages
    scales: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    A: float = 13910.0                # Power-law amplitude (DiT VAE)
    beta: float = 1.54                # Power-law exponent
    omega_max: float = 16.0           # Maximum frequency (latent_dim/2)
    T_total: int = 1000               # Total timestep range
    use_2d: bool = True               # Use 2D radial integration
    alpha: float = 1.0                # Drift weight in complexity
    
    def __post_init__(self):
        """Compute frequency bands from scales."""
        self.frequency_bands = self._compute_frequency_bands()
        self.band_energies = self._compute_band_energies()
        self.drift_norms = self._estimate_drift_norms()
        
    def _compute_frequency_bands(self) -> List[Tuple[float, float]]:
        """
        Map resolution scales to frequency bands.
        
        Scale s_k means the stage handles frequencies up to s_k × ω_max.
        Band k covers [s_{k-1}·ω_max, s_k·ω_max] where s_0 = 0.
        
        Example: scales=[0.25, 0.5, 1.0], ω_max=16
          Stage 0 (coarsest): [0, 4]   - DC + very low freq
          Stage 1 (middle):   [4, 8]   - mid frequencies
          Stage 2 (finest):   [8, 16]  - high frequencies
        """
        bands = []
        prev_scale = 0.0
        for s in self.scales:
            omega_lo = prev_scale * self.omega_max
            omega_hi = s * self.omega_max
            bands.append((omega_lo, omega_hi))
            prev_scale = s
        return bands
    
    def _compute_band_energies(self) -> np.ndarray:
        """Compute 2D band energy for each stage."""
        energies = np.zeros(self.K)
        energy_fn = band_energy_2d if self.use_2d else band_energy
        for k, (w_min, w_max) in enumerate(self.frequency_bands):
            energies[k] = energy_fn(self.A, self.beta, w_min, w_max)
        return energies
    
    def _estimate_drift_norms(self) -> np.ndarray:
        """
        Estimate ||d_k||² for each stage.
        
        For RDD, drift d_k = UP(DOWN(x,s_k)) - x captures the high-frequency
        content removed by downsampling to scale s_k.
        ||d_k||² ≈ E_{above_k} = total energy above the stage's resolution.
        
        Stage 0 (coarsest): d=0 (no drift)
        Stage k>0: ||d_k||² ≈ Σ_{j>k} E_j (energy of frequencies the stage adds)
        
        Actually for RDD, drift for stage k is the difference between
        the target and its low-pass version at scale s_{k-1}:
          ||d_k||² ≈ E_k (the energy of the band this stage handles)
        """
        norms = np.zeros(self.K)
        norms[0] = 0.0  # Coarsest stage: no drift
        for k in range(1, self.K):
            norms[k] = self.band_energies[k]
        return norms
    
    def compute_optimal_training_lengths(self) -> np.ndarray:
        """
        Compute optimal training stage lengths.
        
        Formula: stage_length_k = T_total × √C_k / Σ_j √C_j
        where C_k = E_k^(2D) × W_k × drift_correction
        """
        return optimal_training_allocation(
            self.A, self.beta, self.frequency_bands,
            self.T_total, self.use_2d, self.drift_norms, self.alpha
        )
    
    def compute_inference_steps_euler(self, total_nfe: int) -> np.ndarray:
        """Compute optimal Euler step allocation for given NFE budget."""
        stage_lengths = self.compute_optimal_training_lengths()
        L_normalized = stage_lengths / np.sum(stage_lengths)
        return euler_optimal_allocation(L_normalized, self.drift_norms, total_nfe)
    
    def compute_inference_steps_heun(self, total_nfe: int) -> np.ndarray:
        """Compute optimal Heun step allocation for given NFE budget."""
        stage_lengths = self.compute_optimal_training_lengths()
        L_normalized = stage_lengths / np.sum(stage_lengths)
        return heun_optimal_allocation(L_normalized, total_nfe)
    
    def compute_quality_speed_curve(
        self, nfe_range: np.ndarray, method: str = "euler"
    ) -> np.ndarray:
        """Compute theoretical error curve."""
        stage_lengths = self.compute_optimal_training_lengths()
        L_normalized = stage_lengths / np.sum(stage_lengths)
        return quality_speed_curve(L_normalized, self.drift_norms, nfe_range, method)
    
    def compare_with_empirical(
        self, empirical_lengths: List[int] = [300, 300, 400]
    ) -> Dict:
        """
        Compare optimal allocation with empirical baseline.
        
        Computes:
          - Relative efficiency: how much worse is empirical vs optimal
          - Per-stage analysis
        """
        optimal = self.compute_optimal_training_lengths()
        empirical = np.array(empirical_lengths, dtype=np.float64)
        
        # Compute effective training error ratio
        # Error ∝ Σ_k C_k / stage_length_k (proportional to difficulty/time)
        C = compute_stage_complexity(
            self.A, self.beta, self.frequency_bands, 
            self.use_2d, self.drift_norms, self.alpha
        )
        
        error_optimal = np.sum(C / optimal)
        error_empirical = np.sum(C / empirical)
        efficiency_ratio = error_empirical / error_optimal
        
        return {
            "optimal_lengths": optimal.tolist(),
            "empirical_lengths": empirical.tolist(),
            "complexities": C.tolist(),
            "band_energies": self.band_energies.tolist(),
            "efficiency_ratio": efficiency_ratio,
            "efficiency_loss_pct": (efficiency_ratio - 1) * 100,
        }
    
    def full_analysis(self) -> Dict:
        """Run complete analysis and return all results."""
        optimal_train = self.compute_optimal_training_lengths()
        L_norm = optimal_train / np.sum(optimal_train)
        
        # Inference allocations for various budgets
        nfe_budgets = [6, 10, 20, 50, 100, 250]
        euler_allocs = {}
        heun_allocs = {}
        for nfe in nfe_budgets:
            euler_allocs[nfe] = euler_optimal_allocation(
                L_norm, self.drift_norms, nfe
            ).tolist()
            if nfe >= 2 * self.K:
                heun_allocs[nfe] = heun_optimal_allocation(
                    L_norm, nfe
                ).tolist()
        
        # Quality-speed curves
        nfe_range = np.arange(6, 252, 2)
        euler_errors = self.compute_quality_speed_curve(nfe_range, "euler")
        heun_errors = self.compute_quality_speed_curve(nfe_range, "heun")
        
        # Comparison with empirical
        comparison = self.compare_with_empirical()
        
        # Minimum NFE for various tolerances
        tolerances = [0.1, 0.05, 0.01, 0.005, 0.001]
        min_nfe_euler = {f"{eps}": minimum_nfe_for_tolerance(
            L_norm, self.drift_norms, eps, "euler") for eps in tolerances}
        min_nfe_heun = {f"{eps}": minimum_nfe_for_tolerance(
            L_norm, self.drift_norms, eps, "heun") for eps in tolerances}
        
        return {
            "config": {
                "K": self.K,
                "scales": self.scales,
                "A": self.A,
                "beta": self.beta,
                "omega_max": self.omega_max,
                "T_total": self.T_total,
                "frequency_bands": self.frequency_bands,
            },
            "training": {
                "optimal_stage_lengths": optimal_train.tolist(),
                "band_energies": self.band_energies.tolist(),
                "drift_norms": self.drift_norms.tolist(),
                "comparison_with_empirical": comparison,
            },
            "inference": {
                "euler_allocations": euler_allocs,
                "heun_allocations": heun_allocs,
                "min_nfe_euler": min_nfe_euler,
                "min_nfe_heun": min_nfe_heun,
            },
            "theory": {
                "heun_exact_nfe": 2 * self.K,
                "euler_convergence_rate": "O(1/N)",
                "heun_convergence_rate": "O(1/N²)",
                "linear_velocity_theorem": (
                    "v(t) = (x₁-ε) + (1-2t)·d is linear in t within each stage. "
                    "Heun's method is exact for linear ODEs → 2K NFE suffices."
                ),
            },
        }


# =============================================================================
# Part V: Sensitivity Analysis
# =============================================================================

def sensitivity_to_beta(
    A: float, omega_max: float, K: int, scales: List[float],
    beta_range: np.ndarray
) -> np.ndarray:
    """
    Analyze how optimal allocation changes with power-law exponent β.
    
    Higher β → steeper spectrum → more energy in low frequencies
    → coarsest stage gets proportionally more time.
    """
    results = np.zeros((len(beta_range), K))
    for i, beta in enumerate(beta_range):
        opt = RDDScheduleOptimizer(K=K, scales=scales, A=A, beta=beta, omega_max=omega_max)
        results[i] = opt.compute_optimal_training_lengths()
    return results


def sensitivity_to_K(
    A: float, beta: float, omega_max: float, K_range: List[int]
) -> Dict:
    """
    Analyze effect of number of stages K on optimal allocation.
    
    More stages → finer frequency decomposition → different length distribution.
    """
    results = {}
    for K in K_range:
        # Generate uniform scales
        scales = [(i+1)/K for i in range(K)]
        opt = RDDScheduleOptimizer(K=K, scales=scales, A=A, beta=beta, omega_max=omega_max)
        lengths = opt.compute_optimal_training_lengths()
        results[K] = {
            "scales": scales,
            "lengths": lengths.tolist(),
            "heun_exact_nfe": 2 * K,
        }
    return results


# =============================================================================
# Part VI: Visualization
# =============================================================================

def plot_all(optimizer: RDDScheduleOptimizer, results: Dict):
    """Generate all visualization plots."""
    
    fig_dir = OUTPUT_DIR
    
    # =========================================================================
    # Figure 1: Power Spectrum and Band Energies
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Power spectrum with band coloring
    omega = np.linspace(0.5, optimizer.omega_max, 500)
    P = power_spectrum(omega, optimizer.A, optimizer.beta)
    
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    band_labels = ['Stage 0 (coarse)', 'Stage 1 (mid)', 'Stage 2 (fine)']
    
    axes[0].semilogy(omega, P, 'k-', linewidth=2, label='P(ω) = A·|ω|^{-β}')
    for k, ((w_min, w_max), color, label) in enumerate(
        zip(optimizer.frequency_bands, colors, band_labels)):
        mask = (omega >= max(w_min, 0.5)) & (omega <= w_max)
        axes[0].fill_between(omega[mask], P[mask], alpha=0.3, color=color, label=label)
    
    axes[0].set_xlabel('Frequency ω', fontsize=12)
    axes[0].set_ylabel('Power P(ω)', fontsize=12)
    axes[0].set_title(f'Power Spectrum P(ω) = {optimizer.A:.0f}·|ω|^{{-{optimizer.beta:.2f}}}', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Band energies bar chart
    energies = optimizer.band_energies
    bandwidths = [b[1]-b[0] for b in optimizer.frequency_bands]
    complexities = compute_stage_complexity(
        optimizer.A, optimizer.beta, optimizer.frequency_bands, True)
    
    x = np.arange(optimizer.K)
    width = 0.25
    axes[1].bar(x - width, energies/np.max(energies), width, color=colors, alpha=0.7, label='Band Energy E_k')
    axes[1].bar(x, [b/max(bandwidths) for b in bandwidths], width, color=colors, alpha=0.4, label='Bandwidth W_k')
    axes[1].bar(x + width, complexities/np.max(complexities), width, color=colors, alpha=0.9, 
                edgecolor='black', linewidth=1.5, label='Complexity C_k = E_k×W_k')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Stage {k}\n[{b[0]:.0f}, {b[1]:.0f}]' for k, b in enumerate(optimizer.frequency_bands)])
    axes[1].set_ylabel('Normalized Value', fontsize=12)
    axes[1].set_title('Per-Stage Band Energy, Bandwidth & Complexity', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig1_spectrum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 1: Power spectrum and band energies")
    
    # =========================================================================
    # Figure 2: Optimal Training Lengths vs Empirical
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    optimal = np.array(results["training"]["optimal_stage_lengths"])
    empirical = np.array([300, 300, 400])
    comparison = results["training"]["comparison_with_empirical"]
    
    # Left: Bar comparison
    x = np.arange(optimizer.K)
    width = 0.35
    bars1 = axes[0].bar(x - width/2, optimal, width, color='#2196F3', alpha=0.8, label='Optimal (theory)')
    bars2 = axes[0].bar(x + width/2, empirical, width, color='#FF9800', alpha=0.8, label='Empirical [300,300,400]')
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Stage {k}\n({optimizer.frequency_bands[k][0]:.0f}-{optimizer.frequency_bands[k][1]:.0f} Hz)' 
                             for k in range(optimizer.K)])
    axes[0].set_ylabel('Stage Length (timesteps)', fontsize=12)
    axes[0].set_title(f'Training Stage Length Allocation\n'
                      f'Empirical is {comparison["efficiency_loss_pct"]:.1f}% suboptimal', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 5, f'{h:.0f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 5, f'{h:.0f}', 
                    ha='center', va='bottom', fontsize=10)
    
    # Right: Pie chart of allocation fractions
    optimal_pct = optimal / np.sum(optimal) * 100
    empirical_pct = empirical / np.sum(empirical) * 100
    
    labels = [f'Stage {k}' for k in range(optimizer.K)]
    axes[1].pie(optimal_pct, labels=[f'{l}\n{p:.1f}%' for l, p in zip(labels, optimal_pct)],
                colors=colors, autopct='', startangle=90, 
                wedgeprops=dict(width=0.4, edgecolor='white'))
    axes[1].pie(empirical_pct, labels=None,
                colors=colors, autopct='', startangle=90, radius=0.55,
                wedgeprops=dict(width=0.35, edgecolor='white', alpha=0.5))
    axes[1].set_title('Allocation: Outer=Optimal, Inner=Empirical', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig2_training.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 2: Training stage lengths comparison")
    
    # =========================================================================
    # Figure 3: Inference Step Allocation
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nfe_budgets = [10, 20, 50, 100, 250]
    euler_data = []
    heun_data = []
    
    stage_lengths = np.array(results["training"]["optimal_stage_lengths"])
    L_norm = stage_lengths / np.sum(stage_lengths)
    
    for nfe in nfe_budgets:
        euler_data.append(euler_optimal_allocation(L_norm, optimizer.drift_norms, nfe))
        if nfe >= 2 * optimizer.K:
            heun_data.append(heun_optimal_allocation(L_norm, nfe))
        else:
            heun_data.append(np.ones(optimizer.K, dtype=int))
    
    # Left: Euler allocation
    x = np.arange(optimizer.K)
    width = 0.15
    for i, (nfe, alloc) in enumerate(zip(nfe_budgets, euler_data)):
        offset = (i - len(nfe_budgets)/2 + 0.5) * width
        axes[0].bar(x + offset, alloc, width, alpha=0.8, label=f'NFE={nfe}')
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Stage {k}' for k in range(optimizer.K)])
    axes[0].set_ylabel('Steps per Stage', fontsize=12)
    axes[0].set_title('Euler: Optimal Step Allocation\nn_k ∝ L_k·√R_k', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Heun allocation
    for i, (nfe, alloc) in enumerate(zip(nfe_budgets, heun_data)):
        offset = (i - len(nfe_budgets)/2 + 0.5) * width
        axes[1].bar(x + offset, alloc, width, alpha=0.8, label=f'NFE={nfe}')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Stage {k}' for k in range(optimizer.K)])
    axes[1].set_ylabel('Heun Steps per Stage (×2 = NFE)', fontsize=12)
    axes[1].set_title('Heun: Optimal Step Allocation\nn_k ∝ L_k^{3/2} (imperfect model)', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig3_inference.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 3: Inference step allocation")
    
    # =========================================================================
    # Figure 4: Quality-Speed Tradeoff Curve
    # =========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    nfe_range = np.arange(6, 252, 2)
    euler_errors = optimizer.compute_quality_speed_curve(nfe_range, "euler")
    heun_errors = optimizer.compute_quality_speed_curve(nfe_range, "heun")
    
    # Normalize for visualization
    euler_errors_norm = euler_errors / euler_errors[0]
    heun_errors_norm = heun_errors / heun_errors[0]
    
    ax.semilogy(nfe_range, euler_errors_norm, 'b-', linewidth=2, label='Euler optimal: E ∝ 1/N')
    ax.semilogy(nfe_range, heun_errors_norm, 'r-', linewidth=2, label='Heun optimal: E ∝ 1/N²')
    
    # Mark the 2K=6 exact point for Heun
    ax.axvline(x=2*optimizer.K, color='r', linestyle='--', alpha=0.5)
    ax.annotate(f'Heun exact:\n{2*optimizer.K} NFE', xy=(2*optimizer.K, 0.01),
                fontsize=10, color='red', ha='center')
    
    # Mark typical operating points
    for nfe_mark in [20, 50, 100]:
        euler_e = euler_errors_norm[np.argmin(np.abs(nfe_range - nfe_mark))]
        heun_e = heun_errors_norm[np.argmin(np.abs(nfe_range - nfe_mark))]
        ax.plot(nfe_mark, euler_e, 'bo', markersize=8)
        ax.plot(nfe_mark, heun_e, 'rs', markersize=8)
        speedup = euler_e / heun_e
        ax.annotate(f'{speedup:.1f}× better', xy=(nfe_mark+3, heun_e),
                   fontsize=9, color='darkred')
    
    ax.set_xlabel('Total NFE (Function Evaluations)', fontsize=12)
    ax.set_ylabel('Relative Error (normalized)', fontsize=12)
    ax.set_title('Quality-Speed Tradeoff: Euler vs Heun (RDD-FM)\n'
                 'Heun exploits linear velocity theorem for quadratic convergence', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 255)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig4_quality_speed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 4: Quality-speed tradeoff curve")
    
    # =========================================================================
    # Figure 5: Euler vs Heun Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nfe_test = 20
    euler_steps = euler_optimal_allocation(L_norm, optimizer.drift_norms, nfe_test)
    heun_steps = heun_optimal_allocation(L_norm, nfe_test)
    
    # Left: Direct comparison at NFE=20
    x = np.arange(optimizer.K)
    width = 0.35
    axes[0].bar(x - width/2, euler_steps, width, color='#2196F3', alpha=0.8, 
               label=f'Euler: {np.sum(euler_steps)} steps = {np.sum(euler_steps)} NFE')
    axes[0].bar(x + width/2, heun_steps, width, color='#F44336', alpha=0.8,
               label=f'Heun: {np.sum(heun_steps)} steps = {2*np.sum(heun_steps)} NFE')
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Stage {k}' for k in range(optimizer.K)])
    axes[0].set_ylabel('Steps per Stage', fontsize=12)
    axes[0].set_title(f'NFE={nfe_test}: Euler vs Heun Allocation', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Error comparison across NFE range
    nfe_range = np.arange(6, 102, 2)
    euler_e = optimizer.compute_quality_speed_curve(nfe_range, "euler")
    heun_e = optimizer.compute_quality_speed_curve(nfe_range, "heun")
    
    # Compute effective "equivalent quality NFE"
    axes[1].plot(nfe_range, nfe_range, 'k--', alpha=0.5, label='Equal')
    
    # For each Heun NFE, find Euler NFE with same error
    equiv_euler_nfe = []
    for i, nfe in enumerate(nfe_range):
        target_error = heun_e[i]
        # Find Euler NFE that gives same error
        if target_error > 0:
            # euler_e = C_euler / N → N = C_euler / target_error
            C_euler = euler_e[0] * nfe_range[0]  # numerator constant
            equiv_nfe = C_euler / target_error
            equiv_euler_nfe.append(min(equiv_nfe, 500))
        else:
            equiv_euler_nfe.append(500)
    
    axes[1].plot(nfe_range, equiv_euler_nfe, 'r-', linewidth=2, 
                label='Heun → equivalent Euler NFE')
    axes[1].fill_between(nfe_range, nfe_range, equiv_euler_nfe, alpha=0.2, color='red',
                        label='Heun advantage region')
    
    axes[1].set_xlabel('Heun NFE', fontsize=12)
    axes[1].set_ylabel('Equivalent Euler NFE (same quality)', fontsize=12)
    axes[1].set_title('Heun Effective Speedup over Euler\n(same error → how many Euler NFE needed)', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(6, 100)
    axes[1].set_ylim(0, 300)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig5_euler_vs_heun.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 5: Euler vs Heun comparison")
    
    # =========================================================================
    # Figure 6: Sensitivity Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Sensitivity to β
    beta_range = np.linspace(1.0, 3.0, 50)
    beta_results = sensitivity_to_beta(
        optimizer.A, optimizer.omega_max, optimizer.K, optimizer.scales, beta_range)
    
    for k in range(optimizer.K):
        axes[0].plot(beta_range, beta_results[:, k], linewidth=2, color=colors[k],
                    label=f'Stage {k} [{optimizer.frequency_bands[k][0]:.0f}-{optimizer.frequency_bands[k][1]:.0f}]')
    
    axes[0].axvline(x=optimizer.beta, color='gray', linestyle='--', alpha=0.7)
    axes[0].annotate(f'β={optimizer.beta:.2f}\n(DiT VAE)', 
                    xy=(optimizer.beta, axes[0].get_ylim()[0]),
                    fontsize=10, color='gray', ha='center')
    
    axes[0].set_xlabel('Power-law exponent β', fontsize=12)
    axes[0].set_ylabel('Optimal Stage Length', fontsize=12)
    axes[0].set_title('Sensitivity to β\n(steeper spectrum → more time for coarse stage)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Sensitivity to K (number of stages)
    K_range = [2, 3, 4, 5, 6]
    K_results = sensitivity_to_K(optimizer.A, optimizer.beta, optimizer.omega_max, K_range)
    
    for K_val in K_range:
        lengths = K_results[K_val]["lengths"]
        x = np.arange(K_val) / K_val  # Normalize x-axis
        axes[1].plot(x, lengths, 'o-', linewidth=2, markersize=8, 
                    label=f'K={K_val} (Heun exact: {2*K_val} NFE)')
    
    axes[1].set_xlabel('Stage index (normalized)', fontsize=12)
    axes[1].set_ylabel('Stage Length', fontsize=12)
    axes[1].set_title('Effect of Number of Stages K\non Optimal Allocation', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'optimal_alloc_fig6_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] Figure 6: Sensitivity analysis")


# =============================================================================
# Part VII: SPEED Comparison (Why it fails for RDD)
# =============================================================================

def speed_comparison(optimizer: RDDScheduleOptimizer):
    """
    Demonstrate why SPEED's formula fails for DiT VAE latents and
    how our approach solves this.
    """
    print("\n" + "="*70)
    print("SPEED vs RDD-Optimal: Comparison")
    print("="*70)
    
    # SPEED activation time
    delta = 0.01  # SPEED default tolerance
    
    print(f"\nSPEED Parameters: δ={delta}, A={optimizer.A}, β={optimizer.beta}")
    print(f"RDD Parameters:   K={optimizer.K}, scales={optimizer.scales}")
    print(f"                  ω_max={optimizer.omega_max}")
    
    print(f"\nSPEED activation times (t* for boundary frequencies):")
    for k, (w_min, w_max) in enumerate(optimizer.frequency_bands):
        if w_max > 0:
            P_max = power_spectrum(np.array([w_max]), optimizer.A, optimizer.beta)[0]
            # SPEED formula: t* = 1/(1+√(δ/(P·(1+P-δ))))
            if P_max > delta:
                t_star = 1.0 / (1.0 + np.sqrt(delta / (P_max * (1.0 + P_max - delta))))
            else:
                t_star = 0.0
            print(f"  Stage {k}: ω_max={w_max:.1f}, P(ω_max)={P_max:.1f}, t*={t_star:.6f}")
    
    print(f"\n  → PROBLEM: P(ω) is so large that t* ≈ 1.0 for all frequencies!")
    print(f"    This means SPEED allocates ALL steps to the first stage.")
    print(f"    Root cause: A={optimizer.A:.0f} is ~70× larger than FLUX (A=203.6)")
    
    print(f"\n  → SOLUTION (our approach):")
    print(f"    Instead of activation time, use band energy × bandwidth as complexity proxy.")
    print(f"    This naturally handles large A by distributing proportionally.")
    
    optimal = optimizer.compute_optimal_training_lengths()
    print(f"\n  Optimal allocation: {optimal.astype(int).tolist()}")
    print(f"  vs Empirical:       [300, 300, 400]")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RDD Optimal Stage Allocation: Theoretical Framework")
    print("="*70)
    
    # Initialize optimizer with DiT VAE parameters
    optimizer = RDDScheduleOptimizer(
        K=3,
        scales=[0.25, 0.5, 1.0],
        A=13910.0,
        beta=1.54,
        omega_max=16.0,
        T_total=1000,
        use_2d=True,
        alpha=1.0,
    )
    
    print(f"\nConfiguration:")
    print(f"  Stages: K={optimizer.K}")
    print(f"  Scales: {optimizer.scales}")
    print(f"  Frequency bands: {optimizer.frequency_bands}")
    print(f"  Power spectrum: P(ω) = {optimizer.A}·|ω|^(-{optimizer.beta})")
    print(f"  ω_max = {optimizer.omega_max}")
    
    # Run full analysis
    print(f"\n{'='*70}")
    print("PART I: Training Stage Length Allocation")
    print("="*70)
    
    results = optimizer.full_analysis()
    
    optimal_train = np.array(results["training"]["optimal_stage_lengths"])
    print(f"\n  Band Energies (2D): {[f'{e:.1f}' for e in results['training']['band_energies']]}")
    print(f"  Drift Norms:        {[f'{d:.1f}' for d in results['training']['drift_norms']]}")
    print(f"\n  Optimal Stage Lengths: {optimal_train.astype(int).tolist()}")
    print(f"  Empirical Baseline:    [300, 300, 400]")
    
    comparison = results["training"]["comparison_with_empirical"]
    print(f"\n  Efficiency ratio (empirical/optimal): {comparison['efficiency_ratio']:.4f}")
    print(f"  Empirical is {comparison['efficiency_loss_pct']:.1f}% suboptimal")
    
    print(f"\n{'='*70}")
    print("PART II: Inference Step Allocation")
    print("="*70)
    
    print(f"\n  Linear Velocity Theorem:")
    print(f"    v(t) = (x₁-ε) + (1-2t)·d is linear in t")
    print(f"    → Heun exact with 1 step/stage → {2*optimizer.K} NFE total (perfect model)")
    
    print(f"\n  Euler Optimal Allocation (n_k ∝ L_k·√R_k):")
    for nfe, alloc in results["inference"]["euler_allocations"].items():
        print(f"    NFE={nfe:>3}: steps = {alloc}")
    
    print(f"\n  Heun Optimal Allocation (n_k ∝ L_k^(3/2), imperfect model):")
    for nfe, alloc in results["inference"]["heun_allocations"].items():
        print(f"    NFE={nfe:>3}: heun_steps = {alloc} (actual NFE = {2*sum(alloc)})")
    
    print(f"\n  Minimum NFE for Error Tolerances:")
    print(f"    {'Tolerance':<12} {'Euler NFE':<12} {'Heun NFE':<12} {'Heun Speedup':<12}")
    for eps in results["inference"]["min_nfe_euler"]:
        euler_n = results["inference"]["min_nfe_euler"][eps]
        heun_n = results["inference"]["min_nfe_heun"][eps]
        speedup = euler_n / max(heun_n, 1)
        print(f"    {eps:<12} {euler_n:<12} {heun_n:<12} {speedup:.1f}×")
    
    # SPEED comparison
    speed_comparison(optimizer)
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("PART III: Generating Visualizations")
    print("="*70)
    plot_all(optimizer, results)
    
    # Save results to JSON
    results_file = os.path.join(OUTPUT_DIR, "rdd_optimal_allocation_results.json")
    
    # Convert numpy arrays for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_serializable(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    # Final summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Recommended Configurations")
    print("="*70)
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ RDD-FM Optimal Stage Allocation (K=3, ImageNet 256×256)             │
├─────────────────┬───────────────┬───────────────┬───────────────────┤
│                 │ Stage 0       │ Stage 1       │ Stage 2           │
│                 │ [0-4 Hz]      │ [4-8 Hz]      │ [8-16 Hz]         │
├─────────────────┼───────────────┼───────────────┼───────────────────┤
│ Training Length │ {int(optimal_train[0]):>5}         │ {int(optimal_train[1]):>5}         │ {int(optimal_train[2]):>5}             │
│ (optimal)       │               │               │                   │
├─────────────────┼───────────────┼───────────────┼───────────────────┤
│ Training Length │   300         │   300         │   400             │
│ (empirical)     │               │               │                   │
├─────────────────┼───────────────┼───────────────┼───────────────────┤
│ Euler 50 NFE   │ {results['inference']['euler_allocations'][50][0]:>5}         │ {results['inference']['euler_allocations'][50][1]:>5}         │ {results['inference']['euler_allocations'][50][2]:>5}             │
├─────────────────┼───────────────┼───────────────┼───────────────────┤
│ Heun (perfect) │     1         │     1         │     1             │
│ = 6 NFE exact  │               │               │                   │
└─────────────────┴───────────────┴───────────────┴───────────────────┘

Key Theoretical Results:
  • Training: stage_length_k ∝ √(E_k × W_k)  [energy × bandwidth complexity]
  • Inference Euler: n_k ∝ L_k·√R_k  [stage length × √curvature]
  • Inference Heun: 2K = {2*optimizer.K} NFE exact (Linear Velocity Theorem)
  • Heun convergence: O(1/N²) vs Euler O(1/N) → quadratic advantage
  • Empirical [300,300,400] is {comparison['efficiency_loss_pct']:.1f}% suboptimal
""")
    
    print("Done! All plots saved to:", OUTPUT_DIR)
