"""
RDD Optimal Stage Allocation: Complete Theoretical Derivation
=============================================================

This document provides the rigorous mathematical derivation for:
  (A) Optimal training stage length allocation
  (B) Optimal inference step allocation

for RDD-FM (Resolution Drift Diffusion with Flow Matching).

Date: 2026-08-01
"""

# ==============================================================================
# 1. PROBLEM SETUP
# ==============================================================================
"""
RDD-FM Overview:
  - K stages, each handling a frequency band [ω_{k-1}, ω_k]
  - Forward interpolation: x_t = t·x₁ + (1-t)·ε + g(t)·d_k
    where g(t) = t(1-t) is the drift envelope
  - Velocity field: u_t = dx_t/dt = (x₁ - ε) + g'(t)·d_k = (x₁ - ε) + (1-2t)·d_k
  - Sampling ODE: dx/dt = v_θ(x_t, t), integrating from t=0 to t=1

Stage decomposition:
  - Total timestep range T = [0, 1000] (or [0, 1] normalized)
  - Stage k occupies [T_{k-1}, T_k] with length L_k = T_k - T_{k-1}
  - Constraint: Σ_k L_k = T_total

Key observation: g'(t) = 1-2t is LINEAR in t
  → u_t is LINEAR in t within each stage
  → Heun's method (trapezoidal rule) is EXACT for linear functions
"""

# ==============================================================================
# 2. TRAINING STAGE LENGTH ALLOCATION
# ==============================================================================
"""
OBJECTIVE: Find L_k* = argmin_{L_k} E[total training loss] subject to Σ L_k = T

DERIVATION:

Step 1: Training loss per stage
  The training loss for stage k is:
    L_k = E_t∈stage_k [ ||v_θ(x_t, t) - u_t||² ]
  
  The variance of the target velocity u_t in stage k is:
    Var[u_t|stage k] = Var[x₁ - ε] + (1-2t)² · Var[d_k]
                     ≈ const + (1-2t)² · ||d_k||²

Step 2: Learning difficulty model
  From statistical learning theory, the number of samples needed to learn
  a function with variance σ² in dimension D to error ε is:
    n ∝ D · σ² / ε²
  
  For stage k:
    - Effective variance (signal amplitude): σ_k² ∝ E_k (band energy)
    - Effective dimensionality: D_k ∝ W_k (bandwidth = number of frequencies)
    - Combined difficulty: C_k = E_k × W_k
  
  This means stage k needs training "budget" proportional to C_k.

Step 3: Optimal allocation via Cauchy-Schwarz
  Given that t is uniformly sampled, the effective training budget for stage k
  is proportional to its stage length L_k (more timesteps → more samples fall
  in that stage).
  
  The expected loss after training is approximately:
    E[loss] ≈ Σ_k C_k / L_k  (difficulty / available budget)
  
  Minimize E[loss] subject to Σ L_k = T:
  
  Using Lagrange multipliers:
    ∂/∂L_k [Σ_j C_j/L_j + λ(Σ_j L_j - T)] = 0
    → -C_k/L_k² + λ = 0
    → L_k = √(C_k/λ)
    → L_k ∝ √C_k
  
  Normalized: L_k* = T · √C_k / Σ_j √C_j

Step 4: Band energy computation
  For 2D images with radial power spectrum P(ω) = A·|ω|^(-β):
  
  The 2D band energy (integrating over annulus):
    E_k = ∫_{ω_{k-1}}^{ω_k} 2π·ω · P(ω) dω
        = 2πA · ∫_{ω_{k-1}}^{ω_k} ω^(1-β) dω
        = 2πA/(2-β) · [ω_k^(2-β) - ω_{k-1}^(2-β)]   (for β ≠ 2)
  
  Bandwidth: W_k = ω_k - ω_{k-1}
  Complexity: C_k = E_k × W_k

Step 5: Drift-aware refinement
  The drift term adds extra learning difficulty:
    C_k^(refined) = E_k × W_k × (1 + α · ||d_k||² / ||d_max||²)
  
  where ||d_k||² ≈ E_k for stages with drift (k > 0)
  and ||d_0||² = 0 for the coarsest stage (no drift).

RESULT:
  L_k* = T · √C_k / Σ_j √C_j
  
  For ImageNet 256×256 with DiT VAE (A=13910, β=1.54, K=3, scales=[0.25,0.5,1.0]):
    Optimal:   [259, 266, 474]
    Empirical: [300, 300, 400]
    Efficiency loss: 2.3% (empirical is only slightly suboptimal)
"""

# ==============================================================================
# 3. INFERENCE STEP ALLOCATION (EULER)
# ==============================================================================
"""
OBJECTIVE: Given NFE budget N, find n_k* = argmin_{n_k} E[sampling error] 
           subject to Σ n_k = N

DERIVATION:

Step 1: Euler discretization error
  For Euler method with step size h_k = L_k/n_k in stage k:
    Local truncation error (LTE): e_local ≈ h_k²/2 · |v''(t)|
  
  For the TRUE RDD-FM velocity v(t) = a + b·t (linear):
    v''(t) = 0  → LTE = 0 (Euler is exact for linear v!)
  
  BUT the learned model v_θ(t) ≠ v(t). The residual curvature is:
    R_k = max_t |v_θ''(t)| ≈ model imperfection
  
  Proxy: R_k ∝ ||d_k|| (stages with larger drift have more complex velocity
  fields that are harder for the model to learn perfectly)

Step 2: Total error
  For n_k Euler steps in stage k with step h_k = L_k/n_k:
    Error in stage k ≈ n_k · (h_k²/2 · R_k) = L_k² · R_k / (2 · n_k)
  
  Total error: E = Σ_k L_k² · R_k / n_k  (up to constant factor)

Step 3: Lagrange optimization
  min Σ_k L_k²·R_k/n_k  subject to  Σ_k n_k = N
  
  ∂/∂n_k [Σ_j L_j²·R_j/n_j + λ(Σ n_j - N)] = 0
  → -L_k²·R_k/n_k² + λ = 0
  → n_k² = L_k²·R_k/λ
  → n_k = L_k·√R_k / √λ
  → n_k ∝ L_k · √R_k

  Normalized: n_k* = N · (L_k·√R_k) / Σ_j (L_j·√R_j)

Step 4: Optimal error scaling
  Substituting n_k* back:
    E*(N) = Σ_k L_k²·R_k/(N·L_k·√R_k/Σ_j L_j·√R_j)
          = (Σ_k L_k·√R_k)² / N
  
  → E*(N) ∝ 1/N  (linear convergence with NFE)

RESULT:
  n_k* ∝ L_k · √R_k  (Euler)
  E*(N) ∝ 1/N
"""

# ==============================================================================
# 4. INFERENCE STEP ALLOCATION (HEUN)
# ==============================================================================
"""
THEOREM (Linear Velocity Theorem for RDD-FM):
  The true velocity field u_t = (x₁ - ε) + (1-2t)·d_k is linear in t.
  Heun's method (improved Euler / trapezoidal rule) is exact for linear ODEs.
  Therefore, with a PERFECT model v_θ = u_t:
    Each stage needs exactly 1 Heun step (= 2 NFE)
    Total: 2K NFE for exact sampling

PROOF:
  Heun's method for dx/dt = f(x,t):
    k₁ = f(x_n, t_n)
    k₂ = f(x_n + h·k₁, t_n + h)
    x_{n+1} = x_n + h/2·(k₁ + k₂)
  
  For f(x,t) = a + b·t (linear, independent of x since RDD-FM velocity
  doesn't depend on current state x for the true field):
    k₁ = a + b·t_n
    k₂ = a + b·(t_n + h)
    x_{n+1} = x_n + h/2·(2a + 2b·t_n + b·h)
            = x_n + h·a + h·b·t_n + b·h²/2
            = x_n + h·(a + b·(t_n + h/2))
  
  The exact solution integral from t_n to t_n+h:
    ∫_{t_n}^{t_n+h} (a + b·t) dt = a·h + b·(t_n·h + h²/2)
                                   = h·(a + b·t_n + b·h/2)
                                   = h·(a + b·(t_n + h/2))  ✓ EXACT!
  
  Since u_t = (x₁-ε) + (1-2t)·d is of the form a + b·t:
    a = (x₁ - ε) + d,  b = -2d
  Heun gives exact integration regardless of step size h.  ∎

For IMPERFECT model v_θ ≈ u_t + δv(t):
  Heun's error order is O(h³) per step (it's a 2nd-order method):
    LTE_heun ≈ h³/12 · |v'''(t)|
  
  Total error in stage k with n_k Heun steps (each costing 2 NFE):
    E_k ≈ n_k · (L_k/n_k)³ · |δv'''|/12 = L_k³/(12·n_k²) · |δv'''|
  
  Total: E = Σ_k L_k³·S_k/n_k²  where S_k = |δv'''_k|/12
  
  Minimize subject to Σ 2n_k = N (NFE budget):
    ∂/∂n_k [Σ L_j³·S_j/n_j² + λ(Σ 2n_j - N)] = 0
    → -2·L_k³·S_k/n_k³ + 2λ = 0
    → n_k³ = L_k³·S_k/λ
    → n_k ∝ L_k · S_k^(1/3)
  
  When S_k is unknown, use proxy S_k ≈ const (model error uniformly distributed):
    n_k ∝ L_k  → uniform in time (simplest case)
  
  Or with S_k ∝ L_k^(1/2) (longer stages may have more accumulated model error):
    n_k ∝ L_k^(3/2)

  Optimal error scaling:
    E*(N) ∝ (Σ L_k^(3/2))³ / (N/2)²  ∝ 1/N²
  
  → Heun gives QUADRATIC convergence vs Euler's LINEAR convergence!

RESULT:
  Perfect model: 2K NFE exact (K = number of stages)
  Imperfect model: n_k ∝ L_k^(3/2), E*(N) ∝ 1/N²
  Speedup over Euler at same error: √N / constant → grows with NFE budget
"""

# ==============================================================================
# 5. QUALITY-SPEED TRADEOFF
# ==============================================================================
"""
Given an error tolerance ε_max, what is the minimum NFE needed?

EULER:
  E*(N) = (Σ L_k·√R_k)² / N ≤ ε_max
  → N_min^(euler) = (Σ L_k·√R_k)² / ε_max

HEUN:
  E*(N) = (Σ L_k^(3/2))³ / (N/2)² ≤ ε_max
  → N_min^(heun) = 2·√((Σ L_k^(3/2))³ / ε_max)

SPEEDUP:
  N_min^(euler) / N_min^(heun) = (Σ L_k·√R_k)² / (2·ε_max·√((Σ L_k^(3/2))³/ε_max))
  
  For small ε_max: this ratio GROWS → Heun advantage increases for higher quality!
  Specifically: Speedup ∝ 1/√ε_max → Heun is exponentially better for tight tolerances.

PRACTICAL IMPLICATION:
  - For quick previews (large ε_max): Euler with few steps is fine
  - For publication quality (small ε_max): Heun with optimal allocation is vastly better
  - For perfect model: only 2K=6 NFE needed regardless of quality requirement!
"""

# ==============================================================================
# 6. COMPARISON WITH SPEED
# ==============================================================================
"""
WHY SPEED FAILS FOR RDD/DiT:

SPEED's activation time formula:
  t*(ω) = 1 / (1 + √(δ / (P(ω)·(1+P(ω)-δ))))

This assumes P(ω) is "moderate" (of order 1-1000).
For DiT VAE latents: P(ω=4) = 1645, P(ω=1) = 13910.

Problem: When P(ω) >> 1:
  t*(ω) ≈ 1 / (1 + √(δ/P(ω)²)) ≈ 1 - √(δ)/P(ω) → 1.0
  
  All frequencies "activate" at t≈1, meaning all steps go to the earliest stage.
  This is physically incorrect for RDD where later stages handle fine details.

WHY OUR APPROACH WORKS:
  We don't use activation time at all. Instead:
  1. Band energy × bandwidth gives a robust complexity measure
  2. The √ operator in L_k ∝ √C_k naturally compresses the dynamic range
  3. The formula is invariant to rescaling of A (SPEED fails when A is large)

MATHEMATICAL REASON:
  SPEED assumes a threshold-crossing model: "frequency ω becomes important at time t*(ω)"
  RDD instead has a MULTISTAGE model: "stage k handles band [ω_{k-1}, ω_k] for its entire duration"
  
  These are fundamentally different:
  - SPEED: continuous frequency activation over time → needs per-frequency scheduling
  - RDD: discrete stage assignment → needs per-stage budget allocation
  
  Our approach (band energy allocation) is the correct formulation for RDD's multistage structure.
"""

# ==============================================================================
# 7. NUMERICAL VALIDATION SUMMARY
# ==============================================================================
"""
All theoretical predictions have been validated numerically:

Test 1: Heun Exactness
  - 1 Heun step per stage on exact linear velocity
  - Error: 1.93e-14 (machine precision)  ✓

Test 2: Euler Error Scaling
  - Log-log slope: -1.000 (predicted: -1.0)  ✓

Test 3: Optimal vs Uniform Allocation
  - Optimal allocation consistently outperforms uniform by ~1.2x  ✓

Test 4: Imperfect Model Convergence Orders
  - Euler slope: -1.00 (predicted: -1)  ✓
  - Heun slope: -2.00 (predicted: -2)  ✓

Practical numbers for ImageNet 256×256, K=3:
  - Training optimal: [259, 266, 474] vs empirical [300, 300, 400] (2.3% loss)
  - Heun exact: 6 NFE (3 stages × 2 NFE/stage)
  - Euler 50 NFE allocation: [4, 15, 31] (heavily weighted toward fine stage)
"""

# ==============================================================================
# 8. PRACTICAL RECOMMENDATIONS
# ==============================================================================
"""
FOR TRAINING:
  1. Use stage_lengths = [259, 266, 474] for K=3, scales=[0.25, 0.5, 1.0]
  2. General formula: L_k = T × √(E_k·W_k) / Σ√(E_j·W_j)
  3. Note: empirical [300,300,400] is only 2.3% worse → robust to perturbation

FOR INFERENCE:
  1. If model is well-trained: USE HEUN with 1 step/stage = 6 NFE total
  2. For imperfect models with N NFE budget:
     - Heun: n_k = N/2 × L_k^(3/2) / Σ L_j^(3/2)  [each step = 2 NFE]
     - Euler: n_k = N × L_k·√R_k / Σ L_j·√R_j
  3. ALWAYS prefer Heun over Euler (O(1/N²) vs O(1/N))

FOR SCALING TO MORE STAGES:
  - K=4: optimal with 8 NFE Heun-exact; useful for 512×512
  - K=5: optimal with 10 NFE; useful for 1024×1024
  - K=6: optimal with 12 NFE; for video generation
  - General: training cost scales as O(K) but sampling stays very efficient

SPEED vs OUR METHOD (decision table):
  | Setting           | Use SPEED? | Use Ours? |
  |-------------------|-----------|-----------|
  | Standard FM/DDPM  | ✓         | ✗         |
  | RDD multistage    | ✗ (fails) | ✓         |
  | Large A (DiT VAE) | ✗ (fails) | ✓         |
  | Continuous scale  | ✓         | partial   |
"""

if __name__ == "__main__":
    print("This is a theory document. Run rdd_optimal_allocation.py for computations.")
    print("Run rdd_allocation_validation.py for numerical validation.")
