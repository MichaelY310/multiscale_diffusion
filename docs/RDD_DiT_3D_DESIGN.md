# RDD-DiT-3D: Resolution-Dependent Diffusion for 3D Point Cloud Generation

## 1. Overview

Fuses three ideas for efficient 3D generation:
1. **Set Transformer** — encodes variable-length 3D point sets in each grid cell into a fixed-dim embedding
2. **RDD (Resolution-Dependent Diffusion)** — multi-resolution diffusion: coarse→fine
3. **Empty Patch Pruning** — skips provably-empty regions in finer stages

### Pipeline
```
Input PointCloud [N,3]
  → Spatial Grid (R³ cells) + Set Transformer Encode
  → Volume Z ∈ R^{C × R × R × R}  (C=128, R=16)
  → RDD 3-stage diffusion:
      Stage 1: 4×4×4   (64 tokens, 400 steps)
      Stage 2: 8×8×8   (512 tokens, 300 steps)
      Stage 3: 16×16×16 (4096 tokens → pruned to ~500-1500, 300 steps)
  → Set Transformer Decode occupied cells
  → Output PointCloud
```

## 2. Data Representation

### Volume Encoding
- Normalize mesh vertices to [-0.5, 0.5]³
- Divide into 16×16×16 grid (4096 cells)
- Each cell: collect up to K_MAX=10 points
- Encode each cell with pre-trained Set Transformer → embedding ∈ R^C
- Empty cells → learned empty embedding e_∅ ∈ R^C
- Result: Z ∈ R^{C × 16 × 16 × 16}

### Resolution Hierarchy
- Stage 1: avg_pool3d(Z, 4) → 4×4×4 volume, 64 tokens
- Stage 2: avg_pool3d(Z, 2) → 8×8×8 volume, 512 tokens
- Stage 3: Z directly → 16×16×16 volume, 4096 tokens (pruned during sampling)

## 3. RDD Diffusion Framework

### Stage Boundaries
Total T=1000: [0,400) coarse | [400,700) mid | [700,1000) fine

### Drift (non-coarsest stages)
```
d = Upsample3D(Downsample3D(x_0, 2), 2) - x_0
```
Represents high-frequency detail lost by downsampling.

### Forward Process with Drift
```
q(x_t | x_0) = N(√ᾱ_t · (x_0 + κ(t)·d), (1-ᾱ_t)·I)
κ(t) = t / (T_{c+1} - T_c)   [linearly interpolated]
```

### Training Loss
```
noise_target = ε + √(ᾱ_t/(1-ᾱ_t)) · κ(t) · d
loss = MSE(model(x_t, t, y), noise_target)
```
Coarsest stage: standard ε-prediction (no drift).

## 4. Empty Patch Pruning

### Detection (between stages during sampling)
1. Predict x̂_0 from current noisy state
2. Compute cosine_similarity(each_voxel, e_∅)
3. Mark as empty if sim > τ (τ₁=0.85 after Stage 1, τ₂=0.90 after Stage 2)

### Safety Mechanisms
- **Dilation**: expand active region by 3×3×3 kernel
- **Random retention**: keep 5% of "empty" patches as safety

### Speedup
- Stage 3: typically only 15-40% of 4096 tokens are active
- Only active tokens participate in self-attention

## 5. Model Architecture (DiT3D_RDD)

Single unified Transformer for all resolutions:
- Input projection: Linear(C_in → hidden)
- 3D sincos positional embeddings (precomputed at R=16, interpolated for coarser)
- N × DiTBlock(adaLN-Zero): self-attention + MLP with conditioning
- FinalLayer(adaLN-Zero + Linear → C_out)
- Output reshape to volume

Each voxel = 1 token (no Conv3d patching).
Variable resolution handled by positional embedding interpolation.

Configs: S(384/12/6), B(768/12/12), L(1024/24/16), XL(1152/28/16)

## 6. Training
- Precompute volumes offline (Set Transformer encode)
- Single model, all stages, sum losses
- No pruning during training (all tokens processed)
- EMA for evaluation
- AdamW lr=1e-4

## 7. Sampling
1. Stage 1: generate 4×4×4 from noise
2. Detect empty → compute active_mask for 8×8×8
3. Noise-fill upsample to 8×8×8 (nearest + re-noise)
4. Stage 2: denoise 8×8×8 (sparse attention on active tokens)
5. Refine active_mask for 16×16×16
6. Noise-fill upsample to 16×16×16
7. Stage 3: denoise 16×16×16 (sparse, ~500-1500 tokens)
8. Decode occupied cells with Set Transformer

## 8. Evaluation
- 1-NNA (CD/EMD), COV (CD/EMD), JSD
- Same protocol as DiT-3D on ShapeNet

## 9. Hyperparameters
| Param | Value |
|-------|-------|
| Volume R | 16 |
| Embedding C | 128 |
| K_MAX | 10 |
| T_total | 1000 |
| Stages | 400/300/300 |
| Resolutions | 4/8/16 |
| Hidden | 768 (Base) |
| Depth | 12 |
| Heads | 12 |
| τ₁ | 0.85 |
| τ₂ | 0.90 |
| Dilation | 3×3×3 |
| Random keep | 5% |
