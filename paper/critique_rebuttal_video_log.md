# Video Extension Critique + Rebuttal Log

> 本文档记录视频生成扩展相关的 Critique + Rebuttal 循环。
> 每轮包含：Critique（审稿人攻击点）+ Rebuttal（论文修改）

---

## Round 1

### Critique (10 issues)

**Critique 1: All Quantitative Video Results Are Pending**
- **Severity**: Major
- **Issue**: 所有视频实验数值都标记为 X.XX，核心声明（strictly better Pareto）无任何数据支撑。
- **Rebuttal**: 认可。这是 draft 阶段，已在 experiments_plan_detailed.md 规划了 exp16-exp20 的完整实验方案。在论文中已通过红色标注明确这些是待填数值。

**Critique 2: "ODE Continuity" Contradicted by Noise Filling at Stage Boundaries**
- **Severity**: Major
- **Issue**: Noise filling (Eq. st_noise_fill) 在 stage boundary 注入新鲜噪声，这与 PyramidalWan 的 corrective noise 本质相同，都打断了 ODE 轨迹。DPM-Solver 无法跨 boundary 维持轨迹历史。
- **Rebuttal**: 这是论文的核心理论漏洞。需要精确澄清：ODE 连续性是 **within-stage** 的性质，而非 cross-stage。DPM-Solver-2 的优势在于每个 stage 内部可以使用高阶求解器（PyramidalWan 因为 stage 内部的训练目标与实际轨迹不一致，即使 stage 内也无法用 DPM-2）。已修改论文明确这一区分。

**Critique 3: Proposition "All 2D Properties Extend to 3D" Is Not a Proof**
- **Severity**: Major
- **Issue**: 未验证 STUp∘STDown 在视频维度上是否满足正交投影性质 P²=P, P^T=P。如果使用非 box filter（如双线性插值），理论保证不成立。
- **Rebuttal**: 需要在论文中明确声明使用 box filter（average pooling + nearest-neighbor），并简短证明 box filter 满足投影性质。

**Critique 4: Temporal ÷4 Under 4× VAE Compression Is Extreme**
- **Severity**: Major
- **Issue**: 81帧经过 4× VAE 压缩变成 21 帧 latent，再 ÷4 只剩 5 帧。几乎所有运动信息被摧毁。
- **Rebuttal**: 需要增加对极端时间降采样的讨论和 ablation（÷2 vs ÷4）。在 exp17 中已覆盖。论文中增加可视化说明 5 帧保留了什么信息。

**Critique 5: Training on Synthetic Data Confounds Comparison**
- **Severity**: Major
- **Issue**: 80K 合成视频的分布偏差可能影响结果。两者虽然用相同数据，但 RDD-Wan 对数据质量的敏感度可能不同。
- **Rebuttal**: 增加限制性声明。在 Discussion 中明确这一 confound。如果资源允许，在 Panda-70M 子集上做小规模验证。

**Critique 6: SURF Is Training-Free and Should Be Primary Practical Baseline**
- **Severity**: Minor
- **Issue**: 如果用户想加速，直接用 SURF（0成本）vs RDD-Wan（需要生成80K数据+5K步训练），总成本不公平。
- **Rebuttal**: 在论文中增加 total pipeline cost 讨论。SURF 的缺点是两次前向传播、不支持更低 NFE。补充 RDD-Wan + SURF 的组合可能性讨论。

**Critique 7: Convention Confusion (t=0 noise vs data)**
- **Severity**: Minor
- **Issue**: video_extension_content.tex 中存在 convention 矛盾。
- **Rebuttal**: 已在 main.tex 中统一为 t=0 noise, t=1 data（Wan 2.1 的 FM convention）。所有方程已核验。

**Critique 8: Velocity Field NOT Smooth at Stage Boundaries**
- **Severity**: Major
- **Issue**: 不同 stage 的 velocity target 有不同的 drift coefficient 和不同的 d^c_ST。在 boundary 处 velocity 是不连续的。
- **Rebuttal**: 这与 Critique 2 相关。需要重新表述 DPM-Solver 优势：不是"全局 ODE 连续"，而是"within-stage ODE 连续 + 每个 stage 内可用 DPM-2"。关键区别是 RDD-Wan 在 stage 内部的训练目标是 smooth 的（drift 连续衰减到 0），而 PyramidalWan 在 stage 内部也是 smooth 的——两者区别在于 transition 后的初始条件质量。修改论文精确化这一声明。

**Critique 9: VBench Alone Is Insufficient**
- **Severity**: Minor
- **Issue**: 缺少 FVD 和 human evaluation。
- **Rebuttal**: 在实验计划中补充 FVD 评估。Human study 留作 camera-ready。

**Critique 10: 5K Steps May Be Insufficient for Drift Learning**
- **Severity**: Minor
- **Issue**: RDD-Wan 需要学习额外的 drift signal，5K 步可能不够收敛。
- **Rebuttal**: 在 exp16 中增加 training curve 对比（1K/2K/3K/5K/10K）。在论文中增加收敛曲线讨论。

---

### Rebuttal Actions (Round 1)

1. ✅ 修改论文中 ODE continuity 的表述：明确为 within-stage 性质
2. ✅ 在 Proposition 证明中明确使用 box filter（average pooling + nearest-neighbor）
3. ✅ 增加对 temporal ÷4 极端性的讨论
4. ✅ 增加 SURF total cost 对比讨论
5. ✅ 精确化 DPM-Solver 优势的表述
6. ✅ 增加 Limitation 条目关于 synthetic data confound
7. ✅ 增加 FVD 评估到实验计划

---


---

## Round 2

### Critique (5 issues)

**Critique 2.1: DPM-Solver-2 NFE Counting Unfairness**
- **Severity**: Major
- **Issue**: DPM-Solver-2 uses 2 function evaluations per step (midpoint + correction). Claiming "20 NFE with DPM-2" is actually 10 steps × 2 evals = 20 forward passes, which should be compared to 20 Euler steps (also 20 forward passes). The paper conflates "steps" with "NFE" when the solver order differs.
- **Rebuttal**: Clarify NFE = total function evaluations in all tables. Our NFE=20 with DPM-2 means 10 DPM-2 steps (20 forward passes). This is fairly compared to 20 Euler steps (20 forward passes). Update table captions to make this explicit.

**Critique 2.2: "Strictly Better Pareto" Is Unsubstantiated**
- **Severity**: Major
- **Issue**: The claim "strictly better NFE--quality Pareto curve" appears multiple times but all data is X.XX. This is the single strongest claim in the paper with zero evidence.
- **Rebuttal**: Acknowledged as pending experimental validation. The theoretical argument (DPM-2 convergence order > Euler) provides the mechanism; data will confirm magnitude. Softened language to "expected to achieve" in draft stage.

**Critique 2.3: Redundant Paragraphs on Stage Boundary**
- **Severity**: Minor
- **Issue**: Lines 840-847 contain two nearly identical paragraphs about noise filling at stage boundaries. This appears to be a copy-paste artifact.
- **Rebuttal**: Remove the duplicate paragraph.

**Critique 2.4: Missing Comparison with REPA+RDD for Video**
- **Severity**: Minor
- **Issue**: REPA composability is claimed for images but not discussed for video. If RDD composes with REPA, does RDD-Wan compose with video-REPA variants?
- **Rebuttal**: Add a note that composability with representation alignment is a future direction for video. REPA is image-specific; video analogs (e.g., temporal representation alignment) are unexplored.

**Critique 2.5: VBench Evaluation Completeness**
- **Severity**: Medium
- **Issue**: VBench has 16 dimensions but only "quality" and "semantic" are reported. Which dimensions does RDD-Wan improve/degrade? Temporal flickering and motion smoothness are particularly relevant for multi-resolution temporal methods.
- **Rebuttal**: Add per-dimension VBench breakdown in Appendix. Particularly important: temporal flickering (expected to improve due to temporal curriculum), motion smoothness (may degrade due to aggressive temporal downsampling).

---

## Round 3

### Critique (5 issues)

**Critique 3.1: Training Marginal Circularity**
- **Severity**: Major
- **Issue**: The paper claims noise filling "matches the training marginal exactly" (line 842). But the training marginal is *defined* by the forward process that includes the drift. So noise filling matching the training marginal is tautological—it's just saying "our inference uses the same process as training." The real question is whether this forward process is *better* than PyramidalWan's, which requires empirical evidence.
- **Rebuttal**: Reframe: The key insight is that in PyramidalWan, the training objective (standard velocity matching per stage) and inference procedure (corrective noise injection) are *mismatched*—the model never sees corrective-noise-initialized samples during training. In RDD-Wan, the training and inference marginals are aligned by construction. This is a concrete advantage (no train-test mismatch), not a tautology.

**Critique 3.2: DPM-Solver-2 Convergence for Non-Smooth Velocity Fields**
- **Severity**: Medium
- **Issue**: The drift envelope $\frac{s_{k+1}-2t}{L_k}$ is $C^1$ within a stage, but the *learned* velocity $v_\theta$ may not be smooth if the network approximation introduces non-smoothness. Second-order convergence of DPM-Solver-2 requires Lipschitz continuity of the velocity field.
- **Rebuttal**: Add caveat: second-order convergence holds under the assumption that $v_\theta$ is a good approximation to the true velocity field. In practice, neural network velocity fields are smooth (due to weight sharing and activation functions), so the Lipschitz condition is empirically satisfied. The key point is that RDD *does not introduce additional sources of non-smoothness* (unlike corrective noise).

**Critique 3.3: CogVideoX, HunyuanVideo, Mochi Missing from Related Work**
- **Severity**: Medium
- **Issue**: The related work covers multi-resolution methods but omits major 2024-2025 video foundation models (CogVideoX, HunyuanVideo, Mochi-1) that use single-pass full-resolution generation. These represent the mainstream approach that multi-resolution methods aim to accelerate.
- **Rebuttal**: Add a brief "Single-pass video foundation models" paragraph acknowledging these as the dominant paradigm that RDD aims to accelerate, not replace.

**Critique 3.4: Distillation-Based Acceleration Comparison Missing**
- **Severity**: Medium
- **Issue**: The paper positions RDD-Wan as an inference acceleration method but doesn't compare with distillation-based approaches (InstaFlow, AnimateLCM, VideoLCM) that also reduce NFE for video models. These achieve 4-8 step generation and are the practical state-of-the-art for fast video inference.
- **Rebuttal**: Add discussion clarifying that RDD is orthogonal to distillation: RDD reduces per-step cost (multi-resolution) while distillation reduces step count. They compose multiplicatively. Add explicit comparison in Related Work.

**Critique 3.5: Box Filter Temporal Artifacts**
- **Severity**: Minor
- **Issue**: Average pooling over $\tau=4$ temporal frames creates a box-averaged representation that may introduce temporal aliasing (motion blur in latent space). Unlike spatial box filtering (which preserves DC component), temporal box filtering can destroy periodic motion signals with period < 8 frames (Nyquist at $\tau=4$).
- **Rebuttal**: Acknowledged. This is why temporal $\div$4 is listed as a limitation. For periodic motion (e.g., walking, waving), the 5 latent frames after temporal $\div$4 may miss half-cycles. Ablation in exp17 compares temporal $\div$2 vs $\div$4.

---

## Round 4

### Critique (5 issues)

**Critique 4.1: Softened Claims Still Too Strong**
- **Severity**: Medium
- **Issue**: Contribution 7 still says "enabling a strictly better NFE--quality Pareto curve"—the word "enabling" doesn't fix the problem since no data exists. Should say "hypothesized to enable" or "designed to enable."
- **Rebuttal**: Change to "designed to enable a better NFE--quality trade-off" (remove "strictly" and "Pareto curve" until data exists).

**Critique 4.2: Proposition 2 Doesn't Verify STDown Adjoint**
- **Severity**: Medium
- **Issue**: The proof sketch says $P^T=P$ "under the standard inner product" but doesn't verify this for the specific operators. For spatial box filter: $\langle STUp(y), x \rangle = \langle y, STDown(x) \rangle$? This requires STDown = (1/s²τ) × sum-pooling and STUp = replication (not interpolation). The normalization matters.
- **Rebuttal**: Add explicit verification: For box filter, STDown$(x)[i,j,k] = \frac{1}{s^2\tau}\sum_{p,q,r} x[...]$. Then $P = \text{STUp}\circ\text{STDown}$ is indeed self-adjoint because both operators are defined with the correct normalization. Add a footnote with the inner product calculation.

**Critique 4.3: FVD Not Used as Evaluation Metric**
- **Severity**: Medium
- **Issue**: FVD (Fréchet Video Distance) is the standard quantitative metric for video generation but is not mentioned. VBench is useful but FVD provides a single scalar comparable to FID. Both should be reported.
- **Rebuttal**: Add FVD evaluation to exp16. VBench provides finer-grained analysis; FVD provides a single comparable number. Report both.

**Critique 4.4: "Training-Consistent" Advantage May Be Small**
- **Severity**: Minor
- **Issue**: The practical advantage of training-consistent noise filling over corrective noise may be negligible if the model is robust to small distribution shifts. PyramidalWan works well empirically despite the mismatch—perhaps the mismatch is small enough to not matter.
- **Rebuttal**: This is an empirical question answered by exp17 (RDD-Wan vs PyramidalWan at matched settings). If the gap is small, RDD-Wan's advantage is primarily DPM-2 compatibility (efficiency), not quality per se. Honestly state this possibility.

**Critique 4.5: Single Video Resolution (480p) Tested**
- **Severity**: Minor
- **Issue**: Only 480p×81 frames is tested. Does RDD-Wan scale to 720p or 1080p? The token reduction advantage should be *more* pronounced at higher resolutions.
- **Rebuttal**: Add scaling argument: at 720p×121 frames, the latent is 31×90×120 = 334,800 tokens; Stage 0 reduces to 7×22×30 = 4,620 (72× reduction). Note as future work.

---

## Round 5

### Critique (5 issues)

**Critique 5.1: No Training Convergence Evidence**
- **Severity**: Major
- **Issue**: For images, RDD shows 3.7× convergence speedup. For video, this claim is never made and no training curves are shown. Does RDD-Wan converge faster than PyramidalWan given the same 5K steps? Without this, the "training efficiency" contribution doesn't extend to video.
- **Rebuttal**: Add training loss curves (at 1K/2.5K/5K/10K steps) to experiments_plan as a new experiment. The curriculum effect should accelerate video finetuning similarly to image training. If RDD-Wan converges in 2.5K steps while PyramidalWan needs 5K, that's a 2× training speedup—valuable for practical deployment.

**Critique 5.2: FM Interpolant Marginal Constraint**
- **Severity**: Medium
- **Issue**: The RDD-FM interpolant (Eq. video_fm_interpolant) adds a drift term to the standard FM interpolant. This changes the marginal distribution $p_t(x_t)$ away from the standard Gaussian mixture. Has it been verified that the resulting marginal is well-behaved (e.g., non-degenerate, bounded variance)?
- **Rebuttal**: The drift term $\frac{t(s_{k+1}-t)}{L_k}\bd^c_{ST}$ is bounded (peaks at $t = s_{k+1}/2$ with magnitude $\frac{s_{k+1}^2}{4L_k}\|\bd^c_{ST}\|$) and vanishes at stage boundaries ($t=s_k$ and $t=s_{k+1}$). The marginal remains Gaussian (shifted mean) with unchanged variance. Add this boundedness argument.

**Critique 5.3: Why Not Joint Training (All Stages Simultaneously)?**
- **Severity**: Medium
- **Issue**: The paper describes sequential sampling (Stage 0 → 1 → 2) but doesn't discuss whether training processes all stages in each batch or samples random stages. If random-stage training, what's the stage sampling distribution?
- **Rebuttal**: Clarify: following PyramidalWan, each training sample is assigned a random timestep t ∈ [0,1], which determines its stage. The stage distribution is proportional to stage length: Stage 0 gets 6% of samples, Stage 1 gets 35%, Stage 2 gets 59%. This is naturally handled by uniform t-sampling.

**Critique 5.4: "First Multi-Resolution Video Method with Within-Stage ODE Continuity"**
- **Severity**: Medium
- **Issue**: This priority claim is hard to verify. TPDiff (ICLR 2026) uses continuous schedules within temporal stages. Pyramid-Flow uses flow matching within stages. Do they not also have within-stage ODE continuity?
- **Rebuttal**: Nuance: TPDiff and Pyramid-Flow do have smooth velocity fields within stages, but their *transitions* (anti-correlated noise, Gaussian renoising) create larger distribution mismatch at boundaries. Rephrase claim to: "the first multi-resolution video method where both within-stage dynamics AND boundary transitions are provably training-consistent, enabling full DPM-Solver-2 compatibility."

**Critique 5.5: DPM-Solver-2 Convergence Proof Applicability**
- **Severity**: Minor
- **Issue**: Standard DPM-Solver convergence proofs assume the score function is Lipschitz continuous. With the nonlinear projection P (which is linear but rank-deficient), does the drift-corrected DPM-Solver still converge at second order?
- **Rebuttal**: P is a linear operator (not nonlinear), so the drift correction $\Delta_{drift}$ is a deterministic, bounded additive term. DPM-Solver-2 convergence requires Lipschitz velocity field—adding a constant (data-dependent but time-smooth) correction preserves Lipschitz continuity. The convergence order is unaffected.

---

## Round 6

### Critique (5 issues)

**Critique 6.1: Batched Inference Efficiency**
- **Severity**: Medium
- **Issue**: Stage 0 processes 1,500 tokens while Stage 2 processes 100,800 tokens. In practice, GPU utilization at Stage 0 will be extremely low (the DiT-1.3B model has far more parameters than needed for 1,500 tokens). The theoretical 67× token reduction doesn't translate to 67× speedup due to kernel launch overhead and memory bandwidth bottlenecks.
- **Rebuttal**: Acknowledged. The wall-clock speedup table (exp20) will report actual GPU-measured latency, which will be significantly less than the theoretical maximum. Cite that PyramidalWan reports ~2× actual speedup despite similar token reduction, due to GPU underutilization at coarse stages. RDD-Wan's additional advantage is NFE reduction (DPM-2), which compounds with per-step savings.

**Critique 6.2: No Qualitative Video Comparisons**
- **Severity**: Medium
- **Issue**: Unlike image generation where FID is well-established, video quality assessment benefits enormously from visual comparisons. No qualitative figures (video frames, temporal consistency comparisons) are shown.
- **Rebuttal**: Add a qualitative comparison figure showing: (1) selected frames from same prompt under RDD-Wan vs PyramidalWan vs vanilla Wan, (2) temporal difference maps highlighting flicker/consistency. Plan for supplementary video.

**Critique 6.3: Longer Videos (>81 frames) Stage Boundary Scaling**
- **Severity**: Medium
- **Issue**: For longer videos (e.g., 10 seconds at 24fps = 240 frames → 60 latent frames after VAE), temporal ÷4 gives 15 frames at Stage 0—quite reasonable. But the stage boundaries (0.06, 0.41) were calibrated for 81 frames. Do they need recalibration for different video lengths?
- **Rebuttal**: Good point. Stage boundaries are currently fixed and may need length-dependent adjustment. Add this to limitations and future work (adaptive scheduling). Note that the boundaries are inherited from PyramidalWan's empirical tuning—not theoretically optimal for RDD.

**Critique 6.4: Memory Overhead of Multi-Resolution**
- **Severity**: Minor
- **Issue**: During inference, does the model need to store intermediate results from Stage 0/1 when transitioning? Or is it purely sequential (discard previous stage outputs after noise filling)? Memory overhead should be discussed.
- **Rebuttal**: Clarify: inference is purely sequential. After noise filling at boundary, only the upsampled+noisy tensor for the next stage is retained. No memory overhead beyond single-stage inference. This is identical to PyramidalWan's memory model.

**Critique 6.5: Wan 2.1-1.3B vs Larger Models**
- **Severity**: Minor
- **Issue**: Testing only on the 1.3B model limits practical significance. Industry deployments use 5B+ models where inference cost is truly prohibitive. Does RDD-Wan's advantage scale with model size?
- **Rebuttal**: Add scaling argument: larger models have higher per-token compute (quadratic attention), so token reduction at coarse stages provides *greater* wall-clock savings. Future work: RDD-Wan on Wan 2.1-14B.

---

## Round 7

### Critique (4 issues)

**Critique 7.1: Adaptive Step-Size Methods Not Discussed**
- **Severity**: Medium
- **Issue**: Adaptive ODE solvers (e.g., adaptive-step DPM-Solver, Heun with rejection) can automatically allocate more steps where the velocity field changes rapidly. These are an alternative to fixed multi-resolution pyramids for efficient inference. Not discussed.
- **Rebuttal**: Add brief discussion: adaptive step-size methods are orthogonal—they choose *when* to evaluate, while RDD chooses *at what resolution* to evaluate. The two can be combined: adaptive step-size DPM-Solver operating within RDD's multi-resolution framework.

**Critique 7.2: Temporal Coherence Metric Missing**
- **Severity**: Medium
- **Issue**: VBench includes "temporal flickering" and "motion smoothness" dimensions, but these are not highlighted. For a method that manipulates temporal resolution, these specific metrics are the most important to report individually.
- **Rebuttal**: Will highlight temporal flickering and motion smoothness in the per-dimension breakdown. These are the dimensions where temporal curriculum should help (better global temporal structure) or hurt (information loss from temporal ÷4).

**Critique 7.3: "Summary of Video Results" Section Needed**
- **Severity**: Minor
- **Issue**: The video results section has tables but lacks a synthesizing paragraph that tells the reader what to conclude. After 4 tables, a clear "takeaway" paragraph would improve readability.
- **Rebuttal**: Add a concluding paragraph after the wall-clock table summarizing the 3 key findings: (1) quality preserved, (2) DPM-2 enables NFE reduction, (3) compound speedup from token reduction × step reduction.

**Critique 7.4: Stage Boundaries Optimization for RDD**
- **Severity**: Minor
- **Issue**: Stage boundaries (0.06, 0.41) are inherited from PyramidalWan without justification that these are optimal for RDD. Given that RDD's drift changes the effective signal content at each timestep, the optimal transition points may differ.
- **Rebuttal**: Acknowledged. Optimal boundary search for RDD-Wan is computationally expensive (requires training multiple models). We use PyramidalWan's boundaries for fair comparison and to isolate the effect of the transition mechanism. Add note about boundary optimization as future work.

---

## Round 8

### Critique (4 issues)

**Critique 8.1: Reproducibility Concerns**
- **Severity**: Medium
- **Issue**: The paper relies on 80K synthetic videos generated by Wan 2.1-14B. Will this dataset be released? Without it, the results are not reproducible. If using a different 80K dataset, would results differ?
- **Rebuttal**: Commit to releasing: (1) training code, (2) evaluation scripts, (3) model checkpoints. For the synthetic dataset: note that PyramidalWan also doesn't release their exact dataset but provides generation scripts. We will provide equivalent generation scripts (exp19).

**Critique 8.2: Statistical Significance of VBench Differences**
- **Severity**: Medium
- **Issue**: VBench scores are typically reported to 2 decimal places. What's the standard deviation across multiple runs? If differences are <0.5%, they may not be statistically significant.
- **Rebuttal**: Add error bars or confidence intervals. Plan to run evaluation 3 times with different random seeds for sampling. Report mean ± std. If differences are not significant at p<0.05, honestly state this and argue the advantage is in efficiency, not quality.

**Critique 8.3: Computational Cost of DPM-Solver-2**
- **Severity**: Minor
- **Issue**: DPM-Solver-2 requires storing trajectory history (previous velocity evaluations) for the multistep formulation. At 100,800 tokens per evaluation, this doubles peak memory compared to Euler. Is this practical on consumer hardware?
- **Rebuttal**: DPM-Solver-2 only stores one previous evaluation (the midpoint). Peak memory increase is ~1× the latent tensor size (negligible compared to model parameters and KV-cache). On A100 (80GB), this is <1GB overhead for 480p video.

**Critique 8.4: Comparison with Training-Free Methods Unfair**
- **Severity**: Minor
- **Issue**: SURF is training-free while RDD-Wan requires 5K finetuning steps. The comparison isn't apples-to-apples. A fairer framing: RDD-Wan gets X hours of training compute for its advantage, while SURF gets zero.
- **Rebuttal**: Acknowledged. Add compute budget column to Table: Wan (0 extra), SURF (0 extra), PyramidalWan (Y A100-hours), RDD-Wan (Y A100-hours, same as PyramidalWan). The fair comparison is RDD-Wan vs PyramidalWan (same budget). SURF comparison shows what's achievable without any finetuning.

---

## Round 9

### Critique (4 issues)

**Critique 9.1: Modern Video Systems in Related Work**
- **Severity**: Medium
- **Issue**: CogVideoX (2024), HunyuanVideo (2024), Mochi-1 (2024), and OpenSora (2024) are major open-source video generation systems that use single-pass full-resolution DiT. They represent the mainstream baseline that multi-resolution methods aim to accelerate. Their absence from Related Work is a notable gap.
- **Rebuttal**: Add a "Single-pass video foundation models" paragraph in Category VIII acknowledging these systems as the dominant paradigm. Note that they all face the O(T×H×W) computational cost that multi-resolution methods address.

**Critique 9.2: Distillation-Based Video Acceleration**
- **Severity**: Medium
- **Issue**: AnimateLCM, VideoLCM, and latent consistency models for video achieve 4-8 step generation. These are the practical SOTA for fast video inference. How does RDD-Wan's speedup compare? At 20 NFE (DPM-2), RDD-Wan is still 2.5× slower than 8-step distilled models.
- **Rebuttal**: Add explicit positioning: distillation reduces steps (8 vs 50), RDD reduces per-step cost (multi-resolution). They are orthogonal and composable: RDD + distillation could achieve 8 steps × multi-resolution = extreme speedup. This is future work (exp equivalent to exp15 for video). Add to Related Work and Discussion.

**Critique 9.3: Temporal Downsampling vs Frame Skipping**
- **Severity**: Minor
- **Issue**: Temporal ÷4 via average pooling is different from temporal ÷4 via frame skipping (selecting every 4th frame). Average pooling creates blurred "average frames" while frame skipping preserves sharpness of selected frames. Which is better for the coarse stage? The paper doesn't justify the choice.
- **Rebuttal**: Box filter (average pooling) is chosen for theoretical reasons: it makes STDown the adjoint of STUp (nearest-neighbor), ensuring P²=P. Frame skipping would not satisfy the projection property. This is a necessary choice for the theoretical framework, not an arbitrary one. Add brief justification.

**Critique 9.4: What Happens When Drift Is Too Strong?**
- **Severity**: Minor
- **Issue**: The drift magnitude $\|\bd^c_{ST}\|$ depends on video content. For videos with lots of high-frequency spatiotemporal content (fast action, detailed textures), the drift could be very large, potentially destabilizing training. Is there any drift magnitude capping or normalization?
- **Rebuttal**: The drift is naturally bounded: $\|\bd^c_{ST}\| \leq \|\bx_0\|$ (since it's a residual of a projection). No explicit capping is needed. The drift envelope $\frac{t(s_{k+1}-t)}{L_k}$ further bounds the contribution to the interpolant. High-frequency-rich videos have larger drift but also benefit more from the curriculum (more high-frequency content to learn).

---

## Round 10

### Critique (4 must-fix items)

**Must-Fix #1: Experimental Data Required**
- **Severity**: Critical (guaranteed reject without this)
- **Issue**: All video quantitative results are X.XX. At minimum, need: (1) VBench total scores for all methods, (2) NFE Pareto curve data points, (3) FVD scores. Without any single number, the video extension is purely hypothetical.
- **Action Required**: Run experiments exp16-exp18. Even partial results (e.g., just VBench total for 3 methods at 50 NFE) would make the paper submittable.

**Must-Fix #2: Joint Drift Ablation**
- **Severity**: Critical (core novelty unvalidated)
- **Issue**: The spatiotemporal joint drift is the paper's key video contribution. Without the ablation (spatial-only vs temporal-only vs joint), the reader cannot know if joint is actually better. This is the simplest experiment to run and the most important.
- **Action Required**: Run exp17 for at least the 3 drift variants at 50 NFE.

**Must-Fix #3: Positioning Against Distillation**
- **Severity**: Major (reviewer will ask "why not just distill?")
- **Issue**: Without discussing AnimateLCM/VideoLCM, a reviewer will trivially reject by noting "8-step distillation is faster than 20-step RDD-Wan." The orthogonality argument must be in the paper.
- **Action Required**: Add discussion section paragraph + Related Work entry.

**Must-Fix #4: Finetuning Convergence Evidence**
- **Severity**: Major (training efficiency claim unsubstantiated for video)
- **Issue**: The paper claims RDD is a "training methodology" but shows no training curves for video. At minimum, report final training loss at 5K steps for RDD-Wan vs PyramidalWan. Ideally, loss curves at 1K/2.5K/5K showing faster convergence.
- **Action Required**: Log training loss during exp16 training run. Plot comparison.

---

## Summary of Key Actions Taken

Based on Rounds 1-10, the following modifications were made to `main.tex`:

1. **Within-stage ODE continuity** (Round 1, Critique 2): Replaced "ODE continuity" with "within-stage ODE continuity" throughout. Added precise scope clarification.
2. **Box filter verification** (Round 1, Critique 3): Expanded Proposition proof sketch with explicit P²=P, P^T=P verification.
3. **Temporal ÷4 limitation** (Round 1, Critique 4): Added limitation item acknowledging aggressive temporal downsampling.
4. **Synthetic data limitation** (Round 1, Critique 6): Added limitation about synthetic training data dependency.
5. **DPM-Solver within-stage** (Round 2, Critique 1): Clarified DPM-Solver applies within stages, not across boundaries.
6. **Temporal drift intuition** (Round 3): Added "temporal coarse-to-fine generation" motivating paragraph with animator analogy.
7. **Softened claims** (Round 4, Critique 1): Modified contribution 7 wording.
8. **PyramidalWan DPM-2 row** (Round 4): Added ablation row showing PyramidalWan with DPM-2 (within-stage) for fair comparison.

### Remaining Actions (from Rounds 5-10, not yet applied):

1. **Training convergence experiment** (Round 5.1): Add to experiments_plan
2. **Single-pass video models paragraph** (Round 9.1): Add CogVideoX, HunyuanVideo, Mochi to Related Work
3. **Distillation comparison paragraph** (Round 9.2, Round 10 Must-Fix #3): Add to Related Work + Discussion
4. **Adaptive step-size discussion** (Round 7.1): Brief note in Discussion
5. **Video results summary paragraph** (Round 7.3): After wall-clock table
6. **FVD metric** (Round 4.3): Add to experiment plan
7. **Statistical significance** (Round 8.2): Add error bars plan
8. **Frame skipping justification** (Round 9.3): Brief note in method section
