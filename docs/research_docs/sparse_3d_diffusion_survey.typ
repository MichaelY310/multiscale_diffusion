// Sparse Multi-Resolution 3D Diffusion — Complete Survey
// Compiled with: python3 -c "import typst; typst.compile('sparse_3d_diffusion_survey.typ', output='sparse_3d_diffusion_survey.pdf')"

#set text(font: ("New Computer Modern", "Noto Sans CJK SC"), size: 10.5pt)
#set page(margin: (x: 2cm, y: 2.5cm))
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.65em)
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(0.5em)
  it
  v(0.3em)
}

#align(center)[
  #text(size: 24pt, weight: "bold")[Sparse Multi-Resolution 3D Diffusion]
  #v(0.4em)
  #text(size: 13pt, style: "italic")[A Comprehensive Survey of 3D Representations\
  Compatible with RDD's Constant Upsampling Property]
  #v(0.8em)
  #text(size: 11pt)[YANG, YUAN (杨远) #h(3em) July 2026]
  #v(0.3em)
  #text(size: 10pt, fill: rgb("#555"))[NeurIPS / ICLR Candidate Research Direction]
]

#v(1.5em)
#outline(indent: 1.5em, depth: 3)

= Executive Summary

== Core Thesis

RDD (Residual Drift Diffusion) 的 *constant upsampling*（最近邻上采样）使得每个 voxel 的上采样值 _仅取决于自身_，不依赖邻居。这意味着：被判定为空的 voxel 可以在上采样时被直接 _丢弃_，不影响其余 voxel 的正确性。这使得 diffusion 的计算复杂度从 $O(N^3)$（dense volume）降为 $O(k)$（$k$ = 非空 voxel 数，对典型3D形状 $k ~ N^2 << N^3$）。

本调研的核心目标：找到可以被映射为 $N times N times N$ 网格的3D表示，使其天然适配 RDD 的 sparse multi-resolution diffusion 框架。

== Key Finding: Top 3 Representations

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  [*方法*], [*核心优势*], [*RDD 兼容性*],
  [GaussianCube (NeurIPS'24)], [3DGS → OT → $32^3 times 14$ grid; 已证明 diffusion 可行], [★★★★★],
  [XCube / VDB (CVPR'24)], [原生稀疏 octree; $1024^3$; 工业级 VDB 后端], [★★★★★],
  [K-Planes / TensoRF-VM], [平面因子分解 $O(N^2)$ 内存; 隐式稀疏], [★★★★☆],
)

== Project Architecture Vision (ASCII)

```
RDD Sparse 3D Generation Pipeline
═══════════════════════════════════════════════════════
TRAINING:
  3D Asset ──► Encoder (Sparse VAE) ──► Latent Grid N³×C
                                              │
                                    [Sparsify: mark empty voxels]
                                              │
                                    Sparse Latent (surface voxels only)
                                              │
                                    RDD Forward (multi-scale drift)
                                              │
                                    Train Sparse 3D U-Net / DiT

GENERATION (Multi-Resolution):
  t=T: 8³ dense noise
    │
    ▼ RDD reverse (coarse)
  Clean 8³ coarse structure
    │
    ▼ Constant Upsample → 16³
    │── Identify empty (density < ε)
    │── DISCARD empty ← (RDD enables this!)
    │
    ▼ RDD reverse (medium, sparse 16³)
    │
    ▼ Constant Upsample → 64³ → prune → sparse
    │
    ▼ RDD reverse (fine, sparse 64³)
    │
  Output: Sparse GaussianCube / SDF → Render / Mesh
═══════════════════════════════════════════════════════
```

= Background: RDD's Constant Upsampling Property

== Mathematical Foundation

=== 标准 RDD Forward Process

$ q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) dot x_0 + sqrt(overline(alpha)_t) dot kappa(t) dot bold(d), quad (1-overline(alpha)_t) dot bold(I)) $

其中 drift term:
$ bold(d) = "UP"("DOWN"(x_0)) - x_0 $

- $"DOWN": N^3 -> (N\/r)^3$，average pooling
- $"UP": (N\/r)^3 -> N^3$，*nearest-neighbor replication*（constant）

=== 3D Extension

对于 voxel grid $x_0 in RR^(N times N times N times C)$:
$ bold(d)[i,j,k] = overline(x)_0[floor(i\/r), floor(j\/r), floor(k\/r)] - x_0[i,j,k] $

其中 $overline(x)_0$ 是 $r times r times r$ block 的均值。

=== Sparsity Theorem (核心洞察)

*定理*：若 $x_0$ 在某个 $r times r times r$ block 内所有值为 0，则：
- $"DOWN"(x_0)$ 在该 block 对应位置为 0
- $"UP"("DOWN"(x_0))$ 在该 block 所有位置为 0
- $bold(d) = 0 - 0 = 0$（drift 为零）
- Forward process: $q(x_t | x_0) = cal(N)(0, (1-overline(alpha)_t) bold(I))$
- Reverse process: 该 block 仍收敛到 0

*结论*：空 block 在整个 diffusion 正反过程中保持为零——稀疏计算是 _精确_ 的，非近似。

== Why Bilinear Upsampling Breaks Sparsity

双线性插值的上采样：
$ "UP"_"bilinear"(x)[i] = sum_j w_(i j) dot x[j], quad w_(i j) != 0 "for neighbors" $

即使 $x[j] = 0$，只要邻居 $x[k] != 0$，则 $"UP"_"bilinear"(x)[j] != 0$。空区域的边界会被 "污染"。这意味着使用双线性上采样时，不能安全地丢弃空 voxels。

RDD 的 constant upsampling 完全避免了这个问题。

= Category 1: Sparse Voxel Diffusion (直接相关)

== XCube (CVPR 2024 Highlight) — NVIDIA

*完整标题*: "XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies"\
*会议*: CVPR 2024 (Highlight)\
*ArXiv*: 2312.03806\
*PDF*: `papers_3d_sparse_diffusion/XCube_CVPR2024.pdf`

=== 3D Representation: VDB Sparse Voxel Hierarchy

*VDB (Volumetric Dynamic B+tree)* 是 OpenVDB 工业标准：
- 4 层浅宽树结构：Root (Hash Table) → InternalNode1 ($32^3$ branching) → InternalNode2 ($16^3$) → LeafNode ($8^3$ voxels)
- 每个 LeafNode 存储 $8^3 = 512$ 个 voxel 值
- *内存效率*：只分配非空区域的节点
- *随机访问*：$O(1)$ 平均时间
- *支持的属性*：TSDF、法向量、颜色、语义标签等

*表示定义*：
$ cal(G) = (V, A) $
- $V$：活跃 voxel 的坐标集合 $V = {(i,j,k) : "occupied"}$
- $A$：每个活跃 voxel 的属性向量

*层次结构*：$cal(G) = {G_1, G_2, ..., G_L}$，从粗到细。

=== Architecture: Two-Stage Pipeline

*Stage 1: Sparse Structure VAE*

Encoder:
- 输入：$(G, A)$ — 稀疏 voxel grid + 属性
- 使用 *fVDB* 后端实现 3D sparse convolution
- 下采样：通过 stride-2 sparse conv 逐级降低分辨率
- 输出：latent $X in RR^(N_l times D)$（$N_l$ = 活跃 latent voxel 数, $D=8$）
- KL regularization weight: $beta = 0.03$

Decoder (Structure Prediction):
- *Neural Kernel-inspired* subdivision mask 预测
- 输入 latent $X$ → 预测每个 voxel 是否需要进一步细分
- 渐进式恢复：从 latent 分辨率逐步预测更细层级的活跃 mask
- Upsampling: 每个活跃父节点 → 预测 $2^3=8$ 个子节点中哪些活跃

VAE Loss:
$ cal(L)_"VAE" = EE_q [underbrace(log p(G,A|X), "reconstruction")] - beta dot underbrace(D_"KL" (q(X|G,A) || p(X)), "regularization") $

*Stage 2: Hierarchical Voxel Latent Diffusion*

为每个分辨率层级训练 *独立* 的 diffusion model：

Level 1 ($16^3$, *dense*):
- 标准 3D U-Net (ADM-style)
- 生成粗糙结构：哪些大区域有物体
- 无稀疏性利用（$16^3$ 足够小）

Level 2 ($128^3$, *sparse*):
- 条件：Level 1 的输出特征（concat 或 cross-attention）
- *Sparse 3D U-Net*：仅在 Level 1 预测为活跃的区域计算
- 使用 fVDB 的 sparse convolution kernel

Level 3 ($512^3$ 或 $1024^3$, *sparse*):
- 条件：Level 2 的输出
- 更高分辨率的 sparse 3D U-Net
- 仅在 surface 附近生成精细几何

每层的 diffusion loss：
$ cal(L)_"diff"^((l)) = EE_(t, epsilon) [||epsilon - epsilon_theta^((l)) (x_t^((l)), t, underbrace(c^((l-1)), "coarser level output"))||^2] $

=== Key Technical Details

*fVDB Backend*:
- NVIDIA 自研的 GPU-native VDB 实现
- 比 TorchSparse 快 3×，内存 0.5×
- 支持：sparse conv, sparse attention, sparse pooling
- Kernel: 3D $3 times 3 times 3$ sparse convolution

*Sparse Convolution 实现*:
- 输入：$(V_"active", F_"active")$ — 仅活跃 voxel 的坐标和特征
- 构建 neighbor hash map
- 对每个活跃 voxel，仅与其活跃邻居做卷积
- 输出：新的活跃 voxel 特征

=== Experimental Results

*ShapeNet*:
#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  [*Category*], [*1-NNA CD↓*], [*1-NNA EMD↓*], [*COV↑*],
  [Chair], [49.8], [51.2], [62.3],
  [Airplane], [48.5], [49.1], [58.7],
  [Car], [50.3], [51.8], [55.2],
)
(优于或持平所有 baseline: LION, 3DShape2VecSet, SDFusion)

*Waymo Open Dataset*:
- 场景大小：100m × 100m
- Voxel size: 10cm → 有效分辨率 $1000^3$
- 生成时间：< 30 秒 (including all levels)
- 生成质量：IoU > 0.85 vs ground truth

*Scalability*:
- $1024^3$ 分辨率仅需 ~16GB GPU 内存（thanks to sparsity）
- Active voxels: ~2M out of $10^9$ total (0.2% occupancy)

=== RDD Compatibility Analysis

#rect(stroke: 1.5pt, inset: 10pt)[
*兼容性: ★★★★★*

#table(
  columns: (auto, auto),
  stroke: 0pt,
  [Grid 结构], [VDB 天然是 $N^3$ 稀疏网格],
  [Constant Upsampling], [VDB 支持任意 nearest-neighbor upsample],
  [稀疏性], [核心设计原则，0.2% 占有率],
  [多分辨率], [层次结构天然支持],
)

*XCube 的局限 (RDD 可以改进的点)*:
- 使用 *cascade*（每层独立 diffusion model）→ 引入 error propagation
- 不同层级之间无 weight sharing → 参数量大
- RDD 的 *单链 drift* 可以将所有层级统一为一个模型
- RDD 的数学保证（空区域 drift=0）提供了比 XCube 更强的稀疏性理论基础
]

== OctFusion (2024)

*完整标题*: "OctFusion: Octree-based Diffusion Models for 3D Shape Generation"\
*ArXiv*: 2408.14732\
*PDF*: `papers_3d_sparse_diffusion/OctFusion_2024.pdf`

=== Core Innovation: Unified Multi-Scale U-Net

与 XCube 的 cascade 不同，OctFusion 使用 *单个统一 U-Net* 同时处理所有 octree 层级。

*Octree 表示*:
- Depth $d$: 分辨率 $2^d$（depth 6 = $64^3$, depth 7 = $128^3$, depth 8 = $256^3$）
- 每个叶节点存储特征向量（SDF 值 + 法向量 + 颜色）
- 非叶节点仅作为结构索引

*U-Net 结构*:
- 所有 octree depth 共享同一组 weight
- Encoder: octree depth 8 → 7 → 6 → 5（通过 octree pooling）
- Decoder: depth 5 → 6 → 7 → 8（通过 octree unpooling + prediction）
- Skip connections 在对应 depth 之间

*Weight Sharing 的关键*:
- 相同的 conv kernel 应用于不同分辨率
- 通过 depth embedding 区分不同层级
- 显著减少参数量：仅 *33M*

=== Mathematical Formulation

所有层级共享同一时间步：
$ x_t^((l)) = sqrt(overline(alpha)_t) dot x_0^((l)) + sqrt(1-overline(alpha)_t) dot epsilon^((l)), quad forall l in {1,...,L} $

统一 loss（加权求和）:
$ cal(L) = sum_(l=1)^L w_l dot EE_(t,epsilon) [||epsilon^((l)) - epsilon_theta (x_t^((1:L)), t)||^2] $

其中 $w_l$ 根据层级深度调整（deeper = 更精细 = 较小权重）。

*关键区别*：$epsilon_theta$ 接收 *所有层级* 的 noisy input 并同时预测所有层级的 noise。

=== 实验结果

- *FID* (shading-image based): 所有 ShapeNet 类别平均最优
- *参数量*: 33M (vs XCube ~300M+, TRELLIS ~2B)
- *生成速度*: 50 DDIM steps, 2.5 秒 (RTX 4090)
- *分辨率*: 最高 $256^3$
- *质量*: 连续流形保证（octree 天然保证 watertight）

=== RDD Compatibility: ★★★★★

*OctFusion 是 RDD-3D 的最直接参考架构*:
- 单模型处理所有分辨率 ← RDD 的核心设计
- Weight sharing across scales ← 直接适用于 RDD
- 共享时间步 ← 与 RDD 的单链 noise schedule 完美对齐
- *RDD 额外贡献*：提供数学保证证明空区域可精确丢弃（OctFusion 无此保证）

== TRELLIS (2024) — Microsoft Research

*完整标题*: "Structured 3D Latents for Scalable and Versatile 3D Generation"\
*ArXiv*: 2412.01506\
*PDF*: `papers_3d_sparse_diffusion/TRELLIS_2024.pdf`

=== SLAT: Structured LATent Representation

- *核心思想*：在稀疏 voxel 结构上定义 local latent vectors
- 只在活跃（表面附近）的 voxel 位置存储 latent code
- 每个 latent code 编码该位置的局部几何和纹理

*序列化*：将稀疏 voxel 的活跃位置序列化为 token 序列：
$ "tokens" = {(z_i, "pos"_i) : i in "active voxels"} $

=== Two-Stage Generation

*Stage 1: Structure Generation*
- 生成二值占据网格（哪些 voxel 活跃）
- 类似 XCube 的 structure prediction
- 输出：binary mask $M in {0,1}^(N times N times N)$

*Stage 2: Content Generation*
- 条件于 Stage 1 的结构
- 生成每个活跃 voxel 的 latent vector
- Rectified Flow (straight-line ODE paths)

=== Architecture: Rectified Flow Transformer

- *Backbone*: Transformer with 3D shifted-window attention
- *模型规模*: 最大 2B 参数
- *训练数据*: 大规模策划的 3D 资产
- *Attention pattern*: 3D shifted window → 局部到全局信息流

*Rectified Flow* 定义:
$ d x_t = v_theta(x_t, t) d t $
$ "Training": cal(L) = EE_(t, x_0, x_1) [||v_theta(x_t, t) - (x_1 - x_0)||^2] $
$ x_t = (1-t) x_0 + t x_1 $

=== 实验结果

- *TRELLIS.2*: $512^3$ 3秒, $1024^3$ 17秒, $1536^3$ 60秒 (H100)
- 当前 SOTA 的几何和纹理质量
- 支持：text-to-3D, image-to-3D
- 多种输出格式：mesh, 3DGS, NeRF

=== RDD Compatibility: ★★★★★

- SLAT 的 "sparse local latents" 直接映射到 RDD 的 sparse grid
- 两阶段可被 RDD 的多分辨率单链统一
- Rectified Flow + sparse attention 是 RDD backbone 的选项
- Shifted-window attention 提供高效的局部信息交换

*RDD 改进方向*: 将 structure + content 统一为 RDD 的一个 diffusion process（structure 由 drift 隐式编码）

== Direct3D-S2 (2025)

*完整标题*: "Gigascale 3D Generation Made Easy with Spatial Sparse Attention"\
*ArXiv*: 2505.17412\
*PDF*: `papers_3d_sparse_diffusion/Direct3D_S2_2025.pdf`

=== Spatial Sparse Attention (SSA) — 核心贡献

*问题*：标准 attention 的 $O(n^2)$ 复杂度对于 3D（tokens 数量巨大）不可行。

*SSA 设计*:
- 将 3D 空间分为 non-overlapping windows
- 每个 window 内做 dense attention
- 跨 window 通过 shift 或 dilated pattern 通信
- *关键*：只对 non-empty voxel tokens 计算 attention

*实现细节*:
- 24-layer DiT blocks
- Hidden dimension: 1024
- Grouped-Query Attention (GQA): group=2, 16 heads per group
- *加速比*: 3.9× over dense attention (same quality)
- 训练：仅需 8 GPU 即可 train $1024^3$

=== 与 RDD 的结合

SSA 可以直接作为 RDD-3D sparse denoiser 的 attention 机制:
- 输入：sparse voxel tokens（只有非空位置）
- 位置编码：3D coordinates
- 时间条件：timestep embedding
- 输出：predicted noise for each token

= Category 2: Grid-Structured 3D Representations

== GaussianCube (NeurIPS 2024) — 关键方法

*完整标题*: "GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling"\
*ArXiv*: 2403.19655\
*PDF*: `papers_3d_sparse_diffusion/GaussianCube_NeurIPS2024.pdf`

=== 动机

3D Gaussian Splatting (3DGS) 的问题：
- Gaussian 数量不固定（几千到几百万）
- 空间分布无序（无固定拓扑）
- 无法直接用 CNN 或标准 diffusion 处理

*GaussianCube 的解决方案*: 通过 Optimal Transport 将无序 Gaussians 映射到固定 $N^3$ 网格。

=== Step 1: Density-Constrained 3DGS Fitting

标准 3DGS 优化，但加入 *硬性数量约束*:
- $N_"max" = N^3$（如 $N=32 => 32768$ Gaussians）
- 训练过程中正常 densify/prune
- 若 Gaussian 数量 > $N_"max"$: prune opacity 最低的 Gaussians
- 训练 30K iterations

每个 Gaussian $G_i$ 有 14 个属性通道:
$ G_i = (underbrace(mu_i, "position" in RR^3), underbrace(s_i, "scale" in RR^3), underbrace(q_i, "rotation" in RR^4), underbrace(alpha_i, "opacity" in RR^1), underbrace(c_i, "color" in RR^3)) $

=== Step 2: Optimal Transport Structuralization

定义 $N^3$ 个网格中心点 $g_1, ..., g_(N^3) in RR^3$（均匀分布在 $[-1,1]^3$）。

求解 Linear Assignment Problem (LAP):
$ pi^* = arg min_(pi in Pi_(N^3)) sum_(i=1)^(N^3) ||mu_i - g_(pi(i))||_2^2 $

*含义*：找到 Gaussian → Grid cell 的一一映射，最小化总位移距离。

*算法*: Jonker-Volgenant algorithm with segmented approximation（大规模 LAP 的高效解法）。

=== Step 3: Final GaussianCube

$ "GaussianCube" in RR^(N times N times N times 14) $

每个 grid cell 存储被分配到该位置的 Gaussian 的 *全部属性*:
- Position 通道存储 *offset*: $Delta mu = mu - g_(pi(i))$（相对于 grid center 的残差）
- Scale, rotation, opacity, color 直接存储

=== Diffusion Model

*Architecture*: ADM-style 3D U-Net
- 所有 2D 操作替换为 3D 版本（Conv3D, Attention3D, Up/DownSample3D）
- Full attention at $8^3$ and $4^3$ resolutions
- 输入/输出: $32 times 32 times 32 times 14$
- 1000-step diffusion, cosine noise schedule

*条件*: 类别标签（ShapeNet）或图像（OmniObject3D, via CLIP）

=== Experimental Results

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [*表示方法*], [*维度*], [*PSNR↑*], [*LPIPS↓*], [*表示参数量*],
  [Dense Voxel (SDF)], [$128^3 times 32$], [26.77], [0.1103], [67.1M],
  [Triplane], [$256^2 times 3 times 32$], [29.41], [0.0721], [6.29M],
  [*GaussianCube*], [$32^3 times 14$], [*34.94*], [*0.0347*], [*0.46M*],
)

*核心发现*:
- 比 dense voxel 高 *8.17 dB* PSNR，参数少 *146×*
- 比 triplane 高 *5.53 dB* PSNR，参数少 *14×*
- 可直接渲染（Gaussian Splatting），无需 neural renderer

*Generation FID (ShapeNet)*:
- Chair: 6.82 (vs SDFusion 10.75, 3DShape2VecSet 8.34)
- Airplane: 5.44
- Car: 4.91

=== RDD Compatibility: ★★★★★ (最佳候选)

#rect(stroke: 2pt, inset: 12pt, fill: rgb("#f0fff0"))[
*GaussianCube 可能是 RDD-3D 最理想的表示*:

+ *完美 $N^3$ 网格*: 标准 3D 卷积直接适用
+ *天然稀疏信号*: opacity $alpha_i$ 提供 per-voxel 的 "是否非空" 判断
+ *Constant upsample 兼容*: $16^3 -> 32^3$ 只需将每个 voxel 复制为 $2^3$ 个子 voxel
+ *无需 neural decoder*: 直接 Gaussian Splatting 渲染
+ *已验证可行*: 已有成功的 3D U-Net diffusion 结果
+ *极其紧凑*: 460K params/shape，训练友好

*唯一局限*: 目前最大 $32^3$。RDD 的 sparse 机制可以将其扩展到 $64^3$, $128^3$。
]

== Plenoxels (CVPR 2022)

*完整标题*: "Plenoxels: Radiance Fields without Neural Networks"\
*ArXiv*: 2112.05131

=== Representation

每个 voxel corner 存储:
$ v_(i j k) = (sigma_(i j k), bold(k)_(i j k)) in RR^(1 + 27) $

- $sigma$: volume density
- $bold(k) in RR^(9 times 3 = 27)$: degree-2 spherical harmonic (SH) coefficients (每个颜色通道 9 个)

颜色计算:
$ c(bold(d)) = "sigmoid"(sum_(l=0)^2 sum_(m=-l)^l k_l^m Y_l^m (bold(d))) $

三线性插值获取任意点的属性。

=== 多分辨率与稀疏性

- Coarse-to-fine: $128^3 -> 256^3 -> 512^3$（resize + prune + optimize）
- *Periodic pruning*: 每 N 步删除 $sigma < epsilon$ 的 voxels
- 典型占有率：5–20%
- 使用 sparse storage (只存非空 voxels 的 indices + values)

=== 实验结果

- *训练时间*: 11 min on single V100（vs NeRF 数小时）
- *质量*: PSNR 31.71 (Synthetic), 27.59 (Real Forward-Facing)
- *无 MLP*: 纯显式优化，极快

=== RDD Compatibility: ★★★★☆

- 天然 $N^3$ 网格，28 通道
- 原生稀疏（periodic pruning）
- Coarse-to-fine 与 RDD 多分辨率一致
- *挑战*: 高分辨率内存大（$512^3 times 28$ dense ≈ 14GB）→ RDD sparse 正好解决

== K-Planes (CVPR 2023)

*完整标题*: "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance"\
*ArXiv*: 2301.10241

=== Mathematical Formulation

3D 表示为 3 个平面的 *Hadamard product*:
$ f(x,y,z) = f_(X Y)(x,y) circle.tiny f_(X Z)(x,z) circle.tiny f_(Y Z)(y,z) $

多尺度版本（$S$ 个空间尺度）:
$ f(bold(q)) = "concat"[product_(c in {X Y, X Z, Y Z}) "interp"(P_c^((s)), "proj"_c(bold(q)))]_(s=1)^S $

其中 $P_c^((s)) in RR^(N_s times N_s times M)$ 是第 $s$ 个尺度、第 $c$ 个平面的特征图。

=== Key Property: Implicit Sparsity via Hadamard Product

$ "If" quad f_(X Y)(x,y) = 0 quad "then" quad f(x,y,z) = 0 quad forall z $

任何一个平面在某投影位置为零 → 整个 3D volume 在对应行/列为零。

*这提供了天然的稀疏机制*：无需显式 sparse 数据结构！

=== 转换为 $N^3$ Grid

$ "Grid"[i,j,k] = P_(X Y)[i,j] circle.tiny P_(X Z)[i,k] circle.tiny P_(Y Z)[j,k] $

- 无需 MLP，纯显式计算
- 内存: $3 times S times N^2 times M$ (vs dense $N^3 times C$)
- 典型设置: $N=256, M=32, S=4$: ≈ 25M params ≈ 100MB

=== RDD Compatibility: ★★★★☆

#rect(stroke: 1pt, inset: 8pt)[
*创新方向: 在 2D plane factors 上直接运行 RDD diffusion*

- Upsample 2D planes (nearest-neighbor) 直接等价于 volume upsample
- 内存: $O(N^2)$ 而非 $O(N^3)$
- 可复用现有 2D diffusion 架构！
- RDD constant upsample 在 2D 上同样有效

*局限*: 无法直接从 2D planes 识别 3D sparse regions（需要计算 Hadamard product 后判断）
]

== TensoRF VM Decomposition (ECCV 2022)

*完整标题*: "TensoRF: Tensorial Radiance Fields"\
*ArXiv*: 2203.09517\
*PDF*: `papers_3d_sparse_diffusion/TensoRF_ECCV2022.pdf`

=== Vector-Matrix (VM) Decomposition

将 3D 张量分解为向量和矩阵的外积之和:
$ cal(T) approx sum_(r=1)^(R_1) bold(v)_r^X circle.small bold(M)_r^(Y Z) + sum_(r=1)^(R_2) bold(v)_r^Y circle.small bold(M)_r^(X Z) + sum_(r=1)^(R_3) bold(v)_r^Z circle.small bold(M)_r^(X Y) $

其中:
- $bold(v)_r^X in RR^I$: 沿 X 轴的向量
- $bold(M)_r^(Y Z) in RR^(J times K)$: YZ 平面的矩阵

*查询点 $(x,y,z)$ 的值*:
$ cal(T)(x,y,z) = sum_(r=1)^(R_1) v_r^X (x) dot M_r^(Y Z)(y,z) + sum_(r=1)^(R_2) v_r^Y (y) dot M_r^(X Z)(x,z) + sum_(r=1)^(R_3) v_r^Z (z) dot M_r^(X Y)(x,y) $

=== 关键数据

- 内存: $O(N^2)$，不到 75MB (vs Plenoxels 数 GB)
- 质量: 与 Plenoxels/NeRF++ 持平 at $300^3$
- 训练: ~30 min

=== RDD Compatibility: ★★★★☆

- 可重建任意 $N^3$ grid 供 diffusion 使用
- Factored 结构允许高效 upsample (2D plane + 1D vector 各自 upsample)
- 紧凑到可以作为 latent space

= Category 3: Voxel/Grid-Based 3D Diffusion

== SDFusion (CVPR 2023)

*ArXiv*: 2212.04493\
*PDF*: `papers_3d_sparse_diffusion/SDFusion_CVPR2023.pdf`

=== Pipeline

+ *Input*: Truncated SDF on $64^3$ voxel grid ($C=1$)
+ *VQ-VAE Compression*: $64^3 -> 8^3 times 256$ (8× per axis downsampling)
+ *Latent Diffusion*: 3D U-Net denoiser on $8^3 times 256$ latent
+ *Conditioning*: Image (CLIP), Text (CLIP), Partial shape (concat)

=== 关键技术

- TSDF 截断距离: 3 voxels
- VQ codebook: 1024 entries, dim 256
- U-Net: ~100M params, 1000 steps DDPM
- 支持多模态条件生成和形状补全

=== RDD Compatibility: ★★☆☆☆

- Dense $64^3$ 卷积，无稀疏利用
- 固定分辨率，无 multi-resolution
- *可借鉴*: VQ-VAE 的 latent 压缩策略

== DiffRF (CVPR 2023)

*ArXiv*: 2212.01206\
*PDF*: `papers_3d_sparse_diffusion/DiffRF_CVPR2023.pdf`

=== 核心创新: Rendering-Guided Denoising Loss

标准 DDPM loss 之外，额外加入 *渲染监督*:
$ cal(L) = cal(L)_"DDPM" + lambda dot cal(L)_"render" $
$ cal(L)_"render" = ||hat(I)(hat(V)_0) - I_"gt"||^2 $

其中:
- $hat(V)_0 = f_theta(V_t, t)$: 从 noisy volume 预测的 clean volume
- $hat(I)(dot)$: differentiable volume rendering
- $I_"gt"$: ground truth 多视角图像

=== Pipeline

- 直接在 $128^3 times 4$ voxel grid 上 diffusion (density + RGB)
- 3D U-Net denoiser (2D ADM → 3D)
- 无 latent compression

=== RDD Compatibility: ★★☆☆☆ (dense，但 rendering loss 可借用)

*重要 technique for RDD-3D*: Rendering-guided loss 可以在 RDD 的 3D 生成中使用，用 2D 图像监督 3D 一致性。

== VoxGRAF (NeurIPS 2022)

*ArXiv*: 2206.07695\
*PDF*: `papers_3d_sparse_diffusion/VoxGRAF_NeurIPS2022.pdf`

=== Progressive Sparse Growing

```
z (latent) ──► Dense 32³ feature grid (StyleGAN generator)
                      │
                      ▼ 3D sparse conv
                Grow to 64³
                      │── Free Space Pruning (discard empty)
                      │
                      ▼ 3D sparse conv
                Grow to 128³
                      │── Free Space Pruning
                      │
                      ▼
                Volume Rendering ──► 2D image ──► 2D Discriminator
```

=== Free Space Pruning

- 在每次 grow 后，用一个轻量 density predictor 判断哪些 voxels 为空
- 空 voxels 被标记为 inactive（不参与后续计算）
- 典型剩余比例: 15-25%

=== RDD Compatibility: ★★★★☆

*VoxGRAF 的 progressive growing + free-space pruning 正是 RDD 所实现的范式*（但 VoxGRAF 用 GAN，RDD 用 diffusion）。直接将 GAN 替换为 RDD 就是一个可行方案。

= Category 4: Differentiable Mesh Extraction (Output Decoders)

== FlexiCubes (SIGGRAPH 2023)

*ArXiv*: 2308.05371

=== 原理: 可微分 Dual Marching Cubes

输入: $N^3$ SDF grid + per-vertex/per-edge 可学习参数

每个 cube 有:
- 8 个角点 SDF 值 $s_1,...,s_8$
- 8 个权重 $alpha_1,...,alpha_8 in RR_(gt 0)$（控制 isosurface crossing 位置）
- 12 个 split 参数 $beta_1,...,beta_(12) in [0,1]$（控制四面体化方式）

*Edge crossing 位置*:
$ u_e = (s(x_i) alpha_i dot x_j - s(x_j) alpha_j dot x_i) / (s(x_i) alpha_i - s(x_j) alpha_j) $

=== RDD Integration

RDD 生成 sparse SDF grid → FlexiCubes 提取 mesh。End-to-end differentiable。

== DMTet (NeurIPS 2021)

*ArXiv*: 2111.04276

=== Deep Marching Tetrahedra

- 四面体网格（从 regular cube 细分）
- 每个顶点: SDF 值 $s_i$ + deformation $Delta v_i$
- Mesh vertex 位置:
  $ p_(i j) = (|s_j| dot (v_i + Delta v_i) + |s_i| dot (v_j + Delta v_j)) / (|s_i| + |s_j|) $
- *自适应细分*: 只细化包含 surface 的四面体（= RDD 的 non-empty voxels）
- 分辨率: base $128^3$, 自适应到 $256^3$

== Get3D (NeurIPS 2022)

*ArXiv*: 2209.11163

=== Progressive Feature Volume → DMTet

- StyleGAN2 backbone 生成 feature volume: $4^3 -> 8^3 -> 16^3 -> 32^3 times 64$
- 三线性插值到 DMTet 顶点位置
- MLP 预测 SDF + deformation
- DMTet 提取可微 mesh

*Progressive $4^3 -> 32^3$ feature volume* 与 RDD 的多分辨率阶段直接类比。

= Category 5: Other Important References

== 3DShape2VecSet (SIGGRAPH 2023)

*ArXiv*: 2301.11445\
*PDF*: `papers_3d_sparse_diffusion/3DShape2VecSet_SIGGRAPH2023.pdf`

- Cross-attention encoder: point cloud → fixed-size latent set ($M$ vectors)
- Diffusion on latent set
- Cross-attention decoder: latent set + query points → occupancy
- *RDD Compat: ★★☆☆☆* (token set，非 grid 结构)

== SSDNeRF (ICCV 2023)

*ArXiv*: 2304.06714\
*PDF*: `papers_3d_sparse_diffusion/SSDNeRF_ICCV2023.pdf`

- 在 triplane latent 上做 diffusion
- Triplane: $3 times 64^2 times 32$
- 单图→3D 或无条件生成
- *RDD Compat: ★★★☆☆* (triplane 可转 $N^3$，但 indirect)

== SPVD: Sparse-view Voxel Diffusion (2024)

*ArXiv*: 2408.06145

- Sparse-view 3D reconstruction via voxel diffusion
- 直接在 voxel grid 上 denoise
- 少量视角条件生成完整 3D

== L3DG: Latent 3D Gaussian Diffusion (2024)

*ArXiv*: 2410.13530

- 3DGS latent diffusion
- 与 GaussianCube 类似但用 latent compression

= Comprehensive Comparison Table

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [*方法*], [*表示*], [*Grid*], [*Sparse*], [*Multi-Res*], [*RDD*], [*Max*],
  [XCube'24], [VDB], [Native $N^3$], [Native], [Hierarchy], [★5], [$1024^3$],
  [OctFusion'24], [Octree], [Native $N^3$], [Native], [Unified], [★5], [$256^3$],
  [TRELLIS'24], [SLAT], [Native $N^3$], [Native], [2-stage], [★5], [$1536^3$],
  [*GaussianCube'24*], [*OT-3DGS*], [*$N^3 times 14$*], [*Opacity*], [*Extend*], [*★5*], [*$32^3$+*],
  [Direct3D-S2'25], [Sparse DiT], [Native], [Native], [—], [★4], [$1024^3$],
  [VoxGRAF'22], [Sparse vox], [Native], [Pruning], [Grow], [★4], [$128^3$],
  [Plenoxels'22], [SH voxel], [Native $N^3$], [Pruning], [C2F], [★4], [$512^3$],
  [K-Planes'23], [3 planes], [Indirect], [Implicit], [Multi-S], [★4], [$512^3$],
  [TensoRF-VM'22], [V-M decomp], [Indirect], [Implicit], [Yes], [★4], [$300^3$],
  [Get3D'22], [Feat. vol], [Native], [No], [Prog.], [★3], [$32^3$],
  [SDFusion'23], [Dense SDF], [Native], [No], [No], [★2], [$64^3$],
  [DiffRF'23], [Dense RF], [Native], [No], [No], [★2], [$128^3$],
  [3DShape2VecSet], [Token set], [No], [—], [No], [★2], [N/A],
  [SSDNeRF'23], [Triplane], [Indirect], [No], [No], [★3], [$64^2$],
  [MeshGPT'24], [1D seq], [No], [—], [No], [★1], [N/A],
)

= Novelty Claim & Positioning

== Core Novelty Statement

#rect(stroke: 2pt, inset: 14pt, fill: rgb("#f8f8ff"))[
*论文核心贡献*: 我们发现 RDD 的 constant (nearest-neighbor) upsampling 创造了 3D 生成的独特机遇：由于上采样值 _独立于邻居_，空 voxels 可以被识别并 _精确丢弃_——无信息损失。

这将 $O(N^3)$ 的 dense 体素 diffusion 转化为 $O(k)$ 的 sparse generation，其中 $k$ 是表面 voxel 数（$k prop N^2 << N^3$）。

*数学保证*: 在 RDD 中，若一个 block 的 voxels 具有相同值（constant region），drift $bold(d) = "UP"("DOWN"(x_0)) - x_0 = 0$。空区域在整个 forward/reverse 过程中保持为零——sparse 计算是 _exact_ 的。
]

== Differentiation from Prior Work

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [], [单链], [Constant UP], [精确稀疏], [统一模型],
  [XCube], [✗ (cascade)], [N/A], [✗ (heuristic)], [✗],
  [OctFusion], [✓], [✗ (learned)], [✗ (approx)], [✓],
  [TRELLIS], [✗ (2-stage)], [N/A], [✗], [✗],
  [Cascade DDPM], [✗], [✗ (bilinear)], [✗], [✗],
  [*RDD-3D (ours)*], [*✓*], [*✓*], [*✓*], [*✓*],
)

== Expected Contributions (for NeurIPS/ICLR)

+ *理论贡献*: 证明 constant upsampling + drift correction 在 3D 中实现 exact sparse diffusion
+ *架构贡献*: 首个统一的 sparse multi-resolution 3D diffusion model
+ *实验贡献*: 在 ShapeNet/Objaverse 上展示与 SOTA 持平的质量 + 显著的速度/内存优势
+ *实用价值*: 使 $256^3$ 以上的 3D 生成在单 GPU 上可行

= Detailed Action Plan

== P0: Proof of Concept (4–6 weeks)

=== Task 1: RDD-GaussianCube Baseline
- Clone GaussianCube repo (data preparation pipeline)
- 实现 3D RDD forward process:
  - Input: $32^3 times 14$ GaussianCube
  - DOWN: $32^3 -> 16^3$ (average pooling 3D)
  - UP: $16^3 -> 32^3$ (nearest-neighbor)
  - drift: $bold(d) = "UP"("DOWN"(x_0)) - x_0$
- 训练 3D U-Net denoiser (dense, 作为 baseline)
- 评估: FID, PSNR vs GaussianCube 原始 ADM

=== Task 2: Sparse Masking Validation
- 生成 coarse level 后，threshold opacity: mask = $alpha > epsilon$
- 验证：discarding masked voxels 后生成质量保持
- 测量：实际稀疏比 (target: 70–90% empty)
- Ablation: threshold $epsilon$ = 0.01, 0.05, 0.1

=== Task 3: Upsampling Ablation
- Constant (nearest) vs Bilinear vs Learned upsampling
- 在 $16^3 -> 32^3$ 阶段对比
- 验证 RDD drift 补偿可以弥补 constant upsample 的质量差距

== P1: Sparse Acceleration (4–6 weeks)

=== Task 4: Sparse 3D U-Net
- 选择 backend: MinkowskiEngine / fVDB / TorchSparse
- 替换 dense 3D conv 为 sparse conv（仅在非空 voxel）
- 在 medium/fine resolution 使用 sparse ops
- 测量: FLOPs 减少 + wallclock speedup

=== Task 5: High Resolution Scaling
- Multi-stage RDD: $16^3 -> 32^3 -> 64^3 -> 128^3$
- 每次 upsample 后 prune（sparse 比例逐级增加）
- 验证: wallclock time 线性于 occupied voxels（非立方于分辨率）

=== Task 6: Sparse Attention (Optional)
- 实现 Direct3D-S2 的 Spatial Sparse Attention
- 替代方案: 3D shifted-window attention (TRELLIS style)
- 对比: Sparse U-Net vs Sparse DiT backbone

== P2: Full System & Paper (6–8 weeks)

=== Task 7: Alternative Representations
- Test 1: Sparse SDF grid + FlexiCubes decoder
- Test 2: K-Planes factors + RDD 2D diffusion
- 对比: quality, speed, memory

=== Task 8: Conditioning
- Image: DINOv2/CLIP encoder → cross-attention
- Text: CLIP text → cross-attention
- Partial shape: concat sparse features

=== Task 9: Paper
- 核心贡献定位: "Sparse Multi-Resolution 3D Diffusion via Constant Upsampling"
- Key ablation: $gamma=0$ (no drift) vs full RDD vs cascade
- Comparison: vs XCube, OctFusion, TRELLIS, GaussianCube baseline
- Target: NeurIPS 2026 或 ICLR 2027

= References

+ XCube: Ren et al., "XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies," CVPR 2024. arXiv:2312.03806
+ OctFusion: Xiong et al., "OctFusion: Octree-based Diffusion Models for 3D Shape Generation," 2024. arXiv:2408.14732
+ TRELLIS: Xiang et al., "Structured 3D Latents for Scalable and Versatile 3D Generation," 2024. arXiv:2412.01506
+ GaussianCube: Zhang et al., "GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling," NeurIPS 2024. arXiv:2403.19655
+ Direct3D: Wu et al., "Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer," 2024. arXiv:2405.14832
+ Direct3D-S2: "Gigascale 3D Generation Made Easy with Spatial Sparse Attention," 2025. arXiv:2505.17412
+ 3DShape2VecSet: Zhang et al., SIGGRAPH 2023. arXiv:2301.11445
+ Michelangelo: Zhao et al., NeurIPS 2023. arXiv:2306.17115
+ SDFusion: Cheng et al., CVPR 2023. arXiv:2212.04493
+ DiffRF: Mueller et al., CVPR 2023. arXiv:2212.01206
+ VoxGRAF: Schwarz et al., NeurIPS 2022. arXiv:2206.07695
+ SSDNeRF: Chen et al., ICCV 2023. arXiv:2304.06714
+ TensoRF: Chen et al., ECCV 2022. arXiv:2203.09517
+ K-Planes: Fridovich-Keil et al., CVPR 2023. arXiv:2301.10241
+ Plenoxels: Fridovich-Keil et al., CVPR 2022. arXiv:2112.05131
+ DVGO: Sun et al., CVPR 2022. arXiv:2111.11215
+ EG3D: Chan et al., CVPR 2022. arXiv:2112.07945
+ FlexiCubes: Shen et al., SIGGRAPH 2023. arXiv:2308.05371
+ DMTet: Shen et al., NeurIPS 2021. arXiv:2111.04276
+ Get3D: Gao et al., NeurIPS 2022. arXiv:2209.11163
+ NKSR: Huang et al., CVPR 2023. arXiv:2305.19590
+ MeshGPT: Siddiqui et al., ICLR 2024. arXiv:2311.15475
+ PVD: Zhou et al., ICCV 2021. arXiv:2104.03670
+ L3DG: Lan et al., 2024. arXiv:2410.13530
+ SPVD: 2024. arXiv:2408.06145
+ fVDB: Williams et al., 2024. arXiv:2407.01781
+ NSVF: Liu et al., NeurIPS 2020. arXiv:2007.11571
+ Instant-NGP: Müller et al., SIGGRAPH 2022. arXiv:2201.05989
+ ConvONet: Peng et al., ECCV 2020. arXiv:2003.04618
+ PlenOctrees: Yu et al., ICCV 2021. arXiv:2103.14024
+ Scaffold-GS: Lu et al., CVPR 2024 Highlight. arXiv:2312.00109
+ Octree-GS: Ren et al., 2024. arXiv:2403.17898
+ SCube: Ren et al., 2024. arXiv:2410.20030
+ TRELLIS2 (O-Voxel): Xiang et al., 2025. arXiv:2512.14692
+ Sparc3D: Li et al., 2025. arXiv:2505.14521
+ FLUX3D: ANU/ByteDance, 2025. arXiv:2606.24874
+ ULTRA3D: Chen et al., 2025. arXiv:2505.02610
+ Scaffold Diffusion: 2025. arXiv:2509.00062
+ Hyper3D: 2025. arXiv:2503.10403

= Category 6: Structured Representations for 3D — Implicit, Explicit & Hybrid

本章扩充调研范围，覆盖除 sparse voxel diffusion（Category 1）和 GaussianCube（Category 2）之外的、对 RDD Sparse 3D Diffusion 有启发意义的结构化 3D 表征。

== 6.1 Neural Sparse Voxel Fields (NSVF) — Liu et al., NeurIPS 2020

*核心思想：* Sparse Voxel Octree + Per-vertex Learnable Embeddings + NeRF-style Volume Rendering.

*方法细节：*
- 场景被组织为一棵 octree，每个叶节点是一个 voxel（初始 $16^3$ 或 $32^3$）
- 每个 voxel 的 8 个顶点各存储一个 32D learnable feature embedding $bold(f)_i in RR^(32)$
- 光线与 voxel 相交时，通过 trilinear interpolation 得到 query point 的特征：$bold(f)(bold(x)) = sum_(i=1)^8 w_i bold(f)_i$
- 该特征送入共享 MLP 解码得到 $sigma$ 和 color $bold(c)$
- *Self-Pruning*：训练中定期检测 voxel 内所有采样点的 $sigma$，若最大值低于阈值则删除该 voxel
- *Progressive Training*：分阶段将 voxel 细分（subdivision），从粗到精增加分辨率
- 最终结果：在 Synthetic-NeRF 数据集上，NSVF 仅需约 10% 的空间就达到与 dense grid 相当的质量

*与 RDD 的适配性分析（3/5 stars）：*
- 优点: Octree 天然稀疏，与 RDD 的"空 voxel drift=0"语义一致
- 优点: Self-pruning 策略可直接借鉴为 RDD denoising 过程中的"占据判定"
- 优点: Progressive subdivision 完美对应 RDD 的 constant upsampling（coarse to fine）
- 缺点: Per-vertex embedding + MLP 是隐式表示，不如 GaussianCube 的显式 14-channel 直接
- 缺点: 每次 query 需要 MLP forward pass，不适合直接做 diffusion loss

*借鉴价值：* Self-pruning 判据和 progressive subdivision scheduling 可移植到 RDD sparse pipeline。

== 6.2 Instant-NGP (Multi-Resolution Hash Encoding) — Mueller et al., SIGGRAPH 2022

*核心思想：* L 层多分辨率 hash grid，每层独立映射坐标到 feature vector，替代 MLP 的大部分容量。

*方法细节：*
- 共 $L=16$ 层，分辨率从 $N_(min)=16$ 到 $N_(max)=2048$，按几何级数增长
- 每层维护一个大小为 $T$ 的 hash table（$T = 2^(19)$），每条目存 $F=2$ 维特征
- 对于任意 3D 坐标 $bold(x)$，在第 $l$ 层：找到包含它的体素的 8 个顶点、hash function 映射、查表、trilinear interpolation
- 所有 $L$ 层的特征 concatenate 后送入一个极小的 MLP（2 层 64 宽）
- Hash 冲突通过梯度优化自然解决（高频细节区域获得更多梯度）
- 训练速度：5 秒即可达到可接受质量，远快于 vanilla NeRF 的数小时

*与 RDD 的适配性分析（3/5 stars）：*
- 优点: Multi-resolution 天然对应 RDD 的多分辨率结构
- 优点: Hash table 的 flat structure 支持 O(1) 访问和 GPU 并行
- 优点: 分辨率几何增长 = constant factor upsampling，与 RDD 2x upsample 一致
- 缺点: Hash collision 使得表示不是唯一确定的（non-injective），diffusion 去噪可能不稳定
- 缺点: 仍需 MLP decoder，不能直接渲染
- 缺点: 每层 hash table 大小固定 $T$，无法动态稀疏化

*借鉴价值：* Multi-resolution geometric growth + flat hash table 的思路可用于设计 RDD 的多尺度 feature volume。

== 6.3 Convolutional Occupancy Networks (ConvONet) — Peng et al., ECCV 2020

*核心思想：* 3D Feature Volume + 3D U-Net encoder + point-wise implicit decoder。将全局 MLP 替换为局部卷积特征。

*方法细节：*
- Input: 点云 / partial scan
- Encoder: 将输入离散化到 $N^3$ 的 regular grid，运行 3D U-Net 得到 feature volume $bold(F) in RR^(N times N times N times C)$
- Decoder: 对任意 query point $bold(x)$，通过 trilinear interpolation 从 $bold(F)$ 中提取局部特征，再通过小型 MLP 预测 occupancy $o(bold(x)) in [0,1]$
- 与 OccNet 的关键区别：特征是局部的（卷积诱导的 translation equivariance），而非全局 latent code
- 支持三种 feature 组织方式：3D Volume / Triplane / Points-based
- 在 ShapeNet 和大场景（Synthetic Room）上显著优于 OccNet

*与 RDD 的适配性分析（4/5 stars）：*
- 优点: $N^3$ feature volume 完美适配 RDD 的 grid 结构
- 优点: 3D U-Net 可直接作为 RDD 的 denoising network backbone
- 优点: 卷积的 translation equivariance 与 RDD 的 constant upsample 兼容
- 优点: Feature volume 可以天然稀疏化（空区域 feature = 0）
- 缺点: 仍依赖 MLP decoder 做最终 occupancy 预测
- 缺点: Dense $N^3$ volume 在高分辨率时内存爆炸（需要 sparse conv 改造）

*借鉴价值：* ConvONet 的 3D U-Net + trilinear query 架构是 RDD 3D denoiser 的理想候选 backbone。结合 sparse convolution 可扩展到高分辨率。

== 6.4 PlenOctrees — Yu et al., ICCV 2021

*核心思想：* 将训练好的 NeRF 预计算为 Octree，每个叶节点直接存储 density $sigma$ 和 SH coefficients，实现 150+ FPS 实时渲染。

*方法细节：*
- 先训练一个 NeRF-SH 模型（输出 SH 系数而非 view-dependent color）
- 从训练好的 NeRF-SH 中采样 octree：对每个 voxel 检查中心点 $sigma$，若 $sigma gt tau$ 则保留并细分
- 叶节点存储：$sigma in RR$ + 3 阶 SH 系数 x 3 color channels = $1 + 27 = 28$ floats/voxel
- 可进一步直接优化 octree 上的值（不经过 MLP），类似 fine-tuning
- 最终表示：纯显式 octree，无需任何神经网络推理即可渲染

*与 RDD 的适配性分析（4/5 stars）：*
- 优点: 纯显式表示——每个 voxel 存储 28 floats，无 MLP，可直接做 diffusion
- 优点: Octree 天然稀疏，只存储非空节点
- 优点: 28 channel/voxel 与 GaussianCube 的 14 channel/voxel 类似，都是显式可微渲染
- 优点: Octree 的层级结构 = multi-resolution，完美适配 RDD 的逐级生成
- 缺点: Octree 的不规则拓扑对 3D convolution 不友好（vs regular grid）
- 缺点: SH 表示的色彩空间较 Gaussian Splatting 更受限

*借鉴价值：* "显式 octree + per-voxel attributes (density + SH)" 的范式证明了 tree-structured explicit representation 可以完全替代 MLP-based NeRF。其 28-float/voxel 格式与 RDD diffusion target 高度契合。

== 6.5 Scaffold-GS — Lu et al., CVPR 2024 Highlight

*核心思想：* 在 voxelized anchor grid 上分布 3D Gaussians，每个 anchor 通过 MLP 动态预测其关联 Gaussians 的属性。

*方法细节：*
- 从 SfM 点云出发，voxelize 为规则网格，voxel 中心作为 anchor point
- 每个 anchor 存储：32D feature $bold(f)$, position $bold(x)$, 6D scaling, $k=10$ 个 offset directions, opacity
- 渲染时：3 个小 MLP（opacity/covariance/color）以 $(bold(f), bold(d)_"view", d_"dist")$ 为输入，动态生成 $k$ 个 Neural Gaussians
- 只有预测 opacity 大于 0 的 Gaussians 被实际光栅化
- Growing/Pruning：根据 Gaussian 重要性动态增删 anchor

*与 RDD 的适配性分析（3/5 stars）：*
- 优点: Voxelized anchor grid 提供了规则的 $N^3$ 结构
- 优点: Anchor 的稀疏分布（只在有表面的地方）与 RDD sparse 一致
- 优点: Growing/Pruning 策略可借鉴为 RDD 的占据判定
- 缺点: 需要 MLP 推理才能得到 Gaussian 属性，不如 GaussianCube 的直接存储
- 缺点: Anchor 分布不规则（来自 SfM），不适合 regular grid diffusion

*借鉴价值：* "Anchor + Neural Gaussian" 双层架构启发了 "coarse occupancy grid + fine detail" 的 RDD 两阶段生成。

== 6.6 SCube / VoxSplat — Ren et al., 2024

*核心思想：* Hierarchical voxel latent diffusion + feedforward Gaussian prediction，用于大规模场景重建。是 XCube 的直接后续。

*方法细节：*
- *VoxSplat 表示*：$1024^3$ sparse voxel grid 作为骨架，每个 occupied voxel 内放置一组 3D Gaussians
- *生成流程*：Hierarchical latent diffusion model，从 coarse ($64^3$) 逐级生成到 fine ($1024^3$)
  - Stage 1: 生成粗 occupancy + latent feature
  - Stage 2: Feedforward appearance network 预测每个 voxel 内的 Gaussian 属性
- *条件输入*：仅需 3 张不重叠的 posed images
- *输出*：数百万 Gaussians + skybox texture

*与 RDD 的适配性分析（5/5 stars）：*
- 优点: Hierarchical coarse-to-fine = RDD 的 constant upsampling 逐级细化
- 优点: Sparse voxel + per-voxel Gaussians = GaussianCube 的自然扩展到大场景
- 优点: Latent diffusion on voxel grid = 与 RDD 的架构完全兼容
- 优点: 已证明可扩展到 $1024^3$
- 缺点: 是场景重建而非生成，diffusion 用于 occupancy/latent 而非完整属性
- 缺点: 需要 appearance network（非纯 diffusion）

*借鉴价值：* SCube 证明了 "hierarchical sparse voxel latent diffusion" 在极高分辨率($1024^3$)下的可行性，是 RDD-3D 最直接的架构参考。其 VoxSplat = 稀疏体素骨架上的 Gaussian Splatting，与我们的 "RDD on GaussianCube" 思路几乎完全一致。

== 6.7 TRELLIS2 / O-Voxel — Xiang et al., 2025

*核心思想：* Omni-Voxel (O-Voxel) = field-free 稀疏体素表示，同时编码 geometry + full PBR appearance；配合 Sparse Compression VAE (SC-VAE) 实现 16x 下采样压缩。

*方法细节：*
- *O-Voxel 表示*：每个 occupied voxel 存储：
  - Geometry: SDF 值 + vertex displacements（类似 FlexiCubes）
  - Appearance: Base Color (3) + Metallic (1) + Roughness (1) + Alpha (1) = 6 channels
  - 共约 $10+$ channels/voxel
- *SC-VAE (Sparse Compression VAE)*：
  - 直接在 sparse voxel 上运行 sparse 3D convolution
  - 实现 16x spatial downsampling（$1024^3 -> 64^3$ latent）
  - 约 9.6K latent tokens for $1024^3$ input
  - Sparse Residual Autoencoding：先编码 coarse structure，再逐级编码残差
- *Diffusion*：4B parameter Rectified Flow Transformer on latent tokens
- *性能*：$512^3$ 生成仅需 3 秒，$1024^3$ 仅需 17 秒（H100）

*与 RDD 的适配性分析（5/5 stars）：*
- 优点: O-Voxel 是显式 per-voxel 属性，无 MLP decoder，直接适合做 diffusion target
- 优点: SC-VAE 的 sparse residual encoding 与 RDD 的 residual drift 概念高度吻合
- 优点: 16x downsampling 到 latent space 后做 diffusion = RDD latent diffusion 的自然形式
- 优点: 支持 full PBR（不仅 RGB），表达力极强
- 优点: Field-free——不依赖 implicit function，直接从 voxel extract mesh
- 缺点: SC-VAE 训练复杂度高（4B model on 4M data）
- 缺点: 稀疏模式需要额外生成（occupancy mask 单独一阶段）

*借鉴价值：* TRELLIS2 是当前最接近"完美 RDD-3D target representation"的方案。其 O-Voxel（显式 sparse voxel + per-voxel PBR）与 SC-VAE（sparse residual compression）几乎为 RDD 量身定制。

== 6.8 Sparc3D — Li et al., 2025

*核心思想：* SparseCubes（稀疏可变形 Marching Cubes 表示）+ SparConv-VAE（全 sparse convolution 的 VAE），实现 near-lossless 3D mesh 压缩和 $1024^3$ 分辨率生成。

*方法细节：*
- *SparseCubes 表示*：
  - 仅在表面附近的 voxel 存储 SDF 值 + vertex deformation
  - 通过稀疏 Marching Cubes 直接提取 watertight mesh
  - 与 FlexiCubes 类似但更稀疏（仅 surface band）
- *SparConv-VAE*：
  - 第一个完全基于 sparse 3D convolution 的 VAE encoder-decoder
  - 支持 $1024^3$ 输入，16x downsampling
  - Near-lossless: 重建误差极低
  - 模态一致：encoder 和 decoder 都直接操作 sparse voxel（无 2D-3D 转换损失）
- *生成*：Latent diffusion on compressed latent space
- *性能*：4x faster training than TRELLIS2，Chamfer Distance / F1 SOTA

*与 RDD 的适配性分析（5/5 stars）：*
- 优点: SparseCubes 直接 = sparse $N^3$ grid，完美适配 RDD
- 优点: SparConv-VAE 的 sparse conv 架构可直接用作 RDD denoiser
- 优点: Near-lossless 意味着 VAE bottleneck 不会丢失细节
- 优点: 仅存储 surface-band voxels = 极致稀疏（$O(N^2)$ surface vs $O(N^3)$ volume）
- 优点: Watertight mesh 输出，适合下游应用
- 缺点: 仅做 geometry（SDF），不包含 appearance

*借鉴价值：* Sparc3D 的 "SparseCubes + SparConv-VAE" 提供了 geometry 分支的最优解。若结合 O-Voxel 的 appearance 分支，可构建完整的 RDD-3D 表示。

== 6.9 FLUX3D — ANU/ByteDance, 2025

*核心思想：* Diffusion-Aligned Structured Latents (DA-SLAT) + Sparse-structure Multimodal DiT (SMDiT)。解决 sparse voxel latent 与 2D image token 的跨模态对齐问题。

*方法细节：*
- *表示学习改进 (DA-SLAT)*：
  - 用预训练 diffusion model（FLUX）的 2D features 替代 DINOv2，保留高频纹理信息
  - Decoder-only architecture 解码 sparse voxel latents 为 3DGS
  - 更好的 3DGS 重建保真度
- *生成改进 (SMDiT)*：
  - Sparse-structure Multimodal Diffusion Transformer
  - Modal-Aware Rotary Positional Embedding (MARoPE)：为 2D image tokens 和 3D voxel tokens 分别设计位置编码
  - 实现 geometry-agnostic 的 2D-3D alignment
- *输出*：高保真 3D Gaussian Splatting assets

*与 RDD 的适配性分析（4/5 stars）：*
- 优点: 基于 sparse voxel latent，与 RDD 兼容
- 优点: SMDiT 的 sparse structure-aware attention 可直接借鉴
- 优点: MARoPE 的多模态位置编码对 RDD 的条件生成有参考价值
- 优点: 证明了 diffusion features 优于 discriminative features 做 3D reconstruction
- 缺点: 仍依赖两阶段（occupancy + latent），non-end-to-end
- 缺点: DA-SLAT 的 decoder-only 架构训练成本高

*借鉴价值：* FLUX3D 的核心洞察——"用 diffusion model features 做 3D representation learning 比 DINOv2 更好"——对 RDD 的 encoder 设计有直接指导意义。

== 6.10 Scaffold Diffusion — 2025

*核心思想：* 将 sparse multi-category voxel 视为 token，使用 Masked Discrete Diffusion Language Model (MDLM) 生成 3D 结构。

*方法细节：*
- 针对 Minecraft 房屋等 sparse categorical voxel structures（大于 98% sparsity）
- Voxel 视为 discrete token（类别 = block type + empty）
- 使用 MDLM（Masked Diffusion Language Model）的 discrete diffusion 框架
- 加入 3D positional encoding 保证空间连贯性
- 不需要 VAE，直接在 voxel token space 做 diffusion

*与 RDD 的适配性分析（2/5 stars）：*
- 优点: 证明了 discrete diffusion 可处理大于 98% sparsity 的 3D 结构
- 优点: 直接在 voxel space 做 diffusion，无 VAE 损失
- 缺点: Discrete diffusion 而非 continuous，与 RDD 的 SDE formulation 不兼容
- 缺点: 针对 categorical data（block types），非连续几何属性

*借鉴价值：* 作为对比参照：证明极致稀疏(98%)下的 3D diffusion 是可行的，但 discrete 路线与 RDD 的 continuous SDE 正交。

== 6.11 Hyper3D — 2025

*核心思想：* Hybrid Triplane + Octree Feature 的 3D VAE，结合两者优势：triplane 的效率 + octree 的几何细节保留。

*方法细节：*
- 不再从 mesh surface 均匀采样构造 VAE input
- 使用 pre-trained octree feature model 将 mesh 信息编码为多尺度 octree features
- 结合 triplane representation 做高效的 latent encoding
- 3D VAE 重建质量显著优于纯 triplane 或纯 octree 方法

*与 RDD 的适配性分析（3/5 stars）：*
- 优点: Hybrid triplane + octree 的思路可用于设计 RDD 的 latent space
- 优点: Octree features 天然稀疏
- 缺点: Triplane 的 2D 结构与 RDD 的 $N^3$ grid 不完全匹配
- 缺点: 仍是 implicit (需要 query decoder)

*借鉴价值：* 多表示融合的思路：可以在 RDD 中用 triplane 做 coarse-level conditioning，octree/voxel 做 fine-level diffusion target。

= Updated Comprehensive Comparison Table

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  [*Method*], [*Representation*], [*Sparsity*], [*Explicit?*], [*Resolution*], [*RDD Score*],
  [GaussianCube], [$N^3 times 14$ grid (OT)], [Yes (opacity)], [Yes], [$32^3$--$64^3$], [5/5],
  [O-Voxel/TRELLIS2], [Sparse voxel + PBR], [High], [Yes], [$1024^3$], [5/5],
  [Sparc3D], [SparseCubes + SDF], [High], [Partial], [$1024^3$], [5/5],
  [SCube/VoxSplat], [Sparse voxel + GS], [High], [Partial], [$1024^3$], [5/5],
  [XCube], [VDB sparse hierarchy], [High], [Partial], [$1024^3$], [4/5],
  [OctFusion], [Octree multi-scale], [Medium], [Partial], [$512^3$], [4/5],
  [ConvONet], [$N^3$ feature volume], [Low], [Implicit], [$128^3$], [4/5],
  [PlenOctrees], [Octree + 28 floats/leaf], [Medium], [Yes], [Adaptive], [4/5],
  [FLUX3D], [DA-SLAT sparse voxel], [Medium], [Partial], [High], [4/5],
  [Direct3D-S2], [Sparse latent + SSA], [Medium], [Partial], [$512^3$], [4/5],
  [TRELLIS v1], [SLAT sparse local], [Medium], [Implicit], [$1536^3$], [4/5],
  [NSVF], [Sparse octree + embed], [Medium], [Implicit], [Adaptive], [3/5],
  [Instant-NGP], [Multi-res hash grid], [Low], [Implicit], [$2048^3$], [3/5],
  [Scaffold-GS], [Voxel anchor + MLP GS], [Low], [Implicit], [Adaptive], [3/5],
  [Hyper3D], [Triplane + Octree VAE], [Low], [Implicit], [Medium], [3/5],
  [Scaffold Diffusion], [Discrete voxel tokens], [High], [Yes], [$64^3$], [2/5],
  [TensoRF-VM], [Vector-Matrix decomp], [Low], [Yes], [$300^3$], [3/5],
  [K-Planes], [3 planes Hadamard], [Low], [Partial], [$256^3$], [2/5],
)

*Legend:* Sparsity = supports sparse storage; Explicit = no MLP decoder needed for rendering; RDD Score = compatibility with RDD Sparse 3D Diffusion framework.

= Updated Candidate Ranking for RDD-3D

== Tier 1: Direct Integration Candidates（可直接作为 RDD target）

+ *GaussianCube* — $N^3 times 14$ 显式 grid，OT 结构化，opacity 天然稀疏信号。RDD constant upsample 直接适用。首选。
+ *O-Voxel (TRELLIS2)* — 显式 sparse voxel + full PBR，SC-VAE 的 sparse residual encoding 与 RDD residual drift 概念天然吻合。可扩展到 $1024^3$。
+ *Sparc3D SparseCubes* — 纯 geometry 方案，near-lossless SparConv-VAE，$1024^3$。若结合 appearance 可达 full pipeline。
+ *SCube VoxSplat* — Hierarchical sparse voxel + per-voxel GS，已验证 coarse-to-fine latent diffusion 可达 $1024^3$。架构与 RDD 最接近。

== Tier 2: Strong Architectural References（架构可深度借鉴）

+ *XCube* — Hierarchical VDB latent diffusion 的开创者，cascade 架构参考。
+ *OctFusion* — 单 U-Net 多尺度 octree diffusion，33M 参数轻量级方案。
+ *ConvONet* — 3D Feature Volume + U-Net backbone 的经典范式。
+ *PlenOctrees* — 28-float/voxel 纯显式 octree，证明了 tree-structured explicit 表示的可行性。
+ *FLUX3D* — Diffusion-aligned sparse latent + 跨模态对齐。

== Tier 3: Conceptual Inspiration（概念启发）

+ *NSVF* — Self-pruning + Progressive Subdivision 策略。
+ *Instant-NGP* — Multi-resolution hash encoding + geometric growth。
+ *Scaffold-GS* — Voxelized anchor + dynamic Gaussian prediction 双层架构。
+ *Hyper3D* — Triplane + Octree hybrid VAE 的多表示融合。
+ *Scaffold Diffusion* — 极致稀疏(98%)下 discrete 3D diffusion 的可行性证明。

= Representation Evolution Timeline (2020--2025)

```
2020: ConvONet (ECCV), NSVF (NeurIPS)
2021: PlenOctrees (ICCV), PVD (ICCV)
2022: Instant-NGP (SIGGRAPH), TensoRF (ECCV), Plenoxels (CVPR)
2023: 3DGS (SIGGRAPH), SDFusion (CVPR), K-Planes (CVPR)
2024: GaussianCube (NeurIPS), XCube (CVPR), Scaffold-GS (CVPR),
      OctFusion, SCube/VoxSplat, TRELLIS v1
2025: TRELLIS2/O-Voxel, Sparc3D, FLUX3D, Direct3D-S2,
      ULTRA3D, Scaffold Diffusion, Hyper3D
```

*趋势总结：*
- 2020-2022: Implicit to Hybrid（从纯 MLP 到 feature grid + tiny MLP）
- 2022-2023: Hybrid to Explicit（Plenoxels, 3DGS 证明无需 MLP）
- 2024-2025: Explicit + Sparse + Diffusion（GaussianCube, O-Voxel, Sparc3D, SCube）
- *关键洞察*：领域正在收敛到 "sparse voxel grid + explicit per-voxel attributes + latent diffusion" 范式，这与 RDD-3D 的设计目标完全一致。

= Conclusion and Next Steps

本次扩充调研新增了 13 篇论文/方法的详细分析（共计 30+ 篇覆盖），涵盖：
- 隐式结构化表示（NSVF, Instant-NGP, ConvONet）
- 显式 tree/octree 表示（PlenOctrees, Octree-GS）
- 结构化 Gaussian Splatting（Scaffold-GS, SCube/VoxSplat）
- 最新 sparse voxel diffusion（TRELLIS2, Sparc3D, FLUX3D, ULTRA3D）
- Discrete sparse diffusion（Scaffold Diffusion）
- Hybrid 表示（Hyper3D）

*最终结论：* RDD-3D 的最佳实现路径是：
+ *表示层*：GaussianCube 或 O-Voxel（显式 sparse $N^3$ grid + per-voxel attributes）
+ *压缩层*：SparConv-VAE / SC-VAE（sparse residual encoding，16x 下采样）
+ *Diffusion 层*：RDD on latent space，constant upsample = spatial upsampling in latent
+ *稀疏加速*：opacity/occupancy mask 确定 active voxels，仅对 active voxels 计算 drift
