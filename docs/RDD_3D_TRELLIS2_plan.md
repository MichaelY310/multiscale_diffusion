# RDD-3D 方案：基于 TRELLIS.2 的多分辨率 3D 生成

> **历史方案提醒（2026-08-11）**：本文保留早期构想，但其中“空 voxel drift 天然为零”、直接从 coarse active voxel 展开并删除 children、以及直接平均显式 O-Voxel 属性等表述不够严格。当前可执行且经过相关工作审计的方案见 [`research_docs/RDD_3D_SPARSE_LATENT_EXPERIMENT.md`](research_docs/RDD_3D_SPARSE_LATENT_EXPERIMENT.md)：先由 structure model 给出 fine support，再在 TRELLIS.2 shape SLat 上使用 fixed-support active-child projector 与 detail-subspace flow。

## 1. 动机

RDD (Resolution-Dependent Diffusion) 的核心思想是：在一条连续 ODE/SDE 轨迹中，从粗分辨率逐步演化到细分辨率，空区域 drift=0 跳过计算。
这在 2D 图像生成中已验证有效。现在要将其扩展到 3D。

**为什么选 TRELLIS.2？**
- 每个 voxel 存连续 latent vector（可 average、可加噪）
- 解码完全独立于其他 voxel（Dual Contouring 是确定性几何算法）
- 已验证分辨率越高质量越好（512³ → 1024³ → 1536³）
- CVPR 2026 Best Student Paper，代码开源
- 4B 参数模型 + 800K 数据集，工业级基线

---

## 2. TRELLIS.2 核心机制详解

### 2.1 O-Voxel 表示

O-Voxel 是一种 **field-free（无场）** 的稀疏体素表示。

**每个 active voxel 存储：**

| 字段 | 符号 | 类型 | 含义 |
|------|------|------|------|
| 对偶顶点位置 | v ∈ R³ | float[3] | 该体素内的精确顶点坐标 (范围 [0,1] 相对于体素) |
| 边交叉标志 | δ ∈ {0,1}³ | 3 bits | X/Y/Z 三轴方向的边是否被表面穿过 |
| 分裂权重 | γ ∈ R>0 | float[1] | 四边形→两个三角形的分裂方式 |
| 基础颜色 | c ∈ R³ | float[3] | 线性 RGB |
| 金属度 | m ∈ R | float[1] | PBR 金属度 |
| 粗糙度 | r ∈ R | float[1] | PBR 粗糙度 |
| 不透明度 | α ∈ R | float[1] | 透明度 |

**关键性质：**
- 只有被物体表面穿过的体素才是 active（稀疏性）
- 1024³ 分辨率下，active voxels 通常只有 ~10 万量级（vs 10 亿总体素）
- 不需要任何隐式场（SDF/NeRF/Occupancy），几何信息直接存在 voxel 上

### 2.2 Mesh ↔ O-Voxel 双向转换

**Mesh → O-Voxel（编码，CPU，~几秒）：**
1. 遍历 mesh 所有三角面片
2. 对每条被表面穿过的体素边，记录交点位置和法线
3. 对每个 active voxel，用 QEF（Quadratic Error Function）求解最优对偶顶点位置
4. 从 mesh 纹理采样材质属性（distance-weighted average）

**O-Voxel → Mesh（解码，CUDA，<100ms）：**
1. 每个 active voxel = 一个 mesh 顶点（位置 = v）
2. 对每条被标记为 active 的边（δ=1），连接相邻 4 个 voxel 的顶点 → 形成四边形
3. 按分裂权重 γ 将四边形切成两个三角形
4. 材质：对 mesh 顶点做三线性插值从相邻 voxel 的材质属性中获取

**这就是为什么说"解码独立于其他 voxel"：** 每个 voxel 的顶点位置 v 完全由自己决定，不需要查询其他 voxel 的 latent。连接关系 δ 也是局部的（只看直接邻居）。

### 2.3 SC-VAE（Sparse Compression VAE）

**作用**：把高分辨率 O-Voxel（512³/1024³）压缩为紧凑的 latent space，供扩散模型操作。

**架构**：
- 全稀疏卷积 U-Net（~800M 参数，354M encoder + 474M decoder）
- 16× 空间下采样（1024³ O-Voxel → 64³ latent grid）
- 每个 latent voxel 的维度：32

**SC-VAE 编码器层级结构：**

```
输入: O-Voxel 特征 (1024³, 活跃体素 ~10万)
  ↓ Stage 1: 稀疏卷积 + 残差块 (1024³ → 512³, C=32)
  ↓ Stage 2: 下采样 + 残差块 (512³ → 256³, C=64)  
  ↓ Stage 3: 下采样 + 残差块 (256³ → 128³, C=128)
  ↓ Stage 4: 下采样 + 残差块 (128³ → 64³, C=256)
  ↓ Bottleneck: → latent z ∈ R^32 per active voxel
输出: ~9600 个 32维 latent tokens (64³ grid 中的 active 部分)
```

**Sparse Residual Autoencoding（核心创新）：**
- 下采样时：把 8 个子 voxel 的特征 stack 到通道维度 → group average → 得到粗层特征
- 上采样时：对称地把粗层特征 unstack 回 8 个子 voxel
- 这是**非参数化**的 skip connection，不引入额外可学习参数

**Early-pruning Upsampler：**
- 上采样前先预测哪些子 voxel 是 active 的（binary mask）
- 非 active 的直接跳过 → 大幅节省计算

### 2.4 SC-VAE 训练

**两阶段训练：**

**Stage 1（低分辨率，256³）：**
```
L_s1 = λ_v · MSE(v̂, v)           # 顶点位置
     + λ_δ · BCE(δ̂, δ)           # 边交叉标志
     + λ_ρ · BCE(ρ̂, ρ)           # pruning mask
     + λ_mat · L1(f̂_mat, f_mat)  # 材质属性
     + λ_KL · KL(q(z|x) || N(0,1))  # KL 散度
```

**Stage 2（高分辨率，512³，加入渲染损失）：**
```
L_s2 = L_s1 + L_render

L_render_shape = L1(mask) + 10·L1(depth) + d_p(normal)
L_render_mat   = d_p(base_color) + d_p(metallic_roughness_alpha)

d_p(a, b) = L1(a,b) + 0.2·SSIM + 0.2·LPIPS
```

**两个独立的 SC-VAE：**
- Shape SC-VAE：编码/解码几何 (v, δ, γ)
- Material SC-VAE：编码/解码材质 (c, m, r, α)，以 Shape VAE 的 pruning 结构为条件

### 2.5 三阶段生成 Pipeline

所有生成模型均为 DiT（~1.3B 参数/个，共 4B），使用 Rectified Flow：

```
前向过程: x(t) = (1-t)·x₀ + t·ε   (线性插值)
训练目标: L_CFM = E[||v_θ(x(t), t) - (ε - x₀)||²]
采样: 从 t=1 积分到 t=0，25-50 步
引导: Classifier-Free Guidance (scale ~3-7.5)
时间步采样: logitNorm(1, 1)
```

**Stage 1: Sparse Structure DiT（稠密）**
- 输入：32³ 或 64³ 的全体素噪声
- 输出：binary occupancy（哪些 voxel 是 active 的）
- 条件：DINOv3-L 图像特征（cross-attention）

**Stage 2: Geometry DiT（稀疏）**
- 输入：L 个 32维噪声向量（只在 active voxels 上）
- 输出：L 个 32维 geometry latent z^g
- 条件：图像特征（cross-attention）+ timestep（AdaLN-single）

**Stage 3: Material DiT（稀疏）**
- 输入：L 个 32维噪声 + z^g 拼接（共 64维输入）
- 输出：L 个 32维 material latent z^m
- 条件：同上

**完整推理流程：**
```
图片 → DINOv3 编码 → Stage1(结构) → Stage2(几何latent) → Stage3(材质latent)
     → Shape SC-VAE Decoder(16×上采样) → O-Voxel
     → Material SC-VAE Decoder → PBR 属性
     → Dual Contouring → GLB mesh
```

### 2.6 Test-time Resolution Scaling（级联推理）

论文中从 1024³ 扩展到 1536³ 的方法：
1. 先在 1024³ 生成完整 O-Voxel
2. 对 O-Voxel 做 max-pooling 降到 96³ 的 sparse structure
3. 以此为条件，重新跑 Stage 2（geometry DiT）生成 1536³ 的 O-Voxel

**注意：这不是多分辨率训练，只是 test-time hack。** DiT 本身在固定分辨率上做"平坦"扩散。

---

## 3. RDD 如何嵌入 TRELLIS.2

### 3.1 核心思路

当前 TRELLIS.2 Stage 2 是"所有 active voxels 同时从纯噪声去噪到干净 latent"。
RDD 要改成：**从少量粗 voxels 逐步扩展到全量细 voxels，形成一条连续 ODE 轨迹。**

```
当前 TRELLIS.2 Stage 2:
  t=1 ────────────────────────────────> t=0
  所有 ~9600 个 active voxels 同时去噪（平坦）

RDD 改造后:
  t=1 ──> t=0.75 ──> t=0.5 ──> t=0.25 ──> t=0
  4³       8³        16³       32³        64³
  ~50      ~400      ~2000     ~6000      ~9600
  voxels   voxels    voxels    voxels     voxels
```

### 3.2 多分辨率 Latent 的定义

设 latent grid 有 L 个层级，分辨率分别为 r₁ < r₂ < ... < r_L（如 4³, 8³, 16³, 32³, 64³）。

**粗层 → 细层的上采样规则：**

对于 SC-VAE 的 latent space：
- 粗层级的每个 active voxel（32维向量）对应细层级的 8 个子 voxel
- **上采样** = 父 voxel 的 latent 复制（nearest-neighbor）或 average 到 8 个子 voxel
- **新增的子 voxel**（粗层级为空，细层级变 active）= 从噪声初始化

**drift 定义：**
- 已有 voxel（从粗层继承的）：drift = 细层 clean latent - 粗层上采样 latent
- 新增 voxel：drift = 完整的去噪（等价于普通扩散）
- 粗层已确定为空的区域：drift = 0（不参与任何计算）

### 3.3 具体实现方案

**方案 A：替换 Stage 2 内部为多分辨率调度**

1. 保留 Stage 1 不动（生成最细分辨率的 occupancy structure）
2. 从 Stage 1 的输出中，通过 max-pooling 构建多级 occupancy：
   - active_4³ ⊂ active_8³ ⊂ ... ⊂ active_64³
3. Geometry DiT 的采样过程改为分段：
   - t ∈ [1.0, 0.75]: 只在 active_4³ 的 voxels 上去噪
   - t ∈ [0.75, 0.5]: 扩展到 active_8³，新增 voxels 用噪声+父voxel latent 初始化
   - t ∈ [0.5, 0.25]: 扩展到 active_16³
   - t ∈ [0.25, 0.0]: 扩展到 active_64³（全量）
4. DiT 网络本身不需要大改（已有 RoPE，天然支持不同数量的 token）

**方案 B：完全重新设计为层级生成**

1. 训练多个 SC-VAE 对应不同分辨率（如 128³, 256³, 512³, 1024³）
2. 每个分辨率有对应的 latent space
3. RDD 的连续 ODE 跨越这些层级
4. 优点：更自然；缺点：需要训多个 VAE，工作量大

**推荐方案 A**：改动最小，只需要修改 Stage 2 的采样调度，DiT 权重可以直接复用 TRELLIS.2 的预训练模型。

### 3.4 与 TRELLIS.2 原有级联推理的区别

| | TRELLIS.2 级联推理 | RDD 方案 |
|---|---|---|
| 理论基础 | Test-time hack，无理论保证 | 连续 ODE，有 noise filling 唯一性证明 |
| 分辨率切换 | 生成完 → max-pool → 重新从噪声去噪 | 连续轨迹，无需 renoising |
| 误差累积 | 每次 re-apply 都可能引入新误差 | 单条 ODE 轨迹，误差不累积 |
| 计算效率 | 每个分辨率跑完整 25 步 | 粗层步数少，细层步数多（自适应） |
| 训练方式 | 不需要额外训练 | 需要在多分辨率上 fine-tune |

---

## 4. 论文故事

### 4.1 核心贡献

"我们证明了 Resolution-Dependent Diffusion 可以统一多分辨率 3D 生成为一条连续 ODE 轨迹，
在不增加 FLOPs 的前提下提升高分辨率 3D 资产的生成质量，并消除级联推理中的误差累积。"

### 4.2 卖点

1. **理论贡献**：将 RDD 的 noise filling 理论从 2D 推广到 3D 稀疏体素
2. **实践贡献**：在 TRELLIS.2 的 Stage 2 中，用 RDD 替换平坦采样，零额外训练成本获得质量提升
3. **效率贡献**：粗层级 token 少 → 前期步数计算量小 → 相同总步数下更快

### 4.3 实验设计

- **消融 1**：RDD 渐进采样 vs 平坦采样（固定步数，对比 CD/MD/F-score）
- **消融 2**：不同分辨率调度（4→8→16→32→64 vs 8→32→64 vs 16→64）
- **消融 3**：RDD 连续轨迹 vs TRELLIS.2 原有级联推理（对比误差累积）
- **主实验**：RDD-TRELLIS.2 vs 原版 TRELLIS.2 vs Direct3D-S2 vs OctFusion

---

## 5. 实现路径

### Phase 1: 原型验证（1-2 周）
1. Clone TRELLIS.2 仓库
2. 加载预训练 Stage 2 Geometry DiT
3. 实现多分辨率采样调度器（分段 timestep + token 扩展）
4. 在 50 个测试 prompt 上对比 CD/MD 指标

### Phase 2: 训练 Fine-tune（2-4 周）
1. 在 Objaverse 子集上 fine-tune Stage 2 DiT（加入多分辨率训练）
2. 实现 RDD loss（带 drift 的噪声预测）
3. 对比收敛速度

### Phase 3: 全量实验（2-3 周）
1. 完整 benchmark 对比
2. 消融实验
3. 可视化分析（不同层级的 latent 变化）

---

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| DiT 在 token 数量变化时性能下降 | TRELLIS.2 已有 RoPE，理论上支持可变长度；必要时加 padding |
| 粗层 active voxels 太少（<50），attention 退化 | 最粗层不要低于 8³ |
| SC-VAE latent 做 average 后信息丢失 | average 只是初始化，后续去噪会修正 |
| 与 TRELLIS.2 原版对比不公平（步数不同） | 固定总 NFE 对比 |

---

## 7. 参考文献

- TRELLIS.2: arXiv:2512.14692, Xiang et al., Dec 2025. CVPR 2026 Best Student Paper.
  - 代码: https://github.com/microsoft/trellis.2
- OctFusion: Computer Graphics Forum Vol.44, Xiong et al., 2025. SGP 2025.
  - 代码: https://github.com/octree-nn/octfusion
- Direct3D-S2: arXiv:2505.17412, Wu et al., 2025. NeurIPS 2025.
- XCube: CVPR 2024, Ren et al., NVIDIA.
