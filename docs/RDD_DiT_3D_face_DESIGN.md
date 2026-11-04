# RDD-DiT-3D 完整设计文档
# Vertex 版本 + Face 版本 + Set Transformer

> 最后更新: 2026-07-14
> 涵盖全部已确认的设计决策、已解决的 Bug 和经验教训。

---

## 目录

1. [项目总览](#1-项目总览)
2. [Set Transformer (Vertex 版)](#2-set-transformer-vertex-版)
3. [Face Set Transformer (Face 版)](#3-face-set-transformer-face-版)
4. [RDD-DiT-3D 扩散模型](#4-rdd-dit-3d-扩散模型)
5. [数据预处理流程](#5-数据预处理流程)
6. [训练流程](#6-训练流程)
7. [采样与推理流程](#7-采样与推理流程)
8. [关键设计决策记录](#8-关键设计决策记录)
9. [已解决的 Bug 与经验教训](#9-已解决的-bug-与经验教训)
10. [数据集统计](#10-数据集统计)
11. [文件结构](#11-文件结构)
12. [与 MeshGPT 的对比](#12-与-meshgpt-的对比)

---

## 1. 项目总览

### 1.1 核心思想

将 RDD (Resolution-Dependent Diffusion) 多分辨率扩散框架应用于 3D 形状生成。系统分两个版本：

| | Vertex 版本 | Face 版本 |
|---|---|---|
| 代码路径 | `RDD_DiT_3D/` | `RDD_DiT_3D_face/` |
| 编码单元 | 每个 grid cell 内的**点集** | 每个 grid cell 内的**三角形面片集** |
| AE 输入维度 | 4 (x, y, z, k/K_MAX) | 10 (9坐标 + 1 scale) |
| 集合大小上限 | K_MAX = 10 | F_MAX = 8~24 |
| 解码输出 | 点坐标 [K, 3] | 面片坐标 [F, 9] + existence + scale |
| 后处理 | 无 | merge_vertices (合并重复顶点) |

### 1.2 三阶段训练流程

```
阶段 1: 训练 Set Transformer AE
    输入: mesh/点云 → 按 grid cell 切分 → 变长集合
    输出: 固定维度 embedding (bottleneck_dim)

阶段 2: 离线编码
    用训好的 AE 将所有 mesh 编码为多分辨率稀疏 {embeddings_R, coords_R}

阶段 3: 训练 RDD-DiT 扩散模型
    在稀疏 embedding 空间上进行多分辨率扩散训练
```

### 1.3 全稀疏表示

**不使用稠密 3D volume**（如 `[B, C, R, R, R]`），而是：
- Token 张量: `[B, N_max, C]` — N_max 是 batch 内活跃 token 的最大数
- 坐标张量: `[B, N_max, 3]` — 整数 grid 坐标
- Padding mask: `[B, N_max]` — True = 有效 token

优势：避免 O(R³) 内存开销，大部分 3D 空间是空的。

**纯 Transformer 架构**：无 3D convolution，所有空间信息通过 positional embedding (sincos_from_coords) 注入。

---

## 2. Set Transformer (Vertex 版)

> 代码路径: `set_transformer/train_emd_ae_3d_v2.py`

### 2.1 架构: AEBridgeReconModelV2

```
Input [B, K_MAX, 4]  →  SetEncoder  →  SetAutoencoder (bottleneck)  →  Decoder
                          4层 SAB         Pool → MLP → C dim           重建点坐标
```

### 2.2 SetEncoder

- 输入投影: `Linear(dim_in=4, dim_hidden)`
- 4 层 `MaskedSAB` 自注意力
- 输入特征: `[x, y, z, k/K_MAX]`
  - xyz: 局部归一化坐标
  - k/K_MAX: 点数归一化 (第 k 个点 / 总点数)

### 2.3 SetAutoencoder (Bottleneck Bridge)

**池化选项** (将 [B, K_MAX, d] → [B, d]):
- `mean`: masked 均值池化 (推荐默认)
- `pma1`: PMA 1 个 seed (排列不变)
- `pma4`: PMA 4 seeds → 展平
- `max_mean`: max + mean 拼接
- `concat`: 展平 (破坏排列不变性，已弃用)

**编码 MLP**: pooled → bottleneck_dim (典型 64~256)

**VAE 模式** (可选):
- 输出 mu + logvar
- Reparameterization: z = mu + std * eps
- KL loss: `-0.5 * mean(1 + logvar - mu² - exp(logvar))`

**解码 MLP**: bottleneck_dim → K_MAX × dim_hidden → reshape

### 2.4 Decoder

- Learnable slot queries (区分输出位置)
- MaskedSAB 精炼
- 输出头: Linear → [B, K_MAX, 3] 坐标

### 2.5 排列不变性

- Encoder: MaskedSAB 自注意力 + mean/PMA pool → 对输入集合排列不变
- Decoder: learnable slot_queries 提供固定的"位置编号"，配合 z-x-y 排序的重建 target 进行学习

### 2.6 典型配置

| 参数 | 值 |
|------|---|
| K_MAX | 10 |
| dim_in | 4 |
| dim_hidden | 32 |
| bottleneck_dim | 64~256 |
| num_heads | 4 |
| encoder layers | 4 |

### 2.7 实验结论

8+1 个 v1 实验探索不同 bottleneck 维度和模型大小：
- **Exp1 (bn=128, 深) 综合最优**
- Exp4 (bn=256, 浅) 排序敏感性太高，不推荐用于 diffusion
- Exp9 (enc2/dec2, bn=32) 最小配置

---

## 3. Face Set Transformer (Face 版)

> 代码路径: `RDD_DiT_3D_face/face_set_transformer.py`

### 3.1 整体架构

```
Input: faces [B, F_MAX, 9] + mask [B, F_MAX] + scale [B, 1]
  → FaceSetEncoder (scale 拼接到每个 token 的第 10 维)
    → [B, F_MAX, dim_hidden]
  → FaceSetAutoencoder
    → Pool (mean/attention) → MLP → bottleneck [B, C]
    → MLP → expand → slot_queries → SAB (不传 mask)
    → [B, F_MAX, dim_hidden]
  → FaceSetDecoder
    → coords [B, F_MAX, 9]
    → existence [B, F_MAX] (logits)
    → scale [B, 1] (exp(log_scale))
```

### 3.2 Triangle Soup 表示

每个面片用 9 个浮点数表示 (3 顶点 × 3 坐标)：
```
face = [v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z]
```

**关键特点**：
- 面片自含完整坐标信息，无需顶点索引表
- 相邻面片的共享顶点通过后处理 `merge_vertices` 合并（相距 < threshold 的顶点合并为一个）
- 优势：天然适配集合处理，无需维护复杂拓扑关系

### 3.3 关键设计: Scale 机制

**问题**: 不同 cell 内面片的实际空间范围差异巨大。若只归一化坐标而不记录 scale，则不同大小的面片归一化后产生相同 embedding ("scale 碰撞")。

**解决方案**:
1. **预处理归一化** (`compute_local_coords`):
   ```python
   local = (vertex - cell_origin) / cell_size    # → [0,1]³ 范围内
   centered = local - 0.5                         # → 以 cell 中心为原点
   max_abs = |centered|.max() + 1e-6
   scale = max_abs / 0.5                          # 除以 scale 后最大值恰好 ±0.5
   normalized = centered / scale                  # → [-0.5, 0.5]
   ```

2. **Scale 作为 Encoder 输入**: 拼接为第 10 维 `[B, F_MAX, 10]`

3. **Scale 预测**:
   - Decoder 输出 log_scale (网络输出值)
   - `predicted_scale = exp(log_scale)` → 保证正数
   - Loss: `L1(log(pred_scale), log(target_scale))` (对数空间 L1)

4. **推理还原坐标**:
   ```python
   local_coords = normalized * predicted_scale + 0.5   # → cell 局部 [0,1]
   global_coords = local_coords * cell_size + cell_origin
   ```

### 3.4 Existence Head

每个 slot 输出一个 logit，表示该位置是否有真实面片:
- 训练: `BCE(pred_logits, mask.float())`
- 推理: `sigmoid(logits) > threshold` 筛选有效面片
- 默认 exist_weight = 0.0 (可通过参数开启)

### 3.5 Pooling 方式 (ae_pool)

| 选项 | 实现 | 排列不变性 |
|------|------|-----------|
| `mean` (默认) | masked mean pooling | ✓ |
| `attention` | MaskedPMA (1 seed) | ✓ |

**重要**: 之前的 `concat` 模式已弃用，因为它破坏排列不变性且硬编码了 F_MAX。

### 3.6 训推一致性设计

**核心规则: Decoder 端 (AE dec_sab + FaceSetDecoder) 在训练和推理时都不传 mask。**

原因:
- 推理时 mask 不可知 (扩散输出无 mask 信息)
- 训练时传 mask 但推理时不传 → 行为不一致
- 解决: 训练时也不传 mask，让 Decoder 自主学习处理 padding 位置

### 3.7 面片排序

**输入排序**: 按面片重心的 z→x→y 字典序排序。

原因:
- Set Transformer encoder 是排列不变的 (PMA/mean pool)
- 但 Decoder 需要确定性输出顺序来对齐重建 loss
- 排序替代了复杂的 Hungarian Matching（O(N³) 开销过大）

### 3.8 VAE 模式

- `variational=True` 时启用
- AE encoder MLP 输出 2×bottleneck_dim，分为 mu 和 logvar
- 训练: reparameterization trick + KL loss
- 推理 encode: 仅返回 mu (确定性)
- 高斯先验正则化: 确保 latent space 连续、平滑，利于 diffusion 建模

### 3.9 损失函数

| 损失项 | 公式 | 默认权重 |
|--------|------|---------|
| 重建 L1 | `mean(\|pred - target\|)` on valid faces | 1.0 |
| Existence BCE | `BCE(logits, mask.float())` | 0.0 (param) |
| Scale log-L1 | `L1(log(pred_scale+ε), log(target_scale+ε))` | 1.0 |
| KL | `-0.5 * mean(1 + logvar - mu² - exp(logvar))` | kl_weight (param) |

### 3.10 decode_from_bottleneck

用于采样后从 embedding 重建面片:
```python
def decode_from_bottleneck(self, bottleneck):
    # bottleneck: [B, C]
    dec_out = self.ae.dec_mlp(bottleneck)           # [B, dim_hidden]
    H = expand(B, F_MAX, dim_hidden) + slot_queries # [B, F_MAX, dim_hidden]
    Z = self.ae.dec_sab(H, mask=None)               # 不传 mask!
    coords, existence, scale = self.decoder(Z, mask=None)
    return coords, existence, scale
    # coords: [B, F_MAX, 9]
    # existence: [B, F_MAX] (logits)
    # scale: [B, 1] or None
```

**注意**: decode_from_bottleneck 的返回值必须解包为 `(coords, existence, scale)` 三元组。

### 3.11 encode_cell 接口

```python
def encode_cell(self, faces, mask, scale):
    """
    faces: [B, F_MAX, 9] — 归一化后的面片坐标
    mask:  [B, F_MAX]    — True=有效面片
    scale: [B, 1]        — 该 cell 的 scale 值
    returns: [B, bottleneck_dim] — embedding
    """
```

### 3.12 模型配置

| 配置 | dim_hidden | bottleneck_dim | num_heads | enc/dec layers |
|------|-----------|----------------|-----------|----------------|
| Small | 64 | 64 | 4 | 2/2 |
| Base | 128 | 128 | 4 | 2/2 |
| Large | 256 | 256 | 8 | 3/3 |

---

## 4. RDD-DiT-3D 扩散模型

> 代码路径: `RDD_DiT_3D/model.py`, `RDD_DiT_3D/diffusion_3d.py`
> Face 版本完全复用相同的 model.py 和 diffusion_3d.py

### 4.1 DiT3D_RDD 模型

基于标准 DiT 架构，适配为稀疏 3D:

```
Forward(x_active, coords, t, y, padding_mask):
  x_active: [B, N_max, C]      — 活跃 token (来自 AE 编码)
  coords:   [B, N_max, 3]      — 整数 grid 坐标
  t:        [B]                 — 全局时间步
  y:        [B]                 — 类别标签
  padding_mask: [B, N_max] bool — True=有效

  1. Linear projection: [B, N, C] → [B, N, hidden_size]
  2. + sincos_from_coords(coords) 位置编码
  3. c = TimestepEmbedder(t) + LabelEmbedder(y)
  4. N × DiTBlock (adaLN-Zero: scale, shift, gate)
  5. FinalLayer → [B, N, out_channels]
```

### 4.2 位置编码: sincos_from_coords

- 实时从整数坐标计算 (不预存 buffer)
- `coords / max_resolution` 归一化到 [0, 1)
- 每轴分配 `hidden_size // 3` 维度
- sin/cos 频率编码 (10000 base)
- 优势：分辨率无关，不同阶段用同一份编码逻辑

### 4.3 多分辨率阶段

| 阶段 | 分辨率 | 时间步范围 | Drift | Token 数 (典型) |
|------|--------|-----------|-------|-----------------|
| Stage 0 | 4³ | [0, 400) | 无 | 64 (全网格) |
| Stage 1 | 8³ | [400, 700) | 有 | ~100-300 |
| Stage 2 | 16³ | [700, 1000) | 有 | ~500-2000 |

**注**: Stage 0 可以是全网格（因为 4³=64 token 数很少），后续阶段仅在非空位置生成。

### 4.4 Drift 机制

**公式**: `drift = upsample(downsample(x_0)) - x_0`

物理意义: 高频残差。低频成分通过 downsample→upsample 保留，drift 是被丢弃的高频部分。

**实现** (`compute_drift_sparse`):
1. 下采样坐标: `coords_low = coords // 2`
2. Scatter Add 均值池化 (按低分辨率位置聚合)
3. Gather 回高分辨率位置
4. 残差 = gathered - original

**kappa 调度**: `kappa(t) = t_local / num_timesteps_stage`
- 阶段初期 kappa≈0 (几乎无 drift)
- 阶段末期 kappa≈1 (完全 drift)

### 4.5 前向过程

```
x_t = sqrt(ᾱ_t) * (x_0 + κ_t * drift) + sqrt(1 - ᾱ_t) * noise
```

**训练目标 (noise target)**:
```
target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * κ * drift
```

### 4.6 逆过程 (p_mean_variance)

标准 DDPM 后验 + drift 修正:
```
μ = (1/sqrt_alpha) * (x_t - beta/sqrt_one_minus_alpha_bar * eps_pred) + drift_coef * drift
```

### 4.7 Decouple 模式

模型输出 `[B, N, 2C]`:
- 前 C 通道: epsilon 预测
- 后 C 通道: drift 预测
- 损失: `MSE(eps_pred, noise) + MSE(drift_pred, drift_true)`
- CFG 仅应用于 eps 头

### 4.8 Jump-Step 采样器

实现精确后验 `q(x_s | x_t, x_0)` 的多步跳跃:
- 利用闭式后验均值和方差
- 支持确定性 (DDIM-like) 和随机模式
- 可 10~50x 加速采样

### 4.9 Noise Filling (阶段间过渡)

将粗阶段结果传递到细阶段:
1. 从父体素 gather 嵌入: `x_up = x_clean[parent_indices]`
2. 加噪到目标水平: `x_noisy = sqrt(ᾱ_T) * x_up + sqrt(1-ᾱ_T) * noise`
   - `ᾱ_T` 是细阶段第一步对应的信噪比

### 4.10 空体素检测

在阶段过渡时确定哪些子体素是"空"的（不含任何几何）:
- 方法: 将 denoised x_0 与 empty embedding 计算 cosine similarity
- 辅助: 3D max_pool 膨胀，确保空体素的邻居也被保留
- 高 similarity → 该位置为空 → 在后续阶段跳过

**核心创新**: 低分辨率阶段识别空 patch，后续阶段不在这些位置生成 token，大幅减少计算量。

### 4.11 SpacedDiffusion3D (加速采样)

- 从全局 1000 步中均匀选取子集 (如 80 步)
- 重新计算对应的 betas (保持 alpha_bar 精确匹配)
- 内部维护 timestep_map 进行全局时间步映射

### 4.12 CFG (Classifier-Free Guidance)

- 训练时随机 drop label (替换为 unconditional token)
- 推理: `eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)`
- Decouple 模式下 CFG 仅应用于 eps 头，不应用于 drift 头

### 4.13 模型配置

| 名称 | hidden_size | depth | num_heads | 参数量 |
|------|------------|-------|-----------|-------|
| DiT-S | 384 | 12 | 6 | ~33M |
| DiT-B | 768 | 12 | 12 | ~130M |
| DiT-L | 1024 | 24 | 16 | ~460M |

---

## 5. 数据预处理流程

### 5.1 Vertex 版本

```
输入: .obj/.off 点云文件
  → 体素化: 将空间划分为 R³ grid
  → 每个 cell: 收集内部所有点，pad 到 K_MAX
  → 输入特征: [x_local, y_local, z_local, k/K_MAX]
  → 输出: {cell_data[R³], coords}
```

### 5.2 Face 版本

> 代码路径: `RDD_DiT_3D_face/generate_cell_data.py`

```
输入: mesh (.obj) 文件
  → 体素化: 按面片重心分配到 R³ grid cell
  → 每个 cell:
      1. 收集面片 (每面 9 坐标: 3顶点×3维)
      2. compute_local_coords: 归一化到 [-0.5, 0.5] + 记录 scale
      3. 按重心 z→x→y 排序
      4. pad 到 F_MAX
  → 输出: {faces[F_MAX, 9], mask[F_MAX], scale, cell_coords}
```

### 5.3 MeshAnythingV2 数据集预处理

> 代码路径: `RDD_DiT_3D_face/preprocess_meshanything.py`

1. **Merge 预处理**: 读取 MeshAnythingV2 的 .npz 文件并合并
2. **Grid Density 统计**: 统计每个 cell 中面片数的分布，用于选择 F_MAX
3. **溢出分析**: 超出 F_MAX 的面片被截断，统计丢失比例
4. **过滤**: 面片数为 0 的 cell 不参与训练

### 5.4 compute_local_coords 详细实现

```python
def compute_local_coords(faces_global, cell_origin, cell_size):
    """
    faces_global: [N_faces, 3, 3] — 全局坐标
    cell_origin: [3] — cell 左下角
    cell_size: float — cell 边长
    
    returns:
        normalized: [N_faces, 3, 3] — 范围 [-0.5, 0.5]
        scale: float — 用于还原
    """
    local = (faces_global - cell_origin) / cell_size   # → [0, 1]
    centered = local - 0.5                              # → [-0.5, 0.5] (大致)
    max_abs = centered.abs().max() + 1e-6
    scale = max_abs / 0.5                               # 保证除后精确到 [-0.5, 0.5]
    normalized = centered / scale
    return normalized, scale
```

---

## 6. 训练流程

### 6.1 Set Transformer AE 训练

> 代码路径: `RDD_DiT_3D_face/train_face_set_transformer.py`

```
for batch in dataloader:
    faces, mask, scale = batch           # [B, F_MAX, 9], [B, F_MAX], [B, 1]
    
    # Forward
    recon_coords, exist_logits, pred_scale, kl_loss = model(faces, mask, scale)
    
    # Loss
    recon_loss = L1(recon_coords[mask], faces[mask])
    exist_loss = BCE(exist_logits, mask.float())
    scale_loss = L1(log(pred_scale + ε), log(scale + ε))
    
    total_loss = recon_loss + exist_weight * exist_loss + scale_loss + kl_weight * kl_loss
```

### 6.2 离线编码 (阶段 2)

```python
# 对每个 mesh、每个分辨率 R:
for R in resolutions:
    for cell in non_empty_cells(mesh, R):
        faces, mask, scale = prepare_cell_data(cell)
        embedding = ae_model.encode(faces, mask, scale)  # [1, C]
        # 存储: (embedding, cell_coord)
```

### 6.3 RDD-DiT 扩散训练

> 代码路径: `RDD_DiT_3D/train.py`

```
for batch in dataloader:
    # batch 包含所有分辨率的 embeddings
    x_0, coords, mask = batch
    
    # 随机采样时间步 (跨阶段)
    t = uniform_sample(0, total_timesteps)
    stage = get_stage(t)
    
    # 计算 drift (非第一阶段)
    drift = compute_drift_sparse(x_0, coords, stage) if stage > 0 else 0
    
    # 前向扩散
    noise = torch.randn_like(x_0)
    kappa = get_kappa(t, stage)
    x_t = sqrt_alpha_bar * (x_0 + kappa * drift) + sqrt_one_minus_alpha_bar * noise
    
    # 模型预测
    eps_pred = model(x_t, coords, t, y, mask)
    
    # 目标 (含 drift)
    target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * kappa * drift
    
    loss = MSE(eps_pred[mask], target[mask])  # 仅在有效 token 上计算
```

---

## 7. 采样与推理流程

### 7.1 多阶段采样总览

```
Stage 0 (最粗, e.g. R=4):
  coords_0 = all_grid_coords(4)    # 64 tokens (全网格)
  x_0_clean = ddpm_sample(coords_0, T_stage0)
  
  → 空体素检测 → 确定 Stage 1 的活跃位置

Stage 1 (R=8):
  coords_1 = subdivide(coords_0_active)  # 活跃cell各生8个子cell
  x_1_init = noise_filling(x_0_clean, coords_0, coords_1)
  x_1_clean = ddpm_sample(coords_1, T_stage1, init=x_1_init)
  
  → 空体素检测 → 确定 Stage 2 的活跃位置

Stage 2 (R=16):
  coords_2 = subdivide(coords_1_active)
  x_2_init = noise_filling(x_1_clean, coords_1, coords_2)
  x_2_clean = ddpm_sample(coords_2, T_stage2, init=x_2_init)
```

### 7.2 空体素检测详细流程

```python
def detect_empty_voxels(x_clean, empty_embedding, threshold=0.8):
    """
    x_clean: [B, N, C] — 去噪后的 embeddings
    empty_embedding: [C] — 空 cell 的 reference embedding (训练时统计)
    """
    # Cosine similarity
    sim = F.cosine_similarity(x_clean, empty_embedding, dim=-1)  # [B, N]
    
    # 高相似度 = 空
    is_empty = sim > threshold
    
    # 3D max_pool 膨胀: 保留空体素的邻居
    # ... (稀疏实现)
    
    return ~is_empty  # True = 活跃，参与下一阶段
```

### 7.3 从 Embedding 重建几何

```python
# 最终阶段采样完成后:
for i, (emb, coord) in enumerate(zip(embeddings, coords)):
    # 从 bottleneck 解码
    faces, existence, scale = ae_model.decode_from_bottleneck(emb)
    
    # 筛选有效面片
    valid = torch.sigmoid(existence) > 0.5
    faces_valid = faces[valid]
    
    # 还原坐标
    local = faces_valid * scale + 0.5                   # → cell 局部 [0,1]
    global_coords = local * cell_size + cell_origin(coord)
    
    all_faces.append(global_coords)

# 后处理: 合并共享顶点
mesh = merge_vertices(all_faces, threshold=1e-5)
```

### 7.4 CFG 推理

```python
# 在每个去噪步:
eps_cond = model(x_t, coords, t, y_label, mask)
eps_uncond = model(x_t, coords, t, y_null, mask)
eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

---

## 8. 关键设计决策记录

### 决策 1: Triangle Soup vs Indexed Mesh
- **选择**: Triangle Soup (face 自含完整坐标)
- **理由**: 天然适配集合处理；不需要维护全局顶点索引；排列不变性自然满足
- **代价**: 共享顶点存储冗余 (3x)，靠后处理 merge

### 决策 2: Decoder 不传 Mask (训推一致性)
- **选择**: 训练和推理时 decoder 都不接收 mask
- **理由**: 推理时 mask 不可知；训练传 mask 会造成分布偏移
- **影响**: decoder 需要自主学习哪些 slot 对应真实面片

### 决策 3: Mean Pool (默认) vs Attention Pool
- **选择**: mean pool 作为默认
- **理由**: 实验表明 mean pool 对排序变化最鲁棒；attention pool 可选但无显著优势
- **弃用**: concat pool (破坏排列不变性)

### 决策 4: Scale 预测使用 log 空间
- **选择**: 网络输出 log_scale，exp 激活保证正数
- **理由**: scale 跨数量级变化，log 空间更适合优化；L1(log) loss 对大小 scale 等权重

### 决策 5: 排序替代 Hungarian Matching
- **选择**: z→x→y 字典序排序 + slot_queries 对齐
- **理由**: Hungarian 是 O(N³)，F_MAX=8 虽不大但仍有开销；排序提供确定性对齐且零额外开销
- **保证**: encoder 是排列不变的（只影响 decoder 监督对齐）

### 决策 6: F_MAX = 8 (推荐配置)
- **选择**: R=512 分辨率下，F_MAX=8 覆盖绝大部分 cell
- **数据依据**: MeshAnythingV2 统计中位数 3-4，95% < 8
- **权衡**: 更大的 F_MAX (16/24) 可降低截断率但增加计算开销

### 决策 7: 全稀疏表示 (无 3D convolution)
- **选择**: `[B, N, C]` + positional embedding
- **理由**: 3D 空间极度稀疏 (通常 < 5% 占用)；稠密 volume 浪费 95%+ 内存
- **代价**: 失去了局部感受野的 inductive bias，完全依赖 attention 学习空间关系

---

## 9. 已解决的 Bug 与经验教训

### Bug 1: `sys.path` 冲突
- **现象**: `train.py` import `dataset` 时加载了错误模块
- **原因**: 当前目录优先级高于包路径
- **修复**: `sys.path.insert(0, project_root)` 确保优先级

### Bug 2: `decode_from_bottleneck` 返回值解包
- **现象**: 采样时解包失败
- **原因**: 函数签名改为返回三元组 `(coords, existence, scale)` 后调用侧未更新
- **修复**: 所有调用点统一解包为三元组

### Bug 3: Scale 归一化范围不一致
- **现象**: 预处理输出范围不明确 (有时 [0,1]，有时 [-0.5,0.5])
- **修复**: 明确规定 `compute_local_coords` 输出范围为 [-0.5, 0.5]

### Bug 4: 推理还原公式错误
- **原始公式**: `normalized * scale` (缺少偏移)
- **正确公式**: `normalized * scale + 0.5` (恢复到 [0,1] cell 局部坐标)

### Bug 5: Mask 维度不匹配
- **现象**: encoder 期望 `[B, F_MAX]` bool mask，数据集返回了 int
- **修复**: 数据集侧统一转为 bool

### Bug 6: Diffusion 3D 中坐标归一化除以 0
- **现象**: max_resolution=0 时 sincos_from_coords 除以 0
- **修复**: 添加 `max(max_resolution, 1)` 保护

### Bug 7: 稀疏 drift 计算中 scatter 维度错误
- **现象**: `scatter_add` 的 index 广播不正确
- **修复**: 确保 index 形状与 src 形状精确匹配

### 经验教训

1. **训推一致性最重要**: 任何训练时有但推理时没有的信息（如 mask）都应该在训练时也去除
2. **log 空间处理跨量级变量**: scale、学习率等跨越数量级的值用 log 空间优化
3. **排列不变性要贯穿设计**: encoder 侧保证排列不变，decoder 侧通过 slot_queries + 排序提供对齐
4. **稀疏表示的 padding 要小心**: mask 必须正确传递，loss 只在有效位置计算
5. **先 smoke test，后大规模实验**: 用小数据和简单 case 验证数值正确性

---

## 10. 数据集统计

### 10.1 MeshAnythingV2 数据集

- 来源: MeshAnything v2 生成的 mesh
- 面片数范围: 数十到数千面
- 格式: .npz (vertices + faces)

### 10.2 Grid Density 统计 (R=512)

| 指标 | 值 |
|------|---|
| 中位数 faces/cell | 3-4 |
| 95th percentile | ~8 |
| 99th percentile | ~16 |
| max | ~50+ |

### 10.3 F_MAX 选择依据

| F_MAX | 截断率 (faces被丢弃) | 推荐场景 |
|-------|---------------------|---------|
| 8 | < 5% | 标准训练（推荐） |
| 16 | < 1% | 高精度需求 |
| 24 | < 0.1% | 极端 case 兼容 |

---

## 11. 文件结构

```
multiscale_diffusion/
├── set_transformer/                    # Vertex Set Transformer
│   ├── train_emd_ae_3d_v2.py          # AEBridgeReconModelV2 模型 + 训练
│   ├── analyze_embedding_quality.py    # Embedding 质量分析
│   └── glb_demo/                       # GLB 重建 demo
│
├── RDD_DiT_3D/                         # Vertex 版 RDD-DiT-3D
│   ├── model.py                        # DiT3D_RDD 模型 (+ sincos_from_coords)
│   ├── diffusion_3d.py                 # SpacedDiffusion3D, 前向/逆向过程, drift
│   ├── dataset.py                      # 数据加载 (多分辨率稀疏)
│   ├── train.py                        # 训练脚本
│   └── sample.py                       # 采样脚本 (多阶段 + 空体素检测)
│
├── RDD_DiT_3D_face/                    # Face 版 RDD-DiT-3D
│   ├── face_set_transformer.py         # FaceSetTransformerAE 模型
│   ├── train_face_set_transformer.py   # AE 训练脚本
│   ├── dataset.py                      # Face 数据加载
│   ├── generate_cell_data.py           # 预处理: mesh → cell data
│   ├── preprocess_meshanything.py      # MeshAnythingV2 预处理
│   ├── sample.py                       # 采样脚本 (含坐标还原)
│   └── DESIGN.md                       # 本文档
│
└── 2d/                                 # 2D 图像版 (参考实现)
    ├── RDD_sample.py                   # 2D 采样 (支持多GPU)
    └── RDD_sample_unified.py           # 2D Unified 多阶段采样
```

---

## 12. 与 MeshGPT 的对比

| 维度 | MeshGPT | 本项目 (RDD-DiT-3D Face) |
|------|---------|--------------------------|
| 表示 | 面片序列 (自回归) | 面片集合 (集合处理) |
| 生成方式 | Token-by-token 自回归 | 多分辨率扩散并行生成 |
| 拓扑 | 隐式通过序列顺序 | Triangle Soup + 后处理 merge |
| 面片编码 | Codebook (VQ-VAE) | Continuous latent (VAE/AE) |
| 分辨率 | 单一 | 多分辨率 (粗→细) |
| 可扩展性 | 受限于序列长度 | 稀疏表示支持高分辨率 |
| 条件生成 | Class token | Class label + CFG |
| 面片数上限 | 序列长度 ÷ 9 | N_cells × F_MAX |
| 训练信号 | Cross-entropy (离散) | MSE / L1 (连续) |

**本项目优势**:
1. 并行生成所有面片 (非自回归, 速度快)
2. 多分辨率生成 (粗→细, 质量递进)
3. 稀疏表示无 O(R³) 开销
4. 连续 latent 空间 (无 codebook collapse 问题)

**本项目劣势**:
1. 后处理 merge_vertices 可能引入误差
2. Triangle Soup 不保证 watertight
3. 需要选择 F_MAX 截断

---

## 附录 A: 数学符号表

| 符号 | 含义 |
|------|------|
| R | grid 分辨率 (如 4, 8, 16) |
| C | embedding/channel 维度 (= bottleneck_dim) |
| K_MAX | 每 cell 最大点数 (vertex 版) |
| F_MAX | 每 cell 最大面片数 (face 版) |
| ᾱ_t | cumulative product of (1-β_t) |
| κ_t | kappa, drift 调度系数 |
| N_max | batch 内 padding 后的最大 token 数 |
| B | batch size |

## 附录 B: 关键公式汇总

```
# 前向扩散
x_t = sqrt(ᾱ_t) * (x_0 + κ * drift) + sqrt(1-ᾱ_t) * ε

# 训练目标
target = ε + (sqrt(ᾱ_t) / sqrt(1-ᾱ_t)) * κ * drift

# DDPM 逆过程均值
μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))

# Scale 归一化
normalized = (local - 0.5) / scale,  where scale = max_abs / 0.5

# Scale 还原
local = normalized * scale + 0.5
global = local * cell_size + cell_origin

# KL Divergence
KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

# Noise Filling
x_init_fine = √(ᾱ_T_fine) * x_up + √(1-ᾱ_T_fine) * ε
```
