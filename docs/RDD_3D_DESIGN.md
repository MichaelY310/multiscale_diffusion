# RDD-DiT-3D: Sparse Multiscale 3D Generation via Resolution-Dependent Diffusion

## Overview

RDD-DiT-3D 是一个多分辨率稀疏 3D 生成框架。它将 3D 空间划分为从粗到细的多层 grid（如 R=8→16→32→64），每个 grid cell 用一个 embedding 向量表示其中包含的几何内容。核心创新在于：**每个阶段只在上一阶段判定为"非空"的 cell 处展开子 cell**，从而实现指数级稀疏化。

生成过程类似"递进式望远镜"：先在粗分辨率上确定大致占据区域，再逐步放大细化有内容的局部区域，空白区域直接丢弃不再处理。

---

## 1. 表示方式 (Representation)

每个 grid cell 的内容由一个 **embedding 向量** `e ∈ R^C` 描述。这个 embedding 是由外部 encoder（如 Set Transformer AE）将 cell 内部的点集编码得到的。

| 组件 | 格式 | 含义 |
|------|------|------|
| `embeddings` | `[N, C]` | 每个 cell 的内容向量 |
| `coords` | `[N, 3]` | 每个 cell 在当前分辨率 grid 中的整数坐标 |
| `resolution` | `int` | 当前 grid 分辨率（如 64） |

**空 cell 的表示 = 零向量。** 这是整个稀疏机制的基石。

> **模块化接口**: 如需替换 representation，只需提供一个 encoder/decoder 对满足：
> - `encode(points_in_cell) → embedding ∈ R^C`
> - `decode(embedding, cell_coord, resolution) → points`
> - 空 cell 编码为零向量（或提供自定义 `empty_embedding`）

---

## 2. 多分辨率层级 (Multi-Resolution Hierarchy)

设总分辨率为 `[R_0, R_1, ..., R_K]`，相邻阶段比率为 2（即 `R_{k+1} = 2 * R_k`）。

例如 `[8, 16, 32, 64]` 表示 4 个阶段：
- Stage 0: 8×8×8 = 512 cells（全网格）
- Stage 1: 最多 512×8 = 4096 cells（仅展开非空 parent）
- Stage 2: 继续稀疏展开
- Stage 3: 最终分辨率

每个阶段拥有独立的 diffusion 时间区间。全局 T=1000 步按 `steps_per_stage`（如 [400, 300, 300]）切分给各阶段。

---

## 3. 训练流程

### 3.1 数据准备: Parent-Occupancy 展开

从最高分辨率 GT 数据实时计算各阶段的 token 集合：

```
输入: 最高分辨率 embeddings [N_max, C] + coords [N_max, 3]

Stage 0 (最粗, R_0=8):
    生成全 grid: R_0^3 个 token
    对 GT 中有数据的 cell → mean pool 得到该分辨率的 embedding
    空 cell → 零向量
    记录 occupied mask（哪些 cell 在 GT 中有内容）

Stage k (k>0, R_k):
    从 Stage k-1 的 occupied cells 展开:
        parent_coords[occupied] → children = parent * 2 + offsets_2x2x2
    对每个 child:
        在 GT 中查找 → 有数据: mean pool embedding, 标记 occupied
                       → 无数据: 零向量, 标记 not occupied
```

**关键**: 训练时用 GT occupancy 决定展开，采样时用模型预测的 occupancy 决定展开，两者逻辑完全对齐。

### 3.2 Diffusion 训练

每个阶段独立执行 diffusion 训练：

```python
for each stage k:
    x_0 = embeddings at resolution R_k   # GT token embeddings
    t ~ Uniform(0, T_k)                  # 该阶段内的随机时间步
    
    if k == 0:  # 最粗阶段: 标准 ε-prediction
        x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        loss = MSE(model(x_t, t), ε)
    
    else:  # 非最粗阶段: 带 drift 校正
        drift = lowpass_residual(x_0)  # 低通滤波 - 原始
        κ = t / T_k                     # 时间调制系数
        x_t = sqrt(α_t) * (x_0 + κ*drift) + sqrt(1-α_t) * ε
        noise_target = ε + (sqrt(α_t)/sqrt(1-α_t)) * κ * drift
        loss = MSE(model(x_t, t), noise_target)

total_loss = Σ losses / num_stages
```

### 3.3 模型输入输出

模型是一个 Transformer（DiT 架构）：
- **输入**: token embeddings `[B, N, C]` + 3D coords `[B, N, 3]`（用于 3D RoPE 位置编码）
- **条件**: timestep `t` + class label `y`（adaLN-Zero 调制）
- **输出**: 预测的 noise target `[B, N, C]`
- **Attention mask**: padding token 位置设为 -inf，不参与 attention

位置编码使用 **3D RoPE (Centralized Interpolative)**：将不同分辨率的坐标统一映射到 `[0, base_size]` 范围，确保跨分辨率的位置编码一致性。

---

## 4. 采样流程 — 稀疏生成的核心

### 4.1 整体流程图

```
Stage 0 (R=8, 全网格)
    ┌─────────────────────────────────────┐
    │  512 tokens ← 高斯噪声              │
    │  反向扩散去噪 → denoised embeddings  │
    │  判断 occupancy → M_0 个非空 cells   │
    └───────────────┬─────────────────────┘
                    ↓ 仅展开非空 cells
Stage 1 (R=16)
    ┌─────────────────────────────────────┐
    │  M_0 × 8 = M_1 tokens              │
    │  初始化: 复制 parent emb + 加噪      │
    │  反向扩散去噪 → 判断 occupancy       │
    │  → M_1' 个非空 cells (M_1' < M_1)   │
    └───────────────┬─────────────────────┘
                    ↓ 仅展开非空 cells
Stage 2 (R=32)
    ┌─────────────────────────────────────┐
    │  M_1' × 8 tokens                    │
    │  同上...                             │
    └───────────────┬─────────────────────┘
                    ↓
    ... (继续到最高分辨率) ...
                    ↓
最终输出: 非空 cells 的 embeddings + coords
    → Decoder 还原 3D 几何
```

### 4.2 阶段 0: 全网格去噪

```python
N = R_0^3                              # 如 8^3 = 512
coords = 全网格坐标 [1, 512, 3]         # (0,0,0) 到 (7,7,7)
x = randn(1, 512, C)                   # 纯噪声
x = reverse_diffusion(model, x, coords, R_0)  # DDPM 去噪
```

### 4.3 占据性判断 (Occupancy Detection)

**核心机制**: 因为训练时空 cell 的 GT 为零向量，模型学会了对空区域输出低幅值 embedding。

```python
def judge_occupancy(embeddings, threshold=0.5):
    """
    embeddings: [B, N, C]
    返回: [B, N] bool mask, True = 非空
    """
    norms = embeddings.norm(dim=-1)  # L2 范数
    return norms >= threshold
```

**直觉**: 
- 非空 cell 的 embedding 编码了具体几何信息，L2 norm 较大
- 空 cell 的 embedding ≈ 零向量，L2 norm ≈ 0
- 阈值 0.5 做二值化判断

**替代方案**（代码中也实现了）：
```python
# 基于余弦相似度 + 膨胀
sim = cosine_similarity(embedding, empty_embedding)
active = sim < threshold  # 与空 embedding 不相似 → 非空
active = dilate_3d(active)  # 膨胀避免误剪边界
```

### 4.4 稀疏展开 (Sparse Expansion)

```python
# occupied_indices: 上一阶段非空 cell 的索引
parent_coords = coords[occupied_indices]       # [M_occ, 3]
parent_embs = embeddings[occupied_indices]     # [M_occ, C]

# 展开: 每个 parent → 8 个 children
offsets = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],
           [1,0,0],[1,0,1],[1,1,0],[1,1,1]]   # 2×2×2
children_coords = parent_coords * 2 + offsets  # [M_occ*8, 3]
```

### 4.5 Noise Filling 初始化

新阶段的初始 latent 不是纯噪声，而是从 parent embedding 继承信息：

```python
# 将 parent embedding 复制到其 8 个 children
x_up = parent_embs.repeat_interleave(8, dim=0)  # [M_occ*8, C]

# 加噪到该阶段的起始时间步 T_k
α_bar = alphas_cumprod[T_k - 1]
x_init = sqrt(α_bar) * x_up + sqrt(1 - α_bar) * randn_like(x_up)
```

然后从 `x_init` 开始反向去噪，而非从纯噪声开始。这让细粒度阶段能利用粗粒度的结构信息。

---

## 5. 为什么这样设计能实现稀疏生成

| 机制 | 效果 |
|------|------|
| 空 cell GT = 零向量 | 模型学会输出低 norm = "这里没东西" |
| L2 norm 阈值判断 | 简单高效地区分空/非空 |
| 仅展开非空 parent | 每阶段 token 数量由内容决定，而非 resolution^3 |
| 指数级剪枝 | 3D 物体通常只占空间的 ~5-15%，4 个阶段后 token 数 << R^3 |

**对比全网格方法**: R=64 全网格需要 64^3=262,144 tokens，而稀疏展开后通常只有 ~2000-10000 tokens。

---

## 6. Drift 校正机制 (仅非最粗阶段)

非最粗阶段的 diffusion 包含一个 drift 项，补偿从粗分辨率到细分辨率的信息损失：

```
drift = lowpass(x_0) - x_0
```

其中 lowpass 是将当前分辨率的 embedding 先 mean pool 到一半分辨率再 nearest-neighbor upsample 回来。drift 捕捉了"粗粒度看不到的细节"。

前向过程中 drift 随时间线性 blend in，使模型在去噪过程中逐步恢复细节信息。

---

## 7. 如何替换 Representation

整个框架对 cell embedding 的来源是 **完全解耦** 的。要替换 representation：

### 需要实现的接口:

```python
class YourEncoder:
    def encode(self, points_in_cell: [K, 3]) -> embedding: [C]
        """将 cell 内 K 个点编码为 C 维向量"""
        ...

class YourDecoder:
    def decode(self, embedding: [C], cell_coord: [3], resolution: int) -> points: [K', 3]
        """将 embedding 解码回 cell 内的点"""
        ...
```

### 需要满足的约束:

1. **空 cell 编码为零向量**（或者修改 `judge_occupancy` 使用你的 `empty_embedding`）
2. **固定维度 C**（所有 cell 的 embedding 维度一致）
3. **平移等变性建议**：cell_coord 给出了该 cell 在全局 grid 中的位置，decoder 应将局部坐标 + cell_coord 组合还原全局坐标

### 需要修改的文件:

| 文件 | 修改点 |
|------|--------|
| `dataset.py` | 数据加载：提供各 cell 的 embedding |
| `sample_to_glb.py` | 最终输出：用你的 decoder 将 embedding 解码为点/面 |
| `judge_occupancy()` | 如果空 cell 不用零向量，修改判断逻辑 |
| 模型配置 | `in_channels=C` 匹配你的 embedding 维度 |

**不需要修改**: model.py、train.py、diffusion_3d.py — 它们只关心 `[B, N, C]` 的 token 和 `[B, N, 3]` 的坐标。

---

## 8. 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `stage_resolutions` | 各阶段分辨率 | [8, 16, 32, 64] |
| `steps_per_stage` | 各阶段 diffusion 步数 | [400, 300, 300] |
| `occupancy_threshold` | 空/非空判断阈值 | 0.5 |
| `C` (in_channels) | embedding 维度 | 32-256 |
| `K_MAX` | 每个 cell 内最大点数 | 50 |
| `total_diffusion_steps` | 全局 diffusion 步数 | 1000 |

---

## 9. 计算复杂度分析

设物体占据率为 `p`（通常 5-15%），则：
- Stage 0: `R_0^3` tokens（全网格，固定）
- Stage k: `≈ p * R_k^3` tokens

由于每阶段只处理非空区域，实际 token 数远小于 `R^3`。例如 R=64, p=10%:
- 全网格: 262,144 tokens
- 稀疏: ~26,000 tokens（节省 90%）

Attention 复杂度为 O(N²)，稀疏化后计算量约为全网格的 p² ≈ 1% — 这使得高分辨率 3D 生成在计算上变得可行。

---

## 10. 网络架构设计 (DiT3D_RDD)

### 10.1 设计参考

本模型的 Transformer 架构融合了以下工作的设计思路：

| 参考来源 | 借鉴的设计 |
|----------|-----------|
| **DiT** (Peebles & Xie, 2023) | adaLN-Zero 调制机制、TimestepEmbedder/LabelEmbedder、FinalLayer 零初始化 |
| **Open-Sora / MMDiT** (HPC-AI Tech) | SingleStreamBlock 并行 Attn+MLP 架构、整体代码组织风格 |
| **PaLM** (Google, 2022) | 并行 Attention + MLP（非串行），通过 fused linear 提高计算效率 |
| **HunyuanDiT** (Tencent) | Centralized Interpolative Positional Encoding — 3D RoPE 坐标归一化方式 |
| **QK-Norm** (Henry et al., 2020) | Per-head RMSNorm 稳定 attention logits |

### 10.2 整体架构

```
Input: x_active [B, N, C] + coords [B, N, 3] + t [B] + y [B]
                │
                ▼
        ┌─── x_embedder (Linear: C → hidden_size) ───┐
        │                                              │
        │   t_embedder(t) + y_embedder(y) → vec [B,D] │
        │                                              │
        │   compute_3d_rope(coords) → (cos, sin)       │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
        ┌─────── SingleStreamBlock × depth ────────┐
        │                                           │
        │   adaLN-Zero: shift, scale, gate          │
        │   x_mod = (1+scale) * LayerNorm(x) + shift│
        │                                           │
        │   ┌─── linear1 (fused) ──────────┐        │
        │   │ QKV [3*D] │ MLP_in [4D]      │        │
        │   └────┬───────────────┬──────────┘        │
        │        │               │                   │
        │    QK-Norm          GELU                   │
        │    3D RoPE                                 │
        │    SDPA(attn_mask)                         │
        │        │               │                   │
        │   ┌────┴───────────────┴──────────┐        │
        │   │    linear2 (merge)            │        │
        │   └───────────────┬───────────────┘        │
        │                   │                        │
        │   x = x + gate * output (gated residual)   │
        └───────────────────┬───────────────────────┘
                            │
                            ▼
        ┌─────── FinalLayer ───────────────────────┐
        │  adaLN: shift, scale from vec             │
        │  x = (1+scale)*LayerNorm(x) + shift       │
        │  output = Linear(x) → [B, N, out_channels]│
        └───────────────────┬───────────────────────┘
                            │
                            ▼
        Zero-out padding positions (× padding_mask)
```

### 10.3 关键设计选择

#### (a) 并行 Attention + MLP (PaLM Style)

传统 Transformer 是 Attn → Add → MLP → Add（串行），本模型采用并行方式：

```python
# 一次 linear 同时计算 QKV 和 MLP 输入
qkv, mlp_in = linear1(x_mod).split([3*D, mlp_hidden_dim], dim=-1)

# 并行执行
attn_out = attention(q, k, v)
mlp_out = GELU(mlp_in)

# 合并
output = linear2(cat[attn_out, mlp_out])
x = x + gate * output
```

优势：减少一次 LayerNorm + residual 操作，在不损失表达能力的情况下提高 throughput ~15%。

#### (b) 3D RoPE — Centralized Interpolative

坐标映射策略：
```python
norm_coords = coords.float() * (base_size / (resolution - 1))
```

将 `[0, R-1]` 的整数坐标映射到 `[0, base_size]`（默认 base_size=8）。

**为什么不直接用原始坐标？** 因为不同阶段分辨率不同（如 R=8 和 R=64），如果用原始坐标，高分辨率的位置编码范围远大于低分辨率。Centralized Interpolative 确保所有阶段的坐标范围一致，使得：
- 粗分辨率的 (0,0,0)→(7,7,7) 和细分辨率的 (0,0,0)→(63,63,63) 映射到相同的 [0,8] 范围
- 同一物理位置在不同分辨率下有相近的位置编码
- 模型可以在不同分辨率阶段间迁移位置理解

head_dim 被三等分给 x/y/z 三个轴，每个轴独立计算 RoPE cos/sin。

#### (c) QK-Norm

在 attention 计算前对 Q 和 K 做 per-head RMSNorm：
```python
q = RMSNorm(q)  # per head, per feature
k = RMSNorm(k)
```

防止 attention logits 爆炸，尤其在 bf16 训练 + 高维 token 数下效果显著。

#### (d) adaLN-Zero 调制

每个 block 通过 conditioning vector（timestep + class embedding）调制 LayerNorm 的输出：
```python
shift, scale, gate = Modulation(vec)
x_mod = (1 + scale) * LayerNorm(x) + shift
...
x = x + gate * output  # gate 初始化为 0
```

**零初始化关键**: Modulation 层的输出 Linear 权重和 bias 初始化为 0，意味着训练开始时：
- scale = 0, shift = 0 → LayerNorm 输出不变
- gate = 0 → residual 贡献为 0

整个网络初始时等价于恒等映射，保证训练初期数值稳定。

#### (e) Attention Mask 处理

对于变长序列（不同样本 token 数量不同），通过 padding + mask 处理：
```python
# padding 位置作为 key 时设为 -inf，softmax 后为 0
attn_mask = torch.zeros(B, 1, 1, N)
attn_mask.masked_fill_(~padding_mask[:, None, None, :], float('-inf'))

# SDPA (scaled dot-product attention) 使用 memory-efficient backend
attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

注意：**不能**简单对 K/V 乘零来替代 mask（因为 softmax(0) ≠ 忽略）。

最终输出也对 padding 位置清零：`out *= padding_mask.float()`

### 10.4 模型规格

| 配置 | hidden_size | depth | num_heads | head_dim | 参数量 |
|------|------------|-------|-----------|----------|--------|
| **S** (Small) | 384 | 12 | 6 | 64 | ~33M |
| **B** (Base) | 768 | 12 | 12 | 64 | ~110M |
| **L** (Large) | 1024 | 24 | 16 | 64 | ~381M |
| **XL** (Extra Large) | 1152 | 28 | 16 | 72 | ~534M |

当前实验使用 **B** 配置（110M 参数），在 8×H20 GPU 上以 bf16 + gradient checkpointing 训练。

### 10.5 初始化策略

| 组件 | 初始化 |
|------|--------|
| 所有 Linear | Xavier uniform (weight), 0 (bias) |
| Embedding (class/timestep) | Normal(std=0.02) |
| Modulation 输出 Linear | 全零 (adaLN-Zero) |
| FinalLayer 的 adaLN + output Linear | 全零 |
| d_prediction FinalLayer | 全零 |

全零初始化使网络启动时为恒等映射，保证训练初期不产生大的梯度信号。

### 10.6 Classifier-Free Guidance (CFG)

模型支持 CFG 采样：
```python
# 训练时: 以 dropout_prob 概率将 class label 替换为 null token
# 采样时: 并行计算 conditional 和 unconditional 预测
eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

通过 `LabelEmbedder` 的 `token_drop` 实现训练时的随机 dropout。

### 10.7 Gradient Checkpointing

高 token 数下（如 ~27k tokens）显存瓶颈在 attention 的中间激活。通过 PyTorch 原生 checkpoint 包裹每个 block：
```python
if self.gradient_checkpointing and self.training:
    x = torch.utils.checkpoint.checkpoint(block, x, vec, pe, padding_mask)
```

以 ~30% 计算开销换取 ~60% 显存节省。
