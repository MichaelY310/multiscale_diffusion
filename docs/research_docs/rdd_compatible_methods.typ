// RDD-Compatible 3D Representation Methods — 超详细技术文档
// 编译命令: python3 -c "import typst; typst.compile('rdd_compatible_methods.typ', 'rdd_compatible_methods.pdf')"

#set page(margin: 2cm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[RDD-Compatible 3D Representation Methods]
  #v(0.3em)
  #text(size: 12pt)[8 个与 RDD 架构适配的 3D 表征方法深度解析]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[基于源代码阅读，重点解释 Representation 的输入/过程/输出]
]

#v(1em)

#outline(indent: 1em, depth: 2)

#pagebreak()

= 方法一览与分类

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  [*方法*], [*底层 3D 表征*], [*每 voxel 维度*], [*发表*], [*Diffusion 类型*],
  [GaussianCube], [3D Gaussian Splatting], [14 连续], [ICLR 2024], [连续 (DDPM)],
  [Plenoxels/svox2], [Explicit Radiance Field], [28 连续], [CVPR 2022], [N/A (直接优化)],
  [DirectVoxGO], [Explicit Radiance Field], [4 连续], [CVPR 2022 Oral], [N/A (直接优化)],
  [Direct3D-S2], [SDF 距离场], [1→latent 8\~16], [2025], [连续 (Flow Matching)],
  [SCube], [稀疏点云/表面], [32→latent 8], [CVPR 2025], [连续 (DDPM)],
  [LATO.2], [Mesh 顶点占据], [聚合→latent 8], [2025], [连续 (Flow Matching)],
  [DVD], [Binary Occupancy], [1 bit], [2026], [离散 (USDM)],
  [Scaffold Diffusion], [离散类别体素], [1 int (\~256类)], [NeurIPS 2025], [离散 (MDLM)],
)

#v(1em)

*按输出可渲染性分组*：
- *可直接渲染*（无需额外处理）：GaussianCube (Gaussian Splatting), Plenoxels (体渲染), DirectVoxGO (体渲染)
- *需 Marching Cubes 提取 mesh*：Direct3D-S2 (SDF→mesh)
- *需解码器恢复结构*：SCube (稀疏结构+属性), LATO.2 (顶点+拓扑)
- *纯几何占据*（需后续管线生成外观）：DVD, Scaffold Diffusion

#pagebreak()

= GaussianCube — 3D Gaussian Splatting on Grid

== 一句话总结
将无序的 3D Gaussian Splatting 通过 Optimal Transport 排列到固定 $32 times 32 times 32$ 网格上，每 voxel 存储一个完整 Gaussian 的 14 个参数，直接用 3D U-Net 做 DDPM 扩散。

== 输入
*原始数据*：一组无序的 3D Gaussians（由 3DGS 拟合得到），每个 Gaussian 有位置、颜色、不透明度、缩放、旋转共 14 个参数。

*关键处理*：
1. 固定 Gaussian 数量为 $32^3 = 32768$ 个（通过 densification/pruning 调整）
2. 建立 $32 times 32 times 32$ 的均匀网格点作为"锚点"
3. 用 *Optimal Transport*（线性分配问题）将 32768 个 Gaussian 一一映射到 32768 个网格点，使总位移最小（双射）
4. 每个 Gaussian 的位置存储为相对于所在网格点的 *偏移量*（而非绝对坐标）

== 每 Voxel 存储内容（14 维连续向量）

```
通道 [0:3]  — Position Offset (相对于 voxel 中心的位移, 3维)
通道 [3:6]  — Color DC (RGB SH 0阶系数, 3维)
通道 [6:7]  — Opacity (不透明度, 1维, clamp到[0,1])
通道 [7:10] — Scale (各轴缩放, 3维, clamp到[0,1])
通道 [10:14]— Rotation (四元数, 4维, L2归一化)
```

代码验证（`gaussian_renderer/__init__.py` 第17-38行）：
```python
xyz = volume[:, :3] + std_volume_xyz          # 偏移+锚点 = 绝对位置
features_dc = volume[:, 3:6]                  # RGB color
opacities = volume[:, 6:7].clamp(0, 1)       # opacity
scales = volume[:, 7:10].clamp(0, 1)          # scale
rots = F.normalize(volume[:, 10:14])          # rotation quaternion
```

== 数据格式
- 磁盘存储：`.pt` 文件，形状 `(32, 32, 32, 14)` float32
- 送入模型：permute 为 `(14, 32, 32, 32)` channel-first
- 归一化：逐通道 mean/std 标准化 + clip

代码验证（`dataset/dataset.py` 第119-151行）：
```python
data = torch.load(path).to(torch.float32)  # (32,32,32,14)
volume = volume.permute(3, 0, 1, 2)        # -> (14,32,32,32)
```

== 扩散过程
- 模型：标准 3D U-Net（`model/unet.py`，`dims=3`）
- 上采样：`F.interpolate(mode='nearest')` — *最近邻*
- 目标：直接预测去噪后的 14 通道 voxel grid
- 损失：MSE + 可选渲染 loss（将预测 Gaussian 渲染后与 GT 图像比较）

== 输出与渲染
- 输出：`(14, 32, 32, 32)` 的连续张量
- 渲染：`parse_volume_data()` 拆分 14 通道 → 直接送入 Gaussian Splatting 渲染器 → 2D 图像
- *无需任何 MLP、mesh 提取或后处理*

== RDD 适配性
- ✅ $32^3$ 规则网格
- ✅ 14 维显式 per-voxel 属性，无 MLP
- ✅ Opacity 通道是天然稀疏信号（空区域 opacity≈0 → RDD drift=0）
- ✅ U-Net 使用 nearest 上采样

#pagebreak()

= Plenoxels / svox2 — Explicit Sparse Radiance Field

== 一句话总结
在 $N^3$ 网格上直接存储 density + 球谐系数（28 通道），用三线性插值体渲染出图像，通过梯度直接优化（非神经网络），完全无 MLP。

== 输入
*原始数据*：多视角图像 + 相机位姿（标准 NeRF 输入）

*无需预处理成特定格式*——Plenoxels 本身是一个"表示+优化"方法，不是生成模型。它的输出（优化后的 voxel grid）可以作为其他生成模型的训练数据。

== 数据结构（Dense-Sparse 混合）

```
links: [reso_x, reso_y, reso_z] int32   — N³索引表，-1表示空
density_data: [capacity, 1] float32      — 有效体素的密度
sh_data: [capacity, 27] float32          — 有效体素的球谐系数
```

代码验证（`svox2/svox2.py` 第449-458行）：
```python
self.density_data = nn.Parameter(torch.zeros(self.capacity, 1))
self.sh_data = nn.Parameter(
    torch.zeros(self.capacity, self.basis_dim * 3))  # 9*3=27
```

`links` 是一个 dense $N^3$ int32 张量，每个位置存储一个索引指向 `density_data`/`sh_data` 的行号。值为 -1 表示该 voxel 为空。

C++ 层定义（`data_spec.hpp` 第18-52行）：
```cpp
Tensor links;         // 3D: (reso_x, reso_y, reso_z), int32
Tensor density_data;  // 2D: (N, 1)
Tensor sh_data;       // 2D: (N, basis_dim*3)
```

== 每 Voxel 存储内容（28 维连续向量）

```
通道 [0]     — Density σ (体密度, 1维)
通道 [1:28]  — SH Coefficients (RGB各9个球谐基函数系数, 共27维)
               = 3色 × (l=0,1,2 共 (0+1)²+(1+1)²-1+(2+1)²-4 = 1+3+5 = 9个分量)
```

`data_dim` 属性（第537-543行）：
```python
@property
def data_dim(self):
    return self.sh_data.size(1) + 1  # 27 + 1 = 28
```

== 渲染过程
1. 沿光线采样点
2. 在 $N^3$ `links` 表中查找最近的 8 个体素
3. *三线性插值*获取该点的 density 和 SH 系数（CUDA 实现 `trilerp_cuvol_one`）
4. 用 SH 基函数 × 系数得到视角相关颜色
5. 沿光线做体积积分（alpha compositing）

== 优化过程（非扩散，直接梯度）
- 损失：渲染图像与 GT 的 MSE
- 优化器：直接对 `density_data` 和 `sh_data` 做梯度下降
- 粗到细：`reso_list` 控制分辨率递进（如 128→256→512）
- 自剪枝：低密度体素被移除（`links` 对应位置设为 -1）

== RDD 适配性
- ✅ $N^3$ 规则网格（通过 `links` 索引）
- ✅ 28 维纯显式 per-voxel 属性，*零 MLP*
- ✅ Density 通道是天然稀疏信号
- ⚠️ 渲染时用三线性插值（非 nearest），但作为 *扩散目标* 时不影响（扩散 U-Net 可用 nearest 上采样）

#pagebreak()

= DirectVoxGO (DVGO) — Explicit Dense Voxel Radiance Field

== 一句话总结
密集 $N^3$ 体素网格存储 density(1ch) + RGB/feature(3\~12ch)，coarse 模式完全无 MLP（直接 sigmoid 得 RGB），fine 模式用 3 层浅 MLP 处理视角依赖颜色。

== 输入
与 Plenoxels 相同：多视角图像 + 相机位姿。DVGO 也是一个重建方法而非生成模型。

== 数据结构

核心为 `DenseGrid` 类（`lib/grid.py` 第38-58行）：
```python
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, ...):
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))
        # 形状: [1, C, X, Y, Z]
```

模型包含两个网格：
```python
# lib/dvgo.py 第54-56行: Density 网格 (1通道)
self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size)

# 第73-75行: Coarse 模式 Color 网格 (3通道, 无MLP)
self.k0_dim = 3
self.k0 = grid.create_grid(k0_type, channels=3, world_size=self.world_size)
self.rgbnet = None  # 无 MLP！
```

== 每 Voxel 存储内容

*Coarse 模式*（4 维）：
```
通道 [0]   — Density σ (体密度, 1维)
通道 [1:4] — RGB logit (颜色, 3维, 经sigmoid→RGB)
```

*Fine 模式*（13 维）：
```
通道 [0]    — Density σ (体密度, 1维)
通道 [1:13] — Feature (12维, 送入3层MLP+view direction→RGB)
```

== 渲染过程（以 coarse 为例）

```python
# lib/dvgo.py 第318-347行
density = self.density(ray_pts)              # grid_sample 三线性插值查密度
alpha = self.activate_density(density)       # softplus → alpha
mask = (alpha > self.fast_color_thres)       # 跳过低密度点
k0 = self.k0(ray_pts[mask])                  # grid_sample 查颜色
rgb = torch.sigmoid(k0)                      # 直接 sigmoid, 无MLP!
```

== 稀疏性机制
1. `MaskGrid`（`grid.py` 第204-237行）：布尔 $N^3$ 缓存，标记空区域
2. `update_occupancy_cache()`（`dvgo.py` 第199-209行）：每 1000 步更新，alpha 低于阈值的 voxel 标记为空
3. `fast_color_thres`：渲染时跳过低 alpha 采样点

== 分辨率递进（Coarse-to-Fine）
```python
# grid.py 第60-65行
def scale_volume_grid(self, new_world_size):
    self.grid = nn.Parameter(
        F.interpolate(self.grid.data, size=tuple(new_world_size),
                      mode='trilinear', align_corners=True))
```
默认用三线性插值上采样。*改为 `mode='nearest'` 即可适配 RDD*（一行代码修改）。

== RDD 适配性
- ✅ Dense $N^3$ 网格
- ✅ Coarse 模式 4ch 纯显式无 MLP；Fine 模式仅 view-dependence 用浅 MLP
- ✅ Density 天然稀疏信号 + MaskGrid 裁剪
- ⚠️ 上采样默认 trilinear（改一行代码 `→ nearest` 即可）

#pagebreak()

= Direct3D-S2 — Sparse SDF Latent Diffusion

== 一句话总结
将 mesh 表面附近的稀疏 SDF 体素通过 Sparse Conv VAE 编码为 latent tokens（8x 空间压缩），在 latent 空间用 Sparse DiT 做 Flow Matching 扩散，解码后用 Marching Cubes 提取 mesh。

== 输入
*原始数据*：3D Mesh（三角网格）

*预处理*：
1. 从 mesh 计算 Unsigned Distance Field (UDF)
2. 找到距表面足够近（如距离不超过 3 个 voxel）的所有体素坐标
3. 计算这些位置的 *Signed Distance Field* (SDF)
4. 存储为稀疏格式：`sparse_sdf [N,1]` + `sparse_index [N,3]`

== 每 Voxel 存储内容

*原始层面*（编码器输入）：
```
1 维 SDF 标量值 (正=外部, 负=内部, 0=表面)
```

代码验证（`ss_vae.py` 第75-82行）：
```python
feat = batch['sparse_sdf']       # SDF 值
if feat.ndim == 1:
    feat = feat.unsqueeze(-1)    # → (N, 1)
x = sp.SparseTensor(feat, coords)
```

*Latent 层面*（扩散目标）：
```
embed_dim 维连续向量 (典型值 8~16)
分辨率 = 原始分辨率 / 8 (如 512→64, 1024→128)
```

== 编码过程（3 次 2x 下采样 = 8x 压缩）

```
SDF (N, 1) @ resolution 512
  → SparseLinear(1 → 32)
  → SparseDownBlock3d(32 → 64)   分辨率 512→256
  → SparseDownBlock3d(64 → 128)  分辨率 256→128
  → SparseDownBlock3d(128 → 512) 分辨率 128→64
  → SparseTransformer (Swin窗口注意力)
  → SparseLinear(512 → 2×embed_dim)  [输出 mu+logvar]
  → VAE 重参数化
  = Latent (M, embed_dim) @ resolution 64
```

每个 `SparseDownBlock3d`：SparseGroupNorm → SiLU → SparseDownsample(2x avg pool) → 2层 SparseConv3d(3×3×3) → 残差连接。

== 扩散过程（Sparse DiT + Flow Matching）

- 目标：在 latent space 的稀疏 tokens 上去噪
- 模型：`SparseDiT`（窗口注意力 + 交叉注意力 + AdaLN）
- 条件：图像特征（CLIP/DINOv2）通过 cross-attention 注入
- 位置：稀疏 token 的空间位置由上一阶段确定，扩散只学特征值

== 解码过程（3 次 2x SparseSubdivide = 8x 恢复）

```
Latent (M, embed_dim) @ resolution 64
  → SparseTransformer (Swin窗口注意力)
  → SparseSubdivideBlock3d(512 → 128) 分辨率 64→128
  → SparseSubdivideBlock3d(128 → 64)  分辨率 128→256
  → SparseSubdivideBlock3d(64 → 32)   分辨率 256→512
  → SparseLinear(32 → 1) + Tanh
  = 重建 SDF (N', 1) @ resolution 512, 范围[-1,1]
```

`SparseSubdivide`（`spatial.py` 第89-114行）核心逻辑：
```python
new_coords[:, 1:] *= 2                    # 坐标翻倍
new_coords = new_coords + offsets_2x2x2   # 8个子voxel偏移
new_feats = feats.expand(..., 8, ...)      # 特征直接复制（最近邻）
```

== 输出与 Mesh 提取

```python
# ss_vae.py 第107-128行
sdf = torch.ones((512, 512, 512))                          # 全1（默认外部）
sdf[coords[:,0], coords[:,1], coords[:,2]] = predicted_sdf  # 填入预测SDF
vertices, faces = marching_cubes(sdf, threshold=0.0)        # 等值面提取
```

输出：三角网格（vertices + faces）

== RDD 适配性
- ✅ 稀疏 $N^3$ 网格（SparseTensor 以整数坐标索引）
- ✅ Per-voxel 显式 SDF/latent（无 MLP decoder，解码器是对称的 sparse conv）
- ✅ SDF sign 是天然稀疏信号（仅表面附近非零）
- ✅ SparseSubdivide 是精确的 nearest-neighbor 复制

#pagebreak()

= SCube — Sparse Voxel + Cascade Diffusion

== 一句话总结
将点云（含法线/颜色/语义）通过 Encoder 映射为 per-voxel 特征，经 Sparse VAE 编码到 latent grid，在 latent 上用 Sparse U-Net 做 DDPM 扩散，解码时逐层上采样并预测结构（哪些 voxel 存在）。

== 输入
*原始数据*：带属性的点云（来自 LiDAR 或 mesh 采样）

每个点可包含：
- `xyz`：3D 坐标（必须）
- `normal`：法线方向（可选，3维）
- `color`：RGB 颜色（可选，3维）
- `intensity`：反射强度（可选，1维）
- `semantic`：语义类别标签（可选）

== Encoder：点云 → Per-Voxel 特征

```python
# base_encoder.py 第31-79行
coords = grid.grid_to_world(grid.ijk.float())    # 体素世界坐标
unet_feat = self.pos_embedder(coords)             # 频率位置编码(multires=5)
unet_feat = cat([unet_feat, normals, colors, ...])  # 拼接可选属性
unet_feat = self.mix_fc(unet_feat)                # Linear → c_dim维 (如32维)
```

其中法线等属性通过 `splat_trilinear`（三线性 splat）从原始点云投射到 voxel 网格。

== 每 Voxel 存储内容

*编码器输出*：`c_dim` 维（典型 32）混合特征向量
*VAE Latent*：`f_maps × 2^(num_blocks-1) / cut_ratio` 维（典型 8 维）
*解码器每层输出*：2 维 structure logit（exist/not-exist）

== VAE 编码（MaxPool 下采样）

```
Per-voxel features (c_dim) @ 原始分辨率
  → pre_conv
  → MaxPool 下采样 × num_blocks 层
  → pre_kl_bottleneck → mu, log_sigma
  → 重参数化
  = Latent (latent_dim 维) @ 最粗分辨率
```

== 扩散过程（Sparse U-Net + DDPM）

Sparse U-Net（`unet_sparse.py`）：
- 结构：标准 encoder-decoder + skip connections
- 下采样：`fvnn.AvgPool(stride)`
- 上采样：*`fvnn.UpsamplingNearest(stride)`* — 最近邻
- 时间条件：AdaGN（Adaptive Group Normalization）
- skip connection：通道维度拼接 `fvdb.jcat([pop_data, data], dim=1)`

关键上采样代码（`unet_sparse.py` 第62-64行）：
```python
if self.up:
    self.up_module = fvnn.UpsamplingNearest(self.stride)
```

支持多种条件输入：文本 cross-attention、图像特征、3D 框、深度反投影等。

== VAE 解码（UpsamplingNearest + Structure Pruning）

```
Latent @ 最粗分辨率
  → post_kl_bottleneck
  → 逐层: UpsamplingNearest(2x) → Conv → 预测结构(2ch logit)
        → struct_to_mask() 二值化 → 仅保留预测"存在"的子voxel
  → 最终可选输出: 法线(3ch) / 颜色(3ch) / 语义(Nch)
```

*关键设计*：每层上采样后，模型自己预测哪些新产生的子 voxel 应该存在（self-pruning），从而逐步恢复稀疏结构。

== 输出
- 多层稀疏结构网格（每层的 occupancy decision）
- 可选：每 voxel 的法线、颜色、语义
- 用于下游 Gaussian Splatting 渲染或 mesh 提取

== RDD 适配性
- ✅ fVDB 稀疏网格（整数坐标索引，底层是 $N^3$ 子集）
- ✅ Per-voxel 显式特征（结构 logit + 法线/颜色），VAE decoder 是对称 sparse conv
- ✅ Structure prediction 二值 mask 就是稀疏信号
- ✅ `fvnn.UpsamplingNearest` — 精确最近邻上采样

#pagebreak()

= LATO.2 — Mesh Vertex Occupancy + Topology Flow

== 一句话总结
将 3D Mesh 的顶点量化到 $1024^3$ 网格，通过点云 PointNet 聚合为 per-voxel 特征，经 Sparse VAE 编码到 latent（分辨率 64），解码时通过 4 级 SparseSubdivide（最近邻复制）+ occupancy pruning 逐步展开到 1024 分辨率恢复顶点位置。拓扑（面连接）由独立的 Flow Matching 模型生成。

== 输入
*原始数据*：标准 3D 三角网格文件（`.obj`, `.glb`, `.ply` 等）

*预处理*（`dataset/utils.py` 第11-55行 `quantize_mesh_clustering`）：
1. BBox 归一化到 $[0, 1]$
2. 顶点量化到 $[0, 1023]$ 整数网格
3. 落入同一 voxel 的顶点聚类合并（取均值位置）
4. 计算每个顶点的*亚体素偏移*（sub-voxel offset）：
   ```python
   voxel_center = (v_int + 0.5) / 1024.0
   offsets = (v_mean - voxel_center) * 2.0 * 1024  # 范围(-1,1)
   ```
5. Mesh 表面体素化：找出所有与表面相交的 voxel（`mesh_to_flexible_dual_grid`）
6. Active voxels = 顶点所在 voxels ∪ 表面穿越 voxels

*点云采样*（819,200 个点，每点 15 维特征）：
```
[xyz(3), normal(3), VDF(9)]
```
VDF = Vertex Direction Field，即从采样点到所在三角面三个顶点的方向向量（排序后展平）。

== 每 Voxel 存储内容

*PointNet 聚合后*：`out_channels` 维特征向量
- 将 819K 个点的 15 维特征通过 FC + ResNet 块提取逐点特征
- 每个点分配到其所在 active voxel
- `scatter_mean` 聚合：同一 voxel 内所有点取均值

*VAE Latent*：`latent_dim = 8` 维连续向量（在分辨率 64 下）

== 编码过程

```
Active Voxel Features (M, out_ch) @ resolution 1024
  → SparseLinear → model_channels
  → 4次 SparseResBlock3d + SparseDownsample(2x)
    (1024→512→256→128→64)
  → Sparse Transformer self-attention
  → SparseLinear → 2×latent_dim (mu+logvar)
  → VAE 重参数化
  = Latent (M', 8) @ resolution 64
```

== 解码过程（4 级 SparseSubdivide + Pruning）

*Step 0 — Shell→Vertex 筛选*：
```python
vtx_pruning_head(h)  # 预测哪些 shell voxel 包含 mesh 顶点
```

*Step 1\~4 — 逐级上采样*：
```
每级:
  1. SparseSubdivide: 每voxel → 8个子voxel（坐标×2, 特征复制）
  2. SparseConv3d(3×3×3): 细化特征
  3. Cross-Attention: 全局信息融合
  4. pruning_head: 预测每个子voxel的存活概率
  5. 阈值裁剪（保证每组至少1个存活）

分辨率变化: 64 → 128 → 256 → 512 → 1024
```

`SparseSubdivide`（`modules/sparse/spatial.py` 第126-158行）：
```python
new_coords[:, 1:] *= 2                           # 坐标翻倍
new_coords = new_coords.unsqueeze(1) + n_coords  # +8个偏移
new_feats = input.feats.unsqueeze(1).expand(...)  # 特征直接复制
```

== 输出

*几何输出*：
- 存活 voxel 坐标 @ 1024 分辨率
- OffsetHead 预测亚体素偏移 (N, 3)，范围 (-1, 1)
- 顶点最终位置 = `(voxel_coord + 0.5 + offset × 0.5) / 1024`

*拓扑输出*（独立的 TopologySiTFlow 模型）：
- 输入：已知顶点坐标 + 条件（二值占据体）
- Flow Matching（Rectified Flow, 50 步 Euler 积分）
- TopologyVAE.decode → 对所有顶点对预测连接概率
- 阈值 → 边集合 → 三角面重建

== RDD 适配性
- ✅ 规则 $1024^3$ 整数网格
- ✅ Per-voxel 显式特征（PointNet 聚合），解码器是对称 sparse conv
- ✅ Occupancy pruning 提供稀疏信号（每层只保留活跃 voxel）
- ✅ SparseSubdivide 是精确的 nearest-neighbor 复制

#pagebreak()

= DVD — Discrete Voxel Diffusion (Binary Occupancy)

== 一句话总结
在 $64^3$ 密集二值体素网格上直接做离散扩散（vocab_size=2），每 voxel 只有"空/占用"两种状态，无任何 VAE 或 MLP，是最简单最直接的 3D 形状生成方法。

== 输入
*原始数据*：3D Mesh

*预处理*（`dataset_toolkits/voxelize.py` 第19行）：
```python
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh, voxel_size=1/64,
    min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
```
将 mesh 表面体素化到 $64^3$ 分辨率，产出占用坐标。

*数据加载*（`datasets/sparse_structure.py` 第43-48行）：
```python
position = utils3d.io.read_ply(os.path.join(root, 'voxels', f'{instance}.ply'))[0]
coords = ((torch.tensor(position) + 0.5) * 64).int()
ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)  # 全零
ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1  # 占用位置设1
```

== 每 Voxel 存储内容

```
1 维离散值: 0 (空/air) 或 1 (占用/occupied)
格式: torch.long, 形状 [1, 64, 64, 64]
```

*就是这么简单 —— 每 voxel 只有 1 bit 信息*。

== 离散扩散过程（USDM: Uniform State Discrete Model）

*前向噪声*（`discrete_diffusion.py` 第126-134行）：
```python
noise = torch.randint(0, 2, x_0.shape)           # 从{0,1}均匀采样噪声
alpha_t = noise_schedule(t)                       # LogLinear 调度
move_indices = torch.rand(x_0.shape) < 1 - alpha_t  # 以概率(1-α_t)替换
x_t = torch.where(move_indices, noise, x_0)      # 被选中位置→随机{0,1}
```

*核心思想*：随着 $t$ 增大，越来越多 voxel 被随机噪声替换，最终 $t=1$ 时全部变成均匀随机 $\{0,1\}$。

*去噪模型*：DiT 架构（`sparse_structure_flow.py`）
- 输入 `[B, 1, 64, 64, 64]` 整数网格
- Patchify（patch_size=4）→ $16^3 = 4096$ 个 patch tokens
- Transformer 预测每个 voxel 的 2 维 logit（空/占用概率）
- 损失：NLL（负对数似然）

```python
vocab_size: int = 2  # 第96行
pred_logits_normalized = F.log_softmax(pred_logits, dim=-1)  # [B,...,2]
```

*反向采样*（`ancestral.py`）：从全随机 $\{0,1\}$ 出发，根据后验概率逐步恢复原始占用。

== 输出

```python
@dataclass
class DVDVoxelOutput:
    samples: torch.Tensor   # [B, 64, 64, 64] 二值占用网格
    coords: torch.Tensor    # [N, 4] 占用体素坐标 (batch_idx, x, y, z)
```

输出同时提供 dense 和 sparse 两种表示。

*后续 pipeline*：DVD 生成的占用结构作为 TRELLIS Stage 2 的条件输入，由 TRELLIS 生成最终的 mesh/Gaussian/radiance field。

== RDD 适配性
- ✅ 固定 $64^3$ 密集规则网格
- ✅ 纯二值 per-voxel 属性，零 MLP
- ✅ 0=空 / 1=占用，是最纯粹的稀疏信号（\~80% 为空）
- ✅ 二值 occupancy 的 nearest-neighbor 复制天然无邻居依赖

#pagebreak()

= Scaffold Diffusion — Discrete Category Voxel World

== 一句话总结
在 $32^3$ 网格上，每个 voxel 存储一个离散方块类别 token（\~256 种 Minecraft 方块类型），用 Masked Diffusion Language Model (MDLM) 做离散扩散生成。核心创新是 Scaffold 模式：*仅对 non-air voxels 做扩散*。

== 输入
*原始数据*：Minecraft 建筑的 3D 体素表示

*数据格式*（`datasets.py` 第318-324行）：
```python
voxels = torch.zeros((32, 32, 32), dtype=torch.long)  # 全零=全空气
voxels[new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2]] = block_ids  # 填入方块类别
```

*Tokenizer*（`MinecraftTokenizer`，第29-104行）：
- 建立 `block_id → token_id` 的双向查找表
- 空气 (air) 固定为 token_id = 0
- 总词表：\~256 种方块 + pad_token + mask_token
- Tokenize/Detokenize 是纯查表操作，无神经网络

== 每 Voxel 存储内容

```
1 个整数 token ID ∈ {0, 1, 2, ..., vocab_size-1}
  - 0 = 空气 (air)
  - 1~255 = 各种方块类型 (石头, 木头, 玻璃, ...)
格式: torch.long, 形状 (32, 32, 32)
```

== 两种扩散模式

=== 模式 A：Dense Diffusion（`diffusion.py` + `dit3d_window.py`）

对全部 $32^3 = 32768$ 个 voxel 做扩散：
1. 输入 `(B, 32, 32, 32)` 整数网格
2. `PatchEmbed_Voxel`（Conv3D, patch_size=4）→ patch 嵌入 `(B, 8³=512, hidden_dim)`
3. 3D DiT with window attention 处理
4. Unpatchify → 每个 voxel 的 `vocab_size` 维 logits

=== 模式 B：Scaffold 稀疏模式（`scaffold.py` + `dit3d_vanilla.py`）

*核心创新*——仅对 non-air voxels 做扩散：
```python
# scaffold.py 第801-857行 _get_token_pos_ids()
occupancy_map = (batch > 0).long()              # 哪些不是空气
token_pos = torch.nonzero(occupancy_map)[:, 1:]  # 非空坐标 (L, 3)
token_ids = batch[occupancy_map.bool()]          # 非空token值 (L,)
# 右填充到固定长度 L=max_active_tokens=1024
```

DiT backbone 接收稀疏序列：
```python
# dit3d_vanilla.py 第392-398行
pos_embed = self.pos_embed[x_coord, y_coord, z_coord]  # 坐标索引位置编码
h = self.vocab_embed(indices) + pos_embed + self.coord_embed(token_pos)
```

== 离散扩散机制（MDLM / Absorbing State）

*前向噪声*（`scaffold.py` 第533-547行）：
```python
move_chance = 1 - exp(-sigma)                    # sigma来自LogLinear调度
move_indices = torch.rand(x_0.shape) < move_chance
x_t = torch.where(move_indices, mask_token_id, x_0)  # 替换为[MASK]
```
- PAD token 永不被 mask
- 随 $t$ 增大，越来越多 token 变成 [MASK]

*SUBS 参数化*（第281-302行）：
- 模型预测每个位置的类别概率分布
- [MASK] token 的 logit 设为 $-infinity$（永不输出 MASK）
- 未被 mask 的位置强制保持不变

*损失*：$-log p_theta times frac(d sigma, e^sigma - 1)$ (ELBO 变体)

== 3D 位置编码

预计算 `(grid_size, grid_size, grid_size, embed_dim)` 的固定 sincos 编码：
- embed_dim 三等分，分别对 x/y/z 坐标独立编码
- 每轴：$omega = 1 slash 10000^(2i slash D)$，sin/cos 交替

== 输出
- 每个 voxel 位置的 `vocab_size` 维 logit
- Argmax → 方块类别 ID
- Detokenize → 原始 block_id
- 可直接在 Minecraft 引擎中渲染

== RDD 适配性
- ✅ 固定 $32^3$ 规则网格
- ✅ Per-voxel 离散类别（纯查表 tokenizer），零 MLP decoder
- ✅ Air token (id=0) 是天然稀疏信号，占比超过 95%
- ✅ 坐标式位置编码天然支持分辨率扩展
- ✅ Scaffold 模式的"仅对 non-air 做 diffusion"理念与 RDD 的 zero-drift 完全对齐

#pagebreak()

= 方法间关键对比

== 信息密度对比

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  [*方法*], [*通道数*], [*分辨率*], [*每物体总信息量*],
  [GaussianCube], [14], [$32^3$], [$32^3 times 14 = 458K$ floats],
  [Plenoxels], [28], [$256^3\~512^3$], [仅表面\~百万 floats],
  [DirectVoxGO], [4], [$128^3\~256^3$], [dense $N^3 times 4$],
  [Direct3D-S2], [1 (SDF)], [$512^3\~1024^3$], [仅表面带\~百万 floats],
  [SCube], [32→8 latent], [层级], [稀疏\~几十万 floats],
  [LATO.2], [聚合→8 latent], [64→1024], [稀疏\~几十万 floats],
  [DVD], [1 bit], [$64^3$], [$64^3 = 262K$ bits],
  [Scaffold Diff], [1 int], [$32^3$], [$32^3 = 32K$ ints],
)

== 渲染方式对比

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*方法*], [*渲染方式*], [*是否需要后处理*],
  [GaussianCube], [Gaussian Splatting (直接 rasterize)], [否],
  [Plenoxels], [体积渲染 (ray marching + 三线性插值)], [否],
  [DirectVoxGO], [体积渲染 (ray marching + grid_sample)], [否（coarse）],
  [Direct3D-S2], [Marching Cubes → mesh → 任意渲染器], [是（需 MC）],
  [SCube], [Gaussian Splatting / mesh 提取], [视配置],
  [LATO.2], [直接输出 mesh (vertices + faces)], [否],
  [DVD], [需 TRELLIS Stage 2 生成外观], [是],
  [Scaffold Diff], [Minecraft 引擎直接渲染], [否],
)

== 与 RDD 结合的推荐优先级

1. *GaussianCube* — MVP 首选。$32^3$，14ch，无需改动，直接可渲染验证
2. *DVD* — 最简单的形状生成验证。$64^3$，1bit，RDD zero-drift 优化效果最直观
3. *Scaffold Diffusion* — 理念最对齐。Scaffold 模式 = RDD zero-drift 的离散版本
4. *Direct3D-S2* — 高分辨率生产级管线。SparseSubdivide 架构可直接复用
5. *SCube* — 大规模场景扩展。fVDB + UpsamplingNearest + self-pruning
6. *LATO.2* — Mesh 直接生成。4 级 SparseSubdivide + pruning 的完整参考
7. *DirectVoxGO* — 最简显式辐射场。4ch，改一行代码即可
8. *Plenoxels* — 极致无 MLP。28ch 但分辨率偏高，适合做验证
