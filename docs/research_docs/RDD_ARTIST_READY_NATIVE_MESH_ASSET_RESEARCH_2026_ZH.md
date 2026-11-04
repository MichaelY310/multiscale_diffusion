# RDD × Native Mesh × Artist-Ready Asset：前沿调研与研究方案

> 调研日期：2026-08-24  
> 范围：native mesh generation、artist topology、UV/PBR、parts、rigging、skinning、blendshape、motion、LOD 与 DCC/engine 交付。  
> 结论边界：本文提出的是可检验的研究方案；在完成真实 DCC 导入、动画压力测试和艺术家盲评之前，不能承诺“零人工修改即可投入生产”。

## 1. 核心结论

### 1.1 哪些 native-mesh 方法最适合 RDD

按“数学兼容性 + 能否保留原生 mesh topology”综合排序：

1. **Progressive-mesh / vertex-split 路线（ARMesh、VertexRegen）**：最适合发展成真正 novel 的 Mesh-RDD。它天然给出跨 LOD 的 restriction/prolongation 和 parent-child correspondence，但当前方法是离散、顺序生成，需要改造成阶段内并行生成。
2. **Nexus 的 octree vertex factorization**：最容易把现有规则网格 RDD、动态稀疏和 vertex generation 接进去；edge/face topology 仍应由显式 topology decoder 处理。
3. **LATO.2 的 V-VAE/T-VAE factorization**：稀疏 vertex latent 很适合 RDD，GT adjacency supervision 也很干净；不足是 vertex pair 打分近似二次复杂度，且 topology 无法纠正错误 geometry。
4. **Rigel3D/AniGen 的 surface+skeleton sparse streams**：非常适合把 RDD 扩展到 skeleton/skin，但它们当前不是 native artist mesh——表面仍来自 SDF/FlexiCubes 或连续场。
5. **MeshFlow**：continuous per-vertex topology embedding 可以实验 RDD，但没有天然嵌套网格，pooling topology embedding 不保持组合拓扑，必须先构造 hierarchy。

不适合作为 RDD 主干、但适合作为 decoder/teacher/reward 的方法：QuadGPT、Mesh-Pro、MeshRipple、SATO、FACE、MeshMosaic。它们的 face/strip/token sequence 没有自然的 average pooling 或 nearest-neighbor prolongation。

### 1.2 当前真正空缺的方向

当前尚没有一个公开方法同时做到：

- 多分辨率、计算自适应的 native mesh generation；
- 每个阶段均输出有效、可编辑的 triangle/quad mesh，而非中间 implicit field；
- 明确生成 artist edge flow、UV seams、实际 UV coordinates 和 PBR maps；
- geometry 与 skeleton、skinning 在生成阶段相互约束；
- topology 接受真实动作/变形监督，而不只在静态姿态下对齐表面；
- 同时交付 LOD、命名部件、material slots、rig、动作和标准 glTF/USD/FBX 资产。

这比“把 RDD 用在 TRELLIS latent 上”明显更 novel，也更贴合 RDD 最初的多分辨率目的。

### 1.3 推荐的主方法

建议将主方向命名为类似：

> **StudioRDD / Progressive Native-Mesh RDD：Rig- and UV-aware Hierarchical Mesh Generation**

核心不是把 face tokens 直接做 diffusion，而是建立一个由 artist mesh 派生的、保持拓扑有效性的 progressive hierarchy：

1. 生成 coarse control cage、semantic parts 和 kinematic graph；
2. 每一阶段并行生成合法的 edge/face split decisions；
3. 用 RDD 生成新增 vertices 和连续属性的 detail residual；
4. 用 rig、UV seams、hard edges 和材料边界约束 refinement；
5. 每个阶段天然成为可用 LOD；
6. 只展开有几何、材质或形变细节需求的 surface patches。

这将 RDD 从“规则 voxel 分辨率变化”推广成“显式 mesh LOD 上的拓扑变化生成过程”。

---

## 2. RDD 的真实约束

仓库当前 RDD-3D 已经验证：

- ordinary-average hierarchy 可以在 TRELLIS SLat 上精确构造；
- 由 coarse stage 自己预测 non-empty support、再逐级展开是可行的；
- frozen 29K denoiser + endpoint occupancy head 在 64-asset final test 上达到 support precision/recall/IoU `0.245/0.509/0.191`；
- 对相同 checkpoint，dynamic sparse path 相对 force-keep control 实测约 `3.94x` sampler speedup，peak allocated memory 下降约 `46%`；
- 但当前 25.6M 小模型、96 个训练资产的 mesh F-score 约 `0.159`，只证明 dynamic sparse mechanism，不证明通用 3D generator 质量。

因此，新方案不能继续把“稀疏机制成立”与“能生成 production mesh”混为一谈。新的 research question 应是：

> 能否把 RDD 的跨尺度连续生成与动态计算优势，放进一个显式、拓扑有效、属性持久的 progressive artist-mesh representation，并在相同质量下减少高分辨率 topology/geometry generation 的计算？

### 2.1 适配 RDD 所需的结构

一个 representation 至少应提供：

- 明确的 coarse element 与 fine children 对应关系；
- restriction `R_k`：fine → coarse；
- prolongation `P_k`：coarse → fine；
- 对连续属性有意义的 `P_k R_k x - x` detail residual；
- 能让 inactive/empty/low-detail branches 不被展开；
- topology 的离散变化不能依靠对整数 face index 做线性平均。

规则 octree 满足前五项；progressive mesh 的 edge collapse / inverse vertex split 也满足，但需要把 topology 与连续属性分开建模。

---

## 3. 最新 native-mesh 方法与 RDD 兼容性

| 方法 | 原生输出 | 跨尺度结构 | RDD 兼容性 | 推荐角色 |
|---|---|---|---:|---|
| Nexus | vertices + edges/faces | vertex octree | 5/5 | 最快的第一版 geometry backbone |
| LATO.2 | vertices + adjacency + faces | sparse voxel vertex latent | 4.5/5 | 可复现 vertex/topology baseline |
| ARMesh / VertexRegen | progressive topology operations | edge collapse / vertex split | 5/5（概念） | 最 novel 的最终表示 |
| Rigel3D | mesh + skeleton + skinning | surface/skeleton SLats | 5/5（rig stream） | 最接近的 animation-ready competitor |
| AniGen | shape/skeleton/skin fields | shared sparse field | 5/5（空间层级） | joint generation competitor；非 native mesh |
| MeshFlow | per-vertex geometry/topology latent | 无固定 hierarchy | 2.5/5 | topology embedding/reward baseline |
| MeshMosaic | patch-wise AR mesh | patch hierarchy | 3/5 | high-poly/local refinement baseline |
| SATO | strip tokens + UV islands | 1D strip sequence | 1.5/5 | UV/topology teacher 与 reward |
| QuadGPT / Mesh-Pro | tri/quad face sequence | 1D token sequence | 1/5 | quad topology teacher/reward |
| MeshRipple | frontier BFS faces | 1D frontier | 1/5 | connectedness/topology AR baseline |
| FACE | one-face-one-token codec | VecSet + face sequence | 2/5 | 高效 final decoder baseline |
| Mesh BDF / SQuadGen | topology continuous field | voxel/grid field | 5/5 | 强 RDD baseline，但不满足严格 native-GT-topology 标准 |
| TopoMesh | DMC-compliant topology | fixed grid | 5/5 | direct mesh-loss baseline，但不保留原 artist connectivity |

### 3.1 Nexus：最容易落地，但不是最终 artist-ready 解法

Nexus 将 `p(M|C)` 分成 vertex generation 与 topology generation。vertex 被表示成 octree occupancy，天然适合 RDD 的 `8→16→32→64` 或更高分辨率 staged flow。其 topology autoencoder 在 GT edges 和 GT faces 上使用显式 BCE，然后通过 per-vertex spacetime embedding 生成 connectivity。[论文](https://arxiv.org/html/2607.13563)

可直接改造：

- 用 RDD 替换 vertex octree 的普通 coarse-to-fine flow；
- 复用当前 dynamic occupancy head 决定哪些 octree branches 展开；
- 将 topology generation 延后到 final vertices，作为低风险第一版；
- 第二版在每个 LOD 生成 topology，以建立 topology-aware drift。

限制：

- final topology 仍需全局 pair/cycle reasoning；
- 生成顶点位置可能受 grid snapping；
- face orientation 仍需后处理；
- 生成的是 artist-like triangle connectivity，不是 production quad edge loops；
- topology 只在最终 stage 生成时，不能帮助前面 geometry refinement。

### 3.2 LATO.2：最好的开源 factorized baseline

LATO.2 的 T-VAE 使用 GT adjacency mask，并对正 edges、hard negatives 和 random negatives做 asymmetric focal supervision；V-Flow 与 T-Flow 分别生成 geometry/topology latent。[论文](https://arxiv.org/html/2607.10623)；[代码](https://github.com/LoHhhha/LATO.2)

可直接改造：

- 对 V-VAE sparse latent 建 RDD pyramid；
- coarse stage 先生成 vertex support；
- fine stage只展开高 curvature / high deformation / high texture residual patches；
- 保留 T-VAE/T-Flow 作为 topology head。

限制：

- pairwise edge scoring 随 vertex 数上升成本高；
- face 由三环恢复，错误 edge 会产生连锁错误；
- topology 不能移动错误 vertices；
- 无 UV、material、rig、animation；
- 对 100K-face artist mesh 不现实。

### 3.3 Progressive mesh：最适合形成独立 novelty

VertexRegen 通过 inverse vertex-split 生成 continuous LOD；ARMesh 直接生成 simplicial-complex remeshing operations。它们最大的价值不是当前指标，而是 topology hierarchy 本身是真实 mesh 操作。[VertexRegen](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf)；[ARMesh](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ec7d99ca50540a9c6994197f4355776f-Abstract-Conference.html)

RDD 可以做的独特改进：

- 不是一次 AR 一个 split，而是在同一 resolution stage 内并行预测互不冲突的 split set；
- continuous vertex/detail attributes 用 RDD/FM；
- discrete split templates 用 categorical flow、absorbing diffusion 或 masked parallel decoding；
- split grammar 在构造层面保证 manifoldness、orientation 和局部 quad validity；
- inactive patch 不展开，实现 surface-complexity sparsity。

### 3.4 Rigel3D 与 AniGen：必须正面对比的最接近工作

Rigel3D 同时生成 surface SLat 和 skeleton SLat，解码 mesh、skeleton topology、joint positions、skinning weights，并做 open-vocabulary joint labeling。它有四个 flow：surface occupancy/latent 和 skeleton occupancy/latent。[论文](https://arxiv.org/html/2605.13129)

AniGen 把 Shape、Skeleton、Skin 统一成连续 `S^3` fields，并以 structured flow 共同生成。[论文](https://arxiv.org/html/2604.08746)

它们与拟议方法的关键差异应是：

| | Rigel3D / AniGen | StudioRDD |
|---|---|---|
| 表面 | SDF/FlexiCubes 或 continuous fields | progressive native triangle/quad mesh |
| GT face connectivity | 不保留 | 显式监督并生成 |
| LOD | 不是核心产物 | 每个 stage 都是合法 LOD |
| topology 与 rig | 间接耦合 | topology refinement 直接由 deformation loss 约束 |
| UV | 非核心 | seam + coordinates + packing |
| 动作监督 | 主要静态 rig supervision | motion/deformation stress supervision |
| 稀疏计算 | sparse representation | occupancy + detail + deformation 三类 adaptive refinement |

Rigel3D 明确承认会产生 missing joints、spurious branches、imperfect connectivity；由于没有长动作序列监督，极端姿态的 deformation 可能不自然。AniGen 也承认机械部件闭合时会有间隙，生成 skeleton 更接近 medial axis，而不是 production anatomical rig。

---

## 4. 现有 direct/native mesh 方法的共同缺陷

### 4.1 Autoregressive 主线的结构性问题

1. **长序列与慢推理**：面数越高越严重；100K-face generation 很难靠单序列完成。
2. **error accumulation**：一个错误 vertex/face token 会造成 holes、flying faces、fragmentation 和后续 traversal 崩坏。
3. **量化误差**：坐标离散化损坏 sharp edges、thin structures 和对称性。
4. **顺序依赖**：同一 mesh 有大量等价 serialization；canonical ordering 可能破坏局部性。
5. **polycount 不可控**：EOS 的概率同时承担完整性与模型复杂度控制。
6. **局部有效不等于全局有效**：face sequence CE 不直接保证 watertightness、genus、manifoldness 或全局 edge flow。

MeshRipple 通过 frontier BFS 改善连通性；MeshMosaic 通过 patch generation 扩到 100K faces；FACE 通过 one-face-one-token 缩短序列；但它们分别留下 AR latency、patch seams 和 topology validity 问题。

### 4.2 非 AR native topology 主线的问题

Nexus、LATO.2、MeshFlow 的共同模式是先 vertices，再 pairwise edges，再从 cycles 恢复 faces。问题包括：

- vertex generation 与 topology generation distribution mismatch；
- adjacency 往往接近 `O(N^2)`；
- 一个错误 edge 会同时创造多个错误 cycles；
- face orientation、non-manifold edges、tiny holes 仍需修复；
- 缺少 production quad topology；
- 几乎没有 UV、rig 和 motion-aware losses。

### 4.3 “artist mesh”仍经常只是视觉上像 artist mesh

大多数论文重点报告 CD、HD、NC。它们不能证明：

- 是否有边界洞、自交、退化面、孤立点；
- edge loops 是否适合 shoulder、elbow、mouth、eye deformation；
- poles 是否位于合理位置；
- mesh density 是否跟随曲率与形变复杂度；
- subdivision 后是否稳定；
- 是否能在 Maya/Blender/Unreal 中直接编辑。

真正 production-oriented 的 QuadGPT 使用 edge-loop/fracture reward；Mesh-Pro 使用 mixed tri/quad representation 与 online RL；SATO 生成 UV island boundaries。但即使这些方法也没有共同生成完整 rig、motion-tested skinning 和 corrective shapes。

### 4.4 UV 与材质是断开的

SATO 是当前最接近 joint mesh+UV generation 的方法：它把 UV chart transition token 放入 strip sequence，但只生成 face-to-island segmentation，实际 UV coordinates 仍由 Blender unwrap 后处理。[SATO](https://arxiv.org/html/2604.09132)

ArtUV 分成 SeamGPT seam prediction 和 parameterization refinement，可得到 artist-style UV，但它处理已有 mesh，而不是与 mesh generation 共同生成。[ArtUV](https://arxiv.org/abs/2509.20710)

Material Anything、MatLat、TexGaussian 等可以生成 PBR maps，但通常在 mesh 已完成后独立运行，容易出现：

- UV seam 处颜色/法线不连续；
- unseen region hallucination；
- albedo 与 lighting 混淆；
- metallic/roughness 不物理；
- topology refinement 后 texture correspondence 丢失。

[Material Anything](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Material_Anything_Generating_Materials_for_Any_3D_Object_via_Diffusion_CVPR_2025_paper.pdf) 使用两阶段 image-space material generation + UV-space refinement；这可以作为 PBR baseline，而不是 topology generator。

### 4.5 Rigging 的主要问题不是“有没有骨骼”

当前强方法包括：

- RigAnything：template-free BFS skeleton AR + skinning；
- UniRig：Skeleton Tree Tokenization + bone-point attention；
- Puppeteer：joint-token skeleton、topology-aware skinning 与动画优化；
- Rigel3D/AniGen：geometry 与 rig joint generation；
- AnyTop/SAMoR：arbitrary-skeleton motion。

[RigAnything](https://arxiv.org/abs/2502.09615)、[UniRig](https://arxiv.org/abs/2504.12451)、[Puppeteer](https://papers.neurips.cc/paper_files/paper/2025/hash/689cffc97600f9deb8374fc8fa918b8e-Abstract-Conference.html)、[AnyTop](https://arxiv.org/abs/2502.17327)

但 production rig 通常还需要：

- stable rest pose 与 local bone axes；
- twist/roll bones；
- joint limits；
- IK/FK controls、space switching 和 constraints；
- max influence count、weight pruning 和 normalization；
- pose-space corrective deformation；
- facial blendshapes、visemes、eyes/teeth/tongue/inner mouth；
- cloth/hair secondary motion；
- standardized naming 与 retarget profile。

这些几乎没有被当前“animation-ready generation”论文完整覆盖。

### 4.6 数据集交集极小

- Anymate：约 230K rigged assets，提供 skeleton + skinning，是目前最大的 rigging source 之一。[Anymate](https://arxiv.org/abs/2505.06227)
- Rig-XL：约 14K 多类别 rigged models，格式含 vertices/faces/joints/parents/names/local axes/skin weights。[UniRig/Rig-XL](https://github.com/VAST-AI-Research/UniRig)
- TexVerse-Skeleton/Animation 提供大规模 rig/animation 子集，但 clean quad topology、UV/PBR 和动作质量仍需重新审计。
- Objaverse/Objaverse-XL 数量大，但 topology、UV、naming、rest pose 和 license 质量差异巨大。
- QuadGPT 的大规模 quad 数据有相当部分来自自动 tri-to-quad conversion，不等于全部是手工 production topology。

真正同时具备 `clean quad mesh + semantic parts + UV + PBR + skeleton + weights + multiple animations + facial rig` 的公开数据非常少。这是整个方向最主要的瓶颈，而不是模型大小。

---

## 5. 提议方法：StudioRDD

## 5.1 表示：Progressive Artist Asset Graph

一个资产表示为：

\[
\mathcal A_k=(M_k,G_{part},G_{rig},U_k,T_k,W_k,B_k,C_k),
\]

其中：

- `M_k=(V_k,E_k,F_k)`：第 k 个 LOD 的显式 mesh；
- `G_part`：命名部件、parent-child assembly、material slots；
- `G_rig`：joints/bones/parents/local axes/limits；
- `U_k`：UV seams、chart IDs 和 coordinates；
- `T_k`：PBR mip features；
- `W_k`：per-vertex skin weights；
- `B_k`：blendshape/corrective deformation basis；
- `C_k`：collision/physical tags、hard edges、normals、semantic labels。

GT artist mesh 先通过 constrained progressive simplification 构造：

\[
M_0 \xrightarrow{S_0} M_1 \xrightarrow{S_1}\cdots\xrightarrow{S_{K-1}} M_K.
\]

简化时禁止跨越或强惩罚：

- UV seams；
- material boundaries；
- hard-normal/crease edges；
- part boundaries；
- 高 skin-weight gradient；
- 关节附近 deformation loops；
- facial landmarks 和 blendshape 高位移区域。

每次 collapse 都保存 inverse split record，因此 hierarchy 有真实、可逆的 parent-child correspondence。

## 5.2 连续 RDD 与离散 topology 必须分开

对于连续属性 `x_k`，定义由 progressive hierarchy 决定的 restriction/prolongation：

\[
R_k:x_{k+1}\rightarrow x_k,\qquad
P_k:x_k\rightarrow \tilde x_{k+1}.
\]

不再盲目使用 voxel average，而是：

- positions/detail：mass-lumped barycentric restriction；
- normals/tangents：在 local frame 中平均后归一化；
- UV：chart 内 barycentric restriction，绝不跨 seam；
- skin weights：先转 logits，再 area-weighted pool，最后 softmax；
- rotations：log-map 到 Lie algebra 后聚合；
- material：在 linear color/physical parameter space 或 learned latent 中聚合。

RDD drift 为：

\[
d_k=P_kR_kx_{k+1}-x_{k+1}.
\]

离散 topology 不做线性 drift。每个 stage 单独生成：

- split/no-split mask；
- split template；
- edge/face orientation；
- seam/material/part boundary continuation；
- 可选 handle/genus operation。

可以使用 categorical flow、absorbing diffusion 或 MaskGIT-style parallel decoding。应用 split grammar 后才获得下一 stage 的 domain，再运行 continuous RDD refinement。

## 5.3 推理流程

1. **Asset graph stage**：从图像/文本生成 named parts、rig/kinematic tree、粗 joint locations 和 material slots。
2. **Coarse cage stage**：生成 200–1000 face 的合法 triangle/quad control cage；所有 genus 和 component decisions 尽量在这里完成。
3. **Parallel topology refinement**：在当前 mesh 上预测一组互不冲突的 edge/loop split operations。
4. **RDD continuous refinement**：为新增 vertices 生成 geometry、UV、skin/material/detail residual。
5. **Adaptive activation**：只展开满足至少一个条件的 patch：几何 detail、UV/PBR detail、deformation detail、semantic importance。
6. 重复至目标 LOD；每个中间 `M_k` 均可直接导出。
7. 在 UV mip pyramid 上运行 2D RDD 生成 baseColor/normal/roughness/metallic/AO/emissive。
8. 生成/匹配 motion clips，做 deformation validation 与 corrective basis refinement。
9. 导出 glTF/GLB、USD；研究阶段另导出 Blender scene 和 FBX 以进行真实 DCC 验证。

## 5.4 为什么这比“RDD + Nexus/LATO”更 novel

仅替换 Nexus 的 vertex flow，novelty 很容易被评价为 solver/curriculum modification。StudioRDD 的独立贡献应是：

1. **Topology-changing RDD formulation**：首次在显式 progressive mesh hierarchy 上定义 resolution drift；
2. **Parallel valid topology refinement**：用局部 grammar 代替逐 token AR，同时保证每一 LOD 的合法性；
3. **Rig/UV-aware restriction and prolongation**：跨尺度映射保留 seams、parts、skinning 与 deformation semantics；
4. **Functional adaptive refinement**：计算分配不只看 occupancy，还看 deformation、texture 和 semantic residual；
5. **Native animation-ready generation**：与 Rigel3D/AniGen 的 implicit/field surface 明确区分；
6. **Production evaluation protocol**：用 DCC edit time、motion stress 和 engine import pass rate，而不只用 CD/HD。

“首次”措辞仍需在投稿前重新做一次最新论文检索。

---

## 6. Loss 设计

### 6.1 Geometry 与 topology

\[
L_{mesh}=\lambda_vL_{vertex}+\lambda_nL_{normal}+\lambda_eL_{edge}
+\lambda_fL_{face}+\lambda_{topo}L_{topo-valid}.
\]

- `L_vertex`：有 correspondence 时直接 L1/Huber；无 correspondence 时 CD/point-to-surface；
- `L_normal`：face orientation 与 shading normal；
- `L_edge`：length、sharp-edge、edge-density；
- `L_face`：GT face/split CE；
- `L_topo-valid`：boundary、non-manifold、自交、degenerate face、component/genus；
- quad 模式额外监督 valence、pole location、loop continuity 和 quad ratio。

局部 grammar 应提供硬约束；loss 用于 artist preference，而不是弥补所有 validity。

### 6.2 Deformation-aware topology

训练时从 GT 动作或 joint-limit prior 采样姿态 `q`：

\[
L_{def}=\mathbb E_q\big[
L_{surf}(D(\hat M,\hat W,q),D(M,W,q))
+\lambda_{flip}L_{normal-flip}
+\lambda_{arap}L_{ARAP}
+\lambda_{pen}L_{penetration}
\big].
\]

再加入：

- skin weight soft CE/L1；
- influence sparsity 与 max-4 influence constraint；
- joint-to-surface/bone-inside loss；
- edge-loop density 与 skin-gradient correlation；
- extreme-pose stress loss。

这是区别于仅静态监督 Rigel3D/UniRig 的关键实验。

### 6.3 UV/PBR

- GT seam edge BCE / preference reward；
- chart connectivity 与 semantic consistency；
- overlap penalty；
- conformal、area、symmetric-Dirichlet distortion；
- packing utilization 与 texel-density variance；
- multi-view render loss；
- seam consistency；
- PBR relighting consistency 与 material energy constraints。

### 6.4 RDD 与计算

- standard RDD/FM velocity loss；
- split-mask loss；
- activation/compute budget loss；
- coarse-to-fine representation consistency；
- endpoint pruning calibration；
- 对 geometry/UV/deformation residual 分别记录 activation recall，避免只优化速度而漏掉细节。

---

## 7. 整套动画资产应包含什么

## 7.1 最小“artist-usable”静态资产

- clean manifold mesh 或明确设计的 open boundaries；
- consistent face winding；
- triangle/quad topology；
- named parts 与 material slots；
- normals/tangents、hard edges；
- UV0 texture atlas；
- 可选 UV1 lightmap atlas；
- PBR metallic-roughness maps；
- 3–4 个 LOD；
- collision mesh；
- centimeter/meter units、up axis、pivot/origin。

## 7.2 Rigged asset

- skeleton hierarchy；
- rest/bind pose；
- bone local axes 和 inverse bind matrices；
- per-vertex normalized skin weights；
- joint names、symmetry 和 retarget profile；
- joint limits；
- twist/support bones；
- deform/attachment bones 分离；
- controls/constraints 可在 Blender/Maya 插件中程序化构建。

## 7.3 表情与高级形变

- FACS/ARKit-compatible facial blendshapes；
- visemes；
- eyelids/eyes、teeth、gums、tongue、inner mouth；
- pose-space corrective shapes；
- muscle/volume preservation；
- cloth/hair secondary controls。

RigAnyFace 可从任意 facial topology 生成 FACS poses；OmniFaceRig 进一步处理 inner mouth、teeth、tongue 与最多 155 blendshapes，可作为专用 face post-module，而不应假装一个通用模型已经解决面部 production rig。[RigAnyFace](https://arxiv.org/abs/2511.18601)；[OmniFaceRig](https://arxiv.org/abs/2606.08043)

## 7.4 动作

- locomotion、idle、turn、jump、interaction 基础 clips；
- root motion；
- contacts 与 foot locking；
- loopability；
- arbitrary-skeleton motion/retarget mapping；
- animation state metadata；
- 机械部件需要 joint type/axis/origin/limit，而不是 humanoid motion model。

AnyTop、OmniSkel、SAMoR 是更合理的 motion baseline；不要从头重做全部 motion generation，除非 mesh/rig 主论文已经成立。

---

## 8. 分阶段研究计划

### Phase 0：不要一开始做“全工作室”

先证明核心：

> RDD 能否在 progressive native mesh hierarchy 上，以并行、拓扑有效的方式生成多个 LOD，并在同等几何/拓扑质量下显著减少高分辨率计算？

范围：triangle mesh、无 texture、可选给定 point cloud。数据 20K–50K，2–4 个 LOD。

### Phase 1：Triangle Progressive Mesh RDD

- GT constrained edge-collapse hierarchy；
- coarse mesh generator；
- parallel non-conflicting vertex-split generation；
- continuous RDD vertex residual；
- 与 VertexRegen、ARMesh、Nexus、LATO.2、MeshRipple 对比；
- 指标：validity、edge/face F1、CD/HD/NC、polycount、runtime/VRAM、LOD consistency。

成功门槛：

- validity 不低于 strongest direct-mesh baseline；
- 同 CD/HD 下至少 1.5–2x wall-clock 改善；
- 10K+ faces 时无明显 AR-style error accumulation；
- 每个 LOD 都能单独通过 importer/mesh validator。

### Phase 2：Quad/strip refinement

- 以 QuadGPT、Mesh-Pro、SATO 为 teacher/reward；
- 引入 quad face split、edge-loop insertion 和局部 triangle exception；
- 保持 articulation loops、seams 和 hard edges；
- artist preference model + 少量真正 artist DPO。

不要声称“game-ready”，直到：

- Blender subdivision stress 通过；
- shoulder/elbow/knee/mouth deformation test 通过；
- 盲评中的 manual fix time 显著减少。

### Phase 3：UV + PBR

- joint seam prediction；
- differentiable unwrap/packing；
- 2D RDD UV-mipmap material generation；
- 对比 SATO+Blender unwrap、ArtUV、Material Anything、MatLat；
- 评估 overlap、distortion、packing、seam continuity、relighting。

### Phase 4：Rig-aware native mesh generation

- 先使用 GT rig 测 topology 是否因 deformation supervision 变好；
- 再接 UniRig/RigAnything 的 predicted rig；
- 最后训练 surface/skeleton 双 RDD stream；
- 对比 generate-then-rig、Rigel3D、AniGen；
- 必须使用 held-out motions 和 extreme poses，而不是只比较 rest pose joint distance。

### Phase 5：完整资产包

- face-specific blendshape module；
- AnyTop/SAMoR motion clips；
- part graph/URDF for machinery；
- LOD、collision、engine import；
- Blender add-on 自动构建 controls 和输出 USD/glTF/FBX。

这更适合作为第二篇论文或系统论文，不应全部塞进第一篇方法论文。

---

## 9. 评测协议：怎样证明“艺术家真的能用”

### 9.1 自动 validity gate

每个资产必须记录：

- invalid indices、NaN/Inf；
- degenerate faces；
- boundary/non-manifold edges；
- disconnected components；
- self intersections；
- inconsistent normals；
- UV overlaps、out-of-bounds、zero-area charts；
- texture missing/seam error；
- skeleton cycles、bones outside、duplicate joints；
- skin weights sum、negative weights、influence count；
- bind-pose reconstruction error；
- glTF validator、Blender/Maya/Unreal import pass/fail。

### 9.2 动画压力测试

- 随机 joint-limit poses；
- 标准 walk/run/jump clips；
- 极端 joint bend；
- normal flip、volume loss、surface penetration、foot slide；
- unseen skeleton retarget；
- mechanical open/close closure gap。

### 9.3 真正关键的 artist study

让专业 artist 完成相同任务：

1. 修复 mesh；
2. 调整 edge loops；
3. UV/texture editing；
4. weight painting；
5. retarget 一个动作；
6. 导入 engine。

记录：

- time-to-first-valid-import；
- manual edit minutes；
- topology/UV/rig 分项修改次数；
- 是否接受作为 blockout、background asset、game asset、hero asset；
- 最终盲评偏好。

“艺术家可用”应定义为相对人工/自动 pipeline 显著减少 time-to-ready，而不是完全零修改。Hero asset 零修改在当前技术水平下不诚实。

---

## 10. 风险与 kill criteria

### 10.1 最大风险

1. Progressive hierarchy 不是唯一的，同一 artist mesh 可以有多种 collapse history；
2. parallel splits 可能相互冲突，降低拓扑多样性；
3. quad loop insertion 比 triangle vertex split 难很多；
4. geometry、UV、rig、motion 多 loss 容易互相牵制；
5. clean full-asset data 远少于静态 Objaverse；
6. 动态稀疏节省可能被 graph/half-edge bookkeeping 吃掉；
7. direct mesh generation 与单图 geometry hallucination 混在一起会导致结论不清。

### 10.2 预先固定的停止标准

- triangle Phase 1 不能在 10K-face 档达到至少 1.5x 实测速度并保持 validity：停止扩展全资产；
- parallel topology 的 hole/non-manifold rate 高于 MeshRipple/Nexus：退回 coarse RDD + deterministic refinement；
- rig-aware loss 只改善静态指标、不改善 held-out motion deformation：不写 joint-generation claim；
- UV/PBR 没有降低 artist edit time：作为工程插件，不列核心 contribution；
- 数据清洗成本远高于模型收益：缩小到 characters 或 articulated objects 单域。

---

## 11. 推荐的最现实论文范围

### 主论文版本

标题方向：

> **Progressive Native-Mesh RDD: Parallel Coarse-to-Fine Generation of Valid Artist Meshes**

三项 contribution 足够集中：

1. progressive-mesh resolution drift；
2. parallel valid topology refinement + adaptive compute；
3. deformation-aware topology/LOD evaluation。

任务：point-cloud/image-conditioned native mesh；triangle 主结果、quad 小规模扩展；rig 作为 condition/functional evaluation，而不是同时声称完整 motion generation。

### 后续系统论文

> **StudioRDD: Generating Editable, Rigged and Materialized 3D Asset Packages**

在主方法稳定后加入：UV/PBR、semantic parts、skeleton/skinning、blendshapes、motion、Blender/engine package。

这种拆分比一次承诺“生成所有动画制作资源”更可信，也更容易形成两个清楚、互相递进的研究贡献。

---

## 12. 最终判断

最没有价值的方向是：

- 把 QuadGPT/MeshRipple 的 token sequence 生硬地 average pool；
- 只在 Nexus/LATO 的 vertex flow 上换成 RDD，然后声称 artist-ready；
- 继续使用 implicit mesh decoder，却把最终 GLB 称为 native topology generation；
- 拼接 static generator + UniRig + texture model + motion model，然后把 pipeline integration 当成核心 novelty。

最有价值的方向是：

> 用 progressive mesh operations 为 RDD 建立真正的显式拓扑 hierarchy；让 coarse-to-fine refinement 同时受 geometry、UV、rig 和 deformation 约束；以并行、合法、动态稀疏的方式生成每一层可编辑 LOD。

这是 RDD 与当前 native mesh / animation-ready generation 交叉处最有理论独立性、也最可能产生长期影响的方向。
