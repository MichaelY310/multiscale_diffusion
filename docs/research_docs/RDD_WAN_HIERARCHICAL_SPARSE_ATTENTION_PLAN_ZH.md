# RDD-Wan 分层稀疏注意力：调研、证伪实验与硬件实现方案

更新日期：2026-08-17

## 先说结论

“coarse-to-fine routing + block-sparse kernel”这个宽泛描述本身不新。VSA 已经在高分辨率 attention 内部先池化 Q/K、找关键 tile，再只对候选 tile 做细粒度 attention；更早的 QuadTree Attention 和 BiFormer 也做过 coarse-to-fine/top-k region routing。

RDD 真正可能提供的新东西不是“也做一个 coarse router”，而是：

> RDD 在正式进入高分辨率 stage 之前，本来就完整运行过低分辨率 stage。把已经付过计算成本的低分辨率 attention 当成高分辨率稀疏计算的 scout；跨 stage 搬运候选拓扑，只对不确定候选做很小的 fine correction，并用与 RDD 嵌套网格一致的 3D block-sparse kernel 执行。

我暂时把它叫做 **HARP-RDD（Hierarchical Attention Routing for Pyramidal RDD）**。

这个命题有明显风险。最新的 SparSTAR 在多尺度自回归视频里发现：重要块会随 scale 改变，直接搬运旧 scale 的 mask 不如在目标 scale 重算。CRPA 还说明，不同分辨率的 RoPE 采样率会造成相位错位。因此，默认方案不应该是“盲目直接复用 20% mask”，而应该是：

1. 先测跨 stage 可预测性；
2. 低分辨率 route 只生成一个较宽的候选集；
3. 在候选集内做便宜的 fine residual routing；
4. 对不稳定 layer/head 保留 dense 或目标 scale 自路由；
5. 只有数据证明足够稳定时，才升级成直接 mask reuse。

## 已完成的可行性检查（2026-08-17）

目前仓库和 surfi2 上都还没有训练完成的 RDD-Wan checkpoint。因此这次不能把结果冒充为“RDD-Wan 已验证”；我使用的是官方 Wan2.1-T2V-1.3B 权重、结构化合成 latent、零文本 context，并保持 `4x -> 2x -> 1x` 的 RDD 边界输入。它的作用是尽早证伪明显不成立的设计。真正的 go/no-go 必须在 RDD-Wan checkpoint、真实 VAE latent 和真实 prompt 上复测。

五个代表层（0/7/15/23/29）的平均结果：

| 比较 | route distribution cosine | JS divergence | source top-20% 覆盖 target attention mass |
|---|---:|---:|---:|
| coarse end → middle start | 0.7261 | 0.1242 | 39.90% |
| middle end → fine start | 0.9283 | 0.0369 | 66.11% |
| fine 相邻 timestep | 0.9974 | 0.00127 | 71.14% |

解释：

- `coarse -> middle` 太不稳定，不应作为主要 router；
- `middle -> fine` 明显有预测性，但 20% hard mask 还不够安全；
- `fine timestep -> next fine timestep` 极稳定，说明普通 timestep route reuse 很有希望；
- middle route 的 top-10/20/30/40% 分别覆盖约 55.2/66.1/73.7/79.5% 的 fine attention mass，而 target 自己的 oracle top-20% 也只覆盖约 70.6%。所以这里 attention 本身较弥散，不能只用“覆盖 80%”这一个指标判死刑，必须同时看 sparse attention output error 和最终视频质量。

直接在 fine Q/K/V 上使用 middle route 时，20/30/40% block density 的平均 attention-output cosine 分别是 0.9605/0.9743/0.9820，相对 L2 是 0.2689/0.2053/0.1636。平均值掩盖了很重要的层差异：20% density 在 layer 0 只有 0.8762 cosine，但 layer 7/15/23/29 是 0.9928/0.9696/0.9811/0.9827。因此最合理的第一版 policy 是“前层 dense 或较高密度，稳定的中后层才用 20%–30%”，而不是全模型统一稀疏率。

`source top-20% + local radius 1` 在这个缩小网格上达到 0.9883 output cosine，但其有效密度也膨胀到 45.3%；真实 `6x7x13` 根网格的局部邻域占比会小得多，所以不能把这个数直接外推成真实视频质量。

当前证据支持的不是“20% 直接跨尺度 hard reuse”，而是：

> middle route 产生较宽候选集，fine pooled Q/K 在候选内做 residual correction；进入 fine stage 后，再沿 timestep 复用修正后的 route。

硬件原型已在 RTX PRO 6000 Blackwell 上以实际 Wan padded token 数 `24x28x52=34,944`、12 heads、head dim 128 跑通。64-token FlexAttention block kernel 的 BF16 数值相对误差是 `5.94e-5`；dense SDPA 为 21.57 ms，10%/20%/30% 有效密度分别为 4.67/8.73/12.97 ms，即 4.62x/2.47x/1.66x kernel speedup。这里是单 attention microbenchmark，不是端到端 Wan speedup，也不是最终 FastVideo kernel。

## 它是不是 attention reuse？

部分相关，但不是通常所说的 attention reuse。

常见 reuse 方法（PAB、DiTFastAttn、TeaCache、FasterCache）主要沿去噪时间轴工作：同一分辨率下，相邻 timestep 的 attention output、block output 或特征很相似，于是跳过一次计算，直接使用之前的结果。

HARP-RDD 默认只复用**连接拓扑**：

- 不复用旧的 attention output；
- 不复用旧的 K/V 数值；
- 高分辨率 Q/K/V 仍然重新计算；
- 只复用“这个 query tile 值得看哪些 key tile”的候选索引；
- 对候选 tile 之间的 fine token attention 仍做真实计算。

所以更准确的名字是 **cross-stage route reuse / mask prediction**。它可以再和 timestep feature reuse 正交组合，但第一篇系统工作不应该把两件事混在一起。

## 最接近的工作和边界

| 工作 | 它做什么 | 与 HARP-RDD 的关系 |
|---|---|---|
| [VSA, NeurIPS 2025](https://arxiv.org/abs/2505.13389) | 每个目标 attention 内部池化 token，做 coarse route，再以 block kernel 计算候选块；Wan2.1 attention 6×、端到端 31s→18s | 最接近、必须作为主 baseline。HARP 若仍在 fine stage 完整重算同样 router，就没有方法新意 |
| [DFSAttn, ICML 2026](https://arxiv.org/abs/2605.23445) | 3D Hilbert 重排、hierarchical block scoring、跨 timestep adaptive mask cache、block-sparse kernel；Wan2.1 约 1.79x E2E | 覆盖了“层次打分 + mask cache + kernel”这句宽泛主张。HARP 唯一清楚的区别必须是跨 RDD resolution stage 的 scout/parent memory，而不是普通 timestep cache |
| [Sparse VideoGen](https://arxiv.org/abs/2502.01776) / [SVG2](https://arxiv.org/abs/2505.18875) | 动态识别时空稀疏 head；SVG2 用 semantic permutation、top-p budget 和定制 kernel，Wan2.1 报告最高约 1.89x | 证明 token layout/reordering 与找对 mask 同样重要；必须做相同硬件上的 latency-quality Pareto 对比 |
| [MOD-DiT](https://arxiv.org/abs/2601.11641) | 用早期去噪步拟合随时间变化的 mask 分布，并在线保留历史稀疏信息 | 与“先前计算预测后续 mask”高度相近，但仍是同一 resolution 的时间轴预测，不是 RDD 跨分辨率 parent-child 继承 |
| [SVG-EAR](https://arxiv.org/abs/2603.08982) | 用 block centroid 近似被跳过的 tail，再按估计误差选择必须精算的 block | 直接启发 HARP 的安全版本：parent route 选 exact block，未选 block 用 RDD parent summary 补偿；必须作为 baseline |
| [Sliding Tile Attention, ICML 2025](https://arxiv.org/abs/2502.04507) | 固定局部 3D sliding tile，并做硬件友好的 kernel | kernel 设计基线；没有内容自适应跨 stage routing |
| [QuadTree Attention, ICLR 2022](https://arxiv.org/abs/2201.02767) | 从粗到细选 top-k 区域 | 证明“粗到细 routing”不是新概念，但不是 diffusion trajectory 的跨 stage 免费 scout |
| [BiFormer, CVPR 2023](https://arxiv.org/abs/2303.08810) | region-level 筛选后做 token-level attention | bi-level routing 的通用先例 |
| [DSV](https://arxiv.org/abs/2502.07590) | 动态稀疏训练、定制 kernel、稀疏感知 context parallel | 训练和分布式系统 baseline；不利用 RDD stage 继承关系 |
| [Sparse-vDiT](https://arxiv.org/abs/2506.03065) | 离线按 layer/head 搜索 diagonal、multi-diagonal、stripe 模式 | 硬件友好但大多内容无关；可作为 static-pattern baseline |
| [Radial Attention, NeurIPS 2025](https://arxiv.org/abs/2506.19852) | 根据时空距离能量衰减使用静态 O(n log n) mask | 强局部先验 baseline，可作为 HARP 的必保 local branch |
| [PAB](https://arxiv.org/abs/2408.12588) | 沿 timestep 广播 attention output | 是真正的 time reuse，可与 HARP 正交组合 |
| [DiTFastAttn](https://arxiv.org/abs/2406.08552) | window attention、跨 timestep/CFG attention sharing | 同上；不能被当成 HARP 的 novelty |
| [SparSTAR](https://arxiv.org/abs/2608.10519) | 多尺度自回归视频 block sparsity；实测跨 scale mask 直接迁移不可靠 | 对 HARP 的直接反例风险。任务不同，但必须正面比较“直接迁移 vs 目标 scale 重算 vs residual correction” |
| [CRPA](https://arxiv.org/abs/2511.19778) | 对齐不同分辨率的 RoPE 物理相位 | 解释 route 跨 scale 失效的一个机制；HARP 必须做 native RoPE / phase-aligned / pre-RoPE 三组消融 |
| [VMoBA](https://arxiv.org/abs/2506.23858) | 按 layer 使用 1D/2D/3D block partition，并按 head 动态选择重要块 | 强动态 block-sparse baseline；其目标 scale router 不利用 RDD 已经算过的低分辨率 stage |
| [SLA](https://arxiv.org/abs/2509.24006) | 把高权重部分做 sparse attention、边缘部分做 linear attention，并融合到单 kernel | 比纯 hard mask 更能保留弥散的 attention tail；可以成为 HARP residual branch 的替代实现 |
| [ScalingAttention](https://arxiv.org/abs/2606.23019) | 离线提取 prompt-agnostic、head-wise sparse topology，并用硬件对齐 kernel | 非常强的新 baseline；HARP 必须证明 RDD 动态 scout 在运动/语义依赖上优于纯权重先验，或与其先验并集更好 |
| [FG-Attn](https://arxiv.org/abs/2509.16518) | 比 64x64 block 更细的 Mx1 稀疏粒度，并实现异步 gather kernel | 说明 64-token block 可能浪费块内稀疏；第一版仍用成熟 block kernel，后续可比较细粒度下界 |
| [PyramidalWan](https://arxiv.org/abs/2601.04792) | 把 Wan2.1 变成 4×→2×→1× 时空分辨率的 pyramidal model | 质量与多分辨率计算基线；本身没有把低分辨率 attention 用作 fine sparse router |

## 为什么 RDD-Wan 的网格特别适合做这件事

论文设置是 81 帧、448×832 像素。Wan VAE 后约为 21×56×104 latent，patch size 是 1×2×2，所以 fine DiT token 网格为：

```text
fine:   21 × 28 × 52 = 30,576 tokens
middle: 11 × 14 × 26 =  4,004 tokens
coarse:  6 ×  7 × 13 =    546 tokens
```

给 fine 时间轴补到 24 后：

```text
fine padded = 24 × 28 × 52
            = (6 × 4) × (7 × 4) × (13 × 4)
```

因此一个 coarse token 精确对应一个 fine 的 `4×4×4=64` token tile；middle 补到 `12×14×26` 后，一个 `2×2×2` middle token 组也对应同一个根 tile。三层共享同一个 `6×7×13=546` 根网格。

这带来两个实际价值：

- route 不需要学习任意的坐标映射，parent/child index 是确定的整数关系；
- 64 token 恰好是 VSA 当前 kernel 的 tile volume，能够直接接硬件实现，而不是在 Python 里生成 dense mask。

## 明确的方法方案

### 1. middle stage 充当主要 scout

不要主要依赖 coarsest stage。RDD-Wan 的 coarse stage 结束在 data time 约 0.06，此时信号很弱；middle 结束在约 0.41，语义和运动结构更可靠，而且下一步正好进入最昂贵的 fine stage。

对每个 layer `l`、head `h`，在 middle stage 最后一个 denoiser call 保存聚合到根网格的 route：

```text
R_mid[l,h,q_root,k_root]
```

只保存 top-k block id，不保存完整 attention 矩阵。

### 2. 三路并集保证鲁棒性

每个 fine query root 的候选 key roots 为：

```text
M = M_parent ∪ M_local ∪ M_anchor
```

- `M_parent`：middle route 的 top-k 候选；
- `M_local`：同帧/邻帧的 3D 邻域，防止纹理和运动连续性断裂；
- `M_anchor`：第一帧、时间相邻帧、必要的全局 sink/summary tile。

文本 cross-attention token 很少，第一版保持 dense；只改视频 self-attention。

### 3. 推荐版不是硬复用，而是 candidate + residual correction

直接用 20% transferred mask 风险过高。推荐：

1. 根据当前 pilot，middle route 先保留约 40% 根块，形成 candidate set；
2. fine Q/K 已经由原模型线性层产生；只在 candidate root pairs 上计算很便宜的 pooled score；
3. 从 candidate 中再选到最终 20%–30%；若训练后 20% 质量不够，就保留 30% 或为 attention tail 增加 SLA 式 linear residual；
4. 用 block-sparse attention 对这些块做完整 token attention。

这样把 fine router 的根块打分从 `R²` 缩到约 `0.4R²`，同时避免完全相信跨 scale mask。只有在真实 RDD-Wan 上同时满足低 output error 和视频质量 parity 时，才去掉 residual router；attention mass coverage 只作为诊断指标，不再单独决定 go/no-go。

### 4. RoPE 对齐

Wan 的 3D RoPE 使用 token index；同一物理位置在 coarse 和 fine 的 index 步长不同。至少比较三种 route：

1. native route：直接用 previous-stage post-RoPE attention；
2. content route：用 pre-RoPE pooled Q/K，检查内容本身是否一致；
3. phase-aligned route：把 source root Q/K 按 fine root 的物理坐标重新施加 RoPE，再打分。

如果 2/3 显著优于 1，问题主要是相位而不是语义，HARP 仍然可行；如果三者都差，说明跨 stage route 本身不稳定，应退回 VSA 式目标 scale routing。

### 5. layer/head 自适应，而不是全模型一刀切

probe 为每个 layer/head 输出 coverage 和 output error。根据验证集统计固定一张 policy 表：

```text
stable head   -> direct transferred route
medium head   -> transferred candidate + fine correction
unstable head -> ordinary VSA or dense attention
```

policy 在离线校准后固定，kernel 不做复杂 Python 分支。

### 6. 可选的 timestep reuse

进入 fine stage 后，可在多个相邻 step 复用已经修正过的 fine route index，但仍重新计算候选块内 attention。用路由变化量或 TeaCache 式 timestep signal 决定 refresh。

这是第二层优化：

```text
cross-stage reuse: middle -> fine 的候选先验
cross-step reuse:  fine step n -> n+1 的最终 sparse topology
```

二者要分别消融。

## 计算量到底省在哪里

fine dense self-attention 的 QK 和 AV 核心复杂度与 `N²d` 成正比。`N=30,576` 时它远大于 middle/coarse attention。若最终有效 block density 为 `rho`，attention pair 理论上约减少到 `rho`；例如 `rho=0.2` 是约 5× pair reduction。

但端到端不会有 5×，因为 QKV/O projection、FFN、cross-attention、VAE 和调度没有一起减少。VSA 在 Wan2.1 上报告的是 attention 6×、端到端 31s→18s，这个数量级是更现实的参考。

HARP 相对 VSA 额外减少的是：

- 每层每 step 的完整 fine root-router 打分；
- top-k/排序和动态 mask 建图；
- 如果跨 fine timesteps 复用 route，可进一步减少 route refresh；
- 固定/半固定 CSR topology 更适合 CUDA Graph 和持久 kernel。

必须诚实：router 本身通常不是总计算的大头，所以 HARP 若只比 VSA 少一个 router，端到端增益可能不大。论文价值需要同时证明以下至少一项：

- 相同稀疏率下，RDD scout route 比普通 VSA route 质量更好；
- 相同质量下，HARP 能用更高 sparsity；
- 无需 VSA 那样的大规模 sparse finetuning；
- route amortization 和静态 topology 显著改善实际 kernel/多 GPU 通信，而不只是 FLOPs。

我们的 microbenchmark 进一步量化了这个问题：在 20% density 时 sparse attention 是 8.73 ms，目标 scale 的 VSA-like root router 仅约 0.127 ms，而复用现成 index 的搬运约 0.016 ms。也就是说，**单纯省掉 router 最多只省这一 attention call 的约 1.3%**，不可能单独支撑一篇强硬件论文。真正值得做的目标必须改成：

1. RDD prior 让最终 exact density 比 VSA/DFSAttn 更低；或
2. 用 parent summary 补偿被跳过的弥散 tail，在同样 20% exact density 下质量明显更高；或
3. 在 context parallel 中只交换 route 选中的远端 KV，显著减少通信；或
4. 以更少/不需要 sparse finetuning 达到同等质量。

若四条都不能证明，应该把该方案降级为 RDD-Wan 的工程优化，而不是论文核心贡献。

## 推荐的增强版：exact children + parent-summary tail

当前 attention 的 oracle top-20% 也只覆盖约 70.6%，说明直接删除剩余 80% 的 block 会丢掉大量“每一块都很小、加起来却不小”的 tail。更合理的 RDD 原生版本是：

```text
selected child blocks:  用 fine K/V 做精确 block-sparse attention
skipped root blocks:    用上一个 RDD stage 已缓存的 parent K/V summary 近似
两部分:                 在同一个 online-softmax 归一化中融合
```

若一个 parent root 代表 `n_b` 个 fine key，summary logit 使用：

```text
q_i dot k_parent_b / sqrt(d) + log(n_b)
```

`log(n_b)` 表示这个 summary 代表了多少个 child key；selected root 不再走 summary，避免重复计数。这个做法把被跳过部分从“直接置零”变成低成本 coarse approximation，和 RDD 的低分辨率已计算状态发生真正的方法耦合。风险是 previous-stage K/V 与当前 fine K/V 有分布差，尤其有 RoPE 相位差，因此要比较：

- previous-middle summary（真正可复用方案）；
- 当前 fine K/V pooled summary（需要当前计算的上界）；
- previous-middle + phase alignment / 小型 correction projection；
- 直接丢弃 tail。

这一分支如果有效，比“只省 0.1 ms router”更有研究价值；如果 previous-middle 明显不如 current-fine upper bound，则只训练一个小 correction projector，不能直接声称 hidden-state reuse 成立。

该对照也已经跑完。在 20% exact density 下：

| tail 处理 | output cosine | relative L2 |
|---|---:|---:|
| 直接丢弃 tail | 0.9605 | 0.2689 |
| previous-middle parent summary | 0.9648 | 0.2509 |
| 当前 fine pooled summary（上界） | 0.9891 | 0.1329 |

结果很清楚：原样跨 stage 复用 parent K/V 只带来小幅平均改进，不能直接使用；但“exact sparse children + coarse summary tail”这个计算形式本身非常有效。layer 0 从直接丢 tail 的 0.8762 提升到 parent-tail 的 0.9294，说明 parent 对最不稳定前层有帮助，但其他层存在 stage distribution mismatch。最终实现应在两条路线中二选一：

1. 稳妥版：在 fine 当前 Q/K/V 上 pooling 出 root summary，做 fused sparse + summary attention；RDD middle route 仅负责 candidate prior；
2. 更 RDD-native 的研究版：对 cached parent K/V 加很小的 per-layer/head low-rank或 affine correction，使其逼近 current-fine pooled summary，再融合。

第二条更有 novelty，但必须通过训练证明 correction 成本远小于在 fine stage 重建 router/summary，并且视频质量与端到端 latency都占优。

## 硬件实现路线

### 原型阶段

使用 PyTorch FlexAttention 验证：

- route 语义和数值正确；
- 64-token block mask 真正跳过 QK/AV；
- 测 dense SDPA vs sparse latency；
- 测 route 构建/搬运开销。

FlexAttention 适合验证，不是最终最快实现。

### 论文 kernel 阶段

不要从零再造整个 attention 栈。基于 [FastVideo](https://github.com/hao-ai-lab/FastVideo) 的 VSA 64-token kernel 增加“外部 route index”入口：

```text
harp_sparse_attention(q, k, v, row_ptr, col_idx, block_sizes)
```

数据布局：

- 先把 `T×H×W` token 重排为 tile-major；
- 一个 tile 连续存放 64 个 token；
- `row_ptr/col_idx` 使用 CSR；546 个 root 可用 uint16/int32；
- 每个 query row 的 key block id 排序，增加 K/V 合并读取；
- route buffer 按 layer/head 存放，并保持在 GPU；
- 30 layer、12 head、546 query root、每行 64 个 key、uint16 约 25MB，可接受。

kernel：

- CTA/program 对应 `(batch, head, query_tile)`；
- query tile 常驻 register/shared memory；
- 顺序遍历 CSR 中选中的 key/value tile；
- 使用 FlashAttention 的 online softmax，绝不物化 attention matrix；
- BF16 起步，之后再考虑 FP8/FP4；
- padding token 用 `block_sizes` 屏蔽；
- 前向跑通后再实现 backward，第一篇可以先做 inference-only sparse retrofit。

多 GPU context parallel：route 决定每张卡需要哪些远端 KV tile。可先 all-to-all 交换选中 tile，而不是 dense all-gather；这部分若做扎实，会比“又一个 sparse mask”更像 Hao Zhang 组关注的系统贡献。

## 训练方案

第一阶段不要一上来训练新 backbone。

1. dense RDD-Wan checkpoint 作为 teacher；
2. sparse student 从同一 checkpoint 初始化；
3. 先冻结 backbone，只训练 phase-align/correction router；
4. 再对 attention projection 或 LoRA 做短期微调；
5. sparsity 从 0% 逐渐升到 50%、70%、80%。

建议 loss：

```text
L = L_RDD-FM
  + lambda_out * sum_l ||O_sparse_l - O_dense_l||²
  + lambda_route * sum_l KL(A_dense_root_l || A_router_l)
```

`L_RDD-FM` 保证生成任务；output distillation 约束 block-sparse 替换；route KL 只教 router 找候选，不要求复制完整 token attention。

## 实验矩阵

### Phase 0：不训练，先证伪

对 held-out prompts/latents，逐 layer/head 比较：

- coarse_end → middle_start；
- middle_end → fine_start（主结果）；
- fine_start → fine_next（普通 timestep reuse 对照）；
- native RoPE / pre-RoPE / phase-aligned；
- top 10/20/30/40% 的 attention mass coverage；
- sparse output cosine、relative L2；
- route 在 prompt、seed、motion strength 间的方差。

建议 gate：

- 20% route coverage ≥ 0.80 且 output cosine ≥ 0.95：可以试 direct reuse；
- coverage 0.65–0.80：只做 candidate + residual correction；
- coverage < 0.65：停止“直接跨 stage route”主张，改为 RDD-aware sparsity budget 或 ordinary VSA。

### Phase 1：小规模质量实验

同一个 RDD-Wan checkpoint、prompt、seed、solver、steps：

1. dense attention；
2. local/radial static mask；
3. VSA self-router；
4. direct transferred route；
5. transferred candidate + fine correction；
6. 5 + cross-step route reuse。

先 100 prompts × 3 seeds，筛掉明显坏方案。报告 paired PSNR/SSIM/LPIPS、输出 cosine、VBench 子项、闪烁和 motion smoothness。

### Phase 2：论文评估

- VBench、VBench-2.0 完整 prompt set；
- 至少单卡和双卡结果；
- kernel latency、attention latency、denoiser latency、端到端 latency；
- peak memory、实际 TFLOPs/MFU、route overhead；
- 与 dense RDD-Wan、PyramidalWan、VSA、STA/Radial、PAB/TeaCache 比较；
- 同质量 speedup 和同延迟 quality 两条 Pareto 曲线；
- 真实 RDD-Wan checkpoint，而不是官方 Wan init。

## 需要补哪些前置知识

推荐阅读顺序：

1. 先理解标准 multi-head attention、Q/K/V、softmax；
2. 读 [FlashAttention](https://arxiv.org/abs/2205.14135)：重点是 tiling、HBM/shared memory、online softmax；
3. 读 [Wan2.1](https://arxiv.org/abs/2503.20314) 和官方 `wan/modules/model.py`：重点是 3D token、3D RoPE、self/cross attention；
4. 读自己的 RDD-Wan stage schedule，再读 [PyramidalWan](https://arxiv.org/abs/2601.04792)；
5. **最重要**：读 [VSA](https://arxiv.org/abs/2505.13389) 和 FastVideo 的 `video_sparse_attn.py`；
6. 读 [DFSAttn](https://arxiv.org/abs/2605.23445)，因为它最直接覆盖 hierarchical scoring、mask cache 和 sparse kernel；
7. 读 [STA](https://arxiv.org/abs/2502.04507)，理解“理论 sparse 不等于硬件快”；
8. 读 [FlexAttention](https://arxiv.org/abs/2412.05496)，理解 block mask 如何 lower 成 kernel；
9. 读 [CRPA](https://arxiv.org/abs/2511.19778) 和 [SparSTAR](https://arxiv.org/abs/2608.10519)，专门理解本方案最可能失败的两个原因；
10. 最后读 PAB、TeaCache、FasterCache，区分 mask reuse 与 feature/output reuse。

CUDA/Triton 需要掌握：thread block/warp、HBM/L2/shared memory、coalesced access、Tensor Core tile、occupancy、register pressure、CUDA event benchmark。先能改 FastVideo VSA 的 route 输入，不需要先从零写 FlashAttention。

## 可执行脚本

```bash
# 一次性：隔离环境、Wan 源码、仅下载 transformer
bash rdd_video/attention_analysis/setup_probe_env.sh

# GPU 0 跑模型 route probe，GPU 1 跑真实 Wan shape FlexAttention benchmark
bash rdd_video/attention_analysis/run_study_2gpu.sh

# 挂 tmux
bash rdd_video/attention_analysis/launch_study_tmux.sh

# 有真正 RDD-Wan checkpoint 后复测
RDD_WAN_CHECKPOINT=/path/to/rdd_wan.pt \
RDD_WAN_LATENT=/path/to/cropped_real_vae_latent.pt \
RDD_WAN_CONTEXT=/path/to/precomputed_wan_t5_context.pt \
  bash rdd_video/attention_analysis/run_study_2gpu.sh
```

输出目录默认是：

```text
eval_results/rdd_wan_attention_routing/
```

其中 `model_probe/routing_probe.json` 是方法 go/no-go 的核心证据，`kernel_benchmark.json` 是硬件原型证据。
