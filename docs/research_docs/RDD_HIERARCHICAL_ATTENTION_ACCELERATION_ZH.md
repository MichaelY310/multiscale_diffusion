# RDD 分层注意力路由与分数复用：实现、实验与结论

更新时间：2026-08-20

## 1. 先给结论

这次得到的不是“已经超过 FastVideo 的完整结论”，而是一个已经能运行、确实加速、且把关键假设逐一验证过的 RDD 专用 attention 系统原型。

最值得继续发展的方案是：

> **上一分辨率只负责给下一分辨率提供 block route；当前分辨率仍周期性重算自己的 sparse attention score；两个 refresh 之间复用 score，但始终乘当前的 V。**

具体实现为：

1. 8×8 stage 计算 dense attention，得到第一份空间路由；
2. 8×8 路由决定 16×16 stage 要访问的 2×2 token micro-block；
3. 16×16 只在 refresh step 计算 sparse QK、online softmax 和 AV，中间 step 只计算 V，并执行缓存的 `A @ V_current`；
4. 16×16 stage 末尾的 sparse attention 再聚合成路由，决定 32×32 的 4×4 token blocks；
5. 32×32 重复 sparse refresh + temporal score reuse；
6. 全流程由 Triton kernel 执行；严格配置在 8×8 之后不会构造 dense QK 矩阵。

当前最重要的实验结论有四个：

- **同一 resolution 内可以复用 attention score，但 K=10 太激进；K≈3 更可靠。**
- **跨 resolution 的 route 有用，直接复制完整 attention score 不可靠。**
- **真正的严格级联已经能加速，但当前 ImageNet-256 的 token 太少，额外收益只有约 1.084×。**
- **允许每三步 dense refresh、其余 step 复用 score 时，三段 denoiser 可达到约 1.142×；主要收益来自跳过 Q/K projection，而不只是 sparse QK。**

因此论文里最稳妥的主方法不是“直接复用低分辨率 attention score”，而应是：

> **RDD-native hierarchical route propagation + within-stage attention-score reuse + hardware-aware sparse kernels.**

## 2. 这和 FastVideo / VSA / DraftAttention 到底有什么区别

### 2.1 VSA / FastVideo

VSA 在当前 full-resolution layer 内：

1. 对当前 Q/K/V 做 pooling；
2. 用 pooled Q/K 算 coarse attention；
3. Top-K 选择 tile pairs；
4. fine sparse kernel 只计算选中的 token pairs。

VSA 的 coarse stage FLOPs 很小，但论文报告在 87.5% sparsity 下仍约占 attention runtime 的 14%。其 Wan-2.1 retrofit 在约 23K tokens 上把生成时间从 31 秒降到 18 秒，约 1.7×。

论文：https://arxiv.org/abs/2505.13389

RDD 的不同点是：低分辨率 stage 原本就必须生成，不需要在每个 fine layer、每个 timestep 再人为建立一份 pooled Q/K surrogate。前一 stage 的 attention 是实际生成轨迹的一部分，可以直接提供 route prior。

但是这不意味着 RDD 可以完全相信 coarse score。实验表明跨分辨率完整 score 的数值变化明显，因此正确用法是：

- **跨 resolution 传 route；**
- **在 fine resolution 内重新计算选中 blocks 的 score；**
- **只在相邻 timestep 间复用数值 score。**

### 2.2 DraftAttention

DraftAttention 仍然是 full-resolution diffusion。它从当前 full-resolution feature 中下采样得到 draft Q/K，用低分辨率 draft attention 指导同一步 full-resolution sparse attention，并通过 token reordering 提供硬件友好的结构稀疏。论文报告 H100 上最高约 1.75×。

论文：https://arxiv.org/abs/2505.14708

RDD 的 route 来自真实的前一生成 stage，不需要在当前 fine step 再计算一遍 draft Q/K。这是 RDD 最有价值的系统差异。

### 2.3 attention score reuse 不是单独的新点

下面这些工作已经研究过跨 timestep 的 attention 或 feature reuse：

- PHAST / HURRY：https://arxiv.org/abs/2401.01008
- DiTFastAttn：https://arxiv.org/abs/2406.08552
- PAB：https://arxiv.org/abs/2408.12588
- TeaCache：https://arxiv.org/abs/2411.19108

因此不能把“每 K 步复用 attention”单独写成 novelty。可主张的新组合是：

1. route 来自 RDD 的真实 coarse generative stage；
2. route 以 8→16→32 级联传播，而不是每一步重新建立 coarse surrogate；
3. 跨 stage 只复用结构 route，同 stage 才复用数值 score；
4. route、score refresh、V-only cache hit 由统一的 block-sparse backend 执行。

## 3. 数学上到底复用了什么

标准 attention 为：

\[
A_t=\operatorname{softmax}\left(\frac{Q_tK_t^\top}{\sqrt d}\right),
\qquad
O_t=A_tV_t.
\]

### 3.1 同 resolution score reuse

假设第 \(t_0\) 步是 refresh step，缓存 \(A_{t_0}\)。接下来的若干步使用：

\[
\widetilde O_t=A_{t_0}V_t.
\]

注意这里的 \(V_t\) 仍是当前 step 的 V。这样可以跳过当前 step 的 Q projection、K projection、QK 和 softmax，但不会把内容特征完全冻住。

这比 PAB 风格的整输出广播

\[
\widetilde O_t=A_{t_0}V_{t_0}
\]

保守得多。我们的 probe 也显示 `A_old @ V_current` 显著比 `A_old @ V_old` 接近当前真实输出。

### 3.2 跨 resolution route propagation

把 token attention \(A\) 按空间 blocks 聚合：

\[
R_{ab}
=
\frac{1}{|B_a|}
\sum_{i\in B_a}\sum_{j\in B_b}A_{ij}.
\]

这里 \(R_{ab}\) 的意思是：“query block \(a\) 总体有多关注 key block \(b\)”。对每个 query block 保留 Top-K key blocks，得到下一 resolution 的 block indices。

我们没有把 \(R\) 当作下一 stage 的最终 attention probability；它只决定下一 stage 的 sparse QK 需要计算哪些 block pairs。

### 3.3 为什么不直接跨 resolution 复制 score

直接复制相当于：

\[
\widetilde O^{2r}_t
=
\operatorname{Lift}\left(A^r_{\mathrm{boundary}}\right)V^{2r}_t.
\]

这隐含假设 coarse token 内的所有 fine children 拥有近似相同的 query-key 关系。实际模型不满足这个强假设，尤其 fine stage 会形成新的边缘、纹理和部件关系。

所以：

- route 是低带宽的结构先验，可以跨 stage；
- 完整 probability 是高带宽的数值对象，不宜跨 stage 直接复制。

## 4. attention dynamics 实测

模型：RDD Unified XL/2 400K；相同 class、seed 和采样轨迹。下表为跨若干 denoiser calls 的平均统计。

### 4.1 Unified XL 同 resolution

| Resolution | 间隔 | score cosine | route cosine | `A_old @ V_current` cosine | 整输出广播 cosine |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.9833 | 0.9833 | 0.9944 | 0.9679 |
| 16 | 5 | 0.9467 | 0.9467 | 0.9820 | 0.8990 |
| 16 | 10 | 0.9188 | 0.9188 | 0.9711 | 0.8302 |
| 32 | 1 | 0.9853 | 0.9946 | 0.9960 | 0.9744 |
| 32 | 5 | 0.9548 | 0.9847 | 0.9871 | 0.9390 |
| 32 | 10 | 0.9216 | 0.9720 | 0.9766 | 0.8977 |

关键解释：32×32 的 route 在间隔 10 时仍很稳定，但完整 score 已经明显变化。也就是说，旧 attention 仍能告诉我们“该看哪里”，却不能精确告诉我们“每个 token 应该分多少概率”。这正好支持 route propagation，而不支持长期直接 score copy。

### 4.2 跨 resolution

| 模型 | 边界 | route cosine | Top-40% overlap | 直接 score reuse 输出 cosine | relative L2 |
|---|---|---:|---:|---:|---:|
| Unified XL | 8→16 | 0.7993 | 0.6365 | 0.8252 | 0.5282 |
| Unified XL | 16→32 | 0.7570 | 0.6735 | 0.8050 | 0.5478 |
| Multistage B/L/XL | 8→16 | 0.8483 | 0.5679 | 0.7719 | 0.6641 |
| Multistage B/L/XL | 16→32 | 0.8011 | 0.6389 | 0.7772 | 0.5730 |

这说明 Unified 与 Multistage 都支持“coarse route 有中等相关性”，但都不支持“完整 coarse score 可以原样复制”。

原始分析数据：

- `eval_results/rdd_attention_dynamics/unified_xl_400k_seed20260820.json`
- `eval_results/rdd_attention_dynamics/multistage_300300400_400k_seed20260819.json`

## 5. 已实现的 kernel

文件：

- `2d/attention_analysis/rdd_block_sparse_triton.py`
- `2d/attention_analysis/sample_rdd_hierarchical_kernel.py`
- `2d/attention_analysis/test_rdd_block_sparse_triton.py`
- `2d/attention_analysis/benchmark_rdd_block_sparse_triton.py`

包含六条真正的 Triton 路径：

1. 4×4 token block sparse QK + online softmax + AV；
2. 同上，但同时保存 sparse probabilities；
3. 使用缓存 probability 的 sparse AV；
4. 2×2 token micro-block sparse QK + online softmax + AV；
5. micro-block refresh + probability store；
6. micro-block cached AV。

cache hit 时，采样器还会从 fused QKV linear 的最后三分之一切出 V 权重，只计算 V projection。

正确性测试：

- FP32 micro sparse / store / reuse 相对 dense reference 的 relative L2 约 \(3.23\times10^{-7}\)，cosine 1.0；
- BF16 micro sparse relative L2 约 0.00737，cosine 0.999973；
- 4×4 FP32 kernel relative L2 约 0.0014，主要来自 TF32/tensor-core 数值路径，cosine 大于 0.999999；
- 4×4 BF16 cosine 约 0.999985。

## 6. 速度—质量结果

### 6.1 评估协议

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition；
- 模型：RDD Unified XL/2 400K；
- classes：207、282、323、817；
- CFG：1.5；
- DDPM sampling allocation：80/80/90；
- 相同 initial noise、transition noise 和 seed；
- 每个速度点运行 3 次，报告中位数；
- 时间口径是 **8/16/32 三段 denoiser loop 总时间**；
- 不包含 checkpoint load、VAE decode、PNG 保存；stage 间 noise-filling 也在计时区间外；
- 图像指标是与同噪声 dense 输出的保真度，不是 50K FID。

### 6.2 三次重复的主结果

| 配置 | 时间（秒）↓ | 相对 dense 加速↑ | image cosine↑ | PSNR↑ | post-8 dense QK | 结论 |
|---|---:|---:|---:|---:|---:|---|
| dense RDD | 4.6336 | 1.000× | 1.0000 | ∞ | 4760 | 基线 |
| score K=3，全层 | 4.0581 | **1.142×** | 0.99526 | 31.29 | 1652 refresh | 最快的稳健点 |
| adaptive 5/3/3 | 4.0780 | **1.136×** | **0.99642** | **31.87** | 1537 refresh | 推荐质量优先点 |
| 严格无 dense，balanced | 4.2754 | **1.084×** | 0.99312 | 29.83 | **0** | 满足用户的严格级联要求 |
| 严格无 dense + 跨 stage 直接 score | 4.2415 | 1.092× | 0.98572 | 26.49 | **0** | 明显掉质量，拒绝 |

说明：`post-8 dense QK` 只统计 16/32 stages。8×8 始终 dense。

### 6.3 补充探索点

| 配置 | 加速 | image cosine | 状态 |
|---|---:|---:|---|
| K=5，仅前 7/28 层 | 1.027× | **0.99911** | 极保守、安全 |
| K=5，全层 | 1.180× | 0.98888 | 更快但误差明显，不作为默认 |
| 严格无 dense，aggressive | 1.088× | 0.98917 | 单次探索，质量低于 balanced |

结果文件：

- `eval_results/rdd_hierarchical_kernel_final_repeat3_metrics.json`
- `eval_results/rdd_hierarchical_kernel_nodense_cfg15_metrics.json`
- `eval_results/rdd_hierarchical_kernel_pareto_cfg15_metrics.json`
- `eval_results/rdd_hierarchical_kernel_final_repeat3_comparison.png`

### 6.4 图片里能看到什么

`score K=3` 和 `adaptive 5/3/3` 与 dense 肉眼非常接近。严格 balanced 的主体和布局仍一致，但局部纹理已经有轻微变化。跨 resolution 直接 score reuse 在猫脸和蝴蝶等高频结构上出现明显漂移，和数值指标一致。

## 7. 为什么当前加速没有 VSA 那么大

当前 DiT-XL/2 的三段 token 数只有：

\[
N_8=16,\qquad N_{16}=64,\qquad N_{32}=256.
\]

模型宽度为 \(D=1152\)。粗略把一个 DiT block 的 token-dependent FLOPs 写成：

\[
12ND^2+2N^2D.
\]

在 \(N=256\) 时，真正的 QK+AV 二次项占比约为：

\[
\frac{2N^2D}{12ND^2+2N^2D}
=
\frac{2N}{12D+2N}
\approx 3.57\%.
\]

所以即使把 attention matrix 部分完全消灭，整个 block 也不会快很多。相反，score cache hit 可以少算 Q 与 K 两个 projection，其理论占比约为：

\[
\frac{2ND^2}{12ND^2+2N^2D}
=
\frac{2D}{12D+2N}
\approx16.1\%.
\]

这解释了为什么实测中“score reuse + V-only”比单纯 block sparsity 更重要。

BF16、batch 1 的 kernel microbenchmark 也显示：

| Tokens | Density | sparse vs SDPA | cached AV vs SDPA |
|---:|---:|---:|---:|
| 256 | 25% | 1.12× | 1.32× |
| 256 | 50% | 0.86× | 1.05× |
| 1024 | 25% | 1.10× | 1.72× |
| 1024 | 50% | 0.63× | 1.10× |

目前 kernel 在高 density 时还不如高度优化的 dense SDPA；这不是理论上的 sparse FLOPs 不省，而是当前 tile、循环和 occupancy 还没有达到 FlashAttention/VSA kernel 的成熟度。

## 8. 能不能说超过 FastVideo

### 现在不能直接这样说

FastVideo/VSA 的约 1.7× 是 23K-token 视频模型上的端到端结果；本实验是 256-token ImageNet DiT 上的 denoiser-only 结果。任务、序列长度、硬件和计时范围都不一样。

当前可诚实写：

- RDD 原有 H20 microbenchmark 中，Unified XL 的多分辨率 schedule 相对 full-resolution DiT 为 1.896×；
- 本方法在另一台 Blackwell GPU 上，对 dense RDD denoiser 再提供约 1.14×；
- 两者若近似相乘是约 2.16×，但这只是跨协议估计，**不是测量值，不能放主表冒充结果**；
- 严格无 post-8 dense 的估计组合约为 1.896×1.084≈2.06×，同样只是估计。

### 为什么高分辨率图像有机会真正超过

1024×1024 图像若使用 8× VAE、patch size 2，最终 token 数约为 4096；这时二次 attention 占比显著上升。RDD 又天然拥有 16→32→64→128 latent 的 coarse stages，可以把 route 连续传到 4096-token fine stage，而 VSA/DraftAttention 每个 fine step 仍需要构造当前 step 的 coarse surrogate。

真正有希望超过 VSA 的版本应当：

1. 在 512/1024 图像或视频长序列上训练/微调；
2. 使用 64-token hardware tile，而不是当前为 ImageNet-256 设计的 4/16-token小块；
3. 把 route Top-K、index packing 和 sparse attention 做成 fused kernel；
4. 使用训练期 route consistency / sparse distillation，让 25% 甚至 12.5% density 仍保质量；
5. 对 refresh interval 使用误差触发，而不是固定 K；
6. 最终在同一 H100/Blackwell、同一 batch、同一精度下与 FA3、VSA 和 DraftAttention 做 matched-quality latency。

## 9. 最终推荐方案

### 9.1 现在立刻可用：质量优先

采用 `triton_dense_score_adaptive_5_3_3`：

- early layers 每 5 calls refresh；
- middle / late layers 每 3 calls refresh；
- cache hit 只算 V，并用旧 \(A\) 乘当前 \(V\)；
- 当前四图保真度 0.99642；
- denoiser 加速约 1.136×。

它不是严格无 dense，因为 refresh calls 会计算 dense QK；但这是当前最好用的 training-free 点。

### 9.2 严格满足“只在 8×8 dense”

采用 `triton_no_dense_balanced_score3_all`：

- 8×8 dense；
- 16×16 使用 2×2 micro-block，route density 93.75%；
- 32×32 使用 4×4 block，route density 87.5%；
- 两个 stages 都每 3 calls sparse refresh；
- 其余 calls 只执行 V-only + cached sparse AV；
- post-8 dense refresh 次数严格为 0；
- denoiser 加速约 1.084×，image cosine 0.99312。

### 9.3 不推荐

- 不要把 8×8 probability 直接当 16×16 probability；
- 不要把 16×16 probability 直接当 32×32 probability；
- 不要全层 K=10；
- 不要只看 route cosine 就认为完整 score 可以复用；
- 不要在 256 tokens 上宣称 sparse attention 会有视频长序列那样的巨大收益。

## 10. 下一轮论文级实验

按优先级：

1. **50K ImageNet evaluation**：dense、adaptive 5/3/3、strict balanced，统一算 FID/sFID/IS/precision/recall；
2. **完整 latency**：把 noise-filling、VAE decode 和保存前 tensor materialization 纳入；batch 1/4/16，BF16，报告 p50/p95 和峰值显存；
3. **高分辨率验证**：至少 512，最好 1024；这是检验 sparse kernel 真正价值的关键；
4. **matched-quality comparison**：和 dense RDD、FA3、DraftAttention-style current-step pooling、VSA-style coarse stage 对比；
5. **训练期 sparse adaptation**：从 dense RDD checkpoint 微调，loss 包括 denoising、dense-teacher output distillation、route mass coverage；
6. **自适应 refresh**：当 route overlap、input delta 或 cached-output error proxy 超阈值才 refresh；
7. **kernel v2**：64-token tile、fused Top-K/index packing、persistent scheduling、BF16/FP8、forward/backward kernels。

论文主表至少要同时报告：FID、precision、recall、真实 NFE、denoiser latency、完整 pipeline latency、peak memory、route density 和 refresh rate。

## 11. 复现命令

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
CUDA_VISIBLE_DEVICES=1 \
bash 2d/attention_analysis/run_rdd_hierarchical_kernel_final.sh \
2>&1 | tee eval_results/rdd_hierarchical_kernel_final.log
```

可以通过环境变量覆盖路径和重复次数：

```bash
CUDA_VISIBLE_DEVICES=1 \
TIMING_REPEATS=5 \
OUTPUT_DIR=/scr/yuheng/DevOpsBench/multiscale_diffusion/eval_results/rdd_hier_attention_repeat5 \
bash 2d/attention_analysis/run_rdd_hierarchical_kernel_final.sh
```

## 12. 最终一句话

最有价值的结论不是“coarse attention 可以原样复用”，而是：

> **RDD 的 coarse stage 足以提供下一 stage 的稀疏搜索空间；fine stage 只需要在这个空间里周期性更新数值 attention，其他 timestep 复用 score 并更新 V。**

这个表述符合实验，也比“低分辨率 attention 直接替代高分辨率 attention”更稳、更容易抵御 novelty 和 correctness 质疑。
