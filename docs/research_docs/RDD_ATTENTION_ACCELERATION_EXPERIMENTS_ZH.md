# RDD 2D Attention 加速实验：方法、结果与失败记录

更新时间：2026-08-20  
范围：ImageNet-256 latent RDD，主要模型为 Unified XL/2 400K；采样为 DDPM 80/80/90、CFG 1.5。本文只汇总 2D attention 实验。

## 先回答：之前“效果还可以”的版本到底有多 sparse？

代码里的旧名字 `balanced` 没有标准技术含义，只是当时给“保真度优先的保守配置”起的内部名。以后统一叫：

> **保守级联：16 阶段 Top-15/16，32 阶段 Top-14/16，每 3 步刷新一次当前阶段的 attention probability。**

它不是模型自动决定保留比例，而是人工设置比例；低分辨率 attention 只决定“具体保留哪几个 block”。每个 query 一共有 16 个候选 key blocks：

| 阶段 | 人工 Top-K | 强制 self block 后实际保留 | 实际删掉 |
|---|---:|---:|---:|
| 16×16 latent | Top-15/16 | 15 或 16 个 | 0%–6.25% |
| 32×32 latent | Top-14/16 | 14 或 15 个 | 6.25%–12.5% |

所以它确实**不太 sparse**。它之所以图像接近 dense，主要就是因为只删了很少的 block。

另一个较早且效果很好的 `route40_early30` 是：32 阶段前 30/90 次调用保留约 40.8%，后 60 次恢复 dense。按整段 32 阶段平均，仍保留约 80.3% 的 interactions，也只删了约 19.7%。它同样不是“整段都只保留 40%”。

## 名字翻译

| 旧实验名 | 本文直白名称 | 实际含义 |
|---|---|---|
| `balanced` | 保守级联 | 16 保留 15/16，32 保留 14/16；self block 兜底 |
| `adaptive_5_3_3` | 按网络深度使用 5/3/3 步刷新 | 前 15% layers 每 5 步刷新，中间 70% 和最后 15% 每 3 步刷新；**不是根据输入动态判断** |
| `score3` | 每 3 步刷新 probability | refresh 时算 Q/K、QK、softmax；其余两步复用旧 probability，只重新算当前 V |
| `cross5` | 开头 5 次直接使用上一阶段 probability | 跨分辨率直接复用数值 probability；实验表明质量较差 |
| `cache hit` | 本步成功复用旧 probability | 计算 `A_old @ V_current`，跳过当前 Q/K、QK 和 softmax |
| `cache miss/refresh` | 本步必须更新 probability | 重新计算当前 Q/K、QK、softmax；可能是 sparse refresh，不等于一定算 dense |

---

## 1. 低分辨率 attention 引导高分辨率 attention route

### 方法

1. 8 阶段结束时，把最后一次 8 阶段 attention 按空间 block 汇总成 route。
2. 对每个 query block，按 route 的数值取 Top-K key blocks；这只决定 16 阶段“去哪里算 attention”。
3. 16 阶段结束时，用其最后一次 attention 产生 32 阶段 route。
4. 现有严格 kernel 版本中，一旦进入新阶段，block 名单在该阶段内固定；不会随着 timestep 改变。
5. 新阶段仍在选中的 block 内计算自己的 QK/softmax；并没有把上一阶段完整 attention probability 原样当成答案。

一句话：**上一阶段告诉下一阶段“重点看哪里”，下一阶段自己计算“具体看多少”。**

### 数值结果

跨分辨率相关性并不算高到可以直接复制完整 probability，但足以当候选 route：

| 模型 | 方向 | route cosine（大致范围） | Top-40% 重合率 | 直接复用 coarse probability 后的输出 cosine |
|---|---|---:|---:|---:|
| Unified XL | 8→16 | 0.764–0.803 | 0.615–0.649 | 0.745–0.890 |
| Unified XL | 16→32 | 0.762–0.795 | 0.690–0.702 | 0.760–0.845 |
| Multistage | 8→16 | 0.834–0.876 | 0.527–0.610 | 0.720–0.832 |
| Multistage | 16→32 | 约 0.73–0.82 | 约 0.63–0.65 | 约 0.75–0.82 |

完整采样的 training-free correctness oracle：

| 模型与设置 | 真正被 route 限制的调用 | 实际保留 | image cosine | PSNR | 结论 |
|---|---:|---:|---:|---:|---|
| Unified，前 30/90 次 | 30 | 40.80% | **0.9970** | **33.19 dB** | 很接近 dense，但全段平均并不很 sparse |
| Unified，全部 90 次 | 90 | 40.80% | 0.9884 | 27.02 dB | route 用太久会漂移 |
| Multistage，前 30/90 次 | 30 | 40.63% | 0.9723 | 23.58 dB | 独立模型的 heads 不对齐，明显更差 |
| Multistage，全部 90 次 | 90 | 40.63% | 0.9543 | 21.14 dB | 失败，不宜直接迁移 route |

这些 oracle 先构造 dense QK 再遮罩，因此它们只验证质量，运行时间不能代表加速。

### 图片

![Unified XL 跨阶段 route 对比](../../eval_results/rdd_cross_stage_sparse_samples/unified/unified_xl_comparison.png)

![Multistage 跨阶段 route 对比](../../eval_results/rdd_cross_stage_sparse_samples/multistage/multistage_blxl_comparison.png)

---

## 2. 同一 resolution stage 内复用 attention probability

### 方法

attention probability 是正方形矩阵：

```text
A_t = softmax(Q_t K_t^T)
O_t = A_t V_t
```

每 K 步刷新一次 `A`。中间步骤不是复用旧的完整输出 `A_old V_old`，而是：

```text
O_t ≈ A_old V_current
```

因此当前图像内容仍通过 `V_current` 更新；被省掉的是当前 Q、K、QK 和 softmax。

### 数值结果

以相隔 10 步为例：

| 模型 | 阶段 | probability cosine | route cosine | Top-40% 重合 | `A_old V_current` cosine | 直接复用旧 `AV` cosine |
|---|---:|---:|---:|---:|---:|---:|
| Unified XL | 16 | 0.9220 | 0.9220 | 0.8159 | 0.9707 | 0.8335 |
| Unified XL | 32 | 0.9322 | 0.9756 | 0.8811 | 0.9802 | 0.9070 |
| Multistage | 16 | 0.9405 | 0.9405 | 0.7566 | 0.9415 | 0.8478 |
| Multistage | 32 | 0.9419 | 0.9819 | 0.9048 | 0.9782 | 0.9101 |

结论：route 比完整 probability 稳定；`A_old V_current` 又明显优于复用旧 `AV`。因此当前实现缓存的是 **probability A**，不是缓存最终输出 AV。

真实 kernel 采样结果：

| 刷新方法 | 加速 | image cosine | PSNR | 结论 |
|---|---:|---:|---:|---|
| 每 3 步刷新，所有 layers | **1.142×** | 0.99526 | 31.29 dB | 当前最稳健的速度点 |
| 网络前层每 5 步、其余每 3 步 | 1.136× | **0.99642** | **31.87 dB** | 质量更优；旧名 `adaptive_5_3_3` |
| 每 5 步刷新，所有 layers | 1.180× | 0.98888 | 27.84 dB | 更快但误差明显 |

### 图片

![同阶段与跨阶段 attention 动态](../../eval_results/rdd_attention_dynamics/rdd_attention_dynamics.png)

---

## 3. 级联 route + 同阶段 probability 复用 + Triton kernel

### 方法

这是目前真正端到端避免 16/32 dense QK 的组合版：

```text
8 阶段：dense，最后一步产生 16 阶段 block 名单
16 阶段：只算名单内的 sparse QK；每 3 步刷新 A，其余两步算 A_old @ V_current
16 阶段最后一步：由 sparse A 产生 32 阶段 block 名单
32 阶段：同样做 sparse refresh + probability reuse
```

保守级联的 block 名单在每个 stage 内固定。16/32 的每个 query 都有 16 个候选 key blocks，实际分别保留 15–16 和 14–15 个，所以它保真但不够 sparse。

### 数值结果

| 方法 | 16/32 保留设置 | 16/32 dense QK | 加速 | image cosine | PSNR |
|---|---|---:|---:|---:|---:|
| Dense | 全保留 | 有 | 1.000× | 1.0000 | ∞ |
| 只做每 3 步 probability 复用 | 100% / 100% | 仅 refresh 时有 | **1.142×** | 0.99526 | 31.29 |
| 保守级联 + 每 3 步复用 | Top-15/16；Top-14/16；含 self 兜底 | **0** | **1.084×** | **0.99312** | 29.83 |
| 更激进级联 + 每 3 步复用 | 87.5%；75%（中层 87.5%） | **0** | 1.088× | 0.98917 | 27.80 |
| 保守级联 + 开头直接跨阶段复用 probability | 同保守级联 | **0** | 1.092× | 0.98572 | 26.49 |

为什么保守级联比“只复用 probability”反而慢一点：当前 N=64/256 很小，少删几个 block 节省的 QK 不足以抵消 gather、索引和 sparse kernel 调度成本。当前主要加速来源其实是 cache hit 时连 Q/K projection 都不计算。

### 图片

![真实 Triton kernel：dense、同阶段复用、保守级联和失败的跨阶段数值复用](../../eval_results/rdd_hierarchical_kernel_final_repeat3_comparison.png)

---

## 4. 失败或不够好的尝试

### 4.1 每层只保留 12.5%

每个 query 只保留 2/16 个 key blocks，而且没有 VSA 的 coarse output branch 和可学习 gate。结果主体、纹理和颜色全部崩坏：

| 设置 | 加速 | image cosine | PSNR |
|---|---:|---:|---:|
| 每步重新算 12.5% sparse attention | 0.995× | 0.86416 | 16.05 dB |
| 12.5% + 每 3 步复用 probability | 1.081× | 0.86577 | 16.16 dB |

![12.5% 强稀疏失败结果](../../eval_results/rdd_hierarchical_kernel_vsa125_cfg15/triton_hierarchical_comparison.png)

### 4.2 把同一份 coarse route 用满整个 fine stage

Unified 的前 30 次效果好，但 90 次全用时从 cosine 0.9970 降到 0.9884；Multistage 更严重。说明 route 有生命周期，不能默认 stage 内永远不变。

### 4.3 直接跨分辨率复用完整 probability

route 只描述“哪些区域重要”，完整 probability 还包含每个 token 的精确权重。后者跨分辨率变化较大；`cross5` 的 cosine 只有 0.98572，明显差于保守级联的 0.99312。

### 4.4 上一阶段 coarse K/V summary

Unified 的实验曾尝试用 coarse summary 补回被删掉的全局信息：前 30 次、约 24.96% exact roots 时 cosine 0.9943；整段使用时降到 0.9806。它证明方向可能有用，但当前 training-free 权重和 layer 策略不够稳，真正类似 VSA 的 `coarse output + learnable gate` 仍未实现，需要 finetune。

### 4.5 初始 75%/50% route 搜索

初始 quality oracle 在 16 保留 75%、32 保留 50% 时，image cosine 约 0.961；邻域扩张、每 5/10 步刷新、短暂跨阶段 probability 复用都只提升到约 0.960–0.964。它说明不经过训练就大幅裁剪 whole-stage attention 很难保持原图。

![初始大范围 route / score 搜索](../../eval_results/rdd_hierarchical_attention_comparison.png)

---

## 5. 讨论过但尚未完成的方法

| 方法 | 当前状态 | 为什么值得做 |
|---|---|---|
| 真正动态 route refresh | 未实现 | 当前 route 在 stage 内固定；可每隔若干步用当前 sparse结果更新 route，或低置信度时临时扩大 K |
| 真正 timestep-adaptive cache | 未实现 | 目前所谓 `adaptive_5_3_3` 只是按 layer 深度设置周期；可根据 hidden/route 改变量决定 hit 或 refresh |
| 8 阶段也复用/稀疏 | 未实现 | 8 阶段 Top-K 会变化，但 token 只有 16，潜在 wall-clock 收益很小；更适合长序列 video |
| VSA 式 coarse output + learnable gate | 未实现，需要 finetune | 被删掉的 keys 仍通过低分辨率 summary 提供全局信息，可能让 12.5%–40% 强稀疏可用 |
| sparse teacher distillation | 未实现 | 用 dense teacher 对齐 sparse attention output、denoiser output 和 route，可恢复强稀疏质量 |
| sparse + linear/global tail | 未实现 | 精确 sparse branch 处理强关系，廉价线性 attention 或 coarse summary 处理长尾全局信息 |
| video 上的级联 kernel | 未实现 | 2D latent token 太少；时空 token 数大时，RDD 原生 coarse route 的硬件价值才可能明显放大 |

## 最终结论

1. **跨分辨率可以传 route，不适合直接传完整 probability。**
2. **同分辨率可以短期复用 probability，但必须继续使用当前 V。**
3. 当前最好看的严格级联并不神奇：16 只删 0%–6.25%，32 只删 6.25%–12.5%。
4. 12.5% 强稀疏失败的关键原因不是 kernel，而是没有 coarse/global 补偿和针对 sparse pattern 的 finetune。
5. 当前最可靠的实测加速是每 3 步复用 probability（1.142×）；真正有论文潜力的下一步是“RDD coarse route + 动态刷新 + coarse/global tail + sparse distillation”，并在 video 长序列上验证。

## 原始文件

- 指标：`eval_results/rdd_hierarchical_kernel_final_repeat3_metrics.json`
- 指标：`eval_results/rdd_hierarchical_kernel_nodense_cfg15_metrics.json`
- 指标：`eval_results/rdd_hierarchical_kernel_pareto_cfg15_metrics.json`
- 指标：`eval_results/rdd_hierarchical_kernel_vsa125_cfg15/metrics.json`
- attention 动态：`eval_results/rdd_attention_dynamics/`
- 跨阶段采样：`eval_results/rdd_cross_stage_sparse_samples/`
- 采样实现：`2d/attention_analysis/sample_rdd_hierarchical_kernel.py`
- Triton kernel：`2d/attention_analysis/rdd_block_sparse_triton.py`
