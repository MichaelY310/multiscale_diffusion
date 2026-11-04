# RDD 2D 跨分辨率 attention routing / coarse summary 实测

## 结论先行

这次实验支持下面这个**受限制的结论**：

> RDD 已经完成的低分辨率 stage，确实能为相邻 fine stage 提供有信息量的空间候选路由；在 Unified XL 的 `16×16 → 32×32` 边界，上一 stage 的 pooled K/V 还可以作为被裁掉 fine tokens 的 coarse summary。

但实验**不支持**把整块 hidden state 直接搬到 fine stage，也不支持在所有 stage、所有层使用同一种稀疏策略。最有希望的第一版应当是：

1. 只加速 Unified XL 的 finest `32×32` stage；
2. 缓存 `16×16` stage 边界的 head-specific route 和 pooled K/V；
3. fine query 先只访问 route 指定的 fine blocks；
4. 仅在实测有收益的层用 coarse K/V 表示未展开区域；
5. 保留局部邻域、少量全局安全块和可回退 dense 的门控；
6. 暂时不要把“hidden state reuse”作为主方法。

这是一个机制验证，不是最终加速结果。当前 ImageNet-256 latent 的 fine attention 只有 256 个 tokens，attention matmul 在 DiT-XL block 总 FLOPs 中占比很小；真正的硬件收益需要更长序列，尤其是 video。

## 1. 实验到底比较了什么

### 1.1 模型

两个模型都使用 400K EMA checkpoint，训练边界都是 finest/middle/coarsest = `300/300/400`：

- Unified：`DiTMultiRes-XL/2`，8、16、32 三个分辨率共享同一个模型、同一组 attention heads 和同一 hidden width；
- Multistage：8=`DiT-B/2`、16=`DiT-L/2`、32=`DiT-XL/2`，三个 stage 是独立模型。

### 1.2 公平配对

使用 4 张 ImageNet 原始图片，经 SD VAE posterior mode 得到确定性的 `32×32` latent。每张图使用两组 noise seed。为了比较同一次 coarse-to-fine 生成过程，三个尺度的噪声来自同一个 `32×32` Gaussian noise，再用 average pooling 并恢复单位方差：

```text
epsilon_16 = AvgPool2(epsilon_32) * 2
epsilon_8  = AvgPool4(epsilon_32) * 4
```

比较 RDD stage 边界处相邻的 denoiser calls：

```text
8×8  stage 的 local t=0     → 16×16 stage 的 local t=299
16×16 stage 的 local t=0    → 32×32 stage 的 local t=299
```

后者分别映射到全局 timestep 600/599 和 300/299。也就是说，没有拿相隔很远的噪声状态硬做相关性。

DiT patch size 是 2，所以 latent 8/16/32 对应的 token grid 是 4×4、8×8、16×16。比较 attention 时，把 fine attention 的每个 2×2 token cell 聚合回 coarse grid。

### 1.3 五个归一化深度

每个模型取 0%、25%、50%、75%、100% 深度附近的五层：

- Unified XL：0、7、14、20、27；
- Multistage 的不同深度模型按相对深度配对。

Unified 的 head 编号在两个分辨率间有真实的一一对应。Multistage 的模型独立训练，即使 head 数碰巧相同，head 编号也没有共同语义，所以 Multistage 只比较 head-averaged 空间拓扑，不做虚假的 head-to-head 对齐。

## 2. 指标用人话怎么理解

### predicted-x0 low-pass cosine

把 fine stage 预测的 x0 average-pool 到 coarse 大小，再和 coarse stage 预测的 x0 比较。越接近 1，说明两边对同一张图的低频内容判断越一致。

### route cosine / JS

把 attention 都变成 coarse-cell 到 coarse-cell 的概率图，再比较两张概率图。cosine 越高、JS 越低越相似。但接近均匀的 attention 天生容易得到很高 cosine，所以不能只看这个指标。

### 20% mass coverage

上一 stage 为每个 query 选择 attention 最大的 20% coarse key cells，然后测这些 cells 在 fine stage 的真实 attention 中一共占多少概率质量。

- 随机/均匀选择的基线约为 20%；
- 如果得到 60%，意思是只保留五分之一的空间，却保留了 fine attention 六成的注意力质量。

这是判断 route 是否真正有选择性的核心指标。

### sparse attention output cosine

不是只比较 attention 图，而是真正把 fine attention 限制到 coarse route 允许的 token blocks，然后比较 `softmax(QK)V` 输出与 dense attention 输出。越接近 1 越好。

### parent summary

保留 route 内的 fine K/V；route 外不直接丢弃，而用上一 stage 每个 coarse cell 的 pooled K/V 代表这一整块。这是在测试“coarse summary”是否真的能补回被裁掉的信息。

### hidden / K representation

- linear CKA：两个表示是否包含相似的整体关系结构；
- aligned-token cosine：同一空间位置的向量方向是否真的相近；
- ridge adapter test cosine：只用一半图片拟合线性 adapter，在未见的另一半图片上测试。这个小样本结果只能作风险提示，不能作为训练 adapter 的最终结论。

## 3. 汇总结果

所有数值是 4 张图片、2 个对齐 noise seeds、5 个深度的平均值。

| 模型 | 边界 | pred-x0 low-pass cosine | route cosine | 20% route 覆盖的 fine mass | 20% sparse output cosine | 20% parent summary output cosine |
|---|---:|---:|---:|---:|---:|---:|
| Unified XL | 8→16 | 0.697 | 0.783 | 0.458 | 0.886 | 0.820 |
| Unified XL | 16→32 | **0.924** | 0.796 | **0.564** | **0.885** | **0.931** |
| Multistage B→L | 8→16 | 0.699 | 0.900 | 0.317 | 0.689 | 不可直接复用 |
| Multistage L→XL | 16→32 | **0.922** | 0.899 | 0.415 | 0.743 | 不可直接复用 |

这里 Multistage 的 route cosine 高于 Unified 并不代表它更适合稀疏化。它是 head-average 后的数值，而且部分层的 attention 接近均匀；更有区分力的 mass coverage 和真实 sparse output 都明显弱于 Unified。

## 4. Unified `16→32` 是最重要的正结果

### 4.1 逐层 route 与稀疏输出

| XL 层 | 20% route mass coverage | oracle 20% mass | sparse output cosine @20% | @30% | @40% |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.286 | 0.339 | 0.934 | 0.958 | 0.975 |
| 7 | 0.683 | 0.777 | 0.809 | 0.870 | 0.908 |
| 14 | 0.579 | 0.676 | 0.886 | 0.930 | 0.955 |
| 20 | 0.473 | 0.565 | 0.806 | 0.866 | 0.905 |
| 27 | **0.799** | 0.878 | **0.990** | 0.993 | 0.995 |

两个直接观察：

1. 低分辨率 route 远强于 20% 随机密度基线，尤其 layer 7、14、27；
2. 固定 20% 对 layer 7/20 太激进，但 40% 时五个探测层的局部 attention 输出余弦都至少为 0.905。

### 4.2 coarse summary 不是每层都该使用

固定 20% exact fine blocks 时：

| XL 层 | 直接丢 tail | 加上一 stage coarse K/V summary | 用当前 fine K/V 做 summary（参考上界） |
|---:|---:|---:|---:|
| 0 | 0.934 | **0.984** | 0.998 |
| 7 | 0.809 | **0.935** | 0.996 |
| 14 | 0.886 | **0.951** | 0.995 |
| 20 | 0.806 | **0.861** | 0.991 |
| 27 | **0.990** | 0.923 | 0.987 |

上一 stage summary 在 0/7/14/20 层确实补回了信息，但在最后一层反而破坏已经很好的 exact sparse output。因此 summary 必须做成 layer-specific，而不是统一打开。

一个由当前数据直接导出的第一版策略是：

```text
layer 0/7/14: 20% fine exact blocks + previous-stage summary
layer 20:     40% fine exact blocks, 不用 summary
layer 27:     20% fine exact blocks, 不用 summary
```

这五个探测层的局部 attention 输出余弦约为：

```text
0.984, 0.935, 0.951, 0.905, 0.990，平均约 0.953
```

按 token interaction 粗算，该混合策略约使用 dense attention core 的 36%，即 QK/AV interaction 减少约 64%。这还不是端到端模型 FLOPs 或真实 wall-clock 加速。

### 4.3 应复用 K/route，不应直接复用 hidden

Unified `16→32` 中：

- 中间层 hidden CKA 约 0.87–0.88，说明整体结构相似；
- 但 aligned-token hidden cosine 只有约 0.53–0.62；
- 简单 adapter 在未见图片上的 hidden cosine 多数只有约 0.20；
- pooled K 的 aligned-token cosine 为 0.80–0.87；
- pooled K 的 CKA 为 0.85–0.93；
- 最后一层 pooled K 的小样本 adapter test cosine 约 0.93。

因此“hidden 很像”不能被解读成“可以直接跳过 fine block”。当前证据更具体地支持：

```text
共享空间候选关系（route）
          +
共享被压缩的 K/V 记忆（coarse summary）
```

## 5. 为什么 `8→16` 不应该作为第一版加速目标

Unified `8→16` 的 predicted-x0 low-pass cosine 只有 0.697，而 `16→32` 是 0.924。20% route 虽然仍覆盖 45.8% fine mass，但 coarse summary 只在个别层有帮助，并在 layer 14/20/27 明显变差。

更现实的一点是：16×16 latent 在 patch size 2 后只有 64 个 tokens，其 dense attention 本来就非常便宜。为了这一段引入路由、gather 和 kernel launch，容易得不偿失。

所以第一版应当：

- 8 stage：dense；
- 16 stage：dense，同时产出供下一 stage 使用的 cache；
- 32 stage：使用跨 stage routing / summary。

## 6. Multistage 能用什么，不能用什么

Multistage 的 `16→32` predicted-x0 low-pass cosine 也达到 0.922，证明图像的 coarse semantic/layout 在独立 L 和 XL 模型间仍然高度一致；20% route 的 mass coverage 0.415 也明显高于随机 0.2。

但：

- L 和 XL hidden width 不同；
- 模型独立训练，head 没有一一对应语义；
- K/V 不在共同特征空间；
- head-average route 做直接稀疏 attention 时，平均 output cosine 只有 0.743。

因此 Multistage 可以把 low-resolution route 当成**无 head 的空间 proposal**，然后由 fine XL 自己的廉价 router 重新打分；不能直接把 L 的 K/V 当 XL 的 summary，除非专门训练 head/width adapter。为了把 RDD 主方法保持清楚，第一篇硬件工作不建议先从 Multistage adapter 开始。

## 7. 精化后的方法：RDD Cross-Stage Route Cache（暂名）

### 7.1 16 stage 结束时缓存什么

对选定层缓存：

```text
top-k coarse block indices: [batch, head, coarse_query, k]
pooled K/V summary:          [batch, head, coarse_cell, head_dim]
```

不缓存完整 hidden state。route cache 很小；16 stage 的 token grid 只有 8×8。

### 7.2 32 stage 如何用

每个 2×2 fine query group 对应一个 coarse query cell：

1. 从上一 stage route 取 top-k candidate key cells；
2. 展开这些 cells 内的 fine tokens，计算精确 QK/AV；
3. 强制加入当前 query 的局部邻域和少量 global landmark blocks，避免 route 漏掉新出现的高频关系；
4. 在 layer 0/7/14 等实测有效的层，用 cached coarse K/V 表示未展开区域；
5. 在 layer 20/27 使用纯 exact sparse 或 dense fallback；
6. 根据 entropy、top-k margin 或 coarse/fine route disagreement 自动提高密度。

### 7.3 它和普通 attention reuse 的区别

普通 temporal attention reuse 通常利用“相邻 denoising timesteps 的 attention 很像”。这里利用的是 RDD 特有的结构：

```text
上一分辨率已经完整求解出的空间依赖
              ↓ 跨分辨率映射
下一分辨率 fine attention 的候选结构和长程 coarse memory
```

它不是简单复用相邻 timestep 的同尺寸 attention，也不是把上一层 hidden state 原样拷贝过来。之后可以再与 temporal reuse 正交组合：跨 stage cache 给初始 route，同一 fine stage 内再做 refresh/reuse。

## 8. 算力上限：为什么 2D 只能证明机制

DiT-XL/2 在 `32×32` latent 上有 N=16×16=256 tokens、hidden width D=1152。忽略常数细节，一个 block 中：

```text
QKV + output projection + MLP ≈ 12 N D²
QK + AV attention core         ≈ 2 N² D
```

attention core 占比约为：

```text
2N / (12D + 2N) = 512 / (13824 + 512) ≈ 3.57%
```

即使把 attention core 减少 80%，单个 block 的理论总 FLOPs 也只减少约 2.9%，真实 kernel 的索引/gather 开销还可能吃掉收益。因此：

- 这次 2D 结果回答“跨分辨率关系是否存在”；答案是 Unified `16→32` 存在；
- 不能用它宣称 ImageNet-256 已有明显 wall-clock speedup；
- 序列越长，attention 占比越大，这个方法才越有硬件价值；video 正是更合适的最终落点。

## 9. 下一步实验，按优先级排序

### P0：dense kernel 上做 correctness oracle

先不写 Triton。给 Unified XL attention 加 mask/gather reference 实现，跑完整 denoiser forward：

- 测 epsilon / predicted-x0 的相对误差；
- 跑完整 50K samples，比较 FID、sFID、Precision、Recall；
- density = 20/30/40%；
- direct sparse、summary hybrid、随机 route、local-window、fine-only cheap router 五组对照。

只有局部 attention output cosine 不足以保证最终 FID。

### P0：测 fine stage 内的时间持久性

当前只证明了 stage boundary 的第一步。必须测 cached 16-stage route 与 32-stage 各 timestep 的关系：

```text
t_local = 299, 250, 200, 150, 100, 50, 0
```

如果相关性快速下降，就只在 fine stage 早期使用跨 stage cache，之后每 R 步用 fine route refresh；如果长期稳定，才有理由整段复用。

### P1：训练自由的 adaptive density

用 coarse route entropy、top-k margin、fine proxy disagreement 选择 20/40/100% 密度。目标是困难 query 回退 dense，简单 query 才稀疏。

### P1：真实 block-sparse kernel

将相邻 coarse cells 合成适合 GPU tile 的规则 block，并实现：

- index compaction / block table；
- fused QK + masked softmax + AV；
- optional coarse-summary tail；
- 与 FlashAttention dense baseline 比 wall time、HBM traffic、occupancy。

在 2D 的 256-token 场景，kernel 主要用于正确性和编程验证；最终性能图应放到长序列 video 或更高分辨率生成上。

## 10. 这次结果不能证明什么

- 样本只有 4 张图、2 个 noise seeds；趋势稳定，但不是统计充分的论文结果；
- 使用的是 teacher-forced、对齐噪声的 stage-boundary latent，不是完整自由采样轨迹；
- sparse output 指标只检查单层 attention core，没有让误差经过后续 block 累积；
- 局部 output/summary 指标为控制显存，只计算前两张图和前四个 heads；route/representation 指标使用全部小批量；
- 没有真实 block-sparse kernel，因此没有 wall-clock speedup；
- Multistage 的 head-average route 只能说明空间 proposal，不说明 head-specific K/V 可复用。

## 11. 复现文件

- 原始 JSON：`eval_results/rdd_2d_cross_stage/rdd_2d_cross_stage_probe.json`
- 主探针：`2d/attention_analysis/probe_rdd_2d_cross_stage.py`
- 单机运行：`2d/attention_analysis/run_rdd_2d_cross_stage_probe.sh`
- tmux 启动：`2d/attention_analysis/launch_rdd_2d_cross_stage_probe.sh`
- Unified XL checkpoint 下载：`2d/attention_analysis/download_rdd_unified_xl_400k.sh`

标准运行命令：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash 2d/attention_analysis/launch_rdd_2d_cross_stage_probe.sh
tail -f eval_results/rdd_2d_cross_stage/logs/probe.log
```

