# RDD Unified-L stage 分配：FID 曲线与原因诊断

## 结论先行

在完全相同的评估口径下，三条曲线从 50K 到 250K 的排序始终是：

1. `300/300/400` 最好；
2. `100/300/600` 第二；
3. `600/300/100` 最差。

这不是“100/300/600 的 8×8 没训练够”可以解释的。旧的单独续训 8×8 实验已经只追回约 5.6%–8.7% 的 FID 差距；而且 Unified 训练代码每个 optimizer step 都依次训练 32、16、8 三个 stage 一次，stage 长度本身并不改变某个分辨率获得的 batch 数量。

当前最有证据的解释是两个不同的失败机制：

- `100/300/600`：32×32 的 drift 在高 SNR 区间内变化过快，且过早把仍然清楚可恢复的 16→32 细节删除掉。你的“模型被强迫做很重的超分辨率/残差恢复，而不是普通去噪”判断基本正确。
- `600/300/100`：8×8 coarsest stage 只有最后 100 个 global timestep；RDD 的 stage-local diffusion 在该区间末端仍保留很强的 clean signal，但采样代码却从纯高斯噪声启动，形成明显的 train–sample terminal mismatch。这比“8×8 训练次数少”更可能是它最差的主因。

## 1. 公平比较口径

- 模型：Unified DiT-L/2；
- 数据：原生 ImageNet；
- checkpoint：50K、100K、150K、200K、250K；
- sampler：DDPM；
- 总 NFE：250；
- sampling order `[8,16,32]`：`80/80/90`；
- CFG：1.0；
- stage transition：noise filling；
- 每点：50K samples；
- reference：ADM ImageNet-256。

图见：[unified_l_stage_allocation_fid.svg](../../paper/figures/unified_l_stage_allocation_fid.svg)。原始数值见：[metrics_summary.tsv](../../eval_results/unified_l_boundary_analysis/metrics_summary.tsv)。

### 250K 结果

| allocation | FID ↓ | sFID ↓ | Precision ↑ | Recall ↑ | 相对 300/300/400 的 FID 差距 |
|---|---:|---:|---:|---:|---:|
| 300/300/400 | **11.222** | **4.535** | **0.6948** | 0.6109 | 0 |
| 100/300/600 | 15.777 | 4.954 | 0.6524 | **0.6155** | +4.555（+40.6%） |
| 600/300/100 | 20.125 | 4.671 | 0.6375 | **0.6155** | +8.903（+79.3%） |

50K–250K 区间的梯形积分平均 FID 分别为 20.802、26.434、32.792。不是某一个 checkpoint 的随机异常，整个训练过程中排序都没有翻转。

## 2. 一个容易误解、但非常关键的代码事实

Unified 训练的每个 optimizer step 都执行：

```text
for stage in [32×32, 16×16, 8×8]:
    从这个 stage 的 local timestep 中 uniform 采一个 t
    计算该 stage loss
把三个 loss 相加，再 backward 一次
```

因此把 stage 长度从 100 改成 600，改变的不是“这个分辨率训练多少个 batch”，而是：

1. 它负责哪一段噪声/SNR 区间；
2. local timestep 的数量，以及每个 local timestep 被重复看到的次数；
3. drift 的斜率 `1/L`；
4. epsilon target 中 drift residual 的强度；
5. 固定 `80/80/90` 采样时，每个 stage 的离散化密度；
6. 三个任务共享同一个 Transformer 时的 loss 尺度和梯度冲突。

在 250K、global batch 256 时，每个 stage 总共都见过约 6400 万个样本；只是长度为 100/300/600 时，每个 local t 平均分别见过约 64 万/21.3 万/10.7 万个样本。

## 3. 为什么 100/300/600 差

### 3.1 实际训练 target

代码中的 RDD forward 为：

\[
x_k=\sqrt{\tilde\alpha_k}x_0+
\sqrt{\tilde\alpha_k}\lambda(k)d+
\sqrt{1-\tilde\alpha_k}\epsilon,
\qquad
\lambda(k)=\frac{k}{L}.
\]

epsilon prediction 实际学习的目标不是纯高斯噪声，而是：

\[
\tilde\epsilon
=
\epsilon
+
g(k)d,
\qquad
g(k)=\lambda(k)
\sqrt{\frac{\tilde\alpha_k}{1-\tilde\alpha_k}}.
\]

这里的 `\tilde\alpha_k` 是代码通过 `SpacedDiffusionWithDrift` 为每个 stage 重建出来的 conditional/local cumulative alpha，不是简单把 global `alpha_bar(t)` 原样拿来。

使用与旧 teacher-forced 实验相同的 ImageNet validation SD-VAE latent，16→32 residual 的均方能量为约 0.24137。于是 drift target 相对单位高斯 target 的平均能量为：

\[
\mathbb E_k[g(k)^2]\,\mathbb E[d^2].
\]

| schedule 的 32×32 stage | drift 斜率 `1/L` | `g` RMS | drift target / Gaussian target 能量 |
|---|---:|---:|---:|
| 100/300/600 | 0.0100 | 2.683 | **1.738×** |
| 300/300/400 | 0.00333 | 0.907 | 0.199× |
| 600/300/100 | 0.00167 | 0.382 | 0.035× |

也就是说，在 `100/300/600` 的 32×32 stage 中，仅 drift residual 的平均 target 能量就已经是高斯噪声 target 的 1.74 倍。它并非完全等于传统超分辨率训练，但从优化信号看，模型确实被强烈推动去预测/抵消 `UP(DOWN(x0))-x0` 这一高频残差。

### 3.2 handoff 发生得太早

在 32×32 stage 末端：

| allocation | local terminal SNR | 16→32 residual band terminal SNR |
|---|---:|---:|
| 100/300/600 | 8.710 | **2.102** |
| 300/300/400 | 0.657 | 0.159 |
| 600/300/100 | 0.0266 | 0.0064 |

`100/300/600` 在 32×32→16×16 切换时，高分辨率新增细节的 band SNR 仍大于 2：这些细节还清晰可见，却已经被 drift 快速删除并交给低分辨率表示。这就是“分辨率变化压过普通噪声过程”的更精确数学表达。

### 3.3 训练 loss 也支持 target 尺度异常，但不能单独当作质量指标

同样取 240K–250K、每 100 step 记录一次的 101 个点，总 loss 均值为：

| allocation | mean total loss |
|---|---:|
| 100/300/600 | **1.8364** |
| 300/300/400 | 0.7829 |
| 600/300/100 | 0.5503 |

`100/300/600` 的总 loss 是 baseline 的约 2.35 倍，与 drift target 能量大幅增加方向一致。但不同 schedule 的 target 分布不同，所以绝对 loss 不能直接比较模型“学得好不好”。尤其 `600/300/100` loss 最低但 FID 最差，正好说明它的问题更像生成初始条件/任务定义错配，而不是普通 underfitting。

## 4. 为什么 600/300/100 反而最差

coarsest 8×8 stage 没有 drift。关键在于每个 stage 的 beta schedule 是从 global schedule 截取后重新条件化的。8×8 stage 末端的 local `alpha_bar` 为：

| allocation 的 8×8 区间 | terminal local alpha_bar | terminal SNR |
|---|---:|---:|
| 100/300/600 `[400,1000)` | 0.000207 | 0.000207 |
| 300/300/400 `[600,1000)` | 0.001559 | 0.001562 |
| 600/300/100 `[900,1000)` | **0.146648** | **0.171849** |

采样代码对第一个 coarsest stage 明确执行 `z = torch.randn(...)`，即从纯高斯开始。但 `600/300/100` 的 8×8 模型训练时最噪的样本仍包含：

\[
\sqrt{0.14665}\,x_0 \approx 0.383\,x_0
\]

的 clean-signal 系数。它从未真正学过“从纯噪声开始”的条件，却在生成时被要求这么做。相比之下，baseline 的 clean 系数只有约 `sqrt(0.001559)=0.0395`，已经接近纯噪声。

这个 terminal mismatch 是目前解释 `600/300/100` 最差的第一嫌疑，而且它与 8×8 训练是否充分无关：100-step stage 每个 local timestep 反而被重复训练得最多。

## 5. 指标形态还告诉了我们什么

- 250K 时三个配置的 Recall 都约为 0.61；差异主要在 Precision（0.6948、0.6524、0.6375）。因此主要问题是 fidelity/realism，不是简单 mode collapse。
- `600/300/100` 的 sFID=4.671，已经很接近 baseline 4.535，甚至优于 `100/300/600` 的 4.954，但它的 FID 仍是 20.125。这说明局部空间/纹理统计并没有同等程度地坏掉，更像 coarsest 初始语义、全局形状或条件分布错位，之后高分辨率 stage 只能把错误结构画得更精细。
- `100/300/600` 的 Recall 不差而 Precision 明显下降，与“过急 drift/过早 handoff 造成失真或不自然细节”一致。

## 6. 仍可能存在的次要原因

### 固定 80/80/90 并不对所有边界同样合适

同一 NFE 不等于每个 local interval 的离散误差相同。`600/300/100` 的 32×32 stage 覆盖 600 个训练 timestep，却只有 90 次昂贵的 fine-resolution update；`100/300/600` 的 8×8 stage 覆盖 600 个 timestep，却只有 80 次 coarse update。它可能放大差距，但无法解释 `600/300/100` coarsest terminal SNR 高达 0.172 的根本错配。

### Unified 共享参数的梯度冲突

三个 stage loss 无权重相加。`100/300/600` 的 fine-stage target 尺度很大，可能主导共享 Transformer 的梯度；`600/300/100` 的各 stage 总 loss 很低，却可能把参数优化到容易拟合的局部任务，而生成起点仍错误。需要 per-stage gradient norm/cosine 才能确认。

### 每个 timestep 的覆盖不同

长度 600 的 stage 每个 local t 被看到的次数只有长度 100 的六分之一。这是 per-t coverage，不是 stage 总训练量。它可能影响 600-step fine/coarse interval 的拟合，但不能解释为什么 `600/300/100` 的总 loss 反而最低。

## 7. 最小、最有判别力的下一组实验

不要再无目的地扫很多 boundary。先做下面四个因果实验：

1. **100 schedule 的 fine-drift-scale pilot**：保持 `100/300/600` 不变，只把 32×32 stage 的 drift scale 设为 `1/3`，使 `dλ/dt` 与 300-step baseline 相同。训练到 100K，跑 10K paired FID。若明显追回差距，说明 drift stiffness/target domination 是主因。
2. **100 schedule 的 fine-no-drift pilot**：只关闭 32×32 drift，边界不变。它与实验 1 一起区分“过急 drift”与“即使无 drift，100-step handoff 本身也太早”。
3. **600 schedule 的 terminal-fix pilot**：保持 `600/300/100` 的分辨率边界和 time embedding，但把 8×8 stage 的 local noise schedule 重新归一化到 terminal `alpha_bar≈10^-3`，训练到 100K。若 FID 大幅改善，就直接证实 pure-noise terminal mismatch。
4. **sampling allocation 10K 诊断**：不重训，分别比较 baseline `80/80/90` 与按 local interval/实测 solver error 重分配的 NFE；总 FLOPs 用各分辨率实测单步延迟对齐。若排名基本不变，训练/forward 定义是主因；若明显改善，固定采样分配也在放大差距。

优先级是 3 → 1 → 2 → 4。前两个训练 pilot 到 100K 足以看收敛方向，不必直接烧到 500K。

## 8. 可复现文件

- 曲线图：[unified_l_stage_allocation_fid.svg](../../paper/figures/unified_l_stage_allocation_fid.svg)
- 曲线原始数据：[metrics_summary.tsv](../../eval_results/unified_l_boundary_analysis/metrics_summary.tsv)
- 同 step loss 统计：[training_loss_same_step.tsv](../../eval_results/unified_l_boundary_analysis/training_loss_same_step.tsv)
- 完整机械量和曲线汇总：[unified_stage_allocation_analysis.json](../../eval_results/unified_l_boundary_analysis/unified_stage_allocation_analysis.json)
- 可复现分析脚本：[analyze_unified_stage_schedule_mechanics.py](../../2d/diagnostics/analyze_unified_stage_schedule_mechanics.py)

运行：

```bash
python 2d/diagnostics/analyze_unified_stage_schedule_mechanics.py
```
