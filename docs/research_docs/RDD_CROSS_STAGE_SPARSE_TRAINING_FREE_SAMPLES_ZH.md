# RDD 跨 Stage Sparse Attention：Training-free 完整采样结果

## 结论

在现有 400K checkpoint 上，不修改权重即可完整跑通跨 stage sparse attention。

- Unified XL：上一 16-stage route 对 32-stage 的前一部分采样有实际可用性；`route40_early30` 是当前最佳设置。
- Unified XL：同一份 route 使用完整个 32-stage 会逐渐产生颜色、纹理和局部几何漂移，说明 route 有生命周期。
- Unified XL：coarse K/V summary 可以完整采样，但当前激进 hybrid 不如较保守的 40% route。
- Multistage B/L/XL：head-averaged spatial route 只能作为弱 proposal，直接控制 fine attention 不稳定，不建议继续做 training-free 直接复用。
- 当前代码先计算 dense QK 再施加 mask，是正确性 oracle，不是加速 kernel；因此 sparse wall time 比 dense 更慢。

## 实验配置

```text
checkpoints: 400K EMA
training allocation: finest/middle/coarsest = 300/300/400
sampling: DDPM 80/80/90 = 250 calls
transition: nearest-neighbor noise filling
CFG: 1.5
classes: 207, 282, 323, 817
seed: 20260819
```

Dense 和所有 sparse 版本使用完全相同的：

- class labels；
- initial Gaussian noise；
- 两次 resolution transition noise；
- DDPM 每一步随机数。

因此同一行图片之间的变化来自 attention 替换，而不是换了随机 seed。

## Setting 定义

### route40_early30

- 只作用于 32-stage 的前 30/90 次 denoiser calls；
- 每个 fine query 根据上一 16-stage attention route 选择约 40% coarse roots；
- 强制保留 query 自己所在的 root；
- 其余 60 次调用恢复 dense。

实际平均 exact root density 为 Unified 40.80%、Multistage 40.63%。

### route40_all90

与上面相同，但同一份跨 stage route 用满全部 90 次 fine-stage calls。

### hybrid_early30 / hybrid_all90

仅用于 Unified：

```text
前60% layers: 20% exact fine roots + previous-stage K/V summary
60%-85% layers: 40% exact roots, no summary
最后15% layers: 20% exact roots, no summary
```

平均 exact root density 约 24.96%。summary 层还会访问未展开区域的 coarse K/V，所以这不等于只做 dense attention 的 24.96% 计算量。

## Unified XL 定量结果

| setting | sparse fine calls | exact root density | decoded pixel cosine vs dense | PSNR vs dense | MAE vs dense |
|---|---:|---:|---:|---:|---:|
| dense | 0/90 | 100% | 1.000 | ∞ | 0 |
| route40_early30 | 30/90 | 40.80% | **0.9970** | **33.19 dB** | **0.0245** |
| route40_all90 | 90/90 | 40.80% | 0.9884 | 27.02 dB | 0.0544 |
| hybrid_early30 | 30/90 | 24.96% exact | 0.9943 | 29.92 dB | 0.0336 |
| hybrid_all90 | 90/90 | 24.96% exact | 0.9806 | 24.79 dB | 0.0749 |

主要观察：

1. `route40_early30` 与 dense 非常接近，证明 stage-boundary route 的正相关可以通过完整 diffusion trajectory 传递到图片。
2. `route40_all90` 仍保持类别和大构图，但细节偏差明显增大，说明 frozen route 不应覆盖整个 fine stage。
3. `hybrid_early30` 用更少 exact roots 获得尚可结果，但不如 route40；summary 需要更保守的 layer/density 策略或 finetuning。
4. `hybrid_all90` 证明“上一 stage summary 永久有效”的假设不成立。

当前 PyTorch oracle 的 denoiser wall time：

```text
dense:             12.84 s
route40_early30:   16.06 s
route40_all90:     19.73 s
hybrid_early30:    15.91 s
hybrid_all90:      20.27 s
```

这些时间不能用于加速结论：oracle 仍然生成完整 dense logits，再进行 bool mask、concat 和 softmax，而且没有使用 fused block-sparse kernel。

## Multistage B/L/XL 定量结果

| setting | sparse fine calls | exact root density | decoded pixel cosine vs dense | PSNR vs dense | MAE vs dense |
|---|---:|---:|---:|---:|---:|
| dense | 0/90 | 100% | 1.000 | ∞ | 0 |
| route40_early30 | 30/90 | 40.63% | 0.9723 | 23.58 dB | 0.0753 |
| route40_all90 | 90/90 | 40.63% | 0.9543 | 21.14 dB | 0.1145 |

Multistage 的 L 和 XL 是独立训练的模型：

- head 编号不对应；
- Q/K/V feature space 不对应；
- layer depth 不相同；
- 只能将 source attention 对 heads 求平均后用作 spatial proposal。

即便如此，`early30` 也已经造成明显样本变化；`all90` 对猫脸产生严重结构破坏。Multistage 若要继续，需要训练 fine-model router/adapter，而不是直接 transfer mask。

## 肉眼观察

### Unified XL

- golden retriever：early30 与 dense 基本一致；all90/hybrid-all 的鼻部颜色和背景纹理有漂移。
- tiger cat：对 route 生命周期最敏感；hybrid-all 的脸部、眼睛和绿色背景发生明显改变。
- monarch butterfly：四种 sparse 设置都保持了主体与构图，局部翅膀和花朵纹理有差别。
- sports car：early30 基本保持车辆形状；all90/hybrid-all 的人、车灯和车身边缘发生变化。

### Multistage

- early30 已经改变猫脸和汽车局部几何；
- all90 的猫脸出现明显破碎/重复结构；
- 狗和蝴蝶保持类别，但细节及背景也明显偏离 dense。

## 对方法的更新

当前数据支持：

```text
Unified 16→32 cross-stage route cache
          +
只在 fine stage 早期使用
          +
后续 refresh 或回退 dense
```

当前数据不支持：

```text
一份16-stage route无条件使用整个32-stage
Multistage L route直接控制XL heads
整段使用激进previous-stage K/V summary
```

## 下一轮 training-free 搜索

优先搜索：

```text
fine-call limit: 10, 20, 30, 45
route density:   40%, 50%, 60%
local safety:    self root, 3x3 coarse neighborhood
refresh:         第30步用当前fine pooled Q/K刷新route
summary:         只在实测最稳定的少数layers启用
```

目标不是追求最大像素相似度，而是在 50K FID 几乎不变的前提下减少 attention interactions。

## Finetuning 版本

Finetuning 很可能比 training-free 更稳。建议从 Unified XL checkpoint 开始，冻结或低学习率更新主模型，并使用 dense teacher：

```text
L = L_diffusion
  + lambda_output * ||O_sparse - O_dense||²
  + lambda_eps    * ||eps_sparse - eps_dense||²
  + lambda_route  * KL(A_coarse || Pool(A_fine))
  + lambda_kv     * KV boundary alignment
```

训练 curriculum：

1. 先只在 32-stage 前 10% timesteps 使用 60% route；
2. 逐渐扩展到前 30% timesteps和 40% route；
3. 最后训练 adaptive fallback；
4. 保留一部分 dense batches，避免模型只适应固定 mask。

## 文件

- Unified 对比图：`eval_results/rdd_cross_stage_sparse_samples/unified/unified_xl_comparison.png`
- Multistage 对比图：`eval_results/rdd_cross_stage_sparse_samples/multistage/multistage_blxl_comparison.png`
- Unified metrics：`eval_results/rdd_cross_stage_sparse_samples/unified/metrics.json`
- Multistage metrics：`eval_results/rdd_cross_stage_sparse_samples/multistage/metrics.json`
- 实现：`2d/attention_analysis/sample_rdd_cross_stage_sparse.py`
- 运行脚本：`2d/attention_analysis/run_rdd_cross_stage_sparse_samples.sh`

