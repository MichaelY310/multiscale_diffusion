# 更激进的直接复用：以采样＋VAE减少10%为目标

新增：[10%–50%速度档位、层布局对照与全部视频](RDD_REUSE_SPEED_TIERS_20260915_ZH.html)。本页保留第二轮原结果，不与新一轮混用计时。

本页是2026-09-15第二轮，和[第一轮6层复用](RDD_DIRECT_REUSE_FULLMODEL_20260915_ZH.html)分开记录。所有表格由原始metrics生成；没有结果的项目明确标记等待。

**本轮已全部完成：2组800步整网微调、32组配对图像（224张PNG）、14段主实验视频及2段同latent精度对照。** 找到了采样＋VAE减少10%以上的配置，但**没有证明激进复用后画质已恢复**。

| 代表配置 | 采样＋VAE延迟减少 | 质量结论 |
|---|---:|---|
| 图片，fine复用12/24层，800步整网微调 | 11.26% | 类别Top-1为9/32，原版24/32；明显未保质 |
| 图片，fine复用18/24层，800步整网微调 | 17.78% | 类别Top-1为1/32；不可用，不能靠速度掩盖 |
| Spatial421，1280×768×81帧，两级各复用24/30层，FP32 VAE | 10.99% | 未微调视频有严重块状伪影 |
| Temporal221，832×448×29帧，fine复用24/30层，两边同用BF16 VAE | 12.12% | 狐狸主体保留，细节有变化；尚非正式质量验证 |

速度为共享GPU下暖启动配对测量的平均延迟下降，不是所有部署的保证。训练和评估已结束，未自动延长训练。

## 1. 是否充分利用了低分辨率信息

上一轮没有：图片只用了16→32的6/24层；视频只在最后一个stage复用少数层。这次扩展为：

| 路径 | 做法 | 低分辨率信息从哪里来 |
|---|---|---|
| 图片复用12/24层 | fine的50% blocks直接加上coarse更新，整网微调800步 | 最后一次16×16调用，t=300 |
| 图片复用18/24层 | fine的75% blocks直接加上coarse更新，整网微调800步 | 同上；不新增adapter |
| 视频fine18/fine24 | 最后stage复用18/30或24/30层 | 上一stage真实算过的更新 |
| 视频cascade24 | middle、fine各复用24/30层，保留的真实层错开 | coarse→middle→fine；middle新算的层能补充更新 |
| 视频cascade28 | middle、fine各复用28/30层，只保留首尾层真实计算 | 大量更新从最初coarse级联传递 |

**这叫利用更多已有更新，不是让低分辨率凭空拥有高分辨率细节。** 未复用层仍需负责当前时间、文本条件和细节修正。大幅复用会限制这种修正能力，不能保证多训练就一定追平。

视频factor仍为Spatial `(4,1)→(2,1)→(1,1)`；两个数分别为空间、时间缩小倍数。所有视频同一个狐狸prompt、seed20660913、DMD3NFE、noise_filling。没有同时叠加C2F、StepReuse或改变VSA密度。

## 2. 公平性与10%的硬上限

比较对象始终是**同一个RDD分辨率路径＋VSA20、不做block复用**。精度、seed、帧数、输出、resident/offload、VAE都相同。每次真实解码，不用常数拼总时间。

`总延迟减少比例 = 省下的去噪时间 / (原去噪时间 + 原VAE时间)`。上一轮1280×768×29帧约2.7秒去噪、20秒VAE；即使把去噪全部删掉，总时间也最多减少约12%。所以想达到10%，须接近消除大部分去噪计算。

本轮也测试81帧，同组baseline和候选都是81帧，不能拿29帧候选与81帧baseline比较。更长序列的收益不能冒充所有视频长度都能达到的收益。

**硬件仍与其他训练共享，只使用GPU0–3。** 暖启动和ABBA/BAAB减小漂移，但不能把这些数字当作独占GPU的正式论文结果。视频只用未经RDD微调的FastWan，尚未做视频训练或正式质量评价。

## 3. 新增的实现修复

大规模复用首次触及本任务的显存额度：旧controller把已经消费过的fine缓存留到了VAE。现在只保存下一stage真正需要的更新，消费一次就释放，不把最终fine缓存带入VAE。

这个优化**只用于每stage一次NFE的3步视频**，不能直接套进每stage多次NFE的图片采样。CPU测试及真实视频轨迹逐元素一致性检查要求差异为0。另启用本进程的expandable allocator缓解碎片；不改变其他进程或数学计算。

## 4. 视频实测与所有输出

百分比均为**延迟减少**，不是吞吐提升。秒数是各臂中位数；百分比是配对组均值，所以四舍五入后不必精确互相换算。区间为同次实验组bootstrap，仅表征组间波动。不同配置分别与各自GPU上的同期baseline配对，不把跨卡的绝对时间差直接当成方法收益。

### 29帧：只复用fine

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| fine18 | 2.651→1.450s | 45.15% | 22.734→21.539s | **5.22%** [5.18,5.30]% |
| fine24 | 2.648→1.053s | 60.29% | 22.716→21.189s | **6.74%** [6.61,6.90]% |

输出1280×768×29帧，16fps；`[[4, 1], [2, 1], [1, 1]]`；同模型、同prompt与seed；两边都用fp32 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/five_frame_comparison.jpg)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### fine18

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/fine18.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/fine18.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'2': [1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **1.450s**；中位数比1.829×；配对延迟减少45.15% |
| 采样＋VAE | **21.539s**；中位数比1.055×；配对延迟减少5.22% |

#### fine24

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/fine24.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/fine24.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'2': [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **1.053s**；中位数比2.515×；配对延迟减少60.29% |
| 采样＋VAE | **21.189s**；中位数比1.072×；配对延迟减少6.74% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_1280_fine/metrics.json)

### 29帧：两级级联

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| cascade24 | 2.646→0.780s | 70.52% | 22.700→20.774s | **8.41%** [8.25,8.57]% |
| cascade28 | 2.654→0.453s | 83.15% | 22.685→20.511s | **9.60%** [9.42,9.77]% |

输出1280×768×29帧，16fps；`[[4, 1], [2, 1], [1, 1]]`；同模型、同prompt与seed；两边都用fp32 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/five_frame_comparison.jpg)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### cascade24

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/cascade24.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/cascade24.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'1': [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28], '2': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **0.780s**；中位数比3.393×；配对延迟减少70.52% |
| 采样＋VAE | **20.774s**；中位数比1.093×；配对延迟减少8.41% |

#### cascade28

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/cascade28.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/cascade28.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×29帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], '2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **0.453s**；中位数比5.862×；配对延迟减少83.15% |
| 采样＋VAE | **20.511s**；中位数比1.106×；配对延迟减少9.60% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_1280_cascade/metrics.json)

### 81帧：middle与fine各复用24层

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| cascade24 | 9.593→2.457s | 74.38% | 65.415→58.269s | **10.99%** [10.78,11.13]% |

输出1280×768×81帧，16fps；`[[4, 1], [2, 1], [1, 1]]`；同模型、同prompt与seed；两边都用fp32 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/five_frame_comparison.jpg)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×81帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### cascade24

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/cascade24.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/cascade24.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×81帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'1': [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28], '2': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **2.457s**；中位数比3.904×；配对延迟减少74.38% |
| 采样＋VAE | **58.269s**；中位数比1.123×；配对延迟减少10.99% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_cascade24/metrics.json)

### 81帧：fine复用24层

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| fine24 | 9.565→3.502s | 63.25% | 65.181→59.164s | **9.27%** [9.23,9.31]% |

输出1280×768×81帧，16fps；`[[4, 1], [2, 1], [1, 1]]`；同模型、同prompt与seed；两边都用fp32 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/five_frame_comparison.jpg)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×81帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### fine24

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/fine24.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/fine24.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 1280×768×81帧；`[[4, 1], [2, 1], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'2': [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；fp32 VAE，与baseline相同 |
| 去噪 | **3.502s**；中位数比2.731×；配对延迟减少63.25% |
| 采样＋VAE | **59.164s**；中位数比1.102×；配对延迟减少9.27% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_1280_81f_fine24/metrics.json)

### 额外部署对照：同用BF16 VAE，Temporal221复用18层

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| fine18 | 1.204→0.879s | 27.08% | 3.617→3.296s | **8.82%** [8.62,9.02]% |

输出832×448×29帧，16fps；`[[1, 2], [1, 2], [1, 1]]`；同模型、同prompt与seed；两边都用bf16 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/five_frame_comparison.jpg)

**这是不同部署精度下的独立公平对照，不与上方FP32总时间交叉计算加速。** BF16不是本方法贡献，也不能算成复用省下的时间。

同一个baseline latent的FP32/BF16解码审计：PSNR=57.55dB，MAE=0.001954，max_abs=0.135821，像素范围[-1,1]。不是严格逐元素一致，也不是正式质量等价证明。

[同latent的FP32解码视频](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/baseline_fp32_decode.mp4)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 832×448×29帧；`[[1, 2], [1, 2], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；bf16 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### fine18

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/fine18.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/fine18.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 832×448×29帧；`[[1, 2], [1, 2], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'2': [1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；bf16 VAE，与baseline相同 |
| 去噪 | **0.879s**；中位数比1.370×；配对延迟减少27.08% |
| 采样＋VAE | **3.296s**；中位数比1.097×；配对延迟减少8.82% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16/metrics.json)

### 额外部署对照：同用BF16 VAE，Temporal221复用24层

| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |
|---|---|---|---|---|
| fine24 | 1.221→0.785s | 35.85% | 3.630→3.196s | **12.12%** [11.82,12.46]% |

输出832×448×29帧，16fps；`[[1, 2], [1, 2], [1, 1]]`；同模型、同prompt与seed；两边都用bf16 VAE。

![固定五帧、所有方法](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/five_frame_comparison.jpg)

**这是不同部署精度下的独立公平对照，不与上方FP32总时间交叉计算加速。** BF16不是本方法贡献，也不能算成复用省下的时间。

同一个baseline latent的FP32/BF16解码审计：PSNR=57.55dB，MAE=0.001954，max_abs=0.135821，像素范围[-1,1]。不是严格逐元素一致，也不是正式质量等价证明。

[同latent的FP32解码视频](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/baseline_fp32_decode.mp4)

#### baseline

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/baseline.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/baseline.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 832×448×29帧；`[[1, 2], [1, 2], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；bf16 VAE，与baseline相同 |
| 时间参照 | 上表各行同期baseline；1× |

#### fine24

<video controls preload="none" poster="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/fine24.png" src="../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/fine24.mp4" style="width:100%;max-width:800px"></video>

| 配置 | 内容 |
|---|---|
| 输出 / stage | 832×448×29帧；`[[1, 2], [1, 2], [1, 1]]` |
| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |
| 复用层，0开始 | `{'2': [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28]}`；字典key1为middle，2为fine；scale=1 |
| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；bf16 VAE，与baseline相同 |
| 去噪 | **0.785s**；中位数比1.555×；配对延迟减少35.85% |
| 采样＋VAE | **3.196s**；中位数比1.136×；配对延迟减少12.12% |

[原始配置、每次计时和缓存审计](../../eval_results/aggressive_direct_reuse_20260915/video_832_temporal221_bf16_high24/metrics.json)

## 5. 图片整网微调验证

原RDD-Unified-L/2 400K、300/300/400；24blocks、hidden1024；没有新参数。两个版本都从原400K开始，不从旧adapter初始化。每个实验单卡完整batch256，AdamW lr1e-5，800updates，保存200/400/800快照。沿用第一轮同一4096张teacher轨迹/128张验证数据、fine＋coarse保持损失。

**图片本轮仅16→32复用，不宣称已经训练8→16级联。** 现有语料缺少完整middle-stage训练轨迹，不应把只训练middle终点冒充已训练整个middle阶段。视频级联目前是独立的速度/质量sanity。

checkpoint仍是原模型结构的权重，但新权重必须配合对应12层或18层mask采样。不能把训练时复用、测试时又全算一遍的结果当成相同方法。旧400K及旧采样脚本不受影响。

分类器只用于检查目标类别，不是FID；32组固定配对图，共224张。

| 版本 | 类别Top-1 | Top-5 | 目标概率 |
|---|---:|---:|---:|
| original | 75.00% | 93.75% | 0.3185 |
| raw_reuse12 | 9.38% | 15.62% | 0.0350 |
| reuse12_200 | 25.00% | 37.50% | 0.0533 |
| reuse12_800 | 28.12% | 62.50% | 0.1165 |
| raw_reuse18 | 3.12% | 3.12% | 0.0066 |
| reuse18_200 | 3.12% | 3.12% | 0.0140 |
| reuse18_800 | 3.12% | 9.38% | 0.0082 |

图片推理BF16，DDPM80/80/90，CFG1，noise_filling；下方计时为batch8，实际VAE也同驻留执行。

| 候选 | 去噪延迟减少 | 采样＋VAE延迟减少 |
|---|---:|---:|
| reuse12_800 | 11.43% | 11.26% |
| reuse18_800 | 17.93% | 17.78% |

每张对比从左到右：原RDD、raw12、12层微调200步、12层微调800步、raw18、18层微调200步、18层微调800步；各固定分片前8张，不选最好看的样本。

![七列固定样本](../../eval_results/aggressive_direct_reuse_20260915/image_preview/shard_0/comparison.png)

![七列固定样本](../../eval_results/aggressive_direct_reuse_20260915/image_preview/shard_1/comparison.png)

## 6. 人工检查与结论

### 已确认的29帧结果

只复用fine24层的总延迟减少6.74%，middle和fine各复用24层是8.41%，各复用28层是9.60%。**29帧FP32 VAE这组没有达到10%，不应四舍五入宣称达标。** 28层版本已经省掉83.15%的去噪延迟，但VAE没被省掉。

这回答了“是否能更多利用low-resolution”：能，级联让coarse的信息也替代middle的重复计算；但它不是无限收益，而且和“直接把所有high-resolution hidden换成low-resolution hidden”不同，当前fine hidden仍保留，只把更新量换成低分辨率来源。

### 29帧未微调视频的可见问题

原Spatial421的狐狸已经在眼睛、口鼻处带彩色块状伪影。fine24继续保留大轮廓，但皮毛和眼部细节变差；cascade24/28的细节损失更明显，不能当成画质合格的视频。它们只验证了计算成本，以及未经训练直接替代有多大破坏。

更多训练可能修复，但当前证据不能保证：nearest上采样的同一个coarse cell会给多个fine位置相同的更新；真正区分这些位置、响应新timestep的工作只能由保留的fine层及当前fine特征完成。28/30层复用只留下首尾两层，这个限制很强。不能只追求“跳到足够多”而不验证最终画质。

### 已确认的两个视频速度过线配置

**FP32 VAE、1280×768×81帧**：middle/fine各复用24/30层，采样＋VAE从65.415秒到58.269秒，配对延迟减少**10.99%**，本次组区间[10.78%,11.13%]。同样81帧、只复用fine24层则只减少9.27%。这是本轮“多利用一个coarse阶段能再省一些计算”的直接速度对照，但两配置分别在各自GPU与baseline配对，不能把跨卡绝对差当成独立显著性检验。

**BF16 VAE、832×448×29帧、Temporal221**：fine复用24/30层，采样＋VAE从3.630秒到3.196秒，配对延迟减少**12.12%**，区间[11.82%,12.46%]。两边完全使用同样的BF16 VAE，因此这个12.12%是复用的增量收益；**不能拿3.196秒和旧FP32的约8.9秒相比，说全部差距都是复用贡献。**

从固定五帧检查，81帧Spatial级联版本有严重块状伪影，不能称作可用的高质量结果；较温和的Temporal221保留狐狸和雪地森林主体，大体动作也在，但皮毛、脸和局部形状变化明显。后者是当前更适合进一步训练验证的速度配置，仍不能声称无损。

### 对下一轮训练的具体警惕

图片整网微调的结果并不支持盲目增加复用：32个配对样本里，原模型Top-1为24/32；12层复用从未微调的3/32，提高到200步的8/32、800步的9/32；18层复用在未微调、200步、800步都是1/32。它不是FID，但结合所有固定样本中的脸、鸟翼、物体形状破坏，足以说明当前激进配置未保质。

800步时fine teacher-input epsilon MSE，12层为0.03063，18层为0.06431，均比初始化显著降低；与此同时真实rollout仍很差。因此不应把验证MSE下降当作生图质量已经恢复。

当前全模型训练的coarse保持损失只使用8×8的t=600、16×16的t=300两个边界输入，而整网参数会影响各stage的**全部**timestep；fine训练输入也来自teacher，而不是student实际滚出的状态。扩大复用后，仅仅延长相同损失的训练，可能继续降低teacher-input MSE，却修不好真实采样。

这还不是已证明的单一原因。更有针对性的下一轮应补齐coarse全时间范围的保持/原生diffusion训练，并检查student rollout状态上的蒸馏，再判断是否需要增加真实fine层。不能把“800步还没修好”直接解释为永远训练不好，也不能反过来保证“只要再训练就一定修好”。

还要区分图片与视频：本轮图片在90次fine调用里反复用同一份边界缓存，三步视频只在最后一次fine调用使用它。误差累积次数、模型和训练方式都不同；不能把图片分类器下降直接换算成视频质量下降，也不能用一段尚可的狐狸视频证明图片方法已经保质。


## 如何读这些结果

速度通过和质量通过是两件事。某一配置平均总延迟减少10%不代表所有长度、硬件和共享负载都达到10%；若区间跨过10%，也不能声称稳定越过门槛。视频未微调，若主体崩坏就只能作为计算预算证据。

训练fine预测误差下降，不保证student自己滚出的图片恢复。更多复用可能让coarse预测被拉偏、缓存跨timestep不匹配更大，或者让真实fine层不足以补细节。所有失败输出仍保留。

## 7. 可执行代码与原始数据

- [激进图像训练](../../2d/attention_analysis/train_aggressive_direct_reuse.py)
- [图像mask与共享原实现](../../2d/attention_analysis/aggressive_direct_reuse.py)
- [图像配对采样与真实VAE计时](../../2d/attention_analysis/sample_aggressive_direct_reuse.py)
- [29帧视频benchmark](../../rdd_video/benchmark_aggressive_direct_reuse.py)
- [81帧视频benchmark](../../rdd_video/benchmark_aggressive_direct_reuse_long.py)
- [单次stage缓存生命周期优化](../../rdd_video/attention_analysis/one_call_cascade_controller.py)
- [CPU正确性测试](../../tests/test_aggressive_reuse.py)
- [结果SHA审计](../../eval_results/aggressive_direct_reuse_20260915/summary_and_audit.json)

服务器结果：`/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915`。旧模型、第一轮结果及其他用户任务均未覆盖或停止。
