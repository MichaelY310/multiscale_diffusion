# FastWan 上的 RDD、VSA 与 Attention Reuse

> 本页只展示 FastWan/FastVideo 实验。所有动态分辨率实验均使用 **noise filling**：上采样上一 stage 的 clean prediction，再按新 stage 的噪声强度抽取完整 IID Gaussian。原始 Wan 调试结果不再放在主页面。

## 1. 四个方法分别是什么

| 名称 | 唯一作用 | 不做什么 |
|---|---|---|
| **RDD** | DMD denoising 从低空间/时间分辨率走到完整分辨率 | 本身不规定 attention 必须 dense 或 sparse |
| **C2F-Route(p%)** | 上一分辨率的 block route 决定下一分辨率只计算哪些 Q-K block pairs | 不复用旧 attention 数值；不与在线 VSA router 同时使用 |
| **StepReuse-Out(k)** | 同一 resolution stage 内每 k 次 model call 重算一次 `O=A·V`，中间调用复用旧 `O` | 不减少 denoiser 调用次数；分辨率一变缓存立即失效 |
| **VSA(p%)** | 每层、每步用当前 fine Q/K/V 做 coarse router，再计算选中的 fine block pairs，并融合 learned coarse branch | 它是 FastWan checkpoint 的训练组成；不是 C2F-Route |

FastWan 的 `StepReuse-Out` 在官方 VSA backend 输出处命中：复用旧的 `O=A·V`，跳过本次 VSA coarse/sparse attention，但当前 QKV projection 和 output projection 仍运行。它是保守的 training-free 版本，不代表最终可达到的最大加速。

## 2. 哪些组合真的有意义

| 组合 | 是否运行 | 原因 |
|---|---|---|
| RDD + VSA | 是 | 使用 FastWan 原生 VSA，只改变三次 DMD call 的分辨率路径 |
| RDD 三分辨率 + StepReuse-Out | **无命中** | `(4,4)→(2,2)→(1,1)` 每个 shape 只调用一次 |
| RDD 两分辨率 + StepReuse-Out | 是 | 令前两次 DMD call 使用同一 coarse shape，第二次产生一次真实复用 |
| 固定全分辨率 + VSA + StepReuse-Out | 是 | 三次 DMD call 都共享 full-resolution shape |
| C2F-Route + 在线 VSA router | **否** | 两者都决定 sparse route；C2F 是对在线 VSA router 的替代，不是叠加 |

重要限制：目前没有 RDD-FastWan 训练权重。动态分辨率视频是 **zero-finetune 机制实验**，用于判断速度上限、哪类分辨率切换破坏最小，以及复用能否运行；不能当作最终论文质量。

计时口径：冷启动表保留首次 shape 编译成本。热启动表对每种路径先完整运行 1 次，再运行 3 次并报告 denoise 中位数；不含模型加载、T5 和 VAE。“相对首行”使用同表的固定全分辨率 VSA20 作为 1.00×。任务仍与训练共享 GPU，因此应看中位数和数量级，不把小于约 5% 的差距当成结论。

## 3. 第一批视频：冷启动与 VSA 密度

共同配置：FastWan2.1-T2V-1.3B-Diffusers，832×448，29 帧，seed 20260905，DMD timestep `[1000,757,522]`，shift 8，无额外 CFG。完成 8/8。

| ID | 分辨率路径 | attention 方法 | 名义保留 | 实际保留 | Reuse 命中 | denoise | denoise 加速 | sampler+VAE | 总加速 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_vsa20` | 固定全分辨率 `(1,1)→(1,1)→(1,1)` | VSA(20%) | 20% | — | 0 | 4.20 s | 1.00× | 6.40 s | 1.00× |
| `fixed_vsa20_stepreuse2` | 固定全分辨率 `(1,1)→(1,1)→(1,1)` | VSA(20%) + StepReuse-Out(2) | 20% | — | 30 | 3.95 s | 1.06× | 6.08 s | 1.05× |
| `fixed_vsa20_stepreuse3` | 固定全分辨率 `(1,1)→(1,1)→(1,1)` | VSA(20%) + StepReuse-Out(3) | 20% | — | 60 | 3.93 s | 1.07× | 6.16 s | 1.04× |
| `rdd_spatiotemporal_vsa100` | 时空 RDD `(4,4)→(2,2)→(1,1)` | VSA(100%) | 100% | — | 0 | 7.90 s | 0.53× | 10.15 s | 0.63× |
| `rdd_spatiotemporal_vsa50` | 时空 RDD `(4,4)→(2,2)→(1,1)` | VSA(50%) | 50% | — | 0 | 7.88 s | 0.53× | 10.04 s | 0.64× |
| `rdd_spatiotemporal_vsa20` | 时空 RDD `(4,4)→(2,2)→(1,1)` | VSA(20%) | 20% | — | 0 | 8.16 s | 0.51× | 10.41 s | 0.61× |
| `rdd_spatial_only_vsa20` | 仅空间 RDD `(4,1)→(2,1)→(1,1)` | VSA(20%) | 20% | — | 0 | 8.17 s | 0.51× | 10.42 s | 0.61× |
| `rdd_temporal_only_vsa20` | 仅时间 RDD `(1,4)→(1,2)→(1,1)` | VSA(20%) | 20% | — | 0 | 5.59 s | 0.75× | 7.81 s | 0.82× |

<div class="method-gallery">
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20.mp4"></video>
  <div class="method-copy"><h4><code>fixed_vsa20</code></h4><ul>
    <li>路径：固定全分辨率 <code>(1,1)→(1,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>4.20 s</strong>；记录总时间 <strong>6.40 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20_stepreuse2_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20_stepreuse2.mp4"></video>
  <div class="method-copy"><h4><code>fixed_vsa20_stepreuse2</code></h4><ul>
    <li>路径：固定全分辨率 <code>(1,1)→(1,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(2)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：30 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>3.95 s</strong>；记录总时间 <strong>6.08 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20_stepreuse3_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/fixed_vsa20_stepreuse3.mp4"></video>
  <div class="method-copy"><h4><code>fixed_vsa20_stepreuse3</code></h4><ul>
    <li>路径：固定全分辨率 <code>(1,1)→(1,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(3)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：60 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>3.93 s</strong>；记录总时间 <strong>6.16 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa100_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa100.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_vsa100</code></h4><ul>
    <li>路径：时空 RDD <code>(4,4)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(100%)；名义保留 100%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>7.90 s</strong>；记录总时间 <strong>10.15 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa50_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa50.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_vsa50</code></h4><ul>
    <li>路径：时空 RDD <code>(4,4)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(50%)；名义保留 50%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>7.88 s</strong>；记录总时间 <strong>10.04 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa20_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatiotemporal_vsa20.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_vsa20</code></h4><ul>
    <li>路径：时空 RDD <code>(4,4)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>8.16 s</strong>；记录总时间 <strong>10.41 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatial_only_vsa20_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_spatial_only_vsa20.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatial_only_vsa20</code></h4><ul>
    <li>路径：仅空间 RDD <code>(4,1)→(2,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>8.17 s</strong>；记录总时间 <strong>10.42 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_temporal_only_vsa20_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_f29/rdd_temporal_only_vsa20.mp4"></video>
  <div class="method-copy"><h4><code>rdd_temporal_only_vsa20</code></h4><ul>
    <li>路径：仅时间 RDD <code>(1,4)→(1,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>5.59 s</strong>；记录总时间 <strong>7.81 s</strong></li>
  </ul></div>
</article>
</div>

## 4. FastWan 热启动：RDD × StepReuse-Out

共同配置仍是 FastWan DMD3、832×448、29 帧、相同 prompt/seed/timestep、VSA 保留 20%，只使用 noise filling。`reuse2` 表示同一 shape 的第二次 model call 在每个 transformer layer 复用第一次的 attention output。完成 9/9。

| ID | 分辨率路径 | attention 方法 | 名义保留 | 实际保留 | Reuse 命中 | denoise | denoise 加速 | sampler+VAE | 总加速 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_vsa20_hot` | 固定全分辨率 `(1,1)→(1,1)→(1,1)` | VSA(20%) | 20% | — | 0 | 1.88 s | 1.00× | 4.11 s | 1.00× |
| `fixed_vsa20_reuse2_hot` | 固定全分辨率 `(1,1)→(1,1)→(1,1)` | VSA(20%) + StepReuse-Out(2) | 20% | — | 30 | 1.78 s | 1.06× | 3.98 s | 1.03× |
| `rdd_temporal_2stage_hot` | 仅时间两阶段 RDD `(1,2)→(1,2)→(1,1)` | VSA(20%) | 20% | — | 0 | 1.25 s | 1.51× | 3.45 s | 1.19× |
| `rdd_temporal_2stage_reuse2_hot` | 仅时间两阶段 RDD `(1,2)→(1,2)→(1,1)` | VSA(20%) + StepReuse-Out(2) | 20% | — | 30 | 1.23 s | 1.52× | 3.48 s | 1.18× |
| `rdd_spatial_2stage_hot` | 仅空间两阶段 RDD `(2,1)→(2,1)→(1,1)` | VSA(20%) | 20% | — | 0 | 0.92 s | 2.04× | 3.13 s | 1.31× |
| `rdd_spatial_2stage_reuse2_hot` | 仅空间两阶段 RDD `(2,1)→(2,1)→(1,1)` | VSA(20%) + StepReuse-Out(2) | 20% | — | 30 | 0.91 s | 2.06× | 3.07 s | 1.34× |
| `rdd_spatiotemporal_2stage_hot` | 时空两阶段 RDD `(2,2)→(2,2)→(1,1)` | VSA(20%) | 20% | — | 0 | 1.05 s | 1.79× | 3.24 s | 1.27× |
| `rdd_spatiotemporal_2stage_reuse2_hot` | 时空两阶段 RDD `(2,2)→(2,2)→(1,1)` | VSA(20%) + StepReuse-Out(2) | 20% | — | 30 | 0.84 s | 2.23× | 3.05 s | 1.35× |
| `rdd_spatiotemporal_3stage_hot` | 时空 RDD `(4,4)→(2,2)→(1,1)` | VSA(20%) | 20% | — | 0 | 0.88 s | 2.14× | 3.05 s | 1.35× |

<div class="method-gallery">
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_hot.mp4"></video>
  <div class="method-copy"><h4><code>fixed_vsa20_hot</code></h4><ul>
    <li>路径：固定全分辨率 <code>(1,1)→(1,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>1.88 s</strong>；记录总时间 <strong>4.11 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_reuse2_hot.mp4"></video>
  <div class="method-copy"><h4><code>fixed_vsa20_reuse2_hot</code></h4><ul>
    <li>路径：固定全分辨率 <code>(1,1)→(1,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(2)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：30 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>1.78 s</strong>；记录总时间 <strong>3.98 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_temporal_2stage_hot</code></h4><ul>
    <li>路径：仅时间两阶段 RDD <code>(1,2)→(1,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>1.25 s</strong>；记录总时间 <strong>3.45 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_reuse2_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_temporal_2stage_reuse2_hot</code></h4><ul>
    <li>路径：仅时间两阶段 RDD <code>(1,2)→(1,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(2)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：30 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>1.23 s</strong>；记录总时间 <strong>3.48 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatial_2stage_hot</code></h4><ul>
    <li>路径：仅空间两阶段 RDD <code>(2,1)→(2,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>0.92 s</strong>；记录总时间 <strong>3.13 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_reuse2_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatial_2stage_reuse2_hot</code></h4><ul>
    <li>路径：仅空间两阶段 RDD <code>(2,1)→(2,1)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(2)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：30 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>0.91 s</strong>；记录总时间 <strong>3.07 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_2stage_hot</code></h4><ul>
    <li>路径：时空两阶段 RDD <code>(2,2)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>1.05 s</strong>；记录总时间 <strong>3.24 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_reuse2_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_2stage_reuse2_hot</code></h4><ul>
    <li>路径：时空两阶段 RDD <code>(2,2)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%) + StepReuse-Out(2)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：30 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>0.84 s</strong>；记录总时间 <strong>3.05 s</strong></li>
  </ul></div>
</article>
<article class="method-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_3stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_3stage_hot.mp4"></video>
  <div class="method-copy"><h4><code>rdd_spatiotemporal_3stage_hot</code></h4><ul>
    <li>路径：时空 RDD <code>(4,4)→(2,2)→(1,1)</code></li>
    <li>采样：DMD 3-step / 3 NFE</li>
    <li>attention：VSA(20%)；名义保留 20%；padding 后实际保留 —</li>
    <li>StepReuse 实际命中：0 次层内 attention</li>
    <li>boundary：<strong>noise_filling</strong></li>
    <li>计时：denoise <strong>0.88 s</strong>；记录总时间 <strong>3.05 s</strong></li>
  </ul></div>
</article>
</div>

## 5. 当前结论

- 固定全分辨率 `fixed_vsa20` 正常且清晰；此前 `StepReuse-Out(2)` 仍保留主体，`StepReuse-Out(3)` 更模糊。复用不是无损操作。
- 你指出的旧 #16 已用规范名称重跑为 `rdd_spatiotemporal_vsa100`。它确实保留了狐狸轮廓，但仍有严重块状和颜色伪影。降低到 VSA 50%/20% 后更差：**主因是未训练权重直接走空间 RDD 路径，稀疏化是叠加损伤，不是唯一原因。**
- **热启动后 RDD 明确更快。**固定全分辨率是 1.880 s；标准三分辨率 RDD 是 0.880 s，即 denoiser **2.14×**，连同约 2.2 s VAE 后仍约 **1.35×**。
- **当前“又快、又最不坏”的配置是 temporal 两阶段 + reuse2：**`(1,2)→(1,2)→(1,1)`，1.233 s，denoiser **1.52×**；sampler+VAE 约 3.48 s，相对 4.11 s 为 **1.18×**。中间帧仍是清楚、写实的狐狸；但还需查看完整运动并增加 seeds。
- 时空两阶段 + reuse2 达到 0.842 s，denoiser **2.23×**、sampler+VAE **1.35×**；狐狸和场景可辨认，但风格化与结构伪影明显强于 temporal-only。
- 空间两阶段约 2.04–2.06× denoiser 加速，但画面出现明显块状/绘画化伪影。标准三分辨率虽然有 2.14×，质量仍崩坏。因此 zero-finetune 的主要不兼容来自空间 latent 网格切换。
- StepReuse 的增量并不稳定：固定全分辨率约 +6%，temporal 两阶段约 +1%，时空两阶段约 +25%。当前 GPU 与训练共享，正式论文数字必须在独占 GPU 上增加 warmup/repeats；大于 1.5× 的 RDD 主效应可信，小幅 reuse 差异暂不应过度解读。

## 6. 读图顺序

1. `fixed_vsa20_hot` 是严格速度与画质基线。
2. 比较每对两阶段 `*_hot` 与 `*_reuse2_hot`，只看复用的增量影响。
3. 横向比较 temporal、spatial、spatiotemporal，判断减少哪一维最划算。
4. `rdd_spatiotemporal_3stage_hot` 回答此前慢是否主要由三种 shape 的首次编译造成。

<style>
.method-gallery {display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin:18px 0 30px}
.method-card {overflow:hidden;border:1px solid #d0d7de;border-radius:12px;background:#fff;box-shadow:0 2px 8px rgb(31 35 40 / 6%)}
.method-card video {display:block;width:100%;aspect-ratio:832/448;object-fit:contain;background:#111}
.method-copy {padding:12px 15px 15px}
.method-copy h4 {margin:0 0 7px}
.method-copy ul {margin:0;padding-left:20px}
</style>
