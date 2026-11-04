# FastWan：RDD + Attention Reuse 热启动实验

> 这是一个全新的独立报告，只对应 `fastwan_reuse_warm_f29` 中刚完成的 **9 条 FastWan 视频**。不包含原始 Wan，不包含旧的冷启动视频，不使用 redraw。

## 1. 完整公共配置

| 项目 | 设置 |
|---|---|
| 模型 | `FastWan2.1-T2V-1.3B-Diffusers`，由 FastVideo native `WanTransformerBlock_VSA` 加载 |
| 输出 | `832×448`，29 帧，16 FPS（约 1.81 秒） |
| 完整 latent | `(C,T,H,W)=(16,8,56,104)` |
| Prompt | `A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.` |
| Seed | `20260905` |
| Sampler | FastWan DMD 3-step / 3 NFE |
| DMD timestep | `[1000,757,522]`，shift `8.0` |
| CFG | 无额外 CFG；使用蒸馏后的 FastWan checkpoint |
| Attention | FastWan 原生 VSA；`vsa_sparsity=0.8`，即名义保留 20% fine tile pairs |
| Transition | 只使用 `noise_filling` |
| 热启动 | 每个配置先完整运行 1 条不计时 trajectory，再运行 3 条计时 trajectory，报告中位数 |
| 硬件状态 | 物理 GPU0；与已有训练共享 GPU，因此不是最终论文级独占计时 |

### Stage factor 到底表示什么

每个 pair 是 `(spatial_factor, temporal_factor)`：

- `(1,1)`：完整空间、完整时间；latent 为 `(8,56,104)`。
- `(1,2)`：空间不变，时间减半；latent 为 `(4,56,104)`。
- `(2,1)`：时间不变，H/W 各减半；latent 为 `(8,28,52)`。
- `(2,2)`：时间、H、W 都减半；latent 为 `(4,28,52)`。
- `(4,4)`：时间、H、W 都为完整 latent 的约四分之一；本实验为 `(2,14,26)`。

### Noise filling 的精确含义

从上一 stage 得到 clean prediction 后，先把 clean latent 上采样到下一 stage 的 shape，然后重新抽取该 shape 的完整 IID Gaussian：

$$
x_{\sigma_{next}}=(1-\sigma_{next})\,\operatorname{Upsample}(\hat x_0)+\sigma_{next}\epsilon_{new}.
$$

这里没有保留上一 stage 的旧噪声，也没有 redraw 这个独立实验分支。

### StepReuse-Out(2) 的精确含义

如果连续两个 DMD model call 的 latent shape 相同，每个 transformer layer 的第二次调用直接复用第一次保存的 `O=A·V`。当前实现仍计算当前 Q/K/V projection 和 output projection，只跳过 VSA attention backend。因此：

- 三分辨率 `(4,4)→(2,2)→(1,1)` 没有重复 shape，reuse 必然命中 0 次。
- 两阶段 `coarse→coarse→full` 的第二个 coarse call 会在 30 层各命中一次，共 30 次。
- 固定 `full→full→full` 在第二个 call 命中 30 次；第三个 call 重新计算。

## 2. 全部数值结果

基线是 `fixed_vsa20_hot`。`sampler+VAE` 不含 T5/model load 和视频编码，只是 sampler wrapper 中位数加本次 VAE decode。

| ID | stage factor `(spatial,temporal)` | latent `(T,H,W)` | Reuse 命中 | 3 次 denoise | median | denoise 加速 | sampler+VAE | 总加速 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `fixed_vsa20_hot` | `(1,1)→(1,1)→(1,1)` | `(8,56,104)→(8,56,104)→(8,56,104)` | 0 | 1.892/1.838/1.880 s | **1.880 s** | **1.00×** | 4.106 s | **1.00×** |
| `fixed_vsa20_reuse2_hot` | `(1,1)→(1,1)→(1,1)` | `(8,56,104)→(8,56,104)→(8,56,104)` | 30 | 1.779/1.780/1.823 s | **1.780 s** | **1.06×** | 3.985 s | **1.03×** |
| `rdd_temporal_2stage_hot` | `(1,2)→(1,2)→(1,1)` | `(4,56,104)→(4,56,104)→(8,56,104)` | 0 | 1.246/1.245/1.266 s | **1.246 s** | **1.51×** | 3.448 s | **1.19×** |
| `rdd_temporal_2stage_reuse2_hot` | `(1,2)→(1,2)→(1,1)` | `(4,56,104)→(4,56,104)→(8,56,104)` | 30 | 1.233/1.228/1.285 s | **1.233 s** | **1.52×** | 3.478 s | **1.18×** |
| `rdd_spatial_2stage_hot` | `(2,1)→(2,1)→(1,1)` | `(8,28,52)→(8,28,52)→(8,56,104)` | 0 | 0.921/0.927/0.884 s | **0.921 s** | **2.04×** | 3.126 s | **1.31×** |
| `rdd_spatial_2stage_reuse2_hot` | `(2,1)→(2,1)→(1,1)` | `(8,28,52)→(8,28,52)→(8,56,104)` | 30 | 0.898/0.924/0.913 s | **0.913 s** | **2.06×** | 3.074 s | **1.34×** |
| `rdd_spatiotemporal_2stage_hot` | `(2,2)→(2,2)→(1,1)` | `(4,28,52)→(4,28,52)→(8,56,104)` | 0 | 1.040/1.053/1.058 s | **1.053 s** | **1.79×** | 3.244 s | **1.27×** |
| `rdd_spatiotemporal_2stage_reuse2_hot` | `(2,2)→(2,2)→(1,1)` | `(4,28,52)→(4,28,52)→(8,56,104)` | 30 | 0.835/0.842/0.880 s | **0.842 s** | **2.23×** | 3.047 s | **1.35×** |
| `rdd_spatiotemporal_3stage_hot` | `(4,4)→(2,2)→(1,1)` | `(2,14,26)→(4,28,52)→(8,56,104)` | 0 | 0.896/0.835/0.880 s | **0.880 s** | **2.14×** | 3.050 s | **1.35×** |

## 3. 最重要的结论

1. **热启动后 RDD 的 denoiser 确实更快。**标准三分辨率 RDD 为 0.880 s，对比固定全分辨率 1.880 s，是 2.14×。
2. **目前最可信的质量/速度折中是 temporal 两阶段 RDD。**无复用版 1.246 s，即 denoiser 1.51×、sampler+VAE 1.19×，中间帧保持写实且结构正常。
3. temporal 两阶段加入 reuse2 后是 1.233 s，只比 1.246 s 快约 1%；这不能算明确的额外加速。它的意义主要是确认 FastWan+RDD+reuse 的执行路径和缓存命中正确。
4. 时空两阶段+reuse2 达到 0.842 s，即 denoiser 2.23×、sampler+VAE 1.35×；但画质出现明显结构/风格伪影。
5. 空间降采样和三分辨率 RDD 的 zero-finetune 画质明显更差。因此当前最合理的下一步是训练/微调 RDD-FastWan，而不是继续用未适配 checkpoint 推高空间降采样比例。
6. VAE 约固定占 2.2 s，把 denoiser 的 2× 加速压缩成约 1.35× 总加速。论文若汇报端到端，还需要 VAE 优化或更长视频，让 denoiser 占比更大。

## 4. 九条视频：ID、参数与文件一一对应

<div class="warm-gallery">
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_hot.mp4"></video>
  <div class="warm-copy"><h3><code>fixed_vsa20_hot</code></h3><ul>
    <li>stage factor：<code>(1,1)→(1,1)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(8,56,104)→(8,56,104)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(1)；实际命中 0</li>
    <li>3 次 denoise：1.892/1.838/1.880 s；median <strong>1.880 s</strong>；相对基线 <strong>1.00×</strong></li>
    <li>VAE：2.226 s；sampler+VAE：4.106 s；总加速 1.00×</li>
    <li>目视记录：正常基线：写实狐狸、雪地和树木清楚，主体结构稳定。</li>
    <li>文件校验：<code>sha256:d728c78008f2</code>；419.9 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/fixed_vsa20_reuse2_hot.mp4"></video>
  <div class="warm-copy"><h3><code>fixed_vsa20_reuse2_hot</code></h3><ul>
    <li>stage factor：<code>(1,1)→(1,1)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(8,56,104)→(8,56,104)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(2)；实际命中 30</li>
    <li>3 次 denoise：1.779/1.780/1.823 s；median <strong>1.780 s</strong>；相对基线 <strong>1.06×</strong></li>
    <li>VAE：2.204 s；sampler+VAE：3.985 s；总加速 1.03×</li>
    <li>目视记录：仍是清楚的写实狐狸；相对基线略软，但没有明显块状崩坏。</li>
    <li>文件校验：<code>sha256:e6776a9d3c4f</code>；452.9 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_temporal_2stage_hot</code></h3><ul>
    <li>stage factor：<code>(1,2)→(1,2)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(4,56,104)→(4,56,104)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(1)；实际命中 0</li>
    <li>3 次 denoise：1.246/1.245/1.266 s；median <strong>1.246 s</strong>；相对基线 <strong>1.51×</strong></li>
    <li>VAE：2.201 s；sampler+VAE：3.448 s；总加速 1.19×</li>
    <li>目视记录：当前最佳质量/速度折中：狐狸与森林清楚，zero-finetune 下仍接近正常视频。</li>
    <li>文件校验：<code>sha256:3bd15f954535</code>；638.5 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_temporal_2stage_reuse2_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_temporal_2stage_reuse2_hot</code></h3><ul>
    <li>stage factor：<code>(1,2)→(1,2)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(4,56,104)→(4,56,104)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(2)；实际命中 30</li>
    <li>3 次 denoise：1.233/1.228/1.285 s；median <strong>1.233 s</strong>；相对基线 <strong>1.52×</strong></li>
    <li>VAE：2.244 s；sampler+VAE：3.478 s；总加速 1.18×</li>
    <li>目视记录：主体和场景仍清楚；复用改变了细节与构图，但没有出现空间 RDD 那种严重块状伪影。</li>
    <li>文件校验：<code>sha256:9f5ed248ba31</code>；799.0 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_spatial_2stage_hot</code></h3><ul>
    <li>stage factor：<code>(2,1)→(2,1)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(8,28,52)→(8,28,52)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(1)；实际命中 0</li>
    <li>3 次 denoise：0.921/0.927/0.884 s；median <strong>0.921 s</strong>；相对基线 <strong>2.04×</strong></li>
    <li>VAE：2.204 s；sampler+VAE：3.126 s；总加速 1.31×</li>
    <li>目视记录：狐狸可辨认，但明显绘画化、块状化，局部几何和颜色不稳定。</li>
    <li>文件校验：<code>sha256:85df48392dc5</code>；587.9 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatial_2stage_reuse2_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_spatial_2stage_reuse2_hot</code></h3><ul>
    <li>stage factor：<code>(2,1)→(2,1)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(8,28,52)→(8,28,52)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(2)；实际命中 30</li>
    <li>3 次 denoise：0.898/0.924/0.913 s；median <strong>0.913 s</strong>；相对基线 <strong>2.06×</strong></li>
    <li>VAE：2.160 s；sampler+VAE：3.074 s；总加速 1.34×</li>
    <li>目视记录：与无复用空间版相似，主体可辨认但伪影明显；复用没有修复空间切换问题。</li>
    <li>文件校验：<code>sha256:9a82bf46ea1c</code>；466.2 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_spatiotemporal_2stage_hot</code></h3><ul>
    <li>stage factor：<code>(2,2)→(2,2)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(4,28,52)→(4,28,52)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(1)；实际命中 0</li>
    <li>3 次 denoise：1.040/1.053/1.058 s；median <strong>1.053 s</strong>；相对基线 <strong>1.79×</strong></li>
    <li>VAE：2.190 s；sampler+VAE：3.244 s；总加速 1.27×</li>
    <li>目视记录：狐狸与雪地可辨认，但画面明显风格化，结构和纹理存在 transition 伪影。</li>
    <li>文件校验：<code>sha256:52173d141d84</code>；461.9 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_reuse2_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_2stage_reuse2_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_spatiotemporal_2stage_reuse2_hot</code></h3><ul>
    <li>stage factor：<code>(2,2)→(2,2)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(4,28,52)→(4,28,52)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(2)；实际命中 30</li>
    <li>3 次 denoise：0.835/0.842/0.880 s；median <strong>0.842 s</strong>；相对基线 <strong>2.23×</strong></li>
    <li>VAE：2.205 s；sampler+VAE：3.047 s；总加速 1.35×</li>
    <li>目视记录：仍能辨认狐狸和森林；速度最高之一，但画质明显弱于 temporal-only。</li>
    <li>文件校验：<code>sha256:6318ba873a23</code>；452.8 KiB</li>
  </ul></div>
</article>
<article class="warm-card">
  <video controls preload="metadata" playsinline poster="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_3stage_hot_mid.jpg" src="../../eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/rdd_spatiotemporal_3stage_hot.mp4"></video>
  <div class="warm-copy"><h3><code>rdd_spatiotemporal_3stage_hot</code></h3><ul>
    <li>stage factor：<code>(4,4)→(2,2)→(1,1)</code></li>
    <li>latent (T,H,W)：<code>(2,14,26)→(4,28,52)→(8,56,104)</code></li>
    <li>attention：FastWan 原生 VSA，保留 20%；StepReuse-Out(1)；实际命中 0</li>
    <li>3 次 denoise：0.896/0.835/0.880 s；median <strong>0.880 s</strong>；相对基线 <strong>2.14×</strong></li>
    <li>VAE：2.169 s；sampler+VAE：3.050 s；总加速 1.35×</li>
    <li>目视记录：严重彩色和块状崩坏，只剩局部动物/场景线索；不属于可用画质。</li>
    <li>文件校验：<code>sha256:fa1d585a51da</code>；981.7 KiB</li>
  </ul></div>
</article>
</div>

## 5. 每条实验的低层计时记录

### `fixed_vsa20_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.645, 0.595, 0.631]; [0.563, 0.669, 0.603]; [0.592, 0.622, 0.647]`。
- attention 调用统计：online VSA `90`，reuse hits `0`，recompute/cache writes `0`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `fixed_vsa20_reuse2_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.586, 0.573, 0.600]; [0.636, 0.565, 0.560]; [0.646, 0.577, 0.588]`。
- attention 调用统计：online VSA `60`，reuse hits `30`，recompute/cache writes `60`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_temporal_2stage_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.286, 0.348, 0.602]; [0.305, 0.305, 0.624]; [0.329, 0.278, 0.646]`。
- attention 调用统计：online VSA `90`，reuse hits `0`，recompute/cache writes `0`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_temporal_2stage_reuse2_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.340, 0.280, 0.603]; [0.316, 0.274, 0.625]; [0.350, 0.253, 0.670]`。
- attention 调用统计：online VSA `60`，reuse hits `30`，recompute/cache writes `60`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_spatial_2stage_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.138, 0.126, 0.646]; [0.133, 0.151, 0.623]; [0.150, 0.150, 0.564]`。
- attention 调用统计：online VSA `90`，reuse hits `0`，recompute/cache writes `0`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_spatial_2stage_reuse2_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.150, 0.138, 0.590]; [0.151, 0.145, 0.608]; [0.146, 0.134, 0.623]`。
- attention 调用统计：online VSA `60`，reuse hits `30`，recompute/cache writes `60`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_spatiotemporal_2stage_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.206, 0.224, 0.591]; [0.227, 0.194, 0.621]; [0.212, 0.201, 0.625]`。
- attention 调用统计：online VSA `90`，reuse hits `0`，recompute/cache writes `0`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_spatiotemporal_2stage_reuse2_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.110, 0.124, 0.584]; [0.127, 0.109, 0.594]; [0.123, 0.106, 0.640]`。
- attention 调用统计：online VSA `60`，reuse hits `30`，recompute/cache writes `60`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。

### `rdd_spatiotemporal_3stage_hot`

- 三次完整 trajectory 的 stage-forward 秒数依次为：`[0.124, 0.127, 0.625]; [0.104, 0.127, 0.584]; [0.151, 0.115, 0.602]`。
- attention 调用统计：online VSA `90`，reuse hits `0`，recompute/cache writes `0`。
- 当前进程峰值 CUDA allocation（decode 前）：`21.48 GiB`。


## 6. 为什么冷启动特别慢；换 shape 会不会再次变慢

会。FastWan 的 VSA 使用 shape/block-layout 专门化的 CUDA/Triton kernel。某个 `(T,H,W)`、tile grid、dtype 或稀疏布局第一次出现时，除了真实矩阵计算，还可能发生：

1. Triton kernel 编译与 autotune；
2. cuBLAS/CUDA handle、workspace 和显存池的首次建立；
3. shape-specific VSA metadata、padding/index buffer 的首次构造；
4. lazy module/kernel 初始化。

冷启动日志直接显示了这个效应：

| 路径 | 第一次运行的三个 stage/model-call | 冷启动总计 | 热启动 median |
|---|---:|---:|---:|
| 固定 full→full→full | 2.920 / 0.588 / 0.627 s | 4.20 s | 1.880 s |
| 三分辨率 `(4,4)→(2,2)→(1,1)` | 3.482 / 2.273 / 2.244 s | 8.16 s | 0.880 s |

固定全分辨率只在第一个 call 遇到一种新 shape，后两个 call 已经复用同一 shape 的已编译 kernel；三分辨率 RDD 的三个 call 分别遇到三种新 shape，因此第一次视频连续支付三次初始化成本。这就是旧冷启动比较看起来 RDD 反而慢约两倍的主要原因。

如果先只热身 shape A，再换到此前从未出现的 shape B，B **很可能再次出现一次性变慢**。但只要进程不退出，之后在已经热身过的 A/B 之间切换，通常会复用缓存，不再重复完整编译；实际执行时间仍会随 token 数变化。

部署和正式 benchmark 应当：

- 启动时依次预热所有允许的 RDD stage shape，而不是只预热 full resolution；
- 把任意尺寸归入少数几个固定 `(T,H,W)` bucket，避免无限产生新 kernel key；
- 保持同一 worker 进程，保存 Triton cache，并为每个 stage 预建 metadata/workspace；
- 论文同时报告 cold-first-video 与 warm steady-state latency，主吞吐量表使用 warm median；
- 若服务主要是一次性 CLI 调用，冷启动是真实成本，不能只报告热启动。

## 7. 复现命令

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
GPU=0 WARMUP_RUNS=1 TIMED_RUNS=3 \
  bash rdd_video/run_fastwan_rdd_reuse_warm_benchmark.sh 2>&1 \
  | tee eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29/driver.log
```

<style>
.warm-gallery {display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px;margin:18px 0 30px}
.warm-card {overflow:hidden;border:1px solid #d0d7de;border-radius:12px;background:#fff;box-shadow:0 2px 8px rgb(31 35 40 / 7%)}
.warm-card video {display:block;width:100%;aspect-ratio:832/448;object-fit:contain;background:#111}
.warm-copy {padding:12px 15px 15px}
.warm-copy h3 {margin:0 0 7px}
.warm-copy ul {margin:0;padding-left:20px}
</style>
