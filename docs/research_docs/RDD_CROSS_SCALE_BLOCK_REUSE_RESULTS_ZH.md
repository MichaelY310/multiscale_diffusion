# RDD 跨尺度加速：Cross-Scale Block Residual Reuse（CSBR）

> **最新独立10K结果见[联合训练与新方案报告](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。** 固定较简单的front-keep1跳6层reader，用新种子与原RDD各生成10,000张：FID **11.6741→17.5270**，sFID **17.5614→18.8710**，IS **115.2414→93.6692**，Precision **70.59%→64.76%**，Recall **70.99%→70.11%**。完整审计通过，四张不挑选的固定对比图已放在新报告开头。不是50K FID；结果不支持已经保质。此前coarse记忆修正与50%状态回放也未可靠改善质量，768/1024张PNG归档均已完整下载并核验。未训练视频的13–15%净去噪计算空间不能与这组2D结果拼成训练后视频结论。**整体目标未达成，也不是只差视频训练。** 旧front0的10K开发FID **11.8755／16.8040／20.3236** 是另一组实验，不混用。

> **新增计算验证：** 实际FastVideo中加入完整条件bridge后，384次配对计时显示1280×768×29帧下，跳fine前6层的净denoiser延迟下降13.92–15.25%，前8层下降19.21–20.93%。包括缓存、上采样、attention/FFN和两套模型驻留；未训练Wan bridge、共享GPU、不含VAE，因此只是计算可行性，不是视频质量证明。详细8项结果、波动和限制在上述联合训练报告。2D有/无来源两组800→1600续训已完成，原0–3卡、4×64，旧模型未修改；不能把这轮2D质量和未训练视频计时拼成同一训练后模型的结果。

> **此前：条件V3对照完成，未显示有意义收益；随后定位层与时间的误差。** V3的validation主MSE仅比匹配V2低0.018%，七组生成偏差均值反而高0.051%，详见14.19。128条独立验证轨迹的只读诊断发现：较早t恢复第14层更有效，较晚t恢复第21层更有效；清零/错配coarse缓存会增加误差，见14.20。这里是诊断证据，不是新的画质/加速结果。**最终质量与视频净加速目标尚未达成。**

> **结论先说：** FastWan faux-RDD 实验给出了“若能跳过 fine block”的速度参照：温和设置减少 **12–16% denoiser 延迟**，激进设置减少 **51–61%**。真实 RDD checkpoint 上，直接跨尺度复制仍会破坏图像；训练 adapter 可以减小对 dense 输出的偏离，但目前未达到最终质量要求。旧 pilot 的 late-3 平均 latent RRMSE **0.255** 与 raw-all6 的 **0.447** 使用不同替换层数，不能单独证明 adapter 的收益，也不能将三层的质量与六层的速度组合汇报。**选择性 CSRA 仍是待验证候选，不是已经完成的方法。**

<style>
.rdd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px;margin:14px 0 26px}.rdd-card{border:1px solid #d0d7de;border-radius:10px;padding:12px;background:#fff}.rdd-card video{width:100%;border-radius:7px;background:#111}.rdd-card h4{margin:.7em 0 .3em}.rdd-config{width:100%;border-collapse:collapse;font-size:.84em}.rdd-config th,.rdd-config td{border:1px solid #d8dee4;padding:5px 7px;vertical-align:top}.rdd-config th{width:24%;background:#f6f8fa;text-align:left;white-space:nowrap}.warn{padding:10px 14px;border-left:4px solid #bf8700;background:#fff8c5}.good{padding:10px 14px;border-left:4px solid #2da44e;background:#dafbe1}
</style>

## 1. 方法是什么

Wan 的一个 Transformer block 可以粗略写成“输出 = 输入 + 本 block 做出的修改”。令低分辨率 stage 第 $l$ 层的修改为：

$$
\Delta_l^c=h_{l+1}^c-h_l^c.
$$

CSBR 先在真实的 coarse RDD stage 正常运行该 block 并保存 $\Delta_l^c$。到 fine stage 的同一层时，把它沿时间、高度、宽度放大，然后直接加到当前 fine hidden state：

$$
h_{l+1}^f=h_l^f+s\,U(\Delta_l^c).
$$

这样被选中的 fine block 整个跳过：不计算当前 fine self-attention/QKV/VSA，不计算 cross-attention，也不计算 MLP。参数 $s$ 控制 coarse 修改的注入强度。其余 block 原样运行在线 VSA-20。

~~~text
for each RDD stage coarse -> middle -> fine:
    for each Transformer block l:
        if this is a normal coarse/middle block:
            y = block_l(h)
            cache[stage, l] = stop_gradient(y - h)
            h = y
        elif l is selected for CSBR at the fine stage:
            h = h + scale * resize_T_H_W(cache[previous_stage, l])
        else:
            h = block_l(h)
~~~

这不是普通的跨 timestep attention cache。它复用的是**已经完成的低分辨率生成 stage 的整层语义修改**；普通全分辨率生成器没有这个真实 coarse trajectory。若它额外跑一遍 coarse pass 才能使用 CSBR，就已经承担了 RDD 本来就有、而它原来没有的计算。

## 2. 公平比较口径

| 项目 | 固定设置 |
|---|---|
| 模型 | 同一个 <code>FastWan2.1-T2V-1.3B-Diffusers</code> checkpoint |
| RDD | Spatial <code>(4,1)→(2,1)→(1,1)</code>，<code>noise_filling</code> |
| sampler | DMD 3 NFE，t=<code>1000/757/522</code> |
| attention | 基线和 CSBR 未跳过的 block 都用完整在线 VSA-20 |
| 执行 | resident DiT；同一进程一次加载；逐 shape warmup；候选交错顺序；CUDA Events |
| 输出 | 832×448×29 帧以及 1280×768×29 帧，batch=1 |
| 唯一变化 | fine stage 哪些 block 被 coarse block residual 代替 |

因此这里不是拿 CSBR 和全分辨率 Wan 比，也没有偷换成更低 VSA density；基线本身已经是相同 RDD + VSA-20。

## 3. 稳定主结果

<code>uniform6-s0.75</code> 是温和候选：30 个 fine blocks 中均匀跳过 6 个。<code>uniform24-s0.75</code> 是需要训练的激进候选：跳过 24 个。

| 输出 | 配置 | denoiser 加速 | denoiser 延迟减少 | 含 VAE 加速 | 总延迟减少 | latent cosine |
|---|---|---:|---:|---:|---:|---:|
| 832×448，2 prompt 均值 | uniform6-s0.75 | **1.142×** | **12.4%** | 1.029× | 2.8% | 0.925 |
| 1280×768，2 prompt 均值 | uniform6-s0.75 | **1.185×** | **15.6%** | 1.039× | 3.7% | 0.934 |
| 832×448，激进验证 | uniform24-s0.75 | **2.043×** | **51.1%** | **1.128×** | **11.3%** | 0.818 |
| 1280×768，激进验证 | uniform24-s0.75 | **2.534×** | **60.5%** | **1.170×** | **14.5%** | 0.843 |

同样跳 6 层时，分辨率从 832×448 增大到 1280×768，denoiser 延迟减少从约 12.4% 增至约 15.6%，因为被跳过的 fine block 更贵。VAE 完全没有被 CSBR 加速，所以温和配置的总时间只改善约 3–4%。激进 24 层才足以在当前 VAE 下仍达到 11.3–14.5% 总延迟减少。

### 逐 prompt 复验

| prompt / 输出 | CSBR-6 denoiser | CSBR-6 含 VAE | CSBR-12 denoiser | CSBR-12 含 VAE |
|---|---:|---:|---:|---:|
| city，832×448 | 1.141× | 1.029× | 1.341× | 1.061× |
| coast，832×448 | 1.143× | 1.029× | 1.343× | 1.060× |
| robot，1280×768 | 1.185× | 1.038× | 1.442× | 1.079× |
| horse，1280×768 | 1.184× | 1.039× | 1.440× | 1.080× |

两个 832 prompt 与两个 1280 prompt 的速度几乎重合，说明结论不是由单个 prompt 或单次计时造成的。

## 4. 速度—扰动曲线

下面的 SSIM/PSNR 是同 prompt、同 seed 相对 RDD+VSA 基线的像素扰动指标，不是 FVD/VBench，也不能替代人工画质。

| 832×448 配置 | 跳过 fine blocks | denoiser 加速 | 含 VAE 加速 | SSIM | PSNR |
|---|---:|---:|---:|---:|---:|
| CSBR-6，scale 0.75 | 6 | 1.135× | 1.027× | 0.688 | 23.91 dB |
| CSBR-12，scale 1.00 | 12 | 1.340× | 1.060× | 0.615 | 22.58 dB |
| CSBR-18，scale 0.75 | 18 | 1.618× | 1.092× | 0.615 | 22.81 dB |
| CSBR-24，scale 0.75 | 24 | 2.043× | 1.128× | 0.555 | 21.76 dB |

<div class="warn"><strong>诚实判断：</strong>当前零微调曲线证明了“跳层越多，速度越快、扰动越大”，但不能证明 24 层配置有可接受生成质量。当前 faux-RDD baseline 本身就不是 RDD 训练出的 checkpoint；CSBR-24 明显比基线更糊、更块状。论文主结果必须用 RDD+CSBR 共同微调后的 VBench/FVD/人工 A/B。</div>

## 5. 视频：温和搜索

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_csbr_quality_20260908/spatial421_rdd_vsa20_contact.jpg" src="../../eval_results/rdd_fastwan_csbr_quality_20260908/spatial421_rdd_vsa20.mp4"></video>
<h4>RDD + 在线 VSA-20 基线</h4>
<table class="rdd-config"><tr><th>RDD</th><td>(4,1)→(2,1)→(1,1)</td></tr><tr><th>CSBR</th><td>关闭，0/30 fine blocks</td></tr><tr><th>denoiser</th><td>0.415 s；1.00×</td></tr><tr><th>含 VAE</th><td>1.861 s；1.00×</td></tr></table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_csbr_quality_20260908/csbr_uniform6_s075_contact.jpg" src="../../eval_results/rdd_fastwan_csbr_quality_20260908/csbr_uniform6_s075.mp4"></video>
<h4>CSBR 温和：均匀 6 层，scale 0.75</h4>
<table class="rdd-config"><tr><th>唯一变化</th><td>fine stage 均匀跳过 6/30 blocks</td></tr><tr><th>denoiser</th><td>0.364 s；1.141×；延迟 -12.4%</td></tr><tr><th>含 VAE</th><td>1.810 s；1.029×；延迟 -2.8%</td></tr><tr><th>扰动</th><td>cosine 0.922；SSIM 0.736</td></tr></table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_csbr_quality_20260908/csbr_uniform12_s100_contact.jpg" src="../../eval_results/rdd_fastwan_csbr_quality_20260908/csbr_uniform12_s100.mp4"></video>
<h4>CSBR 中等：均匀 12 层，scale 1.0</h4>
<table class="rdd-config"><tr><th>唯一变化</th><td>fine stage 均匀跳过 12/30 blocks</td></tr><tr><th>denoiser</th><td>0.310 s；1.339×；延迟 -25.3%</td></tr><tr><th>含 VAE</th><td>1.756 s；1.060×；延迟 -5.7%</td></tr><tr><th>扰动</th><td>cosine 0.891；SSIM 0.669</td></tr></table>
</article>
</div>

## 6. 视频：跨 prompt 和分辨率

### 城市，832×448

<div class="rdd-grid">
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_city832/spatial421_rdd_vsa20.mp4"></video><h4>基线 RDD+VSA</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.413 s</td></tr><tr><th>含 VAE</th><td>基线 1.00×</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_city832/csbr_uniform6_s075.mp4"></video><h4>CSBR-6</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.362 s；1.141×；延迟 -12.3%</td></tr><tr><th>含 VAE</th><td>1.029×；延迟 -2.8%</td></tr><tr><th>cosine</th><td>0.921</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_city832/csbr_uniform12_s100.mp4"></video><h4>CSBR-12</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.308 s；1.341×；延迟 -25.4%</td></tr><tr><th>含 VAE</th><td>1.061×；延迟 -5.7%</td></tr><tr><th>cosine</th><td>0.890</td></tr></table></article>
</div>

### 机器人，1280×768

<div class="rdd-grid">
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_robot768/spatial421_rdd_vsa20.mp4"></video><h4>基线 RDD+VSA</h4><table class="rdd-config"><tr><th>denoiser</th><td>1.213 s</td></tr><tr><th>含 VAE</th><td>基线 1.00×</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_robot768/csbr_uniform6_s075.mp4"></video><h4>CSBR-6</h4><table class="rdd-config"><tr><th>denoiser</th><td>1.023 s；1.185×；延迟 -15.6%</td></tr><tr><th>含 VAE</th><td>1.038×；延迟 -3.7%</td></tr><tr><th>cosine</th><td>0.953</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline src="../../eval_results/rdd_csbr_verify_robot768/csbr_uniform12_s100.mp4"></video><h4>CSBR-12</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.841 s；1.442×；延迟 -30.7%</td></tr><tr><th>含 VAE</th><td>1.079×；延迟 -7.3%</td></tr><tr><th>cosine</th><td>0.936</td></tr></table></article>
</div>

## 7. 视频：激进速度上限

<div class="rdd-grid">
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_verify_aggressive832/spatial421_rdd_vsa20_contact.jpg" src="../../eval_results/rdd_csbr_verify_aggressive832/spatial421_rdd_vsa20.mp4"></video><h4>激进 sweep 的基线</h4><table class="rdd-config"><tr><th>CSBR</th><td>关闭</td></tr><tr><th>denoiser</th><td>0.410 s</td></tr><tr><th>含 VAE</th><td>1.00×</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform6_s075_contact.jpg" src="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform6_s075.mp4"></video><h4>CSBR-6：温和点</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.361 s；1.135×</td></tr><tr><th>含 VAE</th><td>1.027×</td></tr><tr><th>SSIM</th><td>0.688</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform18_s075_contact.jpg" src="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform18_s075.mp4"></video><h4>CSBR-18：训练候选</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.253 s；1.618×；延迟 -38.2%</td></tr><tr><th>含 VAE</th><td>1.092×；延迟 -8.5%</td></tr><tr><th>SSIM</th><td>0.615</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform24_s075_contact.jpg" src="../../eval_results/rdd_csbr_verify_aggressive832/csbr_uniform24_s075.mp4"></video><h4>CSBR-24：速度上限候选</h4><table class="rdd-config"><tr><th>denoiser</th><td>0.201 s；2.043×；延迟 -51.1%</td></tr><tr><th>含 VAE</th><td>1.128×；延迟 -11.3%</td></tr><tr><th>SSIM</th><td>0.555</td></tr></table></article>
</div>

## 8. 失败或淘汰的路线

| 路线 | 结果 | 处理 |
|---|---|---|
| 仅复用 coarse self-attention residual | 有加速，但 latent 相似度/速度 Pareto 被整 block CSBR 支配 | 不作为主方法 |
| 仅复用 coarse MLP residual | 约 5% 级别且相似度差 | 从主结果删除 |
| C2F route-only | 相对完整 VSA 的主要结构差异是删除 coarse-output branch；route 本身只省很小一部分 | 不作为 ≥10% 主张 |
| SmallQ-32 | 算子级可快，但 2D 小 stage 端到端约 1% | 仅保留工程记录 |
| generic StepReuse-Out | 全分辨率模型也能做，不是 RDD 独有 | 不满足本任务 novelty 约束 |

CSBR 能越过 10% 的根本原因不是一个很小的 sparse kernel，而是跳过高分辨率 block 的全部主干计算。它省掉的范围远大于 C2F 只省 router。

## 9. 与已有工作的关系

- [Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954) 与 [PyramidalWan](https://arxiv.org/abs/2601.04792) 都采用低到高生成轨迹，是最接近的 trajectory prior art；公开方法中没有发现“把前一真实生成 stage 的 block residual 放大后用于跳过 fine Transformer block”的机制。
- [DraftAttention](https://arxiv.org/abs/2505.14708) 在**当前高分辨率调用内部**下采样 feature，重新计算一次低分辨率 draft Q/K，再指导当前 Q/K/V；它不是复用已经完成的上一生成 stage。
- [VSA](https://arxiv.org/abs/2505.13389) 同样从当前 tokens 池化 coarse tiles，计算 coarse attention/gate，再做 fine sparse attention；CSBR 的公平基线正是相同在线 VSA-20。
- [Sparse VideoGen](https://arxiv.org/abs/2502.01776)、[Re-ttention](https://arxiv.org/abs/2505.22918)、[ProfilingDiT](https://arxiv.org/abs/2504.03140) 和 [OmniCache](https://arxiv.org/abs/2607.23844) 分别研究当前分辨率的稀疏 pattern 或跨 timestep/layer cache；它们可与 CSBR 组合，但不是“前一分辨率生成 stage → 后一分辨率整 block residual”。
- [Scale Space Diffusion](https://arxiv.org/abs/2603.08709) 是 RDD 概率路径/多尺度建模的强相关竞争者，不是这个系统复用机制的直接先例。

<div class="good"><strong>建议论文措辞：</strong>“在我们检索到的公开工作中，尚未发现一种视频生成方法同时具有真实 coarse-to-fine generative trajectory，并把已完成 coarse stage 的 Transformer block residual 直接迁移到 fine stage 来跳过 fine blocks。”不要写成未经限定的“第一个多分辨率生成”或“第一个跨尺度复用”。</div>

CSBR 的 novelty 是 **RDD trajectory × Transformer execution 的协同设计**。RDD 理论本身仍应和 f-DM、Scale Space Diffusion、Pyramidal Flow 分开、诚实讨论；CSBR 不依赖声称 RDD 是所有多分辨率 diffusion 的第一个。

## 10. 论文级下一步：训练、质量验证和净速度实测均未完成

2026-09-14补充：fine邻域修正独立实现并完成四组固定2400训练，保留原coarse全局读取，在bridge输出加入rank128的空间3×3残差分支。19项相关测试及真实FastWan零初始化完整轨迹一致性通过；新front1的384次计时完成，1280下同RDD+VSA的净去噪延迟少13.33%／15.19%（不含VAE、未训练bridge）。但同256图的生成对照没有可靠质量收益，故不继续原样延长。之后固定网络、跳层和预算完成状态回放对照，仍未改善质量；结果及下一轮候选见主报告。详见主报告开头，不把未训练视频速度和2D图像拼成训练后视频方法。

1. 从 RDD-Wan checkpoint 微调，不要从普通 FastWan 的 faux-RDD 结果判断最终质量。训练时随机采样 reuse count ∈ {0, 6, 12, 18, 24}，让一个 checkpoint 支持多个速度档位。
2. 对被替代的 fine block 加蒸馏：让 $sU(\Delta_l^c)$ 拟合 teacher fine block 的 $\Delta_l^f$；同时保留最终 flow/DMD 目标。
3. 先固定 uniform layer pattern；训练稳定后再学习 per-layer gate。gate 必须只读 coarse 或轻量统计，不能重新跑昂贵 fine router。
4. 与同一个 RDD+在线 VSA-20 teacher 比较：VBench、FVD、CLIP/文本一致性、人工 A/B、denoiser 延迟、含 VAE 延迟、峰值显存。
5. 论文主表用 CSBR-6 作为温和点、CSBR-12/18 作中间点、CSBR-24 作速度上限；只在匹配质量后报告“总延迟 ≥10%”。

## 11. 实现与复现入口

- 控制器与核心 CSBR 逻辑：<code>rdd_video/attention_analysis/fastwan_native_attention.py</code>
- 稳定配对 benchmark：<code>rdd_video/benchmark_fastwan_paired_suite.py</code>
- 单元测试：<code>tests/test_fastwan_cross_stage_block_reuse.py</code>
- 温和/激进配置：<code>rdd_video/experiments/fastwan_cross_stage_block_reuse_*.json</code> 与 <code>fastwan_csbr_aggressive.json</code>
- 远程启动器：<code>rdd_video/launch_fastwan_csbr_verify.sh</code>、<code>launch_fastwan_csbr_aggressive_gpu23.sh</code>
- 原始 JSON、全部 timing samples 和视频：<code>eval_results/rdd_fastwan_csbr_quality_20260908</code>、<code>eval_results/rdd_csbr_verify_*</code>

## 12. 完成度审计

| 项目 | 状态 |
|---|---|
| RDD 独有机制 | 执行路径完成；真实 checkpoint 表明直接 skip 的质量假设不成立 |
| 不破坏旧 checkpoint/API | 完成：所有新开关默认关闭；基线原路径可运行 |
| ≥10% denoiser | 仅完成 FastWan faux-RDD 速度上限；尚未达到真实 RDD 匹配画质 |
| ≥10% 当前 sample+VAE | 仅激进 faux-RDD 档达到 11–15%；不能作为最终质量结果 |
| 稳定计时 | 完成：单进程、warmup、交错顺序、median/raw/IQR/MAD |
| 多 prompt / 两分辨率 | 完成 |
| 视频展示 | 完成；明确标注零微调限制 |
| prior-art/novelty 审计 | 完成；采用限定措辞 |
| 论文级匹配画质 | CSRA pilot 的部分子集扰动较小；四层 3K→5K→10K 同配置续训仍有明显面部损坏，未通过质量门槛；尚未完成视频 adapter、VBench/FVD |

## 12. Residual 假设小样本验证（2026-09-09）

已完成 2 prompts × 2 seeds、30 层逐层测量，以及 RDD cross-scale reuse 对 full-resolution previous-timestep reuse 的同跳层预算比较。详细公式、逐层结果、原始数据和六个对比视频见：

- [CSBR residual 假设四组验证](RDD_CSBR_RESIDUAL_HYPOTHESIS_PILOT_ZH.html)
- `eval_results/rdd_csbr_residual_hypotheses_20260909/aggregate_metrics.json`

最重要的结论不是“验证成功”，而是把故事边界测清楚了：

| 结论 | 证据 | 判断 |
|---|---|---|
| middle residual 携带 fine low-frequency update | RDD noise-filling：raw cosine 0.529，low cosine 0.613；wrong-layer 0.003 | **部分支持**，而且是同层特异的 |
| 8→16 也适合整 block reuse | coarse→middle low cosine 只有 0.331 | **不支持**；暂时只做 16→32 |
| RDD residual 比 full-resolution timestep cache 更稳定 | common-innovation：RDD low cosine 0.636；full-res low cosine 0.860 | **当前否定** |
| 相同跳层数时 RDD-CSBR 扰动更小 | skip-6 latent RRMSE：RDD 0.416；full-res 0.226 | **当前否定** |
| 只需恢复 fine high-frequency 即可 | RDD 最优 scalar 后 low-band RRMSE 仍约 0.74 | **否定**；low-band 本身也需蒸馏 |

这意味着论文不能直接写“coarse residual 天然更容易预测”。更准确的表述是：RDD 提供了一个在首次 fine NFE 之前已经存在的、计算上免费的跨尺度 cache；它与 fine low-frequency residual 有中等且 layer-specific 的相关性，但需要 layer selection、per-layer/timestep scale 与 residual distillation 才能转化为质量可接受的跳块加速。正式实验必须加入 full-resolution residual cache 作为强基线。

## 13. 真实 RDD checkpoint 的完整后续验证（2026-09-11）

后续已补齐：统一 BF16 forward 公平性修复、两个真实 dense reverse trajectories、按 frequency/layer/timestep/refresh gap 的分析、三个实际 rollout seeds、RDD 与原始 DiT 的七次交错稳定计时。

最终结论比上面的 pilot 更严格：

- time-conditioned affine 可稳定降低直接 cross-scale 替换误差约 18–21%，但连续替换仍会破坏图像；
- RDD delta transport 只在较长 refresh gap 的低频 residual 上稳定有利，对高频无帮助；
- 混合 6 层的 period-5 rollout 不稳定；收缩到真实轨迹最稳定的 `layer 1/2/3` 后，RDD delta transport 在 3 seeds、period 5/8 上均比同预算 generic cache 低 15–28% latent error，但七次稳定计时的总速度仍只有约 2–2.5%；
- 原始 DiT previous-timestep cache 是强基线，七次稳定计时可减少 5.5–8.2% denoiser 延迟且扰动明显更低；
- training-free CSBR 不再继续手调，下一步是冻结 backbone 的轻量 CSRA adapter pilot。

完整数字、公式、图像、原始结果目录和训练 Go/No-Go 门槛见：

- [真实 RDD checkpoint：跨尺度 block reuse 的最终验证](RDD_REAL_CHECKPOINT_BLOCK_REUSE_VALIDATION_ZH.html)

## 14. CSRA 训练与真实 2D rollout（2026-09-11）

### 14.1 实验设置

| 项目 | RDD 主实验 | 普通 DiT 对照 |
|---|---|---|
| backbone | `RDD-Unified-L/2 @ 400K`，训练边界 300/300/400 | 第三方 DiT-L/2；只作 preliminary，不作论文正式数值 |
| backbone 参数 | 冻结 | 冻结 |
| adapter | rank 64；layers 1/2/3/14/17/20；2.14M trainable params | 同结构、同层、同参数量 |
| batch | 4 GPUs × 64 = 256；无梯度累计 | 4 GPUs × 64 = 256；无梯度累计 |
| 训练 | 5K steps；ImageNet 原版数据 | 5K steps；ImageNet 原版数据 |
| source | 最后一个 16×16 NFE 的真实同层 residual | 上一个 full-resolution NFE 的同层 residual |
| 使用区间 | 整个 32×32 stage | 仅匹配的全局 `t≤299` 区间 |

这里的 CSRA 不是直接复制 coarse residual。它同时读取上采样后的 coarse residual、当前 fine hidden 的高频部分和当前 timestep，再预测应该跳过的 fine block residual。被替换的原 block 不执行。

### 14.2 三个 seed 的端到端结果

下表是同 seed dense rollout 的 latent relative L2；越低越接近原模型。每个 seed 4 张图，共 12 张。

| RDD 配置 | seed 1 | seed 2 | seed 3 | 平均 | 判断 |
|---|---:|---:|---:|---:|---|
| raw coarse，6 层 | 0.482 | 0.481 | 0.379 | **0.447** | 不可作为最终方法 |
| CSRA early3 | 0.390 | 0.332 | 0.279 | **0.334** | 会破坏早期语义计算 |
| CSRA late3 | **0.244** | **0.313** | **0.207** | **0.255** | 当前最稳候选 |
| CSRA all6 | 0.421 | 0.392 | 0.298 | **0.370** | teacher-forcing 分布偏移明显 |

<div class="rdd-grid">
<article class="rdd-card"><img style="width:100%" src="../../eval_results/csra_rollout_rdd_l_5k/comparison_grid.png"><h4>RDD seed 20260911</h4><p>从左到右：dense、raw-all6、CSRA early3、late3、all6。</p></article>
<article class="rdd-card"><img style="width:100%" src="../../eval_results/csra_rollout_rdd_l_5k_seed_20260912/comparison_grid.png"><h4>RDD seed 20260912</h4><p>late3 保持主体与局部纹理最稳定。</p></article>
<article class="rdd-card"><img style="width:100%" src="../../eval_results/csra_rollout_rdd_l_5k_seed_20260913/comparison_grid.png"><h4>RDD seed 20260913</h4><p>同样出现 late3 最优、raw 与 early3 更易漂移的 pattern。</p></article>
</div>

普通 DiT 对照也给出了有用的边界：无 refresh 的 recurrent all6 直接崩坏，latent RRMSE 为 1.285；late3 为 0.292；每 4 次 NFE 做一次 dense refresh 可降至 0.161。RDD 的 all6 虽然仍不够好，但误差只有 0.421，没有出现 DiT recurrent all6 的彻底崩坏。这支持的是一个有限而准确的结论：**RDD 已完成的 coarse stage 提供了非递归、不会逐 NFE 自我污染的 cache；它不是天然精确，但比无限递归旧 full-resolution prediction 更稳定。**

<div class="rdd-grid">
<article class="rdd-card"><img style="width:100%" src="../../eval_results/csra_rollout_dit_l_5k/comparison_grid.png"><h4>DiT-L preliminary control</h4><p>无 refresh 的 early/all 递归预测崩坏；late3 和 period-4 refresh 才稳定。第三方 checkpoint 来源不完整，因此只用于机制诊断。</p></article>
</div>

### 14.3 2026-09-11 视频速度复验

在 `1280×768×29 帧`、FastWan DMD 3-NFE、RDD Spatial 4→2→1、在线 VSA-20、noise filling 下，两个新 prompt 的 9 轮交错计时复现了旧结论：

| 配置 | denoiser 延迟下降 | denoiser 加速 | 含当前 VAE 的端到端延迟下降 |
|---|---:|---:|---:|
| CSBR-6 | 15.3–15.6% | 1.18–1.19× | 3.6–3.9% |
| CSBR-12 | 30.1–30.2% | 1.43× | 约 7.4% |
| CSBR-18 | 45.0% | 1.82× | 约 10.7% |
| CSBR-24 | 59.6% | 2.47× | 约 14.4% |

这仍是“跳层速度上限”，不是训练后视频画质结论。当前 Wan VAE 在该输出尺寸约耗时 8.3 秒，因而严重稀释 denoiser 加速。视频版 CSRA 会增加一个很小的 adapter 成本，所以正式目标应定为：late/safe 6–12 层经过 rollout-aware distillation 后，保持约 **12–30% denoiser 延迟下降**；要让当前 VAE 下的端到端下降超过 10%，通常需要稳定跳过约 18 层或额外优化 VAE。

### 14.4 两轮 on-policy 六层验证：尚未通过质量门槛

共同设置：冻结同一个 `RDD-Unified-L/2 @ 400K`，原生 ImageNet、rank 64、每卡 64 / 全局 batch 256、AdamW LR `2e-4`、3K adapter updates；采样为 DDPM `80/80/90`、CFG 1.0、noise filling。固定 seeds 为 `20260921/20260922/20260923`，每个 seed 四张图。所有层编号从 0 开始。

这里的 **on-policy 只指层内状态**：在一轮 fine forward 中，上游被替换层使用当前 adapter，后续层的监督目标在这个已改变的 hidden 上重新计算。训练输入仍来自加噪的 ImageNet latent，不是完整反向采样轨迹；尚未优化最终 epsilon 或多步 rollout 损失。因此不能认为整个训练与推理的分布差异已经解决。

| 实验 / 层集合 | 使用的层 | seed 20260921 | seed 20260922 | seed 20260923 | 平均 latent relative L2 |
|---|---|---:|---:|---:|---:|
| late6 / 直接复制 | 12,14,16,18,20,22 | 0.438 | 0.422 | 0.384 | 0.415 |
| late6 / adapter 前三层 | 12,14,16 | 0.281 | 0.291 | 0.222 | 0.264 |
| late6 / adapter 后三层 | 18,20,22 | 0.287 | 0.284 | 0.246 | 0.272 |
| late6 / adapter 全六层 | 12,14,16,18,20,22 | 0.338 | 0.327 | 0.301 | 0.322 |
| safe6 / 直接复制 | 8,11,13,15,17,20 | 0.378 | 0.344 | 0.315 | 0.346 |
| safe6 / adapter 前三层 | 8,11,13 | 0.284 | 0.295 | 0.223 | 0.268 |
| safe6 / adapter 后三层 | 15,17,20 | 0.233 | 0.248 | 0.186 | 0.223 |
| safe6 / adapter 全六层 | 8,11,13,15,17,20 | 0.328 | 0.312 | 0.267 | 0.302 |

`safe6` 是候选的实验名，不代表已经安全。它的后三层子集相对更稳，但人脸仍出现明显破坏；全六层仍不合格。两个实验都比各自同层的直接复制误差小；它们与 14.2 的旧实验使用不同层集合、seed 和训练时长，不能把跨表均值下降全部归因于 on-policy 训练。上述 relative L2 测的是对 dense 输出的偏离，不是 FID，也不能单独证明生成质量改善。

六层 adapter 的 2D 延迟下降范围为 late6 `0.56–3.29%`、safe6 `0.78–3.26%`；只有两轮配对计时，而且与既有训练共享 GPU，这些数值不足以作为稳定加速结论。

以下每张图从左到右都是：dense、直接复制六层、adapter 前三层、adapter 后三层、adapter 全六层。应同时检查失败的人脸案例与较稳定的车辆、蝴蝶、蜗牛，不能只选好看的图片。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_late6_3k/seed_20260921/comparison_grid.png" style="width:100%"><h4>late6：20260921</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_late6_3k/seed_20260922/comparison_grid.png" style="width:100%"><h4>late6：20260922</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_late6_3k/seed_20260923/comparison_grid.png" style="width:100%"><h4>late6：20260923</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_safe6_3k/seed_20260921/comparison_grid.png" style="width:100%"><h4>safe6：20260921</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_safe6_3k/seed_20260922/comparison_grid.png" style="width:100%"><h4>safe6：20260922</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_safe6_3k/seed_20260923/comparison_grid.png" style="width:100%"><h4>safe6：20260923</h4></article>
</div>

当前四层候选为 `13,15,17,20`，单独联合训练到 3K；参数 1.43M，仍为每卡 64 / 全局 256。它是否更稳由后续三 seed 验证决定，不能预先称为“最安全”。把视频六层的 15.3–15.6% 收益线性按四层折算，只得到 **约 10.2–10.4% 的粗略跳层参照**，还没有扣除 adapter、搬运和缓存成本，也没有验证 DiT-L 层选择在 Wan 中的对应关系。视频净收益必须实测，含 VAE 的收益另报。

原始数据：`eval_results/csra_onpolicy_late6_3k/seed_*/metrics.json` 和 `eval_results/csra_onpolicy_safe6_3k/seed_*/metrics.json`；每个目录都保存全五组图片。本节完成时 `fine_context_v2` 尚未部署；9 月 12 日已开始独立训练，见 14.11，不属于本节已完成结果。

四层训练已保存 `RDD-CSRA-OnPolicy-Unified-L-400K-r64-subset4/0003000.pt`，末段 loss 约 `1.01918`，`rel_full=0.45498`；这里的训练 `rel_full` 是 relative MSE，与上表 rollout 的 relative L2 并非同一指标。后续结果目录为 `eval_results/csra_onpolicy_subset4_3k`。旧评估脚本的名字需要按实际层列表解释：`csra_all6` 在本轮实际只替换四层；`csra_early3` 是 `13,15,17`；`csra_late3` 实际仅替换 `20` 一层。不能把后一列误报成三层效果。

### 14.5 四层 3K 验证与增加训练量的对照

四层模型的三组采样均已完成。下表使用与 14.4 相同的 seeds、采样设置、每 seed 四张图；数值仍是相对各自 dense 输出的 latent relative L2。

| 方法 | 实际替换层（0-based） | 20260921 | 20260922 | 20260923 | 平均 |
|---|---|---:|---:|---:|---:|
| 直接复制四层 | 13,15,17,20 | 0.3393 | 0.3071 | 0.2831 | 0.3099 |
| adapter 前三层 | 13,15,17 | 0.2369 | 0.2392 | 0.1890 | 0.2217 |
| adapter 最后一层 | 20 | 0.1807 | 0.1928 | 0.1332 | 0.1689 |
| adapter 全四层 | 13,15,17,20 | 0.2868 | 0.2766 | 0.2245 | 0.2627 |

四层的扰动小于此前六层，但删除了部分替换操作，本就更容易保持画质；不能据此宣称 adapter 本身更强。视觉上，蜗牛、蝴蝶和部分车辆保留较好，但 `20260921` 的白帽人物面部仍明显损坏，另一人物及局部纹理也有偏移。**全四层仍未通过质量门槛。** 只替换 layer 20 的列较接近 dense，但不能用一层的质量配上四层或六层的速度进行汇报。

两轮共享 GPU 计时下，全四层 2D 延迟仅下降 `0.80% / 0.26% / 0.55%`。这一小样本没有证明稳定加速，视频的 adapter 成本仍待实测。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_3k/seed_20260921/comparison_grid.png" style="width:100%"><h4>四层 3K：20260921</h4><p>五列依次为 dense、raw 四层、adapter 三层、adapter 单层、adapter 四层。图内旧名称 all6/late3 以这里的实际层数为准；白帽人脸是明确失败案例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_3k/seed_20260922/comparison_grid.png" style="width:100%"><h4>四层 3K：20260922</h4><p>蝴蝶等主体仍可辨认，细节已有变化；此列不能代替失败人物案例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_3k/seed_20260923/comparison_grid.png" style="width:100%"><h4>四层 3K：20260923</h4><p>单层比四层更接近 dense；需要在相同实际替换层数下比较速度和质量。</p></article>
</div>

下一轮固定 `13,15,17,20`、rank64、同一个 RDD teacher、相同 LR/全局 batch，恢复 3K 的 adapter 与 optimizer，增加至累计 10K updates，在 5K 和 10K checkpoint 复核同一组对照。原始 3K checkpoint 与结果保留；续训单独保存于 `RDD-CSRA-OnPolicy-Unified-L-400K-r64-subset4-continue10k`。这是训练量对照，尚无改善结论。当前 trainer 没有保存 dataloader 游标和随机数状态，恢复时会重新启动数据遍历，因此属于恢复权重/优化器后的续训，不是逐 bit 连续运行的复现。

仍需排查的限制包括：adapter 没有直接读取当前 fine hidden 的低频部分；训练输入是加噪真实图像而非模型生成轨迹；没有对最终 epsilon 或多步 rollout 做一致性监督。这些是代码审查提出的潜在原因，目前没有单变量消融证明哪一项主导。若续训不能改善，就需要改输入或训练目标，不能无限重复相同的 3K 选层搜索。

### 14.6 固定四层，从 3K 续训到 5K：改善很小

本次固定 backbone、adapter 架构 `legacy_v1`、rank64、替换层 `13,15,17,20`、LR、batch 和采样配置。读取累计 `0005000.pt`，仍使用 `20260921/20260922/20260923` 三个 seed，每 seed 四张图。12 张 dense 基线 PNG 的 SHA-256 与 3K 评估逐张一致；三组 raw 四层的 relative L2 也完全一致。因而下表没有把不同 seed 或不同基线混在一起。

| 实际替换层 | 20260921：3K → 5K | 20260922：3K → 5K | 20260923：3K → 5K | 三 seed 平均：3K → 5K |
|---|---:|---:|---:|---:|
| raw 四层：13,15,17,20 | 0.339317 → 0.339317 | 0.307142 → 0.307142 | 0.283105 → 0.283105 | 0.309855 → 0.309855 |
| adapter 三层：13,15,17 | 0.236858 → 0.236001 | 0.239171 → 0.240022 | 0.188980 → 0.188446 | 0.221670 → 0.221490 |
| adapter 单层：20 | 0.180661 → 0.177777 | 0.192844 → 0.190594 | 0.133206 → 0.132636 | 0.168904 → 0.167002 |
| adapter 四层：13,15,17,20 | **0.286822 → 0.284896** | **0.276612 → 0.275562** | **0.224533 → 0.224012** | **0.262656 → 0.261490** |

全四层均值只相对下降约 **0.44%**，没有出现质变。逐张查看全部三个 grid：`20260921` 的白帽人物仍有严重面部伪影，`20260923` 的坐姿人物也有明显面部与姿态偏移；蝴蝶、蜗牛等图片相对稳定。**5K 仍未通过质量门槛。** 单层更接近 dense 不能代替四层结果；relative L2 不是 FID，也不代表与真实数据的质量差距。

本轮和 adapter 续训共享 GPU。全四层的两轮配对计时仅显示 `0.78% / 1.06% / 1.14%` 延迟下降，dense 每批约 24 秒；不能与其他 GPU 负载下的旧绝对秒数比较，也不能据此宣称稳定加速。这轮主要回答训练量是否改善输出，而非最终速度。

下面五列依次是：dense、raw 四层、adapter 三层、adapter **单层**、adapter 四层。图内历史名称 `all6`、`late3` 不代表本轮实际层数。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_5k/seed_20260921/comparison_grid.png" style="width:100%"><h4>四层 5K：20260921</h4><p>继续训练后，白帽人物的四层替换仍然失败；不能只展示第一行的动物。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_5k/seed_20260922/comparison_grid.png" style="width:100%"><h4>四层 5K：20260922</h4><p>蝴蝶主体保持较好，其他纹理仍改变；三层子集的数值还略差于其 3K 版本。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_5k/seed_20260923/comparison_grid.png" style="width:100%"><h4>四层 5K：20260923</h4><p>蜗牛较稳定；坐姿人物仍有明显损坏。不能用较好的单层列替代四层的质量判断。</p></article>
</div>

完整五组图片和原始数值保存在 `eval_results/csra_onpolicy_subset4_5k/seed_*/`。10K 仍沿用这三个 seed 与相同层集合，不再引入新选层来混淆训练量的效果。

### 14.7 一个明确的输入限制：当前 adapter 看不到哪部分变化？

代码中的 `low_high_tokens` 把每个通道的 hidden 按 **2×2 token 格子取平均再放大回来**，记为 $P h$；剩余细节是 $(I-P)h$。这只是 block-average 意义的低/高频分解，不是理想傅里叶频率截断。

V1 预测的 block 修改量可以写成：

$$
\widehat{\Delta}(c,h,t)=A(c,t)+B((I-P)h,t),
$$

其中 $c$ 是上采样的 coarse residual，$h$ 是当前 fine hidden。**当前 fine hidden 的平均部分 $Ph$ 没有直接输入 adapter**；它与旧 coarse residual 的相关性只能间接提供信息。

一个通道的 2×2 小格若是 `[1,3;5,7]`，均值为 4，剩余细节是 `[-3,-1;1,3]`。把四个数都加 10 后，均值变为 14，但细节仍是 `[-3,-1;1,3]`。固定同一个 coarse cache 与 timestep，V1 就会预测相同的修改量。

更一般地，对任意在每个 2×2 格子内为常数的变化 $u$（不同格子、不同通道可取不同值），有 $Pu=u$，所以：

$$
\widehat{\Delta}(c,h+u,t)=\widehat{\Delta}(c,h,t).
$$

这是当前输入设计导致的不变性，在精确算术下成立；增大 rank 或训练更久不会消除它。**不是说输出 hidden 的低频被删除了**：残差连接仍执行 $h+\widehat{\Delta}$，原来的 $h$ 保留着。问题是预测的“修改量”不能直接响应当前低频的变化，而完整 teacher block 一般没有这种限制。

这给出了一个可检验的改进方向：保留 coarse cache，同时给 adapter 补充当前 fine hidden 的轻量点式分支，然后在同层数、同训练量下做消融，检查人物损坏是否改善，并实测新增分支成本。它尚不能证明低频缺失主导了所有失败——真实生成轨迹上 $c$ 与 $Ph$ 可能高度相关，训练/采样分布差异和最终预测监督仍是其他候选原因。

实施状态：此前 `fine_context_v2` 的上传曾被自动权限审查拒绝。**9 月 12 日用户明确确认后，上传已成功，测试通过，开始匹配训练。** 原 V1 的 10K 对照完整保留；新分支是否有效仍待生成结果验证。

### 14.8 10K 对照完成：原样加训练量没有解决问题

训练正常保存 `RDD-CSRA-OnPolicy-Unified-L-400K-r64-subset4-continue10k/0010000.pt`，三个指定 seed 的评估均成功结束；完整结果已下载到 `eval_results/csra_onpolicy_subset4_10k`。与 3K 的 12 张 dense 基线 PNG 再次逐张核对 SHA-256，全部相同；raw 四层数值也完全相同。

| adapter 全四层：13,15,17,20 | 3K | 5K | 10K |
|---|---:|---:|---:|
| seed 20260921 | 0.286822 | 0.284896 | 0.283439 |
| seed 20260922 | 0.276612 | 0.275562 | 0.275934 |
| seed 20260923 | 0.224533 | 0.224012 | 0.221967 |
| **平均 latent relative L2** | **0.262656** | **0.261490** | **0.260447** |

从 3K 到 10K，平均误差相对下降 **0.84%**；从 5K 到 10K，只再下降 **0.40%**，其中 seed 20260922 还略有反向变化。该误差是与 dense 输出的差别，不是 FID。末段训练 relative MSE 从 3K 约 `0.45498` 降至 10K `0.44758`，也没有出现显著突破；训练指标与表内 relative L2 的定义不同，不能直接混用。

| 实际使用的替换层 | 3K 平均 | 5K 平均 | 10K 平均 |
|---|---:|---:|---:|
| raw 四层：13,15,17,20 | 0.309855 | 0.309855 | 0.309855 |
| adapter 三层：13,15,17 | 0.221670 | 0.221490 | 0.220072 |
| adapter 单层：20 | 0.168904 | 0.167002 | 0.167632 |
| adapter 四层：13,15,17,20 | 0.262656 | 0.261490 | 0.260447 |

已查看全部三张 10K 对比 grid：白帽人物、坐姿人物的四层替换仍有严重面部伪影，与 3K/5K 的失败 pattern 一致；蝴蝶、蜗牛等例子仍相对稳定。单层列较接近 dense，但仍不能满足“用足够多的层换取视频净加速”的目标。**本次结果不支持“原架构只是训练太少”的乐观解释**，但也不能凭这 12 张图证明所有更长训练都不可能改善，更不能据此否定经过重新设计和训练的跨尺度复用。

10K 的全四层 2D 延迟下降为 `1.36% / 0.86% / 0.80%`。其他训练仍共享 GPU，且仅两轮配对计时；不把它们当作稳定速度证据。5K 评估时本轮 adapter 训练也在运行，10K 评估时它已结束，dense 绝对耗时约从 24 秒降至 12.6–12.7 秒，**不是 checkpoint 训练得更久使模型变快**。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_10k/seed_20260921/comparison_grid.png" style="width:100%"><h4>四层 10K：20260921</h4><p>五列：dense、raw 四层、adapter 三层、adapter 单层、adapter 四层。白帽人物仍是失败案例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_10k/seed_20260922/comparison_grid.png" style="width:100%"><h4>四层 10K：20260922</h4><p>蝴蝶外形较稳定；本 seed 的四层误差略高于 5K，未出现持续显著改善。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_onpolicy_subset4_10k/seed_20260923/comparison_grid.png" style="width:100%"><h4>四层 10K：20260923</h4><p>蜗牛较稳定，但坐姿人物仍明显损坏；结论综合全部图片，不只挑成功例子。</p></article>
</div>

### 14.9 接下来改什么，以及当前执行边界

不再启动一轮相同架构的长训，也不再反复用短训换层集合。按下面顺序排查，每次只改变一个关键变量：

1. **核对训练/推理精度：已完成。** 远端确认训练显式禁用 adapter 的 autocast，而旧采样在 BF16 autocast 下调用它；输入 `.float()` 本身不会禁用 autocast。9 月 12 日对同一 checkpoint 做了三 seed 配对测试，影响极小，见 14.10。3K/5K/10K 共用了旧评估路径，以上训练量对照不因此变成不同精度之间的比较。
2. **补充当前 fine 的输入信息。** 先固定四层 `13,15,17,20`、rank64、teacher 和训练量，只比较 V1 与增加轻量当前-hidden 分支的版本。明确控制训练样本及加噪随机数。不能直接拿草案启动器默认的六层/rank128 与四层/rank64 比，再宣称提升来自 fine context。
3. **若层内拟合改善但生成仍坏，再改监督。** 在真实 reverse rollout 上收集 coarse cache 与 fine 状态，加入最终 epsilon 一致性，而不是只逐层拟合 residual。先做小规模、同预算对照；若有效再扩大训练。冻结 backbone 不等于禁止梯度经过 backbone，需要允许最终损失经未替换 block 回传到 adapter，同时保持原模型权重不更新。
4. **2D 质量过关后，实测视频净收益。** 视频实现必须把 adapter、cache 与上采样成本一起计时，和同一个 RDD+VSA 基线配对比较；同时另报 VAE-inclusive 延迟。现有 FastWan 只是 faux-RDD，不能代替真实 RDD 视频的训练后画质验证。

**执行状态更新：** 旧 10K 训练和评估均已结束。9 月 12 日用户明确确认上传和运行，针对本项目训练/采样源码的部署已成功；远端原文件先备份，仅同步七个相关代码/测试/启动器文件，没有上传私钥、token 或另传 ImageNet 数据集。原 checkpoint 和结果保留。新训练已启动，仍不宣称达到最终目标。

### 14.10 精度消融：不是主要原因

冻结同一个 V1 四层 10K checkpoint，只改变 adapter 运算精度；DiT backbone 仍为 BF16，adapter 输出仍转回 hidden 的 dtype。新开关 `--adapter-precision fp32` 只在 adapter 内禁用 autocast，默认 `legacy_autocast` 保留旧行为。两种设置均为 DDPM `80/80/90`、CFG1.0、noise filling、每 seed 四张图。

| 全四层：13,15,17,20 | 旧 BF16 autocast | adapter FP32 |
|---|---:|---:|
| 20260921 | 0.283439 | 0.283437 |
| 20260922 | 0.275934 | 0.275923 |
| 20260923 | 0.221967 | 0.221947 |
| **平均 latent relative L2** | **0.260447** | **0.260436** |

平均仅相对下降约 **0.0043%**。12 张 dense PNG 的 SHA-256 与旧 10K 对照全部一致，raw 四层误差也完全相同。全部三张 grid 均已查看：白帽人物的面部、坐姿人物仍明显损坏。**FP32 没有解决当前问题**；它只能消除后续实验中这项已知精度差异。共享 GPU 上的两轮计时不作为速度结论。

下面五列已经按真实层数标注：dense、raw 四层、adapter 三层 `13/15/17`、adapter 单层 `20`、adapter 四层。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_subset4_10k_fp32_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>10K / adapter FP32：20260921</h4><p>第三行白帽人物仍是明确失败案例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_subset4_10k_fp32_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>10K / adapter FP32：20260922</h4><p>蝴蝶较稳定，与旧精度结果没有明显变化。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_subset4_10k_fp32_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>10K / adapter FP32：20260923</h4><p>坐姿人物仍损坏；不能据蜗牛一张图判断质量过关。</p></article>
</div>

原始数据：`eval_results/csra_subset4_10k_fp32_20260912/seed_*/metrics.json`，包含五种设置的实际层号、精度、图片和计时。

### 14.11 补充 fine context 的匹配对照：两组训练与最终采样均完成

人话解释：旧 adapter 看得到 coarse 的修改，以及当前 fine 格子内部的差异，却不能直接读取格子的平均状态。V2 保留这些输入，**再加一条读取当前完整 fine hidden 的轻量分支**；不是重算被跳过的完整 Transformer block。

$$
\widehat{\Delta}_{V2}(c,h,t)=\widehat{\Delta}_{V1}(c,h,t)+C(h,t).
$$

`C` 是无仿射 LayerNorm → 降到 rank64 → timestep 调制与 SiLU → 升回 hidden dimension。新分支的最后投影初始化为零，所以加载同一个 V1 checkpoint 时，V2 的初始输出与 V1 相同。它是轻量近似，有额外成本，不能预先认定保留了视频净加速。

| 项目 | V2 实验 | V1 匹配对照 |
|---|---|---|
| architecture | `fine_context_v2` | `legacy_v1` |
| 当前 fine 输入 | 原有高频分支 + 新增完整 hidden 分支 | 仅原有高频分支 |
| backbone | 同一个 RDD-Unified-L/2 @400K，300/300/400，冻结 | 同左 |
| 初始化 | 同一个四层 V1 的 10K 权重；新分支零输出 | 同一个四层 V1 的 10K 权重 |
| optimizer / 计数 | 重建 AdamW；重新从 0 计追加步数 | 同左 |
| 追加训练量 | 3000 updates，1000/2000/3000 保存 | 同左 |
| 层 / rank | `13,15,17,20`（0-based）/ 64 | 同左 |
| GPU / batch | 0–3；每卡64，全局256，不做梯度累积 | 同左 |
| LR / weight decay / clip | `2e-4 / 0.01 / 1.0` | 同左 |
| seed | 20260912；构造模型后重新设置数据/噪声随机流 | 同左 |
| 监督 | `on_policy_layerwise`，仍非最终 epsilon 或完整反向轨迹监督 | 同左 |
| 评估 | 相同三个 seed、DDPM80/80/90、CFG1.0、noise filling、adapter FP32 | 同左 |

两组顺序运行，避免把更多并行任务挤到现有训练上。新目录中的 `0003000.pt` 表示“10K 权重初始化后追加 3K”，不是从头仅训练 3K；也不是恢复旧 optimizer 的连续 13K 训练。相同数据流起点使输入/训练随机性尽量受控，但架构、可训练参数量和实际运行耗时不同，结果应作为**同更新数/同样本预算**对照，不冒充同 FLOPs 对照。

**已验证（9 月 12 日，V1 最终 checkpoint 已保存）：** 六项 CPU 单元测试通过，包括 V1→V2 初始化输出一致、缺失旧权重/层集合不符会拒绝、FP32 精度隔离、新分支可学习；四 GPU 每卡64 的两步测试正常保存并 exit0。正式 V2 与 V1 对照都完成追加 3000 updates 并保存，末段训练 `rel_full` 分别为 `0.38690 / 0.44595`（relative MSE），峰值 allocated 分别为 `9.90 / 9.89 GiB/GPU`。整个顺序训练启动器正常完成，无 OOM，未停止其他任务。训练指标不能代替生成质量：两组后续层的输入状态会因 adapter 不同而改变，损失下降也不等于固定输入 validation 误差同比下降。

远端根目录：`/scr/yuheng/DevOpsBench/multiscale_diffusion`。

| 类型 | 相对该根目录的位置 |
|---|---|
| tmux | `rdd_csra_context_pair_0123_20260912` |
| 启动器 | `2d/exp/train_rdd_csra_context_pair_4gpu.sh` |
| 合并日志 | `logs/train_csra_context_pair_20260912.log` |
| V2 日志 | `logs/train_csra_context_v2_20260912.log` |
| V1 对照日志 | `logs/train_csra_context_v1control_20260912.log` |
| V2 checkpoint | `checkpoints/RDD-CSRA-context-v2-L400K-r64-subset4-20260912/` |
| V1 对照 checkpoint | `checkpoints/RDD-CSRA-context-v1control-L400K-r64-subset4-20260912/` |

**V2 追加 1K、3K 的三 seed 生成均已完成并逐图检查：** 下表仍是与 dense 的 latent relative L2，实际替换四层。旧 V1 10K 是初始化参照，尚不是同额外训练量的架构对照。

| 全四层：13,15,17,20 | 初始化 V1 10K，FP32 | V2 再训练 1K，FP32 | V2 再训练 3K，FP32 |
|---|---:|---:|---:|
| seed 20260921 | 0.283437 | 0.272938 | 0.271286 |
| seed 20260922 | 0.275923 | 0.269032 | 0.271094 |
| seed 20260923 | 0.221947 | 0.210440 | 0.206600 |
| **平均** | **0.260436** | **0.250803** | **0.249660** |

均值相对下降约 **3.70%**，三个 seed 均有小幅改善；12 张 dense PNG 的 SHA-256 与初始化参照全部一致，raw 四层数值也一致。但白帽人物和坐姿人物的面部仍然损坏，**尚未通过质量门槛**。只替换 layer20 的误差降幅更明显，不能拿单层效果代表四层；前三层子集则在两个 seed 上略变差，说明新增分支不是每种组合都改善。当前证据最多说明“值得继续做匹配对照”，不能单凭此表把改善归因于完整 fine context，因为它还增加了训练量。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_1k_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>V2 追加 1K：20260921</h4><p>五列为 dense、raw 四层、adapter 三层、adapter 单层、adapter 四层。第三行面部仍明显损坏。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_1k_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>V2 追加 1K：20260922</h4><p>蝴蝶相对稳定，但全四层仅小幅接近 dense，尚无质变。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_1k_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>V2 追加 1K：20260923</h4><p>坐姿人物在四层列仍明显变形；不能只展示较稳定的蜗牛。</p></article>
</div>

**3K 复核：** 全四层均值相对初始化下降约 **4.14%**，但比 V2 1K 仅再下降 **0.46%**；seed20260922 反而比1K略差。全部三组3K图已检查，人物脸部仍明显受损，未出现质变。只替换 layer20 时，三 seed 误差为 `0.144342/0.175287/0.111396`，改善比全四层明显；这说明“当前输入可能有帮助”与“联合替换足够多层已可用”是两件事，不能混写。仍需等同预算 V1 对照完成后再归因。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_3k_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>V2 追加 3K：20260921</h4><p>白帽人物的四层结果仍有严重伪影；单层列较好不能替代四层证据。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_3k_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>V2 追加 3K：20260922</h4><p>与1K相比，本 seed 的全四层误差略增；不支持“训练继续就一定改善”。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v2_3k_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>V2 追加 3K：20260923</h4><p>误差有所下降，但坐姿人物仍损坏，尚未达到质量要求。</p></article>
</div>

完整结果在 `eval_results/csra_context_v2_1k_20260912/seed_*/` 和 `eval_results/csra_context_v2_3k_20260912/seed_*/`。两次评估的12张 dense PNG 均与初始化参照逐张 SHA-256 相同，raw 四层数值完全一致。评估与其他训练共享 GPU，全四层的两轮计时，1K 为约 `0.96% / 1.04% / 1.27%`、3K 为 `0.71% / 0.95% / 1.13%` 延迟下降，不能作为稳定速度证据。V1 对照最终三 seed 采样也已正常完成、下载并检查，目录 `eval_results/csra_context_v1control_3k_20260912/seed_*/`。

最终同预算比较见下一节。层内拟合改善但图仍坏，下一项是轨迹/最终预测监督，不是把当前训练 loss 下降当作完成。最终仍需真实的 adapter 视频计时和质量边界验证。

### 14.12 V2 / V1 同预算最终结果：小幅有效，但不够用

两者从同一个10K adapter初始化，各追加3K、重新建立optimizer，保持样本预算、seed、rank64和层集合一致；都用FP32 adapter评估。V2增加参数和每步计算，因此不是同FLOPs对照。全部12张 dense PNG 在两组间逐张 SHA-256 相同，raw四层误差也一致。

| 全四层：13,15,17,20 | V1 对照，再训3K | V2 完整fine输入，再训3K |
|---|---:|---:|
| 20260921 | 0.284427 | 0.271286 |
| 20260922 | 0.275839 | 0.271094 |
| 20260923 | 0.222581 | 0.206600 |
| **平均 latent relative L2** | **0.260949** | **0.249660** |

三个seed均改善，平均相对下降约 **4.33%**。这比单纯比较V2与初始化更能支持“新增fine输入有帮助”，但新增分支也增加容量，尚未排除容量因素。V1继续训练没有带来类似改善；V2人物仍然严重损坏，**不能把这轮写成画质过关或已完成方法**。

分层子集揭示了限制：仅用layer20时，V1/V2平均误差约 `0.16756 / 0.14367`；用前三层13/15/17时平均约 `0.22064 / 0.21968`，而且两个seed略退步。收益并不是各层均匀出现；不能用单层质量配合四层跳过量汇报。这里是一个训练seed、三个采样seed的小规模诊断，不是统计充分的FID实验。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_context_v1control_3k_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>V1 同预算对照：20260921</h4><p>与上一节V2相同的五列与seed，白帽人物仍坏；V2也未修复该失败模式。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v1control_3k_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>V1 同预算对照：20260922</h4><p>蝴蝶仍较稳定；不能只用这类主体评价方法。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_context_v1control_3k_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>V1 同预算对照：20260923</h4><p>坐姿人物明显损坏。新监督实验继续保留这个失败案例作检查。</p></article>
</div>

### 14.13 真实生成轨迹 + 最终网络预测监督：有改善，但仍有失败案例

之前训练类似于“每个零件分别做得像”，现在检验“组装后整台机器的输出是否像”。仍然只训练原来四个轻量adapter，不更新RDD的Transformer和VAE权重，不改变noise filling、DDPM80/80/90或CFG1.0。

1. 用原RDD正常从8→16→32采样，在16×16、t=300保存四层的真实residual，保持原始dtype和原始64-token形状。
2. 在32×32阶段的第 `0/12/25/38/51/64/77/89` 次网络调用，保存当前latent、真实t、类别和dense网络最终8通道输出。不是拿真实图片人为加噪来近似这对coarse/fine状态。
3. 训练时在这些fine输入上联合替换四层。误差从**最终输出头**穿过后续未替换层，再传回所有adapter；冻结参数不等于切断对输入的梯度。旧可视化函数带有 `inference_mode`，所以新增独立可微路径，不改旧采样函数。
4. 将最终8通道分开：前4通道为当前模型的去噪预测，后4通道为learned-variance预测。pilot损失为前者MSE加 `0.05 × 后者MSE`，这个权重是试验设置，不是理论最优值；不能把8通道统称为epsilon。

| 项目 | 当前pilot配置 |
|---|---|
| frozen teacher | 原生ImageNet的 `RDD-Unified-L/2 @400K`，300/300/400 |
| adapter初始化 | V2新增fine分支的追加3K checkpoint |
| 训练/验证轨迹 | 512 / 32 条，seed不重合，且不使用三个展示seed |
| 保存fine状态 | 每条8个，共4096 / 256个训练/验证状态 |
| 数据来源 | **dense teacher真实采样轨迹**，不是student自己生成的on-policy轨迹 |
| 训练对象 | 13,15,17,20，rank64，V2轻量adapter；backbone冻结 |
| batch / GPU | 全局256，每卡64，GPU0–3，暂不做梯度累积 |
| optimizer | AdamW，LR1e-4，weight decay0.01，clip1.0 |
| pilot预算 | 先2步真实模型测试，再400步；每50步看独立validation，每100步存checkpoint |
| 数值/显存 | adapter FP32、backbone BF16，后续冻结block做activation checkpointing |
| 模型选择 | `best.pt`按独立validation的主预测MSE选择；还必须检查真实生成，不只看该MSE |

**已完成检查：** 采集器小测试与四个正式分片都通过“最终latent最大差异=0”的完整采样一致性测试；数据采集已完成。四项CPU梯度测试通过：前后adapter均有梯度、teacher无梯度、activation checkpointing不改变测试输出/梯度、两种输出损失正确分开。真实四卡每卡64的两步训练测试也正常完成：初始训练前向与旧采样前向的最大差异为0，所有adapter有梯度，teacher无梯度，无OOM。以下只是验证集预测误差，不是生成FID或画质评分。

| 独立validation，256个状态 | 初始化 | 1步测试后 | 2步测试后 |
|---|---:|---:|---:|
| 去噪主预测MSE | 0.0138915 | 0.0135440 | 0.0135076 |
| variance输出MSE | 0.110174 | 0.0387801 | 0.0364669 |

400步pilot从**原V2追加3K checkpoint**重新初始化、使用新optimizer，未接着两步smoke训练；现已正常完成，独立validation选出的best恰好是400步。训练循环耗时约167秒（不含加载、初始化和初始validation），峰值allocated约3.67GiB/GPU；没有OOM，没有暂停其他任务。训练使用预先保存的状态，不需每步重新编码ImageNet，不能把这个耗时直接和之前含编码的训练比较。

| 独立validation，256个状态 | 初始化 | 100步 | 400步 / best |
|---|---:|---:|---:|
| 去噪主预测MSE | 0.0138915 | 0.00913119 | 0.00846061 |
| variance输出MSE | 0.110174 | 0.00914492 | 0.00702651 |

**实际生成，比loss更重要：** 三个seed各4张，DDPM80/80/90、CFG1.0、noise filling，adapter FP32；全部四层13/15/17/20同时替换。两次生成的12张dense PNG均与V2初始化参照逐张SHA-256相同。下表是latent relative L2，**不是FID、不是画质提升百分比**。

| 采样seed | V2初始化 | 最终输出监督100步 | 最终输出监督400步 |
|---|---:|---:|---:|
| 20260921 | 0.271286 | 0.211698 | 0.208228 |
| 20260922 | 0.271094 | 0.255443 | 0.242820 |
| 20260923 | 0.206600 | 0.177841 | 0.176814 |
| **平均** | **0.249660** | **0.214994** | **0.209287** |

400步比初始化的误差平均减少**16.17%**，三个seed都改善；但100→400仅再减少**2.65%**。逐图检查：狗、车、蜗牛等整体主体保留，raw直接复用的严重破坏被减轻；白帽人物仍有明显面部/肤色异常，坐姿人物的脸也与dense不同，不能宣布质量过关。训练与采样仍共享GPU，计时仅两轮；这轮是质量诊断，不提供稳定加速结论。

**一个新的pattern：必须联合评价四层。** 400步时，四层误差在三个seed都低于“只用前三层adapter”的误差（分别为0.229892/0.247520/0.193571）；但后者比100步不一定更好。这与联合输出训练允许不同层互相补偿相符，尚非因果证明。不能随意关掉其中一层，又认为剩下的adapter仍是针对那个子集训练的；也不能拿单层效果配四层的速度。

以下每张图五列依次为：**dense / raw四层 / adapter三层13,15,17 / adapter单层20 / adapter四层**。最后一列才对应本轮主方法。未挑除失败人物。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_400_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>最终输出监督400步：20260921</h4><p>白帽人物的面部仍失真；整体偏差下降不代表这个失败模式已修复。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_400_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>最终输出监督400步：20260922</h4><p>主体基本可辨，但纹理/背景仍改变。第四列只替换一层，不是四层主方法。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_400_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>最终输出监督400步：20260923</h4><p>车辆、蜗牛相对稳定；坐姿人物的脸仍与dense有明显差别。</p></article>
</div>

<details><summary>展开同seed的100步结果，检查是否继续改善</summary>
<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_100_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>100步：20260921</h4><p>同样保留白帽人物失败案例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_100_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>100步：20260922</h4><p>400步数值改善，但未出现普遍的画质质变。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_pilot_100_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>100步：20260923</h4><p>本seed的100→400四层误差变化很小。</p></article>
</div>
</details>

**后续有限续训：** 从400步恢复权重和optimizer，到**总计2000步，追加1600步**；不是从头2000步，也不把旧V2的步数混入这个计数。保持数据、四层、rank、LR、batch不变，每100步独立validation，按主预测MSE选择checkpoint，再自动对同三个seed生成。四卡每卡64，启动前要求GPU0–3各有至少12GiB余量，不停止其他任务。400步已相当于遍历小数据25遍，2000步相当于125遍；这一轮只回答“继续优化是否还有收益”，不能证明泛化充分。若验证和人脸继续停滞，应扩大独立轨迹、分析student轨迹分布偏移与各timestep误差，不再无限重复同512条轨迹。

这次同时改变了“训练状态来源”和“监督位置”，属于组合可行性pilot，不是单变量归因。如果有效，需要再做相同预算的拆分消融。如果无效，也不能靠不断重复这512条轨迹证明训练量充足；应检查独立validation和生成是否过拟合。它**不是对整个反向扩散过程做多步反向传播**，也没有把student轨迹的分布偏移完全解决。

| 文件/任务 | 位置（远端项目根目录下） |
|---|---|
| 采集代码 | `2d/attention_analysis/collect_csra_rollout_dataset.py` |
| 数据 | `eval_results/csra_rollout_data_20260912/shard_0` 至 `shard_3`，每个包含manifest |
| 可微路径 | `2d/attention_analysis/csra_output_distillation.py` |
| 训练代码 | `2d/attention_analysis/train_csra_rollout_output.py` |
| 测试tmux / log | `rdd_csra_output_smoke_0123_20260912` / `logs/csra_output_smoke_20260912.log` |
| pilot tmux / log | `rdd_csra_output_pilot_0123_20260912` / `logs/csra_output_pilot_20260912.log`（已正常完成） |
| pilot checkpoint | `checkpoints/RDD-CSRA-output-pilot-L400K-20260912/`，与旧实验分开 |
| 100/400步图与metrics | `eval_results/csra_output_pilot_100_20260912/`、`eval_results/csra_output_pilot_400_20260912/`，本机已完整下载 |
| 本机训练证据 | `eval_results/csra_output_pilot_evidence_20260912/`，含日志、complete与各分片manifest |
| 有限续训脚本 | `2d/exp/train_rdd_csra_output_continue2k_4gpu.sh` |
| 续训tmux / log | `rdd_csra_output_continue2k_0123_20260912` / `logs/csra_output_continue2k_20260912.log`（已正常完成2000步；无OOM） |
| 续训checkpoint / 自动预览 | `checkpoints/RDD-CSRA-output-continue2k-L400K-20260912/` / `eval_results/csra_output_continue2k_best_20260912/` |

### 14.14 续训检查：最佳800步，继续训练不能修复人物

续训完整跑到2000步，自动用独立validation选出的checkpoint生成三组预览，实际选中**800步**。不能把下面图片写成2000步生成；2000步checkpoint已保存，但未用它替代validation选择。续训1600步训练循环约661秒，峰值allocated仍约3.67GiB/GPU，无OOM。

| 总步数 | validation主预测MSE | variance MSE | 状态 |
|---|---:|---:|---|
| 400 | 0.00846061 | 0.00702651 | 续训起点 |
| 800 | **0.00839045** | 0.00650020 | validation选择的best |
| 1200 | 0.00848199 | 0.00637990 | 主预测开始退步 |
| 2000 | 0.00869697 | 0.00641252 | 主预测比400步还差 |

训练loss仍下降，但held-out主预测误差回升，符合在小数据上过拟合/泛化停滞的表现；仅凭这一轮不能断言唯一原因是过拟合。下面仍是实际四层13/15/17/20一起替换、相同DDPM80/80/90、CFG1.0、noise filling、FP32 adapter的生成偏差，**不是FID**。

| seed | 400步 | validation-best 800步 |
|---|---:|---:|
| 20260921 | 0.208228 | 0.207491 |
| 20260922 | 0.242820 | 0.241041 |
| 20260923 | 0.176814 | 0.178892 |
| **平均latent relative L2** | **0.209287** | **0.209141** |

均值仅改善约**0.07%**，一个seed退步。全部12张dense PNG与400步参照逐张SHA-256相同，三个完整grid已检查：白帽人物的肤色/五官仍明显异常，坐姿人物脸部也改变，**未通过质量检查**。因此停止对这批数据单纯增加训练步数。共享GPU、仅两轮计时，不用这些数字主张稳定加速。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_continue2k_best_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>最佳800步：20260921</h4><p>五列依次dense、raw四层、adapter三层、adapter单层20、adapter四层。白帽人物失败仍在。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_continue2k_best_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>最佳800步：20260922</h4><p>与400步几乎同一质量水平；不能把微小偏差改善称为质变。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_continue2k_best_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>最佳800步：20260923</h4><p>本seed四层生成偏差反而增加。仍保留而非筛除。</p></article>
</div>

**接下来的诊断，不是继续猜：** 用原来32条独立validation轨迹的同seed，让当前adapter实际完成90次fine采样，在相同8个时间点分别计算：

1. teacher轨迹输入上，adapter输出与teacher输出的误差（原训练验证的口径）。
2. student自己走到的输入上，adapter输出与teacher在**同一student输入**上的输出误差。不能拿两个不同输入的网络输出直接相减来冒充预测误差。
3. 相同噪声起点下，student与teacher轨迹的latent偏离随时间如何变化。
4. 主预测误差用2×2平均/复制投影拆为低频与剩余细节功率；这只描述误差分布，不证明coarse特征一定更准确。

诊断要求coarse缓存完全相同、首次fine输入完全相同、额外观测前后最终student latent最大差异为0；不做训练，不把验证数据加入训练集。代码`2d/attention_analysis/probe_csra_rollout_shift.py`；输出`eval_results/csra_rollout_shift_20260912/shard_0..3`。结果出来后再决定扩大数据还是改用student轨迹监督，不能预先宣称哪种原因成立。

### 14.15 配对轨迹诊断结果：后半程拟合不足，而非普遍的输入分布误差暴涨

已完成四组、共32条独立validation轨迹，每条8个观察点。使用最佳800步四层adapter，与原teacher使用相同seed、类别、DDPM80/80/90与noise filling。四组均通过：coarse缓存最大差异0、首次fine输入完全相同、原teacher保存的预测能逐元素重现、插入观测前后的完整student最终latent最大差异0。频带投影/误差计算的CPU检查也通过。

“teacher输入误差”指学生在teacher原本会走到的输入上预测，再与teacher在该输入上的预测比较；“student输入误差”则让两者都看student实际走到的输入。**每个误差都比较同一输入，不是拿不同输入的输出相减。** 表中按32张的平方误差和目标能量汇总后开平方，并非把四个batch的relative L2直接平均。

| fine NFE（0-based） | 全局t | teacher输入上的主预测相对L2 | student输入上的主预测相对L2 | 两条轨迹的latent相对偏离 |
|---:|---:|---:|---:|---:|
| 0 | 299 | 4.11% | 4.11% | 0.00% |
| 12 | 259 | 5.43% | 5.31% | 1.00% |
| 25 | 215 | 7.71% | 7.26% | 2.34% |
| 38 | 171 | 10.97% | 10.39% | 4.20% |
| 51 | 128 | 14.45% | 13.68% | 7.05% |
| 64 | 84 | 17.62% | 16.89% | 11.22% |
| 77 | 40 | 21.14% | 20.42% | 16.10% |
| 89 | 0 | 43.80% | 41.37% | 19.57% |

最后一列是在对应NFE**调用之前**观测的latent偏离，不是解码图像差异，也不是最终生成FID；32条validation也不是12张展示图的同一组seed。

![32条配对轨迹的预测误差、轨迹偏离与频带误差](../../eval_results/csra_rollout_shift_20260912/rollout_shift.png)

这轮支持和不支持什么：

- **支持“初入fine尚能近似，后半程越来越不够准”。** 单次主预测误差从约4%逐渐升到t=40的约20%，与此同时轨迹逐步偏离。这是时间位置上的相关性，不是单独证明coarse缓存过时就是唯一原因。
- **不支持“student输入使单步误差普遍暴涨”这个简单解释。** 在这些观测点，相对L2反而普遍略小；绝对MSE在t=259到40也都略小。但较小的teacher-student误差不等于生成更好，student已走到不同状态；仍不能排除重新采集student轨迹训练有帮助。
- **不要被t=0的43.8%吓成“绝对误差爆炸”。** teacher输入的主预测绝对MSE在t=40为0.01781，到t=0反而降到0.00308；目标能量下降使相对误差放大。两种误差必须一起看。
- **后半程细节修正是可疑薄弱项，但不是已经证实的唯一瓶颈。** 2×2平均投影保留1/4维，细节子空间占3/4维，所以“75%的总误差在高频”本身没有说服力。按每维归一化后，高频/低频误差功率比在t=128约1.72、t=84约1.67、t=40约1.52；t=299约0.52、t=0约0.92。这只是在latent主预测误差上观测的现象，不是图像感知质量证明。

每组的原始`metrics.json`、保留验证用途的`student_validation_*.pt`、汇总`summary.json`与上图均在`eval_results/csra_rollout_shift_20260912/`，本机已下载。观察额外执行teacher调用，因此这些诊断运行时间不能用于加速对比。

### 14.16 扩大数据的结果：偏差再降7.71%，仍不是质量过关

之前512条训练轨迹被重复用了125遍，validation在800步后回升。先检验增加独立样本能否改善这一限制，而不是同时更换架构、rank、采样器等多项设置。

| 项目 | 新一轮设置 |
|---|---|
| frozen backbone | 原生ImageNet的RDD-Unified-L/2 @400K，300/300/400，不更新backbone或VAE |
| adapter | 同四层13/15/17/20，fine-context V2，rank64，FP32 |
| 初始化 | 上轮validation-best 800步权重；**重建optimizer，新计数0** |
| 采集来源 | 原RDD dense teacher从噪声正常采样，不是student轨迹，也不下载其他checkpoint |
| 新训练 / 验证轨迹 | **4096 / 128条**；各8个fine状态，共32768 / 1024个状态 |
| 采集seed | `20270000 + shard*1000 + batch`；每shard 128个训练batch、4个验证batch，每batch8条 |
| 数据隔离 | 与旧训练/旧验证/展示seed不重合；14.15的32条validation绝不转成训练数据 |
| 采样/边界 | DDPM80/80/90、CFG1.0、noise filling；coarse t300原dtype缓存；四shard均须parity=0 |
| 新训练预算 | **800个新updates**，约遍历新训练集6.25遍；不是恢复到旧总计1600步 |
| batch/GPU | 物理0–3，每卡64，全局256，不做梯度累积；采集每卡batch8 |
| loss/optimizer | 主预测MSE + 0.05×variance MSE；AdamW LR1e-4、WD0.01、clip1.0 |
| 验证/保存 | 每100步validation、每200步保存；按新验证集主预测MSE选best |
| 初始模型保护 | **step0初始化也纳入best候选**，避免所有更新都变坏时仍误选一个更新模型 |
| 后续生成 | 自动为validation-best生成相同三seed，每seed4张，必须检查失败人物 |

本轮没有改变架构和损失，但增加了数据、重建了optimizer并更换训练随机流，因此首先是更充分的数据可行性检查，不冒充严格单变量消融；若有效，再做相同初始化/optimizer预算的旧数据对照。所有新数据都存新目录，旧checkpoint与结果不覆盖。启动前检查每张卡至少12GiB余量；已检查磁盘约465GB空闲、主存约936GiB available，预期新数据仅数GB。其他用户的进程不停止。

| 文件/任务 | 远端项目根目录下的路径 |
|---|---|
| 自动采集→训练→预览启动器 | `2d/exp/train_rdd_csra_data4k_output_4gpu.sh` |
| tmux / 主日志 | `rdd_csra_data4k_and_output_0123_20260912` / `logs/csra_data4k_and_output_20260912.log` |
| 四个采集日志 | `logs/csra_data4k_shard0_20260912.log` 至 `shard3` |
| 数据 | `eval_results/csra_rollout_data4k_20260912/shard_0..3` |
| checkpoint | `checkpoints/RDD-CSRA-output-data4k-L400K-20260912/` |
| 自动预览 | `eval_results/csra_output_data4k_best_20260912/seed_20260921..23` |

训练脚本补充了“初始模型也可获选best”的保护，不改变网络或旧checkpoint读法；旧源文件另存`.tmp/train_csra_rollout_output_before_data4k_20260912.py`。四个采集分片均正常完成，parity均为0，训练前向与旧采样前向的最大差异也为0。四项梯度/冻结参数/损失单元测试重新通过。800步训练循环约337秒，峰值allocated约3.67GiB/GPU，无OOM。

**新验证集上的改善：** 以下所有列都是同一组128条新validation轨迹、1024状态，不把旧32条validation的数值混在一起。

| 在4096条训练轨迹上的新更新数 | 初始化（旧best800权重） | 100 | 400 | 800 / 本轮best |
|---|---:|---:|---:|---:|
| 主预测MSE | 0.00835921 | 0.00793258 | 0.00761224 | **0.00743651** |
| variance MSE | 0.00571412 | 0.00553853 | 0.00525137 | 0.00496796 |

主预测MSE相对初始化下降约11.04%。本轮最佳是**新更新800步**；这个800不是14.14的旧800，初始化来自旧800，但重建了optimizer且计数重新开始。

| 同seed，全四层替换的生成偏差 | 旧512条数据best800 | 新4096条数据再更新800 |
|---|---:|---:|
| 20260921 | 0.207491 | 0.192329 |
| 20260922 | 0.241041 | 0.229248 |
| 20260923 | 0.178892 | 0.157489 |
| **平均latent relative L2** | **0.209141** | **0.193022** |

三个seed均改善，平均相对下降**7.71%**，12张dense PNG与上轮逐张SHA-256相同。所有图已检查：车、蜗牛等整体保留，坐姿人物的躯体与脸部较原raw直接复用稳定；但白帽人物的面部/肤色仍明显异常，狗的姿态也变化，不能认为已经充分保持原生成质量。只替换layer20的部分指标反而略差，再次说明**四层联合训练后的效果不能外推到任意子集**。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_best_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>4096条数据 +800更新：20260921</h4><p>仍为dense/raw四层/adapter三层/adapter单层/adapter四层。白帽人物仍有明显伪影；不是画质过关样例。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_best_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>4096条数据 +800更新：20260922</h4><p>主方法是最后一列。纹理与背景变化仍在，偏差减小不等于FID已改善。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_best_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>4096条数据 +800更新：20260923</h4><p>本seed改善相对明显，但仍需新seed验证，不能反复用这四张挑参数。</p></article>
</div>

完整图、单张PNG、metrics与日志已下载到本机同名目录；四个数据manifest和训练complete.json也已保留。这里是同模型更接近dense的证据，**不是FID、不是稳定计时、不是视频净加速实测**。

### 14.17 补充匹配对照与新seed验证

为了区分“更多数据”与“又训练了800步/重建optimizer”，补一组更严格的对照，而不是直接把14.16的改善全部归因于数据量。

| 配置 | 已完成4096条组 | 512条数据对照 |
|---|---|---|
| 初始化 | 同一个旧best800 checkpoint | 完全相同 |
| optimizer / seed | 新AdamW，LR1e-4，WD0.01，seed20274000 | 完全相同 |
| 新更新 / batch | 800；全局256，每卡64，0–3卡 | 完全相同 |
| 模型 / 复用层 / rank / loss | 冻结RDD-L/2；13,15,17,20；64；主MSE+0.05variance | 完全相同 |
| 训练数据 | 4096条，32768状态 | 旧512条，4096状态 |
| **validation / 选择标准** | **新128条validation，按主MSE选best，含step0** | **使用同一组新128条validation、同一标准** |
| 展示seed | 20260921/22/23 | 完全相同 |

**两组均已完成800次更新。** 控制组初始validation主MSE为`0.00835921204998158`、variance为`0.005714119732147083`，与大数据组初始化完全一致；训练前向parity=0，四项梯度/冻结参数测试通过，无OOM。不同训练集长度使数据排列不同，但更新数/批量相同；这是同样本访问预算，不是把512条也强行当成4096条独立样本。训练脚本新增可选`--validation-data-root`，默认仍为原data-root，不改变旧用法；修改前源文件保存在`.tmp/train_csra_rollout_output_before_valsplit_20260912.py`。

此外，对**已经选好的4096条组best checkpoint**固定生成四个新seed `20275001/20275002/20275003/20275004`，每seed4张，仍为相同DDPM80/80/90、CFG1、noise filling、FP32 adapter。不以这16张挑选checkpoint，全部结果都保留。为只看质量，此处`benchmark-repeats=0`，输出JSON中的首次运行时间**不能用于加速比较**。

| 文件/任务 | 远端根目录下的位置 |
|---|---|
| 启动器 | `2d/exp/train_rdd_csra_data_control_and_fresh_4gpu.sh` |
| tmux / log | `rdd_csra_data_control_and_fresh_0123_20260912` / `logs/csra_data_control_and_fresh_20260912.log` |
| 512条控制组checkpoint | `checkpoints/RDD-CSRA-output-data512-control-L400K-20260912/` |
| 512条控制组三seed预览 | `eval_results/csra_output_data512_control_best_20260912/`，训练后用0/1/2卡 |
| 4096条组新seed预览 | `eval_results/csra_output_data4k_fresh_20260912/`，控制组训练后依次使用3卡 |

**匹配对照结果：** 以下MSE在完全相同的128条validation轨迹、1024个状态上计算；所有指标越低越接近teacher。

| 指标 | 512条组 | 4096条组 |
|---|---:|---:|
| 初始主预测MSE | 0.00835921 | 0.00835921 |
| 完成800更新时的主预测MSE | 0.00853284 | **0.00743651** |
| 最佳validation对应的新更新数 | **100** | **800** |
| 最佳validation主预测MSE | 0.00832303 | **0.00743651** |
| 最佳checkpoint生成，seed20260921 latent relative L2 | 0.206251 | **0.192329** |
| seed20260922 | 0.240232 | **0.229248** |
| seed20260923 | 0.178230 | **0.157489** |
| 三seed平均 | 0.208238 | **0.193022** |

固定800次更新看，4096条组主预测MSE低12.85%；各自按相同validation规则选择最佳checkpoint看，低10.65%。生成偏差低7.31%。512条组到100步就最佳，之后验证误差逐渐回升；4096条组则到800步仍下降。**这支持本实验中增加独立训练轨迹有帮助，不支持“只要增加数据就一定能无损替换fine block”这样的普遍结论。** 两组三seed的12张dense PNG逐张SHA-256完全相同。下面保留全部控制组图片，方便与14.16对应比较。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_data512_control_best_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>512条匹配对照，best新100步：20260921</h4><p>最后一列是联合替换13/15/17/20四层的主方法；白帽人脸仍明显异常。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data512_control_best_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>512条匹配对照：20260922</h4><p>结构大致保持，水纹、珊瑚状纹理与蝶翼细节仍会改变。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data512_control_best_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>512条匹配对照：20260923</h4><p>车与蜗牛比raw稳定；人物脸部和姿态仍变化，不能只看较好的物体样例。</p></article>
</div>

**四个新seed的16张图：** 都使用已经选好的4096条组best800；每行依次为dense、raw四层、adapter三层13/15/17、adapter单层20、adapter四层13/15/17/20。最右列是本轮主方法，不能把中间子集的质量当成四层结果。

| seed / 每组4张 | raw四层latent偏差 | adapter四层latent偏差 |
|---|---:|---:|
| 20275001 | 0.307584 | 0.184798 |
| 20275002 | 0.329456 | 0.222246 |
| 20275003 | 0.273195 | 0.146953 |
| 20275004 | 0.296417 | 0.200601 |
| 平均 | **0.301663** | **0.188649** |

这是相对各自同seed dense的latent误差，不是同真实ImageNet比较的FID；四个batch的误差平均也不是逐张误差的平均。这里相对raw降低37.46%，**不等于画质提高37.46%**。这16张最初未用于checkpoint选择，现在已看过，今后继续用它们比较应称“配对开发集”，不能再称完全未触碰的最终测试集。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_fresh_20260912/seed_20275001/comparison_grid.png" style="width:100%"><h4>新seed20275001</h4><p>乐器、菌类、溪流和动物的整体布局明显比raw稳定；乐器局部形状和动物细节仍有变化。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_fresh_20260912/seed_20275002/comparison_grid.png" style="width:100%"><h4>新seed20275002</h4><p>鸟、黑狗、物体和人狗场景主体保留；黑狗脸部形态、人物细节和背景仍发生变化。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_fresh_20260912/seed_20275003/comparison_grid.png" style="width:100%"><h4>新seed20275003</h4><p>小物件、面包和狼的结构保留较好；女性脸部与衣物图案仍变化。原dense本身也并非完美真实图像。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_fresh_20260912/seed_20275004/comparison_grid.png" style="width:100%"><h4>新seed20275004</h4><p>鸟、蓝色液体容器、麝牛的结构明显好于raw；小物体和文字仍不可靠。不能据此声称文字能力得到保留。</p></article>
</div>

所有7组完整PNG、grid、metrics、日志与控制组complete.json均已下载本机，汇总包为`eval_results/csra_data_control_fresh_results_20260912.tar.gz`。本轮仍未运行用于论文结论的FID或稳定视频计时。整个任务最多使用物理0–3卡，不停止其他任务，**还没有达到“只差视频训练”的阶段**。

### 14.18 4096条数据延长到3200步：验证改善，但生成收益不一致

这一轮先不换架构：4096条组800次更新仅相当于约6.25遍数据，验证误差尚在持续下降，而512条匹配对照已经回升。因此检验大数据组是否仍有训练不足，不继续反复训练已明确过拟合的小数据组。

| 项目 | 设置 |
|---|---|
| 恢复来源 | `RDD-CSRA-output-data4k-L400K-20260912/0000800.pt`；恢复权重和AdamW状态，不重建optimizer |
| 步数 | 800 → 最多总计3200，即最多再2400次更新；不是再加3200 |
| 数据 / 模型 / 层 | 保持4096 train +128 validation；冻结RDD-L/2@400K；四层13/15/17/20，rank64，fine-context V2 |
| batch / 设备 | 全局256，每卡64，不累积；物理0–3卡 |
| loss / LR / seed | 主MSE +0.05variance MSE；1e-4；20274000，均不变 |
| 验证 / 保存 | 每200步validation，每800步保存；若全程不改善则使用原800步checkpoint |
| 采样 | 按validation选best后，复跑3个原seed与4个配对开发seed，各4张；DDPM80/80/90、CFG1、noise filling |
| 计时口径 | 此轮preview不做测速，`benchmark-repeats=0`；首次运行时间不能用于速度结论 |
| 启动器 | `2d/exp/train_rdd_csra_data4k_continue3200_4gpu.sh` |
| tmux / log | `rdd_csra_data4k_continue3200_0123_20260912` / `logs/csra_data4k_continue3200_20260912.log` |
| checkpoints | `checkpoints/RDD-CSRA-output-data4k-continue3200-L400K-20260912/` |
| 预览输出 | `eval_results/csra_output_data4k_continue3200_best_20260912/`、`eval_results/csra_output_data4k_continue3200_fresh_20260912/` |

如果验证误差也不再改善且面部问题仍明显，就不能继续以“再多训练一点”代替分析，应测试当前t/类别条件或fine细节修正分支是否不足。即使图像可行，视频还必须测**包含adapter、缓存、插值成本的净加速**；之前直接跳层的数字不能替代这个结果。

**训练和7组完整预览均已完成。** 恢复时主MSE `0.007436514453729615`、variance MSE `0.00496796373045072` 与原800步完全一致，训练/采样前向parity=0。四项CPU回归测试通过，每卡训练峰值allocated约3.67GiB，无OOM。最终选择的是**2800步，而非3200步**。

| 总更新步数（此4096条数据组） | 800 | 1600 | 2400 | 2800（best） | 3200 |
|---|---:|---:|---:|---:|---:|
| 同128条validation主MSE | 0.00743651 | 0.00726892 | 0.00718364 | **0.00714728** | 0.00716176 |

| 同seed四层adapter生成偏差 | 800步 | 2800步 |
|---|---:|---:|
| 20260921 | 0.192329 | 0.187470 |
| 20260922 | 0.229248 | 0.224193 |
| 20260923 | 0.157489 | 0.151585 |
| **原三组均值** | **0.193022** | **0.187749（下降2.73%）** |
| 20275001 | 0.184798 | 0.183852 |
| 20275002 | 0.222246 | 0.225932 |
| 20275003 | 0.146953 | 0.156083 |
| 20275004 | 0.200601 | 0.206860 |
| **较新四组均值** | **0.188649** | **0.193182（增加2.40%）** |

所有28张dense PNG与800步那轮逐张SHA-256相同，采样配置没有变化。表中仍是latent relative L2，不是FID。较新四组仅16张，不能由2.40%变化推断总体FID显著变差；但足以说明**不能只拿原三组的改善宣布方法普遍更好**。白帽人的红色面部异常、其他人物与纹理变化仍存在，尚未达到质量目标。故不再延长这一个同配置训练，转向有明确假设的候选对照。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_best_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>2800步：20260921</h4><p>最右列为四层adapter，白帽人脸仍有明显肤色异常；狗与摄影器材相对稳定。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_best_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>2800步：20260922</h4><p>保留总体布局，但纹理和局部形状仍变化，不能仅看主体存在就判质量过关。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_best_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>2800步：20260923</h4><p>原展示集中相对稳定的一组；人物身体/脸部与原dense仍不同。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_fresh_20260912/seed_20275001/comparison_grid.png" style="width:100%"><h4>2800步：20275001</h4><p>新开发集中唯一latent偏差小幅下降的seed，整体形状较好。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_fresh_20260912/seed_20275002/comparison_grid.png" style="width:100%"><h4>2800步：20275002</h4><p>黑狗脸部、背景与人物細節变化；平均偏差比800步略高。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_fresh_20260912/seed_20275003/comparison_grid.png" style="width:100%"><h4>2800步：20275003</h4><p>面包和动物总体保留；女性脸和衣物仍不一致。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_output_data4k_continue3200_fresh_20260912/seed_20275004/comparison_grid.png" style="width:100%"><h4>2800步：20275004</h4><p>鸟、容器、麝牛比raw合理，但比800步并非更接近dense。</p></article>
</div>

以上每张grid都是dense / raw四层 / adapter三层 / adapter单层 / adapter四层。完整图片、指标与日志已下载到本机同名目录，打包文件为`eval_results/csra_data4k_continue3200_results_20260912.tar.gz`。

### 14.19 当前候选：把原模型的“当前时间＋类别”交给adapter

**疑点不是旧adapter完全不知道时间。** V2已有缩放到0–1的时间正弦特征，也能从hidden/coarse间接获得语义；但它没有直接使用原DiT每一层都在用的、训练好的时间与类别条件向量。原block的更新由当前条件调节，旧coarse residual却来自t=300；让adapter直接拿到当前条件，是一个值得单独检验的补充，尚未证明一定有效。

用人话说：旧版根据“低分辨率怎么改过画面＋当前细节”猜这层该怎么改；新版额外告诉它“现在进行到哪一步、正在生成哪一类东西”。这个信息**原DiT已经算过**，不为adapter重新执行一遍embedding。

~~~python
# 每次原DiT forward中本来就有：
condition = t_embedder(t) + y_embedder(label)

# 新增的很小的、每个样本只算一次的分支：
gates = small_mlp(layer_norm(condition))  # H -> 2r -> 6r
# gates对coarse、高频、完整fine三个rank-r分支各给一个scale和shift。
# 输出仍是预测该block residual，加入当前hidden；不增加全局attention。
delta = adapter(coarse_residual, current_hidden, t, condition)
h = h + delta
~~~

| 项目 | V2对照 | 条件V3 |
|---|---|---|
| 架构名 | `fine_context_v2` | `teacher_condition_v3` |
| 恢复权重 | 同一个4096条组best2800 | 完全相同，新增条件输出投影初始化为0 |
| optimizer / seed | 新AdamW /20276000 | 相同 |
| 数据 / 更新数 | 4096训练轨迹、128验证轨迹；最多800个新更新 | 相同 |
| batch / 设备 | 全局256，每卡64，不累积；0–3卡 | 相同；两组训练依次跑，不多占卡 |
| 其他 | 四层13/15/17/20，rank64，FP32；主MSE+0.05variance；LR1e-4 | 相同 |
| 唯一架构变化 | 无新条件分支 | 已有条件→小MLP→三个分支的FiLM调节 |
| 选模型 / 生成 | 同validation主MSE规则选best；7组配对开发seed，全部展示 | 相同 |

这同时增加了一点参数，若有效，不能直接把收益全部归因于条件信息；还需要参数匹配或打乱条件的消融。它仍依赖RDD真实coarse缓存，但也不能只凭这个模块就声称所有全分辨率缓存方法都无法使用条件修正。

启动器：`2d/exp/train_rdd_csra_condition_pair_4gpu.sh`。已在tmux `rdd_csra_condition_pair_0123_20260912`启动，主日志`logs/csra_condition_pair_20260912.log`，两组单独训练日志`logs/csra_condition_v2_20260912.log`和`logs/csra_condition_v3_20260912.log`。结果分别为`eval_results/csra_condition_v2_20260912/`、`eval_results/csra_condition_v3_20260912/`，各7个seed目录。先完成两组训练，再在0/1/2卡各跑一组原seed、3卡依次跑四组较新seed，均不进行正式测速。

**兼容性检查已通过后才启动训练：** 15项CPU测试通过，包括V1/V2旧行为、V2→V3零初始化输出相等、新分支梯度以及backbone冻结。真实RDD-L checkpoint在t=299与t=0、共4个状态的GPU检查中，两者都冻结时输出最大差异0，两者都可训练时最大差异也为0；四层条件分支梯度均非零，backbone没有梯度，峰值allocated约1.06GiB。

检查过程中发现并保留了一次失败记录：最初验证脚本把冻结V2和可训练V3混合比较，末端输出最大差异0.125。逐层追踪显示同输入、同权重的`high_down`首次出现约2.3e-5数值差异，沿后续BF16计算传播；把两者状态对齐后消失。**这不是把0.125强行当成通过阈值，而是修正了比较模式。** 仍不能声称同一模型的训练/冻结两种CUDA执行路径逐bit相同；正式新旧比较必须使用一致模式。通过日志为`logs/csra_condition_gpu_smoke_aligned_20260912.log`，早期诊断日志也保留。

旧源文件已备份到远端`.tmp/csra_source_before_condition_20260912.tar.gz`，未改动原模型或任何旧checkpoint。条件转换只允许显式V2→V3，默认旧架构不变；缺少条件、错误层号/rank、丢失原参数都会明确报错，避免静默加载错误模型。

**2026-09-13 完成结果。** 两组均完成800个新更新，validation均选择800步；每组7个seed、每seed4张、每张5种mask的全部生成已完成。14组全部指标、280张单图、14张grid和日志均已下载，逐一查看了全部grid；两组28张dense PNG在本机逐张SHA-256完全一致。归档为`eval_results/csra_condition_pair_complete_20260913.tar.gz`，SHA-256=`6b74608cc5abebbb287864c07b6ccd104bbb5a4e9e8a16ae7cdd298f41af5169`。

| 指标（越低越好） | V2匹配对照 | 条件V3 |
|---|---:|---:|
| 可训练参数 | 2,055,168 | 2,778,112 |
| 验证主预测MSE | 0.0071354045 | 0.0071341264 |
| 验证variance MSE | 0.0046603345 | 0.0046410869 |
| 原三组最终latent偏差均值 | 0.186262 | 0.186099 |
| 较新四组最终latent偏差均值 | 0.192608 | 0.192900 |
| 全七组最终latent偏差均值 | 0.189888 | 0.189985 |

V3增加35.18% adapter参数，主MSE仅下降0.018%；原三组偏差略降0.088%，较新四组反而增加0.152%，全部七组增加0.051%。幅度很小，没有重复训练种子的显著性检验。结论是**这次直接补条件没有显示明确收益，不再把同一配置继续训练当成主要方向**，不是证明任何条件建模都无效。已查看的V2/V3白帽人物仍有鼻口、肤色变化；两版本总体非常接近。

| seed | V2四层 | V3四层 |
|---|---:|---:|
| 20260921 | 0.187980 | 0.188083 |
| 20260922 | 0.220638 | 0.219869 |
| 20260923 | 0.150168 | 0.150343 |
| 20275001 | 0.182875 | 0.183796 |
| 20275002 | 0.222526 | 0.222999 |
| 20275003 | 0.156401 | 0.156199 |
| 20275004 | 0.208629 | 0.208604 |

以下按相同seed成对放置V2/V3的全部图片，列顺序始终为dense、raw四层、adapter三层、adapter单层、adapter四层。最后一列才是当前四层主方法。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>20260921，V2</h4><p>第五列是四层主方法；第三/四列分别只替换三层/一层。白帽人脸仍有明显偏差。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20260921/comparison_grid.png" style="width:100%"><h4>20260921，V3</h4><p>增加条件后与V2极接近，没有解决人脸问题。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>20260922，V2</h4><p>主体保留，局部纹理与dense不同。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20260922/comparison_grid.png" style="width:100%"><h4>20260922，V3</h4><p>数值略改善，视觉差别小，不能据此判总体质量更好。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>20260923，V2</h4><p>汽车、蜗牛相对稳定。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20260923/comparison_grid.png" style="width:100%"><h4>20260923，V3</h4><p>人物脸部和身体变化仍存在。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20275001/comparison_grid.png" style="width:100%"><h4>20275001，V2</h4><p>形状基本保留，与raw相比改善明显。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20275001/comparison_grid.png" style="width:100%"><h4>20275001，V3</h4><p>与V2非常接近，新增条件没有带来明显视觉改善。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20275002/comparison_grid.png" style="width:100%"><h4>20275002，V2</h4><p>黑狗和人物的局部细节仍有变化。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20275002/comparison_grid.png" style="width:100%"><h4>20275002，V3</h4><p>黑狗轮廓保留，人物与标牌细节差异仍在。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20275003/comparison_grid.png" style="width:100%"><h4>20275003，V2</h4><p>女性脸部与衣物纹样变化仍在。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20275003/comparison_grid.png" style="width:100%"><h4>20275003，V3</h4><p>女性脸部、手臂和衣纹依旧偏离dense，不能判画质过关。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v2_20260912/seed_20275004/comparison_grid.png" style="width:100%"><h4>20275004，V2</h4><p>容器轮廓改变；形状看似合理不等于复现dense。</p></article>
<article class="rdd-card"><img src="../../eval_results/csra_condition_v3_20260912/seed_20275004/comparison_grid.png" style="width:100%"><h4>20275004，V3</h4><p>容器、文字的变化与V2相似，新增条件没有解决这些例子。</p></article>
</div>

### 14.20 只读诊断：到底哪一层、哪个时间段需要修复？coarse缓存还在起作用吗？

条件V3未显示收益后，不立即训练另一个更大的模型，先做反事实检查。使用V2最新best800，4K数据组原来独立的128条validation轨迹、每条8个状态，共1024状态；**没有把验证数据改作训练，没有生成额外完整图片，也不拿此处耗时测速**。

做法很简单：同一个输入，先测四层adapter的误差；然后每次把其中一层换回真正的dense block，其他三层保持不变，看最终去噪预测误差减少多少。另测把coarse缓存清零、在同一batch中错配给另一个样本。最后两个是分布外干预，不能直接视为删除缓存并重新训练后的效果。

| 干预 | 全1024状态主MSE | 相对四层adapter |
|---|---:|---:|
| 四层adapter原样运行 | 0.00713532 | 1.0000 |
| 第14层恢复dense（代码13） | 0.00660010 | 0.9250 |
| 第16层恢复dense（代码15） | 0.00655371 | 0.9185 |
| 第18层恢复dense（代码17） | 0.00662976 | 0.9291 |
| 第21层恢复dense（代码20） | 0.00666957 | 0.9347 |
| 所有coarse缓存清零 | 0.00806320 | 1.1300 |
| 所有coarse缓存错配样本 | 0.00768492 | 1.0770 |

**缓存有作用，但不能说它独自解释了全部收益。** 清零使误差增加13.00%，错配增加7.70%；这证明当前网络对配对缓存敏感，并不证明它比同预算的无缓存重新训练模型更好。恢复一个dense block只减少约6.5%–8.2%的平均主误差，也说明问题不只来自一层。

更有价值的是时间差异。下表都是“干预后MSE / 原四层MSE”，小于1表示恢复该层有帮助：

| 全局t | 恢复第14层 | 恢复第16层 | 恢复第18层 | 恢复第21层 | 缓存清零 | 缓存错配 |
|---:|---:|---:|---:|---:|---:|---:|
| 299 | 0.776 | 0.921 | 0.926 | 1.084 | 1.372 | 1.165 |
| 259 | 0.809 | 0.915 | 0.954 | 1.031 | 1.237 | 1.153 |
| 215 | 0.816 | 0.903 | 0.954 | 1.010 | 1.148 | 1.134 |
| 171 | 0.854 | 0.908 | 0.963 | 0.974 | 1.133 | 1.108 |
| 128 | 0.900 | 0.911 | 0.939 | 0.970 | 1.140 | 1.086 |
| 84 | 0.945 | 0.918 | 0.920 | 0.926 | 1.133 | 1.058 |
| 40 | 0.987 | 0.918 | 0.900 | 0.857 | 1.084 | 1.041 |
| 0 | 1.113 | 1.050 | 0.993 | 0.959 | 1.113 | 1.091 |

可以提出但尚未验证的方向：**跨stage刚开始时保护较早层，后段保护较深层，而不是整段90次调用始终采用同一替换mask。** 这不是已经得到新加速方法：恢复一层会增加计算，若作为候选，必须计算相同预算下的收益，不能用更少替换层的画质搭配原四层的速度。

另一个警讯：部分时间恢复真正dense层反而增大最终误差，说明“各层adapter可能共同补偿误差”，也可能与干预改变后续输入有关；不能把单层的错误贡献简单相加。只监督最终输出，未必让每层都真正模仿了对应dense residual。

实现为`2d/attention_analysis/probe_csra_layer_repair.py`，启动器`2d/exp/probe_csra_layer_repair_4gpu.sh`。19项CPU测试通过；四个GPU分片完整结束，原dense预测逐元素重放一致，adapter诊断路径与正式推理路径逐元素一致，峰值allocated约1.13GiB/GPU。原始结果与日志已下载到`eval_results/csra_layer_repair_20260913/`；可用`python 2d/attention_analysis/summarize_csra_condition_and_repair.py`重新计算表格。

**下一项容量检查已启动：** V2/V3三个分支的末尾都是1024←64线性投影，因此在忽略浮点舍入时，每个token的预测residual只能落在最多192维的固定仿射子空间中。即使条件分支变复杂，也不会扩大这三个末端投影张成的空间。这是架构事实，但不代表192维一定不够。新的只读探针计算真实teacher residual在该子空间之外还有多少能量，同时检查加入原coarse residual直接旁路后的理论拟合下限；它不是已经训练好的新方案。对应`probe_csra_capacity_floor.py`，输出`eval_results/csra_capacity_floor_20260913/`。后续必须同时检查teacher输入和student输入，防止将输入变化误当成容量不足。

### 14.21 容量检查完成：当前小网络存在一个明确的输出限制，但不是唯一原因

**这不是说hidden本身只有192维。** hidden依然有1024个通道；是三个输出投影各只能给出64种固定方向，它们相加最多覆盖192种方向。前面怎么调timestep或类别系数，都不能凭空增加末端投影之外的方向。记三个末端权重横向拼接为W、偏置之和为b，则预测更新可写成：

$$
\widehat\Delta=W a(h,c,t)+b,\qquad \operatorname{rank}(W)\le 3\times64=192.
$$

用W列空间的正交基Q，计算真实block更新中无法被**当前这套已学习投影**表示的部分：

$$
E_{\rm floor}=\left\|(I-QQ^\top)(\Delta_{\rm dense}-b)\right\|^2.
$$

这是固定W时、允许前面任意预测系数a的理想误差下限。它**不是所有rank64模型重新训练后的最优下限**，也不是FID下限；重新训练W可以改变这个空间，直接加入其他输出路径也可以改变表示范围。

在相同128条验证轨迹、1024状态上，分别测dense路径输入和student路径实际到达该层的输入。每次都让原dense block在**同一个当前输入**上重新计算目标，避免混用不同hidden。四层测得的W rank均为192。

| 层（代码编号） | dense输入：子空间下限 / 实际局部MSE | student输入：子空间下限 / 实际局部MSE |
|---|---:|---:|
| 13 | 41.15% | 41.15% |
| 15 | 45.28% | 47.50% |
| 17 | 52.83% | 54.48% |
| 20 | 53.64% | 55.34% |

**含义：** 在当前投影不变的情况下，就算前面的系数预测完美，仍有相当一部分“模仿这一层dense更新”的误差无法消除。因此，只给现有三个分支增加时间/类别调节，很可能碰不到这个限制。但剩下约45%–59%的局部误差仍在可表示空间里，所以不能说全部问题都由rank解释；这也不能证明每层局部MSE降低就必然改善生成。

还检验了一个看似自然但没有得到支持的捷径：**直接加上原coarse residual（固定scale=1）**，再让现有子空间补差。student输入下，四层的理想MSE下限分别从`0.347/0.465/0.839/2.703`升到`0.393/0.603/1.128/3.480`。也就是说，这种固定旁路的方向并不合适，不能不训练就强行添加。这里没排除学习门控、改变投影或其他尺度的可能性。

下一项更针对性的候选曾是：保持现有coarse缓存和三分支，**增加一条很便宜的当前fine特征逐通道门控输出**，以提供192维固定子空间之外的修正方向。它尚未实现/训练。根据最新要求，当前优先运行下节的**整个主模型与 adapter 联合训练对照**；没有同时改动 adapter 架构，以免混淆原因。

数值检查也保留：第一版投影误用了开启TF32的FP32矩阵乘法，导致“输出严格在其子空间内”的过严断言失败。独立微型检查测到约`1e-7`相对MSE的舍入偏差后，将SVD/投影分析改用FP64，并明确排除TF32/BF16舍入；**原adapter的推理设置没有改变**。修复后四分片完成，最大输出子空间相对MSE为`1.159e-7`，21项CPU测试通过，真实dense整网与局部block重放均一致。失败日志保存在`eval_results/csra_capacity_floor_failed_tf32_20260913/`，成功结果在`eval_results/csra_capacity_floor_20260913/`，均已下载。

### 14.22 整个主模型联合训练结果与后续实验

完整配置、研究方案和本轮 kernel/区域诊断记录放在 [RDD 联合训练与跨尺度加速方案](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。

原先的 V2/V3 与容量诊断都已完成，不是仍在训练。本轮新建 `rdd_csra_joint_pair_0123_20260913` tmux，依次跑全模型+adapter与新冻结对照。两组使用相同 Unified-L 400K、相同 V2 adapter 初始化、相同4096/128轨迹、相同四层、相同随机序列与损失，先各800步。每卡64/global256，不使用梯度累计。

关键变化是：训练时让当前学生重新产生16×16缓存，fine损失可以通过缓存回传给主模型；同时约束8×8和16×16输出接近原教师。输入仍来自旧教师轨迹，不是已实现在线学生rollout训练。新组统一使用FP32主参数+BF16 autocast，并保留原教师BF16输出作为目标，避免只给联合组改变精度。

最初24项单元测试和真实64张全模型反向传播通过；该轮峰值allocated约12.16GiB/卡、reserved约13.04GiB/卡。第100步fine验证MSE曾从`0.007162915`降至`0.007028686`，但下文完整结果表明这不能代表画质恢复。该轮训练与生成已结束，不是仍在等待。

配对生成明确使用**原始教师dense**作为第一列，再放新冻结组与联合组，不拿更新后的主模型当成原始dense。依然DDPM80/80/90、CFG1、noise filling。旧图片与checkpoint保留。

**两组800步与七组配对生成均已完成，旧tmux正常退出。** 联合组best800的fine验证MSE为`0.006370536`，冻结组best100为`0.007131910`；但完整生成相对原dense的latent偏差均值从`0.1899906`增至`0.2230209`（+17.38%，不是FID或画质下降百分比），6/7组更偏离。逐一查看全部grid后确认，白帽人物的严重变形、桌面和背景变化说明本配置没有修复质量。不能由单步MSE下降宣布成功。

七组图、完整数值和解释已放在 [新报告第2.5节](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。全部84张单图、7张grid、指标与日志已下载本机 `eval_results/csra_joint_pair_20260913/`。以下直接保留三组关键失败，三列均为**原dense / adapter-only / 联合训练**；没有覆盖旧实验。

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_joint_pair_20260913/seed_20260921/comparison_grid.png" style="width:100%"><h4>20260921：联合训练未修复白帽人物，反而严重变形</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_joint_pair_20260913/seed_20275003/comparison_grid.png" style="width:100%"><h4>20275003：桌面物件变化，面包总体稳定</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_joint_pair_20260913/seed_20275004/comparison_grid.png" style="width:100%"><h4>20275004：背景变化增加，并非所有内容都崩坏</h4></article>
</div>

**阶段互换也已完成，完整结果见新报告2.6节。** 三个失败seed中只换joint fine时，最终偏差分别为0.178800/0.146503/0.206165，均低于原冻结组；只换joint coarse时却变为0.273213/0.231443/0.268093。更新后的主干完全不替换block也明显偏离，说明不能把退化只归因于adapter。16阶段结束前，联合模型的输入已经相对原轨迹偏离7.5%–11.8%。这支持优先修复coarse路径保持问题，不是总体质量证明。

固定原8/16后，训练fine实际执行主干+adapter的后续对照也已完成：两组各800步与七组生成，无OOM。冻结组best100，fine联合组best800；完整latent偏差均值从`0.1901036`降至`0.1805941`（约-5.00%，6/7组改善），没有前一轮的极端头部拉长，但白帽人物肤色和脸部形状仍明显不对。七组coarse端点parity全部0；28张dense PNG与前轮SHA-256完全相同。所有图片、指标和日志已下载，详见新报告2.7节。**两套coarse/fine权重有额外显存代价，不能宣布视频净加速已达标。**

<div class="rdd-grid">
<article class="rdd-card"><img src="../../eval_results/csra_preserved_pair_20260913/seed_20260921/comparison_grid.png" style="width:100%"><h4>固定原coarse后：原dense / adapter-only / fine主干联合训练。脸部问题仍未解决。</h4></article>
<article class="rdd-card"><img src="../../eval_results/csra_preserved_pair_20260913/seed_20275003/comparison_grid.png" style="width:100%"><h4>桌面布局保持，平均偏差下降；不是FID证据。</h4></article>
</div>

32条validation轨迹的学生状态诊断也完成。fine联合组在教师状态上的主输出MSE约0.006849，在学生自身状态上约0.006093；归一化误差同样降低。没有证据认为平均局部误差在离开教师轨迹后被放大，但不排除有方向的误差累积和个别失败。所有诊断parity为0，见新报告2.8节。

V4对照也已全部完成：新增549,120个逐通道修正参数后，adapter-only的七组最终latent偏差均值为0.1900435，fine联合组为0.1804812。相对各自V2只变化约-0.032%/-0.062%，没有实质质量突破，白帽人物问题仍在。28张dense PNG与V2完全相同，coarse端点parity均为0；七组新图和所有配置已放在[新报告第2.9节](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。不把V4算有效改进，不继续扩大它。

| 当前状态 | 说明 |
|---|---|
| 1600续训已完成 | 两组1600步、七组生成全部完成，原续训tmux正常退出；联合best1600，冻结最优仍为历史best100 |
| 1600实际结果 | 七组平均latent偏差：冻结0.190104、联合800步0.180594、联合1600步0.177146；比800再降1.91%，5/7组改善，但脸部未修复 |
| 无来源对照已完成 | 无coarse缓存输入的模型联合训练1600步，七组生成完成、tmux正常退出；同初始化、数据、层数、精度、LR和batch |
| 关键反证 | 最终latent平均偏差：有coarse0.177146，无coarse0.176095；验证MSE0.00592335 vs0.00590971。没有观察到有coarse优势，不能把当前V2收益主要归因于跨尺度 |
| 质量限制 | 无coarse略低0.594%、4/7组更低，并非显著性结论；两者脸部都未修复 |
| V5候选也已完成 | 有coarse/无coarse各800步与七组生成全部完成；平均latent偏差0.181276/0.179628，验证MSE0.00634336/0.00635149，没有稳定来源优势，人物脸部仍失败 |
| V5完整记录 | `rdd_csra_memory_pair_0123_20260913`已正常退出；84张单图、7张grid和日志已下载，完整图与配置见新报告第8节，不追加训练预算 |
| 分时间诊断已完成 | 128轨迹×8状态：t299有coarse总误差少2.62%/3.13%（V2/V5），后半段基本无优势；低频预测误差早期改善较多，但没有新的画质/速度结论 |
| V6完整hidden对照已完成 | 相同V2无来源1600步初始化，有h/无h各新增800步；best700/800，七组平均latent偏差0.174397/0.173949，仍无稳定来源优势，白帽人物失败。全部84单图、7grid与日志已取回，见新报告§8，不追加相同配置预算 |
| 新只读检查已完成 | 原模型保留fine MLP，只替换coarse K/V；旧KV预测误差约为当前KV池化的2.36倍，改坐标/时间调制未解决。详见新报告§8，不报速度 |
| 三组KV预览已完成 | 全90步/前20步的旧KV平均latent偏差0.266364/0.145775；当前KV池化0.178604/0.077651。短窗口保留脸部更好，但同窗口当前池化仍更接近dense；完整63张PNG已取回 |
| 前缀信息探针已完成 | 192拟合/64校准/128验证轨迹：正确coarse使前14层hidden预测MSE降低13.64%；局部平均误差少22.44%，局部差异仅少1.93%。正确位置也优于等参数错移/换图/均值来源控制 |
| 原fine后缀检查已完成 | 接入原后10层后，正确coarse的主输出MSE比无来源少21.58%，但绝对误差仍大；1024状态oracle逐位parity，不是画质/速度结果 |
| V7前缀14层已完成 | 各800步、3组生成，有来源validation主MSE少17.12%、平均latent偏差少9.96%，但人物严重变形。全部3张grid和36张单图已下载，dense哈希及coarse parity匹配；未达质量/视频速度目标。见新报告§8 |
| 更温和8层已完成 | 重新拟合目标h_after7，两组各1600步、3组生成全部完成；有来源validation主MSE少12.02%、平均latent偏差少4.31%，但仅2/3组占优，人物仍有失败。全部新图与指标见新报告§8，不是视频速度结果 |
| 决策 | 不延长V2/V4预算；保留负面证据，不能只根据推理时清零变差来证明来源价值 |
| 来源验证 | 正确coarse预测MSE0.006849；父格错位0.007034、换图0.007254、清零0.007544；仅证明当前网络使用coarse，不证明无coarse训练一定更差 |
| 新研究 | 当前fine Q读取旧coarse K/V；需注意空间2倍时pair比例25%，可能比VSA20更多。详见新报告第8节，不是已实现的加速 |

<details><summary>V4关键失败对比：原dense / adapter-only / fine联合训练</summary><img src="../../eval_results/csra_preserved_v4_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V4联合训练仍未修复脸部"></details>

1600步的全部七组图和配置已合并到[新报告2.10节](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。28张dense PNG与800步参考SHA-256相同，14项coarse端点parity均为0；84张单图、7张grid、metrics与日志已下载本机。下面三列为**原dense / adapter-only best100 / fine联合best1600**。这些是开发用小样本，不是FID或画质提升百分比。

<details open><summary>1600步：人物脸部仍是关键失败，不能只看平均数</summary><img src="../../eval_results/csra_preserved_continue1600_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="1600步人物失败"></details>

完整无coarse训练对照和全部七组图片详见[新报告2.12节](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。84张单图、7张grid、metrics与日志已下载。28张dense及28张有coarse参考PNG均与前轮SHA-256相同，14项coarse端点parity为0。它是在同一RDD内比较来源作用，不是full-resolution DiT对照；全零分支部分参数的输入梯度为0，保留同样参数张量不等于完全相同的有效容量。即便共同暖启动可能有利于coarse组，无coarse仍追平，足以否定目前已有强来源优势的说法，但不是证明所有跨尺度架构都无用。

<details open><summary>充分训练后：原dense / 有coarse / 无coarse。人物失败两者都存在。</summary><img src="../../eval_results/csra_no_coarse_control_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="训练后有无coarse比较"></details>

V5主动记忆读取的完整七组图也已合并到[新报告第8节](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。28张dense PNG与旧参考SHA-256相同，14项coarse端点parity为0。无coarse平均偏差略低0.91%、4/7组更低，不能解释成显著胜负，但不支持当前V5已经获得跨尺度优势。

<details><summary>V5：原dense / 主动读取coarse / 充分训练的无coarse</summary><img src="../../eval_results/csra_memory_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V5有无来源比较"></details>

V6完整hidden的七组图已汇入[新报告§8](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.html)。有来源仅3/7组latent偏差更低，平均还略高约0.26%，不是显著胜负，更不是跨尺度优势证据。下面同样为原dense/有h/无h。

<details><summary>V6完整hidden：相同初始化、同时训练fine主干与adapter</summary><img src="../../eval_results/csra_hidden_pair_20260913/seed_20260921/comparison_grid.png" style="max-width:100%;width:768px" alt="V6完整hidden对照"></details>

当前仍没有“训练后画质已恢复且视频净加速10%–20%”的完整证据，更没有三种已证明仅RDD可用的方法。更新后的调研明确区分HPDM的跨分辨率特征先例、InvarDiff的时间/层粒度缓存，以及Helios的金字塔采样。新增DisCa和SeaCache只作为训练目标/缓存诊断的参考，不当成RDD专属贡献。
