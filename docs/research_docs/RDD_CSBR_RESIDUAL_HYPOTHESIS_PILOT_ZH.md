# CSBR residual 假设：四组小样本验证

> **先说结论：** 这轮结果只支持一个较弱、但真实的结论：middle-stage residual 确实携带与同层 fine residual 相关的低频更新；然而它目前只是**中等相关**，并且明显不如 full-resolution previous-timestep residual 稳定。因而“RDD coarse residual 天生比 full-resolution cache 更容易复用”这个强故事，已被当前零微调 FastWan pilot **否定**。CSBR 的计算加速仍然真实，但论文级质量需要专门蒸馏/微调，而且必须把 full-resolution cache 作为强基线。

<style>
.rdd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px;margin:14px 0 26px}.rdd-card{border:1px solid #d0d7de;border-radius:10px;padding:12px;background:#fff}.rdd-card video,.rdd-card img{width:100%;border-radius:7px;background:#111}.rdd-card h4{margin:.7em 0 .3em}.rdd-config{width:100%;border-collapse:collapse;font-size:.88em}.rdd-config th,.rdd-config td{border:1px solid #d8dee4;padding:5px 7px;vertical-align:top}.rdd-config th{width:30%;background:#f6f8fa;text-align:left;white-space:nowrap}.warn{padding:10px 14px;border-left:4px solid #bf8700;background:#fff8c5}.bad{padding:10px 14px;border-left:4px solid #cf222e;background:#ffebe9}.good{padding:10px 14px;border-left:4px solid #2da44e;background:#dafbe1}
</style>

## 1. 实验到底测了什么

| 项目 | 设置 |
|---|---|
| checkpoint | FastWan2.1-T2V-1.3B-Diffusers；**没有**经过 RDD/CSBR 训练 |
| 输出 | 832×448，29 frames，16 fps |
| sampler | DMD 3 NFE；`t=1000/757/522` |
| attention | 所有正常执行的 block 都是在线 VSA-20 |
| RDD 路径 | spatial `(4,1)→(2,1)→(1,1)`；latent `8×14×26 → 8×28×52 → 8×56×104` |
| full-res 路径 | `(1,1)→(1,1)→(1,1)`；三个 NFE 都是 `8×56×104` |
| 样本 | 2 prompts × 2 seeds；每个 run/boundary 有 4×30=120 个逐层观测 |
| RDD transition | 主口径 `noise_filling`；另跑 `innovation_preserving`，用于排除换噪声带来的干扰 |
| full-res transition | 主理论对照使用 `innovation_preserving`；另跑 `noise_filling` 作为重抽噪声控制 |

对每个 Transformer block 定义整块 residual：

$$
\Delta_l = h_{l+1}-h_l.
$$

对 middle→fine 的 RDD 比较，先把 middle residual 最近邻放大：

$$
\widetilde\Delta_l^{m}=U(\Delta_l^m).
$$

然后把真实 fine teacher residual 用“2×2 coarse cell 内平均，再最近邻放大”拆成正交的两部分：

$$
\Delta_l^f=P_{\mathrm{low}}\Delta_l^f+P_{\mathrm{high}}\Delta_l^f.
$$

这个 $P_{\mathrm{low}}$ 是与 RDD stage 几何严格对齐的 piecewise-constant 子空间，不是随意设置的 FFT cutoff。

<div class="warn"><strong>必须牢记：</strong>$U(\Delta_l^m)$ 本身就在 low 子空间，所以它与 $P_{\mathrm{high}}\Delta_l^f$ 的 cosine 接近 0 是由正交定义保证的。这个数字不能用作 CSBR 成功证据。真正有辨别力的是 low-band cosine、matched-layer 对 wrong-layer、最优缩放后误差，以及和 full-resolution cache 的公平比较。</div>

## 2. 实验一：CSBR 是否主要携带 low-frequency update

### 2.1 聚合结果

| trajectory / boundary | raw cosine | 对 fine low 的 cosine | wrong-layer 对 fine low | matched−wrong | fine low 能量 | fine high 能量 | 最优 scalar 后 low RRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| RDD noise-filling，coarse→middle | 0.287 | 0.331 | -0.004 | +0.335 | 70.4% | 29.6% | 0.911 |
| RDD noise-filling，middle→fine | 0.529 | **0.613** | 0.003 | **+0.611** | 72.1% | 27.9% | 0.760 |
| RDD common-innovation，coarse→middle | 0.300 | 0.346 | -0.004 | +0.351 | 70.2% | 29.8% | 0.905 |
| RDD common-innovation，middle→fine | 0.547 | **0.636** | -0.003 | **+0.639** | 71.9% | 28.1% | 0.742 |

可以得到四个具体判断：

1. **“有 low-frequency 关联”得到支持。** middle→fine 时，low cosine 从 raw 的 0.529 提高到 0.613；common-innovation 下从 0.547 提高到 0.636。
2. **这种关联是 layer-specific 的。** 同层 matched cosine 为 0.61–0.64，而错开 7 层的 wrong-layer 平均接近 0；不是“随便拿一个 coarse tensor 都差不多”。
3. **它还远远不是准确替代。** 即使每层允许选择一个最优 scalar，middle→fine low-band RRMSE 仍有 0.74–0.76。
4. **8→16 比 16→32 更难复用。** coarse→middle 的 low cosine 只有 0.33–0.35；当前 CSBR 只做 middle→fine 是合理的。

### 2.2 不同层差异非常大

RDD noise-filling 的 middle→fine low cosine：

| 最高的层 | cosine | 最低的层 | cosine |
|---:|---:|---:|---:|
| 0 | 0.93 | 24 | 0.39 |
| 29 | 0.90 | 12 | 0.40 |
| 1 | 0.85 | 9 | 0.40 |
| 15 | 0.82 | 8 | 0.44 |
| 2 | 0.80 | 11 | 0.44 |
| 17 | 0.77 | 16 | 0.44 |

这解释了为什么“均匀每隔几层跳一次”并不理想。现有 CSBR-6 的 `[0,5,10,15,20,25]` 平均 low cosine 为 0.667；同样跳 6 层，若只按本 pilot 的相似度选 `[0,29,1,15,2,17]`，预期 residual 匹配会明显更好。后者必须在独立 prompts 上验证，不能直接当正式结果。

## 3. 实验二：与 full-resolution timestep cache 公平比较

每一行都保持同一个 FastWan checkpoint、相同 3 NFE timestep、相同 VSA-20、相同最终形状，并让 fine NFE 跳过完全相同数量的 block。数值是候选最终 latent 相对其**各自不复用 baseline**的变化；越接近 cosine=1、RRMSE=0 越好。

| 方法 | transition | 跳过 fine blocks | scale | latent cosine | latent RRMSE |
|---|---|---:|---:|---:|---:|
| RDD cross-scale reuse | common innovation | 6 | 0.75 | 0.916 | 0.416 |
| Full-res previous-timestep reuse | common innovation | 6 | 0.75 | **0.976** | **0.226** |
| RDD cross-scale reuse | common innovation | 12 | 1.00 | 0.872 | 0.492 |
| Full-res previous-timestep reuse | common innovation | 12 | 1.00 | **0.983** | **0.180** |
| RDD cross-scale reuse | noise filling | 6 | 0.75 | 0.912 | 0.433 |
| Full-res previous-timestep reuse | noise filling | 6 | 0.75 | 0.932 | 0.396 |

<div class="bad"><strong>结论：强假设被否定。</strong>在当前零微调 checkpoint 上，full-resolution previous-timestep residual reuse 明显比 RDD cross-scale reuse 更接近各自 teacher。不能在论文中声称“low-resolution residual 天然比 full-resolution residual 更稳定、更容易复用”。如果经过 RDD+CSBR 专门训练后能够反超，这才会成为方法证据。</div>

`noise_filling` 的 full-resolution 控制会在同一形状上重新抽 Gaussian，并不是标准 full-resolution trajectory；它只说明“换噪声会让 timestep cache 变差”。主要公平结论应看 common-innovation 两行。

## 4. 实验三：跨 timestep / 跨 resolution 稳定性

| 比较 | raw cosine | low-band cosine | high-band cosine |
|---|---:|---:|---:|
| RDD middle(t=757) → fine(t=522)，noise filling | 0.529 | 0.613 | 不适用；正交后恒近 0 |
| RDD middle(t=757) → fine(t=522)，common innovation | 0.547 | 0.636 | 不适用；正交后恒近 0 |
| Full-res t=757 → t=522，common innovation | **0.773** | **0.860** | 0.617 |
| Full-res t=757 → t=522，重新抽噪声 | 0.618 | 0.785 | 0.267 |

这组结果与实验二一致：

- RDD middle residual 对 fine low residual 有中等相关；
- 但 common-innovation full-resolution cache 的低频与整体 residual 都更稳定；
- full-res 的 high-frequency residual 确实比 low-frequency 更快变化（0.617 < 0.860），这支持“低频更容易 cache”的一般规律，却没有证明 RDD 的跨尺度 cache 更强。

## 5. 实验四：误差、方差与训练难度

| 统计量，middle/t=757 → fine/t=522 | RDD noise filling | RDD common innovation | Full-res common innovation |
|---|---:|---:|---:|
| low/raw 最优 scalar，30 层平均 | 0.489 | 0.514 | 0.748 |
| 最优 scalar 后 RRMSE | 0.760 | 0.742 | **0.602** |
| 每层跨 4 cases 的 scale 标准差，再对层平均 | 0.035 | 0.037 | **0.029** |
| 每层跨 4 cases 的 cosine 标准差，再对层平均 | 0.034 | 0.039 | **0.028** |
| scalar-adapter gradient 标准差，再对层平均 | 0.0082 | 0.0089 | **0.0046** |

此外，真实 fine residual 的能量约 **72% 在 stage-low、28% 在 stage-high**。直接把 upsampled middle residual 当 fine update 时，误差的 high/low RMS 比约 0.60；也就是说当前最大问题不只是“CSBR 看不到 high-frequency”，而是它连 fine 的 low-frequency update 也还拟合得不够准。

<div class="warn"><strong>实验四只完成了一半。</strong>这里的 gradient 是冻结 residual 后，单参数模型 $sU(\Delta^m)$ 在 $s=1$ 的 MSE gradient proxy；它不是 full-network gradient variance。真正的训练难度比较必须在 RDD-Wan/Full-Wan 的训练或蒸馏 loop 中，对相同 batch、相同 loss、相同参数子集采集 per-example/per-microbatch gradients。目前 proxy 同样没有显示 RDD 更容易训练。</div>

## 6. 代表性视频：结果与数值一致

第一个 prompt 是黄色雨衣女性走在霓虹街道。RDD baseline 本身就是未适配 checkpoint 上的 faux-RDD，因此只能看“相对破坏”，不能用它判断训练后 RDD 的最终质量。

<div class="rdd-grid">
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_base_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_base.mp4"></video><h4>RDD noise-filling baseline</h4><table class="rdd-config"><tr><th>复用</th><td>无</td></tr><tr><th>作用</th><td>只作为 faux-RDD 相对参考</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_csbr6_s075_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_csbr6_s075.mp4"></video><h4>RDD CSBR-6，scale 0.75</h4><table class="rdd-config"><tr><th>latent</th><td>cosine 0.912；RRMSE 0.433（4-case 平均）</td></tr><tr><th>decoded case-0</th><td>cosine 0.885；PSNR 15.23 dB</td></tr><tr><th>观察</th><td>主体仍在，但纹理与局部结构明显破坏</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_csbr12_s100_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/rdd_nf_csbr12_s100.mp4"></video><h4>RDD CSBR-12，scale 1.0</h4><table class="rdd-config"><tr><th>latent</th><td>cosine 0.867；RRMSE 0.504（4-case 平均）</td></tr><tr><th>decoded case-0</th><td>cosine 0.847；PSNR 14.19 dB</td></tr><tr><th>观察</th><td>比 CSBR-6 更强的块状/纹理破坏</td></tr></table></article>
</div>

<div class="rdd-grid">
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_base_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_base.mp4"></video><h4>Full-resolution common-innovation baseline</h4><table class="rdd-config"><tr><th>复用</th><td>无</td></tr><tr><th>观察</th><td>正常、清晰的 FastWan 输出</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_cache6_s075_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_cache6_s075.mp4"></video><h4>Full-res cache-6，scale 0.75</h4><table class="rdd-config"><tr><th>latent</th><td>cosine 0.976；RRMSE 0.226（4-case 平均）</td></tr><tr><th>decoded case-0</th><td>cosine 0.987；PSNR 24.90 dB</td></tr><tr><th>观察</th><td>有脸部/细节变化，但远小于 RDD CSBR-6</td></tr></table></article>
<article class="rdd-card"><video controls preload="metadata" playsinline poster="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_cache12_s100_contact.jpg" src="../../eval_results/rdd_csbr_residual_hypotheses_20260909/case_0/full_ip_cache12_s100.mp4"></video><h4>Full-res cache-12，scale 1.0</h4><table class="rdd-config"><tr><th>latent</th><td>cosine 0.983；RRMSE 0.180（4-case 平均）</td></tr><tr><th>decoded case-0</th><td>cosine 0.993；PSNR 27.21 dB</td></tr><tr><th>观察</th><td>代表性视频最接近 baseline</td></tr></table></article>
</div>

decoded 数值是本次脚本直接对 VAE 输出 tensor 计算的成对 cosine/PSNR；它只能量化相对 baseline 的扰动，正式论文仍应统一使用 VBench、FVD、CLIP/文本一致性和人工 A/B。

## 7. 对 CSBR 方法设计的直接影响

### 7.1 现在不能再讲的故事

- 不能说 low-resolution residual 天生比 full-resolution timestep residual 更稳定；当前结果相反。
- 不能拿 cross-scale tensor 对正交 high-band 的 cosine≈0 当证明。
- 不能把 uniform-6 / uniform-12 当成合理的最终 layer mask；层间差异过大。
- 不能声称只解决 high-frequency correction 就够了；当前 low-band 本身也有 0.74–0.76 的最优缩放后 RRMSE。

### 7.2 仍然成立的价值

- RDD 给出了一个真实、已经支付过计算的低分辨率 trajectory；CSBR 可以在第一个 fine NFE 就使用它，而普通 full-resolution previous-timestep cache 必须先有一次同形状 fine NFE。
- 16→32 的同层 residual 确实有中等、layer-specific 的 low-frequency 关联。
- 跳过 whole fine block 带来的 10%–60% denoiser 加速是执行层面的真实收益；问题在于如何训练到可接受质量。

### 7.3 建议的下一版，不再盲目 uniform reuse

1. **Layer-selective CSBR。** 先从 `[0,29,1,15,2,17]` 这样的高相关层候选出发，在独立 prompts 上重新筛选；同时保留足够的 fine anchor blocks。
2. **Per-layer、per-boundary、timestep-conditioned scale。** 现有一个固定 `0.75/1.0` 不合理；本 pilot 的 middle→fine 最优 scalar 全层平均约 0.49–0.51，并且层间差异很大。
3. **Residual distillation。** teacher 目标必须直接包含 fine low residual；先学习
   $$
   \widehat\Delta_l^f=a_l(t_s,t_f)U(\Delta_l^m)+C_l(h_l^f),
   $$
   其中 $C_l$ 是轻量 high-resolution correction。仅调 scalar 不够，因为最优 scalar 后 low RRMSE 仍约 0.74。
4. **交错 anchor/reuse。** middle 和 fine 使用互补 layer mask，避免某一条 reused residual 连续跨两个 boundary 传播；但它只能缓解误差累积，不能替代 distillation。
5. **正式基线必须包含 full-resolution cache。** 只有在相同跳层数、相同 denoiser FLOPs/延迟下，RDD-CSBR 训练后质量优于它，才能主张 RDD trajectory 带来独特复用优势。

## 8. 文件与可复现入口

- 单 case 诊断脚本：`rdd_video/diagnose_csbr_residual_hypotheses.py`
- 四卡 launcher：`rdd_video/launch_csbr_residual_hypotheses_4gpu.sh`
- 聚合脚本：`rdd_video/summarize_csbr_residual_hypotheses.py`
- 聚合 JSON：`eval_results/rdd_csbr_residual_hypotheses_20260909/aggregate_metrics.json`
- residual 汇总 CSV：`eval_results/rdd_csbr_residual_hypotheses_20260909/residual_summary.csv`
- 输出扰动 CSV：`eval_results/rdd_csbr_residual_hypotheses_20260909/latent_comparisons.csv`
- 每个 case 的原始逐层数据与日志：`eval_results/rdd_csbr_residual_hypotheses_20260909/case_*/metrics.json`、`run.log`
