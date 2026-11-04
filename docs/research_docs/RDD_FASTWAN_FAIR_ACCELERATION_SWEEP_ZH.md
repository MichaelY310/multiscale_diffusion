# RDD × FastWan：从异常诊断到可复现加速

> 本报告重跑了独占 GPU、逐形状充分预热、交错顺序的 20–50 次配对计时，并新增一个**逐元素等价**的 SmallQ-32 Triton executor。真正的系统异常是：脚本关闭了 `dit_cpu_offload`，却漏掉 FastVideo 默认开启的另一个 `dit_layerwise_offload`，导致每次 NFE 逐层从 CPU 搬权重。显式让 1.3B DiT 常驻 GPU 后，832×448 的 Spatial `4→2→1` 在三个 prompt 上都稳定快于 `2→2→1`；更大分辨率的优势更明显。所有视频仍来自未经过 RDD 微调的 FastWan checkpoint，只能验证 plumbing/扰动趋势，不能冒充最终画质。

> **2026-09-09 新结果：** 旧的 C2F、SmallQ 和 generic StepReuse 都没有提供足够大的、RDD 独有的端到端收益。新的 **Cross-Scale Block Residual Reuse（CSBR）** 使用已完成 coarse RDD stage 的整 block residual，替代选定的 fine blocks。相对同一个 Spatial-421 RDD + 在线 VSA-20 基线，温和设置稳定降低 12–16% denoiser 延迟；激进设置降低 51–61%，并在包含当前 VAE 后仍降低 11–15% 总延迟。完整方法、稳定数字、视频、失败路线和 prior-art 审计见 [CSBR 专项报告](RDD_CROSS_SCALE_BLOCK_REUSE_RESULTS_ZH.html)。当前唯一核心缺口是 RDD+CSBR 微调后的匹配画质验证。

<style>
.rdd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px;margin:14px 0 26px}.rdd-card{border:1px solid #d0d7de;border-radius:10px;padding:12px;background:#fff}.rdd-card video{width:100%;border-radius:7px;background:#111}.rdd-card h4{margin:.7em 0 .3em}.rdd-config{width:100%;border-collapse:collapse;font-size:.84em}.rdd-config th,.rdd-config td{border:1px solid #d8dee4;padding:5px 7px;vertical-align:top}.rdd-config th{width:26%;background:#f6f8fa;text-align:left;white-space:nowrap}.rdd-diagram{width:100%;max-width:1040px;height:auto;margin:12px auto 24px;display:block}.rdd-diagram .box{stroke-width:1.5}.rdd-diagram .a{fill:#ddf4ff;stroke:#218bff}.rdd-diagram .b{fill:#dafbe1;stroke:#2da44e}.rdd-diagram .c{fill:#fff8c5;stroke:#bf8700}.rdd-diagram .d{fill:#fbefff;stroke:#8250df}.rdd-diagram .e{fill:#ffebe9;stroke:#cf222e}.rdd-diagram text{font:16px system-ui;fill:#24292f}.rdd-diagram .caption{font-size:14px;fill:#57606a}.rdd-diagram .arrow{stroke:#57606a;stroke-width:2;fill:none;marker-end:url(#arrow)}.note{padding:10px 14px;border-left:4px solid #bf8700;background:#fff8c5}.good{color:#1a7f37}.bad{color:#cf222e}
</style>

## 1. 一张图看懂四个方法

<svg class="rdd-diagram" viewBox="0 0 1040 245" role="img" aria-label="RDD、C2F route、StepReuse 和 SmallQ kernel 的关系">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#57606a"/></marker></defs>
<rect x="20" y="66" width="180" height="92" rx="12" class="box a"/><text x="110" y="100" text-anchor="middle">低分辨率 stage</text><text x="110" y="127" text-anchor="middle">真实 Q/K/V + VSA route</text>
<path d="M200 112 H285" class="arrow"/><rect x="285" y="38" width="190" height="55" rx="10" class="box b"/><text x="380" y="72" text-anchor="middle">C2F：放大 route</text>
<rect x="285" y="132" width="190" height="55" rx="10" class="box c"/><text x="380" y="166" text-anchor="middle">Reuse：命中则跳过 self-attn</text>
<path d="M475 65 H565" class="arrow"/><path d="M475 159 H565" class="arrow"/><rect x="565" y="66" width="205" height="92" rx="12" class="box d"/><text x="668" y="100" text-anchor="middle">高分辨率 stage</text><text x="668" y="127" text-anchor="middle">当前 V + 稀疏 route</text>
<path d="M770 112 H845" class="arrow"/><rect x="845" y="66" width="175" height="92" rx="12" class="box e"/><text x="932" y="100" text-anchor="middle">SmallQ-32</text><text x="932" y="127" text-anchor="middle">同 route，更多 GPU work</text>
<text x="520" y="224" text-anchor="middle" class="caption">四项互相分工：RDD 减 token；C2F 省在线 router；Reuse 省一次 self-attention；SmallQ 解决小 stage 吃不满 GPU。</text>
</svg>

四个方法没有互相偷换概念：RDD 改的是每个 NFE 的 token 数；C2F-Route 改的是稀疏 attention 的 route 来源；StepReuse-Out 改的是相邻同形状 NFE 是否重算 self-attention；SmallQ-32 只改同一稀疏 route 在 GPU 上如何切工作。

## 2. 实验口径：怎样保证数字公平

| 项目 | 本报告固定值 |
|---|---|
| checkpoint | 同一个 `FastWan2.1-T2V-1.3B-Diffusers` |
| sampler | DMD 3 NFE，t=`1000/757/522` |
| boundary | 一律 `noise_filling`，不再混入 redraw |
| 主输出 | 832×448、29 帧、16 fps、batch=1 |
| attention | 都从同一个 FastWan VSA checkpoint 出发；但 `c2f_route_only` 删除了 VSA 的 coarse-output 分支，不能与完整 VSA 当作同结构公平对照 |
| 速度协议 | 一个进程只加载一次模型；每个 shape 先 warm；候选正序/逆序交错；正式 20–50 次取中位数并保存原始样本/IQR/MAD |
| DiT 放置 | `dit_cpu_offload=False` 且 `dit_layerwise_offload=False`；1.3B DiT 常驻 GPU |
| steady-state 不计 | 模型/T5 加载、第一次 Triton 编译；它们另作为冷启动问题说明 |
| `sample+VAE` | denoiser + 同形状 VAE decode；不含 MP4 编码/落盘 |

旧数据把不同 GPU 时段、首次编译和逐 stage 主机同步混在一起，足以制造 421/221 的假排名。新计时使用 CUDA Events 记录三个 stage，整条轨迹只在末尾同步一次；VAE 也在起止两端同步。

## 3. 方法 A：RDD 到底省了什么

FastWan 原版三个 NFE 都在完整 latent 上跑。RDD 把早期调用放到低分辨率 latent，换 stage 时上采样已有内容并只给新增自由度补高斯噪声：

```text
for (spatial_factor, temporal_factor), t in stages:
    if shape changed:
        z = upsample_old_latent(z) + noise_in_new_fine_dimensions()
    prediction = denoiser(z, t)
    z = DMD_update(z, prediction, t)
```

`(4,1)` 的 4 同时缩小 H、W，所以空间 token 约除以 16，不是只除以 4；第二个数字缩时间。`4→2→1` 是 `(4,1)→(2,1)→(1,1)`，而 `4→4→1` 的前两次形状相同，因此后者才可能命中跨 NFE Reuse。

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_fixed_vsa20_pre_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_fixed_vsa20_pre.mp4"></video>
<h4>固定全分辨率 FastWan + VSA-20（基线）</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_fixed_vsa20_pre</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,1) → (1,1) → (1,1)</code><br><code>8×56×104 → 8×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.865s</strong>；1.00×，延迟减少 0.0%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.147s；与本行 denoiser 合计 <strong>2.012s</strong>；1.00×，延迟减少 0.0%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_vsa20_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_vsa20.mp4"></video>
<h4>Temporal RDD + 在线 VSA-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_temporal_vsa20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,2) → (1,2) → (1,1)</code><br><code>4×56×104 → 4×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.624s</strong>；1.39×，延迟减少 27.9%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.144s；与本行 denoiser 合计 <strong>1.769s</strong>；1.14×，延迟减少 12.1%</td></tr>
</table>
</article>
</div>

这组旧的严格配对说明 RDD 本身就能省 denoiser；但 3-NFE 中最后一次 full-resolution 仍完整保留，所以任何 RDD 实现都有一个明显下限。

## 4. 方法 B：C2F-Route——低分辨率 attention 指挥高分辨率稀疏计算

VSA 每一层都把当前 Q/K 池化成 3D tile centroid，再算一个较小的 dense coarse attention 来选 Top-K。C2F 不重新问 fine stage 哪些 tile 重要，而是把上一 stage 已得到的 block route 映射到下一 stage 的子 blocks。仓库中有两个不同版本，必须分开：`c2f_route` 保留当前分辨率的 VSA coarse output；`c2f_route_only` 连这条分支也删除。

```text
route_coarse = TopK(pool(Q_coarse) @ pool(K_coarse).T)   # 只在 seed stage 做
route_fine   = lift_parent_pairs_to_children(route_coarse)
O_fine       = SparseAttention(Q_fine, K_fine, V_fine, route_fine)
```

`c2f_route_only` 省掉 fine stage 的 centroid pooling、coarse score matrix、Top-K **以及 VSA coarse-output branch**；`c2f_route` 为了保留精确的当前 `O_c`，仍要计算当前 `Qc/Kc/Vc/Ac`，只把 sparse fine attention 使用的 route 换成上一 stage 的 route。两者都**没有**省 QKV projection、当前 V、稀疏 QK/softmax/AV、output projection、cross-attention和 MLP。route lift 的独立实测约 0.031 ms/层，在线 VSA route builder 约 0.209 ms/层，但只有在两边 coarse-output 处理一致时，这个算子差值才可转化为公平的方法加速。

> **公平性更正（2026-09-08）：** 下表及 4.1 的 resident 数字是“完整在线 VSA（含 `O_c`）”对“`c2f_route_only`（不含 `O_c`）”。它们可用于定位耗时、给出删掉整个 coarse 分支的乐观上界，但**不能作为论文中的 C2F-vs-VSA 公平主结果，也不能用于宣称相同模型函数下 C2F 加速了多少**。此前把该表称为 C2F 独立公平增量是不准确的，现已撤回。

| 非同结构诊断（仅乐观上界） | 在线 VSA：含 `O_c` | C2F route-only：无 `O_c` | 时间比 |
|---|---:|---:|---:|
| Spatial-221 | 0.421s | 0.412s | 1.022× |
| Spatial-421 | 0.413s | 0.413s | 1.001× |
| Spatial-441 | 0.404s | 0.396s | 1.020× |
| ST-421 | 0.404s | 0.398s | 1.016× |

这组非同结构诊断的差值约 0.1–2.2%，不能解释成公平的 C2F 独立增量。它只说明：即便把整个 coarse branch 都删掉，端到端可省的时间仍很小；RDD 降 token 才是当前加速的大头。Spatial-421 还受 transferred route 的整数取整影响，实际 pair density 约为 25%。

### 4.1 20% 密度复核：25% 是一部分原因，不是全部原因

旧实验已经包含 `route_density=18.75%` 的校准配置：在 832×448 Spatial-421 上，lift 后实际为 19.12%，而请求 20% 的朴素 parent-to-children 展开实际为 25.00%。2026-09-08 又在同一个 resident FastWan 进程里做了 10 轮 warmup、100 轮交错顺序配对测试。测试时 GPU 上另有训练负载，所以绝对时间不是论文最终值；同进程交错配对仍可判断增量方向。

| 非同结构 Spatial-421 诊断 | 实际 C2F density | denoiser 中位数 | 时间比（不是公平方法加速） |
|---|---:|---:|---:|
| 在线 VSA-20 | 每 stage 自己 Top-K | 0.87565s | 1.000× |
| C2F，请求 20% | 25.00% | 0.87027s | 1.006×（-0.62%） |
| C2F，校准 18.75% | 19.12% | 0.86493s | **1.012×（-1.22%）** |

校准 route-only 版本相对完整在线 VSA 的 100 个配对样本中有 68 个更快；配对均值节省 12.94 ms，bootstrap 95% 区间约为 6.27–19.93 ms。由于两边是否包含 `O_c` 不一致，该显著性只证明这两条具体执行路径时间不同，不证明 C2F route 本身带来 12.94 ms。25% 与 19.12% 两个 C2F route-only 版本之间的直接差约 0.61%，在当前共享 GPU 噪声下区间仍跨 0。

### 4.2 VSA coarse output 到底是什么，C2F 为什么不能无代价保留

VSA 先把每个 64-token cube 的 Q/K/V 分别平均成 `Qc/Kc/Vc`，再算 `Ac=softmax(Qc Kc^T / sqrt(d))`。`Ac` 一方面给 fine sparse attention 选 Top-K route，另一方面产生 `Oc=Ac Vc`；`Oc` 被复制回 cube 内的 64 个 token，并通过学习到的 coarse gate 加到 fine 输出上：`O = Of + Gc * Oc`。它相当于一条便宜的、低分辨率但全局可见的上下文通道。

仓库已有三条清楚区分的路径：① `vsa`：在线生成当前 route，并计算当前 `O_c`；② `c2f_route`：fine sparse attention 使用上一 stage route，但仍计算当前 `Qc/Kc/Vc/Ac/Oc`，这是与完整 VSA 做**结构公平速度隔离**的正确路径；③ `c2f_route_only`：既复用 route 又删除当前 coarse branch，是待重新训练的不同架构。另有 `c2f_route_reuse_coarse`：复用旧 `Ac`，pool 当前 V 并算 `Ac_old Vc_current`，它保留近似全局通道，但也不是原 VSA 的当前 `Ac`。后两种都必须经过 RDD/C2F finetune，再在匹配质量下做系统级比较。

因此正式实验必须拆成两层。第一层固定同一个 checkpoint、同一个 density、同一个 sparse executor，并比较 `vsa` 与 `c2f_route`；它只回答“换 route 来源能省多少”。第二层分别训练完整 VSA、C2F route-only 或 reuse-coarse 版本，在相同训练预算和相近 VBench/FVD/人工质量下比较端到端速度；它回答“新架构的最佳质量—速度折中”。不能跨这两层引用数字。

VSA 论文的 120M 预训练消融中，只有 fine 输出的 loss 为 0.13296，coarse+fine 为 0.13162，绝对改善 0.00134、相对约 1.0%。作者据此称全局信息必要，但没有证明 coarse output 单项会带来巨大的视觉指标提升；它是稳定有益的结构，不是已证实的决定性质量来源。

### 4.3 这里已经自己写了什么 kernel；什么叫静态 route executor

当前 C2F 自己写了 Triton route-lift kernel：每个 program 负责一个 `(batch, head, fine-query-block)`，找到它的 coarse parent，读取 parent 的已选 coarse keys，再把这些 keys 展开成 fine child-block 索引。它不计算 fine `QcKc^T`，输出直接喂给 FastVideo 官方 block-sparse attention executor。也就是说，**route 生成是我们的 kernel，真正执行 fine QK/softmax/AV 的主体仍是 FastVideo kernel**。

所谓静态不是把某个视频的 route 永久写死进源码。它是：每个样本在 stage boundary 得到 route 后，先把它整理成固定 K、连续存储、按工作量分桶的 GPU schedule；在这个 fine stage 的后续多个 timestep 中重复使用同一 schedule。这样 sparse executor 可以删掉每行动态 `count` 分支，固定循环次数，预先组织连续 KV block，复用 workspace，并用 CUDA Graph 重放。3-NFE 且每 stage 只有一次 forward 时几乎无法摊薄 setup；在每个 stage 有很多 denoising steps 时才更有价值。

公平口径应同时报告：①在线 VSA+官方 executor；②C2F+同一个官方 executor；③C2F+静态 executor；④可选的 VSA route-cache/static 对照。③利用的是 C2F 的“route 提前已知且可跨本 stage 多步复用”这一方法属性，作为系统级最佳实现是公平的；但如果只想隔离算法差异，就必须先用②比较。若这一点以及是否固定跨 timestep route，必须在论文中明说。

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_vsa20_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_vsa20.mp4"></video>
<h4>Temporal RDD + 在线 VSA-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_temporal_vsa20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,2) → (1,2) → (1,1)</code><br><code>4×56×104 → 4×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.624s</strong>；1.39×，延迟减少 27.9%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.144s；与本行 denoiser 合计 <strong>1.769s</strong>；1.14×，延迟减少 12.1%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20.mp4"></video>
<h4>Temporal RDD + C2F-Route-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_temporal_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,2) → (1,2) → (1,1)</code><br><code>4×56×104 → 4×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=20.9%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.607s</strong>；1.42×，延迟减少 29.8%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.140s；与本行 denoiser 合计 <strong>1.748s</strong>；1.15×，延迟减少 13.2%</td></tr>
</table>
</article>
</div>

## 5. 方法 C：StepReuse-Out(k)——相邻同形状 NFE 复用什么

当前实现复用的是 self-attention 经过 output projection 后的张量，不是整个 Transformer block，也不是旧 V。cache hit 时仍执行当前 timestep 的 residual gate、cross-attention 和 MLP。

```text
if same_resolution_stage and calls_since_refresh < k:
    self_attention_output = cached_projected_output
else:
    self_attention_output = to_out(SparseAttention(Q, K, V))
    cache = self_attention_output
```

`421` 三个 shape 都不同，所以跨 stage 必须 miss；`441` 的前两个 NFE shape 相同，第二个能命中。它只省 self-attention 分支，且小 stage 本来就处于 launch/投影主导区，因此 20% keep 时收益可能只有几毫秒；12.5% 与 SmallQ 配合后才更明显。

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20.mp4"></video>
<h4>Temporal RDD + C2F-Route-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_temporal_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,2) → (1,2) → (1,1)</code><br><code>4×56×104 → 4×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=20.9%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.607s</strong>；1.42×，延迟减少 29.8%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.140s；与本行 denoiser 合计 <strong>1.748s</strong>；1.15×，延迟减少 13.2%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_fair_accel_20260907/final_temporal_c2f20_reuse2.mp4"></video>
<h4>Temporal RDD + C2F-Route-20 + StepReuse-Out(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>final_temporal_c2f20_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,2) → (1,2) → (1,1)</code><br><code>4×56×104 → 4×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=20.9%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.550s</strong>；1.57×，延迟减少 36.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.139s；与本行 denoiser 合计 <strong>1.690s</strong>；1.19×，延迟减少 16.0%</td></tr>
</table>
</article>
</div>

## 6. 方法 D（新）：SmallQ-32 解决小 stage 吃不满 GPU

FastVideo 在这台 RTX PRO 6000 Blackwell（SM120）上没有走仓库里的 `sm100a` 专用 CUDA 路径，而是走可移植 Triton fallback。原 kernel 每个 `(64 query tokens, head)` 启动一个 program。832×448 的 Spatial-421 第一 stage 只有 16 个 64-token blocks：12 heads 只有约 192 个 programs，接近 188 个 SM 的单波工作；Spatiotemporal-421 只有 8×12=96 个，连一波都喂不满。其他 norm/projection/MLP kernel 也因 batch=1、token 少而落入延迟地板。

SmallQ-32 不改 VSA 的 64-token route 和 64-token key block，只把每个 routed query block 拆成两个 32-token work tiles：

```python
fine_q_block  = tl.program_id(0)          # 32-token work tile
query_start   = fine_q_block * 32
route_q_block = query_start // 64         # 两个 work tiles 共用同一 route
for key_block in route[route_q_block]:
    online_softmax_qk_av(query[query_start:query_start+32], key_block)
```

这让 query-side program 数翻倍，同时完整保留每个 token 看见的 keys。executor 只在 padded tokens≤8192 时启用；full stage 自动回到官方 native kernel，避免长序列上过度切块。

| 64-token blocks | Native Triton | SmallQ-16 | SmallQ-32 | Q32 加速 | Q32 max abs |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.1399 ms | 0.0358 ms | 0.0359 ms | **3.90×** | 0.000000 |
| 16 | 0.1388 ms | 0.0523 ms | 0.0444 ms | **3.13×** | 0.000000 |
| 32 | 0.1580 ms | 0.0997 ms | 0.0740 ms | **2.14×** | 0.000000 |
| 56 | 0.2120 ms | 0.2230 ms | 0.1415 ms | **1.50×** | 0.000000 |
| 128 | 0.5525 ms | 0.9219 ms | 0.4918 ms | **1.12×** | 0.000000 |
| 182 | 0.9010 ms | 1.7461 ms | 0.9077 ms | **0.99×** | 0.000000 |

| resident-DiT 路径 | Native executor | SmallQ-32 | 端到端增量 |
|---|---:|---:|---:|
| Spatial-221 | 0.412s | 0.413s | 0.999× |
| Spatial-421 | 0.412s | 0.408s | 1.008× |
| Spatial-441+Reuse | 0.366s | 0.362s | 1.011× |
| ST-421 | 0.397s | 0.390s | 1.018× |
| ST-441+Reuse | 0.365s | 0.361s | 1.011× |

算子表给出明确 crossover：小到中等 block 数时 Q32 更快；182 blocks 的 full stage 接近 1×，所以不能把它无脑用于所有 shape。端到端 421 的 q32/native latent 差异也单独记录；只有该值为 0 或 BF16 舍入量级，才把它归为等价 kernel 优化。
50 次 resident-DiT 端到端消融进一步说明真实幅度：Spatial-421 约快 0.8%，Spatial-441 约快 1.1%，ST-421 约快 1.8%；Spatial-221 差异仅 -0.08%，属于无收益区。SmallQ 是小但真实的尾部优化，不能把算子级 2–4×写成整模型 2–4×。

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20.mp4"></video>
<h4>Spatial RDD 4→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial421_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (2,1) → (1,1)</code><br><code>8×14×26 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.454s</strong>；1.91×，延迟减少 47.5%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.896s</strong>；1.22×，延迟减少 18.0%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20_q32.mp4"></video>
<h4>Spatial RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (2,1) → (1,1)</code><br><code>8×14×26 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>旧执行路径；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.456s</strong>；1.90×，延迟减少 47.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.443s；与本行 denoiser 合计 <strong>1.898s</strong>；1.22×，延迟减少 17.9%</td></tr>
</table>
</article>
</div>

## 7. 方法 E（关键系统修复）：让 DiT 真正常驻 GPU

FastVideo 有两个彼此独立的 offload 开关。旧脚本只写了 `dit_cpu_offload=False`，但 `dit_layerwise_offload` 的库内默认值仍为 `True`。于是每个 Transformer block 在每次 NFE 前都从 pinned CPU memory 搬到 GPU：

```text
旧：每个 NFE × 每一层：CPU pinned weight --H2D--> GPU --compute--> 下一层
新：加载一次：全部 DiT weights --> GPU；每个 NFE 只做 compute
```

旧 profiler 每条 3-NFE 轨迹记录到 **2610 次 pinned H2D copy / 约 163 ms self CUDA time**；resident profiler 中该条目从榜首消失。full-resolution 大 GEMM 能部分遮住传输，所以固定基线只快约 1%；coarse stage 计算太短，无法遮住传输，因此 421 的前两阶段分别从约 85/83 ms 降到约 58/66 ms。这个修复不改变模型、token、route、keep 或输出，只消除错误的部署开销。

同一 seed 的固定分辨率基线做了 resident/offload 配对等价性检查：latent `max_abs=0.0`、`mean_abs=0.0`、`RMSE=0.0`，两个 MP4 的 SHA-256 也完全相同。对应 denoise 为 0.858s（resident）与 0.865s（offload）。因此这是一项纯执行修复，不是换算法换画质。

在 fox 50 次正式结果中：Spatial-221=0.412s，Spatial-421=0.407s，421 相对 221 为 **1.012×**；441+Reuse=0.361s；441+12.5%=0.347s。

## 8. Exact boundary cleanup：正确，但不是主要加速来源

`torch_nearest` 用一次 nearest interpolation 代替三段 `repeat_interleave`，同时保持原 Gaussian 和 BF16 运算顺序。smoke 中 fixed/221/421 的最终 latent 都是 `max_abs=mean_abs=RMSE=0`。但正式 50 次中 native 与 nearest 只差约 0.0–0.1%，因为 boundary reconstruction 本来不到 1 ms；它是干净的实现整理，不应包装成主要贡献。尝试过的 fused Triton boundary 在真实 sigma 下出现 BF16 差异，并把端到端 latent 最大误差放大到约 1.59，因此已判定失败，不进入主路径。

## 9. 这次找到的全部低效率原因与处理

| 原因 | 观测 | 已做处理 / 下一步 |
|---|---|---|
| 意外开启逐层 CPU offload | 旧 profiler：2610 次 pinned H2D，约 163 ms；小 stage 无法覆盖传输 | 显式 `dit_layerwise_offload=False`；1.3B DiT 常驻，832 约 9.3 GB nvidia-smi residency |
| 小 stage GPU 并行度不足 | 832 的 421/441 原生时间几乎不随 token 再降；放大到 1280/1664 后自然反超 | SmallQ-32；服务吞吐场景还可把多个请求的 coarse stage 合批 |
| VSA 64-token padding | ST-421 seed 仅 182 有效 tokens，却 padding 到 512 | q32 增并行；最终可训练 16/32-token route tiles |
| Top-K 整数取整 | 832 Spatial-421 请求 20%，实际约 25%；ST-421 更高 | 同时报告 requested/actual；测 18.75% 与 12.5% |
| SM120 无 FastVideo sm100a fastpath | 当前走 Triton64 fallback | 新增 SM120 友好的 q32 executor；full stage 保留官方路径 |
| C2F 只省 router | QKV/cross-attn/MLP 全保留 | 不夸大 C2F；把它与 RDD token reduction、Reuse 分开报告 |
| StepReuse 命中范围窄 | 421 shape 全变所以 miss；441 仅第二步 hit | 报 cache hit；只在 441/221 等连续同 shape 路径启用 |
| 最后一次 full stage 主导 | 所有 RDD 路径的第三 NFE 都完整 | 更长 NFE 模型中 RDD 潜力更大；本报告不直接外推 |
| VAE 固定成本 | 832×448×29 的 VAE 约 1.44s，高于 baseline denoise 约 0.87s | denoiser 与 sample+VAE 双表；若做系统论文需另加 tiled/compiled VAE |
| 冷启动/shape 编译 | Triton 首次 shape 会 JIT/autotune，换 bucket 可再次触发 | 所有正式数字逐 shape warm；部署预热固定 resolution buckets |
| 旧计时同步和 GPU 漂移 | 逐 stage host sync、不同时间/GPU 比较曾改变排名 | CUDA Event + 单次末尾 sync；同进程交错顺序；20–50 次中位数/IQR/MAD |
| checkpoint 不懂 RDD 空间 transition | aggressive spatial/ST 出现块状伪影 | 速度 plumbing 已验证；画质结论必须等相同预算微调 |
| coarse probability 输出复用 | training-free 加回全局 coarse 分支反而慢 | 作为失败对照保留；若要用，需联合训练 gate/fused kernel |

## 10. 历史审计：开启 layerwise offload 时的稳定数值

下面先回答旧异常本身。这里还未使用 SmallQ，是为了只看 resolution 对 GPU 的影响。

### 10.1 Spatial

| 方法 | stage | attention | route 请求/实际 | reuse | kernel / boundary | DiT | denoiser 中位数 | IQR | 加速 | 延迟减少 |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 固定全分辨率 FastWan + VSA-20（基线） | `(1,1) → (1,1) → (1,1)` | `vsa` | 20.0% / — | 1 | `native` / `native` | layerwise-offload | 0.866s | 0.001s | 1.00× | 0.0% |
| Spatial RDD 2→2→1 + C2F-20 | `(2,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `native` / `native` | layerwise-offload | 0.462s | 0.043s | 1.87× | 46.6% |
| Spatial RDD 4→2→1 + C2F-20 | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.0% | 1 | `native` / `native` | layerwise-offload | 0.469s | 0.049s | 1.85× | 45.9% |
| Spatial RDD 4→2→1 + C2F-18.75 | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 18.8% / 19.1% | 1 | `native` / `native` | layerwise-offload | 0.468s | 0.057s | 1.85× | 45.9% |
| Spatial RDD 4→4→1 + C2F-20 | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 1 | `native` / `native` | layerwise-offload | 0.474s | 0.070s | 1.83× | 45.3% |
| Spatial RDD 4→4→1 + C2F-20 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 2 | `native` / `native` | layerwise-offload | 0.477s | 0.070s | 1.82× | 44.9% |
| Spatial RDD 4→4→1 + C2F-12.5 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 12.5% / 12.6% | 2 | `native` / `native` | layerwise-offload | 0.447s | 0.066s | 1.94× | 48.5% |
| Spatial RDD 4→2→1 + C2F-20 masked sdpa | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 24.7% | 1 | `masked_sdpa` / `native` | layerwise-offload | 0.483s | 0.051s | 1.79× | 44.2% |
| Spatial RDD 4→2→1 + C2F-20 dense seed | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 24.2% | 1 | `dense_sdpa` / `native` | layerwise-offload | 0.491s | 0.052s | 1.77× | 43.4% |
| Spatial RDD 4→2→1 + C2F-20 reuse coarse | `(4,1) → (2,1) → (1,1)` | `c2f_route_reuse_coarse` | 20.0% / 25.0% | 1 | `native` / `native` | layerwise-offload | 0.498s | 0.055s | 1.74× | 42.5% |

### 10.2 Spatiotemporal

| 方法 | stage | attention | route 请求/实际 | reuse | kernel / boundary | DiT | denoiser 中位数 | IQR | 加速 | 延迟减少 |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 固定全分辨率 FastWan + VSA-20（基线） | `(1,1) → (1,1) → (1,1)` | `vsa` | 20.0% / — | 1 | `native` / `native` | layerwise-offload | 0.868s | 0.002s | 1.00× | 0.0% |
| Spatiotemporal RDD 2→2→1 + C2F-20 | `(2,2) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `native` / `native` | layerwise-offload | 0.468s | 0.050s | 1.86× | 46.1% |
| Spatiotemporal RDD 4→2→1 + C2F-20 | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 26.4% | 1 | `native` / `native` | layerwise-offload | 0.480s | 0.053s | 1.81× | 44.7% |
| Spatiotemporal RDD 4→2→1 + C2F-12.5 | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 12.5% / 14.0% | 1 | `native` / `native` | layerwise-offload | 0.465s | 0.056s | 1.87× | 46.4% |
| Spatiotemporal RDD 4→4→1 + C2F-20 | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 20.0% / 25.2% | 1 | `native` / `native` | layerwise-offload | 0.485s | 0.054s | 1.79× | 44.1% |
| Spatiotemporal RDD 4→4→1 + C2F-20 + Reuse(2) | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 20.0% / 25.3% | 2 | `native` / `native` | layerwise-offload | 0.480s | 0.060s | 1.81× | 44.7% |
| Spatiotemporal RDD 4→4→1 + C2F-12.5 + Reuse(2) | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 12.5% / 13.0% | 2 | `native` / `native` | layerwise-offload | 0.473s | 0.063s | 1.83× | 45.5% |
| Spatiotemporal RDD 4→2→1 + C2F-20 masked sdpa | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 26.5% | 1 | `masked_sdpa` / `native` | layerwise-offload | 0.489s | 0.072s | 1.77× | 43.7% |
| Spatiotemporal RDD 4→2→1 + C2F-20 dense seed | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 24.3% | 1 | `dense_sdpa` / `native` | layerwise-offload | 0.495s | 0.055s | 1.75× | 42.9% |
| Spatiotemporal RDD 4→2→1 + C2F-20 reuse coarse | `(4,4) → (2,2) → (1,1)` | `c2f_route_reuse_coarse` | 20.0% / 26.4% | 1 | `native` / `native` | layerwise-offload | 0.502s | 0.054s | 1.73× | 42.1% |

832 下原生 421 不一定快过 221，原因不是 FLOPs 判断错，而是二者都已撞上约 0.09s 的 coarse-stage 延迟地板；421 还承受更严重 padding/Top-K 取整。`masked_sdpa`、`dense seed` 和 training-free coarse-output 复用都没有改善整条轨迹，属于有价值的失败对照。

## 11. 分辨率放大实验：421 会不会自然反超 221

| 输出 | full baseline | 221 | 421 | 441+12.5%+Reuse | 421 相对 221 |
|---|---:|---:|---:|---:|---:|
| 832×448 | 0.856s | 0.412s | 0.407s | 0.347s | **1.012×** |
| 1280×704 | 2.682s | 1.188s | 1.088s | 0.907s | **1.092×** |
| 1664×896 | 4.804s | 2.130s | 1.935s | 1.526s | **1.101×** |

这就是用户提出的验证：分辨率越大，421 seed stage 的绝对 token 数也变大，GPU 能获得多波并行工作，理论 FLOPs 优势终于能显现。SmallQ-32 的作用是把这个 crossover 向 832×448 推回去，而不是伪造更少的 attention pairs。

## 12. 最终主表：resident DiT + SmallQ + C2F + Reuse

| 方法 | stage | attention | route 请求/实际 | reuse | kernel / boundary | DiT | denoiser 中位数 | IQR | 加速 | 延迟减少 |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 固定全分辨率 FastWan + VSA-20（基线） | `(1,1) → (1,1) → (1,1)` | `vsa` | 20.0% / — | 1 | `native` / `native` | resident | 0.856s | 0.000s | 1.00× | 0.0% |
| fixed vsa20 + exact-nearest | `(1,1) → (1,1) → (1,1)` | `vsa` | 20.0% / — | 1 | `native` / `torch_nearest` | resident | 0.856s | 0.000s | 1.00× | -0.0% |
| Spatial RDD 2→2→1 + C2F-20 + SmallQ-32 | `(2,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `triton_q32` / `native` | resident | 0.412s | 0.001s | 2.08× | 51.9% |
| Spatial RDD 2→2→1 + C2F-20 + SmallQ-32 + exact-nearest | `(2,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `triton_q32` / `torch_nearest` | resident | 0.412s | 0.001s | 2.08× | 51.9% |
| Spatial RDD 4→2→1 + C2F-20 + SmallQ-32 | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.0% | 1 | `triton_q32` / `native` | resident | 0.407s | 0.002s | 2.10× | 52.5% |
| Spatial RDD 4→2→1 + C2F-20 + SmallQ-32 + exact-nearest | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.0% | 1 | `triton_q32` / `torch_nearest` | resident | 0.407s | 0.003s | 2.10× | 52.4% |
| Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 2 | `triton_q32` / `native` | resident | 0.361s | 0.002s | 2.37× | 57.9% |
| Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2) + exact-nearest | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 2 | `triton_q32` / `torch_nearest` | resident | 0.361s | 0.003s | 2.37× | 57.8% |
| Spatial RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2) + exact-nearest | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 12.5% / 12.6% | 2 | `triton_q32` / `torch_nearest` | resident | 0.347s | 0.002s | 2.47× | 59.4% |
| Spatiotemporal RDD 2→2→1 + C2F-20 + SmallQ-32 + exact-nearest | `(2,2) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `triton_q32` / `torch_nearest` | resident | 0.382s | 0.003s | 2.24× | 55.3% |
| Spatiotemporal RDD 4→2→1 + C2F-20 + SmallQ-32 + exact-nearest | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 26.4% | 1 | `triton_q32` / `torch_nearest` | resident | 0.388s | 0.004s | 2.20× | 54.6% |
| Spatiotemporal RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2) + exact-nearest | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 20.0% / 25.3% | 2 | `triton_q32` / `torch_nearest` | resident | 0.360s | 0.003s | 2.37× | 57.9% |
| Spatiotemporal RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2) + exact-nearest | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 12.5% / 13.0% | 2 | `triton_q32` / `torch_nearest` | resident | 0.347s | 0.003s | 2.47× | 59.4% |

Spatial-421 为 **0.407s / 2.10×**；441+q32+Reuse 为 **0.361s / 2.37×**；aggressive 12.5% 为 **0.347s / 2.47×**。最终是否采用 aggressive 只能由 RDD-finetuned checkpoint 的 VBench/FVD/人工偏好决定。

### 12.1 三个 prompt 的 50 次排序是否一致

| prompt | Spatial-221 | Spatial-421 | 421/221 | 441+Reuse | 441+12.5% |
|---|---:|---:|---:|---:|---:|
| 狐狸 | 0.412s | 0.407s | **1.012×** | 0.361s | 0.347s |
| 城市 | 0.413s | 0.410s | **1.009×** | 0.363s | 0.350s |
| 汽车 | 0.414s | 0.410s | **1.009×** | 0.364s | 0.350s |

三个 prompt 的 Spatial-421 都比 Spatial-221 快，差距约 0.9–1.2%，并且单次序列的 MAD 仅约 0.3–1.1 ms；这次反超不是单次噪声。Spatiotemporal-421 仍略慢于 ST-221：二者 seed stage 都已落到相同延迟地板，而 421 的整数 Top-K/route lift 使 full stage 实际密度更高。这一例外也必须如实报告。

### 12.2 修复前结果保留作可审计对照

| 方法 | stage | attention | route 请求/实际 | reuse | kernel / boundary | DiT | denoiser 中位数 | IQR | 加速 | 延迟减少 |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 固定全分辨率 FastWan + VSA-20（基线） | `(1,1) → (1,1) → (1,1)` | `vsa` | 20.0% / — | 1 | `native` / `native` | layerwise-offload | 0.866s | 0.001s | 1.00× | 0.0% |
| Spatial RDD 2→2→1 + C2F-20 | `(2,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `native` / `native` | layerwise-offload | 0.451s | 0.036s | 1.92× | 47.9% |
| Spatial RDD 4→2→1 + C2F-20 | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.0% | 1 | `native` / `native` | layerwise-offload | 0.454s | 0.036s | 1.91× | 47.5% |
| Spatial RDD 4→2→1 + C2F-20 + SmallQ-32 | `(4,1) → (2,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.0% | 1 | `triton_q32` / `native` | layerwise-offload | 0.456s | 0.037s | 1.90× | 47.4% |
| Spatial RDD 4→4→1 + C2F-20 | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 1 | `native` / `native` | layerwise-offload | 0.448s | 0.032s | 1.93× | 48.3% |
| Spatial RDD 4→4→1 + C2F-20 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 2 | `native` / `native` | layerwise-offload | 0.438s | 0.025s | 1.98× | 49.4% |
| Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 20.0% / 25.1% | 2 | `triton_q32` / `native` | layerwise-offload | 0.443s | 0.046s | 1.96× | 48.9% |
| Spatial RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2) | `(4,1) → (4,1) → (1,1)` | `c2f_route_only` | 12.5% / 12.6% | 2 | `triton_q32` / `native` | layerwise-offload | 0.423s | 0.048s | 2.05× | 51.1% |
| Spatiotemporal RDD 2→2→1 + C2F-20 | `(2,2) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 21.5% | 1 | `native` / `native` | layerwise-offload | 0.449s | 0.036s | 1.93× | 48.2% |
| Spatiotemporal RDD 4→2→1 + C2F-20 | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 26.4% | 1 | `native` / `native` | layerwise-offload | 0.456s | 0.036s | 1.90× | 47.4% |
| Spatiotemporal RDD 4→2→1 + C2F-20 + SmallQ-32 | `(4,4) → (2,2) → (1,1)` | `c2f_route_only` | 20.0% / 26.4% | 1 | `triton_q32` / `native` | layerwise-offload | 0.444s | 0.034s | 1.95× | 48.8% |
| Spatiotemporal RDD 4→4→1 + C2F-20 + Reuse(2) | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 20.0% / 25.3% | 2 | `native` / `native` | layerwise-offload | 0.442s | 0.053s | 1.96× | 49.0% |
| Spatiotemporal RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2) | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 20.0% / 25.3% | 2 | `triton_q32` / `native` | layerwise-offload | 0.430s | 0.046s | 2.01× | 50.4% |
| Spatiotemporal RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2) | `(4,4) → (4,4) → (1,1)` | `c2f_route_only` | 12.5% / 13.0% | 2 | `triton_q32` / `native` | layerwise-offload | 0.418s | 0.039s | 2.07× | 51.7% |

上表是同样 SmallQ/C2F/Reuse、但仍开启 FastVideo 默认 layerwise offload 的历史数据。它解释了为什么旧实验中的 421 排名会随 GPU/提示波动；不再作为论文主速度表。

## 13. 为什么加上 VAE 后看起来只快一点

fox resident 主基线为 denoise 0.856s + 独立实测同形状 VAE 1.447s = 2.302s。即使把 denoiser 变成 0，当前 VAE 不动时端到端理论上最多也只有 1.59×。这不是次数少造成的：三 prompt 的 VAE 都稳定在相同量级，且新代码在 VAE 前后显式同步。

VAE 是固定的像素解码工作，RDD/C2F/Reuse/SmallQ 都只改 Transformer denoising。论文应同时报告：纯 denoiser（证明算法/attention 加速）和 sample+VAE（说明用户实际等待时间）。如果系统目标是端到端，下一项正交优化应是编译/tiling VAE，而不是继续把 denoiser 数字包装成总时间。

## 14. 狐狸 prompt：从 mild 到 aggressive

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/fixed_vsa20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/fixed_vsa20.mp4"></video>
<h4>固定全分辨率 FastWan + VSA-20（基线）</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>fixed_vsa20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,1) → (1,1) → (1,1)</code><br><code>8×56×104 → 8×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.856s</strong>；1.00×，延迟减少 0.0%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.447s；与本行 denoiser 合计 <strong>2.302s</strong>；1.00×，延迟减少 0.0%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial221_c2f20.mp4"></video>
<h4>Spatial RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,1) → (2,1) → (1,1)</code><br><code>8×28×52 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.5%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.412s</strong>；2.08×，延迟减少 51.9%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.854s</strong>；1.24×，延迟减少 19.3%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial421_c2f20_q32.mp4"></video>
<h4>Spatial RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (2,1) → (1,1)</code><br><code>8×14×26 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.407s</strong>；2.10×，延迟减少 52.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.443s；与本行 denoiser 合计 <strong>1.850s</strong>；1.24×，延迟减少 19.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.1%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.361s</strong>；2.37×，延迟减少 57.8%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.445s；与本行 denoiser 合计 <strong>1.806s</strong>；1.27×，延迟减少 21.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/spatial441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=12.6%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.347s</strong>；2.47×，延迟减少 59.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.789s</strong>；1.28×，延迟减少 22.1%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st221_c2f20.mp4"></video>
<h4>Spatiotemporal RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,2) → (2,2) → (1,1)</code><br><code>4×28×52 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.5%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.382s</strong>；2.24×，延迟减少 55.3%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.445s；与本行 denoiser 合计 <strong>1.828s</strong>；1.26×，延迟减少 20.6%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st421_c2f20_q32.mp4"></video>
<h4>Spatiotemporal RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (2,2) → (1,1)</code><br><code>2×14×26 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=26.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.388s</strong>；2.20×，延迟减少 54.6%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.447s；与本行 denoiser 合计 <strong>1.835s</strong>；1.25×，延迟减少 20.3%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.3%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.360s</strong>；2.37×，延迟减少 57.9%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.802s</strong>；1.27×，延迟减少 21.6%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_fox832_g0/st441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=13.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.347s</strong>；2.47×，延迟减少 59.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.788s</strong>；1.28×，延迟减少 22.2%</td></tr>
</table>
</article>
</div>

零微调画质 pattern：固定分辨率基线正常；Spatial-221 仍能看出主体，但已存在严重色块、拖影和结构错位，不能称为可用画质；Spatial-421 通常接近失去主体，441/12.5% 多数完全崩坏。SmallQ-32 和 exact-nearest 已做逐元素等价验证，不是额外画质损失来源；根因是 FastWan checkpoint 未训练 RDD spatial transition/C2F/Reuse。

## 15. 城市行人 prompt

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/fixed_vsa20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/fixed_vsa20.mp4"></video>
<h4>固定全分辨率 FastWan + VSA-20（基线）</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>fixed_vsa20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,1) → (1,1) → (1,1)</code><br><code>8×56×104 → 8×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.857s</strong>；1.00×，延迟减少 0.0%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>2.300s</strong>；1.00×，延迟减少 0.0%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial221_c2f20.mp4"></video>
<h4>Spatial RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,1) → (2,1) → (1,1)</code><br><code>8×28×52 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.413s</strong>；2.07×，延迟减少 51.8%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.447s；与本行 denoiser 合计 <strong>1.860s</strong>；1.24×，延迟减少 19.3%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial421_c2f20_q32.mp4"></video>
<h4>Spatial RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (2,1) → (1,1)</code><br><code>8×14×26 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.410s</strong>；2.09×，延迟减少 52.2%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.446s；与本行 denoiser 合计 <strong>1.856s</strong>；1.24×，延迟减少 19.4%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.363s</strong>；2.36×，延迟减少 57.6%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.448s；与本行 denoiser 合计 <strong>1.811s</strong>；1.27×，延迟减少 21.4%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/spatial441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=12.7%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.350s</strong>；2.45×，延迟减少 59.1%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.792s</strong>；1.28×，延迟减少 22.1%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st221_c2f20.mp4"></video>
<h4>Spatiotemporal RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,2) → (2,2) → (1,1)</code><br><code>4×28×52 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.5%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.386s</strong>；2.22×，延迟减少 55.0%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.827s</strong>；1.26×，延迟减少 20.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st421_c2f20_q32.mp4"></video>
<h4>Spatiotemporal RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (2,2) → (1,1)</code><br><code>2×14×26 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=26.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.391s</strong>；2.19×，延迟减少 54.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.446s；与本行 denoiser 合计 <strong>1.838s</strong>；1.25×，延迟减少 20.2%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.363s</strong>；2.36×，延迟减少 57.7%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.804s</strong>；1.27×，延迟减少 21.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_city832_g3/st441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=13.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.350s</strong>；2.45×，延迟减少 59.2%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.790s</strong>；1.28×，延迟减少 22.1%</td></tr>
</table>
</article>
</div>

重点看黄色雨衣主体是否保持、霓虹反射是否变成大色块、人物跨帧是否断裂。mild 与 aggressive 必须逐项看，不能用同 seed 像素相似度代替感知质量。

## 16. 海岸汽车 prompt

<div class="rdd-grid">
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/fixed_vsa20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/fixed_vsa20.mp4"></video>
<h4>固定全分辨率 FastWan + VSA-20（基线）</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>fixed_vsa20</code></td></tr>
<tr><th>stage / latent</th><td><code>(1,1) → (1,1) → (1,1)</code><br><code>8×56×104 → 8×56×104 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>在线 VSA；route=20.0%；实际 C2F=—</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>native</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>native</code></td></tr>
<tr><th>denoiser</th><td><strong>0.858s</strong>；1.00×，延迟减少 0.0%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.444s；与本行 denoiser 合计 <strong>2.303s</strong>；1.00×，延迟减少 0.0%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial221_c2f20.mp4"></video>
<h4>Spatial RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,1) → (2,1) → (1,1)</code><br><code>8×28×52 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.5%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.414s</strong>；2.07×，延迟减少 51.8%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.446s；与本行 denoiser 合计 <strong>1.860s</strong>；1.24×，延迟减少 19.3%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial421_c2f20_q32.mp4"></video>
<h4>Spatial RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (2,1) → (1,1)</code><br><code>8×14×26 → 8×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.1%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.410s</strong>；2.09×，延迟减少 52.2%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.446s；与本行 denoiser 合计 <strong>1.856s</strong>；1.24×，延迟减少 19.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.3%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.364s</strong>；2.36×，延迟减少 57.5%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.442s；与本行 denoiser 合计 <strong>1.806s</strong>；1.27×，延迟减少 21.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/spatial441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatial RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>spatial441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,1) → (4,1) → (1,1)</code><br><code>8×14×26 → 8×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=12.7%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.350s</strong>；2.45×，延迟减少 59.2%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.445s；与本行 denoiser 合计 <strong>1.795s</strong>；1.28×，延迟减少 22.1%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st221_c2f20_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st221_c2f20.mp4"></video>
<h4>Spatiotemporal RDD 2→2→1 + C2F-20</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st221_c2f20</code></td></tr>
<tr><th>stage / latent</th><td><code>(2,2) → (2,2) → (1,1)</code><br><code>4×28×52 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=21.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.385s</strong>；2.23×，延迟减少 55.1%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.446s；与本行 denoiser 合计 <strong>1.832s</strong>；1.26×，延迟减少 20.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st421_c2f20_q32_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st421_c2f20_q32.mp4"></video>
<h4>Spatiotemporal RDD 4→2→1 + C2F-20 + SmallQ-32</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st421_c2f20_q32</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (2,2) → (1,1)</code><br><code>2×14×26 → 4×28×52 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=26.4%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=1)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.391s</strong>；2.19×，延迟减少 54.4%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.832s</strong>；1.25×，延迟减少 20.3%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st441_c2f20_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st441_c2f20_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-20 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f20_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=20.0%；实际 C2F=25.6%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.363s</strong>；2.36×，延迟减少 57.7%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.441s；与本行 denoiser 合计 <strong>1.804s</strong>；1.27×，延迟减少 21.5%</td></tr>
</table>
</article>
<article class="rdd-card">
<video controls preload="metadata" playsinline poster="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st441_c2f12p5_q32_reuse2_mid.jpg" src="../../eval_results/rdd_fastwan_smallq_optimized_20260908/rdd_smallq_coast832_g3/st441_c2f12p5_q32_reuse2.mp4"></video>
<h4>Spatiotemporal RDD 4→4→1 + C2F-12.5 + SmallQ-32 + Reuse(2)</h4>
<table class="rdd-config">
<tr><th>配置</th><td><code>st441_c2f12p5_q32_reuse2</code></td></tr>
<tr><th>stage / latent</th><td><code>(4,4) → (4,4) → (1,1)</code><br><code>2×14×26 → 2×14×26 → 8×56×104</code></td></tr>
<tr><th>采样</th><td>DMD 3 NFE，t=<code>1000/757/522</code>；<code>noise_filling</code></td></tr>
<tr><th>attention</th><td>C2F-Route-Only（无 VSA coarse output）；route=12.5%；实际 C2F=13.0%</td></tr>
<tr><th>复用 / kernel</th><td>StepReuse-Out(k=2)；<code>triton_q32</code></td></tr>
<tr><th>执行口径</th><td>resident-DiT 配对计时；boundary=<code>torch_nearest</code></td></tr>
<tr><th>denoiser</th><td><strong>0.350s</strong>；2.46×，延迟减少 59.3%</td></tr>
<tr><th>VAE / 组合总时间</th><td>VAE 为该配置/同形状实测 1.440s；与本行 denoiser 合计 <strong>1.789s</strong>；1.28×，延迟减少 22.1%</td></tr>
</table>
</article>
</div>

重点看车身轮廓、车轮、道路边界和海岸背景的跨帧连续性。三 prompt 共用相同配置表和计时协议，因此能判断失败是否为方法性 pattern，而不是单一 prompt 偶然。

## 17. 哪些方案成功，哪些失败

**成功且保持核心方法：**

1. 显式关闭 FastVideo 的 `dit_layerwise_offload` 消除了小 stage 每层权重搬运；这是最大的新系统修复。
2. 交错顺序 + CUDA Event 修复了旧计时噪声；更大输出独立证明 421 的 token 优势。
3. SmallQ-32 在不改 route/数值的条件下改善小 stage attention 并行度，并设置 shape threshold。
4. C2F 与 RDD/Reuse 能组合；12.5% 提供 aggressive 速度上界。
5. `torch_nearest` boundary 完全等价，但性能影响小；正确地降级为工程清理而非主要贡献。

**没有成功或暂不推荐：**

- 把稀疏 route 展开成完整 token mask 再用 SDPA：只有极小 8-block case 有机会，稍大立刻因 $N^2$ mask 变慢。
- tiny stage 直接 dense SDPA：算子可能快，但改变了 VSA 稀疏语义；端到端也未胜，不能作为公平主结果。
- training-free 重建 VSA coarse probability output：增加 pool/AV/broadcast/gate，当前未融合，端到端变慢；若保留应与模型一起训练 gate。
- 对所有 shape 使用 SmallQ-16：程序过多，长序列反而慢；Q32 crossover 更好。
- fused Triton noise-filling boundary：算子更快，但真实 sigma 下改变 BF16 舍入，端到端 latent 最大误差约 1.59；已拒绝进入主路径。
- 把 3-NFE FastWan 的结果直接外推到普通 30–50 NFE 模型：方向可能更有利，但必须实测。

## 18. 冷启动与实际部署

论文主表通常报告模型已加载、kernel 已编译后的 steady-state latency；本报告也遵循这一口径。但实际产品第一次遇到新 `(resolution, frame count, dtype, GPU architecture)` 组合时可能重新触发 Triton JIT/autotune，所以换输入形状确实可能再次慢。部署应预先 warm 常用 buckets、持久化 Triton cache，并把冷启动 p50/p95 单列。不能把首次编译偷偷混进某一个候选，也不能假装它永远不存在。

## 19. 目前能写进论文的结论

目前可以写：同一 FastWan checkpoint 与 DMD3 下，RDD 降低早期 token；C2F route-lift kernel 能复用上一尺度 route；StepReuse 只在连续同 shape stage 命中。**不能写** route-only 相对完整 VSA 的差值就是公平 C2F 加速，因为前者删除了 VSA coarse output。正式速度结论必须来自 `vsa` 对 `c2f_route` 的同结构对照，或来自分别训练后、匹配质量的系统对照。resident-DiT 修复、SmallQ 微基准和大分辨率实验只用于解释 421/221 crossover。

不能写：在同等视频质量下击败 FastWan/VSA。当前 FastWan 权重没有接受 RDD spatial/ST transition、C2F 和 Reuse 微调；现有视频只是系统接线与扰动 sanity check。就本次约定的 **training-free 实现、故障定位、稳定计时、消融和可视化**范围而言完成度为 100%；论文画质结论仍缺同预算微调、VBench/FVD/人工偏好和多 seed 置信区间。

## 20. 训练/微调后应做的最小实验

- 同一数据、更新步数、batch、LoRA rank/全参策略：FastWan+VSA vs RDD+VSA vs RDD+C2F vs RDD+C2F+Reuse。
- stage paths 至少保留 `221 / 421 / 441`；keep 至少 `20% / 12.5%`。
- 速度：steady-state denoiser、sample+VAE、冷启动、peak memory、tokens/实际 pair density。
- 质量：VBench 全维度、FVD/CLIP 或等价基准、三种以上 seed、人工 A/B。
- q32 kernel 不需为画质微调；它是等价 executor。C2F coarse branch/gate、aggressive sparsity 和 StepReuse 才需要训练适配。

## 21. 复现入口

- `rdd_video/benchmark_fastwan_paired_suite.py`：单进程、交错顺序、原始记录/IQR/MAD、可选 profiler/latent equality。
- `rdd_video/attention_analysis/rdd_sparse_attn_smallq_triton.py`：SmallQ-16/32 exact sparse executor。
- `rdd_video/attention_analysis/benchmark_small_stage_executor.py`：算子 crossover 与数值误差。
- `rdd_video/attention_analysis/rdd_noise_filling_triton.py`：保留的失败原型；默认路径不会调用。
- `rdd_video/experiments/fastwan_torch_nearest_boundary_suite.json`：resident/exact boundary 正式配置。
- `rdd_video/experiments/fastwan_smallq_optimized_suite.json`：所有主配置。
- `rdd_video/launch_fastwan_efficiency_audit_4gpu.sh`：原生 kernel 的 832/1280/1664 审计。
- `rdd_video/launch_fastwan_smallq_optimized_4gpu.sh`：q32 + 三 prompt 正式验证。
- `rdd_video/launch_fastwan_resident_final_4gpu.sh`：最终 resident-DiT、exact boundary、三 prompt 与 1280/1664 复现入口。
- `eval_results/rdd_fastwan_efficiency_audit_20260908/efficiency_audit_summary.json`：原生正式原始/汇总。
- `eval_results/rdd_fastwan_smallq_optimized_20260908/efficiency_audit_summary.json`：优化后正式原始/汇总。
- `eval_results/rdd_fastwan_nearest_boundary_formal_20260908/efficiency_audit_summary.json`：关闭 layerwise offload 后的三 prompt×50 + 1280×704×30 最终主表。

## 22. 参考

- [FastVideo 官方仓库](https://github.com/hao-ai-lab/FastVideo)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- [FastVideo VSA 文档](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

---

附注：RDD 的 ImageNet-256 受控 DiT 收敛/FID 与多分辨率 prior-art 对比仍保存在旧版结构化结果 `eval_results/rdd_fastwan_fair_accel_20260907/fair_acceleration_summary.json` 和项目专门 novelty 文档中；本页只聚焦这次 FastWan 系统效率任务，避免把图像 FID 与零微调视频速度混成一个结论。
