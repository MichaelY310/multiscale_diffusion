# RDD、f-DM、REG 与 NeurIPS 实验定位审计

最后更新：2026-08-10

范围：只讨论 ImageNet 2D / DiT。Video 部分未修改。

## 结论先行

1. **SiT-L/2 + REG 是一个很强且可信的 400K 训练参考。** REG 已被 NeurIPS 2025 接收为 Oral，不只是未经评审的 arXiv 工作。它在 SiT-L/2 上只增加约2M参数，在400K、无CFG、250步SDE采样下报告 FID 4.6；相同论文中的 SiT-XL/2 + REG 为3.4。RDD-XL/2当前400K、DDPM250、无CFG为7.06。因此，RDD不能再写成“400K最强训练加速方法”。
2. **REG与RDD大体正交，但不是完全无重叠。** RDD改变空间分辨率路径/forward mean；REG改变表示、输入token和辅助损失。二者在代码结构上可组合，但都试图让高噪声阶段更早获得低频/语义结构，收益可能部分冗余。没有组合实验前，只能写“plausibly complementary”。
3. **f-DM是RDD最直接的先行工作。** 它已经有阶段内连续插值、精确Gaussian reverse conditional、noise rescaling和边界noise resampling。旧文档中“f-DM只做离散跳变”“RDD首次精确后验/首次noise filling”的表述不成立。
4. **RDD仍有论文价值，但核心卖点需要变窄。** 可防守的差异是：在latent DiT/VP diffusion里使用加性resolution drift和单一effective-noise head；标准共享backbone；任意stage-local jump的显式更新；drift-corrected DPM-Solver；以及原生ImageNet上的受控收敛曲线与实际速度证据。
5. **最像RDD且已经成功的工作不止f-DM。** NeurIPS 2025 Spotlight DFM、ICLR 2026 LapFlow、ICLR 2024 Relay、ICLR 2024 MDM、NeurIPS 2024 DiMR，以及CVPR 2026 SFD都说明“渐进/多尺度/语义先行可以加速生成学习”已经是一条成熟研究线。RDD必须用因果矩阵说明自己的具体参数化为什么有用，而不是依靠宏观idea的新颖性。

## 1. SiT-L/2 + REG 到底是什么

### 1.1 SiT-L/2

SiT使用与DiT相同的Transformer backbone，但将离散DDPM训练放进stochastic interpolant / flow与diffusion统一框架。`L/2`表示Large规模、latent patch size为2：

- SiT-L/2 baseline：458M参数；
- SiT-L/2 + REG：460M参数；
- ImageNet-256输入经标准Stable Diffusion VAE编码为32×32×4 latent；
- REG论文使用batch 256，并按SiT/REPA协议训练。

### 1.2 REG做了什么

REG不是普通正则项。训练时它同时构造两条被加噪的状态：

1. VAE image latent `z_t`；
2. 预训练视觉foundation encoder（默认DINOv2）产生的一个global class token `cls_t`。

`cls_t`投影后作为额外一个token与image tokens拼接。模型同时预测image velocity与class-token velocity，并保留REPA式中间特征对齐损失：

`L_total = L_image_velocity + beta * L_cls_velocity + lambda * L_REPA`。

推理时不再调用DINO encoder；image latent与class token都从Gaussian noise开始并联合去噪。额外只有一个token，论文报告相对REPA约+0.30%参数、+0.38% FLOPs、+0.49% latency。

### 1.3 可信结果

REG论文在ImageNet-256、无CFG、SDE Euler–Maruyama 250 steps下报告：

| 模型 | 参数 | Updates | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| SiT-L/2 | 458M | 400K | 18.8 | 5.29 | 72.0 | 0.64 | 0.64 |
| SiT-L/2 + REPA | 458M | 400K | 约9.7--10.0 | 约5.2 | 约109 | 约0.69 | 约0.65 |
| **SiT-L/2 + REG** | 460M | 400K | **4.6** | 5.21 | 167.6 | 0.75 | 0.63 |
| SiT-XL/2 + REG | 677M | 400K | **3.4** | 4.87 | 184.1 | 0.76 | 0.64 |

这些数值与RDD不是完全同协议：REG是SiT/SDE sampler，RDD当前主点是DDPM250；实现和随机性也不同。但数据、VAE、batch、nominal updates和无CFG条件足够接近，REG必须出现在论文的400K上下文表中。

## 2. REG和RDD是否正交

### 2.1 结构上为什么可以组合

| 轴 | RDD | REG |
|---|---|---|
| 修改对象 | forward mean、resolution path、stage transition、solver | token/state representation、预测头、辅助alignment loss |
| 空间结构 | 32/16/8 latent grid分阶段 | 各阶段额外一个semantic token |
| 主要监督 | effective noise / drift-aware target | image velocity + class-token velocity + REPA |
| 外部预训练encoder | 不需要 | 训练时需要DINOv2 |
| 推理额外网络 | 不需要 | 不需要，只多一个生成token |

因此可以把每个RDD stage的image tokens与一个noised semantic token拼接，并给class token定义同一global time对应的noise schedule。

### 2.2 为什么不能直接说“完全正交”

- RDD的解释是coarse-to-fine/frequency curriculum让高噪声学习更容易；REG也让全局语义提前进入生成路径。
- 两者可能都在解决同一个低频语义瓶颈，组合收益可能小于相加。
- RDD每个global update同时训练三个resolution losses。若REG也对三个stage各算DINO alignment，训练开销和loss weighting会改变。
- class token在32/16/8三个阶段是否共享同一噪声、是否在边界resample、是否也需要drift，均不是REG原论文回答的问题。

结论：**理论/实现轴大体正交，优化作用可能重叠。** 最安全的论文措辞是“RDD modifies a different axis and is structurally composable with REG; empirical complementarity remains to be tested.”

## 3. 强、可信、与RDD较正交的方法

优先级按“可信度 × 组合干净程度 × 实验成本”排序。

| 方法 | 发表状态 | 与RDD正交度 | 建议 | 原因 |
|---|---|---:|---|---|
| **SiT / RDD-FM** | ECCV 2024 | 高 | **投稿首选组合** | 改训练objective/continuous-time path，不依赖外部encoder；最能证明RDD不限于DDPM |
| **REPA** | ICLR 2025 Oral | 高到中 | FM失败时的首选备选 | 只加中间表示对齐，工程量小；但语义curriculum有部分重叠 |
| **Limited-interval guidance** | NeurIPS 2024 | 很高 | **低成本必测** | 纯sampling-time，只在中间noise区间开CFG，可同时改善质量和Recall |
| **REG** | NeurIPS 2025 Oral | 中高 | 非必做、潜在最强 | 结构可组合且结果极强；但包含REPA、额外token/head，作用重叠和工程量更大 |
| **LightningDiT + VA-VAE** | CVPR 2025 Oral | 高到中 | 后续强组合 | VAE/训练recipe不同；需重新校准RDD latent down/up与drift，不能直接公平套用 |
| **FasterDiT** | NeurIPS 2024 | 中高 | 暂不作为主组合 | timestep PDF和v方向loss可组合，但RDD每步对三个stage取样，需stage-conditional重推权重 |
| **Autoguidance** | NeurIPS 2024 Oral | 很高 | 有现成弱checkpoint时可选 | 纯推理guidance；可用早期/小RDD作guide，但会增加每步forward成本 |
| **DiMR** | NeurIPS 2024 | 中 | 不作为最小投稿包 | 改多分辨率架构，但与RDD都依赖多尺度，机制重叠且实现成本大 |

不建议一次做RDD+REPA、RDD+REG、RDD+DiMR、RDD+FasterDiT。一个干净的正交组合比四个调参不足的组合更有说服力。

## 4. 与RDD idea相似的工作及结果位置

下表只做研究定位，不能作为iso-compute排名：架构、参数、训练空间、sampler、CFG和评估代码不同。

| 方法 | 与RDD相似处 | 关键区别 | 公开结果的含义 |
|---|---|---|---|
| **f-DM, ICLR 2023** | forward mean中进行progressive signal transformation；阶段内插值；边界噪声处理 | U-Net/pixel或变换空间、双预测、维度可变 | ImageNet-256公开表中DS约FID8.2、VQVAE变体约6.8；与RDD7.06同一量级，是最直接prior |
| **Relay Diffusion, ICLR 2024 Spotlight** | coarse-to-fine、跨分辨率SNR、block/noise bridge | 两个模型、image-space blurring diffusion、训练量巨大 | ImageNet有指导FID1.99（class-balanced 1.87）；最终质量强，但不是400K效率对照 |
| **MDM, ICLR 2024** | 单系统联合多分辨率denoising和progressive training | NestedUNet在每步联合处理多尺度，不改成RDD drift path | 证明多尺度联合监督本身就能改善优化，因此`d=0`控制必做 |
| **DiMR, NeurIPS 2024** | 多分辨率预测、低到高细化 | 架构/多尺度target创新，标准diffusion forward | 400 epochs/约500K updates报告无CFG4.87，800 epochs为4.50；强但计算和参数协议不同 |
| **DFM, NeurIPS 2025** | timestep分段、coarse-to-fine、单模型、训练收敛加速 | 每个decomposition level独立Flow Matching；可用Laplacian/spectral表示 | 宏观idea与RDD高度相似，且已被NeurIPS接收；RDD不能宣称首个渐进连续transport |
| **LapFlow, ICLR 2026** | time segments、多尺度latent、训练和采样都省计算 | Laplacian residual + causal MoT + flow paths | ImageNet-256 XL/2 600K无CFG14.38；绝对FID不及RDD400K，但接受度说明研究方向成立 |
| **SFD, CVPR 2026** | 让高层语义先于纹理，异步coarse-to-fine，极快收敛 | 额外Semantic VAE、semantic/texture双latent、LightningDiT | 400K无guidance FID3.53，guided最低1.04；与RDD解释高度相关但不是空间resolution drift |
| **REG, NeurIPS 2025 Oral** | 提前学习/生成全局语义，显著加速收敛 | semantic token和表示对齐，不做空间resolution transition | L/2 400K无CFG4.6、XL/2 3.4；说明RDD7.06很强但不是离群到无人可比 |

最诚实的判断：RDD400K相对vanilla DiT的提升很大，并且在“只改forward process、标准SD-VAE/DiT、不依赖外部视觉teacher”这个受限类别里很有竞争力；但放到所有现代training accelerators中，REG、SFD、DiMR等已经更强。RDD的影响力取决于能否证明这种收益来自drift参数化而非三尺度联合监督/额外loss，并展示至少一个干净的跨objective或跨分辨率扩展。

## 5. f-DM问题的逐条修正

### 5.1 旧表述中不成立的内容

- “f-DM在stage boundary做离散跳变”——错误。f-DM Eq.4在相邻变换结果之间做stage-internal interpolation。
- “f-DM没有exact posterior”——错误。其Appendix A.2推导了`q(z_s | z_t, x)`。
- “RDD首次在边界补Gaussian noise”——错误。f-DM Algorithm 1和Appendix A.4有noise resampling，并为下采样/平均的逆过程补充缺失Gaussian分量。
- “f-DM不连续所以只能heuristic sampling”——过强。f-DM有ancestral和DDIM构造；其forward不一定是全局Markov，但这不等于没有正确reverse conditional。
- “RDD strict generalization of f-DM”——目前未证明。安全说法是RDD是f-DM大框架下一个面向latent DiT和solver的简化/特化参数化。

### 5.2 仍可防守的RDD差异

- 一个effective-noise head，而不是signal + degradation双预测；
- 加性projection drift给出简洁的stage-local Markov transition；
- 保持标准latent DiT共享backbone，只加resolution conditioning；
- 显式drift-corrected DPM1/DPM2；
- 原生ImageNet、同VAE/同backbone/同updates的完整50K convergence curve；
- 已测的per-step训练代价和sampling latency，而不是只报FID。

### 5.3 必须用实验回答，而不能靠文字回答

1. 标准DiT vs progressive `d=0` vs f-DM-style双预测 vs full RDD；
2. noise filling × DDPM/DPM2的2×2；
3. matched actual NFE曲线；
4. full RDD与最强因果baseline的第二训练seed；
5. 一个正交组合（RDD-FM优先，RDD+REPA备选）。

## 6. 已修改的论文和实验计划

### 论文 `paper/main.tex`

- 删除f-DM“离散跳变/heuristic interpolation”的错误定位；
- 明确承认f-DM已有插值、exact reverse conditional、noise rescaling/resampling；
- 把noise-filling唯一性收窄到RDD marginal和isotropic Gaussian family；
- 加入DFM、LapFlow、REPA、REG、SFD和limited-interval guidance；
- 将REG L/2、REG XL/2、REPA和SFD的400K公开结果加入上下文表；
- 将“RDD与这些方法正交”改为“结构上可组合但需要实证”；
- 保持Video实验段落不变。

### 计划 `docs/theory/experiment_plan.md`

- 将`d=0`从有预算才做提升为投稿阻塞项；
- 新增同协议f-DM-style adapted baseline；
- 将正交性验证限制为一个组合：RDD-FM/SiT首选，RDD+REPA备选；
- matched-NFE从大规模sampler zoo缩减为64/124/244三个DDPM/DPM2锚点；
- 用limited-interval guidance替代密集CFG/stage-wise CFG排列；
- 只要求full RDD与最强因果baseline做第二训练seed；
- Spotlight stretch最多选一个：ImageNet-512 matched pair或更长训练曲线。

## 7. 主来源

- [f-DM, ICLR 2023](https://openreview.net/forum?id=iBdwKIsg4m)
- [SiT, ECCV 2024](https://arxiv.org/abs/2401.08740)
- [REPA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html)
- [REG, NeurIPS 2025 Oral](https://openreview.net/forum?id=koEALFNBj1)
- [Relay Diffusion, ICLR 2024](https://openreview.net/forum?id=qTlcbLSm4p)
- [Matryoshka Diffusion Models, ICLR 2024](https://openreview.net/forum?id=tOzCcDdH9O)
- [DiMR, NeurIPS 2024](https://papers.nips.cc/paper/2024/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html)
- [DFM, NeurIPS 2025](https://openreview.net/forum?id=3isHlkiykj)
- [LapFlow, ICLR 2026](https://openreview.net/forum?id=kdrc4o6okz)
- [SFD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Semantics_Lead_the_Way_Harmonizing_Semantic_and_Texture_Modeling_with_CVPR_2026_paper.pdf)
- [FasterDiT, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/661de50100d3115cf4317bb8b5219e56-Abstract-Conference.html)
- [Limited-interval guidance, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html)
- [Autoguidance, NeurIPS 2024 Oral](https://openreview.net/forum?id=bg6fVPVs3s)
- [LightningDiT / VA-VAE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yao_Reconstruction_vs._Generation_Taming_Optimization_Dilemma_in_Latent_Diffusion_Models_CVPR_2025_paper.pdf)
