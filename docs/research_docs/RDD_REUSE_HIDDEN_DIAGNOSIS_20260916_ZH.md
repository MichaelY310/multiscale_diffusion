# RDD-DMD 跨尺度复用：hidden 监督前的只读诊断

## 结论

当前三步、共享 DiT-L/2、8/20 层复用的训练没有 Transformer hidden 对齐项。可以增加，但不能把“hidden 更像教师”等同于画质修复。本次只增加并运行独立诊断脚本，没有修改两组训练目标、权重、预算或 checkpoint。

目前有两类不同问题：缓存能表达的更新还没有对齐；缓存本身不能表达的细粒度更新也确实存在。额外监督主要帮助前者，不能凭空增加 fine 计算能力。

## 当前 loss（完成 1000 步 ramp 后）

所有 paired clean MSE 都比较相同初始噪声与类别下，原不跳层 DMD 和学生各自 rollout 的阶段输出。on-policy MSE 则在学生自己的阶段输入上询问冻结教师，两者不是一个目标。

| 项目 | MSE 对照权重 | DMD＋LPIPS 权重 |
|---|---:|---:|
| 最终 32×32 paired clean MSE | 1 | 0.1 |
| 8×8 paired clean MSE | 0.5 | 0.5 |
| 16×16 paired clean MSE | 0.5 | 0.5 |
| 三阶段 on-policy clean MSE 的平均值 | 0.25 | **0** |
| 三阶段 variance head MSE 的平均值 | 0.01 | 0.01 |
| RDD 分布匹配 surrogate | 0 | 1 |
| 解码图片的 LPIPS | 0 | 0.25 |
| 生成器 Transformer hidden 对齐 | **0** | **0** |

LPIPS 使用图像特征比较，不是本生成器内部 Transformer hidden 的监督。DMD 的额外 score 模型只用于训练。loss 数值大小不能直接证明哪一项梯度主导。

## 诊断设置与数值

- 两组都读取新增 5000 步、累计压缩 10000 步的固定快照，不读取正在写入的 latest。
- 每组相同 8 个新样本，batch=1，seed=6192026。不是 256 张正式验证，更不是 FID。
- FP32 权重、BF16 autocast、TF32 关闭；GPU 0，峰值 allocated 3.438 GiB。
- 学生完整 rollout 后，把每个 stage 的学生输入、相同 t、相同 y 送入原始冻结、不跳层 DMD 教师，比较同层 hidden。
- 附加诊断：将学生 fine 第 2 层后的 hidden 原样交给教师第 3—22 层，得到这个相同输入下教师希望执行的累计更新。将它与学生该段实际复用的累计更新比较。
- 诊断带 trace 与原始默认 forward 的三阶段 clean 输出逐元素一致；学生和教师都通过检查。没有反向传播、VAE 解码或模型更新。

下表先逐样本归一化，再平均；NMSE 是平方误差／目标能量，不是图像质量百分比。

| fine 第 3—22 层累计更新 | MSE 对照 | DMD＋LPIPS |
|---|---:|---:|
| 教师更新中 coarse-cell 内变化的能量比例 | 8.658% | 8.731% |
| 学生更新中 coarse-cell 内变化的能量比例 | 0.000679% | 0.000602% |
| 学生更新相对同输入教师更新的 NMSE | 0.224373 | 0.223407 |
| 可表示部分单独比较的 NMSE | 0.150830 | 0.149066 |

这里的“细节”指 hidden token 网格在每个 2×2 子块内的变化：先平均池化到上一层 token 网格，再 nearest 上采样，剩余部分就是这些变化。**不是 RGB 图片的 FFT，也不是全部图像高频。** 原始 hidden 网格为 4×4 → 8×8 → 16×16，因为 patch size=2。

学生的极小非零值主要与有限精度加法有关；理想精度下这段缓存累计更新在该细节子空间为零。新配方当前约 22.34% 的归一化更新误差中，约 8.73 个百分点来自不可表示细节，其余约 13.61 个百分点来自可表示部分失配。因此不能只归因于“缺少高频”。以上仅限这 8 个样本、这段教师变换与这种特征坐标。

| 新配方、同一个学生 stage 输入 | 第 2 层 hidden NMSE | 第 12 层 | 第 22 层 | 第 24 层 |
|---|---:|---:|---:|---:|
| 16×16 | 0.03632 | 0.24380 | 0.22866 | 0.09999 |
| 32×32 | 0.14664 | 0.30172 | 0.19380 | 0.27676 |

误差没有随深度单调增加，不能宣称“第 N 层就是唯一出错点”。fine 前两层已有偏差；共享参数微调也可能改变 hidden 坐标，原始 hidden NMSE 本身不证明生成质量。

新配方最终 clean MSE：原教师自己轨迹参照 0.08154；教师接收学生同输入的参照 0.05112。两者目标不同，不能把差值解释成误差的因果分解；同输入仍有明显失配，说明不只是两个 rollout 输入不一样。

## 结构上为什么不能靠任意 hidden loss 保证修好

fine 第 3—22 层全部是旧更新的上采样加法。固定缓存，在精确算术下：

$$h_{22}^{S}=h_2^{S}+\sum_{l=3}^{22}U(\Delta_l^{\mathrm{middle}}).$$

这 20 个位置不是 20 次读取当前 fine hidden 的非线性变换。若 $P$ 是平均池化再 nearest 上采样，则：

$$ (I-P)(h_{22}^{S}-h_2^{S})=0. $$

对于任意目标累计更新 $d$，只用低网格更新 $z$ 做逐点拟合，有投影恒等式：

$$\|Uz-d\|^2=\|Uz-Pd\|^2+\|(I-P)d\|^2.$$

本次度量的教师细节比例就是这种**固定输入、固定目标更新拟合**的不可表示部分。它不是最终画质误差下界：学生前两层、最后两层、输出头仍可适应，原有 fine hidden 的细节也没有被删除。

此外复用的是旧 timestep 下的整个 block 更新，包含旧时间调制效果；复用支路不重新读取当前 fine hidden 或当前 t。剩余真实计算层仍会使用当前 t。这里是可测试的结构限制，不是“发现了采样程序错误”。

## 建议怎样加 hidden loss（尚未加入训练）

采用冻结的原三步 DMD 作 hidden 教师，不使用未蒸馏 diffusion 模型作同层 hidden 目标。教师接收学生实际 stage 输入 `x.detach()`、相同 t、y；只在教师路径 stop-gradient，学生及跨 stage 缓存保留梯度。

第一组只加少量接口监督：

1. 16×16 与 32×32 的第 24 层后：完整 hidden 归一化 MSE，约束交给输出头的特征。
2. 32×32 第 22 层后：低网格投影后的 hidden MSE，约束即将交给最后两个真实 block 的 coarse 信息，不硬逼缓存逐层创造其无法表达的细节。
3. 初始不逐层监督全部 24 层；不加 attention matrix loss；不同时改 mask、采样和其他 loss。

对选定接口定义：

$$L_{h,l}=\frac{\operatorname{mean}\left[(h_l^{S}-\operatorname{sg}(h_l^{T}))^2\right]}{\operatorname{mean}\left[(h_l^{T})^2\right]+\epsilon}.$$

上式 mean 对平方后的逐元素值求平均；教师不接收梯度。第 22 层版本先对双方使用 $P$。

推荐训练集上先校准辅助项的梯度量级（如主目标生成器梯度范数的约 5–10%，只是试验起点），逐步 ramp，并记录梯度方向是否冲突；不要按 loss 标量随意选一个很大的权重。已有训练每步计算同输入教师前向，可从中捕获接口特征，不需要再加第三种教师。推理不增加网络、参数或算子。

应从同一个 checkpoint 做有／无 hidden loss 的等预算小对照，再判断是否延长；当前两组 30000 步实验保持原目标不变。恢复 on-policy 输出项也可以单独消融，但不要和 hidden loss 同时改变后把收益全部归因于 hidden。

如果 hidden 误差下降但完整图像、独立验证和正式分布指标不改善，应考虑把四个真实 fine block 分散布置等结构对照，并重新测视频速度，而不是持续加大 hidden 权重。

## 已有负面经验和边界

旧的非 DMD、prefix-reader 实验曾做过 hidden 对齐：有 coarse 组 hidden NMSE 降低 16.85%，但最终 latent 偏差只降低 0.25%，没有明显图像恢复。那不是本轮直接复用架构，不能据此断言当前方案无效；但足以说明 hidden 对齐不是必然有效的补救。

中间特征蒸馏是已有方法，不作为新理论创新：[FitNets](https://arxiv.org/abs/1412.6550)、[TinyBERT](https://aclanthology.org/2020.findings-emnlp.372/)。它们支持“可以采用中间监督”的方法选择，不证明此 RDD 配置会成功。

原始诊断：[metrics.json](../../eval_results/dmd_reuse_objective_20260915/hidden_audit_20260916/metrics.json)。脚本：[diagnose_dmd_reuse_hidden.py](../../2d/attention_analysis/diagnose_dmd_reuse_hidden.py)。旧对照：[历史报告](RDD_JOINT_TRAINING_AND_SCALE_SPECIFIC_ACCELERATION_ZH.md)。
