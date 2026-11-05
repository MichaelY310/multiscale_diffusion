# FM fresh 与旧 RDD-DDPM fresh：算子相同，噪声日程不同

本检查针对历史 DDPM-XL 的 `300/300/400`、DDPM `80/80/90`、CFG1 配方，
以及 FM-XL v2 的 `[0,.3,.6,1]`、Euler `80/80/90`、CFG1 配方。
没有访问另一服务器的新生成图，也没有重跑 50K FID。

## 结论

两种 fresh 都执行 `alpha * nearest(predicted_clean) + sigma * new_iid_noise`。
都不保留旧 noisy 的残差，不用 bilinear、不在边界裁剪 clean，也不额外补后验协方差。
DDPM 的边界函数为 `2d/sample_fid_unified.py::_redraw_upsample`；FM 为
`diffusion/flow_matching_with_drift.py::upsample_with_noise_fm`。
unified 的 native DDPM 用 `RDDFresh`，native FM 调用同一个 FM backend。

把系数、clean、随机流设成一样，三种边界实现的输出逐字节一致，最大误差为 0。
这是边界操作测试：用于隔离操作的人工系数不代表真实 VP/DDPM 日程。
实际运行时，两者的系数和提供 clean 估计的状态不相同。

## 旧 DDPM 使用阶段内累计 alpha

`diffusion/respace.py::SpacedDiffusionWithDrift` 从
`last_alpha_cumprod = global_alphas_cumprod[T_c - 1]` 开始构建阶段 betas，
随后 `GaussianDiffusionWithDrift` 对这些阶段 betas 重新累乘。
因此有效累计值是

```math
alpha_bar_stage(T) = alpha_bar_global(T) / alpha_bar_global(T_c - 1)
```

`T_c=0` 时分母为 1。不是直接沿用全局 `alpha_bar_global(T)`。
训练和采样均调用这个构造器；本次逐个核对采样时间对应的训练累计 alpha，
最大差异 `3.33e-16`。此行为在历史提交 `c90213b` 中已经存在，
不是此次 unified 移植或 fresh 改名引入的。

下面是每个阶段**出口**的前向分布系数；X 表示该分辨率真实 clean，eps 表示标准噪声。
这些是训练路径定义，不是假定实际网络生成状态严格符合该理想分布。

| 切换 | 旧 DDPM 出口 | FM 出口 |
|---|---|---|
| 8→16 | `0.993956 X + 0.109781 eps`，模型时间600 | `0.3 X + 0.7 eps`，时间0.3 |
| 16→32 | `0.996957 X + 0.077949 eps`，模型时间300 | `0.6 X + 0.4 eps`，时间0.6 |

DDPM 出口接近 clean；FM 出口仍相当 noisy。不能仅因两边都叫 fresh，
就推断 clean 估计的难度或质量代价相同。
这不意味着每阶段执行了完整1000步：DDPM模型时间仍是
`999→600、599→300、299→0`，此配置实际采样更新仍是 `80/80/90`。
区别是模型时间对应的有效噪声日程。

fresh 构建**下一阶段入口**时，系数也不同：

| 切换 | DDPM `(alpha,sigma)` | FM `(alpha,sigma)` |
|---|---|---|
| 8→16 | `(0.255505,0.966808)`，模型时间599 | `(0.3,0.7)` |
| 16→32 | `(0.629619,0.776904)`，模型时间299 | `(0.6,0.4)` |

DDPM每阶段重新归一化的行为已参与原权重训练，不能为追求表面相同而直接修改旧采样日程。

## clean 恢复是否出错

DDPM从模型的联合噪声目标恢复 clean；FM从 velocity 恢复 clean，并包含 drift 的投影逆。
对3阶段的入口、中点和出口，使用真实条件 velocity 作为 oracle，FM恢复 clean 的
最大误差 `8.89e-16`（float64）。两个高分辨率阶段的训练入口也符合
`start * UP(DOWN(clean)) + (1-start) * noise`，最大误差 `3.06e-16`。
这些验证说明已实现的代数关系正确，不衡量实际训练网络的预测误差。

发现 FM 源文件旧注释错误地把“drift为零”写成“状态已经clean”，并把无drift的
`x+(1-t)v` 公式泛化到所有阶段出口。实际代码已使用正确的 drift-aware inverse；
本次只修正说明文字，数值算法不变。

## 对新 fresh FID 33.2233 的意义

较 noisy 的出口使 fresh 对 clean 估计偏差与后验均值替换更敏感；
这是为何旧 DDPM 的 fresh 有效、但 FM fresh 可能退化的具体机制之一。
不能由该代数检查证明新的 FID 退化全部来自它，仍需同一服务器、权重、资产和评估流程下的
FM filling/fresh 对照；四卡分片代码也未在本次审查中取得。

## 复查

```bash
CUDA_VISIBLE_DEVICES='' UMDIFF_CPU_ONLY=1 \
  python unified_multires/tools/audit_rdd_fresh_semantics.py \
  --output unified_multires/runs/fresh_audit/report.json
```

只运行CPU代数与日程检查，不下载模型/数据、不创建训练checkpoint。
[原始结果及源码hash](FM_DDPM_FRESH_AUDIT.json)。
