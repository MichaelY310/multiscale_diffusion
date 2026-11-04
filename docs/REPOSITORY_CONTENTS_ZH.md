# GitHub main 的内容与复现范围

2026-11-04：此前 GitHub main 是9月30日整理的55文件精简发布版。本次从现有main历史接续，补入迁移快照内的旧RDD项目源码、配置、研究文档与测试，并同步最新边界命名。远程已有结果图和`results/metrics.json`保留，没有重写旧提交。

## 代码在哪里

| 目录 | 内容 |
|---|---|
| `2d/`、`DiT/`、`diffusion/` | 图像DDPM/FM、共享DiT与独立多模型、loss、边界、求解器、采样与FID入口 |
| `rdd_video/` | Wan、视频RDD、蒸馏和加速研究；有独立依赖与模型资产要求 |
| `RDD_DiT_3D/` | 3D与稀疏潜空间研究 |
| `experiments/`、`2d/exp/`、`scripts/` | 原实验配置、调度/诊断/训练脚本；历史机器路径、GPU编号与定时策略须按环境修改 |
| `tests/` | 核心与研究组件测试；并非每个可选集成在本次发布都重新执行 |
| `docs/`、`paper/` | 历史理论、实验说明及研究草稿；保留当时内容，不将早期推测视为新验收结论 |
| `assets/`、`results/metrics.json` | 既有公开结果图与精选指标 |

所有这些源码在main，不需要切换Git分支。新多论文统一框架`unified_multires/umdiff`是工作区中的另一个工程，本仓库的“Unified DiT”表示旧RDD共享多分辨率模型，两者不是同一仓库。

模型权重、数据集、运行日志、大型tensor、Python环境和外部repo不进入Git。实验资产继续通过[HF归档](https://huggingface.co/yang335/search_agent_hl_to_i18n_transport)访问；并不是每个视频微调权重都已找到。机器交接提示、会话日志与独立CUDA练习不属于此次源码发布。

## 边界名称

- `fresh`/`refresh`：丢弃旧噪声，预测clean加全新IID噪声。
- `noise_filling`：从旧noisy与预测clean恢复残差，归一化上采样保留，并补新增子空间。
- `legacy_initial_noise`：保留阶段初始噪声，区别于从末端恢复残差。

旧DDPM主实验的noise_filling实际为fresh；旧FM的noise_filling是恢复残差。已有FID不变。旧DDPM JSON若仍写noise_filling，要么改成fresh复现旧行为，要么显式填写`boundary_semantics: recovered_v2`选择新变体。详见[命名说明](BOUNDARY_NAMES_20261104_ZH.md)。

## 最新数值证据

`validation/20261104/`保留固定CPU测试报告：40条DDPM/DPM、16条共享FM、8条独立三模型FM完整采样；同输入、权重和随机流下结果逐字节相同。训练比较module输出、activation梯度、参数梯度，以及Adam/EMA/RNG；既有完整XL/L检查单列。原DiT和每阶段solver backend复用，两边分别执行阶段组合与训练更新。

本轮新轨迹使用随机非零小网络，保留实际latent网格/采样步数。报告内源码SHA和绝对路径指当时受测迁移工作区，不冒称在GitHub任意后续提交重新执行。视频12项只认证边界函数；不等于完整Wan、3D或附加加速系统均已对齐。正式400K训练和50K FID没有因此重跑。

本次发布另在独立Git checkout运行CPU核心测试与代码语法检查；见`validation/20261104/PUBLISH_VALIDATION.json`。Shell和文本源码的Windows CRLF行尾在发布副本中规范为LF；所有Python文件与已测快照的AST一致，算法实现没有因此改变。
