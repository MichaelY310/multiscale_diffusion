# RDD 实验命名与配置对照

**不要再把 `300300400`、`908080`、`808090` 当作独立的实验名称。**
从现在起，每个数都写明对应的 latent 网格。训练时段、采样预算、实际网络调用次数分开记。
本文及 [逐条结果表](experiment_registry/results.csv) 是当前索引；原始日志和 checkpoint 内部字段保留原样，作为历史证据。

## 最容易混淆的几组

| 配置 | 8×8 | 16×16 | 32×32 | 说明 |
|---|---:|---:|---:|---|
| 旧 DDPM `stage_lengths=300,300,400` 的训练时段长度 | 400 | 300 | 300 | 原参数按高→低；模型时标分别 999→600、599→300、299→0 |
| FM-XL v2 `stage_lengths=300,300,400` 的训练时段长度 | 300 | 300 | 400 | 原参数按低→高；FM 时间分别 0→0.3、0.3→0.6、0.6→1 |
| corrected FM-L 旧标签 `300300400` | 400 | 300 | 300 | 实际 CLI 为 `400,300,300`；时间 0→0.4、0.4→0.7、0.7→1 |
| 旧 DDPM 主采样，文件名常写 `908080` | 80 | 80 | 90 | 实际 `num_steps=80,80,90`；DDPM 为 250 NFE |
| FM-XL v2 Euler，文件名写 `808090` | 80 | 80 | 90 | 250 次 Euler 更新，加两次边界预测，共 252 NFE |
| FM-XL v2 midpoint | 40 | 40 | 45 | 125 次 RK2 更新，加两次边界预测，共 252 NFE |
| DDPM DPM-Solver-2，旧文件名 `908080` | 80 | 80 | 90 | 原实现共 494 NFE，不是 250 NFE |
| 当前共同对比 RDD 的另一份配方 | 90 | 80 | 80 | 确实与历史主采样不同；没有将参数改成历史配方 |
| paired / paper recipe 的 RDD DPM2 配方 | 100 | 75 | 75 | 另一个分配；并非历史 DPM2 主表的 80/80/90 |

上表的 8/16/32 都是 **latent 空间的边长**；同一 SD-VAE 下，对应图像边长 64/128/256。
DDPM 的时标不是 FM 的信号系数：其原实现还会在阶段内部重新归一化 alpha，不能仅看全局数字推断噪声水平。

## 每个实验到底是什么

- [116 条历史正式结果 CSV](experiment_registry/results.csv)：每条记录包含训练 checkpoint 步数、每个网格的训练区间、采样步数、solver、NFE、CFG、边界和五项指标；没有改动原指标。
- [完整结果 JSON](experiment_registry/results.json)：包含原始表行和修正理由；采样边界消融的 inference 时段也与训练时段分开。
- [全部已扫描的 RDD 采样配置 CSV](experiment_registry/configs.csv)：配置存在不等于已经跑完；包含主框架、原生复现和旧 JSONL。
- [模型命名表](experiment_registry/models.json)：13 组正式图像模型的训练时钟和旧 HF 目录名。
- [仓库旧入口 → 新入口](experiment_registry/repository_aliases.json)：新文件名明确写 `train_r8-400_r16-300_r32-300`、`sample_r8-80_r16-80_r32-90`；旧入口仅作为兼容符号链接。
- [HF 文件迁移计划](experiment_registry/hf_rename_plan.json)：每个文件记录旧路径、新路径、字节数、LFS SHA256 或 Git blob。是否已执行看迁移回执，不能把计划当已完成。
- [其他服务器最近的 FM 对照](experiment_registry/remote_fm_reports.json)：fresh FID 33.2233 与 noise_filling FID 14.5032 均为 8×8=80、16×16=80、32×32=90、252 NFE。按用户回报单列，不冒称本机重跑。

视频仓库的 `rdd-c2f` 分支也已明确命名为
`wan_finetune__train_down4-400_down2-300_down1-300.yaml`：空间缩小 4/2/1 倍，
训练时间分别 0→0.4、0.4→0.7、0.7→1。该 pilot 采样是 20/15/15 次 Euler 更新，
加两次边界预测；每个 CFG 分支 52 次，分别执行条件与无条件模型共 104 次调用。
它不是 ImageNet FM-XL v2 的 0/.3/.6/1、80/80/90 配置。
见 [视频配置改名与验证](https://github.com/MichaelY310/FastVideo/commit/02308c6)。

新实验的名称形式：

```text
rdd-ddpm-shared-xl2__train_r8-400_r16-300_r32-300
__ckpt-400000__dpm2__sample_r8-80_r16-80_r32-90__cfg-1.5__fresh
```

这分别表示模型、训练区间长度、训练更新步数、求解器、每个网格的采样预算、CFG 与边界。
仅模型/checkpoint 的名称不包含采样参数，因为同一权重可以用于多种采样实验。
裸 `300300400` 既可能是训练区间，也可能是一次高预算采样，禁止靠字符串猜。

## 核对方式与保留范围

1. 已对照固定 HF revision 中的旧 JSONL；不是依据迁移后文件名倒推。见 [原始配置核对](validation/20261104/SAMPLING_STEP_ORDER_ARCHIVE.json)。
2. 已实际在 CPU 执行原采样阶段循环，记录网格、输入时标和网络调用数。DDPM 的两种分配、DPM2 两组预算、FM Euler/midpoint 均有记录，见 [调用审计](validation/20261104/SAMPLING_STEP_ORDER_CALLS.json)。这不是重新评估 FID。
3. 修复了旧结果生成器两处列方向错误：FM-XL TSV 和 interval-proportional TSV 原本已经按低→高，但被放在了高→低列。新表逐格展开；历史输入另存，不覆盖证据。
4. 所有重命名保持数值配置不变。旧 DDPM 的历史 `noise_filling` 名字实际代表 fresh；FM 的历史 `noise_filling` 则保留残差，两者仍按 [边界定义](BOUNDARY_NAMES_20261104_ZH.md) 区分。
5. 旧服务器路径、旧 Git 提交、固定 HF revision、论文截图和原始日志不可被当作可以全局重写的当前配置。它们保留旧名，并通过索引查新名。没有足够配置证据的历史记录明确标为未核实，不赋予猜测的分配。

新旧文件名的兼容链接不是另一份实验。新采样入口会打印逐网格配置，并在输出目录写 `resolution_identity.json`；同目录再次运行时若分配改变则报错。这个文件只检查分辨率相关身份，不替代完整 checkpoint、seed、数据和 RNG 的验证。

重建索引（CPU，不加载模型）：

```bash
python scripts/build_rdd_naming_registry.py
python scripts/inventory_rdd_configs.py
python scripts/validate_rdd_naming.py
```

HF 重命名使用服务端复制和同一提交内删除旧路径，不重新下载或上传 checkpoint；旧 revision 的链接继续可用。执行工具会检查远端父提交、源文件身份、目标路径冲突以及提交后的所有 hash。
