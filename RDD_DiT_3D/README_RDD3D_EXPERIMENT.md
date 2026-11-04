# RDD-3D：从数据到论文表格的直接运行指南

> **路线更新（2026-08-11）：** 当前主方案由 RDD 自身逐级生成并剪枝 empty voxel，不再由 TRELLIS Structure Model 预先提供 support。完整设计见 [`RDD_3D_DYNAMIC_SPARSE_GENERATION.md`](../docs/research_docs/RDD_3D_DYNAMIC_SPARSE_GENERATION.md)。本页后续 fixed-support 内容仅保留作历史受控对照。

当前 dynamic sparse 32-asset gate v2（默认 10K steps、raw weights；v1 的
2K teacher-forced pilot 未通过自由生成 gate）：

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
RDD3D_DYNAMIC_GPU=4 RDD3D_DYNAMIC_RUN_NAME=dynamic_sparse_32asset_gate_v2 \
  bash scripts/rdd3d/run_dynamic_sparse_pilot_tmux.sh
tail -f logs/rdd3d_dynamic_sparse_pilot.log
```

完整的表示审计、数学推导、相关工作与当前实测数值见
[`docs/research_docs/RDD_3D_SPARSE_LATENT_EXPERIMENT.md`](../docs/research_docs/RDD_3D_SPARSE_LATENT_EXPERIMENT.md)。本页只保留可执行顺序。

## 0. 实验范围

主实验只替换 TRELLIS.2 的 **shape SLat geometry flow**：

`input image -> shared TRELLIS.2 structure support -> {flat FM | RDD FM} -> shared SC-VAE shape decoder`

Texture/PBR 不参与第一张主表。RDD 与 flat 使用相同 support、1.3B backbone 初始化、条件图、训练数据、优化器步数、40 NFE 和 decoder。主输出分辨率是 1024³（64³ SLat）；512³ 只做 scaling ablation。

## 1. 一次性环境

```bash
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
bash scripts/rdd3d/run_setup_trellis2_tmux.sh
tail -f logs/rdd3d_trellis2_setup.log
```

代码固定 TRELLIS.2 commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`。正式训练使用官方 DINOv3-L/16。需要由账号持有人在模型页面接受 license，然后在服务器登录 Hugging Face：

```text
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
```

DINOv2 入口只用于无 gated 权重时检查 data/DDP/loss/sampler，DINOv2 数值不能放进论文质量表。

## 2. 数据：先 1K 容量 probe，再 10K 决策集

TRELLIS-500K Sketchfab 源已验证可以通过 Objaverse v1/Hugging Face 下载，不需要 GitHub 整仓 zip。probe 中一个 GLB 为约 50.7 MB；实际大小长尾明显，所以不要直接盲下 50K。

先准备 1K train，记录 `du -sh`、成功率和耗时：

```bash
RDD3D_SPLIT=train \
RDD3D_SOURCE=sketchfab \
RDD3D_SUBSET_COUNT=1000 \
RDD3D_OVOXEL_RESOLUTION=1024 \
RDD3D_GPU=4 \
bash scripts/rdd3d/run_trellis2_formal_data_tmux.sh

tail -f logs/rdd3d_data_train_1024.log
```

通过容量检查后，用同一 seed 建 10K train。确定性 hash ranking 保证前 1K 是 10K 的稳定子集：

```bash
RDD3D_SPLIT=train \
RDD3D_SOURCE=sketchfab \
RDD3D_SUBSET_COUNT=10000 \
RDD3D_OVOXEL_RESOLUTION=1024 \
RDD3D_GPU=4 \
bash scripts/rdd3d/run_trellis2_formal_data_tmux.sh
```

再建不与 train 重叠的 1K validation：

```bash
TRAIN_ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion/datasets/rdd3d_formal/ObjaverseXL_sketchfab_train_10000

RDD3D_SPLIT=validation \
RDD3D_SOURCE=sketchfab \
RDD3D_SUBSET_COUNT=1000 \
RDD3D_SUBSET_SEED=8407 \
RDD3D_EXCLUDE_METADATA="$TRAIN_ROOT/metadata.csv" \
RDD3D_OVOXEL_RESOLUTION=1024 \
RDD3D_GPU=4 \
bash scripts/rdd3d/run_trellis2_formal_data_tmux.sh
```

每个 split 都保存 `rdd3d_subset_manifest.json`，包含 source、seed、aesthetic threshold、SHA256 列表和 selection digest。不要从 train 中随机抽 validation。

## 3. 10K paired decision run

先确认 DINOv3 可加载，再同时在两组 GPU 上训练：

```bash
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
TRAIN_ROOT="$ROOT/datasets/rdd3d_formal/ObjaverseXL_sketchfab_train_10000"

RDD3D_SUBSET_COUNT=10000 \
RDD3D_DATA_ROOT="$TRAIN_ROOT" \
RDD3D_MODEL_RESOLUTION=1024 \
RDD3D_MAX_STEPS=5000 \
RDD3D_RUN_NAME=formal_sketchfab_10000_5000step \
RDD3D_IMAGE_ENCODER=dinov3 \
RDD3D_EMA_RATE=0.999 \
RDD3D_RDD_GPUS=4,5 \
RDD3D_FLAT_GPUS=6,7 \
bash scripts/rdd3d/run_trellis2_paired_formal.sh

tail -f logs/trellis2_rdd_1024_formal_sketchfab_10000_5000step_train.log
tail -f logs/trellis2_flat_1024_formal_sketchfab_10000_5000step_train.log
```

默认 optimizer 是官方 1024 fine-tune 配置：AdamW、LR `2e-5`、weight decay `0.01`、BF16 AMP、CFG dropout `0.1`；每模型 2 GPUs、1 shape/GPU。RDD stage 为 16/32/64 等概率，global-time fractions 为 fine/middle/coarse 0.5/0.3/0.2，并使用 band-width-normalized loss。两边都从同一个官方 1024 SLat checkpoint 初始化。5K decision run 使用匹配的 EMA=0.999；官方 0.9999 在 500/5K-step objective-switch fine-tune 中过于滞后，短程 gate 同时报告 raw checkpoint。

## 4. 受控 ground-truth-support 评估

这一步先隔离 geometry flow；每边使用 validation SLat 的同一个真实 support：

```bash
VAL_ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion/datasets/rdd3d_formal/ObjaverseXL_sketchfab_validation_1000

RDD3D_EVAL_DATA_ROOT="$VAL_ROOT" RDD3D_MODEL_RESOLUTION=1024 \
RDD3D_RUN_NAME=formal_sketchfab_10000_5000step \
RDD3D_EVAL_LIMIT=100 RDD3D_EVAL_GPU=4 \
bash scripts/rdd3d/run_trellis2_known_support_eval_tmux.sh rdd

RDD3D_EVAL_DATA_ROOT="$VAL_ROOT" RDD3D_MODEL_RESOLUTION=1024 \
RDD3D_RUN_NAME=formal_sketchfab_10000_5000step \
RDD3D_EVAL_LIMIT=100 RDD3D_EVAL_GPU=6 \
bash scripts/rdd3d/run_trellis2_known_support_eval_tmux.sh flat
```

输出含 squared Chamfer-L2、F-score@0.01、precision/recall、geometry sampling wall time、peak VRAM、latent MSE 与失败列表。参考 mesh 是原始 validation SLat 的 SC-VAE decode；最终论文还需补 raw normalized mesh 的指标。

## 5. 相同 predicted support 评估

先只运行一次官方 structure model，把 support 固定落盘：

```bash
VAL_ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion/datasets/rdd3d_formal/ObjaverseXL_sketchfab_validation_1000
SUPPORT_ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion/eval_results/rdd3d_predicted_supports/1024

RDD3D_EVAL_DATA_ROOT="$VAL_ROOT" \
RDD3D_SUPPORT_OUTPUT="$SUPPORT_ROOT" \
RDD3D_MODEL_RESOLUTION=1024 \
RDD3D_EVAL_LIMIT=100 \
RDD3D_EVAL_GPU=4 \
bash scripts/rdd3d/run_trellis2_predicted_supports_tmux.sh
```

然后 RDD/flat 都读取这一个目录：

```bash
RDD3D_EVAL_DATA_ROOT="$VAL_ROOT" RDD3D_EVAL_SUPPORTS="$SUPPORT_ROOT" \
RDD3D_RUN_NAME=formal_sketchfab_10000_5000step \
RDD3D_MODEL_RESOLUTION=1024 RDD3D_EVAL_LIMIT=100 RDD3D_EVAL_GPU=4 \
bash scripts/rdd3d/run_trellis2_known_support_eval_tmux.sh rdd

RDD3D_EVAL_DATA_ROOT="$VAL_ROOT" RDD3D_EVAL_SUPPORTS="$SUPPORT_ROOT" \
RDD3D_RUN_NAME=formal_sketchfab_10000_5000step \
RDD3D_MODEL_RESOLUTION=1024 RDD3D_EVAL_LIMIT=100 RDD3D_EVAL_GPU=6 \
bash scripts/rdd3d/run_trellis2_known_support_eval_tmux.sh flat
```

## 6. Go / no-go

只有同时满足以下条件才扩到 50K、20K steps：

1. RDD 1024 geometry-denoiser wall time相对 flat 至少 1.3×；
2. ground-truth support 下 F-score/CD 退化不超过 3%；
3. predicted support 下结论仍成立；
4. RDD 没有更高的 empty mesh/NaN/断裂失败率；
5. projector、detail-subspace noise 和 8/12/20 schedule 的消融能解释收益。

若 10K/5K-step 后 RDD 仍明显低于 flat，不要用更大数据掩盖路径问题；先检查 global-time target、stage balance 和 pretrained-flat-to-RDD objective shift。若修正后仍差，则把 3D 降为 representation/efficiency appendix，而不是继续烧 50K。
