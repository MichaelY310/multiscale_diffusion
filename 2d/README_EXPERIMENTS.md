# RDD 实验手册

## 1. 环境准备

- 虚拟环境激活: `source /home/tiger/multiscale_diffusion/2d/2d/bin/activate`
- 包管理必须使用 `uv pip`（不要用原生 pip）
- 关键依赖: torch, diffusers, numpy (2.2.6), scipy, accelerate

## 2. 训练入口

### 2.1 RDD DDPM — Multistage（独立模型）

- **Python 入口**: `train_RDD_imagenet256.py`
- **架构**: 每个分辨率一个独立 DiT（32×32→XL, 16×16→L, 8×8→B）
- **Shell 脚本（当前活跃）**:
  - `train_RDD_imagenet256_300_300_400_original.sh`
  - `train_RDD_imagenet256_100_300_600_original.sh`
- **关键参数**:
  ```bash
  --model-32 DiT-XL/2 --model-16 DiT-L/2 --model-8 DiT-B/2 \
  --stage-lengths 300,300,400 \
  --resume-32 <ckpt_path> --resume-16 <ckpt_path> --resume-8 <ckpt_path>
  ```

### 2.2 RDD DDPM — Unified（单一模型）

- **Python 入口**: `train_RDD_imagenet256_unified.py`
- **架构**: 单个 DiTMultiRes 处理所有分辨率
- **Shell 脚本**:
  - `train_RDD_imagenet256_unified_XL_original.sh`
  - `train_RDD_imagenet256_unified_L_original.sh`
- **关键参数**:
  ```bash
  --model DiT-XL/2 --resolutions 32,16,8 \
  --stage-lengths 300,300,400 --resume <ckpt_path>
  ```

### 2.3 RDD Flow Matching — Multistage

- **Python 入口**: `train_RDD_FM_imagenet256.py`
- **Shell 脚本**: `train_RDD_FM_imagenet256_multistage.sh`
- **Stage 映射**:
  - coarsest (8×8): t∈[0, 0.3)
  - middle (16×16): t∈[0.3, 0.6)
  - finest (32×32): t∈[0.6, 1.0)

### 2.4 RDD Flow Matching — Unified

- **Python 入口**: `train_RDD_FM_imagenet256_unified.py`
- **Shell 脚本**: `train_RDD_FM_unified_XL_v2.sh`

### 2.5 DiT Baseline（原版）

- **Python 入口**: `DiT/train_imagenet256.py`
- **Shell 脚本**: `DiT/train_imagenet256_original.sh`
- **关键参数**:
  ```bash
  --model DiT-XL/2 --image-size 256 \
  --data-source imagenet --data-path .../ILSVRC/Data/CLS-LOC/train \
  --resume <ckpt_path>
  ```

### 2.6 通用训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| `--global-batch-size` | 256 | 8 GPU × 32 per GPU |
| `--ckpt-every` | 5000 | 每 5K steps 保存一次 |
| `--data-source` | imagenet | 数据来源 |
| `--data-path` | `.../ILSVRC/Data/CLS-LOC/train` | ImageNet 训练集路径 |

- **数据预处理**: ADM center_crop_arr（random flip + center crop to 256×256）
- **结果保存路径**: `/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/`
- **续训机制**: 自动从 checkpoint 续训，log 以 append 模式写入

## 3. 采样 & 评估

### 3.1 批量 FID 评估 (sample_fid_unified.py)

读取 JSONL 配置文件，逐个实验采样 + 自动计算 FID。

**使用方法**:
```bash
# 8 GPU 并行采样
for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid_unified.py \
        --config exp/new_ablation_400k.jsonl \
        --rank $RANK --world-size 8 \
        --total-samples 50000 --batch-size 256 \
        --seed 0 &
done
wait
```

- **JSONL 每行字段**: `model_type`, `ckpt`/`ckpt_dir`, `sampler`, `num_steps`, `cfg_scale`, `stage_lengths`, `output_dir` 等
- **结果输出**: 自动输出到 `eval_results_unified/` 下的 txt 文件
- **断点续采**: 已有的 PNG 会自动跳过

### 3.2 单次预览采样 (RDD_sample_unified.py)

快速生成少量图片预览效果。

**使用方法**:
```bash
python RDD_sample_unified.py \
    --model DiT-XL/2 \
    --ckpt path/to/checkpoint.pt \
    --vae ema \
    --stage-lengths 300,300,400 \
    --resolutions 32,16,8 \
    --num-steps-per-stage 300,300,400 \
    --total-samples 16 --batch-size 16 \
    --class-label random --cfg-scale 1.5 \
    --save-grid --output-dir samples_preview
```

### 3.3 当前活跃评估脚本

| 脚本 | 用途 |
|------|------|
| `exp/new_ablation_400k.sh` | 16 个消融实验（DPM-Solver-2、Stage split、CFG sweep） |
| `exp/ablation_batch_400k.sh` | 早期消融实验 |

- **FID 评估标准**: 50K samples，参考集 ADM 256×256

## 4. 目录结构约定

- Checkpoint 保存在 `RESULTS_DIR/imagenet256-{type}/checkpoints/` 下
- 文件名格式: `{step:07d}.pt`（如 `0400000.pt`）
- 只保留 50K 倍数的 checkpoint + 最新一个
- 不再使用 `001-`, `002-` 等递增子目录

## 5. 关键配置文件

| 路径 | 说明 |
|------|------|
| `exp/new_ablation_400k.jsonl` | 新消融实验配置 |
| `exp/ablation_batch_400k.jsonl` | 旧消融实验配置 |
| `eval_results_unified/` | 所有 FID 评估结果 |
| `speed_benchmark/` | 训练速度基准日志 |
