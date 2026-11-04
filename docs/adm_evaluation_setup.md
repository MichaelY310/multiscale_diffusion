# ADM Evaluation Setup Guide

本项目使用 OpenAI guided-diffusion 的评估工具计算 FID/sFID/IS/Precision/Recall。

## 依赖文件

### 1. Reference NPZ（真实图片统计）

ImageNet 256×256:
```bash
wget -O /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz \
  "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
```

ImageNet 512×512（如需 512 实验）:
```bash
wget -O /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet512.npz \
  "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz"
```

### 2. InceptionV3 模型（~100MB，evaluator.py 自动下载）

evaluator.py 运行时会自动下载 `classify_image_graph_def.pb` 到当前工作目录。如果网络不通，手动放置：
```bash
# 已有备份:
/home/tiger/multiscale_diffusion/2d/classify_image_graph_def.pb

# 或手动下载:
wget "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
tar -xzf inception-2015-12-05.tgz classify_image_graph_def.pb
```

### 3. evaluator.py 脚本

位置: `/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py`

Python 依赖（在 2d venv 中已安装）:
```
numpy
scipy
tensorflow (或 tensorflow-cpu)
```

## 评估流程

### Step 1: 采样 50K 图片

用 `sample_fid_unified.py`（RDD）或 DiT 原版 `sample_fid.py` 生成 50000 张 PNG。

### Step 2: PNG 打包为 NPZ

```bash
python /home/tiger/multiscale_diffusion/2d/png_to_npz.py <png_dir> --num-samples 50000 --output /tmp/samples.npz
```

### Step 3: 计算指标

```bash
cd /home/tiger/multiscale_diffusion/2d  # 确保 classify_image_graph_def.pb 在 cwd
python /home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py \
  /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz \
  /tmp/samples.npz
```

输出示例:
```
Inception Score: 215.83
FID: 3.94
sFID: 6.14
Precision: 0.8265
Recall: 0.5309
```

## 文件路径汇总

| 文件 | 路径 |
|------|------|
| Reference NPZ (256) | `/mnt/bn/.../RDD/VIRTUAL_imagenet256_labeled.npz` |
| InceptionV3 pb | `/home/tiger/multiscale_diffusion/2d/classify_image_graph_def.pb` |
| evaluator.py | `/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py` |
| png_to_npz.py | `/home/tiger/multiscale_diffusion/2d/png_to_npz.py` |
| Python env | `source /home/tiger/multiscale_diffusion/2d/2d/bin/activate` |

## 自动化脚本

已有封装好的批量 FID 计算脚本:
- RDD: `2d/exp/compute_fid_rdd_convergence.sh`
- DiT: `2d/exp/compute_fid_dit_convergence.sh`

这些脚本自动完成 PNG→NPZ→evaluator 全流程，跳过已有结果和文件不够的目录。
