"""
训练 Face Set Transformer AutoEncoder.

该脚本用于训练 Face Set Transformer AE，将每个 grid cell 内的三角面片集合
编码为固定维度的 bottleneck embedding，并重建原始面片坐标。

两阶段流水线：
  Stage 1: python train_face_set_transformer.py --use-synthetic   (合成数据预训练验证)
           python train_face_set_transformer.py --data-dir <dir>  (真实 mesh cell 数据)
  Stage 2: python dataset.py --preprocess   (用训好的 AE 编码所有 mesh → embeddings)
  Stage 3: python train.py                  (在 face embeddings 上训练扩散模型)
"""

import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from face_set_transformer import (
    FaceSetTransformerAE,
    face_set_transformer_loss,
)


# ===========================================================================
# 合成数据生成（用于无真实 mesh 数据时的初步验证）
# ===========================================================================

def _sort_faces_by_centroid(faces_tensor, num_faces, dim_in=9):
    """
    按面片重心的 z → x → y 顺序排序（字典序）。
    faces_tensor: [F_MAX, dim_in] 或 [num_faces, dim_in]
    num_faces: 有效面片数
    返回排序后的 faces_tensor（原地修改）。
    """
    if num_faces <= 1:
        return faces_tensor
    valid = faces_tensor[:num_faces]  # [num_faces, 9]
    # 每个 face 的坐标为 [v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]
    # reshape 为 [num_faces, 3, 3] → 计算重心 [num_faces, 3]
    verts = valid.reshape(num_faces, 3, 3)  # [num_faces, 3 vertices, 3 coords]
    centroids = verts.mean(dim=1)  # [num_faces, 3] = (x, y, z)
    # 按 z → x → y 字典序排序
    sort_keys = centroids[:, 2] * 1e8 + centroids[:, 0] * 1e4 + centroids[:, 1]
    _, indices = sort_keys.sort()
    faces_tensor[:num_faces] = valid[indices]
    return faces_tensor


def generate_synthetic_face_data(batch_size, f_max=10, dim_in=9, use_scale=False):
    """
    生成合成的 face 数据用于训练验证。

    每个样本：
    - 随机 1~f_max 个三角面片
    - 每个面片 = 3 个顶点 × 3 坐标
    - use_scale=False: 坐标在 [0, 1] 范围（旧行为）
    - use_scale=True:  模拟真实数据流程：
        1. 在 200x200x200 空间中随机生成三角形顶点 (范围 [0, 200])
        2. 保证每个三角形的重心在正中间 1x1x1 cube 内 (即 [99.5, 100.5])
        3. 以 cube 中心 (100,100,100) 为缩放中心，缩放整个 triangle set
           直到刚好被包含在 1x1x1 cube 中（某顶点恰好碰到 cube 边界）
        4. 记录缩放比例 scale（原始尺寸 / 归一化尺寸）
        5. 返回 normalized faces (在 [-0.5, 0.5] 内) 和 scale

    Returns:
        faces: [B, F_MAX, dim_in]
        mask:  [B, F_MAX] bool
        scales: [B, 1] (仅 use_scale=True，否则 None)
    """
    faces = torch.zeros(batch_size, f_max, dim_in)
    mask = torch.zeros(batch_size, f_max, dtype=torch.bool)
    scales = torch.zeros(batch_size, 1) if use_scale else None

    for i in range(batch_size):
        num_faces = torch.randint(1, f_max + 1, (1,)).item()

        if use_scale:
            # 在 200x200x200 空间中生成三角形
            # 每个面片有 3 个顶点，每个顶点 3 坐标 → dim_in=9
            # 为了保证放大 case 也存在，混合两种生成策略：
            #   - 50% 概率：顶点在较大范围 [0, 200] 生成（需要缩小）
            #   - 50% 概率：顶点在较小范围（中心附近 0.1~1.0 范围内）（需要放大）
            center = 100.0  # 200x200x200 cube 的中心

            if torch.rand(1).item() < 0.5:
                # 需要缩小的 case：顶点散布在较大范围
                spread = torch.rand(1).item() * 99.0 + 1.0  # spread in [1, 100]
                centroid_range = 1.0  # 重心在 [99.5, 100.5]
            else:
                # 需要放大的 case：三角形非常小，集中在中心
                spread = torch.rand(1).item() * 0.1 + 0.001  # spread in [0.001, 0.1]
                centroid_range = 0.05  # 重心也非常接近中心

            # 生成 num_faces 个三角形的顶点
            # 每个三角形: 先生成重心在 [99.5, 100.5]³ 内，然后顶点围绕重心分布
            all_verts = []  # 存放所有顶点坐标 [num_faces * 3, 3] 展平为 [num_faces, 9]
            raw_coords = torch.zeros(num_faces, dim_in)

            for fi in range(num_faces):
                # 重心在中心附近 centroid_range 范围内
                centroid = torch.rand(3) * centroid_range + (center - centroid_range / 2)
                # 3 个顶点围绕重心分布，偏移范围由 spread 控制
                offsets = (torch.rand(3, 3) - 0.5) * 2 * spread  # [-spread, spread]
                verts = centroid.unsqueeze(0) + offsets  # [3, 3]
                # 调整使重心精确: verts 的均值 = centroid
                actual_centroid = verts.mean(0)
                verts = verts - actual_centroid + centroid
                raw_coords[fi] = verts.reshape(9)

            # 以 center (100,100,100) 为缩放中心
            # 先平移使 center 在原点
            centered = raw_coords.reshape(-1, 3) - center  # [num_faces*3, 3]

            # 缩放使得刚好包含在 [-0.5, 0.5]³ 内
            # scale_val = 所有顶点到中心的最大绝对坐标值（某个顶点恰好碰到边界）
            max_abs = centered.abs().max().item()
            if max_abs < 1e-8:
                max_abs = 1e-8
            # scale_val 是原始空间到归一化空间的比例因子
            # normalized = centered / (max_abs * 2) 会在 [-0.5, 0.5]
            # 但我们用 normalized = centered / max_abs 使其在 [-1, 1]... 不对
            # 按需求：缩放到刚好在 1x1x1 cube 内 → 坐标在 [-0.5, 0.5]
            scale_val = max_abs / 0.5  # 即 max_abs * 2
            normalized = centered / scale_val  # 在 [-0.5, 0.5] 范围，某顶点恰好 ±0.5

            # reshape 回 [num_faces, 9]
            normalized = normalized.reshape(num_faces, dim_in)
            faces[i, :num_faces] = normalized
            scales[i, 0] = scale_val
        else:
            faces[i, :num_faces] = torch.rand(num_faces, dim_in)

        mask[i, :num_faces] = True
        # 按重心 z-x-y 排序，保证确定性输出顺序
        _sort_faces_by_centroid(faces[i], num_faces, dim_in)

    return faces, mask, scales


class SyntheticFaceDataset(Dataset):
    """合成数据集：预生成固定数量的样本用于训练。"""

    def __init__(self, num_samples, f_max=10, dim_in=9, use_scale=False):
        super().__init__()
        self.num_samples = num_samples
        self.f_max = f_max
        self.dim_in = dim_in
        self.use_scale = use_scale
        # 预生成所有数据
        print(f"生成 {num_samples} 个合成样本 (f_max={f_max}, dim_in={dim_in}, use_scale={use_scale})...")
        self.faces, self.masks, self.scales = generate_synthetic_face_data(
            num_samples, f_max, dim_in, use_scale
        )
        print("合成数据生成完成。")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_scale:
            return self.faces[idx], self.masks[idx], self.scales[idx]
        return self.faces[idx], self.masks[idx], torch.zeros(1)  # dummy scale


# ===========================================================================
# 真实 Mesh Cell 数据集
# ===========================================================================

class FaceCellDataset(Dataset):
    """
    加载预处理后的 per-cell 面片数据。

    支持两种格式：
    1. 单个 .pt 文件（list of dicts），由 generate_cell_data.py 生成：
       [{'faces': [F_MAX, 9], 'mask': [F_MAX], 'scale': [1], 'num_faces': int}, ...]
    2. 目录下多个 .pt 文件（旧格式，每个文件对应一个 cell）

    __getitem__ 返回 (faces [F_MAX, 9], mask [F_MAX], scale [1])
    """

    def __init__(self, data_dir, f_max=10, dim_in=9):
        super().__init__()
        self.f_max = f_max
        self.dim_in = dim_in

        # 判断 data_dir 是单个 .pt 文件还是目录
        if os.path.isfile(data_dir) and data_dir.endswith('.pt'):
            # 新格式：单个 .pt 文件包含 list of dicts
            print(f"FaceCellDataset: 从单文件加载 {data_dir} ...")
            self.data_list = torch.load(data_dir, map_location='cpu')
            self.mode = 'list'
            print(f"  加载完成: {len(self.data_list)} 个 cells (f_max={f_max})")
        else:
            # 旧格式：目录下多个 .pt 文件
            import glob
            self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            assert len(self.file_list) > 0, f"在 {data_dir} 中未找到 .pt 文件"
            self.mode = 'files'
            print(f"FaceCellDataset: 加载 {len(self.file_list)} 个 cell 文件 (f_max={f_max})")

    def __len__(self):
        if self.mode == 'list':
            return len(self.data_list)
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.mode == 'list':
            item = self.data_list[idx]
            faces = item['faces'].clone()    # [F_MAX, 9] already padded
            mask = item['mask']      # [F_MAX] bool
            scale = item.get('scale', torch.zeros(1))  # [1]
            # 按重心排序
            num_faces = mask.sum().item()
            _sort_faces_by_centroid(faces, num_faces, self.dim_in)
            return faces, mask, scale

        # 旧格式：从文件加载
        data = torch.load(self.file_list[idx], map_location='cpu')
        faces_raw = data['faces']  # [F_k, 9]
        num_faces = data['num_faces']

        F_k = faces_raw.shape[0]
        assert F_k <= self.f_max, (
            f"文件 {self.file_list[idx]} 含有 {F_k} 个面片，超过 f_max={self.f_max}"
        )

        faces = torch.zeros(self.f_max, self.dim_in)
        mask = torch.zeros(self.f_max, dtype=torch.bool)
        faces[:F_k] = faces_raw[:F_k]
        mask[:F_k] = True
        # 按重心排序
        _sort_faces_by_centroid(faces, F_k, self.dim_in)

        scale = data.get('scale', torch.zeros(1))
        return faces, mask, scale


# ===========================================================================
# 训练循环
# ===========================================================================

def train(args):
    """主训练函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据集 ---
    if args.use_synthetic:
        print("=" * 60)
        print("模式: 合成数据训练（用于验证模型能否收敛）")
        print("=" * 60)
        dataset = SyntheticFaceDataset(
            num_samples=args.num_synthetic_samples,
            f_max=args.f_max,
            dim_in=args.dim_in,
            use_scale=args.predict_scale,
        )
    else:
        assert args.data_dir is not None, "非合成模式需要指定 --data-dir"
        print("=" * 60)
        print(f"模式: 真实 mesh cell 数据训练 (data_dir={args.data_dir})")
        print("=" * 60)
        dataset = FaceCellDataset(
            data_dir=args.data_dir,
            f_max=args.f_max,
            dim_in=args.dim_in,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- 模型 ---
    model = FaceSetTransformerAE(
        dim_in=args.dim_in,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        bottleneck_dim=args.bottleneck_dim,
        ae_pool=args.ae_pool,
        f_max=args.f_max,
        decode_mode=args.decode_mode,
        num_bins=args.num_bins,
        ln=True,
        variational=args.variational,
        predict_scale=args.predict_scale,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print(f"  dim_hidden={args.dim_hidden}, bottleneck_dim={args.bottleneck_dim}")
    print(f"  f_max={args.f_max}, decode_mode={args.decode_mode}")
    print(f"  ae_pool={args.ae_pool}, num_heads={args.num_heads}")
    print(f"  encoder_layers={args.num_encoder_layers}, decoder_layers={args.num_decoder_layers}")

    # --- 优化器 & 调度器 ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- 保存目录 ---
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint 保存目录: {save_dir}")
    print(f"训练配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 60)

    # --- 训练 ---
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_exist = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(loader):
            faces = batch_data[0].to(device)
            mask = batch_data[1].to(device)
            target_scale = batch_data[2].to(device) if args.predict_scale else None

            # 前向 (scale 同时作为 encoder 输入和 decoder 预测目标)
            pred_coords, pred_existence, bottleneck, kl_loss, pred_scale = model(faces, mask, scale=target_scale)

            # 损失
            loss, loss_dict = face_set_transformer_loss(
                pred_coords=pred_coords,
                pred_existence=pred_existence,
                target_coords=faces,
                mask=mask,
                mode=args.decode_mode,
                num_bins=args.num_bins,
                exist_weight=args.exist_weight,
                pred_scale=pred_scale,
                target_scale=target_scale,
                scale_weight=args.scale_weight,
            )

            # VAE: 加 KL 正则项
            if args.variational:
                loss = loss + args.kl_weight * kl_loss

            # 反向
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_recon += loss_dict['recon']
            epoch_exist += loss_dict.get('exist', 0.0)
            epoch_kl = epoch_kl + kl_loss.item() if args.variational else 0.0
            num_batches += 1
            global_step += 1

            # 日志
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / num_batches
                avg_recon = epoch_recon / num_batches
                avg_exist = epoch_exist / num_batches
                lr_now = optimizer.param_groups[0]['lr']
                log_msg = (
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} (recon={avg_recon:.4f}, exist={avg_exist:.4f}"
                )
                if args.variational:
                    avg_kl = epoch_kl / num_batches
                    log_msg += f", kl={avg_kl:.4f}"
                if args.predict_scale and 'scale' in loss_dict:
                    log_msg += f", scale={loss_dict['scale']:.4f}"
                log_msg += f") | LR: {lr_now:.2e}"
                print(log_msg)

        # Epoch 结束
        scheduler.step()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)

        # Checkpoint 保存
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(save_dir, f"face_ae_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args),
            }, ckpt_path)
            print(f"  → 保存 checkpoint: {ckpt_path}")

        # 最优模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(save_dir, "face_ae_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'args': vars(args),
            }, best_path)

        if (epoch + 1) % args.save_every == 0:
            print(
                f"  Epoch {epoch+1} 完成 | "
                f"平均Loss: {avg_epoch_loss:.4f} | "
                f"最优Loss: {best_loss:.4f}"
            )

    print("=" * 60)
    print(f"训练完成！最优 loss = {best_loss:.4f}")
    print(f"最优模型保存于: {os.path.join(save_dir, 'face_ae_best.pt')}")
    print("=" * 60)


# ===========================================================================
# 参数解析
# ===========================================================================

def create_argparser():
    parser = argparse.ArgumentParser(
        description="训练 Face Set Transformer AutoEncoder"
    )

    # 模型架构
    parser.add_argument("--dim-hidden", type=int, default=128,
                        help="Set Transformer 隐藏维度")
    parser.add_argument("--bottleneck-dim", type=int, default=128,
                        help="Bottleneck embedding 维度 (即扩散模型的 in_channels)")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="多头注意力头数")
    parser.add_argument("--num-encoder-layers", type=int, default=2,
                        help="编码器 SAB 层数")
    parser.add_argument("--num-decoder-layers", type=int, default=2,
                        help="解码器 SAB 层数")
    parser.add_argument("--ae-pool", type=str, default="mean",
                        choices=["mean", "attention"],
                        help="AE bottleneck pooling 方式 (mean/attention)")
    parser.add_argument("--decode-mode", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="解码模式: regression(MSE) 或 classification(CE)")

    # 数据参数
    parser.add_argument("--f-max", type=int, default=10,
                        help="每个 cell 最大面片数")
    parser.add_argument("--dim-in", type=int, default=9,
                        help="每个面片的输入维度 (3顶点×3坐标)")
    parser.add_argument("--num-bins", type=int, default=128,
                        help="分类模式的 bin 数量")
    parser.add_argument("--variational", action="store_true",
                        help="使用 VAE 模式（加 KL 正则化使 latent 接近 N(0,I)）")
    parser.add_argument("--kl-weight", type=float, default=1e-3,
                        help="VAE 模式下 KL loss 的权重")
    parser.add_argument("--predict-scale", action="store_true",
                        help="启用 scale 预测（输入归一化到 [-0.5,0.5]，额外预测 scale）")
    parser.add_argument("--scale-weight", type=float, default=1.0,
                        help="scale loss 的权重")
    parser.add_argument("--exist-weight", type=float, default=0.0,
                        help="existence loss 的权重（默认 0 不使用）")

    # 训练参数
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--log-every", type=int, default=50,
                        help="每 N 步打印一次日志")
    parser.add_argument("--save-every", type=int, default=20,
                        help="每 N 个 epoch 保存一次 checkpoint")

    # 数据来源
    parser.add_argument("--data-dir", type=str, default=None,
                        help="预处理后的 per-cell 面片数据目录")
    parser.add_argument("--save-dir", type=str, default="./results_face_ae",
                        help="Checkpoint 保存目录")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="使用合成数据训练（用于初步验证模型）")
    parser.add_argument("--num-synthetic-samples", type=int, default=100000,
                        help="合成数据样本数量")

    return parser


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    args = create_argparser().parse_args()
    train(args)
