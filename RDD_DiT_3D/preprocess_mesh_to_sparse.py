"""
预处理 MeshAnythingV2 数据集 → 稀疏 .pt 文件，供 RDD-DiT-3D 训练使用。

Pipeline:
1. 加载预训练 Set Transformer (V1 或 V2)
2. 加载 MeshAnythingV2 merged npz（包含顶点+面片）
3. 对每个 mesh:
   a. 提取顶点，归一化到 [0,1]^3
   b. 按 K_MAX 过滤（跳过存在任何 cell > K_MAX 顶点的 mesh）
   c. 在最高分辨率 R=256 下:
      - 按顶点位置分配到 R^3 grid
      - 每个非空 cell 用 Set Transformer encode → embedding
      - 记录 (embedding, grid_coord)
   d. 保存为 .pt 文件

注意: 只在最高分辨率编码一次。低分辨率数据在训练时通过 mean pool 实时计算。

Usage:
  python RDD_DiT_3D/preprocess_mesh_to_sparse.py \
      --input /path/to/test_merged.npz \
      --ckpt /path/to/set_transformer_checkpoint.pt \
      --output-dir /path/to/output/ \
      --resolution 256 \
      --k-max 10 \
      --filter-by-kmax \
      --batch-size 512
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# 添加 set_transformer 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'set_transformer'))


def load_set_transformer(ckpt_path, device):
    """加载预训练 Set Transformer，自动检测 V1/V2."""
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']

    # 检测版本
    if 'decoder_type' in args or 'use_k_feature' in args:
        # V2
        from train_emd_ae_3d_v2 import AEBridgeReconModelV2
        model = AEBridgeReconModelV2(
            dim_hidden=args.get('dim_hidden', 32),
            num_heads=args.get('num_heads', 4),
            num_encoder_layers=args.get('num_encoder_layers', 4),
            num_decoder_layers=args.get('num_decoder_layers', 4),
            bottleneck_dim=args.get('bottleneck_dim', 64),
            ae_num_heads=args.get('ae_num_heads', 4),
            ae_num_mlp_layers=args.get('ae_num_mlp_layers', 3),
            ae_mlp_factor=args.get('ae_mlp_factor', 4),
            ae_pool=args.get('ae_pool', 'pma1'),
            decoder_type=args.get('decoder_type', 'ar'),
            kl_weight=args.get('kl_weight', 0.0),
            use_k_feature=args.get('use_k_feature', True),
            ln=args.get('ln', True),
            k_max=args.get('k_max', 10),
        )
        use_k_feature = args.get('use_k_feature', True)
    else:
        # V1
        from train_emd_ae_3d import AEBridgeReconModel
        k_max = args.get('k_max', 10)
        model = AEBridgeReconModel(
            dim_hidden=args['dim_hidden'],
            num_heads=args['num_heads'],
            num_encoder_layers=args['num_encoder_layers'],
            num_decoder_layers=args['num_decoder_layers'],
            bottleneck_dim=args['bottleneck_dim'],
            ae_num_heads=args['ae_num_heads'],
            ae_num_mlp_layers=args['ae_num_mlp_layers'],
            ae_mlp_factor=args['ae_mlp_factor'],
            ae_pool=args['ae_pool'],
            k_max=k_max,
            ln=args.get('ln', False),
        )
        use_k_feature = True

    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    k_max = args.get('k_max', 10)
    bottleneck_dim = args.get('bottleneck_dim', 64)
    dim_in = 4 if use_k_feature else 3

    print(f"Loaded Set Transformer from {ckpt_path}")
    print(f"  step={ckpt['step']}, dim_hidden={args.get('dim_hidden')}, "
          f"bottleneck_dim={bottleneck_dim}, ae_pool={args.get('ae_pool')}, "
          f"k_max={k_max}, use_k_feature={use_k_feature}")

    return model, k_max, dim_in, bottleneck_dim, use_k_feature


def normalize_vertices(vertices):
    """归一化顶点到 [0, 1]^3."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    span = (vmax - vmin).max()
    if span < 1e-8:
        return np.full_like(vertices, 0.5)
    normalized = (vertices - center) / span + 0.5  # [0, 1]^3
    normalized = np.clip(normalized, 0.0, 1.0 - 1e-7)
    return normalized


def assign_to_grid(vertices, R):
    """
    将顶点分配到 R^3 grid cells.
    vertices: [N, 3] in [0, 1)^3
    返回: dict {(i,j,k): [indices]}
    """
    grid_coords = np.floor(vertices * R).astype(np.int64)
    grid_coords = np.clip(grid_coords, 0, R - 1)

    cells = {}
    for idx in range(len(vertices)):
        key = (grid_coords[idx, 0], grid_coords[idx, 1], grid_coords[idx, 2])
        if key not in cells:
            cells[key] = []
        cells[key].append(idx)
    return cells


def check_kmax_constraint(vertices, R, k_max):
    """检查该 mesh 在分辨率 R 下是否所有 cell 顶点数 ≤ k_max."""
    grid_coords = np.floor(vertices * R).astype(np.int64)
    grid_coords = np.clip(grid_coords, 0, R - 1)
    _, counts = np.unique(grid_coords, axis=0, return_counts=True)
    return counts.max() <= k_max


def _encode_bottleneck(model, X, mask):
    """
    Extract bottleneck embedding from AEBridgeReconModel (V1) or V2.
    Returns: [B, bottleneck_dim]
    """
    if hasattr(model, 'encode'):
        return model.encode(X, mask)
    else:
        Z = model.encoder(X, mask)
        _, bottleneck = model.ae(Z, mask)
        return bottleneck


@torch.no_grad()
def encode_cells_batched(model, cells, vertices, k_max, dim_in, use_k_feature, device, batch_size=512):
    """
    批量编码所有 cells (仅在最高分辨率 R 调用一次).
    返回: embeddings [N_cells, C], coords [N_cells, 3]
    """
    cell_keys = list(cells.keys())
    N_cells = len(cell_keys)

    all_X = torch.zeros(N_cells, k_max, dim_in)
    all_mask = torch.zeros(N_cells, k_max, dtype=torch.bool)
    all_coords = torch.zeros(N_cells, 3, dtype=torch.long)

    for ci, key in enumerate(cell_keys):
        indices = cells[key][:k_max]  # 截断到 k_max
        k = len(indices)
        pts = vertices[indices]  # [k, 3]

        # 局部归一化到 [0, 1]
        cell_min = pts.min(axis=0)
        cell_max = pts.max(axis=0)
        span = cell_max - cell_min
        span = np.where(span < 1e-8, 1.0, span)
        local_pts = (pts - cell_min) / span

        # 字典序排序
        order = sorted(range(k), key=lambda r: (local_pts[r, 0], local_pts[r, 1], local_pts[r, 2]))
        local_pts = local_pts[order]

        all_X[ci, :k, :3] = torch.from_numpy(local_pts).float()
        if use_k_feature:
            all_X[ci, :k, 3] = k / k_max
        all_mask[ci, :k] = True
        all_coords[ci] = torch.tensor(key, dtype=torch.long)

    # 批量编码
    embeddings_list = []
    for start in range(0, N_cells, batch_size):
        end = min(start + batch_size, N_cells)
        X_batch = all_X[start:end].to(device)
        mask_batch = all_mask[start:end].to(device)
        emb = _encode_bottleneck(model, X_batch, mask_batch)
        embeddings_list.append(emb.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)  # [N_cells, C]
    return embeddings, all_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='MeshAnythingV2 merged npz 文件路径')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='预训练 Set Transformer checkpoint 路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出 .pt 文件目录')
    parser.add_argument('--resolution', type=int, default=256,
                        help='编码分辨率 (只在此分辨率下编码一次)')
    parser.add_argument('--k-max', type=int, default=10,
                        help='每个 cell 最大顶点数')
    parser.add_argument('--filter-by-kmax', action='store_true', default=True,
                        help='过滤掉存在超过 K_MAX 顶点的 cell 的 mesh')
    parser.add_argument('--no-filter', dest='filter_by_kmax', action='store_false')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Set Transformer 编码的 batch size')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    R = args.resolution

    # 加载 Set Transformer
    model, k_max_model, dim_in, bottleneck_dim, use_k_feature = load_set_transformer(args.ckpt, device)
    k_max = args.k_max
    assert k_max <= k_max_model, f"Requested k_max={k_max} > model k_max={k_max_model}"

    # 加载数据
    print(f"\n加载数据: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    npz_list = data['npz_list']
    N = len(npz_list)
    print(f"共 {N} 个 mesh")
    print(f"编码分辨率: R={R}, K_MAX={k_max}, filter={args.filter_by_kmax}")

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    saved = 0
    skipped = 0

    for idx in range(N):
        item = npz_list[idx]
        vertices = item['vertices'].astype(np.float64)

        if len(vertices) == 0:
            skipped += 1
            continue

        verts_norm = normalize_vertices(vertices)

        # K_MAX 过滤
        if args.filter_by_kmax:
            if not check_kmax_constraint(verts_norm, R, k_max):
                skipped += 1
                continue

        # 只在 R 编码一次
        cells = assign_to_grid(verts_norm, R)
        embeddings, coords = encode_cells_batched(
            model, cells, verts_norm, k_max, dim_in, use_k_feature, device, args.batch_size
        )

        result = {
            'embeddings': embeddings,  # [N_cells, C]
            'coords': coords,          # [N_cells, 3]
            'resolution': R,
            'label': 0,
        }

        save_path = os.path.join(args.output_dir, f"{saved:06d}.pt")
        torch.save(result, save_path)
        saved += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{idx+1}/{N}] saved={saved}, skipped={skipped}, "
                  f"{rate:.1f} mesh/s, elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成! 共处理 {N} 个 mesh:")
    print(f"  保存: {saved} ({saved/N*100:.1f}%)")
    print(f"  过滤: {skipped} ({skipped/N*100:.1f}%)")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/N:.2f}s/mesh)")
    print(f"  输出目录: {args.output_dir}")
    print(f"  每个 .pt: embeddings [{bottleneck_dim}-d] + coords [3-d] at R={R}")


if __name__ == '__main__':
    main()
