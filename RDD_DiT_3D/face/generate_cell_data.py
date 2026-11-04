"""
从预处理后的 mesh 数据（test_merged.npz）生成 Face Set Transformer 训练用的 per-cell 数据。

流程：
1. 加载 npz（每个样本含 vertices [V,3] 和 faces [F,3]，已归一化到 [0,1)）
2. 对每个样本，在指定分辨率 R 下：
   a. 计算每个 face 的重心，按重心分配到 grid cell
   b. 过滤掉超过 F_MAX 的 cell
   c. 计算局部坐标：local = (vertex - cell_origin) / cell_size
   d. 动态缩放：centered = local - 0.5, scale = |centered|.max(), normalized = centered / scale
   e. 保存 (normalized_faces [F_k, 9], mask [F_MAX], scale [1])

输出格式：单个 .pt 文件，包含一个 list of dicts:
    [{'faces': [F_MAX, 9], 'mask': [F_MAX], 'scale': [1], 'num_faces': int}, ...]

用法:
    python generate_cell_data.py \
        --input /path/to/test_merged.npz \
        --output /path/to/cell_data_R32_F24.pt \
        --resolution 32 \
        --f-max 24
"""

import argparse
import os
import numpy as np
import torch
from collections import defaultdict


def compute_cell_data(vertices, faces, resolution, f_max):
    """
    将单个 mesh 转换为 cell-level 面片数据（含 scale 归一化）。

    Args:
        vertices: [V, 3] numpy, 已归一化到 [0, 1)
        faces: [F, 3] numpy int, 面片顶点索引
        resolution: grid 分辨率 R
        f_max: 每个 cell 最大面片数

    Returns:
        list of dict: 每个 dict 对应一个非空 cell:
            {'faces': [F_MAX, 9] tensor, 'mask': [F_MAX] bool tensor,
             'scale': [1] tensor, 'num_faces': int}
    """
    R = resolution
    cell_size = 1.0 / R

    # 计算每个面片的重心
    v1 = vertices[faces[:, 0]]  # [F, 3]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    centroids = (v1 + v2 + v3) / 3.0  # [F, 3]

    # 分配到 grid cell
    grid_coords = np.floor(centroids * R).astype(np.int64)
    grid_coords = np.clip(grid_coords, 0, R - 1)

    # 按 cell 分组
    cell_dict = defaultdict(list)
    for face_idx in range(len(faces)):
        key = tuple(grid_coords[face_idx])
        cell_dict[key].append(face_idx)

    results = []
    skipped_cells = 0

    for (gi, gj, gk), face_indices in cell_dict.items():
        num_faces = len(face_indices)
        if num_faces > f_max:
            skipped_cells += 1
            continue

        # Cell 原点（世界坐标）
        cell_origin = np.array([gi * cell_size, gj * cell_size, gk * cell_size])

        # 计算每个 face 的局部坐标
        local_faces_list = []
        for fi in face_indices:
            v_idx = faces[fi]  # [3]
            verts = vertices[v_idx]  # [3, 3]
            # 局部坐标 = (vertex - cell_origin) / cell_size
            local_verts = (verts - cell_origin) / cell_size  # [3, 3]
            local_faces_list.append(local_verts.reshape(9))

        local_faces = np.stack(local_faces_list, axis=0)  # [F_k, 9]

        # 动态缩放：以 cell 中心 (0.5, 0.5, 0.5) 为缩放中心
        # centered = local - 0.5（相对于 cell 中心的偏移）
        # 缩放到刚好在 [-0.5, 0.5] 内（某顶点恰好碰到边界）
        centered = local_faces.reshape(-1, 3) - 0.5  # [F_k*3, 3]
        max_abs = np.abs(centered).max()
        if max_abs < 1e-8:
            max_abs = 1e-8
        scale_val = max_abs / 0.5  # 原始范围 / 归一化范围
        normalized = (centered / scale_val).reshape(-1, 9)  # 在 [-0.5, 0.5]

        # Pad 到 F_MAX
        padded_faces = np.zeros((f_max, 9), dtype=np.float32)
        padded_faces[:num_faces] = normalized.astype(np.float32)

        mask = np.zeros(f_max, dtype=bool)
        mask[:num_faces] = True

        results.append({
            'faces': torch.from_numpy(padded_faces),        # [F_MAX, 9]
            'mask': torch.from_numpy(mask),                  # [F_MAX]
            'scale': torch.tensor([scale_val], dtype=torch.float32),  # [1]
            'num_faces': num_faces,
        })

    return results, skipped_cells


def main():
    parser = argparse.ArgumentParser(
        description="从 merged npz 生成 Face Set Transformer per-cell 训练数据"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入 .npz 文件（test_merged.npz）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 .pt 文件路径")
    parser.add_argument("--resolution", type=int, default=32,
                        help="Grid 分辨率 R（默认 32）")
    parser.add_argument("--f-max", type=int, default=24,
                        help="每个 cell 最大面片数（超过的 cell 会被跳过）")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="最大处理样本数（-1=全部）")
    args = parser.parse_args()

    print(f"配置: R={args.resolution}, F_MAX={args.f_max}")
    print(f"加载 {args.input} ...")
    data = np.load(args.input, allow_pickle=True)
    npz_list = data['npz_list']
    N = len(npz_list) if args.max_samples < 0 else min(len(npz_list), args.max_samples)
    print(f"处理 {N} 个样本 ...")

    all_cells = []
    total_skipped = 0
    total_cells = 0

    for i in range(N):
        if i % 500 == 0:
            print(f"  [{i}/{N}] 已收集 {len(all_cells)} 个 cells")

        item = npz_list[i]
        vertices = item['vertices'].astype(np.float64)
        faces = item['faces'].astype(np.int64)

        cells, skipped = compute_cell_data(vertices, faces, args.resolution, args.f_max)
        all_cells.extend(cells)
        total_skipped += skipped
        total_cells += len(cells) + skipped

    print(f"\n=== 统计 ===")
    print(f"总样本数: {N}")
    print(f"总 cells: {total_cells}")
    print(f"保留 cells: {len(all_cells)} ({len(all_cells)/max(total_cells,1)*100:.1f}%)")
    print(f"跳过 cells (>{args.f_max} faces): {total_skipped} ({total_skipped/max(total_cells,1)*100:.1f}%)")

    # 统计 scale 和 num_faces 分布
    scales = np.array([c['scale'].item() for c in all_cells])
    num_faces_arr = np.array([c['num_faces'] for c in all_cells])
    print(f"\nScale 分布:")
    print(f"  min={scales.min():.3f}, median={np.median(scales):.3f}, "
          f"mean={scales.mean():.3f}, p95={np.percentile(scales, 95):.3f}, max={scales.max():.3f}")
    print(f"Faces/cell 分布:")
    print(f"  min={num_faces_arr.min()}, median={int(np.median(num_faces_arr))}, "
          f"mean={num_faces_arr.mean():.1f}, p95={int(np.percentile(num_faces_arr, 95))}, "
          f"max={num_faces_arr.max()}")

    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    torch.save(all_cells, args.output)
    print(f"\n已保存 {len(all_cells)} 个 cell 到 {args.output}")
    print(f"文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
