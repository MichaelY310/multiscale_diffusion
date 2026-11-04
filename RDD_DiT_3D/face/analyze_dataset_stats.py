"""
统计 MeshAnythingV2 数据集的两个关键指标：
1. Grid density: 不同分辨率下每个 cell 内面片数的分布
2. Overflow distance: 面片顶点超出所分配 cell 边界的最大距离（以 cell_size 为单位）

这两个指标用于决定 F_MAX 和数据过滤阈值。

用法:
    python analyze_dataset_stats.py \
        --input /mnt/bn/.../MeshAnythingV2/test_merged.npz \
        --resolutions 256 512
"""

import argparse
import numpy as np
from collections import defaultdict


def analyze_single_mesh(vertices, faces, resolution):
    """
    对单个 mesh 分析 grid density 和 overflow distance。

    Args:
        vertices: [V, 3] numpy, 已归一化到 [0, 1)
        faces: [F, 3] numpy int, 面片顶点索引
        resolution: grid 分辨率 R

    Returns:
        faces_per_cell: list of int — 每个非空 cell 的面片数
        overflow_distances: list of float — 每个非空 cell 的最大溢出距离
            (以 cell_size 为单位，0 表示全部在 cell 内)
    """
    R = resolution
    cell_size = 1.0 / R

    # 每个面片的三个顶点坐标
    v1 = vertices[faces[:, 0]]  # [F, 3]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    # 按重心分配 cell
    centroids = (v1 + v2 + v3) / 3.0  # [F, 3]
    grid_coords = np.floor(centroids * R).astype(np.int64)
    grid_coords = np.clip(grid_coords, 0, R - 1)

    # 按 cell 分组
    cell_dict = defaultdict(list)
    for face_idx in range(len(faces)):
        key = tuple(grid_coords[face_idx])
        cell_dict[key].append(face_idx)

    faces_per_cell = []
    overflow_distances = []

    for (gi, gj, gk), face_indices in cell_dict.items():
        faces_per_cell.append(len(face_indices))

        # 计算该 cell 的边界
        cell_origin = np.array([gi * cell_size, gj * cell_size, gk * cell_size])
        cell_end = cell_origin + cell_size

        # 收集该 cell 内所有面片的所有顶点
        all_verts = []
        for fi in face_indices:
            v_idx = faces[fi]  # [3]
            verts = vertices[v_idx]  # [3, 3]
            all_verts.append(verts)

        if len(all_verts) == 0:
            overflow_distances.append(0.0)
            continue

        all_verts = np.concatenate(all_verts, axis=0)  # [N*3, 3]

        # 计算超出 cell 边界的距离
        # overflow = max(0, vertex - cell_end) + max(0, cell_origin - vertex)
        overflow_low = np.maximum(0, cell_origin - all_verts)   # 低于下界
        overflow_high = np.maximum(0, all_verts - cell_end)     # 高于上界
        overflow = overflow_low + overflow_high                  # [N*3, 3]

        # 最大溢出距离（以 cell_size 为单位）
        max_overflow = overflow.max() / cell_size
        overflow_distances.append(max_overflow)

    return faces_per_cell, overflow_distances


def print_distribution(name, values):
    """打印分布统计。"""
    arr = np.array(values)
    print(f"\n  {name} ({len(arr)} cells):")
    print(f"    min={arr.min():.4f}, median={np.median(arr):.4f}, mean={arr.mean():.4f}")
    print(f"    p75={np.percentile(arr, 75):.4f}, p90={np.percentile(arr, 90):.4f}, "
          f"p95={np.percentile(arr, 95):.4f}, p99={np.percentile(arr, 99):.4f}")
    print(f"    max={arr.max():.4f}")


def print_histogram(name, values, bins):
    """打印直方图。"""
    arr = np.array(values)
    print(f"\n  {name} 直方图:")
    for i in range(len(bins) - 1):
        count = np.sum((arr >= bins[i]) & (arr < bins[i+1]))
        pct = count / len(arr) * 100
        bar = '█' * int(pct / 2)
        print(f"    [{bins[i]:6.2f}, {bins[i+1]:6.2f}) : {count:7d} ({pct:5.1f}%) {bar}")
    count_over = np.sum(arr >= bins[-1])
    pct = count_over / len(arr) * 100
    bar = '█' * int(pct / 2)
    print(f"    [{bins[-1]:6.2f},    inf) : {count_over:7d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="MeshAnythingV2 数据集 grid density + overflow distance 统计"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入 .npz 文件（test_merged.npz）")
    parser.add_argument("--resolutions", type=int, nargs='+', default=[256, 512],
                        help="Grid 分辨率列表（默认 256 512）")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="最大处理样本数（-1=全部）")
    args = parser.parse_args()

    print(f"加载 {args.input} ...")
    data = np.load(args.input, allow_pickle=True)
    npz_list = data['npz_list']
    N = len(npz_list) if args.max_samples < 0 else min(len(npz_list), args.max_samples)
    print(f"处理 {N} 个样本, 分辨率: {args.resolutions}")

    for R in args.resolutions:
        print(f"\n{'='*70}")
        print(f" Resolution R = {R} (cell_size = {1.0/R:.6f})")
        print(f"{'='*70}")

        all_faces_per_cell = []
        all_overflow_distances = []
        per_mesh_max_faces = []
        per_mesh_max_overflow = []

        for i in range(N):
            if i % 500 == 0:
                print(f"  [{i}/{N}] ...")

            item = npz_list[i]
            vertices = item['vertices'].astype(np.float64)
            faces_idx = item['faces'].astype(np.int64)

            if len(faces_idx) == 0:
                continue

            fpc, ovf = analyze_single_mesh(vertices, faces_idx, R)
            all_faces_per_cell.extend(fpc)
            all_overflow_distances.extend(ovf)

            if fpc:
                per_mesh_max_faces.append(max(fpc))
            if ovf:
                per_mesh_max_overflow.append(max(ovf))

        # === 统计 1: Faces per Cell ===
        print(f"\n--- 统计 1: Faces per Cell ---")
        print_distribution("Faces/cell (全局)", all_faces_per_cell)
        print_distribution("Per-mesh max faces/cell", per_mesh_max_faces)

        # 直方图
        fpc_arr = np.array(all_faces_per_cell)
        bins_fpc = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 64]
        print(f"\n  Faces/cell 累积分布:")
        for threshold in [4, 8, 12, 16, 24, 32]:
            count = np.sum(fpc_arr <= threshold)
            pct = count / len(fpc_arr) * 100
            print(f"    ≤ {threshold:3d}: {count:8d} / {len(fpc_arr)} ({pct:.2f}%)")

        # === 统计 2: Overflow Distance ===
        print(f"\n--- 统计 2: Overflow Distance (以 cell_size 为单位) ---")
        print_distribution("Overflow (全局)", all_overflow_distances)
        print_distribution("Per-mesh max overflow", per_mesh_max_overflow)

        ovf_arr = np.array(all_overflow_distances)
        # overflow = 0 表示完全在 cell 内
        in_cell = np.sum(ovf_arr == 0)
        print(f"\n  完全在 cell 内 (overflow=0): {in_cell} / {len(ovf_arr)} ({in_cell/len(ovf_arr)*100:.2f}%)")

        print(f"\n  Overflow 累积分布:")
        for threshold in [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            count = np.sum(ovf_arr <= threshold)
            pct = count / len(ovf_arr) * 100
            print(f"    ≤ {threshold:.1f} cell_size: {count:8d} / {len(ovf_arr)} ({pct:.2f}%)")

        # === 联合分析: 过滤后的数据量 ===
        print(f"\n--- 联合过滤分析 ---")
        print(f"  F_MAX (faces/cell) × overflow_threshold 过滤后的保留 cell 比例:")
        for f_max in [4, 8, 12, 16, 24]:
            for ovf_thresh in [0.5, 1.0, 1.5, 2.0, 999]:
                mask = (fpc_arr <= f_max) & (ovf_arr <= ovf_thresh)
                count = mask.sum()
                pct = count / len(fpc_arr) * 100
                ovf_label = f"≤{ovf_thresh:.1f}" if ovf_thresh < 100 else "不限"
                print(f"    F_MAX={f_max:2d}, overflow {ovf_label:>5s}: "
                      f"{count:8d} / {len(fpc_arr)} ({pct:.1f}%)")
            print()

    print("\n=== 统计完成 ===")


if __name__ == "__main__":
    main()
