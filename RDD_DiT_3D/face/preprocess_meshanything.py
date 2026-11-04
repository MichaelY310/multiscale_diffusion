"""
预处理 MeshAnythingV2 数据集：
1. 合并重复顶点
2. 归一化坐标到 [0, 1)
3. 保存为独立 .pt 文件（方便后续按需加载）

用法:
    python preprocess_meshanything.py \
        --input /mnt/bn/.../MeshAnythingV2/test.npz \
        --output-dir /mnt/bn/.../MeshAnythingV2/processed_test \
        --merge-threshold 1e-6
"""

import argparse
import os
import numpy as np
from pathlib import Path


def merge_vertices(vertices, faces, threshold=1e-6):
    """合并距离 < threshold 的重复顶点，更新 face 索引。"""
    from scipy.spatial import cKDTree

    if threshold <= 0:
        # 精确去重
        unique_verts, inverse = np.unique(vertices, axis=0, return_inverse=True)
    else:
        # 基于距离的合并
        tree = cKDTree(vertices)
        # 找到所有距离 < threshold 的点对
        pairs = tree.query_pairs(r=threshold, output_type='ndarray')

        # Union-Find
        n = len(vertices)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # 压缩路径
        for i in range(n):
            parent[i] = find(i)

        # 构建新索引
        unique_parents, inverse = np.unique(parent, return_inverse=True)
        unique_verts = vertices[unique_parents]

    # 更新 face 索引
    new_faces = inverse[faces]

    # 去除退化面（三个顶点有重复的）
    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
            (new_faces[:, 1] != new_faces[:, 2]) & \
            (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[valid]

    # 去除重复面
    sorted_faces = np.sort(new_faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    new_faces = new_faces[unique_idx]

    return unique_verts, new_faces


def normalize_to_unit(vertices):
    """归一化顶点到 [0, 1) 范围。"""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = (vmax - vmin).max()
    if extent < 1e-10:
        return vertices - vmin, vmin, extent
    normalized = (vertices - vmin) / extent * 0.999
    return normalized, vmin, extent


def process_single(item, merge_threshold):
    """处理单个 mesh 样本。"""
    vertices = item['vertices'].astype(np.float64)
    faces = item['faces'].astype(np.int64)

    # 1. Merge vertices
    merged_verts, merged_faces = merge_vertices(vertices, faces, threshold=merge_threshold)

    # 2. Normalize to [0, 1)
    norm_verts, offset, scale = normalize_to_unit(merged_verts)

    return {
        'vertices': norm_verts.astype(np.float32),   # (V', 3)
        'faces': merged_faces.astype(np.int64),       # (F', 3)
        'num_vertices': len(norm_verts),
        'num_faces': len(merged_faces),
        'offset': offset.astype(np.float32),          # 原始偏移（用于还原）
        'scale': float(scale),                         # 原始缩放（用于还原）
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="输入 .npz 文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 .npz 文件路径")
    parser.add_argument("--merge-threshold", type=float, default=1e-6,
                        help="顶点合并距离阈值（0=精确去重）")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="最大处理样本数（-1=全部）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=True)
    npz_list = data['npz_list']
    N = len(npz_list) if args.max_samples < 0 else min(len(npz_list), args.max_samples)
    print(f"Processing {N} samples (merge_threshold={args.merge_threshold}) ...")

    results = []
    stats_v = []
    stats_f = []

    for i in range(N):
        if i % 500 == 0:
            print(f"  [{i}/{N}]")

        result = process_single(npz_list[i], args.merge_threshold)
        results.append(result)
        stats_v.append(result['num_vertices'])
        stats_f.append(result['num_faces'])

    # 保存为单个 npz
    output_path = args.output
    np.savez(output_path, npz_list=np.array(results, dtype=object))

    # 打印统计
    stats_v = np.array(stats_v)
    stats_f = np.array(stats_f)

    print(f"\n=== 处理完成: {N} samples → {output_path} ===")
    print(f"Vertices: min={stats_v.min()}, median={int(np.median(stats_v))}, "
          f"mean={stats_v.mean():.0f}, p95={int(np.percentile(stats_v, 95))}, max={stats_v.max()}")
    print(f"Faces:    min={stats_f.min()}, median={int(np.median(stats_f))}, "
          f"mean={stats_f.mean():.0f}, p95={int(np.percentile(stats_f, 95))}, max={stats_f.max()}")


if __name__ == "__main__":
    main()
