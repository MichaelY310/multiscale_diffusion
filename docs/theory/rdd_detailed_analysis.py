"""
RDD 理论加速 - 逐 voxel 深度统计
================================
基于 ShapeNet Car 点云数据，详细分析每个 2×2×2 block 的情况
"""
import torch
import numpy as np
import json
import os

PLY_PATH = "/home/tiger/multiscale_diffusion/.gc_download/example_data/100715345ee54d7ae38b52b4ee9d36a3/points3d.ply"
OUTPUT_DIR = "/home/tiger/multiscale_diffusion/research_experiments/rdd_sparsity_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_ply_points(ply_path):
    """读取 binary PLY"""
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        num_vertices = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                break
        data = np.frombuffer(f.read(), dtype=np.uint8)
        vertex_size = 6 * 4 + 3
        points = np.zeros((num_vertices, 3), dtype=np.float32)
        for i in range(num_vertices):
            offset = i * vertex_size
            points[i] = np.frombuffer(data[offset:offset+12], dtype=np.float32)
    return points


def voxelize(points, resolution, bound=0.45):
    """体素化点云"""
    pts = (points - (-bound)) / (2 * bound) * resolution
    pts = np.clip(pts, 0, resolution - 1).astype(int)
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    for p in pts:
        grid[p[0], p[1], p[2]] = True
    return grid


def multilevel_analysis(grid, max_res):
    """多层级逐 block 详细分析"""
    
    print(f"\n{'='*70}")
    print(f" 多层级 RDD 加速分析 (分辨率 {max_res}³, ShapeNet Car 点云)")
    print(f"{'='*70}")
    
    # 建立多分辨率金字塔
    levels = []
    res = max_res
    while res >= 4:
        levels.append(res)
        res //= 2
    levels.reverse()  # [4, 8, 16, 32, ...]
    
    # 计算各分辨率下的占据网格
    grids_at_level = {}
    grids_at_level[max_res] = grid
    
    for res in levels[:-1]:  # 除了最高分辨率
        factor = max_res // res
        g = np.zeros((res, res, res), dtype=bool)
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    block = grid[i*factor:(i+1)*factor, j*factor:(j+1)*factor, k*factor:(k+1)*factor]
                    g[i, j, k] = block.any()
        grids_at_level[res] = g
    
    # ========================================
    # 核心分析：RDD 推理流程模拟
    # ========================================
    print(f"\n RDD 推理流程模拟:")
    print(f" Stage 1: 在 {levels[0]}³ 做 FULL diffusion (所有 voxel)")
    for i in range(1, len(levels)):
        print(f" Stage {i+1}: 从 {levels[i-1]}³ 上采样到 {levels[i]}³ (仅计算 exposed voxels)")
    
    total_rdd_ops = 0
    total_full_ops = 0
    
    stage_details = []
    
    for level_idx, res in enumerate(levels):
        g = grids_at_level[res]
        occupied = g.sum()
        total = res ** 3
        
        if level_idx == 0:
            # 最低分辨率：全量计算
            exposed = total
            waste = total - occupied
            waste_ratio = 0  # 这不算浪费，因为必须做 full diffusion
            
            print(f"\n Stage {level_idx+1} ({res}³):")
            print(f"   全量 diffusion: {total} voxels")
            print(f"   其中 occupied: {occupied} ({100*occupied/total:.1f}%)")
            
        else:
            # 非最低分辨率：分析 2×2×2 暴露
            half_res = res // 2
            parent_grid = grids_at_level[res // 2] if res // 2 in grids_at_level else None
            
            # 对于从 parent 到 child 的上采样
            # 每个 parent voxel 产生 8 个 children
            # 如果 parent 非空 → 8 个 children 全部暴露
            block_count = {n: 0 for n in range(9)}
            exposed = 0
            
            for i in range(half_res):
                for j in range(half_res):
                    for k in range(half_res):
                        children = g[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2]
                        n_occ = children.sum()
                        block_count[n_occ] += 1
                        if n_occ > 0:
                            exposed += 8  # 全部 8 个都暴露
            
            waste = exposed - occupied
            waste_ratio = waste / exposed if exposed > 0 else 0
            
            print(f"\n Stage {level_idx+1} ({res}³):")
            print(f"   Occupied: {occupied}/{total} ({100*occupied/total:.1f}%)")
            print(f"   Exposed (must compute): {exposed} ({100*exposed/total:.1f}%)")
            print(f"   Waste: {waste} ({100*waste_ratio:.1f}% of exposed)")
            print(f"   2×2×2 block 分布:")
            print(f"   {'Occupied/8':<12} {'Count':<8} {'%':<8} {'Waste/block':<12}")
            print(f"   {'-'*44}")
            for n in range(9):
                if block_count[n] > 0:
                    pct = 100 * block_count[n] / (half_res**3)
                    w = 8 - n if n > 0 else 0
                    marker = " ← 最大浪费" if n == 1 else ""
                    print(f"   {n}/8          {block_count[n]:<8d} {pct:<7.1f}% {w:<12d}{marker}")
        
        total_rdd_ops += exposed
        total_full_ops += total  # full-space 每个 stage 都算全部
        
        stage_details.append({
            "stage": level_idx + 1,
            "resolution": res,
            "total_voxels": total,
            "occupied": int(occupied),
            "exposed": int(exposed),
            "waste": int(waste),
            "waste_ratio": float(waste_ratio),
        })
    
    # ========================================
    # 与 full-space diffusion 对比
    # ========================================
    # Full-space: 所有 stage 都在最高分辨率
    full_space_all_max = max_res ** 3 * len(levels)
    # Full-space multires (不稀疏): 每个 stage 在对应分辨率
    full_multires = sum(r**3 for r in levels)
    
    print(f"\n{'─'*70}")
    print(f" 计算量对比:")
    print(f"{'─'*70}")
    print(f"  方案 A - Full-space (全在最高分辨率): {full_space_all_max:>12,} voxel-ops ({len(levels)} × {max_res}³)")
    print(f"  方案 B - 多分辨率 (但不稀疏):        {full_multires:>12,} voxel-ops")
    print(f"  方案 C - RDD 稀疏多分辨率:            {total_rdd_ops:>12,} voxel-ops")
    print(f"")
    print(f"  加速比:")
    print(f"    RDD vs Full-space:                 {full_space_all_max/total_rdd_ops:.2f}x")
    print(f"    RDD vs 多分辨率(非稀疏):            {full_multires/total_rdd_ops:.2f}x (纯稀疏加速)")
    print(f"    多分辨率 vs Full-space:             {full_space_all_max/full_multires:.2f}x (纯多分辨率加速)")
    
    # ========================================
    # 考虑实际 DPM-Solver 的额外加速
    # ========================================
    print(f"\n{'─'*70}")
    print(f" 结合 DPM-Solver 高阶采样器的实际加速:")
    print(f"{'─'*70}")
    print(f"  PyramidalWan 基线: 20-20-10 步 = 50 NFE")
    print(f"  RDD + DPM-Solver-2: 10-10-5 步 = 25 NFE (× 0.5 步数)")
    print(f"")
    print(f"  最终理论加速 = 稀疏加速 × 步数减少:")
    sparse_speedup = full_multires / total_rdd_ops
    dpm_speedup = 2.0  # DPM-Solver-2 理论上减半步数
    total_speedup = sparse_speedup * dpm_speedup
    full_speedup = (full_space_all_max / total_rdd_ops) * dpm_speedup
    print(f"    vs 多分辨率非稀疏: {sparse_speedup:.2f}x (稀疏) × 2x (DPM) = {total_speedup:.2f}x")
    print(f"    vs Full-space:     {full_space_all_max/total_rdd_ops:.2f}x (稀疏+多分辨率) × 2x (DPM) = {full_speedup:.2f}x")
    
    return {
        "stages": stage_details,
        "full_space_cost": full_space_all_max,
        "multires_cost": full_multires,
        "rdd_cost": total_rdd_ops,
        "speedup_vs_fullspace": full_space_all_max / total_rdd_ops,
        "speedup_vs_multires": full_multires / total_rdd_ops,
        "speedup_multires_only": full_space_all_max / full_multires,
        "with_dpm_solver": full_speedup,
    }


def opacity_threshold_analysis():
    """
    分析不同 opacity 阈值下的稀疏度变化
    基于 GaussianCube mean.pt (训练集平均 opacity)
    """
    print(f"\n{'='*70}")
    print(f" Opacity 阈值敏感度分析")
    print(f"{'='*70}")
    
    data_dir = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/GaussianCube_ShapeNet"
    mean_path = os.path.join(data_dir, "mean.pt")
    
    if not os.path.exists(mean_path):
        print("[WARN] mean.pt not found, skipping opacity analysis")
        return None
    
    mean = torch.load(mean_path, map_location='cpu')
    # opacity 是第 6 通道 (0-indexed)
    opacity = mean[:, :, :, 6].numpy()
    
    print(f"\n  GaussianCube 训练集平均 opacity 统计:")
    print(f"  Shape: {opacity.shape}")
    print(f"  Range: [{opacity.min():.4f}, {opacity.max():.4f}]")
    print(f"  Mean: {opacity.mean():.4f}")
    print(f"  Median: {np.median(opacity):.4f}")
    
    # 直方图
    print(f"\n  Opacity 直方图 (32³ voxels = 32768 total):")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(opacity, bins=bins)
    for i in range(len(bins)-1):
        bar = '█' * (hist[i] * 60 // opacity.size)
        print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:>6d} ({100*hist[i]/opacity.size:>5.1f}%) {bar}")
    
    # 不同阈值下的"空 voxel"数量
    print(f"\n  不同 opacity 阈值下的稀疏度:")
    print(f"  {'阈值':<8} {'空voxel数':<12} {'占据率':<10} {'RDD可跳过':<12}")
    print(f"  {'─'*45}")
    
    results = []
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        empty = (opacity < thresh).sum()
        occupied = 32768 - empty
        occ_ratio = occupied / 32768
        print(f"  {thresh:<8.2f} {empty:<12d} {100*occ_ratio:<9.1f}% {100*(1-occ_ratio):<11.1f}%")
        results.append({"threshold": thresh, "empty": int(empty), "occupied": int(occupied), "occ_ratio": float(occ_ratio)})
    
    print(f"\n  ⚠️  注意: 这是 mean.pt (训练集平均值)")
    print(f"      单个样本的 opacity 分布会更极端（更多 0 和更多 1）")
    print(f"      std.pt 的 opacity std 平均值 = 0.37 → 说明个体差异大")
    print(f"      实际单样本的占据率应该比 mean 显示的 更稀疏")
    
    # 用 mean - std 估计单样本的"确定空"区域
    std_path = os.path.join(data_dir, "std.pt")
    if os.path.exists(std_path):
        std = torch.load(std_path, map_location='cpu')
        std_opacity = std[:, :, :, 6].numpy()
        
        # 如果 mean + std < threshold，那大概率是空的
        confident_empty = ((opacity + std_opacity) < 0.3).sum()
        confident_occupied = 32768 - confident_empty
        print(f"\n  保守估计 (mean + std < 0.3 视为空): {confident_empty} 空 / {confident_occupied} 占据")
        print(f"  保守占据率: {100*confident_occupied/32768:.1f}%")
        
        # 如果 mean - std > threshold，那大概率是非空的
        confident_full = ((opacity - std_opacity) > 0.5).sum()
        print(f"  确定非空 (mean - std > 0.5): {confident_full} ({100*confident_full/32768:.1f}%)")
        
        # 估计典型单样本
        print(f"\n  估计典型单样本的 opacity 分布:")
        print(f"  假设服从正态 N(mean, std²)，对每个 voxel 采样:")
        np.random.seed(42)
        sample_opacity = np.random.normal(opacity, std_opacity)
        sample_opacity = np.clip(sample_opacity, 0, 1)
        
        for thresh in [0.1, 0.2, 0.3, 0.5]:
            empty = (sample_opacity < thresh).sum()
            print(f"    阈值{thresh:.1f}: {empty} 空 voxels ({100*empty/32768:.1f}%), 占据率 {100*(1-empty/32768):.1f}%")
    
    return results


def main():
    print("RDD 3D Sparse Diffusion - 逐 Voxel 深度分析")
    print("=" * 70)
    
    # 读取点云
    points = read_ply_points(PLY_PATH)
    print(f"ShapeNet Car 点云: {len(points)} points")
    print(f"范围: x=[{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"      y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"      z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    all_results = {}
    
    # 在 32³ 和 64³ 分辨率下分析
    for res in [32, 64]:
        bound = 0.45  # 和 GaussianCube init_volume_grid 一致
        grid = voxelize(points, res, bound)
        result = multilevel_analysis(grid, res)
        all_results[f"shapenet_car_{res}"] = result
    
    # Opacity 阈值分析
    opacity_results = opacity_threshold_analysis()
    if opacity_results:
        all_results["opacity_analysis"] = opacity_results
    
    # 保存
    output_file = os.path.join(OUTPUT_DIR, "rdd_detailed_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
