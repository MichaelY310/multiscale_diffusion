"""分析不同 Resolution + K_MAX 配置下的训练效率."""
import numpy as np

data_path = '/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/MeshAnythingV2/test_merged.npz'
data = np.load(data_path, allow_pickle=True)
npz_list = data['npz_list']
print(f'Total meshes: {len(npz_list)}')

configs = [
    # (R, K_MAX, stage_resolutions)
    (32, 100, [8, 16, 32]),
    (32, 50, [8, 16, 32]),
    (64, 50, [8, 16, 32, 64]),
    (64, 30, [8, 16, 32, 64]),
    (64, 20, [8, 16, 32, 64]),
    (64, 10, [8, 16, 32, 64]),
    (128, 30, [8, 16, 32, 64, 128]),
    (128, 20, [8, 16, 32, 64, 128]),
    (128, 10, [8, 16, 32, 64, 128]),
    (256, 50, [8, 16, 32, 64, 128, 256]),
    (256, 30, [8, 16, 32, 64, 128, 256]),
    (256, 10, [8, 16, 32, 64, 128, 256]),  # 当前配置
]


def analyze_config(npz_list, R, K_MAX, stage_resolutions, num_samples=2000):
    passed = 0
    vertex_counts = []
    occupied_cells_at_R = []
    final_tokens = []

    for i in range(min(num_samples, len(npz_list))):
        sample = npz_list[i]
        verts = sample['vertices']
        V = verts.shape[0]

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        extent = (vmax - vmin).max()
        if extent < 1e-8:
            continue

        normalized = (verts - vmin) / extent * (R - 1)
        grid_coords = np.floor(normalized).astype(np.int32)
        grid_coords = np.clip(grid_coords, 0, R - 1)

        flat = grid_coords[:, 0] * R * R + grid_coords[:, 1] * R + grid_coords[:, 2]
        unique_cells, counts = np.unique(flat, return_counts=True)
        max_per_cell = counts.max()

        if max_per_cell > K_MAX:
            continue

        passed += 1
        n_occupied = len(unique_cells)
        vertex_counts.append(V)
        occupied_cells_at_R.append(n_occupied)

        # Parent-occupancy 展开: token 数 = occupied_at_second_highest_R * 8
        if len(stage_resolutions) >= 2:
            second_R = stage_resolutions[-2]
            ratio = R // second_R
            coords_at_second = grid_coords // ratio
            flat_second = (coords_at_second[:, 0] * second_R * second_R +
                           coords_at_second[:, 1] * second_R +
                           coords_at_second[:, 2])
            n_occ_second = len(np.unique(flat_second))
            tokens_at_R = n_occ_second * 8
        else:
            tokens_at_R = R ** 3

        final_tokens.append(tokens_at_R)

    return {
        'passed': passed,
        'total': min(num_samples, len(npz_list)),
        'pass_rate': passed / min(num_samples, len(npz_list)),
        'vertex_counts': np.array(vertex_counts) if vertex_counts else np.array([0]),
        'occupied_cells': np.array(occupied_cells_at_R) if occupied_cells_at_R else np.array([0]),
        'final_tokens': np.array(final_tokens) if final_tokens else np.array([0]),
    }


print(f'\nAnalyzing {len(configs)} configurations on first 2000 samples...\n')
header = "{:<18} | {:>6} | {:>10} | {:>14} | {:>11} | {:>11} | {:>8} | {:>6}".format(
    "Config", "Pass%", "Verts(med)", "OccCells(med)", "Tokens(med)", "Tokens(P95)", "Tok/Vert", "Stages")
print(header)
print('-' * len(header))

for R, K, stages in configs:
    result = analyze_config(npz_list, R, K, stages)
    label = f"R={R:3d} K={K:3d}"
    if result['passed'] == 0:
        print(f"{label:<18} | {result['pass_rate']*100:5.1f}% |        N/A |            N/A |         N/A |         N/A |      N/A | {len(stages):>6}")
        continue

    vc_med = np.median(result['vertex_counts'])
    oc_med = np.median(result['occupied_cells'])
    tk_med = np.median(result['final_tokens'])
    tk_p95 = np.percentile(result['final_tokens'], 95)
    tok_vert_ratio = tk_med / vc_med if vc_med > 0 else 0

    print(f"{label:<18} | {result['pass_rate']*100:5.1f}% | {vc_med:10.0f} | {oc_med:14.0f} | {tk_med:11.0f} | {tk_p95:11.0f} | {tok_vert_ratio:8.2f} | {len(stages):>6}")

print()
print("关键指标:")
print("  Pass%: 数据通过率")
print("  Occ cells(med): 最高 R 下 occupied grid cell 中位数")
print("  Tokens(med): parent-occupancy 展开后最高 R stage 的 token 数中位数")
print("  Tok/Vert: token数/vertex数, 越接近1越高效 (=1 意味着没有浪费)")
print("  理想情况: 高 Pass%, 低 Tok/Vert, 合理的 Tokens(P95)")
