"""
Dataset for RDD-DiT-3D (全稀疏版本).

数据格式: 每个 .pt 文件只包含最高分辨率 R 的 embeddings + coords。
低分辨率数据在训练时按照 parent-occupancy 展开逻辑实时计算。

核心设计:
  - R_0 (最粗): 全 grid (R_0^3 tokens)，GT 有数据的 cell 通过 mean pool 获得 embedding，
                空 cell 的 embedding 为零向量。
  - R_k (k>0): 从 R_{k-1} 中 "非空" 的 parent cell 展开 2^3=8 个子 cell。
                "非空" 定义: 该 parent cell 对应的 GT 数据中有至少一个 occupied token。
  - 这与采样时一致: 采样时从 coarse → fine，每阶段去噪后判断哪些 cell 非空，再展开。
  - 训练时，空 cell 也参与训练，模型需要学会输出接近零的 embedding（表示空）。
"""

import os
import glob
import torch
from torch.utils.data import Dataset


class SparseVolumeDataset(Dataset):
    """
    加载预处理的稀疏 .pt 文件。

    每个文件包含:
    {
        'embeddings': [N, C],   # R=256 下的 cell embeddings
        'coords': [N, 3],       # R=256 下的 integer grid coords
        'resolution': int,      # 编码分辨率 (256)
        'label': int,           # class label
    }
    """

    def __init__(self, data_dir, max_tokens=None, stage_resolutions=None):
        """
        Args:
            data_dir: 预处理 .pt 文件目录
            max_tokens: 若设置，过滤掉最高分辨率 stage token 数超过此值的样本
            stage_resolutions: 与 max_tokens 配合使用，用于计算 token 数
        """
        super().__init__()
        self.data_dir = data_dir
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        assert len(all_files) > 0, f"No .pt files found in {data_dir}"

        if max_tokens is not None and stage_resolutions is not None:
            # 过滤: 通过检查最高分辨率 stage 的 token 数
            self.file_list = []
            max_R = max(stage_resolutions)
            second_R = sorted(stage_resolutions)[-2]  # R=128
            ratio = max_R // second_R  # 256/128 = 2
            # token 数 = occupied_at_second_R * (ratio^3)
            # 等价于检查 second_R 下 occupied count <= max_tokens / ratio^3
            max_parent_occ = max_tokens // (ratio ** 3)
            source_ratio = None  # 延迟计算
            for f in all_files:
                data = torch.load(f, map_location='cpu', weights_only=False)
                coords = data['coords']
                source_R = data['resolution']
                r = source_R // second_R
                coords_at_secondR = coords // r
                flat = _coords_to_flat(coords_at_secondR, second_R)
                n_occupied = flat.unique().shape[0]
                if n_occupied <= max_parent_occ:
                    self.file_list.append(f)
            print(f"[Dataset] max_tokens={max_tokens} filter: {len(self.file_list)}/{len(all_files)} samples retained "
                  f"(max R={second_R} occupied <= {max_parent_occ})")
        else:
            self.file_list = all_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx], map_location='cpu')
        return {
            'embeddings': data['embeddings'],  # [N, C]
            'coords': data['coords'],          # [N, 3]
            'resolution': data['resolution'],
            'label': data['label'],
        }


def _coords_to_flat(coords, R):
    """Convert [N, 3] int coords to flat indices in R^3 grid."""
    return coords[:, 0] * (R * R) + coords[:, 1] * R + coords[:, 2]


def _flat_to_coords(flat, R):
    """Convert flat indices back to [N, 3] coords."""
    x = flat // (R * R)
    y = (flat % (R * R)) // R
    z = flat % R
    return torch.stack([x, y, z], dim=-1).long()


def compute_stage_data(embeddings, coords, source_R, stage_resolutions):
    """
    为单个样本计算各 stage 的 token 集合，遵循 parent-occupancy 展开逻辑。
    全量向量化实现，无 Python for 循环。

    逻辑:
      1. 对最粗分辨率 R_0: 全 grid。用 GT 数据做 mean pool 填充有内容的 cell，空 cell = 零向量。
      2. 对后续分辨率 R_k: 找到 R_{k-1} 中非空的 parent cell，展开其 2^3 个子 cell。
         子 cell 如果有 GT 数据则 mean pool，否则为零向量。

    Args:
        embeddings: [N, C] 最高分辨率的 GT embeddings
        coords: [N, 3] 最高分辨率的 GT coords (integers in [0, source_R))
        source_R: int, 最高分辨率 (256)
        stage_resolutions: list of int, 从粗到细 [8, 16, 32, 64, 128, 256]

    Returns:
        dict: stage_embs[R], stage_coords[R], stage_occupied[R] for each R
    """
    C = embeddings.shape[1]

    # 预计算: 各分辨率的 GT occupied flat indices + mean-pooled embeddings
    # 用 sparse tensor 存储, 便于后续 O(1) 索引
    gt_at_res = {}
    for R in stage_resolutions:
        ratio = source_R // R
        coords_at_R = coords // ratio  # [N, 3]
        flat_at_R = _coords_to_flat(coords_at_R, R)  # [N]
        unique_flat, inverse = flat_at_R.unique(return_inverse=True)

        M = unique_flat.shape[0]
        sum_emb = torch.zeros(M, C, dtype=embeddings.dtype)
        count = torch.zeros(M, dtype=embeddings.dtype)
        sum_emb.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, C), embeddings)
        count.scatter_add_(0, inverse, torch.ones(embeddings.shape[0], dtype=embeddings.dtype))
        mean_emb = sum_emb / count.clamp(min=1.0).unsqueeze(-1)

        # 创建 dense lookup: flat_idx -> embedding (用 R^3 大小的数组，对 R<=32 可行)
        # 对高分辨率，用 sparse 方式 (sorted flat + searchsorted)
        gt_at_res[R] = {
            'unique_flat': unique_flat,  # [M] sorted by unique
            'mean_emb': mean_emb,        # [M, C]
        }

    # 预计算 offsets (2x2x2)
    offsets_2 = torch.tensor([[ox, oy, oz] for ox in range(2) for oy in range(2) for oz in range(2)], dtype=torch.long)  # [8, 3]

    # Stage-by-stage 展开
    stage_embs = {}
    stage_coords = {}
    stage_occupied = {}

    for stage_idx, R in enumerate(stage_resolutions):
        gt_info = gt_at_res[R]
        gt_flat_sorted, sort_idx = gt_info['unique_flat'].sort()
        gt_emb_sorted = gt_info['mean_emb'][sort_idx]  # [M, C]

        if stage_idx == 0:
            # 最粗分辨率: 全 grid
            N_full = R ** 3
            all_flat = torch.arange(N_full, dtype=torch.long)
            all_coords_R = _flat_to_coords(all_flat, R)  # [R^3, 3]

            # 向量化填充: searchsorted 找到 GT 位置
            emb_full = torch.zeros(N_full, C, dtype=embeddings.dtype)
            occ_mask = torch.zeros(N_full, dtype=torch.bool)

            # 直接用 unique_flat 做 index assignment
            emb_full[gt_info['unique_flat']] = gt_info['mean_emb']
            occ_mask[gt_info['unique_flat']] = True

            stage_embs[R] = emb_full
            stage_coords[R] = all_coords_R
            stage_occupied[R] = occ_mask

        else:
            prev_R = stage_resolutions[stage_idx - 1]
            prev_occ = stage_occupied[prev_R]  # [N_prev] bool
            prev_coords_all = stage_coords[prev_R]  # [N_prev, 3]

            occupied_parent_coords = prev_coords_all[prev_occ]  # [M_occ, 3]

            if occupied_parent_coords.shape[0] == 0:
                stage_embs[R] = torch.zeros(0, C, dtype=embeddings.dtype)
                stage_coords[R] = torch.zeros(0, 3, dtype=torch.long)
                stage_occupied[R] = torch.zeros(0, dtype=torch.bool)
                continue

            # 展开 children: [M_occ, 1, 3] * 2 + [1, 8, 3] offsets
            children_coords = (occupied_parent_coords * 2).unsqueeze(1) + offsets_2.unsqueeze(0)
            children_coords = children_coords.reshape(-1, 3)  # [M_occ * 8, 3]
            N_children = children_coords.shape[0]

            # 向量化查找: searchsorted
            children_flat = _coords_to_flat(children_coords, R)  # [N_children]
            pos = torch.searchsorted(gt_flat_sorted, children_flat)
            pos_clamped = pos.clamp(max=gt_flat_sorted.shape[0] - 1)
            found = (gt_flat_sorted[pos_clamped] == children_flat)

            emb_children = torch.zeros(N_children, C, dtype=embeddings.dtype)
            occ_children = found

            # 批量填充找到的 embeddings
            if found.any():
                emb_children[found] = gt_emb_sorted[pos_clamped[found]]

            stage_embs[R] = emb_children
            stage_coords[R] = children_coords
            stage_occupied[R] = occ_children

    return stage_embs, stage_coords, stage_occupied


def sparse_collate_fn(batch, stage_resolutions=None):
    """
    Custom collate: 按 parent-occupancy 展开逻辑计算各 stage 的 token 集合。

    训练时的 token 集合与采样时完全一致:
      - R_0: 全 grid (R_0^3)
      - R_k: 从 R_{k-1} 的 occupied parents 展开 2^3 个 children

    Returns dict:
        embeddings_{R}: [B, N_max_R, C]
        coords_{R}: [B, N_max_R, 3]
        padding_mask_{R}: [B, N_max_R] bool (True=valid token, False=padding)
        occupied_{R}: [B, N_max_R] bool (True=GT occupied, False=empty)
        label: [B] long
    """
    B = len(batch)
    source_R = batch[0]['resolution']
    result = {'label': torch.tensor([b['label'] for b in batch], dtype=torch.long)}

    # 计算每个样本的各 stage 数据
    all_stage_data = []
    for i in range(B):
        emb = batch[i]['embeddings']
        coords = batch[i]['coords']
        stage_embs, stage_coords, stage_occupied = compute_stage_data(
            emb, coords, source_R, stage_resolutions
        )
        all_stage_data.append((stage_embs, stage_coords, stage_occupied))

    # Collate: pad to max N per stage
    for R in stage_resolutions:
        per_sample_emb = [all_stage_data[i][0][R] for i in range(B)]
        per_sample_coords = [all_stage_data[i][1][R] for i in range(B)]
        per_sample_occ = [all_stage_data[i][2][R] for i in range(B)]

        Ns = [e.shape[0] for e in per_sample_emb]
        N_max = max(Ns) if Ns else 0
        if N_max == 0:
            N_max = 1  # avoid zero-dim tensors

        C = per_sample_emb[0].shape[1] if per_sample_emb[0].shape[0] > 0 else batch[0]['embeddings'].shape[1]

        emb_padded = torch.zeros(B, N_max, C)
        coord_padded = torch.zeros(B, N_max, 3, dtype=torch.long)
        mask = torch.zeros(B, N_max, dtype=torch.bool)
        occ = torch.zeros(B, N_max, dtype=torch.bool)

        for i in range(B):
            n = Ns[i]
            if n > 0:
                emb_padded[i, :n] = per_sample_emb[i]
                coord_padded[i, :n] = per_sample_coords[i]
                mask[i, :n] = True
                occ[i, :n] = per_sample_occ[i]

        result[f'embeddings_{R}'] = emb_padded
        result[f'coords_{R}'] = coord_padded
        result[f'padding_mask_{R}'] = mask
        result[f'occupied_{R}'] = occ

    return result
