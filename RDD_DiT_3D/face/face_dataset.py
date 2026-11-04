"""
Dataset and preprocessing for RDD-DiT-3D (Face 版本 — 基于三角形面片).

FaceSparseDataset: 加载预计算的稀疏 .pt 文件 (active embeddings + coords)
MeshFacePreprocessor: 将原始 mesh (vertices + faces) 转换为多分辨率稀疏表示
    (embeddings, coords) 使用预训练的 FaceSetTransformerAE 编码器。

与 vertex 版本 (RDD_DiT_3D/dataset.py) 的区别：
  - 输入是三角形面片 (faces) 而不是点云
  - 使用 face centroid 来分配 grid cell
  - 每个 cell 编码的是该 cell 内面片的局部坐标 [F_k, 9]
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class FaceSparseDataset(Dataset):
    """
    加载预计算的稀疏面片表示 (.pt 文件).

    每个文件包含:
    {
        'embeddings_R': [N_R, C],     # 分辨率 R 下的 active embeddings
        'coords_R': [N_R, 3],         # 分辨率 R 下的整数 grid 坐标
        'label': int,                  # 类别标签
    }
    对于 3 级 RDD (resolutions [4, 8, 16])，文件包含：
        embeddings_4, coords_4, embeddings_8, coords_8, embeddings_16, coords_16
    """

    def __init__(self, data_dir, stage_resolutions=(4, 8, 16)):
        super().__init__()
        self.data_dir = data_dir
        self.stage_resolutions = stage_resolutions
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        assert len(self.file_list) > 0, f"No .pt files found in {data_dir}"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx], map_location='cpu')
        label = data['label']

        # 收集每个分辨率的数据
        result = {'label': label}
        for R in self.stage_resolutions:
            emb_key = f'embeddings_{R}'
            coord_key = f'coords_{R}'
            result[emb_key] = data[emb_key]     # [N_R, C]
            result[coord_key] = data[coord_key]  # [N_R, 3]

        return result


def face_sparse_collate_fn(batch):
    """
    自定义 collate 函数，处理变长稀疏 tokens。
    对每个分辨率，pad 到 batch 内最大 N，返回 padding_mask。

    Returns dict:
        embeddings_R: [B, N_max_R, C]
        coords_R: [B, N_max_R, 3]
        padding_mask_R: [B, N_max_R] bool (True=有效)
        label: [B] long
    """
    # 从 batch 中推断分辨率列表
    stage_resolutions = []
    for key in batch[0]:
        if key.startswith('embeddings_'):
            R = int(key.split('_')[1])
            stage_resolutions.append(R)
    stage_resolutions = sorted(stage_resolutions)

    B = len(batch)
    result = {'label': torch.tensor([b['label'] for b in batch], dtype=torch.long)}

    for R in stage_resolutions:
        emb_key = f'embeddings_{R}'
        coord_key = f'coords_{R}'

        # 找到该分辨率下的最大 N
        Ns = [batch[i][emb_key].shape[0] for i in range(B)]
        N_max = max(Ns)
        C = batch[0][emb_key].shape[1]

        # Pad
        emb_padded = torch.zeros(B, N_max, C)
        coord_padded = torch.zeros(B, N_max, 3, dtype=torch.long)
        mask = torch.zeros(B, N_max, dtype=torch.bool)

        for i in range(B):
            n = Ns[i]
            emb_padded[i, :n] = batch[i][emb_key]
            coord_padded[i, :n] = batch[i][coord_key]
            mask[i, :n] = True

        result[emb_key] = emb_padded
        result[coord_key] = coord_padded
        result[f'padding_mask_{R}'] = mask

    return result


class MeshFacePreprocessor:
    """
    将 Mesh (vertices + faces) 转换为多分辨率稀疏面片 embeddings。

    Pipeline:
    1. 归一化顶点到 [-0.5, 0.5]^3
    2. 对每个目标分辨率 R：
       a. 计算每个面片的质心，按质心分配到 R^3 的 grid cells
       b. 对每个非空 cell 内的面片计算局部坐标，用 FaceSetTransformerAE 编码 → embedding
       c. 记录 cell 的 (i, j, k) 坐标
    3. 保存为稀疏格式：(embeddings_R, coords_R) per resolution
    """

    def __init__(self, encoder, resolution_list=(4, 8, 16), f_max=10, device='cuda'):
        """
        Args:
            encoder: 预训练的 FaceSetTransformerAE (有 .encode() 方法)
            resolution_list: 需要预计算的分辨率列表
            f_max: 每个 cell 内最大面片数
            device: 计算设备
        """
        self.encoder = encoder
        self.resolution_list = resolution_list
        self.f_max = f_max
        self.device = device
        self.encoder.eval()

    def normalize_mesh(self, vertices):
        """
        归一化顶点坐标到 [-0.5, 0.5]^3。

        Args:
            vertices: [V, 3] tensor

        Returns:
            normalized vertices: [V, 3] tensor
        """
        center = (vertices.max(0)[0] + vertices.min(0)[0]) / 2
        vertices = vertices - center
        scale = vertices.abs().max()
        if scale > 0:
            vertices = vertices / (2 * scale)
        return vertices

    def assign_faces_to_grid(self, vertices, faces, resolution):
        """
        根据面片质心将面片分配到 grid cells。

        Args:
            vertices: [V, 3] 归一化后的顶点坐标 (在 [-0.5, 0.5]^3 中)
            faces: [F, 3] int tensor, 面片顶点索引
            resolution: 网格分辨率 R

        Returns:
            dict: (i, j, k) -> list of face indices
        """
        R = resolution
        # 计算每个面片的质心: (v1 + v2 + v3) / 3
        v1 = vertices[faces[:, 0]]  # [F, 3]
        v2 = vertices[faces[:, 1]]  # [F, 3]
        v3 = vertices[faces[:, 2]]  # [F, 3]
        centroids = (v1 + v2 + v3) / 3.0  # [F, 3]

        # 将质心映射到 grid 坐标: ((centroid + 0.5) * R).long().clamp(0, R-1)
        grid_coords = ((centroids + 0.5) * R).long().clamp(0, R - 1)  # [F, 3]

        cells = {}
        for face_idx in range(faces.shape[0]):
            key = tuple(grid_coords[face_idx].tolist())
            if key not in cells:
                cells[key] = []
            cells[key].append(face_idx)
        return cells

    def compute_local_coords(self, vertices, faces, face_indices, cell_origin, cell_size):
        """
        计算指定面片在 cell 内的局部坐标，并执行 centered + scale 归一化。

        流程:
          1. local = (vertex - cell_origin) / cell_size → 重心在 [0, 1]³
          2. centered = local - 0.5 → 以 cell 中心为原点
          3. scale = |centered|.max() + 1e-6 → 最大绝对值
          4. normalized = centered / scale → 在 [-0.5, 0.5] 内，某顶点恰好碰到 ±0.5

        Args:
            vertices: [V, 3] 归一化后的全局顶点坐标
            faces: [F, 3] int tensor, 面片顶点索引
            face_indices: 当前 cell 内的面片索引列表
            cell_origin: [3] tensor, cell 的原点坐标 (世界坐标系)
            cell_size: float, cell 的边长

        Returns:
            normalized_faces: [F_k, 9] tensor, 归一化后的局部坐标 (在 [-0.5, 0.5] 范围)
            scale: float, 缩放因子 (用于还原: centered = normalized * scale, local = centered + 0.5)
        """
        local_faces = []
        for fi in face_indices:
            # 获取该面片的 3 个顶点
            v_indices = faces[fi]  # [3]
            verts = vertices[v_indices]  # [3, 3]
            # 计算局部坐标
            local_verts = (verts - cell_origin.unsqueeze(0)) / cell_size  # [3, 3]
            # 展平为 9 维
            local_faces.append(local_verts.reshape(9))

        local_faces_tensor = torch.stack(local_faces, dim=0)  # [F_k, 9]

        # Centered: 以 cell 中心 (0.5) 为原点
        centered = local_faces_tensor - 0.5

        # Scale: max_abs / 0.5，保证归一化后在 [-0.5, 0.5]
        # 还原公式: centered = normalized * scale, 即 normalized = centered / scale
        # 因此 scale 需要满足: |centered|.max() / scale = 0.5
        # → scale = |centered|.max() / 0.5
        max_abs = centered.abs().max().item() + 1e-6
        scale = max_abs / 0.5  # 除以 scale 后最大值恰好为 ±0.5

        # Normalize: 缩放到 [-0.5, 0.5]，某顶点恰好碰到 ±0.5
        normalized = centered / scale

        return normalized, scale

    @torch.no_grad()
    def encode_cell(self, face_coords_normalized, num_faces, scale_val):
        """
        编码单个 cell 内的面片集合。

        Args:
            face_coords_normalized: [F_k, 9] 归一化后的局部坐标 ([-0.5, 0.5])
            num_faces: 有效面片数量
            scale_val: float, 该 cell 的 scale 因子

        Returns:
            [C] embedding 向量
        """
        # Pad 到 f_max
        F_k = face_coords_normalized.shape[0]
        if F_k < self.f_max:
            padding = torch.zeros(self.f_max - F_k, 9, device=self.device)
            input_padded = torch.cat([face_coords_normalized, padding], dim=0)  # [f_max, 9]
        else:
            input_padded = face_coords_normalized[:self.f_max]  # 截断到 f_max

        # 创建 mask: True 表示有效位置
        mask = torch.zeros(self.f_max, dtype=torch.bool, device=self.device)
        mask[:min(F_k, self.f_max)] = True

        # Scale tensor: [1, 1]
        scale_tensor = torch.tensor([[scale_val]], dtype=torch.float32, device=self.device)

        # 编码: [1, F_MAX, 9] + mask [1, F_MAX] + scale [1, 1] → [1, C]
        embedding = self.encoder.encode(
            input_padded.unsqueeze(0),  # [1, f_max, 9]
            mask.unsqueeze(0),          # [1, f_max]
            scale=scale_tensor           # [1, 1]
        )
        return embedding.squeeze(0)  # [C]

    @torch.no_grad()
    def process(self, vertices, faces, label):
        """
        将单个 mesh 转换为多分辨率稀疏面片表示。

        Args:
            vertices: [V, 3] tensor, 顶点坐标
            faces: [F, 3] int tensor, 面片顶点索引
            label: int 类别标签

        Returns:
            dict: {embeddings_R, coords_R for each R, label}
        """
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).float()
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).long()

        vertices = vertices.to(self.device)
        faces = faces.to(self.device)

        # 归一化顶点
        vertices = self.normalize_mesh(vertices)

        result = {'label': label}

        for R in self.resolution_list:
            # 将面片分配到 grid cells
            cells = self.assign_faces_to_grid(vertices, faces, R)

            embeddings_list = []
            coords_list = []

            cell_size = 1.0 / R  # cell 边长

            for (i, j, k), face_indices in cells.items():
                # 计算 cell 原点 (世界坐标系)
                # cell_origin = grid_coord / R - 0.5
                cell_origin = torch.tensor(
                    [i / R - 0.5, j / R - 0.5, k / R - 0.5],
                    dtype=torch.float32, device=self.device
                )

                # 计算该 cell 内面片的归一化局部坐标 + scale
                normalized_coords, scale_val = self.compute_local_coords(
                    vertices, faces, face_indices[:self.f_max],
                    cell_origin, cell_size
                )

                # 编码（传入 scale）
                embedding = self.encode_cell(normalized_coords, len(face_indices), scale_val)
                embeddings_list.append(embedding)
                coords_list.append(torch.tensor([i, j, k], dtype=torch.long))

            if len(embeddings_list) > 0:
                result[f'embeddings_{R}'] = torch.stack(embeddings_list).cpu()  # [N_R, C]
                result[f'coords_{R}'] = torch.stack(coords_list).cpu()          # [N_R, 3]
            else:
                # 边界情况：没有非空 cell（正常数据不应出现）
                # 用一次 dummy forward 获取 embedding 维度
                dummy_input = torch.zeros(1, self.f_max, 9, device=self.device)
                dummy_mask = torch.ones(1, self.f_max, dtype=torch.bool, device=self.device)
                dummy_scale = torch.ones(1, 1, device=self.device)
                C = self.encoder.encode(dummy_input, dummy_mask, scale=dummy_scale).shape[1]
                result[f'embeddings_{R}'] = torch.zeros(0, C)
                result[f'coords_{R}'] = torch.zeros(0, 3, dtype=torch.long)

        return result

    def preprocess_dataset(self, meshes, output_dir):
        """
        批量预处理 mesh 数据集。

        Args:
            meshes: list of (vertices, faces, label) tuples
                - vertices: [V_i, 3] 顶点坐标
                - faces: [F_i, 3] 面片顶点索引
                - label: int 类别标签
            output_dir: 保存 .pt 文件的目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for idx, (vertices, faces, label) in enumerate(meshes):
            result = self.process(vertices, faces, label)
            save_path = os.path.join(output_dir, f"{idx:06d}.pt")
            torch.save(result, save_path)
            if (idx + 1) % 100 == 0:
                print(f"已预处理 {idx + 1}/{len(meshes)}")
        print(f"完成。已保存 {len(meshes)} 个稀疏面片表示到 {output_dir}")
