"""
RDD-DiT-3D Face Sampling Script (全稀疏版本).

功能：通过扩散模型生成 face embeddings，然后通过 FaceSetTransformerAE 解码为
triangle soup，最后合并顶点形成完整 mesh。

与 RDD_DiT_3D/sample.py 的 sample_batch_v2 逻辑一致，增加了面片解码和 mesh 后处理步骤。
"""

import os
import sys
import argparse
import torch
import numpy as np

# 导入 vertex 版本的模型和扩散模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model import DiT3D_RDD, DiT3D_RDD_models
from diffusion_3d import (
    GaussianDiffusion3D,
    SpacedDiffusion3D,
    create_diffusion_3d,
    compute_drift_sparse,
    upsample_coords,
    compute_active_mask_sparse,
    get_named_beta_schedule,
)

# 导入本目录的 face decoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_set_transformer import FaceSetTransformerAE


# ===========================================================================
# Mesh 解码与后处理
# ===========================================================================

def decode_embeddings_to_mesh(embeddings, coords, face_decoder, resolution,
                              f_max=10, exist_threshold=0.5):
    """
    将生成的 face embeddings 解码为 triangle soup。

    Args:
        embeddings: [N_active, C] — 活跃 cell 的 face embeddings
        coords: [N_active, 3] — 整数 grid 坐标
        face_decoder: FaceSetTransformerAE 模型 (有 decode_from_bottleneck 方法)
        resolution: int — 当前网格分辨率 (如 16 或 512)
        f_max: int — 每个 cell 最多解码的面片数
        exist_threshold: float — 面片存在性阈值 (sigmoid 后)

    Returns:
        vertices: [N_total_verts, 3] numpy array — 所有三角形顶点
        faces: [N_total_faces, 3] numpy array — 面片索引 (每个三角形 3 个顶点索引)
    """
    device = embeddings.device
    N_active = embeddings.shape[0]
    cell_size = 1.0 / resolution

    all_vertices = []  # 收集所有顶点
    all_faces = []     # 收集所有面片索引
    vertex_offset = 0  # 当前顶点偏移量

    # 批量解码所有 cell 的 embeddings
    with torch.no_grad():
        # decode_from_bottleneck 输入: [B, bottleneck_dim]
        # 输出: faces_local [B, F_MAX, 9], existence [B, F_MAX], scale [B, 1] or None
        pred_coords, pred_existence, pred_scale = face_decoder.decode_from_bottleneck(embeddings)

    # pred_coords: [N_active, F_MAX, 9] — 局部坐标
    # pred_existence: [N_active, F_MAX] — 存在性 logits
    existence_prob = torch.sigmoid(pred_existence)  # [N_active, F_MAX]

    # 转换为 numpy 进行后续处理
    pred_coords_np = pred_coords.cpu().numpy()       # [N_active, F_MAX, 9]
    existence_prob_np = existence_prob.cpu().numpy()  # [N_active, F_MAX]
    coords_np = coords.cpu().numpy()                 # [N_active, 3]
    pred_scale_np = pred_scale.cpu().numpy() if pred_scale is not None else None  # [N_active, 1]

    for i in range(N_active):
        # 计算该 cell 的全局原点
        cell_origin = coords_np[i].astype(np.float64) / resolution - 0.5  # [3]

        # 筛选存在的面片
        exist_mask = existence_prob_np[i] > exist_threshold  # [F_MAX]
        n_valid = exist_mask.sum()
        if n_valid == 0:
            continue

        # 获取有效面片的局部坐标 [n_valid, 9]
        faces_local = pred_coords_np[i][exist_mask]  # [n_valid, 9]

        # 将 9 维展开为 3 个顶点 x 3 坐标: [n_valid, 3, 3]
        faces_local_3v = faces_local.reshape(n_valid, 3, 3)

        # 局部坐标还原：
        # 如果有 scale: normalized coords 在 [-0.5, 0.5]
        #   centered = normalized * scale_val → 局部偏移（相对于 cell 中心）
        #   local = centered + 0.5 → 在 [0, 1] 附近的 cell 局部坐标
        #   global = local * cell_size + cell_origin
        # 如果没有 scale: 直接 local * cell_size + cell_origin
        if pred_scale_np is not None:
            scale_val = pred_scale_np[i, 0]  # 标量
            faces_local_3v = faces_local_3v * scale_val + 0.5  # 还原到 cell 局部坐标 [0,1] 附近

        faces_global = faces_local_3v * cell_size + cell_origin[np.newaxis, np.newaxis, :]

        # 每个面片贡献 3 个顶点和 1 个三角形
        for f_idx in range(n_valid):
            v0, v1, v2 = faces_global[f_idx]  # 3 个顶点，每个 [3]
            all_vertices.append(v0)
            all_vertices.append(v1)
            all_vertices.append(v2)
            # 三角形索引
            all_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3

    if len(all_vertices) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    vertices = np.array(all_vertices, dtype=np.float64)  # [N_total_verts, 3]
    faces = np.array(all_faces, dtype=np.int64)          # [N_total_faces, 3]
    return vertices, faces


def merge_triangle_soup(vertices, faces, merge_threshold=0.005):
    """
    合并 triangle soup 中距离小于阈值的顶点，形成共享顶点的 mesh。

    Args:
        vertices: [M, 3] numpy array — 顶点坐标
        faces: [F, 3] numpy array — 面片顶点索引

    Returns:
        merged_vertices: [M', 3] — 合并后顶点
        merged_faces: [F', 3] — 更新后面片索引 (去除退化面)
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    N = len(vertices)

    # 使用空间哈希加速顶点合并
    # 将顶点量化到 merge_threshold 的格子中
    quantized = np.round(vertices / merge_threshold).astype(np.int64)

    # 为每个量化位置分配统一索引
    # 使用字典进行快速查找
    vertex_map = {}  # quantized_key -> new_index
    old_to_new = np.zeros(N, dtype=np.int64)
    new_vertices = []
    new_idx = 0

    for i in range(N):
        key = (quantized[i, 0], quantized[i, 1], quantized[i, 2])
        if key not in vertex_map:
            vertex_map[key] = new_idx
            new_vertices.append(vertices[i])
            new_idx += 1
        old_to_new[i] = vertex_map[key]

    merged_vertices = np.array(new_vertices, dtype=np.float64) if new_vertices else np.zeros((0, 3))

    # 更新面片索引
    merged_faces = old_to_new[faces]  # [F, 3]

    # 去除退化面（两个或更多顶点索引相同的三角形）
    valid_mask = (
        (merged_faces[:, 0] != merged_faces[:, 1]) &
        (merged_faces[:, 1] != merged_faces[:, 2]) &
        (merged_faces[:, 0] != merged_faces[:, 2])
    )
    merged_faces = merged_faces[valid_mask]

    return merged_vertices, merged_faces


# ===========================================================================
# Mesh I/O
# ===========================================================================

def save_mesh(vertices, faces, path):
    """
    保存 mesh 到文件。支持 .obj 和 .ply 格式。

    Args:
        vertices: [V, 3] numpy array
        faces: [F, 3] numpy array (0-indexed)
        path: 输出文件路径
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.obj':
        _save_obj(vertices, faces, path)
    elif ext == '.ply':
        _save_ply(vertices, faces, path)
    elif ext == '.glb':
        _save_glb(vertices, faces, path)
    else:
        # 默认保存为 obj
        _save_obj(vertices, faces, path)


def _save_obj(vertices, faces, path):
    """保存为 OBJ 格式 (文本)。"""
    with open(path, 'w') as f:
        f.write("# Generated by RDD-DiT-3D Face Sampling\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ 面片索引从 1 开始
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _save_ply(vertices, faces, path):
    """保存为 PLY 格式 (文本)。"""
    n_verts = len(vertices)
    n_faces = len(faces)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _save_glb(vertices, faces, path):
    """保存为 GLB 格式 (需要 trimesh 库)。"""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(path)
    except ImportError:
        # trimesh 不可用，降级保存为 obj
        print(f"[WARNING] trimesh 未安装，降级保存为 .obj 格式")
        obj_path = path.replace('.glb', '.obj')
        _save_obj(vertices, faces, obj_path)


# ===========================================================================
# 模型加载
# ===========================================================================

def _load_model(ckpt_path, model_name, in_channels, num_classes, max_resolution, device):
    """加载 DiT3D_RDD 模型，自动检测 decouple 模式。"""
    weights = torch.load(ckpt_path, map_location=device)
    if "ema" in weights:
        state_dict = weights["ema"]
    elif "model" in weights:
        state_dict = weights["model"]
    else:
        state_dict = weights

    d_prediction = any("d_final_layer" in k for k in state_dict.keys())

    model = DiT3D_RDD_models[model_name](
        in_channels=in_channels,
        num_classes=num_classes,
        max_resolution=max_resolution,
        d_prediction=d_prediction,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, d_prediction


def _load_face_decoder(ckpt_path, device, **kwargs):
    """加载 FaceSetTransformerAE face decoder。"""
    # 默认配置
    config = dict(
        dim_in=9,
        dim_hidden=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        bottleneck_dim=128,
        ae_num_heads=4,
        ae_num_mlp_layers=3,
        ae_mlp_factor=4,
        ae_pool='mean',
        f_max=10,
        decode_mode='regression',
        num_bins=128,
        ln=False,
        predict_scale=False,
    )
    config.update(kwargs)

    model = FaceSetTransformerAE(**config).to(device)

    weights = torch.load(ckpt_path, map_location=device)
    if "model" in weights:
        state_dict = weights["model"]
    elif "model_state_dict" in weights:
        state_dict = weights["model_state_dict"]
    elif "ema" in weights:
        state_dict = weights["ema"]
    else:
        state_dict = weights

    # 如果 checkpoint 保存了 args，用其覆盖默认 config 重建模型
    if 'args' in weights and not kwargs:
        saved_args = weights['args']
        for key in ['dim_hidden', 'bottleneck_dim', 'num_heads', 'f_max',
                    'ae_pool', 'decode_mode', 'predict_scale', 'num_encoder_layers',
                    'num_decoder_layers']:
            if key.replace('-', '_') in saved_args:
                config[key] = saved_args[key.replace('-', '_')]
        model = FaceSetTransformerAE(**config).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ===========================================================================
# 采样核心逻辑
# ===========================================================================

@torch.no_grad()
def sample_batch_v2(models, diffusions, batch_size, device, args, empty_embedding=None):
    """
    渐进式多阶段采样 (全稀疏)。

    与 RDD_DiT_3D/sample.py 的 sample_batch_v2 逻辑一致：
    - Stage 0: 在最粗分辨率的全网格上从纯噪声开始去噪
    - 每个 stage 结束后: 通过余弦相似度裁剪空 token，上采样坐标到下一分辨率
    - 使用 noise_filling_sparse 在新坐标上填充噪声

    Returns:
        x: [B, N_final, C] 去噪后的 token embeddings
        coords: [B, N_final, 3] 最终分辨率坐标
        y: [B] 类别标签
    """
    steps_per_stage = [int(s) for s in args.steps_per_stage.split(",")]
    stage_resolutions = [int(r) for r in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)
    thresholds = [args.threshold_1, args.threshold_2]
    all_betas = get_named_beta_schedule(args.schedule, args.diffusion_steps)

    C = args.in_channels
    use_cfg = args.cfg_scale > 1.0

    if not isinstance(models, list):
        models = [models] * num_stages

    # 类别标签
    if args.class_label is not None:
        y = torch.full((batch_size,), args.class_label, dtype=torch.long, device=device)
    else:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)

    if use_cfg:
        y_null = torch.full_like(y, args.num_classes)
        y_cfg = torch.cat([y, y_null], dim=0)

    x = None
    coords = None

    for stage_idx in range(num_stages):
        diff = diffusions[stage_idx]
        R = stage_resolutions[stage_idx]
        model = models[stage_idx]

        if stage_idx == 0:
            # 第一阶段：最粗分辨率全网格
            N = R ** 3
            grid = torch.arange(R, device=device)
            gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing='ij')
            all_coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
            coords = all_coords.unsqueeze(0).expand(batch_size, N, 3).contiguous()
            x = torch.randn(batch_size, N, C, device=device)
        # else: x 和 coords 已由上一 stage 的上采样步骤设置

        # 构建 model_fn 闭包
        N_cur = coords.shape[1]
        padding_mask = torch.ones(batch_size, N_cur, dtype=torch.bool, device=device)

        def _make_fn(m, crds, pmask):
            def model_fn(x_t, t_global, y_in):
                if use_cfg:
                    x_d = torch.cat([x_t, x_t], dim=0)
                    t_d = torch.cat([t_global, t_global], dim=0)
                    c_d = torch.cat([crds, crds], dim=0)
                    p_d = torch.cat([pmask, pmask], dim=0)
                    out = m(x_d, c_d, t_d, y_cfg, padding_mask=p_d)
                    if hasattr(m, 'd_prediction') and m.d_prediction:
                        out = out[..., :m.out_channels]
                    cond, uncond = out.chunk(2, dim=0)
                    return uncond + args.cfg_scale * (cond - uncond)
                else:
                    out = m(x_t, crds, t_global, y_in, padding_mask=pmask)
                    if hasattr(m, 'd_prediction') and m.d_prediction:
                        out = out[..., :m.out_channels]
                    return out
            return model_fn

        model_fn = _make_fn(model, coords, padding_mask)

        # 逆扩散去噪
        shape = (batch_size, N_cur, C)
        if args.sampler == "jump_step":
            x = diff.jump_step_sample_loop(
                model_fn, shape, y, device,
                coords=coords, resolution=R,
                x_init=x, deterministic=not args.jump_step_stochastic,
            )
        else:
            x = diff.p_sample_loop(model_fn, shape, y, device, coords=coords, resolution=R, x_init=x)

        # 阶段过渡：裁剪 + 上采样
        if stage_idx < num_stages - 1:
            next_R = stage_resolutions[stage_idx + 1]
            t_start_next = sum(steps_per_stage[:stage_idx + 1])
            t_end_next = t_start_next + steps_per_stage[stage_idx + 1]
            betas_next = all_betas[t_start_next:t_end_next]

            # 裁剪空 token
            if args.use_pruning and empty_embedding is not None:
                threshold = thresholds[stage_idx] if stage_idx < len(thresholds) else 0.90
                active_mask = compute_active_mask_sparse(
                    x, coords, empty_embedding, threshold, R,
                    dilation_kernel=args.dilation_kernel,
                    random_keep_ratio=args.random_keep,
                )

                # 每个 batch 收集活跃 token
                max_active = int(active_mask.sum(dim=1).max().item())
                x_active = torch.zeros(batch_size, max_active, C, device=device)
                coords_active = torch.zeros(batch_size, max_active, 3, dtype=torch.long, device=device)
                for b in range(batch_size):
                    idx = active_mask[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    x_active[b, :n] = x[b, idx]
                    coords_active[b, :n] = coords[b, idx]
            else:
                x_active = x
                coords_active = coords

            # 上采样坐标到下一分辨率
            coords_high = upsample_coords(coords_active, next_R)

            # 噪声填充：从干净的 x_active 在 coords_active → 在 coords_high 处加噪
            # 使用下一 stage 的 diffusion 对象的 alphas_cumprod[-1] 确保噪声水平
            # 与采样 loop 的起始时刻一致（对 SpacedDiffusion 至关重要）。
            next_diff = diffusions[stage_idx + 1]
            x = GaussianDiffusion3D.noise_filling_sparse(
                x_active, coords_active, coords_high,
                steps_per_stage[stage_idx + 1], betas_next,
                alpha_cumprod_T=float(next_diff.alphas_cumprod[-1])
            )
            coords = coords_high

    return x, coords, y


@torch.no_grad()
def sample_and_decode(models, diffusions, face_decoder, batch_size, device, args,
                      empty_embedding=None):
    """
    采样并解码为 mesh。

    1. 调用 sample_batch_v2 生成 embeddings + coords
    2. 对 batch 中每个样本：
       - 提取活跃 token
       - 调用 decode_embeddings_to_mesh 解码为 triangle soup
       - 调用 merge_triangle_soup 合并顶点

    Returns:
        meshes: list of (vertices, faces) tuples
        labels: [B] 类别标签
    """
    # 获取最终分辨率
    stage_resolutions = [int(r) for r in args.stage_resolutions.split(",")]
    final_resolution = stage_resolutions[-1]

    # 采样生成 embeddings
    x, coords, y = sample_batch_v2(models, diffusions, batch_size, device, args, empty_embedding)

    meshes = []
    f_max = getattr(face_decoder, 'f_max', 10)

    for b in range(batch_size):
        # 提取该样本的有效 token (非零坐标)
        # 注意：padding 部分的坐标为 0，需要判断真正活跃的 token
        # 通过检查 embedding 的范数来判断是否为 padding
        emb_b = x[b]           # [N, C]
        coords_b = coords[b]   # [N, 3]

        # 非 padding token：norm > 0 的 token 视为活跃
        norms = emb_b.norm(dim=-1)  # [N]
        active_mask = norms > 1e-6
        emb_active = emb_b[active_mask]      # [N_active, C]
        coords_active = coords_b[active_mask]  # [N_active, 3]

        if emb_active.shape[0] == 0:
            meshes.append((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
            continue

        # 解码 embeddings 为 triangle soup
        vertices, faces = decode_embeddings_to_mesh(
            emb_active, coords_active, face_decoder, final_resolution,
            f_max=f_max, exist_threshold=args.exist_threshold,
        )

        # 合并顶点形成完整 mesh
        vertices, faces = merge_triangle_soup(vertices, faces, args.merge_threshold)

        meshes.append((vertices, faces))

    return meshes, y


# ===========================================================================
# 参数解析
# ===========================================================================

def create_argparser():
    parser = argparse.ArgumentParser(description="Sample from RDD-DiT-3D Face (Sparse)")

    # --- Model ---
    parser.add_argument("--model", type=str, default="DiT3D-RDD-B",
                        choices=list(DiT3D_RDD_models.keys()))
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=55)
    parser.add_argument("--max-resolution", type=int, default=16,
                        help="位置编码归一化的最大分辨率")

    # --- Checkpoint: unified mode ---
    parser.add_argument("--ckpt", type=str, default=None,
                        help="统一模型 checkpoint")
    # --- Checkpoint: multi-model mode ---
    parser.add_argument("--ckpt-per-stage", type=str, nargs="+", default=None,
                        help="每个 stage 的 checkpoint，如 --ckpt-per-stage s0.pt s1.pt s2.pt")
    parser.add_argument("--model-per-stage", type=str, nargs="+", default=None,
                        help="每个 stage 的模型名 (默认全部使用 --model)")

    # --- Face Decoder ---
    parser.add_argument("--face-decoder-ckpt", type=str, required=True,
                        help="FaceSetTransformerAE 的 checkpoint 路径")
    parser.add_argument("--exist-threshold", type=float, default=0.5,
                        help="面片存在性阈值 (sigmoid 概率)")
    parser.add_argument("--merge-threshold", type=float, default=0.005,
                        help="顶点合并距离阈值")
    parser.add_argument("--output-format", type=str, default="obj",
                        choices=["obj", "ply", "glb"],
                        help="输出 mesh 格式")

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default="./samples_face")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-label", type=int, default=None,
                        help="指定类别标签；None 表示均匀随机采样所有类别")

    # --- Diffusion config ---
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--steps-per-stage", type=str, default="400,300,300")
    parser.add_argument("--stage-resolutions", type=str, default="4,8,16")
    parser.add_argument("--sampling-steps", type=str, default=None,
                        help="每个 stage 的采样步数，如 '100,80,80'。None 表示使用完整步数。")

    # --- Sampler ---
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "jump_step"],
                        help="采样算法: ddpm 或 jump_step")
    parser.add_argument("--jump-step-stochastic", action="store_true", default=False,
                        help="jump-step 采样器中是否添加后验噪声")

    # --- Pruning ---
    parser.add_argument("--use-pruning", action="store_true", default=True,
                        help="启用空 token 裁剪")
    parser.add_argument("--threshold-1", type=float, default=0.85,
                        help="Stage 0 后的空检测阈值")
    parser.add_argument("--threshold-2", type=float, default=0.90,
                        help="Stage 1 后的空检测阈值")
    parser.add_argument("--dilation-kernel", type=int, default=3)
    parser.add_argument("--random-keep", type=float, default=0.05)
    parser.add_argument("--empty-embedding-path", type=str, default=None,
                        help="空 embedding .pt 文件路径")

    # --- 其他 ---
    parser.add_argument("--ref-coords-dir", type=str, default=None,
                        help="预计算的各分辨率参考活跃坐标目录 (用于裁剪交集)")
    parser.add_argument("--seed", type=int, default=0)

    return parser


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    args = create_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    steps_per_stage = [int(s) for s in args.steps_per_stage.split(",")]
    num_stages = len(steps_per_stage)

    # --- 加载扩散模型 ---
    if args.ckpt_per_stage:
        assert len(args.ckpt_per_stage) == num_stages
        model_names = args.model_per_stage or [args.model] * num_stages
        models = []
        for i, ckpt_path in enumerate(args.ckpt_per_stage):
            m, d_pred = _load_model(ckpt_path, model_names[i], args.in_channels,
                                    args.num_classes, args.max_resolution, device)
            models.append(m)
            print(f"Stage {i}: 加载 {model_names[i]} from {ckpt_path} (decouple={d_pred})")
    elif args.ckpt:
        model, d_pred = _load_model(args.ckpt, args.model, args.in_channels,
                                    args.num_classes, args.max_resolution, device)
        models = model
        print(f"统一模型: {args.model} (decouple={d_pred})")
    else:
        raise ValueError("必须提供 --ckpt (统一模型) 或 --ckpt-per-stage (多模型)")

    # --- 加载 Face Decoder ---
    face_decoder = _load_face_decoder(args.face_decoder_ckpt, device)
    print(f"Face Decoder 已加载: {args.face_decoder_ckpt}")

    # --- 创建扩散过程 ---
    sampling_steps = None
    if args.sampling_steps:
        sampling_steps = tuple(int(s) for s in args.sampling_steps.split(","))
    diffusions = create_diffusion_3d(
        schedule_name=args.schedule,
        steps_per_stage=steps_per_stage,
        diffusion_steps=args.diffusion_steps,
        sampling_steps_per_stage=sampling_steps,
    )

    # --- 加载空 embedding ---
    empty_embedding = None
    if args.empty_embedding_path and os.path.exists(args.empty_embedding_path):
        empty_embedding = torch.load(args.empty_embedding_path, map_location=device)
        print(f"已加载空 embedding: {empty_embedding.shape}")

    # --- 采样循环 ---
    num_generated = 0
    batch_idx = 0
    while num_generated < args.num_samples:
        bs = min(args.batch_size, args.num_samples - num_generated)
        print(f"采样 batch {batch_idx} ({num_generated}/{args.num_samples})...")

        # 采样并解码为 mesh
        meshes, labels = sample_and_decode(
            models, diffusions, face_decoder, bs, device, args, empty_embedding
        )

        # 保存每个 mesh
        for i in range(bs):
            vertices, faces = meshes[i]
            sample_idx = num_generated + i
            label = labels[i].item()

            # 保存 mesh 文件
            mesh_filename = f"sample_{sample_idx:06d}_class{label}.{args.output_format}"
            mesh_path = os.path.join(args.output_dir, mesh_filename)
            save_mesh(vertices, faces, mesh_path)

            # 同时保存原始 embedding (可选，用于后续分析)
            pt_path = os.path.join(args.output_dir, f"sample_{sample_idx:06d}_class{label}.pt")
            torch.save({
                'vertices': vertices,
                'faces': faces,
                'label': label,
                'n_vertices': len(vertices),
                'n_faces': len(faces),
            }, pt_path)

        num_generated += bs
        batch_idx += 1

    print(f"完成。共生成 {num_generated} 个 mesh 样本，保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
