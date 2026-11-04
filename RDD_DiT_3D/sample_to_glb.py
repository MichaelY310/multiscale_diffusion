"""
RDD-DiT-3D 端到端采样脚本：Diffusion 采样 → Set Transformer Decode → 输出 GLB。

流程：
  1. 加载训练好的 DiT3D-RDD 模型
  2. 多阶段渐进采样，得到 [B, N_final, 256] embeddings + [B, N_final, 3] coords
  3. 对每个 occupied cell，通过 Set Transformer AR decode 还原 3D 顶点
  4. 输出点云为 GLB 文件（每个点作为独立顶点，无 face）

注意：生成的是点云而非 mesh（因为 face 拓扑信息不在 embedding 里）。
如果需要 mesh，后续可以用 Ball Pivoting / Poisson 重建。

Usage:
  python sample_to_glb.py \
    --dit-ckpt /path/to/dit_checkpoint.pt \
    --st-ckpt /path/to/set_transformer_step_8400000.pt \
    --num-samples 4 --batch-size 4 \
    --output-dir ./generated_glbs
"""

import os
import sys
import argparse
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DiT3D_RDD, DiT3D_RDD_models
from diffusion_3d import (
    create_diffusion_3d,
    GaussianDiffusion3D,
    get_named_beta_schedule,
)
from set_transformer_ae import AEBridgeReconModel, K_MAX, DIM_IN
from set_transformer_ae import load_model as load_st_model


def create_argparser():
    parser = argparse.ArgumentParser(description="RDD-DiT-3D → GLB end-to-end sampling")

    # --- DiT model ---
    parser.add_argument("--dit-ckpt", type=str, required=True, help="DiT3D-RDD checkpoint")
    parser.add_argument("--model", type=str, default="DiT3D-RDD-B",
                        choices=list(DiT3D_RDD_models.keys()))
    parser.add_argument("--in-channels", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--base-size", type=int, default=8,
                        help="Base size for position embedding (must match training)")
    parser.add_argument("--max-resolution", type=int, default=256,
                        help="Max resolution for decode coordinate mapping")

    # --- Occupancy threshold ---
    parser.add_argument("--occupancy-threshold", type=float, default=0.5,
                        help="L2 norm threshold for judging non-empty cells. "
                             "Cells with embedding norm < threshold are pruned.")

    # --- Set Transformer ---
    parser.add_argument("--st-ckpt", type=str,
                        default="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/set_transformer/checkpoints_emd_ae_concat_3d/step_8400000.pt",
                        help="Set Transformer AE checkpoint")
    parser.add_argument("--st-batch-size", type=int, default=512,
                        help="Batch size for Set Transformer decode (cells per forward)")

    # --- Diffusion ---
    parser.add_argument("--stage-resolutions", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--steps-per-stage", type=str, default="100,150,150,200,200,200")
    parser.add_argument("--sampling-steps", type=str, default=None,
                        help="Per-stage sampling steps for speedup, e.g. '30,30,30,30,30,30'")
    parser.add_argument("--schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "jump_step"])
    parser.add_argument("--jump-step-stochastic", action="store_true", default=False)

    # --- Sampling ---
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default="./generated_glbs")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser


def load_dit_model(ckpt_path, model_name, in_channels, num_classes, base_size, device):
    """Load DiT3D-RDD from checkpoint."""
    weights = torch.load(ckpt_path, map_location=device)
    if "ema" in weights:
        state_dict = weights["ema"]
    elif "model" in weights:
        state_dict = weights["model"]
    else:
        state_dict = weights

    d_prediction = any("d_final_layer" in k for k in state_dict.keys())

    # Infer class_dropout_prob from checkpoint embedding table size
    emb_key = "y_embedder.embedding_table.weight"
    class_dropout_prob = 0.0
    if emb_key in state_dict:
        emb_size = state_dict[emb_key].shape[0]
        if emb_size == num_classes:
            class_dropout_prob = 0.0
        else:
            class_dropout_prob = 0.1  # has null token

    model = DiT3D_RDD_models[model_name](
        in_channels=in_channels,
        num_classes=num_classes,
        base_size=base_size,
        d_prediction=d_prediction,
        class_dropout_prob=class_dropout_prob,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, d_prediction


def judge_occupancy(embeddings, threshold):
    """
    判断去噪后的 embeddings 哪些 cell 为"非空"。

    策略: 计算每个 cell embedding 的 L2 norm，norm >= threshold 视为非空。
    训练时空 cell 的 GT embedding 为零向量，所以模型应学会对空 cell 输出接近零的 embedding。

    Args:
        embeddings: [B, N, C] 去噪后的 embeddings
        threshold: float, L2 norm 阈值
    Returns:
        occupancy: [B, N] bool, True=非空
    """
    norms = embeddings.norm(dim=-1)  # [B, N]
    return norms >= threshold


@torch.no_grad()
def diffusion_sample(model, diffusions, batch_size, device, args):
    """
    Run progressive multi-stage diffusion sampling with parent-occupancy expansion.

    逻辑（与训练完全一致）:
      - R_0 (最粗): 全 grid (R_0^3 tokens)
      - R_k (k>0): 仅从 R_{k-1} 中**非空** parent 展开 2^3=8 个 children
      - 非空判断: embedding L2 norm >= occupancy_threshold

    Returns: list of (embeddings_i [N_i, C], coords_i [N_i, 3]) per sample
    """
    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)
    C = args.in_channels
    use_cfg = args.cfg_scale > 1.0
    occ_threshold = args.occupancy_threshold

    # Class labels
    if args.class_label is not None:
        y = torch.full((batch_size,), args.class_label, dtype=torch.long, device=device)
    else:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)

    if use_cfg:
        y_null = torch.full_like(y, args.num_classes)
        y_cfg = torch.cat([y, y_null], dim=0)

    # 逐 sample 处理（token 数量因 occupancy 不同而不同）
    all_results = []

    for sample_i in range(batch_size):
        y_single = y[sample_i:sample_i+1]
        if use_cfg:
            y_cfg_single = torch.cat([y_single, torch.full_like(y_single, args.num_classes)], dim=0)

        x = None
        coords = None

        for stage_idx in range(num_stages):
            diff = diffusions[stage_idx]
            R = stage_resolutions[stage_idx]

            if stage_idx == 0:
                # 全 grid at coarsest resolution
                N = R ** 3
                grid = torch.arange(R, device=device)
                gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing='ij')
                coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
                coords = coords.unsqueeze(0)  # [1, R^3, 3]
                x = torch.randn(1, N, C, device=device)
            else:
                # 从上一阶段非空 parent 展开 children
                prev_R = stage_resolutions[stage_idx - 1]
                ratio = R // prev_R  # = 2

                # 判断上一阶段哪些 cell 非空
                occ_mask = judge_occupancy(x, occ_threshold)  # [1, N_prev]
                occupied_indices = occ_mask[0].nonzero(as_tuple=True)[0]  # [M_occ]

                if occupied_indices.shape[0] == 0:
                    print(f"  Stage {stage_idx} (R={R}): all cells empty, stopping.")
                    break

                # 非空 parent 的坐标和 embedding
                parent_coords = coords[0, occupied_indices]  # [M_occ, 3]
                parent_embs = x[0, occupied_indices]  # [M_occ, C]

                # 展开 2^3=8 个子 cell
                offsets = []
                for ox in range(ratio):
                    for oy in range(ratio):
                        for oz in range(ratio):
                            offsets.append([ox, oy, oz])
                offsets = torch.tensor(offsets, dtype=torch.long, device=device)  # [8, 3]

                # children coords: [M_occ, 8, 3]
                children_coords = (parent_coords * ratio).unsqueeze(1) + offsets.unsqueeze(0)
                children_coords = children_coords.reshape(-1, 3)  # [M_occ*8, 3]
                N_children = children_coords.shape[0]
                coords = children_coords.unsqueeze(0)  # [1, M_occ*8, 3]

                # Noise filling: repeat parent embedding to 8 children, then add noise
                alpha_bar_T = float(diff.alphas_cumprod[-1])
                x_up = parent_embs.unsqueeze(1).expand(-1, ratio**3, -1).reshape(N_children, C)
                x_up = x_up.unsqueeze(0)  # [1, N_children, C]
                noise = torch.randn_like(x_up)
                x = (alpha_bar_T ** 0.5) * x_up + ((1 - alpha_bar_T) ** 0.5) * noise

            # Build model_fn
            N_cur = coords.shape[1]
            padding_mask = torch.ones(1, N_cur, dtype=torch.bool, device=device)

            def _make_fn(m, crds, pmask, y_in, y_cfg_in, res, _use_cfg, _cfg_scale):
                def model_fn(x_t, t_global, y_unused):
                    if _use_cfg:
                        x_d = torch.cat([x_t, x_t], dim=0)
                        t_d = torch.cat([t_global, t_global], dim=0)
                        c_d = torch.cat([crds, crds], dim=0)
                        p_d = torch.cat([pmask, pmask], dim=0)
                        out = m(x_d, c_d, t_d, y_cfg_in, padding_mask=p_d, resolution=res)
                        if hasattr(m, 'd_prediction') and m.d_prediction:
                            out = out[..., :m.out_channels]
                        cond, uncond = out.chunk(2, dim=0)
                        return uncond + _cfg_scale * (cond - uncond)
                    else:
                        out = m(x_t, crds, t_global, y_in, padding_mask=pmask, resolution=res)
                        if hasattr(m, 'd_prediction') and m.d_prediction:
                            out = out[..., :m.out_channels]
                        return out
                return model_fn

            model_fn = _make_fn(model, coords, padding_mask, y_single,
                                y_cfg_single if use_cfg else None,
                                R, use_cfg, args.cfg_scale)

            # Reverse diffusion
            shape = (1, N_cur, C)
            x = diff.p_sample_loop(model_fn, shape, y_single, device, coords=coords,
                                   resolution=R, x_init=x)

            # 统计非空 cell
            occ_count = int(judge_occupancy(x, occ_threshold).sum().item())
            print(f"  Sample {sample_i}, Stage {stage_idx} (R={R}): "
                  f"denoised {N_cur} tokens, {occ_count} non-empty")

        # 最终阶段：只保留非空 cell 的 embeddings 用于后续 decode
        final_occ = judge_occupancy(x, occ_threshold)[0]  # [N]
        final_emb = x[0, final_occ]  # [M, C]
        final_coords = coords[0, final_occ]  # [M, 3]
        all_results.append((final_emb, final_coords))
        print(f"  Sample {sample_i}: final {final_emb.shape[0]} occupied cells")

    return all_results


@torch.no_grad()
def decode_embeddings_to_points(st_model, embeddings, coords, max_resolution, device, st_batch_size=512):
    """
    Decode Set Transformer embeddings back to 3D points.

    Args:
        st_model: AEBridgeReconModel
        embeddings: [N, 256] tensor - one sample's embeddings
        coords: [N, 3] tensor - integer grid coordinates
        max_resolution: int (e.g. 256)
        device: torch device
        st_batch_size: batch size for decode

    Returns:
        all_points: [M, 3] numpy array - reconstructed 3D points in [0,1)^3
    """
    N = embeddings.shape[0]
    bottleneck_dim = embeddings.shape[1]
    R = max_resolution

    all_points = []

    # Process in batches
    for start in range(0, N, st_batch_size):
        end = min(start + st_batch_size, N)
        batch_emb = embeddings[start:end]  # [bs, 256]
        batch_coords = coords[start:end]   # [bs, 3]
        bs = batch_emb.shape[0]

        # We need to decode from embedding → points
        # The AE decoder takes [B, bottleneck_dim] → [B, K_MAX, d] via dec_mlp + dec_sab
        # Then the AR decoder takes memory → predicted points

        # Step 1: AE decode (bottleneck → memory)
        dec_out = st_model.ae.dec_mlp(batch_emb)  # [bs, K_MAX*d] or [bs, d]
        if st_model.ae.pool == 'concat':
            H2 = dec_out.reshape(bs, st_model.ae.k_max, st_model.ae.dim_hidden)
        else:
            H2 = dec_out.unsqueeze(1).expand(bs, st_model.ae.k_max, st_model.ae.dim_hidden)
        # Apply dec_sab with full mask (all positions active — we don't know k yet)
        full_mask = torch.ones(bs, K_MAX, dtype=torch.bool, device=device)
        memory = st_model.ae.dec_sab(H2, full_mask)  # [bs, K_MAX, d]

        # Step 2: AR decode from memory
        mem_mask = ~full_mask  # all False = no padding
        pos0 = torch.zeros(1, dtype=torch.long, device=device)
        tgt = (st_model.bos.view(1, 1, -1).expand(bs, 1, -1).clone()
               + st_model.pos_embed(pos0).unsqueeze(0))

        batch_points = [[] for _ in range(bs)]
        batch_done = [False] * bs

        for t in range(K_MAX):
            T_cur = tgt.size(1)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                T_cur, device=device)
            out = st_model.transformer_decoder(
                tgt=tgt, memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_mask,
            )
            last = out[:, -1:, :]
            val = torch.sigmoid(st_model.value_head(last))   # [bs, 1, 3]
            end_prob = torch.sigmoid(st_model.end_head(last))  # [bs, 1, 1]

            for b in range(bs):
                if not batch_done[b]:
                    batch_points[b].append(val[b, 0].cpu().numpy())
                    if end_prob[b, 0, 0].item() > 0.5:
                        batch_done[b] = True

            if all(batch_done):
                break

            # Prepare next token
            pos_t = torch.tensor([t + 1], dtype=torch.long, device=device)
            next_emb = st_model.tgt_proj(val) + st_model.pos_embed(pos_t).unsqueeze(0)
            tgt = torch.cat([tgt, next_emb], dim=1)

        # Convert local points to global coordinates
        for b in range(bs):
            cell_coord = batch_coords[b].cpu().numpy().astype(np.float64)  # [3]
            pts_local = np.array(batch_points[b])  # [k, 3] in [0,1]^3

            if len(pts_local) == 0:
                continue

            # Local → global: pt_global = (cell_coord + pt_local) / R
            pts_global = (cell_coord + pts_local) / R
            all_points.append(pts_global)

    if len(all_points) > 0:
        return np.concatenate(all_points, axis=0)
    else:
        return np.zeros((0, 3))


def points_to_glb(points, output_path):
    """Save a point cloud as GLB (using trimesh PointCloud → export)."""
    # Create a simple point cloud mesh (vertices only, no faces)
    cloud = trimesh.PointCloud(points)
    cloud.export(output_path)
    return output_path


def main():
    args = create_argparser().parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]

    # ── Load DiT model ──
    print(f"Loading DiT3D-RDD from: {args.dit_ckpt}")
    dit_model, d_pred = load_dit_model(
        args.dit_ckpt, args.model, args.in_channels,
        args.num_classes, args.base_size, device
    )
    print(f"  Model: {args.model}, decouple={d_pred}")

    # ── Load Set Transformer ──
    print(f"Loading Set Transformer from: {args.st_ckpt}")
    st_model = load_st_model(args.st_ckpt, device)

    # ── Create diffusions ──
    sampling_steps = None
    if args.sampling_steps:
        sampling_steps = tuple(int(x) for x in args.sampling_steps.split(","))
    diffusions = create_diffusion_3d(
        schedule_name=args.schedule,
        steps_per_stage=steps_per_stage,
        diffusion_steps=args.diffusion_steps,
        sampling_steps_per_stage=sampling_steps,
    )

    # ── Sampling loop ──
    num_generated = 0
    batch_idx = 0

    while num_generated < args.num_samples:
        bs = min(args.batch_size, args.num_samples - num_generated)
        print(f"\n{'='*60}")
        print(f"Sampling batch {batch_idx} ({num_generated}/{args.num_samples})...")
        print(f"{'='*60}")

        # Stage 1: Diffusion sampling (returns list of (emb, coords) per sample)
        sample_results = diffusion_sample(dit_model, diffusions, bs, device, args)

        # Stage 2: Decode each sample
        for i in range(bs):
            sample_idx = num_generated + i
            print(f"\n  Decoding sample {sample_idx}...")

            # Get this sample's embeddings and coords (already filtered to non-empty)
            emb_i, coord_i = sample_results[i]  # [N, C], [N, 3]

            # Decode embeddings → 3D points
            points = decode_embeddings_to_points(
                st_model, emb_i, coord_i,
                args.max_resolution, device,
                st_batch_size=args.st_batch_size,
            )

            print(f"    Generated {len(points)} vertices from {emb_i.shape[0]} cells")

            if len(points) == 0:
                print(f"    WARNING: No points generated for sample {sample_idx}!")
                continue

            # Save as GLB
            out_path = os.path.join(args.output_dir, f"sample_{sample_idx:04d}.glb")
            points_to_glb(points, out_path)
            print(f"    Saved: {out_path}")

            # Also save raw data
            raw_path = os.path.join(args.output_dir, f"sample_{sample_idx:04d}_raw.pt")
            torch.save({
                'embeddings': emb_i.cpu(),
                'coords': coord_i.cpu(),
                'points': torch.from_numpy(points),
                'label': args.class_label,
            }, raw_path)

        num_generated += bs
        batch_idx += 1

    print(f"\n{'='*60}")
    print(f"Done! Generated {num_generated} samples in {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
