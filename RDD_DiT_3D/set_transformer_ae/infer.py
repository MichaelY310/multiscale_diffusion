"""
Inference script for the AE-Bridge + Concat Pool 3D model (train_emd_ae_3d.py).

Demonstrates:
  1. Encoding a random 3D point set → 256-dim embedding
  2. Decoding the embedding back → reconstructed 3D point set
  3. Comparing original vs reconstructed points + EMD

Usage:
  python infer_ae_3d.py --ckpt /path/to/step_8400000.pt --k 5 --seed 42
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import wasserstein_distance

try:
    from .ae_model import AEBridgeReconModel, K_MAX, DIM_IN
except ImportError:
    from ae_model import AEBridgeReconModel, K_MAX, DIM_IN


def load_model(ckpt_path: str, device: torch.device):
    """Load AEBridgeReconModel from checkpoint, inferring architecture from saved args."""
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    model = AEBridgeReconModel(
        dim_hidden=a['dim_hidden'],
        num_heads=a['num_heads'],
        num_encoder_layers=a['num_encoder_layers'],
        num_decoder_layers=a['num_decoder_layers'],
        bottleneck_dim=a['bottleneck_dim'],
        ae_num_heads=a['ae_num_heads'],
        ae_num_mlp_layers=a['ae_num_mlp_layers'],
        ae_mlp_factor=a['ae_mlp_factor'],
        ae_pool=a['ae_pool'],
        ln=a.get('ln', False),
    )
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"Loaded model from {ckpt_path} (step {ckpt['step']})")
    print(f"  dim_hidden={a['dim_hidden']}, bottleneck_dim={a['bottleneck_dim']}, "
          f"ae_pool={a['ae_pool']}")
    return model


def prepare_input(pts: torch.Tensor, device: torch.device):
    """
    pts: [k, 3] tensor of 3D points in [0,1]^3
    Returns X [1, K_MAX, 4], mask [1, K_MAX]
    """
    k = pts.shape[0]
    # Canonical lexicographic sort (same as training)
    order = sorted(range(k), key=lambda r: tuple(pts[r].tolist()))
    pts = pts[order]

    X = torch.zeros(1, K_MAX, DIM_IN, device=device)
    mask = torch.zeros(1, K_MAX, dtype=torch.bool, device=device)
    X[0, :k, :3] = pts
    X[0, :k, 3] = k / K_MAX
    mask[0, :k] = True
    return X, mask, pts


def encode(model: AEBridgeReconModel, X: torch.Tensor, mask: torch.Tensor):
    """Extract the 256-dim embedding from the AE bottleneck."""
    with torch.no_grad():
        Z = model.encoder(X, mask)          # [1, K_MAX, d]
        _, embedding = model.ae(Z, mask)    # [1, bottleneck_dim]
    return embedding


def decode(model: AEBridgeReconModel, X: torch.Tensor, mask: torch.Tensor):
    """Full forward inference: encode → decode → generate points."""
    with torch.no_grad():
        pred_values, pred_ends = model.forward_inference(X, mask)
    # pred_values: [1, K_MAX, 3], pred_ends: [1, K_MAX]
    return pred_values[0], pred_ends[0]


def extract_points(pred_values, pred_ends, threshold=0.5):
    """Truncate generated sequence at first end > threshold."""
    pts = []
    for t in range(pred_values.shape[0]):
        pts.append(pred_values[t].cpu().numpy())
        if pred_ends[t].item() > threshold:
            break
    return np.array(pts)


def compute_emd(true_pts, gen_pts):
    """Mean per-coordinate Wasserstein distance as 3D EMD proxy."""
    emds = []
    for j in range(3):
        emds.append(wasserstein_distance(true_pts[:, j], gen_pts[:, j]))
    return np.mean(emds)


def run(args):
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    # Generate random 3D point set
    torch.manual_seed(args.seed)
    k = args.k if args.k > 0 else np.random.randint(1, K_MAX + 1)
    pts_raw = torch.rand(k, 3)

    # Prepare input (sorts lexicographically)
    X, mask, pts_sorted = prepare_input(pts_raw, device)

    # Encode → embedding
    embedding = encode(model, X, mask)
    print(f"\n{'='*60}")
    print(f"Input: {k} random 3D points (seed={args.seed})")
    print(f"{'='*60}")
    for i in range(k):
        p = pts_sorted[i]
        print(f"  pt[{i}]  x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")

    print(f"\n[Embedding] shape={list(embedding.shape)}, "
          f"norm={embedding.norm().item():.4f}")
    print(f"  first 10 dims: {embedding[0, :10].cpu().numpy().round(4)}")

    # Decode → reconstruct
    pred_values, pred_ends = decode(model, X, mask)
    gen_pts = extract_points(pred_values, pred_ends)

    print(f"\n{'='*60}")
    print(f"Reconstruction: generated {len(gen_pts)} points")
    print(f"{'='*60}")

    true_np = pts_sorted.cpu().numpy()
    max_len = max(k, len(gen_pts))
    print(f"  {'':4s} {'orig_x':>8s} {'orig_y':>8s} {'orig_z':>8s}  |  "
          f"{'recon_x':>8s} {'recon_y':>8s} {'recon_z':>8s}")
    print(f"  {'':4s} {'-'*26}  |  {'-'*26}")
    for i in range(max_len):
        if i < k:
            ox = f"{true_np[i, 0]:.4f}"
            oy = f"{true_np[i, 1]:.4f}"
            oz = f"{true_np[i, 2]:.4f}"
        else:
            ox = oy = oz = "   -  "
        if i < len(gen_pts):
            rx = f"{gen_pts[i, 0]:.4f}"
            ry = f"{gen_pts[i, 1]:.4f}"
            rz = f"{gen_pts[i, 2]:.4f}"
        else:
            rx = ry = rz = "   -  "
        print(f"  [{i}]  {ox:>8s} {oy:>8s} {oz:>8s}  |  {rx:>8s} {ry:>8s} {rz:>8s}")

    # Compute EMD
    if len(gen_pts) > 0:
        emd = compute_emd(true_np, gen_pts)
        print(f"\n  EMD (mean per-coord Wasserstein): {emd:.6f}")
        print(f"  Generated count: {len(gen_pts)}, True count: {k}")

    # Batch evaluation
    if args.eval_batches > 0:
        print(f"\n{'='*60}")
        print(f"Batch evaluation ({args.eval_batches} batches × {args.batch_size} samples)")
        print(f"{'='*60}")
        from train_emd_ae_3d import evaluate
        avg_emd = evaluate(model, args.eval_batches, args.batch_size, device)
        print(f"  Average val_emd: {avg_emd:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AE-Bridge 3D inference & embedding demo')
    parser.add_argument('--ckpt', type=str,
                        default='/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/'
                                'RDD/set_transformer/checkpoints_emd_ae_concat_3d/step_8400000.pt')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of input points (0 = random 1-10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--eval_batches', type=int, default=10,
                        help='Number of batches for avg EMD evaluation (0 to skip)')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    run(args)
