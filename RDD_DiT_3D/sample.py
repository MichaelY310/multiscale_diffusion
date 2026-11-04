"""
RDD-DiT-3D Sampling Script (全稀疏版本).

Progressive multi-stage sampling with noise-filling upsample and empty patch pruning.
Supports: unified model, multi-model (per-stage checkpoints), decouple mode, jump-step sampler.
"""

import os
import argparse
import torch
import numpy as np

from model import DiT3D_RDD, DiT3D_RDD_models
from diffusion_3d import (
    create_diffusion_3d,
    GaussianDiffusion3D,
    compute_active_mask_sparse,
    upsample_coords,
    get_named_beta_schedule,
)


def create_argparser():
    parser = argparse.ArgumentParser(description="Sample from RDD-DiT-3D (Sparse)")
    # --- Model ---
    parser.add_argument("--model", type=str, default="DiT3D-RDD-B",
                        choices=list(DiT3D_RDD_models.keys()))
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=55)
    parser.add_argument("--max-resolution", type=int, default=16,
                        help="Max resolution for pos embed normalization")
    # --- Checkpoint: unified mode ---
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Unified model checkpoint")
    # --- Checkpoint: multi-model mode ---
    parser.add_argument("--ckpt-per-stage", type=str, nargs="+", default=None,
                        help="Per-stage checkpoints, e.g. --ckpt-per-stage s0.pt s1.pt s2.pt")
    parser.add_argument("--model-per-stage", type=str, nargs="+", default=None,
                        help="Per-stage model names (defaults to --model for all)")
    # --- Output ---
    parser.add_argument("--output-dir", type=str, default="./samples")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-label", type=int, default=None,
                        help="If None, sample all classes uniformly")
    # --- Diffusion config ---
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--steps-per-stage", type=str, default="400,300,300")
    parser.add_argument("--stage-resolutions", type=str, default="4,8,16")
    parser.add_argument("--sampling-steps", type=str, default=None,
                        help="Per-stage sampling steps, e.g. '100,80,80'. If None, use full steps.")
    # --- Sampler ---
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "jump_step"],
                        help="Sampling algorithm: ddpm or jump_step")
    parser.add_argument("--jump-step-stochastic", action="store_true", default=False,
                        help="Add posterior noise in jump-step sampler")
    # --- Pruning ---
    parser.add_argument("--use-pruning", action="store_true", default=True,
                        help="Enable empty patch pruning during sampling")
    parser.add_argument("--threshold-1", type=float, default=0.85,
                        help="Empty detection threshold after stage 0")
    parser.add_argument("--threshold-2", type=float, default=0.90,
                        help="Empty detection threshold after stage 1")
    parser.add_argument("--dilation-kernel", type=int, default=3)
    parser.add_argument("--random-keep", type=float, default=0.05)
    parser.add_argument("--empty-embedding-path", type=str, default=None,
                        help="Path to empty embedding .pt file")
    # --- Reference coords for high-res stages (precomputed) ---
    parser.add_argument("--ref-coords-dir", type=str, default=None,
                        help="Dir with reference active coords per resolution (for pruning intersection)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_model(ckpt_path, model_name, in_channels, num_classes, max_resolution, device):
    """Load a model from checkpoint, auto-detecting decouple mode."""
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


@torch.no_grad()
def sample_batch(models, diffusions, batch_size, device, args, empty_embedding=None):
    """
    Progressive multi-stage sampling for one batch (全稀疏).

    At stage 0: start with all possible coords at the coarsest resolution (full grid).
    After each stage: prune empty tokens via cosine similarity, upsample coords to next resolution.

    Returns:
        x: [B, N_final, C] denoised token embeddings
        coords: [B, N_final, 3] final resolution coordinates
        y: [B] class labels
    """
    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)
    thresholds = [args.threshold_1, args.threshold_2]
    all_betas = get_named_beta_schedule(args.schedule, args.diffusion_steps)

    C = args.in_channels
    use_cfg = args.cfg_scale > 1.0

    # Normalize models to list
    if not isinstance(models, list):
        models = [models] * num_stages

    # Class labels
    if args.class_label is not None:
        y = torch.full((batch_size,), args.class_label, dtype=torch.long, device=device)
    else:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)

    # For CFG
    if use_cfg:
        y_null = torch.full_like(y, args.num_classes)
        y_cfg = torch.cat([y, y_null], dim=0)

    x = None
    coords = None

    for stage_idx in range(num_stages):
        diff = diffusions[stage_idx]
        R = stage_resolutions[stage_idx]
        model = models[stage_idx]

        # Prepare coords and initial noise for this stage
        if stage_idx == 0:
            # First stage: full grid (all R^3 positions are active)
            N = R ** 3
            # Generate all coords for R^3 grid
            grid = torch.arange(R, device=device)
            gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing='ij')
            all_coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # [N, 3]
            coords = all_coords.unsqueeze(0).expand(batch_size, N, 3).contiguous()  # [B, N, 3]
            x = torch.randn(batch_size, N, C, device=device)
        else:
            # Noise-filling upsample from previous stage
            t_start = sum(steps_per_stage[:stage_idx])
            t_end = t_start + steps_per_stage[stage_idx]
            stage_betas = all_betas[t_start:t_end]
            x = GaussianDiffusion3D.noise_filling_sparse(
                x, coords, coords,  # coords updated below after pruning
                steps_per_stage[stage_idx], stage_betas
            )

        # Build model_fn closure
        padding_mask = torch.ones(batch_size, coords.shape[1], dtype=torch.bool, device=device)

        def _make_model_fn(m, crds, pmask, use_cfg_, cfg_scale, y_c):
            def model_fn(x_t, t_global, y_in):
                if use_cfg_:
                    x_double = torch.cat([x_t, x_t], dim=0)
                    t_double = torch.cat([t_global, t_global], dim=0)
                    crds_double = torch.cat([crds, crds], dim=0)
                    pmask_double = torch.cat([pmask, pmask], dim=0)
                    out = m(x_double, crds_double, t_double, y_c, padding_mask=pmask_double)
                    # Only take eps head for CFG
                    if hasattr(m, 'd_prediction') and m.d_prediction:
                        out = out[..., :m.out_channels]
                    cond, uncond = out.chunk(2, dim=0)
                    return uncond + cfg_scale * (cond - uncond)
                else:
                    out = m(x_t, crds, t_global, y_in, padding_mask=pmask)
                    if hasattr(m, 'd_prediction') and m.d_prediction:
                        out = out[..., :m.out_channels]
                    return out
            return model_fn

        model_fn = _make_model_fn(
            model, coords, padding_mask,
            use_cfg, args.cfg_scale,
            y_cfg if use_cfg else y,
        )

        # Run reverse diffusion
        shape = (batch_size, coords.shape[1], C)
        if args.sampler == "jump_step":
            x = diff.jump_step_sample_loop(
                model_fn, shape, y, device,
                coords=coords, resolution=R,
                x_init=x,
                deterministic=not args.jump_step_stochastic,
            )
        else:
            x = diff.p_sample_loop(
                model_fn, shape, y, device,
                coords=coords, resolution=R, x_init=x,
            )

        # After this stage: prune empty tokens and upsample coords for next stage
        if stage_idx < num_stages - 1 and args.use_pruning and empty_embedding is not None:
            threshold = thresholds[stage_idx] if stage_idx < len(thresholds) else 0.90
            next_R = stage_resolutions[stage_idx + 1]

            # Compute active mask at current resolution
            active_mask = compute_active_mask_sparse(
                x, coords, empty_embedding, threshold, R,
                dilation_kernel=args.dilation_kernel,
                random_keep_ratio=args.random_keep,
            )  # [B, N] bool

            # Filter to active tokens only, then upsample coords
            # For simplicity: per-batch gather active tokens, then upsample
            new_x_list = []
            new_coords_list = []
            max_n_high = 0

            for b in range(batch_size):
                active_idx = active_mask[b].nonzero(as_tuple=True)[0]
                x_active = x[b, active_idx]          # [n_active, C]
                coords_active = coords[b, active_idx]  # [n_active, 3]

                # Upsample to next resolution
                coords_high_b = upsample_coords(
                    coords_active.unsqueeze(0), next_R
                ).squeeze(0)  # [n_high, 3]

                new_x_list.append(x_active)
                new_coords_list.append(coords_high_b)
                max_n_high = max(max_n_high, coords_high_b.shape[0])

            # Pad to max_n_high for batched processing
            coords_high = torch.zeros(batch_size, max_n_high, 3, dtype=torch.long, device=device)
            for b in range(batch_size):
                n = new_coords_list[b].shape[0]
                coords_high[b, :n] = new_coords_list[b]

            coords = coords_high

            # noise_filling will be called at the top of next stage iteration
            # But we need clean x at pruned coords. Use noise_filling_sparse:
            # Re-gather active x, pad to match coords_high
            x_clean_padded = torch.zeros(batch_size, max_n_high, C, device=device)
            # Not directly usable — noise_filling_sparse handles parent lookup.
            # Instead, store the active x and coords_low for noise_filling at next iteration:
            # We'll override the noise_filling at top of loop.
            # Actually noise_filling_sparse needs coords_low (the pruned active coords at current res)
            # and coords_high (the new high-res coords). Let's do it here directly.
            t_start_next = sum(steps_per_stage[:stage_idx + 1])
            t_end_next = t_start_next + steps_per_stage[stage_idx + 1]
            betas_next = all_betas[t_start_next:t_end_next]

            # Build padded x_clean from active tokens
            max_n_low = max(active_mask[b].sum().item() for b in range(batch_size))
            x_clean = torch.zeros(batch_size, max_n_low, C, device=device)
            coords_low = torch.zeros(batch_size, max_n_low, 3, dtype=torch.long, device=device)
            for b in range(batch_size):
                active_idx = active_mask[b].nonzero(as_tuple=True)[0]
                n = active_idx.shape[0]
                x_clean[b, :n] = x[b, active_idx]
                coords_low[b, :n] = coords[b // 1, :n]  # already filtered above
                coords_low[b, :n] = new_coords_list[b][:n] // 2 if False else coords[b, :n]
            # Simpler: gather from original
            x_clean2 = torch.zeros(batch_size, max_n_low, C, device=device)
            coords_low2 = torch.zeros(batch_size, max_n_low, 3, dtype=torch.long, device=device)
            for b in range(batch_size):
                active_idx = active_mask[b].nonzero(as_tuple=True)[0]
                n = active_idx.shape[0]
                x_clean2[b, :n] = x[b, active_idx]
                coords_low2[b, :n] = coords[b, active_idx] if stage_idx == 0 else coords[b, active_idx]

            # Wait - coords was already set to coords_high above. Need original coords.
            # Fix: save original coords before overwriting
            pass  # The logic below handles this correctly in a cleaner way.

        elif stage_idx < num_stages - 1:
            # No pruning: upsample all coords
            next_R = stage_resolutions[stage_idx + 1]
            coords = upsample_coords(coords, next_R)

    return x, coords, y


@torch.no_grad()
def sample_batch_v2(models, diffusions, batch_size, device, args, empty_embedding=None):
    """
    Cleaner progressive sampling implementation.
    Uses noise_filling_sparse correctly by keeping track of (x_clean, coords_low) before upsampling.
    """
    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)
    thresholds = [args.threshold_1, args.threshold_2]
    all_betas = get_named_beta_schedule(args.schedule, args.diffusion_steps)

    C = args.in_channels
    use_cfg = args.cfg_scale > 1.0

    if not isinstance(models, list):
        models = [models] * num_stages

    # Class labels
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
            # Full grid at coarsest resolution
            N = R ** 3
            grid = torch.arange(R, device=device)
            gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing='ij')
            all_coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
            coords = all_coords.unsqueeze(0).expand(batch_size, N, 3).contiguous()
            x = torch.randn(batch_size, N, C, device=device)
        # else: x and coords already set from previous stage's upsample

        # Build model_fn
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

        # Reverse diffusion
        shape = (batch_size, N_cur, C)
        if args.sampler == "jump_step":
            x = diff.jump_step_sample_loop(
                model_fn, shape, y, device,
                coords=coords, resolution=R,
                x_init=x, deterministic=not args.jump_step_stochastic,
            )
        else:
            x = diff.p_sample_loop(model_fn, shape, y, device, coords=coords, resolution=R, x_init=x)

        # Transition to next stage
        if stage_idx < num_stages - 1:
            next_R = stage_resolutions[stage_idx + 1]
            t_start_next = sum(steps_per_stage[:stage_idx + 1])
            t_end_next = t_start_next + steps_per_stage[stage_idx + 1]
            betas_next = all_betas[t_start_next:t_end_next]

            # Pruning
            if args.use_pruning and empty_embedding is not None:
                threshold = thresholds[stage_idx] if stage_idx < len(thresholds) else 0.90
                active_mask = compute_active_mask_sparse(
                    x, coords, empty_embedding, threshold, R,
                    dilation_kernel=args.dilation_kernel,
                    random_keep_ratio=args.random_keep,
                )

                # Gather active tokens per batch
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

            # Upsample coords to next resolution
            coords_high = upsample_coords(coords_active, next_R)

            # Noise filling: from clean x_active at coords_active → noisy at coords_high
            # Use the next stage diffusion's alphas_cumprod[-1] to ensure noise level
            # matches the sampling loop's starting timestep (important for SpacedDiffusion).
            next_diff = diffusions[stage_idx + 1]
            x = GaussianDiffusion3D.noise_filling_sparse(
                x_active, coords_active, coords_high,
                steps_per_stage[stage_idx + 1], betas_next,
                alpha_cumprod_T=float(next_diff.alphas_cumprod[-1])
            )
            coords = coords_high

    return x, coords, y


def main():
    args = create_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    num_stages = len(steps_per_stage)

    # --- Load model(s) ---
    if args.ckpt_per_stage:
        assert len(args.ckpt_per_stage) == num_stages
        model_names = args.model_per_stage or [args.model] * num_stages
        models = []
        for i, ckpt_path in enumerate(args.ckpt_per_stage):
            m, d_pred = _load_model(ckpt_path, model_names[i], args.in_channels,
                                    args.num_classes, args.max_resolution, device)
            models.append(m)
            print(f"Stage {i}: loaded {model_names[i]} from {ckpt_path} (decouple={d_pred})")
    elif args.ckpt:
        model, d_pred = _load_model(args.ckpt, args.model, args.in_channels,
                                    args.num_classes, args.max_resolution, device)
        models = model
        print(f"Unified model: {args.model} (decouple={d_pred})")
    else:
        raise ValueError("Must provide --ckpt (unified) or --ckpt-per-stage (multi-model)")

    # --- Diffusion ---
    sampling_steps = None
    if args.sampling_steps:
        sampling_steps = tuple(int(x) for x in args.sampling_steps.split(","))
    diffusions = create_diffusion_3d(
        schedule_name=args.schedule,
        steps_per_stage=steps_per_stage,
        diffusion_steps=args.diffusion_steps,
        sampling_steps_per_stage=sampling_steps,
    )

    # --- Empty embedding ---
    empty_embedding = None
    if args.empty_embedding_path and os.path.exists(args.empty_embedding_path):
        empty_embedding = torch.load(args.empty_embedding_path, map_location=device)
        print(f"Loaded empty embedding: {empty_embedding.shape}")

    # --- Sampling loop ---
    num_generated = 0
    batch_idx = 0
    while num_generated < args.num_samples:
        bs = min(args.batch_size, args.num_samples - num_generated)
        print(f"Sampling batch {batch_idx} ({num_generated}/{args.num_samples})...")

        x, coords, labels = sample_batch_v2(
            models, diffusions, bs, device, args, empty_embedding
        )

        for i in range(bs):
            save_path = os.path.join(args.output_dir, f"sample_{num_generated + i:06d}.pt")
            torch.save({
                'embeddings': x[i].cpu(),
                'coords': coords[i].cpu(),
                'label': labels[i].item(),
            }, save_path)

        num_generated += bs
        batch_idx += 1

    print(f"Done. Generated {num_generated} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
