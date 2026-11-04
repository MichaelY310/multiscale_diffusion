"""
GaussianDiffusion3D: 3D volume diffusion with RDD drift and empty patch pruning utilities.

Fully sparse representation: operates on [B, N, C] token tensors + [B, N, 3] coordinate tensors,
where N is the max number of active tokens per sample.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ==============================================================================
#                              Helpers
# ==============================================================================

def mean_flat(tensor):
    """Mean over all non-batch dimensions. For [B, N, C] → [B]."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    Broadcasts to 3D shape [B, N, C] or similar.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device, dtype=torch.float32)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ==============================================================================
#                      Legacy dense helpers (for comparison)
# ==============================================================================

def downsample_3d(x, factor):
    """[Legacy] Average-pool 3D volume by factor. x: [B, C, R, R, R]"""
    return F.avg_pool3d(x, kernel_size=factor, stride=factor)


def upsample_3d(x, factor):
    """[Legacy] Trilinear upsample 3D volume by factor. x: [B, C, R, R, R]"""
    return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=False)


# ==============================================================================
#                      Sparse coordinate utilities
# ==============================================================================

def _coords_to_flat_index(coords, R):
    """
    Convert [B, N, 3] integer coordinates to flat indices in [0, R^3).
    coords[..., 0] = x, coords[..., 1] = y, coords[..., 2] = z
    Returns: [B, N] long tensor
    """
    return coords[..., 0] * (R * R) + coords[..., 1] * R + coords[..., 2]


def _flat_index_to_coords(flat_idx, R):
    """
    Convert [B, N] flat indices in [0, R^3) to [B, N, 3] integer coordinates.
    """
    x = flat_idx // (R * R)
    rem = flat_idx % (R * R)
    y = rem // R
    z = rem % R
    return torch.stack([x, y, z], dim=-1)


def compute_drift_sparse(x_0, coords, resolution, padding_mask=None):
    """
    Compute drift for sparse tokens.
    
    Downsample: coords // 2, mean pool tokens sharing the same low-res coordinate.
    Upsample: copy low-res embedding back to each high-res token.
    drift = upsampled - x_0

    Args:
        x_0: [B, N, C] token embeddings
        coords: [B, N, 3] integer coordinates at current resolution
        resolution: int, current spatial resolution R
        padding_mask: [B, N] bool, True=valid, False=padding. If None, all valid.
    Returns:
        drift: [B, N, C]
    """
    B, N, C = x_0.shape
    R_low = resolution // 2

    # Low-res coordinates via floor division
    coords_low = coords // 2  # [B, N, 3]

    # Zero out padding tokens before pooling
    if padding_mask is not None:
        mask_float = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
        x_0_masked = x_0 * mask_float
    else:
        x_0_masked = x_0

    if R_low <= 64:
        # Dense scatter for small resolutions
        flat_low = _coords_to_flat_index(coords_low, R_low)  # [B, N]
        num_low_voxels = R_low ** 3
        flat_low_exp = flat_low.unsqueeze(-1).expand(B, N, C)  # [B, N, C]

        sum_buf = torch.zeros(B, num_low_voxels, C, device=x_0.device, dtype=x_0.dtype)
        sum_buf.scatter_add_(1, flat_low_exp, x_0_masked)

        count_buf = torch.zeros(B, num_low_voxels, 1, device=x_0.device, dtype=x_0.dtype)
        if padding_mask is not None:
            ones = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
        else:
            ones = torch.ones(B, N, 1, device=x_0.device, dtype=x_0.dtype)
        count_buf.scatter_add_(1, flat_low.unsqueeze(-1), ones)
        count_buf = count_buf.clamp(min=1.0)

        mean_low = sum_buf / count_buf  # [B, num_low_voxels, C]
        x_up = torch.gather(mean_low, 1, flat_low_exp)  # [B, N, C]
    else:
        # Sparse mean pooling for large resolutions
        # Group tokens by their low-res coordinate using unique
        x_up = torch.zeros_like(x_0)
        for b in range(B):
            cl = coords_low[b]  # [N, 3]
            # Encode as flat index for grouping (use large stride to avoid collisions)
            flat_b = cl[:, 0] * (R_low * R_low) + cl[:, 1] * R_low + cl[:, 2]
            unique_flat, inverse = flat_b.unique(return_inverse=True)
            # Compute mean per group
            num_groups = unique_flat.shape[0]
            group_sum = torch.zeros(num_groups, C, device=x_0.device, dtype=x_0.dtype)
            group_count = torch.zeros(num_groups, 1, device=x_0.device, dtype=x_0.dtype)
            group_sum.scatter_add_(0, inverse.unsqueeze(-1).expand(N, C), x_0_masked[b])
            if padding_mask is not None:
                group_count.scatter_add_(0, inverse.unsqueeze(-1), padding_mask[b].unsqueeze(-1).float())
            else:
                group_count.scatter_add_(0, inverse.unsqueeze(-1), torch.ones(N, 1, device=x_0.device))
            group_mean = group_sum / group_count.clamp(min=1.0)
            # Gather back
            x_up[b] = group_mean[inverse]

    drift = x_up - x_0
    # Zero out drift at padding positions
    if padding_mask is not None:
        drift = drift * padding_mask.unsqueeze(-1).float()
    return drift


def upsample_coords(coords_low, target_resolution, active_coords_high=None):
    """
    Expand low-res active coordinates to high-res sub-coordinates.
    Each (x, y, z) → 8 children: (2x+dx, 2y+dy, 2z+dz) for dx,dy,dz in {0,1}.

    Args:
        coords_low: [B, N_low, 3] low-res integer coordinates
        target_resolution: int, the target high resolution R
        active_coords_high: [B, N_full, 3] or None. If provided, intersect with this set.
    Returns:
        coords_high: [B, N_high, 3] high-res coordinates (padded with 0 if needed)
    """
    B, N_low, _ = coords_low.shape
    device = coords_low.device

    # Generate 8 sub-offsets
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in range(2) for dy in range(2) for dz in range(2)],
        device=device, dtype=coords_low.dtype
    )  # [8, 3]

    # Expand: [B, N_low, 1, 3] * 2 + [1, 1, 8, 3] offsets
    base = coords_low * 2  # [B, N_low, 3]
    base = base.unsqueeze(2).expand(B, N_low, 8, 3)  # [B, N_low, 8, 3]
    offsets_exp = offsets.reshape(1, 1, 8, 3).expand(B, N_low, 8, 3)
    children = base + offsets_exp  # [B, N_low, 8, 3]
    children = children.reshape(B, N_low * 8, 3)  # [B, N_low*8, 3]

    # Clamp to valid range
    children = children.clamp(min=0, max=target_resolution - 1)

    if active_coords_high is not None:
        # Intersect with active high-res set using flat indices
        R = target_resolution
        children_flat = _coords_to_flat_index(children, R)  # [B, N_low*8]
        active_flat = _coords_to_flat_index(active_coords_high, R)  # [B, N_full]

        results = []
        max_n = 0
        for b in range(B):
            active_set = set(active_flat[b].cpu().tolist())
            child_flat_b = children_flat[b].cpu().tolist()
            # Deduplicate and intersect
            kept = sorted(set(child_flat_b) & active_set)
            max_n = max(max_n, len(kept))
            results.append(kept)

        # Pad to max_n
        coords_high = torch.zeros(B, max_n, 3, device=device, dtype=coords_low.dtype)
        for b in range(B):
            n = len(results[b])
            if n > 0:
                flat_t = torch.tensor(results[b], device=device, dtype=torch.long)
                coords_high[b, :n] = _flat_index_to_coords(flat_t, R)

        return coords_high
    else:
        # Deduplicate per batch
        R = target_resolution
        children_flat = _coords_to_flat_index(children, R)  # [B, N_low*8]
        results = []
        max_n = 0
        for b in range(B):
            unique_flat = children_flat[b].unique()
            max_n = max(max_n, unique_flat.shape[0])
            results.append(unique_flat)

        coords_high = torch.zeros(B, max_n, 3, device=device, dtype=coords_low.dtype)
        for b in range(B):
            n = results[b].shape[0]
            coords_high[b, :n] = _flat_index_to_coords(results[b], R)

        return coords_high


def compute_active_mask_sparse(x0_pred, coords, empty_embedding, threshold, resolution,
                               dilation_kernel=3, random_keep_ratio=0.05):
    """
    Compute active mask for sparse tokens with sparse dilation (memory-efficient).

    Args:
        x0_pred: [B, N, C] predicted clean tokens
        coords: [B, N, 3] integer coordinates
        empty_embedding: [C] embedding of the empty token
        threshold: float, cosine similarity threshold above which a token is empty
        resolution: int, spatial resolution R
        dilation_kernel: int, kernel size for 3D dilation
        random_keep_ratio: float, fraction of tokens randomly kept as safety
    Returns:
        active_mask: [B, N] bool tensor (True = active / not empty)
    """
    B, N, C = x0_pred.shape
    R = resolution

    # Cosine similarity with empty embedding
    e = empty_embedding.reshape(1, 1, C)  # [1, 1, C]
    sim = F.cosine_similarity(x0_pred, e.expand(B, N, C), dim=-1)  # [B, N]
    active = sim < threshold  # True = not empty

    # Sparse dilation: for each active token, mark all tokens within
    # dilation_kernel//2 Manhattan distance as active
    # This avoids allocating a dense R^3 volume
    if R <= 64:
        # Small resolution: use dense method (fast)
        flat_idx = _coords_to_flat_index(coords, R)  # [B, N]
        vol = torch.zeros(B, R * R * R, device=x0_pred.device, dtype=torch.float32)
        vol.scatter_(1, flat_idx, active.float())
        vol = vol.reshape(B, 1, R, R, R)
        pad = dilation_kernel // 2
        vol = F.max_pool3d(vol, kernel_size=dilation_kernel, stride=1, padding=pad)
        vol_flat = vol.reshape(B, R * R * R)
        active_dilated = torch.gather(vol_flat, 1, flat_idx) > 0.5
    else:
        # Large resolution: use sparse neighbor lookup
        pad = dilation_kernel // 2
        active_dilated = active.clone()
        for b in range(B):
            active_coords_b = coords[b][active[b]]  # [M, 3] active token coords
            if active_coords_b.shape[0] == 0:
                continue
            # For each token in batch b, check if any active token is within pad distance
            # Vectorized: compute L-inf distance between each token and all active tokens
            all_coords_b = coords[b]  # [N, 3]
            # Use chunked processing to avoid N*M memory
            chunk_size = 4096
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = all_coords_b[start:end].unsqueeze(1)  # [chunk, 1, 3]
                diff = (chunk - active_coords_b.unsqueeze(0)).abs()  # [chunk, M, 3]
                linf = diff.max(dim=-1)[0]  # [chunk, M]
                within = (linf <= pad).any(dim=1)  # [chunk]
                active_dilated[b, start:end] |= within

    active_mask = active_dilated

    # Random safety retention
    if random_keep_ratio > 0:
        rand_keep = torch.rand(B, N, device=x0_pred.device) < random_keep_ratio
        active_mask = active_mask | rand_keep

    return active_mask


# ==============================================================================
#                         GaussianDiffusion3D (Sparse)
# ==============================================================================

class GaussianDiffusion3D:
    """
    3D Gaussian Diffusion with optional drift for RDD multi-resolution stages.
    Sparse representation: operates on [B, N, C] tokens + [B, N, 3] coordinates.

    For the coarsest stage (c=0): standard epsilon-prediction, no drift.
    For finer stages (c>0): drift from low-pass residual.
    """

    def __init__(self, betas, stage_index=0, T_start=0, T_end=None, resolution=None):
        """
        betas: 1-D numpy array for this stage's timesteps.
        stage_index: 0=coarsest, 1=mid, 2=fine
        T_start: global timestep offset (for kappa computation)
        T_end: global timestep end (T_start + len(betas))
        resolution: int, the spatial resolution for this stage (used in drift computation)
        """
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])
        self.stage_index = stage_index
        self.T_start = T_start
        self.T_end = T_end if T_end is not None else T_start + self.num_timesteps
        self.resolution = resolution

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _kappa(self, t_local):
        """Linear kappa for drift: kappa(t) = t_local / num_timesteps."""
        return t_local.float() / self.num_timesteps if self.num_timesteps > 0 else t_local.float() * 0

    def compute_drift(self, x_0, coords, resolution, padding_mask=None):
        """
        Compute drift = upsample(downsample(x_0)) - x_0 in sparse representation.
        Only for non-coarsest stages.

        Args:
            x_0: [B, N, C]
            coords: [B, N, 3]
            resolution: int
            padding_mask: [B, N] bool, True=valid, False=padding. If None, all valid.
        Returns:
            drift: [B, N, C]
        """
        if self.stage_index == 0:
            return torch.zeros_like(x_0)
        return compute_drift_sparse(x_0, coords, resolution, padding_mask=padding_mask)

    def q_sample(self, x_0, t, noise=None, drift=None):
        """
        Forward diffusion: q(x_t | x_0) with optional drift.
        x_0: [B, N, C]
        t: [B] local timesteps
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        if drift is not None and self.stage_index > 0:
            kappa = self._kappa(t).reshape(-1, 1, 1)
            x_t = sqrt_alpha * (x_0 + kappa * drift) + sqrt_one_minus_alpha * noise
        else:
            x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t

    def training_losses(self, model, x_0, t, y, coords, noise=None, padding_mask=None):
        """
        Compute training loss for one stage.

        Args:
            model: predicts noise given (x_t, t_global, y)
            x_0: [B, N, C] ground truth tokens at this resolution
            t: [B] local timesteps within this stage
            y: [B] class labels
            coords: [B, N, 3] integer coordinates
        Returns:
            dict with 'loss' key
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        # Compute noise target with drift correction
        if self.stage_index > 0:
            kappa = self._kappa(t).reshape(-1, 1, 1)
            sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            noise_target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * kappa * drift
        else:
            noise_target = noise

        t_global = t + self.T_start
        model_output = model(x_t, t_global, y)

        # MSE loss only on valid (non-padding) tokens
        mse = (model_output - noise_target) ** 2  # [B, N, C]
        if padding_mask is not None:
            # padding_mask: [B, N] bool, True=valid
            mask = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
            mse = mse * mask
            # Mean over valid tokens: sum(N,C) / (N_valid * C)
            C = mse.shape[-1]
            n_valid = padding_mask.sum(dim=1).clamp(min=1).float()  # [B]
            loss = mse.sum(dim=(1, 2)) / (n_valid * C)
        else:
            loss = mean_flat(mse)
        return {"loss": loss}

    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None, padding_mask=None):
        """
        Decouple training: model outputs [eps_pred | d_pred] concatenated along last dim.
        eps head predicts combined noise target, d head predicts raw drift.
        Only for non-coarsest stages (stage_index > 0).

        Args:
            model: returns [B, N, 2*C] — first C channels are eps, last C are drift pred
            x_0: [B, N, C]
            t: [B] local timesteps
            y: [B] class labels
            coords: [B, N, 3]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = t + self.T_start
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]

        C = x_0.shape[-1]
        eps_pred = model_output[..., :C]   # [B, N, C]
        d_pred = model_output[..., C:]     # [B, N, C]

        # Combined noise target for eps head
        if self.stage_index > 0:
            kappa = self._kappa(t).reshape(-1, 1, 1)
            sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            noise_target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * kappa * drift
        else:
            noise_target = noise

        # MSE loss only on valid (non-padding) tokens
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
            mse_eps_raw = (eps_pred - noise_target) ** 2 * mask
            mse_d_raw = (d_pred - drift) ** 2 * mask
            C = eps_pred.shape[-1]
            n_valid = padding_mask.sum(dim=1).clamp(min=1).float()  # [B]
            mse_eps = mse_eps_raw.sum(dim=(1, 2)) / (n_valid * C)
            mse_d = mse_d_raw.sum(dim=(1, 2)) / (n_valid * C)
        else:
            mse_eps = mean_flat((eps_pred - noise_target) ** 2)
            mse_d = mean_flat((d_pred - drift) ** 2)
        loss = mse_eps + mse_d

        return {"loss": loss, "mse_eps": mse_eps, "mse_d": mse_d}

    def _compute_posterior_mean_coef_drift(self):
        """
        Precompute drift coefficient for the single-step DDPM posterior.
        Matches 2D implementation:
            coef_drift[t] = sqrt(ab_prev) / (1 - ab_t) *
                            ((t-T_c) - alpha*(t-T_c) - alpha + ab) / DT
        where (t-T_c) = kappa * DT, DT = T_end - T_start.
        Simplifies to:
            = sqrt(ab_prev) / (1 - ab) * (kappa*(1-alpha) + (ab - alpha)/DT)
        """
        if self.stage_index == 0:
            return np.zeros(self.num_timesteps, dtype=np.float64)

        DT = float(self.T_end - self.T_start)
        if DT == 0:
            return np.zeros(self.num_timesteps, dtype=np.float64)

        alphas = 1.0 - self.betas
        t_local = np.arange(self.num_timesteps)
        kappa = t_local / self.num_timesteps

        coef = (
            np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-20) *
            (kappa * (1.0 - alphas) + (self.alphas_cumprod - alphas) / DT)
        )
        return coef

    def p_mean_variance(self, model, x_t, t, y, clip_denoised=True,
                        coords=None, resolution=None):
        """
        Compute predicted mean and variance for p(x_{t-1} | x_t).
        x_t: [B, N, C]
        t: [B] local timesteps
        coords, resolution: needed for drift correction in non-coarsest stages.
        """
        t_global = t + self.T_start
        model_output = model(x_t, t_global, y)

        # Predict x_0 from noise prediction
        sqrt_recip = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * model_output

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-5, 5)

        # Posterior mean (standard DDPM)
        coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * pred_x0 + coef2 * x_t

        # Drift correction for non-coarsest stages
        if self.stage_index > 0 and coords is not None and resolution is not None:
            d = self.compute_drift(pred_x0, coords, resolution)
            coef_drift = _extract_into_tensor(
                self._compute_posterior_mean_coef_drift(), t, x_t.shape)
            posterior_mean = posterior_mean + coef_drift * d

        posterior_log_var = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return {
            "mean": posterior_mean,
            "log_variance": posterior_log_var,
            "pred_x0": pred_x0,
        }

    @torch.no_grad()
    def p_sample(self, model, x_t, t, y, coords=None, resolution=None):
        """Single denoising step. x_t: [B, N, C]."""
        out = self.p_mean_variance(model, x_t, t, y, coords=coords, resolution=resolution)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}

    @torch.no_grad()
    def p_sample_loop(self, model, shape, y, device, coords=None, resolution=None,
                      x_init=None, progress=False):
        """
        Full reverse diffusion loop for this stage.
        shape: (B, N, C)
        coords: [B, N, 3] passed to model via closure if needed
        resolution: int, spatial resolution (needed for drift correction)
        x_init: initial noisy tokens (for non-first stages, from noise_filling_sparse)
        """
        B = shape[0]
        x = x_init if x_init is not None else torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc=f"Stage {self.stage_index}")

        for i in indices:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            out = self.p_sample(model, x, t, y, coords=coords, resolution=resolution)
            x = out["sample"]

        return x

    @torch.no_grad()
    def jump_step_sample(self, model, x, t, t_next, y, coords=None, resolution=None,
                         deterministic=True):
        """
        Multi-step Bayesian posterior sampler (jump-step).
        Computes exact posterior q(x_s | x_t, x_0) where s = t_next.

        x: [B, N, C]
        t, t_next: [B] local timesteps (t > t_next in reverse direction)
        coords: [B, N, 3] needed for drift computation at non-coarsest stages
        resolution: int, spatial resolution (needed for drift)
        deterministic: if True, return posterior mean (DDIM-like); else add posterior noise.
        """
        # Predict x_0
        out = self.p_mean_variance(model, x, t, y)
        pred_xstart = out["pred_x0"]

        # Get alpha_bar values at t and t_next
        ab_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        t_next_clamped = t_next.clamp(min=0)
        ab_s = _extract_into_tensor(self.alphas_cumprod, t_next_clamped, x.shape)
        final_mask = (t_next < 0).float().reshape(-1, 1, 1)
        ab_s = ab_s * (1 - final_mask) + final_mask

        # Posterior coefficients (exact multi-step)
        Sigma = (ab_s - ab_t) * (1.0 - ab_s) / (ab_s * (1.0 - ab_t) + 1e-20)
        coeff_xt = (1.0 - ab_s) / (1.0 - ab_t + 1e-20) * torch.sqrt(ab_t / (ab_s + 1e-20))
        coeff_x0 = (ab_s - ab_t) / (torch.sqrt(ab_s + 1e-20) * (1.0 - ab_t + 1e-20))

        mu = coeff_xt * x + coeff_x0 * pred_xstart

        # Drift correction for non-coarsest stages
        # Uses exact posterior formula: coeff_d = Sigma * (D/sigma2^2 - A*B/sigma1^2)
        # where A=sqrt(ab_t/ab_s), B=sqrt(ab_t)*(kappa_t-kappa_s),
        #       D=sqrt(ab_s)*kappa_s, sigma1^2=1-ab_t/ab_s, sigma2^2=1-ab_s
        if self.stage_index > 0 and coords is not None and resolution is not None:
            d = self.compute_drift(pred_xstart, coords, resolution)
            DT = float(self.T_end - self.T_start)

            # kappa values at t and t_next
            t_global = t + self.T_start
            t_next_global = t_next_clamped + self.T_start
            kappa_t = ((t_global.float() - self.T_start) / DT).reshape(-1, 1, 1)
            kappa_s = ((t_next_global.float() - self.T_start) / DT).reshape(-1, 1, 1)

            # Exact posterior drift coefficient (matching 2D implementation)
            sigma1_sq = 1.0 - ab_t / (ab_s + 1e-20)
            sigma2_sq = 1.0 - ab_s
            A_coef = torch.sqrt(ab_t / (ab_s + 1e-20))
            B_coef = torch.sqrt(ab_t) * (kappa_t - kappa_s)
            D_coef = torch.sqrt(ab_s) * kappa_s

            coeff_d = Sigma * (D_coef / (sigma2_sq + 1e-20) - A_coef * B_coef / (sigma1_sq + 1e-20))
            mu = mu + coeff_d * d

        if deterministic:
            return {"sample": mu, "pred_x0": pred_xstart}
        else:
            noise = torch.randn_like(x)
            Sigma_clamped = Sigma.clamp(min=0)
            sample = mu + torch.sqrt(Sigma_clamped) * noise
            return {"sample": sample, "pred_x0": pred_xstart}

    @torch.no_grad()
    def jump_step_sample_loop(self, model, shape, y, device, coords=None, resolution=None,
                              progress=False, x_init=None, deterministic=True):
        """
        Full jump-step reverse loop for this stage.
        shape: (B, N, C)
        """
        B = shape[0]
        x = x_init if x_init is not None else torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc=f"Stage {self.stage_index} (jump-step)")

        for idx_i, i in enumerate(indices):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            if idx_i < len(indices) - 1:
                t_next = torch.full((B,), indices[idx_i + 1], device=device, dtype=torch.long)
            else:
                t_next = torch.full((B,), -1, device=device, dtype=torch.long)
            out = self.jump_step_sample(model, x, t, t_next, y,
                                        coords=coords, resolution=resolution,
                                        deterministic=deterministic)
            x = out["sample"]

        return x

    @staticmethod
    def noise_filling_sparse(x_clean, coords_low, coords_high, num_timesteps_next, betas_next,
                             alpha_cumprod_T=None):
        """
        Noise-filling upsample in sparse format.
        For each high-res coord, find its low-res parent (coords_high // 2),
        gather the corresponding embedding from x_clean, then q_sample at t=T-1.

        Args:
            x_clean: [B, N_low, C] denoised tokens at low resolution
            coords_low: [B, N_low, 3] low-res integer coordinates
            coords_high: [B, N_high, 3] high-res integer coordinates
            num_timesteps_next: int, number of timesteps in the next stage
            betas_next: 1-D numpy array, beta schedule for the next stage
            alpha_cumprod_T: float or None. If provided, use this value instead of
                computing from betas_next. This is needed when using SpacedDiffusion
                to ensure noise level matches the sampling loop's starting timestep.
        Returns:
            x_noisy: [B, N_high, C] noisy tokens at high resolution
        """
        B, N_low, C = x_clean.shape
        N_high = coords_high.shape[1]
        device = x_clean.device

        # Determine low-res resolution from coords_low
        R_low = int(coords_low.max().item()) + 1

        # Flat indices of low-res coords
        flat_low = _coords_to_flat_index(coords_low, R_low)  # [B, N_low]

        # Parent coords of high-res coords
        parent_coords = coords_high // 2  # [B, N_high, 3]
        flat_parent = _coords_to_flat_index(parent_coords, R_low)  # [B, N_high]

        # Build a dense lookup: for each low-res flat index, store the position in x_clean
        # We scatter the position indices into a buffer
        pos_buf = torch.zeros(B, R_low ** 3, device=device, dtype=torch.long)
        # Fill with indices [0, N_low) per batch
        indices_low = torch.arange(N_low, device=device).unsqueeze(0).expand(B, N_low)
        pos_buf.scatter_(1, flat_low, indices_low)

        # Gather: for each high-res token, find its parent's position in x_clean
        parent_pos = torch.gather(pos_buf, 1, flat_parent)  # [B, N_high]

        # Gather embeddings
        parent_pos_exp = parent_pos.unsqueeze(-1).expand(B, N_high, C)
        x_up = torch.gather(x_clean, 1, parent_pos_exp)  # [B, N_high, C]

        # q_sample at t = num_timesteps_next - 1
        if alpha_cumprod_T is not None:
            ac_T = alpha_cumprod_T
        else:
            alphas_cumprod = np.cumprod(1.0 - betas_next, axis=0)
            ac_T = alphas_cumprod[-1]
        sqrt_alpha_T = math.sqrt(ac_T)
        sqrt_one_minus_alpha_T = math.sqrt(1.0 - ac_T)

        noise = torch.randn(B, N_high, C, device=device, dtype=x_clean.dtype)
        x_noisy = sqrt_alpha_T * x_up + sqrt_one_minus_alpha_T * noise
        return x_noisy


# ==============================================================================
#                         SpacedDiffusion3D (Sparse)
# ==============================================================================

def space_timesteps_3d(num_timesteps, section_counts):
    """
    Select a subset of timesteps from [0, num_timesteps) evenly.
    section_counts: int or str (number of desired steps).
    Returns: set of selected timestep indices.
    """
    if isinstance(section_counts, str):
        section_counts = int(section_counts)
    if section_counts >= num_timesteps:
        return set(range(num_timesteps))
    stride = num_timesteps / section_counts
    return set(int(i * stride) for i in range(section_counts))


class SpacedDiffusion3D(GaussianDiffusion3D):
    """
    Wraps GaussianDiffusion3D with timestep spacing for fewer sampling steps.
    Sparse format: all tensors are [B, N, C].
    """

    def __init__(self, global_betas, use_timesteps, stage_index=0, T_start=0, T_end=None, resolution=None):
        """
        global_betas: full 1000-step beta schedule
        use_timesteps: set of global timestep indices to use for this stage
        """
        global_alphas_cumprod = np.cumprod(1.0 - global_betas, axis=0)

        last_alpha_cumprod = global_alphas_cumprod[T_start - 1] if T_start > 0 else 1.0
        new_betas = []
        self.timestep_map = []

        for i in sorted(use_timesteps):
            new_betas.append(1.0 - global_alphas_cumprod[i] / last_alpha_cumprod)
            last_alpha_cumprod = global_alphas_cumprod[i]
            self.timestep_map.append(i)

        new_betas = np.array(new_betas, dtype=np.float64)
        self.timestep_map = np.array(self.timestep_map)

        super().__init__(new_betas, stage_index=stage_index, T_start=T_start, T_end=T_end, resolution=resolution)
        self.original_num_timesteps = T_end - T_start if T_end else len(global_betas)

    def _map_timestep(self, t):
        """Map local spaced index to global timestep for model conditioning."""
        return torch.tensor(self.timestep_map, device=t.device, dtype=torch.long)[t]

    def _kappa(self, t_local):
        """Kappa uses mapped global timesteps for correct drift scaling."""
        t_global = torch.tensor(self.timestep_map, device=t_local.device, dtype=torch.float32)[t_local]
        DT = float(self.T_end - self.T_start)
        return (t_global - self.T_start) / DT if DT > 0 else t_local.float() * 0

    def _compute_posterior_mean_coef_drift(self):
        """
        Override: use timestep_map for kappa in the drift coefficient.
        coef = sqrt(ab_prev) / (1-ab) * (kappa*(1-alpha) + (ab-alpha)/DT)
        where kappa = (mapped_global_t - T_start) / DT.
        """
        if self.stage_index == 0:
            return np.zeros(self.num_timesteps, dtype=np.float64)

        DT = float(self.T_end - self.T_start)
        if DT == 0:
            return np.zeros(self.num_timesteps, dtype=np.float64)

        alphas = 1.0 - self.betas
        # Use mapped global timesteps for kappa
        kappa = (self.timestep_map - self.T_start) / DT

        coef = (
            np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-20) *
            (kappa * (1.0 - alphas) + (self.alphas_cumprod - alphas) / DT)
        )
        return coef

    def p_mean_variance(self, model, x_t, t, y, clip_denoised=True,
                        coords=None, resolution=None):
        """Override: model receives global timestep via timestep_map."""
        t_global = self._map_timestep(t)
        model_output = model(x_t, t_global, y)

        sqrt_recip = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * model_output

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-5, 5)

        coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * pred_x0 + coef2 * x_t

        # Drift correction for non-coarsest stages
        if self.stage_index > 0 and coords is not None and resolution is not None:
            d = self.compute_drift(pred_x0, coords, resolution)
            coef_drift = _extract_into_tensor(
                self._compute_posterior_mean_coef_drift(), t, x_t.shape)
            posterior_mean = posterior_mean + coef_drift * d

        posterior_log_var = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return {
            "mean": posterior_mean,
            "log_variance": posterior_log_var,
            "pred_x0": pred_x0,
        }

    def training_losses(self, model, x_0, t, y, coords, noise=None, padding_mask=None):
        """Override: use mapped timesteps for kappa and model conditioning."""
        if noise is None:
            noise = torch.randn_like(x_0)

        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        if self.stage_index > 0:
            kappa = self._kappa(t).reshape(-1, 1, 1)
            sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            noise_target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * kappa * drift
        else:
            noise_target = noise

        t_global = self._map_timestep(t)
        model_output = model(x_t, t_global, y)

        # MSE loss only on valid (non-padding) tokens
        mse = (model_output - noise_target) ** 2  # [B, N, C]
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
            mse = mse * mask
            C = mse.shape[-1]
            n_valid = padding_mask.sum(dim=1).clamp(min=1).float()  # [B]
            loss = mse.sum(dim=(1, 2)) / (n_valid * C)
        else:
            loss = mean_flat(mse)
        return {"loss": loss}

    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None, padding_mask=None):
        """Override: decouple training with mapped timesteps."""
        if noise is None:
            noise = torch.randn_like(x_0)

        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = self._map_timestep(t)
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]

        C = x_0.shape[-1]
        eps_pred = model_output[..., :C]
        d_pred = model_output[..., C:]

        if self.stage_index > 0:
            kappa = self._kappa(t).reshape(-1, 1, 1)
            sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            noise_target = noise + (sqrt_alpha / sqrt_one_minus_alpha) * kappa * drift
        else:
            noise_target = noise

        # MSE loss only on valid (non-padding) tokens
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()  # [B, N, 1]
            mse_eps_raw = (eps_pred - noise_target) ** 2 * mask
            mse_d_raw = (d_pred - drift) ** 2 * mask
            C = eps_pred.shape[-1]
            n_valid = padding_mask.sum(dim=1).clamp(min=1).float()  # [B]
            mse_eps = mse_eps_raw.sum(dim=(1, 2)) / (n_valid * C)
            mse_d = mse_d_raw.sum(dim=(1, 2)) / (n_valid * C)
        else:
            mse_eps = mean_flat((eps_pred - noise_target) ** 2)
            mse_d = mean_flat((d_pred - drift) ** 2)
        loss = mse_eps + mse_d

        return {"loss": loss, "mse_eps": mse_eps, "mse_d": mse_d}

    def jump_step_sample(self, model, x, t, t_next, y, coords=None, resolution=None,
                         deterministic=True):
        """Override: use timestep_map for kappa in drift correction."""
        out = self.p_mean_variance(model, x, t, y)
        pred_xstart = out["pred_x0"]

        ab_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        t_next_clamped = t_next.clamp(min=0)
        ab_s = _extract_into_tensor(self.alphas_cumprod, t_next_clamped, x.shape)
        final_mask = (t_next < 0).float().reshape(-1, 1, 1)
        ab_s = ab_s * (1 - final_mask) + final_mask

        Sigma = (ab_s - ab_t) * (1.0 - ab_s) / (ab_s * (1.0 - ab_t) + 1e-20)
        coeff_xt = (1.0 - ab_s) / (1.0 - ab_t + 1e-20) * torch.sqrt(ab_t / (ab_s + 1e-20))
        coeff_x0 = (ab_s - ab_t) / (torch.sqrt(ab_s + 1e-20) * (1.0 - ab_t + 1e-20))

        mu = coeff_xt * x + coeff_x0 * pred_xstart

        if self.stage_index > 0 and coords is not None and resolution is not None:
            d = self.compute_drift(pred_xstart, coords, resolution)
            DT = float(self.T_end - self.T_start)
            # Use mapped global timesteps for kappa
            t_global_mapped = torch.tensor(
                self.timestep_map, device=t.device, dtype=torch.float32)[t]
            t_next_global_mapped = torch.tensor(
                self.timestep_map, device=t.device, dtype=torch.float32)[t_next_clamped]
            kappa_t = ((t_global_mapped - self.T_start) / DT).reshape(-1, 1, 1)
            kappa_s = ((t_next_global_mapped - self.T_start) / DT).reshape(-1, 1, 1)

            # Exact posterior drift coefficient (matching 2D implementation)
            sigma1_sq = 1.0 - ab_t / (ab_s + 1e-20)
            sigma2_sq = 1.0 - ab_s
            A_coef = torch.sqrt(ab_t / (ab_s + 1e-20))
            B_coef = torch.sqrt(ab_t) * (kappa_t - kappa_s)
            D_coef = torch.sqrt(ab_s) * kappa_s

            coeff_d = Sigma * (D_coef / (sigma2_sq + 1e-20) - A_coef * B_coef / (sigma1_sq + 1e-20))
            mu = mu + coeff_d * d

        if deterministic:
            return {"sample": mu, "pred_x0": pred_xstart}
        else:
            noise = torch.randn_like(x)
            Sigma_clamped = Sigma.clamp(min=0)
            sample = mu + torch.sqrt(Sigma_clamped) * noise
            return {"sample": sample, "pred_x0": pred_xstart}


# ==============================================================================
#                         Factory
# ==============================================================================

def create_diffusion_3d(
    schedule_name="linear",
    steps_per_stage=(400, 300, 300),
    diffusion_steps=1000,
    sampling_steps_per_stage=None,
    stage_resolutions=None,
):
    """
    Create diffusion objects for each stage.

    Args:
        schedule_name: beta schedule type
        steps_per_stage: training step allocation per stage (defines global schedule split)
        diffusion_steps: total timesteps in full schedule
        sampling_steps_per_stage: if provided, creates SpacedDiffusion3D with fewer steps.
                                  e.g. (100, 80, 80) for accelerated sampling.
                                  If None, returns full-step GaussianDiffusion3D.
        stage_resolutions: tuple of ints, spatial resolution for each stage (e.g. (8,16,32,64,128,256)).
                          Used for drift computation. If None, resolution is inferred from coords.
    Returns: list of GaussianDiffusion3D or SpacedDiffusion3D (one per stage)
    """
    all_betas = get_named_beta_schedule(schedule_name, diffusion_steps)
    diffusions = []
    t_start = 0

    for i, num_steps in enumerate(steps_per_stage):
        t_end = t_start + num_steps
        res_i = stage_resolutions[i] if stage_resolutions is not None else None

        if sampling_steps_per_stage is not None:
            sample_steps = sampling_steps_per_stage[i]
            use_timesteps = set()
            if sample_steps >= num_steps:
                use_timesteps = set(range(t_start, t_end))
            else:
                stride = num_steps / sample_steps
                for j in range(sample_steps):
                    use_timesteps.add(t_start + int(j * stride))

            diffusions.append(SpacedDiffusion3D(
                global_betas=all_betas,
                use_timesteps=use_timesteps,
                stage_index=i,
                T_start=t_start,
                T_end=t_end,
                resolution=res_i,
            ))
        else:
            stage_betas = all_betas[t_start:t_end]
            diffusions.append(GaussianDiffusion3D(
                betas=stage_betas,
                stage_index=i,
                T_start=t_start,
                T_end=t_end,
                resolution=res_i,
            ))
        t_start = t_end

    return diffusions
