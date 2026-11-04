#!/usr/bin/env python3
"""Apply padding_mask changes to diffusion_3d.py"""

filepath = '/home/tiger/multiscale_diffusion/RDD_DiT_3D/diffusion_3d.py'

with open(filepath, 'r') as f:
    content = f.read()

# 1. Modify compute_drift_sparse function signature and body
old_func = '''def compute_drift_sparse(x_0, coords, resolution):
    """
    Compute drift for sparse tokens.
    
    Downsample: coords // 2, mean pool tokens sharing the same low-res coordinate.
    Upsample: copy low-res embedding back to each high-res token.
    drift = upsampled - x_0

    Args:
        x_0: [B, N, C] token embeddings
        coords: [B, N, 3] integer coordinates at current resolution
        resolution: int, current spatial resolution R
    Returns:
        drift: [B, N, C]
    """
    B, N, C = x_0.shape
    R_low = resolution // 2

    # Low-res coordinates via floor division
    coords_low = coords // 2  # [B, N, 3]

    if R_low <= 64:
        # Dense scatter for small resolutions
        flat_low = _coords_to_flat_index(coords_low, R_low)  # [B, N]
        num_low_voxels = R_low ** 3
        flat_low_exp = flat_low.unsqueeze(-1).expand(B, N, C)  # [B, N, C]

        sum_buf = torch.zeros(B, num_low_voxels, C, device=x_0.device, dtype=x_0.dtype)
        sum_buf.scatter_add_(1, flat_low_exp, x_0)

        count_buf = torch.zeros(B, num_low_voxels, 1, device=x_0.device, dtype=x_0.dtype)
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
            group_sum.scatter_add_(0, inverse.unsqueeze(-1).expand(N, C), x_0[b])
            group_count.scatter_add_(0, inverse.unsqueeze(-1), torch.ones(N, 1, device=x_0.device))
            group_mean = group_sum / group_count.clamp(min=1.0)
            # Gather back
            x_up[b] = group_mean[inverse]

    drift = x_up - x_0
    return drift'''

new_func = '''def compute_drift_sparse(x_0, coords, resolution, padding_mask=None):
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
    return drift'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("SUCCESS: compute_drift_sparse replaced")
else:
    print("ERROR: compute_drift_sparse not found exactly")
    # Try to debug - print lines around the function
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def compute_drift_sparse' in line:
            print(f"Found at line {i+1}: repr={repr(line)}")
            for j in range(i, min(i+5, len(lines))):
                print(f"  Line {j+1}: {repr(lines[j])}")
            break

# 2. Modify compute_drift method in GaussianDiffusion3D
old_compute_drift = '''    def compute_drift(self, x_0, coords, resolution):
        """
        Compute drift = upsample(downsample(x_0)) - x_0 in sparse representation.
        Only for non-coarsest stages.

        Args:
            x_0: [B, N, C]
            coords: [B, N, 3]
            resolution: int
        Returns:
            drift: [B, N, C]
        """
        if self.stage_index == 0:
            return torch.zeros_like(x_0)
        return compute_drift_sparse(x_0, coords, resolution)'''

new_compute_drift = '''    def compute_drift(self, x_0, coords, resolution, padding_mask=None):
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
        return compute_drift_sparse(x_0, coords, resolution, padding_mask=padding_mask)'''

if old_compute_drift in content:
    content = content.replace(old_compute_drift, new_compute_drift)
    print("SUCCESS: compute_drift method replaced")
else:
    print("ERROR: compute_drift method not found exactly")

# 3. Modify first training_losses (line 391)
old_tl1 = '    def training_losses(self, model, x_0, t, y, coords, noise=None):\n        """\n        Compute training loss for one stage.'
new_tl1 = '    def training_losses(self, model, x_0, t, y, coords, noise=None, padding_mask=None):\n        """\n        Compute training loss for one stage.'

if old_tl1 in content:
    content = content.replace(old_tl1, new_tl1, 1)
    print("SUCCESS: first training_losses signature replaced")
else:
    print("ERROR: first training_losses signature not found")

# 3b. Modify the drift call in first training_losses
old_drift1 = '''        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        # Compute noise target with drift correction'''

new_drift1 = '''        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        # Compute noise target with drift correction'''

if old_drift1 in content:
    content = content.replace(old_drift1, new_drift1, 1)
    print("SUCCESS: first training_losses drift call replaced")
else:
    print("ERROR: first training_losses drift call not found")

# 4. Modify first training_losses_decouple (line 426)
old_tld1 = '    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None):\n        """\n        Decouple training: model outputs [eps_pred | d_pred] concatenated along last dim.'
new_tld1 = '    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None, padding_mask=None):\n        """\n        Decouple training: model outputs [eps_pred | d_pred] concatenated along last dim.'

if old_tld1 in content:
    content = content.replace(old_tld1, new_tld1, 1)
    print("SUCCESS: first training_losses_decouple signature replaced")
else:
    print("ERROR: first training_losses_decouple signature not found")

# 4b. Modify the drift call in first training_losses_decouple
old_drift_d1 = '''        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = t + self.T_start
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]'''

new_drift_d1 = '''        resolution = self.resolution if self.resolution is not None else int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = t + self.T_start
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]'''

if old_drift_d1 in content:
    content = content.replace(old_drift_d1, new_drift_d1, 1)
    print("SUCCESS: first training_losses_decouple drift call replaced")
else:
    print("ERROR: first training_losses_decouple drift call not found")

# 5. Modify second training_losses (line 831, the override)
old_tl2 = '    def training_losses(self, model, x_0, t, y, coords, noise=None):\n        """Override: use mapped timesteps for kappa and model conditioning."""'
new_tl2 = '    def training_losses(self, model, x_0, t, y, coords, noise=None, padding_mask=None):\n        """Override: use mapped timesteps for kappa and model conditioning."""'

if old_tl2 in content:
    content = content.replace(old_tl2, new_tl2, 1)
    print("SUCCESS: second training_losses signature replaced")
else:
    print("ERROR: second training_losses signature not found")

# 5b. Modify the drift call in second training_losses
old_drift2 = '''        resolution = int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution)
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
        loss = mean_flat((model_output - noise_target) ** 2)
        return {"loss": loss}'''

new_drift2 = '''        resolution = int(coords.max().item()) + 1
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
        loss = mean_flat((model_output - noise_target) ** 2)
        return {"loss": loss}'''

if old_drift2 in content:
    content = content.replace(old_drift2, new_drift2, 1)
    print("SUCCESS: second training_losses drift call replaced")
else:
    print("ERROR: second training_losses drift call not found")

# 6. Modify second training_losses_decouple (line 853, the override)
old_tld2 = '    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None):\n        """Override: decouple training with mapped timesteps."""'
new_tld2 = '    def training_losses_decouple(self, model, x_0, t, y, coords, noise=None, padding_mask=None):\n        """Override: decouple training with mapped timesteps."""'

if old_tld2 in content:
    content = content.replace(old_tld2, new_tld2, 1)
    print("SUCCESS: second training_losses_decouple signature replaced")
else:
    print("ERROR: second training_losses_decouple signature not found")

# 6b. Modify the drift call in second training_losses_decouple
old_drift_d2 = '''        resolution = int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = self._map_timestep(t)
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]'''

new_drift_d2 = '''        resolution = int(coords.max().item()) + 1
        drift = self.compute_drift(x_0, coords, resolution, padding_mask=padding_mask)
        x_t = self.q_sample(x_0, t, noise=noise, drift=drift)

        t_global = self._map_timestep(t)
        model_output = model(x_t, t_global, y)  # [B, N, 2*C]'''

if old_drift_d2 in content:
    content = content.replace(old_drift_d2, new_drift_d2, 1)
    print("SUCCESS: second training_losses_decouple drift call replaced")
else:
    print("ERROR: second training_losses_decouple drift call not found")

with open(filepath, 'w') as f:
    f.write(content)

print("\nDone! File written.")
