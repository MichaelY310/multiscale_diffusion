"""
DiT3D_RDD: 全稀疏 3D Diffusion Transformer for multi-resolution RDD.
参考 Open-Sora MMDiT 架构：
- 3D RoPE 位置编码（Centralized Interpolative Positional Encoding, HunyuanDiT style）
- SingleStreamBlock（并行 Attention + MLP，PaLM 风格）
- QK Norm（per-head RMSNorm）
- adaLN-Zero 调制
- 支持 padding mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###############################################################################
#                          3D RoPE Position Encoding                           #
###############################################################################

def compute_3d_rope(coords, head_dim, resolution, base_size=8, theta=10000):
    """
    3D RoPE with Centralized Interpolative Positional Encoding.

    Instead of normalizing by max_resolution (which makes low-res positions clustered near 0),
    we map all resolutions to [0, base_size] range, ensuring consistent positional encoding.

    Args:
        coords: [B, N, 3] int tensor
        head_dim: int
        resolution: int, current stage resolution (e.g. 8, 16, 32, ...)
        base_size: float, base size for interpolation (default 8)
        theta: int, RoPE theta
    Returns:
        cos: [B, N, head_dim]
        sin: [B, N, head_dim]
    """
    B, N, _ = coords.shape

    # Axes dimension allocation (head_dim split into 3 parts for x, y, z)
    base = head_dim // 3
    base = base - (base % 2)  # ensure even
    axes_dim = [base, base, head_dim - 2 * base]
    if axes_dim[2] % 2 != 0:
        axes_dim[1] += 1
        axes_dim[2] -= 1

    assert sum(axes_dim) == head_dim, f"axes_dim {axes_dim} sum != head_dim {head_dim}"
    assert all(d % 2 == 0 for d in axes_dim), f"axes_dim must all be even: {axes_dim}"

    # Centralized Interpolative: map [0, R-1] -> [0, base_size]
    if resolution > 1:
        norm_coords = coords.float() * (base_size / (resolution - 1))
    else:
        norm_coords = coords.float() * 0.0  # single voxel case

    cos_parts = []
    sin_parts = []
    for axis in range(3):
        dim_a = axes_dim[axis]
        pos = norm_coords[:, :, axis]  # [B, N]
        # Compute frequencies
        scale = torch.arange(0, dim_a, 2, dtype=torch.float32, device=coords.device) / dim_a
        omega = 1.0 / (theta ** scale)  # [dim_a//2]
        # pos: [B, N], omega: [dim_a//2] -> angles: [B, N, dim_a//2]
        angles = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
        cos_a = angles.cos().repeat_interleave(2, dim=-1)  # [B, N, dim_a]
        sin_a = angles.sin().repeat_interleave(2, dim=-1)  # [B, N, dim_a]
        cos_parts.append(cos_a)
        sin_parts.append(sin_a)

    cos = torch.cat(cos_parts, dim=-1)  # [B, N, head_dim]
    sin = torch.cat(sin_parts, dim=-1)  # [B, N, head_dim]
    return cos, sin


def rotate_half(x):
    """[..., 0::2] -> swap pairs with negation."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(x, cos, sin):
    """
    x: [B, H, N, D]
    cos, sin: [B, N, D] -> broadcast to [B, 1, N, D]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


###############################################################################
#                              Building Blocks                                  #
###############################################################################

class RMSNorm(nn.Module):
    """Per-feature RMSNorm (with learnable scale). Works in native dtype (bf16/fp16)."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (x * rrms) * self.scale


class QKNorm(nn.Module):
    """Per-head RMSNorm for Q and K (stabilizes attention)."""
    def __init__(self, head_dim):
        super().__init__()
        self.query_norm = RMSNorm(head_dim)
        self.key_norm = RMSNorm(head_dim)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class Modulation(nn.Module):
    """adaLN-Zero modulation: produces shift, scale, gate from conditioning vector."""
    def __init__(self, dim, double=False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        # vec: [B, D] -> [B, 1, multiplier*D] -> chunk
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        if self.is_double:
            return (out[0], out[1], out[2]), (out[3], out[4], out[5])
        return out[0], out[1], out[2]  # shift, scale, gate


###############################################################################
#                    SingleStreamBlock (Open-Sora / PaLM style)                #
###############################################################################

class SingleStreamBlock(nn.Module):
    """
    并行 Attention + MLP block (PaLM 风格)，带 adaLN-Zero 调制。
    单次前向通过一个大 Linear 同时计算 QKV 和 MLP 输入，
    然后并行执行 attention 和 MLP activation，最后通过一个 Linear 合并输出。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Pre-norm
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Fused linear: QKV (3*hidden) + MLP_in (mlp_hidden_dim)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)

        # QK Norm (per-head RMSNorm)
        self.qk_norm = QKNorm(self.head_dim)

        # MLP activation
        self.mlp_act = nn.GELU(approximate="tanh")

        # Output linear: attn_out (hidden) + mlp_out (mlp_hidden) -> hidden
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # adaLN-Zero modulation (3 params: shift, scale, gate)
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, vec, pe, padding_mask=None):
        """
        Args:
            x: [B, N, D]
            vec: [B, D] — conditioning vector (timestep + class)
            pe: (cos, sin) tuple — cos/sin each [B, N, head_dim]
            padding_mask: [B, N] bool — True=valid, False=padding
        """
        # Modulation
        shift, scale, gate = self.modulation(vec)  # each [B, 1, D]
        x_mod = (1 + scale) * self.pre_norm(x) + shift

        # Fused QKV + MLP input
        qkv, mlp_in = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim],
            dim=-1
        )

        # Reshape QKV: [B, N, 3*D] -> [B, H, N, head_dim] * 3
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv.unbind(0)  # each [B, H, N, head_dim]

        # QK Norm
        q, k = self.qk_norm(q, k)

        # Apply 3D RoPE to Q and K
        cos_pe, sin_pe = pe
        q = apply_rotary_emb(q, cos_pe, sin_pe)
        k = apply_rotary_emb(k, cos_pe, sin_pe)

        # Attention with padding mask
        attn_mask = None
        if padding_mask is not None:
            # padding_mask: [B, N] True=valid, False=padding
            # Key mask: [B, 1, 1, N] — padding keys get -inf score
            attn_mask = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(~padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.hidden_size)

        # Parallel MLP
        mlp_out = self.mlp_act(mlp_in)

        # Merge attention output and MLP output
        output = self.linear2(torch.cat([attn_out, mlp_out], dim=-1))

        # Gated residual
        x = x + gate * output
        return x


###############################################################################
#                              Final Layer                                      #
###############################################################################

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


###############################################################################
#                     Main Model: DiT3D_RDD (Open-Sora style)                  #
###############################################################################

class DiT3D_RDD(nn.Module):
    """
    全稀疏 3D DiT for RDD，参考 Open-Sora MMDiT 架构：
    - 3D RoPE 位置编码（Centralized Interpolative Positional Encoding）
    - SingleStreamBlock（并行 Attn + MLP）
    - QK Norm
    - adaLN-Zero 调制

    forward 接口：(x_active, coords, t, y, padding_mask, resolution)
    """
    def __init__(self, in_channels=128, hidden_size=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=55,
                 learn_sigma=False, base_size=8, d_prediction=False,
                 rope_theta=10000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.base_size = base_size
        self.d_prediction = d_prediction
        self.rope_theta = rope_theta
        self.gradient_checkpointing = False

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Conditioning embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Output heads
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        if d_prediction:
            self.d_final_layer = FinalLayer(hidden_size, in_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # Embedder init
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init modulation output (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.modulation.lin.weight, 0)
            nn.init.constant_(block.modulation.lin.bias, 0)

        # Zero-init final layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.d_prediction:
            nn.init.constant_(self.d_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.d_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.d_final_layer.linear.weight, 0)
            nn.init.constant_(self.d_final_layer.linear.bias, 0)

    def forward(self, x_active, coords, t, y, padding_mask=None, resolution=None):
        """
        全稀疏 forward。

        Args:
            x_active: [B, N_max, C] — padding 后的 active token embeddings
            coords: [B, N_max, 3] — 对应的整数 grid 坐标 (x, y, z)
            t: [B] — global timesteps
            y: [B] — class labels
            padding_mask: [B, N_max] bool — True=valid, False=padding
            resolution: int — current stage resolution (e.g. 8, 16, 32, ...)

        Returns:
            output: [B, N_max, out_channels] 或 [B, N_max, out_channels + in_channels]
        """
        # Project input
        x = self.x_embedder(x_active)  # [B, N, hidden]

        # Compute 3D RoPE with centralized interpolative encoding
        res = resolution if resolution is not None else int(coords.max().item()) + 1
        cos_pe, sin_pe = compute_3d_rope(
            coords, self.head_dim, res, base_size=self.base_size, theta=self.rope_theta
        )
        pe = (cos_pe, sin_pe)

        # Conditioning vector
        vec = self.t_embedder(t) + self.y_embedder(y, self.training)  # [B, hidden]

        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, vec, pe, padding_mask, use_reentrant=False
                )
            else:
                x = block(x, vec, pe, padding_mask=padding_mask)

        # Output heads
        main_out = self.final_layer(x, vec)
        if self.d_prediction:
            d_out = self.d_final_layer(x, vec)
            out = torch.cat([main_out, d_out], dim=-1)
        else:
            out = main_out

        # Zero-out padding positions
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).float()

        return out

    def forward_with_cfg(self, x_active, coords, t, y, cfg_scale, padding_mask=None, resolution=None):
        """
        Classifier-free guidance forward.
        Input should be doubled: first half conditional, second half unconditional.
        """
        half_b = x_active.shape[0] // 2
        x_combined = torch.cat([x_active[:half_b], x_active[:half_b]], dim=0)
        coords_combined = torch.cat([coords[:half_b], coords[:half_b]], dim=0)
        pm_combined = None
        if padding_mask is not None:
            pm_combined = torch.cat([padding_mask[:half_b], padding_mask[:half_b]], dim=0)

        model_out = self.forward(
            x_combined, coords_combined, t, y,
            padding_mask=pm_combined, resolution=resolution
        )

        eps = model_out[:, :, :self.in_channels]
        rest = model_out[:, :, self.in_channels:]

        cond_eps, uncond_eps = eps[:half_b], eps[half_b:]
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest[:half_b].repeat(2, 1, 1)], dim=-1)


###############################################################################
#                          Model Configs                                        #
###############################################################################

def DiT3D_RDD_S(**kwargs):
    return DiT3D_RDD(hidden_size=384, depth=12, num_heads=6, **kwargs)

def DiT3D_RDD_B(**kwargs):
    return DiT3D_RDD(hidden_size=768, depth=12, num_heads=12, **kwargs)

def DiT3D_RDD_L(**kwargs):
    return DiT3D_RDD(hidden_size=1024, depth=24, num_heads=16, **kwargs)

def DiT3D_RDD_XL(**kwargs):
    return DiT3D_RDD(hidden_size=1152, depth=28, num_heads=16, **kwargs)

DiT3D_RDD_models = {
    'DiT3D-RDD-S': DiT3D_RDD_S,
    'DiT3D-RDD-B': DiT3D_RDD_B,
    'DiT3D-RDD-L': DiT3D_RDD_L,
    'DiT3D-RDD-XL': DiT3D_RDD_XL,
}
