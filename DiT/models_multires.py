# Multi-Resolution DiT with 2D RoPE (Centralized Interpolative Positional Encoding)
# Based on: Original DiT (Meta) + HunyuanDiT's multi-resolution RoPE strategy
#
# Key difference from original DiT:
#   - Replaces fixed sin-cos pos_embed with 2D RoPE applied inside attention
#   - Supports arbitrary input resolutions at both training and inference time
#   - Single model handles all resolution stages in RDD

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp


#################################################################################
#                            2D RoPE Implementation                             #
#################################################################################

def get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    """
    Compute 1D rotary positional embeddings (cos, sin) for given positions.
    Returns: (cos, sin) each of shape (S, D) where D = dim.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim))
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=torch.float32)
    else:
        pos = pos.float()
    freqs = torch.outer(pos, freqs)  # (S, D/2)
    cos = freqs.cos().repeat_interleave(2, dim=1)  # (S, D)
    sin = freqs.sin().repeat_interleave(2, dim=1)  # (S, D)
    return cos, sin


def get_2d_rotary_pos_embed(head_dim, grid_h, grid_w, base_size=None):
    """
    Compute 2D RoPE for a grid of (H, W) positions.

    If base_size is provided, uses Centralized Interpolative Positional Encoding:
      positions are linearly interpolated into [0, base_size] range regardless of
      actual grid size, ensuring consistent positional encoding across resolutions.

    If base_size is None, uses extend mode (positions = 0, 1, ..., H-1).

    Args:
        head_dim: dimension per attention head (will use head_dim/2 for each axis)
        grid_h: number of patches in height
        grid_w: number of patches in width
        base_size: base resolution for centralized interpolation (e.g. 16 for 32x32/patch2)

    Returns:
        (cos, sin): each of shape (H*W, head_dim)
    """
    half_dim = head_dim // 2

    if base_size is not None:
        # Centralized Interpolative Positional Encoding (HunyuanDiT paper Eq. 2)
        # Map coordinates to [0, base_size] range centered
        pos_h = torch.linspace(0, base_size, steps=grid_h)
        pos_w = torch.linspace(0, base_size, steps=grid_w)
    else:
        # Extend mode: integer positions
        pos_h = torch.arange(grid_h, dtype=torch.float32)
        pos_w = torch.arange(grid_w, dtype=torch.float32)

    # Get 1D embeddings for each axis
    cos_h, sin_h = get_1d_rotary_pos_embed(half_dim, pos_h)  # (H, half_dim)
    cos_w, sin_w = get_1d_rotary_pos_embed(half_dim, pos_w)  # (W, half_dim)

    # Create 2D grid: expand to (H, W, half_dim) then flatten to (H*W, half_dim)
    cos_h = cos_h.unsqueeze(1).expand(-1, grid_w, -1).reshape(grid_h * grid_w, half_dim)
    sin_h = sin_h.unsqueeze(1).expand(-1, grid_w, -1).reshape(grid_h * grid_w, half_dim)
    cos_w = cos_w.unsqueeze(0).expand(grid_h, -1, -1).reshape(grid_h * grid_w, half_dim)
    sin_w = sin_w.unsqueeze(0).expand(grid_h, -1, -1).reshape(grid_h * grid_w, half_dim)

    # Concatenate: first half_dim for height, second half_dim for width
    cos = torch.cat([cos_h, cos_w], dim=1)  # (H*W, head_dim)
    sin = torch.cat([sin_h, sin_w], dim=1)  # (H*W, head_dim)
    return cos, sin


def rotate_half(x):
    """Rotate pairs of elements: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embedding to tensor x.
    x: (B, num_heads, S, head_dim)
    cos, sin: (S, head_dim)  -- will be broadcast to batch and head dims
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, S, head_dim)
    return x * cos + rotate_half(x) * sin


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with classifier-free guidance dropout."""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                         Attention with RoPE                                   #
#################################################################################

class AttentionWithRoPE(nn.Module):
    """
    Multi-head self-attention with 2D Rotary Positional Embedding.
    Replaces timm's Attention to inject RoPE into Q and K.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos, rope_sin):
        """
        x: (B, S, D)
        rope_cos, rope_sin: (S, head_dim) -- precomputed for current resolution
        """
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, H, S, head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = self.proj(x)
        return x


#################################################################################
#                              Core DiT Blocks                                  #
#################################################################################

class DiTBlockMultiRes(nn.Module):
    """DiT block with adaLN-Zero conditioning and RoPE-based attention."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithRoPE(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, rope_cos, rope_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_cos, rope_sin)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                         Multi-Resolution PatchEmbed                           #
#################################################################################

class PatchEmbedMultiRes(nn.Module):
    """
    2D Image to Patch Embedding that accepts any spatial resolution.
    No fixed num_patches assertion — works with variable H, W.
    """
    def __init__(self, patch_size=2, in_channels=4, embed_dim=1152):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) — H, W can be arbitrary multiples of patch_size
        x = self.proj(x)          # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, T, D) where T = (H/p)*(W/p)
        return x


#################################################################################
#                          DiT Multi-Resolution Model                           #
#################################################################################

class DiTMultiRes(nn.Module):
    """
    Diffusion Transformer that handles arbitrary resolution inputs via 2D RoPE.

    Key design:
    - Uses Centralized Interpolative Positional Encoding (from HunyuanDiT paper)
    - All resolutions share the same positional encoding range [0, base_size]
    - Compatible with DiT-XL/2, DiT-L/2, etc. naming convention
    - Single model can process 8x8, 16x16, 32x32, 64x64 latents
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        d_prediction=False,
        base_size=16,       # base grid size for RoPE interpolation (e.g. 16 patches = 32x32 latent / patch_size 2)
        rope_theta=10000.0,
        repa_dim=None,
        repa_depth=8,
        repa_projector_dim=2048,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.d_prediction = d_prediction
        self.base_size = base_size
        self.rope_theta = rope_theta
        self.repa_dim = repa_dim
        self.repa_depth = repa_depth
        if repa_dim is not None and not (1 <= repa_depth <= depth):
            raise ValueError(f"repa_depth must be in [1, {depth}], got {repa_depth}")

        # Patch embedding (resolution-agnostic)
        self.x_embedder = PatchEmbedMultiRes(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Transformer blocks with RoPE attention
        self.blocks = nn.ModuleList([
            DiTBlockMultiRes(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Optional REPA projection head.  It is absent by default, so legacy
        # model state_dicts and forward outputs remain byte-for-byte compatible.
        if repa_dim is not None:
            self.repa_projector = nn.Sequential(
                nn.Linear(hidden_size, repa_projector_dim),
                nn.SiLU(),
                nn.Linear(repa_projector_dim, repa_dim),
            )

        # Optional drift prediction head
        if d_prediction:
            self.d_final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.d_prediction:
            nn.init.constant_(self.d_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.d_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.d_final_layer.linear.weight, 0)
            nn.init.constant_(self.d_final_layer.linear.bias, 0)

    def get_rope(self, grid_h, grid_w, device):
        """
        Compute 2D RoPE (cos, sin) for a (grid_h, grid_w) token grid.
        Uses centralized interpolative encoding: all resolutions map to [0, base_size].
        """
        cos, sin = get_2d_rotary_pos_embed(self.head_dim, grid_h, grid_w, base_size=self.base_size)
        return cos.to(device), sin.to(device)

    def unpatchify(self, x, h, w, channels=None):
        """
        x: (N, T, patch_size**2 * C)
        Returns: (N, C, H, W) where H = h * patch_size, W = w * patch_size
        """
        c = channels if channels is not None else self.out_channels
        p = self.patch_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, t, y, return_features=False):
        """
        Forward pass of DiT Multi-Resolution.
        x: (N, C, H, W) — H, W can be any multiple of patch_size
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        B, C, H, W = x.shape
        p = self.patch_size
        grid_h, grid_w = H // p, W // p

        # Patchify
        x = self.x_embedder(x)  # (N, T, D) where T = grid_h * grid_w

        # Compute RoPE for this resolution
        rope_cos, rope_sin = self.get_rope(grid_h, grid_w, x.device)

        # Condition embedding
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        # Transformer blocks
        repa_features = None
        for block_index, block in enumerate(self.blocks, start=1):
            x = block(x, c, rope_cos, rope_sin)
            if return_features and block_index == self.repa_depth:
                if self.repa_dim is None:
                    raise RuntimeError(
                        "return_features=True requires constructing DiTMultiRes with repa_dim"
                    )
                repa_features = self.repa_projector(x)

        # Output
        main_out = self.final_layer(x, c)
        main_out = self.unpatchify(main_out, grid_h, grid_w)

        if self.d_prediction:
            d_out = self.d_final_layer(x, c)
            d_out = self.unpatchify(d_out, grid_h, grid_w, channels=self.in_channels)
            main_out = torch.cat([main_out, d_out], dim=1)
        if return_features:
            if repa_features is None:
                raise RuntimeError("REPA features were not captured; check repa_depth")
            return main_out, repa_features
        return main_out

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        if self.d_prediction:
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            cond_rest, uncond_rest = torch.split(rest, len(rest) // 2, dim=0)
            rest = torch.cat([cond_rest, cond_rest], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)


#################################################################################
#                              Model Configs                                    #
#################################################################################

def DiTMultiRes_XL_2(**kwargs):
    return DiTMultiRes(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiTMultiRes_XL_4(**kwargs):
    return DiTMultiRes(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiTMultiRes_XL_8(**kwargs):
    return DiTMultiRes(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiTMultiRes_L_2(**kwargs):
    return DiTMultiRes(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiTMultiRes_L_4(**kwargs):
    return DiTMultiRes(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiTMultiRes_L_8(**kwargs):
    return DiTMultiRes(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiTMultiRes_B_2(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTMultiRes_B_4(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTMultiRes_B_8(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTMultiRes_S_2(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiTMultiRes_S_4(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTMultiRes_S_8(**kwargs):
    return DiTMultiRes(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiTMultiRes_models = {
    'DiT-XL/2': DiTMultiRes_XL_2,  'DiT-XL/4': DiTMultiRes_XL_4,  'DiT-XL/8': DiTMultiRes_XL_8,
    'DiT-L/2':  DiTMultiRes_L_2,   'DiT-L/4':  DiTMultiRes_L_4,   'DiT-L/8':  DiTMultiRes_L_8,
    'DiT-B/2':  DiTMultiRes_B_2,   'DiT-B/4':  DiTMultiRes_B_4,   'DiT-B/8':  DiTMultiRes_B_8,
    'DiT-S/2':  DiTMultiRes_S_2,   'DiT-S/4':  DiTMultiRes_S_4,   'DiT-S/8':  DiTMultiRes_S_8,
}
