"""
Face Set Transformer: 编码/解码一个 grid cell 内的三角形面片集合。

架构与 vertex 版本的 Set Transformer AE-Bridge 保持一致：
  FaceSetEncoder  →  FaceSetAutoencoder (bottleneck)  →  FaceSetDecoder

区别：
  - 输入：每个 face 是 9 维 (3 顶点 × 3 坐标的局部坐标)
  - 输出：重建每个 face 的 9 坐标
  - F_MAX=10 (resolution=512 下统计的最大值)
  - 解码方式支持回归 (MSE) 和分类 (CE with num_bins)

Usage:
    model = FaceSetTransformerAE(bottleneck_dim=128)
    faces = torch.randn(B, F_MAX, 9)   # 局部坐标
    mask  = torch.ones(B, F_MAX, dtype=torch.bool)
    recon, bottleneck = model(faces, mask)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 基础模块 (从 set_transformer/modules.py 复制并扩展)
# ===========================================================================

class MaskedMAB(nn.Module):
    """Multi-head Attention Block with optional key-padding mask."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, Q, K, mask=None):
        """
        Q: [B, N_q, dim_Q]
        K: [B, N_k, dim_K]
        mask: [B, N_k] bool — True=valid, False=padding
        """
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)

        B, N_q, _ = Q_.shape
        N_k = K_.shape[1]
        head_dim = self.dim_V // self.num_heads

        # Reshape for multi-head: [B*H, N, head_dim]
        Q_ = Q_.reshape(B, N_q, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * self.num_heads, N_q, head_dim)
        K_ = K_.reshape(B, N_k, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * self.num_heads, N_k, head_dim)
        V_ = V_.reshape(B, N_k, self.num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * self.num_heads, N_k, head_dim)

        # Attention
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(head_dim)  # [B*H, N_q, N_k]

        if mask is not None:
            # mask: [B, N_k] -> [B*H, 1, N_k]
            mask_expanded = mask.unsqueeze(1).repeat(1, self.num_heads, 1).reshape(B * self.num_heads, 1, N_k)
            A = A.masked_fill(~mask_expanded, float('-inf'))

        A = torch.softmax(A, dim=-1)
        A = A.nan_to_num(0.0)  # handle all-masked rows

        O = Q_ + A.bmm(V_)  # [B*H, N_q, head_dim]
        O = O.reshape(B, self.num_heads, N_q, head_dim).permute(0, 2, 1, 3).reshape(B, N_q, self.dim_V)

        O = O if not hasattr(self, 'ln0') else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if not hasattr(self, 'ln1') else self.ln1(O)
        return O


class MaskedSAB(nn.Module):
    """Self-Attention Block with mask."""

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MaskedMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask)


class MaskedISAB(nn.Module):
    """Induced Set Attention Block with mask."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MaskedMAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MaskedMAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.expand(X.size(0), -1, -1), X, mask)
        return self.mab1(X, H)


class MaskedPMA(nn.Module):
    """Pooling by Multi-head Attention with mask.
    将 [B, N, d] 压缩为 [B, num_seeds, d]。
    用 learnable seeds 对输入做 cross-attention。"""

    def __init__(self, dim, num_heads, num_seeds=1, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MaskedMAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        """X: [B, N, d] → [B, num_seeds, d]"""
        return self.mab(self.S.expand(X.size(0), -1, -1), X, mask)


# ===========================================================================
# Helper
# ===========================================================================

def _mlp(dim_in, dim_out, dim_hidden, num_layers):
    assert num_layers >= 2
    layers = []
    for i in range(num_layers):
        in_d = dim_in if i == 0 else dim_hidden
        out_d = dim_out if i == num_layers - 1 else dim_hidden
        layers.append(nn.Linear(in_d, out_d))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ===========================================================================
# Face Set Encoder
# ===========================================================================

class FaceSetEncoder(nn.Module):
    """
    编码 F_MAX 个 face（每个 DIM_IN 维）→ [B, F_MAX, dim_hidden]

    当 predict_scale=True 时，scale 会被拼接到每个 face 的特征上，
    使得 dim_in 有效变为 dim_in + 1，确保 scale 信息编码进 embedding。

    输入: [B, F_MAX, dim_in] + (可选) scale [B, 1]
    输出: [B, F_MAX, dim_hidden]
    """

    def __init__(self, dim_in=9, dim_hidden=128, num_heads=4,
                 num_layers=2, ln=False, predict_scale=False):
        super().__init__()
        self.predict_scale = predict_scale
        # 如果 predict_scale，每个 token 多拼一维 scale
        actual_dim_in = dim_in + 1 if predict_scale else dim_in
        self.proj = nn.Linear(actual_dim_in, dim_hidden)
        self.layers = nn.ModuleList([
            MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
            for _ in range(num_layers)
        ])

    def forward(self, X, mask=None, scale=None):
        """
        X: [B, F_MAX, dim_in] → [B, F_MAX, dim_hidden]
        scale: [B, 1] — 当 predict_scale=True 时必须提供
        """
        if self.predict_scale and scale is not None:
            # 将 scale broadcast 到每个 face token: [B, 1] → [B, F_MAX, 1]
            B_sz, F_sz, _ = X.shape
            scale_expanded = scale.unsqueeze(1).expand(B_sz, F_sz, 1)
            X = torch.cat([X, scale_expanded], dim=-1)  # [B, F_MAX, dim_in+1]
        Z = F.relu(self.proj(X))
        for layer in self.layers:
            Z = layer(Z, mask)
        return Z


# ===========================================================================
# Face Set Autoencoder (Bottleneck Bridge)
# ===========================================================================

class FaceSetAutoencoder(nn.Module):
    """
    Encodes [B, F_MAX, d] → [B, bottleneck_dim] → [B, F_MAX, d].

    pool 选项:
      'mean': masked_mean → MLP → bottleneck（排列不变，最简单）
      'attention': PMA(num_seeds=1) → squeeze → MLP → bottleneck（排列不变，信息更丰富）

    Decoder 端使用 learnable slot queries 区分各输出位置。

    variational=True: VAE 模式，编码为 mu/logvar，reparameterize 采样，
                      额外返回 kl_loss。latent 分布被正则化为 N(0,I)。
    """

    def __init__(self, dim_hidden, f_max=10,
                 bottleneck_dim=128, num_heads=4,
                 num_mlp_layers=3, mlp_factor=4,
                 pool='mean', ln=False, variational=False):
        super().__init__()
        self.f_max = f_max
        self.dim_hidden = dim_hidden
        self.pool = pool
        self.variational = variational
        mlp_hidden = mlp_factor * dim_hidden

        enc_in = dim_hidden  # pool 输出维度（mean 或 PMA squeeze 后）
        dec_out = dim_hidden

        # Encode
        self.enc_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        if pool == 'attention':
            self.pma = MaskedPMA(dim_hidden, num_heads, num_seeds=1, ln=ln)
        if variational:
            self.enc_mlp = _mlp(enc_in, bottleneck_dim * 2, mlp_hidden, num_mlp_layers)
        else:
            self.enc_mlp = _mlp(enc_in, bottleneck_dim, mlp_hidden, num_mlp_layers)

        # Decode
        self.dec_mlp = _mlp(bottleneck_dim, dec_out, mlp_hidden, num_mlp_layers)
        # Learnable slot queries: 让 decoder 的每个位置有不同初始化
        self.slot_queries = nn.Parameter(torch.randn(1, f_max, dim_hidden) * 0.02)
        self.dec_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, Z, mask=None):
        """
        Z: [B, F_MAX, d]
        mask: [B, F_MAX] bool

        Returns:
            (decoded: [B, F_MAX, d], bottleneck: [B, bottleneck_dim], kl_loss: scalar)
        """
        B = Z.size(0)
        H = self.enc_sab(Z, mask)

        # Pooling (排列不变)
        if self.pool == 'attention':
            pooled = self.pma(H, mask).squeeze(1)  # [B, 1, d] → [B, d]
        else:  # 'mean'
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                counts = m.sum(1).clamp(min=1)
                pooled = (H * m).sum(1) / counts
            else:
                pooled = H.mean(1)

        enc_out = self.enc_mlp(pooled)

        if self.variational:
            mu, logvar = enc_out.chunk(2, dim=-1)
            bottleneck = self._reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            bottleneck = enc_out
            kl_loss = torch.tensor(0.0, device=Z.device)

        # Decode: expand + slot queries 区分各位置 (不传 mask，保证训推一致)
        dec_out = self.dec_mlp(bottleneck)  # [B, dim_hidden]
        H2 = dec_out.unsqueeze(1).expand(B, self.f_max, self.dim_hidden)
        H2 = H2 + self.slot_queries.expand(B, -1, -1)  # 加 learnable 位置区分

        out = self.dec_sab(H2, None)

        return out, bottleneck, kl_loss


# ===========================================================================
# Face Set Decoder (Output Head)
# ===========================================================================

class FaceSetDecoder(nn.Module):
    """
    从 AE 解码器输出 [B, F_MAX, dim_hidden] → 重建 face 坐标。

    支持两种输出模式：
    - mode='regression': 直接输出 [B, F_MAX, 9] 坐标 (MSE loss)
    - mode='classification': 输出 [B, F_MAX, 9, num_bins] logits (CE loss)

    额外输出 existence logit: [B, F_MAX, 1] (该 slot 是否有 face)
    """

    def __init__(self, dim_hidden=128, dim_out=9, num_heads=4,
                 num_layers=2, mode='regression', num_bins=128, ln=False,
                 predict_scale=False):
        super().__init__()
        self.mode = mode
        self.dim_out = dim_out
        self.num_bins = num_bins
        self.predict_scale = predict_scale

        # Transformer decoder layers (self-attention refinement)
        self.layers = nn.ModuleList([
            MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
            for _ in range(num_layers)
        ])

        # Output projection
        if mode == 'regression':
            self.out_proj = nn.Linear(dim_hidden, dim_out)
        elif mode == 'classification':
            self.out_proj = nn.Linear(dim_hidden, dim_out * num_bins)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Existence head
        self.exist_proj = nn.Linear(dim_hidden, 1)

        # Scale head: 预测 log_scale，通过 exp 输出正实数
        # 网络输出 log_scale ∈ (-∞, +∞)，scale = exp(log_scale) ∈ (0, +∞)
        # 对于 scale 范围 [0.02, 240]，log_scale 范围 [-3.9, 5.5]，非常合理
        if predict_scale:
            self.scale_proj = nn.Linear(dim_hidden, 1)

    def forward(self, Z, mask=None):
        """
        Z: [B, F_MAX, dim_hidden]
        mask: [B, F_MAX] bool

        Returns:
            coords: [B, F_MAX, 9] (regression) or [B, F_MAX, 9, num_bins] (classification)
            existence: [B, F_MAX] logits (sigmoid → probability)
            scale: [B, 1] 正实数（仅 predict_scale=True），否则 None
        """
        for layer in self.layers:
            Z = layer(Z, mask)

        # Coordinates
        raw = self.out_proj(Z)  # [B, F_MAX, 9] or [B, F_MAX, 9*num_bins]
        if self.mode == 'classification':
            B_s, F_s, _ = raw.shape
            coords = raw.reshape(B_s, F_s, self.dim_out, self.num_bins)
        else:
            coords = raw

        # Existence
        existence = self.exist_proj(Z).squeeze(-1)  # [B, F_MAX]

        # Scale: 预测 log_scale，exp 输出正数
        if self.predict_scale:
            if mask is not None:
                m = mask.unsqueeze(-1).float()  # [B, F_MAX, 1]
                pooled = (Z * m).sum(1) / m.sum(1).clamp(min=1)  # [B, dim_hidden]
            else:
                pooled = Z.mean(1)
            log_scale = self.scale_proj(pooled)  # [B, 1]
            scale = torch.exp(log_scale)  # [B, 1]，保证正数
        else:
            scale = None

        return coords, existence, scale


# ===========================================================================
# Full Model: Face Set Transformer AE
# ===========================================================================

class FaceSetTransformerAE(nn.Module):
    """
    完整的 Face Set Transformer AutoEncoder。

    Pipeline:
      Input [B, F_MAX, dim_in]
        → FaceSetEncoder → [B, F_MAX, dim_hidden]
        → FaceSetAutoencoder → bottleneck [B, bottleneck_dim]
                             → decoded [B, F_MAX, dim_hidden]
        → FaceSetDecoder → coords [B, F_MAX, 9(, num_bins)]
                         → existence [B, F_MAX]
    """

    def __init__(self,
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
                 variational=False,
                 predict_scale=False):
        super().__init__()
        self.f_max = f_max
        self.dim_hidden = dim_hidden
        self.bottleneck_dim = bottleneck_dim
        self.decode_mode = decode_mode
        self.variational = variational
        self.predict_scale = predict_scale

        # 1. Set Encoder
        self.encoder = FaceSetEncoder(
            dim_in=dim_in, dim_hidden=dim_hidden,
            num_heads=num_heads, num_layers=num_encoder_layers, ln=ln,
            predict_scale=predict_scale)

        # 2. AE Bridge (bottleneck)
        self.ae = FaceSetAutoencoder(
            dim_hidden=dim_hidden, f_max=f_max,
            bottleneck_dim=bottleneck_dim,
            num_heads=ae_num_heads,
            num_mlp_layers=ae_num_mlp_layers,
            mlp_factor=ae_mlp_factor,
            pool=ae_pool, ln=ln,
            variational=variational)

        # 3. Decoder (output head)
        self.decoder = FaceSetDecoder(
            dim_hidden=dim_hidden, dim_out=9,
            num_heads=num_heads, num_layers=num_decoder_layers,
            mode=decode_mode, num_bins=num_bins, ln=ln,
            predict_scale=predict_scale)

    def forward(self, faces, mask=None, scale=None):
        """
        faces: [B, F_MAX, dim_in] — normalized 局部坐标（[-0.5, 0.5] 范围）
        mask:  [B, F_MAX] bool — True=有效 face, False=padding
        scale: [B, 1] — 缩放因子（predict_scale=True 时作为 encoder 输入）

        Returns:
            coords: [B, F_MAX, 9] or [B, F_MAX, 9, num_bins]
            existence: [B, F_MAX] logits
            bottleneck: [B, bottleneck_dim]
            kl_loss: scalar (仅 variational=True 时有意义，否则为 0)
            pred_scale: [B, 1] 正实数（仅 predict_scale=True），否则 None
        """
        # Encode (scale 作为额外输入)
        Z = self.encoder(faces, mask, scale=scale)  # [B, F_MAX, dim_hidden]

        # Bottleneck (统一返回 3 个值)
        # 注意: AE decode 和 FaceSetDecoder 均不传 mask，保证训推一致
        Z_dec, bottleneck, kl_loss = self.ae(Z, mask)

        # Decode (不传 mask，训推行为一致)
        coords, existence, pred_scale = self.decoder(Z_dec, None)

        return coords, existence, bottleneck, kl_loss, pred_scale

    def encode(self, faces, mask=None, scale=None):
        """只编码，返回 bottleneck embedding。用于预处理阶段。
        VAE 模式下返回 mu（确定性），不做 reparameterize。"""
        Z = self.encoder(faces, mask, scale=scale)
        if self.variational:
            B = Z.size(0)
            H = self.ae.enc_sab(Z, mask)
            # 复用 AE 的 pooling 逻辑
            if self.ae.pool == 'attention':
                pooled = self.ae.pma(H, mask).squeeze(1)
            else:
                if mask is not None:
                    m = mask.unsqueeze(-1).float()
                    counts = m.sum(1).clamp(min=1)
                    pooled = (H * m).sum(1) / counts
                else:
                    pooled = H.mean(1)
            enc_out = self.ae.enc_mlp(pooled)
            mu, _ = enc_out.chunk(2, dim=-1)
            return mu
        else:
            _, bottleneck, _ = self.ae(Z, mask)
            return bottleneck

    def decode_from_bottleneck(self, bottleneck):
        """从 bottleneck 解码。用于采样后的重建。
        不接受 mask 参数，保证训推一致（decoder 始终不用 mask）。
        Returns:
            coords: [B, F_MAX, 9]
            existence: [B, F_MAX]
            scale: [B, 1] or None
        """
        B = bottleneck.shape[0]

        # Decode through AE decoder side (含 slot_queries, 无 mask)
        dec_out = self.ae.dec_mlp(bottleneck)
        H2 = dec_out.unsqueeze(1).expand(B, self.f_max, self.dim_hidden)
        H2 = H2 + self.ae.slot_queries.expand(B, -1, -1)
        Z_dec = self.ae.dec_sab(H2, None)

        # Output head (无 mask)
        coords, existence, scale = self.decoder(Z_dec, None)
        return coords, existence, scale


# ===========================================================================
# Loss Functions
# ===========================================================================

def face_reconstruction_loss(pred_coords, target_coords, mask,
                             mode='regression', num_bins=128):
    """
    计算 face 重建损失。

    Args:
        pred_coords: [B, F_MAX, 9] (regression) or [B, F_MAX, 9, num_bins] (classification)
        target_coords: [B, F_MAX, 9] — 归一化的局部坐标
        mask: [B, F_MAX] bool — 有效 face
        mode: 'regression' or 'classification'
        num_bins: 分类模式的 bin 数量

    Returns:
        loss: scalar
    """
    if mode == 'regression':
        # MSE only on valid faces
        diff = (pred_coords - target_coords) ** 2  # [B, F_MAX, 9]
        diff = diff * mask.unsqueeze(-1).float()
        n_valid = mask.float().sum().clamp(min=1)
        loss = diff.sum() / (n_valid * 9)

    elif mode == 'classification':
        # Quantize target to bins
        # target_coords 范围 [-0.5, 0.5]（归一化后）
        # 映射到 [0, num_bins): (coord + 0.5) * num_bins，clamp 到有效范围
        target_bins = ((target_coords + 0.5) * num_bins).long()
        target_bins = target_bins.clamp(0, num_bins - 1)  # [B, F_MAX, 9]

        # pred_coords: [B, F_MAX, 9, num_bins]
        B_sz, F_sz, D_sz, C_sz = pred_coords.shape
        pred_flat = pred_coords.reshape(-1, C_sz)  # [B*F*9, num_bins]
        target_flat = target_bins.reshape(-1)  # [B*F*9]
        mask_flat = mask.unsqueeze(-1).expand(B_sz, F_sz, D_sz).reshape(-1)  # [B*F*9]

        ce = F.cross_entropy(pred_flat, target_flat, reduction='none')  # [B*F*9]
        ce = ce * mask_flat.float()
        loss = ce.sum() / mask_flat.float().sum().clamp(min=1)

    return loss


def existence_loss(pred_existence, mask):
    """
    BCE loss for face existence prediction.

    Args:
        pred_existence: [B, F_MAX] logits
        mask: [B, F_MAX] bool — True=face exists
    """
    target = mask.float()
    return F.binary_cross_entropy_with_logits(pred_existence, target)


def face_set_transformer_loss(pred_coords, pred_existence, target_coords, mask,
                              mode='regression', num_bins=128,
                              exist_weight=0.0,
                              pred_scale=None, target_scale=None,
                              scale_weight=1.0):
    """
    Combined loss = reconstruction + existence + scale.

    exist_weight: 默认 0.0，不计入 existence loss。可设为正数启用。

    Args:
        pred_scale: [B, 1] 预测的 scale（predict_scale=True 时）
        target_scale: [B, 1] 真实的 scale
        scale_weight: scale loss 的权重
    """
    recon = face_reconstruction_loss(pred_coords, target_coords, mask, mode, num_bins)
    total = recon
    loss_dict = {'recon': recon.item()}

    if exist_weight > 0:
        exist = existence_loss(pred_existence, mask)
        total = total + exist_weight * exist
        loss_dict['exist'] = exist.item()

    if pred_scale is not None and target_scale is not None:
        # L1 loss on log-scale (对 scale 的相对误差更合理)
        scale_loss = F.l1_loss(torch.log(pred_scale + 1e-6),
                               torch.log(target_scale + 1e-6))
        total = total + scale_weight * scale_loss
        loss_dict['scale'] = scale_loss.item()

    return total, loss_dict


# ===========================================================================
# Model Configs
# ===========================================================================

def FaceSetTransformerAE_Small(**kwargs):
    """Small: dim_hidden=64, bottleneck=64"""
    defaults = dict(dim_hidden=64, num_heads=4, bottleneck_dim=64,
                    num_encoder_layers=2, num_decoder_layers=2)
    defaults.update(kwargs)
    return FaceSetTransformerAE(**defaults)


def FaceSetTransformerAE_Base(**kwargs):
    """Base: dim_hidden=128, bottleneck=128"""
    defaults = dict(dim_hidden=128, num_heads=4, bottleneck_dim=128,
                    num_encoder_layers=2, num_decoder_layers=2)
    defaults.update(kwargs)
    return FaceSetTransformerAE(**defaults)


def FaceSetTransformerAE_Large(**kwargs):
    """Large: dim_hidden=256, bottleneck=256"""
    defaults = dict(dim_hidden=256, num_heads=8, bottleneck_dim=256,
                    num_encoder_layers=3, num_decoder_layers=3)
    defaults.update(kwargs)
    return FaceSetTransformerAE(**defaults)
