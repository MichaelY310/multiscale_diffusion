"""
AE-Bridge EMD Reconstruction V2 — Improved architecture with multiple pooling modes.

Improvements over v1:
  1. PMA pooling option (permutation-invariant, no need for sorting)
  2. LayerNorm in all attention blocks (stabilizes gradients)
  3. GELU activation in MLPs (smoother than ReLU)
  4. Optional KL regularization on bottleneck (for better latent geometry)
  5. Parallel (non-autoregressive) decoder option: predicts all K_MAX slots + count
  6. Deeper/residual MLP option for encode/decode
  7. Configurable bottleneck_dim (can test smaller dims like 32/64)
  8. Optional input without k/K_MAX feature (rely on mask only)

Usage:
  python train_emd_ae_3d_v2.py --config <preset_name>
  # or specify all args manually
"""

import os
import time
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance

try:
    from .masked_attention import MaskedMAB, MaskedSAB, MaskedPMA
except ImportError:
    from masked_attention import MaskedMAB, MaskedSAB, MaskedPMA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_MAX = 10


# ---------------------------------------------------------------------------
# Helper: build MLP with options
# ---------------------------------------------------------------------------

def _mlp(dim_in: int, dim_out: int, dim_hidden: int, num_layers: int,
         activation: str = 'gelu', use_layernorm: bool = False) -> nn.Sequential:
    assert num_layers >= 2
    act_fn = nn.GELU if activation == 'gelu' else nn.ReLU
    layers: list[nn.Module] = []
    for i in range(num_layers):
        in_d = dim_in if i == 0 else dim_hidden
        out_d = dim_out if i == num_layers - 1 else dim_hidden
        layers.append(nn.Linear(in_d, out_d))
        if i < num_layers - 1:
            if use_layernorm:
                layers.append(nn.LayerNorm(out_d))
            layers.append(act_fn())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SetEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, num_heads: int,
                 num_layers: int = 4, ln: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden)
        self.layers = nn.ModuleList([
            MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
            for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        Z = F.gelu(self.proj(X))
        for layer in self.layers:
            Z = layer(Z, mask)
        return Z


# ---------------------------------------------------------------------------
# SetAutoencoder with multiple pooling modes
# ---------------------------------------------------------------------------

class SetAutoencoder(nn.Module):
    """
    Pool modes:
      - 'concat': flatten all K_MAX slots (requires sorted input)
      - 'mean': masked mean pooling
      - 'pma1': PMA with 1 seed (true permutation invariant)
      - 'pma4': PMA with 4 seeds → flatten (richer, still permutation invariant)
      - 'max_mean': masked max + masked mean → concat (permutation invariant)
    """

    def __init__(self, dim_hidden: int, k_max: int,
                 bottleneck_dim: int = 64,
                 num_heads: int = 4,
                 num_mlp_layers: int = 3,
                 mlp_factor: int = 4,
                 pool: str = 'pma1',
                 ln: bool = True,
                 activation: str = 'gelu',
                 mlp_layernorm: bool = False,
                 kl_weight: float = 0.0):
        super().__init__()
        self.k_max = k_max
        self.dim_hidden = dim_hidden
        self.pool = pool
        self.kl_weight = kl_weight
        mlp_hidden = mlp_factor * dim_hidden

        # Determine pooled dimension based on pool mode
        if pool == 'concat':
            pooled_dim = k_max * dim_hidden
        elif pool == 'mean':
            pooled_dim = dim_hidden
        elif pool == 'pma1':
            pooled_dim = dim_hidden
        elif pool == 'pma4':
            pooled_dim = 4 * dim_hidden
        elif pool == 'max_mean':
            pooled_dim = 2 * dim_hidden
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

        self.pooled_dim = pooled_dim

        # Pooling modules
        self.enc_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        if pool == 'pma1':
            self.pma = MaskedPMA(dim_hidden, num_heads, num_seeds=1, ln=ln)
        elif pool == 'pma4':
            self.pma = MaskedPMA(dim_hidden, num_heads, num_seeds=4, ln=ln)

        # Encode MLP
        if kl_weight > 0:
            # VAE-style: output mu and logvar
            self.enc_mlp_mu = _mlp(pooled_dim, bottleneck_dim, mlp_hidden,
                                   num_mlp_layers, activation, mlp_layernorm)
            self.enc_mlp_logvar = _mlp(pooled_dim, bottleneck_dim, mlp_hidden,
                                       num_mlp_layers, activation, mlp_layernorm)
        else:
            self.enc_mlp = _mlp(pooled_dim, bottleneck_dim, mlp_hidden,
                                num_mlp_layers, activation, mlp_layernorm)

        # Decode MLP: bottleneck → k_max * dim_hidden (always decode to full grid)
        dec_out = k_max * dim_hidden
        self.dec_mlp = _mlp(bottleneck_dim, dec_out, mlp_hidden,
                            num_mlp_layers, activation, mlp_layernorm)
        self.dec_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)

    def _pool(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = H.size(0)
        if self.pool == 'concat':
            return H.reshape(B, -1)
        elif self.pool == 'mean':
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (H * m).sum(1) / m.sum(1).clamp(min=1)
            return H.mean(1)
        elif self.pool in ('pma1', 'pma4'):
            out = self.pma(H, mask)  # [B, num_seeds, d]
            return out.reshape(B, -1)  # [B, num_seeds*d]
        elif self.pool == 'max_mean':
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                H_masked = H * m + (~mask).unsqueeze(-1).float() * (-1e9)
                max_pool = H_masked.max(1)[0]
                mean_pool = (H * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                max_pool = H.max(1)[0]
                mean_pool = H.mean(1)
            return torch.cat([max_pool, mean_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None):
        B = Z.size(0)
        H = self.enc_sab(Z, mask)
        pooled = self._pool(H, mask)

        # Encode to bottleneck
        kl_loss = torch.tensor(0.0, device=Z.device)
        if self.kl_weight > 0:
            mu = self.enc_mlp_mu(pooled)
            logvar = self.enc_mlp_logvar(pooled)
            if self.training:
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                bottleneck = mu + eps * std
                kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
                kl_loss = self.kl_weight * kl_loss
            else:
                bottleneck = mu
        else:
            bottleneck = self.enc_mlp(pooled)

        # Decode
        dec_out = self.dec_mlp(bottleneck)
        H2 = dec_out.reshape(B, self.k_max, self.dim_hidden)
        out = self.dec_sab(H2, mask)
        return out, bottleneck, kl_loss


# ---------------------------------------------------------------------------
# Parallel Decoder (non-autoregressive)
# ---------------------------------------------------------------------------

class ParallelDecoder(nn.Module):
    """
    Non-autoregressive decoder: directly predicts K_MAX coordinate slots + presence mask.
    No need for sorting assumption on output side (use Hungarian matching or slot-based).
    """

    def __init__(self, dim_hidden: int, k_max: int, num_heads: int,
                 num_layers: int = 4, ln: bool = True):
        super().__init__()
        self.k_max = k_max
        # Learned slot queries
        self.slot_queries = nn.Parameter(torch.randn(1, k_max, dim_hidden) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim_hidden, nhead=num_heads,
                dim_feedforward=dim_hidden * 4, dropout=0.0,
                batch_first=True, activation='gelu',
            ) for _ in range(num_layers)
        ])
        self.coord_head = nn.Linear(dim_hidden, 3)
        self.presence_head = nn.Linear(dim_hidden, 1)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor = None):
        """
        memory: [B, K_MAX, d] from AE decoder
        Returns: coords [B, K_MAX, 3], presence [B, K_MAX]
        """
        B = memory.size(0)
        tgt = self.slot_queries.expand(B, -1, -1)
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=memory_mask)
        coords = torch.sigmoid(self.coord_head(tgt))       # [B, K_MAX, 3]
        presence = torch.sigmoid(self.presence_head(tgt).squeeze(-1))  # [B, K_MAX]
        return coords, presence


# ---------------------------------------------------------------------------
# Autoregressive Decoder (same as v1 but with GELU)
# ---------------------------------------------------------------------------

class AutoregressiveDecoder(nn.Module):
    def __init__(self, dim_hidden: int, k_max: int, num_heads: int,
                 num_layers: int = 4, ln: bool = True):
        super().__init__()
        self.k_max = k_max
        self.dim_hidden = dim_hidden
        self.bos = nn.Parameter(torch.randn(dim_hidden))
        self.tgt_proj = nn.Linear(3, dim_hidden)
        self.pos_embed = nn.Embedding(k_max + 1, dim_hidden)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_hidden, nhead=num_heads,
            dim_feedforward=dim_hidden * 4, dropout=0.0,
            batch_first=True, activation='gelu',
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(dim_hidden, 3)
        self.end_head = nn.Linear(dim_hidden, 1)

    def forward_train(self, memory, mem_mask, sorted_values):
        B, device = memory.size(0), memory.device
        bos_emb = self.bos.view(1, 1, -1).expand(B, 1, -1)
        val_emb = self.tgt_proj(sorted_values)
        tgt = torch.cat([bos_emb, val_emb], dim=1)
        T = tgt.size(1)
        pos = torch.arange(T, device=device)
        tgt = tgt + self.pos_embed(pos).unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        out = self.transformer_decoder(
            tgt=tgt, memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=mem_mask,
        )[:, :self.k_max, :]
        pred_values = torch.sigmoid(self.value_head(out))
        pred_ends = torch.sigmoid(self.end_head(out))
        return pred_values, pred_ends

    @torch.no_grad()
    def forward_inference(self, memory, mem_mask):
        B, device = memory.size(0), memory.device
        pos0 = torch.zeros(1, dtype=torch.long, device=device)
        tgt = (self.bos.view(1, 1, -1).expand(B, 1, -1).clone()
               + self.pos_embed(pos0).unsqueeze(0))
        all_values, all_ends = [], []
        for t in range(self.k_max):
            T_cur = tgt.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T_cur, device=device)
            out = self.transformer_decoder(
                tgt=tgt, memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_mask,
            )
            last = out[:, -1:, :]
            val = torch.sigmoid(self.value_head(last))
            end = torch.sigmoid(self.end_head(last))
            all_values.append(val)
            all_ends.append(end.squeeze(-1))
            pos_t = torch.tensor([t + 1], dtype=torch.long, device=device)
            next_emb = self.tgt_proj(val) + self.pos_embed(pos_t).unsqueeze(0)
            tgt = torch.cat([tgt, next_emb], dim=1)
        return torch.cat(all_values, dim=1), torch.cat(all_ends, dim=1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class AEBridgeReconModelV2(nn.Module):
    def __init__(self, dim_hidden: int = 32, num_heads: int = 4,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 bottleneck_dim: int = 64,
                 ae_num_heads: int = 4, ae_num_mlp_layers: int = 3,
                 ae_mlp_factor: int = 4, ae_pool: str = 'pma1',
                 decoder_type: str = 'ar',  # 'ar' or 'parallel'
                 kl_weight: float = 0.0,
                 use_k_feature: bool = True,
                 ln: bool = True,
                 activation: str = 'gelu',
                 mlp_layernorm: bool = False,
                 k_max: int = K_MAX):
        super().__init__()
        self.k_max = k_max
        self.dim_hidden = dim_hidden
        self.decoder_type = decoder_type
        self.use_k_feature = use_k_feature
        self.ae_pool = ae_pool

        dim_in = 4 if use_k_feature else 3

        # 1. Encoder
        self.encoder = SetEncoder(dim_in, dim_hidden, num_heads,
                                  num_encoder_layers, ln=ln)

        # 2. AE Bridge
        self.ae = SetAutoencoder(
            dim_hidden=dim_hidden, k_max=k_max,
            bottleneck_dim=bottleneck_dim,
            num_heads=ae_num_heads,
            num_mlp_layers=ae_num_mlp_layers,
            mlp_factor=ae_mlp_factor,
            pool=ae_pool, ln=ln,
            activation=activation,
            mlp_layernorm=mlp_layernorm,
            kl_weight=kl_weight,
        )

        # 3. Decoder
        if decoder_type == 'ar':
            self.decoder = AutoregressiveDecoder(
                dim_hidden, k_max, num_heads, num_decoder_layers, ln=ln)
        else:
            self.decoder = ParallelDecoder(
                dim_hidden, k_max, num_heads, num_decoder_layers, ln=ln)

    def encode(self, X: torch.Tensor, mask: torch.Tensor):
        Z = self.encoder(X, mask)
        _, bottleneck, _ = self.ae(Z, mask)
        return bottleneck

    def forward_train(self, X, mask, sorted_values):
        Z = self.encoder(X, mask)
        memory, bottleneck, kl_loss = self.ae(Z, mask)
        mem_mask = ~mask
        if self.decoder_type == 'ar':
            pred_values, pred_ends = self.decoder.forward_train(memory, mem_mask, sorted_values)
            return pred_values, pred_ends, kl_loss
        else:
            coords, presence = self.decoder(memory, mem_mask)
            return coords, presence, kl_loss

    @torch.no_grad()
    def forward_inference(self, X, mask):
        Z = self.encoder(X, mask)
        memory, bottleneck, _ = self.ae(Z, mask)
        mem_mask = ~mask
        if self.decoder_type == 'ar':
            return self.decoder.forward_inference(memory, mem_mask)
        else:
            coords, presence = self.decoder(memory, mem_mask)
            return coords, presence


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_batch(batch_size: int, device: torch.device, use_k_feature: bool = True,
                   sort_input: bool = True):
    dim_in = 4 if use_k_feature else 3
    ks_np = np.random.randint(1, K_MAX + 1, size=batch_size)
    X = torch.zeros(batch_size, K_MAX, dim_in)
    mask = torch.zeros(batch_size, K_MAX, dtype=torch.bool)
    sorted_values = torch.zeros(batch_size, K_MAX, 3)
    for i, k in enumerate(ks_np):
        pts = torch.rand(k, 3)
        if sort_input:
            order = sorted(range(k), key=lambda r: tuple(pts[r].tolist()))
            pts = pts[order]
        X[i, :k, :3] = pts
        if use_k_feature:
            X[i, :k, 3] = k / K_MAX
        mask[i, :k] = True
        sorted_values[i, :k] = pts
    ks = torch.tensor(ks_np, dtype=torch.long, device=device)
    return X.to(device), mask.to(device), sorted_values.to(device), ks


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_loss_ar(pred_values, pred_ends, sorted_values, ks):
    B, T, _ = pred_values.shape
    device = pred_values.device
    positions = torch.arange(T, device=device).unsqueeze(0)
    token_mask = positions < ks.unsqueeze(1)
    end_target = (positions == (ks.unsqueeze(1) - 1)).float()
    value_loss = F.mse_loss(pred_values[token_mask], sorted_values[token_mask])
    end_loss = F.binary_cross_entropy(
        pred_ends.squeeze(-1)[token_mask], end_target[token_mask])
    return value_loss + end_loss


def compute_loss_parallel(coords, presence, sorted_values, ks, use_hungarian=False):
    """
    Simple slot-based loss (no Hungarian matching needed if input is sorted 
    and slots correspond to positions).
    """
    B, T, _ = coords.shape
    device = coords.device
    positions = torch.arange(T, device=device).unsqueeze(0)
    token_mask = positions < ks.unsqueeze(1)
    # Coordinate loss (only for present slots)
    coord_loss = F.mse_loss(coords[token_mask], sorted_values[token_mask])
    # Presence loss
    presence_target = token_mask.float()
    presence_loss = F.binary_cross_entropy(presence, presence_target)
    return coord_loss + presence_loss


# ---------------------------------------------------------------------------
# EMD evaluation
# ---------------------------------------------------------------------------

def compute_emd_batch_ar(pred_values, pred_ends, sorted_values, ks):
    pv = pred_values.cpu().numpy()
    pe = pred_ends.cpu().numpy()
    sv = sorted_values.cpu().numpy()
    ks_ = ks.cpu().numpy()
    emds = []
    for i, k in enumerate(ks_):
        gen = []
        for t in range(K_MAX):
            gen.append(pv[i, t])
            if pe[i, t] > 0.5:
                break
        gen = np.array(gen)
        true_pts = sv[i, :k]
        emd = np.mean([wasserstein_distance(gen[:, j], true_pts[:, j]) for j in range(3)])
        emds.append(emd)
    return float(np.mean(emds))


def compute_emd_batch_parallel(coords, presence, sorted_values, ks):
    c = coords.cpu().numpy()
    p = presence.cpu().numpy()
    sv = sorted_values.cpu().numpy()
    ks_ = ks.cpu().numpy()
    emds = []
    for i, k in enumerate(ks_):
        # Take slots with highest presence scores
        n_pred = min(int((p[i] > 0.5).sum()), K_MAX)
        if n_pred == 0:
            n_pred = 1
        # Sort by presence descending, take top n_pred
        idx = np.argsort(-p[i])[:n_pred]
        gen = c[i, idx]
        true_pts = sv[i, :k]
        emd = np.mean([wasserstein_distance(gen[:, j], true_pts[:, j]) for j in range(3)])
        emds.append(emd)
    return float(np.mean(emds))


def evaluate(model, num_batches, batch_size, device, use_k_feature, sort_input):
    model.eval()
    scores = []
    for _ in range(num_batches):
        X, mask, sv, ks = generate_batch(batch_size, device, use_k_feature, sort_input)
        if model.decoder_type == 'ar':
            pv, pe = model.forward_inference(X, mask)
            scores.append(compute_emd_batch_ar(pv, pe, sv, ks))
        else:
            coords, presence = model.forward_inference(X, mask)
            scores.append(compute_emd_batch_parallel(coords, presence, sv, ks))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, model, optimizer, scheduler, args):
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return int(ckpt['step'])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Need sorting for concat pool or AR decoder
    sort_input = (args.ae_pool == 'concat') or (args.decoder_type == 'ar')

    model = AEBridgeReconModelV2(
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        bottleneck_dim=args.bottleneck_dim,
        ae_num_heads=args.ae_num_heads,
        ae_num_mlp_layers=args.ae_num_mlp_layers,
        ae_mlp_factor=args.ae_mlp_factor,
        ae_pool=args.ae_pool,
        decoder_type=args.decoder_type,
        kl_weight=args.kl_weight,
        use_k_feature=args.use_k_feature,
        ln=True,
        activation='gelu',
        mlp_layernorm=args.mlp_layernorm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"  Pool: {args.ae_pool}, Decoder: {args.decoder_type}")
    print(f"  Bottleneck: {args.bottleneck_dim}, dim_hidden: {args.dim_hidden}")
    print(f"  KL weight: {args.kl_weight}, use_k_feature: {args.use_k_feature}")
    print(f"  Sort input: {sort_input}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=1e-6)

    # Resume
    start_step = 1
    latest_ckpt = os.path.join(args.ckpt_dir, 'latest.pt')
    if os.path.exists(latest_ckpt):
        try:
            start_step = load_checkpoint(latest_ckpt, model, optimizer, scheduler, device) + 1
            print(f"Resumed from step {start_step}")
        except Exception as e:
            print(f"Checkpoint incompatible ({e}), starting fresh")

    # Metrics
    write_header = (not os.path.exists(args.metrics_file)) or (start_step == 1)
    metrics_fh = open(args.metrics_file, 'a', buffering=1)
    if write_header:
        metrics_fh.write('step,loss,val_emd\n')

    tick = time.time()
    for step in range(start_step, args.num_steps + 1):
        model.train()
        X, mask, sv, ks = generate_batch(args.batch_size, device,
                                          args.use_k_feature, sort_input)

        if args.decoder_type == 'ar':
            pv, pe, kl_loss = model.forward_train(X, mask, sv)
            loss = compute_loss_ar(pv, pe, sv, ks) + kl_loss
        else:
            coords, presence, kl_loss = model.forward_train(X, mask, sv)
            loss = compute_loss_parallel(coords, presence, sv, ks) + kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            train_elapsed = time.time() - tick
            eval_tick = time.time()
            emd = evaluate(model, num_batches=5, batch_size=args.batch_size,
                           device=device, use_k_feature=args.use_k_feature,
                           sort_input=sort_input)
            eval_elapsed = time.time() - eval_tick
            cur_lr = scheduler.get_last_lr()[0]
            print(f"step {step:>7d} | loss={loss.item():.6f} "
                  f"| val_emd={emd:.6f} | lr={cur_lr:.2e} "
                  f"| train={train_elapsed:.2f}s | eval={eval_elapsed:.2f}s")
            metrics_fh.write(f'{step},{loss.item():.6f},{emd:.6f}\n')
            tick = time.time()
            save_checkpoint(latest_ckpt, step, model, optimizer, scheduler, args)

        if step % args.perm_save_freq == 0:
            perm = os.path.join(args.ckpt_dir, f'step_{step:07d}.pt')
            save_checkpoint(perm, step, model, optimizer, scheduler, args)

    metrics_fh.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AE-Bridge V2: Improved Set Transformer')

    # General
    parser.add_argument('--num_steps', type=int, default=2000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_threads', type=int, default=4)

    # Architecture
    parser.add_argument('--dim_hidden', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--bottleneck_dim', type=int, default=64)
    parser.add_argument('--ae_num_heads', type=int, default=4)
    parser.add_argument('--ae_num_mlp_layers', type=int, default=3)
    parser.add_argument('--ae_mlp_factor', type=int, default=4)
    parser.add_argument('--ae_pool', type=str, default='pma1',
                        choices=['concat', 'mean', 'pma1', 'pma4', 'max_mean'])
    parser.add_argument('--decoder_type', type=str, default='ar',
                        choices=['ar', 'parallel'])
    parser.add_argument('--kl_weight', type=float, default=0.0)
    parser.add_argument('--use_k_feature', type=int, default=1,
                        help='1=use k/K_MAX as 4th input dim, 0=only xyz')
    parser.add_argument('--mlp_layernorm', action='store_true')

    # I/O
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--perm_save_freq', type=int, default=500000)
    parser.add_argument('--metrics_file', type=str, default='metrics_v2.csv')

    args = parser.parse_args()
    args.use_k_feature = bool(args.use_k_feature)
    torch.set_num_threads(args.num_threads)
    train(args)
