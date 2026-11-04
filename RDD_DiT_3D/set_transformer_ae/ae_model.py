"""
AE-Bridge EMD Reconstruction Experiment.

Architecture:
  SetEncoder  (same as train_emd.py: Linear + N×MaskedSAB)
      ↓  [B, K_MAX, d]
  SetAutoencoder  ← NEW bottleneck bridge
      Encoder side  (--ae_pool concat, default):
        MaskedSAB  →  flatten [B, K_MAX*d]  →  MLP  →  [B, bottleneck_dim]
      Encoder side  (--ae_pool mean):
        MaskedSAB  →  masked-mean [B, d]    →  MLP  →  [B, bottleneck_dim]
      Decoder side  (concat):
        MLP  →  [B, K_MAX*d]  →  reshape [B, K_MAX, d]  →  MaskedSAB  →  [B, K_MAX, d]
      Decoder side  (mean):
        MLP  →  [B, d]  →  expand [B, K_MAX, d]  →  MaskedSAB  →  [B, K_MAX, d]
      ↓  [B, K_MAX, d]
  Transformer Decoder  (same as train_emd.py: autoregressive, cross-attends to K_MAX tokens)

Tunable autoencoder hyper-parameters (see argparse at the bottom):
  --bottleneck_dim   : dimensionality of the compressed embedding  (default 256)
  --ae_num_heads     : number of attention heads inside the AE     (default 4)
  --ae_num_mlp_layers: depth of encode/decode MLPs (≥2)           (default 3)
  --ae_mlp_factor    : hidden width = ae_mlp_factor × dim_hidden   (default 4)
  --ae_pool          : pooling before MLP — 'concat' (flatten) or 'mean' (default concat)
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance

try:
    from .masked_attention import MaskedMAB, MaskedSAB, MaskedPMA   # noqa: F401
except ImportError:
    from masked_attention import MaskedMAB, MaskedSAB, MaskedPMA   # noqa: F401

# ---------------------------------------------------------------------------
# Constants  (identical to train_emd.py)
# ---------------------------------------------------------------------------

K_MAX  = 10
DIM_IN = 4   # [x, y, z, k/K_MAX]


# ---------------------------------------------------------------------------
# Encoder  (identical to train_emd.py SetEncoder)
# ---------------------------------------------------------------------------

class SetEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, num_heads: int,
                 num_layers: int = 2, ln: bool = False):
        super().__init__()
        self.proj   = nn.Linear(dim_in, dim_hidden)
        self.layers = nn.ModuleList([
            MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
            for _ in range(num_layers)
        ])

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """X: [B, K_MAX, dim_in]  →  [B, K_MAX, dim_hidden]"""
        Z = F.relu(self.proj(X))
        for layer in self.layers:
            Z = layer(Z, mask)
        return Z


# ---------------------------------------------------------------------------
# Helper: build a ReLU MLP
# ---------------------------------------------------------------------------

def _mlp(dim_in: int, dim_out: int, dim_hidden: int, num_layers: int) -> nn.Sequential:
    assert num_layers >= 2, "MLP must have at least 2 layers"
    layers: list[nn.Module] = []
    for i in range(num_layers):
        in_d  = dim_in    if i == 0             else dim_hidden
        out_d = dim_out   if i == num_layers - 1 else dim_hidden
        layers.append(nn.Linear(in_d, out_d))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# SetAutoencoder  (the new bridge module)
# ---------------------------------------------------------------------------

class SetAutoencoder(nn.Module):
    """
    Encodes [B, K_MAX, d] → [B, bottleneck_dim] → [B, K_MAX, d].

    pool='concat'  (default):
      Encode: MaskedSAB → flatten [B, K_MAX*d] → MLP → [B, bottleneck_dim]
      Decode: MLP → [B, K_MAX*d] → reshape [B, K_MAX, d] → MaskedSAB

    pool='mean':
      Encode: MaskedSAB → masked-mean [B, d] → MLP → [B, bottleneck_dim]
      Decode: MLP → [B, d] → expand [B, K_MAX, d] → MaskedSAB
    """

    def __init__(self, dim_hidden: int, k_max: int,
                 bottleneck_dim: int = 256,
                 num_heads: int = 4,
                 num_mlp_layers: int = 3,
                 mlp_factor: int = 4,
                 pool: str = 'concat',
                 ln: bool = False):
        super().__init__()
        assert pool in ('concat', 'mean'), f"pool must be 'concat' or 'mean', got {pool!r}"
        self.k_max      = k_max
        self.dim_hidden = dim_hidden
        self.pool       = pool
        mlp_hidden      = mlp_factor * dim_hidden

        # Input/output dim of the MLPs depends on pooling mode
        enc_in  = k_max * dim_hidden if pool == 'concat' else dim_hidden
        dec_out = k_max * dim_hidden if pool == 'concat' else dim_hidden

        # ---- Encode side ----
        self.enc_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.enc_mlp = _mlp(enc_in, bottleneck_dim, mlp_hidden, num_mlp_layers)

        # ---- Decode side ----
        self.dec_mlp = _mlp(bottleneck_dim, dec_out, mlp_hidden, num_mlp_layers)
        self.dec_sab = MaskedSAB(dim_hidden, dim_hidden, num_heads, ln=ln)

    def forward(self, Z: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Z    : [B, K_MAX, d]
        mask : [B, K_MAX] bool  (True = real element)

        Returns
        -------
        out        : [B, K_MAX, d]
        bottleneck : [B, bottleneck_dim]
        """
        B = Z.size(0)
        H = self.enc_sab(Z, mask)                           # [B, K_MAX, d]

        # --- Pool ---
        if self.pool == 'concat':
            pooled = H.reshape(B, -1)                       # [B, K_MAX * d]
        else:
            # Masked mean: average only over real (non-padded) positions
            if mask is not None:
                m      = mask.unsqueeze(-1).float()         # [B, K_MAX, 1]
                counts = m.sum(1).clamp(min=1)              # [B, 1]
                pooled = (H * m).sum(1) / counts            # [B, d]
            else:
                pooled = H.mean(1)                          # [B, d]

        bottleneck = self.enc_mlp(pooled)                   # [B, bottleneck_dim]

        # --- Decode ---
        dec_out = self.dec_mlp(bottleneck)                  # [B, K_MAX*d] or [B, d]
        if self.pool == 'concat':
            H2 = dec_out.reshape(B, self.k_max, self.dim_hidden)  # [B, K_MAX, d]
        else:
            H2 = dec_out.unsqueeze(1).expand(B, self.k_max, self.dim_hidden)  # [B, K_MAX, d]

        out = self.dec_sab(H2, mask)                        # [B, K_MAX, d]
        return out, bottleneck


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AEBridgeReconModel(nn.Module):
    """
    SetEncoder  →  SetAutoencoder  →  Transformer Decoder.

    The decoder is identical to train_emd.py: autoregressive, cross-attends
    to the K_MAX per-token memory produced by the AE decoder side.
    """

    def __init__(self, dim_hidden: int = 64, num_heads: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 bottleneck_dim: int = 256,
                 ae_num_heads: int = 4, ae_num_mlp_layers: int = 3,
                 ae_mlp_factor: int = 4, ae_pool: str = 'concat',
                 k_max: int = K_MAX, ln: bool = False):
        super().__init__()
        self.k_max      = k_max
        self.dim_hidden = dim_hidden

        # ---- 1. Set Encoder ----
        self.encoder = SetEncoder(
            DIM_IN, dim_hidden, num_heads, num_encoder_layers, ln=ln)

        # ---- 2. AE Bridge ----
        self.ae = SetAutoencoder(
            dim_hidden=dim_hidden,
            k_max=k_max,
            bottleneck_dim=bottleneck_dim,
            num_heads=ae_num_heads,
            num_mlp_layers=ae_num_mlp_layers,
            mlp_factor=ae_mlp_factor,
            pool=ae_pool,
            ln=ln,
        )

        # ---- 3. Transformer Decoder ----
        self.bos       = nn.Parameter(torch.randn(dim_hidden))
        self.tgt_proj  = nn.Linear(3, dim_hidden)   # embed a 3D point token
        self.pos_embed = nn.Embedding(k_max + 1, dim_hidden)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        self.value_head = nn.Linear(dim_hidden, 3)   # predict 3D point
        self.end_head   = nn.Linear(dim_hidden, 1)

    # ------------------------------------------------------------------

    def _encode(self, X: torch.Tensor, mask: torch.Tensor):
        """Run encoder + AE, return memory [B, K_MAX, d] for the transformer decoder."""
        Z      = self.encoder(X, mask)           # [B, K_MAX, d]
        mem, _ = self.ae(Z, mask)                # [B, K_MAX, d]
        return mem

    def _tgt_sequence(self, B: int, device: torch.device,
                      prev_values: torch.Tensor) -> torch.Tensor:
        T       = prev_values.size(1)
        bos_emb = self.bos.view(1, 1, -1).expand(B, 1, -1)
        val_emb = self.tgt_proj(prev_values)
        tgt     = torch.cat([bos_emb, val_emb], dim=1)
        pos     = torch.arange(T + 1, device=device)
        return tgt + self.pos_embed(pos).unsqueeze(0)

    # ------------------------------------------------------------------

    def forward_train(self, X: torch.Tensor, mask: torch.Tensor,
                      sorted_values: torch.Tensor):
        B, device = X.size(0), X.device
        memory   = self._encode(X, mask)          # [B, K_MAX, d]
        mem_mask = ~mask                           # True = padding (PyTorch convention)

        prev_vals   = sorted_values                               # [B, K_MAX, 3]
        tgt         = self._tgt_sequence(B, device, prev_vals)   # [B, K_MAX+1, d]
        T           = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=mem_mask,
        )[:, :self.k_max, :]                      # [B, K_MAX, d]

        pred_values = torch.sigmoid(self.value_head(out))
        pred_ends   = torch.sigmoid(self.end_head(out))
        return pred_values, pred_ends

    @torch.no_grad()
    def forward_inference(self, X: torch.Tensor, mask: torch.Tensor):
        B, device = X.size(0), X.device
        memory   = self._encode(X, mask)
        mem_mask = ~mask

        pos0 = torch.zeros(1, dtype=torch.long, device=device)
        tgt  = (self.bos.view(1, 1, -1).expand(B, 1, -1).clone()
                + self.pos_embed(pos0).unsqueeze(0))

        all_values, all_ends = [], []
        for t in range(self.k_max):
            T_cur       = tgt.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T_cur, device=device)
            out  = self.transformer_decoder(
                tgt=tgt, memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=mem_mask,
            )
            last = out[:, -1:, :]
            val  = torch.sigmoid(self.value_head(last))  # [B, 1, 3]
            end  = torch.sigmoid(self.end_head(last))    # [B, 1, 1]
            all_values.append(val)                       # keep [B, 1, 3]
            all_ends.append(end.squeeze(-1))             # [B, 1]

            pos_t    = torch.tensor([t + 1], dtype=torch.long, device=device)
            next_emb = (self.tgt_proj(val)               # [B, 1, 3] → [B, 1, d]
                        + self.pos_embed(pos_t).unsqueeze(0))
            tgt = torch.cat([tgt, next_emb], dim=1)

        # [B, K_MAX, 3],  [B, K_MAX]
        return torch.cat(all_values, dim=1), torch.cat(all_ends, dim=1)


# ---------------------------------------------------------------------------
# Data generation, loss, EMD  (identical to train_emd.py)
# ---------------------------------------------------------------------------

def generate_batch(batch_size: int, device: torch.device):
    ks_np         = np.random.randint(1, K_MAX + 1, size=batch_size)
    X             = torch.zeros(batch_size, K_MAX, DIM_IN)   # [B, K_MAX, 4]
    mask          = torch.zeros(batch_size, K_MAX, dtype=torch.bool)
    sorted_values = torch.zeros(batch_size, K_MAX, 3)        # [B, K_MAX, 3]
    for i, k in enumerate(ks_np):
        pts   = torch.rand(k, 3)
        # Canonical lexicographic sort for deterministic ordering
        order = sorted(range(k), key=lambda r: tuple(pts[r].tolist()))
        pts   = pts[order]
        X[i, :k, :3] = pts
        X[i, :k,  3] = k / K_MAX
        mask[i, :k]  = True
        sorted_values[i, :k] = pts          # already in canonical order
    ks = torch.tensor(ks_np, dtype=torch.long, device=device)
    return X.to(device), mask.to(device), sorted_values.to(device), ks


def compute_loss(pred_values, pred_ends, sorted_values, ks):
    B, T, _ = pred_values.shape
    device  = pred_values.device
    positions  = torch.arange(T, device=device).unsqueeze(0)
    token_mask = positions < ks.unsqueeze(1)
    end_target = (positions == (ks.unsqueeze(1) - 1)).float()
    value_loss = F.mse_loss(
        pred_values[token_mask], sorted_values[token_mask])   # [n, 3] vs [n, 3]
    end_loss = F.binary_cross_entropy(
        pred_ends.squeeze(-1)[token_mask], end_target[token_mask])
    return value_loss + end_loss


def compute_emd_batch(pred_values, pred_ends, sorted_values, ks):
    # pred_values: [B, K_MAX, 3],  pred_ends: [B, K_MAX],  sorted_values: [B, K_MAX, 3]
    pv  = pred_values.cpu().numpy()
    pe  = pred_ends.cpu().numpy()
    sv  = sorted_values.cpu().numpy()
    ks_ = ks.cpu().numpy()
    emds = []
    for i, k in enumerate(ks_):
        gen = []
        for t in range(K_MAX):
            gen.append(pv[i, t])          # 3D point
            if pe[i, t] > 0.5:
                break
        gen      = np.array(gen)          # [n_gen, 3]
        true_pts = sv[i, :k]              # [k, 3]
        # Mean per-coordinate Wasserstein as 3D EMD proxy
        emd = np.mean([wasserstein_distance(gen[:, j], true_pts[:, j])
                       for j in range(3)])
        emds.append(emd)
    return float(np.mean(emds))


def evaluate(model, num_batches, batch_size, device):
    model.eval()
    scores = []
    for _ in range(num_batches):
        X, mask, sv, ks = generate_batch(batch_size, device)
        pv, pe = model.forward_inference(X, mask)
        scores.append(compute_emd_batch(pv, pe, sv, ks))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, model, optimizer, scheduler, args):
    torch.save({
        'step':      step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args':      vars(args),
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
    global K_MAX, DIM_IN
    K_MAX = args.k_max
    DIM_IN = 4  # [x, y, z, k/K_MAX]

    device = torch.device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = AEBridgeReconModel(
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        bottleneck_dim=args.bottleneck_dim,
        ae_num_heads=args.ae_num_heads,
        ae_num_mlp_layers=args.ae_num_mlp_layers,
        ae_mlp_factor=args.ae_mlp_factor,
        ae_pool=args.ae_pool,
        k_max=K_MAX,
        ln=args.ln,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters())
    flat_dim  = K_MAX * args.dim_hidden
    pool_desc = (f"flatten [B,{flat_dim}]" if args.ae_pool == 'concat'
                 else f"masked-mean [B,{args.dim_hidden}]")
    print(f"Model parameters: {n_params:,}")
    print(f"  Encoder output  : [B, {K_MAX}, {args.dim_hidden}]")
    print(f"  AE pool         : {args.ae_pool}  ({pool_desc}  →  bottleneck {args.bottleneck_dim})")
    print(f"  Decoder memory  : [B, {K_MAX}, {args.dim_hidden}]")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=1e-6)

    # ---- Resume ----
    start_step  = 1
    latest_ckpt = os.path.join(args.ckpt_dir, 'latest.pt')
    if os.path.exists(latest_ckpt):
        try:
            start_step = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, device) + 1
            print(f"Resumed from checkpoint — continuing at step {start_step}")
        except Exception as e:
            print(f"Checkpoint incompatible ({e}), starting from scratch")
    else:
        print("No checkpoint found — starting from scratch")

    # ---- Metrics CSV ----
    write_header = (not os.path.exists(args.metrics_file)) or (start_step == 1)
    metrics_fh   = open(args.metrics_file, 'a', buffering=1)
    if write_header:
        metrics_fh.write('step,loss,val_emd\n')
    print(f"Metrics  → {args.metrics_file}")
    print(f"Ckpts    → {args.ckpt_dir}/")

    tick = time.time()
    for step in range(start_step, args.num_steps + 1):

        model.train()
        X, mask, sv, ks = generate_batch(args.batch_size, device)
        pv, pe = model.forward_train(X, mask, sv)
        loss   = compute_loss(pv, pe, sv, ks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---- Every 100 steps: log + save rolling checkpoint ----
        if step % 100 == 0:
            emd     = evaluate(model, num_batches=5,
                               batch_size=args.batch_size, device=device)
            elapsed = time.time() - tick
            cur_lr  = scheduler.get_last_lr()[0]
            print(f"step {step:>6d} | loss={loss.item():.6f} "
                  f"| val_emd={emd:.6f} | lr={cur_lr:.2e} | {elapsed:.1f}s")
            metrics_fh.write(f'{step},{loss.item():.6f},{emd:.6f}\n')
            tick = time.time()
            save_checkpoint(latest_ckpt, step, model, optimizer, scheduler, args)

        # ---- Every perm_save_freq steps: permanent checkpoint ----
        if step % args.perm_save_freq == 0:
            perm = os.path.join(args.ckpt_dir, f'step_{step:07d}.pt')
            save_checkpoint(perm, step, model, optimizer, scheduler, args)
            print(f"[step {step}] Permanent checkpoint → {perm}")

    metrics_fh.close()

    # ---- Final per-k EMD ----
    print("\n=== Final EMD per set size k ===")
    model.eval()
    N_eval = 200
    for k in range(1, K_MAX + 1):
        X    = torch.zeros(N_eval, K_MAX, DIM_IN, device=device)
        mask = torch.zeros(N_eval, K_MAX, dtype=torch.bool, device=device)
        sv   = torch.zeros(N_eval, K_MAX, 3, device=device)
        for b in range(N_eval):
            pts   = torch.rand(k, 3)
            order = sorted(range(k), key=lambda r: tuple(pts[r].tolist()))
            pts   = pts[order]
            X[b, :k, :3] = pts
            X[b, :k,  3] = k / K_MAX
            sv[b, :k]    = pts
        mask[:, :k] = True
        ks_f = torch.full((N_eval,), k, dtype=torch.long, device=device)
        pv, pe = model.forward_inference(X, mask)
        emd = compute_emd_batch(pv, pe, sv, ks_f)
        print(f"  k={k:>2d}: mean EMD = {emd:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AE-Bridge: SetEncoder + SetAutoencoder + Transformer Decoder')

    # General
    parser.add_argument('--num_steps',          type=int,   default=100000)
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--lr',                 type=float, default=1e-3)
    parser.add_argument('--device',             type=str,   default='cpu')
    parser.add_argument('--num_threads',        type=int,   default=4)
    parser.add_argument('--ln',                 action='store_true')

    # Encoder / Decoder (same knobs as train_emd.py)
    parser.add_argument('--dim_hidden',         type=int,   default=64)
    parser.add_argument('--num_heads',          type=int,   default=4)
    parser.add_argument('--num_encoder_layers', type=int,   default=2)
    parser.add_argument('--num_decoder_layers', type=int,   default=2)

    # AE bridge (new)
    parser.add_argument('--bottleneck_dim',     type=int,   default=256,
                        help='Dimension of the compressed AE embedding')
    parser.add_argument('--ae_num_heads',       type=int,   default=4,
                        help='Number of attention heads inside the AE SABs')
    parser.add_argument('--ae_num_mlp_layers',  type=int,   default=3,
                        help='Depth of encode/decode MLPs inside the AE (≥2)')
    parser.add_argument('--ae_mlp_factor',      type=int,   default=4,
                        help='MLP hidden width = ae_mlp_factor × dim_hidden')
    parser.add_argument('--ae_pool',            type=str,   default='concat',
                        choices=['concat', 'mean'],
                        help='Pooling before AE encode MLP: flatten (concat) or masked-mean')

    # I/O
    parser.add_argument('--ckpt_dir',           type=str,   default='checkpoints_emd_ae')
    parser.add_argument('--perm_save_freq',     type=int,   default=100000,
                        help='Save a permanent (non-overwritten) checkpoint every N steps')
    parser.add_argument('--metrics_file',       type=str,   default='train_emd_ae_metrics.csv')
    parser.add_argument('--k_max',              type=int,   default=10,
                        help='Maximum number of points per grid cell')

    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(max(1, args.num_threads // 2))
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
    train(args)
