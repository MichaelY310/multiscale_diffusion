"""
Masked attention modules for Set Transformer.

Provides MaskedMAB, MaskedSAB, MaskedPMA with key-padding mask support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


K_MAX  = 10  # maximum / padding length
DIM_IN = 3   # input element dimensionality ([0,1]^DIM_IN)


class MaskedMAB(nn.Module):
    """
    Multihead Attention Block with optional key-padding mask.

    mask : BoolTensor [B, N_k]  —  True = real token, False = padding.
    Padded key positions are filled with -inf before softmax so they
    contribute zero weight and do not pollute the output.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)   # [B*H, Nq, d]
        K_ = torch.cat(K.split(dim_split, 2), 0)   # [B*H, Nk, d]
        V_ = torch.cat(V.split(dim_split, 2), 0)   # [B*H, Nk, d]

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)  # [B*H, Nq, Nk]

        if mask is not None:
            # mask: [B, Nk] -> tile for H heads -> [B*H, Nk]
            mask_tiled = mask.repeat(self.num_heads, 1)          # [B*H, Nk]
            mask_tiled = mask_tiled.unsqueeze(1)                 # [B*H, 1, Nk]
            A = A.masked_fill(~mask_tiled, float('-inf'))

        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MaskedSAB(nn.Module):
    """Self-Attention Block with key-padding mask."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MaskedMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


class MaskedPMA(nn.Module):
    """Pooling by Multihead Attention with key-padding mask."""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MaskedMAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        S = self.S.repeat(X.size(0), 1, 1)
        return self.mab(S, X, mask=mask)
