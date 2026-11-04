"""扫描预处理数据，找到 R=256 阶段 token 数最多的样本。"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from dataset import compute_stage_data

data_dir = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/RDD-DiT-3D/preprocessed_R256_K10"
stage_resolutions = [8, 16, 32, 64, 128, 256]

import glob
files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
print(f"Total files: {len(files)}")

max_r256_tokens = 0
max_file = ""
max_r128_occupied = 0

# 也统计分布
token_counts_256 = []

for i, f in enumerate(files):
    data = torch.load(f, map_location='cpu')
    emb = data['embeddings']
    coords = data['coords']
    source_R = data['resolution']
    
    stage_embs, stage_coords, stage_occupied = compute_stage_data(emb, coords, source_R, stage_resolutions)
    
    r128_occ = stage_occupied[128].sum().item()
    r256_tokens = stage_embs[256].shape[0]
    token_counts_256.append(r256_tokens)
    
    if r256_tokens > max_r256_tokens:
        max_r256_tokens = r256_tokens
        max_file = f
        max_r128_occupied = r128_occ
    
    if (i + 1) % 500 == 0:
        print(f"  Scanned {i+1}/{len(files)} | Current max R256 tokens: {max_r256_tokens} (R128 occ: {max_r128_occupied})")

print(f"\n{'='*60}")
print(f"Max R=256 tokens: {max_r256_tokens}")
print(f"  = R=128 occupied cells × 8 = {max_r128_occupied} × 8 = {max_r128_occupied * 8}")
print(f"  File: {os.path.basename(max_file)}")
print(f"\nDistribution stats:")
token_counts_256 = torch.tensor(token_counts_256, dtype=torch.float)
print(f"  Mean:   {token_counts_256.mean().item():.0f}")
print(f"  Median: {token_counts_256.median().item():.0f}")
print(f"  P90:    {token_counts_256.quantile(0.9).item():.0f}")
print(f"  P95:    {token_counts_256.quantile(0.95).item():.0f}")
print(f"  P99:    {token_counts_256.quantile(0.99).item():.0f}")
print(f"  Max:    {token_counts_256.max().item():.0f}")
print(f"  Min:    {token_counts_256.min().item():.0f}")
