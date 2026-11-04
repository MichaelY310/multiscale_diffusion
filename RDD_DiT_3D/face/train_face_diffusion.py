"""
Train RDD-DiT-3D (Face version).
Uses the same diffusion training logic as vertex version.
"""
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from train import (
    main as train_main,
    train_unified,
    train_multi_model,
    create_argparser,
    setup_distributed,
    cleanup,
    update_ema,
)
from model import DiT3D_RDD, DiT3D_RDD_models
from diffusion_3d import create_diffusion_3d, GaussianDiffusion3D
from dataset import SparseVolumeDataset, sparse_collate_fn


if __name__ == "__main__":
    print("=" * 60)
    print("Face RDD-DiT-3D Diffusion Model Training")
    print("=" * 60)
    print("Reuses vertex version training logic.")
    print("Ensure --in-channels matches Face Set Transformer bottleneck_dim.")
    print("=" * 60)
    train_main()
