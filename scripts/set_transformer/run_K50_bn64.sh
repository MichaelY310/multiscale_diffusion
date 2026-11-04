#!/bin/bash
# Set Transformer K=50, bottleneck_dim=64, 4层enc/4层dec, dim_hidden=32
# GPU 2
cd /home/tiger/multiscale_diffusion/RDD_DiT_3D/set_transformer_ae

CUDA_VISIBLE_DEVICES=2 python ae_model.py --device cuda \
  --dim_hidden 32 --num_heads 4 --num_encoder_layers 4 --num_decoder_layers 4 \
  --bottleneck_dim 64 \
  --k_max 50 \
  --ae_pool concat --ae_num_heads 4 --ae_num_mlp_layers 4 --ae_mlp_factor 8 \
  --lr 1e-4 --num_steps 5000000 --batch_size 256 \
  --perm_save_freq 500000 \
  --ckpt_dir checkpoints_K50_bn64 --metrics_file csv/metrics_K50_bn64.csv
