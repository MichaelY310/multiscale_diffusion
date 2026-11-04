#!/bin/bash
cd /home/tiger/multiscale_diffusion/RDD_DiT_3D/set_transformer_ae
python ae_model.py --device cuda \
  --dim_hidden 16 --num_heads 4 --num_encoder_layers 4 --num_decoder_layers 4 \
  --bottleneck_dim 128 \
  --ae_pool concat --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --lr 1e-4 --num_steps 5000000 --batch_size 64 \
  --perm_save_freq 500000 \
  --ckpt_dir checkpoints_v1_exp7_d16_bn128 --metrics_file metrics_v1_exp7.csv
