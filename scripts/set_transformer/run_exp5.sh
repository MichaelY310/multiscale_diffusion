#!/bin/bash
cd /home/tiger/multiscale_diffusion/RDD_DiT_3D/set_transformer_ae

CUDA_VISIBLE_DEVICES=0 python ae_model_v2.py \
    --dim_hidden 32 --num_heads 4 --num_encoder_layers 4 --num_decoder_layers 4 \
    --ae_pool pma1 --bottleneck_dim 64 \
    --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
    --decoder_type parallel --use_k_feature 0 \
    --lr 3e-4 --weight_decay 1e-4 --num_steps 2000000 --batch_size 64 \
    --ckpt_dir checkpoints_v2_exp5_pma1_parallel_64 --metrics_file metrics_v2_exp5.csv \
    --perm_save_freq 500000
