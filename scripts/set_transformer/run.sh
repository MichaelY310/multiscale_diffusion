# 中间加上autoencoder
# concat
cd /scr/yuan/set_transformer && nohup python train_emd_ae.py \
  --num_steps 1000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 32 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool concat --bottleneck_dim 256 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_concat --metrics_file train_emd_ae_metrics_concat.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_concat.log 2>&1 &

# 64
cd /scr/yuan/set_transformer && nohup python train_emd_ae.py \
  --num_steps 1000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 32 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool concat --bottleneck_dim 64 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_concat_64 --metrics_file train_emd_ae_metrics_concat_64.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_concat_64.log 2>&1 &

# mean
cd /scr/yuan/set_transformer && nohup python train_emd_ae.py \
  --num_steps 1000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 256 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool mean --bottleneck_dim 256 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_mean --metrics_file train_emd_ae_metrics_mean.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_mean.log 2>&1 &

# 64
cd /scr/yuan/set_transformer && nohup python train_emd_ae.py \
  --num_steps 5000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 64 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool mean --bottleneck_dim 64 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_mean_64 --metrics_file train_emd_ae_metrics_mean_64.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_mean_64.log 2>&1 &

  # 原始
  nohup python train_emd.py \
    --num_steps 500000 \
    --batch_size 64 \
    --lr 1e-4 \
    --dim_hidden 32 \
    --num_heads 4 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --ln \
    --ckpt_dir checkpoints_emd_d32 \
    --metrics_file train_emd_d32_metrics.csv \
    --num_threads 10     > train_emd_d32.log 2>&1 &

# 替换成embedding
nohup python train_emd_bottleneck.py \
    --num_steps 500000 \
    --batch_size 64 \
    --lr 1e-4 \
    --dim_hidden 384 \
    --num_heads 16 \
    --num_encoder_layers 8 \
    --num_decoder_layers 8 \
    --num_threads 10     > train_emd_bottleneck_384.log 2>&1 &



# 1d to 3d
cd /scr/yuan/set_transformer && nohup python train_3d_embedding.py \
  --pretrained checkpoints_emd_ae_concat/latest.pt \
  --fused_dim 256 --fused_num_mlp_layers 3 --fused_mlp_factor 4 \
  --num_steps 100000 --batch_size 64 --lr 1e-3 \
  --ckpt_dir checkpoints_3d_emb --metrics_file train_3d_emb_metrics.csv \
  --device cpu --num_threads 4 \
  > train_3d_emb.log 2>&1 &



# 3D
# 中间加上autoencoder
# concat
cd /scr/yuan/set_transformer && nohup python train_emd_ae_3d.py \
  --num_steps 10000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 32 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool concat --bottleneck_dim 256 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_concat_3d --metrics_file train_emd_ae_metrics_concat_3d.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_concat_3d.log 2>&1 &
# mean
cd /scr/yuan/set_transformer && nohup python train_emd_ae_3d.py \
  --num_steps 10000000 --batch_size 64 --lr 1e-4 \
  --dim_hidden 256 --num_heads 4 \
  --num_encoder_layers 4 --num_decoder_layers 4 \
  --ae_pool mean --bottleneck_dim 256 --ae_num_heads 4 --ae_num_mlp_layers 3 --ae_mlp_factor 4 \
  --ckpt_dir checkpoints_emd_ae_mean_3d --metrics_file train_emd_ae_metrics_mean_3d.csv \
  --device cpu --num_threads 10 \
  > train_emd_ae_mean_3d.log 2>&1 &