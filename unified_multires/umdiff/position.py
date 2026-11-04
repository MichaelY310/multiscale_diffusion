"""PixelFlow's pinned Diffusers 0.32 NumPy-grid RoPE, without deprecated API."""
import numpy as np

def rotary_2d(embed_dim,crops_coords,grid_size,use_real=True):
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_from_grid
    start,stop=crops_coords
    h=np.linspace(start[0],stop[0],grid_size[0],endpoint=False,dtype=np.float32)
    w=np.linspace(start[1],stop[1],grid_size[1],endpoint=False,dtype=np.float32)
    grid=np.stack(np.meshgrid(w,h),axis=0).reshape(2,1,*grid_size)
    return get_2d_rotary_pos_embed_from_grid(embed_dim,grid,use_real=use_real)
