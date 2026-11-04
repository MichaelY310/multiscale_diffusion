import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from probe_rdd_native_coarse_kv import parent_mean,parent_center_rope,project,block_with_memory
from models_multires import DiTBlockMultiRes,get_2d_rotary_pos_embed,apply_rotary_emb


class NativeKVTests(unittest.TestCase):
    def test_parent_pool(self):
        x=torch.arange(16.).reshape(1,1,16,1)
        torch.testing.assert_close(parent_mean(x).flatten(),torch.tensor([2.5,4.5,10.5,12.5]))

    def test_dense_path_exact_and_modes_finite(self):
        torch.manual_seed(941)
        block=DiTBlockMultiRes(16,4).eval()
        h,c=torch.randn(2,16,16),torch.randn(2,16)
        src=torch.randn(2,4,16)
        rope=get_2d_rotary_pos_embed(4,4,4,base_size=4)
        sr=get_2d_rotary_pos_embed(4,2,2,base_size=4)
        _,k,v,_,norm=project(block,src,c)
        cache={'raw_k':k,'v':v,'normalized':norm,'rotated_k':apply_rotary_emb(k,*sr),
               'center_rope':parent_center_rope(4,4,4,torch.device('cpu'))}
        torch.testing.assert_close(block(h,c,*rope),block_with_memory(block,h,c,rope,cache,'dense_parity'),atol=0,rtol=0)
        for mode in ('cached_native','cached_center','cached_recondition','current_parent'):
            out=block_with_memory(block,h,c,rope,cache,mode)
            self.assertEqual(out.shape,h.shape)
            self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
