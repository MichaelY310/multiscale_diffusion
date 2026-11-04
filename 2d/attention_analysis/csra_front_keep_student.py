"""Opt-in conditional cross-scale bridge AFTER one real fine block.

No changes to old prefix students or old checkpoint formats. Six skipped blocks
are shifted from0..5 to1..6; executed blocks are0,7..23 (18 total).
"""
import torch
from torch.utils.checkpoint import checkpoint
from csra_prefix_student import ConditionalPrefixStudent

FORMAT = 'csra_front1_skip6_conditional_v1'


def check_initializer(payload):
    from probe_csra_front_keep import check_contract, layout
    check_contract(payload,1)
    expected = layout(1)
    if any(payload.get(k)!=v for k,v in expected.items()):
        raise ValueError('Initializer does not target front1, skipped1..6')
    if payload['fits']['fine_only']['weight'].shape != (2048,1024):
        raise ValueError('Need same fine-only L/2 initializer for both arms')


class FrontKeepStudent(ConditionalPrefixStudent):
    def __init__(self,*args,front_keep=1,**kwargs):
        if front_keep!=1:
            raise ValueError('Only the predeclared front1 placement is supported')
        super().__init__(*args,**kwargs)
        self.front_keep = front_keep
        if self.prefix_depth+front_keep >= len(self.backbone.blocks):
            raise ValueError('Must retain a suffix')
        # Superclass froze the old prefix. Replace that mask explicitly.
        for i,block in enumerate(self.backbone.blocks):
            block.requires_grad_(not(front_keep<=i<front_keep+self.prefix_depth))

    def forward(self,batch,return_hidden=False):
        x,t,y = batch['x'],batch['t'],batch['labels']
        m = self.backbone
        gh,gw = x.shape[-2]//m.patch_size,x.shape[-1]//m.patch_size
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
            h = m.x_embedder(x)
            c = m.t_embedder(t)+m.y_embedder(y,False)
            cos,sin = m.get_rope(gh,gw,x.device)
            def execute(block,value):
                if self.checkpoint_blocks and torch.is_grad_enabled():
                    return checkpoint(block,value,c,cos,sin,use_reentrant=False)
                return block(value,c,cos,sin)
            for block in m.blocks[:self.front_keep]:
                h = execute(block,h)
            source = None if self.coarse_source=='zero' else batch['coarse'][self.source_layer]
            h = self.lift(h,c,source).to(h.dtype)
            bridge_hidden = h
            for block in m.blocks[self.front_keep+self.prefix_depth:]:
                h = execute(block,h)
            out = m.unpatchify(m.final_layer(h,c),gh,gw)
            return (out,bridge_hidden) if return_hidden else out


def check_resume(payload,args,initializer_sha):
    if (payload.get('format')!=FORMAT or payload.get('front_keep')!=1
            or payload.get('prefix_depth')!=6 or payload.get('source_layer')!=13
            or payload.get('initializer_sha256')!=initializer_sha):
        raise ValueError('Wrong front1 resume source/target/initializer')
    for key in ('base_ckpt','data_root','hidden_root','initializer','coarse_source','seed'):
        if payload['args'][key]!=vars(args)[key]:
            raise ValueError(f'Resume changed {key}')
