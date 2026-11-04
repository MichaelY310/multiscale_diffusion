#!/usr/bin/env python3
"""Read-only: keep fine MLP/gates, replace only attention K/V by coarse memory.

This diagnostic intentionally computes the original full QKV to keep Q bitwise
matched. It is NOT an accelerated implementation or a latency benchmark.
The current-parent control is pooled fine K/V, not VSA (no sparse/global gates).
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from train_rdd_csra_adapter import load_frozen_teacher
from probe_rdd_vs_dit_block_reuse import manual_forward
from models_multires import apply_rotary_emb, get_1d_rotary_pos_embed, modulate

MODES=('cached_native','cached_center','cached_recondition','current_parent')
LAYERS=[13,15,17,20]


def parent_mean(x):
    """B,heads,child_tokens,dim -> 2x2 averaged parent tokens, row-major."""
    b,h,n,d=x.shape
    g=int(n**.5)
    if g*g!=n or g%2:
        raise ValueError('Expected even square token grid')
    return x.reshape(b,h,g//2,2,g//2,2,d).mean((3,5)).reshape(b,h,n//4,d)


def parent_center_rope(head_dim, fine_grid, base_size, device):
    pos=torch.linspace(0,base_size,fine_grid).reshape(-1,2).mean(1)
    c,s=get_1d_rotary_pos_embed(head_dim//2,pos)
    g=len(pos)
    cos=torch.cat((c[:,None].expand(-1,g,-1).reshape(g*g,-1),c[None].expand(g,-1,-1).reshape(g*g,-1)),1)
    sin=torch.cat((s[:,None].expand(-1,g,-1).reshape(g*g,-1),s[None].expand(g,-1,-1).reshape(g*g,-1)),1)
    return cos.to(device),sin.to(device)


def project(block,h,c):
    parts=block.adaLN_modulation(c).chunk(6,1)
    normalized=block.norm1(h)
    shaped=modulate(normalized,parts[0],parts[1])
    b,n,d=h.shape
    q,k,v=block.attn.qkv(shaped).reshape(b,n,3,block.attn.num_heads,block.attn.head_dim).permute(2,0,3,1,4).unbind(0)
    return q,k,v,parts,normalized


def block_with_memory(block,h,c,rope,source,mode):
    q,k,v,parts,_=project(block,h,c)
    if mode=='cached_native':
        k,v=source['rotated_k'],source['v']
    elif mode=='cached_center':
        k,v=apply_rotary_emb(source['raw_k'],*source['center_rope']),source['v']
    elif mode=='cached_recondition':
        inp=modulate(source['normalized'],parts[0],parts[1])
        b,n,d=inp.shape
        _,k,v=block.attn.qkv(inp).reshape(b,n,3,block.attn.num_heads,block.attn.head_dim).permute(2,0,3,1,4).unbind(0)
        k=apply_rotary_emb(k,*source['center_rope'])
    elif mode=='current_parent':
        k,v=apply_rotary_emb(parent_mean(k),*source['center_rope']),parent_mean(v)
    elif mode=='dense_parity':
        k=apply_rotary_emb(k,*rope)
    else:
        raise ValueError(mode)
    q=apply_rotary_emb(q,*rope)
    attn=((q@k.transpose(-2,-1))*block.attn.scale).softmax(-1)
    a=block.attn.proj((attn@v).transpose(1,2).reshape(h.shape))
    mid=h+parts[2][:,None]*a
    result=mid+parts[5][:,None]*block.mlp(modulate(block.norm2(mid),parts[3],parts[4]))
    return result


@torch.inference_mode()
def main(args):
    out=Path(args.output_dir)
    if (out/'metrics.json').exists(): return
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0]<8*2**30:
        raise RuntimeError('Need8GiB free; never stop another process')
    base=Path('checkpoints/RDD-Unified-L/0400000.pt')
    model=load_frozen_teacher(base,device,'DiT-L/2','rdd')
    root=Path(args.data_root)/f'shard_{args.shard}'
    mp=root/'manifest.json'
    manifest=json.loads(mp.read_text())
    digest=hashlib.sha256(mp.read_bytes()).hexdigest()
    replay=Path(args.source_root)/root.name
    side=json.loads((replay/'manifest.json').read_text())
    if not manifest['complete'] or manifest['parity_max_abs']!=0 or side['source_manifest_sha256']!=digest or not side['complete']:
        raise ValueError('Corpus/source parity or manifest mismatch')
    if Path(manifest['base_checkpoint']).resolve()!=base.resolve(): raise ValueError('Wrong model')
    items=[i for i in manifest['files'] if i['split']=='validation'][:1]
    if len(items)!=1: raise ValueError('Missing validation batch')
    rows=[]
    for item in items:
        if Path(item['name']).name!=item['name']: raise ValueError('Unsafe record')
        ref=torch.load(root/item['name'],map_location='cpu',weights_only=True)
        saved=torch.load(replay/item['name'],map_location='cpu',weights_only=True)
        if saved['seed']!=item['seed'] or saved['cache_parity_max_abs']!=0: raise ValueError('Source mismatch')
        src=saved['sources']['16']; labels=ref['labels'].to(device)
        if not torch.equal(src['labels'],ref['labels']) or not torch.all(src['t']==300): raise ValueError('Source time/label mismatch')
        coarse_x,coarse_t=src['x'].to(device),src['t'].to(device)
        pred,trace=manual_forward(model,coarse_x,coarse_t,labels,multires=True,capture_layers=set(LAYERS))
        torch.testing.assert_close(pred.cpu(),src['target'],atol=0,rtol=0)
        caches={}
        rope=model.get_rope(16,16,device)
        coarse_rope=model.get_rope(8,8,device)
        centers=parent_center_rope(model.head_dim,16,model.base_size,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=model.t_embedder(coarse_t)+model.y_embedder(labels,False)
            for l in LAYERS:
                torch.testing.assert_close(trace[l]['residual'].cpu(),ref['coarse'][l],atol=0,rtol=0)
                _,k,v,_,normalized=project(model.blocks[l],trace[l]['input'],c)
                caches[l]={'raw_k':k,'v':v,'normalized':normalized,'rotated_k':apply_rotary_emb(k,*coarse_rope),'center_rope':centers}
        for i in range(len(ref['t'])):
            x,t,target=(ref[key][i].to(device) for key in ('x','t','target'))
            dense,fine_trace=manual_forward(model,x,t,labels,multires=True,capture_layers=set(LAYERS))
            torch.testing.assert_close(dense,target,atol=0,rtol=0)
            rng=torch.cuda.get_rng_state(device)
            # Exact full-block replacements avoid extra h+(result-h) rounding.
            for mode in ('dense_parity',)+MODES:
                replacements={l:(lambda h,c,l=l,mode=mode:block_with_memory(model.blocks[l],h,c,rope,caches[l],mode)) for l in LAYERS}
                pred,_=manual_forward(model,x,t,labels,multires=True,replacements=replacements,
                    conditioned_replacements=True,replacement_mode='hidden')
                if mode=='dense_parity':
                    torch.testing.assert_close(pred,dense,atol=0,rtol=0)
                    continue
                error=(pred[:,:4].float()-target[:,:4].float()).square().mean((1,2,3))
                power=target[:,:4].float().square().mean((1,2,3))
                rows.extend({'seed':item['seed'],'image':j,'t':int(t[j]),'mode':mode,'main_mse':float(error[j]),'target_power':float(power[j])} for j in range(len(labels)))
            if not torch.equal(rng,torch.cuda.get_rng_state(device)): raise RuntimeError('Diagnostic consumedRNG')
    assert len(rows)==8*8*len(MODES)
    summary={mode:{'states':sum(r['mode']==mode for r in rows),
        'main_mse':sum(r['main_mse'] for r in rows if r['mode']==mode)/64} for mode in MODES}
    out.mkdir(parents=True,exist_ok=True)
    temp=out/'metrics.json.tmp'
    temp.write_text(json.dumps({'complete':True,'config':vars(args),'layers':LAYERS,'rows':rows,'summary':summary,
        'dense_parity_max_abs':0.,'max_memory_gib':torch.cuda.max_memory_allocated()/2**30,
        'warning':'Read-only teacher-state diagnostic. Keeps CURRENT fine MLP and all gates. Does not skip full QKV yet (reference Q is exact), not latency/FID/rollout. current_parent is pooled-fine global attention, NOT VSA.'},indent=2,allow_nan=False))
    temp.replace(out/'metrics.json')
    print(json.dumps({'shard':args.shard,'summary':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--source-root',default='eval_results/csra_joint_sources_20260913')
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--output-dir',required=True)
    main(p.parse_args())
