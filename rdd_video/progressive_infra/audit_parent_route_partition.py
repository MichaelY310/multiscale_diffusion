"""CPU-only, real-mask screen of one persistent parent hypergraph placement.

No SP timing or image-quality claim. The optimizer's cost counts DISTINCT KV
requests, not attention edge cuts. All layers/heads share one balanced layout.
Actual mask capture/CPU optimization costs must not be excluded if deployed.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def request_counts(masks, owner, ranks):
    return np.stack([masks[:, owner==r].sum(1,dtype=np.int16) for r in range(ranks)])


def stats(counts, owner, ranks):
    payload=[]
    for r in range(ranks):
        payload.append((counts[r,:,owner!=r]>0).sum())
    total=sum(payload)
    possible=counts.shape[1]*sum((owner!=r).sum() for r in range(ranks))
    return dict(remote_union=total/possible,remote_blocks_total=int(total),
                per_rank_remote_blocks=[int(x) for x in payload],
                maximum_rank_blocks=int(max(payload)))


def main(a):
    out=Path(a.directory)
    meta=json.loads((out/'metrics.json').read_text())
    shape=meta['shape']
    hw=(shape[3]//8)*(shape[4]//8)
    n=shape[2]//16*hw
    archive=np.load(out/'packed_masks.npz')
    root=np.stack([np.unpackbits(archive[f's0_l{l}'],count=12*n*n).reshape(12,n,n)
                   for l in range(30)]).reshape(-1,n,n).astype(bool)
    ranks=4
    if n%ranks:
        raise ValueError('Exactly balanced four ranks required')
    starts={
        'contiguous':np.arange(n)*ranks//n,
        'persistent_spatial':(np.arange(n)%hw)*ranks//hw,
    }
    start=time.perf_counter()
    # Spectral seed only. The actual accepted objective below is demand union.
    affinity=np.zeros((n,n),dtype=np.float64)
    for m in root:
        # Weight uncommon keys; globally popular keys poorly identify groups.
        weight=1/np.maximum(m.sum(0),1)
        f=m.astype(np.float64)*np.sqrt(weight)[None,:]
        affinity+=f@f.T
    affinity/=root.shape[0]
    degree=np.maximum(affinity.sum(1),1e-9)
    _,vec=np.linalg.eigh(affinity/np.sqrt(degree[:,None]*degree[None,:]))
    embedding=vec[:,-ranks:]
    embedding/=np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1e-9)
    rng=np.random.default_rng(20261003)
    seeds=rng.choice(n,ranks,replace=False)
    centers=embedding[seeds]
    for _ in range(12):
        costs=((embedding[:,None,:]-centers[None,:,:])**2).sum(-1)
        _,cols=linear_sum_assignment(np.repeat(costs,n//ranks,axis=1))
        owner=cols//(n//ranks)
        centers=np.stack([embedding[owner==r].mean(0) for r in range(ranks)])
    starts['balanced_spectral']=owner
    report=dict(mode='CPU hypergraph screen, NOT measured SP acceleration',shape=shape,
        root_blocks=n,heads_and_layers=root.shape[0],ranks=ranks,
        cost='Distinct remote KV requests, all30layers/12heads; one assignment',layouts={})
    for name,initial in starts.items():
        owner=initial.copy()
        counts=request_counts(root,owner,ranks)
        initial_stats=stats(counts,owner,ranks)
        objective=initial_stats['remote_blocks_total']
        accepted=0
        for _ in range(a.swaps):
            x,y=rng.choice(n,2,replace=False)
            r,s=owner[x],owner[y]
            if r==s:
                continue
            cr=counts[r]-root[:,x,:]+root[:,y,:]
            cs=counts[s]-root[:,y,:]+root[:,x,:]
            new_owner=owner.copy()
            new_owner[x],new_owner[y]=s,r
            before=(counts[r,:,owner!=r]>0).sum()+(counts[s,:,owner!=s]>0).sum()
            after=(cr[:,new_owner!=r]>0).sum()+(cs[:,new_owner!=s]>0).sum()
            if after < before:
                owner=new_owner
                counts[r],counts[s]=cr,cs
                objective+=int(after-before)
                accepted+=1
        actual=stats(counts,owner,ranks)
        if actual['remote_blocks_total']!=objective or not np.all(np.bincount(owner)==n//ranks):
            raise AssertionError('Incremental objective or balance mismatch')
        online=[]
        for stage in (1,2):
            ratio=2**stage
            nf=n*ratio
            ids=np.arange(nf)
            parent=(ids//hw//ratio)*hw+ids%hw
            fo=owner[parent]
            masks=np.stack([np.unpackbits(archive[f's{stage}_l{l}'],count=12*nf*nf).reshape(12,nf,nf)
                            for l in range(30)]).reshape(-1,nf,nf).astype(bool)
            online.append(dict(stage=stage,**stats(request_counts(masks,fo,ranks),fo,ranks)))
        report['layouts'][name]=dict(initial=initial_stats,optimized=actual,
            accepted_swaps=accepted,owner=owner.tolist(),online_under_same_assignment=online)
    report['CPU_planning_seconds']=time.perf_counter()-start
    report['warning']='CPU planning, mask capture, redistribution not included in any speed claim. Ordinary previous/draft masks can use identical optimizer.'
    (out/'partition_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',required=True)
    p.add_argument('--swaps',type=int,default=5000)
    main(p.parse_args())
