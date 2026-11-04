"""Read-only source-paired probe. No quality/training/latency equivalence claim."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_direct_reuse_fullmodel import PairDataset
from train_rdd_csra_adapter import load_frozen_teacher
from direct_reuse_fullmodel import DirectReuseStudent,MASKS
from direct_reuse_gate_refresh import GateRefreshStudent


@torch.inference_mode()
def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3'):raise ValueError('GPU0-3 only')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0]<8*2**30:raise RuntimeError('Need8GiB free')
    torch.cuda.set_per_process_memory_fraction(.08,dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base='checkpoints/RDD-Unified-L/0400000.pt'
    model=load_frozen_teacher(Path(base),dev,'DiT-L/2','rdd')
    data=PairDataset('eval_results/csra_rollout_data4k_20260912','eval_results/csra_joint_sources_20260913',base,'validation',max_files=1)
    direct=DirectReuseStudent(model,MASKS['reuse6'],False);gate=GateRefreshStudent(model,MASKS['reuse6'],False)
    rows=[]
    for batch in DataLoader(data,batch_size=8,shuffle=False):
        b={k:v.to(dev) for k,v in batch.items()}
        middle,c1=direct.stage(b['source16_x'],b['source16_t'],b['labels'],capture=True)
        gate_middle,c2=gate.stage(b['source16_x'],b['source16_t'],b['labels'],capture=True)
        torch.testing.assert_close(middle,gate_middle,atol=0,rtol=0)
        for name,net,cache in [('direct_delta',direct,c1),('original_gate_refresh',gate,c2)]:
            p=net.stage(b['x'],b['t'],b['labels'],cache=cache)[0]
            mse=(p[:,:4].float()-b['target'][:,:4].float()).square().flatten(1).mean(1)
            for t,e in zip(b['t'].tolist(),mse.tolist()):rows.append(dict(method=name,t=t,mse=e))
    summary={name:sum(r['mse'] for r in rows if r['method']==name)/len(data) for name in ('direct_delta','original_gate_refresh')}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(complete=True,images=data.images,states=len(data),rows=rows,summary=summary,
        capture_path_max_abs=0.,new_parameters=0,warning='Teacher-state epsilon MSE only; NOT image quality or FID'),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
