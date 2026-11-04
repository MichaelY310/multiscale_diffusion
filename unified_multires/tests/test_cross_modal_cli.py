import json
import os
import subprocess
import sys
import torch


def test_video_local_shards_cli_exact_resume_and_sampling(tmp_path):
    def run(*args):
        result=subprocess.run([sys.executable,'-m','umdiff','--threads','2',*map(str,args)],capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    config=tmp_path/'rdd_video.json'
    run('configure','--method','rdd','--shape','2,3,8,8','--width',24,'--depth',1,'--attention-heads',3,'--patch','1,1,1','--sampling-steps',3,'--decoupled','--output',config)
    data=tmp_path/'data';data.mkdir()
    torch.save({'latents':torch.randn(4,2,3,8,8)},data/'a.pt')
    torch.save({'latents':torch.randn(4,2,3,8,8)},data/'b.pt')
    (data/'manifest.json').write_text(json.dumps({'format':'umdiff_shards_v1','shards':[{'file':'a.pt','count':4},{'file':'b.pt','count':4}]}))
    common=['--config',config,'--data',data,'--batch-size',2,'--seed',17,'--device','cpu']
    run('train',*common,'--steps',4,'--output',tmp_path/'full')
    run('train',*common,'--steps',2,'--output',tmp_path/'split')
    run('train',*common,'--steps',2,'--output',tmp_path/'split','--resume',tmp_path/'split/checkpoint.pt')
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False);b=torch.load(tmp_path/'split/checkpoint.pt',weights_only=False)
    def compare(x,y):
        if torch.is_tensor(x):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:compare(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for v,w in zip(x,y):compare(v,w)
        else:assert x==y
    for k in ('model','ema','optimizer','stream','step','images_seen'):compare(a[k],b[k])
    for name in ('full','split'):
        run('sample','--checkpoint',tmp_path/name/'checkpoint.pt','--count',2,'--seed',71,'--output',tmp_path/(name+'.pt'))
    x=torch.load(tmp_path/'full.pt',weights_only=True);y=torch.load(tmp_path/'split.pt',weights_only=True)
    assert x.shape==(2,2,3,8,8) and torch.equal(x,y) and torch.isfinite(x).all()
