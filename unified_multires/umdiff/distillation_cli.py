"""Local-embedding training and inference for the two-optimizer RMD task."""
import json,random
from pathlib import Path
import numpy as np
import torch
from .registry import build
from .distillation import RMDTrainer,RMDObjective
from .randomness import GlobalRandomStream

def validate(config):
    allowed={'name','task','model','image_shape','training','distillation','evidence','checkpoints'}
    if set(config)-allowed:raise ValueError('Unknown RMD task configuration fields')
    if len(config['image_shape'])!=4:raise ValueError('RMD input shape is C,T,H,W')
    if set(config.get('training',{}))-{'optimizer','precision','accumulation_steps'}:raise ValueError('Unknown RMD task training recipe field')

def train(args,config,device):
    validate(config);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model=build('model',config['model']).to(device);batch=args.batch_size
    if args.synthetic:
        rng=torch.Generator().manual_seed(args.seed+1024);dim=config['model']['args']['text_dim']
        text=torch.randn(batch,3,dim,generator=rng);negative=torch.randn(1,3,dim,generator=rng)
    elif args.data:
        data=torch.load(args.data,map_location='cpu',weights_only=True)
        if set(data)!={'prompt_embeds','negative_prompt_embeds'}:raise ValueError('RMD local input requires prompt_embeds and negative_prompt_embeds')
        text=data['prompt_embeds'];negative=data['negative_prompt_embeds']
        if len(text)<batch:raise ValueError('Need at least one complete prompt batch')
    else:raise ValueError('RMD requires local embedding data or --synthetic')
    trainer=RMDTrainer(config,model,negative.to(device),args.seed,initialize=not bool(args.resume))
    from .training import data_binding
    trainer.data_binding=data_binding(args.data,batch,args.seed,synthetic=args.synthetic)
    if args.resume:trainer.load(args.resume)
    template=torch.empty(batch,*config['image_shape'],device=device);trace=[]
    for i in range(args.steps):
        indices=torch.tensor([(trainer.step_index*batch+j)%len(text) for j in range(batch)])
        current=text[indices].to(device);values=trainer.step(template,current);trainer.sample_text=current.detach()
        row=dict(step=trainer.step_index,score_loss=values[0].item(),generator_loss=values[1].item());trace.append(row)
        if trainer.step_index%args.log_every==0 or i==args.steps-1:print(json.dumps(row),flush=True)
        if args.save_every and trainer.step_index%args.save_every==0:trainer.save(Path(args.output)/'checkpoint.pt')
    trainer.save(Path(args.output)/'checkpoint.pt');(Path(args.output)/'trace.json').write_text(json.dumps(trace,indent=2))

def sample(args,checkpoint,device):
    config=checkpoint['config'];validate(config)
    if args.steps is not None:raise ValueError('Set distillation.k_step in the config; RMD sampling does not use generic --steps')
    if args.labels:raise ValueError('RMD uses text embeddings; provide --conditioning')
    torch.manual_seed(args.seed);model=build('model',config['model']).to(device).eval();model.load_state_dict(checkpoint['model'])
    if args.conditioning:
        data=torch.load(args.conditioning,map_location='cpu',weights_only=True);text=data['prompt_embeds']
    else:text=checkpoint['sample_text'].cpu()
    if len(text)==1:text=text.expand(args.count,-1,-1)
    if len(text)<args.count:raise ValueError('Not enough supplied prompt embeddings for --count')
    text=text[:args.count].to(device);stream=GlobalRandomStream()
    precision=config.get('training',{}).get('precision','float32');dtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[precision]
    objective=RMDObjective(checkpoint['negative'].to(device),stream,dtype=dtype,**config.get('distillation',{}))
    torch.manual_seed(args.seed);initial=stream.normal(torch.empty(args.count,*config['image_shape'],device=device,dtype=dtype))
    with torch.no_grad(),torch.autocast(device.type,dtype=dtype if precision!='float32' else torch.bfloat16,enabled=precision!='float32'):result,_,_=objective.predictor.generate(model,initial,text.to(dtype),objective.steps,objective.transition,objective.switch,objective.target)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(result.cpu(),output)
    print(json.dumps(dict(output=str(output),space='BCTHW latent; supply local Wan VAE for decoded pixels',weights='generator (source RMD has no EMA)')))
