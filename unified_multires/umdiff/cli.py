"""Offline training, exact resume, sampling and configuration inspection."""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .registry import build,REGISTRY
from .engine import Method
from .training import Trainer,LocalImages
from .randomness import RandomStream
from .tree import map_tree

def make_method(config):
    if config.get('task')=='rdd_fm_native':
        from .rdd_fm_native import NativeRDDFM
        return NativeRDDFM(config)
    if config.get('task')=='multires':
        from .multires_process import MultiresMethod
        return MultiresMethod(config)
    if config.get('task')=='codec':
        from .codec_training import CodecMethod
        return CodecMethod(config)
    return Method(config)

def read_config(path):
    config=json.loads(Path(path).read_text())
    if not isinstance(config,dict):raise ValueError('Configuration must be an object')
    if 'training' in config:
        unknown=set(config['training'])-{'optimizer','time_sampler','conditioning','inference_only','initialization','freeze','data','input_transform','input_transform_scope','precision','accumulation_steps'}
        if unknown:raise ValueError(f'Unknown training fields: {unknown}')
        if config.get('task')!='rmd' and set(config['training'])&{'precision','accumulation_steps'}:
            raise ValueError('For ordinary/codec training, precision and accumulation_steps belong in training.optimizer')
    return config

def choose_device(value):
    if value=='cpu':return torch.device('cpu')
    if value not in ('cuda','cuda:0'):raise ValueError('Only CPU or the single visible GPU is supported by this CLI')
    # Job authorization/deadlines belong to the launch environment, not a
    # hostname and an expired historical date embedded in the library.
    if os.environ.get('UMDIFF_CPU_ONLY')=='1':raise RuntimeError('This shell is CPU-only (UMDIFF_CPU_ONLY=1)')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; use --device cpu or select a working CUDA environment')
    return torch.device('cuda:0')

class OfflineTensors(torch.utils.data.Dataset):
    def __init__(self,path):
        value=torch.load(path,map_location='cpu',weights_only=True)
        if torch.is_tensor(value):self.images=value;self.labels=None;self.conditions={}
        else:
            allowed={'images','latents','labels','conditions'}
            if not isinstance(value,dict) or set(value)-allowed:raise ValueError('Tensor dataset must contain images or latents, optional labels/conditions')
            if ('images' in value)==('latents' in value):raise ValueError('Supply exactly one of images and latents')
            key='latents' if 'latents' in value else 'images';self.images=value[key];self.labels=value.get('labels');self.conditions=value.get('conditions',{})
        if not isinstance(self.conditions,dict) or set(self.conditions)&{'images','labels'}:raise ValueError('Condition fields must be a dictionary without reserved images/labels keys')
        if self.images.ndim not in (4,5) or not self.images.is_floating_point():raise ValueError('Local tensor input must be floating BCHW or BCTHW in the intended representation/normalization')
        if self.labels is not None and len(self.labels)!=len(self.images):raise ValueError('Label length mismatch')
        if any(len(v)!=len(self.images) for v in self.conditions.values()):raise ValueError('Conditioning length mismatch')
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        out={'images':self.images[i],**{k:v[i] for k,v in self.conditions.items()}}
        if self.labels is not None:out['labels']=self.labels[i]
        return out

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self,path,size,data_recipe=None):
        recipe=data_recipe or {};self.images=LocalImages(path,size,crop=recipe.get('crop','legacy'),horizontal_flip=recipe.get('horizontal_flip',False),expand_ratio=recipe.get('expand_ratio',1.125),flip_rng=recipe.get('flip_rng','python'),normalization=recipe.get('normalization','signed_uint8'));names=sorted({p.parent.name for p in self.images.paths});self.labels={k:i for i,k in enumerate(names)}
    def __len__(self):return len(self.images)
    def __getitem__(self,i):return {'images':self.images[i],'labels':torch.tensor(self.labels[self.images.paths[i].parent.name])}

class TensorShards(torch.utils.data.Dataset):
    """Local manifest + tensor shards; only one shard is resident at a time."""
    def __init__(self,path):
        self.path=Path(path).resolve();manifest=self.path/'manifest.json';value=json.loads(manifest.read_text())
        if value.get('format')!='umdiff_shards_v1' or set(value)-{'format','shards'}:raise ValueError('Expected umdiff_shards_v1 manifest with shards')
        self.entries=value['shards'];self.ends=[];self.files=[manifest];count=0
        if not self.entries:raise ValueError('Empty tensor shard manifest')
        for entry in self.entries:
            if set(entry)!={'file','count'} or not isinstance(entry['count'],int) or entry['count']<1:raise ValueError('Each shard requires file and positive sample count')
            file=(self.path/entry['file']).resolve()
            if not file.is_relative_to(self.path) or not file.is_file():raise ValueError('Shard must be an existing file inside its dataset directory')
            count+=entry['count'];self.ends.append(count);self.files.append(file)
        self.cached_index=None;self.cached_data=None
    def __len__(self):return self.ends[-1]
    def __getitem__(self,index):
        import bisect
        if not 0<=index<len(self):raise IndexError(index)
        shard=bisect.bisect_right(self.ends,index)
        if shard!=self.cached_index:
            data=OfflineTensors(self.files[shard+1])
            if len(data)!=self.entries[shard]['count']:raise ValueError('Tensor shard sample count differs from manifest')
            self.cached_data=data;self.cached_index=shard
        return self.cached_data[index-(self.ends[shard-1] if shard else 0)]

def conditions(training,batch,device,batch_size,phase='train',stream=None):
    kwargs={};context={}
    for key,spec in training.get('conditioning',{}).items():
        if spec.get('phase','both') not in ('both','train','sample'):raise ValueError('Condition phase must be both, train or sample')
        if spec.get('phase','both') not in ('both',phase):continue
        kind=spec.get('kind')
        if kind=='labels':
            if phase=='train' and 'labels' not in batch:
                raise ValueError('Conditional training requires real labels in the input batch')
            value=batch.get('labels',torch.zeros(batch_size,dtype=torch.long)).to(device)
            if value.dtype!=torch.long or value.shape!=(batch_size,):
                raise ValueError('Labels must be an int64 vector with one value per example')
            maximum=spec['classes'] if spec.get('allow_null',False) else spec['classes']-1
            if value.min()<0 or value.max()>maximum:raise ValueError('Label outside configured class range')
            probability=spec.get('dropout',0.) if phase=='train' else 0.
            if not 0<=probability<=1:raise ValueError('Invalid conditioning dropout')
            if probability:
                rng=spec.get('dropout_rng','torch')
                if rng=='python':draws=torch.tensor([random.random() for _ in range(batch_size)],device=device)
                elif rng=='torch':draws=torch.rand(batch_size,device=device) if stream is None else stream.uniform((batch_size,),device=device)
                else:raise ValueError('Condition dropout_rng must be python or torch')
                value=torch.where(draws<probability,spec.get('null_label',spec['classes']),value)
            if spec.get('one_hot',False):value=torch.nn.functional.one_hot(value,num_classes=spec['classes']+1)[...,:spec['classes']].float()
        elif kind=='field':
            if spec['field'] not in batch:raise ValueError(f'Missing local conditioning field {spec["field"]}')
            value=batch[spec['field']].to(device)
            if len(value)!=batch_size:raise ValueError('Conditioning batch size differs from --count / batch size')
            if spec.get('wrap_list',False):value=[value]
        elif kind=='mapping':
            value,extra=conditions({'conditioning':spec['fields']},batch,device,batch_size,phase,stream)
            if extra:raise ValueError('Nested conditioning mapping cannot target context')
        elif kind=='zeros':value=torch.zeros(batch_size,*spec['shape'],device=device)
        elif kind=='ones':value=torch.ones(batch_size,*spec['shape'],device=device)
        elif kind=='empty_dict':value={}
        else:raise ValueError(f'Unknown conditioning kind {kind}')
        (context if spec.get('destination')=='context' else kwargs)[key]=value
    return kwargs,context

def setup(config,device,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    method=make_method(config);model=build('model',config['model']).to(device)
    training=config.get('training',{})
    from .recipes import initialize,freeze_parameters
    initialization=training.get('initialization',[])
    if isinstance(initialization,dict):initialization=[initialization]
    model.initialization_evidence=[initialize(model,spec) for spec in initialization]
    model.frozen_parameter_names=freeze_parameters(model,training.get('freeze',[]))
    if training.get('inference_only'):raise ValueError('This paper releases inference only; use its workflow/sampler with locally supplied assets')
    trainer=Trainer(method,model,seed=seed,**training.get('optimizer',{}))
    return method,model,trainer

def train(args):
    config=read_config(args.config);device=choose_device(args.device);training=config.get('training',{})
    if args.synthetic and config.get('evidence',{}).get('synthetic_training_allowed') is False:
        raise ValueError('Paper recipe templates require local real data; use native_fixtures for synthetic checks')
    if config.get('task')=='rmd':
        from .distillation_cli import train as train_rmd
        return train_rmd(args,config,device)
    method,model,trainer=setup(config,device,args.seed)
    sampler=build('time_sampler',training.get('time_sampler',{'name':'uniform'}))
    trainer.time_sampler=sampler
    transform=build('input_transform',training['input_transform']) if training.get('input_transform') else None
    transform_scope=training.get('input_transform_scope','global_batch')
    if transform_scope not in ('global_batch','microbatch'):raise ValueError('Unknown codec invocation scope')
    shape=config.get('image_shape',[3,32,32]);batch_size=args.batch_size
    if transform_scope=='microbatch':
        def checked_transform(images,stream):
            value=transform(images,stream) if transform is not None else images
            if list(value.shape[1:])!=shape:raise ValueError(f'Microbatch codec shape {list(value.shape[1:])} differs from config {shape}')
            return value
        trainer.input_transform=checked_transform
    if args.synthetic:
        # A fixed deterministic random fixture; explicit flag prevents accidental
        # long training on random images when a local data path was forgotten.
        generator=torch.Generator().manual_seed(args.seed+1024)
        fixture=torch.randn(batch_size,*shape,generator=generator)
        dataset=None
    elif args.data:
        path=Path(args.data)
        recipe=training.get('data',{})
        if recipe.get('kind')=='relay_zip':
            from .data_pipeline import NativeRelayDataset
            dataset=NativeRelayDataset(path,**recipe.get('author_args',{}))
        elif path.is_dir():
            from .input_transforms import LocalMomentDataset
            recipe=training.get('data',{})
            if (path/'manifest.json').exists():dataset=TensorShards(path)
            elif recipe.get('kind')=='moments_npy':
                dropout=recipe.get('cfg_dropout',0.)
                if dropout and any(spec.get('dropout',0.) for spec in training.get('conditioning',{}).values()):raise ValueError('Reader CFG dropout and training conditioning dropout cannot both be enabled')
                dataset=LocalMomentDataset(path,cfg_dropout=dropout,null_label=recipe.get('null_label',1000),strict_author_layout=recipe.get('strict_author_layout',True))
            else:dataset=ImageFolder(path,recipe.get('image_size',shape[-1]),recipe)
        else:dataset=OfflineTensors(path)
        if len(dataset)<batch_size:raise ValueError('Dataset is smaller than one complete batch')
    else:raise ValueError('Provide --data LOCAL_DIRECTORY_OR_PT, or explicit --synthetic')
    from .training import data_binding
    files=dataset.images.paths if isinstance(dataset,ImageFolder) else getattr(dataset,'files',None)
    trainer.data_binding=data_binding(args.data,batch_size,args.seed,files=files,synthetic=args.synthetic)
    reader_recipe=training.get('data',{})
    reader_kind=reader_recipe.get('reader','indexed')
    if reader_kind not in ('indexed','stateful_author'):raise ValueError('Unknown data reader')
    if dataset is not None and reader_kind=='stateful_author':
        from .data_pipeline import StatefulOfflineLoader
        trainer.attach_data_pipeline(StatefulOfflineLoader(dataset,batch_size,seed=args.seed,
            shuffle=reader_recipe.get('shuffle',False),drop_last=reader_recipe.get('drop_last',False),
            num_workers=reader_recipe.get('num_workers',0),rank=reader_recipe.get('rank',0),
            world_size=reader_recipe.get('world_size',1),prefetch_factor=reader_recipe.get('prefetch_factor',2),
            persistent_workers=reader_recipe.get('persistent_workers',False)))
        trainer.data_binding['reader_contract']=trainer.data_pipeline.contract
    if args.resume:trainer.load(args.resume)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    trace=[];permutation_cache={}
    for _ in range(args.steps):
        if dataset is None:
            batch={'images':fixture}
            label_specs=[v for v in training.get('conditioning',{}).values() if v.get('kind')=='labels']
            if label_specs:batch['labels']=torch.arange(batch_size)%min(v['classes'] for v in label_specs)
        elif trainer.data_pipeline is not None:batch=next(trainer.data_pipeline)
        else:
            # Deterministic traversal indexed by the checkpoint step. No worker
            # prefetch RNG or implicit sampler state is lost on resume.
            drop_last=training.get('data',{}).get('drop_last',False)
            epoch_size=(len(dataset)//batch_size)*batch_size if drop_last else len(dataset)
            absolute=[trainer.step_index*batch_size+i for i in range(batch_size)]
            indices=[position%epoch_size for position in absolute]
            if training.get('data',{}).get('shuffle',False):
                epochs={position//epoch_size for position in absolute}
                permutation_cache={epoch:permutation_cache.get(epoch) for epoch in epochs}
                for epoch in epochs:
                    if permutation_cache[epoch] is None:permutation_cache[epoch]=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(args.seed+epoch))
                indices=[int(permutation_cache[position//epoch_size][position%epoch_size]) for position in absolute]
            batch=torch.utils.data.default_collate([dataset[i] for i in indices])
        images=batch['images'].to(device)
        kwargs,context=conditions(training,batch,device,len(images),stream=trainer.stream)
        if transform is not None and transform_scope=='global_batch':images=transform(images,trainer.stream)
        if transform_scope!='microbatch' and list(images.shape[1:])!=shape:raise ValueError(f'Input shape {list(images.shape[1:])} differs from config {shape}; declare training.input_transform for a local latent codec')
        if transform_scope=='microbatch':
            result=trainer.step(images,None,model_kwargs=kwargs,context=context,time_sampler=sampler)
        else:
            t,time_context=sampler(len(images),device,trainer.stream);context.update(time_context)
            result=trainer.step(images,t,model_kwargs=kwargs,context=context)
        row={'step':trainer.step_index,'loss':result.loss.mean().item()};trace.append(row)
        if trainer.step_index%args.log_every==0 or _==args.steps-1:print(json.dumps(row),flush=True)
        if args.save_every and trainer.step_index%args.save_every==0:trainer.save(output/'checkpoint.pt')
    trainer.save(output/'checkpoint.pt');(output/'trace.json').write_text(json.dumps(trace,indent=2))
    print(json.dumps({'checkpoint':str(output/'checkpoint.pt'),'device':str(device),'steps':trainer.step_index,'synthetic':args.synthetic}))

@torch.no_grad()
def sample(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=checkpoint['config'];device=choose_device(args.device)
    if config.get('task')=='rmd':
        from .distillation_cli import sample as sample_rmd
        return sample_rmd(args,checkpoint,device)
    method=make_method(config);model=build('model',config['model']).to(device);model.load_state_dict(checkpoint['model' if args.raw_weights else 'ema']);model.eval()
    stream=RandomStream(args.seed,device);shape=config['image_shape'];name=config.get('sampler',{}).get('name')
    initial_shape=list(shape)
    if hasattr(method,'initial_shape'):initial_shape=method.initial_shape()
    elif getattr(method,'sampler',None) is not None and hasattr(method.sampler,'initial_shape'):initial_shape=method.sampler.initial_shape(shape)
    if config.get('task')=='codec':initial_shape=list(config['sampling_shape'])
    if name=='pixelflow':
        factor=2**(method.sampler.stages-1);initial_shape[-2:]=[n//factor for n in shape[-2:]]
    elif name=='ssd':initial_shape[-2:]=[method.sampler.batch.resolutions[-1]]*2
    elif name=='relay_chain':initial_shape[-2:]=[model['low'].img_resolution]*2
    elif name=='rdd_native_stages':initial_shape[-2:]=[method.sampler.resolutions[-1]]*2
    initial=stream.normal(torch.empty(args.count,*initial_shape,device=device))
    if getattr(method,'batch',None) is not None and name is None and hasattr(method.clock,'schedule'):
        clean=method.representation.encode(initial);initial=dict(enumerate(stream.normal(x) for x in clean))
    elif config.get('representation',{}).get('name') in ('dct_ycbcr','dct_ycbcr_grid'):
        initial=stream.normal(method.representation.encode(initial))
    training=config.get('training',{});batch={}
    if args.conditioning:
        batch=torch.load(args.conditioning,map_location='cpu',weights_only=True)
        if not isinstance(batch,dict) or not all(torch.is_tensor(v) for v in batch.values()):raise ValueError('Local sample conditioning must be a dictionary of tensors')
    if args.labels:batch['labels']=torch.tensor(args.labels if len(args.labels)>1 else args.labels*args.count)
    if 'labels' in batch and len(batch['labels'])!=args.count:raise ValueError('Provide one label or exactly --count labels')
    kwargs,context=conditions(training,batch,device,args.count,phase='sample')
    if 'class_labels' in context:kwargs['class_labels']=context['class_labels']
    if args.steps is not None and getattr(method,'sampler',None) is not None:
        raise ValueError('This method uses its configured paper sampler. Change sampler.args in the training config; --steps applies to the generic solver only.')
    result=method.sample(model,initial,steps=args.steps or 20,stream=stream,model_kwargs=kwargs)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(map_tree(lambda x:x.cpu(),result),output)
    if torch.is_tensor(result) and result.ndim==4 and result.shape[1] in (1,3):
        from PIL import Image
        pixels=result if result.dtype==torch.uint8 else ((result.clamp(-1,1)+1)*127.5).round().to(torch.uint8)
        for i,pixel in enumerate(pixels.cpu()):
            array=pixel.permute(1,2,0).numpy();Image.fromarray(array[...,0] if pixel.shape[0]==1 else array).save(output.with_name(output.stem+f'_{i}.png'))
    print(json.dumps({'output':str(output),'device':str(device),'space':'RGB pixels' if torch.is_tensor(result) and result.ndim==4 and result.shape[1]==3 else 'configured representation/latent; supply a local decoder for RGB'}))

@torch.no_grad()
def decode(args):
    from .input_transforms import local_autoencoder
    device=choose_device(args.device);model=local_autoencoder(args.vae).to(device).eval()
    latents=torch.load(args.latents,map_location=device,weights_only=True)
    if not torch.is_tensor(latents) or latents.ndim!=4 or latents.shape[1]!=model.config.latent_channels:raise ValueError('Image VAE decoding requires a BCHW latent tensor matching the local codec')
    scale=args.scale if args.scale is not None else model.config.scaling_factor
    if scale<=0:raise ValueError('Latent scaling must be positive')
    shift=getattr(model.config,'shift_factor',None) or 0.
    pixels=model.decode(latents.float()/scale+shift,return_dict=False)[0]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(pixels.cpu(),output)
    from PIL import Image
    for i,pixel in enumerate(((pixels.clamp(-1,1)+1)*127.5).round().byte().cpu()):Image.fromarray(pixel.permute(1,2,0).numpy()).save(output.with_name(output.stem+f'_{i}.png'))
    print(json.dumps({'output':str(output),'vae':str(Path(args.vae).resolve()),'scale':scale,'shift':shift,'space':'decoded RGB'}))

def infer(args):
    from .workflows import run
    config=read_config(args.config);device=choose_device(args.device)
    if args.embeddings:
        embeddings=torch.load(args.embeddings,map_location='cpu',weights_only=True)
        if not isinstance(embeddings,dict) or not all(torch.is_tensor(x) for x in embeddings.values()):raise ValueError('Embeddings must be a local dictionary of tensors')
    elif args.synthetic_embeddings:
        if not config.get('synthetic_fixture'):raise ValueError('Synthetic embeddings require an explicitly marked fixture config')
        shape=config['synthetic_embedding_shape'];generator=torch.Generator().manual_seed(args.seed+1024)
        embeddings={'prompt_embeds':torch.randn(shape,generator=generator),'pooled_prompt_embeds':torch.randn(shape[0],shape[-1],generator=generator)}
        if config['name']=='nami':
            embeddings.update(negative_prompt_embeds=torch.randn(shape,generator=generator),negative_pooled_prompt_embeds=torch.randn(shape[0],shape[-1],generator=generator))
        elif config['name']=='mrflow_qwen':
            embeddings={'prompt_embeds':embeddings['prompt_embeds'],'prompt_embeds_mask':torch.ones(shape[:2],dtype=torch.long),
                'negative_prompt_embeds':torch.randn(shape,generator=generator),'negative_prompt_embeds_mask':torch.ones(shape[:2],dtype=torch.long)}
        elif config['name']=='speed_pixelgen':embeddings={'cond':embeddings['prompt_embeds'],'uncond':torch.randn(shape,generator=generator)}
        elif config['name']=='speed_wan':embeddings={'ctx_cond':embeddings['prompt_embeds'],'ctx_uncond':torch.randn(shape,generator=generator)}
    else:raise ValueError('Supply --embeddings LOCAL_PT or explicit --synthetic-embeddings')
    result=run(config,embeddings,device,args.seed)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(map_tree(lambda x:x.cpu(),result),output)
    from PIL import Image
    for key,value in result.items():
        if key=='latents':continue
        pixels=value.cpu() if value.dtype==torch.uint8 else ((value.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).cpu()
        for i,pixel in enumerate(pixels):
            frames=pixel.permute(1,0,2,3) if pixels.ndim==5 else pixel[None]
            for j,frame in enumerate(frames):Image.fromarray(frame.permute(1,2,0).numpy()).save(output.with_name(output.stem+f'_{key}_{i}'+(f'_frame{j}' if pixels.ndim==5 else '')+'.png'))
    print(json.dumps({'output':str(output),'device':str(device),'synthetic_fixture':config.get('synthetic_fixture',False)}))

def main():
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=2);sub=parser.add_subparsers(dest='command',required=True)
    from .cross_modal import configure,METHODS
    p=sub.add_parser('configure');p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--shape',default='3,32,32',help='C,H,W or C,T,H,W in model domain');p.add_argument('--backbone',default='grid_dit',choices=['grid_dit','dit_grid','wan_grid','grid_conv']);p.add_argument('--width',type=int,default=96);p.add_argument('--depth',type=int,default=2);p.add_argument('--attention-heads',type=int,default=4);p.add_argument('--patch',default='1,2,2');p.add_argument('--position',choices=['sincos','rope'],default='sincos');p.add_argument('--grids',help='JSON coarse-to-fine grids');p.add_argument('--first-frame',action='store_true');p.add_argument('--separate-stages',action='store_true');p.add_argument('--decoupled',action='store_true');p.add_argument('--model-json');p.add_argument('--process-args',default='{}');p.add_argument('--sampling-steps',type=int,default=20);p.add_argument('--inference-policy',choices=['bottleneck','speed']);p.add_argument('--inference-args',default='{}');p.add_argument('--output',required=True);p.set_defaults(run=configure)
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--data');p.add_argument('--synthetic',action='store_true');p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--steps',type=int,default=20);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--save-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=10);p.set_defaults(run=train)
    p=sub.add_parser('sample');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--conditioning');p.add_argument('--count',type=int,default=2);p.add_argument('--steps',type=int);p.add_argument('--seed',type=int,default=123);p.add_argument('--device',default='cpu');p.add_argument('--labels',type=int,nargs='+');p.add_argument('--raw-weights',action='store_true');p.set_defaults(run=sample)
    p=sub.add_parser('infer');p.add_argument('--config',required=True);p.add_argument('--embeddings');p.add_argument('--synthetic-embeddings',action='store_true');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=123);p.add_argument('--device',default='cpu');p.set_defaults(run=infer)
    p=sub.add_parser('decode');p.add_argument('--latents',required=True);p.add_argument('--vae',required=True);p.add_argument('--scale',type=float);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.set_defaults(run=decode)
    p=sub.add_parser('components');p.set_defaults(run=lambda args:print(json.dumps({g:sorted(k[1] for k in REGISTRY if k[0]==g) for g in sorted(set(k[0] for k in REGISTRY))},indent=2)))
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if (getattr(args,'steps',None) is not None and args.steps<1) or getattr(args,'log_every',1)<1 or getattr(args,'count',1)<1 or getattr(args,'batch_size',1)<1:parser.error('Steps, batch/count and log interval must be positive')
    args.run(args)

if __name__=='__main__':main()
