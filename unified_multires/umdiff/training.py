"""Deterministic optimizer/EMA state, atomic checkpoints and offline data."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import os,random,hashlib
import numpy as np
import torch
from .randomness import RandomStream,GlobalRandomStream

def data_binding(path,batch_size,seed,files=None,synthetic=False):
    """Resume identity: path/file metadata, batch and traversal seed.

    This detects ordinary replacement/reordering; it is not a content hash of
    large datasets. External datasets should additionally have a frozen manifest.
    """
    result={'batch_size':batch_size,'seed':seed,'synthetic':synthetic}
    if synthetic:return result
    path=Path(path).resolve();digest=hashlib.sha256();count=0
    for file in files or [path]:
        file=Path(file).resolve();stat=file.stat()
        digest.update((str(file)+'\0'+str(stat.st_size)+'\0'+str(stat.st_mtime_ns)+'\n').encode());count+=1
    result.update(path=str(path),file_count=count,metadata_sha256=digest.hexdigest())
    return result

class Trainer:
    def __init__(self,method,model,lr=1e-3,betas=(.9,.999),weight_decay=0.,ema_decay=.999,grad_clip=None,seed=0,optimizer_name='adamw',eps=1e-8,lr_schedule=None,precision='float32',accumulation_steps=1,ema_update_every=1,ema_warmup=False,loss_reduction='mean',ema_style='add_alpha',randomness='separate',loss_multiplier=1.,ema_schedule=None,image_count_multiplier=1,gradient_repair=None,accumulation_normalization='mean',optimizer_parameters='trainable',input_transform=None,distributed_sync='every_microbatch',nonfinite_policy='raise',ema_on_skipped_step=True):
        self.method=method;self.model=model
        factory={'adamw':torch.optim.AdamW,'adam':torch.optim.Adam}.get(optimizer_name)
        if factory is None:raise ValueError('Supported optimizers: adamw, adam')
        if optimizer_parameters not in ('trainable','all'):raise ValueError('Unknown optimizer parameter policy')
        parameters=[p for p in model.parameters() if p.requires_grad or optimizer_parameters=='all']
        if not any(p.requires_grad for p in parameters):raise ValueError('No trainable parameters')
        self.optimizer=factory(parameters,lr=lr,betas=tuple(betas),weight_decay=weight_decay,eps=eps)
        self.ema=deepcopy(model).eval().requires_grad_(False)
        self.decay=ema_decay;self.grad_clip=grad_clip;self.step_index=0;self.data_binding=None;self.images_seen=0
        if gradient_repair is not None and (set(gradient_repair)!={'nan','posinf','neginf'} or precision=='float16'):raise ValueError('Gradient repair requires nan/posinf/neginf and a non-scaled precision policy')
        self.gradient_repair=gradient_repair
        if randomness not in ('separate','native_global'):raise ValueError('Unknown randomness policy')
        self.stream=RandomStream(seed,next(model.parameters()).device) if randomness=='separate' else GlobalRandomStream()
        if accumulation_steps<1 or ema_update_every<1:raise ValueError('Accumulation/EMA interval must be positive')
        if accumulation_normalization not in ('mean','sum'):raise ValueError('Unknown accumulation normalization')
        self.accumulation_normalization=accumulation_normalization
        if precision not in ('float32','bfloat16','float16'):raise ValueError('Unknown training precision')
        if loss_reduction not in ('mean','sum','normalized_weights'):raise ValueError('Unknown training loss reduction')
        if ema_style not in ('add_alpha','scaled_add','lerp','source_lerp'):raise ValueError('Unknown EMA arithmetic')
        if loss_multiplier<=0:raise ValueError('Loss multiplier must be positive')
        if not isinstance(image_count_multiplier,int) or image_count_multiplier<1:raise ValueError('Image count multiplier must be a positive integer')
        if ema_schedule is not None and (ema_schedule.get('name')!='image_halflife' or ema_schedule.get('half_life_images',0)<=0 or ema_update_every!=1):raise ValueError('Image half-life EMA requires a positive half life and an update every step')
        self.loss_multiplier=loss_multiplier;self.ema_schedule=ema_schedule;self.image_count_multiplier=image_count_multiplier
        self.ema_style=ema_style
        self.base_lr=lr;self.schedule=lr_schedule;self.precision=precision;self.accumulation=accumulation_steps;self.ema_every=ema_update_every;self.ema_warmup=ema_warmup;self.reduction=loss_reduction
        if self.schedule and self.schedule.get('update_timing','before_step') not in ('before_step','after_step'):raise ValueError('Unknown LR update timing')
        self.recipe={'lr':lr,'betas':list(betas),'weight_decay':weight_decay,'optimizer_name':optimizer_name,'eps':eps,'lr_schedule':lr_schedule,'precision':precision,'accumulation_steps':accumulation_steps,'ema_update_every':ema_update_every,'ema_warmup':ema_warmup,'loss_reduction':loss_reduction,'ema_style':ema_style,'randomness':randomness,'loss_multiplier':loss_multiplier,'ema_schedule':ema_schedule,'image_count_multiplier':image_count_multiplier}
        self.recipe['gradient_repair']=gradient_repair
        self.recipe['accumulation_normalization']=accumulation_normalization
        self.recipe['optimizer_parameters']=optimizer_parameters
        if distributed_sync not in ('every_microbatch','last_microbatch'):raise ValueError('Unknown distributed synchronization policy')
        if distributed_sync=='last_microbatch' and not hasattr(model,'no_sync'):raise ValueError('Last-microbatch synchronization requires a distributed no_sync model')
        if nonfinite_policy not in ('raise','grad_scaler') or (nonfinite_policy=='grad_scaler' and precision!='float16'):raise ValueError('GradScaler non-finite policy requires float16 scaling')
        self.distributed_sync=distributed_sync;self.nonfinite_policy=nonfinite_policy;self.ema_on_skip=ema_on_skipped_step
        self.recipe.update(distributed_sync=distributed_sync,nonfinite_policy=nonfinite_policy,ema_on_skipped_step=ema_on_skipped_step)
        self.input_transform=input_transform;self.data_pipeline=None;self.pending_reader_state=None;self.time_sampler=None
        device=next(model.parameters()).device
        self.scaler=torch.amp.GradScaler(device.type,enabled=precision=='float16')
    def attach_data_pipeline(self,pipeline):
        if self.data_pipeline is not None:raise ValueError('A data pipeline is already attached')
        if self.pending_reader_state is not None:
            pipeline.load_state_dict(self.pending_reader_state);self.pending_reader_state=None
        self.data_pipeline=pipeline
    def step(self,images,t,noise=None,model_kwargs=None,active=None,context=None,time_sampler=None,before_gradient_processing=None):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        from .tree import map_tree
        from .recipes import lr_factor
        if self.schedule and self.schedule.get('update_timing','before_step')=='before_step':
            for group in self.optimizer.param_groups:group['lr']=self.base_lr*lr_factor(self.step_index,self.schedule,images_seen=self.images_seen)
        if len(images)%self.accumulation:raise ValueError('Batch must divide accumulation_steps exactly')
        width=len(images)//self.accumulation;results=[]
        def sliced(value,a,z):
            if torch.is_tensor(value):return value[a:z] if value.ndim and len(value)==len(images) else value
            if isinstance(value,dict):return {k:sliced(v,a,z) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return type(value)(sliced(v,a,z) for v in value)
            return value
        for i in range(self.accumulation):
            a=i*width;z=a+width;ctx=dict(sliced(context or {},a,z));ctx.setdefault('step',self.step_index)
            micro_images=images[a:z]
            if self.input_transform is not None:micro_images=self.input_transform(micro_images,self.stream)
            micro_t=sliced(t,a,z)
            if time_sampler is not None:
                micro_t,time_context=time_sampler(len(micro_images),micro_images.device,self.stream);ctx.update(time_context)
            sync_scope=self.model.no_sync() if self.distributed_sync=='last_microbatch' and i<self.accumulation-1 else nullcontext()
            with sync_scope:
              with torch.autocast(images.device.type,dtype={'bfloat16':torch.bfloat16,'float16':torch.float16}.get(self.precision,torch.bfloat16),enabled=self.precision!='float32'):
                result=self.method.training(self.model,micro_images,micro_t,self.stream,sliced(noise,a,z),sliced(model_kwargs,a,z),sliced(active,a,z),ctx)
                if self.reduction=='normalized_weights':
                    if 'loss_weights' not in ctx:raise ValueError('Normalized-weight loss requires per-example loss_weights')
                    loss=(result.loss*ctx['loss_weights']).sum()/context['loss_weights'].sum()
                elif self.reduction=='sum':loss=result.loss.sum()
                else:loss=result.loss.mean()/(self.accumulation if self.accumulation_normalization=='mean' else 1)
                if self.loss_multiplier!=1.:loss=loss*self.loss_multiplier
              if not torch.isfinite(loss) and self.nonfinite_policy=='raise':raise FloatingPointError('Non-finite training loss')
              self.scaler.scale(loss).backward();results.append(result)
        self.scaler.unscale_(self.optimizer)
        if before_gradient_processing is not None:before_gradient_processing(self.model)
        if self.gradient_repair is not None:
            for param in self.model.parameters():
                if param.grad is not None:torch.nan_to_num(param.grad,out=param.grad,**self.gradient_repair)
        for parameter in self.model.parameters():
            if self.nonfinite_policy=='raise' and parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise FloatingPointError('Non-finite gradient')
        if self.grad_clip is not None:torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip,error_if_nonfinite=self.nonfinite_policy=='raise')
        scale_before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
        self.last_update_skipped=self.scaler.get_scale()<scale_before
        if self.schedule and self.schedule.get('update_timing','before_step')=='after_step':
            for group in self.optimizer.param_groups:group['lr']=self.base_lr*lr_factor(self.step_index,self.schedule,images_seen=self.images_seen)
        self.step_index+=1
        with torch.no_grad():
            if self.step_index%self.ema_every==0 and (not self.last_update_skipped or self.ema_on_skip):
                decay=min(self.decay,(1+self.step_index)/(10+self.step_index)) if self.ema_warmup else self.decay
                if self.ema_schedule is not None:
                    half_life=self.ema_schedule['half_life_images'];ramp=self.ema_schedule.get('rampup_ratio')
                    if ramp is not None:half_life=min(half_life,self.images_seen*ramp)
                    decay=.5**(len(images)*self.image_count_multiplier/max(half_life,1e-8))
                for target,source in zip(self.ema.parameters(),self.model.parameters()):
                    if self.ema_style=='scaled_add':target.mul_(decay).add_((1-decay)*source)
                    elif self.ema_style=='source_lerp':target.copy_(source.detach().lerp(target,decay))
                    elif self.ema_style=='lerp':target.lerp_(source,1-decay)
                    else:target.mul_(decay).add_(source,alpha=1-decay)
                for target,source in zip(self.ema.buffers(),self.model.buffers()):target.copy_(source)
        self.images_seen+=len(images)*self.image_count_multiplier
        if len(results)>1:
            from .engine import TrainingResult
            def joined(values):
                value=values[0]
                if torch.is_tensor(value):
                    if value.ndim and all(v.shape[1:]==value.shape[1:] for v in values):return torch.cat(values)
                    if value.ndim==0:return torch.stack(values)
                    return tuple(values)
                if isinstance(value,dict):
                    if any(v.keys()!=value.keys() for v in values):return tuple(values)
                    return {k:joined([v[k] for v in values]) for k in value}
                if isinstance(value,(tuple,list)):
                    if any(len(v)!=len(value) for v in values):return tuple(values)
                    return type(value)(joined([v[i] for v in values]) for i in range(len(value)))
                return value if all(v is value for v in values) else tuple(values)
            result=TrainingResult(torch.cat([r.loss.reshape(-1) for r in results]),joined([r.prediction for r in results]),joined([r.target for r in results]),joined([r.state for r in results]),joined([r.terms for r in results]),tuple(results))
        return result
    def state_dict(self):
        return {'format':1,'config':self.method.config,'model':self.model.state_dict(),'ema':self.ema.state_dict(),
                'optimizer':self.optimizer.state_dict(),'step':self.step_index,'stream':self.stream.state_dict(),
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                'trainer':{'ema_decay':self.decay,'grad_clip':self.grad_clip},'recipe':self.recipe,'scaler':self.scaler.state_dict(),'data_binding':self.data_binding,'images_seen':self.images_seen,
                'reader_state':self.data_pipeline.state_dict() if self.data_pipeline is not None else None,
                'method_state':self.method.state_dict() if hasattr(self.method,'state_dict') else None,
                'time_sampler_state':self.time_sampler.state_dict() if self.time_sampler is not None and hasattr(self.time_sampler,'state_dict') else None}
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(self.state_dict(),temporary);temporary.replace(path)
    def load(self,path):
        # Only load local checkpoints you trust; optimizer RNG includes NumPy.
        state=torch.load(path,map_location=next(self.model.parameters()).device,weights_only=False)
        if state['format']!=1 or state['config']!=self.method.config:raise ValueError('Checkpoint method configuration mismatch')
        if self.data_binding is not None and state.get('data_binding')!=self.data_binding:raise ValueError('Checkpoint data identity, batch size or traversal seed mismatch')
        self.data_binding=state.get('data_binding')
        if state['trainer']!={'ema_decay':self.decay,'grad_clip':self.grad_clip}:raise ValueError('Checkpoint trainer settings mismatch')
        saved_recipe=dict(state.get('recipe',self.recipe));saved_recipe.setdefault('ema_style','add_alpha')
        saved_recipe.setdefault('randomness','separate')
        saved_recipe.setdefault('loss_multiplier',1.);saved_recipe.setdefault('ema_schedule',None)
        saved_recipe.setdefault('image_count_multiplier',1)
        saved_recipe.setdefault('gradient_repair',None)
        saved_recipe.setdefault('accumulation_normalization','mean')
        saved_recipe.setdefault('optimizer_parameters','trainable')
        saved_recipe.setdefault('distributed_sync','every_microbatch')
        saved_recipe.setdefault('nonfinite_policy','raise');saved_recipe.setdefault('ema_on_skipped_step',True)
        if saved_recipe!=self.recipe:raise ValueError('Checkpoint optimizer/precision/LR recipe mismatch')
        if state.get('method_state') is None and getattr(self.method,'requires_checkpoint_state',False):
            raise ValueError('Checkpoint is missing required method sampler history')
        if state.get('time_sampler_state') is None and self.time_sampler is not None and getattr(self.time_sampler,'requires_checkpoint_state',hasattr(self.time_sampler,'state_dict')):
            raise ValueError('Checkpoint is missing required time sampler history')
        self.model.load_state_dict(state['model']);self.ema.load_state_dict(state['ema']);self.optimizer.load_state_dict(state['optimizer'])
        self.step_index=state['step'];self.images_seen=state.get('images_seen',0);self.stream.load_state_dict(state['stream'])
        self.scaler.load_state_dict(state.get('scaler',{}))
        if state.get('method_state') is not None:
            if not hasattr(self.method,'load_state_dict'):raise ValueError('Checkpoint contains unsupported method state')
            self.method.load_state_dict(state['method_state'])
        if state.get('time_sampler_state') is not None:
            if self.time_sampler is None or not hasattr(self.time_sampler,'load_state_dict'):raise ValueError('Attach the stateful time sampler before loading its checkpoint')
            self.time_sampler.load_state_dict(state['time_sampler_state'])
        if self.data_pipeline is not None:
            if state.get('reader_state') is None:raise ValueError('Stateful reader checkpoint has no reader state')
            self.data_pipeline.load_state_dict(state['reader_state'])
        else:self.pending_reader_state=state.get('reader_state')
        torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])

class LocalImages(torch.utils.data.Dataset):
    def __init__(self,directory,size=32,crop='legacy',horizontal_flip=False,expand_ratio=1.125,flip_rng='python',normalization='signed_uint8'):
        self.paths=sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
        self.size=size;self.crop=crop;self.flip=horizontal_flip;self.flip_rng=flip_rng
        if crop not in ('legacy','adm_center','pixelflow_random','pre_resized'):raise ValueError('Unknown image crop recipe')
        if flip_rng not in ('python','torch'):raise ValueError('Unknown horizontal flip RNG')
        if normalization not in ('signed_uint8','torchvision'):raise ValueError('Unknown pixel normalization')
        self.normalize=normalization
        self.transform=None
        if crop=='pixelflow_random':
            from torchvision import transforms
            self.transform=transforms.Compose([transforms.Resize(round(size*expand_ratio),interpolation=transforms.InterpolationMode.LANCZOS),transforms.RandomCrop(size)])
        if not self.paths:raise ValueError('No local images found; no data is downloaded')
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            image=image.convert('RGB')
            if self.crop=='pre_resized':
                if image.size!=(self.size,self.size):raise ValueError('Native pre-resized data must have the declared image dimensions')
            elif self.crop=='adm_center':
                while min(image.size)>=2*self.size:image=image.resize(tuple(n//2 for n in image.size),resample=Image.Resampling.BOX)
                scale=self.size/min(image.size);image=image.resize(tuple(round(n*scale) for n in image.size),resample=Image.Resampling.BICUBIC)
                array=np.array(image);top=(array.shape[0]-self.size)//2;left=(array.shape[1]-self.size)//2;image=Image.fromarray(array[top:top+self.size,left:left+self.size])
            elif self.transform is not None:image=self.transform(image)
            else:
                w,h=image.size;side=min(w,h);image=image.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2)).resize((self.size,self.size),Image.Resampling.BICUBIC)
            if self.flip:
                draw=random.random() if self.flip_rng=='python' else torch.rand(1).item()
                if draw<.5:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.normalize=='torchvision':
                from torchvision import transforms
                return transforms.Normalize([.5]*3,[.5]*3,inplace=True)(transforms.ToTensor()(image))
            array=np.array(image,dtype=np.float32)/127.5-1
        return torch.from_numpy(array).permute(2,0,1)
