"""Composable training objective and stagewise/product-state inference."""
from dataclasses import dataclass
import torch
from .registry import build
from .tree import first,map_tree,mean_flat

@dataclass
class TrainingResult:
    loss:torch.Tensor
    prediction:object
    target:object
    state:object
    terms:dict
    microbatches:tuple=()

class Method:
    def __init__(self,config):
        allowed={'name','representation','noise','path','paths','target','metric','clock','solver','stages','leaf_weights','evidence','training','model','image_shape','batch','precondition','prediction_adapter','sampler','guidance','training_stages','training_stage_reduction','training_aggregation'}
        if set(config)-allowed:raise ValueError(f'Unknown method fields: {set(config)-allowed}')
        self.config=config
        self.training_stages=[]
        self.stage_reduction=config.get('training_stage_reduction','sum_vectors')
        if self.stage_reduction not in ('sum_vectors','sum_stage_means'):raise ValueError('Unknown training stage reduction')
        if 'training_stage_reduction' in config and 'training_stages' not in config:raise ValueError('Stage reduction requires a training stage program')
        if 'training_stages' in config:
            if not config['training_stages']:raise ValueError('Empty training stage program')
            if set(config)&{'batch','path','paths','target','metric'}:raise ValueError('Training stage program owns its objective components')
            for spec in config['training_stages']:
                if set(spec)-{'method','input','time_sampler','weight'}:raise ValueError('Unknown training stage fields')
                self.training_stages.append((Method(spec['method']),build('representation',spec.get('input',{'name':'identity'})),build('time_sampler',spec['time_sampler']),spec.get('weight',1.)))
        self.representation=build('representation',config.get('representation',{'name':'identity'}))
        self.noise=build('noise',config.get('noise',{'name':'iid'}))
        self.target=build('target',config.get('target',{'name':'joint'}))
        self.metric=build('metric',config.get('metric',{'name':'mse'}))
        self.paths=[build('path',p) for p in config['paths']] if 'paths' in config else [build('path',config.get('path',{'name':'flow'}))]
        self.clock=build('clock',config.get('clock',{'name':'linear'}))
        self.solver=build('solver',config.get('solver',{'name':'euler'}))
        self.weights=config.get('leaf_weights',[1.]*len(self.paths))
        self.batch=build('batch',config['batch']) if 'batch' in config else None
        self.aggregation_samplers=[]
        if 'training_aggregation' in config:
            policy=config['training_aggregation']
            if self.batch is None or self.training_stages:
                raise ValueError('Batch aggregation requires a batch objective and no training_stages')
            if set(policy)!={'name','samplers'} or policy['name']!='sum_stage_means' or not policy['samplers']:
                raise ValueError('Training aggregation requires sum_stage_means and nonempty phase samplers')
            self.aggregation_samplers=[build('time_sampler',s) for s in policy['samplers']]
        if config.get('batch',{}).get('name')=='pixelflow' and 'model' in config:
            spec=config['model']
            if spec['name']=='adapter':spec=spec['args']['model']
            if spec['name']=='native_pixelflow':
                params=spec.get('args',{})
                if self.batch.patch!=params.get('patch_size',2) or self.batch.head_dim!=params.get('attention_head_dim',16):raise ValueError('PixelFlow packing patch and rotary dimension must match the network')
        if self.batch is not None:
            ignored=set(config)&{'noise','path','paths','metric'}
            if ignored:raise ValueError(f'Native batch owns {ignored}; configure its arguments or use the generic composable path engine')
            if 'target' in config:
                expected={'relay':('joint','clean')}.get(config['batch']['name'])
                actual=(config['target']['name'],config['target'].get('args',{}).get('quantity'))
                if actual!=expected:raise ValueError('Native batch has fixed author prediction semantics; use the generic path/target components for a controlled target transplant')
        self.sampler=build('sampler',config['sampler']) if 'sampler' in config else None
        self.guidance=build('guidance',config['guidance']) if 'guidance' in config else None
        if self.guidance is not None and (self.sampler is not None or not hasattr(self.clock,'schedule')):raise ValueError('Window guidance requires the product-clock solver; sampler policies belong in sampler.args')
        if self.batch is not None and self.sampler is not None:
            if hasattr(self.batch,'gammas') and hasattr(self.sampler,'gammas'):
                if not torch.equal(self.batch.gammas,self.sampler.gammas) or list(self.batch.scales)!=list(self.sampler.scales) or self.batch.shift!=self.sampler.shift or self.batch.power!=self.sampler.power:raise ValueError('Nested train/sample schedule mismatch')
                if self.batch.prediction!=self.sampler.quantity:raise ValueError('Nested train/sample prediction parameterization mismatch')
            if config['batch']['name']=='lapflow' and config['sampler']['name']=='lapflow':
                if self.batch.levels!=self.sampler.levels or self.batch.activations!=self.sampler.activations or self.batch.noise_factor!=self.sampler.noise_factor or self.batch.variance!=self.sampler.variance or config['batch'].get('args',{}).get('curve','linear')!=self.sampler.curve or config['batch'].get('args',{}).get('epsilon',0.)!=0.:
                    raise ValueError('LapFlow train/sample path, noise or variance contract mismatch')
            if config['batch']['name']=='ssd' and config['sampler']['name']=='ssd':
                if self.batch.resolutions!=self.sampler.batch.resolutions or self.batch.rescale!=self.sampler.batch.rescale:raise ValueError('SSD train/sample clock mismatch')
            if config['batch']['name']=='rdd_ddpm' and config['sampler']['name']=='rdd_ddpm':
                import numpy as np
                a=self.batch.path;b=self.sampler.path
                if not np.array_equal(a.betas,b.betas) or not np.array_equal(a.timestep_map,b.timestep_map) or (a.drift_start,a.drift_end,a.multiplier)!=(b.drift_start,b.drift_end,b.multiplier) or self.batch.learned!=self.sampler.learned or self.batch.decoupled!=self.sampler.decoupled:raise ValueError('RDD train/sample path or prediction-head mismatch')
            if config['batch']['name']=='relay' and config['sampler']['name']=='relay':
                for a,b in [('mean','mean'),('std','std'),('floor','floor'),('blur','blur'),('patch','patch'),('length','prob')]:
                    if getattr(self.batch,a)!=getattr(self.sampler,b):raise ValueError(f'Relay train/sample clock mismatch: {a}')
        self.precondition=build('precondition',config['precondition']) if 'precondition' in config else None
        if len(self.weights)!=len(self.paths):raise ValueError('Leaf weights mismatch')
        self.stages=[]
        for s in config.get('stages',[]):
            if set(s)-{'path','clock','boundary','steps','model_stage'}:raise ValueError('Unknown stage fields')
            if 'clock' not in s or s.get('steps',20)<1:raise ValueError('Each stage requires a clock and positive step count')
            self.stages.append({k:build(k,v) if k in ('path','clock','boundary','representation') else v for k,v in s.items()})
    def training(self,model,images,t,stream,noise=None,model_kwargs=None,active=None,context=None):
        if not self.aggregation_samplers:
            return self._training_once(model,images,t,stream,noise,model_kwargs,active,context)
        if t is not None or noise is not None or active is not None:
            raise ValueError('All-stage aggregation owns its time/noise and active-mode draws')
        if context is not None and set(context)&{'stage','importance_weights'}:
            raise ValueError('All-stage aggregation owns phase selection and outer weights')
        results=[];terms={}
        for i,sampler in enumerate(self.aggregation_samplers):
            time,extra=sampler(len(images),images.device,stream)
            result=self._training_once(model,images,time,stream,model_kwargs=model_kwargs,context={**(context or {}),**extra})
            results.append(result)
            terms.update({f'stage{i}/{k}':v for k,v in result.terms.items()})
            terms[f'stage{i}/objective']=result.loss.mean()
        # RDD contract: batch mean per stage, then SUM; no 1/K or p(stage).
        loss=sum(r.loss.mean() for r in results)
        return TrainingResult(loss,[r.prediction for r in results],[r.target for r in results],[r.state for r in results],terms)

    def _training_once(self,model,images,t,stream,noise=None,model_kwargs=None,active=None,context=None):
        if self.training_stages:
            if t is not None or noise is not None:raise ValueError('Training stages own their time/noise draws; use their configured samplers')
            results=[];terms={}
            for i,(method,projection,sampler,weight) in enumerate(self.training_stages):
                x=projection.encode(images);clock,extra=sampler(len(x),x.device,stream);ctx={**(context or {}),**extra}
                result=method.training(model,x,clock,stream,model_kwargs=model_kwargs,active=active,context=ctx);results.append((weight,result))
                terms.update({f'stage{i}/{k}':v for k,v in result.terms.items()})
            loss=sum(w*(r.loss.mean() if self.stage_reduction=='sum_stage_means' else r.loss) for w,r in results)
            return TrainingResult(loss,[r.prediction for w,r in results],[r.target for w,r in results],[r.state for w,r in results],terms)
        if self.batch is not None:
            prepared=self.batch(images,t,stream,noise,context)
            kwargs={**prepared.model_kwargs,**(model_kwargs or {})}
            network_time=prepared.time if prepared.network_time is None else prepared.network_time
            counts=prepared.metadata.get('active_counts')
            model_state=prepared.state if counts is None else [x[:n] for x,n in zip(prepared.state,counts)]
            pred=self.precondition(model,model_state,network_time,**kwargs) if self.precondition else model(model_state,network_time,**kwargs)
            if counts is not None:
                pred=[torch.cat([p,p.new_zeros(len(prepared.state[i])-len(p),*p.shape[1:])]) for i,p in enumerate(pred)]
            if 'prediction_adapter' in self.config:
                pred=build('prediction_adapter',self.config['prediction_adapter'])(pred,prepared)
            loss,terms=prepared.loss(pred)
            if context is not None and 'importance_weights' in context and not prepared.metadata.get('handles_importance_weights'):loss=loss*context['importance_weights']
            return TrainingResult(loss,pred,prepared.target,prepared.state,terms)
        clean=self.representation.encode(images)
        eps=self.noise(clean,stream) if noise is None else noise
        product=isinstance(clean,(list,tuple))
        if product:
            if len(clean)!=len(self.paths):raise ValueError('Path count must match representation leaves')
            samples=[p.sample(x,t[:,i] if t.ndim==2 else t,e) for i,(p,x,e) in enumerate(zip(self.paths,clean,eps))]
            states=[s.state for s in samples];targets=[self.target.encode(s) for s in samples]
        else:
            samples=[self.paths[0].sample(clean,t,eps)]
            states=samples[0].state;targets=self.target.encode(samples[0])
        network_t=self.clock.network(t) if hasattr(self.clock,'network') else t
        prediction=model(states,network_t,**(model_kwargs or {}))
        if product:
            if len(prediction)!=len(targets):raise ValueError('Model output leaves differ')
            losses=[];terms={}
            for i,(pred,target) in enumerate(zip(prediction,targets)):
                value,parts=self.metric(pred,target)
                if active is not None:value=value*active[:,i]
                losses.append(value*self.weights[i]);terms.update({f'leaf{i}/{k}':v for k,v in parts.items()})
            loss=sum(losses)
        else:loss,terms=self.metric(prediction,targets)
        return TrainingResult(loss,prediction,targets,states,terms)
    @property
    def requires_checkpoint_state(self):
        return (any(method.requires_checkpoint_state or getattr(sampler,'requires_checkpoint_state',hasattr(sampler,'state_dict')) for method,projection,sampler,weight in self.training_stages)
                or any(getattr(s,'requires_checkpoint_state',hasattr(s,'state_dict')) for s in self.aggregation_samplers))
    def state_dict(self):
        state={'format':1,'stages':[{'method':method.state_dict(),'sampler':sampler.state_dict() if hasattr(sampler,'state_dict') else None} for method,projection,sampler,weight in self.training_stages]}
        if self.aggregation_samplers:
            state['aggregation']=[s.state_dict() if hasattr(s,'state_dict') else None for s in self.aggregation_samplers]
        return state
    def load_state_dict(self,state):
        if state['format']!=1 or len(state['stages'])!=len(self.training_stages):raise ValueError('Method stage-state mismatch')
        for entry,(method,projection,sampler,weight) in zip(state['stages'],self.training_stages):
            method.load_state_dict(entry['method'])
            if entry['sampler'] is None and getattr(sampler,'requires_checkpoint_state',hasattr(sampler,'state_dict')):
                raise ValueError('Checkpoint is missing required stage sampler history')
            if entry['sampler'] is not None:
                if not hasattr(sampler,'load_state_dict'):raise ValueError('Sampler cannot restore recorded state')
                sampler.load_state_dict(entry['sampler'])
        entries=state.get('aggregation',[])
        if len(entries)!=len(self.aggregation_samplers):raise ValueError('Aggregation sampler state mismatch')
        for entry,sampler in zip(entries,self.aggregation_samplers):
            if entry is None and getattr(sampler,'requires_checkpoint_state',hasattr(sampler,'state_dict')):
                raise ValueError('Missing required aggregation sampler state')
            if entry is not None:
                if not hasattr(sampler,'load_state_dict'):raise ValueError('Aggregation sampler cannot restore recorded state')
                sampler.load_state_dict(entry)
    @torch.no_grad()
    def sample(self,model,initial,steps=20,stream=None,model_kwargs=None,return_trajectory=False):
        from .solvers import DDPM,Heun
        kwargs=model_kwargs or {};state=initial;trajectory=[map_tree(lambda x:x.clone(),state)]
        if self.sampler is not None:
            out=self.sampler(model,initial,stream=stream,model_kwargs=kwargs,return_trajectory=return_trajectory)
            if return_trajectory:return self.representation.decode(out[0]),out[1]
            return self.representation.decode(out)
        if self.batch is not None and not hasattr(self.clock,'schedule'):
            raise ValueError('This batch component requires an explicit compatible sampler; generic Flow inference would use the wrong path')
        if isinstance(self.solver,DDPM):
            if stream is None:raise ValueError('DDPM requires an explicit random stream')
            path=self.paths[0]
            if self.stages:raise ValueError('Discrete multistage inference requires explicit boundary inputs; use sample_stages')
            for index in range(len(path.betas)-1,-1,-1):
                t=torch.full((state.shape[0],),index,device=state.device,dtype=torch.long)
                pred=model(state,t,**kwargs);clean=self.target.clean(pred,state,t,path)
                state=self.solver.step(path,clean,state,t,stream.normal(state));trajectory.append(state.clone())
        elif hasattr(self.clock,'schedule'):
            # DFM product-state solver, fp64 accumulation with fp32 model inputs.
            original=first(state).dtype;state=map_tree(lambda x:x.double(),state)
            for row in self.clock.schedule(first(state).device):
                t=row['times'].expand(first(state).shape[0],-1)
                pred=self.guidance(model,state,row['times'],row['drop_mask'],kwargs,original) if self.guidance else model(map_tree(lambda x:x.to(original),state),t,drop_mask=row['drop_mask'],**kwargs)
                if isinstance(pred,tuple) and isinstance(pred[0],(dict,list)):pred=pred[0]
                keys=list(state) if isinstance(state,dict) else range(len(state))
                values=[state[k]+row['delta_t'][i]*pred[k].double() if row['drop_mask'][i] and k in pred else state[k] for i,k in enumerate(keys)] if isinstance(state,dict) else [state[k]+row['delta_t'][i]*pred[k].double() if row['drop_mask'][i] else state[k] for i,k in enumerate(keys)]
                state=dict(zip(keys,values)) if isinstance(state,dict) else values
                trajectory.append(map_tree(lambda x:x.clone(),state))
        else:
            if self.stages:raise ValueError('Use sample_stages for explicit boundary noise and conditioning')
            grid=self.clock.grid(steps,first(state).device,first(state).dtype)
            for t0,t1 in zip(grid[:-1],grid[1:]):
                def velocity(x,t):
                    batch_t=t.expand(first(x).shape[0]);net_t=self.clock.network(batch_t)
                    pred=model(x,net_t,**kwargs)
                    return self._velocity(pred,x,batch_t)
                if isinstance(self.solver,Heun):state=self.solver.integrate(velocity,state,t0,t1)
                else:state=self.solver.step(state,velocity(state,t0),t1-t0)
                trajectory.append(map_tree(lambda x:x.clone(),state))
        decoded=self.representation.decode(state)
        return (decoded,trajectory) if return_trajectory else decoded
    def _velocity(self,pred,state,t,index=0,path=None):
        if getattr(self.target,'quantity',None)=='velocity':return pred
        path=path or self.paths[index]
        clean=self.target.clean(pred,state,t,path)
        from .tree import expand
        a,hi,s,da,dhi,ds=path.coefficients(expand(t,state))
        if torch.any(s.abs()<1e-12):raise ValueError('Velocity decoding is singular at sigma=0; use an interior terminal time')
        if torch.equal(hi,a):mean=a*clean;derivative=da*clean
        else:
            low=path.projection(clean);high=clean-low
            mean=a*low+hi*high;derivative=da*low+dhi*high
        eps=(state-mean)/s
        return derivative+ds*eps
    @torch.no_grad()
    def sample_stages(self,model,initial,boundary_inputs,model_kwargs=None,return_trajectory=False):
        """Explicit boundary contexts include supplied noise, clean estimates, etc.

        No boundary chooses an arbitrary missing clean/noise tensor on your behalf.
        """
        if len(boundary_inputs)!=max(0,len(self.stages)-1):raise ValueError('One boundary context per transition required')
        state=initial;trajectory=[]
        for i,stage in enumerate(self.stages):
            clock=stage['clock'];grid=clock.grid(stage.get('steps',20),state.device,state.dtype)
            path=stage.get('path',self.paths[min(i,len(self.paths)-1)])
            for t0,t1 in zip(grid[:-1],grid[1:]):
                def velocity(value,time):
                    t=time.expand(value.shape[0])
                    pred=model(value,clock.network(t),stage=stage.get('model_stage',i),**(model_kwargs or {}))
                    return self._velocity(pred,value,t,path=path)
                from .solvers import Heun,DDPM
                if isinstance(self.solver,DDPM):raise ValueError('Use a discrete stage sampler for DDPM transitions')
                if isinstance(self.solver,Heun):state=self.solver.integrate(velocity,state,t0,t1)
                else:state=self.solver.step(state,velocity(state,t0),t1-t0)
                trajectory.append(state.clone())
            if i<len(self.stages)-1:
                state=stage['boundary'](state,**boundary_inputs[i]).state;trajectory.append(state.clone())
        return (state,trajectory) if return_trajectory else state
