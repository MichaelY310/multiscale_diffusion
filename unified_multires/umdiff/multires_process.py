"""Backbone-independent image/video stage programs.

Paper mathematical processes are separated from author architectures, training
services and unpublished recipes. See README_ZH.md, cross-modal section.
Times below are always data->noise unless explicitly stated otherwise.
"""
import math
from copy import deepcopy
import torch
from .registry import register, build
from .engine import TrainingResult
from .grids import GridMap, resize_grid, grid_shape, haar_split, haar_merge
from .tree import mean_flat, expand
from .trajectories import Trace


class MultiresMethod:
    def __init__(self, config):
        allowed = {'task', 'name', 'image_shape', 'model', 'training', 'process', 'grids', 'first_frame', 'evidence', 'training_stage_reduction'}
        if set(config)-allowed:
            raise ValueError(f'Unknown multires fields: {set(config)-allowed}')
        self.config = deepcopy(config)
        grids = config['grids']
        if tuple(grids[-1]) != tuple(config['image_shape'][1:]):
            raise ValueError('Last process grid must equal the full training grid')
        self.process = build('process', config['process'], grids=grids, first_frame=config.get('first_frame', False))
        reduction = config.get('training_stage_reduction', 'sum_vectors')
        if reduction not in ('sum_vectors', 'sum_stage_means'):
            raise ValueError('Unknown multires training stage reduction')
        self.process.stage_reduction = reduction
        self.batch = None
        self.sampler = self.process

    def training(self, model, images, t, stream, noise=None, model_kwargs=None, active=None, context=None):
        if tuple(images.shape[1:]) != tuple(self.config['image_shape']):
            raise ValueError('Training tensor differs from configured C,[T,]H,W shape')
        if noise is not None and noise.shape != images.shape:
            raise ValueError('Explicit noise must have the full input shape')
        if t is not None and (t.shape != (len(images),) or not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1))):
            raise ValueError('Training time must be a finite B-vector in [0,1]')
        if active is not None:
            raise ValueError('Configure stage selection in the process; external active masks are unsupported')
        return self.process.training(model, images, t, stream, noise, model_kwargs or {}, context or {})

    @torch.no_grad()
    def sample(self, model, initial, steps=20, stream=None, model_kwargs=None, return_trajectory=False):
        if stream is None:
            raise ValueError('An explicit random stream is required')
        if tuple(initial.shape[1:]) != tuple(self.initial_shape()):
            raise ValueError('Initial noise differs from the configured coarse grid')
        return self.process.sample(model, initial, stream, model_kwargs or {}, return_trajectory)

    def initial_shape(self):
        return [self.config['image_shape'][0], *self.process.grids[0]]


class StageProgram:
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, solver='euler', clean_pyramid_fields=None):
        self.grids = [tuple(g) for g in grids]
        if not self.grids or len({len(g) for g in self.grids}) != 1:
            raise ValueError('Nonempty grids of a common rank are required')
        self.maps = [GridMap(b, a, first_frame) for a, b in zip(self.grids[:-1], self.grids[1:])]
        self.full_maps = [GridMap(self.grids[-1], g, first_frame) for g in self.grids]
        self.first_frame = first_frame
        self.steps = [steps]*len(grids) if isinstance(steps, int) else list(steps)
        if len(self.steps) != len(grids) or min(self.steps) < 1:
            raise ValueError('One positive step count per stage is required')
        self.weights = [1/len(grids)]*len(grids) if stage_weights is None else list(stage_weights)
        if len(self.weights) != len(grids) or min(self.weights) < 0 or not any(self.weights):
            raise ValueError('Invalid stage loss weights')
        if solver not in ('euler', 'heun'):
            raise ValueError('Stage ODE solver must be euler or heun')
        self.solver = solver
        self.clean_fields = clean_pyramid_fields
        if clean_pyramid_fields is not None and len(clean_pyramid_fields) != len(grids):
            raise ValueError('Provide a model-domain clean tensor field for every stage')

    def clean_pyramid(self, images, context):
        if self.clean_fields is not None:
            out = [context[key] for key in self.clean_fields]
            for g, x in zip(self.grids, out):
                if grid_shape(x) != g or x.shape[:2] != images.shape[:2]:
                    raise ValueError('Supplied encoded stage tensor has an incompatible shape')
            return out
        out = [images]
        for op in reversed(self.maps):
            out.append(op.down(out[-1]))
        return out[::-1]

    def noise_pyramid(self, noise):
        out = [noise]
        for op in reversed(self.maps):
            out.append(op.down_noise(out[-1]))
        return out[::-1]

    def call(self, model, state, time, stage, kwargs):
        return model(state, time, stage=stage, **kwargs)

    def reduce_training_losses(self, losses):
        if getattr(self, 'stage_reduction', 'sum_vectors') == 'sum_stage_means':
            return sum(loss.mean() for loss in losses)
        return sum(losses)

    def training(self, model, images, t, stream, noise, kwargs, context):
        clean = self.clean_pyramid(images, context)
        noises = self.noise_pyramid(stream.normal(images) if noise is None else noise)
        predictions, targets, states, terms = [], [], [], {}
        losses = []
        for i, (x, eps) in enumerate(zip(clean, noises)):
            u = stream.uniform((len(images),), device=images.device).to(images.dtype) if t is None else t
            if u.shape != (len(images),) or torch.any((u < 0) | (u > 1)):
                raise ValueError('Training time must be a B-vector in [0,1]')
            state, target, clock, meta = self.prepare(clean, eps, u, i)
            prediction = self.call(model, state, clock, i, kwargs)
            loss, detail = self.loss(prediction, target, meta)
            losses.append(loss*self.weights[i]); predictions.append(prediction); targets.append(target); states.append(state)
            terms.update({f'stage{i}/{k}': v for k, v in detail.items()})
            if getattr(self, 'stage_reduction', 'sum_vectors') == 'sum_stage_means':
                terms[f'stage{i}/objective'] = losses[-1].mean()
        return TrainingResult(self.reduce_training_losses(losses), predictions, targets, states, terms)

    def loss(self, prediction, target, meta):
        if prediction.shape != target.shape:
            raise ValueError('Model output shape differs from the stage target')
        value = mean_flat((prediction-target).square())
        return value, {'mse': value}

    def integrate(self, fn, state, grid, trajectory):
        for t0, t1 in zip(grid[:-1], grid[1:]):
            v = fn(state, t0)
            nxt = state+(t1-t0)*v
            if self.solver == 'heun' and t1 != 0:
                nxt = state+(t1-t0)*(v+fn(nxt, t1))/2
            state = nxt
            trajectory.append(state)
        return state


@register('process','relay')
class RelayProcess(StageProgram):
    """Complete low EDM -> spatial upsample -> high blur-diffusion cascade.

    Shared or independent common-model parameters are an explicit transplant.
    Source loss/sampler components remain in use; no random high-res image is
    substituted for the generated low-resolution relay condition.
    """
    def __init__(self,grids,first_frame=False,steps=20,stage_weights=None,patch=4,
                 sigma_data=.5,block_scale=.15,low_sampler=None,high_sampler=None):
        super().__init__(grids,first_frame,steps,stage_weights)
        if len(grids)!=2 or self.maps[0].factors[-2:]!=(patch,patch) or (len(grids[0])==3 and grids[0][0]!=grids[1][0]):
            raise ValueError('Relay requires two grids with spatial ratio=patch and fixed T')
        from .batches import EDMBatch,RelayBatch
        from .relay import EDMSampler,RelaySampler
        self.sigma_data=sigma_data
        self.low_batch=EDMBatch(sigma_data=sigma_data)
        self.high_batch=RelayBatch(patch=patch,sigma_data=sigma_data,block_scale=block_scale)
        self.low_sampler=EDMSampler(num_steps=self.steps[0],**(low_sampler or {}))
        self.high_sampler=RelaySampler(steps=self.steps[1],patch=patch,block_scale=block_scale,**(high_sampler or {}))

    def view(self,model,stage,kwargs):
        data=self.sigma_data
        class View:
            sigma_min=0.;sigma_max=float('inf')
            def round_sigma(self,sigma):return torch.as_tensor(sigma)
            def __call__(self,x,sigma,class_labels=None,augment_labels=None):
                if augment_labels is not None:raise ValueError('Common Relay recipe has no geometric augmentation conditioning')
                if class_labels is not None:raise ValueError('Pass common model conditioning explicitly as y/text')
                x=x.float();clock=sigma.float().reshape(-1)
                if len(clock)==1:clock=clock.expand(len(x))
                s=clock.reshape(-1,*([1]*(x.ndim-1)));d=data
                raw=model(x/(s*s+d*d).sqrt(),clock.log()/4,stage=stage,**kwargs)
                return d*d/(s*s+d*d)*x+s*d/(s*s+d*d).sqrt()*raw
        return View()

    def training(self,model,images,t,stream,noise,kwargs,context):
        low=self.maps[0].down(images)
        normal=None if t is None else torch.erfinv(2*t.clamp(1e-6,1-1e-6)-1)*math.sqrt(2)
        u=stream.uniform((len(images),),device=images.device) if t is None else t
        lo=self.low_batch(low,normal,stream,None if noise is None else self.maps[0].down_noise(noise))
        hi_noise=None
        if noise is not None:
            hi_noise=[noise] if not self.high_batch.block else [noise,stream.normal(images),stream.normal(images)]
        hi=self.high_batch(images,u,stream,hi_noise)
        results=[];terms={};predictions=[]
        for i,batch in enumerate((lo,hi)):
            pred=self.view(model,i,kwargs)(batch.state,batch.time,**batch.model_kwargs)
            loss,detail=batch.loss(pred);results.append(loss*self.weights[i]);predictions.append(pred)
            terms.update({f'stage{i}/{k}':v for k,v in detail.items()})
        return TrainingResult(self.reduce_training_losses(results),predictions,[lo.target,hi.target],[lo.state,hi.state],terms)

    def sample(self,model,initial,stream,kwargs,return_trajectory):
        low=self.low_sampler(self.view(model,0,kwargs),initial,stream=stream)
        high=self.maps[0].up(low)
        if not return_trajectory:
            return self.high_sampler(self.view(model,1,kwargs),high,stream=stream)
        out,trace=self.high_sampler(self.view(model,1,kwargs),high,stream=stream,return_trajectory=True)
        return (out,[low,high,*trace]) if return_trajectory else out


@register('process', 'pyramid_flow')
class PyramidFlowProcess(StageProgram):
    """Pyramid Flow's spatial noisy endpoints and local flow objective.

For 2x spatial transitions, gamma=1/3 and the pinned author's 1000-time grid
are retained. Other integral block geometries use the orthogonal extension.
"""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, solver='euler',
                 train_steps=1000, stage_range=None, shift=1., clean_pyramid_fields=None, network_clock='author',
                 boundary_policy='orthogonal'):
        super().__init__(grids, first_frame, steps, stage_weights, solver, clean_pyramid_fields)
        n = len(grids)
        ranges = torch.linspace(0, 1, n+1).tolist() if stage_range is None else stage_range
        if len(ranges) != n+1 or ranges[0] != 0 or ranges[-1] != 1 or any(a >= b for a, b in zip(ranges, ranges[1:])):
            raise ValueError('Stage ranges must strictly partition [0,1]')
        if train_steps < n or shift <= 0:
            raise ValueError('Invalid training clock')
        self.train_steps = train_steps
        if boundary_policy not in ('orthogonal','native_psd'):
            raise ValueError('Unknown Pyramid Flow boundary policy')
        if boundary_policy=='native_psd' and any(op.factors not in ((2,2),(1,2,2)) for op in self.maps):
            raise ValueError('Native PSD compatibility is a spatial2x boundary')
        self.boundary_policy=boundary_policy
        sigma = torch.arange(train_steps, 0, -1, dtype=torch.float32)/train_steps
        sigma = shift*sigma/(1+(shift-1)*sigma)
        self.original = [sigma[min(int(r*train_steps), train_steps-1)].item() for r in ranges[:-1]]
        self.ends = [sigma[int(r*train_steps)].item() if r < 1 else 0. for r in ranges[1:]]
        if network_clock not in ('author','local'):
            raise ValueError('Pyramid Flow network clock must be author or local')
        if network_clock == 'author' and any(op.first_frame for op in self.maps):
            raise ValueError('Original spatial clock has scalar block volumes; select local for a new first-frame temporal extension')
        self.network_clock = network_clock
        self.starts = []
        for i, g in enumerate(self.grids):
            dummy = torch.empty(1, 1, *g)
            omega = 1. if i == 0 else self.maps[i-1].omega(dummy)
            if not torch.is_tensor(omega) or omega.ndim==0:
                ori_signal=1-self.original[i]
                self.starts.append(1-ori_signal/(float(omega)*(1-ori_signal)+ori_signal))
            else:self.starts.append(self.original[i]*omega/(1+(omega-1)*self.original[i]))
        if network_clock == 'author':
            import numpy as np
            distance = [float(a)-b for a,b in zip(self.starts,self.ends)]
            total = sum(distance)
            self.clock_limits = []
            raw_limits=[]
            for i in range(n):
                low = sum(distance[:i])/total
                high = sum(distance[:i+1])/total if i+1<n else 1.
                a=(sigma*train_steps)[int(low*train_steps)].item()
                b=(sigma*train_steps)[min(int(high*train_steps),train_steps-1)].item()
                raw_limits.append((a,b));self.clock_limits.append((a/train_steps,b/train_steps))
            # Do not divide Python endpoints by1000 and multiply back. The
            # round trip can cross an FP32 midpoint in the inference linspace.
            # Author passes FP32 tensor endpoints to NumPy. Retain its dtype
            # promotion too: Python float endpoints silently create FP64 tables.
            self.clock_raw_tables = [torch.as_tensor(np.linspace(torch.tensor(a,dtype=torch.float32),torch.tensor(b,dtype=torch.float32),train_steps+1)[:-1]) for a,b in raw_limits]
            self.clock_tables = [table.float()/train_steps for table in self.clock_raw_tables]
            self.ratios = torch.from_numpy(np.linspace(1,0,train_steps+1)[:-1])

    def noise_pyramid(self, noise):
        # Retain the released bilinear arithmetic for its spatial geometry.
        if all(op.factors == ((1,2,2) if noise.ndim==5 else (2,2)) for op in self.maps):
            from .operators import resize
            out=[noise]
            for _ in self.maps:out.append(resize(out[-1],scale=.5,mode='bilinear')*2)
            return out[::-1]
        return super().noise_pyramid(noise)

    def network_time(self, rho, stage):
        if self.network_clock == 'local':return rho
        a,b = self.clock_limits[stage]
        return b+(a-b)*rho

    def params(self, i, x):
        # A Python scalar preserves the author's 1-start subtraction BEFORE
        # tensor casting; early float32 conversion changes endpoint gradients.
        start = self.starts[i] if isinstance(self.starts[i],float) else torch.as_tensor(self.starts[i],device=x.device,dtype=x.dtype)
        return start, self.ends[i]

    def prepare(self, clean, eps, u, i):
        x = clean[i]; start, end = self.params(i, x)
        noisy = eps if i == 0 else start*eps+(1-start)*self.maps[i-1].up(clean[i-1])
        cleaner = end*eps+(1-end)*x
        if self.network_clock == 'author':
            index = (u*self.train_steps).long().clamp_max(self.train_steps-1)
            local = self.ratios.to(device=u.device)[index]
            clock = self.clock_tables[i].to(device=u.device)[index].to(u.dtype)
        else:
            local = u; clock = u
        rho = local.reshape(-1,*([1]*(x.ndim-1)))
        return (rho*noisy+(1-rho)*cleaner).to(x.dtype), noisy-cleaner, clock, {}

    def transition(self, state, i, stream):
        op = self.maps[i]
        lifted = op.up(state); sigma = self.ends[i]
        if self.boundary_policy=='native_psd':
            # The author's gamma=1/3 covariance is singular. This explicit PSD
            # compatibility policy retains the established native CPU block
            # RNG order and source alpha/beta arithmetic, without jitter.
            gamma=1/3
            ori_sigma=1-self.original[i+1]
            alpha=1/(math.sqrt(1+1/gamma)*(1-ori_sigma)+ori_sigma)
            beta=alpha*(1-ori_sigma)/math.sqrt(gamma)
            dc=torch.ones(4,4,dtype=torch.float64)/4
            factor=(math.sqrt(1+gamma)*(torch.eye(4,dtype=torch.float64)-dc)).float()
            b,c=lifted.shape[:2];h,w=lifted.shape[-2:];frames=1 if lifted.ndim==4 else lifted.shape[2]
            raw=stream.normal(torch.empty(b*c*frames*(h//2)*(w//2),4,device='cpu'))@factor.T
            noise=raw.reshape(b,c,frames,h//2,w//2,2,2).permute(0,1,2,3,5,4,6).reshape(b,c,frames,h,w)
            if lifted.ndim==4:noise=noise.squeeze(2)
            return alpha*lifted+beta*noise.to(lifted)
        innovation = stream.normal(lifted)
        omega = op.omega(lifted)
        return (lifted+omega*sigma*(innovation-op.project(innovation)))/(1+(omega-1)*sigma)

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        if grid_shape(initial) != self.grids[0]:
            raise ValueError('Incorrect initial grid')
        state = initial; trajectory = Trace(return_trajectory,state)
        for i, steps in enumerate(self.steps):
            if self.network_clock=='author':
                import numpy as np
                grid=torch.from_numpy(np.append(np.linspace(self.ratios[0].item(),self.ratios[-1].item(),steps),0.)).to(state.device)
                clocks=torch.from_numpy(np.linspace(self.clock_raw_tables[i][0].item(),self.clock_raw_tables[i][-1].item(),steps)).to(state.device)
                for j,t in enumerate(clocks):
                    velocity=self.call(model,state,(t.to(state)/self.train_steps).expand(len(state)),i,kwargs)
                    dt=(grid[j+1]-grid[j]).to(state)
                    nxt=state+dt*velocity
                    if self.solver=='heun' and j+1<len(clocks):
                        nxt=state+dt*(velocity+self.call(model,nxt,(clocks[j+1].to(state)/self.train_steps).expand(len(state)),i,kwargs))/2
                    state=nxt;trajectory.append(state)
            else:
                grid = torch.linspace(1, 0, steps+1, device=state.device, dtype=state.dtype)
                state = self.integrate(lambda x, t: self.call(model, x, t.expand(len(x)), i, kwargs), state, grid, trajectory)
            if i < len(self.maps):
                state = self.transition(state, i, stream); trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'pyramidal_wan_path')
class PyramidalWanProcess(PyramidFlowProcess):
    """Published nonlinear natural/global clocks and Haar boundaries.

Default is the mathematical process on model-domain tensors. For RGB-before-
VAE semantics supply clean_pyramid_fields; DMD/teacher regularization is not
silently fabricated. Existing native paper teacher backend remains separate.
"""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, solver='euler',
                 natural_bounds=(1., .9412, .5858, 0.), clean_pyramid_fields=None):
        StageProgram.__init__(self, grids, first_frame, steps, stage_weights, solver, clean_pyramid_fields)
        if len(natural_bounds) != len(grids)+1 or natural_bounds[0] != 1 or natural_bounds[-1] != 0 or any(a <= b for a, b in zip(natural_bounds, natural_bounds[1:])):
            raise ValueError('Natural boundaries must descend strictly from 1 to 0')
        self.bounds = list(natural_bounds)

    def global_sigma(self, natural, i, x):
        omega = self.full_maps[i].omega(x)
        u = expand(natural, x) if natural.ndim == 1 else natural
        return u/(omega-(omega-1)*u)

    def frame_clock(self, sigma, x):
        if x.ndim == 5 and sigma.ndim == 5 and sigma.shape[2] > 1:
            return sigma.expand(len(x), 1, x.shape[2], 1, 1)[:, 0, :, 0, 0]
        return sigma.reshape(-1).expand(len(x))

    def prepare(self, clean, eps, u, i):
        x = clean[i]
        natural = self.bounds[i+1]+u*(self.bounds[i]-self.bounds[i+1])
        noisy_s = self.global_sigma(x.new_tensor(self.bounds[i]), i, x)
        clean_s = self.global_sigma(x.new_tensor(self.bounds[i+1]), i, x)
        sigma = self.global_sigma(natural, i, x)
        noisy = eps if i == 0 else (1-noisy_s)*self.maps[i-1].up(clean[i-1])+noisy_s*eps
        cleaner = (1-clean_s)*x+clean_s*eps
        rho = (sigma-clean_s)/(noisy_s-clean_s)
        return (1-rho)*cleaner+rho*noisy, (noisy-cleaner)/(noisy_s-clean_s), self.frame_clock(sigma, x), {}

    def transition(self, state, i, stream):
        op = self.maps[i]; lifted = op.up(state)
        sigma = self.global_sigma(state.new_tensor(self.bounds[i+1]), i, lifted)
        omega = op.omega(lifted); innovation = stream.normal(lifted)
        return (lifted+omega*sigma*(innovation-op.project(innovation)))/(1+(omega-1)*sigma)

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = initial; trajectory = Trace(return_trajectory,state)
        for i, steps in enumerate(self.steps):
            natural = torch.linspace(self.bounds[i], self.bounds[i+1], steps+1, device=state.device, dtype=state.dtype)
            for n0, n1 in zip(natural[:-1], natural[1:]):
                s0, s1 = self.global_sigma(n0, i, state), self.global_sigma(n1, i, state)
                v = self.call(model, state, self.frame_clock(s0, state), i, kwargs)
                nxt = state+(s1-s0)*v
                if self.solver == 'heun' and n1 != 0:
                    v1 = self.call(model, nxt, self.frame_clock(s1, nxt), i, kwargs)
                    nxt = state+(s1-s0)*(v+v1)/2
                state = nxt; trajectory.append(state)
            if i < len(self.maps):
                state = self.transition(state, i, stream); trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'pixelflow')
class PixelFlowProcess(StageProgram):
    """Released spatial endpoint/clock on an unpacked common backbone.

    The original packed batch is replaced by one full minibatch per stage.
    Spatial factor-two grids and the source jittered block-noise law remain.
    Video shares this spatial process across frames with a joint video model.
    """
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None,
                 train_steps=1000, shift=1., gamma=-1/3, stage_range=None,
                 solver='euler', guidance_scale=0., guidance_scales=None,
                 atol=1e-6, rtol=.001, null_label=1000):
        super().__init__(grids,first_frame,steps,stage_weights)
        from .clocks import PixelFlowClock
        if any(op.factors != ((1,2,2) if len(op.factors)==3 else (2,2)) for op in self.maps):
            raise ValueError('Native PixelFlow clock requires factor-two spatial transitions and fixed T')
        self.clock=PixelFlowClock(train_steps,len(grids),gamma,stage_range=stage_range)
        self.shift,self.gamma=shift,gamma
        if solver not in ('euler','native_dopri5'):
            raise ValueError('PixelFlow supports Euler or the released Dopri5 policy')
        if solver=='native_dopri5' and (len(set(self.steps))!=1 or self.steps[0]<2 or shift!=1.):
            raise ValueError('Released Dopri5 uses a common output-point count >=2 and an unshifted local clock')
        self.solver,self.cfg,self.null=solver,guidance_scale,null_label
        if guidance_scales is None:
            # Published class-conditional recipe, including the redundant
            # unconditioned branch at CFG=1 and its floating-point arithmetic.
            if len(grids)>4:raise ValueError('Supply explicit CFG scales beyond the four released stages')
            guidance_scales=[(guidance_scale-1)*v+1 for v in (0.,1/6,2/3,1.)[:len(grids)]]
        if len(guidance_scales)!=len(grids):raise ValueError('One CFG scale per stage required')
        if solver=='native_dopri5' and guidance_scale<=0:raise ValueError('Native PixelFlow Dopri5 requires its doubled CFG state')
        self.cfg_scales=list(guidance_scales);self.atol,self.rtol=atol,rtol

    def prepare(self,clean,eps,u,i):
        index=(u*self.clock.train_steps).long().clamp_max(self.clock.train_steps-1)
        tau=expand(index.to(u.dtype)/self.clock.train_steps,clean[i])
        start=self.clock.starts[i];end=self.clock.ends[i]
        previous=clean[i] if i==0 else self.maps[i-1].up(clean[i-1])
        x0=start*previous+(1-start)*eps
        x1=end*clean[i]+(1-end)*eps
        clock=self.clock.network_times[i].to(u)[index]/self.clock.train_steps
        return tau*x1+(1-tau)*x0,x1-x0,clock,{}

    def sample(self,model,initial,stream,kwargs,return_trajectory):
        from .noise import equicorrelated_blocks
        state=initial;trace=Trace(return_trajectory,state)
        kwargs=dict(kwargs)
        cfg=self.cfg>0
        if cfg:
            labels=kwargs.get('y')
            if labels is None:raise ValueError('PixelFlow CFG requires class labels y; set guidance_scale=0 for an unconditional backbone')
            kwargs['y']=torch.cat([torch.full_like(labels,self.null),labels])
        for i,count in enumerate(self.steps):
            if i:
                state=self.maps[i-1].up(state)
                shape=state.shape
                flat=state if state.ndim==4 else state.permute(0,2,1,3,4).reshape(-1,shape[1],*shape[-2:])
                noise=equicorrelated_blocks(flat,self.gamma,stream=stream)
                if len(shape)==5:noise=noise.reshape(shape[0],shape[2],shape[1],*shape[-2:]).permute(0,2,1,3,4)
                original=self.clock.original[i]
                a=1/(math.sqrt(1-1/self.gamma)*(1-original)+original)
                b=a*(1-original)/math.sqrt(-self.gamma)
                state=a*state+b*noise;trace.append(state)
            if self.solver=='native_dopri5':
                from .native_models import activate
                activate('pixelflow',['pixelflow'])
                from pixelflow.solver_ode_wrapper import ODE
                # Same published integrator, independent stage/noise/model
                # assembly. Keep BOTH halves through boundaries: the author
                # draws independent boundary innovations for the second half.
                if i==0:state=torch.cat([state,state])
                def velocity(x,time):
                    half=x[:len(x)//2]
                    pred=self.call(model,torch.cat([half,half]),time/self.clock.train_steps,i,kwargs)
                    negative,positive=pred.chunk(2)
                    guided=negative+self.cfg_scales[i]*(positive-negative)
                    return torch.cat([guided,guided])
                ode=ODE(t0=0,t1=1,sampler_type='dopri5',num_steps=count,atol=self.atol,rtol=self.rtol)
                values=ode.sample(state,velocity,self.clock.network_times[i][0].item(),self.clock.network_times[i][-1].item())
                state=values[-1]
                for value in values[1:]:trace.append(value)
                if i==len(self.steps)-1:state=state[:len(initial)]
                continue
            grid,network=self.clock.grid(count,i,self.shift,state.device)
            for j,t in enumerate(network):
                inputs=torch.cat([state,state]) if cfg else state
                # Author casts the unnormalized clock BEFORE dividing in the
                # backbone adapter. Dividing float64 first can round differently.
                v=self.call(model,inputs,(t.to(state)/self.clock.train_steps).expand(len(inputs)),i,kwargs)
                if cfg:
                    negative,positive=v.chunk(2)
                    v=negative+self.cfg_scales[i]*(positive-negative)
                state=(state.float()+(grid[j+1].float()-grid[j].float())*v).to(v.dtype)
                trace.append(state)
        return (state,list(trace)) if return_trajectory else state


@register('process', 'fdm')
class FDMProcess(StageProgram):
    """f-DM-DS Eq.4/8/9 and Algorithm 1, including Appendix A.4 coupling."""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, bounds=None,
                 rescaling='vp', prediction='epsilon', loss_weight='p2', eta=1.,
                 signal_up='linear', noise_lift='average', terminal=.9999):
        super().__init__(grids, first_frame, steps, stage_weights)
        self.bounds = list(torch.linspace(1, 0, len(grids)+1).tolist() if bounds is None else bounds)
        if len(self.bounds) != len(grids)+1 or self.bounds[0] != 1 or self.bounds[-1] != 0 or any(a <= b for a, b in zip(self.bounds, self.bounds[1:])):
            raise ValueError('f-DM bounds must descend from 1 to 0')
        if rescaling not in ('vp', 'sp') or prediction not in ('epsilon', 'clean') or loss_weight not in ('p2', 'snr', 'uniform'):
            raise ValueError('Unsupported f-DM parameterization or weighting')
        if signal_up not in ('nearest', 'linear') or noise_lift not in ('average', 'drop') or not 0 <= eta <= 1 or not .9 < terminal < 1:
            raise ValueError('Invalid f-DM sampler options')
        self.rescaling, self.prediction, self.weighting = rescaling, prediction, loss_weight
        self.eta, self.signal_up, self.noise_lift, self.terminal = eta, signal_up, noise_lift, terminal

    def up_signal(self, x, i):
        return self.maps[i].resize_signal(x,self.signal_up)

    def coeff(self, t, i, x):
        u = expand(t, x)
        a = torch.cos(math.pi*u/2)
        s = torch.sin(math.pi*u/2)/self.full_maps[i].omega(x)
        if self.rescaling == 'vp':
            norm = (a.square()+s.square()).sqrt(); a, s = a/norm, s/norm
        return a, s

    def prepare(self, clean, eps, u, i):
        # Never evaluate the epsilon->signal decoder at the singular t=1.
        upper = min(self.bounds[i], self.terminal)
        t = self.bounds[i+1]+u*(upper-self.bounds[i+1])
        t = t.clamp_min(1e-5)
        a, s = self.coeff(t, i, clean[i])
        fraction = expand((t-self.bounds[i+1])/(self.bounds[i]-self.bounds[i+1]), clean[i])
        previous = clean[i] if i == 0 else self.up_signal(clean[i-1], i-1)
        signal = (1-fraction)*clean[i]+fraction*previous
        residual = clean[i]-signal
        state = a*signal+s*eps
        return state, {'signal': signal, 'residual': residual}, t, {'alpha': a, 'sigma': s, 'state': state}

    def decode(self, pred, state, a, s):
        expected = {'epsilon', 'residual'} if self.prediction == 'epsilon' else {'signal', 'residual'}
        if not isinstance(pred, dict) or set(pred) != expected:
            raise ValueError(f'f-DM requires named heads {sorted(expected)}')
        signal = (state-s*pred['epsilon'])/a if self.prediction == 'epsilon' else pred['signal']
        eps = pred['epsilon'] if self.prediction == 'epsilon' else (state-a*signal)/s
        return signal, pred['residual'], eps

    def loss(self, prediction, target, meta):
        a, s = meta['alpha'], meta['sigma']
        signal, residual, _ = self.decode(prediction, meta['state'], a, s)
        snr = a.square()/s.square()
        weight = 1/(1+snr) if self.weighting == 'p2' else snr if self.weighting == 'snr' else 1.
        first = mean_flat(weight*(signal-target['signal']).square())
        second = mean_flat(weight*(residual-target['residual']).square())
        return first+second, {'signal': first, 'residual': second}

    def lift_noise(self, eps, innovation, i):
        op = self.maps[i]
        if self.noise_lift == 'average':
            return op.lift_noise(eps, innovation)
        if op.first_frame:
            raise ValueError('Drop-noise coupling with separate first frame is not implemented; use average')
        out = innovation.clone()
        slices = (slice(None), slice(None), *[slice(0, None, f) for f in op.factors])
        out[slices] = eps
        return out

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = initial*self.coeff(initial.new_tensor(self.terminal), 0, initial)[1]
        trajectory = Trace(return_trajectory,state)
        for i, steps in enumerate(self.steps):
            high, low = min(self.bounds[i], self.terminal), self.bounds[i+1]
            grid = torch.linspace(high, low, steps+1, device=state.device, dtype=state.dtype)
            for t, s_time in zip(grid[:-1], grid[1:]):
                tv = t.expand(len(state)); sv = s_time.expand(len(state))
                a, sigma = self.coeff(tv, i, state); a_s, sig_s = self.coeff(sv, i, state)
                pred = self.call(model, state, tv, i, kwargs)
                signal, residual, eps = self.decode(pred, state, a, sigma)
                corrected = signal+residual*((t-s_time)/(t-low))
                conditional = (sigma.square()-(a/a_s).square()*sig_s.square()).clamp_min(0)
                variance = self.eta**2*sig_s.square()*conditional/sigma.square()
                state = a_s*corrected+(sig_s.square()-variance).clamp_min(0).sqrt()*eps
                if self.eta and s_time > 0:
                    state = state+variance.sqrt()*stream.normal(state)
                trajectory.append(state)
            if i < len(self.maps):
                # Endpoint denoised estimate includes the full within-stage
                # degradation correction. Couple the *current* residual noise.
                coarse_clean = corrected
                coarse_eps = (state-a_s*coarse_clean)/sig_s
                fine_clean = self.up_signal(coarse_clean, i)
                fine_eps = self.lift_noise(coarse_eps, stream.normal(fine_clean), i)
                a_next, s_next = self.coeff(state.new_full((len(state),), low), i+1, fine_clean)
                state = a_next*fine_clean+s_next*fine_eps; trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'edify_laplacian')
class EdifyProcess(StageProgram):
    """Dimension-varying EDM mathematical instance, not the full Edify system.

The report leaves attenuation schedules open. Here they are explicit linear
functions of sigma inside each stage; low bands stay constant. Eq.8 controls
the boundary, preserving low noise and filling only its orthogonal complement.
"""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, solver='heun',
                 sigma_max=80., sigma_min=.002, transition_sigmas=None, rho=7.,
                 sigma_data=.5, preconditioning=True):
        super().__init__(grids, first_frame, steps, stage_weights, solver)
        if first_frame and any(a[0] != b[0] for a, b in zip(grids[:-1], grids[1:])):
            raise ValueError('Edify scalar-sigma recipe supports uniform blocks; separate-first-frame time pooling needs a tensor clock recipe')
        self.ends = list(transition_sigmas or [1./2**i for i in range(len(grids)-1)])+[sigma_min]
        self.starts = [sigma_max]+[self.ends[i]*math.sqrt(math.prod(op.factors)) for i, op in enumerate(self.maps)]
        if len(self.ends) != len(grids) or sigma_min <= 0 or any(a <= b for a, b in zip(self.starts, self.ends)) or rho <= 0:
            raise ValueError('Invalid EDM sigma intervals')
        self.rho, self.sigma_data, self.preconditioning = rho, sigma_data, preconditioning

    def attenuation(self, sigma, i):
        if i == 0:
            return torch.ones_like(sigma), torch.zeros_like(sigma)
        # Fully recover this band before it becomes the next stage's low band.
        distance = self.starts[i]-self.ends[i]
        return (self.starts[i]-sigma)/distance, -torch.ones_like(sigma)/distance

    def call(self, model, state, time, stage, kwargs):
        if not self.preconditioning:
            return super().call(model, state, time, stage, kwargs)
        s = expand(time, state); d = self.sigma_data
        raw = model(state/(s*s+d*d).sqrt(), time.log()/4, stage=stage, **kwargs)
        return d*d/(s*s+d*d)*state+s*d/(s*s+d*d).sqrt()*raw

    def prepare(self, clean, eps, u, i):
        sigma = (self.ends[i]**(1/self.rho)+u*(self.starts[i]**(1/self.rho)-self.ends[i]**(1/self.rho)))**self.rho
        s = expand(sigma, clean[i]); alpha, _ = self.attenuation(s, i)
        low = clean[i] if i == 0 else self.maps[i-1].project(clean[i])
        state = low+alpha*(clean[i]-low)+s*eps
        return state, clean[i], sigma, {'sigma': s}

    def loss(self, prediction, target, meta):
        s = meta['sigma']; d = self.sigma_data
        value = mean_flat((prediction-target).square()*(s*s+d*d)/(s*d).square())
        return value, {'edm': value}

    def velocity(self, clean, state, sigma, i):
        s = expand(sigma, state); alpha, derivative = self.attenuation(s, i)
        low = clean if i == 0 else self.maps[i-1].project(clean)
        high = clean-low
        return derivative*high+(state-low-alpha*high)/s

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = initial*self.starts[0]; trajectory = Trace(return_trajectory,state)
        for i, steps in enumerate(self.steps):
            grid = torch.linspace(self.starts[i]**(1/self.rho), self.ends[i]**(1/self.rho), steps+1, device=state.device, dtype=state.dtype)**self.rho
            state = self.integrate(lambda x, s: self.velocity(self.call(model, x, s.expand(len(x)), i, kwargs), x, s.expand(len(x)), i), state, grid, trajectory)
            if i < len(self.maps):
                op = self.maps[i]; state = op.up(state)
                noise = stream.normal(state)
                state = state+self.starts[i+1]*(noise-op.project(noise)); trajectory.append(state)
        # One explicit terminal denoising evaluation, counted in diagnostics.
        state = self.call(model, state, state.new_full((len(state),), self.ends[-1]), len(self.grids)-1, kwargs)
        trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'pddpm')
class PDDPMProcess(StageProgram):
    def __init__(self, grids, first_frame=False, steps=1000, stage_weights=None,
                 beta_start=.0001, beta_end=.02, restart_steps=None, up='linear',train_steps=1000):
        super().__init__(grids, first_frame, steps, stage_weights)
        if not 0 < beta_start <= beta_end < 1 or up not in ('linear', 'nearest') or train_steps<2:
            raise ValueError('Invalid DDPM beta table or upsampler')
        self.train_steps=train_steps
        self.betas = torch.linspace(beta_start, beta_end, train_steps, dtype=torch.float64)
        self.abar = (1-self.betas).cumprod(0)
        self.restart = list(restart_steps or [train_steps]+[max(1, train_steps//4)]*(len(grids)-1))
        if len(self.restart) != len(grids) or self.restart[0] != train_steps or any(not 1 <= k <= train_steps for k in self.restart):
            raise ValueError('Invalid per-stage reverse lengths')
        self.up = up

    def prepare(self, clean, eps, u, i):
        index = (u*self.train_steps).long().clamp_max(self.train_steps-1)
        a = expand(self.abar.to(clean[i])[index].sqrt(), clean[i])
        s = (1-a.square()).sqrt()
        return a*clean[i]+s*eps, eps, index.to(clean[i].dtype)/self.train_steps, {}

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = initial; trajectory = Trace(return_trajectory,state)
        for i, count in enumerate(self.restart):
            if i:
                state = self.maps[i-1].resize_signal(state,self.up)
                a = self.abar[count-1].to(state)
                state = a.sqrt()*state+(1-a).sqrt()*stream.normal(state); trajectory.append(state)
            # Respacing changes the inference budget, never the trained marginal.
            indices=torch.linspace(count-1,0,min(count,self.steps[i]),dtype=torch.float64).round().long().tolist()+[-1]
            for k,previous in zip(indices[:-1],indices[1:]):
                a = self.abar[k].to(state)
                ap = self.abar[previous].to(state) if previous>=0 else state.new_tensor(1.)
                beta=1-a/ap
                eps = self.call(model, state, state.new_full((len(state),), k/self.train_steps), i, kwargs)
                clean = (state-(1-a).sqrt()*eps)/a.sqrt()
                state = beta*ap.sqrt()/(1-a)*clean+(1-ap)*(1-beta).sqrt()/(1-a)*state
                if previous>=0:
                    state = state+(beta*(1-ap)/(1-a)).sqrt()*stream.normal(state)
                trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'rdd')
class RDDProcess(StageProgram):
    """Configurable RDD-FM transplant; native old-image recipes use NativeRDDFM."""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None, solver='euler',
                 bounds=None, drift=True, target=None, metric=None, boundary='fresh'):
        super().__init__(grids, first_frame, steps, stage_weights, solver)
        from .paths import RDDFlow, Flow
        self.bounds = list(torch.linspace(0, 1, len(grids)+1).tolist() if bounds is None else bounds)
        if len(self.bounds) != len(grids)+1 or self.bounds[0] != 0 or self.bounds[-1] != 1 or any(a >= b for a, b in zip(self.bounds, self.bounds[1:])):
            raise ValueError('RDD times must ascend from 0 to 1')
        self.paths = []
        for i, (a, b) in enumerate(zip(self.bounds[:-1], self.bounds[1:])):
            path = RDDFlow(a, b, drift=drift)
            path.projection = self.maps[i-1] if i else GridMap(self.grids[0], self.grids[0])
            self.paths.append(path)
        self.target = build('target', target or {'name': 'joint', 'args': {'quantity': 'velocity'}})
        self.metric = build('metric', metric or {'name': 'mse'})
        boundary = {'refresh': 'fresh', 'coupled': 'noise_filling'}.get(boundary, boundary)
        if boundary not in ('fresh', 'noise_filling'):
            raise ValueError('RDD boundary must be fresh (refresh) or noise_filling (legacy alias: coupled)')
        self.boundary = boundary

    def prepare(self, clean, eps, u, i):
        t = self.bounds[i]+u*(self.bounds[i+1]-self.bounds[i])
        # Non-velocity parameterizations need an interior noise endpoint.
        t = t.clamp(1e-5, 1-1e-5)
        sample = self.paths[i].sample(clean[i], t, eps)
        return sample.state, self.target.encode(sample), t, {}

    def loss(self, prediction, target, meta):
        return self.metric(prediction, target)

    def velocity(self, pred, x, t, i):
        if getattr(self.target, 'quantity', None) == 'velocity':
            return pred
        clean = self.target.clean(pred, x, t, self.paths[i])
        a, hi, s, da, dhi, ds = self.paths[i].coefficients(expand(t, x))
        low = self.paths[i].projection(clean); high = clean-low
        return da*low+dhi*high+ds*(x-a*low-hi*high)/s

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = initial; trajectory = Trace(return_trajectory,state)
        for i, steps in enumerate(self.steps):
            a, b = self.bounds[i:i+2]
            quantity = getattr(self.target, 'quantity', None)
            if quantity != 'velocity':
                a = max(a, 1e-5); b = min(b, 1-1e-5)
            grid = torch.linspace(a, b, steps+1, device=state.device, dtype=state.dtype)
            state = self.integrate(lambda x, t: self.velocity(self.call(model, x, t.expand(len(x)), i, kwargs), x, t.expand(len(x)), i), state, grid, trajectory)
            if i < len(self.maps):
                tv = state.new_full((len(state),), b)
                pred = self.call(model, state, tv, i, kwargs)
                clean = self.target.clean(pred, state, tv, self.paths[i])
                op = self.maps[i]; lifted = op.up(clean)
                eps = stream.normal(lifted)
                if self.boundary == 'noise_filling':
                    coarse_eps = (state-b*clean)/(1-b)
                    eps = op.lift_noise(coarse_eps, eps)
                state = b*lifted+(1-b)*eps; trajectory.append(state)
        return (state, list(trajectory)) if return_trajectory else state


@register('process', 'nonuniform_haar')
class NonUniformProcess(StageProgram):
    """Non-Uniform Diffusion §3.3: Haar product states, VP clocks, weighted DSM.

Backbone sees all active bands jointly. Independent per-stage parameter sets
can be provided by a routing model; sharing is an explicit controlled choice.
"""
    def __init__(self, grids, first_frame=False, steps=20, stage_weights=None,
                 logsnr_max=9.21, logsnr_min=-10., likelihood_weighting=True,
                 sampling='sde', terminal=1e-5):
        super().__init__(grids, first_frame, steps, stage_weights)
        if first_frame or sampling not in ('sde', 'ode') or logsnr_max <= logsnr_min or not 0 < terminal < 1/len(grids):
            raise ValueError('Nonuniform Haar requires uniform blocks, valid SNR range and sde/ode sampling')
        self.axes = []
        for op in self.maps:
            if any(f not in (1, 2) for f in op.factors) or all(f == 1 for f in op.factors):
                raise ValueError('Each Haar level must double selected axes')
            self.axes.append(tuple(j+2 for j, f in enumerate(op.factors) if f == 2))
        self.horizons = [1.-i/len(grids) for i in range(len(grids))]
        self.max_snr, self.min_snr = logsnr_max, logsnr_min
        self.weighted, self.sampling, self.terminal = likelihood_weighting, sampling, terminal

    def encode(self, x):
        parts = []
        for axes in reversed(self.axes):
            leaves = haar_split(x, axes); x = leaves[0]
            parts.append(torch.cat(leaves[1:], 1))
        return [x, *parts[::-1]]

    def decode(self, parts):
        x = parts[0]
        for detail, axes in zip(parts[1:], self.axes):
            x = haar_merge([x, *detail.chunk(2**len(axes)-1, 1)], axes)
        return x

    def coeff(self, t, i, x):
        slope = (self.min_snr-self.max_snr)/self.horizons[i]
        logsnr = self.max_snr+slope*expand(t, x)
        abar = logsnr.sigmoid()
        return abar.sqrt(), (1-abar).sqrt(), -slope*(1-abar)

    def training(self, model, images, t, stream, noise, kwargs, context):
        clean = self.encode(images)
        eps = self.encode(stream.normal(images) if noise is None else noise)
        losses, predictions, targets, states, terms = [], [], [], [], {}
        for stage in range(len(clean)):
            high = self.horizons[stage]
            low = self.horizons[stage+1] if stage+1 < len(clean) else self.terminal
            u = stream.uniform((len(images),), device=images.device) if t is None else t
            clock = low+(high-low)*u
            state, target, weights = {}, {}, {}
            for j in range(stage+1):
                a, s, beta = self.coeff(clock, j, clean[j])
                state[j] = a*clean[j]+s*eps[j]; target[j] = eps[j]
                weights[j] = beta/s.square() if self.weighted else 1.
            clocks = clock[:, None].expand(-1, stage+1)
            pred = model(state, clocks, stage=stage, **kwargs)
            # 1/2 ||score error||^2_Λ, normalized per scalar coefficient.
            size = sum(v[0].numel() for v in state.values())
            value = sum(((pred[j]-target[j]).square()*weights[j]).flatten(1).sum(1) for j in state)/(2*size)
            losses.append(value*self.weights[stage]); predictions.append(pred); targets.append(target); states.append(state)
            terms[f'stage{stage}/weighted_dsm'] = value
            if getattr(self, 'stage_reduction', 'sum_vectors') == 'sum_stage_means':
                terms[f'stage{stage}/objective'] = losses[-1].mean()
        return TrainingResult(self.reduce_training_losses(losses), predictions, targets, states, terms)

    def sample(self, model, initial, stream, kwargs, return_trajectory):
        state = {0: initial}; trajectory = Trace(return_trajectory,state)
        channels = initial.shape[1]
        for stage, steps in enumerate(self.steps):
            high = self.horizons[stage]
            low = self.horizons[stage+1] if stage+1 < len(self.grids) else self.terminal
            if stage:
                detail_shape = (len(initial), channels*(2**len(self.axes[stage-1])-1), *self.grids[stage-1])
                state[stage] = stream.normal(initial.new_empty(detail_shape))
            grid = torch.linspace(high, low, steps+1, device=initial.device, dtype=initial.dtype)
            for t0, t1 in zip(grid[:-1], grid[1:]):
                t = t0.expand(len(initial)); pred = model(state, t[:, None].expand(-1, stage+1), stage=stage, **kwargs)
                for j in state:
                    a, s, beta = self.coeff(t, j, state[j])
                    score = -pred[j]/s
                    velocity = -.5*beta*state[j]-beta*score*(1. if self.sampling == 'sde' else .5)
                    state[j] = state[j]+(t1-t0)*velocity
                    if self.sampling == 'sde':
                        state[j] = state[j]+(beta*(t0-t1)).sqrt()*stream.normal(state[j])
                trajectory.append(state)
        out = self.decode([state[j] for j in range(len(self.grids))])
        return (out, list(trajectory)) if return_trajectory else out
