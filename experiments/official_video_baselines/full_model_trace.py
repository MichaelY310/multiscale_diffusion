"""Bounded verification hashes of every tensor; supports BF16 and DTensor."""
import hashlib
import torch
import dataclasses


def hashed(value):
    if isinstance(value, torch.Tensor):
        if hasattr(value, 'full_tensor'):
            value=value.full_tensor()
        t=value.detach().cpu().contiguous()
        return {'shape':list(t.shape),'dtype':str(t.dtype),
                'sha256':hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):return {str(k):hashed(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [hashed(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {f.name:hashed(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'object_type':type(value).__module__+'.'+type(value).__qualname__}


def trace_method(method):
    """Record ordinary FastVideo methods without changing model computation."""
    trace={'raw':[],'inputs':[],'predictions':[],'losses':[],
           'gradients':[],'updates':[],'teacher_inputs':[],'teacher_predictions':[]}
    original=method.single_train_step
    def step(batch,iteration):
        trace['raw'].append({'batch':hashed(batch),'iteration':iteration,
            'cpu_rng':hashed(torch.get_rng_state()),'cuda_rng':hashed(torch.cuda.get_rng_state()),
            'generator':hashed(method.cuda_generator.get_state())})
        value=original(batch,iteration)
        trace['losses'].append(hashed(value[0]))
        return value
    method.single_train_step=step
    for role,model in method._role_models.items():
        prefix='' if role=='student' else 'teacher_'
        model.transformer.register_forward_pre_hook(
            lambda m,a,k,p=prefix:trace[p+'inputs'].append(hashed({'args':a,'kwargs':k})),with_kwargs=True)
        model.transformer.register_forward_hook(
            lambda m,a,y,p=prefix:trace[p+'predictions'].append(hashed(y)))
    for optimizer in method.get_optimizers(0):
        raw=getattr(optimizer,'optimizer',optimizer)
        raw.register_step_pre_hook(lambda *a:trace['gradients'].append(
            hashed({k:p.grad for k,p in method.student.transformer.named_parameters()})))
        raw.register_step_post_hook(lambda *a,o=optimizer:trace['updates'].append({
            'weights':hashed(method.student.transformer.state_dict()),'adamw':hashed(o.state_dict())}))
    return trace


def trace_native(runner, optimizer):
    trace={'raw':[],'inputs':[],'predictions':[],'losses':[],'gradients':[],'updates':[]}
    original_class=type(runner)
    def traced(self, video,text,identifier=['video'],**kwargs):
        trace['raw'].append({'video':hashed(video),'text':hashed(text),'identifier':identifier,
                             'cpu_rng':hashed(torch.get_rng_state()),'cuda_rng':hashed(torch.cuda.get_rng_state())})
        return original_class.__call__(self,video,text,identifier,**kwargs)
    runner.__class__=type('VerificationTrace'+original_class.__name__,(original_class,),{'__call__':traced})
    runner.dit.register_forward_pre_hook(lambda m,a,kw:trace['inputs'].append(hashed(kw)),with_kwargs=True)
    runner.dit.register_forward_hook(lambda m,a,y:trace['predictions'].append(hashed(y)))
    loss=runner.calculate_loss
    def traced_loss(pred,target):
        value=loss(pred,target)
        trace['losses'].append({'loss':hashed(value[0]),'targets':hashed(target)})
        return value
    runner.calculate_loss=traced_loss
    raw=getattr(optimizer,'optimizer',optimizer)
    raw.register_step_pre_hook(lambda *a:trace['gradients'].append(hashed({k:p.grad for k,p in runner.dit.named_parameters()})))
    raw.register_step_post_hook(lambda *a:trace['updates'].append({'weights':hashed(runner.dit.state_dict()),'adamw':hashed(optimizer.state_dict())}))
    return trace
