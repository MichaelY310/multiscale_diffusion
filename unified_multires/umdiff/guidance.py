"""Guidance is an inference policy; it never changes the training target."""
import torch
from .registry import register
from .tree import first,map_tree

@register('guidance','cfg_window')
class WindowGuidance:
    """Product-state CFG: one doubled call, independent clock per leaf."""
    def __init__(self,scale=1.,low=0.,high=1.,null_label=1000,label_key='y'):
        if low>high:raise ValueError('Empty guidance window')
        self.scale=scale;self.low=low;self.high=high;self.null=null_label;self.key=label_key
    def __call__(self,model,state,times,drop_mask,kwargs,dtype):
        active=(times>=self.low)&(times<=self.high)
        guided=self.scale>1 and bool(active.any())
        x=map_tree(lambda v:v.to(dtype),state);t=times.expand(len(first(state)),-1)
        call=dict(kwargs)
        if guided:
            if self.key not in call:raise ValueError('Guidance requires supplied conditional labels')
            y=call[self.key];call[self.key]=torch.cat([y,torch.full_like(y,self.null)])
            x=map_tree(lambda v:torch.cat([v,v]),x);t=torch.cat([t,t]).to(dtype)
        pred=model(x,t,drop_mask=drop_mask,**call)
        if isinstance(pred,tuple) and isinstance(pred[0],(dict,list)):pred=pred[0]
        if guided:
            keys=list(pred) if isinstance(pred,dict) else range(len(pred));outputs={}
            for k in keys:
                conditional,unconditional=pred[k].double().chunk(2)
                outputs[k]=unconditional+self.scale*(conditional-unconditional) if active[int(k)] else conditional
            pred=outputs if isinstance(pred,dict) else [outputs[k] for k in keys]
        return pred

def threshold_sample(sample,kind='clip',image_scale=1.):
    """Native MDM clip / dynamic / dynamic-if conventions."""
    if kind=='none':return sample
    if kind=='clip':return (sample*image_scale).clip(-1,1)/image_scale
    if kind not in ('dynamic','dynamic_if'):raise ValueError('Unknown threshold policy')
    original=sample.dtype;shape=sample.shape;value=sample*image_scale
    if original not in (torch.float32,torch.float64):value=value.float()
    flat=value.reshape(len(value),-1)
    q,maximum=(.995,100.) if kind=='dynamic' else (.95,1.5)
    scale=torch.quantile(flat.abs(),q,dim=1).clamp(min=1,max=maximum)[:,None]
    return (torch.clamp(flat,-scale,scale)/scale).reshape(shape).to(original)/image_scale
