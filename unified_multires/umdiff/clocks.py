"""Local path time, network time and active product-state clocks are distinct."""
import math
import numpy as np
import torch
from .registry import register

@register('clock','linear')
class LinearClock:
    def __init__(self,start=0.,end=1.,network_scale=1.,network_offset=0.):
        self.start=start;self.end=end;self.scale=network_scale;self.offset=network_offset
    def grid(self,steps,device='cpu',dtype=torch.float32):
        return torch.linspace(self.start,self.end,steps+1,device=device,dtype=dtype)
    def network(self,t):return t*self.scale+self.offset

@register('clock','pixelflow')
class PixelFlowClock:
    def __init__(self,train_steps=1000,stages=4,gamma=-1/3,stage_range=None):
        if stages<=0 or stages>train_steps:raise ValueError('Invalid stage count')
        self.train_steps=train_steps;self.stages=stages;self.gamma=gamma
        ranges=[j/stages for j in range(stages+1)] if stage_range is None else list(stage_range)
        if len(ranges)!=stages+1 or ranges[0]!=0 or ranges[-1]!=1 or any(a>=b for a,b in zip(ranges,ranges[1:])):
            raise ValueError('PixelFlow stage_range must partition [0,1]')
        ts=torch.linspace(0,train_steps-1,train_steps,dtype=torch.float32)
        normalized=ts/train_steps
        self.original=[];self.starts=[];self.ends=[]
        for j in range(stages):
            # Keep original arithmetic when no controlled partition is supplied.
            ia=int(train_steps*j/stages) if stage_range is None else int(train_steps*ranges[j])
            ib=int(train_steps*(j+1)/stages) if stage_range is None else int(train_steps*ranges[j+1])
            a=normalized[ia].item()
            b=normalized[ib].item() if j+1<stages else 1.
            self.original.append(a)
            self.starts.append(a/(math.sqrt(1-1/gamma)*(1-a)+a) if j else a)
            self.ends.append(b)
        distance=[b-a for a,b in zip(self.starts,self.ends)]
        total=sum(distance);self.network_times=[]
        for j in range(stages):
            lo=0. if j==0 else sum(distance[:j])/total
            hi=1. if j==stages-1 else sum(distance[:j+1])/total
            a=ts[int(train_steps*lo)].item();b=ts[min(int(train_steps*hi),train_steps-1)].item()
            count=train_steps if j==stages-1 else train_steps+1
            grid=torch.linspace(a,b,count,dtype=torch.float64)
            self.network_times.append(grid if j==stages-1 else grid[:-1])
    def grid(self,steps,stage=0,shift=1.,device='cpu'):
        end=(self.train_steps-1)/self.train_steps
        t=np.linspace(0,end,steps,dtype=np.float64)
        t=t/(shift+(1-shift)*t)
        a=self.network_times[stage][0].item();b=self.network_times[stage][-1].item()
        k=(b-a)/end;network=k*t+a
        return torch.from_numpy(np.append(t,1.)).to(device),torch.from_numpy(network).to(device)

@register('clock','dfm')
class DFMClock:
    def __init__(self,steps=(4,4),thresholds=(.5,)):
        self.steps=steps;self.thresholds=thresholds
        if len(thresholds)!=len(steps)-1 or not all(x>y for x,y in zip(thresholds,thresholds[1:])):raise ValueError('Invalid thresholds')
    def schedule(self,device='cpu'):
        ends=torch.cat([torch.tensor(self.thresholds),torch.tensor([0.])])
        table=torch.ones(sum(self.steps),len(self.steps))
        cursor=0
        for i,n in enumerate(self.steps):
            indices=torch.flip(torch.arange(n,dtype=torch.float64),dims=[0])+1
            chunk=indices*((1.-float(ends[i]))/n)+float(ends[i])
            table[cursor:cursor+n,i]=chunk
            for j in range(i):table[cursor:cursor+n,j]=chunk*(float(ends[j])-float(ends[j+1]))+float(ends[j+1])
            cursor+=n
        table=table.to(device);rows=[]
        for i,current in enumerate(table):
            nxt=table[i+1] if i+1<len(table) else torch.zeros_like(current)
            delta=nxt-current;mask=torch.ones_like(current)
            negative=torch.where(delta<0)[0]
            last=int(negative[-1]) if len(negative) else -1
            mask[last+1:]=0
            if torch.any(delta>0):raise ValueError('DFM times must decrease')
            rows.append({'times':current,'next_times':nxt,'delta_t':delta,'drop_mask':mask})
        return rows

def speed_activation(power,delta):
    if not 0<delta<1 or power*(1+power-delta)<=0:raise ValueError('Invalid spectrum or threshold')
    return 1/(1+math.sqrt(delta/(power*(1+power-delta))))

@register('clock','nami')
class NAMIClock:
    def __init__(self,train_steps=1000,stage_range=(0,1/3,2/3,1),gamma=1/3,shift=1.):
        times=torch.from_numpy(np.linspace(1,train_steps,train_steps,dtype=np.float32)[::-1].copy());sigmas=times/train_steps
        sigmas=shift*sigmas/(1+(shift-1)*sigmas);times=sigmas*train_steps
        distances=[];stages=len(stage_range)-1
        for i in range(stages):
            start=sigmas[int(stage_range[i]*train_steps)].item();end=sigmas[int(stage_range[i+1]*train_steps)].item() if i+1<stages else 0.
            if i:
                original=1-start;start=1-original/(math.sqrt(1+1/gamma)*(1-original)+original)
            distances.append(start-end)
        self.network=[];self.sigmas=[];total=sum(distances)
        for i in range(stages):
            low=sum(distances[:i])/total;high=sum(distances[:i+1])/total if i+1<stages else 1.
            maximum=times[int(low*train_steps)].item();minimum=times[min(int(high*train_steps),train_steps-1)].item()
            count=int(maximum-minimum)+1
            self.network.append(torch.from_numpy(np.linspace(maximum,minimum,count)[:-1]))
            self.sigmas.append(torch.from_numpy(np.linspace(1,0,count)[:-1]))
    def grid(self,steps,stage,device='cpu'):
        t=self.network[stage];s=self.sigmas[stage]
        times=torch.from_numpy(np.linspace(t[0].item(),t[-1].item(),steps)).to(device)
        sigmas=torch.from_numpy(np.linspace(s[0].item(),s[-1].item(),steps)).to(device)
        return times,torch.cat([sigmas,torch.zeros(1,device=device)])
