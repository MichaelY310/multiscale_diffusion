"""Tensor trees preserve dictionary keys, scale order and heterogeneous shapes."""
import torch

def map_tree(fn, tree, *others):
    if isinstance(tree, torch.Tensor):
        if not all(isinstance(o, torch.Tensor) for o in others):
            raise TypeError('Tensor tree structures differ')
        return fn(tree, *others)
    if isinstance(tree, dict):
        if not all(isinstance(o, dict) and list(o)==list(tree) for o in others):
            raise ValueError('Tensor tree dictionary keys/order differ')
        return {k:map_tree(fn, v, *(o[k] for o in others)) for k,v in tree.items()}
    if isinstance(tree, (list, tuple)):
        if not all(type(o) is type(tree) and len(o)==len(tree) for o in others):
            raise ValueError('Tensor tree sequence structures differ')
        return type(tree)(map_tree(fn,v,*(o[i] for o in others)) for i,v in enumerate(tree))
    raise TypeError(f'Unsupported state leaf: {type(tree)}')

def leaves(tree):
    out=[]
    map_tree(lambda x:out.append(x),tree)
    return out

def first(tree):
    return leaves(tree)[0]

def add(x,y,scale=1):
    return map_tree(lambda a,b:a+scale*b,x,y)

def clone(tree):
    return map_tree(lambda x:x.clone(),tree)

def detach(tree):
    return map_tree(lambda x:x.detach(),tree)

def expand(t,x):
    t=torch.as_tensor(t,device=x.device,dtype=x.dtype)
    if t.ndim==0:
        return t
    if t.ndim==1 and t.shape[0]==x.shape[0]:
        return t.reshape(-1,*([1]*(x.ndim-1)))
    if t.ndim==x.ndim:
        return t
    raise ValueError(f'Time {t.shape} is not scalar or per-sample for state {x.shape}')

def mean_flat(x):
    return x.flatten(1).mean(1)
