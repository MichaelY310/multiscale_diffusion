"""Explicit random tape for native/port coupling, and serializable generators."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from .tree import first, map_tree

class RandomStream:
    def __init__(self, seed=0, device='cpu'):
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.device_generators={str(self.generator.device):self.generator}

    def for_device(self,device):
        key=str(torch.device(device))
        if key not in self.device_generators:
            self.device_generators[key]=torch.Generator(device=device).manual_seed(self.generator.initial_seed())
        return self.device_generators[key]

    @contextmanager
    def native_rng(self,device='cpu'):
        """Process-local native calls consume this stream, preserving outer RNG."""
        device=torch.device(device);devices=[] if device.type=='cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.for_device('cpu').get_state())
            if devices:torch.cuda.set_rng_state(self.for_device(device).get_state(),devices[0])
            try:yield
            finally:
                self.for_device('cpu').set_state(torch.get_rng_state())
                if devices:self.for_device(device).set_state(torch.cuda.get_rng_state(devices[0]))

    def normal(self, x):
        # Match randn_like's preserve_format behavior. Video resize/packing can
        # yield noncontiguous dense views: a contiguous shape-only draw consumes
        # the same RNG but assigns its numbers to different tensor elements.
        return torch.empty_like(x).normal_(generator=self.for_device(x.device))

    def uniform(self, shape, *, device='cpu', dtype=torch.float32):
        return torch.rand(shape,device=device,dtype=dtype,generator=self.for_device(device))

    def randint(self,low,high,shape,device='cpu'):
        return torch.randint(low,high,shape,device=device,generator=self.for_device(device))

    def state_dict(self):
        return {'generator':self.generator.get_state(),'devices':{k:g.get_state() for k,g in self.device_generators.items()}}

    def load_state_dict(self,state):
        self.generator.set_state(state['generator'].cpu())
        for key,value in state.get('devices',{}).items():self.for_device(key).set_state(value.cpu())

class GlobalRandomStream:
    """Compatibility stream for author functions using global torch RNG.

    Independent experiments should prefer RandomStream. This stream is useful
    when exactly coupling native stochastic model layers and noising draws.
    """
    def for_device(self,device):return None
    def normal(self,x):return torch.randn_like(x)
    def uniform(self,shape,device='cpu',dtype=torch.float32):return torch.rand(shape,device=device,dtype=dtype)
    def randint(self,low,high,shape,device='cpu'):return torch.randint(low,high,shape,device=device)
    def state_dict(self):return {'torch':torch.get_rng_state()}
    def load_state_dict(self,state):torch.set_rng_state(state['torch'].cpu())
    @contextmanager
    def native_rng(self,device='cpu'):yield

class NoiseTape:
    """Supply raw normal draws to native functions without changing their bodies.

    Each entry is consumed exactly once; shape validation catches RNG-order drift.
    This is test-only patching, never used by a production trainer.
    """
    def __init__(self, tensors):
        self.tensors=list(tensors)
        self.cursor=0

    def normal_like(self,x,**kwargs):
        if self.cursor>=len(self.tensors):
            raise RuntimeError('Noise tape exhausted')
        value=self.tensors[self.cursor]
        self.cursor+=1
        if value.shape!=x.shape:
            raise ValueError(f'Noise tape draw {self.cursor}: {value.shape} != {x.shape}')
        return value.to(device=kwargs.get('device',x.device),dtype=kwargs.get('dtype',x.dtype))

    def assert_consumed(self):
        if self.cursor!=len(self.tensors):
            raise AssertionError(f'Unused noise tape: {len(self.tensors)-self.cursor}')

    @contextmanager
    def patch_randn_like(self):
        with patch('torch.randn_like',self.normal_like):
            yield self
        self.assert_consumed()
