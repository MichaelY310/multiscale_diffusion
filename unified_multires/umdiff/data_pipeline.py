"""Stateful author-style DataLoader traversal; rank-local worker RNG survives resume."""
import copy
import random
import numpy as np
import torch
from pathlib import Path


class NativeRelayDataset(torch.utils.data.Dataset):
    """Use the pinned author's ZIP/labels/xflip reader and exact pixel mapping.

    Source reader reuse is intentional: ZIP metadata is not reinterpreted as
    ImageFolder class directory labels. No dataset conversion/download occurs.
    """
    def __init__(self,path,**kwargs):
        from .native_models import activate
        import importlib
        activate('relay',['dnnlib','training'])
        source=importlib.import_module('training.dataset')
        self.dataset=source.ImageFolderDataset(path=str(path),**kwargs)
        root=Path(path)
        self.files=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        pixels,labels=self.dataset[index]
        return {'images':torch.from_numpy(pixels).to(torch.float32)/127.5-1,
                'class_labels':torch.from_numpy(labels)}


class WorkerStateDataset(torch.utils.data.Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):return self.dataset[index]
    def state_dict(self):
        # Main-process training RNG is owned by Trainer, not this dataset.
        if torch.utils.data.get_worker_info() is None:return {}
        value={'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
        if hasattr(self.dataset,'state_dict'):value['dataset']=self.dataset.state_dict()
        return value
    def load_state_dict(self,value):
        if not value:return
        torch.set_rng_state(value['torch']);np.random.set_state(value['numpy']);random.setstate(value['python'])
        if 'dataset' in value:self.dataset.load_state_dict(value['dataset'])


class StatefulOfflineLoader:
    def __init__(self,dataset,batch_size,seed=0,shuffle=True,drop_last=True,num_workers=0,
                 rank=0,world_size=1,prefetch_factor=2,persistent_workers=False):
        from torchdata.stateful_dataloader import StatefulDataLoader
        if not 0<=rank<world_size or num_workers<0 or batch_size<1:raise ValueError('Invalid reader topology')
        self.contract=dict(batch_size=batch_size,seed=seed,shuffle=shuffle,drop_last=drop_last,
            num_workers=num_workers,rank=rank,world_size=world_size,prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,dataset_size=len(dataset))
        self.sampler=torch.utils.data.DistributedSampler(dataset,num_replicas=world_size,rank=rank,
            shuffle=shuffle,seed=seed,drop_last=False)
        args=dict(batch_size=batch_size,sampler=self.sampler,drop_last=drop_last,num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),snapshot_every_n_steps=1)
        if num_workers:args.update(multiprocessing_context='spawn',prefetch_factor=prefetch_factor,persistent_workers=persistent_workers)
        self.loader=StatefulDataLoader(WorkerStateDataset(dataset),**args)
        if not len(self.loader):raise ValueError('No complete rank-local batch')
        self.epoch=0;self.iterator=None;self.batches=0
    def __iter__(self):return self
    def __next__(self):
        if self.iterator is None:self.sampler.set_epoch(self.epoch);self.iterator=iter(self.loader)
        try:value=next(self.iterator)
        except StopIteration:
            self.epoch+=1;self.sampler.set_epoch(self.epoch);self.iterator=iter(self.loader);value=next(self.iterator)
        self.batches+=1;return value
    def state_dict(self):
        return copy.deepcopy(dict(format=1,contract=self.contract,epoch=self.epoch,batches=self.batches,loader=self.loader.state_dict()))
    def load_state_dict(self,value):
        if value['format']!=1 or value['contract']!=self.contract:raise ValueError('Reader dataset/topology/worker contract mismatch')
        self.epoch=value['epoch'];self.batches=value['batches'];self.sampler.set_epoch(self.epoch)
        self.loader.load_state_dict(value['loader']);self.iterator=None
