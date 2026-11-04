"""Offline input codecs belong to the data recipe, separate from diffusion."""
from pathlib import Path
import numpy as np
import torch
from .registry import register,build

@register('model','local_autoencoder')
def local_autoencoder(path):
    from diffusers import AutoencoderKL
    path=Path(path)
    if not path.is_dir() or not (path/'config.json').is_file():raise FileNotFoundError('Supply an existing local Diffusers VAE directory')
    return AutoencoderKL.from_pretrained(str(path),local_files_only=True)

@register('model','local_wan_transformer')
def local_wan_transformer(path):
    from diffusers import WanTransformer3DModel
    if not Path(path).is_dir():raise FileNotFoundError('Supply an existing local Wan transformer directory')
    return WanTransformer3DModel.from_pretrained(path,local_files_only=True)

@register('input_transform','vae_moments')
class VAEMoments:
    def __init__(self,scale=.18215,min_logvar=-30.,max_logvar=20.):self.scale=scale;self.minimum=min_logvar;self.maximum=max_logvar
    @torch.no_grad()
    def __call__(self,moments,stream):
        if moments.shape[1]%2:raise ValueError('Moments contain concatenated mean and log variance')
        mean,logvar=moments.chunk(2,dim=1);std=torch.exp(.5*logvar.clamp(self.minimum,self.maximum))
        return self.scale*(mean+std*stream.normal(mean))

@register('input_transform','local_vae')
class LocalVAE:
    def __init__(self,model,scale=.18215,mean_only=False,freeze_parameters=True):
        self.model=build('model',model).eval()
        if freeze_parameters:self.model.requires_grad_(False)
        self.scale=scale;self.mean=mean_only
    @torch.no_grad()
    def __call__(self,images,stream):
        self.model.to(images.device);posterior=self.model.encode(images).latent_dist
        return (posterior.mean if self.mean else posterior.sample(generator=stream.for_device(images.device)))*self.scale

class LocalMomentDataset(torch.utils.data.Dataset):
    """Author DiMR numbered local .npy moments/label records; no downloader."""
    def __init__(self,path,cfg_dropout=0.,null_label=1000,strict_author_layout=True):
        self.files=sorted(Path(path).glob('*.npy'),key=lambda p:int(p.stem))
        if not self.files:raise ValueError('No local moment/label .npy records')
        if strict_author_layout and [int(p.stem) for p in self.files]!=list(range(len(self.files))):raise ValueError('Author moments layout requires consecutive0..N-1.npy records')
        if not 0<=cfg_dropout<=1:raise ValueError('Invalid reader CFG dropout')
        self.dropout=cfg_dropout;self.null_label=null_label
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        moments,label=np.load(self.files[index],allow_pickle=True)
        if self.dropout:
            import random
            if random.random()<self.dropout:label=self.null_label
        return {'images':torch.as_tensor(moments).float(),'labels':torch.tensor(int(label))}
