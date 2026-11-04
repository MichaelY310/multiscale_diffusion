"""Autoencoder reconstruction/KL and configurable auxiliary codec losses."""
import torch
from .registry import build,register
from .engine import TrainingResult
from .tree import mean_flat

class CodecMethod:
    def __init__(self,config):
        allowed_fields={'task','name','model','image_shape','sampling_shape','training','evidence','codec'}
        if set(config)-allowed_fields:raise ValueError(f'Unknown codec method fields: {set(config)-allowed_fields}')
        self.config=config;settings=config.get('codec',{})
        allowed={'mse_weight','kl_weight','regularizers','perceptual','regularizer_selection','regularizer_ignore_perceptual','kl_schedule','kl_anneal_steps','decoder_r1_weight','decoder_r1_frequency','mae_weight','pseudo_huber_weight','pseudo_huber_breadth','high_frequency','reconstruction_style'}
        if set(settings)-allowed:raise ValueError('Unknown codec objective setting')
        self.mse=settings.get('mse_weight',1.);self.kl=settings.get('kl_weight',1e-4)
        self.regularizers=[build('regularizer',s) for s in settings.get('regularizers',[])]
        self.settings=settings;self.perceptual=build('model',settings['perceptual']['model']) if settings.get('perceptual') else None
        if self.perceptual is not None:self.perceptual.eval().requires_grad_(False)
        if settings.get('regularizer_selection','all') not in ('all','random_one'):raise ValueError('Unknown regularizer selection')
        if settings.get('kl_schedule') and self.kl!=0:raise ValueError('Native KL schedule requires kl_weight=0')
        if settings.get('reconstruction_style','generic') not in ('generic','native'):raise ValueError('Unknown reconstruction reduction style')
        if settings.get('decoder_r1_frequency',1)<1:raise ValueError('R1 frequency must be positive')
    def native_reconstruction(self,prediction,images):
        """Author metric order and keepdim=True matter for floating backward."""
        prediction=prediction[:,None];images=images[:,None];values=[]
        if self.perceptual is not None:values.append(self.perceptual(prediction,images).reshape(len(images),-1).mean(1,keepdim=True)*self.settings['perceptual'].get('weight',1.))
        if self.mse:values.append((prediction-images).square().reshape(len(images),-1).mean(1,keepdim=True)*self.mse)
        if self.settings.get('mae_weight'):values.append((prediction-images).abs().reshape(len(images),-1).mean(1,keepdim=True)*self.settings['mae_weight'])
        if self.settings.get('pseudo_huber_weight'):
            width=self.settings.get('pseudo_huber_breadth',.01);values.append(((prediction-images).square()+width**2).sqrt().sub(width).reshape(len(images),-1).mean(1,keepdim=True)*self.settings['pseudo_huber_weight'])
        return sum(values)
    def training(self,model,images,t,stream,noise=None,model_kwargs=None,active=None,context=None):
        network=model.model if hasattr(model,'model') else model
        posterior=network.encode(images).latent_dist
        # Author concatenates statistics into contiguous BTCHW before sampling.
        # randn_like on a channels-last view maps RNG draws to different pixels.
        mean=posterior.mean.contiguous();logvar=posterior.logvar.contiguous()
        innovation=stream.normal(mean) if noise is None else noise
        latents=mean+innovation*torch.exp(.5*logvar)
        if self.settings.get('decoder_r1_weight',0) and not latents.requires_grad:latents=latents.requires_grad_(True)
        # Author concatenates decoded BTCHW chunks. This produces contiguous
        # pixels even when the decoder emits channels-last storage. VGG's CPU
        # kernels otherwise use a different accumulation order in backward.
        decode=lambda x:network.decode(x,return_dict=False)[0].contiguous()
        prediction=decode(latents)
        step=(context or {}).get('step',0);kl=self.kl
        schedule=self.settings.get('kl_schedule',[])
        if schedule:kl=next((weight for boundary,weight in schedule if step<boundary),schedule[-1][1])
        elif self.settings.get('kl_anneal_steps'):kl=kl*min(1.,step/self.settings['kl_anneal_steps'])
        native_style=self.settings.get('reconstruction_style')=='native'
        if self.perceptual is not None:self.perceptual.to(images.device)
        kl_loss=kl*mean_flat(-.5*(1+logvar-mean.pow(2)-logvar.exp()))
        if native_style:terms={'reconstruction':self.native_reconstruction(prediction,images),**({'kl':kl_loss} if kl else {})}
        else:
            terms={'reconstruction':self.mse*mean_flat((prediction-images)**2),'kl':kl_loss}
            if self.settings.get('mae_weight'):terms['mae']=self.settings['mae_weight']*mean_flat((prediction-images).abs())
            if self.settings.get('pseudo_huber_weight'):
                width=self.settings.get('pseudo_huber_breadth',.01)
                terms['pseudo_huber']=self.settings['pseudo_huber_weight']*mean_flat(((prediction-images)**2+width**2).sqrt()-width)
            if self.perceptual is not None:terms['perceptual']=self.settings['perceptual'].get('weight',1.)*self.perceptual(prediction,images).reshape(len(images),-1).mean(1)
        regularizers=list(enumerate(self.regularizers))
        if self.settings.get('regularizer_selection')=='random_one' and regularizers:
            import numpy as np
            regularizers=[regularizers[int(np.random.choice(len(regularizers)))]]
        def reconstruction(a,b):
            if native_style:return self.native_reconstruction(a,b)
            value=self.mse*mean_flat((a-b)**2)
            if self.perceptual is not None:value=value+self.settings['perceptual'].get('weight',1.)*self.perceptual(a,b).reshape(len(a),-1).mean(1)
            if self.settings.get('mae_weight'):value=value+self.settings['mae_weight']*mean_flat((a-b).abs())
            if self.settings.get('pseudo_huber_weight'):
                width=self.settings.get('pseudo_huber_breadth',.01)
                value=value+self.settings['pseudo_huber_weight']*mean_flat(((a-b)**2+width**2).sqrt()-width)
            return value
        for i,regularizer in regularizers:
            if not self.settings.get('regularizer_ignore_perceptual',True):
                if not getattr(regularizer,'supports_reconstruction',False):raise ValueError('Regularizer does not accept a reconstruction objective')
                terms[f'aux{i}']=regularizer(decode,latents,images,reconstruction=reconstruction)
            else:terms[f'aux{i}']=regularizer(decode,latents,images)
        frequency=self.settings.get('decoder_r1_frequency',1);r1=self.settings.get('decoder_r1_weight',0.)
        if r1 and step%frequency==0:
            gradient=torch.autograd.grad(prediction.sum(),latents,create_graph=True,only_inputs=True)[0]
            terms['decoder_r1']=r1*frequency*mean_flat(gradient.square())
        if self.settings.get('high_frequency'):
            from .regularizers import native_dct_utilities
            config=dict(self.settings['high_frequency']);weight=config.pop('weight',1.)
            terms['high_frequency']=weight*native_dct_utilities()['reg_dc_dct2d_high_freqs'](latents,**config)
        return TrainingResult(sum(terms.values()),prediction,images,latents,terms)
    @torch.no_grad()
    def sample(self,model,initial,**kwargs):
        network=model.model if hasattr(model,'model') else model
        return network.decode(initial,return_dict=False)[0]

@register('model','autoencoder_kl')
def autoencoder_kl(**kwargs):
    from diffusers import AutoencoderKL
    return AutoencoderKL(**kwargs)
