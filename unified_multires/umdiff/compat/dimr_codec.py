"""Local DiMR VAE with the author's codec and explicit latent arithmetic.

No asset downloads. The default architecture is the released SD-style codec;
ddconfig/embed_dim overrides are explicit architecture fixtures, not conversion.
"""
from pathlib import Path

import torch

from .. import native
from ..registry import build, register
from ..input_transforms import VAEMoments
from ..randomness import GlobalRandomStream


@register('model', 'local_dimr_autoencoder')
def local_dimr_autoencoder(pretrained_path, scale_factor=.18215, ddconfig=None, embed_dim=4):
    path = Path(pretrained_path)
    if not path.is_file():
        raise FileNotFoundError('Supply an existing local DiMR autoencoder state dict')
    source = native.load('dimr/libs/autoencoder.py')
    if ddconfig is None:
        if embed_dim != 4:
            raise ValueError('Default released DiMR VAE has embed_dim=4')
        return source.get_model(str(path), scale_factor=scale_factor)
    return source.FrozenAutoencoderKL(ddconfig, embed_dim, str(path), scale_factor)


@register('input_transform', 'dimr_vae')
class DiMRVAE:
    """Frozen RGB-to-latent transform and source-order float RGB decoding.

The public training call uses no_grad, as in the author trainer. encode_moments
and decode retain input derivatives for separately labelled diagnostic probes.
"""
    def __init__(self, model):
        if model.get('name') != 'local_dimr_autoencoder':
            raise ValueError('DiMRVAE requires its declared native codec architecture')
        self.model = build('model', model).eval().requires_grad_(False)
        self.posterior = VAEMoments(scale=self.model.scale_factor)

    def encode_moments(self, images):
        self.model.to(images.device)
        return self.model.quant_conv(self.model.encoder(images))

    @torch.no_grad()
    def __call__(self, images, stream):
        return self.posterior(self.encode_moments(images), stream)

    @torch.no_grad()
    def encode(self, images):
        return self(images, GlobalRandomStream())

    def decode(self, latents):
        self.model.to(latents.device)
        # Preserve reciprocal multiplication from the author (not z / scale).
        unscaled = (1. / self.model.scale_factor) * latents
        return self.model.decoder(self.model.post_quant_conv(unscaled))
