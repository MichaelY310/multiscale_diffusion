"""CAT published nested-codec assembly, as an explicitly loaded extension.

This ports the released forward/KL assembly. The NeurIPS supplement does not
publish a diffusion trainer or the complete codec training objective. No such
training-reproduction claim is made by this adapter.
"""
from pathlib import Path
import importlib
import sys
import torch


def load_author(source_directory):
    """Import only the two model files; never execute asset/scoring scripts."""
    path = Path(source_directory).resolve()
    for filename in ("nested_vae.py", "nested_vae_modules.py"):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    sys.path.insert(0, str(path))
    return importlib.import_module("nested_vae").MultiScaleAutoencoderKL


def forward_components(network, sample, compression_rates=None,
                       sample_posterior=True, generator=None):
    """Independent composition of the author's encoder, posterior and decoder."""
    if compression_rates is None:
        compression_rates = torch.tensor([1] * len(sample)).to(sample.device).long()
    encoded, spatial_dims = network.encode(sample, compression_rates=compression_rates)
    posterior = encoded.latent_dist
    z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
    decoded, mid_loss = network.decode(
        z, compression_rates, spatial_dims,
        label=sample if network.mid_block_supervision else None,
    )
    kl = posterior.kl()
    objective = torch.sum(kl) / kl.shape[0] * network.kl_loss_weight + 0
    if mid_loss > 0:
        objective = objective + mid_loss / 4.0
    return decoded.sample, objective


class CATCodecAdapter(torch.nn.Module):
    """Native API adapter with component-composed forward and reusable weights."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, sample, compression_rates=None, sample_posterior=True,
                return_dict=True, generator=None):
        pixels, kl = forward_components(self.network, sample, compression_rates,
                                        sample_posterior, generator)
        return (pixels, kl) if return_dict else (pixels,)

    def encode(self, *args, **kwargs):
        return self.network.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.network.decode(*args, **kwargs)
