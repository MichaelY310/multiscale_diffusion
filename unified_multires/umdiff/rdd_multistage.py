"""Independent native DiTs selected by the actual latent resolution.

This is the old MultiStage topology, distinct from a shared DiTMultiRes. Each
child keeps its original fixed positional embedding and checkpoint key names.
"""
from pathlib import Path
import torch
from .registry import build, register


@register('model', 'native_rdd_multistage')
class NativeRDDMultistage(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        if not isinstance(models, dict) or not models:
            raise ValueError('Native multistage requires resolution -> model-spec mapping')
        keys = [str(int(key)) for key in models]
        if len(set(keys)) != len(keys) or any(int(k) <= 0 for k in keys):
            raise ValueError('Stage resolutions must be distinct positive integers')
        self.stages = torch.nn.ModuleDict()
        for resolution, spec in models.items():
            key = str(int(resolution))
            if spec.get('name') != 'native_rdd':
                raise ValueError('Legacy MultiStage uses the original fixed-grid DiT (native_rdd)')
            if int(spec.get('args', {}).get('input_size', 32)) != int(key):
                raise ValueError('Stage DiT input_size must equal its routing resolution')
            self.stages[key] = build('model', spec)
        channels = {model.in_channels for model in self.stages.values()}
        if len(channels) != 1:
            raise ValueError('Stage models must use the same latent channel count')
        self.in_channels = channels.pop()

    def for_resolution(self, resolution):
        key = str(int(resolution))
        if key not in self.stages:
            raise ValueError(f'No native model configured for {key} x {key}')
        return self.stages[key]

    def _route(self, x):
        if x.ndim != 4 or x.shape[-2] != x.shape[-1]:
            raise ValueError('Legacy stage routing requires square BCHW latents')
        return self.for_resolution(x.shape[-1])

    def forward(self, x, t, **kwargs):
        kwargs.pop('stage', None)  # Optional framework routing hint, not model conditioning.
        return self._route(x)(x, t, **kwargs)

    def forward_with_cfg(self, x, t, **kwargs):
        kwargs.pop('stage', None)
        return self._route(x).forward_with_cfg(x, t, **kwargs)

    def load_legacy_checkpoints(self, checkpoints, key='ema'):
        """Load explicitly mapped, trusted local stage checkpoints; never download.

        No architecture/step inference or EMA->raw fallback is performed. Mixed
        training-step stage bundles must be declared by the caller's mapping.
        """
        if {str(int(k)) for k in checkpoints} != set(self.stages):
            raise ValueError('One checkpoint must be supplied for every stage')
        evidence = []
        for resolution, path in checkpoints.items():
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location='cpu', weights_only=False)
            if not isinstance(state, dict) or key not in state:
                raise ValueError(f'{path}: required checkpoint key {key!r} is missing')
            model = self.for_resolution(resolution)
            model.load_state_dict(state[key], strict=True)
            evidence.append({'resolution': int(resolution), 'checkpoint': str(path.resolve()),
                             'weight_key': key, 'parameter_tensors': len(state[key])})
        return evidence
