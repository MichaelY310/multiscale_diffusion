"""Exact first-frame-preserving RDD projections for native odd-frame Wan latents.

Core rdd.flow stays unchanged. Average/repeat projections remain idempotent;
first-frame noise uses spatial scale, tail noise uses spatial*sqrt(temporal).
"""
from typing import Any
import math
import torch
import torch.nn.functional as F
from rdd.flow import RDDPath
from native_wan_adapter import NativeWanComparisonModel


class FirstFrameProjectionPath(RDDPath):

    @staticmethod
    def pool(x: Any, spatial: Any, temporal: Any) -> Any:
        if x.ndim != 5 or (x.shape[2] - 1) % temporal or x.shape[3] % spatial or x.shape[4] % spatial:
            raise ValueError('Exact first-frame/tail projection requires divisible spatial dimensions and T-1')
        first = F.avg_pool3d(x[:, :, :1], (1, spatial, spatial))
        if x.shape[2] == 1:
            return first
        tail = F.avg_pool3d(x[:, :, 1:], (temporal, spatial, spatial))
        return torch.cat((first, tail), 2)

    @staticmethod
    def up(x: Any, spatial: Any, temporal: Any) -> Any:
        first = x[:, :, :1].repeat_interleave(spatial, 3).repeat_interleave(spatial, 4)
        tail = x[:, :, 1:].repeat_interleave(temporal, 2).repeat_interleave(spatial, 3).repeat_interleave(spatial, 4)
        return torch.cat((first, tail), 2)

    def project(self: Any, x: Any, stage: Any) -> Any:
        if stage == 0:
            return x
        old, new = self.factors[stage - 1:stage + 1]
        spatial, temporal = old[0] // new[0], old[1] // new[1]
        return self.up(self.pool(x, spatial, temporal), spatial, temporal)

    def noise_at_stage(self: Any, full_noise: Any, stage: Any) -> Any:
        spatial, temporal = self.factors[stage]
        noise = self.pool(full_noise, spatial, temporal)
        return torch.cat((noise[:, :, :1] * spatial, noise[:, :, 1:] * spatial * math.sqrt(temporal)), 2)

    def transition(self: Any, clean: Any, full_shape: Any, next_stage: Any, generator: Any) -> Any:
        old, new = self.factors[next_stage - 1:next_stage + 1]
        lifted = self.up(clean, old[0] // new[0], old[1] // new[1])
        expected = (1 + (full_shape[2] - 1) // new[1], full_shape[3] // new[0], full_shape[4] // new[0])
        if tuple(lifted.shape[2:]) != expected:
            raise ValueError('Projection transition shape changed')
        t = self.boundaries[next_stage]
        return t * lifted + (1 - t) * torch.randn(
            lifted.shape, device=lifted.device, dtype=lifted.dtype, generator=generator)


class FirstFrameDriftFreePath(FirstFrameProjectionPath):

    def coefficients(self: Any, t: Any, stage: Any) -> Any:
        return torch.zeros_like(t), torch.zeros_like(t)

    def clean_from_velocity(self: Any, x: Any, w: Any, t: Any, stage: Any) -> Any:
        return x - (1 - t.reshape(-1, 1, 1, 1, 1).to(x)) * w


class ProjectedRDDWanModel(NativeWanComparisonModel):

    def __init__(self: Any, *, projection: Any = 'joint', comparison: Any = 'rdd', **kwargs: Any) -> None:
        if projection not in ('spatial', 'temporal', 'joint') or comparison not in ('rdd', 'staged_drift_free'):
            raise ValueError(
                'RDD projection ablation supports spatial/temporal/joint and separately trained drift-free')
        super().__init__(comparison=comparison, **kwargs)
        factors = {
            'spatial': ((4, 1), (2, 1), (1, 1)),
            'temporal': ((1, 4), (1, 2), (1, 1)),
            'joint': ((4, 4), (2, 2), (1, 1))
        }[projection]
        cls = FirstFrameProjectionPath if comparison == 'rdd' else FirstFrameDriftFreePath
        self.rdd_path = cls(factors=factors, boundaries=self.rdd_path.boundaries)
        self.projection = projection

    def _prepare_dit_inputs(self: Any, batch: Any, generator: Any) -> Any:
        full = batch.latents.float()
        tc = self.training_config
        expected = (tc.data.num_latent_t, tc.data.num_height // 8, tc.data.num_width // 8)
        if tuple(full.shape[2:]) != expected:
            raise ValueError('Recache; silent crop is prohibited')
        stage = self.rdd_stage
        path = self.rdd_path
        clean = path.stage_clean(full, stage)
        a, b = path.boundaries[stage:stage + 2]
        t = torch.rand(full.shape[0], device=full.device, generator=generator) * (b - a) + a
        if self.rdd_fixed_t is not None:
            t.fill_(self.rdd_fixed_t)
        full_noise = torch.randn(full.shape, device=full.device, generator=generator, dtype=torch.float32)
        noise = path.noise_at_stage(full_noise, stage)
        noisy, target = path.noisy_and_target(clean, noise, t, stage)
        self.rdd_target = target
        self.comparison_full, self.comparison_full_noise, self.comparison_t = full, full_noise, t
        batch.latents = clean.permute(0, 2, 1, 3, 4)
        batch.noisy_model_input, batch.noise = noisy, noise
        batch.sigmas = (1 - t).reshape(-1, 1, 1, 1, 1)
        batch.timesteps = (1 - t) * 1000
        batch.raw_latent_shape = clean.shape
        batch.conditional_dict = {
            'encoder_hidden_states': batch.encoder_hidden_states,
            'encoder_attention_mask': batch.encoder_attention_mask
        }
        batch.unconditional_dict = {
            'encoder_hidden_states': self.negative_prompt_embeds,
            'encoder_attention_mask': self.negative_prompt_attention_mask
        }
        return batch
