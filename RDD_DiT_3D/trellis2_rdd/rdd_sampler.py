"""Piecewise-global Euler sampler for the topology-conditioned RDD SLat flow."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from easydict import EasyDict as edict

from trellis2.modules import sparse as sp
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler

from .hierarchy import build_masked_latent_pyramid, expand_parent_features, project_detail


class SparseRDDDetailEulerCfgSampler(FlowEulerCfgSampler):
    def __init__(
        self,
        sigma_min: float = 1e-5,
        resolutions=(8, 16, 32),
        time_fractions=(0.5, 0.3, 0.2),
    ):
        super().__init__(sigma_min=sigma_min)
        self.resolutions = tuple(int(r) for r in resolutions)
        fractions = np.asarray(time_fractions, dtype=np.float64)
        if len(fractions) != len(self.resolutions) or np.any(fractions <= 0):
            raise ValueError("time_fractions must be positive and match resolutions")
        fractions = fractions / fractions.sum()
        self.boundaries = np.concatenate(([0.0], np.cumsum(fractions)))

    def _band(self, stage_index: int) -> tuple[float, float]:
        band = len(self.resolutions) - 1 - stage_index
        return float(self.boundaries[band]), float(self.boundaries[band + 1])

    @staticmethod
    def _sparse(level, feats):
        batch = torch.zeros(
            (level.model_coords.shape[0], 1),
            dtype=torch.int32,
            device=level.model_coords.device,
        )
        coords = torch.cat((batch, level.model_coords.to(torch.int32)), dim=1)
        return sp.SparseTensor(feats=feats, coords=coords)

    @torch.no_grad()
    def sample(
        self,
        model,
        fine_coords: torch.Tensor,
        cond: Any,
        neg_cond: Any,
        *,
        channels: int = 32,
        steps=(8, 12, 20),
        guidance_strength: float = 3.0,
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        if fine_coords.ndim != 2 or fine_coords.shape[1] not in (3, 4):
            raise ValueError("fine_coords must have shape [N,3] or [N,4]")
        if fine_coords.shape[1] == 4:
            if not bool(torch.all(fine_coords[:, 0] == fine_coords[0, 0])):
                raise ValueError("sample one topology at a time")
            fine_coords = fine_coords[:, 1:]
        if len(steps) != len(self.resolutions):
            raise ValueError("steps must match resolutions")

        template = build_masked_latent_pyramid(
            fine_coords.long(),
            torch.zeros((fine_coords.shape[0], channels), device=fine_coords.device),
            fine_resolution=self.resolutions[-1],
            resolutions=self.resolutions,
        )
        parent_sample = None
        history = []
        for stage_index, (level, num_steps) in enumerate(zip(template.levels, steps)):
            if stage_index == 0:
                feats = torch.randn(
                    (level.logical_coords.shape[0], channels), device=fine_coords.device
                )
            else:
                upsampled = expand_parent_features(
                    template.levels[stage_index - 1], level, parent_sample.feats
                )
                detail_noise = project_detail(level.logical_coords, torch.randn_like(upsampled))
                feats = upsampled + detail_noise
            sample = self._sparse(level, feats)

            low, high = self._band(stage_index)
            times = np.linspace(high, low, int(num_steps) + 1)
            for t, t_prev in zip(times[:-1], times[1:]):
                pred = self._inference_model(
                    model,
                    sample,
                    float(t),
                    cond,
                    neg_cond=neg_cond,
                    guidance_strength=guidance_strength,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )
                if stage_index > 0:
                    pred = pred.replace(project_detail(level.logical_coords, pred.feats))
                sample = sample - float(t - t_prev) * pred
                if verbose:
                    history.append({"resolution": level.resolution, "t": float(t_prev)})
            parent_sample = sample

        return edict(samples=parent_sample, history=history)
