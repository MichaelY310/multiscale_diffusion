"""RDD detail-subspace flow matching trainer for the official TRELLIS.2 code.

This module intentionally lives outside ``third_party/TRELLIS.2``.  The small
launcher in :mod:`train_trellis2_rdd` registers the classes at runtime, keeping
the pinned upstream checkout unmodified.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from trellis2.modules import sparse as sp
from trellis2.trainers.flow_matching.mixins.classifier_free_guidance import (
    ClassifierFreeGuidanceMixin,
)
from trellis2.trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin
from trellis2.trainers.flow_matching.sparse_flow_matching import SparseFlowMatchingTrainer

from .hierarchy import build_masked_latent_pyramid, project_detail


class SparseRDDDetailFlowMatchingTrainer(SparseFlowMatchingTrainer):
    """Flow-match one shared model over coarse means and sparse latent details.

    ``resolutions`` are ordered coarse-to-fine.  ``time_fractions`` are ordered
    fine-to-coarse because reverse-time sampling visits the coarse band first.
    ``stage_probabilities`` control training exposure independently of interval
    width; equal probabilities are a robust default for fine-tuning.
    """

    def __init__(
        self,
        *args,
        resolutions=(8, 16, 32),
        time_fractions=(0.5, 0.3, 0.2),
        stage_probabilities=None,
        normalize_global_velocity_loss=True,
        **kwargs,
    ):
        self.rdd_resolutions = tuple(int(r) for r in resolutions)
        self.rdd_fine_resolution = self.rdd_resolutions[-1]
        if any(b != 2 * a for a, b in zip(self.rdd_resolutions, self.rdd_resolutions[1:])):
            raise ValueError("RDD resolutions must form a dyadic hierarchy")
        fractions = torch.tensor(time_fractions, dtype=torch.float64)
        if len(fractions) != len(self.rdd_resolutions) or bool(torch.any(fractions <= 0)):
            raise ValueError("time_fractions must be positive and match resolutions")
        fractions = fractions / fractions.sum()
        # Boundaries are in global clean->noise time, bands fine->coarse.
        self.rdd_time_boundaries = torch.cat((torch.zeros(1, dtype=torch.float64), fractions.cumsum(0)))
        if stage_probabilities is None:
            stage_probabilities = [1.0 / len(self.rdd_resolutions)] * len(self.rdd_resolutions)
        probs = torch.tensor(stage_probabilities, dtype=torch.float64)
        if len(probs) != len(self.rdd_resolutions) or bool(torch.any(probs <= 0)):
            raise ValueError("stage_probabilities must be positive and match resolutions")
        self.rdd_stage_probabilities = probs / probs.sum()
        self.normalize_global_velocity_loss = bool(normalize_global_velocity_loss)
        super().__init__(*args, **kwargs)

    def _stage_time(self, stage_index: int, local_p: torch.Tensor) -> torch.Tensor:
        # stage_index is coarse->fine; time bands are fine->coarse.
        band = len(self.rdd_resolutions) - 1 - stage_index
        low = self.rdd_time_boundaries[band].to(local_p.device, local_p.dtype)
        high = self.rdd_time_boundaries[band + 1].to(local_p.device, local_p.dtype)
        return low + (high - low) * local_p

    def _stage_width(self, stage_index: int, reference: torch.Tensor) -> torch.Tensor:
        band = len(self.rdd_resolutions) - 1 - stage_index
        return (self.rdd_time_boundaries[band + 1] - self.rdd_time_boundaries[band]).to(
            reference.device, reference.dtype
        )

    def _make_training_state(self, x_0: sp.SparseTensor):
        batch_size = x_0.shape[0]
        stage_ids = torch.multinomial(
            self.rdd_stage_probabilities.to(x_0.device), batch_size, replacement=True
        )
        local_p = self.sample_t(batch_size).to(x_0.device).float()

        x_feats, target_feats, coords_list, logical_list = [], [], [], []
        global_t = []
        for batch_index, layout in enumerate(x_0.layout):
            fine_coords = x_0.coords[layout, 1:].long()
            fine_feats = x_0.feats[layout]
            pyramid = build_masked_latent_pyramid(
                fine_coords,
                fine_feats,
                fine_resolution=self.rdd_fine_resolution,
                resolutions=self.rdd_resolutions,
            )
            stage = int(stage_ids[batch_index])
            p = local_p[batch_index]
            sigma = self.sigma_min + (1.0 - self.sigma_min) * p

            if stage == 0:
                level = pyramid.levels[0]
                noise = torch.randn_like(level.feats)
                state = (1.0 - p) * level.feats + sigma * noise
                target = (1.0 - self.sigma_min) * noise - level.feats
            else:
                transition = pyramid.transitions[stage - 1]
                level = transition.child
                detail_noise = project_detail(level.logical_coords, torch.randn_like(level.feats))
                state = transition.upsampled + (1.0 - p) * transition.detail + sigma * detail_noise
                target = (1.0 - self.sigma_min) * detail_noise - transition.detail

            # The path is parameterized by global t, not stage-local p.
            target = target / self._stage_width(stage, p)

            batch_column = torch.zeros(
                (level.model_coords.shape[0], 1), dtype=x_0.coords.dtype, device=x_0.coords.device
            )
            coords_list.append(torch.cat((batch_column, level.model_coords.to(x_0.coords.dtype)), dim=1))
            logical_list.append(level.logical_coords)
            x_feats.append(state)
            target_feats.append(target)
            global_t.append(self._stage_time(stage, p))

        x_t = sp.SparseTensor.from_tensor_list(x_feats, coords_list)
        target = sp.SparseTensor.from_tensor_list(target_feats, coords_list)
        return x_t, target, torch.stack(global_t), stage_ids, logical_list

    def _project_prediction(self, pred: sp.SparseTensor, stage_ids: torch.Tensor, logical_list):
        projected = []
        for batch_index, layout in enumerate(pred.layout):
            values = pred.feats[layout]
            if int(stage_ids[batch_index]) > 0:
                values = project_detail(logical_list[batch_index], values)
            projected.append(values)
        return pred.replace(torch.cat(projected, dim=0))

    def training_losses(self, x_0: sp.SparseTensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        x_t, target, global_t, stage_ids, logical_list = self._make_training_state(x_0)
        cond = self.get_cond(cond, **kwargs)
        pred = self.training_models["denoiser"](x_t, global_t * 1000, cond, **kwargs)
        pred = self._project_prediction(pred, stage_ids, logical_list)

        # ``target`` is the derivative with respect to *global* time, hence the
        # division by band width in _make_training_state.  Its squared norm is
        # mechanically amplified by 1 / width**2.  Multiplying each per-instance
        # MSE by width**2 leaves the optimum vector field unchanged inside that
        # band, while preventing the narrow coarse band from dominating gradients.
        raw_instance_losses = torch.stack(
            [F.mse_loss(pred.feats[s], target.feats[s]) for s in pred.layout]
        )
        if self.normalize_global_velocity_loss:
            loss_weights = torch.stack(
                [
                    self._stage_width(int(stage_ids[i]), raw_instance_losses[i]).square()
                    for i in range(len(raw_instance_losses))
                ]
            )
        else:
            loss_weights = torch.ones_like(raw_instance_losses)
        weighted_instance_losses = raw_instance_losses * loss_weights

        terms = edict()
        terms["mse"] = weighted_instance_losses.mean()
        terms["raw_global_velocity_mse"] = raw_instance_losses.detach().mean()
        terms["loss"] = terms["mse"]
        per_instance = raw_instance_losses.detach().cpu().numpy()
        weighted_per_instance = weighted_instance_losses.detach().cpu().numpy()
        for stage_index, resolution in enumerate(self.rdd_resolutions):
            selected = stage_ids.detach().cpu().numpy() == stage_index
            if selected.any():
                terms[f"stage_{resolution}"] = {
                    "raw_global_velocity_mse": float(per_instance[selected].mean()),
                    "weighted_mse": float(weighted_per_instance[selected].mean()),
                    "tokens": float(np.mean([pred.seqlen[i].item() for i in np.flatnonzero(selected)])),
                }
        return terms, {}

    def snapshot(self, *args, **kwargs):
        """Disable upstream's flat-FM training snapshots for the RDD model.

        ``BasicTrainer.run`` forces snapshots at initialization and completion,
        even when ``i_sample`` is beyond ``max_steps``.  The inherited sparse-FM
        snapshot uses a flat sampler on the fine support and therefore does not
        represent the model trained here.  Generation is intentionally delegated
        to the topology-conditioned paired evaluator.
        """

        if self.is_master:
            print("Skipping flat-FM snapshot for RDD; use the paired RDD evaluator.")


class SparseRDDDetailFlowMatchingCFGTrainer(
    ClassifierFreeGuidanceMixin, SparseRDDDetailFlowMatchingTrainer
):
    pass


class ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer(
    ImageConditionedMixin, SparseRDDDetailFlowMatchingCFGTrainer
):
    pass
