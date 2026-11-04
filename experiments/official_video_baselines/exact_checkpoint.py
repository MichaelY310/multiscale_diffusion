"""Full-role DCP and rank-local data state for the pinned FastVideo runtime.

The upstream trainable-only ModelWrapper compares canonical DCP keys with
DDP-prefixed named_parameters, which silently filters every model tensor out.
Do not modify the shared FastVideo tree: use its public DCP state API here.
"""
from typing import Any
from pathlib import Path
import hashlib
import json
import os
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (get_model_state_dict, set_model_state_dict, StateDictOptions)
from fastvideo.train.utils.checkpoint import CheckpointManager


class FullRoleState:

    def __init__(self: Any, model: Any) -> None:
        self.model = model

    def state_dict(self: Any) -> Any:
        value = get_model_state_dict(self.model)
        if not value:
            raise ValueError('A full trainable role cannot have an empty checkpoint')
        return value

    def load_state_dict(self: Any, value: Any) -> Any:
        if not value:
            raise ValueError('Reject a checkpoint with no trainable model tensors')
        set_model_state_dict(self.model, model_state_dict=value, options=StateDictOptions(strict=True))


class ExactCheckpointManager(CheckpointManager):

    def _contract(self: Any) -> Any:
        raw = self._raw_config or {}
        training = raw.get('training', {})
        return {
            k: raw.get(k, {})
            for k in ('models', 'method', 'pipeline', 'callbacks')
        } | {
            'training': {
                k: training.get(k)
                for k in ('data', 'optimizer', 'distributed', 'dit_precision', 'model')
            },
            'gradient_accumulation_steps': training.get('loop', {}).get('gradient_accumulation_steps', 1),
            'configured_max_train_steps': training.get('loop', {}).get('max_train_steps'),
            'implementation': self._implementation_contract()
        }

    def _implementation_contract(self: Any) -> Any:
        # Resume only with the same objective, codec and storage implementation.
        # Documentation or invocation caps do not alter the training contract.
        helpers = Path(__file__).parent
        files = [
            helpers / name
            for name in ('native_wan_adapter.py', 'fastvideo_native_pyramid.py', 'fastvideo_pyramidal_wan_paper.py',
                         'pyramidal_wan_paper_math.py', 'pyramidal_wan_network.py', 'pyramid_compat.py',
                         'exact_checkpoint.py', 'fastvideo_native_checkpoint.py', 'pyramid_runner_state.py',
                         'pyramid_dcp_state.py', 'spatiotemporal_rdd.py')
        ]
        rdd = helpers.parents[2] / 'FastVideo-RDD-minimal/rdd'
        files += [rdd / name for name in ('flow.py', 'finetune.py', 'comparison/paths.py', 'comparison/training.py')]
        return {
            str(path.relative_to(helpers.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }

    def _write_metadata(self: Any, path: Any, step: Any) -> Any:
        super()._write_metadata(path, step)
        file = path / 'metadata.json'
        metadata = json.loads(file.read_text())
        metadata['training_contract'] = self._contract()
        metadata['checkpoint_protocol'] = 'exact_full_roles_atomic_v2'
        file.write_text(json.dumps(metadata, indent=2))

    def _checkpoint_dir(self: Any, step: Any) -> Any:
        if getattr(self, '_pending_step', None) == step:
            return Path(self.output_dir) / ('.checkpoint-' + str(step) + '-' + os.environ['RDD_JOB_TOKEN'] + '.pending')
        return super()._checkpoint_dir(step)

    def maybe_resume(self: Any, *, resume_from_checkpoint: Any) -> Any:
        if resume_from_checkpoint:
            from fastvideo.train.utils.checkpoint import _resolve_resume_checkpoint
            path = _resolve_resume_checkpoint(resume_from_checkpoint, output_dir=self.output_dir)
            if path is None:
                raise FileNotFoundError('Resume must resolve to an existing complete checkpoint')
            metadata = self.load_metadata(path)
            if metadata.get('checkpoint_protocol') != 'exact_full_roles_atomic_v2':
                raise ValueError('Resume requires the current atomic v2 source/precision/schedule contract')
            if metadata.get('checkpoint_protocol') == 'exact_full_roles_atomic_v2':
                if not (path / 'complete.json').is_file():
                    raise ValueError('Checkpoint is not committed')
                if metadata.get('training_contract') != self._contract():
                    raise ValueError('Training objective, base model, data, precision or optimizer changed on resume')
                rank = dist.get_rank() if dist.is_initialized() else 0
                rng_name = ('native_rng_rank' if hasattr(self.method.student, 'runner') else 'rng_state_rank')
                if not (path / (rng_name + str(rank) + '.pt')).is_file():
                    raise ValueError('Committed checkpoint has no rank-local RNG snapshot')
        return super().maybe_resume(resume_from_checkpoint=resume_from_checkpoint)

    def _build_states(self: Any) -> Any:
        values = super()._build_states()
        for role, model in self.method._role_models.items():
            if model._trainable:
                values[f'roles.{role}.transformer'] = FullRoleState(model.transformer)
        if 'dataloader' in values:
            from pyramid_dcp_state import NativeDCPDataLoader
            if not isinstance(values['dataloader'], NativeDCPDataLoader):
                # StatefulDataLoader also changes its nested snapshot shape
                # between an empty fresh iterator and a consumed iterator.
                values['dataloader'] = NativeDCPDataLoader(values['dataloader'])
        if dist.is_initialized() and dist.get_world_size() > 1 and 'dataloader' in values:
            # DCP replicated keys otherwise select one rank's sampler/bucket
            # state for all ranks. Distinct keys preserve each data stream.
            values[f'dataloader_rank{dist.get_rank()}'] = values.pop('dataloader')
        return values

    def save(self: Any, step: Any) -> Any:
        if self._last_saved_step == step:
            return
        if self.config.keep_last:
            raise ValueError('Checkpoint deletion is disabled; keep_last must be zero')
        destination = self._checkpoint_dir(step)
        self._pending_step = step
        pending = self._checkpoint_dir(step)
        try:
            occupied = destination.exists() or pending.exists()
            if dist.is_initialized():
                observations = [None] * dist.get_world_size()
                # All ranks must finish checking BEFORE any rank can mkdir.
                # Otherwise a slower rank mistakes rank0's new pending dir for
                # a pre-existing checkpoint and fails inside DCP's collective.
                dist.all_gather_object(observations, occupied)
                occupied = any(observations)
            if occupied:
                raise FileExistsError('Existing or pending checkpoint protected; use a fresh output directory')
            super().save(step)
            if not dist.is_initialized() or dist.get_rank() == 0:
                (pending / 'complete.json').write_text(
                    json.dumps({
                        'step': step,
                        'complete': True,
                        'protocol': 'exact_full_roles_atomic_v2'
                    }))
                pending.rename(destination)
            if dist.is_initialized():
                dist.barrier()
        finally:
            self._pending_step = None
