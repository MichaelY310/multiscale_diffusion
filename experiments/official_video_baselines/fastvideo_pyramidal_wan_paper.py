"""Strict *paper implementation* of plain PyramidalWan in current FastVideo.

This is not author-source reproduction. Published equations, three-dimensional
pyramids, first-frame conditioning and packed self/cross masks are preserved.
Undisclosed choices live in the run configuration and paper contract.
"""
from typing import Any
import json
import hashlib
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.train.models.base import ModelBase
from fastvideo.train.methods.base import TrainingMethod
from pyramidal_wan_network import PaperWanTransformer
from pyramidal_wan_paper_math import NATURAL_CLEAN, NATURAL_NOISY, prepare, teacher_boundary


class PaperCacheDataset(Dataset):

    def __init__(self: Any, manifest: Any, split: Any = 'train', shape: Any = 'paper') -> None:
        self.manifest = Path(manifest)
        data = json.loads(self.manifest.read_text())
        if data.get('synthetic') is not False or data['posterior'].split(';')[0] != 'mode':
            raise ValueError('Require the audited real, separately RGB-encoded integration cache')
        self.rows = [x for x in data['rows'] if x['split'] == split]
        if not self.rows or shape not in ('paper', 'small'):
            raise ValueError('No matching real cache rows or invalid pyramid shape')
        self.shape = shape

    def __len__(self: Any) -> Any:
        return len(self.rows)

    def __getitem__(self: Any, index: Any) -> Any:
        row = self.rows[index]
        sample = torch.load(row['sample'], map_location='cpu', weights_only=True)
        if sample['id'] != row['id'] or sample['synthetic'] or sample['split'] != row['split']:
            raise ValueError('Cache identity/provenance mismatch')
        values = sample[self.shape + '_raw_pixel_pyramid']
        if len(values) != 3 or any(not torch.isfinite(x).all() for x in values):
            raise ValueError('Invalid pixel-before-VAE cache')
        if self.shape == 'paper' and [tuple(x.shape) for x in values] != [(1, 16, 21, 56, 104), (1, 16, 11, 28, 52),
                                                                          (1, 16, 6, 14, 26)]:
            raise ValueError('Published 81×448×832 pyramid dimensions changed')
        return {'id': sample['id'], 'pyramid': values, 'text': sample['text_embedding']}


def collate_samples(values: Any) -> Any:
    return values


class PaperPyramidalWanModel(ModelBase):

    def __init__(self: Any,
                 *,
                 init_from: Any,
                 manifest: Any = None,
                 cache_shape: Any = 'paper',
                 packed_backend: Any = 'flex',
                 trainable: Any = True,
                 training_config: Any,
                 gradient_checkpointing: Any = True) -> None:
        super().__init__(trainable=trainable)
        from diffusers import WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
        self.training_config = training_config
        self.init_from = Path(init_from)
        self.cache_shape = cache_shape
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
        dtype = torch.float32 if trainable else torch.bfloat16
        dit = WanTransformer3DModel.from_pretrained(str(self.init_from),
                                                    subfolder='transformer',
                                                    torch_dtype=dtype,
                                                    local_files_only=True).to(device)
        if trainable:
            self.transformer = PaperWanTransformer(dit, backend=packed_backend).to(device)
            self.transformer.gradient_checkpointing = bool(gradient_checkpointing)
        else:
            self.transformer = dit.requires_grad_(False).eval()
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.init_from),
                                                                               subfolder='scheduler',
                                                                               local_files_only=True)
        vae = json.loads((self.init_from / 'vae/config.json').read_text())
        self.mean = torch.tensor(vae['latents_mean'], device=device).reshape(1, 16, 1, 1, 1)
        self.std = torch.tensor(vae['latents_std'], device=device).reshape(1, 16, 1, 1, 1)
        if trainable:
            if manifest is None:
                raise ValueError('Paper training requires a cache manifest')
            self.manifest = str(Path(manifest).resolve())
            dataset = PaperCacheDataset(manifest, shape=cache_shape)
            rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
            sampler = DistributedSampler(dataset,
                                         num_replicas=world,
                                         rank=rank,
                                         shuffle=True,
                                         seed=training_config.data.seed,
                                         drop_last=True)
            self.dataloader = StatefulDataLoader(dataset,
                                                 batch_size=training_config.data.train_batch_size,
                                                 sampler=sampler,
                                                 num_workers=0,
                                                 drop_last=True,
                                                 collate_fn=collate_samples,
                                                 generator=torch.Generator().manual_seed(training_config.data.seed +
                                                                                         10000 + rank))
        self.start_step = 0

    @property
    def device(self: Any) -> Any:
        return self.mean.device

    def normalize(self: Any, value: Any) -> Any:
        return (value.to(self.device, dtype=torch.float32) - self.mean) * (1.0 / self.std)

    def prepare_batch(self: Any, raw_batch: Any, *, generator: Any, latents_source: Any = 'data') -> Any:
        if latents_source != 'data':
            raise ValueError('PyramidalWan requires separately RGB-encoded clean pyramid')
        return [{
            'id': row['id'],
            'pyramid': [self.normalize(x) for x in row['pyramid']],
            'text': row['text'].to(self.device, dtype=torch.bfloat16)
        } for row in raw_batch]

    def add_noise(self: Any, clean_latents: Any, noise: Any, timestep: Any) -> Any:
        raise NotImplementedError('Use Eq.1–4 stage endpoints, not the vanilla Wan path')

    def predict_noise(self: Any, noisy_latents: Any, timestep: Any, batch: Any, *, conditional: Any,
                      **kwargs: Any) -> Any:
        if not conditional:
            raise ValueError('Training uses paired conditional teacher/student forwards')
        return self.transformer(noisy_latents,
                                timestep=timestep,
                                encoder_hidden_states=batch['text'],
                                return_dict=False)[0]

    def backward(self: Any, loss: Any, ctx: Any, *, grad_accum_rounds: Any = 1) -> Any:
        (loss / max(1, grad_accum_rounds)).backward()


class _PaperControl:

    def __init__(self: Any, method: Any) -> None:
        self.method = method

    def state_dict(self: Any) -> Any:
        m = self.method
        return {
            'manifest': m.student.manifest,
            'teacher_weight': m.teacher_weight,
            'manifest_sha256': hashlib.sha256(Path(m.student.manifest).read_bytes()).hexdigest(),
            'cache_shape': m.student.cache_shape,
            'optimizer': {
                'lr': m.training_config.optimizer.learning_rate,
                'betas': list(m.training_config.optimizer.betas),
                'weight_decay': m.training_config.optimizer.weight_decay,
                'eps': float(m.method_config.get('optimizer_eps', 1e-8))
            },
            'clip_grad': m.clip_grad,
            'teacher_init': str(m.teacher.init_from),
            'stages_per_batch': m.per_stage,
            'natural_boundaries': list(NATURAL_CLEAN),
            'time_sampling': 'uniform_natural_appendix_A3',
            'paper_version': 'arxiv:2601.04792v1'
        }

    def load_state_dict(self: Any, value: Any) -> Any:
        if value != self.state_dict():
            raise ValueError('Paper training contract changed on checkpoint resume')


class PaperPyramidalWanMethod(TrainingMethod):

    def __init__(self: Any, *, cfg: Any, role_models: Any) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        self.teacher = role_models.get('teacher')
        if not isinstance(self.student, PaperPyramidalWanModel) or self.teacher is None or self.teacher._trainable:
            raise ValueError('Trainable paper student and frozen original Wan teacher are required')
        self.per_stage = int(cfg.method.get('examples_per_stage', 2))
        if self.per_stage < 1 or cfg.training.data.train_batch_size != 3 * self.per_stage:
            raise ValueError('Every rank must draw the same number of examples from each of the three stages')
        self.teacher_weight = float(cfg.method.get('teacher_weight', 1.0))
        if self.teacher_weight <= 0:
            raise ValueError('Plain PyramidalWan requires partial teacher supervision')
        self.clip_grad = float(cfg.method.get('clip_grad', 1.0))
        # Replicated DDP preserves the exact packed per-rank paper computation.
        # SP/TP would need an additional block-mask/RoPE parity test.
        if cfg.training.distributed.sp_size != 1 or cfg.training.distributed.tp_size != 1:
            raise ValueError('Paper implementation currently requires SP=TP=1')
        if dist.is_initialized() and dist.get_world_size() > 1:
            from torch.nn.parallel import DistributedDataParallel
            self.student.transformer = DistributedDataParallel(self.student.transformer,
                                                               device_ids=[self.student.device.index],
                                                               broadcast_buffers=False)
            self.role_modules['student']['transformer'] = self.student.transformer
        oc = cfg.training.optimizer
        self.optimizer = torch.optim.AdamW(self.student.transformer.parameters(),
                                           lr=oc.learning_rate,
                                           betas=tuple(oc.betas),
                                           weight_decay=oc.weight_decay,
                                           eps=float(cfg.method.get('optimizer_eps', 1e-8)),
                                           foreach=False)
        from torch.optim.lr_scheduler import LambdaLR
        self.lr_scheduler = LambdaLR(self.optimizer, lambda step: 1.)
        self.control = _PaperControl(self)
        self.last_stage_losses: list[Any] = []

    def on_train_start(self: Any) -> Any:
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.cuda_generator = torch.Generator(device=self.student.device).manual_seed(self.training_config.data.seed +
                                                                                      rank)
        self.student.transformer.train()
        self.teacher.transformer.eval()

    def single_train_step(self: Any, batch: Any, iteration: Any) -> Any:
        if time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', 'inf')) - 5:
            raise RuntimeError('Stop before the independent GPU deadline')
        samples = self.student.prepare_batch(batch, generator=self.cuda_generator)
        if len(samples) != 3 * self.per_stage:
            raise ValueError('Incomplete packed stage-balanced batch; increase cache size or use drop_last')
        prepared, texts, teacher_targets = [], [], []
        # Teacher is evaluated on full resolution for *each* example, including
        # coarse stages. Same epsilon couples the original and pyramidal paths.
        for n, sample in enumerate(samples):
            i = n // self.per_stage
            full = sample['pyramid'][0]
            eps = torch.randn(full.shape, device=full.device, dtype=full.dtype, generator=self.cuda_generator)
            z = torch.rand(1, device=full.device, generator=self.cuda_generator)
            z = z * (NATURAL_NOISY[i] - NATURAL_CLEAN[i]) + NATURAL_CLEAN[i]
            p = prepare(sample['pyramid'], eps, z, i)
            prepared.append(p)
            texts.append(sample['text'])
            # Frozen native BF16 teacher retains its original FP32 time layers.
            # Extra AMP would silently change the pretrained teacher function.
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                x = (1 - z.reshape(1, 1, 1, 1, 1)) * full + z.reshape(1, 1, 1, 1, 1) * eps
                flow = self.teacher.transformer(x.to(torch.bfloat16),
                                                timestep=z * 1000,
                                                encoder_hidden_states=sample['text'],
                                                return_dict=False)[0].float()
                partial = x - (z.reshape(1, 1, 1, 1, 1) - NATURAL_CLEAN[i]) * flow
                teacher_targets.append(teacher_boundary(partial, i))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predictions = self.student.transformer([p.noisy for p in prepared], [p.natural for p in prepared], texts)
        fm, teacher = [], []
        for p, prediction, target in zip(prepared, predictions, teacher_targets, strict=False):
            fm.append(F.mse_loss(prediction.float(), p.target))
            partial = p.noisy - (p.sigma - p.cleaner_sigma) * prediction.float()
            teacher.append(F.mse_loss(partial, target))
        fm_loss, teacher_loss = torch.stack(fm).mean(), torch.stack(teacher).mean()
        total = fm_loss + self.teacher_weight * teacher_loss
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite published-equation training objective')
        return {
            'total_loss': total,
            'flow_loss': fm_loss,
            'teacher_loss': teacher_loss
        }, {}, {
            'paper/teacher_forwards': len(samples),
            'paper/packed_examples': len(samples),
            **{
                f'paper/stage{i}_fm': torch.stack(fm[i * self.per_stage:(i + 1) * self.per_stage]).mean().detach()
                for i in range(3)
            }
        }

    def optimizers_schedulers_step(self: Any, iteration: Any) -> Any:
        if self.clip_grad > 0:
            norm = torch.nn.utils.clip_grad_norm_(self.student.transformer.parameters(), self.clip_grad)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite PyramidalWan gradient')
        self.optimizer.step()
        self.lr_scheduler.step()

    def get_optimizers(self: Any, iteration: Any) -> Any:
        return [self.optimizer]

    def get_lr_schedulers(self: Any, iteration: Any) -> Any:
        return [self.lr_scheduler]

    @property
    def _optimizer_dict(self: Any) -> Any:
        return {'student': self.optimizer}

    @property
    def _lr_scheduler_dict(self: Any) -> Any:
        return {'student': self.lr_scheduler}

    def checkpoint_state(self: Any) -> Any:
        values = super().checkpoint_state()
        values['pyramidal_wan_contract'] = self.control
        return values
