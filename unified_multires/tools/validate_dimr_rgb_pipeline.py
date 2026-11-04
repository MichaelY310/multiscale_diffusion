#!/usr/bin/env python3
"""CPU author DiMR RGB/VAE/loss/backward/sampler/decode conformance.

The VAE is frozen in the author training recipe. Codec input-adjoint probes are
separate diagnostics, not invented author VAE training losses. No downloads.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_deps'), str(ROOT / 'tools')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

from umdiff import build, native
from umdiff.engine import Method
from umdiff.input_transforms import VAEMoments
from umdiff.randomness import GlobalRandomStream
from activation_trace import ActivationAudit, byte_view
from validate_full_forward_backward import snapshot_rng, restore_rng, fingerprint
from validate_training import nonzero_output


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-vae', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(10212)
    np.random.seed(10212)
    random.seed(10212)
    start = time.time()
    checks, traces = [], []
    report = dict(status='running', gpu_used=False, downloads=False, full_vae=args.full_vae)
    active = None

    def exact(name, expected, actual):
        ok = (expected.shape == actual.shape and expected.dtype == actual.dtype
              and torch.equal(byte_view(expected), byte_view(actual)))
        checks.append(dict(name=name, byte_exact=ok, shape=list(expected.shape),
                           dtype=str(expected.dtype), max_abs=float((expected.detach().double()-actual.detach().double()).abs().max())))
        if not ok:
            raise AssertionError(name)

    def finish(audit):
        audit.finish_segment()
        data = audit.close()
        traces.append(data)
        if data['structural_mismatches'] or any(v['pairs'] != v['byte_exact'] for v in data['totals'].values()):
            raise AssertionError('Layer/gradient mismatch: ' + data['name'])

    try:
        source = native.load('dimr/libs/autoencoder.py')
        side = 256 if args.full_vae else 128
        cfg = dict(double_z=True, z_channels=4, resolution=256, in_channels=3,
                   out_ch=3, ch=128 if args.full_vae else 32, ch_mult=[1, 2, 4, 4],
                   num_res_blocks=2 if args.full_vae else 1, attn_resolutions=[], dropout=0.)
        # Materialize a nonzero random state in the exact source key layout,
        # then run the unmodified source checkpoint-loading constructor twice.
        seed_model = torch.nn.Module()
        seed_model.encoder = source.Encoder(**cfg)
        seed_model.decoder = source.Decoder(**cfg)
        seed_model.quant_conv = torch.nn.Conv2d(8, 8, 1)
        seed_model.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
        with tempfile.TemporaryDirectory(prefix='umdiff_dimr_rgb_') as tmp:
            weights = Path(tmp) / 'random_vae.pt'
            torch.save(seed_model.state_dict(), weights)
            report['temporary_random_checkpoint'] = dict(sha256=sha(weights), bytes=weights.stat().st_size)
            del seed_model
            vae_a = source.FrozenAutoencoderKL(cfg, 4, str(weights))
            spec = dict(name='local_dimr_autoencoder', args=dict(pretrained_path=str(weights)))
            if not args.full_vae:
                spec['args']['ddconfig'] = cfg
            codec = build('input_transform', dict(name='dimr_vae', args=dict(model=spec)))
            report['vae_parameters'] = sum(p.numel() for p in vae_a.parameters())
            for key, value in vae_a.state_dict().items():
                exact('loaded_vae/' + key, value, codec.model.state_dict()[key])
            c = json.loads((ROOT/'configs/paper_recipes/dimr_DiMR-G-2R_imagenet256.json').read_text())
            model_cfg = c['model']['args']['model']['args']['config']
            latent_side = side // 8
            for i, stage in enumerate(model_cfg['stage_configs']):
                stage.update(dim=32, hidden_dim=64, num_attention_heads=4, num_blocks=1,
                             max_height=latent_side // 2 if i == 0 else latent_side,
                             max_width=latent_side // 2 if i == 0 else latent_side)
            c['image_shape'] = [4, latent_side, latent_side]
            net_a = nonzero_output(build('model', c['model']))
            net_b = copy.deepcopy(net_a)
            method = Method(c)
            author = native.extract('dimr/train.py', ['check_zip', 'get_skip', 'stp', 'mos', 'Schedule', 'LSimple'],
                dict(np=np, torch=torch, multi_scale_targets=native.load('dimr/sde.py').multi_scale_targets))
            schedule = author['Schedule'](np.array(c['batch']['args']['betas']))
            clock = build('time_sampler', c['training']['time_sampler'])
            stream = GlobalRandomStream()
            image = torch.rand(1, 3, side, side) * 2 - 1
            report.update(rgb_shape=list(image.shape), effective_model=c['model'], vae_config=cfg,
                          diffusion_parameters=sum(p.numel() for p in net_a.parameters()))
            optimizer_a = torch.optim.AdamW(net_a.parameters(), lr=2e-4, betas=(.99, .99), weight_decay=.03)
            optimizer_b = torch.optim.AdamW(net_b.parameters(), lr=2e-4, betas=(.99, .99), weight_decay=.03)

            # Online RGB and source-style cached-moment training both exercised.
            for step, mode in enumerate(('rgb', 'cached_moments')):
                net_a.train(); net_b.train()
                optimizer_a.zero_grad(set_to_none=True); optimizer_b.zero_grad(set_to_none=True)
                active = ActivationAudit('train_' + mode, [('vae', vae_a, codec.model), ('network', net_a, net_b)], args.output/'layers')
                active.begin(step); active.side('native')
                before = snapshot_rng()
                with torch.no_grad():
                    ma = vae_a.encode_moments(image)
                    za = vae_a.sample(ma)
                labels_a = torch.tensor([1000 if random.random() < .1 else 7])
                loss_a = author['LSimple'](za, lambda x, log_snr, **kw: net_a(x, log_snr, **kw),
                    schedule, c['batch']['args']['loss_weights'], y=labels_a).mean()
                active.side('native', 'backward_recompute'); loss_a.backward()
                after = snapshot_rng()
                restore_rng(before); active.side('unified')
                with torch.no_grad():
                    mb = codec.encode_moments(image)
                    zb = VAEMoments()(mb, stream) if mode == 'cached_moments' else codec.posterior(mb, stream)
                labels_b = torch.tensor([1000 if random.random() < .1 else 7])
                times, _ = clock(len(zb), zb.device, stream)
                loss_b = method.training(net_b, zb, times, stream, model_kwargs={'y': labels_b}).loss.mean()
                active.side('unified', 'backward_recompute'); loss_b.backward()
                exact(mode+'/moments', ma, mb); exact(mode+'/latent', za, zb)
                exact(mode+'/labels', labels_a, labels_b); exact(mode+'/loss', loss_a, loss_b)
                if fingerprint(after) != fingerprint(snapshot_rng()):
                    raise AssertionError('RNG mismatch ' + mode)
                active.parameters(net_a, net_b)
                active.parameters(vae_a, codec.model, role='vae')
                finish(active); active = None
                if any(p.grad is not None for p in vae_a.parameters()) or any(p.grad is not None for p in codec.model.parameters()):
                    raise AssertionError('Author codec must remain frozen')
                optimizer_a.step(); optimizer_b.step()
                for key, value in net_a.state_dict().items():
                    exact(mode+'/updated_network/'+key, value, net_b.state_dict()[key])

            # Public transform itself, including its no_grad boundary.
            before = snapshot_rng()
            with torch.no_grad():
                expected = vae_a.encode(image)
            after = snapshot_rng(); restore_rng(before)
            actual = codec(image, stream)
            exact('public_rgb_input_transform', expected, actual)
            if actual.requires_grad or fingerprint(after) != fingerprint(snapshot_rng()):
                raise AssertionError('Input-transform freeze/RNG boundary')

            # The released sampler uses the shared author DPM solver; its model
            # wrapper, guidance and latent-to-RGB assembly execute independently.
            net_a.eval(); net_b.eval()
            dpm = native.load('dimr/dpm_solver.py')
            noise_schedule = dpm.NoiseScheduleVP(schedule='discrete', betas=torch.tensor(c['batch']['args']['betas']).float())
            initial = torch.randn_like(za)
            for guidance, power in ((0., None), (.5, -1), (1.5, 2)):
                active = ActivationAudit('sample_cfg_' + str(guidance), [('vae', vae_a, codec.model), ('network', net_a, net_b)], args.output/'layers')
                active.begin(0); active.side('native')
                labels = torch.tensor([7])
                if power is None:
                    env = dict(cfg_nnet=lambda x, t, log_snr, **kw: net_a(x, log_snr, **kw)[-1])
                else:
                    env = native.statements('dimr/eval.py', 93, 115, dict(
                        nnet=lambda x, log_snr, **kw: net_a(x, log_snr, **kw),
                        dataset=SimpleNamespace(K=1000), device='cpu', cfg_scale=guidance,
                        cfg_scale_pow=power, torch=torch, math=math))
                env.update(N=1000, kwargs={'y': labels})
                env = native.statements('dimr/eval.py', 134, 142, env)
                with torch.no_grad():
                    latent_a = dpm.DPM_Solver(env['model_fn'], noise_schedule, predict_x0=True, thresholding=False).sample(initial, steps=12, eps=.001, T=1.)
                    pixels_a = vae_a.decode(latent_a)
                    active.side('unified')
                    sampler = build('sampler', dict(name='dimr_dpm', args=dict(betas=c['batch']['args']['betas'], steps=12, guidance_scale=guidance, guidance_power=power)))
                    latent_b = sampler(net_b, initial, stream, {'y': labels})
                    pixels_b = codec.decode(latent_b)
                exact('cfg'+str(guidance)+'/latent', latent_a, latent_b)
                exact('cfg'+str(guidance)+'/float_rgb', pixels_a, pixels_b)
                finish(active); active = None

            # Separate codec input derivatives. These are deliberately not
            # called author training: its VAE is frozen and sampled no_grad.
            for part in ('encoder', 'decoder'):
                xa = (image if part == 'encoder' else za).detach().clone().requires_grad_(True)
                xb = xa.detach().clone().requires_grad_(True)
                active = ActivationAudit('diagnostic_' + part, [('vae', vae_a, codec.model)], args.output/'layers')
                active.begin(0); active.side('native')
                ya = vae_a.encode_moments(xa) if part == 'encoder' else vae_a.decode(xa)
                ya.square().mean().backward()
                active.side('unified')
                yb = codec.encode_moments(xb) if part == 'encoder' else codec.decode(xb)
                yb.square().mean().backward()
                exact(part+'/diagnostic_output', ya, yb)
                exact(part+'/diagnostic_input_gradient', xa.grad, xb.grad)
                finish(active); active = None

            # A mismatched latent scale is visible; random weights do not make
            # the codec boundary test vacuous.
            with torch.no_grad():
                wrong = codec.decode(za * 1.01)
                correct = vae_a.decode(za)
            delta = float((wrong - correct).abs().max())
            if not delta > 0:
                raise AssertionError('Latent scaling negative control was ineffective')
            report['wrong_scale_negative_control'] = dict(max_abs=delta, detected=True)
        report.update(status='passed', temporary_checkpoint_removed=not weights.exists(),
            scope='CPU FP32; synthetic RGB, common nonzero random checkpoints; source VAE architecture '+('full' if args.full_vae else 'reduced')+
                  '; reduced source DiMR network. RGB/moments/posterior/scaling, two independent training F/B+AdamW updates, 12-step DPM with/without CFG, float decoding, separate encoder/decoder input-adjoint diagnostics. Shared author backbone/VAE/DPM classes. No formal training or pretrained quality.',
            frozen_vae_gradients='Absent on both sides in actual training, as required by the author. Diagnostic codec gradients are input derivatives only.')
    except Exception:
        report.update(status='failed', failure=traceback.format_exc())
        print(report['failure'], flush=True)
    if active is not None:
        traces.append(active.close())
    report.update(checks=checks, traces=traces, seconds=time.time()-start,
                  cuda_initialized=torch.cuda.is_initialized(), source_executions=native.AUDIT,
                  harness_sha256=sha(__file__), implementation_sha256=sha(ROOT/'umdiff/compat/dimr_codec.py'))
    (args.output/'dimr_rgb_pipeline.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'seconds', 'cuda_initialized')}), flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
