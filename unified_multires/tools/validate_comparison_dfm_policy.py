"""Author DFM single-phase objective/time law on the common DiT, CPU only.

The author SiT is intentionally replaced on BOTH sides by the same common DiT.
This verifies the objective transplant, not architectural or final-FID equality.
Version3's all-phase SUM deliberately differs from the author's random phase
mixture. Here that outer policy is disabled on both sides for component parity;
validate_all_stage_sum.py separately verifies the new outer objective.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_deps'), str(ROOT / 'tools')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
import yaml
from umdiff import build
from umdiff.cli import make_method, setup
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream, RandomStream
from activation_trace import ActivationAudit, byte_view
from validate_cli import equal

OUT = None
CONFIG = ROOT / 'configs/comparison_imagenet256/main/dfm_xl_2.json'


def capture():
    return torch.get_rng_state(), np.random.get_state()


def restore(state):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    activate('dfm', ['models', 'dfm_utils', 'loss'])
    import loss as author
    source_config = ROOT / 'sources/dfm/experiments/train/dfm_sit_b_256.yaml'
    official = yaml.safe_load(source_config.read_text())
    spec = json.loads(CONFIG.read_text())
    c = copy.deepcopy(spec['method'])
    outer_policy = c.pop('training_aggregation', None)
    if outer_policy is not None:
        # Compare the retained single-phase kernel against the author. This is
        # NOT a comparison of the intentionally different all-stage objective.
        clock_spec = copy.deepcopy(outer_policy['samplers'][0])
        clock_spec['args'].pop('stage')
        c['training']['time_sampler'] = clock_spec
    c['model']['args']['architecture'].update(hidden_size=32, depth=2, num_heads=4)
    for key in ('input_transform', 'input_transform_scope'):
        c['training'].pop(key)
    args = c['batch']['args']
    assert args['weighting'] == official['weighting'] == 'lognormal'
    assert args['stage_weights'] == official['stage_weights']
    assert official['enc_type'] == 'none' and official['proj_coeff'] == args['repa_weight'] == 0
    clock = build('time_sampler', c['training']['time_sampler'])
    method = make_method(c)
    torch.manual_seed(17)
    np.random.seed(17)
    a = build('model', c['model'])
    with torch.no_grad():
        for parameter in a.parameters():
            parameter.normal_(0, .03)
    b = copy.deepcopy(a)
    x = torch.randn(2, 4, 32, 32)
    labels = torch.tensor([11, 47])
    op = c['training']['optimizer']
    optimizers = [torch.optim.AdamW(net.parameters(), lr=op['lr'], betas=tuple(op['betas']),
                                   eps=op['eps'], weight_decay=op['weight_decay']) for net in (a, b)]
    objective = author.DFMSILoss(num_stages=official['stages_count'],
                               stage_weights=official['stage_weights'],
                               weighting=official['weighting'], path_type=official['path_type'],
                               prediction=official['prediction'])
    audit = ActivationAudit('dfm_common_log_normal', [('network', a, b)], OUT)
    state_a = state_b = capture()
    stages = {0: 0, 1: 0}
    maximum = 0.
    losses = []
    for step in range(64):
        audit.begin(step)
        audit.side('native')
        restore(state_a)
        optimizers[0].zero_grad(set_to_none=True)
        ref, projection = objective(lambda z, t, **kw: (a(z, t, **kw), None), x,
                                    model_kwargs={'y': labels})
        assert projection == 0
        ref = ref.mean()
        audit.side('native', 'backward_recompute')
        ref.backward()
        optimizers[0].step()
        state_a = capture()

        audit.side('unified')
        restore(state_b)
        optimizers[1].zero_grad(set_to_none=True)
        t, context = clock(len(x), x.device, GlobalRandomStream())
        stages[context['stage']] += 1
        result = method.training(b, x, t, GlobalRandomStream(), model_kwargs={'y': labels}, context=context)
        got = result.loss.mean()
        audit.side('unified', 'backward_recompute')
        got.backward()
        optimizers[1].step()
        state_b = capture()
        audit.parameters(a, b)
        audit.finish_segment()
        assert torch.equal(byte_view(ref.detach()), byte_view(got.detach())), ('loss', step)
        maximum = max(maximum, abs(ref.item() - got.item()))
        losses.append(got.item())
        assert equal(state_a, state_b), ('continuous RNG', step)
        assert equal(a.state_dict(), b.state_dict()), ('weights', step)
        assert equal(optimizers[0].state_dict(), optimizers[1].state_dict()), ('optimizer', step)
    activation = audit.close()
    (OUT / 'DFM_ACTIVATIONS.json').write_text(json.dumps(activation, indent=2) + '\n')
    assert all(stages.values()), stages
    assert not activation['structural_mismatches']
    assert all(v['pairs'] == v['byte_exact'] for v in activation['totals'].values())

    # Exercise the actual trainer's time-sampler wiring and the configured sampler.
    c['training']['optimizer']['accumulation_steps'] = 2
    run_method, net, trainer = setup(c, torch.device('cpu'), 27)
    run_clock = build('time_sampler', c['training']['time_sampler'])
    trainer.time_sampler = run_clock
    trainer_losses = []
    for _ in range(2):
        out = trainer.step(x, None, model_kwargs={'y': labels}, time_sampler=run_clock)
        assert torch.isfinite(out.loss).all()
        trainer_losses.append(out.loss.mean().item())
    stream = RandomStream(71)
    initial = dict(enumerate(stream.normal(v) for v in method.representation.encode(x[:1])))
    with torch.no_grad():
        sample = run_method.sample(trainer.ema.eval(), initial, stream=stream, model_kwargs={'y': labels[:1]})
    assert sample.shape == (1, 4, 32, 32) and torch.isfinite(sample).all()
    sources = [source_config, ROOT / 'sources/dfm/loss.py', ROOT / 'sources/dfm/dfm_utils/laplacian_decomposer.py',
               ROOT / 'umdiff/batches.py', ROOT / 'umdiff/time_sampling.py', ROOT / 'umdiff/comparison_models.py',
               Path(__file__)]
    report = dict(status='passed', device='cpu', cuda_initialized=torch.cuda.is_initialized(),
                  seconds=time.monotonic()-started, continuous_updates=64,
                  actual_grids=[16, 32], synthetic_model_domain_input=list(x.shape),
                  model_override={'hidden_size': 32, 'depth': 2, 'num_heads': 4},
                  stages_observed=stages, loss_max_abs=maximum, losses=losses,
                  activation_counts=activation['totals'], continuous_torch_numpy_rng_equal=True,
                  updated_weights_and_adam_state_equal_each_step=True,
                  author_loss_unmodified=True, production_components_modified=False,
                  main_config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  trainer={'updates': 2, 'accumulation_steps': 2, 'losses': trainer_losses},
                  sampler={'configured_phase_steps': [30, 10], 'shape': list(sample.shape),
                           'sha256': hashlib.sha256(sample.numpy().tobytes()).hexdigest()},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  exclusions=['Original SiT architecture equality', 'VAE/data loader and full ImageNet run',
                              'Full XL width/depth', 'CUDA/BF16/DDP', 'Final image quality'],
                  outer_training_aggregation_disabled_for_component_check=outer_policy,
                  scope='Author single-phase objective and RNG policy with the same common DiT on both sides; CPU FP32, '
                        'nonzero random weights, synthetic SD-shaped latents. No per-step reseeding. '
                        'Module boundaries are traced; internal functional operations are not individually traced.')
    assert not report['cuda_initialized']
    (OUT / 'DFM_PARITY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'seconds', 'stages_observed', 'activation_counts')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    OUT = parser.parse_args().output
    if (OUT / 'DFM_PARITY.json').exists():
        raise ValueError('Use a new output directory to retain previous reports')
    OUT.mkdir(parents=True, exist_ok=True)
    main()
