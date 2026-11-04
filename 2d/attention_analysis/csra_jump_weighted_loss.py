"""Opt-in local-jump supervision. No inference or historical loss is changed.

P is 2x2 average followed by nearest-neighbor expansion, an orthogonal
projection. For the EXISTING 80/80/90 sampler, a main-head error e changes the
next returned sample by g_low P(e) + g_high (I-P)(e). Its squared norm gives
the weights here. This is NOT a final-image/FID sensitivity or a new sampler.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from csra_output_distillation import prediction_loss


TRAIN_TIMES = (299, 259, 215, 171, 128, 84, 40, 0)
SOURCE_FILES = (
    'diffusion/gaussian_diffusion_with_drift.py',
    '2d/sample_fid_unified.py',
    '2d/attention_analysis/sample_rdd_stagewide_block_reuse_preview.py',
    '2d/attention_analysis/audit_csra_jump_head_sensitivity.py',
    '2d/attention_analysis/csra_jump_weighted_loss.py',
)


def make_calibration(audit):
    rows = audit['rows']
    by_time = {r['t']: r for r in rows}
    if len(rows) != 90 or len(by_time) != 90 or not set(TRAIN_TIMES) <= set(by_time):
        raise ValueError('Expected actual 90 fine sampling times, including all eight train times')
    if any(not math.isfinite(r[k]) for r in rows for k in ('low_gain', 'high_gain')):
        raise ValueError('Nonfinite gain')
    if by_time[0]['low_gain'] != 0 or by_time[0]['high_gain'] != 0:
        raise ValueError('Historical fine local0 must remain an identity sample update')
    if any(r['returned_quantity'] != 'sample' or r['mixed_field_max_abs_error'] > 3e-6 for r in rows):
        raise ValueError('Synthetic actual-jump audit failed')
    # A 2x2 cell has one low and three high degrees of freedom. Unit white
    # main-head error therefore has expected low/high MSE 1/4 and 3/4.
    # ONE common scalar over the fixed uniform train times, not one per t/arm.
    normalizer = sum(.25*by_time[t]['low_gain']**2 + .75*by_time[t]['high_gain']**2
                     for t in TRAIN_TIMES) / len(TRAIN_TIMES)
    if not normalizer > 0 or not math.isfinite(normalizer):
        raise ValueError('Invalid common normalizer')
    weights = [dict(t=t, low=by_time[t]['low_gain']**2/normalizer,
                    high=by_time[t]['high_gain']**2/normalizer) for t in TRAIN_TIMES]
    return dict(version='csra_prefix6_jump_loss_v1', complete=True,
                steps=[80, 80, 90], train_times=list(TRAIN_TIMES), weights=weights,
                normalizer=normalizer, variance_weight=.05, gains=rows,
                normalization='mean_train_times(0.25*g_low^2 + 0.75*g_high^2); unit-isotropic-error expectation',
                recipe='mean_image(w_low*MSE(Pe) + w_high*MSE(e-Pe)) + 0.05*variance_MSE',
                warning='Only a local next-sample error proxy. Not final FID, no empirical validation/FID fitting. '
                        't0 main weight is zero for this historical fine sampler; coarse boundary pred_x0 is untouched.')


def weight_table(calibration, device):
    table = torch.full((1000, 2), float('nan'), device=device, dtype=torch.float32)
    for row in calibration['weights']:
        table[row['t']] = torch.tensor([row['low'], row['high']], device=device)
    return table


def weighted_prediction_loss(prediction, target, weights, variance_weight=.05):
    """weights[B,2] are detached; tensors/targets retain old prediction format.

    Use wh*||e||^2 + (wl-wh)*||Pe||^2, equivalent by orthogonality to
    wl*||Pe||^2 + wh*||(I-P)e||^2. Equal unit weights reduce to old MSE without
    an extra subtraction/projection roundoff term in the gradient.
    """
    _, metrics = prediction_loss(prediction, target, variance_weight)
    if prediction.shape[-1] % 2 or prediction.shape[-2] % 2:
        raise ValueError('Even image dimensions required for orthogonal 2x2 cells')
    if weights.shape != (prediction.shape[0], 2):
        raise ValueError('Expected [batch,2] low/high weights')
    weights = weights.detach().to(device=prediction.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Nonfinite, unknown-timestep or negative weights')
    error = prediction[:, :4].float() - target[:, :4].float()
    main_per_image = error.square().flatten(1).mean(1)
    # Mean of pooled squares equals mean of squared expanded constant field.
    low_per_image = F.avg_pool2d(error, 2).square().flatten(1).mean(1)
    low, high = weights.unbind(1)
    weighted_main = (high*main_per_image + (low-high)*low_per_image).mean()
    variance = (prediction[:, 4:].float()-target[:, 4:].float()).square().mean()
    return weighted_main + variance_weight*variance, {
        **metrics, 'weighted_main_mse': weighted_main.detach(),
        'low_mse': low_per_image.detach().mean(),
        'high_mse': (main_per_image-low_per_image).detach().mean(),
    }


def load_verified_calibration(path):
    value = json.loads(Path(path).read_text())
    expected = make_calibration({'rows': value['gains']})
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Calibration changed or incomplete')
    for name in SOURCE_FILES:
        if value['source_sha256'][name] != hashlib.sha256(Path(name).read_bytes()).hexdigest():
            raise ValueError(f'Calibration no longer matches code: {name}')
    return value


def main(output):
    from audit_csra_jump_head_sensitivity import audit
    if output.exists():
        value = load_verified_calibration(output)
    else:
        value = make_calibration(audit())
        value['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ('normalizer', 'weights', 'normalization', 'warning')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
