"""Oracle hidden-error decomposition, not a deployable accelerated method.

At the six-block cut, restore either each parent cell's mean error or its
within-cell error, then execute the SAME trained suffix. No training/sampling.
Both direct-control fixed1600 arms are tested on the same 128 development images.
"""
import argparse
import gc
from pathlib import Path
import torch
from probe_csra_front_keep import entries, load, file_sha, BASE, BASE_SHA
from probe_csra_prefix_information import TIMES, write_json
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_staged_front_quality import checked_loader
from audit_csra_front_reader import suffix
from csra_prefix_probe_math import split_error

ROOT = Path('eval_results/csra_bridge_bands_20260914')
TEMPLATE = 'checkpoints/RDD-CSRA-front1-direct_control-{arm}-L400K-20260914/latest.pt'


def decompose(error):
    """Orthogonal block-constant projection; not an ideal Fourier low-pass."""
    low, high = split_error(error.float())
    dims = (1, 2)
    total = error.float().square().mean(dims)
    lo, hi = low.square().mean(dims), high.square().mean(dims)
    torch.testing.assert_close(total, lo+hi, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close((low*high).mean(dims), torch.zeros_like(total), atol=1e-6, rtol=0)
    high_mean, _ = split_error(high)
    torch.testing.assert_close(high_mean, torch.zeros_like(high_mean), atol=2e-6, rtol=0)
    return low, high, total, lo, hi


@torch.inference_mode()
def run(args):
    destination = ROOT/f'shard_{args.shard}.json'
    if destination.exists():
        raise FileExistsError('Inspect existing completed audit; never overwrite')
    torch.set_num_threads(2)
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 18*2**30:
        raise RuntimeError('Need18GiB headroom; never stop another process')
    torch.cuda.set_per_process_memory_fraction(.18, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    if file_sha(BASE) != BASE_SHA:
        raise ValueError('Teacher checkpoint changed')
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    all_entries, fingerprints, _ = entries(Path('eval_results/csra_front_keep6_information_20260914'))
    selected = [e for e in all_entries if e[0].parent.name == f'shard_{args.shard}'
                and e[1]['role'] == 'validation']
    if len(selected) != 4:
        raise ValueError('Require32 development images per shard')
    rows, configs, parity = [], {}, 0
    for arm in ('aligned', 'zero'):
        student, config = checked_loader(arm, dev, TEMPLATE, 6, 1600)
        config['checkpoint_sha256'] = file_sha(config['checkpoint'])
        configs[arm] = config
        for entry in selected:
            rec = load(entry)
            ref = torch.load(Path('eval_results/csra_rollout_data4k_20260912')/entry[0].parent.name/entry[0].name,
                             map_location='cpu', weights_only=True)
            if rec['seed'] != ref['seed'] or not torch.equal(rec['labels'], ref['labels']):
                raise ValueError('Source/target mismatch')
            source, labels = rec['coarse'].to(dev), rec['labels'].to(dev)
            for ti, tv in enumerate(TIMES):
                x, t, target = (ref[k][ti].to(dev) for k in ('x', 't', 'target'))
                if not torch.all(t == tv):
                    raise ValueError('Wrong time')
                rng = torch.cuda.get_rng_state(dev)
                actual, hidden = student(dict(x=x, t=t, labels=labels, coarse={13:source}), return_hidden=True)
                model = student.backbone
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    c = model.t_embedder(t)+model.y_embedder(labels, False)
                # Re-executing the suffix must exactly equal the student forward.
                repeated = suffix(model, hidden, c, 7)
                torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
                true_hidden = rec['target1'][ti].to(dev)
                exact = suffix(teacher, true_hidden, rec['condition'][ti].to(dev), 7)
                torch.testing.assert_close(exact, target, atol=0, rtol=0)
                error = true_hidden.float()-hidden.float()
                low, high, total, lo, hi = decompose(error)
                outputs = {'actual':actual}
                for name, correction in (('oracle_cell_mean', low), ('oracle_within_cell', high)):
                    outputs[name] = suffix(model, (hidden.float()+correction).to(hidden.dtype), c, 7)
                outputs['oracle_full'] = suffix(model, true_hidden, c, 7)
                if not torch.equal(rng, torch.cuda.get_rng_state(dev)):
                    raise RuntimeError('Read-only audit changed RNG')
                measured = {name:(pred.float()-target.float()).square().mean((2, 3))
                            for name, pred in outputs.items()}
                power = true_hidden.float().square().mean((1, 2)).clamp_min(1e-12)
                for i in range(len(x)):
                    rows.append(dict(arm=arm, seed=rec['seed'], image=i, t=tv, label=int(labels[i]),
                        hidden_mse=float(total[i]), cell_mean_mse=float(lo[i]), within_cell_mse=float(hi[i]),
                        hidden_nmse=float(total[i]/power[i]),
                        cell_mean_energy_per_dof=float(lo[i]/.25),
                        within_cell_energy_per_dof=float(hi[i]/.75),
                        errors={name:dict(main_mse=float(e[i,:4].mean()), variance_mse=float(e[i,4:].mean()))
                                for name, e in measured.items()}))
                parity += 1
            print(f'{arm} shard{args.shard}: {len(rows)} states checked', flush=True)
        del student, model, actual, repeated, hidden, outputs, measured, exact
        gc.collect()
        torch.cuda.empty_cache()
    if len(rows) != 512 or parity != 64:
        raise ValueError('Incomplete fixed128 audit')
    write_json(destination, dict(complete=True, version='csra_bridge_cell_bands_v1', shard=args.shard,
        configurations=configs, fingerprints=fingerprints, rows=rows, times=TIMES,
        forward_parity_max_abs=0., teacher_suffix_parity_max_abs=0., rng_unchanged=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Oracle diagnostic only: true hidden computes skipped teacher blocks. NOT deployable speed/quality. '
                'P is the orthogonal mean over each2x2 child-token cell, not an ideal Fourier low-pass. '
                'Its complement has75% of degrees of freedom versus25% for cell mean. '
                'Suffix nonlinearity means the two output improvements need not add. '
                'Both source/zero are separately trained direct-control1600 models. '
                'Uses repeated128 development images at8 teacher-state anchors, not unseen holdout or free rollouts.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, choices=range(4), required=True)
    run(p.parse_args())
