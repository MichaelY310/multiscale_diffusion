"""Read-only fixed800 source/frontend/suffix interventions on identical states.

Counterfactual mixed networks are diagnostics, NOT trained methods or quality
estimates. No student parameter, trajectory, checkpoint or RNG is changed.
"""
import argparse
from pathlib import Path
import torch
from probe_csra_front_keep import entries, load, file_sha, BASE, BASE_SHA
from probe_csra_prefix_information import TIMES, write_json
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_front_keep_quality import load_student as load_front1
from sample_csra_prefix_quality_pilot import load_student as load_front0
from diagnose_csra_front_placement_times import TEMPLATES

ROOT = Path('eval_results/csra_front_reader_audit_20260914')


def suffix(backbone, hidden, condition, start):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cos, sin = backbone.get_rope(16, 16, hidden.device)
        for block in backbone.blocks[start:]:
            hidden = block(hidden, condition, cos, sin)
        return backbone.unpatchify(backbone.final_layer(hidden, condition), 16, 16)


def source_control(source, mode):
    if mode == 'aligned': return source
    if mode == 'zero': return torch.zeros_like(source)
    if mode == 'wrong_image': return source.roll(1, 0)
    if mode == 'spatial_roll':
        b, n, h = source.shape
        if n != 64: raise ValueError('Expected native8x8 token grid')
        return source.reshape(b, 8, 8, h).roll((1, 1), (1, 2)).reshape_as(source)
    raise ValueError(mode)


@torch.inference_mode()
def run(args):
    destination = ROOT/f'front{args.front}'/f'shard_{args.shard}.json'
    if destination.exists(): raise FileExistsError('Read completed audit, never overwrite')
    torch.set_num_threads(2)
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 18*2**30:
        raise RuntimeError('Need18GiB free; never stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.18, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    if file_sha(BASE) != BASE_SHA: raise ValueError('Teacher SHA changed')
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    loader = load_front0 if args.front == 0 else load_front1
    student, config = loader('aligned', dev, TEMPLATES[args.front], 6, 800)
    config['checkpoint_sha256'] = file_sha(config['checkpoint'])
    all_entries, fingerprints, counts = entries(Path('eval_results/csra_front_keep6_information_20260914'))
    selected = [e for e in all_entries if e[0].parent.name == f'shard_{args.shard}'
                and e[1]['role'] == 'validation']
    if len(selected) != 4: raise ValueError('Require32 fixed development images/shard')
    rows, parity_records = [], 0
    for entry in selected:
        rec = load(entry)
        ref = torch.load(Path('eval_results/csra_rollout_data4k_20260912')/entry[0].parent.name/entry[0].name,
                         map_location='cpu', weights_only=True)
        if rec['seed'] != ref['seed'] or not torch.equal(rec['labels'], ref['labels']):
            raise ValueError('Source/image mismatch')
        source, labels = rec['coarse'].to(dev), rec['labels'].to(dev)
        for ti, tv in enumerate(TIMES):
            x, t, target = (ref[k][ti].to(dev) for k in ('x', 't', 'target'))
            batch = dict(x=x, t=t, labels=labels, coarse={13:source})
            rng = torch.cuda.get_rng_state(dev)
            plain, plain_hidden = student(batch, return_hidden=True)
            model = student.backbone
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model.x_embedder(x)
                c = model.t_embedder(t)+model.y_embedder(labels, False)
                cos, sin = model.get_rope(16, 16, dev)
                for block in model.blocks[:args.front]: h = block(h, c, cos, sin)
                bridge = student.lift(h, c, source).to(h.dtype)
            actual = suffix(model, bridge, c, args.front+6)
            torch.testing.assert_close(actual, plain, atol=0, rtol=0)
            torch.testing.assert_close(bridge, plain_hidden, atol=0, rtol=0)
            h_teacher = rec[f'input{args.front}'][ti].to(dev)
            c_teacher = rec['condition'][ti].to(dev)
            hidden_teacher = rec[f'target{args.front}'][ti].to(dev)
            exact = suffix(teacher, hidden_teacher, c_teacher, args.front+6)
            torch.testing.assert_close(exact, target, atol=0, rtol=0)
            outputs = {'trained_aligned':actual}
            for mode in ('zero', 'wrong_image', 'spatial_roll'):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = student.lift(h, c, source_control(source, mode)).to(h.dtype)
                outputs[f'trained_source_{mode}'] = suffix(model, hidden, c, args.front+6)
            # Restore only the input feature or both the input and conditioning.
            # Neither mixed network has been trained: do not call an increase a
            # causal importance score, or call a decrease a deployable method.
            for name, fc in (('teacher_front_input', c), ('teacher_front_and_condition', c_teacher)):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = student.lift(h_teacher, fc, source).to(h.dtype)
                outputs[name] = suffix(model, hidden, fc, args.front+6)
            outputs['teacher_cut_trained_suffix'] = suffix(model, hidden_teacher, c, args.front+6)
            outputs['trained_cut_teacher_suffix'] = suffix(teacher, bridge, c_teacher, args.front+6)
            if not torch.equal(rng, torch.cuda.get_rng_state(dev)):
                raise RuntimeError('Read-only interventions changed RNG')
            parity_records += 1
            measured = {name:(pred.float()-target.float()).square().mean((2, 3))
                        for name, pred in outputs.items()}
            hmse = (bridge.float()-hidden_teacher.float()).square().mean((1, 2))
            hp = hidden_teacher.float().square().mean((1, 2)).clamp_min(1e-12)
            input_shift = (h.float()-h_teacher.float()).square().mean((1, 2))
            input_power = h_teacher.float().square().mean((1, 2)).clamp_min(1e-12)
            for i in range(len(x)):
                rows.append(dict(seed=rec['seed'], image=i, t=tv, label=int(labels[i]),
                    bridge_hidden_nmse=float(hmse[i]/hp[i]), frontend_shift_nmse=float(input_shift[i]/input_power[i]),
                    errors={name:dict(main_mse=float(e[i, :4].mean()), variance_mse=float(e[i, 4:].mean()))
                            for name, e in measured.items()}))
        print(f'front{args.front} shard{args.shard}: {len(rows)} states checked', flush=True)
    if len(rows) != 256 or parity_records != 32: raise ValueError('Incomplete audit')
    write_json(destination, dict(complete=True,front=args.front,shard=args.shard,config=config,
        files=fingerprints,rows=rows,split='existing128 development images',times=TIMES,
        forward_parity_max_abs=0.,teacher_suffix_parity_max_abs=0.,rng_unchanged=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Same fixed800 trained aligned student; zeros/wrong images/rolled source are inference interventions, not separately trained controls. Teacher frontend/hidden/suffix swaps mix representations and may be out of distribution. No parameter updates, sampling, FID or speed claims. All128 dev images already used in development.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front', type=int, choices=(0, 1), required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    run(parser.parse_args())
