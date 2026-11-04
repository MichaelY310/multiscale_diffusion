"""Read-only, small-batch hidden audit. Never updates weights or training jobs.

Compare original dense DMD and reused students on the SAME student stage input.
The projection floor concerns matching a teacher update, not a quality bound.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'DiT'))
from models_multires import DiTMultiRes_models
from dmd_cascade_reuse import CascadeDMD, coupled_noises, lift


class TracedCascade(CascadeDMD):
    def __init__(self, backbone, family):
        super().__init__(backbone, family, False)
        self.trace = {}

    def stage(self, x, t, y, stage, cache=None):
        m = self.backbone
        grid = x.shape[-1] // m.patch_size
        h = m.x_embedder(x)
        cos, sin = m.get_rope(grid, grid, x.device)
        condition = m.t_embedder(t) + m.y_embedder(y, False)
        capture = set(self.masks.get(stage + 1, ()))
        next_cache, records = {}, []
        for i, block in enumerate(m.blocks):
            before = h
            if i in self.masks[stage]:
                h = h + lift(cache[i], grid).to(h.dtype)
            else:
                h = block(h, condition, cos, sin)
            if i in capture:
                next_cache[i] = h - before
            records.append({'before': before.detach().cpu(), 'after': h.detach().cpu()})
        self.trace[stage] = records
        return m.unpatchify(m.final_layer(h, condition), grid, grid), next_cache


def project(z, scale=2):
    b, n, c = z.shape
    g = int(n ** .5)
    image = z.float().transpose(1, 2).reshape(b, c, g, g)
    coarse = F.avg_pool2d(image, scale, scale)
    return coarse.repeat_interleave(scale, -2).repeat_interleave(scale, -1).flatten(2).transpose(1, 2)


def energy(z):
    return float(z.float().square().mean())


def nmse(z, target):
    return energy(z.float() - target.float()) / max(energy(target), 1e-12)


def load_model(path, family, device):
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    m = DiTMultiRes_models['DiT-L/2'](num_classes=1000, base_size=16)
    m.load_state_dict(ck['model'], strict=True)
    model = TracedCascade(m.to(device).eval().requires_grad_(False), family)
    del ck
    gc.collect()
    return model


@torch.no_grad()
def main(a):
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f'Preserve existing diagnostic: {output}')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    teacher = load_model(a.teacher, 'dense', device)
    result = {'samples_per_model': a.samples, 'seed': a.seed,
              'teacher': a.teacher, 'read_only': True, 'NOT_FID': True,
              'warning': 'Hidden coordinates may change during full finetuning; '
                         'teacher-update projection floors are NOT image-quality bounds.',
              'models': {}}
    for name, checkpoint in (('control', a.control), ('dmd_lpips', a.dmd)):
        student = load_model(checkpoint, 'high_heavy', device)
        generator = torch.Generator(device=device).manual_seed(a.seed)
        rows = []
        for index in range(a.samples):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                noises, y = coupled_noises(1, device, generator)
                pred = student(noises, y)
                ref = teacher(noises, y)
                if index == 0:
                    # Same backbone and exact forward path, without tracing.
                    plain = CascadeDMD(student.backbone, 'high_heavy', False)(noises, y)
                    for s in range(3):
                        torch.testing.assert_close(pred['cleans'][s], plain['cleans'][s], rtol=0, atol=0)
                    plain_t = CascadeDMD(teacher.backbone, 'dense', False)(noises, y)
                    for s in range(3):
                        torch.testing.assert_close(ref['cleans'][s], plain_t['cleans'][s], rtol=0, atol=0)
                    del plain, plain_t
                row = {'index': index, 'label': int(y.item()),
                       'paired_final_mse': energy(pred['cleans'][-1] - ref['cleans'][-1]),
                       'stages': {}}
                for s in (1, 2):
                    original_clean = ref['cleans'][s]
                    # Teacher and student now receive identical x, t, y at this stage.
                    local_clean = teacher.clean_at(pred['inputs'][s], y, s)[0]
                    st, tt = student.trace[s], teacher.trace[s]
                    info = {'paired_clean_mse': energy(pred['cleans'][s] - original_clean),
                            'same_input_clean_mse': energy(pred['cleans'][s] - local_clean),
                            'hidden_nmse_by_layer': [nmse(p['after'], q['after']) for p, q in zip(st, tt)]}
                    if s == 2:
                        # Replay teacher blocks 3..22 on the EXACT student's h_after2.
                        # This isolates that segment's transformation, not prefix error.
                        h2 = st[1]['after'].to(device)
                        h = h2
                        m = teacher.backbone
                        times = torch.full_like(y, 299)
                        condition = m.t_embedder(times) + m.y_embedder(y, False)
                        cos, sin = m.get_rope(16, 16, device)
                        for block in m.blocks[2:22]:
                            h = block(h, condition, cos, sin)
                        td = (h.float() - h2.float()).cpu()
                        sd = st[21]['after'].float() - st[1]['after'].float()
                        low = project(td)
                        info['segment_3_to_22'] = {
                            'teacher_highpass_energy_fraction': energy(td-low)/max(energy(td), 1e-12),
                            'student_highpass_energy_fraction': energy(sd-project(sd))/max(energy(sd), 1e-12),
                            'update_nmse_to_teacher_same_h2': nmse(sd, td),
                            'lowpass_update_nmse': nmse(project(sd), low),
                            'oracle_projection_nmse': nmse(low, td),
                        }
                    row['stages'][str((8, 16, 32)[s])] = info
                rows.append(row)
                print(json.dumps({'model': name, 'sample': index + 1,
                                  'paired_final_mse': row['paired_final_mse'],
                                  'fine_segment': row['stages']['32']['segment_3_to_22']}), flush=True)
                del pred, ref
        result['models'][name] = {'checkpoint': checkpoint, 'parity_exact': True, 'rows': rows}
        del student
        gc.collect()
        torch.cuda.empty_cache()
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    result['complete'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--dmd', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=6192026)
    p.add_argument('--memory-fraction', type=float, default=.08)
    main(p.parse_args())
