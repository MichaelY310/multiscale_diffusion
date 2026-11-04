"""Bounded architecture pilot, using the EXACT existing teacher-only trainer.

The factory/protocol callbacks below are rebound only inside this standalone
process. No old source/checkpoint is rewritten. Dataset, order, optimizer reset,
loss, batch and fixed400 endpoint are identical to the completed teacher_only
control. Extra coarse-memory computation is the sole inference-graph change.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train_csra_state_replay as shared
from csra_state_replay_data import ROOT as DATA_ROOT, PROTOCOL as DATA_PROTOCOL, PairedReplayDataset, SEED
from csra_refined_student import RefinedFrontStudent, FORMAT
from probe_csra_front_keep import file_sha
from train_csra_rollout_output import to_device

ROOT = Path('eval_results/csra_memory_refiner_20260914')
PROTOCOL = 'csra_current_coarse_memory_fixed400_v1'
CHECKPOINT = Path('checkpoints/RDD-CSRA-memory-refiner-L400K-20260914/latest.pt')
old_factory, old_source_hashes = shared.make_model, shared.source_hashes


def source_hashes():
    paths = ['2d/attention_analysis/train_csra_memory_refiner.py',
             '2d/attention_analysis/csra_refined_student.py',
             'rdd_video/attention_analysis/csra_memory_refiner.py']
    return dict(old_source_hashes(), **{p: file_sha(p) for p in paths})


def dataset_hashes():
    # Dataset format remains the OLD replay format. Never compare it against
    # the NEW candidate checkpoint/training protocol or relabel the data.
    manifests = sorted((DATA_ROOT/'data').glob('shard_*/manifest.json'))
    if len(manifests) != 4: raise ValueError('Need four complete data shards')
    hashes = {}
    for path in manifests:
        d = json.loads(path.read_text())
        if d.get('protocol') != DATA_PROTOCOL or not d.get('complete') or d.get('smoke'):
            raise ValueError('Wrong source dataset protocol')
        for item in d['files']:
            if Path(item['name']).name != item['name'] or file_sha(path.parent/item['name']) != item['sha256']:
                raise ValueError('Dataset content changed')
        hashes[str(path)] = file_sha(path)
    return hashes


def make_model(device, trainable=False):
    parent, meta = old_factory(device, trainable=False)
    init = torch.load(meta['args']['initializer'], map_location='cpu', weights_only=True)
    # Same deterministic new-branch initialization on all ranks, no changes to
    # the shared trainer's later RNG seed or paired data order.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED + 701)
        model = RefinedFrontStudent(parent.backbone, init['fits']['fine_only'], 'aligned',
            prefix_depth=6, source_layer=13, checkpoint_blocks=False,
            neighbor_enabled=False, neighbor_rank=128, refine_depth=2).to(device)
    model.initialize_from_neighbor_parent(parent.state_dict())
    model.eval().requires_grad_(False)
    if trainable:
        # Actual64 is also checked, not just toy CPU tensors. It tests the new
        # zero-output branch while executing all of it (not a disabled flag).
        val = PairedReplayDataset(DATA_ROOT/'data', 'validation', fixed_role='teacher')
        batch = to_device(next(iter(DataLoader(val, batch_size=64, shuffle=False))), device)
        with torch.no_grad():
            before, after = parent(batch), model(batch)
            torch.testing.assert_close(after, before, atol=0, rtol=0)
        meta['zero_init_actual_batch64_max_abs'] = 0.
        model.requires_grad_(True)
        model.neighbor.requires_grad_(False)
        for block in model.backbone.blocks[1:7]: block.requires_grad_(False)
        model.checkpoint_blocks = True
    meta.update(candidate_format=FORMAT, refinement_depth=2,
        correction='native coarse SDPA/FFN, current pooled fine and time, zero-output init; all branch compute retained')
    return model, meta


def install_factories():
    shared.PROTOCOL = PROTOCOL
    shared.make_model = make_model
    shared.source_hashes = source_hashes
    shared.dataset_hashes = dataset_hashes


def load_result(device):
    install_factories()
    return shared.load_result(CHECKPOINT, device, 'teacher_only')


def train(args):
    gate = json.loads((ROOT/'cost_gate.json').read_text())
    if not gate.get('complete') or not gate.get('passes_large_video_cost_gate'):
        raise ValueError('Must pass predeclared large-video cost gate before training')
    # This format has only one new arm; its pre-existing control used the same
    # teacher-only data, reset optimizer and SEED. Do not enable mixed50 here.
    args.arm = 'teacher_only'
    install_factories()
    shared.train(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--smoke', action='store_true')
    train(p.parse_args())
