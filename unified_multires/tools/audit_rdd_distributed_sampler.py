#!/usr/bin/env python3
"""Two actual CPU Gloo ranks: native RDD collective history versus unified clock."""
import argparse, hashlib, importlib.util, json, os, sys, tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from umdiff.time_sampling import LossSecondMoment
from run_scoped_validation import snapshot
SOURCE = ROOT.parent/'diffusion/timestep_sampler.py'


def equal(a, b):
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    else:
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def worker(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(directory/'rendezvous').as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        spec = importlib.util.spec_from_file_location('rdd_sampler_source', SOURCE)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        native = author.LossSecondMomentResampler(SimpleNamespace(num_timesteps=8), history_per_term=2)
        port = LossSecondMoment(8, history_per_term=2)
        cases = []
        np.random.seed(91+rank)
        for step in range(12):
            times = (torch.arange(4)+4*rank+step) % 8
            losses = (times.float()+1.)*(step+1.)/(rank+3.)
            native.update_with_local_losses(times, losses)
            port.update(times, losses)
            equal(native._loss_history, port.history)
            equal(native._loss_counts, port.counts)
            equal(native.weights(), port.weights())
            before = np.random.get_state()
            ta, wa = native.sample(5, 'cpu')
            after = np.random.get_state()
            np.random.set_state(before)
            tb, context = port(5, 'cpu', None)
            equal(ta, tb)
            equal(wa, context['importance_weights'])
            got = np.random.get_state()
            assert after[0] == got[0] and after[2:] == got[2:]
            equal(after[1], got[1])
            cases.append(dict(step=step, status='passed', warmed_up=bool((port.counts==2).all())))
        assert not np.all(port.weights() == port.weights()[0])
        digest = hashlib.sha256(port.history.tobytes()+port.counts.tobytes()).hexdigest()
        states = [None, None]
        dist.all_gather_object(states, digest)
        assert len(set(states)) == 1
        assert not torch.cuda.is_initialized()
        (directory/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,status='passed',cases=cases,history_sha256=digest,same_global_history_on_all_ranks=True)))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = snapshot()
    with tempfile.TemporaryDirectory(prefix='rdd_clock_gloo_') as tmp:
        mp.spawn(worker, args=(tmp,), nprocs=2, join=True)
        ranks = [json.loads((Path(tmp)/f'rank{i}.json').read_text()) for i in range(2)]
    assert snapshot() == before
    result = dict(status='passed', ranks=ranks, world_size=2, updates=12, local_batch=4,
                  source_snapshot=before, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), gpu_used=False,
                  scope='Actual source update_with_local_losses and unified all_gather_object: history, counts, probabilities, sampled times, importance weights and NumPy RNG byte-exact. Balanced local batches only. This is the clock collective, not a full multi-rank RDD network/GPU trajectory.',
                  temporary_payloads_deleted=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='passed', ranks=2, updates=12)))


if __name__ == '__main__':
    main()
