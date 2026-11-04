import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

path=Path(__file__).resolve().parents[1]/'2d/attention_analysis/sample_csra_prefix_fid_pilot.py'
spec=importlib.util.spec_from_file_location('csra_fid_pilot',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PlanTests(unittest.TestCase):
    def test_exact_count_and_bounded_batches(self):
        self.assertEqual(sum(n for r in range(4) for _,n,_ in m.batches(r)),10000)
        for r in range(4):
            self.assertEqual(m.batches(r)[-1][1],4)
            self.assertTrue(all(0<n<=64 for _,n,_ in m.batches(r)))

    def test_unique_seeds(self):
        seeds=[s for r in range(4) for _,_,s in m.batches(r)]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertFalse(set(seeds)&{20460913+r*10000 for r in range(4)})

    def test_smoke_and_protocol(self):
        self.assertEqual(sum(n for r in range(4) for _,n,_ in m.batches(r,True)),16)
        self.assertEqual(m.protocol(False)['steps'],[80,80,90])
        self.assertEqual(m.protocol(False)['samples_per_arm'],10000)
        self.assertNotEqual(m.protocol(True),m.protocol(False))

    def test_invalid_gpu_rank(self):
        for r in (-1,4,7):
            with self.assertRaises(ValueError): m.batches(r)


@unittest.skipIf(np is None, 'NumPy required for assembly tests')
class AssemblyTests(unittest.TestCase):
    def setup_chunks(self, root):
        config=dict(complete=True,protocol=m.protocol(True),base_sha256='test',checkpoints={})
        for rank in range(4):
            folder=root/f'shard_{rank}';folder.mkdir()
            m.write_json(folder/'complete.json',config)
            for arm in m.ARMS:
                dest=folder/arm;dest.mkdir()
                for index,count,seed in m.batches(rank,True):
                    chunk=dest/f'chunk_{index:03d}.npz'
                    labels=[rank]*count
                    np.savez_compressed(chunk,arr_0=np.full((count,256,256,3),rank,np.uint8),arr_1=labels)
                    m.write_json(chunk.with_suffix('.json'),dict(seed=seed,count=count,labels=labels,
                                 endpoints={'8':'fixed','16':'fixed'},sha256=m.sha(chunk)))

    def test_assembly_order(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.setup_chunks(root)
            m.assemble(SimpleNamespace(root=root,smoke=True))
            for arm in m.ARMS:
                with np.load(root/f'{arm}.npz') as data:
                    self.assertEqual(data['arr_0'].shape,(16,256,256,3))
                    self.assertEqual(data['arr_1'].tolist(),[r for r in range(4) for _ in range(4)])
            self.assertTrue(json.loads((root/'sampling_complete.json').read_text())['complete'])

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.setup_chunks(root)
            p=root/'shard_0/original_dense/chunk_000.json'
            data=json.loads(p.read_text());data['sha256']='wrong';m.write_json(p,data)
            with self.assertRaises(ValueError): m.assemble(SimpleNamespace(root=root,smoke=True))


if __name__=='__main__': unittest.main()
