import importlib.util
from pathlib import Path
import unittest
import tempfile
import json


spec = importlib.util.spec_from_file_location('quality_summary',
    Path(__file__).resolve().parents[1]/'2d/attention_analysis/summarize_csra_prefix_quality_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestPairedSummary(unittest.TestCase):
    def test_identical_pairs(self):
        m = module.paired_summary([0, 1, 0, 1], [0, 1, 0, 1], repetitions=100)
        self.assertEqual(m['paired_bootstrap_95'], [0., 0.])
        self.assertEqual(m['equal'], 4)

    def test_signed_difference(self):
        m = module.paired_summary([1, 1, 1], [0, 0, 0], repetitions=100)
        self.assertEqual(m['mean_difference'], 1.)
        self.assertEqual(m['paired_bootstrap_95'], [1., 1.])
        self.assertEqual(m['left_greater'], 3)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            module.paired_summary([1], [])

    def test_explicit_depth_budget_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in range(4):
                rows=[dict(variant=v,image=rank*64+i,local_image=i,target=i)
                      for v in module.VARIANTS for i in range(64)]
                m=dict(complete=True,rank=rank,seed=20460913+rank*10000,images_per_variant=64,rows=rows,
                       coarse_endpoint_max_abs={'aligned':0.,'zero':0.},steps=[80,80,90],cfg=1.,
                       upsample='noise_filling',configs={a:{'step':800} for a in ('aligned','zero')},
                       prefix_depth=6,labels=list(range(64)),peak_memory_gib=4.,
                       relative_latent_l2={v:0. for v in module.VARIANTS})
                folder=root/f'shard_{rank}';folder.mkdir();(folder/'metrics.json').write_text(json.dumps(m))
            grouped,_=module.load_verified(root,prefix_depth=6,checkpoint_step=800,seed_base=20460913)
            self.assertEqual(len(grouped['aligned']),256)
            for wrong in ({'prefix_depth':8},{'checkpoint_step':3200},{'seed_base':20360913}):
                kwargs=dict(prefix_depth=6,checkpoint_step=800,seed_base=20460913);kwargs.update(wrong)
                with self.assertRaises(ValueError):module.load_verified(root,**kwargs)


if __name__ == '__main__':
    unittest.main()
