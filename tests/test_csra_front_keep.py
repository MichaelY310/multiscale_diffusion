import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from probe_csra_front_keep import layout, common, check_contract, hidden_error_row
from csra_prefix_student import check_prefix_initializer


class FrontKeepTests(unittest.TestCase):
    def test_same_six_skips_and_eighteen_executed(self):
        for front in (0,1):
            spec = layout(front)
            self.assertEqual(len(spec['skipped_layers']),6)
            self.assertEqual(len(spec['executed_layers']),18)
            self.assertEqual(sorted(spec['skipped_layers']+spec['executed_layers']),list(range(24)))
            self.assertEqual(spec['target_layer'],front+5)
        self.assertIn(0,layout(1)['executed_layers'])
        self.assertNotIn(0,layout(0)['executed_layers'])
        for bad in (-1,2,True):
            with self.assertRaises(ValueError): layout(bad)

    def test_contract_cannot_silently_initialize_old_student(self):
        spec = common()
        check_contract(spec,1)
        for key,value in [('skip_count',7),('source_layer',12),('fronts',[0,2]),('parity_max_abs',.1),('base_sha256','wrong')]:
            bad = copy.deepcopy(spec); bad[key] = value
            with self.assertRaises(ValueError): check_contract(bad,1)
        with self.assertRaises(ValueError): check_prefix_initializer(spec,6)

    def test_normalization_and_orthogonal_energy(self):
        gen = torch.Generator().manual_seed(818)
        inp = torch.randn(2,16,8,generator=gen)
        target = inp+torch.randn(2,16,8,generator=gen)
        pred = target+.1*torch.randn(2,16,8,generator=gen)
        row = hidden_error_row(pred,target,inp)
        torch.testing.assert_close(row['hidden_mse'],row['low_mse']+row['high_mse'])
        doubled = hidden_error_row(2*pred,2*target,2*inp)
        for key in ('target_nmse','delta_nmse'):
            torch.testing.assert_close(row[key],doubled[key])
        self.assertTrue(all(torch.isfinite(v).all() for v in hidden_error_row(inp*0,inp*0,inp*0).values()))


if __name__=='__main__': unittest.main()
