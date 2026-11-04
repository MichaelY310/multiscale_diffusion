import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"2d"/"attention_analysis"))
from probe_csra_coarse_source import intervene


class SourceInterventionTests(unittest.TestCase):
    def test_modes_do_not_mutate_source_and_preserve_permutation_values(self):
        x = torch.arange(2*4*3).reshape(2,4,3).float()
        bank = {13: x.clone(), 15: x.clone()+100}
        before = {k: v.clone() for k,v in bank.items()}
        self.assertIs(intervene(bank,"aligned"), bank)
        for mode in ("shift_parent", "other_image"):
            changed = intervene(bank, mode)
            self.assertFalse(torch.equal(changed[13],bank[13]))
            for k in bank:
                torch.testing.assert_close(changed[k].flatten().sort().values,
                                           bank[k].flatten().sort().values)
        self.assertEqual(intervene(bank,"zero")[13].count_nonzero(),0)
        for k in bank:
            torch.testing.assert_close(before[k],bank[k],atol=0,rtol=0)

    def test_single_image_and_unknown_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            intervene({13:torch.ones(1,4,3)},"other_image")
        with self.assertRaises(ValueError):
            intervene({},"unknown")


if __name__ == "__main__":
    unittest.main()
