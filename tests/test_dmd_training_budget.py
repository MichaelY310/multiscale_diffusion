from itertools import islice
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from dmd_training_budget import budget_metadata, budget_reached, training_updates


class BudgetTests(unittest.TestCase):
    def test_finite_original_behavior(self):
        self.assertEqual(list(training_updates(1998,2000)),[1999,2000])
        self.assertEqual(list(training_updates(2000,2000)),[])
        self.assertFalse(budget_reached(1999,2000))
        self.assertTrue(budget_reached(2000,2000))
        self.assertTrue(budget_reached(2001,2000))
        self.assertEqual(budget_metadata(2000), dict(additional_updates=2000,
                         total_compression_updates_target=7000, unlimited_training=False))

    def test_unlimited_resume_is_not_finished_or_reset(self):
        self.assertEqual(list(islice(training_updates(2000,0),5)),[2001,2002,2003,2004,2005])
        self.assertEqual(list(islice(training_updates(10**15,0),2)),[10**15+1,10**15+2])
        for step in (0,2000,30000,10**15):
            self.assertFalse(budget_reached(step,0))
        self.assertEqual(budget_metadata(0), dict(additional_updates=None,
                         total_compression_updates_target=None, unlimited_training=True))

    def test_invalid_budget(self):
        for steps in (-1,1.5):
            with self.assertRaises(ValueError): budget_metadata(steps)
            with self.assertRaises(ValueError): training_updates(0,steps)
        with self.assertRaises(ValueError): training_updates(-1,0)


if __name__=='__main__': unittest.main()
