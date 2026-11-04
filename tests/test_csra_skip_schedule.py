import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2d/attention_analysis'))
from csra_skip_schedule import schedules, validate_indices, ScheduledPrefixRollout


class Plans(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual({k: len(v) for k, v in schedules().items()}, dict(
            original_dense=0, all90=90, early75=75, early60=60, late60=60,
            periodic60=60, sandwich60=60, periodic45=45, early30=30))
        for v in schedules().values():
            self.assertEqual(len(v), len(validate_indices(v)))

    def test_invalid(self):
        for indices in [(0, 0), (-1,), (90,), (True,), (1.5,)]:
            with self.assertRaises(ValueError): validate_indices(indices)

    def test_actual_calls_and_cache(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is needed for mock forward test')
        def manual(teacher, x, t, y, **kw):
            return x + 1, {13: {'output': x + 2}} if kw.get('capture_layers') else {}
        def student(batch):
            self.assertIn(13, batch['coarse'])
            return batch['x'] + 3
        wrapper = ScheduledPrefixRollout(None, student, schedules()['late60'], manual)
        y = torch.tensor([1])
        wrapper.forward(torch.zeros(1, 4, 16, 16), torch.tensor([300]), y)
        for i in range(90):
            out = wrapper.forward(torch.zeros(1, 4, 32, 32), torch.tensor([299-i]), y)
            self.assertEqual(float(out[0, 0, 0, 0]), 1 if i < 30 else 3)
        self.assertEqual((wrapper.fine_calls, wrapper.reused_fine_calls), (90, 60))

    def test_missing_source(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is needed for mock forward test')
        wrapper = ScheduledPrefixRollout(None, lambda x: x, (0,), lambda *a, **kw: (None, {}))
        with self.assertRaises(RuntimeError):
            wrapper.forward(torch.zeros(1, 4, 32, 32), torch.tensor([299]), torch.tensor([1]))


if __name__ == '__main__': unittest.main()
