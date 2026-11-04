import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from audit_csra_jump_head_sensitivity import audit


class JumpHeadSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit()

    def test_learned_variance_is_not_used_by_jump_sample(self):
        self.assertIn('LEARNED_RANGE', self.result['model_var_type'])
        self.assertGreater(self.result['p_mean_variance_logvar_change'], 0)
        self.assertTrue(self.result['all_90_times_variance_parity'])
        self.assertEqual(self.result['jump_sample_variance_intervention_max_abs'], 0)

    def test_every_time_matches_spatial_linear_response(self):
        rows = self.result['rows']
        self.assertEqual(len(rows), 90)
        self.assertEqual(rows[0]['t'], 299)
        self.assertEqual(rows[-1]['t'], 0)
        self.assertEqual(rows[-1]['low_gain'], 0)
        self.assertEqual(rows[-1]['high_gain'], 0)
        self.assertLess(max(r['mixed_field_max_abs_error'] for r in rows), 3e-6)


if __name__ == '__main__':
    unittest.main()
