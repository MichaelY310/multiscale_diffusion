import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('staged_protocol', Path(__file__).parents[1]/'2d/attention_analysis/csra_staged_protocol.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTests(unittest.TestCase):
    def test_exact_budgets(self):
        self.assertEqual(sum(module.phase('staged', s) == 'bridge_only' for s in range(1600)), 800)
        self.assertTrue(all(module.phase('direct_control', s) == 'joint' for s in range(1600)))
        self.assertEqual(module.phase('staged', 799), 'bridge_only')
        self.assertEqual(module.phase('staged', 800), 'joint')

    def test_invalid(self):
        for mode, step in [('bad', 0), ('staged', -1), ('staged', 1600)]:
            with self.assertRaises(ValueError): module.phase(mode, step)

    def test_parent_budget(self):
        ck = dict(format='csra_front1_skip6_conditional_v1',step=800,front_keep=1,prefix_depth=6,
                  source_layer=13,base_sha256='b',initializer_sha256='i',
                  args=dict(coarse_source='aligned',seed=20279000,data_root='eval_results/csra_rollout_data4k_20260912',
                            hidden_root='eval_results/csra_hidden_sources_20260913'))
        module.check_parent(ck,'aligned','b','i')
        for key, value in [('step',1600),('training_protocol','another'),('front_keep',0)]:
            with self.assertRaises(ValueError): module.check_parent({**ck,key:value},'aligned','b','i')
