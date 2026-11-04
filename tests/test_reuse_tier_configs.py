"""CPU checks of immutable experiment masks without loading FastVideo/CUDA."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def configs():
    tree=ast.parse((ROOT/'rdd_video/benchmark_reuse_latency_tiers.py').read_text())
    keep=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('spread','config'):
            keep.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('OLD_CONFIGS','CONFIGS') for t in node.targets):
            keep.append(node)
    scope={}
    exec(compile(ast.Module(body=keep,type_ignores=[]),'<config definitions>','exec'),scope)
    return scope['CONFIGS']


class ConfigTests(unittest.TestCase):
    def test_all_masks_valid(self):
        for name,c in configs().items():
            self.assertTrue(0<c['scale']<=1,name)
            for stage,layers in c['masks'].items():
                self.assertIn(stage,(1,2));self.assertEqual(len(layers),len(set(layers)),name)
                self.assertTrue(all(0<=l<30 for l in layers),name)

    def test_label_means_number_reused_not_executed(self):
        c=configs()
        for n in (18,24,28):
            self.assertEqual(len(c[f'fine{n}']['masks'][2]),n)
            self.assertNotIn(0,c[f'fine{n}']['masks'][2])
            self.assertNotIn(29,c[f'fine{n}']['masks'][2])
        self.assertEqual(len(c['cascade12_24']['masks'][1]),12)
        self.assertEqual(len(c['cascade12_24']['masks'][2]),24)

    def test_scale_is_not_a_different_skip_mask(self):
        c=configs()
        self.assertEqual(c['fine24']['masks'],c['fine24_s075']['masks'])
        self.assertEqual(c['fine24']['masks'],c['fine24_s050']['masks'])

    def test_prefix_control_actually_skips_first_layer(self):
        c=configs()
        self.assertEqual(c['prefix24']['masks'][2],list(range(24)))
        self.assertEqual(len(c['fine24']['masks'][2]),len(c['prefix24']['masks'][2]))


if __name__=='__main__':unittest.main()
