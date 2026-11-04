import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from csra_neighbor_protocol import (PROTOCOL,FORMAT,PARENT_SHA,OPTIMIZER_POLICY,FIXED_ARGS,
                                   check_parent,check_new)


class NeighborProtocolTests(unittest.TestCase):
    def test_exact_parent_identity_and_schedule(self):
        args={**FIXED_ARGS,'coarse_source':'aligned','mode':'direct_control'}
        ck=dict(format='csra_front1_skip6_conditional_v1',training_protocol='front1_bridge_warmup800_joint800_v1',
                step=1600,front_keep=1,prefix_depth=6,source_layer=13,base_sha256='base',initializer_sha256='init',args=args)
        check_parent(ck,'aligned',PARENT_SHA['aligned'],'base','init')
        for key,value in [('step',800),('prefix_depth',8),('format',FORMAT)]:
            with self.assertRaises(ValueError): check_parent({**ck,key:value},'aligned',PARENT_SHA['aligned'],'base','init')
        with self.assertRaises(ValueError):check_parent(ck,'aligned','other','base','init')

    def test_new_format_fixed_budget_and_code(self):
        args={**FIXED_ARGS,'coarse_source':'aligned','neighbor':'on','smoke':False}
        ck=dict(format=FORMAT,training_protocol=PROTOCOL,step=2400,front_keep=1,prefix_depth=6,
            source_layer=13,neighbor_rank=128,parent={'sha256':PARENT_SHA['aligned']},args=args,
            base_sha256='base',initializer_sha256='init',code_sha256={'code':'sha'},optimizer_policy=OPTIMIZER_POLICY)
        check_new(ck,args,{'code':'sha'},'base','init',True)
        for key,value in [('step',1602),('neighbor_rank',64),('code_sha256',{}),('optimizer_policy','migrate')]:
            with self.assertRaises(ValueError):check_new({**ck,key:value},args,{'code':'sha'},'base','init',True)
        bad=copy.deepcopy(ck);bad['args']['neighbor']='off'
        with self.assertRaises(ValueError):check_new(bad,args,{'code':'sha'},'base','init',True)
        bad=copy.deepcopy(ck);bad['args']['seed']=0
        with self.assertRaises(ValueError):check_new(bad,bad['args'],{'code':'sha'},'base','init',True)


if __name__ == '__main__':unittest.main()
