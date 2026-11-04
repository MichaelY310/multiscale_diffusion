"""Locate loss of the front1 linear-probe advantage at fixed trained800.

Same existing all90-time diagnostic with explicit loaders/steps; both original
teacher states and each student's own rollout. No model changes or training.
"""
import argparse
from pathlib import Path
from diagnose_csra_prefix6_error_budget import run
from sample_csra_prefix_quality_pilot import load_student as load_front0
from sample_csra_front_keep_quality import load_student as load_front1

TEMPLATES={0:'checkpoints/RDD-CSRA-prefix6-conditional-{arm}-L400K-20260913/best.pt',
           1:'checkpoints/RDD-CSRA-front1-skip6-{arm}-L400K-20260914/latest.pt'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--front',type=int,choices=(0,1),required=True)
    p.add_argument('--arm',choices=('aligned','zero'),required=True)
    p.add_argument('--seed',type=int,choices=(20860914,20870914),required=True)
    args=p.parse_args()
    args.batch=16
    args.root=Path('eval_results/csra_front_placement_times_20260914')/f'front{args.front}'
    run(args,student_loader=load_front0 if args.front==0 else load_front1,
        checkpoint_template=TEMPLATES[args.front],checkpoint_step=800)
