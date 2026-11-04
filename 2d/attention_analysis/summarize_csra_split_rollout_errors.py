"""Aggregate diagnostic curves without conflating model agreement with quality."""
import argparse
import json
from pathlib import Path
import statistics

def mean(rows,key):return statistics.mean(v for row in rows for v in row[key])

def main(root):
    root=Path(root);arms={}
    for arm in ('split800','split1600','front800','front1600'):
        m=json.loads((root/arm/'metrics.json').read_text())
        if not m['complete'] or len(m['records'])!=2:raise ValueError('Incomplete probe')
        phases={};curve=[];memory={}
        for rep in m['records']:
            if rep['diagnostic_vs_standard_max_abs']!=0 or rep['coarse_parity_max_abs']!=0:raise ValueError('Probe perturbs path')
            if len(rep['teacher_path'])!=90 or len(rep['student_path'])!=90:raise ValueError('Call count')
        for index in range(90):
            teacher=[r['teacher_path'][index]['on_original_state'] for r in m['records']]
            student=[r['student_path'][index]['on_student_state'] for r in m['records']]
            curve.append(dict(index=index,t=m['records'][0]['teacher_path'][index]['t'],
                on_original={key:mean(teacher,key) for key in teacher[0]},
                on_student={key:mean(student,key) for key in student[0]}))
        for name,indices in [('early',range(30)),('middle',range(30,60)),('late',range(60,90)),('all',range(90))]:
            record={}
            for path in ('on_original','on_student'):
                record[path]={key:statistics.mean(curve[i][path][key] for i in indices) for key in curve[0][path]}
            record['state_shift_error_ratio']=record['on_student']['epsilon_mse']/record['on_original']['epsilon_mse']
            record['high_band_fraction_on_student']=record['on_student']['high_band_mse']/record['on_student']['epsilon_mse']
            phases[name]=record
        for path,error_key in [('teacher_path','on_original_state'),('student_path','on_student_state')]:
            selected=[r[path][i] for r in m['records'] for i in m['protocol']['checked_fine_indices']]
            baseline=statistics.mean(v for r in selected for v in r[error_key]['epsilon_mse'])
            memory[path]={'aligned_mse':baseline}
            for variant in selected[0]['perturbations']:
                mse=statistics.mean(v for r in selected for v in r['perturbations'][variant]['epsilon_mse'])
                memory[path][variant]=dict(mse=mse,ratio_to_aligned=mse/baseline)
        arms[arm]=dict(phases=phases,memory=memory,curve=curve,config=m['config'])
    result=dict(complete=True,arms=arms,warning='Only16 paired samples per arm, same seeds. Agreement to original RDD is not ground truth or FID. Early/middle/late are30 fine calls each. Perturbation results average10 selected calls, not all90.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{'phases':d['phases'],'memory':d['memory']} for a,d in arms.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
