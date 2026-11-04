"""Independent nonuniform integration, stage NFE and v0.33 scheduler checks."""
import json
from pathlib import Path
import torch
from pyramidal_wan_sampling import scheduler_class,stage_schedulers
from pyramidal_wan_paper_math import to_global,NATURAL_CLEAN,NATURAL_NOISY
from rdd.comparison.sampling import ab2_velocity,allocate_steps
from rdd.comparison.paths import make_path

results={}
# v(t)=2t+3, unequal h=.1,.23. AB2 integrates a linear field exactly.
t=.7;h=.23;previous_h=.1
integral=h*ab2_velocity(torch.tensor(2*t+3,dtype=torch.float64),torch.tensor(2*(t-previous_h)+3,dtype=torch.float64),h,previous_h)
assert abs(float(integral)-(h*(2*t+3)+h*h))<1e-12
results['nonuniform_AB2_exact_linear_field']=True
for name in ['fixed','rdd','pyramid_flow','pyramidal_wan']:
 path=make_path(name,(0.,.4,.7,1.))
 for solver in ['euler','ab2','heun']:
  for budget in [25,50]:
   counts=allocate_steps(path,budget,solver)
   actual=sum(counts)*(2 if solver=='heun' else 1)+(2 if name=='rdd' else 0)
   assert actual<=budget and budget-actual<=1
results['all_method_solver_budgets_include_endpoints']=True
for stage,count in [(2,20),(1,20),(0,10)]:
 z,schedulers=stage_schedulers(stage,count,'cpu')
 assert z[0]==NATURAL_NOISY[stage] and z[-1]==NATURAL_CLEAN[stage]
 for first,scheduler in zip([True,False],schedulers):
  x=torch.randn(2,3)
  for n in range(count):
   v=torch.full_like(x,2.5)
   expected=x+(to_global(z[n+1],stage,first)-to_global(z[n],stage,first))*v
   x=scheduler.step(v,scheduler.timesteps[n],x,return_dict=False)[0]
   torch.testing.assert_close(x,expected,rtol=1e-6,atol=1e-6)
results['official_v033_scheduler_global_first_tail_updates']=True
out=Path(__file__).resolve().parents[3]/'SOLVER_MATH_CPU_20261006.json'
out.write_text(json.dumps({'status':'passed','checks':results},indent=2));print(results)
