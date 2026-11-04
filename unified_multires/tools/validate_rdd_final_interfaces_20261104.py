"""Small interface checks after native trajectory validation; no GPU/assets."""
import ast,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from umdiff import build
from validate_rdd_fm_native_20261104 import source,exact

def main():
 torch.set_num_threads(2);old,_=source();checks=[]
 for factor in (2,4):
  for clock in (.3,.6):
   torch.manual_seed(841);clean=torch.randn(1,4,8,8);state=torch.randn_like(clean)
   rng=torch.get_rng_state();expected=old.noise_filling_transition_fm(state,clean,clock,8*factor)
   torch.set_rng_state(rng);noise=torch.randn(1,4,8*factor,8*factor)
   actual=build('boundary',{'name':'rdd_noise_filling','args':{'factor':factor}})(state,clean=clean,alpha=clock,sigma=1-clock,previous_alpha=clock,previous_sigma=1-clock,noise=noise).state
   exact(expected,actual,'component filling')
   a=build('boundary',{'name':'rdd_fresh','args':{'factor':factor}})(state,clean=clean,alpha=clock,sigma=1-clock,noise=noise).state
   b=build('boundary',{'name':'rdd_refresh','args':{'factor':factor}})(state,clean=clean,alpha=clock,sigma=1-clock,noise=noise).state
   exact(a,b,'fresh alias');checks.append({'factor':factor,'clock':clock,'component_matches_frozen_fm_byte_exact':True,'fresh_refresh_byte_exact':True})
 # Only execute the preview guard: no VAE imports or sampler allocations.
 path=WORK/'2d/RDD_sample_unified.py'
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_sample_one_batch')
 env={};exec(compile(ast.Module([node],type_ignores=[]),str(path),'exec'),env)
 guards=[]
 class ReachedBody(Exception):pass
 class Resolutions:
  def __len__(self):raise ReachedBody()
 for boundary,ddim,jump,dpm,reject in [('noise_filling',True,False,False,True),('noise_filling',False,False,False,True),('noise_filling',True,True,True,True),('noise_filling',False,True,False,False),('noise_filling',False,False,True,False),('fresh',True,False,False,False),('refresh',False,False,False,False)]:
  args=SimpleNamespace(upsample_mode=boundary,use_ddim=ddim,use_ddpm_jump_step=jump,use_dpm_solver=dpm)
  try:env['_sample_one_batch'](1,None,Resolutions(),None,None,None,None,args,'cpu')
  except ValueError:assert reject
  except ReachedBody:assert not reject
  else:raise AssertionError('Guard fixture did not terminate')
  guards.append({'boundary':boundary,'ddim':ddim,'jump':jump,'dpm':dpm,'expected_rejection':reject,'passed':True})
 # Prove the post-test helper edit only changed its docstring. Reconstruct
 # the previously tested bytes and compare against the recorded source hash.
 path2=WORK/'2d/sample_fid_unified.py';current=path2.read_text()
 before=current.replace('    tensors. At ``sigma_h == 0``, an IID standard Gaussian coarse innovation\n    independent of the fresh draw gives standard Gaussian fine noise. This\n    assumption does not automatically hold for a model-recovered residual.','    tensors.  At ``sigma_h == 0`` the combined fine noise is still exactly\n    standard Gaussian, so canonical noise filling and innovation preserving\n    have identical one-stage marginals but different cross-stage couplings.')
 summary=json.loads((ROOT/'evidence/rdd_reproduction_20261104/ddpm_boundary/expanded_matrix/SUMMARY.json').read_text())
 hashes=summary['source_sha256_before'] if 'source_sha256_before' in summary else summary.get('source_sha256',{})
 sha=hashlib.sha256(before.encode()).hexdigest()
 # Source report nested schema varies; exact hash must occur in the report.
 assert sha in json.dumps(summary),sha
 report={'status':'passed','standalone_boundaries':checks,'preview_sampler_guards':guards,'post_trajectory_docstring_edit':{'previous_verified_sha256':sha,'current_sha256':hashlib.sha256(current.encode()).hexdigest(),'only_docstring_changed':True},'cuda_initialized':torch.cuda.is_initialized()}
 (ROOT/'evidence/rdd_reproduction_20261104/FINAL_INTERFACES.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
