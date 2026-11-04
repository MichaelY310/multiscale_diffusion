"""RMD inference-only conformance, no checkpoint or training required."""
import argparse,copy,json,time,traceback
from pathlib import Path
import torch
import validate_rmd as fixtures
from validate_comparison_sampling import check,clone,CHECKS
from umdiff.distillation import RMDObjective
from umdiff.randomness import GlobalRandomStream,RandomStream


@torch.no_grad()
def run():
    torch.manual_seed(20261015)
    _,_,_,_,source,_,model,template,text,_,_=fixtures.fixtures()
    a=model.eval();b=copy.deepcopy(a);x=torch.randn_like(template)
    from diffusers import AutoencoderKLWan
    vae=AutoencoderKLWan(base_dim=8,z_dim=4,dim_mult=[1,1],num_res_blocks=1,
        temperal_downsample=[False],latents_mean=[0.]*4,latents_std=[1.]*4).eval()
    variants=[]
    for steps in (4,6):
        for transition in (0,1,2):
            for eta in (0.,.5,1.):
                switch=(steps-1)//2-1
                na=[];nb=[]
                ha=a.register_forward_hook(lambda m,args,out:na.append(clone(out)))
                hb=b.register_forward_hook(lambda m,args,out:nb.append(clone(out)))
                torch.manual_seed(4701)
                expected=source.generate_new_upsample(a,None,x,x,text,None,steps=steps,eta=eta,
                    return_mid=True,flow_shift_trans=transition,relusion_shift=switch,target_size=(8,12))
                rng=torch.get_rng_state().clone();torch.manual_seed(4701)
                objective=RMDObjective(source.uncond_prompt_embeds,GlobalRandomStream(),eta=eta,
                    flow_shift_trans=transition,target_size=(8,12))
                final,clean,noisy=objective.predictor.generate(b,x,text,steps,transition,switch,(8,12))
                ha.remove();hb.remove()
                tag=f'steps{steps}/transition{transition}/eta{eta}'
                check(tag+'/clean_trajectory',expected[0],clean);check(tag+'/noisy_trajectory',expected[1],noisy)
                check(tag+'/model_outputs',na,nb);check(tag+'/RNG',rng,torch.get_rng_state())
                check(tag+'/final',expected[0][-1],final)
                check(tag+'/decoded_float_RGB',vae.decode(expected[0][-1],return_dict=False)[0],vae.decode(final,return_dict=False)[0])
                outer=torch.get_rng_state().clone()
                isolated=RMDObjective(source.uncond_prompt_embeds,RandomStream(4701),eta=eta,
                    flow_shift_trans=transition,target_size=(8,12))
                out=isolated.predictor.generate(b,x,text,steps,transition,switch,(8,12))
                check(tag+'/private_rng_trajectory',(final,clean,noisy),out)
                check(tag+'/outer_RNG',outer,torch.get_rng_state())
                variants.append(dict(steps=steps,transition=transition,eta=eta,model_calls=len(na)))
    return variants


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);start=time.monotonic()
    report=dict(device='cpu',model=fixtures.MODEL,scope='Original Predictor.generate_new_upsample, random reduced Wan and random tiny Wan VAE; no pretrained quality claim')
    try:report.update(variants=run(),status='passed')
    except Exception:report.update(status='failed',failure=traceback.format_exc())
    report.update(checks=CHECKS,seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','seconds','failure')}),flush=True)
    return report['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
