"""Real conditioning, native AR sampler, native VAE and bridge decode parity."""
import argparse
import json
import os
from pathlib import Path
import time
import torch

WORK=Path(__file__).resolve().parents[3]
MODEL=WORK/'assets_20261006/pyramid-flow-miniflux'
CACHE=WORK/'fair_data_20261006/native_pf_v1'
SOURCE=Path('/scr/yuheng/bugswarm/official_video_baselines_20261004/reference/Pyramid-Flow')
AUGMENT=', hyper quality, Ultra HD, 8K'


def encode(output):
    from pyramid_dit import FluxTextEncoderWithMask
    rows=[json.loads(x) for x in (CACHE/'validation.jsonl').read_text().splitlines()]
    output.mkdir(parents=True,exist_ok=False)
    encoder=FluxTextEncoderWithMask(str(MODEL),torch_dtype=torch.bfloat16).eval().cuda()
    manifest={}
    for n,prompt in enumerate(['']+[r['text']+AUGMENT for r in rows]):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            context,mask,pooled=encoder([prompt],torch.device('cuda'))
        value={'prompt_embed':context.cpu(),'prompt_attention_mask':mask.cpu(),
               'pooled_prompt_embed':pooled.cpu()}
        filename=f'prompt_{n:03}.pt';torch.save(value,output/filename)
        manifest[prompt]=filename
    (output/'manifest.json').write_text(json.dumps({'prompts':manifest,'rows':rows,
        'synthetic':False,'native_prompt_augmentation':AUGMENT,'negative_prompt':'',
        'source_commit':'a012faa1dc4d71301a7a153c7f9554c081947ea2'},indent=2))


class CachedNativeEncoder:
    def __init__(self, cache):
        self.cache=cache
        self.prompts=json.loads((cache/'manifest.json').read_text())['prompts']

    def __call__(self, prompts, device):
        prompts=[prompts] if isinstance(prompts,str) else prompts
        values=[torch.load(self.cache/self.prompts[p],map_location='cpu',weights_only=True) for p in prompts]
        return tuple(torch.cat([v[key] for v in values]).to(device) for key in
                     ('prompt_embed','prompt_attention_mask','pooled_prompt_embed'))


def sample(args):
    from pyramid_dit import PyramidDiTForVideoGeneration
    from video_vae import CausalVideoVAE
    from pyramid_compat import sample_psd_block_noise
    from full_model_trace import hashed
    if args.mode=='reference':
        runner=PyramidDiTForVideoGeneration(str(MODEL),model_dtype='bf16',model_name='pyramid_flux',
            model_variant='diffusion_transformer_384p',load_text_encoder=False,load_vae=False,
            use_flash_attn=False,max_temporal_length=max(8,args.temp),frame_per_unit=1,video_sync_group=1)
    else:
        from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
        from fastvideo.train.utils.config import load_run_config
        from fastvideo.train.utils.instantiate import instantiate
        maybe_init_distributed_environment_and_model_parallel(1,1)
        cfg=load_run_config(args.config)
        model_config=dict(cfg.models['student']);model_config['recipe']=dict(model_config['recipe'])
        model_config['recipe']['max_temporal_length']=max(model_config['recipe']['max_temporal_length'],args.temp)
        plugin=instantiate(model_config,training_config=cfg.training)
        runner=plugin.runner
    runner.dit.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    runner.vae=CausalVideoVAE.from_pretrained(str(MODEL/'causal_video_vae'),
        torch_dtype=torch.bfloat16,interpolate=False).to('cuda').eval()
    runner.vae.enable_tiling()
    runner.text_encoder=CachedNativeEncoder(args.prompt_cache)
    runner.sample_block_noise=lambda *pos:sample_psd_block_noise(runner.scheduler.config.gamma,*pos)
    rows=json.loads((args.prompt_cache/'manifest.json').read_text())['rows']
    counts=[]
    handle=runner.dit.register_forward_pre_hook(lambda m,a,k:counts.append(k['sample'][0][-1].shape[0]),with_kwargs=True)
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for n,row in enumerate(rows[:args.max_prompts]):
        torch.manual_seed(42+n)
        generator=torch.Generator(device='cuda').manual_seed(42+n)
        before=len(counts)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            latent=runner.generate(prompt=row['text'],negative_prompt='',height=384,width=640,
                temp=args.temp,num_inference_steps=args.image_counts,video_num_inference_steps=args.video_counts,
                guidance_scale=5.,video_guidance_scale=5.,output_type='latent',generator=generator)
        torch.cuda.synchronize();denoise=time.perf_counter()-start
        if not torch.isfinite(latent).all():raise FloatingPointError('Native sampler produced nonfinite latent')
        latent_hash=hashed(latent)
        start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            images=runner.decode_latent(latent.clone(),save_memory=True)
        torch.cuda.synchronize();decode=time.perf_counter()-start
        import numpy as np
        pixels=torch.from_numpy(np.stack([np.asarray(x) for x in images]))
        from diffusers.utils import export_to_video
        export_to_video(images,str(args.output/f'prompt_{n:03}.mp4'),fps=16)
        result={'id':row['id'],'caption':row['text'],'actual_positive_prompt':row['text']+AUGMENT,
            'negative_prompt':'','seed':42+n,'latent':latent_hash,'decoded_pixels':hashed(pixels),
            'frames':len(images),'resolution':[384,640],'fps':16,'cfg':5.,
            'actual_nfe':len(counts)-before,'branch_forward_calls':sum(counts[before:]),
            'denoise_seconds':denoise,'vae_seconds':decode,'quality_benchmark':False,
            'psd_noise_compat':'same Gaussian law at singular gamma=1/3; distinct RNG mapping from unavailable Cholesky'}
        expected=sum(args.image_counts)+(args.temp-1)*sum(args.video_counts)
        assert result['actual_nfe']==expected and result['branch_forward_calls']==2*expected
        result['image_steps']=args.image_counts;result['video_steps']=args.video_counts
        results.append(result)
    handle.remove()
    (args.output/'result.json').write_text(json.dumps({'status':'passed','mode':args.mode,
        'synthetic':False,'source_commit':'a012faa1dc4d71301a7a153c7f9554c081947ea2','samples':results},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['encode','reference','fastvideo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-cache',type=Path)
    p.add_argument('--config')
    p.add_argument('--max-prompts',type=int,default=2)
    p.add_argument('--temp',type=int,default=4,help='Native latent frames; decoded=(temp-1)*8+1')
    p.add_argument('--image-counts',type=lambda s:list(map(int,s.split(','))),default=[2,2,2])
    p.add_argument('--video-counts',type=lambda s:list(map(int,s.split(','))),default=[2,2,2])
    args=p.parse_args()
    if not os.environ.get('RDD_JOB_TOKEN') or time.time()>=float(os.environ.get('RDD_DEADLINE_UNIX','0')):
        raise RuntimeError('Independent GPU guard required')
    torch.use_deterministic_algorithms(True)
    from pyramid_smoke import check_source
    check_source(SOURCE)
    encode(args.output) if args.mode=='encode' else sample(args)
    check_source(SOURCE)


if __name__=='__main__':main()
