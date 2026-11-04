#!/usr/bin/env python3
"""Independent source-config comparisons; never infer best FID from defaults."""
import ast,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import yaml
from umdiff import native
OUT=ROOT/'evidence/current_recipe_parity_20261009'
CHECKS=[]
def obj(x):
    if hasattr(x,'to_dict'):return obj(x.to_dict())
    if isinstance(x,dict):return {k:obj(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [obj(v) for v in x]
    if hasattr(x,'__dict__'):return obj(vars(x))
    return x
def check(name,actual,expected,basis):
    row=dict(name=name,actual=obj(actual),expected=obj(expected),basis=basis,passed=obj(actual)==obj(expected));CHECKS.append(row)
    if not row['passed']:raise AssertionError(name)
def read(name):return json.loads((ROOT/'configs/paper_recipes'/(name+'.json')).read_text())
def model(c):return c['model']['args']['model'] if c['model']['name']=='adapter' else c['model']
def factory(repo,file,function):
    node=next(n for n in ast.parse((ROOT/'sources'/repo/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
    call=next(n.value for n in node.body if isinstance(n,ast.Return))
    return {k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg is not None}

def main():
    for path in sorted((ROOT/'sources/dctdiff/configs').glob('*.py')):
        source=native.load('dctdiff/configs/'+path.name).get_config();c=read('dctdiff_'+path.stem);net=obj(source.nnet);net.pop('name')
        check(path.stem+'/model',model(c)['args'],net,'Source config.nnet, all fields')
        for field in ['lr','betas','weight_decay']:check(path.stem+'/optimizer/'+field,c['training']['optimizer'][field],source.optimizer[field],'Source config.optimizer')
        check(path.stem+'/sample_steps',c['sampler']['args']['steps'],source.sample.sample_steps,'Source config.sample')
        check(path.stem+'/SNR',c['path']['args']['snr_scale'],source.dataset.SNR_scale,'Source config.dataset')
        check(path.stem+'/data_crop',c['training']['data']['crop'],'pre_resized','Source DCT dataset consumes already resized RGB before OpenCV DCT')
    for path in sorted((ROOT/'sources/dimr/configs').glob('*.py')):
        source=native.load('dimr/configs/'+path.name).get_config();c=read('dimr_'+path.stem)
        check(path.stem+'/model',model(c)['args']['config'],source.nnet.model_args,'Source config.nnet.model_args, all fields')
        check(path.stem+'/weights',c['batch']['args']['loss_weights'],source.loss_coeffs,'Source config.loss_coeffs')
        check(path.stem+'/CFG',c['sampler']['args']['guidance_scale'],source.sample.scale,'Source config.sample.scale')
        check(path.stem+'/null_dropout',c['training']['conditioning']['y']['dropout'],source.dataset.p_uncond,'Source dataset CFGDataset probability')
    p=yaml.safe_load((ROOT/'sources/pixelflow/configs/pixelflow_xl_c2i.yaml').read_text());c=read('pixelflow_xl_c2i')
    check('pixelflow/model',model(c)['args'],p['model']['params'],'Exact released XL YAML model params')
    check('pixelflow/packing_patch',c['batch']['args']['patch'],p['model']['params']['patch_size'],'Packing uses the network patch size')
    check('pixelflow/rotary_dim',c['batch']['args']['attention_head_dim'],p['model']['params']['attention_head_dim'],'Packing uses network head dimension')
    check('pixelflow/lr',c['training']['optimizer']['lr'],float(p['train']['lr']),'YAML numeric string explicitly parsed')
    for level in [2,3]:
        c=read('ppflow_xl_'+str(level));expected=factory('ppflow','models.py','PPFlow_XL_'+str(level))
        for k,v in expected.items():check('ppflow'+str(level)+'/model/'+k,model(c)['args'][k],v,'Source PPFlow_XL factory')
        check('ppflow'+str(level)+'/sampler',c['sampler']['name'],'ppflow_native','Unchanged published transport backend')
        check('ppflow'+str(level)+'/heun',c['sampler']['args']['sampling_method'],'heun2','Source train_utils.py ODE default')
    for profile,function in [('dfm_sit_xl','DFM_SIT_XL_2'),('dfm_sit_b_released_yaml','DFM_SIT_B_2')]:
        c=read(profile);expected=factory('dfm','models/sit_dfm.py',function)
        check(profile+'/native_debug_rng',c['batch']['args']['native_debug_rng'],True,'Source loss.py logging draws advance NumPy state and subsequent stage choice')
        for k,v in expected.items():
            if k!='patch_sizes':check(profile+'/model/'+k,model(c)['args'][k],v,'Source released SiT factory')
    check('dfm_repa/native_debug_rng',read('dfm_sit_xl_repa_localfeatures')['batch']['args']['native_debug_rng'],True,'REPA must retain the native logging RNG policy')
    train=yaml.safe_load((ROOT/'sources/dfm/experiments/train/dfm_sit_b_256.yaml').read_text());sample=yaml.safe_load((ROOT/'sources/dfm/experiments/generation/dfm_sit_b_256.yaml').read_text());c=read('dfm_sit_b_released_yaml')
    check('dfm_yaml/weighting',c['training']['time_sampler']['args']['weighting'],train['weighting'],'Released training YAML')
    check('dfm_yaml/generation_steps',c['clock']['args']['steps'],sample['num_steps_per_scale'],'Generation YAML differs from training-preview step count')
    check('dfm_yaml/threshold',c['clock']['args']['thresholds'],sample['stage_sampling_thresholds'],'Released generation YAML')
    check('dfm_yaml/cfg',c['guidance']['args']['scale'],sample['cfg_scale'],'Released generation YAML')
    c=read('lapflow_imagenet_xl_2_paper')
    for key,actual,expected in [('LR',c['training']['optimizer']['lr'],1e-4),('EMA',c['training']['optimizer']['ema_decay'],.9999),('T',c['batch']['args']['activations'],[.5,0.]),('VAE',c['training']['input_transform']['args']['model']['args']['path'],'@eqvae@')]:check('lapflow/paper/'+key,actual,expected,'Paper Appendix G https://arxiv.org/html/2602.19461 ; training flags only, best CFG not identified')
    for size in [256,1024]:
        source=yaml.safe_load((ROOT/f'sources/mdm/ml-mdm-matryoshka/configs/models/cc12m_{size}x{size}.yaml').read_text());c=read('mdm_cc12m'+str(size));net=obj(source['unet_config']);actual=model(c)['args']['config']
        # The source CLI supplies T5 dimension, then the constructor propagates
        # it to inner configs. Local checkpoint replaces an S3 run identifier.
        net['conditioning_feature_dim']=2048
        if size==256:
            net['inner_config']['conditioning_feature_dim']=2048
            if net.get('initialize_inner_with_pretrained')=='None':net['initialize_inner_with_pretrained']=None
        else:net['initialize_inner_with_pretrained']='@mdm_inner256@'
        check('mdm'+str(size)+'/network',actual,net,'All source YAML dimensions; explicit T5 input dimension and strict local init identifier')
        check('mdm'+str(size)+'/inference_grid',c['sampler']['args']['inference_steps'],250,'Source generate_sample.py demo default; best FID not identified')
    s=yaml.safe_load((ROOT/'sources/ssd/configs/training.yaml').read_text());c=read('ssd_celeba64')
    for k,v in [('lr',float(s['lr'])),('ema_decay',float(s['ema_rate'])),('weight_decay',s['weight_decay'])]:check('ssd/optimizer/'+k,c['training']['optimizer'].get(k,{'lr':1e-4,'ema_decay':.9999,'weight_decay':0.}[k]),v,'Source training YAML')
    r=yaml.safe_load((ROOT/'sources/rmd/configs/train_wan_1_3b.yaml').read_text());c=read('rmd_wan13b')
    for k in ['k_step','eta','cfg','flow_shift','flow_shift_trans','low_rs_step','warmup_step','split_timestep','t_offset_factor','score_weighting_mode']:check('rmd/'+k,c['distillation'][k],r[k],'Released Wan1.3B training YAML')
    for stage in ['low','high']:
        c=read('relay_imagenet_'+stage);check('relay_'+stage+'/globalbatch',c['evidence']['author_global_batch'],4096,'Source README ImageNet reproduction commands,8 ranks')
        check('relay_'+stage+'/global_image_clock',c['training']['optimizer']['image_count_multiplier'],8,'Source cur_nimg counts all8 ranks')
        check('relay_'+stage+'/gradient_repair',c['training']['optimizer']['gradient_repair'],{'nan':0.,'posinf':1e5,'neginf':-1e5},'Source training_loop.py nan_to_num before optimizer')
    pins=[];changed=[]
    for c in [json.loads(p.read_text()) for p in (ROOT/'configs/paper_recipes').glob('*.json') if p.name!='INDEX.json']:
        for source in c['evidence'].get('source_pins',[]):
            root=ROOT.parent if source.get('root')=='workspace' else ROOT/'sources';path=root/(source['file'].removeprefix('multiscale_diffusion/') if source.get('root')=='workspace' else source['file'])
            actual=hashlib.sha256(path.read_bytes()).hexdigest();pins.append(source)
            if actual!=source['sha256']:changed.append(str(path))
    if changed:raise AssertionError('Source pins changed')
    before=OUT/'before_configs';diffs=[]
    def changes(a,b,path=''):
        if isinstance(a,dict) and isinstance(b,dict):
            return sum([changes(a.get(k),b.get(k),path+'/'+k) for k in sorted(a.keys()|b.keys())],[])
        return [] if a==b else [dict(field=path,before=a,after=b)]
    for path in sorted((ROOT/'configs/paper_recipes').glob('*.json')):
        if path.name=='INDEX.json':continue
        current=json.loads(path.read_text());old=before/path.name
        diffs.append(dict(recipe=path.stem,added=not old.exists(),changes=changes(json.loads(old.read_text()),current) if old.exists() else [],claim=current['evidence']['claim']))
    report=dict(status='passed',checks=CHECKS,source_pin_checks=len(pins),source_pin_changes=changed,template_diffs=diffs,
        recipe_claim_counts=dict(Counter(d['claim'] for d in diffs)),all_best_recipes_identified=False,
        limitations=['Checks validate declared fields against published config/code; no best-run inference for unspecified defaults.',
                     'Diffusability best training instructions/checkpoints remain TODO in https://github.com/snap-research/diffusability .',
                     'DFM release explicitly uses a different architecture from the paper; source-native equivalence does not establish original-paper equivalence.',
                     'RMD attribution unconfirmed; no full video resolution/BF16/DDP quality certification.',
                     'LapFlow public CLI defaults and paper AppendixG differ; best CFG/evaluator/checkpoint still need original experiment metadata.'])
    OUT.mkdir(exist_ok=True);(OUT/'config_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(CHECKS),recipes=len(diffs),source_pins=len(pins)),ensure_ascii=False))
if __name__=='__main__':main()
