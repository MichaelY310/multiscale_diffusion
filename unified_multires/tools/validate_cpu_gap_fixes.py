#!/usr/bin/env python3
"""Executable regression checks for specific original/unified audit findings."""
import argparse,ast,copy,hashlib,json,os,random,sys,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import native,build
from activation_trace import byte_view
import conformance
CHECKS=[]

def record(name,fn):
    try:details=fn();CHECKS.append(dict(name=name,status='passed',details=details))
    except Exception:CHECKS.append(dict(name=name,status='failed',failure=traceback.format_exc()))

def reject(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Expected rejection')

def labels():
    from umdiff.cli import conditions
    from umdiff.source_objectives import setup
    for name in ('relay_imagenet_low','relay_imagenet_high','dctdiff_imgnet64_uvit_mid_2by2'):
        config=json.loads((ROOT/'configs/paired_experiments'/(name+'.json')).read_text())
        from validate_paired_experiments import reduced
        config,_=reduced(config);images=torch.randn(2,*config['image_shape'])
        reject(lambda:conditions(config['training'],{'images':images},torch.device('cpu'),2))
        reject(lambda:setup('dctdiff' if name.startswith('dct') else 'relay_low' if name.endswith('low') else 'relay_high',config,images))
        for bad in (torch.tensor([0.,1.]),torch.tensor([0]),torch.tensor([-1,1]),torch.tensor([0,1000])):
            reject(lambda:conditions(config['training'],{'images':images,'labels':bad},torch.device('cpu'),2))
        conditions(config['training'],{'images':images,'labels':torch.tensor([1,937])},torch.device('cpu'),2)
    return {'missing_and_invalid_labels':'rejected','valid_distinct_classes':'accepted','profiles':3}

def comparator():
    reject(lambda:conformance.compare('negative/dtype',torch.tensor([1.],dtype=torch.float32),torch.tensor([1.],dtype=torch.float64)))
    conformance.compare('negative/signed_zero',torch.tensor([0.]),torch.tensor([-0.]),atol=0,rtol=0)
    assert not conformance.CHECKS[-1]['bitwise']
    reject(lambda:conformance.compare('negative/nan',torch.tensor([float('nan')]),torch.tensor([float('nan')])))
    conformance.compare('positive/inf',torch.tensor([float('inf')]),torch.tensor([float('inf')]))
    assert conformance.CHECKS[-1]['bitwise']
    return {'dtype_and_nan':'rejected','signed_zero':'byte_difference_detected','identical_infinity':'accepted'}

def index_statements(path,function,start_name,stop_name,namespace):
    # Locate the actual index block by named assignments in the named function.
    # Fixed line numbers silently drift when a reader branch is added above it.
    tree=ast.parse(path.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function]
    assert len(functions)==1,(path,function)
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    candidates=[]
    for parent in ast.walk(functions[0]):
        for _,body in ast.iter_fields(parent):
            if not isinstance(body,list):continue
            starts=[i for i,n in enumerate(body) if assigns(n,start_name)]
            for start in starts:
                stops=[i for i in range(start+1,len(body)) if assigns(body[i],stop_name)]
                if stops:candidates.append(body[start:stops[0]])
    assert len(candidates)==1,(path,function,start_name,stop_name,len(candidates))
    nodes=candidates[0]
    assert any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='indices' for node in nodes for n in ast.walk(node))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def data_order():
    # Execute the actual production index statements, not a copied algorithm.
    from torch.utils.data import DataLoader,DistributedSampler
    records=[]
    for shuffle in (False,True):
        dataset=list(range(5));sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=shuffle,seed=29)
        reference=[]
        for epoch in range(3):
            sampler.set_epoch(epoch)
            reference.extend([v.tolist() for v in DataLoader(dataset,batch_size=2,sampler=sampler,drop_last=True)])
        recipe=dict(drop_last=True,shuffle=shuffle);cache={};permutation_cache={}
        for step,expected in enumerate(reference):
            a=index_statements(ROOT/'umdiff/cli.py','train','drop_last','batch',dict(torch=torch,training={'data':recipe},dataset=dataset,batch_size=2,trainer=SimpleNamespace(step_index=step),args=SimpleNamespace(seed=29),permutation_cache=permutation_cache))
            b=index_statements(ROOT/'tools/paired_experiments.py','execute','epoch_size','value',dict(torch=torch,recipe=recipe,dataset=dataset,batch=2,step=step,seed=29,cache=cache))
            assert a['indices']==b['indices']==expected,(step,a['indices'],b['indices'],expected)
            cache=b['cache'];permutation_cache=a['permutation_cache']
            # Resume with no cache at the same persisted step.
            a['permutation_cache']={};resumed=index_statements(ROOT/'umdiff/cli.py','train','drop_last','batch',a)
            assert resumed['indices']==expected
        records.append(dict(shuffle=shuffle,ids=reference,resume_exact=True))
    return records

def mdm_rng():
    from umdiff.source_objectives import setup
    from umdiff.native_models import import_mdm
    from validate_full_forward_backward import snapshot_rng,fingerprint
    from validate_paired_experiments import reduced
    import_mdm() # ordinary module import is outside minibatch binding
    results=[]
    for name in ('mdm_cc12m256','mdm_cc12m1024'):
        config,_=reduced(json.loads((ROOT/'configs/paired_experiments'/(name+'.json')).read_text()))
        images=torch.randn(2,*config['image_shape']);text=torch.randn(2,3,32)
        before=fingerprint(snapshot_rng());setup('mdm1024' if name.endswith('1024') else 'mdm',config,images,torch.tensor([0,1]),{'lm_outputs':text,'lm_mask':torch.ones(2,3)})
        after=fingerprint(snapshot_rng());assert before==after
        results.append(dict(profile=name,torch_numpy_python_rng_unchanged=True))
    return results

def relay_recipe():
    low=json.loads((ROOT/'configs/paired_experiments/relay_imagenet_low.json').read_text())
    high=json.loads((ROOT/'configs/paired_experiments/relay_imagenet_high.json').read_text())
    for cfg,ls,denom in ((low,25,32),(high,1,256)):
        assert cfg['model']['args']['augment_dim']==9
        assert cfg['batch']['args']['augmentation']['p']==.12
        opt=cfg['training']['optimizer'];assert opt['loss_reduction']=='sum' and opt['loss_multiplier']==ls/denom
    assert high['training']['optimizer']['ema_schedule']['half_life_images']==500000
    assert low['training']['optimizer']['ema_schedule']['half_life_images']==50000000
    return {'augmentation_p':.12,'augment_dim':9,'low_ema_images':50000000,'high_ema_images':500000,'loss':'unreduced.sum() * native ls/t; no mean reordering'}

def mdm_accumulation():
    from umdiff.training import Trainer
    from umdiff.engine import TrainingResult
    a=torch.nn.Linear(2,1);b=copy.deepcopy(a);images=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    for chunk in images.chunk(2):
        loss=a(chunk).square().mean()
        # Preserve the author's FP32 ordering: divide the reporting loss
        # only after backward. This intentionally accumulates a gradient sum.
        native.statements('mdm/ml-mdm-matryoshka/ml_mdm/trainer.py',75,77,dict(loss=loss,num_grad_accumulations=2))
    class Objective:
        config={}
        def training(self,net,x,*args,**kw):return TrainingResult(net(x).square(),None,None,None,{})
    trainer=Trainer(Objective(),b,lr=0.,accumulation_steps=2,accumulation_normalization='sum')
    trainer.step(images,None)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(byte_view(p.grad),byte_view(q.grad))
    for name,normalization in [('mdm_cc12m256','sum'),('mdm_cc12m1024','mean')]:
        cfg=json.loads((ROOT/'configs/paired_experiments'/(name+'.json')).read_text())
        assert cfg['training']['optimizer']['accumulation_normalization']==normalization
    return {'microbatches':2,'raw_gradients_byte_exact':True,'source_fp32_behavior':'backward before reporting-loss division','fp16_policy':'mean; normalization before scaler/backward'}

def resolved_plans():
    from umdiff.cli import read_config
    d=ROOT/'evidence/forward_backward_20261012/plans'
    a=json.loads((d/'paper_original_plan.json').read_text())['plans'];b=json.loads((d/'paper_unified_plan.json').read_text())['plans']
    assert len(a)==len(b)==33
    for x,y in zip(a,b):
        assert x['contract']==y['contract'] and x['contract_sha256']==y['contract_sha256']
        path=Path(x['contract']['resolved_config_path']);content=json.loads(path.read_text())
        assert content==x['contract']['method_config']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==x['resolved_config_sha256']
        assert str(path) in x['production_entrypoints']['unified']['argv']
        if x['paper'] not in ('diffusability','rmd'):assert read_config(path)==content
    return {'paired_profiles':33,'on_disk_values_match_contract':True,'unified_argv_loads_resolved':True,'native_cli_certified':False}

def pixelflow_api():
    from umdiff.compat.pixelflow_api import PixelFlowPipeline as UnifiedAPI
    from umdiff.native_models import activate
    model=build('model',{'name':'native_pixelflow','args':{'in_channels':3,'out_channels':3,'num_attention_heads':2,'attention_head_dim':16,'depth':2,'patch_size':2,'num_classes':1000}}).eval()
    from diffusers.models.embeddings import _get_2d_rotary_pos_embed_np
    from pixelflow.pipeline_pixelflow import PixelFlowPipeline
    import pixelflow.pipeline_pixelflow as pipeline
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    pipeline.get_2d_rotary_pos_embed=_get_2d_rotary_pos_embed_np
    a=PixelFlowPipeline(PixelFlowScheduler(1000,4),model);b=UnifiedAPI(copy.deepcopy(model))
    rows=[]
    for cfg,shift,ode in [(0.,1.,False),(4.,1.,False),(0.,2.,False),(4.,2.,False),(4.,1.,True)]:
        kwargs=dict(height=32,width=32,num_inference_steps=3,guidance_scale=cfg,shift=shift,device=torch.device('cpu'),use_ode_dopri5=ode)
        torch.manual_seed(3201);expected=a([1,937],**kwargs)
        torch.manual_seed(3201);actual=b([1,937],**kwargs)
        assert type(actual) is type(expected) is np.ndarray and actual.dtype==expected.dtype==np.float32
        assert actual.shape==expected.shape==(2,32,32,3)
        assert actual.tobytes(order='C')==expected.tobytes(order='C')
        rows.append(dict(cfg=cfg,shift=shift,dopri5=ode,pixels_and_api_byte_exact=True))
    return {'cases':rows,'adapter':'umdiff.compat.pixelflow_api.PixelFlowPipeline','adapter_sha256':hashlib.sha256((ROOT/'umdiff/compat/pixelflow_api.py').read_bytes()).hexdigest(),'backbone':'small released network, shared CPU SDPA; no pretrained claim'}

def dct_float():
    import cv2
    import tempfile
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset
    from umdiff.training import LocalImages
    from umdiff.dctcodec import DCTYCbCr
    source=ROOT/'sources/dctdiff/utils.py';tree=ast.parse(source.read_text())
    names={'DCT_to_RGB'}
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    decode=next(n for n in nodes if n.name=='DCT_to_RGB')
    removed=[n for n in decode.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='uint8']
    assert len(removed)==1
    decode.body=[n for n in decode.body if n not in removed]
    env={'np':np,'cv2':cv2}
    env.update(native.extract('dctdiff/DCT_utils.py',['idct_transform','combine_blocks'],env))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    utility=native.load('dctdiff/DCT_utils.py')
    namespace=dict(torch=torch,np=np,cv2=cv2,os=os,Image=Image,Dataset=Dataset,transforms=transforms)
    namespace.update({key:getattr(utility,key) for key in ('zigzag_order','reverse_zigzag_order','split_into_blocks','dct_transform')})
    datasets=native.extract('dctdiff/datasets.py',['DCT_4YCbCr','DCT_4YCbCr_cond'],namespace)
    rows=[]
    for path in sorted((ROOT/'configs/paired_experiments').glob('dctdiff_*.json')):
        cfg=json.loads(path.read_text());r=cfg['representation']['args'];codec=DCTYCbCr(**r)
        with tempfile.TemporaryDirectory(prefix='umdiff_dct_source_pixels_') as directory:
            folder=Path(directory)/'synthetic';folder.mkdir()
            file=folder/'frame.png';pixels=torch.randint(0,256,(r['resolution'],r['resolution'],3),dtype=torch.uint8)
            Image.fromarray(pixels.numpy()).save(file)
            parameters=dict(img_sz=r['resolution'],tokens=(r['resolution']//(2*r['block']))**2,low_freqs=r['frequencies'],block_sz=r['block'],Y_bound=r['bound'])
            conditional=cfg['model']['args']['num_classes']>0
            dataset=datasets['DCT_4YCbCr_cond'](train_files=[str(file)],labels=[937],**parameters) if conditional else datasets['DCT_4YCbCr'](directory,**parameters)
            torch.manual_seed(4103);native_tokens=dataset[0]
            if conditional:
                native_tokens,label=native_tokens;assert int(label)==937
            torch.manual_seed(4103);signed_pixels=LocalImages(directory,r['resolution'],crop='pre_resized',horizontal_flip=True,flip_rng='torch')[0]
            port_tokens=codec.encode(signed_pixels[None])[0]
            assert torch.equal(byte_view(native_tokens),byte_view(port_tokens)),path.name
        tokens=torch.randn(1,(r['resolution']//(2*r['block']))**2,6*r['frequencies'])*.7
        expected=env['DCT_to_RGB'](tokens[0].numpy(),tokens=tokens.shape[1],low_freqs=r['frequencies'],block_sz=r['block'],reverse_order=codec.reverse,resolution=r['resolution'],Y_bound=r['bound'])
        actual=codec.decode(tokens,output_dtype='float64')[0].permute(1,2,0).numpy()
        assert actual.dtype==expected.dtype==np.float64
        assert np.array_equal(actual.view(np.uint8),expected.view(np.uint8)),path.name
        quantized=codec.decode(tokens)[0].permute(1,2,0).numpy()
        assert np.array_equal(quantized,expected.astype(np.uint8))
        rows.append(dict(profile=path.stem,resolution=r['resolution'],source_png_flip_encode_byte_exact=True,conditional_label_checked=conditional,floating_pixels_byte_exact=True,uint8_byte_exact=True))
    return {'profiles':rows,'author_modification':'Only final np.uint8(rgb_reconstructed) removed to expose existing float64 boundary','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}

def parser_checks():
    import argparse
    from paired_experiments import load_manifest,compile_plan
    from author_cli_plans import commands
    manifest=load_manifest();results=[]
    for entry in manifest['profiles']:
        paper=entry['paper']
        if paper not in ('dfm','lapflow','rdd'):continue
        mode='paper' if isinstance(entry['paper_contract'].get('global_batch'),int) else 'fair-rdd'
        plan=compile_plan(manifest,entry,mode);cmd=commands(entry,plan['contract'])['original']['argv']
        if paper=='dfm':
            source=ROOT/'sources/dfm/train.py';tree=ast.parse(source.read_text())
            nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULTS' for t in n.targets)]
            nodes.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_parser','get_model_type'))
            registry={}
            for model_file in ('sit.py','sit_dfm.py','sit_elit.py'):
                model_tree=ast.parse((ROOT/'sources/dfm/models'/model_file).read_text())
                definition=next(n.value for n in model_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SiT_models' for t in n.targets))
                registry.update({ast.literal_eval(key):None for key in definition.keys})
            env={'argparse':argparse,'ALL_MODELS':registry};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
            parsed=env['build_parser']().parse_args(cmd[2:]);assert parsed.exp_name==entry['id'] and parsed.model==('DFM-SiT-B/2' if '_b_' in entry['id'] else 'DFM-SiT-XL/2')
            assert env['get_model_type'](parsed.model)=='dfm'
        else:
            source=ROOT/'sources/lapflow/feb12_train_ours.py' if paper=='lapflow' else ROOT.parent/'2d/train_RDD_imagenet256_unified.py'
            tree=ast.parse(source.read_text());block=next(n for n in reversed(tree.body) if isinstance(n,ast.If))
            nodes=[]
            for n in block.body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets):break
                nodes.append(n)
            env={'argparse':argparse};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
            args=cmd[2:] if paper=='lapflow' else cmd[6:]
            parsed=env['parser'].parse_args(args)
            if paper=='lapflow':assert parsed.dataset=='imagenet_256' and parsed.image_size==256 and parsed.num_levels==2 and parsed.use_mask and parsed.num_classes==1000
            else:assert parsed.data_source=='imagenet'
        results.append(dict(profile=entry['id'],source=str(source),argv=cmd,parsed=vars(parsed),mode=mode,status='passed'))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261012)
    for name,fn in [('label_guards',labels),('strict_comparator_negative_controls',comparator),('drop_last_shuffle_resume',data_order),('mdm_setup_rng',mdm_rng),('mdm_native_fp32_accumulation',mdm_accumulation),('relay_recipe',relay_recipe),('resolved_pair_values',resolved_plans),('dct_float_decode_13_profiles',dct_float),('actual_author_parsers',parser_checks),('pixelflow_author_output_api',pixelflow_api)]:record(name,fn)
    report=dict(status='passed' if all(c['status']=='passed' for c in CHECKS) else 'failed',checks=CHECKS,device='cpu',gpu_used=False,cuda_initialized=torch.cuda.is_initialized())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
