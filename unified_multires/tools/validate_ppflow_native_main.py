#!/usr/bin/env python3
"""Run the author's main/body on local fixtures with an explicit CPU overlay.

Retains the real Gloo DDP, ImageFolder/transforms/sampler, VAE posterior,
transport/loss/backward/AdamW/EMA/log/save. Only launch/device operations and
asset constructors are overlaid. This is not the original CUDA launch.
"""
import argparse,ast,copy,hashlib,json,logging,os,random,sys,tempfile,time,traceback
from collections import OrderedDict
from glob import glob
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from umdiff import build,native
from umdiff.engine import Method
from umdiff.training import Trainer
from umdiff.input_transforms import LocalVAE
from umdiff.native_models import activate
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from compare_released_profiles import compare_state


class CPUOverlay(ast.NodeTransformer):
    def __init__(self):self.changes=[]
    def visit_Assert(self,node):
        if ast.unparse(node.test)=='torch.cuda.is_available()':
            self.changes.append('CUDA availability assert removed for explicit CPU fixture');return ast.copy_location(ast.Pass(),node)
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and ast.unparse(node.targets[0])=='device':
            self.changes.append('device = torch.device(cpu)');node.value=ast.parse("torch.device('cpu')",mode='eval').body
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('torch.cuda.set_device','torch.cuda.synchronize'):
            self.changes.append(ast.unparse(node.value)+' removed on CPU');return ast.copy_location(ast.Pass(),node)
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if ast.unparse(node.func)=='dist.init_process_group':
            node.args=[ast.Constant('gloo')];node.keywords=[ast.keyword('init_method',ast.Name('gloo_init',ast.Load())),ast.keyword('rank',ast.Constant(0)),ast.keyword('world_size',ast.Constant(1))]
            self.changes.append('NCCL launch → single-rank Gloo file rendezvous')
        if ast.unparse(node.func)=='DDP':
            node.keywords=[k for k in node.keywords if k.arg!='device_ids'];self.changes.append('DDP CPU device_ids omitted')
        return node


def main():
    p=argparse.ArgumentParser();p.add_argument('--levels',type=int,choices=[2,3],default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(91);np.random.seed(91);random.seed(91)
    started=time.time();audit=None;report=dict(gpu_used=False,formal_training=False,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        activate('ppflow',['models','modules','transport'])
        from transport import create_transport,Sampler
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel as DDP
        from diffusers import AutoencoderKL
        c=json.loads((ROOT/f'configs/native_fixtures/ppflow_{a.levels}.json').read_text())
        c['training']['optimizer']=dict(lr=1e-4,weight_decay=0.,ema_decay=.9999,optimizer_parameters='all',randomness='native_global')
        c['model']['args']['class_dropout_prob']=.1
        source=native.ROOT/'sources/ppflow/train.py';tree=ast.parse(source.read_text())
        names=['main','update_ema','requires_grad','cleanup','create_logger','center_crop_arr'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        transform=CPUOverlay();nodes=[transform.visit(copy.deepcopy(n)) if n.name=='main' else n for n in nodes]
        module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
        with tempfile.TemporaryDirectory(prefix='umdiff_ppflow_main_') as tmp:
            tmp=Path(tmp);data=tmp/'images';data.mkdir();pixel_gen=np.random.default_rng(97)
            for i in range(32):
                target=data/f'{i%2:04d}';target.mkdir(exist_ok=True)
                Image.fromarray(pixel_gen.integers(0,256,(137,149,3),dtype=np.uint8)).save(target/f'{i:04d}.png')
            vae=AutoencoderKL(in_channels=3,out_channels=3,down_block_types=('DownEncoderBlock2D',)*4,
                up_block_types=('UpDecoderBlock2D',)*4,block_out_channels=(8,8,8,8),latent_channels=4,layers_per_block=1,norm_num_groups=4).eval()
            vae.save_pretrained(tmp/'vae',safe_serialization=True)
            codec=LocalVAE({'name':'local_autoencoder','args':{'path':str(tmp/'vae')}},freeze_parameters=False)
            model=nonzero_output(build('model',c['model']));port_model=copy.deepcopy(model)
            trainer=Trainer(Method(c),port_model,input_transform=codec,**c['training']['optimizer'])
            audit=ActivationAudit(f'ppflow{a.levels}_native_main',[('network',model,port_model),('vae',vae,codec.model)],a.output/'layers')
            rows=[];frame={};raw_encode=vae.encode;opt_holder={}
            def encode(images,*args,**kwargs):
                frame.update(images=images.detach().clone(),rng=snapshot_rng());audit.begin(trainer.step_index);audit.side('native')
                return raw_encode(images,*args,**kwargs)
            vae.encode=encode
            real_adam=torch.optim.AdamW
            def capture_optimizer(*args,**kwargs):
                optimizer=real_adam(*args,**kwargs);opt_holder['optimizer']=optimizer;return optimizer
            # Only the author's constructor call is redirected. Trainer's
            # optimizer was already constructed independently above.
            torch.optim.AdamW=capture_optimizer
            env=dict(torch=torch,dist=torch.distributed,DDP=DDP,os=os,glob=glob,deepcopy=copy.deepcopy,logging=logging,np=np,
                OrderedDict=OrderedDict,Image=Image,time=time.time,transforms=transforms,ImageFolder=ImageFolder,DataLoader=DataLoader,
                DistributedSampler=DistributedSampler,PPFlow_models={'local_fixture':lambda **kw:model},
                create_transport=create_transport,Sampler=Sampler,AutoencoderKL=SimpleNamespace(from_pretrained=lambda *args,**kwargs:vae),
                gloo_init='file://'+str(tmp/'rendezvous'),find_model=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Unexpected asset load')))
            os.environ['ENTITY']='local';os.environ['PROJECT']='local'
            exec(compile(module,str(source)+'::CPU-overlay','exec'),env);original_ema=env['update_ema']
            def paired_ema(ema,wrapped,decay=.9999):
                original_ema(ema,wrapped,decay)
                if decay==0:return
                after=snapshot_rng();raw_grads={k:None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()}
                # Labels come from the actual model invocation, including its
                # source class-dropout draw (the call input contains pre-drop labels).
                y=frame['labels'];audit.side('unified');restore_rng(frame['rng'])
                def check_raw(net):
                    differences=[]
                    for k,p in net.named_parameters():
                        expected=raw_grads[k]
                        if (expected is None)!=(p.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(p.grad))):differences.append(k)
                    frame['raw_gradient_differences']=differences
                result=trainer.step(frame['images'],None,model_kwargs={'y':y},before_gradient_processing=check_raw)
                audit.parameters(model,port_model);audit.finish_segment()
                checks={k:compare_state(u,v) for k,u,v in [('model',model.state_dict(),port_model.state_dict()),('optimizer',opt_holder['optimizer'].state_dict(),trainer.optimizer.state_dict()),('ema',ema.state_dict(),trainer.ema.state_dict())]}
                if frame['raw_gradient_differences'] or any(v['exact']!=v['tensors'] for v in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()):
                    report['failed_state']=checks;report['failed_raw_gradients']=frame['raw_gradient_differences'];raise AssertionError('Native main state/RNG differs')
                rows.append(dict(step=trainer.step_index,state=checks,rng_byte_exact=True,loss=result.loss.mean().item()));restore_rng(after)
            env['update_ema']=paired_ema
            def labels_hook(net,args,kwargs):
                if 'rng' in frame and trainer.step_index==len(rows):frame['labels']=kwargs['y'].detach().clone()
            hook=model.register_forward_pre_hook(labels_hook,with_kwargs=True)
            args=SimpleNamespace(global_batch_size=8,global_seed=917,results_dir=str(tmp/'results'),model='local_fixture',path_type='Linear',prediction='velocity',loss_weight=None,
                image_size=128,num_classes=1000,ckpt=None,initialize_from=None,patch_level=a.levels,train_eps=0.,sample_eps=0.,vae='ema',data_path=str(data),num_workers=0,
                wandb=False,cfg_scales=[1.],epochs=1,log_every=2,ckpt_every=2,sample_every=999)
            try:env['main'](args)
            finally:torch.optim.AdamW=real_adam;hook.remove()
            if len(rows)!=4:raise AssertionError('Author main did not execute four updates')
            checkpoints=[]
            for path in sorted((tmp/'results').rglob('*.pt')):
                state=torch.load(path,weights_only=False);checkpoints.append(dict(name=path.name,keys=list(state),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                if path.stem=='0000004':
                    for key,other in [('model',port_model.state_dict()),('ema',trainer.ema.state_dict()),('opt',trainer.optimizer.state_dict())]:
                        check=compare_state(state[key],other)
                        if check['exact']!=check['tensors']:raise AssertionError('Source serialized '+key+' differs')
            act=audit.close();audit=None
            if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('Native main layer/gradient mismatch')
            report.update(status='passed',steps=rows,activation=act,checkpoints=checkpoints,cpu_overlay=transform.changes,
                source_main_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_main_body_executed=True,
                native_checkpoint_limitations=['No RNG/reader/step fields','Resume resets train_steps to zero and overwrites args from checkpoint'],
                scope='Actual PPFlow main with explicit CPU launch overlay, reduced nonzero author model, random local real VAE and generated PNGs;native full per-batch posterior/CFG/transport/backward/optimizer/EMA/log/save. No CUDA/pretrained-quality certification.')
    except Exception:
        report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    if audit is not None:report['activation']=audit.close()
    report.update(seconds=time.time()-started,source_executions=native.AUDIT,cuda_initialized=torch.cuda.is_initialized())
    (a.output/f'ppflow{a.levels}_native_main.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','seconds')}),flush=True)
    return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
