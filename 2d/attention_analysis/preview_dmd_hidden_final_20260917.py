"""Small inference-only four-way preview; no training, checkpoint writes or selection."""
import argparse
import gc
import hashlib
import html
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'DiT'))
from models_multires import DiTMultiRes_models
from dmd_cascade_reuse import CascadeDMD, coupled_noises


@torch.inference_mode()
def main(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an existing preview: {out}')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.08)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    specs = [
        ('dense', Path(a.teacher), 'dense'),
        ('start', Path(a.initial), 'high_heavy'),
        ('old', Path(a.old_run)/'checkpoints/latest.pt', 'high_heavy'),
        ('hidden', Path(a.hidden_run)/'checkpoints/latest.pt', 'high_heavy'),
    ]
    cohorts = [('fixed', 9100000), ('fresh', 20260917)]
    labels = None
    signatures = None
    latents = {}
    info = {}
    for name, path, family in specs:
        print(f'Loading {name}: {path}', flush=True)
        stat_before = path.stat()
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        step = int(ck.get('step', 0))
        if family == 'high_heavy':
            assert ck['family'] == family
        if name == 'start':
            assert step == 5000
        if name in ('old', 'hidden'):
            assert ck['objective'] == 'dmd_lpips'
            enabled = ck['protocol'].get('hidden_distillation', {}).get('enabled', False)
            assert enabled == (name == 'hidden')
        info[name] = dict(checkpoint=str(path), checkpoint_step=step,
                          cumulative_reuse_updates=(5000+step if name in ('old','hidden') else
                                                    5000 if name=='start' else None),
                          bytes=stat_before.st_size, mtime_ns=stat_before.st_mtime_ns,
                          family=family)
        model = DiTMultiRes_models['DiT-L/2'](num_classes=1000, base_size=16)
        model.load_state_dict(ck['model'], strict=True)
        del ck
        net = CascadeDMD(model.cuda().eval().requires_grad_(False), family, False)
        current_labels, current_sigs, batches = [], [], []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for cohort, seed in cohorts:
                rng = torch.Generator(device='cuda').manual_seed(seed)
                noise, y = coupled_noises(8, torch.device('cuda'), rng)
                current_labels.extend(y.cpu().tolist())
                current_sigs.append(hashlib.sha256(b''.join(
                    tensor.cpu().numpy().tobytes() for tensor in [y, *noise])).hexdigest())
                z = net(noise,y)['cleans'][-1]
                assert torch.isfinite(z).all()
                batches.append(z.cpu().float())
        if labels is None:
            labels, signatures = current_labels, current_sigs
        else:
            assert labels == current_labels and signatures == current_sigs
        latents[name] = torch.cat(batches)
        stat_after = path.stat()
        assert (stat_before.st_ino,stat_before.st_size,stat_before.st_mtime_ns) == (
            stat_after.st_ino,stat_after.st_size,stat_after.st_mtime_ns), 'Checkpoint changed during read'
        del net, model, noise, z, batches
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Sampled {name}: 16 paired images; step={step}', flush=True)

    out.mkdir(parents=True)
    vae = AutoencoderKL.from_pretrained(a.vae, local_files_only=True).cuda().eval().requires_grad_(False)
    pictures = {}
    for name, z in latents.items():
        pictures[name] = []
        for start in range(0,16,2):
            decoded = vae.decode(z[start:start+2].cuda()/0.18215).sample.float().clamp(-1,1)
            arrays = ((decoded+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
            for i, array in enumerate(arrays,start):
                im = Image.fromarray(array)
                im.save(out/f'{name}_{i:02d}_class{labels[i]}.png')
                pictures[name].append(im)
    del vae, decoded
    gc.collect()
    torch.cuda.empty_cache()

    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path,17) if Path(font_path).exists() else ImageFont.load_default()
    titles = ['Dense RDD-DMD', 'Reuse start: A5000', f'Old: +{info["old"]["checkpoint_step"]}',
              f'Hidden: +{info["hidden"]["checkpoint_step"]}']
    keys = list(latents)
    from torchvision.models import ResNet50_Weights
    categories = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    sheets=[]
    for page in range(4):
        sheet = Image.new('RGB',(1024,4*286+54),'white')
        draw = ImageDraw.Draw(sheet)
        for col,title in enumerate(titles):
            draw.text((col*256+5,5),title,font=font,fill='black')
        cohort,seed = cohorts[page//2]
        draw.text((5,29),f'{cohort} seed={seed}; same class/noise; training budgets DIFFER; no image selection',font=font,fill='black')
        for row in range(4):
            i=page*4+row
            draw.text((5,54+row*286),f'#{i:02d} class {labels[i]}: {categories[labels[i]]}',font=font,fill='black')
            for col,key in enumerate(keys):
                sheet.paste(pictures[key][i],(col*256,54+row*286+25))
        file=f'comparison_{page+1}.jpg'
        sheet.save(out/file,quality=95)
        sheets.append(file)
    manifest=dict(complete=True,sample_count_per_model=16,cohorts=cohorts,
                  labels=labels,categories=[categories[i] for i in labels],
                  noise_signatures=signatures,models=info,sheets=sheets,
                  selection='first8 fixed validation rank0 plus first8 new-seed samples; no filtering',
                  nfe=3,global_t=[999,599,299],resolutions=[8,16,32],cfg='no extra CFG',
                  noise_filling=True,model='shared DiT-L/2,24blocks',
                  reuse='middle8/24; fine20/24; previous-stage block residual; no adapter',
                  precision='FP32 weights, BF16 denoiser autocast; FP32 VAE; TF32 disabled',
                  latent_mse_to_dense={k:float((v-latents['dense']).square().mean()) for k,v in latents.items()},
                  max_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  NOT_FID=True,training_budget_matched=False,training_resumed=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows = ''.join(f'<tr><td>{html.escape(t)}</td><td>{html.escape(info[k]["checkpoint"])}</td>'
                   f'<td>{info[k]["cumulative_reuse_updates"] if info[k]["cumulative_reuse_updates"] is not None else "未进行复用微调"}</td></tr>'
                   for k,t in zip(keys,titles))
    galleries = ''.join(f'<h2>{"原固定种子" if i<2 else "新种子"}：样本{i*4}—{i*4+3}</h2>'
                        f'<a href="{name}"><img src="{name}" alt="四列同种子对比"></a>' for i,name in enumerate(sheets))
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDD复用最新四列对比 · 2026-09-17</title>
<style>body{{max-width:1100px;margin:24px auto;padding:0 16px;font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.7;color:#19212c}}img{{width:100%;height:auto}}table{{display:block;overflow-x:auto;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}td:nth-child(2){{max-width:600px}}a{{color:#174baa}}.warning{{background:#fff5dd;padding:14px}}h1{{font-size:25px}}</style>
<h1>RDD跨尺度复用：最新效果四列对比</h1>
<p>从左到右：<b>原始不跳层RDD-DMD → A@5000复用起点 → 旧DMD＋LPIPS最新 → 新增hidden loss最新</b>。</p>
<p class="warning">旧配方新增{info['old']['checkpoint_step']}步，hidden配方新增{info['hidden']['checkpoint_step']}步；两者预算不同，不能据此判断hidden loss的净收益。A@5000已经做过5000步复用微调，不是未经训练的跳层模型。左列也是多分辨率RDD，不是原始DiT。</p>
<table><tr><th>列</th><th>实际checkpoint</th><th>累计复用微调步数</th></tr>{rows}</table>
<p>共享DiT-L/2、24层、ImageNet256；3NFE，t=999/599/299；8→16→32；noise filling；无额外CFG。复用组中间阶段复用8层，fine复用20层。</p>
<p>固定seed=9100000的前8张，加新seed=20260917的前8张。四个模型严格共享每张图的类别与三阶段噪声；不筛图、不修图。PNG单图保留，JPEG仅用于拼接预览。没有启动任何训练，16张展示不是FID，也不是视频质量证据。</p>
{galleries}<p><a href="manifest.json">完整生成配置与checkpoint记录</a></p></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',required=True)
    p.add_argument('--hidden-run',required=True)
    p.add_argument('--initial',required=True)
    p.add_argument('--teacher',required=True)
    p.add_argument('--vae',required=True)
    p.add_argument('--output',required=True)
    main(p.parse_args())
