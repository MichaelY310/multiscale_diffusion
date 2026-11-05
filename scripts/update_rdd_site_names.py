#!/usr/bin/env python3
"""Apply the reviewed registry to an existing Site without changing raw evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote

p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
a=p.parse_args();dist=a.site/'dist';registry=a.registry
results=json.loads((registry/'results.json').read_text())['results'];by_id={r['record_id']:r for r in results}
models=json.loads((registry/'models.json').read_text());by_group={m['legacy_hf_group']:m for m in models}
plan=json.loads((registry/'hf_rename_plan.json').read_text());path_map={x['old_path']:x['new_path'] for x in plan['operations']}

def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def title(m):return m['canonical_model'].split('__')[0]+' · 训练 '+m['training_display']

def model(m):
    group=m.get('group_path','').removeprefix('checkpoints/')
    spec=by_group.get(group)
    if spec:
        m.setdefault('legacy_name',m['name']);m['canonical_name']=spec['canonical_model']
        m['name']=spec['canonical_model'];m['display_name']=title(spec)
        c=m.setdefault('training_config',{})
        c['training_by_latent_resolution']=spec['training_stages'];c['training_display']=spec['training_display']
        if 'train_allocation_fine_middle_coarse' in c:
            c['legacy_training_label']=c.pop('train_allocation_fine_middle_coarse')
    for x in m.get('checkpoint_assets',[]):
        if x['path'] in path_map:x['canonical_path']=path_map[x['path']]

def run(r):
    row=by_id.get(r.get('id'))
    if not row:return
    assert r.get('metrics',row['metrics'])==row['metrics'], r['id']
    r.setdefault('legacy_name',r.get('name'));r['canonical_name']=row['canonical_name']
    r['name']=row['canonical_model'].split('__')[0]+' · ckpt '+row['train_step']+' · '+row['identity']['solver']+' · CFG '+row['cfg']
    r['steps_coarse_to_fine']=[s['steps'] for s in row['identity']['sampling_stages']]
    r['sampling_display']=row['sampling_display'];r['training_display']=row['training_display'];r['resolution_identity']=row['identity']
    if 'config' in r:
        c=r['config']
        for k in ('sampling_steps_fine_middle_coarse','train_allocation_fine_middle_coarse'):
            if k in c:c['legacy_ledger_'+k]=c.pop(k)
        c['resolution_identity']=row['identity'];c['canonical_name']=row['canonical_name']
    r['naming_source']='naming/results.json#'+row['record_id']

index=read(dist/'lab/index.json')
for m in index['models']:model(m)
model_names={m['id']:m.get('display_name',m['name']) for m in index['models']}
for r in index['runs']:
    run(r);r['model_names']=[model_names[x] for x in r.get('model_ids',[]) if x in model_names]
for asset in index['assets']:
    if asset['path'] in path_map:asset['canonical_path']=path_map[asset['path']]
for path in (dist/'lab/runs').glob('*.json'):
    rows=read(path)
    for r in rows:
        run(r);r['model_names']=[model_names[x] for x in r.get('model_ids',[]) if x in model_names]
    write(path,rows)
for path in (dist/'lab/models').glob('*.json'):
    m=read(path);model(m)
    for r in m.get('runs',[]):
        if isinstance(r,dict):run(r)
    write(path,m)

naming=dist/'naming';naming.mkdir(exist_ok=True)
for file in ('results.json','results.csv','models.json','configs.csv','repository_aliases.json','hf_rename_plan.json','remote_fm_reports.json'):
    if (registry/file).resolve()!=(naming/file).resolve():shutil.copyfile(registry/file,naming/file)
for label,file in [('逐网格正式结果 CSV','results.csv'),('已扫描采样配置 CSV','configs.csv'),('完整命名与原始证据 JSON','results.json')]:
    item=dict(label=label,href='naming/'+file)
    if item not in index['downloads']:index['downloads'].insert(0,item)
index['meta']['naming_schema']='rdd_resolution_identity_v1';write(dist/'lab/index.json',index)

# Correct the obsolete compact dataset too, so its downloadable fields cannot
# reintroduce the reversed FM/proportional columns. Its original source mirrors
# and the new registry's original_ledger_row remain untouched.
compact=dist/'experiments/catalog.json'
if compact.exists():
    data=read(compact)
    for r in data['formal']:
        row=by_id[r['record_id']];r.setdefault('legacy_model',r['model'])
        r['model']=row['canonical_model'];r['sampling_display']=row['sampling_display'];r['training_display']=row['training_display']
        r['sampling_steps_fine_middle_coarse']='/'.join(str(s['steps']) for s in reversed(row['identity']['sampling_stages']))
        r['train_allocation_fine_middle_coarse']='/'.join(str(s['duration']) for s in reversed(row['identity']['training_stages']))
        r['resolution_identity']=row['identity']
    write(compact,data)
    (compact.with_suffix('.js')).write_text('window.RDD_EXPERIMENTS='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    for name in ('curated_valid_results.csv','HISTORICAL_IMAGE_RESULTS_RELABELED.csv'):
        path=dist/'experiments'/name
        if not path.exists():continue
        backup=path.with_name('legacy_raw_'+name)
        if not backup.exists():shutil.copyfile(path,backup)
        with backup.open(encoding='utf-8-sig',newline='') as inp: csv_rows=list(csv.DictReader(inp))
        for r in csv_rows:
            row=by_id[r['record_id']]
            r['legacy_model']=r['model'];r['model']=row['canonical_model']
            r['sampling_steps_fine_middle_coarse']='/'.join(str(s['steps']) for s in reversed(row['identity']['sampling_stages']))
            r['train_allocation_fine_middle_coarse']='/'.join(str(s['duration']) for s in reversed(row['identity']['training_stages']))
            r['sampling_display']=row['sampling_display'];r['training_display']=row['training_display']
        with path.open('w',encoding='utf-8-sig',newline='') as out:
            w=csv.DictWriter(out,list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)

js=dist/'lab.js';s=js.read_text()
s=s.replace("cfg.train_allocation_fine_middle_coarse", "cfg.training_display||cfg.legacy_training_label")
s=s.replace('训练阶段分配（原字段）','各 latent 网格的训练时段长度')
s=s.replace('阶段步数 ${value(r.steps_coarse_to_fine)}','${value(r.sampling_display||r.steps_coarse_to_fine)}')
s=s.replace("field('采样步数（低 → 高）',r.steps_coarse_to_fine)","field('采样步数（逐 latent 网格）',r.sampling_display||r.steps_coarse_to_fine)")
s=s.replace('${esc(a.path)}</td><td>${bytes(a.bytes)}','${esc(a.canonical_path||a.path)}</td><td>${bytes(a.bytes)}')
s=s.replace('权重按归档路径列出','权重按明确的训练网格名称列出；下载链接保留已校验的固定归档 revision')
js.write_text(s)

oldjs=dist/'experiments.js'
if oldjs.exists():
    s=oldjs.read_text().replace('${esc(r.train_allocation_fine_middle_coarse)}','${esc(r.training_display||r.train_allocation_fine_middle_coarse)}')
    s=s.replace('${esc(r.sampling_steps_fine_middle_coarse)}（高→低）','${esc(r.sampling_display||r.sampling_steps_fine_middle_coarse)}')
    oldjs.write_text(s)

esc=lambda s:html.escape(str(s))
rows=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['record_id'],r['canonical_model'].split('__')[0],r['train_step'],r['training_display'],r['sampling_display'],r['identity']['solver'],r['identity']['expected_native_nfe'],r['cfg'],r['boundary'],r['metrics']['fid']])+'</tr>' for r in results)
modelrows=''.join('<tr><td>'+esc(m['legacy_model'])+'</td><td>'+esc(m['canonical_model'])+'</td><td>'+esc(' · '.join(f"{s['latent_resolution']}×{s['latent_resolution']}: {s['start']}→{s['end']}" for s in m['training_stages']))+'</td></tr>' for m in models)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>RDD 实验名称与逐网格配置</title>
<style>body{font:15px/1.65 system-ui,sans-serif;color:#192938;background:#f7f4ed;margin:0;padding:28px}main{max-width:1500px;margin:auto}a{color:#215cc4}h1{font-size:30px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}th{background:#e8e7e2;white-space:nowrap}.scroll{overflow:auto;background:white;border:1px solid #ddd;border-radius:8px}.note{padding:16px;background:#e9eef4;border-radius:8px;margin:20px 0}code{background:#e8e7e2;padding:3px}input{padding:12px;margin:12px 0;width:min(500px,85%);font:inherit}nav a{margin-right:18px}.scroll table{min-width:900px}td:nth-child(4),td:nth-child(5){min-width:180px}</style>
<main><nav><a href="experiments.html">实验首页</a><a href="models.html">全部模型</a><a href="sampling.html">全部采样</a><a href="boundaries.html">fresh / noise filling</a></nav>
<h1>每个数字，都写明对应哪个分辨率</h1>
<div class="note"><strong>旧 DDPM 文件名的 908080，主实验实际是 8×8 用 80 步、16×16 用 80 步、32×32 用 90 步。</strong><br>
FM 的 808090 也是这个分配。不能直接比较文件名；另外的共同比较配置确实使用了 8×8=90、16×16=80、32×32=80。</div>
<p>以下 8/16/32 是 latent 网格边长。训练区间长度、采样预算、NFE 是三个不同的量。旧 DDPM 的训练 300300400 表示 r8=400、r16=300、r32=300；FM-XL v2 的相同字符串却表示 r8=300、r16=300、r32=400。</p>
<p>同样的采样 80/80/90：DDPM 是 250 NFE；FM Euler 含两次边界预测，是 252 NFE；原 DDPM DPM-Solver-2 是 494 NFE。FM midpoint 的 40/40/45 也是 252 NFE。</p>
<p><a href="naming/results.csv" download>下载全部正式结果 CSV</a> · <a href="naming/configs.csv" download>下载全部已扫描配置 CSV</a> · <a href="naming/results.json">原始行、来源与逐网格完整配置</a></p>
<h2>116 条历史正式结果</h2><p>指标未改动，也未重新跑 FID。仅配置存在的任务不混进这张完成结果表。边界按当前 fresh / noise_filling 定义记录。</p>
<input id="filter" type="search" placeholder="筛选模型、步数、边界或记录编号…" aria-label="筛选实验"><span id="count">116 条</span>
<div class="scroll"><table><thead><tr><th>ID</th><th>模型</th><th>训练更新步数</th><th>训练时段长度</th><th>采样步数</th><th>求解器</th><th>NFE</th><th>CFG</th><th>边界</th><th>FID</th></tr></thead><tbody id="records">'''+rows+'''</tbody></table></div>
<h2>另一台服务器最近回报的 FM 对照</h2><p>两次均为 FM-XL v2、400K EMA、Euler、8×8=80 / 16×16=80 / 32×32=90、252 NFE、CFG1、50K。fresh 的 FID 为 33.2233，noise_filling 的 FID 为 14.5032。它们不是步数分配不同。本次只整理用户提供的回报，未在本机重新测量；<a href="naming/remote_fm_reports.json">配置、五项指标与来源路径</a>。</p>
<h2>模型旧名 → 正确的训练配置</h2><div class="scroll"><table><thead><tr><th>历史标签</th><th>当前明确名称</th><th>训练时间区间</th></tr></thead><tbody>'''+modelrows+'''</tbody></table></div>
<div class="note">原始日志、旧 Git/HF revision 和 checkpoint 内部元信息保留原样，避免破坏证据；当前索引和入口使用新名字。归档下载链接仍可访问。未核实的历史配置不会靠文件名猜测。HF 迁移计划与真实执行回执分开，不能把计划当作已经改名。</div>
</main><script>const input=document.querySelector('#filter');input.oninput=()=>{let n=0;for(const row of document.querySelectorAll('#records tr')){row.hidden=!row.textContent.toLowerCase().includes(input.value.toLowerCase());if(!row.hidden)n++;}document.querySelector('#count').textContent=n+' 条';};</script></html>'''
(dist/'naming.html').write_text(page)
for path in dist.glob('*.html'):
    if path.name=='naming.html':continue
    s=path.read_text()
    if 'href="naming.html"' not in s:
        banner='<div style="padding:10px 24px;background:#e9eef4;color:#192938;font:14px system-ui"><a href="naming.html">实验命名已核对：按 8×8 / 16×16 / 32×32 查看训练与采样分配 →</a></div>'
        s=s.replace('<body>', '<body>'+banner,1) if '<body>' in s else s.replace('</header>', '</header>'+banner,1)
    path.write_text(s)

if plan.get('status')=='applied_and_hash_verified':
    # Link current names at the verified rename revision, retaining the old
    # immutable URL as provenance. Never mutate mirrored source documents.
    base='https://huggingface.co/'+plan['repo_id']
    urls={base+'/'+op+'/'+plan['source_revision']+'/'+quote(old,safe='/'):
          base+'/'+op+'/'+plan['applied_revision']+'/'+quote(new,safe='/')
          for old,new in path_map.items() for op in ('blob','resolve')}
    def relink(x):
        if isinstance(x,dict):
            for k,v in list(x.items()):
                if k in ('raw_record','original_ledger_row') or k.startswith('legacy_'):continue
                if isinstance(v,str) and v in urls:
                    x.setdefault('legacy_'+k,v);x[k]=urls[v]
                else:relink(v)
        elif isinstance(x,list):
            for y in x:relink(y)
    files=[dist/'lab/index.json',dist/'experiments/catalog.json']+list((dist/'lab/models').glob('*.json'))+list((dist/'lab/runs').glob('*.json'))
    for path in files:
        data=read(path);relink(data);write(path,data)
    (dist/'experiments/catalog.js').write_text('window.RDD_EXPERIMENTS='+json.dumps(read(dist/'experiments/catalog.json'),ensure_ascii=False,separators=(',',':'))+';\n')
print(json.dumps({'formal_rows_updated':len(results),'models_named':len(models),'metric_values_changed':0,'raw_source_mirrors_changed':0}))
