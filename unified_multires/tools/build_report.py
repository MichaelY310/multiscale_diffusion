#!/usr/bin/env python3
"""Build offline paper/component coverage and an evidence-derived HTML report."""
import datetime,html,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def esc(value):return html.escape(str(value))

NOTES={
 'relay':('Actual EDMPrecond/DhariwalUNet; heat DCT, rolling-window noise, weighted clean loss, stochastic second-order sampler.',
          'get_alpha_t repeat_dims=64 changed to H/patch for the small geometry; persistence is an identity import shim. Native FFT DCT arithmetic retained. Two-model relay quality/training budget not reproduced.'),
 'ssd':('Actual FlexiUNet; cumulative antialiased bilinear M, true adjoint, previous-resolution clean target, anisotropic posterior, 40-step Lanczos and 20-step power clamp.',
        'Native BilinearOp.MT missing H_out/W_out are assigned in the oracle fixture. Single-process rank/barrier are CPU shims. Mean and covariance arithmetic intentionally differ as in source.'),
 'speed':('Full released FLUX generation and local random VAE; DCT/Haar/FFT expansion, bf16, decoded pixels.',
          'Inference-only release. Author FlowMatch scheduler backend explicitly reused. NumPy fp32 scaling precedes destination cast; native non-Hermitian FFT real-part behavior and RNG order retained. No unpublished fine-tuning recipe claimed.'),
 'pixelflow':('Actual packed DiT; heterogeneous resolutions, per-image loss, endpoint path, bilinear D/nearest U, correlated renoising, separate local/global/model clocks and CFG 0/4.',
              'Both paths replace GPU FlashAttention with CPU variable-length SDPA. Diffusers 0.32 NumPy RoPE helper restored under 0.40. No claim of GPU kernel bitwise equivalence. gamma=-1/3 and 1e-6 covariance jitter retained.'),
 'dfm':('Actual released SiT with product Laplacian state, independent band noise, time/drop masks, active unnormalized velocity losses; REPA projector with local synthetic teacher features.',
        'Release README states its architecture differs from the paper. Two-level complete sampler passes. Released three-level [3,3,3]/threshold [.6,.3] scheduler rejects positive dt=.14; preserved as expected error. qk_norm=False fixes a newer timm constructor API. Native debug logging consumes NumPy RNG; comparisons explicitly couple stage/time/noise. For continuous global-seed alignment, disable the two random debug-log conditions in loss.py:202/233 or move logging to a separate RNG.'),
 'lapflow':('Actual released MoT (feb12_train_ours / ours_sample), bilinear D/nearest U Laplacian, delayed band paths; 2/3-level components; two-level full 128-update native training and sampling.',
            'Three-level preset has 4-update CLI/resume/sample coverage, not 128-update complete native-loop coverage. Two-level noise decomposes one full epsilon; default three-level uses full epsilon plus independent coarse leaves. Source last-loop head behavior retained. CPU replaces hardcoded time.cuda(); same torchdiffeq backend reused, including unused coarse tail integration.'),
 'ppflow':('Actual PPFlow DiT, 2/3 stages, invertible channel-to-space and width permutation, velocity loss and entire Euler trajectories.',
           'Balanced batch must divide by 2/4. Native strict rejection mode can undersupply; expected error retained. learn_sigma=False has a source unpatchify channel bug; source-default True retained, variance output ignored by FM. num_steps counts endpoints, hence n-1 updates.'),
 'nami':('Full author-vendored FLUX pipeline, progressive depth, learned nearest affine bridge, native schedule, CFG 1/3; 16-channel 32/64/128 latent geometry and 1024-square decoded pixels.',
         'Inference-only release. Old Transformers FLAX_WEIGHTS_NAME constant restored for import. Vendored pipeline/transformer hashes equal src copies. Native ignores supplied latents and draws full-resolution noise before downsampling; port exposes this full initial noise. No bridge-training objective published/recreated.'),
 'mrflow':('Actual author main with local random FLUX/VAE and actual RRDB architecture from exact RealESRGAN gitlink: generate→decode→PIL uint8→SR→resize→posterior encode→12+1-step sigma .12 refine.',
           'Only asset loaders supply local modules; deprecated cached_download import removed in compatibility namespace. Source tile=192,padding=24,symmetric pad=15,batch=4 and uint8 truncation retained; multi-tile non-square batch=3 tested. Random narrow RRDB is not pretrained SR quality.'),
 'dctdiff':('Actual U-ViT, YCbCr/chroma subsampling/block DCT/zigzag/eta codec, VP objective, native DPM-Solver and decoded uint8 pixels.',
            'DPM-Solver backend explicitly reused (dpm_solver_pytorch, not a substitute DPM-Solver++). Decoder retains coefficient clipping [-2,2], zero omitted frequencies, double IDCT and uint8 truncation.'),
 'diffusability':('Actual AutoencoderKL wrapper, stochastic latent/KL, pixel MSE and scale-equivariance coefficient .3 with mean/bilinear downsampling; trained decoder parity.',
                 'Codec auxiliary objective, not a new progressive denoising sampler. LPIPS/GAN/DCT-cut/temporal optional branches are excluded; no unpublished full recipe claimed. Mean/logvar contiguous layout matches native BTCHW random-number assignment. Native B×1 reconstruction+B-vector KL broadcasting differs in reduction order; gradients/weights agree.'),
 'mdm':('Actual NestedUNet, full/coarse states, independent Gaussian draws, DEEPFLOYD SNR shift, native v→epsilon loss and full 16-step posterior trajectories.',
        'Unused mlx data-reader/torchinfo/S3 imports shimmed on CPU. Source low-noise pooled tensor is overwritten by normal_(), so independent noise is preserved. Legacy residual branch has undefined ratio; default no-residual mode tested. Training index t=1..N maps to model t-1.'),
 'dimr':('Actual MRModel Transformer/ConvNeXt/TDRMSN, full VP input, normalized-convolution multi-resolution epsilon targets, weights [.25,1], native DPM final latent.',
         'Native DPM backend reused explicitly. Train network clock n/1000*8-4 differs from infer T*N and is preserved. No VAE asset downloaded, so final state is latent, not decoded paper image.'),
 'rmd':('Actual WanTransformer3DModel with released Predictor, loss helpers and phase1/phase2 bodies; flow modes 0/1/2 × eta 0/.5/1, two transition modes trained 128 updates; random Wan VAE decoded pixels.',
        'Repository self-identifies official and metadata matches paper author names, but paper page does not link it; author ownership unconfirmed. Full original train() separately executes 2 CPU steps with actual Accelerator/DataLoader/two optimizers/checkpoint export. Synthetic embeddings/local random loaders and small LATENT_SIZES replace assets/geometry. Optional GAN branch excluded. Width-preserving normalization and phase1 higher-shift behavior retained.'),
 'rdd':('Migrated actual DiT/GaussianDiffusion loss: drift off/on, fixed/learned-range variance, joint/decoupled heads, 6 variants ×128 updates, complete DDPM latent outputs.',
        'Native effective epsilon uses sqrt(alpha_bar/(1-alpha_bar)); learned variance VLB keeps mean detached. Fresh default lift differs from historical dense posterior. Historical factor-two posterior mean/cov/sample independently compared bitwise. No ImageNet quality comparison in this fixture.'),
}

SKIPS={
 'fdm':'Author project Code & Data remains coming soon; no executable official implementation confirmed.',
 'edify':'No executable official implementation of this technical report confirmed.',
 'pddpm':'No executable official implementation confirmed.',
 'ddit':'No executable official implementation confirmed.',
 'cat':'No executable official implementation confirmed.',
 'lot':'2026-10-05 paper/project; project has no executable code link at audit.',
 'fresco':'Downloaded official repository has README/LICENSE only; executable release pending.',
 'avits':'Downloaded official repository is README only; executable release pending.',
}

def summarize(path):
    document=read(path);checks=document.get('checks',[])
    groups={}
    for label,fragment in [('loss','loss'),('gradient','gradient'),('updated_parameters','optimizer'),('ema','ema'),('sampling','inference')]:
        selected=[c for c in checks if fragment in c.get('scope','')]
        groups[label]=dict(count=len(selected),bitwise=sum(c.get('bitwise',False) for c in selected),max_abs=max((c['max_abs'] for c in selected),default=None))
    samples=[dict(name=c['name'],scope=c['scope'],**c['artifact']) for c in checks if 'artifact' in c]
    return dict(file=str(path.relative_to(ROOT)),status=document['status'],checks=len(checks),passed=sum(c.get('passed',False) for c in checks),bitwise=sum(c.get('bitwise',False) for c in checks),max_abs=max((c['max_abs'] for c in checks),default=0),groups=groups,sample_artifacts=samples,seconds=document.get('seconds'),details=document.get('details',{}),source_executions=document.get('source_executions',[]))

def compact_details(value):
    if isinstance(value,dict):return {k:compact_details(v) for k,v in value.items() if k not in ('losses','checks')}
    if isinstance(value,list):return [compact_details(v) for v in value]
    return value

def checkpoint_dirs(value):
    if isinstance(value,str) and value.startswith('runs/native_parity/'):return [value]
    if isinstance(value,dict):return [p for v in value.values() for p in checkpoint_dirs(v)]
    if isinstance(value,list):return [p for v in value for p in checkpoint_dirs(v)]
    return []

def main():
    audited=datetime.datetime.now(datetime.timezone.utc).isoformat();sources=read(EVIDENCE/'sources.json')
    activation_path=EVIDENCE/'activation_parity/SUMMARY.json'
    activation=read(activation_path) if activation_path.exists() else None
    compatibility_path=EVIDENCE/'compatibility_audit/AUDIT.json'
    compatibility=read(compatibility_path) if compatibility_path.exists() else None
    extensions_path=EVIDENCE/'recipe_extensions/SUMMARY.json'
    extensions=read(extensions_path) if extensions_path.exists() else None
    literature=read(ROOT.parent/'RDD_UNIFIED_COMPONENT_NOVELTY_AUDIT_20261008.json')
    review=read(EVIDENCE/'PREIMPLEMENTATION_REVIEW.json')
    papers=literature['methods']
    extras=[
      dict(id='mdm',name='Matryoshka Diffusion Models',year='ICLR 2024',paper='https://arxiv.org/abs/2310.15111',components=['Nested full/coarse product state','Per-level VP path','Independent noise; shifted SNR','Joint multiresolution posterior','Joint training/sampling, not only a boundary resize','v parameterization converted to epsilon loss','NestedUNet with shared cross-scale computation','Discrete posterior sampler']),
      dict(id='dimr',name='DiMR: Multi-Resolution Diffusion Models',year='2024',paper='https://arxiv.org/abs/2406.09416',components=['Full latent VP state','Ordinary VP input','IID full input; targets downsampled with normalized convolution','No progressive-state boundary in tested release','Multi-resolution network/targets at each step','Weighted multi-resolution epsilon losses','MRModel Transformer/ConvNeXt/TDRMSN','Native DPM-Solver']),
      dict(id='lot',name='LoT: Level-of-Token Diffusion',year='2026-10-05',paper='https://arxiv.org/abs/2610.05816',components=['Full-resolution image, adaptive rectangular token layout','Asymmetric flow matching','Paper base noise/path','Layout split/merge with reconstruction map','Learned adaptive rectangular granularity','Full-resolution velocity reconstruction','Variable-area patch token model','Reduced tokens; include layout overhead']),
    ]
    papers=papers+extras
    alias={'spd':'speed','se_ae':'diffusability'}
    rows=[]
    for paper in papers:
        key=alias.get(paper['id'],paper['id']);paper=dict(paper,id=key)
        status='source_native_small_fixture' if key in NOTES else 'paper_mapping_native_skipped'
        configs=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs').rglob('*.json')) if p.stem==key or p.stem.startswith(key+'_')]
        # Original audit is literature evidence, not an execution status. New review supersedes it.
        paper.pop('code_status',None);paper.pop('group',None)
        components=paper.pop('components')
        mapping=dict(zip(['E_representation','A_signal_path','C_noise_marginal','K_coupling_boundary','R_time_scale_allocation','T_W_target_loss','H_backbone','S_solver_cost'],components))
        if key=='relay':mapping['C_noise_marginal']='IID plus overlapping cyclic rolling-window noise; not block-constant repetition.'
        if key=='lapflow':mapping['C_noise_marginal']='2-level: one full IID epsilon Laplacian-decomposed. Default 3-level: full epsilon plus independent coarse epsilon leaves. Native branches differ.'
        if key=='ppflow':mapping['E_representation']='Full degrees of freedom retained by invertible channel↔space and width packing; source layout, not lossy image downsampling.'
        row=dict(**paper,status=status,source_provenance='self_declared_official_not_independently_linked' if key=='rmd' else ('author_repository' if key in sources else 'no_executable_official_source_confirmed'),literature_component_map=mapping,configs=configs,paper_quality_verified=False)
        if key in NOTES:
            row.update(native_scope=NOTES[key][0],compatibility_and_limits=NOTES[key][1])
        else:row.update(skip_reason=SKIPS[key],implementation='Literature component mapping / extension interfaces only, not implemented full paper algorithm.')
        if key in sources:row['source']={k:sources[key][k] for k in ('repo','commit','fetched_at_utc')}
        if key in review.get('public_repositories',{}):row['preimplementation_review']=review['public_repositories'][key]
        if key=='rmd':row['preimplementation_review']=read(EVIDENCE/'RMD_PREIMPLEMENTATION_REVIEW.json')
        row['configured_components']=[dict(file=p,configuration=read(ROOT/p)) for p in configs]
        rows.append(row)
    rows.append(dict(id='rdd',name='RDD (migrated local code)',year='local project',paper=None,status='source_native_small_fixture',native_scope=NOTES['rdd'][0],compatibility_and_limits=NOTES['rdd'][1],configs=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs/native_fixtures').glob('rdd*.json'))],paper_quality_verified=False,source_provenance='migrated_local_repository'))
    axes=[dict(id=k,description=v) for k,v in [
      ('E','codec, color, quantization, normalization and product-state order'),('Pi','token/patch layout, position, masks, invertible versus lossy packing'),('F/A','stage signal map and time derivative'),('B/C','noise transform, covariance, scale-dependent SNR'),('K','cross-time/scale joint random coupling and RNG order'),('phi','local, global, model and solver time conversion'),('rho','time/stage distribution and active/drop masks'),('H','backbone, heads, parameter sharing and conditioning'),('T','prediction parameterization, variance and inverse decoder'),('W','metric, per-image/token reduction and auxiliary objectives'),('Q','boundary stochastic kernel, resized state/noise/cache and codec roundtrip'),('S','ODE/SDE/discrete solver and actual model/CFG calls'),('O','initialization, multiple networks, optimizers, LR, clipping, EMA, precision, checkpoint'),('D','data, preprocessing, latent scaling, teacher/decoder assets')]]
    coverage=dict(audited_at_utc=audited,paper_count=22,executable_paper_releases=14,established_author_repositories=13,self_declared_source_with_identity_limit=1,native_skipped_papers=8,additional_local_method='RDD',axes=axes,math='FRAMEWORK_MATH_ZH.md',methods=rows)
    if activation and activation['status']=='passed':
        by_method={r['id']:r for r in activation['rows']}
        for row in rows:
            if row['id'] in by_method:
                r=by_method[row['id']]
                row['executed_backbone_module_activation_audit']=dict(evidence=r['evidence'],steps=128,variants=r['variants'],totals=r['totals'],exclusions=r['exclusions'],scope=activation['definitions']['scope'])
    if compatibility:
        coverage['compatibility_detail_audit']='evidence/compatibility_audit/AUDIT.json'
        by_method={r['id']:r for r in compatibility['methods']}
        for row in rows:
            if row['id'] in by_method:
                r=by_method[row['id']]
                row['full_paper_recipe_coverage']=False
                row['compatibility_findings']=r['findings']
                row['compatibility_assessment']=r['assessment']
    if extensions:
        coverage['recipe_extensions_20261009']={'report':'evidence/recipe_extensions/REPORT_ZH.md','summary':'evidence/recipe_extensions/SUMMARY.json','full_paper_quality_verified':False,'validated_training_variants':extensions['extension_training_variants'],'original_tests_retained_as_historical':True}
        for row in rows:
            if row['id'] in {r['method'] for r in extensions['variants']}|{'pixelflow','rmd'}:row['recipe_extension_evidence']='evidence/recipe_extensions/SUMMARY.json'
    # This generator primarily summarizes historical small-fixture reports.
    # Preserve the later executed CPU audit and partial CAT release discovery.
    previous_path=ROOT/'PAPER_COVERAGE.json'
    previous=read(previous_path) if previous_path.exists() else {}
    cpu_audit_path=EVIDENCE/'forward_backward_20261012/AUDIT.json'
    if cpu_audit_path.exists() and read(cpu_audit_path).get('status')=='passed' and 'cpu_verification_20261012' in previous:
        for field in ('cpu_verification_20261012','native_skipped_papers','partial_published_codec_releases'):
            coverage[field]=previous[field]
        old_cat=next(r for r in previous['methods'] if r['id']=='cat')
        for row in rows:
            if row['id']=='cat':
                for field in ('status','source_provenance','skip_reason','implementation','configured_components','source_release','cpu_evidence','full_cpu_evidence','official_training_loss_verified','verified_components'):
                    row[field]=old_cat[field]
    write(ROOT/'PAPER_COVERAGE.json',coverage)
    reports={}
    for row in rows:
        key=row['id'];reports[key]={}
        for kind in ('conformance','training','inference'):
            path=EVIDENCE/f'{key}_{kind}.json'
            if path.exists():reports[key][kind]=summarize(path)
    all_stats=[v for method in reports.values() for v in method.values()]
    training=[method['training'] for method in reports.values() if 'training' in method]
    sampling=[c['groups']['sampling'] for c in all_stats if c['groups']['sampling']['count']]
    losses=[c['groups']['loss'] for c in training]
    cli=read(EVIDENCE/'cli_validation.json');suite=read(EVIDENCE/'suite.json');offline=read(EVIDENCE/'offline_entrypoints.json')
    summary=dict(audited_at_utc=audited,device='cpu',gpu_used=False,no_model_or_dataset_download=True,paper_count=22,source_repositories=len(sources),executable_papers=14,native_skipped_papers=8,component_methods=sum('conformance' in r for r in reports.values()),training_methods=len(training),inference_only_methods=3,training_steps_per_variant=128,training_comparisons=sum(r['checks'] for r in training),training_bitwise_comparisons=sum(r['bitwise'] for r in training),all_comparisons=sum(r['checks'] for r in all_stats),all_bitwise_comparisons=sum(r['bitwise'] for r in all_stats),failed_comparisons=sum(r['checks']-r['passed'] for r in all_stats),updated_parameter_max_abs=max(r['groups']['updated_parameters']['max_abs'] or 0 for r in training),gradient_max_abs=max(r['groups']['gradient']['max_abs'] or 0 for r in training),ema_max_abs=max(r['groups']['ema']['max_abs'] or 0 for r in training),sampling_comparisons=sum(r['count'] for r in sampling),sampling_bitwise_comparisons=sum(r['bitwise'] for r in sampling),sampling_max_abs=max(r['max_abs'] or 0 for r in sampling),loss_scalar_max_abs=max(r['max_abs'] or 0 for r in losses),cli_presets=len(cli['cases']),cli_passed=sum(c['status']=='passed' for c in cli['cases']),suite_cases=len(suite['cases']),suite_passed=sum(c['status']=='passed' for c in suite['cases']),offline_command_cases=len(offline['cases']),offline_passed=sum(c['exit_code']==0 for c in offline['cases']),paper_quality_verified=False,quality_limit='Random small author architectures / synthetic local inputs. No FID, trained full-scale assets, real dataset quality or GPU-kernel certification.',methods={k:{t:dict(v,details=compact_details(v['details'])) for t,v in r.items()} for k,r in reports.items()})
    summary['sampling_artifact_pairs']=sum(len(s['sample_artifacts']) for s in all_stats)
    summary['supplementary_dfm_continuous_rng']=read(EVIDENCE/'dfm_rng_alignment.json')
    if activation:
        summary['supplementary_activation_parity']=dict(evidence='evidence/activation_parity/SUMMARY.json',status=activation['status'],variants=activation['variants'],steps_per_variant=128,totals=activation['totals'],scope=activation['definitions']['scope'],limitations=activation['limitations'])
    if compatibility:
        summary['supplementary_compatibility_audit']=dict(evidence='evidence/compatibility_audit/AUDIT.json',report='evidence/compatibility_audit/REPORT_ZH.html',findings=len(compatibility['findings']),cpu_counterexamples=compatibility['counterexample_count'],full_paper_recipe_coverage=False,previous_tested_scope_parity_remains_valid=True,conclusion=compatibility['conclusion'])
    if extensions:summary['recipe_extensions_20261009']=coverage['recipe_extensions_20261009']
    write(EVIDENCE/'FINAL_SUMMARY.json',summary)
    parts=[]
    for row in rows:
        key=row['id'];stats=reports[key];train=stats.get('training');infer=stats.get('inference');conf=stats.get('conformance')
        claim='逐元素验证通过' if stats else '按要求跳过 native'
        if key=='rmd':claim+='；来源身份待独立确认'
        count=' / '.join(f'{t}: {s["bitwise"]:,}/{s["checks"]:,} bitwise' for t,s in stats.items())
        anchor=f'paper-{key}';paperlink=f'<a href="{esc(row["paper"])}">论文</a>' if row.get('paper') else '本地 RDD'
        source=row.get('source');repo=f'<a href="{esc(source["repo"])}">源码</a> @ <code>{source["commit"][:12]}</code>' if source else ''
        configlinks=' '.join(f'<a href="{esc(c)}">{esc(Path(c).stem)}</a>' for c in row.get('configs',[]))
        componenthtml=''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k,v in row.get('literature_component_map',{}).items())
        evidencehtml=''.join(f'<li><a href="{esc(s["file"])}">{t} 原始证据 JSON</a>：{s["checks"]:,} 检查；{s["bitwise"]:,} bitwise；最大绝对差 {s["max_abs"]:.3g}</li>' for t,s in stats.items())
        details=''.join(f'<details><summary>{esc(t)}：骨干、优化器、复用 backend、checkpoint 与源码锚点</summary><pre>{esc(json.dumps(compact_details(s["details"]),ensure_ascii=False,indent=2))}</pre><pre>{esc(json.dumps(s["source_executions"],ensure_ascii=False,indent=2))}</pre></details>' for t,s in stats.items())
        scope=row.get('native_scope',row.get('skip_reason',''));limits=row.get('compatibility_and_limits',row.get('implementation',''))
        sample_rows=[a for s in stats.values() for a in s['sample_artifacts']]
        pictures=[a for a in sample_rows if 'native_png' in a]
        gallery=''
        if pictures:
            item=pictures[-1]
            gallery='<p><b>随机小模型 / 合成输入数值样例，非论文质量</b>：'+esc(item['name'])+'</p><div class="gallery">'+''.join(f'<figure><img loading="lazy" src="{esc(item[k+"_png"])}"><figcaption>{label}</figcaption></figure>' for k,label in [('native','原版'),('unified','统一实现'),('difference','绝对差分（未放大）')])+'</div>'
        if sample_rows:
            gallery+='<details><summary>实际采样 tensor 对与全部图像</summary><ul>'+''.join(f'<li>{esc(a["name"])}：<a href="{esc(a["tensor_pair"])}">native/unified .pt</a>'+(' · '+' '.join(f'<a href="{esc(a[k+"_png"])}">{k}</a>' for k in ('native','unified','difference')) if 'native_png' in a else '（保持原始 latent / 轨迹坐标）')+'</li>' for a in sample_rows)+'</ul></details>'
        checkpoints=sorted(set(p for s in stats.values() for p in checkpoint_dirs(s['details'])))
        if checkpoints:gallery+='<details><summary>训练后成对 checkpoint（随机数值 fixture）</summary><ul>'+''.join(f'<li>{esc(p)}：<a href="{esc(p)}/native.pt">native.pt</a> · <a href="{esc(p)}/unified.pt">unified.pt</a></li>' for p in checkpoints)+'</ul></details>'
        parts.append(f'<article id="{anchor}" data-status="{"verified" if stats else "skipped"}"><h3>{esc(row["name"])} <small>{esc(row["year"])}</small></h3><p class="status">{esc(claim)}</p><p>{paperlink} {repo}</p><p>{esc(scope)}</p><p class="limit">{esc(limits)}</p><p>{configlinks}</p><p>{esc(count)}</p>{gallery}<details><summary>统一组件映射（文献）</summary><table>{componenthtml}</table></details><ul>{evidencehtml}</ul>{details}</article>')
    axeshtml=''.join(f'<tr><td>{esc(a["id"])}</td><td>{esc(a["description"])}</td></tr>' for a in axes)
    table=''.join(f'<tr><td><a href="#paper-{r["id"]}">{esc(r["name"])}</a></td><td>{"source-native 小模型" if reports[r["id"]] else "文献映射 / 跳过 native"}</td><td>{"128 steps" if "training" in reports[r["id"]] else ("公开推理 workflow" if "inference" in reports[r["id"]] else "无可执行官方源码")}</td></tr>' for r in rows)
    differences=read(EVIDENCE/'NUMERICAL_DIFFERENCES.json')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unified Multiresolution：逐篇复现报告</title>
<style>body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#18232d}main{max-width:1180px;margin:auto;padding:30px}h1,h2,h3{line-height:1.25}a{color:#1755a2}article,.panel{background:white;padding:22px;margin:20px 0;border-radius:12px;border:1px solid #d9e1e6}small{font-weight:400;color:#596772}table{border-collapse:collapse;width:100%;margin:10px 0}td,th{border-bottom:1px solid #dce3e8;text-align:left;vertical-align:top;padding:9px}code,pre{font-family:ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f9;padding:14px;font-size:13px}summary{cursor:pointer;color:#23578c;margin:10px 0}.status{color:#07604a;font-weight:650}.limit{border-left:4px solid #c18b2a;padding-left:12px}.cards{display:flex;gap:12px;flex-wrap:wrap}.cards>div{background:#eaf2f7;border-radius:10px;padding:15px;min-width:150px}.cards b{display:block;font-size:26px}input,select{padding:9px;font:inherit;max-width:100%;margin:6px}nav{position:sticky;top:0;background:#f3f5f7eF;padding:8px;z-index:1}svg{width:100%;max-width:750px}.gallery{display:flex;flex-wrap:wrap;gap:12px}.gallery figure{margin:0;flex:1;min-width:150px;max-width:360px}.gallery img{width:100%;image-rendering:pixelated;border:1px solid #ddd}button{font:inherit;padding:6px}@media print{nav{display:none}article{break-inside:avoid}}</style>
<main><h1>多分辨率 Diffusion / Flow：统一框架与逐篇数值复现</h1><p>审计时间：@TIME@ · surfi1 · 全部 CPU / offline · 没有下载模型或训练数据</p>
<p><strong>2026-10-09 当前补齐状态：</strong><a href="evidence/recipe_extensions/REPORT_ZH.md">配方补齐与最终复核</a>、<a href="evidence/recipe_extensions/SUMMARY.json">17个新增128步变体的证据</a>、<a href="configs/paper_recipes/README_ZH.md">28份模板用法</a>。下方原配方缺项及旧计数是历史测试范围；真实数据最佳训练指标仍未认证。</p>
<div class="cards"><div><b>22</b>论文 + 本地 RDD</div><div><b>14</b>公开可执行方法</div><div><b>@TRAIN@</b>128 步训练对照类</div><div><b>@CHECKS@</b>逐 tensor 比较</div><div><b>@CLI@</b>CLI 配置通过</div></div>
<div class="panel"><h2>已验证的结论与范围</h2><p>在相同本地随机作者骨干、初始权重、合成输入、条件、时间和 RNG 下，独立统一实现与原仓库算法对照。每类训练含 128 次更新（变体分别执行），再对训练后的完整采样结果作比较。在128步 native 训练对照中，更新参数、EMA 和被测最终采样全部 bitwise 一致。部分 scalar loss 的 reduction 次序差异低于 @LOSS@；最大梯度差 @GRAD@。原始逐参数证据可在每篇卡片下载。</p><p class="limit"><b>这不认证论文 FID 或全规模生成质量。</b>未下载预训练模型、数据、teacher 或评测资产；随机小模型的像素一致性只能证明所测配置的实现一致性。实际训练要匹配完整架构、数据/codec/预处理、teacher、预算和 recipe。八篇未有可执行官方源码的方法按用户要求跳过 native。RMD 仓库自称官方，但论文页未链接它；来源确认有限制。</p><p>17 个 source-only 仓库在实现前完成审读并固定 commit；RealESRGAN 与 RMD 后续扩展也各有实现前记录。骨干类直接使用作者实现，路径/loss/boundary/training orchestration 独立实现；相同 DPM/ODE backend 的复用逐项写明。除 RMD 外，128 步 AdamW/EMA 是两端共用的受控优化器，不冒称每篇论文完整训练 recipe。训练 oracle 执行作者 loss、模块方法或原始训练语句体；各仓库分布式训练 launcher 未纳入本轮测试。RMD 另外完整启动了原版 train() 入口。</p>
<p><a href="README_ZH.md">运行说明</a> · <a href="FRAMEWORK_MATH_ZH.md">完整数学定义</a> · <a href="PAPER_COVERAGE.json">22 篇组件映射 JSON</a> · <a href="evidence/FINAL_SUMMARY.json">机器可读汇总</a> · <a href="evidence/sources.json">源码 pins/hash</a> · <a href="SOURCES_AND_LICENSES.md">源码与许可证</a></p></div>
@COMPATIBILITY@
@ACTIVATION@
<div class="panel"><h2>统一框架</h2><pre>M = (E, Π, {F, B, K, φ, ρ, H, T, W}ₖ, {Qₖ}, S, O, D)
zₖ,ₜ = Fₖ(t,E(x),ξ)；线性高斯子族 zₜ=Aₜx+Bₜε，Cₜ=BₜBₜᵀ
uₜ=∂ₜFₖ；Qₖ 是跨阶段状态/缓存的随机核
相同边缘 ≠ 相同跨尺度联合过程；改变 target 还要匹配 loss metric
joint / decoupled、isometric / fresh / Gaussian posterior、不同频谱/codec 可独立审计</pre><table>@AXES@</table><p>普通 flow/DDPM 与可组合 band 路径由通用引擎支持。精确 native packing/RNG/loss 通过显式 batch/sampler 策略支持；不接受会被忽略的通用覆盖。新算法可注册组件或策略。可学习参数必须归属 model，蒸馏支持显式多模型两优化器。详情与奇点、Jacobian、SNR/时钟推导见数学文档。</p></div>
<div class="panel"><h2>执行覆盖</h2><table><tr><th>方法</th><th>结论范围</th><th>执行类型</th></tr>@TABLE@</table><p>组件 15 组（含 RDD/RMD）；训练 12 类（含 RDD）；SPEED/NAMI/MrFlow 是推理发布，不伪造未发布训练目标。Diffusability 是 codec 辅助训练，不与 progressive sampler 混称。</p><p><a href="evidence/suite.json">总测试 suite</a> · <a href="evidence/cli_validation.json">全部 CLI 连续/恢复测试</a> · <a href="evidence/offline_entrypoints.json">本地 tensor/图片/embedding/权重入口</a> · <a href="evidence/rmd_offline_assets.json">三模型本地 RMD 蒸馏</a> · <a href="evidence/rmd_native_entrypoint.json">RMD 原版 train()</a> · <a href="evidence/dfm_rng_alignment.json">DFM 最小日志修补后连续32步 RNG/AdamW 对齐</a></p></div>
<nav><label>筛选 <input id="search" type="search" placeholder="方法、组件、兼容修改…"></label><select id="filter"><option value="all">全部方法</option><option value="verified">有原版数值对照</option><option value="skipped">跳过 native</option></select><span id="shown"></span></nav>
@PAPERS@
<div class="panel"><h2>数值差异、发现原因与处理</h2><p>不会只用 forward 接近判断复现成功；Relay 的 DCT roundoff 曾被 Adam 放大到权重差 7.8e-5，修复算术顺序后 128 步与最终像素一致。所有发现记录在 <a href="evidence/NUMERICAL_DIFFERENCES.json">差异账本</a>。</p><pre>@DIFF@</pre><p>剩余 native 限制：<a href="evidence/KNOWN_NATIVE_LIMITATIONS.json">完整清单</a>。源文件不改写；CPU/旧依赖兼容修改位于 oracle/adapter namespace。</p></div>
<div class="panel"><h2>直接复跑</h2><pre>cd /scr/yuheng/bugswarm/rdd_workspace_20261005
source unified_multires/activate_cpu.sh
"$UMDIFF_PYTHON" -m umdiff.cli components
"$UMDIFF_PYTHON" -m umdiff.cli train --config unified_multires/configs/native_fixtures/rdd_joint.json --synthetic --steps 20 --batch-size 8 --output unified_multires/runs/my_rdd
"$UMDIFF_PYTHON" -m umdiff.cli sample --checkpoint unified_multires/runs/my_rdd/checkpoint.pt --labels 0 1 --count 2 --output unified_multires/runs/my_rdd/sample.pt
"$UMDIFF_PYTHON" unified_multires/tools/validate_all.py --training-steps 128
"$UMDIFF_PYTHON" unified_multires/tools/validate_cli.py
"$UMDIFF_PYTHON" unified_multires/tools/validate_offline_entrypoints.py
"$UMDIFF_PYTHON" unified_multires/tools/build_report.py</pre><p>示例输出是 latent。真实本地数据、条件、checkpoint、像素方法与扩展插件命令见 README。checkpoint 不替用户核验整个数据集身份；恢复需保持配置/数据/batch size/seed 一致。GPU 未使用；截止后无本任务 GPU 进程。</p><p><a href="evidence/ENVIRONMENT.json">执行环境</a> · <a href="evidence/INTEGRITY_AUDIT.json">完整性审计</a> · <a href="evidence/SAMPLE_ARTIFACT_AUDIT.json">实际输出与证据核对</a> · <a href="evidence/FINAL_PROCESS_AUDIT.json">进程/GPU 审计</a></p></div>
<script>const cards=[...document.querySelectorAll('article')];function update(){const q=document.querySelector('#search').value.toLowerCase(),f=document.querySelector('#filter').value;let n=0;for(const c of cards){c.hidden=!(c.textContent.toLowerCase().includes(q)&&(f==='all'||c.dataset.status===f));if(!c.hidden)n++}document.querySelector('#shown').textContent=n+' / '+cards.length}document.querySelector('#search').addEventListener('input',update);document.querySelector('#filter').addEventListener('change',update);update();</script></main></html>'''
    replacements={'TIME':audited,'TRAIN':len(training),'CHECKS':f'{summary["all_comparisons"]:,}','CLI':summary['cli_passed'],'LOSS':f'{summary["loss_scalar_max_abs"]:.3g}','GRAD':f'{summary["gradient_max_abs"]:.3g}','AXES':axeshtml,'TABLE':table,'PAPERS':'\n'.join(parts),'DIFF':esc(json.dumps(differences,ensure_ascii=False,indent=2))}
    replacements['ACTIVATION']=''
    replacements['COMPATIBILITY']=''
    if compatibility:
        replacements['COMPATIBILITY']='<div class="panel"><h2>完整recipe兼容性补充审计</h2><p><b>已有逐元素相等仅限所测native配置，尚未完整覆盖所有论文配方。</b>逐篇源码复核发现DCTdiff频率loss权重、若干方法CFG、MDM混合batch、Relay双模型链、预训练转换及SSD/RDD非默认network clock等缺口。已用12个CPU反例确认关键差异，包括真实缩小作者UViT梯度与PPFlow两/三层guided采样。生产算法未在本轮修改，缺口保持公开。</p><p><a href="evidence/compatibility_audit/REPORT_ZH.html">24项逐篇兼容性审计与补齐要求</a> · <a href="evidence/compatibility_audit/PROBES.json">反例数值</a> · <a href="evidence/compatibility_audit/AUDIT.json">机器可读代码证据</a></p></div>'
    if activation and activation['status']=='passed':
        replacements['ACTIVATION']='<div class="panel"><h2>逐层 activation 补验</h2><p>11 篇训练论文 + 本地 RDD，21 变体各 128 步。390,544 对模块输入、348,384 对模块输出、227,552 对 activation 梯度和 178,336 对参数梯度均通过数值、shape/dtype 和字节一致性检查，最大差 0。21 对最终 model/EMA/完整优化器状态也与此前无 hooks 的结果一致。</p><p>检查实际执行的共享作者网络模块边界，包括 DiMR backward 重算；未执行分支和模块内部每个 functional 运算未单独认证。Diffusability 检查 VAE 内部模块，原版 BTCHW facade 根接口不作同名匹配；额外无 loss encoder 诊断调用不计入 trace。既有 endpoint 断言保留。</p><p><a href="evidence/activation_parity/REPORT_ZH.html">逐篇逐层报告</a> · <a href="evidence/activation_parity/SUMMARY.json">补验汇总</a> · <a href="evidence/activation_parity/selftest.json">故意差异自测</a></p></div>'
    for key,value in replacements.items():page=page.replace('@'+key+'@',str(value))
    (ROOT/'REPRODUCTION_REPORT_ZH.html').write_text(page)
    print(json.dumps({k:v for k,v in summary.items() if k!='methods'},ensure_ascii=False))
    if any(s['status']!='passed' for s in all_stats) or summary['failed_comparisons'] or summary['cli_passed']!=summary['cli_presets'] or summary['suite_passed']!=summary['suite_cases'] or summary['offline_passed']!=summary['offline_command_cases']:raise SystemExit('Evidence contains failures')
if __name__=='__main__':main()
