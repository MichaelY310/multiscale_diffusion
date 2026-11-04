#!/usr/bin/env python3
"""Publish current, scoped evidence without replacing historical reports."""
import datetime,hashlib,html,json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from compare_released_profiles import PROFILES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/current_recipe_parity_20261009'
ROWS=[
 ('Relay Diffusion',['relay_low','relay_high'],'README ImageNet64→256，192/256宽度','完整双骨干；global4096/8-rank DDP、xformers、FP16与真实数据未执行'),
 ('Scale Space Diffusion',['ssd'],'发布 CelebA64/FlexiUNet/ConvexDecay0.5','source未给完整最佳训练预算；真实CelebA/FID未执行'),
 ('PixelFlow',['pixelflow'],'发布 XL/256，patch4/head72，4 stages','30步/阶段、CFG2.4是公开eval默认；CUDA FlashAttention/BF16未认证'),
 ('DFM',['dfm','dfm_yaml'],'公开 SiT-XL 与 B YAML/lognormal','release明确不同论文架构；真实REPA teacher/原论文最佳配置未认证'),
 ('LapFlow',['lapflow','lapflow_paper'],'公开 MoT-XL；同时测CLI及论文ImageNet训练flags','论文T0.5/LR1e-4/EMA0.9999/EQVAE；最佳CFG和EQVAE资产未认证'),
 ('PPFlow',['ppflow','ppflow3'],'发布 XL-2/XL-3，Heun2/250points','真实SiT初始化缺失；stage CFG为eval默认/示例；未公开4/5stage/FLUX不覆盖'),
 ('DCTdiff',['dctdiff','dctdiff_imagenet'],'公开 U-ViT mid-deep，CIFAR32/ImageNet64','公开U-ViT最大配置；未公开DiT、JPEG统计/原FID资产未覆盖'),
 ('MDM',['mdm','mdm1024'],'CC12M256/1024完整2/3级NestedUNet','1024权重16:4:1/power2；真实inner256预训练/T5/reader/curriculum未认证'),
 ('DiMR',['dimr'],'公开 DiMR-G-2R/ImageNet256','原VAE moments、FP16/DDP与完整FID未执行'),
 ('Improving the Diffusability of Autoencoders',[],'随机SE2/4+原VGG loss/冻结分支128步','best训练说明与checkpoint未公开；仅组件对照，不把自定SE配方称best'),
 ('RMD',[],'发布Wan1.3B配置flags及完整HF骨干','完整骨干仅小视频网格/FP32；21×60×104→21×90×160、BF16/累积/DDP未合并认证；repo作者归属未确认'),
 ('RDD',['rdd'],'本地统一DiT-XL，32→16→8/300+300+400','本地native实现；非某外部论文best；真实VAE/数据/FID未执行'),
]

def read(path):return json.loads(path.read_text())
def assert_exact(audit):
    if audit.get('structural_mismatches'):raise ValueError('Module call structure differs')
    for value in audit['totals'].values():
        if value['pairs']!=value['byte_exact'] or value['max_abs']!=0:raise ValueError('Not byte exact')
def add_totals(total,audit):
    assert_exact(audit)
    for kind,value in audit['totals'].items():
        row=total.setdefault(kind,dict(pairs=0,byte_exact=0,max_abs=0.))
        row['pairs']+=value['pairs'];row['byte_exact']+=value['byte_exact'];row['max_abs']=max(row['max_abs'],value['max_abs'])

def main():
    reports={name:read(OUT/(name+'.json')) for name in PROFILES}
    for report in reports.values():
        if report['status']!='byte_exact' or report['steps']!=2:raise ValueError('Full-network short run missing or failed')
    full_rmd=read(OUT/'components/rmd_full_backbone.json');rmd_flags=read(OUT/'components/rmd_released_flags.json');diff=read(OUT/'components/diffusability.json')
    if full_rmd['status']!='passed' or rmd_flags['status']!='passed' or diff['status']!='passed':raise ValueError('Required component/full-RMD run failed')
    config_audit=read(OUT/'config_audit.json');schema=read(OUT/'recipe_schema.json')
    if config_audit['status']!='passed' or schema['status']!='passed':raise ValueError('Configuration check failed')
    samplers={}
    for dir in ['sampling','sampling_profiles']:
        for path in sorted((OUT/dir).glob('*.json')):
            report=read(path)
            if path.name=='status.json':continue
            if report['status']!='passed' or any(not c['passed'] or not c['bitwise'] for c in report['checks']):raise ValueError('Sampling test is not exact: '+str(path))
            samplers[str(path.relative_to(OUT))]=dict(status=report['status'],checks=len(report['checks']),scope=report['scope'])
    regressions=[]
    for dir in ['regression','regression_reduction','regression_pixelflow']:
        for path in sorted((OUT/dir).glob('*.json')):
            report=read(path)
            if path.name=='status.json':continue
            if report['status']!='passed':raise ValueError('Regression failed')
            if isinstance(report.get('activation'),list):
                for audit in report['activation']:assert_exact(audit)
            else:assert_exact(report)
            regressions.append(dict(file=str(path.relative_to(OUT)),method=report['method'],steps=report['steps']))
    totals={}
    for report in reports.values():add_totals(totals,report['activation'])
    for audit in full_rmd['activation']:add_totals(totals,audit)
    for audit in diff['activation']:assert_exact(audit)
    for audit in rmd_flags['activation']:assert_exact(audit)
    continuous=read(OUT/'continuous_rng/dfm.json')
    if continuous['status']!='passed' or {(v['weighting'],v.get('repa',False)) for v in continuous['variants']}!={('uniform',False),('lognormal',False),('uniform',True)}:
        raise ValueError('DFM continuous RNG run missing or failed')
    for variant in continuous['variants']:
        if (variant['steps']!=128 or variant['per_step_reseeding'] or variant['author_logging_guards_patched']
            or not variant['continuous_torch_numpy_rng_byte_exact'] or variant['loss_scalar_max_abs']!=0
            or variant['optimizer']['tensors']!=variant['optimizer']['exact'] or variant['optimizer']['max_abs']!=0):
            raise ValueError('DFM continuous run is not exact')
        assert_exact(variant['activation'])
    manifest=read(ROOT/'evidence/sources.json');checked=0;changed=[]
    for repo,entry in manifest.items():
        for file,expected in entry.get('selected_files',{}).items():
            checked+=1;path=ROOT/'sources'/repo/file
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:changed.append(str(path))
    if changed:raise ValueError('Pinned source changed')
    profile_rows=[]
    for title,names,selection,limits in ROWS:
        data=[reports[name] for name in names]
        if title=='RMD':data=[full_rmd]
        params=[r.get('parameters') or r['details']['variants'][0]['parameters'] for r in data]
        profile_rows.append(dict(paper=title,profiles=names or ['components/rmd_full_backbone','components/rmd_released_flags'] if title=='RMD' else names or ['components/diffusability'],selection=selection,
            parameters=params,full_backbone_tested=bool(data),full_input_resolution_tested=bool(names),training_steps=2 if data else 128,
            executed_shared_module_activation_byte_exact=True,parameter_gradient_byte_exact=True,
            author_best_recipe_fully_executed=False,paper_quality_verified=False,limits=limits))
    diffs=[]
    for report in reports.values():
        current=read(ROOT/report['template']);before=report['config']
        diffs.append(dict(profile=report['profile'],template_sha_matches_current=report['template_sha256']==hashlib.sha256((ROOT/report['template']).read_bytes()).hexdigest(),
           model_matches_current=before['model']==current['model'],batch_matches_current=before.get('batch')==current.get('batch'),
           note='Snapshot is the tested configuration. Later MDM sampling-only fields are covered separately; effective CPU precision/smaller batch are recorded explicitly.'))
    if not all(d['model_matches_current'] and d['batch_matches_current'] for d in diffs):raise ValueError('Model/batch changed after full run')
    loss_max=max(abs(row['native_loss']-row['unified_loss']) for report in reports.values() for row in report['trajectory'])
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['umdiff','tools','tests','configs/paper_recipes'] for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    findings=deepcopy(read(ROOT/'evidence/recipe_extensions/SUMMARY.json')['findings'])
    for finding in findings:
        if finding['id']=='RDD-02':
            finding['implemented']+=' 本轮正式三层DiT-XL完整网络2步与正式DPM2/CFG采样策略对照通过。'
            finding['remaining']='各分辨率独立多网络的全部旧实验组合未逐一认证；真实资产/完整训练质量未认证。'
        elif finding['id']=='MDM-02':
            finding['implemented']+=' 本轮完整1024三级NestedUNet2步，以及原版250步resampling/power2/CFG采样对照通过。'
            finding['remaining']='真实inner256预训练资产、完整预训练课程、在线T5/CC12M reader与其生产调度仍未认证。'
        elif finding['id']=='DFM-02':
            finding['implemented']+=' native_debug_rng显式兼容原版logging的NumPy draw；uniform/lognormal及REPA各连续128步，无逐步reseeding、无修改作者logging。'
    summary=dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed_for_declared_current_conformance_scopes',
        every_requested_paper_reviewed=True,all_best_recipes_fully_covered=False,user_best_paper_training_results_achieved=False,
        paper_quality_verified=False,gpu_used=False,model_or_dataset_downloaded=False,
        full_image_network_profiles=len(reports),full_video_backbone_profiles=1,full_profiles_training_steps=2,
        full_network_module_totals=totals,full_image_loss_scalar_max_abs=loss_max,rows=profile_rows,
        current_sampling_tests=samplers,reduced_regressions=regressions,configuration_checks=len(config_audit['checks']),
        recipe_templates=len(schema['recipes']),recipe_claim_counts=config_audit['recipe_claim_counts'],source_integrity=dict(repositories=len(manifest),files=checked,changed=changed),
        tested_config_snapshot_consistency=diffs,code_sha256=code,
        residual_findings=findings,
        continuous_rng=dict(file='continuous_rng/dfm.json',status=continuous['status'],variants=[{k:v[k] for k in ('weighting','repa','teacher_features','steps','per_step_reseeding','author_logging_guards_patched','continuous_torch_numpy_rng_byte_exact','loss_scalar_max_abs','optimizer')} for v in continuous['variants']]),
        restrictions=['Executed module boundaries, not every functional/ATen intermediate or unexecuted branch.',
            'Source module classes and several exact solver backends reused explicitly; not independent backbone reimplementation.',
            'Random constructor weights/synthetic inputs; no original pretrained asset, real data training or official FID/VBench rerun.',
            'CPU FP32 architecture comparison; GPU FP16/BF16/FlashAttention/xformers/DDP not certified.',
            'Native optimizer is independently constructed from source-audited hyperparameters; LR/EMA arithmetic is paired, not a full original distributed CLI.',
            'PixelFlow source sampler returns a NumPy array and unified returns a Torch tensor. Same shape/dtype values and bytes are compared; container API types differ.',
            'Diffusability facade interfaces differ; shared internal VAE modules are compared. Best-run instructions/checkpoints remain unavailable.'])
    md=['# 2026-10-09 修改后逐篇配置、完整网络与采样复核','',
        '**已经逐篇重测，但没有“全部补齐”或“最佳论文指标已复现”的结论。** 原24项中的8项partial及1项未公开范围仍存在；本轮另发现并修正了正式配置/采样细节。',
        '',f'共{len(reports)}个完整图像网络配置与1个完整Wan-1.3B骨干配置，各2次配对更新。图像测试保留正式架构/输入网格；Wan骨干保留全部30层/特征维度，但视频网格缩小。全CPU、随机初始化、合成输入，没有下载模型/数据。',
        '',f'实际执行模块的activation/gradient和权重/优化器/EMA（适用时）逐字节相等。图像测试loss标量最大差：{loss_max}。缩小非零作者网络的128步回归和独立公开sampling策略对照另列，不能把后者称完整规模预训练推理。',
        '', '|论文|所选公开/论文配置|完整骨干参数量|修改后训练对照|未认证范围|','|---|---|---:|---|---|']
    for row in profile_rows:
        params=' / '.join(f'{p/1e6:.1f}M' for p in row['parameters']) or '仅缩小组件'
        status='完整骨干2步：activation/gradient相等' if row['full_backbone_tested'] else '组件128步：内部activation/gradient相等'
        md.append(f"|{row['paper']}|{row['selection']}|{params}|{status}|{row['limits']}|")
    md+=['','逐篇JSON及完整snapshot见本目录；总入口 [SUMMARY.json](SUMMARY.json)，独立配置审计 [config_audit.json](config_audit.json)。所有作者源文件固定commit/hash，未改写作者仓库。','',
        '|完整网络检查种类|tensor对数|逐字节相等|最大差|','|---|---:|---:|---:|']
    for kind,value in totals.items():md.append(f"|{kind}|{value['pairs']}|{value['byte_exact']}|{value['max_abs']}|")
    md+=['','## 本轮具体修复','',
        '- PixelFlow正式XL packing从错误patch2/head16改为patch4/head72，并拒绝网络与packing配置不一致；原版每图整块MSE归约取代先channel后token归约。',
        '- DFM发布B YAML的lognormal时间分布，原版跨非batch维度MSE归约，generation30+10与threshold0.1/CFG1.5；XL公开默认40+10。',
        '- DFM原版概率logging会消耗NumPy随机数并影响后续尺度选择。正式native配方显式启用native_debug_rng，REPA保留该设置，作者源码未改写；uniform/lognormal及REPA各连续128步，不逐步重设seed，模块activation/gradient、loss、优化器和Torch/NumPy RNG全部相等。REPA teacher为合成本地feature。见 [continuous_rng/dfm.json](continuous_rng/dfm.json)。',
        '- PPFlow接原版ODE/SDE后端，正式ODE Heun2/250保存点、stage CFG；SDE CLI的sigma diffusion与私有RNG隔离。250点Heun2是249间隔、498次固定求解NFE。原版SBDM/t0=0奇点显式拒绝，不伪装成原版有效输出。',
        '- MDM保留1000训练gamma，按原版(N+1)/(K+1)取整选250推理步、跨步后验、CFG7.5、eta0、输出裁剪/重标定；补1024三级架构、weights16:4:1、SNR power2及严格本地inner256初始化。',
        '- Relay改为README的完整低/高网络、低阶段loss scaling25、EMA50M图像halflife、全局图像时钟、nan_to_num梯度策略；没有把单进程图像时钟称8卡DDP复现。',
        '- LapFlow新增论文ImageNet配方T0.5/LR1e-4/EMA0.9999/EQVAE，与CLI默认T0.9/LR5e-4/EMA0.9分别测试。',
        '- DCT数据严格要求作者预先resize尺寸；正式模板标明源训练精度。MDM本地asset名字含数字时的render bug、tracer单元素stride字节视图也已修复。',
        '- RMD额外检查原版15步warmup和非整数倍非方形transition；完整Wan架构2步、两transition缩小网络128步都通过。',
        '', '## 采样结果对照','',
        '公开sampler步数/CFG保留，骨干宽度/深度缩小并启用非零输出头。DCT/DiMR/PixelFlow/DFM/LapFlow/SSD/RDD保留各正式输入网格；Relay用8/16网格并披露原source alpha的尺寸适配；PPFlow与MDM网格缩小。原版模块/solver复用均在工具中明确标注。latent输出不假称使用原VAE解码。PixelFlow作者返回NumPy、unified返回Torch；比较相同shape/dtype的数值和原始字节，容器API类型不同。',
        '', '|报告|检查数|结果|','|---|---:|---|']
    for file,r in samplers.items():md.append(f"|[{file}]({file})|{r['checks']}|逐字节相等|")
    md+=['','RMD另有训练后完整latent轨迹与**小随机Wan VAE**像素对照；Diffusability训练后codec及SE/DCT分支对照见 [components](components/)。它们都不认证预训练画质。sampling_profiles/*.pt保存native/unified终态tensor，DCT另保存uint8输出；可以直接检查数值。','',
         '## 还不能成立的结论','',
         '1. “已经复现每篇最好训练结果”：没有原真实数据、准确预训练资产/teacher、原版完整训练预算和官方FID/视频评估。没有任何论文的此项认证。',
         '2. “所有默认都是best”：PixelFlow/PPFlow/MDM采样default/example不等于最佳FID设置；Diffusability作者尚未发布完整训练说明和checkpoint；DFM公开实现明确不同论文架构。',
         '3. “所有细节都补齐”：PPFlow未公开stage/FLUX、DCT未公开DiT、LapFlow原版square采样bug/独立专家全部组合、MDM reader/预训练课程、Diffusability temporal/GAN等、RMD GAN/归属、GPU/DDP及真实data/asset链仍未关闭。',
         '4. “任意组件组合继承native一致”：controlled transplant需要自己的测试；本轮数值结果只对应JSON记载的native配置和实际执行分支。',
         '', '## 来源、检查与复跑','',
         f"配置审计{len(config_audit['checks'])}项通过；{len(schema['recipes'])}模板meta构造通过（本地资产明确deferred，不能代替训练）。{checked}个源文件hash与{len(manifest)}仓库commit保持固定；pytest结果见 [pytest.log](pytest.log)。历史失败及归约前snapshot保存在attempts/，没有覆盖旧10月8/9日的报告。",
         '', '```bash','source unified_multires/activate_cpu.sh',
         '"$UMDIFF_PYTHON" unified_multires/tools/audit_current_profile_configs.py',
         '"$UMDIFF_PYTHON" unified_multires/tools/compare_released_profiles.py pixelflow --steps 2',
         '"$UMDIFF_PYTHON" unified_multires/tools/validate_profile_inference.py pixelflow',
         '"$UMDIFF_PYTHON" unified_multires/tools/validate_current_sampling.py mdm',
         '"$UMDIFF_PYTHON" unified_multires/tools/validate_dfm_current_rng.py',
         '"$UMDIFF_PYTHON" unified_multires/tools/validate_rmd_current_profile.py --full-backbone --steps 2',
         '"$UMDIFF_PYTHON" unified_multires/tools/build_current_profile_report.py','```','',
         'GPU旧窗口已过期，本轮仅CPU；没有修改共享dirty FastVideo，也没有恢复旧监控。',
         '', '论文配方依据：[LapFlow AppendixG](https://arxiv.org/html/2602.19461)、[DCTdiff公开U-ViT实验](https://arxiv.org/html/2412.15032)；源码缺口：[Diffusability官方仓库](https://github.com/snap-research/diffusability)、DFM固定release README。']
    markdown='\n'.join(md)+'\n';(OUT/'REPORT_ZH.md').write_text(markdown)
    page='<!doctype html><meta charset="utf-8"><title>修改后逐篇复核</title><style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:16px/1.6 sans-serif;color:#17233c}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #cbd3df;vertical-align:top}th{background:#e7eef9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:20px}a{color:#165fbc}</style><h1>修改后逐篇配置与数值复核</h1><p>17份完整骨干短程对照；不认证论文最佳训练指标。<a href="SUMMARY.json">完整JSON</a> · <a href="REPORT_ZH.md">Markdown报告</a></p><table><tr><th>论文</th><th>选用配置</th><th>结果</th><th>边界</th></tr>'
    for row in profile_rows:page+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [row['paper'],row['selection'],'完整骨干2步逐字节一致' if row['full_backbone_tested'] else '仅组件128步一致',row['limits']])+'</tr>'
    page+='</table><pre>'+html.escape(markdown)+'</pre>';(OUT/'REPORT_ZH.html').write_text(page)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','full_image_network_profiles','full_video_backbone_profiles','configuration_checks','recipe_templates','source_integrity','full_image_loss_scalar_max_abs']},ensure_ascii=False))
if __name__=='__main__':main()
