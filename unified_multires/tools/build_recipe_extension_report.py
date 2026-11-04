#!/usr/bin/env python3
"""Close historical findings without overwriting their counterexample evidence."""
import json,hashlib,datetime,html
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/recipe_extensions'
CASES=['dctdiff','dimr','ppflow','mdm','mdm_vdm','ssd','dfm','lapflow','lapflow3','relay','rdd','diffusability']
CLOSURES={
'DCT-01':('resolved','EBFR 原版归一化 zigzag 频率权重；真实缩小 UViT 128 步梯度与模块激活对照。',['dctdiff.json'],''),
'DCT-02':('partial','本地系数数组的 percentile bound/entropy 统计；原版 RGB block-DCT upsampling；ODE/SDE/DPM 后端。',['auxiliary.json'],'论文 DiT 分支在所固定 release 中未公开；新数据必须重算统计并选择正确数据预处理。'),
'DIMR-01':('resolved','Python RNG 的 null-label dropout；constant/cosine-decay CFG。',['dimr.json'],''),
'DIMR-02':('resolved','512 分支 std normalization/scaling_factor。',['dimr.json'],'已验证该数学/采样分支，不是 512px 正式训练结果。'),
'PPF-01':('resolved','stage-wise CFG、第一组三通道、时间分段 bisect 约定。',['ppflow.json'],''),
'PPF-02':('resolved','本地 SiT→PPFlow 初始化的 repeat/divide 转换，2/3stage 128 步。',['ppflow.json'],'测试输入权重为本地合成 SiT 形状状态，尚无真实预训练资产质量对照。'),
'PPF-03':('deferred_source_unreleased','保持 2/3stage 发布范围。',[],'未公开的4/5stage、任意分段、FLUX不假冒作者复现。'),
'MDM-01':('resolved','mixed_ratio 计算裁剪、padding mask 与有效样本逆比例重加权。',['mdm.json'],''),
'MDM-02':('partial','严格本地 inner 初始化；negative-first CFG；clip/dynamic/dynamic-if；VDM 权重与 normalized reduction。',['mdm.json','mdm_vdm.json','auxiliary.json'],'完整1024三级预训练课程、在线 T5/CC12M reader 与其生产调度仍未认证。'),
'SSD-01':('resolved','同分辨率多时间采样、resolution-transition 单时间、importance/min-SNR 权重。',['ssd.json'],''),
'SSD-02':('resolved','训练与采样 rescale_timesteps 统一并拒绝不一致配置。',['ssd.json'],''),
'RELAY-01':('resolved','低分辨率 EDM→nearest resize→高分辨率 blur，双模型工作流。',['relay.json'],''),
'RELAY-02':('resolved','p_mean/p_std/min_scale 等时钟参数与采样契约校验。',['relay.json'],''),
'DFM-01':('resolved','逐 band 时间窗口 CFG；含普通与 REPA 两条 128 步分支。',['dfm.json'],''),
'DFM-02':('partial','保持 released architecture，RNG 与 REPA projector 分支实际验证。',['dfm.json'],'作者自己说明 released architecture 不同论文；所下载三尺度 scheduler 的 dt 检查限制保留，不编造未公开网络。'),
'LAP-01':('resolved','两/三层第一组三通道 CFG，三层双 batch 随机抽样和左结合重建。',['lapflow.json','lapflow3.json'],''),
'LAP-02':('partial','两层 variance-aware、noise factor、linear/cosine/square 训练；三层 linear MoT 128 步。',['lapflow.json','lapflow3.json'],'原版 square sampler 参数顺序有 bug：显式拒绝该推理组合，未把自行修正伪装原版。独立专家骨干组合未全部认证。'),
'DIFF-01':('partial','随机2x/4x SE、实际 VGG pyramid、本地 pretrained VGG 注入、freeze、DCT cut、KL schedule/anneal、R1/high-frequency、MSE/MAE/pseudo-Huber。',['diffusability.json','auxiliary.json'],'GAN/temporal/tokenizer/distillation-AE/其它 perceptual family 及完整独立 R1 optimizer phase 留在外部任务；codec→latent→下游生成器仍需真实训练。'),
'RMD-01':('partial','Wan released 两个 transition、两个 optimizer、precision 与2-microbatch accumulation。',['regression_rmd_final/rmd.json','runtime.json'],'作者归属未由论文官方链接确认；GAN/其他骨干、GPU BF16 与完整视频质量均未认证。'),
'RDD-01':('resolved','mapped model time 与 posterior/VB index 分离。',['rdd.json'],''),
'RDD-02':('resolved','多 loss 训练程序、DiTMultiRes、fresh/isometric、CFG、source ddpm/jump、DPM1/2、DDIM/ancestral 分支。',['rdd.json','auxiliary.json'],'训练测试两层；正式三层配方只有结构检查。各分辨率独立多网络的全部旧实验组合未逐一认证。'),
'PF-01':('partial','设备专用 generator 与 native_rng scope；保留 packed PixelFlow 回归。',['auxiliary.json','regression/pixelflow.json'],'CUDA 已过授权期；没有进行 GPU RNG/FlashAttention kernel 认证。当前 CPU 使用 SDPA。'),
'COMMON-01':('partial','本地初始化、optimizer/LR/EMA/AMP/accumulation、image-clock/epoch-clock、crop/flip/moments、tensor shards、checkpoint data binding 和本地 image VAE decode；28个模板。',['runtime.json','auxiliary.json','recipe_schema.json','cli_validation.json'],'真实数据/初始化资产、预处理/统计身份、有效global batch、完整发布最佳recipe与官方质量评价仍需验证；不是所有分支的多卡训练实现。'),
'COMMON-02':('resolved','Native batch 自己拥有目标/path；非法 silent overwrite、clock/head/variance 不匹配直接拒绝；通用 component 组合保持 controlled transplant 身份。',['auxiliary.json'],'数学可表达不意味着作者 native 组件任意互换。')}

def read(path):return json.loads(path.read_text())
def main():
    reports={name:read(OUT/(name+'.json')) for name in CASES}
    if any(r['status']!='passed' or r['steps']!=128 for r in reports.values()):raise ValueError('Required 128-step extension case not passed')
    auxiliary=read(OUT/'auxiliary.json');runtime=read(OUT/'runtime.json');schema=read(OUT/'recipe_schema.json');cli=read(OUT/'cli_validation.json');entrypoint=read(OUT/'entrypoint.json')
    if any(r['status']!='passed' for r in (auxiliary,runtime,schema,entrypoint)) or any(r['status']!='passed' for r in cli['cases']):raise ValueError('Runtime/schema/CLI check failed')
    source_manifest=read(ROOT/'evidence/sources.json');changed=[];checked=0
    for repo,entry in source_manifest.items():
        for file,expected in entry.get('selected_files',{}).items():
            checked+=1;path=ROOT/'sources'/repo/file
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:changed.append(str(path))
    if changed:raise ValueError('Author source changed: '+str(changed))
    totals={};variants=[]
    for method,report in reports.items():
        for audit in report['activation']:
            if audit['structural_mismatches']:raise ValueError('Activation structural mismatch')
            variants.append({'method':method,'name':audit['name'],'steps':audit['steps'],'executed_module_boundary_scope':True,'all_tensor_or_aten_operations':False})
            for kind,value in audit['totals'].items():
                row=totals.setdefault(kind,{'pairs':0,'exact':0,'byte_exact':0,'within_tolerance':0,'max_abs':0.})
                for key in ('pairs','exact','byte_exact','within_tolerance'):row[key]+=value[key]
                row['max_abs']=max(row['max_abs'],value['max_abs'])
                if value['pairs']!=value['byte_exact']:raise ValueError('New branch is not byte exact')
    historical=read(ROOT/'evidence/compatibility_audit/AUDIT.json');findings=[]
    for old in historical['findings']:
        status,change,files,remaining=CLOSURES[old['id']]
        findings.append({'id':old['id'],'historical_title':old['title'],'status':status,'implemented':change,'evidence':files,'remaining':remaining,'paper_quality_verified':False})
    regressions={p.stem:read(p) for p in (OUT/'regression').glob('*.json') if p.name!='run_status.json'}
    if len(regressions)!=12 or any(r['status']!='passed' for r in regressions.values()):raise ValueError('Original 12-method regression failed')
    rmd_final=read(OUT/'regression_rmd_final/rmd.json')
    if rmd_final['status']!='passed':raise ValueError('Final RMD regression failed')
    index=read(ROOT/'configs/paper_recipes/INDEX.json')
    codefiles=list((ROOT/'umdiff').glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'configs/paper_recipes').glob('*.json'))
    snapshot={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(codefiles)}
    summary={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed_for_declared_component_and_recipe_tests','user_best_paper_training_results_achieved':False,'paper_quality_verified':False,'gpu_used':False,'model_or_dataset_downloaded':False,'dependency_added':'torch-dct==0.1.6 code-only wheel in project _deps; no shared environment modification','extension_cases':len(reports),'extension_training_variants':len(variants),'training_steps_per_variant':128,'variants':variants,'executed_shared_module_totals':totals,'checks':sum(len(r['checks']) for r in reports.values()),'runtime_checks':len(runtime['checks']),'auxiliary_checks':len(auxiliary['checks']),'original_regression_methods':len(regressions),'original_regression_variants':sum(len(r['variants']) for r in regressions.values()),'rmd_final_regression_variants':len(rmd_final['variants']),'recipe_templates':len(index),'recipe_schema_check':'Full architecture construction on meta device; local pretrained models deferred. No real-asset training certificate.','recipe_claim_counts':dict(Counter(r['claim'] for r in index)),'cli_resume_cases':[r['name'] for r in cli['cases']],'source_integrity':{'repositories':len(source_manifest),'files':checked,'changed':changed},'finding_status_counts':dict(Counter(f['status'] for f in findings)),'findings':findings,'umdiff_tools_recipe_sha256':snapshot,'remaining_before_best_quality':['Select exact released dataset/split/preprocessing/statistics and official evaluator assets.','Bind original local pretrained VAE/SiT/VGG/T5/Wan/vision initialization and verify actual weight loading.','Identify best-run recipe where release supplies only defaults; match effective global batch, update budget, teacher, precision and EMA.','Run paired small real-data training and guided decoded inference before a formal run.','Obtain a new GPU authorization; the Oct8 LA09:00 window has expired.','Run full training and official FID/IS/CLIP or video metrics with repeated seeds; no such result obtained this turn.'],'scope':'Reduced author networks, synthetic inputs, selected enabled branches and shared native backbones. Exact executed module boundaries do not certify every ATen operation, full production kernels, arbitrary controlled mixes, or generated-image quality.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 2026-10-09 配方补齐与最终兼容性复核','', '主要公开算法分支已补齐并验证；**尚未复现各篇论文最佳训练指标**。这里只认证下面实际执行的 CPU 测试，不把同一网络类复用、小合成训练或模板构造视为真实数据质量。','',f'新增 {len(variants)} 个训练变体各128步；旧12类21变体回归通过。新增分支实际执行模块的输入/输出、activation梯度、参数梯度均逐字节相等；范围不含未执行分支或每个 functional/ATen 操作。训练 scalar loss 有少量归约舍入差，不影响已测参数梯度和更新。', '',f'额外 runtime {len(runtime["checks"])} 项、auxiliary {len(auxiliary["checks"])} 项通过；28个正式/组件模板构造通过（meta检查不是训练）；CLI完整/恢复/采样对照：'+', '.join(summary['cli_resume_cases'])+'。','',f'作者源文件固定17仓库、{checked}份hash全部未变。原审计的反例保留为历史证据，已被修复的缺项以本报告为准。','', '|检查种类|tensor对数|逐字节相等|最大差|','|---|---:|---:|---:|']
    for kind,value in totals.items():lines.append(f'|{kind}|{value["pairs"]}|{value["byte_exact"]}|{value["max_abs"]}|')
    lines+=['','## 逐项关闭历史发现','','resolved表示具体公开分支已实现和测试；partial表示代码已补但仍有明确未验证/未公开范围。所有项的论文质量认证仍为false。','','|ID|状态|已补内容|保留范围|','|---|---|---|---|']
    for f in findings:
        evidence=' '.join(f'[{p}]({p})' for p in f['evidence']);lines.append(f'|{f["id"]}|{f["status"]}|{f["implemented"]} {evidence}|{f["remaining"] or "见共同真实数据质量限定"}|')
    lines+=['','## 新一轮发现及修正','','- Diffusability：原版拼接 decoder chunks 得到 contiguous 输出。不同内存布局的 VGG backward 累加顺序导致梯度差，随后 Adam 放大。端口 decoder输出 contiguous 后，128步的内部激活和参数梯度恢复逐字节相等。VGG使用合成权重，只测算子正确性。','- LapFlow square：训练导数的乘法结合顺序需要与作者一致；已对齐。原版 square sampling 另有实质参数顺序 bug，继续拒绝。','- LapFlow 三层 CFG：原版为2B抽取两个低分辨率噪声，虽然模型只使用前B，后B仍消耗 RNG；原版重建采用 `(r0+up(r1))+up(up(r2))`。修复随机抽样和FP32结合顺序后，128步训练、guided终态与RNG均相等。','- MDM：正式嵌套 resnet_config 必须构造作者 dataclass；完整CC12M结构meta检查和非默认配置forward已通过。可选VDM权重/归一化128步对照通过，CC12M发布默认仍是关闭VDM。','- 数据/训练：原版全局RNG配合本地VAE采样现在可用；梯度累积返回完整batch诊断；配置精度字段不得被忽略；metadata数据身份、batch和seed变化会拒绝恢复。大数据支持单shard常驻和epoch permutation缓存。','- Relay：增加按已见图像数计时的LR rampup与EMA halflife、source.lerp的算术和loss multiplier。128步控制目标对照及64步恢复通过；这不是该论文正式训练。','', '## 统一框架如何容纳这些分支','','path/target/metric/representation/boundary仍是算法组件；batch负责多尺度布局与有效样本；guidance独立于training target；training_stages组合多阶段损失；初始化、数据、optimizer、precision、EMA是可检验recipe。原网络与solver作为明确披露的native backend，接口不声称其内部实现通用互换。','', '## 正式复现入口','','参见 [正式配方使用说明](../../configs/paper_recipes/README_ZH.md)、[配方索引](../../configs/paper_recipes/INDEX.json)、[完整机器证据](SUMMARY.json)。23份released配置模板、1份local RDD模板、4份最佳运行未定位的component模板。部分steps/batch为null或epoch描述，需要先按真实数据冻结；不能一概视为完整最佳配方。','', '## 达到用户质量目标仍需的工作','']
    lines += ['- '+item for item in summary['remaining_before_best_quality']]
    lines+=['','没有下载模型或数据，没有GPU进程，没有修改共享dirty FastVideo；torch-dct小型代码依赖安装在本项目_deps。CPU合成checkpoint在runs/recipe_extensions，不可当作生成质量模型。历史尝试/失败均保留在attempts；最终通过的JSON不掩盖残余问题。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    table=''.join('<tr><td>'+html.escape(f['id'])+'</td><td>'+html.escape(f['status'])+'</td><td>'+html.escape(f['implemented'])+'</td><td>'+html.escape(f['remaining'])+'</td></tr>' for f in findings)
    (OUT/'REPORT_ZH.html').write_text('<!doctype html><meta charset="utf-8"><title>配方补齐与兼容性复核</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}table{border-collapse:collapse}a{color:#0369a1}</style><h1>配方补齐与兼容性复核</h1><p>CPU组件验证通过，尚无论文最佳训练指标。详见<a href="REPORT_ZH.md">完整说明</a>及<a href="SUMMARY.json">机器证据</a>。</p><p>'+html.escape(str(summary['finding_status_counts']))+'</p><table><tr><th>ID</th><th>状态</th><th>已实现</th><th>仍需处理</th></tr>'+table+'</table>')
    for file in (ROOT/'PAPER_COVERAGE.json',ROOT/'evidence/FINAL_SUMMARY.json'):
        value=read(file);value['recipe_extensions_20261009']={'report':'evidence/recipe_extensions/REPORT_ZH.md','summary':'evidence/recipe_extensions/SUMMARY.json','full_paper_quality_verified':False,'validated_training_variants':len(variants),'original_tests_retained_as_historical':True}
        if isinstance(value.get('methods'),list):
            for row in value['methods']:
                if row['id'] in set(CASES)|{'pixelflow','rmd'}:row['recipe_extension_evidence']='evidence/recipe_extensions/SUMMARY.json';row['paper_quality_verified']=False
        file.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    limits=read(ROOT/'evidence/KNOWN_NATIVE_LIMITATIONS.json');limits.setdefault('previous_all_methods_scope',limits['all_methods']);limits['all_methods']='Latest extension evidence: recipe_extensions/SUMMARY.json. CPU synthetic reduced author networks and selected branches; full-scale data/GPU/FID quality not certified. Native recipes now support additional optimizer/LR/EMA/precision/data rules.'
    limits['LapFlow']='2/3-level MoT linear train x128 and guided inference checked; 2-level cosine/square train x128, variance/noise optional branches checked. Broken source square sampler explicitly deferred.'
    limits['Diffusability']='Random SE2/4 and actual VGG features, DCTcut, freeze, schedules and additional primitives checked. Synthetic VGG weights, reduced image codec only; full pretrained quality/temporal/GAN pipelines unverified.'
    limits['checkpoint_scope']='CLI binds path/file metadata, effective batch size and traversal seed. This is not a full dataset content hash; external frozen manifest required. Legacy checkpoints without binding cannot resume through bound CLI; direct historical parity checkpoints remain readable.'
    limits['recipe_extensions_20261009']='recipe_extensions/SUMMARY.json';(ROOT/'evidence/KNOWN_NATIVE_LIMITATIONS.json').write_text(json.dumps(limits,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('extension_training_variants','checks','finding_status_counts','source_integrity')},ensure_ascii=False))
if __name__=='__main__':main()
