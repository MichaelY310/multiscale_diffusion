#!/usr/bin/env python3
"""Build a fail-closed report from the fixed-source CPU suite, never from labels."""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_scoped_validation import snapshot,jobs
from audit_snapshot_compatibility import compatibility


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=ROOT/'evidence/unified_scope_20261012');p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args();base=a.evidence.resolve()
    expected={j['name'] for j in jobs(base,True)};current=snapshot();latest={};running=[]
    for attempt in sorted((base/'suite_attempts').iterdir()):
        if not (attempt/'SUMMARY.json').exists():continue
        recorded=read(attempt/'source_snapshot.json')
        summary=read(attempt/'SUMMARY.json')
        if summary['status']=='running' and (recorded==current or any(compatibility(base,recorded,current,n) for n in expected)):
            running.append(str(attempt.relative_to(base)))
        for row in summary['jobs']:
            match=compatibility(base,recorded,current,row['name'])
            if match:latest[row['name']]={**row,'attempt':str(attempt.relative_to(base)),'snapshot_compatibility':match}
    missing=sorted(expected-latest.keys());failed=[n for n in expected&latest.keys() if latest[n]['status']!='passed']
    if (missing or failed or running) and not a.allow_incomplete:raise SystemExit('Incomplete acceptance: '+repr(dict(missing=missing,failed=failed,running=running)))
    contracts=read(ROOT/'SUPPORTED_REPRODUCTIONS.json');pins=read(ROOT/'evidence/sources.json');methods=[]
    variants={'relay':['profile_sampling_relay_low','profile_sampling_relay_high'],'ssd':['profile_sampling_ssd','ssd_branches','native_ssd'],
        'pixelflow':['profile_sampling_pixelflow'],'dfm':['profile_sampling_dfm','profile_sampling_dfm_yaml'],
        'lapflow':['profile_sampling_lapflow_paper','fix_lapflow','extension_lapflow3'],'ppflow':['sampling_ppflow','fix_ppflow','native_ppflow2','native_ppflow3'],
        'dctdiff':['profile_sampling_dctdiff','profile_sampling_dctdiff_imagenet'],'mdm':['sampling_mdm','extension_mdm_vdm'],
        'dimr':['profile_sampling_dimr'],'rdd':['profile_sampling_rdd','rdd_optional'],
        'diffusability':['codec_full','codec_lifecycle'],'rmd':['rmd_full','rmd_lifecycle'],
        'speed':['inference_speed','inference_pixelgen','inference_wan','workflow_speed_pixelgen','workflow_speed_wan'],
        'nami':['inference_nami'],'mrflow':['inference_mrflow','inference_qwen','workflow_mrflow_qwen'],
        'cat':['cat','cat_full','cat_api']}
    primary_totals=defaultdict(lambda:dict(pairs=0,byte_exact=0,max_abs=0.,nonzero_native=0,nonzero_unified=0))
    for contract in contracts['methods']:
        m=contract['id'];required=variants[m][:]
        if contract['kind'] in ('generation','codec','distillation'):
            required+=['long_'+m]
            # PixelFlow extensions are sampling/packing/API checks; RMD's
            # warmup/transitions live in rmd_full. Neither has an extension_ job.
            if m not in ('rmd','pixelflow'):required+=['extension_'+m]
            if m not in ('rmd','diffusability'):required+=['full_current','reduced_current','lifecycle']
        pending=[n for n in required if n not in latest or latest[n]['status']!='passed']
        if m=='pixelflow':
            extra=base/'sampling_dopri5/pixelflow.json'
            if not extra.exists() or read(extra).get('status')!='passed':pending.append('Dopri5 sampling')
            elif read(extra).get('source_snapshot')!=current:pending.append('stale Dopri5 source snapshot')
        if m=='rdd':
            extra=base/'rdd_distributed_sampler.json'
            if not extra.exists() or read(extra).get('status')!='passed':pending.append('RDD collective sampler')
            elif read(extra).get('source_snapshot')!=current:pending.append('stale RDD collective sampler snapshot')
        proof=[];counts=defaultdict(lambda:dict(pairs=0,byte_exact=0,max_abs=0.,nonzero_native=0,nonzero_unified=0))
        for profile in contract['profiles']:
            path=base/'full_current'/(profile+'.json')
            if path.exists():
                report=read(path)
                if report['status']!='passed':pending.append(profile)
                act=report.get('activation',{})
                for k,v in act.get('totals',{}).items():
                    counts[k]['pairs']+=v['pairs'];counts[k]['byte_exact']+=v['byte_exact'];counts[k]['max_abs']=max(counts[k]['max_abs'],v['max_abs'])
                    primary_totals[k]['pairs']+=v['pairs'];primary_totals[k]['byte_exact']+=v['byte_exact'];primary_totals[k]['max_abs']=max(primary_totals[k]['max_abs'],v['max_abs'])
                    for key in ('nonzero_native','nonzero_unified'):
                        counts[k][key]+=v[key];primary_totals[k][key]+=v[key]
                proof.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),scope=report.get('scope'),deviations=report.get('deviations')))
            elif contract['kind']=='generation':
                pending.append('missing full image report: '+profile)
        for f,h in contract['source_file_sha256'].items():
            if sha(ROOT/f)!=h:pending.append('source changed: '+f)
        contract={k:v for k,v in contract.items() if k!='claim'}
        methods.append(dict(**contract,status='passed' if not pending else 'pending',pending=pending,
            claim='Selected supported contract passed; exclusions retained' if not pending else 'Pending fixed-source acceptance',
            native_source_commit=pins.get(m,{}).get('commit'),proof_jobs=required,full_image_module_tensor_pair_counts=dict(counts),full_image_reports=proof,
            forward='byte_exact_on_tested_contract' if not pending else 'pending',
            backward=('not_applicable_no_released_training_objective' if contract['kind']=='inference' else 'diagnostic_adjoint_only_not_author_training' if contract['kind']=='codec_component' else 'byte_exact_on_tested_contract') if not pending else 'pending'))
    supplements={}
    for name in ('config_audit','author_reader_contracts','comparator_additional_negative_controls','checkpoint_state_bytes','postflight_source_integrity','trace_report_readback','sampling_dopri5/pixelflow','rdd_distributed_sampler'):
        path=base/(name+'.json')
        supplements[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),report=read(path)) if path.exists() else dict(report=dict(status='missing'))
    supplemental_failures=[name for name,entry in supplements.items() if entry['report'].get('status')!='passed']
    complete=not missing and not failed and not running and not supplemental_failures and all(m['status']=='passed' for m in methods)
    report=dict(status='passed' if complete else 'incomplete',accepted_scope_complete=complete,methods=methods,
        job_count=len(expected),jobs=latest,missing_jobs=missing,failed_jobs=failed,running_attempts=running,source_snapshot=current,
        supplemental_checks=supplements,supplemental_failures=supplemental_failures,
        full_image_module_tensor_pair_totals=dict(primary_totals),
        claim='Selected public core algorithms and declared compatibility configurations only; module-boundary CPU parity, not every possible input or unpublished branch.',
        paper_best_results_reproduced=0,all_original_clis_certified=False,gpu_used=False,formal_training=False,
        source_integrity=read(base/'source_integrity.json') if (base/'source_integrity.json').exists() else None,
        cleanup=read(base/'cleanup.json') if (base/'cleanup.json').exists() else None,
        harness_repair=read(base/'harness_repair.json') if (base/'harness_repair.json').exists() else None,
        report_finalization_repair=read(base/'report_finalization_repair.json') if (base/'report_finalization_repair.json').exists() else None,
        environment=read(base/'environment.json') if (base/'environment.json').exists() else None)
    (base/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 保留原版核心的 Unified：逐仓库验收','',
        '**当前验收：'+('全部所选任务通过。' if complete else '尚未完成，不得据本表宣称全部通过。')+'** 本表只认证明确支持的公开算法配置。最佳论文质量没有通过正式训练复现。','',
        '统一点、兼容选项和可重跑命令见 [支持约定](../../SUPPORTED_REPRODUCTIONS_ZH.md)。机器结果见 [AUDIT.json](AUDIT.json)。','',
        f'固定被测实现的任务 {len(expected)} 项；31份完整 image 配置各执行两次独立 forward/backward（这一项不做 optimizer 更新），另列完整 codec 和30层 Wan 骨干。长训练对照使用缩小网络：12方法各128步，12组扩展各128步。CPU、同一非零随机初始化和合成输入；每项实际网格、精度、调用范围见原始报告。','',
        '| 方法 | 支持内容 | forward / backward | 有意未覆盖或不同的部分 | 对照证据 |','|---|---|---|---|---|']
    for row in methods:
        status='待完成' if row['status']!='passed' else '逐字节一致 / N/A（仅推理）' if row['kind']=='inference' else '逐字节一致 / 仅诊断梯度' if row['kind']=='codec_component' else '逐字节一致 / 逐字节一致'
        m=row['id'];links=[]
        paths=[]
        if row['kind'] in ('generation','codec','distillation'):
            paths.append(('128步与逐层梯度',f'long_release/{m}.json'))
            if m!='rmd':paths.append(('扩展分支',f'extensions_current/{m}/{m}.json'))
            if row['full_image_reports']:paths.append(('完整模型',str(Path(row['full_image_reports'][0]['path']).relative_to(base.relative_to(ROOT)))))
        elif m=='speed':paths=[('FLUX','inference_layers/speed_inference.json'),('PixelGen','inference_new/pixelgen.json'),('Wan','inference_new/wan.json')]
        elif m=='nami':paths=[('逐层推理','inference_layers/nami_inference.json')]
        elif m=='mrflow':paths=[('FLUX','inference_layers/mrflow_inference.json'),('Qwen','inference_new/qwen.json')]
        elif m=='cat':paths=[('codec','cat_codec/cat_codec.json'),('完整512','cat_codec_full/cat_codec.json')]
        if m=='rdd':paths.extend([('27组目标与恢复','rdd_optional/SUMMARY.json'),('采样器Gloo','rdd_distributed_sampler.json')])
        if m=='pixelflow':paths.append(('Dopri5像素/RNG','sampling_dopri5/pixelflow.json'))
        if m=='ppflow':paths.append(('作者main','native_ppflow/ppflow2_native_main.json'))
        if m=='rmd':paths.append(('完整Wan骨干','rmd/rmd_full_backbone.json'))
        if m=='diffusability':paths.append(('完整codec','codec_current/diffusability.json'))
        links=['['+label+']('+path+')' for label,path in paths if (base/path).exists()]
        lines.append('| '+row['name']+' | '+row['core_preserved_zh']+' | '+status+' | '+row['intentional_scope_limits_zh']+' | '+'、'.join(links)+' |')
    lines+=['','## 差异的性质','',
        '- 所选基础方法出现非零数值差异，不能以“统一需要”自动豁免；必须修复或在对应方法列保留未完成。',
        '- RDD decoupled 加非默认权重时，原辅助函数漏 drift loss；精确兼容开关和保留 drift 的有意修正版分别测试。基础 RDD 不受影响。',
        '- LapFlow square 原实参错误与 PPFlow no-sigma 原几何错误，均保留原失败/缺陷证据，最小修补后对照；默认公开主配置不被悄悄改写。',
        '- LapFlow 两个分解/重构恒等式存在 FP32 舍入，不属于两套实现的 native 训练/采样失配。',
        '- CPU 注意力/依赖适配、缩小的长期测试、合成资产及共同工程是有意验收设置，不能外推 GPU kernel、真实 pretrained 质量或所有论文变体。',
        '- DiMR 的 reader CFG 与主进程 CFG 是明确的可选放置位置，不同时启用。作者 reader 逐条对照、null label 与随机状态另有补充验收。',
        '- 未发布代码、GAN/私有课程等不在选定支持范围；必要 teacher/codec 不作为 trivial trick 删除。若需要这些分支，必须另立相应合同。','',
        '本轮未完成的所选任务：'+('无。' if complete else repr(dict(missing=missing,failed=failed,running=running,supplemental_failures=supplemental_failures))),
        '',
        '测试修复追踪：'+('初轮75/76通过；独立旧回归脚本按已失效的行号截取reader，产生NameError。仅将该脚本改成按AST命名结构定位，随后重跑该项及pytest。生产实现、配置及其余测试代码未变；各任务保留实际源码快照与兼容判定，未重写旧失败或将其改成passed。见[修复记录](harness_repair.json)。' if report['harness_repair'] else '本目录没有登记独立测试脚本的修补。'),
        '',
        '汇总工具另移除了不存在的 `extension_pixelflow` 任务依赖；实际 PixelFlow 证据是完整配置、128步、Euler与Dopri5对照。之前的incomplete报告保留，见[汇总修复记录](report_finalization_repair.json)。' if report['report_finalization_repair'] else '',
        '',
        '补充验收：[159项配置字段与88项源码固定记录](config_audit.json)、[作者数据reader](author_reader_contracts.json)、[比较器负对照](comparator_additional_negative_controls.json)、[完整checkpoint状态回读](checkpoint_state_bytes.json)、[结束时源码核验](postflight_source_integrity.json)、[逐模块统计独立复核](trace_report_readback.json)。负对照会识别复数虚部差异和正负零的字节差异；不会把 allclose 等同于逐字节相同。',
        '',
        '临时产物：'+(f"已清理{report['cleanup']['deleted_files']}个文件、{report['cleanup']['bytes']/1024**3:.3f}GiB；报告、日志、配置和哈希保留，见[清理清单](cleanup.json)。" if report['cleanup'] and report['cleanup'].get('applied') else '尚未执行本轮最终清理。'),
        '',
        '仅数学映射/无可认证完整作者实现：'+ '、'.join(contracts['excluded_math_only'])+'。CAT只认证已发布codec。早先Pyramid Flow/PyramidalWan FastVideo工作独立验收。','']
    (base/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=report['status'],methods=len(methods),jobs=len(expected),missing=missing,failed=failed),ensure_ascii=False))
    return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())
