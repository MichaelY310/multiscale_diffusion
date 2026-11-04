#!/usr/bin/env python3
"""Aggregate the completed activation audit, including independent checkpoints."""
import datetime
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import torch
from validate_activations import METHODS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_tree(a,b):
    result=dict(tensor_pairs=0,byte_equal_tensor_pairs=0,scalar_pairs=0,differences=[])
    def visit(a,b,path='$'):
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            result['tensor_pairs']+=1
            equal=a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
            result['byte_equal_tensor_pairs']+=int(equal)
            if not equal and len(result['differences'])<20:
                result['differences'].append(path)
        elif isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():result['differences'].append(path+'/keys')
            for k in a.keys() & b.keys():visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):result['differences'].append(path+'/length')
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        else:
            result['scalar_pairs']+=1
            if a!=b and len(result['differences'])<20:result['differences'].append(path)
    visit(a,b)
    result['passed']=not result['differences'] and result['tensor_pairs']==result['byte_equal_tensor_pairs']
    return result


def main():
    directory=ROOT/'evidence/activation_parity'
    reports={m:json.loads((directory/(m+'.json')).read_text()) for m in METHODS}
    if any(r['steps']!=128 or r['status']!='passed' for r in reports.values()):raise SystemExit('Require all twelve 128-step reports to pass')
    selftest=json.loads((directory/'selftest.json').read_text())
    if selftest['status']!='passed':raise SystemExit('Tracer self-test failed')
    audit_path=ROOT/'evidence/FORWARD_BACKWARD_COVERAGE_AUDIT_20261008.json'
    old=json.loads(audit_path.read_text());metadata={r['id']:r for r in old['rows']}
    rows=[];totals={k:dict(pairs=0,exact=0,byte_exact=0,max_abs=0.) for k in ('forward_input','forward_output','activation_gradient','parameter_gradient')}
    checkpoint_records=[]
    for method,report in reports.items():
        coverage=[v for variant in report['variants'] for v in variant['coverage'].values()]
        row=dict(id=method,title=metadata[method]['title'],paper=metadata[method]['paper'],
                 variants=len(report['variants']),steps_per_variant=128,totals=report['totals'],
                 executed_module_pairs=sum(c['native_calls']>0 for c in coverage),
                 unexecuted_module_pairs=sum(not c['native_calls'] and not c['excluded'] for c in coverage),
                 backward_recompute_calls=sum(c['native_backward_recompute_calls'] for c in coverage),
                 structural_mismatches=sum(len(v['structural_mismatches']) for v in report['variants']),
                 exclusions=report['exclusions'],evidence='evidence/activation_parity/'+method+'.json')
        for kind,s in report['totals'].items():
            for field in ('pairs','exact','byte_exact'):totals[kind][field]+=s[field]
            totals[kind]['max_abs']=max(totals[kind]['max_abs'],s['max_abs'])
        rows.append(row)
    # Hooks must not silently change the model trajectory. Compare every final
    # model/EMA/full optimizer state both between paths and to the earlier
    # 128-step no-hook baseline, rather than hashing serialization ZIP headers.
    for directory_pair in sorted((ROOT/'runs/activation_parity/activation_parity').iterdir()):
        if not directory_pair.is_dir():continue
        a=torch.load(directory_pair/'native.pt',map_location='cpu',weights_only=True)
        b=torch.load(directory_pair/'unified.pt',map_location='cpu',weights_only=True)
        records={'native_vs_unified':compare_tree(a,b)}
        for side,current in [('native',a),('unified',b)]:
            baseline=ROOT/'runs/native_parity'/directory_pair.name/(side+'.pt')
            previous=torch.load(baseline,map_location='cpu',weights_only=True)
            records[side+'_vs_earlier_no_hook_baseline']=compare_tree(previous,current)
        checkpoint_records.append(dict(variant=directory_pair.name,comparisons=records))
    if len(checkpoint_records)!=21:raise SystemExit('Expected 21 checkpoint pairs')
    sources=json.loads((ROOT/'evidence/sources.json').read_text());source_checks={}
    for name,source in sources.items():
        path=ROOT/'sources'/name
        changed=[relative for relative,digest in source['selected_files'].items() if not (path/relative).exists() or sha(path/relative)!=digest]
        head=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        source_checks[name]=dict(files=len(source['selected_files']),changed_files=changed,pinned_commit=source['commit'],head_commit=head,pin_matches=head==source['commit'])
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoints_passed=all(s['passed'] for r in checkpoint_records for s in r['comparisons'].values())
    exact=all(s['pairs']==s['exact']==s['byte_exact'] and s['max_abs']==0 for s in totals.values())
    sources_unchanged=all(not s['changed_files'] and s['pin_matches'] for s in source_checks.values())
    summary=dict(audited_at_utc=now,status='passed' if exact and checkpoints_passed and sources_unchanged else 'needs_review',
                 device='cpu',gpu_used=False,methods=12,external_papers=11,variants=21,steps_per_variant=128,
                 totals=totals,rows=rows,checkpoint_comparisons=checkpoint_records,
                 checkpoint_paths_match_earlier_uninstrumented_runs=checkpoints_passed,
                 sources=source_checks,all_pinned_sources_unchanged=sources_unchanged,
                 selftest='evidence/activation_parity/selftest.json',
                 definitions=dict(forward='Every tensor input/output leaf of every executed shared nn.Module, matched by qualified path, call occurrence, tensor-tree path and execution phase.',
                                  backward='Output tensor hook captures dL/dactivation; gradient absence/unused outputs checked separately; parameter gradients compared before clipping.',
                                  exact='Shape/dtype + torch.equal; additionally identical contiguous uint8 representations. Max absolute differences also measured.',
                                  scope='CPU fp32 or author-propagated fp64, reduced random author network, synthetic input, existing native-specific presets, 128 steps per variant. Shared author classes reused; not an independent rewrite of every neural layer.'),
                 limitations=['Functional operations inside module.forward are not separate traced layers; module coverage explicitly lists containers/branches never executed.',
                              'Diffusability: author BTCHW facade root has no matching unified Module; all shared VAE submodules are traced. Existing native/unified latent/reconstruction/mean/logvar assertions remain. Extra diagnostic-only encoder call is suspended from tracing, with both loss graphs unchanged.',
                              'DiMR checkpoint recomputation is also traced and matched during backward. MDM positional/keyword inputs are bound to their forward signature before comparison. Equal infinite LapFlow attention-mask entries are handled as equal.',
                              'Same controlled RNG reseeding per training step as the existing suite; earlier DFM continuous-RNG test remains separate.',
                              'The existing source attribution/compatibility limitations still apply (including DFM released REPA code, LapFlow two-level full training, PixelFlow CPU SDPA, RMD provenance). No paper-scale quality, GPU kernel or arbitrary component-combination claim.'],
                 files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/activation_trace.py',ROOT/'tools/validate_activations.py',ROOT/'tools/validate_training.py',ROOT/'tools/validate_training_more.py',ROOT/'tools/validate_rmd.py',ROOT/'tools/build_activation_report.py']})
    (directory/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if summary['status']!='passed':
        print(json.dumps(dict(status=summary['status'],exact=exact,checkpoints_passed=checkpoints_passed,sources_unchanged=sources_unchanged)))
        raise SystemExit('Review SUMMARY.json before updating coverage claims')
    table=[]
    for row in rows:
        cells=[html.escape(row['title']),str(row['variants']),str(row['executed_module_pairs'])]
        for kind in ('forward_output','activation_gradient','parameter_gradient'):
            s=row['totals'][kind];cells.append(str(s['byte_exact'])+'/'+str(s['pairs'])+'，差值 0')
        cells.append('<a href="'+row['id']+'.json">逐层明细</a>')
        table.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐层 activation 验证</title>
<style>body{font:16px/1.65 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#19243a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd4e1;padding:9px;text-align:left}th{background:#eaf1fa}code{background:#edf2f8;padding:2px 5px}li{margin:8px 0}</style>
<h1>12 类方法逐层 activation 与 gradient 验证</h1>
<p>CPU · 缩小作者网络 · 合成输入 · native 专用配置 · 21 种变体各 128 步。下表所有比较同时通过值、shape/dtype 和原始字节一致性检查。</p>
<p>检查每个实际执行的 nn.Module 的 tensor 输入、输出及输出反向梯度，包括 DiMR 的 backward checkpoint 重算。参数梯度在 clip 前比较。作者 backbone 类被复用，验证的是 native 与 unified 的路径、目标和训练组装；不是独立重写所有网络层。</p>
<table><thead><tr><th>方法</th><th>变体</th><th>执行模块对</th><th>Forward 输出</th><th>Activation 梯度</th><th>参数梯度</th><th>证据</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table>
<p>每项输入比较同样完全一致。21 对最终 checkpoint 的 model / EMA（如有）/ 完整优化器状态，既与另一条路径一致，也与先前无 hook 的 128 步结果一致，说明记录器没有改变所测训练轨迹。17 个固定源码仓库全部保持原样。</p>
<p>Diffusability 仅比较共享 VAE 内部全部执行模块：原版 BTCHW facade 与 unified BCHW 调用接口不同；额外无 loss 的 encoder 诊断调用不计入逐层调用匹配，实际两侧 loss graph 未改。已有 mean/logvar/latent/reconstruction 检查仍执行。</p>
<p>这不覆盖模块内部每一个 functional/ATen 运算、未执行分支、原论文大模型与真实数据质量，也不认证 GPU kernel。逐模块覆盖、未执行模块、requires-grad 但未用到的输出和无梯度输出均见明细。</p>
<p><a href="SUMMARY.json">完整汇总与 checkpoint/源码完整性</a> · <a href="selftest.json">记录器的故意差异自测</a> · <a href="../../README_ZH.md">运行说明</a></p>
</html>'''
    (directory/'REPORT_ZH.html').write_text(body)
    print(json.dumps(dict(status='passed',totals=totals,checkpoint_pairs=len(checkpoint_records),source_files=sum(s['files'] for s in source_checks.values()))))


if __name__=='__main__':main()
