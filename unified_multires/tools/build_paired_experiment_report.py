#!/usr/bin/env python3
"""Assemble evidence without turning short source-core checks into FID claims."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/paired_experiments_20261010'

def main():
    manifest=json.loads((ROOT/'configs/paired_experiments/manifest.json').read_text())
    train={}
    for mode in ('paper','fair-rdd'):
        rows=[]
        for path in sorted((OUT/'validation'/mode).glob('*.json')):
            if path.name=='SUMMARY.json':continue
            c=json.loads(path.read_text())
            if c.get('profile') in ('mdm_cc12m256','mdm_cc12m1024') and mode=='paper':
                path=OUT/'mdm_final'/path.name;c=json.loads(path.read_text())
            rows.append(dict(profile=c['profile'],status=c['status'],evidence=str(path.relative_to(ROOT))))
        train[mode]=rows
    checks=[]
    for path in sorted((OUT/'auxiliary').rglob('*.json')):
        if '/activation/' in str(path):continue
        c=json.loads(path.read_text())
        if 'status' in c:
            checks.append(dict(profile=c.get('profile',c.get('method')),status=c['status'],
              numeric_checks=len(c.get('checks',[])),all_numeric_exact=all(x.get('bitwise',False) for x in c.get('checks',[])),evidence=str(path.relative_to(ROOT))))
    for file in ('dopri5/pixelflow.json','euler_regression/pixelflow.json','launcher_smoke.json','full_batch/rdd_unified_xl_300_300_400.json'):
        path=OUT/file;c=json.loads(path.read_text());checks.append(dict(profile=file,status=c['status'],evidence=str(path.relative_to(ROOT))))
    pins=json.loads((ROOT/'evidence/sources.json').read_text());count=0
    for repo,pin in pins.items():
        for file,expected in pin['selected_files'].items():
            if hashlib.sha256((ROOT/'sources'/repo/file).read_bytes()).hexdigest()!=expected:raise ValueError('Changed author source: '+repo+'/'+file)
            count+=1
    code_files=[ROOT/'umdiff/source_objectives.py',ROOT/'umdiff/samplers.py',ROOT/'run_original_papers.sh',ROOT/'run_unified_papers.sh']
    code_files += [ROOT/'tools'/name for name in ('paired_experiments.py','author_cli_plans.py','validate_paired_experiments.py','validate_paired_launchers.py','build_paired_experiments.py')]
    code_files=sorted(set(code_files+list((ROOT/'umdiff').rglob('*.py'))+list((ROOT/'tools').rglob('*.py'))))
    all_ok=all(row['status'] in ('passed','byte_exact') for rows in train.values() for row in rows) and all(c['status'] in ('passed','byte_exact') for c in checks)
    result=dict(status='passed_scoped_checks' if all_ok else 'incomplete',training=train,other_checks=checks,
      source_files_verified=count,source_repositories=len(pins),gpu_used=False,formal_training_started=False,
      native_cli_pair_certified=False,full_paper_result_certified=False,
      configuration_manifest_sha256=hashlib.sha256((ROOT/'configs/paired_experiments/manifest.json').read_bytes()).hexdigest(),
      paper_suite_note='Two MDM fixtures were rerun after correcting reduced nested conditioning/time widths; original failed attempts retained.',
      code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files})
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 论文主要结果配方与成对实验审计（2026-10-10）','',
      '**没有启动正式实验，没有使用GPU，没有下载模型/数据。不能保证全部论文最佳结果的完整复现。**','',
      '已准备两份脚本、一份共用清单和33份配置。对应原版/unified的完整配置合同相同。默认只准备真实CLI入口计划；未经认证的作者完整CLI配对会被阻止。另实现了单进程CPU的作者源码/unified组件成对训练工具，见[使用说明](../../PAIRED_EXPERIMENTS_ZH.md)。','',
      '## 主要结果使用的实验配置','',
      '下面区分论文正文/附录、公开训练示例与发布权重。未定位的值保留未知，不能把CLI默认值补成最佳结果参数。具体公开模板的全部架构/优化器/EMA/路径/采样器参数在共用清单和33份JSON中。','',
      '| 方法与来源 | 数据/任务 | global batch | 优化器、LR | 训练预算与初始化 | 主要评估/限制 |','|---|---|---|---|---|---|']
    def cell(v):return ('未知' if v is None else str(v)).replace('|','/').replace('\n',' ')
    for name,p in manifest['papers'].items():
        link='['+p['title']+']('+p['paper_url']+')' if p['paper_url'] else p['title']
        lines.append('| '+' | '.join(map(cell,[link,p['dataset'],p['batch'],str(p['optimizer'])+'; LR '+str(p['lr']),str(p['budget'])+'; '+p['initialization'],p['evaluation']]))+' |')
    lines+=['','### 每篇不能省略的配置差异','']
    for name,p in manifest['papers'].items():
        repo=p.get('repository');ref=(' [作者代码]('+repo['repo']+')，固定commit `'+repo['commit']+'`。') if repo else ''
        lines+=['**'+p['title']+'**。'+ref,'']
        lines+=['- '+gap for gap in p['gaps']];lines.append('')
    lines+=['## 是否统一为RDD参数','',
      '可以做新的受控对照，不能同时承诺原论文成绩。建议保留“原配方最佳质量”与“RDD预算对齐”两张表。共同global batch和更新步数只保证训练图片数量相同，不保证FLOPs、优化难度、模型容量或预训练成本相同。源码后端对齐检验的是移植忠实度，不是方法质量。','',
      'RDD对齐参数：ImageNet256类条件，global256=1进程×micro16×accum16，AdamW(.9,.999)/eps1e-8/LR1e-4/wd0，无warmup/clip，EMA.9999，FP32，seed0，400k更新。保留原方法backbone、VAE、初始化和推理策略。历史RDD-L/XL是8进程×32，stage300/300/400，L最后492000、XL407600；400k是本次共同预算选择，不冒充历史结束步。','',
      '可比组11配置、6种方法：RDD-L/XL、PPFlow2/3、DiMR-G/XL、PixelFlow Euler/Dopri5、DFM公开XL/B、LapFlow论文flags。Relay两网络、SSD/DCT其他分辨率、MDM文本数据、Diffusability编码器、RMD蒸馏以及inference-only方法保留自己的实验类别。','',
      '还应比较相同的50k评估样本、ADM参考统计、CFG选择预算；汇报实际NFE、训练/推理FLOPs、预训练与VAE/teacher/SR成本，并做多seed。把所有方法都降到RDD的LR/wd并不能证明它们各自已达到最好成绩。','',
      '## 本次验证','',
      '| 范围 | 结果 | 不能推断的内容 |','|---|---|---|',
      f'| 公开配方源码后端 | {len(train["paper"])}配置、各2次累积更新；最终逐字节相同；MDM是修正缩小网络维度后的单独复测 | 不是完整作者CLI，也不是完整模型/数据训练 |',
      f'| RDD对齐源码后端 | {len(train["fair-rdd"])}配置、各2次累积更新；activation/gradient/model/Adam/EMA/连续RNG逐字节相同 | 多数逻辑batch和网络缩小，FP32CPU |',
      '| 保留实际RDD batch | global256/micro16/accum16，缩小网络2次更新逐字节相同 | 不是历史8卡轨迹或真实VAE像素 |',
      '| 实际新reader/loop/save | 两个新进程、缩小RDD、local合成fixture、2更新；权重/optimizer/EMA/TorchRNG/trace相同 | checkpoint容器SHA可因归档名字不同而不同，不能用容器字节代替张量比较 |',
      '| PixelFlow最佳采样分支 | 原版Dopri5与统一版256浮点像素/RNG完全相同；Euler另通过回归 | 随机缩小网络；没有最佳checkpoint/FID |',
      '| SPEED/NAMI/MrFlow | 原版完整公开推理函数与统一实现、随机localVAE/SR的latent和像素逐元素相同 | 缩小随机模型、合成embedding；没有预训练质量认证 |',
      '| Diffusability | 原SE/VGG/冻结codec组件2更新通过 | 未公开的完整训练配方/下游生成主表 |',
      '| RMD | 发布flags、17更新覆盖15步warmup后的fake/student阶段及两transition，通过 | 缩小Wan、CPU；不是14B最佳配置/GAN主表 |',
      '| 回归与完整性 | pytest27通过；作者17仓库'+str(count)+'文件hash相同 | 不能替代正式质量训练 |','',
      '发现并修复的差异：PixelFlow主表采用Dopri5而旧模板采用Euler；Dopri5保留跨阶段双份CFG状态，下一阶段也对双份状态采样block noise，不能套Euler的单份状态。新前端还把两侧都约定为microbatch标量loss，修复FP32日志均值约3e-8～2e-7的归约顺序差异；修复前参数梯度和更新已经相同。作者repo的models/libs/sampler同名，验证必须隔离进程。MDM缩小模型时投影与time embedding宽度必须在嵌套层同时匹配；这属于测试夹具修正，没有修改作者源码或完整模型配置。','',
      '验证的activation只覆盖所执行的模块边界；不宣称检查了全部functional/ATen节点或所有未执行分支。使用作者backbone和部分作者solver是明确的设计，不能把复用同一实现当作独立算法再现证据。latent结果没有通过原始真实VAE解码时，不称真实像素一致。','',
      '## 要满足“完整复现主要成绩”还缺什么','',
      '1. 完成并逐篇验收真正作者CLI与unified生产入口的reader、DDP/Accelerate、seed、worker、AMP、累积/更新上限、LR/EMA图像时钟、初始化与恢复。当前计划对此全部保持未认证；不是仅缺一个路径。',
      '2. 对原论文有明确信息的公开分支提供本地真实数据、原始VAE/SiT/teacher/bridge/SR和评估资产；不自动下载，不换成随机或其他模型。',
      '3. 取得缺失的主表训练参数/数据/分支。DFM论文与公开SiT版本不同；MDM Flickr数据/课程未完整发布；Diffusability主表instructions/checkpoints未公开；部分PPFlow/DCT/NAMI训练分支未公开。这些无法由统一batch补齐。',
      '4. GPU窗口重新明确后，比较实际dtype/kernel、完整模型及多卡真实小数据更新/采样/恢复；CPU通过不代表GPU逐位相同。',
      '5. 最后才运行全预算训练、作者协议采样和FID/其他指标，并报告seed方差。数学/组件一致性不能承诺训练后的指标恰好等于论文。','',
      '## 未公开/未确认可执行作者实现的8篇','']
    for row in manifest['unreleased']:lines.append('- ['+row['title']+']('+row['paper_url']+')：'+row['reason'])
    lines+=['','完整机器证据：[SUMMARY.json](SUMMARY.json)。失败尝试保留在attempts与历史日志，不用失败夹具冒充方法不一致，也不把短跑成功冒充最佳画质。测试payload在保留SHA/张量统计后清理，清单见ARTIFACT_CLEANUP.json。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],paper_profiles=len(train['paper']),fair_profiles=len(train['fair-rdd']),other_checks=len(checks),sources=count)))

if __name__=='__main__':main()
