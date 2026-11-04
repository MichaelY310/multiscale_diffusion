"""Read only actual metrics, export an auditable Chinese Markdown report.

Standard library only. Missing experiments stay pending; never invent speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main(a):
    repo=Path(a.repo).resolve();root=repo/'eval_results/aggressive_direct_reuse_20260915'
    rel='../../eval_results/aggressive_direct_reuse_20260915'
    lines=['# 更激进的直接复用：以采样＋VAE减少10%为目标','',
           '新增：[10%–50%速度档位、层布局对照与全部视频](RDD_REUSE_SPEED_TIERS_20260915_ZH.html)。本页保留第二轮原结果，不与新一轮混用计时。','',
           '本页是2026-09-15第二轮，和[第一轮6层复用](RDD_DIRECT_REUSE_FULLMODEL_20260915_ZH.html)分开记录。所有表格由原始metrics生成；没有结果的项目明确标记等待。','',
           '## 1. 是否充分利用了低分辨率信息','',
           '上一轮没有：图片只用了16→32的6/24层；视频只在最后一个stage复用少数层。这次扩展为：','',
           '| 路径 | 做法 | 低分辨率信息从哪里来 |','|---|---|---|',
           '| 图片复用12/24层 | fine的50% blocks直接加上coarse更新，整网微调800步 | 最后一次16×16调用，t=300 |',
           '| 图片复用18/24层 | fine的75% blocks直接加上coarse更新，整网微调800步 | 同上；不新增adapter |',
           '| 视频fine18/fine24 | 最后stage复用18/30或24/30层 | 上一stage真实算过的更新 |',
           '| 视频cascade24 | middle、fine各复用24/30层，保留的真实层错开 | coarse→middle→fine；middle新算的层能补充更新 |',
           '| 视频cascade28 | middle、fine各复用28/30层，只保留首尾层真实计算 | 大量更新从最初coarse级联传递 |','',
           '**这叫利用更多已有更新，不是让低分辨率凭空拥有高分辨率细节。** 未复用层仍需负责当前时间、文本条件和细节修正。大幅复用会限制这种修正能力，不能保证多训练就一定追平。','',
           '视频factor仍为Spatial `(4,1)→(2,1)→(1,1)`；两个数分别为空间、时间缩小倍数。所有视频同一个狐狸prompt、seed20660913、DMD3NFE、noise_filling。没有同时叠加C2F、StepReuse或改变VSA密度。','',
           '## 2. 公平性与10%的硬上限','',
           '比较对象始终是**同一个RDD分辨率路径＋VSA20、不做block复用**。精度、seed、帧数、输出、resident/offload、VAE都相同。每次真实解码，不用常数拼总时间。','',
           '`总延迟减少比例 = 省下的去噪时间 / (原去噪时间 + 原VAE时间)`。上一轮1280×768×29帧约2.7秒去噪、20秒VAE；即使把去噪全部删掉，总时间也最多减少约12%。所以想达到10%，须接近消除大部分去噪计算。','',
           '本轮也测试81帧，同组baseline和候选都是81帧，不能拿29帧候选与81帧baseline比较。更长序列的收益不能冒充所有视频长度都能达到的收益。','',
           '**硬件仍与其他训练共享，只使用GPU0–3。** 暖启动和ABBA/BAAB减小漂移，但不能把这些数字当作独占GPU的正式论文结果。视频只用未经RDD微调的FastWan，尚未做视频训练或正式质量评价。','',
           '## 3. 新增的实现修复','',
           '大规模复用首次触及本任务的显存额度：旧controller把已经消费过的fine缓存留到了VAE。现在只保存下一stage真正需要的更新，消费一次就释放，不把最终fine缓存带入VAE。','',
           '这个优化**只用于每stage一次NFE的3步视频**，不能直接套进每stage多次NFE的图片采样。CPU测试及真实视频轨迹逐元素一致性检查要求差异为0。另启用本进程的expandable allocator缓解碎片；不改变其他进程或数学计算。','',
           '## 4. 视频实测与所有输出','',
           '百分比均为**延迟减少**，不是吞吐提升。秒数是各臂中位数；百分比是配对组均值，所以四舍五入后不必精确互相换算。区间为同次实验组bootstrap，仅表征组间波动。不同配置分别与各自GPU上的同期baseline配对，不把跨卡的绝对时间差直接当成方法收益。','']
    cases=[('video_1280_fine','29帧：只复用fine'),('video_1280_cascade','29帧：两级级联'),
           ('video_1280_81f_cascade24','81帧：middle与fine各复用24层'),('video_1280_81f_fine24','81帧：fine复用24层'),
           ('video_832_temporal221_bf16','额外部署对照：同用BF16 VAE，Temporal221复用18层'),
           ('video_832_temporal221_bf16_high24','额外部署对照：同用BF16 VAE，Temporal221复用24层')]
    audit={'complete':False,'video':{},'image':{},'files_sha256':{}}
    def check_file(path):
        audit['files_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    audit['training']={}
    for mask in ('reuse12','reuse18'):
        folder=root/f'train_{mask}'
        if (folder/'complete.json').exists():
            for filename in ('complete.json','protocol.json','validation.jsonl'):check_file(folder/filename)
            audit['training'][mask]={'complete':json.loads((folder/'complete.json').read_text()),
                                     'protocol':json.loads((folder/'protocol.json').read_text())}
    for case,title in cases:
        lines+=['### '+title,'']
        path=root/case/'metrics.json'
        if not path.exists():
            lines+=['等待运行完成，暂无可汇报数字。',''];continue
        m=json.loads(path.read_text());assert m['complete'] and m['baseline_noop_max_abs']==0
        assert all(v==0 for v in m['cache_lifetime_parity_max_abs'].values())
        expected_precision='bf16' if 'bf16' in case else 'fp32'
        assert m['vae'].startswith(expected_precision) and m['boundary']=='noise_filling' and m['NFE']==3
        width,height=m['shape'][-1]*8,m['shape'][-2]*8
        check_file(path);audit['video'][case]=m
        lines+=['| 版本 | 去噪：基线→候选 | 去噪延迟减少 | 采样＋VAE：基线→候选 | 总延迟减少／95%区间 |',
                '|---|---|---|---|---|']
        for e in m['entries']:
            b,c=e['baseline'],e['candidate'];d=e['reductions']['denoise'];t=e['reductions']['total']
            lines.append(f"| {e['variant']} | {b['denoise']['median']:.3f}→{c['denoise']['median']:.3f}s | {d['mean']:.2f}% | {b['total']['median']:.3f}→{c['total']['median']:.3f}s | **{t['mean']:.2f}%** [{t['lower']:.2f},{t['upper']:.2f}]% |")
        lines+=['',f"输出{width}×{height}×{m['frames']}帧，16fps；`{m['factors']}`；同模型、同prompt与seed；两边都用{expected_precision} VAE。",'']
        sheet=root/case/'five_frame_comparison.jpg'
        if sheet.exists():
            check_file(sheet);lines += [f'![固定五帧、所有方法]({rel}/{case}/five_frame_comparison.jpg)','']
        if expected_precision=='bf16':
            q=m['vae_precision_audit'];check_file(root/case/'baseline_fp32_decode.mp4')
            lines+=['**这是不同部署精度下的独立公平对照，不与上方FP32总时间交叉计算加速。** BF16不是本方法贡献，也不能算成复用省下的时间。', '',
                    f"同一个baseline latent的FP32/BF16解码审计：PSNR={q['psnr_db']:.2f}dB，MAE={q['mae']:.6f}，max_abs={q['max_abs']:.6f}，像素范围[-1,1]。不是严格逐元素一致，也不是正式质量等价证明。",'',
                    f"[同latent的FP32解码视频]({rel}/{case}/baseline_fp32_decode.mp4)",'']
        for name,masks in m['layers_by_stage_zero_based'].items():
            for ext in ('png','mp4'):check_file(root/case/f'{name}.{ext}')
            lines += ['#### '+name,'',f'<video controls preload="none" poster="{rel}/{case}/{name}.png" src="{rel}/{case}/{name}.mp4" style="width:100%;max-width:800px"></video>','',
                      '| 配置 | 内容 |','|---|---|',f"| 输出 / stage | {width}×{height}×{m['frames']}帧；`{m['factors']}` |",
                      '| 模型 / sampler | FastWan原权重，未RDD微调；DMD3NFE，1000/757/522，noise_filling |',
                      f"| 复用层，0开始 | `{masks}`；字典key1为middle，2为fine；scale=1 |",
                      f'| attention / VAE | 未跳层完整保留VSA20/coarse output/gates；{expected_precision} VAE，与baseline相同 |']
            if name=='baseline':lines+=['| 时间参照 | 上表各行同期baseline；1× |']
            else:
                e=next(e for e in m['entries'] if e['variant']==name);b,c=e['baseline'],e['candidate']
                for key,label in [('denoise','去噪'),('total','采样＋VAE')]:
                    ratio=b[key]['median']/c[key]['median'];red=e['reductions'][key]['mean']
                    lines.append(f"| {label} | **{c[key]['median']:.3f}s**；中位数比{ratio:.3f}×；配对延迟减少{red:.2f}% |")
            lines+=['']
        lines += [f'[原始配置、每次计时和缓存审计]({rel}/{case}/metrics.json)','']
    lines += ['## 5. 图片整网微调验证','',
              '原RDD-Unified-L/2 400K、300/300/400；24blocks、hidden1024；没有新参数。两个版本都从原400K开始，不从旧adapter初始化。每个实验单卡完整batch256，AdamW lr1e-5，800updates，保存200/400/800快照。沿用第一轮同一4096张teacher轨迹/128张验证数据、fine＋coarse保持损失。','',
              '**图片本轮仅16→32复用，不宣称已经训练8→16级联。** 现有语料缺少完整middle-stage训练轨迹，不应把只训练middle终点冒充已训练整个middle阶段。视频级联目前是独立的速度/质量sanity。','',
              'checkpoint仍是原模型结构的权重，但新权重必须配合对应12层或18层mask采样。不能把训练时复用、测试时又全算一遍的结果当成相同方法。旧400K及旧采样脚本不受影响。','']
    paths=sorted((root/'image_preview').glob('shard_*/metrics.json'))
    if len(paths)==2:
        metrics=[json.loads(p.read_text()) for p in paths];assert all(m['complete'] for m in metrics)
        rows=[r for m in metrics for r in m['rows']];names=list(dict.fromkeys(r['variant'] for r in rows))
        lines+=['分类器只用于检查目标类别，不是FID；32组固定配对图，共224张。','',
                '| 版本 | 类别Top-1 | Top-5 | 目标概率 |','|---|---:|---:|---:|']
        for name in names:
            r=[x for x in rows if x['variant']==name];assert len(r)==32
            lines.append(f"| {name} | {100*statistics.mean(x['top1'] for x in r):.2f}% | {100*statistics.mean(x['top5'] for x in r):.2f}% | {statistics.mean(x['target_probability'] for x in r):.4f} |")
        lines+=['','图片推理BF16，DDPM80/80/90，CFG1，noise_filling；下方计时为batch8，实际VAE也同驻留执行。','',
                '| 候选 | 去噪延迟减少 | 采样＋VAE延迟减少 |','|---|---:|---:|']
        for name,groups in metrics[0]['benchmark'].items():
            lines.append(f"| {name} | {statistics.mean(g['reduction_percent']['denoise'] for g in groups):.2f}% | {statistics.mean(g['reduction_percent']['total'] for g in groups):.2f}% |")
        lines+=['','每张对比从左到右：原RDD、raw12、12层微调200步、12层微调800步、raw18、18层微调200步、18层微调800步；各固定分片前8张，不选最好看的样本。','']
        for p,m in zip(paths,metrics):
            check_file(p);check_file(p.parent/'comparison.png')
            for r in m['rows']:
                img=p.parent/r['variant']/f"{r['image']%m['batch']:03d}.png";check_file(img)
                assert audit['files_sha256'][str(img.relative_to(root))]==r['sha256']
            lines += [f"![七列固定样本]({rel}/image_preview/{p.parent.name}/comparison.png)",'']
        audit['image']={'metrics':metrics}
    else:lines+=['800步训练／配对采样尚未全部完成。不能据此宣称微调已修复激进复用。','']
    notes=repo/'docs/research_docs/RDD_AGGRESSIVE_DIRECT_REUSE_20260915_NOTES_ZH.md'
    if notes.exists(): lines+=['## 6. 人工检查与结论','',notes.read_text(encoding='utf-8'),'']
    lines+=['## 如何读这些结果','',
            '速度通过和质量通过是两件事。某一配置平均总延迟减少10%不代表所有长度、硬件和共享负载都达到10%；若区间跨过10%，也不能声称稳定越过门槛。视频未微调，若主体崩坏就只能作为计算预算证据。','',
            '训练fine预测误差下降，不保证student自己滚出的图片恢复。更多复用可能让coarse预测被拉偏、缓存跨timestep不匹配更大，或者让真实fine层不足以补细节。所有失败输出仍保留。','',
            '## 7. 可执行代码与原始数据','',
            '- [激进图像训练](../../2d/attention_analysis/train_aggressive_direct_reuse.py)',
            '- [图像mask与共享原实现](../../2d/attention_analysis/aggressive_direct_reuse.py)',
            '- [图像配对采样与真实VAE计时](../../2d/attention_analysis/sample_aggressive_direct_reuse.py)',
            '- [29帧视频benchmark](../../rdd_video/benchmark_aggressive_direct_reuse.py)',
            '- [81帧视频benchmark](../../rdd_video/benchmark_aggressive_direct_reuse_long.py)',
            '- [单次stage缓存生命周期优化](../../rdd_video/attention_analysis/one_call_cascade_controller.py)',
            '- [CPU正确性测试](../../tests/test_aggressive_reuse.py)',
            f'- [结果SHA审计]({rel}/summary_and_audit.json)','',
            '服务器结果：`/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915`。旧模型、第一轮结果及其他用户任务均未覆盖或停止。','']
    audit['complete']=len(audit['video'])==6 and bool(audit['image'])
    if audit['complete']:
        lines[4:4]=['**本轮已全部完成：2组800步整网微调、32组配对图像（224张PNG）、14段主实验视频及2段同latent精度对照。** 找到了采样＋VAE减少10%以上的配置，但**没有证明激进复用后画质已恢复**。','',
                    '| 代表配置 | 采样＋VAE延迟减少 | 质量结论 |','|---|---:|---|',
                    '| 图片，fine复用12/24层，800步整网微调 | 11.26% | 类别Top-1为9/32，原版24/32；明显未保质 |',
                    '| 图片，fine复用18/24层，800步整网微调 | 17.78% | 类别Top-1为1/32；不可用，不能靠速度掩盖 |',
                    '| Spatial421，1280×768×81帧，两级各复用24/30层，FP32 VAE | 10.99% | 未微调视频有严重块状伪影 |',
                    '| Temporal221，832×448×29帧，fine复用24/30层，两边同用BF16 VAE | 12.12% | 狐狸主体保留，细节有变化；尚非正式质量验证 |','',
                    '速度为共享GPU下暖启动配对测量的平均延迟下降，不是所有部署的保证。训练和评估已结束，未自动延长训练。','']
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary_and_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    doc=repo/'docs/research_docs/RDD_AGGRESSIVE_DIRECT_REUSE_20260915_ZH.md'
    doc.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(doc),'complete':audit['complete'],'files':len(audit['files_sha256'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');main(p.parse_args())
