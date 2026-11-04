"""Evidence-first report. Never invent configurations to fill a speed tier."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXPECTED=['temporal221_832_29','temporal421_832_29','spatial221_1280_81',
 'temporal421_1280_61','temporal221_832_29_head_controls','temporal421_1600_61',
 'temporal421_832_253','temporal221_832_29_car','temporal221_832_29_person','temporal221_1280_85',
 'spatial221_1280_81_trilinear']

def diagram(path,masks,title):
    # A code-native diagram, not a retouched generated image.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="318" viewBox="0 0 1080 318">',
      '<rect width="1080" height="318" fill="#ffffff"/>',
      '<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="#b06010"/></marker></defs>',
      '<g font-family="Arial,Microsoft YaHei,sans-serif" fill="#15202b">',
      f'<text x="18" y="28" font-size="20">{html.escape(title)}</text>',
      '<text x="18" y="54" font-size="14">蓝色 = 真正执行整个 block；橙色 = 跳过整个 block，加入上一阶段同层更新的上采样。</text>']
    for stage,label in enumerate(['低分辨率','中间分辨率','最终分辨率']):
        y=92+stage*75
        parts.append(f'<text x="18" y="{y+22}" font-size="15">{label}</text>')
        for layer in range(30):
            x=140+layer*30;reuse=layer in masks.get(stage,[])
            parts.append(f'<rect x="{x}" y="{y}" width="26" height="30" rx="3" fill="{"#ffe0af" if reuse else "#b9dafa"}"/>')
            parts.append(f'<text x="{x+13}" y="{y+21}" text-anchor="middle" font-size="12">{layer}</text>')
            if reuse:
                parts.append(f'<path d="M{x+13} {y-43} L{x+13} {y-4}" stroke="#b06010" fill="none" marker-end="url(#arrow)"/>')
    parts+=['<text x="18" y="310" font-size="13">数字是实际 block 编号（0–29）。箭头搬运的是 Δh，不是把当前高分辨率 h 整个覆盖。</text>','</g></svg>']
    path.write_text('\n'.join(parts),encoding='utf-8')

def main(a):
    data=ROOT/'eval_results/reuse_latency_tiers_20260915'
    docs=ROOT/'docs/research_docs'
    cases={p.parent.name:json.loads(p.read_text()) for p in sorted(data.glob('*/metrics.json'))}
    missing=[n for n in EXPECTED if not cases.get(n,{}).get('complete')]
    if missing and not a.allow_incomplete:raise RuntimeError(f'Incomplete cases: {missing}')
    out=['# RDD 跨分辨率残差复用：10%–50% 速度档位实测',
      '\n<style>#content{overflow-wrap:anywhere}#content table{display:block;max-width:100%;overflow-x:auto}#content img,#content video{max-width:100%}</style>',
      '\n这里的“加速 20%”统一指 **采样＋真实 VAE 解码的延迟减少 20%**，即耗时变为 0.8 倍、吞吐意义上的速度为 1.25×；不是“去噪快20%”。',
      '\n比较对象都是 **同尺寸、同分辨率路径的不复用 RDD + 原生 VSA20**。不与普通全分辨率生成混比。',
      '\n重要限制：视频权重仍是未经过 RDD/复用微调的 FastWan。下列实验验证实现、实际时间和未经微调的可见破坏；**没有证明训练后视频质量，也没有正式 VBench/FVD**。',
      f'\n本轮已完成 {len(cases)} 个场景、{sum(len(m["entries"]) for m in cases.values())} 个方法配置。' + (f' 仍缺少：{missing}' if missing else ''),
      '\n## 1. 推荐结论与画面检查']
    notes=docs/'RDD_REUSE_SPEED_TIERS_20260915_NOTES_ZH.md'
    out.append(notes.read_text(encoding='utf-8') if notes.exists() else '\n尚未完成画面审阅；不能作质量推荐。')
    out+=['\n## 2. 方法：究竟复用了哪里？',
      '\n每个阶段仍然先做 patch embedding、时间/文字条件和位置编码；接着进入 **30 个 Transformer block**，最后用原输出头预测。',
      '\n一个正常 block 包含自注意力（原生 VSA 的 fine/coarse 分支与 gate）、文字 cross-attention 和 MLP。正常执行时记下 **输出 hidden − 输入 hidden**，也就是这一层对输入作了什么更新。',
      '\n在下一分辨率的选中层，不计算这个 block，改为：',
      '\n'+r'$$h_{l+1}^{\mathrm{fine}}=h_l^{\mathrm{fine}}+s\,U(\Delta_l^{\mathrm{previous}}),\qquad \Delta_l=h_{l+1}-h_l.$$',
      '\n`U` 是按 token 网格沿变化的空间/时间维度做 nearest 上采样；`s` 是更新强度。本轮试了1、0.75、0.5。**当前 fine hidden 保留**，只替换这一层的更新。执行的 native block 不删 VSA coarse output；被跳过 block 的 coarse output 已包含在旧 Δ 中，但不是当前 timestep 重新计算的结果。',
      '\n这省下的是该层 QKV、注意力、cross-attention、MLP 等全部计算，不只是注意力矩阵。没有新 adapter，也没有偷偷改 VAE。模型权重在本轮测试中全部冻结；需要质量恢复时应训练整网。',
      '\n### A. 只复用最终阶段：优先候选',
      '\n低/中阶段完整执行，最终阶段选18/24/28层搬运中阶段更新。`fine24`是复用24层、保留6层，**不是只计算24层**。`fine28`只留首尾2层。',
      '\n![只复用最终阶段](RDD_REUSE_TIERS_FINE24.svg)',
      '\n下面是20%档实际测试的28层版本：只有最终阶段的第0、29层真实计算。前两次调用仍完整执行，Temporal221中它们形状相同。',
      '\n![20%档的实际跳层图](RDD_REUSE_TIERS_FINE28.svg)',
      '\n### B. 两次级联：更快，也更难训练',
      '\n低阶段先完整算；中阶段复用低阶段选中层；最终阶段复用中阶段选中层。如果中阶段这一层也是复用的，那么最终阶段继承的是已转移过一次的更新。`cascade12_24`表示中阶段复用12层、最终复用24层。',
      '\n![级联复用](RDD_REUSE_TIERS_CASCADE24.svg)',
      '\n`cascade28` 在中/最终阶段都只留 block0、29真实执行，是极端测试，不等于已达到可用画质。',
      '\n### C. 改变真实计算放在哪里',
      '\n`prefix24`直接跳0–23、只算24–29；`head1_tail5`跳1–24、保留首层0和末尾25–29；`head2_tail4`跳2–25；`head3_tail3`跳3–26。它们都跳24层，但保留的计算位置不同。',
      '\n![保留首层与末尾](RDD_REUSE_TIERS_HEAD1.svg)',
      '\n不要把“跳24层”当作足够的配置描述。是否跳首层、在哪些层读取当前文字/时间条件、是否在中间插入新计算，都可能改变效果。',
      '\n### 内部残差上采样的平滑对照',
      '\n另测了`spatial221_1280_81_trilinear`：仅把内部Δh的nearest搬运换成trilinear插值，噪声填充和采样边界完全不变。它是同一复用方法的插值配置，不是新VAE或新采样器。画面仍有明显细节损失，不能说平滑插值已经解决空间复用问题。',
      '\n### D. 缓存实现与适用范围',
      '\n只保存下一个阶段将要消费的层；读完即释放，最终阶段不再保留无用缓存。我们已在同输入下检查原缓存与短生命周期实现的 latent 精确一致。这个优化只改变内存寿命。',
      '\n**当前视频控制器只支持每阶段一次调用**，即这次DMD 3NFE。不能直接拿去跑同stage90步：旧图像实现是整个fine阶段重复使用同一边界缓存，误差累积与本次视频不同。',
      '\n## 3. 公平性、计时口径、为什么有些档位空着',
      '\n所有配置：FastWan2.1-T2V-1.3B（加载约1.49B参数），30层，原生VSA保留20%，DMD三次调用`t=1000/757/522`，`noise_filling`，模型常驻GPU、TF32关闭、两边相同BF16 VAE。没有改采样NFE，没有新增蒸馏权重。',
      '\n`(空间缩小倍数, 时间缩小倍数)`：Temporal221=`(1,2)→(1,2)→(1,1)`；Temporal421=`(1,4)→(1,2)→(1,1)`；Spatial221=`(2,1)→(2,1)→(1,1)`。这是latent的倍数，不能把29输出帧机械除2；具体latent见每个视频表。相同形状的前两次调用不被重新命名为跨分辨率复用。',
      '\n每个方法先热身，再在同卡上按ABBA/BAAB交替跑完整轨迹；每次都实际VAE解码。表中时间是各臂中位数，百分比是**逐组配对延迟减少的平均值**，所以用两个四舍五入中位数算出的百分比可能略有差异。区间仅反映本次3–4个组的重采样，不能当作独立机器/大规模质量保证。',
      '\nGPU0–3与已有训练共享，未停止他人任务。每进程显存上限约28.49GiB；1920×1024的61/125帧两次探测在本进程上限处退出，没有计时结果，不能记成速度0或正常完成。编码MP4、加载模型/T5、冷编译不计入采样＋VAE。',
      '\n假设VAE不变，最乐观的上限（连去噪也全部删掉）是：',
      '\n'+r'$$\text{总延迟最多减少}\leq\frac{T_{\mathrm{denoise}}}{T_{\mathrm{denoise}}+T_{\mathrm{VAE}}}.$$',
      '\n真实方法还要计算最低分辨率与保留层，因此一定比这个上限低。不能把去噪60%写成总时间60%，也不能把832与1600两种输出横向混成同一比较。',
      '\n## 4. 所有方法的实测时间',
      '\n| 场景 | 配置 | 去噪：基线→方法(s) | 采样＋VAE：基线→方法(s) | 总延迟减少；组区间 | 去噪延迟减少 |',
      '|---|---|---:|---:|---:|---:|']
    for name,m in cases.items():
        for e in m['entries']:
            b,c,r=e['baseline'],e['candidate'],e['reductions']
            out.append(f'| {name} | {e["variant"]} | {b["denoise"]["median"]:.3f}→{c["denoise"]["median"]:.3f} | {b["total"]["median"]:.3f}→{c["total"]["median"]:.3f} | {r["total"]["mean"]:.2f}%；[{r["total"]["lower"]:.2f}, {r["total"]["upper"]:.2f}] | {r["denoise"]["mean"]:.2f}% |')
    out+=['\n## 5. 如何训练，而不是继续盲目加大复用',
      '\n**本轮没有新增训练，以下是下一轮实验计划，不是已经得到的质量。** 首先必须有不复用的RDD视频基线，或先完成FastWan→RDD路径适配。原FastWan权重不等于已经学习RDD drift 的teacher；不能把其低分辨率输出直接当成正确RDD监督。',
      '\n1. **从A开始，训练整个backbone。** 冻结T5和VAE，保留共享参数；计算coarse更新时不能detach，让fine误差能反传给产生coarse更新的网络。首选24层的保首层/均匀布局；28层和级联先作压力测试。',
      '\n实现上还有一步不能省：本轮视频控制器是inference-only，缓存用了detach。真正训练必须改为保留计算图的student路径，并用activation checkpointing重算；不能仅把当前控制器切到train模式。图像DirectReuseStudent已有不detach的实现，但视频版尚未完成这种训练验证。',
      '\n2. **训练时就运行真实复用路径。** 用student真实coarse→middle→fine状态；冻结的正确RDD teacher也在这些student输入上给出监督，而不只使用预存teacher轨迹输入。这用于减少“训练见到的状态”和“采样实际进入的状态”不一致。',
      '\n3. **同时保护各stage。** 保留原生RDD目标，并给低、中、最终阶段都加teacher保持损失。若是多步图像模型，覆盖整个stage时间范围，不是仅t600/t300两个边界；若是本次3步视频，覆盖实际三个调用及适度邻域。建议先用等权stage蒸馏，避免只优化fine误差毁掉coarse。',
      '\n4. **渐进加大难度。** 先只复用最终阶段12→18→24层，再试中阶段6→12层＋最终24层。每次只有在独立prompt/seed上恢复质量才升级；不要默认训练更久必然能修好28/30层。最终可增加native block的间隔分布或首尾保留数。',
      '\n5. **起步参数（待验证）**：BF16混合精度，AdamW学习率5e-6到1e-5、weight_decay0.01、梯度裁剪1；视频每卡1个样本，4卡梯度累计到global batch16起步；activation checkpointing。先500–1000更新检查损失/样片，再决定5K/20K，不预承诺收敛。图像可沿用原全局256，但必须修复teacher-input分布和coarse全时域保持。',
      '\n共享显卡环境下，整网视频训练应先验证FSDP/optimizer-state sharding的一步通信和显存，再从29帧起步；本轮采样显存通过不代表整网Adam状态与反向激活也能装下。scale=0.75的部署配置也应在训练时用0.75，不把不同配置混为一谈。',
      '\n6. **需要怎样的证据**：固定未训练prompt/seed，比较同一RDD baseline的文本一致性、运动/闪烁、细节、人脸与肢体；再扩大到VBench/FVD或适合的正式评估。最终速度也必须用训练后权重重测。当前几段狐狸/汽车/行人不能替代此步骤。',
      '\n速度目标不是质量保证：10%档优先训练单次fine复用；20%档先保住单次fine，再尝试少量middle复用；30%及以上若只能靠极端级联才达到，就必须把它当作高风险结构压缩，而不是现成推荐。',
      '\n## 6. 全部视频：按场景放在同一个对比组']
    link_count=0
    for name,m in cases.items():
        out.append(f'\n### {name}\n\nPrompt：{m["prompt"]}；seed={m["seed"]}。')
        rel=f'../../eval_results/reuse_latency_tiers_20260915/{name}'
        out.append(f'\n![所有方法固定五帧，不挑最好帧]({rel}/five_frame_comparison.jpg)')
        entries={e['variant']:e for e in m['entries']}
        for method,masks in m['layers_by_stage_zero_based'].items():
            e=entries.get(method);shape=m['shape'];t,h,w=shape[-3:]
            stages=' → '.join(f'{t//ft}×{h//fs}×{w//fs}' for fs,ft in m['factors'])
            keep='；'.join(f'stage{stage}真实层={sorted(set(range(30))-set(layers))}' for stage,layers in masks.items()) or '全部stage的全部30层真实执行'
            out.append(f'\n#### {method}\n\n<video controls preload="none" poster="{rel}/{method}.png" src="{rel}/{method}.mp4" style="width:100%;max-width:900px"></video>\n')
            out+=['| 配置 | 取值 |','|---|---|',
              f'| 输出 | {w*8}×{h*8}×{m["frames"]}帧，16fps，{m["frames"]/16:.2f}s |',
              f'| latent（T×H×W） | {stages} |',f'| factors（空间,时间） | {m["factors"]} |',
              f'| 复用层（0-based） | {masks or "无"} |',f'| 保留真实层 | {keep} |',
              f'| 更新强度s | {m["reuse_scale"][method]} |',
              f'| 内部残差上采样 | {m.get("reuse_resize","nearest")}；不改变noise_filling边界 |',
              '| 采样与后端 | DMD3NFE，1000/757/522；noise_filling；native VSA20；相同BF16 VAE |']
            if e:
                d=e['reductions']['denoise']['mean'];total=e['reductions']['total']['mean']
                out += [f'| 去噪 | {e["candidate"]["denoise"]["median"]:.3f}s；配对减少{d:.2f}%；{1/(1-d/100):.3f}× |',
                  f'| 采样＋VAE | {e["candidate"]["total"]["median"]:.3f}s；配对减少{total:.2f}%；{1/(1-total/100):.3f}× |']
            else:
                out.append('| 时间基准 | 每个方法都有独立ABBA基线；对应精确时间见总表，不用单次展示调用代替基线 |')
            out.append(f'| 原始记录 | [metrics.json]({rel}/metrics.json) |')
            for suffix in ('.png','.mp4'):
                if not (data/name/f'{method}{suffix}').exists() and not a.allow_incomplete:raise FileNotFoundError(data/name/f'{method}{suffix}')
                link_count+=1
    # Diagrams sourced from real recorded masks, not invented block layouts.
    first=cases.get('temporal221_832_29',{})
    second=cases.get('temporal421_832_29',{})
    third=cases.get('temporal221_832_29_head_controls',{})
    for filename,m,method,title in [('RDD_REUSE_TIERS_FINE24.svg',first,'fine24','A：中间阶段 → 最终阶段，复用24层'),
      ('RDD_REUSE_TIERS_CASCADE24.svg',second,'cascade24','B：低 → 中 → 高，两次各复用24层'),
      ('RDD_REUSE_TIERS_HEAD1.svg',third,'head1_tail5','C：最终阶段保首层＋末尾5层，中间24层复用'),
      ('RDD_REUSE_TIERS_FINE28.svg',cases.get('temporal221_1280_85',{}),'fine28','20%档：前两阶段完整计算，最终阶段只算第0、29层')]:
        if m:diagram(docs/filename,{int(k):v for k,v in m['layers_by_stage_zero_based'][method].items()},title)
    out+=['\n## 7. 复现与文件',
      '\n主测试：`rdd_video/benchmark_reuse_latency_tiers.py`；层位置对照：`rdd_video/benchmark_reuse_latency_tiers_controls.py`；缓存实现：`rdd_video/attention_analysis/one_call_cascade_controller.py`，底层复用：`fastwan_native_attention.py`。这些是独立实验入口，没有改旧checkpoint或FastVideo主模型。',
      '\n服务器输出：`/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915`。本地对应目录在`eval_results/reuse_latency_tiers_20260915`。三个主要结构图和所有mask都从实际metrics生成。']
    report=docs/'RDD_REUSE_SPEED_TIERS_20260915_ZH.md'
    report.write_text('\n'.join(out)+'\n',encoding='utf-8')
    if not a.allow_incomplete:
        manifest=json.loads((data/'bundle_manifest.json').read_text())
        errors=[]
        for relative,expected in manifest.items():
            file=data/relative
            if not file.is_file(): errors.append('missing:'+relative);continue
            if file.stat().st_size!=expected['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=expected['sha256']:
                errors.append('mismatch:'+relative)
        if errors:raise RuntimeError('Bundle verification failed:'+str(errors))
        (data/'delivery_audit.json').write_text(json.dumps(dict(complete=True,verified_files=len(manifest)),indent=2))
    audit=dict(complete=not missing,cases=len(cases),configurations=sum(len(m['entries']) for m in cases.values()),media_links=link_count,missing=missing,
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (data/'report_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-incomplete',action='store_true');main(p.parse_args())
