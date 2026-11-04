"""Evidence-labeled explanation of the current CPU/offline infra prototypes."""
import argparse
import html
import json
from pathlib import Path


def table(headers,rows):
    cell=lambda tag,x:f'<{tag}>{html.escape(str(x))}</{tag}>'
    return '<table><thead><tr>'+''.join(cell('th',x) for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(cell('td',x) for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def section(out):
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    a=read('stage_birth_cpu_checks.json'); c=read('root_local_cpu_checks.json')
    mild=c['settings']['3840']; aggressive=c['settings']['960']
    body='<h2 id="methods-refined">16. 打磨后的三条候选：哪些可实现，哪些值得研究？</h2>'
    body+='<p class="warning">2026-10-03：仅CPU和离线CUDA编译。没有GPU实验、新视频、训练后质量或≥10%速度证明。A/B是工程优化，C是需训练的新attention图；三者均尚未证明满足“progressive独特优势”的要求。</p>'
    body+=table(['方法','修改对象','是否需要训练','最诚实的定位'],[
        ['A：融合stage birth','上采样/noise/布局转换','无新参数；数值/RNG需验证','减少边界内存读写；不是主要计算省略'],
        ['B：分辨率感知SP','每个stage参与计算的GPU组','不需要','有增长shape才有明显动机；动态SP本身通用'],
        ['C：ROOT feature＋当前局部attention','fine self-attention的K/V来源与连接图','需要联合/full-model微调','计算预算大，但普通局部＋全局bank也能用']])
    body+='<h3 id="method-a">A. 融合stage birth：减少中间搬运，不删除fine信息</h3>'
    body+='<p>stage结束先按原RDD公式恢复clean endpoint c。下一stage仍使用x=αU(c)+σε，保留每个fine位置的新噪声。传统执行可能分别写出上采样张量、噪声张量、乘法结果、相加结果，再转换为patch输入；新程序读取目标卡需要的parent voxel，直接写最终局部patch。不是“第四个pixel不算了”。</p>'
    body+='<pre>clean parent → small packet → target GPU\n                              └─ repeat address\n                                 global-coordinate RNG\n                                 α·parent + σ·noise\n                                 patch packing\n                                 ↓ one final buffer\n                         original patch projection\n                         original full Transformer</pre>'
    body+='<p>比如clean parent=10、α=.7、σ=.3，四个child噪声为1/−1/2/0，输入就是7.3/6.7/7.6/7.0，而不是四个相同值。不能因clean上采样相同而省掉所有child计算。</p>'
    body+='<p>Philox以global B/C/T/Y/X位置寻址：从1卡换成4卡仍生成相同CPU随机数，不用先生成完整fine噪声再跨卡发送。但这是新的随机序列，不复现torch.randn；GPU数学函数尚未执行验证。最优静态SP也应获得相同RNG/fusion。</p>'
    body+=table(['本轮改进','意义'],[
        ['fp32_fused','FP32混合后转BF16；不能冒充原BF16路径'],
        ['bf16_sigma','直接接收原scheduler sigma；系数、noise、每次乘法及加法BF16舍入'],
        ['bf16_scalar','保留Python/FP32 scalar系数，但noise、乘法、加法按BF16存储舍入'],
        ['patch projection CPU adapter','把原Conv3d[D,C,1,2,2]按原顺序转成patch GEMM；FP32与Conv3d对照通过'],
        [f"{a['tests']}项CPU检查",'空间/时间扩张、物理GPU映射、分片不变RNG、舍入、权重顺序、SP开销计入']])
    body+='<p>为何可能仍不到10%：三步采样只有两个birth，Transformer大矩阵乘和VAE完全没有减少；而且Box–Muller的log/sincos、索引读取、packet packing也有成本。若原实现已经高效融合，它可能只有很小收益。CUDA producer已离线编译；还没有GPU projection/runtime/NCCL集成，不能称完整可运行采样器。</p>'
    body+='<h3 id="method-b">B. 分辨率感知SP：少卡算小stage，多卡算大stage</h3>'
    body+='<p>SP把同一个视频的tokens分给多卡。固定4卡的每层通信都有启动/同步开销；root tokens很少时，分4卡可能比1卡还慢。RDD从root到fine的token量增长，因而可以在边界换组。权重从开始已复制在所有卡上，不移动TP权重；只搬C16 latent，不搬D1536 hidden。</p>'
    body+='<pre>root N/16       middle N/4       fine N\nGPU0            GPU0/GPU1        GPU0/GPU1/GPU2/GPU3\n   └─ measured boundary ──┴─ measured boundary ───┘\n\n1→2→4只是候选，不是预先认定的赢家。</pre>'
    body+='<p>准确目标：T=S启动＋Σ每stage时延＋Σbirth/重分片/换组时延＋A最终latent组装＋相同VAE。对照是同资源、同batch、同kernel的最优合法固定SP，而不是随便选固定4卡。若12-head backend支持SP3，也必须加入；非法/放不进显存的配置应明确标注，而不是悄悄排除更快对照。</p>'
    body+='<p>规划器现在接受实际物理卡组，例如(2,3)→(0,1,2,3)，并计入startup/final assembly；不能把本地组rank误当物理GPU。参数只来自实际profile；目前没有profile。虚构单元测试时间只测试优化器，不构成加速证据。</p>'
    body+='<p>普通固定resolution也能动态调SP，尤其服务负载变动时。RDD更适配的是“单请求shape固定增长”，不是发明动态SP。<a href="https://arxiv.org/html/2604.04335v1">GenServe</a>已有弹性SP，因此B属于值得做的工程优化，不是单独的新论文主张。单视频延迟与四路DP吞吐也需分开报告。</p>'
    body+='<h3 id="method-c">C. ROOT feature＋当前局部attention：新的训练图，不跳层</h3>'
    body+='<p>这条不是旧C2F route：旧方法仅用coarse决定fine选哪些区域，fine仍做那些区域的当前QK/AV。新候选让fine只读取本地当前K/V，并读取已生成ROOT提供的global memory；远处fine pixels不再逐对交互。</p>'
    body+='<h4>改进一：只保存一份ROOT最终feature</h4>'
    body+='<p>从ROOT最后一个block、输出head前保存F_root。它经过整个ROOT网络，比缓存初始Gaussian早层KV更值得测试，但仍是root-generating feature，不等于“编码干净root图片的feature”。下一stage允许修改低频；没有强行锁定最终mean=c。</p>'
    body+='<pre>ROOT final feature F [R,D] ── broadcast once ──→ fine GPU\n                                                     │\nfine layer l: current norm/time adapter on F           │\n             charged Wk_l / Wv_l projections ←────────┘\n             temporary Kroot_l / Vroot_l\n             use in THIS layer, then release\n\nNo 30-layer permanent KV cache; no old block output reuse.</pre>'
    body+=table(['Nfine61440 / D1536 / 30层BF16','完整ROOT：R3840','pool2×2 ROOT：R960'],[
        ['常驻feature',f"{mild['root_feature_mib']:.2f} MiB",f"{aggressive['root_feature_mib']:.2f} MiB"],
        ['当前层KV scratch',f"{mild['layer_kv_scratch_mib']:.2f} MiB",f"{aggressive['layer_kv_scratch_mib']:.2f} MiB"],
        ['旧全部30层KV（对照，不保留）',f"{1024*mild['all_layer_kv_gib_reference']:.2f} MiB",f"{1024*aggressive['all_layer_kv_gib_reference']:.2f} MiB"],
        ['4卡：一份feature复制到其他3卡的总payload',f"{mild['copied_feature_total_mib']:.2f} MiB",f"{aggressive['copied_feature_total_mib']:.2f} MiB"],
        ['所有4卡重复投影30层ROOT K/V总算术量',f"{mild['root_project_tflop_aggregate']:.3f} TFLOP",f"{aggressive['root_project_tflop_aggregate']:.3f} TFLOP"]])
    body+='<p>完整ROOT数值3840来自spatial421的N/16；960来自额外2×2空间池化，不是少算fine tokens。常驻feature缩小不是免费加速：层间KV重投影、time adapter、cache IO及广播都必须计入；如果NVFP4有助投影，应给所有对照相同精度配置。</p>'
    body+='<h4>改进二：保留所有fine层，只改变attention连边</h4>'
    body+='<pre>current fine hidden ── fresh Q/K/V ─┬─ local current K/V\n                                   ├─ ROOT feature → current-layer K/V\n                                   └─ ONE joint softmax → O_local+root\n\ncurrent fine Q/K/V ── original cube64 means\n                      current global coarse attention\n                      original gate_compress → O_current_coarse\n\nO_total = O_local+root + gate_compress ⊙ O_current_coarse\n↓ original output projection, residual, text attention, FFN\n↓ next fully live block</pre>'
    body+='<p>必须区分两个coarse：前一个是上个RDD stage的ROOT memory，后一个是当前fine层VSA的pooling全局分支。保留第二个避免不公平删掉VSA的质量保护。所有30层、所有fine tokens、QKV/output projection、FFN和文字cross-attention继续计算。</p>'
    body+='<p>对每个空间tile，联合logits为[QK_localᵀ/√d, QK_rootᵀ/√d+β]，对全部local＋ROOT keys做一次softmax，再乘对应V。不是softmax(local)V_local＋softmax(root)V_root。β在缩放后加；等logit时，3840 ROOT与1024 local会让ROOT自动占约79%权重，所以初始20% prior使用β=log(.2/.8)+log(1024/R)。这是需要学习/对照的建模选择，不是质量保证；ragged窗口应按有效key数校正。</p>'
    body+='<h4>改进三：真正的kernel执行方式</h4>'
    body+='<p>fine grid为16×48×80 patch tokens时，每个窗口包含全部16个latent时间位置和8×8空间patch，共1024 tokens；共60窗口。每卡拥有完整窗口和全部heads，窗口不重叠/不shift，避免新增D维halo通信。RoPE使用原物理坐标，重排token不等于重定义位置；ROOT K必须按明确物理中心旋转一次，不能二次RoPE。</p>'
    body+='<p>CUDA实现应按Q128/K64等tile流式读取local和共享ROOT K/V，维护同一组online-softmax m/l/accumulator，最后才输出。不为每个窗口复制ROOT bank，也不保存完整attention方阵。current VSA cube64分支需要交换compact当前summary；因此只能称“去掉原始fine KV跨卡交换”，不是零通信。若加shift/halo，通信成本必须重新计算。</p>'
    body+=table(['fine-only设置（root/middle保持原VSA20）','mild','aggressive'],[
        ['local tokens / ROOT slots','1024 / 3840','1024 / 960'],
        ['每fine query keys','4864','1984'],
        ['VSA20每query算术参照','12288','12288'],
        ['fine稀疏QK/AV pair量相对VSA20',f"{100*mild['fine_sparse_pair_fraction_of_vsa20']:.2f}%",f"{100*aggressive['fine_sparse_pair_fraction_of_vsa20']:.2f}%"],
        ['3NFE矩阵FLOP减少估算（含text矩阵、所有4卡ROOT投影）',f"{mild['self_ffn_text_matmul_reduction_percent']:.2f}%",f"{aggressive['self_ffn_text_matmul_reduction_percent']:.2f}%"],
        ['初始ROOT prior / post-scale bias',f"20% / {c['bias3840']:.3f}",f"20% / {c['bias960']:.3f}"],
        ['实际总时间 / 生成质量','未测','未测']])
    body+='<p>FLOPs≠延迟。预算已计VSA gate的D→D projection，仍未计norm、RoPE、pooling/gate activation、cache、实际kernel利用率及VAE。若总时间中VAE占63%，要少10%总延迟，去噪净减少需约10%/(1−.63)=27%；要少20%需约54%。矩阵FLOP百分比不能等同总加速。更快codec可以限定scope，但双方必须用同一codec。</p>'
    body+='<h4>怎么训练，以及为什么不能保证恢复质量</h4>'
    body+='<p>teacher固定为同一RDD＋VSA模型，student只在fine阶段换连接图。先使用teacher的coarse/middle输入训练memory consumer，再加入student自己生成的prefix，联合微调ROOT producer和完整fine模型；训练时不能detach ROOT feature。保留原生成loss，加入teacher输出/选定hidden层和低频端点蒸馏。可训练期逐渐从VSA过渡到新分支，但最终计时必须去掉VSA fine稀疏分支，不能把双路径训练时间当部署时间。三NFE测试必须用真正三NFE teacher；原始Wan不能随便只跑三步就叫DMD。</p>'
    body+='<p>local窗口删掉了远距离fine交互，训练不保证能全部补回。应测跨窗物体运动、遮挡显露、同步细纹理、小字、多对象联系与随机种子多样性；不是只看PSNR或几个静态片段。</p>'
    body+='<h4>强普通对照与novelty底线</h4>'
    body+='<p>必须用同一个局部＋全局kernel比较：①ROOT feature；②当前fine池化K/V，同样bank大小（允许直接复用当前已投影K/V，不强迫多算投影）；③普通模型旧full prediction/features的bank；④local＋当前VSA coarse，不加ROOT；⑤原RDD＋VSA。训练预算、codec、dtype、宽度/层数/采样步数和GPU预算相同。特别是960-bank对照可复用已经为VSA coarse branch产生的summary，可能比ROOT更快。</p>'
    body+='<p><a href="https://arxiv.org/abs/2205.13213">HiLo</a>已有local＋pooled-global attention；<a href="https://arxiv.org/html/2511.16117v1">DCS-LDM</a>已有level-causal KV caching；<a href="https://arxiv.org/html/2512.21338v1">HiStream</a>已有low/high生成和dual caches。不能把“局部＋全局”“跨scale cache”本身说成原创。只有生成的ROOT信息在强普通对照下带来更优quality–latency曲线，才有progressive条件生成的研究贡献。</p>'
    body+='<h3>当前实现与推荐顺序</h3>'
    body+=table(['项目','已验证','尚缺'],[
        ['A/B',f"{a['tests']}项CPU；CUDA birth离线编译；projection原权重FP32顺序对照",'GPU/RNG数值、NCCL、运行时接入、实际SP profile'],
        ['C',f"{c['tests']}项CPU：dense/streaming输出和梯度、mask/tail、bias、additive coarse、ROOT梯度",'RoPE adapter、Wan集成、Triton GPU编译/正确性、微调、视频与实际速度'],
        ['CPU streaming memory','forward不复制ROOT bank，不构造完整score','autograd保留chunks；高效训练需重算/fused backward'],
        ['新生成或GPU使用','没有','沿用用户当前不跑GPU的要求']])
    body+='<p>推荐：A/B作为精确执行基线打磨；C作为需要证明的研究候选。不要把A/B包装成必然10%主方法；不要承诺C微调后一定无损。NVFP4目前作为双方共有的实现工具，上一节固定anchor原型尚未满足现有Wan契约，也不作为第四个已可用方法。</p>'
    body+='<p>代码：'+ ' · '.join(f'<a href="../../rdd_video/progressive_infra/{name}">{name}</a>' for name in
        ('stage_birth_reference.py','stage_birth_patch.cu','root_local_attention_reference.py','test_stage_birth_cpu.py','test_root_local_cpu.py'))+'</p>'
    body+='<p><a href="stage_birth_cpu_checks.json">A/B CPU evidence</a> · <a href="stage_birth_compile_checks.json">Offline compiler evidence</a> · <a href="root_local_cpu_checks.json">C CPU/operator/budget evidence</a></p>'
    ready=out/'root_local_implementation_cpu_checks.json'
    if ready.exists():
        checks=json.loads(ready.read_text())
        body+='<h3 id="root-local-ready">17. 方法C算子实现：已准备开测，尚未GPU运行</h3>'
        body+='<p>已新增Triton联合local＋ROOT forward，直接按物理window map读写，不按窗口复制ROOT bank。在线softmax在local和ROOT两段之间共享归一化状态；没有跳层或旧attention输出缓存。</p>'
        body+=table(['项目','状态'],[
            ['新增实现测试',f"{checks['tests']}项CPU测试通过：实际shape布局、cube顺序、query/key尾块、BF16概率舍入模拟、host guards和paired统计"],
            ['旧数学reference','6项CPU输出/梯度/mask/联合softmax测试通过'],
            ['GPU状态','未编译/未运行此新Triton kernel；没有新速度或质量数据'],
            ['测试范围','两张卡分别做SP1独立重复；随机BF16张量，不是实际coarse输出或双卡SP，尚不含实际Wan/RoPE、通信、训练或VAE'],
            ['真实baseline','直接调用已有FastVideo64 video_sparse_attn；在线router、当前coarse output和gate都计时'],
            ['方法C成本','ROOT两个projection、布局转换、local＋ROOT kernel及当前coarse/gate都计时'],
            ['强普通对照','当前已投影K/V池化；N/64直接复用当前VSA summaries，使用同一新kernel'],
            ['计时','先正确性gate，再warmup和kernel配置选择，轮换顺序paired测量，bootstrap区间；小样本区间不能代替长期稳定性']])
        body+=table(['patch-grid','Nfine','local','ROOT候选'],[
            ['8×28×52','11648','128','728 / 182'],['16×48×80','61440','1024','3840 / 960']])
        body+='<p>N/16对应spatial421低分辨率token数量；N/64在当前synthetic测试中只是压缩bank的容量消融，并未实现真实ROOT压缩、layer norm/time modulation或physical RoPE。尤其728-root的7×13空间网格不能直接等分2×2池化成182，不可把所有N/64 case叫空间pool2×2。</p>'
        body+='<p>代码：'+ ' · '.join(f'<a href="../../rdd_video/progressive_infra/{name}">{name}</a>' for name in
            ('root_local_triton.py','root_local_plan.py','benchmark_root_local_gpu.py','test_root_local_implementation_cpu.py'))+'</p>'
        body+='<p>CPU dry-run：<code>python rdd_video/progressive_infra/benchmark_root_local_gpu.py --dry-run</code>。CPU证据：<a href="root_local_implementation_cpu_checks.json">implementation checks</a>。正式GPU第一关仍可能暴露编译、数值或性能问题；当前“可开测”不等于“已验证完整视频能跑”。</p>'
    from build_advanced_infra_report import section as advanced_section
    body+=advanced_section(out)
    return body


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='eval_results/progressive_infra_20261003')
    out=Path(p.parse_args().output)
    style='body{font:16px/1.65 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#202530}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccd1d9;padding:8px;text-align:left}th{background:#eef2f7}.warning{background:#fff1cb;padding:16px}pre{background:#eef2f7;padding:16px;overflow:auto}h3{margin-top:40px}'
    (out/'methods_refined.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RDD infra candidates: refined</title><style>'+style+'</style>'+section(out),encoding='utf-8')
    print(out/'methods_refined.html')


if __name__=='__main__': main()
