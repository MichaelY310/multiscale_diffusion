"""Build a evidence-only HTML summary of progressive-infra falsification probes."""
import argparse
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main(a):
    out = Path(a.output)
    def read(name):
        return json.loads((out/name).read_text(encoding="utf-8"))
    math = read("birth_math.json")
    rdd = read("teacher_rdd/metrics.json")
    base = read("teacher_base/metrics.json")
    contract = read("parent_contract/metrics.json")
    decoder = read("decoder_bound/metrics.json")
    locality = read("dmd_locality_fp32/metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for report, name in ((rdd, "RDD-finetuned Wan"), (base, "Original Wan on RDD input")):
        rows = [r for r in report["records"] if r["family"] == "rdd"]
        times = sorted({r["data_time"] for r in rows})
        values = [statistics.mean(r["adjusted_prediction_std_rms"] for r in rows if r["data_time"] == t) for t in times]
        axes[0].plot(times, values, "o-", label=name)
    axes[0].set(xlabel="Data-time t (fine stage starts .7)", ylabel="Noise-adjusted prediction std RMS",
                title="Two clips, four noise draws each")
    axes[0].legend(fontsize=8)
    rows = locality["records"]
    names = ["local_radius2", "radius2to6", "far_radius6plus"]
    for i, name in enumerate(names):
        values = [r["bands"][name]["energy_fraction"]*100 for r in rows]
        axes[1].scatter([i]*len(values), values, s=16)
    axes[1].set(xticks=range(3), xticklabels=["Radius <=2", "Radius 2-6", "Radius >6"],
                ylabel="Fraction of endpoint response energy (%)", title="True 3-NFE image RDD-DMD, FP32")
    fig.savefig(out/"diagnostics.png", dpi=180)
    plt.close(fig)
    maps = torch.load(out/"dmd_locality_fp32/response_energy_maps.pt", weights_only=True)
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8), constrained_layout=True)
    for ax, energy, row in zip(axes.flat, maps, rows):
        ax.imshow((energy.clamp_min(1e-12).log10()).numpy(), vmin=-8, vmax=-2, cmap="magma")
        ax.set_title(f'Class {row["label"]}, cell {row["changed_parent_cell"]}', fontsize=8)
        ax.axis("off")
    fig.suptitle("Log10 endpoint response energy; only one input 2x2 cell changed")
    fig.savefig(out/"locality_maps.png", dpi=180)
    plt.close(fig)
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    teacher_rows = []
    for row in rdd["records"]:
        if row["family"] == "rdd" and row["data_time"] == .7:
            matching = next(r for r in base["records"] if r["row"] == row["row"] and r["family"] == "rdd" and r["data_time"] == .7)
            teacher_rows.append([row["row"], f'{row["adjusted_prediction_std_rms"]:.4f}',
                f'{matching["adjusted_prediction_std_rms"]:.4f}',
                f'{100*row["adjusted_prediction_std_rms"]/row["adjusted_prediction_mean_rms"]:.1f}%',
                f'{row["analytic_noise_bf16_roundoff_rms"]:.4f}'])
    body = '<h1>Progressive-resolution infra: validation, 2026-10-03</h1>'
    if (out/'root_local_cpu_checks.json').exists():
        body += '<p><a href="#methods-refined">跳到最新：第16节，打磨后的A/B/C</a> · <a href="methods_refined.html">只读最新方法说明</a>（前面保留历史诊断）</p>'
    body += '<p class="warning">状态：尚未找到同时通过“progressive 专属性、≥10% 实测端到端加速、可接受质量”三项要求的新方法。以下是诊断证据，不是加速成功报告。</p>'
    body += '<p>判定标准经用户确认：按生成过程区分，但必须有强普通模型对照。固定分辨率联合去噪不能直接使用尚未完成的最终parent；改造成低频先完成的条件生成后可以使用。不是声称所有普通架构在逻辑上都无法模拟。</p>'
    body += '<h2>1. Stage-birth 数学检查</h2><p>P=平均池化后重复，Q=I−P。在精确 stage 起点，Qx=σQε，不包含干净细节。普通 FM 在非零 data-time 时，Qx=tQx₀+σQε，仍含细节。普通 FM 的初始纯噪声时刻也具有独立性，不能声称该性质仅 RDD 曾拥有。</p>'
    body += table(["20万 Gaussian 样本", "结果"], [["RDD birth parent-only Bayes 额外 MSE", "0"],
        ["普通 FM t=.7，相同 Gaussian 条件假设", ".005737（解析值）"],
        ["最后 stage 只做一次 Euler", "输出均值 .28，而目标 .4；输出方差≈0，而目标 .01"]])
    body += '<p>FM 瞬时条件均值不等于 DMD 的随机 endpoint transport。必须保留或学习细节随机性，不能据此直接删除 fine 网络。</p>'
    body += '<h2>2. 真正 RDD-Wan teacher</h2><p>原始非蒸馏 Wan2.1-1.3B；RDD checkpoint-5003；两个 held-out clips，每个四次 Q-noise 扰动；固定 clean、caption、Pε。只测 conditional velocity，未生成新视频。原始 Wan 也接受相同 RDD 输入，属于 OOD 对照，且训练预算不同，不能当严格架构比较。</p>'
    body += table(["clip", "RDD 修正后 std", "原 Wan 同输入 std", "RDD std / 内容 RMS", "BF16 参考噪声"], teacher_rows)
    body += '<p>RDD 微调降低了对新噪声的敏感性，但剩余变化显著高于舍入参考，现有 teacher 不能精确替换成 parent-only 推理。</p><img src="diagnostics.png">'
    body += '<h2>3. Parent 是否已经不可变？</h2><p>真实 RDD-Wan，Euler50、CFG5、noise_filling、两次额外 boundary 评估，共104次 conditional/unconditional model calls。此处不是 DMD3。</p>'
    body += table(["clip", "middle clean 到最终 pooled latent 的相对变化", "RMSE"],
        [[r["row"], f'{100*r["parent_revision_relative_norm"]:.2f}%', f'{r["parent_revision_rmse"]:.4f}'] for r in contract["records"]])
    body += '<p>提前永久复用 parent decode 需要改变训练契约，当前 RDD 并不保证 D(x_final)=c。普通固定分辨率模型若不固定 parent，不能使用同一个精确执行依赖消除；但旧预测的 speculative decode 仍是必要负对照。</p>'
    body += '<h2>4. Parent 解码隐藏预算</h2><p>相同原生 BF16 WanVAE，GPU0 RTX PRO6000，固定随机 latent，两次热身，四组 ABBA/BAAB。仅测组件；现有 nonlinear VAE 不能直接复用这些 parent 特征。</p>'
    body += table(["输出形状", "full VAE 中位数", "half-spatial parent 中位数"],
        [[r["shape"], f'{r["full_median"]:.3f}s', f'{r["parent_median"]:.3f}s'] for r in decoder["rows"]])
    body += '<p>1280×768×61 旧 RDD 约9.13s，最后 fine NFE 约2.25s。1.57s parent 工作若能替代必需的解码工作且完全重叠，约有旧总时间17.2%的预算；不是已实现提速。真正节省≤min(parent decode, fine NFE)−传输/竞争。新增辅助 parent decode 本身不会让旧 VAE 更快。</p>'
    body += '<h2>5. 真正 RDD-DMD 的细节噪声响应</h2><p>图像 Unified DiT-L/2，DMD checkpoint step6774，原3 NFE=999/599/299，8→16→32；不跳层。四个类别、各两个位置，改变单个2×2 cell 的 zero-mean noise，parent 输入保持到 FP32 舍入量级。FP32 复核；相同单样本重复输出完全一致。</p>'
    body += table(["类别", "cell", "半径≤2响应能量", "半径>6响应能量"],
        [[r["label"], r["changed_parent_cell"], f'{100*r["bands"]["local_radius2"]["energy_fraction"]:.1f}%',
          f'{100*r["bands"]["far_radius6plus"]["energy_fraction"]:.1f}%'] for r in rows])
    body += '<p>存在强局部性，也存在远距离响应；小型 local innovation head 有动机，但不是对现成 teacher 的精确替代。普通 lossless space-to-depth + 同预算 head 必须作为反例测试。</p><img src="locality_maps.png">'
    body += '<h2>6. 下一步候选及淘汰条件</h2>'+table(["候选", "节省", "尚未通过的条件"],
        [["Birth-content trunk + innovation head", "大模型只算 parent tokens，fresh细节由小分支生成", "普通 packed-input 对照可能同样有效；未训练、未计时"],
         ["Committed-parent conditional codec overlap", "必需的 parent 解码与 fine NFE 并行", "需新模型/codec；比较原VAE、新codec串行/并行和通用快decoder；未实现"]])
    body += '<p>不能把后者的“新 codec 更小”与“progressive 允许提前执行”混成同一个贡献。</p>'
    if (out/'clean_decoder_reuse/metrics.json').exists() and (out/'clean_decoder_pooled_control/metrics.json').exists():
        real=read('clean_decoder_reuse/metrics.json')
        generic=read('clean_decoder_pooled_control/metrics.json')
        decoder_rows=[]
        for r,g in zip(real['rows'],generic['rows']):
            for name in ('last4','last6'):
                native=r['medians']['native']
                decoder_rows.append([r['row'],name,
                    f"{r['medians'][name]:.3f}s / {100*(1-r['medians'][name]/native):.1f}%",
                    f"{r['quality'][name]['psnr_vs_native_db']:.2f}dB",
                    f"{g['medians'][name]:.3f}s",
                    f"{g['quality'][name]['psnr_vs_native_db']:.2f}dB"])
        body += '<h2>7. 新实测：VAE residual 复用通过速度，但被普通模型反例淘汰</h2>'
        body += '<p>只替换 decoder 最后4或6个 residual blocks 的更新，当前输入的 shortcut 保留。fine 解码前先实际解码 parent 并捕获更新；全部 capture/缓存/传输/fine 时间计入串行时间。两个真实 RDD 的 latent，不改变 denoiser。</p>'
        body += table(['clip','复用','真实 parent：串行 / decoder 延迟减少','PSNR vs native','普通 pooled-final：串行','PSNR vs native'],decoder_rows)
        body += '<p>普通对照直接池化最终 latent，使用完全相同的复用方法；速度和质量几乎相同。因此这不是符合要求的 progressive 专属方案，也不是已测端到端 ≥10%。4次组件重复是筛查，不是多视频质量评测。约30–33dB的 last6 明显损失眼镜边缘、毛发和细节。</p>'
        body += '<table><tr><th>Native decoder</th><th>Last6 residual reuse（真实 parent）</th></tr><tr><td><img src="clean_decoder_reuse/row0_native.png"></td><td><img src="clean_decoder_reuse/row0_last6.png"></td></tr></table>'
    body += '<h2>8. 新候选：已完成低频 + 当前新生高频</h2>'
    body += '<p>不是跳层：每层宽度和30层均保留。时间翻倍时，c=(x_even+x_odd)/√2、h=(x_even−x_odd)/√2。完成 c 后只生成 h，重建 x_even=(c+h)/√2、x_odd=(c−h)/√2；读取上一阶段实际生成时留下的 K/V。它们不是重新编码的“当前 clean parent K/V”。</p>'
    body += table(['3-NFE / final T=16','第1次 live T','第2次 live T','第3次 live T','最后 Q/K 长度比例'],
        [['Native temporal RDD 4→2→1',4,8,16,'Q=N,K=N'],
         ['Committed-band + ancestor KV',4,4,8,'Q=N/2,K=N'],
         ['必须训练的普通 temporal patch2 反例',2,4,8,'Q=N/2,K=N/2']])
    body += '<p class="warning">这不是现有 checkpoint 的精确优化：需要完成 coarse marginal、冻结低频的条件生成契约、band embedding 和新的蒸馏训练。现有 RDD 的 parent 后续仍改变7–10%。普通 larger temporal patch 可无损打包两个位置，甚至理论计算更少；未排除该反例就不能声称 RDD 独有。</p>'
    body += '<p>适用训练路径：整个三层级联合 DMD，先重建最终完整 latent，再按原 teacher 的全坐标 corruption 加噪，并使用普通 joint teacher/fake score 做分布匹配。不要把普通 teacher 直接当作 clean-DC/noised-detail 的条件 score，也不要用一个 Euler birth velocity MSE 来训练唯一的 fine NFE：后者会消掉高频随机性。coarse 必须同时训练，错误的冻结 DC 无法被 fine 修复。</p>'
    for directory in ('temporal_band_memory','temporal_band_memory832'):
        if not (out/directory/'metrics.json').exists():
            continue
        bands=read(directory+'/metrics.json')
        body += '<h3>硬件筛查 '+html.escape(directory)+'</h3>'
        body += '<p>现有 FastWan 未训练权重，3 NFE，BF16，VSA保留20%且保留coarse output/gate，完整30层，包含 parent KV 生成/写入与VAE。共享GPU环境有其他任务；不是隔离计时或质量等价结论。</p>'
        body += table(['配置','Native RDD+VSA','新 band + ancestor KV'],[
            ['输出',f"{bands['full_shape'][4]*8}×{bands['full_shape'][3]*8}×{(bands['full_shape'][2]-1)*4+1} frames",'相同'],
            ['NFE / t','3 / 1000,757,522','相同'],['宽度 / 层数','1536 / 30','相同'],
            ['VSA保留 / coarse gate','20% / 保留','相同'],['temporal live T','4→8→16','4→4→8'],
            ['去噪中位数',f"{bands['baseline']['denoise']['median']:.3f}s",f"{bands['candidate']['denoise']['median']:.3f}s"],
            ['去噪+VAE中位数',f"{bands['baseline']['total']['median']:.3f}s",f"{bands['candidate']['total']['median']:.3f}s"],
            ['配对总延迟减少','—',f"{bands['reductions']['total']['mean']:.2f}%（4组ABBA/BAAB）"],
            ['质量','未为RDD重新训练的FastWan','彩色块，未训练、未达标']])
        body += '<table><tr><th>Native RDD-VSA（同样未为RDD训练的FastWan）</th><th>新 band 过程（未训练，只验证执行）</th></tr><tr>'+''.join('<td><video controls preload="metadata" style="max-width:100%" src="'+directory+'/'+name+'.mp4"></video></td>' for name in ('baseline','bands'))+'</tr></table>'
        body += '<p>速度门槛和质量/专属性门槛独立：即使这里达到10%，也不等于任务完成。</p>'
    if (out/'ordinary_patch2_1280/metrics.json').exists():
        control=read('ordinary_patch2_1280/metrics.json')
        body += '<h3>反例已经实测：普通 temporal patch2 更快</h3><p>把相邻两帧的全部128个输入坐标映射进同一个1536维token；投影数值秩128，不是平均丢掉一帧。两帧输出都重新预测。相同30层、VSA20、coarse gate、3NFE和原VAE；没有parent KV复用。新的embedding/head未训练，因此只比较硬件成本，不比较生成质量。</p>'
        body += table(['1280×768×61','去噪','去噪+VAE','总延迟减少'],[
            ['本次matched native RDD+VSA',f"{control['baseline']['denoise']['median']:.3f}s",f"{control['baseline']['total']['median']:.3f}s",'—'],
            ['普通双帧打包',f"{control['candidate']['denoise']['median']:.3f}s",f"{control['candidate']['total']['median']:.3f}s",f"{control['reductions']['total']['mean']:.2f}%"]])
        body += '<p>这直接否定“缩短token序列的速度收益只有progressive可获得”。是否有训练后质量优势仍未验证；不能把未训练的两种结构当作质量等价20%加速。</p>'
    body += '<h2>9. 分布式候选：先测真实路由 union，再考虑 kernel</h2><p>只保留20%的每个 query 的 keys，不代表一个GPU只需要20%的KV。Ulysses按head分片但每rank拥有全部query，通常仍需要全部keys。Ring/query-sharded SP必须测每个rank、每个head的远程key并集。若关键路径通信占比 f，远程并集比例 u，乐观延迟减少上界为 f×(1−u)，还没扣规划、拷贝和同步成本。</p>'
    body += '<p><a href="https://arxiv.org/html/2609.32197v1">SparSP</a> 已经用当前 pooled K summary 与 V projection 重叠，按需传KV、优化partition和传输；它是必须比较的普通在线稀疏通信基线。C2F 的候选贡献只能是提前知道真实请求而额外省去关键路径依赖，不是把别人已有的通信优化归给RDD。</p>'
    if (out/'route_layout_audit1280/route_union.json').exists():
        union=read('route_layout_audit1280/route_union.json')
        result=[]
        for stage in (1,2):
            for layout in ('contiguous_sequence','persistent_spatial','persistent_columns'):
                values=[]
                for kind in ('online_vsa','root_lifted_c2f_math_control'):
                    v=[x['remote_key_union_mean'] for row in union if row['stage']==stage and row['kind']==kind
                       for x in row['query_sharded'] if x['ranks']==4 and x['layout']==layout]
                    values.append(statistics.mean(v))
                result.append([stage,layout,f'{100*values[0]:.2f}%',f'{100*values[1]:.2f}%',f'{100*(1-values[1]/values[0]):.2f}%'])
        body += table(['stage','4rank布局','在线VSA远程并集','C2F远程并集','通信字节变化'],result)
        body += '<p>这里1280的双方keep均精确20%；832因ceil稍有差别。按每个方法最好的已测布局比较，三个stage的token加权总通信量只少约7.96%，不是15.8%端到端。还未运行SP kernel；该比例不是计时结果。</p>'
    if (out/'route_layout_audit1280/partition_audit.json').exists():
        partition=read('route_layout_audit1280/partition_audit.json')
        body += '<h3>用parent路由提前确定同一组query归属</h3><p>把每个head/layer/key的需求看作hyperedge，目标是少传不同的KV，而不是简单少切attention边。30层、12heads共同使用一个均衡分片；fine子区域继承parent归属。仅CPU诊断。</p>'
        body += table(['初始化','优化后远程并集','最忙rank总请求'],[
            [name,f"{100*r['optimized']['remote_union']:.2f}%",r['optimized']['maximum_rank_blocks']]
            for name,r in partition['layouts'].items()])
        body += f"<p>整个CPU审计花了{partition['CPU_planning_seconds']:.2f}s，包含三种初始化的优化和其他计数，不是单个部署规划器的延迟。实际部署必须另测chosen-plan规划、D2H、重分片等开销，不能把它们从计时删掉。root结束后才拿到计划，不能倒算root节省。计入root原始布局，乐观字节收益约13.45%，还没有扣上述开销、计算及VAE；没有达到已测10%端到端。普通previous/draft路由也能使用同一优化器。</p>"
    body += '<h2>10. 同parent子区域共享KV读取：端到端已测，未通过门槛</h2><p>lifted route的多个query子区域选中相同key列表。新Triton程序一次处理128或256行query，载入一份fine K/V后供多个子区域使用；每行softmax独立，Q/K/V全部使用当前值。没有复用旧attention输出，没有删coarse output/gate。</p>'
    body += '<p>真实QK/AV算术量不变，可能减少L2读取和load指令；也可能因寄存器压力而变慢。必须比较官方Q64、提前建好的route、只改变CTA排列的Q64，再比较Q128/Q256。普通网络若设计相同共享路由也能用这个executor，严格专属性仍不能直接宣称。</p>'
    for directory in ('ancestry_sparse','ancestry_sparse_core'):
        if (out/directory/'metrics.json').exists():
            kernel=read(directory+'/metrics.json')
            body += '<h3>'+html.escape(kernel.get('timing_scope',directory))+'</h3>'
            body += table(['executor','官方cached-route基线','候选','算子延迟减少','相对L2误差'],[
                [name,f"{r['native_ms']:.3f}ms",f"{r['candidate_ms']:.3f}ms",f"{r['reduction_percent']:.2f}%",f"{kernel['checks'][name]['relative_l2']:.5f}"]
                for name,r in kernel['timings'].items()])
            body += '<p>仅synthetic BF16 QKV + 真实root layer14的route；不是整模型/VAE提速，也没有质量测试。需要实际层集成和强普通共享路由对照。</p>'
    end_rows=[]
    for directory in ('c2f_current_router_61_v2','c2f_exactlift_125'):
        if not (out/directory/'metrics.json').exists():
            continue
        run=read(directory+'/metrics.json')
        end_rows.append([f"1280×768×{(run['full_shape'][2]-1)*4+1}",
            f"{run['online_vsa']['denoise']['median']:.3f}s",f"{run['candidate']['denoise']['median']:.3f}s",
            f"{run['vs_rdd_vsa']['denoise']['mean']:.2f}%",
            f"{run['online_vsa']['total']['median']:.3f}s",f"{run['candidate']['total']['median']:.3f}s",
            f"{run['vs_rdd_vsa']['total']['mean']:.2f}%"])
    if end_rows:
        body += '<h3>实际3NFE，完整30层 + 当前coarse output/gate + 原生WanVAE</h3>'
        body += table(['视频','RDD+VSA去噪','Q128去噪','延迟减少','RDD+VSA总时间','Q128总时间','延迟减少'],end_rows)
        body += '<p>每组双方交错测试，4组；均为热启动，所有route lift/capture均计时。共享服务器不是隔离环境；这些区间只描述本次组间变化，不是跨服务器置信保证。官方Q64和新Q128的相同C2F支持单独对照；VSA对照包含当前在线router。</p>'
    if (out/'c2f_current_router_61_v2/metrics.json').exists():
        run=read('c2f_current_router_61_v2/metrics.json')
        delta=run['ancestral_vs_current_grouped']['total']
        body += table(['61帧强普通路由对照','去噪','总时间'],[
            ['继承parent路由 + Q128',f"{run['candidate']['denoise']['median']:.3f}s",f"{run['candidate']['total']['median']:.3f}s"],
            ['当前Q/K生成共享路由 + 相同Q128',f"{run['current_grouped']['denoise']['median']:.3f}s",f"{run['current_grouped']['total']['median']:.3f}s"]])
        body += f"<p>继承路由相比当前共享路由只少{delta['mean']:.3f}%总延迟（组间区间{delta['lower']:.3f}%到{delta['upper']:.3f}%）。因此不能把query-grouping的通用收益归给progressive。当前共享router对RoPE后的Q/K再池化：计时反例成立，若比较质量还应补pre-RoPE版本。各视频是原FastWan的未训练RDD输入测试，不是本次微调后结果。</p>"
        body += '<table><tr><th>在线RDD+VSA</th><th>继承路由Q128</th><th>当前共享路由Q128</th></tr><tr>'+''.join('<td><video controls preload="metadata" style="max-width:100%" src="c2f_current_router_61_v2/'+v+'.mp4"></video></td>' for v in ('online_vsa','c2f_siblings','current_grouped'))+'</tr></table>'
    if (out/'fast_codec_budget61/metrics.json').exists():
        codec=read('fast_codec_budget61/metrics.json')
        body += '<h2>11. 新方向：已完成parent的提前渲染</h2><p>低频先完成，细节分支随后生成当前h。干净parent先解码为RGB，renderer读取当前c/h，预测flow和RGB残差。它不复用旧去噪block输出；但必须训练一个保证parent不再变化的过程及新codec。codec变小是通用收益；提前启动才是待验证的progressive贡献。</p>'
        body += table(['1280×768×61组件','8次热启动中位数'],[
            [name,f"{row['median_seconds']:.4f}s"] for name,row in codec['timings'].items()])
        body += f"<p>官方Flash-VAED输出相对原WanVAE PSNR为{codec['flash_native_psnr_db']:.2f}dB，这是decoder差异而非生成质量评分。当前renderer只有{codec['renderer']['parameters']:,}参数、未训练，非零h扰动响应仅{codec['renderer']['zero_h_response_rmse']:.6f}；不能拿平滑插帧的外观宣称质量恢复。时间basis及Wan首帧4n+1规则仍需训练验证。</p>"
        body += '<p>强普通对照必须允许parent解码与h-only特征准备并行，并运行最优chunk流水线；直接Flash解码、相同codec串行/并行、普通patch2、旧clean预测的提前解码也需分别比较。未来若开启量化/SP，双方使用同样配置。仅凭这里组件数值不能宣称已测≥10%总加速。</p>'
    if (out/'ppr_cpu_checks.json').exists():
        cpu=read('ppr_cpu_checks.json')
        body += '<h2>12. 当前实现：PPR，Pipelined Progressive Rendering</h2>'
        body += '<p>用户最新要求：只实现，不运行GPU。此前两卡流水线pilot在FP32 decoder attention后端选择处退出，没有端到端计时结果；已修复后端scope代码但没有重启。已确认本任务GPU6/7进程退出。</p>'
        body += '<p>mean/detail约定：c=(z_even+z_odd)/2，h=(z_even−z_odd)/2；重建z_even=c+h，z_odd=c−h。每个阶段的完成c成为不可变条件，下一阶段只预测fresh h。保留c在数学上精确、浮点中有舍入；与上方正交Haar试验的√2归一化不同。</p>'
        body += '<p>执行依赖：完成c → GPU-B解码c，同时GPU-A生成h → 当前c/h预测flow与RGB残差 → 从parent RGB采样并补细节。所有fine Transformer blocks仍计算；没有旧hidden、旧attention、旧KV复用。未来三NFE新条件模型必须训练，不能直接把普通Wan velocity当clean-detail predictor。</p>'
        body += table(['实现','内容','当前验证'],[
            ['committed_band_process.py','均值/细节契约、零均值noise filling、3个clean-endpoint调用、codec蒸馏loss','CPU'],
            ['progressive_renderer.py','当前c/h生成flow+RGB correction；forecast对照可获得完整delta_c','CPU可反向传播'],
            ['progressive_render_triton.py','融合flow插值、parent RGB三线性采样、correction及clamp','只语法检查；未CUDA编译/运行'],
            ['benchmark_progressive_render_pipeline.py','两卡早启动与相同codec晚启动；晚启动允许h准备并行','已实现；未得到成功GPU结果']])
        body += f"<p>CPU单元测试{cpu['tests']}项，passed={cpu['passed']}。没有训练后质量、没有CUDA正确性、没有≥10%流水线计时证明。</p>"
        body += '<p>普通模型强反例：从中间x₀预测c_hat并提前解码，最后renderer得到final c、h、delta_c=c−c_hat修正。两者使用相同forecast-capable renderer；progressive的delta_c=0。必须比较质量和延迟，不能仅因普通forecast不是精确parent就宣布它无效。time-basis、首帧特殊结构、大运动和遮挡仍需验证。</p>'
    if (out/'spatial_detail_cpu_checks.json').exists():
        cpu=read('spatial_detail_cpu_checks.json')
        body += '<h2>13. 用户提出的新方向：四个子位置只预测三个自由度</h2>'
        body += '<p>PPR暂不作为符合要求的方案。此节只做数学推导和CPU检查，没有GPU运行。若最终mean=c，则x4=4c−x1−x2−x3。例如c=10，前三个值8、9、11，第四个必须12。但现有RDD没有这个不可变条件：两段真实teacher记录中最终parent仍变化7.36%和10.12%。</p>'
        body += '<p>采用三个零均值正交细节分量：x=c+S h，S的四行依次为(1,1,1)、(−1,1,−1)、(1,−1,−1)、(−1,−1,1)。此mean归一化下独立细节noise标准差为sigma/2，不是三个pixel各自独立采Gaussian再强行算第四个。</p>'
        body += table(['层级','四像素/patch含义','实际可能节省'],[
            ['仅缩小输出head','Wan本来把2×2latent放入一个token','不会省整个Transformer的25%'],
            ['三个detail tokens','父2×2patch对应fine4×4区域；3个2×2detail band代替4个fine patch tokens','25%该stage线性token工作上限，尚未实现完整Wan'],
            ['缓存低频linear anchor','输入hidden分解成mean和3个detail；已知mean的W1结果从parent读取','CPU已实现FFN高频路径；全模型需新norm/modulation契约']])
        body += '<h3>已实现的FFN代数原型</h3><p>预存a=c_hidden W1+b1。计算三行detail_hidden W1，重建四行当前preactivation；四行GELU全部重新算。先投影GELU到三个高频分量，再计算三行W2。CPU验证与相同输入的dense四行FFN之高频输出及梯度一致；并非与普通FFN完整输出一致。</p>'
        body += f"<p>CPU检查{cpu['tests']}项，passed={cpu['passed']}。独立检查还验证了普通模型也能把LL+每个detail打包到相同三个token并无损保留全部四个band；因此三token数量本身不能证明progressive独有。</p>"
        body += '<p>额外陷阱：fine每个child的LayerNorm平均值不等于coarse LayerNorm，当前timestep modulation也可能不同。不能直接用旧Wan preactivation冒充当前精确mean；需要scale-separated normalization和训练过的固定anchor结构。普通模型可计算当前mean W1后使用相同高频W2路径，因此只能把确实省掉的parent线性工作归给progressive，不能将两个GEMM的全部25%都说成独特贡献。</p>'
        body += '<p>强对照：相同3band geometry/width/depth的普通joint学生输入当前noisy LL，每个band同时预测detail和LL，再平均LL预测重建；普通current-mean/forecast-mean线性对照；原始RDD+VSA。同一fastcodec、相同资源和质量要求。空间421两次birth的线性token量可从1.3125N降到N（23.81%上限），但parent读取、band layout、VSA padding、attention、norm和VAE均未计入。不是速度实测或完整方法。</p>'
    body += '<p><a href="https://arxiv.org/abs/2208.05003">WSGM</a> 已有条件wavelet因子化；<a href="https://arxiv.org/abs/2211.16152">WaveDiff</a> 是wavelet空间压缩的重要普通对照。不能把Haar分解本身当新方法。</p>'
    if (out/'anchor_nvfp4_cpu_checks.json').exists():
        fp4=read('anchor_nvfp4_cpu_checks.json')
        compiled=read('anchor_nvfp4_compile_checks.json') if (out/'anchor_nvfp4_compile_checks.json').exists() else {}
        body += '<h2 id="nvfp4">14. 算子级原型：固定 coarse anchor + 当前细节 + NVFP4 融合</h2>'
        body += '<p class="warning">已实现CPU参考和CUDA producer，未运行GPU；不是可用的完整Wan加速器。未实现原生NVFP4 GEMM接入，未微调、未验证生成质量，也没有≥10%计时。两个agent分别提出和反驳方案后，判定单独这条路线尚不满足目标。</p>'
        body += '<p>四个子位置共享的是clean上采样分量。noise_filling之后实际输入为alpha·U(c)+sigma·epsilon，epsilon含四个独立噪声。因此实际输入不是四行完全相同，不能直接删掉三行矩阵乘法。原始Wan的child LayerNorm也不保证每层mean等于旧parent hidden；本原型必须训练新的固定anchor契约，不能插入现有checkpoint声称精确等价。</p>'
        body += '<pre>coarse stage: a = c_hidden W1 + b1  [higher precision, cached]\n\nfine stage:   h[3,D] -- W1 --> v[3,G]\n                               |\n                         a + S v [4,G]\n                               |\n                       FOUR fresh GELUs\n                               |\n                    Sᵀ/4 projection [3,G]\n                               |\n                 group16 amax + E4M3 scale\n                 E2M1 conversion + byte pack\n                               |\n                   native NVFP4 W2 [not wired]\n\nCUDA prototype fuses the middle stages; no 4×G temporary.</pre>'
        body += table(['部分','具体实现','状态'],[
            ['CPU oracle','anchor_nvfp4_reference.py；E2M1 RNE、低nibble优先、group16 E4M3、global FP32 scale、tail padding和clip计数',f"{fp4['tests']}项CPU单元测试通过"],
            ['CUDA producer','anchor_nvfp4_epilogue.cu；每16线程共享一次parent读取，四次fresh GELU，三band投影及FP4打包','已编译SM120a PTX和cubin；0 spill；资源见Compiler checks；未launch'],
            ['格式检查','anchor_fp4_host_check.cpp；只调用CUDA header的CPU转换函数','codebook、midpoint ties和E4M3编码一致'],
            ['复现','compile_anchor_nvfp4.py；只编译和运行CPU converter','不创建CUDA context，不运行kernel'],
            ['完整执行缺口','W1/W2 NVFP4 GEMM、CUTLASS scale swizzle、模型norm/训练、GPU正确性和计时','没有实现/验证；logical scales不能直接冒充硬件layout']])
        body += '<p>NVFP4不是把BF16转成FP4再转回BF16做GEMM。后续必须使用真正的SM120 block-scaled Tensor Core GEMM；本producer只负责产生激活和逻辑scale。global scale若用当前amax，必须计入额外扫描；若固定校准，必须报告clip。非线性后的detail不是Gaussian，不能沿用parent scale省略实际block amax。</p>'
        body += table(['CPU合成审计（不是图像质量）','结果'],[
            ['shape / 数据','parent[128,8960] std .5；detail[128,3,8960] std .2；seed74'],
            ['仅epilogue量化相对L2',f"{100*fp4['isolated_synthetic_epilogue_relative_l2']:.2f}%（未量化W1/W2）"],
            ['超出rounded local scale范围的值',f"{fp4['clipped_values']:,}/{fp4['values']:,}；包括scale RNE向下取整后的轻微饱和"],
            ['普通current-mean对照','相同mean、detail和scale时产生逐byte相同的输出；只有current mean W1计算不能直接缓存'],
            ['独特省掉的线性工作上限',f"{100*fp4['budget']['unique_removed_linear_fraction']:.2f}%（D1536/G8960，含self和text Q/output线性；不含attention、norm、VAE和缓存）"],
            ['anchor缓存示例',f"Nc15360、G8960、30层BF16：{fp4['budget']['parent_preactivation_cache_gib']:.2f}GiB；每层读取约275MB"]])
        body += '<p>重要：三行W2、融合GELU/投影/量化是普通模型也能获得的收益。应给普通current-mean、all-mode NVFP4和混合精度对照相同融合。unique部分仅是已完成parent提供的mean W1，不是整个FFN的25%。上述约9%的线性FLOP比例不是硬件延迟上限；混合精度、带宽和kernel形状需另测，当前不支持10%总加速承诺。普通全FP4也不必保留BF16 DC，不能人为削弱它。</p>'
        body += '<p>因此保留这个独立实验原型用于硬件验证，不作为主方法推荐，不改现有RDD/FastVideo执行路径。若后续继续，应先验证是否存在progressive更适合低比特的质量优势，再考虑扩展固定anchor算子；不能靠把通用NVFP4加速归给RDD完成目标。</p>'
        body += '<p>Primary sources: <a href="https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms">CUTLASS SM120</a>提供NVFP4硬件GEMM路径；<a href="https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html">Transformer Engine NVFP4</a>说明格式及支持范围，不能假设SM120支持其训练autocast；<a href="https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html">CUDA FP4 conversion API</a>用于实际转换；<a href="https://research.nvidia.com/labs/eai/publication/svdquant/">SVDQuant/Nunchaku</a>是高精度分支+低比特融合的普通模型强对照。</p>'
        body += '<p><a href="anchor_nvfp4_cpu_checks.json">CPU metrics</a> · <a href="anchor_nvfp4_compile_checks.json">Compiler checks</a></p>'
    if (out/'stage_birth_cpu_checks.json').exists():
        birth=read('stage_birth_cpu_checks.json')
        bc=read('stage_birth_compile_checks.json') if (out/'stage_birth_compile_checks.json').exists() else {}
        body += '<h2 id="birth-infra">15. 新一轮 infra 筛选：stage-local birth / elastic SP 原型</h2>'
        body += '<p class="warning">沿用用户“不运行GPU”的要求。本轮没有GPU实验、没有新增视频或≥10%计时结果。只实现CPU规划/reference、CUDA birth算子，离线编译及CPU测试；尚未接入FastVideo/NCCL。不是已达标的主方法。</p>'
        body += table(['方向','真正节省什么','反例 / 当前结论'],[
            ['NVFP4 + coarse anchor','在新固定anchor模型中省掉parent W1，融合非线性/投影/量化','上一节独特线性工作约9%；普通current-mean/all-NVFP4也能用融合。当前Wan noise/LN不满足固定anchor，不能直接零成本接入'],
            ['fine ΔK/ΔV 低比特通信','若细节容易量化，可以减少通信字节','没有稀疏/低秩时QΔK及AΔV仍要算；普通当前均值或时间差也能压缩。尚无progressive专属质量优势证据'],
            ['ROOT全局memory + 当前局部attention','保留所有blocks/FFN；删掉fine远程原始KV交换和多数attention pairs','强普通current pooled bank使用同样kernel也省大部分工作。需要训练，不是精确替换，暂不作为达标推荐'],
            ['CUDA graphs / persistent weights','减少launch开销或重复准备','普通3NFE同样能capture；padding到full shape可能抹掉RDD收益。通用优化，不作为独特贡献'],
            ['stage-local birth + elastic SP','直接写局部patch；小stage少卡、大stage多卡；低维latent重分片','已做算子原型；收益未知。动态SP已有文献，不宣称其本身原创或普通模型不能用']])
        body += '<h3>已实现的算子：在目标卡上扩分辨率，不先造完整fine latent</h3>'
        body += '<pre>上一个stage：原RDD drift inverse → clean parent c [B,16,Tc,Hc,Wc]\n\nparent patch owners\n       |  仅发送目标rank需要的parent voxel，BF16\n       v\n目标rank的一次CUDA birth程序：\n    读取parent packet\n    整数repeat上采样\n    按global pixel地址生成Philox噪声\n    x = alpha · U(c) + (1-alpha) · epsilon\n    直接写 [B, local patch tokens, 16, 4]\n       |\n       v\n原始Wan patch projection → 原始完整Transformer\n\n候选SP: root 1卡 → middle 2卡 → fine 4卡\n实际SP必须由实测profile选，不预先假定1→2→4最优。</pre>'
        body += '<p>仅支持Wan patch(1,2,2)与整数repeat空间/时间扩张。转换输入必须是原RDD恢复出的clean endpoint，不是带噪stage末状态；保留原drift inverse、边界额外评估和noise_filling。所有卡从开始常驻同一组权重，只改SP，不宣称动态TP无需搬权重。本算子不跳层、不缓存旧block输出，不改变训练目标。</p>'
        body += table(['代码','已实现内容','未实现部分'],[
            ['stage_birth_reference.py','parent voxel包、源rank归属、分片无关Philox、birth/patch CPU oracle、穷举SP计划','真实SP时延表；NCCL执行'],
            ['stage_birth_rng.h / stage_birth_rng_host_check.cpp','Philox4x32-10；3组官方known-answer检查','没有GPU数学函数正确性检查'],
            ['stage_birth_patch.cu','BF16输入输出；相邻两pixel共用一次Philox；三种舍入模式；上采样/noise filling/patch packing融合','launch安全wrapper；GPU projection/runtime；GPU验证'],
            ['test_stage_birth_cpu.py / compile_stage_birth.py',f"{birth['tests']}项CPU检查passed={birth['passed']}；SM120a PTX/cubin compiled={bc.get('compiled',False)}",'没有kernel launch和速度证据']])
        body += '<p>编译资源记录见Compiler checks，不是GPU occupancy/性能实测。最新CPU检查加入原始Conv3d权重顺序、BF16中间乘法/加法舍入和直接传入原scheduler sigma；此前FP32融合模式仍单独保留，不能混为相同舍入口径。使用新coordinate-addressed RNG；不复现torch.randn原始seed序列，CPU/GPU log/sincos及数值路径也不承诺逐bit一致。未来双方必须使用同一RNG与完整舍入口径，另保留native boundary质量对照；GPU正确性尚未验证。“不要求训练”仅指没有新网络参数，不是已证明生成完全不变。完整最新解释见第16节。</p>'
        body += table(['CPU几何计数（BF16、B1/C16）','发送的远程parent值','目标rank映射总量','说明'],[
            [f"{r['parent']} → {r['fine']} / SP{r['sp']}",f"{r['remote_value_bytes']/1048576:.4f} MiB",f"{r['local_map_bytes']/1048576:.4f} MiB",'非通信计时；索引计划应预建'] for r in birth['packet_geometry']])
        body += '<p>不得用“BF16 hidden D1536比latent C16大96倍”宣称加速96倍：最优静态SP基线同样可在NFE边界只移动C16 latent。本轮packet统计只验证几何，metadata/索引读取、收发packing、group同步及原patch projection仍有成本。</p>'
        body += '<h3>公平速度判定</h3><pre>T(schedule) = startup + Σ stage_time(stage, SP)\n            + Σ measured_birth_and_reshard(source_SP, destination_SP)\n            + final assembly + same VAE time\n\nbaseline = min(T(fixed legal SP1/2/3/4))\nreduction = 1 - T(best elastic schedule) / baseline</pre>'
        body += '<p>所有静态/动态候选均使用相同模型、dtype、VSA、kernel、RNG、VAE、batch和可用GPU预算；给静态基线同样的birth融合。计入同SP的birth成本及不同SP的group/收发成本。规划器的虚构数字只用于单元测试，不写成提速证据；目前没有真实profile，因此加速百分比为null。还需与多请求DP吞吐、最优固定SP及普通固定resolution动态SP强对照比较。</p>'
        body += '<p>与progressive的关系：同一请求的token量随stage变化，使每个stage不同SP有结构性动机；普通固定shape无此分辨率增长，但仍可因负载/缓存/通信变化调SP。因此不能声称“普通diffusion绝对不能用”。<a href="https://arxiv.org/html/2604.04335v1">GenServe</a>已实现弹性SP及紧凑latent重配置；<a href="https://arxiv.org/abs/2510.01565">TetriServe</a>已有step-level动态SP。本方向是已有系统思想针对RDD birth的实现，不足以单独建立novelty。</p>'
        body += '<h3>ROOT memory + fresh local attention：最新设计已移至第16节</h3>'
        body += '<p>此前设想保留30层ROOT K/V，示例约0.659/0.165 GiB；现在改为一份ROOT最终feature＋每fine层临时投影。旧25.7%/35.7%只计self-attention/FFN且未扣新投影，不作为当前预算。第16节使用包括text矩阵和所有4卡重复ROOT投影的更新计数；没有实测速度。</p>'
        body += '<p><a href="stage_birth_cpu_checks.json">CPU checks / packet计数</a> · <a href="stage_birth_compile_checks.json">Offline compiler / CPU RNG checks</a></p>'
    if (out/'root_local_cpu_checks.json').exists():
        from build_method_explanation import section
        body += section(out)
    body += '<h2>Primary references</h2><p><a href="https://arxiv.org/html/2511.16117v1">DCS-LDM</a> 已有 level-causal tokenizer/DiT 与 coarse KV caching；<a href="https://github.com/Aoko955/Flash-VAED">Flash-VAED</a> 是必须考虑的通用 Wan 快 decoder 对照；<a href="https://github.com/OpenImagingLab/FlashVSR">FlashVSR</a> 和<a href="https://arxiv.org/html/2609.37831v1#S4.SS4">ReCaVSR</a>提供LR-conditioned fast decoder参考。提前执行必需parent工作必须与最优chunk pipeline/快速codec公平比较；不能把更小codec的通用收益归给progressive。以上不是我们的实测结果。</p>'
    body += '<h2>Raw evidence</h2><ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in (
        "birth_math.json", "teacher_rdd/metrics.json", "teacher_base/metrics.json", "parent_contract/metrics.json",
        "decoder_bound/metrics.json", "dmd_locality_fp32/metrics.json", "temporal_band_memory/metrics.json",
        "temporal_band_memory832/metrics.json", "ordinary_patch2_1280/metrics.json",
        "route_layout_audit1280/route_union.json", "route_layout_audit1280/partition_audit.json"))+'</ul>'
    style = 'body{font:16px/1.6 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#202530}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccd1d9;padding:8px;text-align:left}th{background:#eef2f7}img{max-width:100%}.warning{background:#fff1cb;padding:16px}h2{margin-top:36px}'
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Progressive infra validation</title><style>'+style+'</style>'+body, encoding="utf-8")
    print(out/"index.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="eval_results/progressive_infra_20261003")
    main(p.parse_args())
