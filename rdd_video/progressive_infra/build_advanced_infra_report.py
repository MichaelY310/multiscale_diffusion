"""Research extension: CPU evidence, not GPU or serving-performance claims."""
import html
import json
from pathlib import Path


def table(headers,rows):
    cell=lambda tag,x:f'<{tag}>{html.escape(str(x))}</{tag}>'
    return '<table><thead><tr>'+''.join(cell('th',x) for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(cell('td',x) for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def section(out):
    evidence=out/'advanced_infra_cpu_checks.json'
    if not evidence.exists():return ''
    checks=json.loads(evidence.read_text(encoding='utf-8'))
    c=checks['communication']['3840']
    body='<h2 id="advanced-infra">18. C 的进阶：算子/通信与推理系统分开研究</h2>'
    body+='<p class="warning">2026-10-03：保留方法名C，暂停GPU。这是论文/官方源码调研和7项CPU契约检查，不是新的速度实验。没有≥10%实测、NCCL执行、量化kernel、serving runtime或新生成视频。已有C Triton forward也仍未GPU编译/运行。</p>'
    body+='<h3>方向一：C-SP——在fine KV跨卡搬运之前改变执行布局</h3>'
    body+='<p>每卡拥有完整空间窗口和全部heads，所有时间帧留在同一窗口内。fresh fine Q/K/V、gate、FFN和文字cross-attention继续算。local attention只读本卡KV，远程信息读此前生成ROOT feature的当前层投影；ROOT/local共享一次softmax。保留原VSA当前cube64全局coarse output及加法gate，不删除它制造速度优势。</p>'
    body+=table(['数据','原VSA Ulysses','C-SP候选'],[
        ['fine Q/K/V/gate','正向all-to-all，重分heads','窗口内计算，不做这次global交换'],
        ['fine attention output','反向all-to-all','本地已有全部heads的输出'],
        ['当前VSA coarse output','原语义','本地pool cube，交换当前Kc/Vc；本地Qc算全局摘要attention'],
        ['前一stage ROOT','不使用','广播一份F_root；逐层临时projection/norm/time/RoPE仍收费'],
        ['改变什么','在线selected fine pairs','local＋ROOT连接图，需训练'],
        ['不改变什么','—','宽度、层数、FFN、文字attention、采样步数、VAE']])
    body+='<p>切入点是FastVideo <code>fastvideo/attention/layer.py:DistributedAttention_VSA.forward</code>的fine QKVG交换之前；不能先完成旧Ulysses再调用C。projection直接写目标tile布局，ROOT只读bank不按窗口复制。保留物理RoPE坐标和完整cube ownership。通信/compute overlap必须满足数据依赖，不能用异步掩盖真实等待。</p>'
    body+='<pre>Native remote sent bytes / rank / layer\n  = 5 · N · D · w · (P−1)/P² [Q,K,V,GATE forward + O back]\nC CURRENT64 all-gather remote sent bytes / rank / layer\n  = 2 · (N/64) · D · w · (P−1)/P\nROOT feature logical delivery once = (P−1) · R · D · w\nROOT projection all ranks/layer = 4 · P · R · D² FLOPs</pre>'
    mib=lambda x:f'{x/2**20:.3f} MiB'
    body+=table(['分析示例：N61440 / D1536 / BF16 / SP4 / 30层','payload，不是时间'],[
        ['Native VSA每rank每层remote sent',mib(c['native_remote_sent_bytes_per_rank_layer'])],
        ['C当前coarse summaries每rank每层',mib(c['c_current_summary_remote_sent_bytes_per_rank_layer'])],
        ['该项per-layer字节减少',f"{100*c['c_per_layer_payload_reduction']:.1f}%"],
        ['Native该fine stage每rank',mib(c['native_remote_sent_bytes_per_rank_fine_stage'])],
        ['C该fine stage每rank当前摘要',mib(c['c_current_summary_remote_sent_bytes_per_rank_fine_stage'])],
        ['ROOT一次跨3peer逻辑交付总量',mib(c['root_feature_logical_delivery_bytes_once'])],
        ['ROOT单层KV scratch',mib(c['root_layer_kv_scratch_bytes'])],
        ['普通current N/16 pooled-bank同图每rank每层',mib(c['ordinary_current_bank_remote_sent_bytes_per_rank_layer'])],
        ['普通N/64 bank复用已有当前coarse summaries','额外bank通信0；这是更强反例'],
        ['单视频/含VAE实测加速','未测；97.5%字节减少不是97.5%加速']])
    body+='<p>公式只算平衡、无padding、B1、完整窗口且无shift/halo时的远程发送payload；不是收发总量或网络hop。此例60个full-time 8×8窗口，每rank15个。启动/拓扑/带宽/重叠、ROOT源瓶颈、普通text/TP通信、root/middle工作及VAE都未计时。</p>'
    body+='<p><b>强普通对照：</b>当前fine已投影K/V按4×4空间pool到同尺寸N/16 bank，用相同window ownership和joint-softmax，也不用raw fine A2A，且无需额外D×D projection。ROOT独有增量是少刷新/交换bank及可能更好的条件信息，不是所有通信收益。普通N/64 bank可直接复用当前coarse summaries。若普通bank在同质量下更快，progressive独特贡献就不成立。</p>'
    body+='<h4>量化：附加优化，不作为第二个保证10%的方法</h4>'
    body+='<p>F_root固定，不代表K_l(t)固定：当前layer/time的modulation、Wk/Wv、QK norm及RoPE可能不同。只能逐层即时量化，或提前付出对应layer/t的projection、量化和缓存费用；不能把一份旧packed KV供所有层/时刻使用。BankVersion契约包含producer、weights/LoRA、layer、t、RoPE和precision layout。</p>'
    body+='<p>可以试ROOT K低比特、V较高精度，先保留local/CURRENT64 BF16；普通pooled bank也要获得相同kernel。单F_root压缩只省几十MiB，难独立达到10%端到端。scale必须从当前tensor验证，不能继承parent amax。分别center ROOT/local K时必须恢复各bank的q·mean(K)/√d，否则改变两bank之间的概率质量。CPU只验证centering恒等式，不验证FP4精度/速度。</p>'
    body+='<p><a href="https://arxiv.org/html/2609.33135v1">MpFA</a>指出softmax后量化和搬运可能抵消全FP4收益，采用QK4/PV8；其B200结论不能照搬surfi2。<a href="https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms">CUTLASS SM120</a>的MMA/布局契约不同，需要为RTX PRO 6000单独实现验证。</p>'
    body+='<h3>方向二：progressive stage-bucket推理调度</h3>'
    body+='<p>目标是多请求吞吐/p95，不自动改善单视频。ROOT/middle/fine分别维护shape-compatible队列；小stage合大batch，fine按实测显存和pair预算拆小batch。每请求保留独立seed/RNG、scheduler、stage、ROOT handle、取消状态，禁止跨请求混用ROOT。权重常驻，双方offload相同。</p>'
    body+=table(['阶段','例：每请求tokens','token容量示例batch','上线还要检查'],[
        ['ROOT',3840,16,'pair预算、峰值显存、deadline、实际batch吞吐'],
        ['middle',15360,4,'同上；birth/重分片/组切换'],
        ['fine',61440,1,'同上；bank生命周期、相同VAE']])
    body+='<p>16/4/1只是固定61440-token容量的算术示例，不是profile最优配置；capacity同时受token、B·N²/pair和实测memory限制。不能把小stage padding到fine来组成混合batch。</p>'
    body+='<p>执行算法：①进入对应stage bucket；②按profile/deadline选batch和SP；③同组请求native-sized执行；④boundary/noise_filling后移入下一桶；⑤独立decode与释放cache。预建有限SP组和各stage shape graphs，collective同序；预取依赖bank-ready事件。3NFE只有两个boundary，换组和排队可能不划算，必须允许最佳固定SP获胜。</p>'
    body+='<p>对照最佳固定SP＋相同stage batching、已有step-elastic scheduler＋相同融合、普通fixed-resolution continuous batching。报告requests/s、p50/p95、GPU-seconds/video、峰值显存、cold/steady-state与完整video延迟；不以吞吐倍率冒充单视频加速。<a href="https://arxiv.org/abs/2604.04335">GenServe</a>、<a href="https://arxiv.org/abs/2606.13501">GF-DiT</a>已有弹性资源分配，概念不是原创。</p>'
    body+='<h3>已有工作与官方实现</h3>'
    body+=table(['工作/仓库','已有优化','区别/集成缺口'],[
        ['HPDM (CVPR2024)','previous-scale深层activations条件＋cache','DCF grid_sample/concat/projection，不是C的joint-softmax；跨尺度feature cache已有'],
        ['HiStream','低/高resolution生成＋dual history caches','AR历史；从最终HR/downsample构造双cache，不等于本stage ROOT bank'],
        ['Jenga','progressive resolution＋dynamic attention carving','已经组合多resolution与稀疏推理'],
        ['PSA','当前KV多级pool、multi-level masks与kernel','无previous生成stage；强普通attention对照'],
        ['SparSP','依赖感知placement、summary路由、KV直传','summary只定route；仍传选中full KV，不是C compressed-value bank'],
        ['FastVideo','SP/VSA、regional compile、优化Ulysses','不拿naive A2A作唯一baseline；所检文件未确认原生C/stage-bucket'],
        ['vLLM-Omni','Helios pyramid sampler、step执行、通用SP/连续batch','Helios step mode当前max_num_seqs=1；Wan VSA当前不兼容SP'],
        ['SGLang SPD','spectral progressive生成','官方目前SP/torch.compile不兼容；是集成缺口，不是没有progressive inference']])
    refs=[('HPDM §4.2/4.4/4.5','https://arxiv.org/html/2406.07792v1'),
          ('HiStream','https://arxiv.org/html/2512.21338v1'),('Jenga','https://arxiv.org/html/2505.16864v1'),
          ('PSA','https://arxiv.org/html/2512.04025v1'),('SparSP','https://arxiv.org/html/2609.32197v1'),
          ('FastVideo VSA/SP','https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/attention/layer.py'),
          ('vLLM-Omni Helios','https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/helios/pipeline_helios.py'),
          ('Continuous batching limitations','https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/'),
          ('Wan VSA constraints','https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/attention_backends/fastvideo_vsa/'),
          ('Fused Ulysses baseline','https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/distributed/a2a_permute.py'),
          ('SGLang progressive limitations','https://github.com/sgl-project/sglang/blob/main/docs/docs/sglang-diffusion/progressive_resolution.mdx')]
    body+='<p>'+ ' · '.join(f'<a href="{url}">{title}</a>' for title,url in refs)+'</p>'
    body+='<h3>当前决定与验证gate</h3>'
    body+='<p>优先C-SP，再做stage-bucket，量化暂定位附加。先验证GPU算子和真实QKV布局，再测router/pool/projection/通信关键路径。只有同checkpoint、dtype、3NFE、VAE的RDD＋VSA对照下，端到端延迟稳定减少≥10%才记达标。微调后与current-pooled/previous-full controls比较quality–latency；训练不保证消除跨窗细节损失。</p>'
    body+=table(['CPU新增工作','结论'],[
        ['7项测试','window/cube所有权、payload与普通反例、容量、layer/time身份、独立bank centering'],
        ['此前FLOP预算修正','加入VSA D→D gate projection；仍不是延迟'],
        ['NCCL/quantized/serving runtime','未实现；不以CPU contracts冒充'],
        ['GPU/实际加速','没有运行；measured_speedup=null']])
    body+='<p><a href="advanced_infra_cpu_checks.json">CPU evidence</a> · <a href="../../rdd_video/progressive_infra/advanced_infra_plan.py">CPU contracts</a> · <a href="../../rdd_video/progressive_infra/test_advanced_infra_cpu.py">Tests</a></p>'
    return body


if __name__=='__main__':
    out=Path('eval_results/progressive_infra_20261003')
    style='body{font:16px/1.65 system-ui;max-width:1050px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccd1d9;padding:8px;text-align:left}th,pre{background:#eef2f7}pre{padding:16px;overflow:auto}.warning{background:#fff1cb;padding:16px}'
    (out/'advanced_infra.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C: operator and serving extensions</title><style>'+style+'</style>'+section(out),encoding='utf-8')
    print(out/'advanced_infra.html')
