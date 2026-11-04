#!/usr/bin/env python3
"""Publish completed CPU evidence into existing handoffs, preserving history."""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/forward_backward_20261012'
MARK='<!-- CPU_PARITY_20261012 -->'
END='<!-- /CPU_PARITY_20261012 -->'

def prepend(path,body):
    original=path.read_text()
    original=re.sub(re.escape(MARK)+r'.*?'+re.escape(END)+r'\n*','',original,flags=re.S)
    path.write_text(MARK+'\n'+body.strip()+'\n'+END+'\n\n'+original)

def main():
    global BASE
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=BASE)
    BASE=parser.parse_args().evidence.resolve()
    assert BASE.is_relative_to(ROOT/'evidence') and BASE!=ROOT/'evidence'
    audit=json.loads((BASE/'AUDIT.json').read_text())
    cleanup=json.loads((BASE/'cleanup.json').read_text())
    assert audit['status']=='passed' and audit['training_profiles']==33 and cleanup['applied']
    report=ROOT/'CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md'
    ledger=ROOT/'CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md'
    summary=f'''> **2026-10-12 最新 CPU 逐层验收（覆盖下文修复前状态）**：12种训练/codec/蒸馏方法、33份选中配置全部完成当前源码的逐层 forward/backward 对照；捕获的模块输入、输出、activation gradient和处理前parameter gradient全部shape/dtype/原始字节相同、max_abs=0。31份image配置使用完整发布backbone和输入网格；另有完整SD式VAE及30层Wan1.3B（视频网格缩小）。12方法各128步、12扩展case各128步也全部重跑。两侧复用作者backbone，独立组装目标/loss/backward；CPU FP32、共用非零随机初始化/合成模型域输入，不能外推所有分支、原CUDA kernel或论文最佳质量。
>
> 真实修复包含Relay增强/结构/缩放/EMA、PPFlow路径运算树、RDD阶段汇总/VP噪声/FM读出、LapFlow/DFM归约与CLI、PixelFlow packing/API、MDM RNG/累积、DCT编码/解码/label、reader epoch/drop_last，以及Diffusability广播顺序。17源快照1211个选定文件未修改。SPEED/NAMI/MrFlow公开推理逐层通过，**作者backward N/A**。新找到CAT官方补充4份Python源码，缩小和完整512配置codec输出/KL/诊断adjoint通过；作者diffusion trainer/训练loss未发布，不计入12种训练核心。
>
> [完整报告]({report})、[148项逐项记录]({ledger})、[机器索引]({BASE/'AUDIT.json'})；重跑 `unified_multires/run_cpu_layer_parity.sh`。**完整官方CLI认证仍为0/33，论文最佳主表正式复现仍为0行；148项没有全部关闭**。原CLI生命周期、真实资产/数据、CUDA/DDP和质量实验的前提逐项保留，没有用合成CPU检查冒充完成。39对临时checkpoint的14,948个完整状态张量及param_groups/scalars已重新加载逐字节对照；随后清理{cleanup['deleted_files']}个临时payload、{cleanup['bytes']/2**30:.3f}GiB，报告/日志/配置/hash保留。仅CPU、CUDA未初始化、无正式长训练/模型数据下载/dirty FastVideo或共享环境修改。旧GPU授权不恢复。
'''
    for path in (ROOT.parent/'MIGRATION_HANDOFF_ZH.md',ROOT.parent/'CONTINUE_PROMPT_ZH.md',ROOT.parent/'docs/PROJECT_HANDOFF_ZH.md'):
        prepend(path,summary)
    prepend(ROOT/'README_ZH.md',f'''**2026-10-12 当前CPU逐层对照已执行完成**：[完整结果](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)、[原148项逐项处理](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md)。12种训练/codec/蒸馏方法、33份配置（31完整image网络+完整SD式codec+完整30层Wan；Wan缩小网格）捕获的模块输入/输出、activation gradient和处理前parameter gradient均逐字节相同、max_abs=0；12方法128步和12扩展128步全部重新通过。共用作者backbone、随机非零权重、合成输入、CPU FP32；不是独立网络重写或最佳质量认证。

SPEED/NAMI/MrFlow公开推理通过、作者backward N/A。CAT官方补充4份源码新增codec适配：缩小网络16case、完整classification.py512配置8case逐层/诊断adjoint通过，无公开diffusion trainer，未计入12种训练核心。33个完整native CLI仍未认证，论文最佳主表未正式复现。原文未修复条目以本轮ledger为准。

重跑：`bash unified_multires/run_cpu_layer_parity.sh`；只用现有CPU环境，完整检查可能持续数小时。39对临时checkpoint已实际重载并比较完整model/EMA/Adam/param_groups/scalars，之后清理{cleanup['bytes']/2**30:.3f}GiB测试payload，保留全部报告/hash。下文历史checkpoint路径不保证仍有payload，不能用其恢复正式训练。''')
    prepend(ROOT/'ORIGINAL_UNIFIED_GAPS_TODO_ZH.md',f'''**2026-10-12 当前状态更新**：本文件下方是2026-10-10的原始审计条目与反例描述，保留为历史记录。它们的**当前状态以 [148项逐项处理记录](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md) 为准**，不能再把旧“尚未修复”文字当作当前实现结论。

本轮已实际完成12种训练/codec/蒸馏方法、33份选中配置的当前逐层forward/backward对照：31份完整image backbone/输入网格、完整SD式VAE、完整30层Wan（缩视频网格）；实际捕获的输出、activation gradient、处理前parameter gradient全部逐字节相同、max_abs=0。12方法128步及12扩展128步全部重跑，39对完整状态checkpoint已重载验证并清理临时payload。另3个公开推理workflow通过，作者训练backward N/A；CAT新公开补充codec完成缩小/完整512诊断对照，但不是其diffusion trainer复现。见 [完成报告](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)。

**未把148项全部标成关闭**：这份原清单同时包含私有/未发布trainer、未知best-run、真实资产/数据、CUDA多卡、正式长训练和主表FID；仅CPU forward/backward不能完成这些验收。完整官方CLI仍0/33认证、最佳主表仍0行正式复现。所有已经修掉的CPU错误与限定实测、仍缺的前提分别列在新ledger。''')
    print(json.dumps(dict(status='passed',updated_documents=5,cleanup_gib=cleanup['bytes']/2**30)))
if __name__=='__main__':main()
