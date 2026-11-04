"""Build a compact Chinese progress report from locally pulled evidence only."""
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'eval_results/dmd_cascade_reuse_20260915'
DOC=ROOT/'docs/research_docs/RDD_DMD_CASCADE_FINETUNE_20260915_ZH.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    completion=read(DATA/'completion_audit.json') or {}
    completed=completion.get('all_complete',False)
    final_summary=[]
    if completed:
        final_summary=['', '## 最终结论：速度达标，但本次微调没有保住画质', '',
            '**两组各5000更新、全部18组固定预览已完成并核对；实验执行完成，不等于方法成功。** 视频采样＋VAE延迟门槛均通过，但图像微调后仍有明显碎块、纹理涂抹和物体结构损失。不能说“只差训练”：整网微调已经做过，本轮配方尚未解决问题。', '',
            '| 配置 | 视频采样＋VAE延迟减少 | 图像最终clean MSE↓ | 类别Top-1 | 类别Top-5 |',
            '|---|---:|---:|---:|---:|',
            '| 原图像DMD teacher | 视频基线为同路径RDD＋VSA | 0（与自身比较） | 57/64 | 62/64 |']
        for family,label,speed in [('high_heavy','A：中阶段复用8层，高阶段20层','22.17%'),('balanced_plus','B：中阶段复用16层，高阶段18层','21.66%')]:
            validation=read(DATA/family/'validation_0005000.json')
            preview=read(DATA/family/'preview_0005000/complete.json')
            score=preview['class_consistency']['student']
            final_summary.append(f"| {label} | **{speed}** | {validation['clean_mse_8_16_32'][2]:.4f} | {score['top1_correct']}/64 | {score['top5_correct']}/64 |")
        final_summary+=['', '这里20%指**延迟减少**，不是速度倍数；22.17%约等于1.285×、21.66%约等于1.277×。图像指标仅来自同一组64张开发验证图，不是FID，也不是视频质量测量。视频只做计算成本测试，未输出本轮新视频、未微调视频模型。', '',
            '[A最终原图／学生对比](../../eval_results/dmd_cascade_reuse_20260915/high_heavy/preview_0005000/comparison_first16.jpg) · [B最终原图／学生对比](../../eval_results/dmd_cascade_reuse_20260915/balanced_plus/preview_0005000/comparison_first16.jpg)（每组左原图、右学生；固定前16组，不挑图）。', '',
            '**目前最有用的线索是A优于B，而不是两者已可用。** 两组粗阶段MSE接近（0.0253／0.0257），中阶段已明显分开（0.0224／0.0395），最终为0.0750／0.0956。保留中阶段真实更新、减少连续两次搬运最粗缓存，更值得优先验证；但层位置和重叠也同时变化，不能单独归因。', '',
            '1000→5000步，A的最终MSE仅从0.0850降到0.0750，B从0.1087降到0.0956；类别Top-1并不单调改善。因此不能保证继续延长训练就能恢复，更不能把MSE下降写成保质成功。', '',
            '**下一步应区分三个可能原因，而非直接宣布上视频可用：**（1）仅用配对MSE是否倾向平均化纹理——需要与保留原DMD分布目标的训练做受控比较；（2）连续两级复用的层重叠是否过多——保持成本接近，减少同一最粗残差被连续搬运；（3）可训练实际层是否足以补回细节——再做小规模层位置对照。它们是待验证假设，本轮没有替它们给出实验结论。', '',
            '完成审计：[checkpoint、随机数状态和18组预览检查](../../eval_results/dmd_cascade_reuse_20260915/completion_audit.json)。其中`all_complete`仅指文件／执行完整；`quality_passed=false`是本次人工查看图像后的验收结论，不是自动FID判定。']
    text=['# RDD-DMD 两级复用：20%视频成本门槛与图像整网微调',
          '', '<style>#content{overflow-wrap:anywhere}#content table{display:block;max-width:100%;overflow-x:auto}#content img{max-width:100%}</style>',
          '', f"报告生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。进度是最近拉取快照，不是网页实时刷新。",
          '', '**基线始终是相同分辨率路径的 RDD＋原生 VSA20，不是全分辨率 FastVideo。** 本轮视频只计时，不导出新视频；训练的是用户提供的真正三步 RDD 图像生成器。视频微调与保质结果尚未完成。',
          *final_summary,
          '', '## 1. 两组方法与对应关系', '',
          '| 组别 | 图像24层：8→16／16→32复用 | 视频30层：低→中／中→高复用 | 要比较什么 |',
          '|---|---|---|---|',
          '| A：高阶段重用更多 `high_heavy` | 8／20 | 10／25 | 保留较多中阶段真实计算，主要省高阶段 |',
          '| B：向中阶段分摊 `balanced_plus` | 16／18 | 20／23 | 更多依赖最粗阶段，给高阶段多留真实层 |',
          '', '两组都让8阶段完整计算24层，保留各阶段首尾各2层。A的16／32阶段分别真算16／4层；B分别真算8／6层。复用的是整块 residual `delta = block_output - block_input`，目标层执行 `h = h + nearest(delta)`；不覆盖完整hidden，不加adapter，scale=1。',
          '', '**这是激进级联，不是两级完全错开的方案。** A图像有8层连续从8阶段传到16再传到32；B有15层。A的32阶段缓存中另有12层是在16阶段真算出来的；B只有3层。它们的差异正好用于比较“中阶段的新计算”与“高阶段的新计算”哪一个更值得保留。',
          '', '图像与视频使用同一份比例／层位置生成代码：`diffusion/cross_scale_reuse_config.py`。30层的整数舍入会产生约一个block的比例差；不宣称层语义或权重一一对应。',
          '', '## 2. 视频速度：实测而非按跳层数推算', '',
          '统一1280×768、61帧、batch1、DMD 3 NFE、noise filling、VSA保留20%、BF16主干与BF16 VAE。模型驻留、逐shape预热，每条计时轨迹真正执行VAE。四组ABBA/BAAB，百分比为配对延迟减少；区间仅反映本次组间变化，不能消除共享GPU的系统干扰。初始化、T5、编译和媒体编码不计入采样＋VAE。',
          '', '空间倍率4→2→1的latent是 `16×24×40 → 16×48×80 → 16×96×160`，与图像8→16→32的两次空间放大对应。时间金字塔另列，不拿它冒充空间金字塔达标。',
          '', '| 路径 | 候选 | 去噪减少 | 采样＋VAE减少 | 组bootstrap区间 | 基线→候选总秒数（中位数） |',
          '|---|---|---:|---:|---|---|']
    for folder,label in [('video_speed','时间4→2→1'),('video_spatial_speed','空间4→2→1'),('video_spatial_plus_speed','空间4→2→1补测')]:
        data=read(DATA/folder/'metrics.json')
        if data:
            for e in data['entries']:
                r=e['reductions']['total'];d=e['reductions']['denoise']
                text.append(f"| {label} | {e['variant']} | {d['mean']:.2f}% | **{r['mean']:.2f}%** | [{r['lower']:.2f}, {r['upper']:.2f}]% | {e['baseline']['total']['median']:.3f}→{e['candidate']['total']['median']:.3f} |")
    text+=['', '原 `balanced`（图像16／16、视频20／20）虽然在时间路径通过20%，空间路径只有19.39%，已排除出正式训练。改为B的16／18后必须重新计时通过，不能四舍五入凑20%。A和B的空间测试在不同GPU分别与自身基线配对；不把跨GPU小差值当显著优劣。',
           '', '## 3. 真实DMD checkpoint与训练', '',
           '起点：用户 `results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256/checkpoints/latest.pt`，实际DMD更新6774。DiT-L/2、24层、hidden1024、457,840,672参数；global timestep为999／599／299，各分辨率一次调用。文件中的`ema`逐张量等于`student`，并非独立EMA。原文件不修改；已提取不可变生成器快照。',
           '', '| 参数 | 两组共用设置 |', '|---|---|',
           '| 训练对象 | 原生成器整个共享主干；冻结同起点DMD teacher；无新增参数 |',
           '| GPU | A=0,1,2,3；B=4,5,6,7；不终止已有任务 |',
           '| Batch | 每卡64，总256，四卡DDP，无梯度累积；activation checkpointing并非拆batch |',
           '| 优化器 | AdamW，lr=1e-5，100步warmup，betas=.9/.999，wd=.01，梯度clip=1 |',
           '| 精度 | FP32主参数／优化器，BF16 autocast，TF32关闭 |',
           '| 预算 | 每组新训5000更新；1、200、800、1000、2000等固定快照，不挑best掩盖失败 |',
           '| 数据 | 在线新噪声＋ImageNet类别；原DMD teacher生成配对轨迹，无需下载视频训练集 |',
           '| 条件 | 条件生成，无额外CFG；起点DMD继承其原训练teacher_CFG=1.5，不再乘一次 |',
           '| 验证 | 固定独立64个seed/类别，完整student三步采样；保存全部原图和学生图 |',
           '', '**这轮是对已有DMD生成器的压缩蒸馏，不是继续训练fake-score critic的原始DMD过程。** 原模型opt/critic不加载；新优化器从零开始，两组初始化与训练seed相同。',
           '', '针对旧实验失败的改动：',
           '', '1. 对完整8→16→32三次调用反传；两次noise filling和跨尺度cache都不detach。',
           '2. 同时约束8、16、32的clean预测，防止只修fine而损坏缓存生产者。',
           '3. teacher还在student实际输入上提供监督，减轻只学teacher轨迹的分布偏差。',
           '4. 前200更新把中／高阶段输入从偏teacher逐渐过渡到100% student；推理和全部验证始终100% student。训练到200后不再teacher forcing。',
           '5. 三步DMD只使用999／599／299，故监督这三个实际时刻；不照搬250步模型“补全全部timestep”的建议。',
           '', '损失为 final配对clean MSE ＋ 0.5×(8和16配对clean MSE之和) ＋ 0.25×三个阶段的当前student输入监督均值 ＋ 0.01×variance头保持损失。teacher全部冻结；这里variance不是flow velocity。以上是稳定性措施，不是收敛／画质保证。',
           '', '## 4. 训练记录与全部出图证据', '',
           '| 组别 | 最新本地记录 | 状态 | 峰值显存（GiB） |', '|---|---:|---|---:|']
    for family in ('high_heavy','balanced_plus'):
        run=DATA/family;progress=read(run/'progress.json')
        if progress:text.append(f"| {family} | {progress.get('step','—')} | {progress.get('state','training')} | {progress.get('max_rank_peak_gib',progress.get('peak_gib','—'))} |")
        else:text.append(f'| {family} | 待拉取 | 不据此声称已启动 | — |')
    text+=['', '下表均用同一批独立验证噪声、完整student三步推理，**不使用训练期的teacher输入混合**。MSE是对原DMD生成器的保真程度，不是FID，也不代表真实图像分布距离。', '',
           '| 组别 | 微调更新 | 8阶段clean MSE | 16阶段clean MSE | 32阶段clean MSE | 最终相对L2 |',
           '|---|---:|---:|---:|---:|---:|']
    for family in ('high_heavy','balanced_plus'):
        for path in sorted((DATA/family).glob('validation_*.json')):
            v=read(path)
            errors=v['clean_mse_8_16_32']
            text.append(f"| {family} | {v['step']} | {errors[0]:.4f} | {errors[1]:.4f} | {errors[2]:.4f} | {v['final_relative_l2']:.4f} |")
    for family in ('high_heavy','balanced_plus'):
        for preview in sorted((DATA/family).glob('preview_*')):
            data=read(preview/'complete.json')
            if not data:continue
            text+=['',f"### {family}：更新{data['step']}"]
            scores=data.get('class_consistency')
            if isinstance(scores,dict):
                text.append(f"固定{data['samples']}图：teacher类别Top-1={scores['teacher']['top1_correct']}/{data['samples']}；student={scores['student']['top1_correct']}/{data['samples']}。**不是FID，不等于画质合格率。**")
            image=preview/'comparison_first16.jpg'
            if image.exists():text+=['',f'![固定前16组，左原DMD、右复用学生](../../{image.relative_to(ROOT).as_posix()})']
    if (DATA/'high_heavy/preview_0000200/complete.json').exists():
        text+=['', '**200步时的历史判断：尚未恢复到可用水平。** A比零训练时更接近原图的布局和颜色，但仍有明显碎块、模糊和物体结构缺失。当时因此保留原配置继续到5000步；最终结论见页首。速度门槛通过不等于画质门槛通过。']
    va=read(DATA/'high_heavy/validation_0001000.json')
    vb=read(DATA/'balanced_plus/validation_0001000.json')
    if va and vb:
        gap=100*(1-va['clean_mse_8_16_32'][2]/vb['clean_mse_8_16_32'][2])
        text+=['', '### 1000步配对检查：多保留中间阶段计算的A暂时更好', '',
               f"同样1000更新、同样64个验证seed，A最终clean MSE={va['clean_mse_8_16_32'][2]:.4f}，B={vb['clean_mse_8_16_32'][2]:.4f}，A比B低{gap:.1f}%。800步时也是同一排序。1000步类别Top-1分别为A的9/64、B的5/64，原DMD为57/64；这只是小样本保真筛查，不是FID。", '',
               '一个有用的对照：图像三阶段分别有16/64/256个tokens。A真实block数为24/16/4，B为24/8/6，所以两组的线性投影/MLP理论工作量按token×block计，恰好同为2432：', '',
               '```text', 'A: 24×16 + 16×64 + 4×256 = 2432', 'B: 24×16 +  8×64 + 6×256 = 2432', '```', '',
               '这不表示完整FLOPs或延迟相同：QK/AV按token平方增长，缓存与插值也有额外开销。它说明A的初步优势不能简单解释为多做了线性层计算；保留中阶段真实更新、减少连续从8阶段搬运的缓存，是值得继续检验的方向。两组层位置与重叠也不同，不能据此单独归因，更不能只用一个训练seed宣称规律成立。', '',
               '**1000步预览仍不合格。** 某些轮廓恢复，但大量细节仍呈碎块/涂抹；当时保留两组继续训练，没有因A暂时领先而提前停掉B，也不把loss下降当作保质成功。']
    va2=read(DATA/'high_heavy/validation_0002000.json')
    vb2=read(DATA/'balanced_plus/validation_0002000.json')
    if va and vb and va2 and vb2:
        improve_a=100*(1-va2['clean_mse_8_16_32'][2]/va['clean_mse_8_16_32'][2])
        improve_b=100*(1-vb2['clean_mse_8_16_32'][2]/vb['clean_mse_8_16_32'][2])
        text+=['', '### 2000步检查：A仍领先，但画质恢复开始放缓', '',
               f"固定验证最终clean MSE：A={va2['clean_mse_8_16_32'][2]:.4f}，B={vb2['clean_mse_8_16_32'][2]:.4f}。相对1000步仅进一步下降{improve_a:.1f}%和{improve_b:.1f}%；不能把训练时间加倍误写成效果明显提升。", '',
               '两组2000步预览仍有明显涂抹、碎块与结构缺失，尚未达到可用画质。类别Top-1为A的10/64、B的3/64（teacher57/64）；B从1000步5/64降到3/64，提示MSE下降不保证语义指标改善，但这么小的样本不足以断言总体退化。当时保存、验证、显存和梯度正常，未改变对照配置，继续了原定5000步；最终记录见页首。']
    text+=['', '## 5. 安全、完成状态与归档', '',
           '本任务首次B补测与自己的smoke发生启动竞争，视频计时进程OOM；未停止其他训练。失败日志保留在 `video_spatial_plus_speed/launcher.log`。随后添加本任务逐卡flock，重跑计时；训练有进程显存上限，监控预览有额外空闲显存检查。',
           '', '首次两组训练各到200步时，NCCL在保存随机数状态的object通信路径报illegal memory access；未成功生成当时的checkpoint。已改用CPU Gloo传输此类元数据，梯度仍走NCCL；另在第1步就验证保存/评估路径。仅重新启动本任务两组，失败200更新不计入新训练进度；旧日志保留，不假装从未保存的200步恢复。',
           '', '远程所有新结果位于 `/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915`，不覆盖用户checkpoint。',
           '', '```bash',
           'tail -n 30 /home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915/high_heavy/launcher.log',
           'tail -n 30 /home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915/balanced_plus/launcher.log',
           '```', '',
           ('两组训练及本任务服务器预览监控均已自然退出；不要再用旧tmux attach命令查进度。最终每组5000步，保存优化器与四rank随机数状态，全部9个时间点×2组预览完整。本次新增GPU占用已释放，其他训练仍在运行，不代表整机空闲。' if completed else '服务器记录进度／异常并解码固定预览；训练loss下降不等于画质达标。'),
           '', '最终可继续训练的checkpoint（服务器路径）：', '',
           '```text',
           '/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915/high_heavy/checkpoints/latest.pt',
           '/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915/balanced_plus/checkpoints/latest.pt',
           '```', '',
           '每组另保留`step_0005000.pt`生成器快照。原始用户DMD checkpoint未改。本地拉回了日志、协议、验证JSON、所有18张对比网格和完成审计；大权重与每次64张teacher＋64张student的单张PNG保留服务器，不声称已完整下载到本地。',
           '', '尚未完成的证据：两组训练后正式FID、视频质量／视频微调、独占GPU正式计时、多训练seed。当前20%是特定形状的计算收益，不是所有视频配置的保证。']
    DOC.write_text('\n'.join(text)+'\n',encoding='utf-8');print(DOC)


if __name__=='__main__':main()
