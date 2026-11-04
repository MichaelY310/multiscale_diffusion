"""Refresh only the clearly delimited evidence section from LOCAL pulled files."""
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'eval_results/dmd_reuse_objective_20260915'
DOC=ROOT/'docs/research_docs/RDD_REUSE_DMD_OBJECTIVE_LONGRUN_20260915_ZH.md'
MARK='<!-- LIVE_EVIDENCE -->'
def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
def main():
    body=DOC.read_text(encoding='utf-8').split(MARK)[0].rstrip()
    lines=[MARK,'','## 7. 实际进度与最新图片（非实时页面）','',
           f'本地证据刷新：{datetime.now(timezone.utc).isoformat(timespec="seconds")}。检查日志时以服务器最新内容为准。','',
           '| 组别 | 新增／累计更新 | 状态 | 实测峰值分配显存 |','|---|---|---|---|']
    for family in ('control','dmd_lpips'):
        data=read(DATA/family/'progress.json')
        if not data:
            lines.append(f'| {family} | 尚未拉回 | 不据此声称训练成功 | — |');continue
        peak=data.get('max_rank_peak_gib',data.get('peak_gib'))
        peak=f'{peak:.2f} GiB' if isinstance(peak,(int,float)) else '—'
        state=data.get('state','unknown')
        if state=='critic_warmup':state+=f"，fake预热{data.get('warm_step',0)}/200"
        lines.append(f"| {family} | {data.get('step',0)}／{data.get('total_compression_updates',5000)} | {state} | {peak} |")
    control_progress=read(DATA/'control/progress.json')
    control_final=read(DATA/'control/holdout256/preview_0030000/complete.json')
    control_val=read(DATA/'control/holdout256/validation_0030000.json')
    if (control_progress and control_progress.get('complete') and
        control_progress.get('step')==30000 and control_final and control_final.get('complete') and control_val):
        score=control_final['class_consistency']['student'];teacher=control_final['class_consistency']['teacher']
        count=control_final['samples']
        lines+=['','**MSE对照组已完成全部预算：新增30000步，累计35000步，正常结束并保存最终checkpoint。** '
                '固定64张和独立256张最终预览均已完成并人工检查。与早期相比有局部边缘／可辨识度改善，'
                '但人物、婴儿、犬类面部和灯罩结构仍明显破碎或涂抹，不能称为恢复原模型质量。',
                '',f"最终独立256张：类别Top-1 {score['top1_correct']}/{count}（{score['top1_correct']/count:.1%}），"
                f"原DMD {teacher['top1_correct']}/{count}（{teacher['top1_correct']/count:.1%}）；"
                f"最终latent MSE {control_val['clean_mse_8_16_32'][-1]:.5f}。这些不是FID。"
                '该结果是对照组预算终点，不能与仍在训练的新配方作同预算胜负结论；不自动延长对照组训练。',
                '', '[对照组最终独立验证前16组图片](../../eval_results/dmd_reuse_objective_20260915/control/holdout256/preview_0030000/comparison_first16.jpg)'
                '；两组等步数比较仍见下文。']
    lines+=['','两组目标函数不同，**不能比较训练total loss大小判断谁的图更好**。下表使用相同固定验证集的MSE；分布匹配可能允许图像偏离原轨迹，需要同时看感知与分布指标。','',
            '| 组别 | 新增更新 | 8／16／32 MSE | 类别Top-1（64张） |','|---|---:|---|---|']
    for family in ('control','dmd_lpips'):
        for path in sorted((DATA/family).glob('validation_*.json')):
            data=read(path);step=data['step'];preview=read(DATA/family/f'preview_{step:07d}/complete.json')
            score='未解码'
            if preview and isinstance(preview.get('class_consistency'),dict):score=f"{preview['class_consistency']['student']['top1_correct']}/{preview['samples']}"
            error='／'.join(f'{e:.4f}' for e in data['clean_mse_8_16_32'])
            lines.append(f'| {family} | {step} | {error} | {score} |')
    preview_steps=[]
    for family in ('control','dmd_lpips'):
        preview_steps.append({int(p.name.split('_')[-1]) for p in (DATA/family).glob('preview_*') if (p/'complete.json').exists()})
    common_steps=set.intersection(*preview_steps)
    matched_step=max(common_steps) if common_steps else None
    if matched_step is not None:
        lines+=['',f'下面主图统一使用两组都完成的**新增{matched_step}步**，避免把较早完成的对照组与训练步数更少的新方案直接比较。各自更晚的单组预览另给链接。']
    if matched_step==200:
        lines+=['','**200步人工检查：还没有明确的画质恢复。** 两组都保留了部分布局和颜色，但动物面部、人物、灯罩纹理仍有明显涂抹和结构缺失，肉眼差别较小。固定64张Top-1均14/64；Top-5为对照28/64、新方案33/64。最终latent MSE为0.07497／0.07437，这个小差距不足以说明新方案已明显更好。此时DMD权重只完成20%的渐进增加，继续预定训练，不改变配置，也不把它冒充完整配方的最终结论。']
    if matched_step==1000:
        lines+=['','**1000步人工检查：出现了局部、可见的改善，但距离恢复原模型质量还很远。** 新方案的蚂蚁轮廓、鬣狗面部和毛发、汽车车灯与保险杠、鞋子边缘比同一步数MSE对照清楚；独立验证前16组也出现同方向变化。但婴儿和人物面部、犬类结构、灯罩花纹仍明显损坏，多张图仍有涂抹和块状伪影。这里展示固定前16组，不挑成功图，也不能据此推断全部图像质量。',
                '', '固定64张类别Top-1：对照12/64，新方案27/64，原DMD57/64；独立256张：对照52/256（20.3%），新方案87/256（34.0%），原DMD208/256（81.3%）。这是类别可辨识度的改善，不是FID，也没有达到与原模型相当。',
                '', '值得注意：独立256张最终latent MSE从对照0.07913变成新方案0.08366，**误差略大但可辨识度更好**。这支持继续检查感知／分布目标，而不是仅追逐逐点MSE；当前组合实验还不能区分LPIPS与DMD各自贡献。1000步才刚完成权重渐增，继续既定预算，在2000／5000等步数复核，不能保证后续一定恢复。']
    if matched_step==2000:
        lines+=['','**2000步人工检查：维持了1000步时的局部改善，尚未出现整体画质跃升。** 已逐一查看两组固定前16组及独立验证前16组。新方案的鬣狗面部、毛发、汽车边缘与鞋子仍比同一步数MSE对照清楚，但人物／婴儿面部、部分犬类、蚂蚁结构和灯罩花纹仍严重损坏，不能称为画质恢复。',
                '', '固定64张类别Top-1为对照15/64、新方案32/64；独立256张为对照51/256（19.9%）、新方案96/256（37.5%），原DMD仍为208/256（81.3%）。新方案相较自身1000步的87/256仅多9张被正确分类；这是小规模验证上的渐进改善，不是统计显著性结论，也不是FID。',
                '', '独立256张最终latent MSE为对照0.07789、新方案0.08351。新方案并没有靠更小的逐点MSE获得更高类别可辨识度，但当前配方无法分离DMD与LPIPS的贡献。训练、预算与推理配置均不改变；下一次计划的等步数图像检查为新增5000步。']
    if matched_step==5000:
        lines+=['','**5000步人工检查：新配方的局部优势仍在，但增加训练尚未解决整体结构缺失。** 已核对两组同seed、同更新数的固定前16组和独立验证前16组。新方案的汽车车灯／格栅、鞋带与鞋边、犬类毛发比MSE对照清楚；婴儿／人物面部、部分动物五官、灯罩及文字仍明显扭曲或涂抹。相对新方案2000步，主要是渐进细节变化，不是一次画质跃升。',
                '', '独立256张类别Top-1：对照57/256（22.3%），新方案105/256（41.0%），原DMD208/256（81.3%）；新方案自身2000步为96/256（37.5%）。固定64张Top-1却从新方案2000步的32/64变成30/64，说明小样本指标并非单调增长，不能只挑一项指标宣布成功。',
                '', '独立256张最终latent MSE为对照0.07480、新方案0.08038；两组都比各自2000步降低，但新方案仍有更高逐点误差。维持原定训练目标与30000新增更新预算，下一次计划等步数图像检查为10000步。尚未进行正式FID／sFID／IS／precision／recall比较，也没有新的训练后视频质量结论。']
    if matched_step==10000:
        lines+=['','**10000步人工检查：类别可辨识度继续缓慢增加，但没有明显的整体画质跃升。** 已核对两组同seed、同更新数的固定前16组和独立验证前16组，并回看新方案5000步图片。新方案在鬣狗毛发、汽车格栅与鞋子边缘等局部仍优于同一步数MSE对照；与自身5000步相比，大部分构图和失真模式几乎不变。婴儿、人物和犬类五官、蚂蚁、灯罩花纹与文字仍明显破损，不能称为恢复原DMD质量。',
                '', '独立256张类别Top-1：对照65/256（25.4%），新方案117/256（45.7%），原DMD208/256（81.3%）。新方案5000步为105/256（41.0%），增加12张；固定64张仅从30/64变为31/64，同一步数对照20/64。这仍是小样本诊断，不是FID或统计显著性结论。',
                '', '独立256张最终latent MSE：对照0.07289、新方案0.07822，新方案自身5000步为0.08038。继续原定30000新增更新预算，下一次计划等步数检查15000步；没有因为这次小幅变化启动大规模FID，也不声称视频质量已恢复。',
                '', '**这里仍是没有hidden蒸馏的原长训练。** 另行实现的hidden loss只完成独立三步工程smoke，未并入本组，不能把本页10000步结果归因于新增hidden监督。']
    matched_holdouts=[]
    if matched_step is not None:
        for family in ('control','dmd_lpips'):
            folder=DATA/family/'holdout256'/f'preview_{matched_step:07d}'
            preview=read(folder/'complete.json')
            val=read(DATA/family/'holdout256'/f'validation_{matched_step:07d}.json')
            if preview and val and preview.get('complete'):
                matched_holdouts.append((family,folder,preview,val))
    if len(matched_holdouts)==2:
        lines+=['',f'**相同新增{matched_step}步、相同独立256个seed**（未用于训练；不是FID）：','',
                '| 组别 | 最终latent MSE↓ | 类别Top-1 | 类别Top-5 |','|---|---:|---:|---:|']
        teacher=matched_holdouts[0][2]['class_consistency']['teacher']
        count=matched_holdouts[0][2]['samples']
        lines.append(f"| 原DMD参考 | — | {teacher['top1_correct']}/{count} | {teacher['top5_correct']}/{count} |")
        for family,folder,preview,val in matched_holdouts:
            score=preview['class_consistency']['student'];count=preview['samples']
            lines.append(f"| {family} | {val['clean_mse_8_16_32'][-1]:.5f} | {score['top1_correct']}/{count} | {score['top5_correct']}/{count} |")
    for family in ('control','dmd_lpips'):
        available=[p for p in sorted((DATA/family).glob('preview_*')) if (p/'complete.json').exists()]
        if available:
            preview=DATA/family/f'preview_{matched_step:07d}' if matched_step is not None else available[-1]
            data=read(preview/'complete.json');im=preview/'comparison_first16.jpg'
            lines+=['',f"### {family}：新增{data['step']}步，固定前16组",'','每组左边原DMD、右边复用学生；不挑图。']
            if im.exists():lines+=['',f'![原DMD与复用学生](../../{im.relative_to(ROOT).as_posix()})']
            if available[-1]!=preview:
                latest=available[-1];latest_data=read(latest/'complete.json')
                lines+=['',f"该组另已完成新增{latest_data['step']}步：[最新单组预览](../../{(latest/'comparison_first16.jpg').relative_to(ROOT).as_posix()})，目前不能与另一组不同更新数的图作等步数结论。"]
        if len(matched_holdouts)==2:
            holdout=next(folder for f,folder,_,_ in matched_holdouts if f==family)
            im=holdout/'comparison_first16.jpg'
            if im.exists():
                lines+=['',f'独立验证集：同为新增{matched_step}步的固定前16组，每组左原DMD、右复用学生。','',
                        f'![独立验证：原DMD与复用学生](../../{im.relative_to(ROOT).as_posix()})']
        holdouts=[p for p in sorted((DATA/family/'holdout256').glob('preview_*')) if (p/'complete.json').exists()]
        if holdouts:
            path=holdouts[-1];data=read(path/'complete.json')
            lines+=['',f"额外独立验证：新增{data['step']}步，{data['samples']}张，完整指标[JSON](../../{(path/'complete.json').relative_to(ROOT).as_posix()})。这也不是FID。"]
    DOC.write_text(body+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print(DOC)
if __name__=='__main__':main()
