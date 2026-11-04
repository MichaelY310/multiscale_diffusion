"""Refresh hidden pilot evidence from small LOCAL pulled artifacts only."""
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'eval_results/dmd_reuse_hidden_20260916/pilot_hidden'
BASE=ROOT/'eval_results/dmd_reuse_objective_20260915/dmd_lpips'
DOC=ROOT/'docs/research_docs/RDD_REUSE_HIDDEN_DISTILLATION_IMPLEMENTATION_20260916_ZH.md'
MARK='<!-- HIDDEN_PILOT_LIVE -->'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def score(folder,step):
    report=read(folder/f'preview_{step:07d}/complete.json')
    if report and isinstance(report.get('class_consistency'),dict):
        s=report['class_consistency']['student']
        return f"{s['top1_correct']}/{report['samples']}"
    return '尚未解码'


def main():
    body=DOC.read_text(encoding='utf-8').split(MARK)[0].rstrip()
    lines=[MARK,'','## 8. 本次正式 pilot：实测进度与图片','',
           f'证据刷新（非实时）：{datetime.now(timezone.utc).isoformat(timespec="seconds")}。','']
    progress=read(DATA/'progress.json')
    if progress:
        budget_label='无步数上限' if progress.get('unlimited_training') else '首轮预算2000步'
        lines += [f"状态：`{progress.get('state')}`；新增 {progress.get('step',0)} 步（{budget_label}），累计 {progress.get('total_compression_updates',5000)} 步。"]
        if progress.get('state')=='critic_warmup':
            lines += [f"当前先做 score critic 预热 {progress.get('warm_step',0)}/200；生成器此时尚未更新，属于正常流程。"]
        if progress.get('hidden_terms'):
            lines += ['',f"最近训练 batch 的 hidden NMSE：{json.dumps(progress['hidden_terms'])}；不是画质分数。"]
        if progress.get('max_rank_peak_gib') is not None:
            lines += ['',f"最近一次更新，各 rank 最大 allocated 显存：{progress['max_rank_peak_gib']:.3f} GiB。"]
    else:lines += ['当前尚未拉回正式 pilot 进度；不能用 smoke 冒充正式训练。']
    lines += ['','本轮只在旧目标上增加 hidden 监督。使用原无hidden实验的同起点、同训练步数、同seed结果作为对照，避免重复训练对照组。旧组不重新启动、不改变其目标。','',
              '| 新增更新 | 新 hidden：64张 MSE | 旧无hidden：64张 MSE | 新／旧类别Top-1 |',
              '|---:|---:|---:|---|']
    steps=[]
    for path in sorted(DATA.glob('validation_*.json')):
        v=read(path);s=v['step'];b=read(BASE/path.name);steps.append(s)
        bv=f"{b['clean_mse_8_16_32'][-1]:.5f}" if b else '无同step验证'
        lines += [f"| {s} | {v['clean_mse_8_16_32'][-1]:.5f} | {bv} | {score(DATA,s)} / {score(BASE,s)} |"]
    lines += ['','### 独立256张验证（不是FID）','',
              '| 新增更新 | 新 hidden：MSE | 旧无hidden：MSE | 新／旧类别Top-1 |','|---:|---:|---:|---|']
    for path in sorted((DATA/'holdout256').glob('validation_*.json')):
        v=read(path);s=v['step'];b=read(BASE/'holdout256'/path.name)
        bv=f"{b['clean_mse_8_16_32'][-1]:.5f}" if b else '无同step验证'
        lines += [f"| {s} | {v['clean_mse_8_16_32'][-1]:.5f} | {bv} | {score(DATA/'holdout256',s)} / {score(BASE/'holdout256',s)} |"]
    lines += ['','### 固定64张的hidden误差轨迹','',
              '这里只记录新组对同输入冻结教师的hidden误差；旧组没有对应记录，不能当作两组hidden误差的比较。',
              '', '| 新增更新 | 16×16末层 NMSE | 32×32第22层池化 NMSE | 32×32末层 NMSE |',
              '|---:|---:|---:|---:|']
    for s in steps:
        v=read(DATA/f'validation_{s:07d}.json');h=v.get('hidden_nmse',{})
        if h:
            lines += [f"| {s} | {h['middle_block24']:.5f} | {h['fine_block22_lowpass']:.5f} | {h['fine_block24']:.5f} |"]
    common=[s for s in steps if (DATA/f'preview_{s:07d}/complete.json').exists() and (BASE/f'preview_{s:07d}/complete.json').exists()]
    if common:
        step=max(common)
        if step==200:
            lines += ['','**200步人工检查：尚未看到新增hidden监督带来明确的画质改善。** 已对照相同种子的固定前16组图片；两组主要构图和失真非常相近，动物面部、人物、蚂蚁与灯罩结构仍明显缺失或涂抹。',
                      '', '固定64张类别Top-1两组均为14/64；Top-5为新hidden组28/64、旧无hidden组33/64。最终latent MSE为新0.07486、旧0.07437。不能把这一早期小样本差异当作统计显著结论。',
                      '', '新组同一验证集的fine第24层hidden NMSE从训练前0.22826降到0.16588（约下降27.3%），但图片没有同步出现明确恢复。这说明hidden误差下降不能直接代表画质改善；旧组没有对应hidden记录，不能据此算两组hidden优势。此时hidden权重0.02/0.05、原目标ramp也仅完成20%，继续既定2000步预算、不改变配置。独立256张验证从500步开始。']
        if step==1000:
            lines += ['','**1000步同预算检查：目前没有证据表明额外hidden监督优于原配方。** 已逐一对照两组固定64张与独立256张的前16组图片。主要物体布局、细节变化和错误形态很接近；犬类／人物五官、婴儿、灯罩和文字仍明显损坏，没有看到稳定一致的视觉优势。',
                      '', '同seed独立256张：新hidden组Top-1为88/256，旧无hidden组87/256；Top-5为146/256与149/256；最终latent MSE为0.08392与0.08366。固定64张两组Top-1均为27/64，Top-5为39/64与40/64。差一张不能当作有效提升；这些指标也不是FID。',
                      '', '新组固定64张fine末层hidden NMSE从初始0.22826降到0.11551（约下降49.4%），但这一下降尚未转化为相对无hidden基线的画质收益。不能把新组相较自身500步的类别改善全部归因于hidden loss，因为旧配方随训练也会改善。',
                      '', '当前刚完成原目标1000步ramp；按计划继续到2000步，在完整权重下再检查同预算独立256张对照。不改变损失、复用结构或预算，也不保证再训练就会改善。']
        lines += ['',f'### 同新增{step}步、相同种子的固定前16组','',
                  '每一对左边为原始不跳层DMD，右边为该组复用学生。所有图固定取前16组，不挑选成功样本。']
        for label,folder in [('原目标，无hidden监督',BASE),('新增hidden监督',DATA)]:
            for sub,title in [('', '固定64张中的前16组'),('holdout256','独立256张中的前16组')]:
                im=folder/sub/f'preview_{step:07d}/comparison_first16.jpg'
                if im.exists():lines += ['',f'#### {label}：{title}','',f'![{label}](../../{im.relative_to(ROOT).as_posix()})']
    else:lines += ['','尚未拉回可与旧组同step对比的新预览，暂不声称画质改善。']
    ready=[s for s in steps if (DATA/f'preview_{s:07d}/complete.json').exists()
           and (DATA/f'preview_{s:07d}/comparison_first16.jpg').exists()]
    if ready and (not common or max(ready)>max(common)):
        newest=max(ready)
        lines += ['',f'### 新组最新：新增{newest}步（没有对应旧组同step预览）','',
                  '下图仍是左原始不跳层DMD、右新增hidden的复用学生；不能与上方不同更新数的旧组作同预算胜负结论。']
        if newest==500:
            lines += ['','**500步人工检查：没有明显整体画质跃升。** 已查看固定64张和独立256张各自的前16组；相对本组200步，主要是局部纹理与边缘变化，动物／人物五官、婴儿、灯罩、文字和部分物体形状仍明显破损。独立集也延续了相同的失真模式。',
                      '', '固定64张：Top-1从200步14/64到500步19/64，最终latent MSE从0.07486到0.07370；首次独立256张为68/256（26.6%），原不跳层DMD为208/256（81.3%），新组最终latent MSE为0.08098。旧无hidden实验没有500步验证记录，所以本节点不能判断新增loss相对旧配方的净收益。',
                      '', '固定64张fine第24层hidden NMSE从初始0.22826降至0.12559（约下降45.0%），仍不能据此宣称图片修复。hidden权重已达到0.05，原DMD／LPIPS目标仍在1000步ramp中；继续原定2000步预算，1000步再进行两组同更新数、同seed的独立256张对照。']
        if newest==1500:
            lines += ['', '**1500步人工检查：仍未看到明显的整体画质恢复。** 已查看固定64张、独立256张各自的前16组。铠甲、栗子等部分对象仍可辨认，但蚂蚁、犬类五官、人物、婴儿、灯罩、眼镜盒和马桶等样本仍有显著结构丢失或涂抹。这里是固定展示，不是只挑失败图。',
                      '', '固定64张Top-1为28/64、Top-5为43/64，最终latent MSE为0.07747；独立256张Top-1为92/256、Top-5为148/256，MSE为0.08450。旧组没有1500步的验证，不能据此宣布hidden监督优于旧目标；1000步同预算对照仍是当前可用的直接比较。',
                      '', '固定64张fine末层hidden NMSE继续降到0.10942（相对初始下降约52.1%），仍然不等于画质恢复。按既定计划完成2000步后再做同种子、同预算的最终比较，不自动加训练步数或修改复用层。']
        for sub,title in [('', '固定64张中的前16组'),('holdout256','独立256张中的前16组')]:
            im=DATA/sub/f'preview_{newest:07d}/comparison_first16.jpg'
            if im.exists():lines += ['',f'#### {title}','',f'![新hidden组最新预览](../../{im.relative_to(ROOT).as_posix()})']
    lines += ['','**评价原则：hidden误差下降不等于画质变好；三步smoke不是质量实验。** 当前不作训练后视频质量声明。执行安排以文首最新用户授权为准；预算改变不意味着已经证明了画质收益。']
    DOC.write_text(body+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print(DOC)


if __name__=='__main__':main()
