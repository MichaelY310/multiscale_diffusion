#!/usr/bin/env python3
"""Render the paper/source/config audit without changing parity baselines."""
import datetime
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/compatibility_audit'
SOURCES = json.loads((ROOT/'evidence/sources.json').read_text())
PROBES = json.loads((OUT/'PROBES.json').read_text())
FINDINGS = []


def anchor(relative, needle):
    path = ROOT/relative
    lines = path.read_text().splitlines()
    line = next(i for i,s in enumerate(lines, 1) if needle in s)
    parts = Path(relative).parts
    public = None
    if parts[0] == 'sources':
        key = parts[1]
        public = f'{SOURCES[key]["repo"]}/blob/{SOURCES[key]["commit"]}/{"/".join(parts[2:])}#L{line}'
    return dict(file=relative, line=line, needle=needle,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), public_url=public)


def finding(id, methods, priority, category, title, detail, impact, fix, refs, probes=()):
    item = dict(id=id, methods=methods, priority=priority, category=category, title=title,
                detail=detail, impact=impact, required_work=fix,
                evidence=[anchor(*ref) for ref in refs], probes=list(probes), resolved=False)
    FINDINGS.append(item)
    return id


finding('DCT-01',['dctdiff'],'P1','missing_recipe_component','缺少原版 EBFR 频率 loss 权重',
        '作者将 Y/Cb/Cr 的频率标准差按 zigzag 截断、各自归一化，再按 Y×4、Cb、Cr 拼接，逐元素乘平方误差。现有 MSE 只支持 head 标量权重；128 步 oracle 显式 reweight=1。Prepared 虽可携带元素权重，当前 DCT 配置走 generic path，并没有把作者统计量接入该通路。',
        '同一输入与预测也会产生不同 loss 和梯度；不是精度舍入。现有测试证明的是未加权分支，不能写成完整 DCTdiff 默认目标。',
        '增加带 shape/颜色/zigzag 顺序校验的元素权重 metric；以真实非均匀小权重做 native loss/gradient 对照。',
        [('sources/dctdiff/train.py','Y_reweight ='),('sources/dctdiff/sde.py','loss = loss * reweight'),
         ('umdiff/targets.py','class MSE:'),('tools/validate_training.py','clean,reweight=1.')],
        ['dctdiff_frequency_reweight','dctdiff_actual_uvit_weighted_gradient'])
finding('DCT-02',['dctdiff'],'P2','missing_auxiliary_branch','统计准备、DCT Upsampling 和论文 DiT 分支未完整接入',
        'codec 已实现 YCbCr、4:2:0、zigzag、bound 缩放与解码量化，但 fixture bound=256 不是按目标数据估计的参数。作者独立 DCT_upsampling 使用逐块余弦频率校正；现有 spectral boundary 是 SPEED 的全图频谱扩展，不能替代。论文还有 DiT 结果，但固定作者 README 仍说明 DiT 将另行发布。',
        '缺少的是统计/辅助组件与论文分支，不是已测 UViT 的神经层。不可把同名 DCT 操作视为同一算法；未发布 DiT 不能声称源码等价。',
        '提供离线统计参数导入和独立 block-DCT cosine-corrected upsampler；DiT 标明 paper-only，直到有可核验源码或独立实现证据。',
        [('sources/dctdiff/DCT_datasets_statis.py','np.percentile'),('sources/dctdiff/DCT_analysis_upsampling.py','def DCT_upsampling'),
         ('sources/dctdiff/README.md','We will also release'),('umdiff/boundaries.py','class SpectralBoundary:')])
finding('DIMR-01',['dimr'],'P1','missing_recipe_component','CFG 训练 dropout 与两种推理 guidance 均未接入',
        '作者 CFGDataset 以 p_uncond=.1 将标签换成1000；MRModel本身不替代这个数据包装。当前 labels conditioner 不丢标签且 classes=1000 会拒绝空标签1000。eval.py 同时有 cond+s(cond−uncond) 和基于 timestep 的 cosine decay CFG；当前 sampler 只取 conditional 最后一层预测。',
        '现有模型内部 TD-LN/多尺度结构已包含，但论文报告的 guided 采样配方不包含在当前入口中。离线预处理空标签可以走 field conditioner，仍须显式实现相同 dropout 分布和采样调用。',
        '将 condition dropout 与 guidance 分成组件，支持 null token、作者 scale 约定、time decay 和精确 call batching；用同一网络比较 guided 最终 latent。',
        [('sources/dimr/datasets.py','class CFGDataset'),('sources/dimr/configs/DiMR-XL-2R_imagenet256.py','p_uncond=0.1'),
         ('sources/dimr/eval.py','def cfg_nnet'),('umdiff/samplers.py','class DiMRDPMSampler:'),('umdiff/cli.py','Label outside configured class range')],
        ['dimr_constant_and_decaying_cfg','dimr_cfg_training_label_dropout'])
finding('DIMR-02',['dimr'],'P2','missing_auxiliary_branch','512px scaling_factor 采样选项未接入',
        '原版 eval model_fn 可先按每样本标准差归一化 state，并将 scaling_factor 交给 DPM solver；当前 DiMRDPMSampler 的配置只有 betas 和 steps。三分辨率架构可由作者类配置，当前128步证据是两分辨率。',
        '两分辨率无 guidance fixture 的相等不覆盖该选项或三分辨率完整 recipe。',
        '增加显式状态预处理/solver 参数契约与对应原版分支测试，不能仅改 image_shape。',
        [('sources/dimr/eval.py','x = x / x.std'),('umdiff/samplers.py','def __init__(self,betas,steps=12)')])
finding('PPF-01',['ppflow'],'P1','missing_recipe_component','stage-wise CFG 与前三通道 mask 未接入 sampler',
        '原模型确实含 forward_with_cfg，但现有 sampler 永远调 forward_wo_cfg。作者用 bisect_left 处理 stage 边界，条件 batch 在前、null batch 在后；只对前三通道做 CFG，第四 latent 通道及其余输出保持原值。',
        '已有源码类不等于用户入口已启用该功能。普通全通道 CFG 会改变作者结果；边界 t=.5/.75 的归属也影响 stage scale。',
        '接入作者 guidance 调用、通道 mask、batch 顺序和 stage boundary，逐步比较状态与最终输出。',
        [('sources/ppflow/models.py','def forward_with_cfg'),('sources/ppflow/models.py','eps, rest = model_out[:, :3]'),
         ('sources/ppflow/sample.py','model_kwargs = dict(y=y, cfg_scales'),('umdiff/samplers.py','net.forward_wo_cfg')],
        ['ppflow_stage_cfg_and_three_channel_mask','ppflow_actual_backbone_guided_sampling'])
finding('PPF-02',['ppflow'],'P1','missing_recipe_component','预训练 SiT 到 pyramid 的初始化转换缺失',
        '原版 initialize_from 会复制共享参数，把 Patchify 权重按通道重复并除4/2，Unpatchify 权重/偏置按输出重复。当前 Trainer 只随机构造模型或恢复相同 unified config 的完整 checkpoint，没有该转换入口。',
        '从头训练的 parity 不证明论文微调结果可复现。只使用 strict=False 加载而不完成层转换，也不等价。',
        '增加独立离线 checkpoint importer，逐参数比较转换后的 model/EMA；在载入初值后验证第一个更新和采样。',
        [('sources/ppflow/train.py','if args.initialize_from is not None:'),('sources/ppflow/train.py','repeat_factor=4, scale_factor=4.0'),
         ('umdiff/cli.py','def setup('),('umdiff/training.py','Checkpoint method configuration mismatch')])
finding('PPF-03',['ppflow'],'P2','paper_release_scope','固定发布只有2/3stage，论文新版的4/5stage、其他分段及FLUX不能据此认证',
        '当前作者模型和 training transport 自身也硬编码2/3级与 .5/.75 分段；unified 忠实覆盖这个发布子集。论文新版本额外讨论4/5级、其他分段和FLUX。原版 transport 还存在 SDE/其他预测选项，当前 sampler 只包装 linear velocity Euler。',
        '部分限制来自作者 release 与论文版本之间，不能都归咎于 unified；也不能将论文所有消融写成现有配置已可用。',
        '固定论文版本与发布分支清单，增加 capability validation；扩展分段/layout/SDE 时另作数值验证。',
        [('sources/ppflow/models.py','self.t1 = 0.5'),('sources/ppflow/transport/transport.py','if patch_level == 2:'),
         ('umdiff/batches.py','PPFlow release supports 2 or 3 stages'),('umdiff/samplers.py','class PPFlowSampler:')])
finding('MDM-01',['mdm'],'P1','missing_recipe_component','mixed-resolution batch 的计算裁剪与有效样本重加权缺失',
        '作者 CC12M 256 配方 mixed_ratio=2:1 -> 累积分数[2/3,1]：高分辨率只计算前2/3 batch，输出补零；其 loss 除2/3，尾部 loss 清零。现有 MDMBatch 是各尺度全 batch 加静态 leaf_weights。',
        '不只是算力差异，也改变当前 batch 的 loss/梯度。用一个 head scalar 不能替代 per-example mask。',
        '增加每个尺度的有效 batch mask、计算裁剪、inverse-inclusion weighting，覆盖 mask=0、取整和多尺度顺序。',
        [('sources/mdm/ml-mdm-matryoshka/configs/models/cc12m_256x256.yaml',"mixed_ratio: '2:1'"),
         ('sources/mdm/ml-mdm-matryoshka/ml_mdm/diffusion.py','x_t = [x[: int(m'),
         ('sources/mdm/ml-mdm-matryoshka/ml_mdm/diffusion.py','loss_ = loss_ / self.mixed_ratio'),('umdiff/batches.py','class MDMBatch:')],
        ['mdm_mixed_resolution_batch_mask'])
finding('MDM-02',['mdm'],'P1','missing_recipe_component','progressive 初始化、CFG、dynamic threshold 未完整封装',
        'native factory 明确禁止 initialize_inner_with_pretrained；没有按训练阶段预训练/冻结/扩展网络的 curriculum。MDMSampler 仅普通模型调用和固定 clip。作者还有文本 CFG、dynamic/DYNAMIC_IF threshold，以及 VDM 加权归一化 loss 的可选分支。',
        '共享 NestedUNet 架构和所测 v→epsilon 目标坐标转换已包含，不能把它们说成缺失。论文 progressive/mixed training 与可选阈值策略仍未完整覆盖。',
        '提供本地预训练导入和 curriculum；把 threshold/guidance 放在各尺度相同的位置；VDM 权重保留 sum(loss*w)/sum(w) 而不是简单 mean。',
        [('umdiff/native_models.py',"Pretrained initialization disabled"),('umdiff/samplers.py','class MDMSampler:'),
         ('sources/mdm/ml-mdm-matryoshka/ml_mdm/samplers.py','def _threshold_sample'),
         ('sources/mdm/ml-mdm-matryoshka/ml_mdm/trainer.py','(losses * weights).sum() / weights.sum()')])
finding('SSD-01',['ssd'],'P1','missing_recipe_component','默认同分辨率多t batch 不被当前 batch 接受',
        '作者默认 one_res_many_t_except_one_t_for_res_change：先抽一个t，若无分辨率切换，再在同输入/输出分辨率集合为每个样本抽t；切换时才统一t。unified 无该 time_sampler，SSDBatch 强制整批t完全相同。作者 min_snr_5 默认权重在 unified 已有可配置实现，当前fixture却使用constant。',
        '同t均匀策略可保持单样本均匀目标，但 batch 相关性和梯度方差不同，且作者默认 batch 会报错。min-SNR是已实现但默认recipe未接线，不应混同缺少后验数学。',
        '加入按 (rt,rs) 分组的 time_sampler，并让 batch 对每样本t计算相同shape的状态；补min_snr_5 native loss/gradient分支。',
        [('sources/ssd/configs/ssd.yaml','one_res_many_t_except_one_t_for_res_change'),
         ('sources/ssd/guided_diffusion/resample.py','valid_t = all_t'),('sources/ssd/configs/diffusion.yaml','min_snr_5'),
         ('umdiff/batches.py','Native SSD uses one timestep/resolution')],['ssd_default_same_resolution_many_times'])
finding('SSD-02',['ssd'],'P1','silent_contract_mismatch','rescale_timesteps 可用于训练但 sampler 漏掉',
        'SSDBatch 支持 t*1000/N；ScaleSpaceSampler 无该参数且直接把压缩索引i交给网络。8步反例中，训练和原版网络时间875，unified采样时间7。',
        '默认rescale=False的测试不受影响，开启合法训练选项后会训练/推理时间错配。',
        '让 batch/sampler 共享一个 network clock，加载时验证契约；增加非默认 rescale 对照。',
        [('sources/ssd/guided_diffusion/scale_space_diffusion.py','def _scale_timesteps'),
         ('umdiff/batches.py','t.float()*1000/len(self.path.betas)'),('umdiff/ssd.py','clean=model(pre,t,cond_rs=cond')],['ssd_train_sample_time_rescaling'])
finding('RELAY-01',['relay'],'P1','missing_pipeline','验证的是高分辨率 blur stage，完整两模型链未封装',
        '作者 ImageNet 入口先用独立低分辨率网络跑 EDM，再 resize，将干净结果交给另一个网络的截断 blur sampler。unified relay sampler 只实现后一段；generic CLI 为它生成随机高分辨率 initial，而没有低分辨率 EDM 模型和交接。',
        '后段 sampler 在共同 supplied input 上相等，不代表从噪声到最终图的原版 Relay 流程已相等。模型数、训练/推理算力也不能只计后段。',
        '增加双模型 workflow、两个 checkpoint、低尺度 EDM sampler、明确的 clean/noise 交接及端到端预算。',
        [('sources/relay/generate_imagenet.py','images = edm_sampler(net_first'),
         ('sources/relay/generate_imagenet.py','images = blur_sampler(net_second'),('umdiff/relay.py','class RelaySampler:'),
         ('umdiff/cli.py','initial_shape=list(shape)')])
finding('RELAY-02',['relay'],'P2','silent_contract_mismatch','自定义训练 blur 统计量未与 sampler 自动绑定',
        'RelayBatch 可配置 p_mean/p_std/min_scale；RelaySampler.alpha 仍固定−1.2/1.2/.001。两边默认相同，但合法训练自定义值不会自动传到推理。',
        '默认值的原版相等证据保留；修改训练组件后不能假设 sampler 同步兼容。',
        '共享 heat/sigma clock配置，加载时检查训练/采样参数一致。',
        [('umdiff/batches.py','def __init__(self,patch=4,p_mean='),('umdiff/relay.py','np.log(sigma.cpu())+1.2')])
finding('DFM-01',['dfm'],'P1','missing_recipe_component','各尺度自己的 CFG window 未接入',
        '作者 sampler 根据各 band 当前time判断guidance_low/high；若任一尺度需CFG，先给所有尺度倍增batch，再只在各自window内混合conditional/unconditional。当前 Method 的 product-state 循环仅调用一次普通model。',
        '相同schedule下无CFG可逐位相等，打开原版CFG后终态改变。单一全局time的guidance不能直接替代该行为。',
        'guidance policy必须读取product-state clock与active mask，保留batch顺序、精度转换和调用次数。',
        [('sources/dfm/dfm_utils/samplers_dfm.py','any_scale_needs_cfg = False'),
         ('sources/dfm/dfm_utils/samplers_dfm.py','guidance_low=0.0'),('umdiff/engine.py','pred=model(map_tree')],['dfm_per_scale_guidance_window'])
finding('DFM-02',['dfm'],'P1','source_paper_divergence','release自己说明架构不同论文；RNG和3scale限制不能隐去',
        '固定README明确将release称为基于SiT的重实现，架构不完全等同论文。loss中的随机logging消耗NumPy RNG；3scale特定schedule会产生正dt并被原版拒绝。已有证据保留这些行为并单列日志RNG修正对照，没有改作者源码。',
        '即便所有层与release相同，也不是论文原架构认证；仅固定共同seed不保证连续轨迹相等。',
        '使用“released SiT reimplementation parity”标签；记录per-stage/time/noise随机耦合、logging policy和合法schedule域。',
        [('sources/dfm/README.md','The architecture does not exactly follow'),('sources/dfm/loss.py','np.random.rand() < 0.01'),
         ('umdiff/clocks.py','DFM times must decrease')])
finding('LAP-01',['lapflow'],'P1','missing_recipe_component','原版 CFG 与前三通道约定未接入',
        'MultiScaleDiT.forward_with_cfg 对每个 band 的前三通道做CFG，并在 variance-aware 分支先/后缩放；unified LapFlowSampler 直接调model普通forward。',
        '共享MoT/causal mask/尺度head架构已包含，现有两层128步不是guided采样的认证。',
        '在tuple-state上加入原版guidance和var_scale的运算顺序；验证所有band和拼回的latent。',
        [('sources/lapflow/models/MultiScaleDiT.py','def forward_with_cfg'),
         ('sources/lapflow/sampler/ours_sample.py','one_model.forward_with_cfg'),('umdiff/samplers.py','batch_t=t.expand(len(x[0]))')])
finding('LAP-02',['lapflow'],'P2','missing_auxiliary_branch','variance-aware、square path及部分训练采样选项未接入',
        '源generate_xt_and_u可由当前噪声的std计算var_scale，支持square_path、noise_var_factor等；LapFlowBatch只封装linear/cosine delayed path和epsilon。原版三层噪声规则已包含，但128步全训练认证仅两层，三层为组件和4步CLI。',
        '可选研究分支不能从主分支证据推出。EQ-VAE/SD-VAE及latent scaling必须由离线数据保持，换VAE不是纯sampler等价。',
        '为每个选项声明capability；按真实随机draw生成precondition metadata，新增分支测试；三层128步另补。',
        [('sources/lapflow/sampler/ours_sample.py','if args.variance_aware else None'),
         ('sources/lapflow/sampler/ours_sample.py','def square_path'),('umdiff/batches.py','class LapFlowBatch:')])
finding('DIFF-01',['diffusability'],'P1','missing_recipe_component','随机2x/4x正则、LPIPS和微调参数策略未完整包含',
        '论文/源码每次随机选一个scale_factor；当前若配置两个ScaleEquivariance会每步全部求和。CodecMethod只有MSE+固定KL+MSE正则，无LPIPS、冻结指定输出层、KL schedule、DCT-cut/decoder R1等选项。论文SE主配方还关闭KL；当前fixture的KL=.0001是小测试选择。',
        '当前通过的是共享VAE内部、mean/bilinear单一factor的MSE/KL辅助训练，不是完整autoencoder微调，更不是后续DiT生成训练链。GAN不是此论文SE主配方的必要缺项，不能仅因旧limit提到GAN便要求加GAN。',
        '增加随机regularizer selector、可注入本地perceptual网络、参数冻结和loss schedule；明确codec训练→latent提取→下游diffusion三阶段。',
        [('sources/diffusability/src/training/loss.py','np.random.choice(self.cfg.scale_equiv_reg.scale_factors)'),
         ('sources/diffusability/configs/loss/autoencoder.yaml','perceptual_loss: {weight: 1.0}'),
         ('umdiff/codec_training.py',"allowed={'mse_weight','kl_weight','regularizers'}"),('umdiff/regularizers.py','class ScaleEquivariance:')],
        ['diffusability_random_scale_vs_sum'])
finding('RMD-01',['rmd'],'P1','source_provenance_and_coverage','与自称官方的Wan release一致，未证明全部论文模型一致',
        '仓库归属仍未从论文官方链接独立确认。两种transition、teacher/fake/generator顺序、width未reduce的源码行为已验证；paper包含SDXL/PixArt/SD3.5/Wan，当前实现和训练测试是Wan3D release。可选GAN未封装。',
        '必须把该release数值认证与论文全部backbone/蒸馏机制认证分开。不能将三个网络简化成普通单网络MSE训练。',
        '先核实来源；继续保留专用多网络trainer，按具体backbone和原版版本分别接入；GAN另列实验分支。',
        [('sources/rmd/docs/method.md','algorithm exactly as implemented'),('umdiff/distillation.py','class RMDTrainer:'),
         ('umdiff/distillation.py','width is intentionally not reduced'),('sources/rmd/src/rmd/models/gan.py','class ')])
finding('RDD-01',['rdd'],'P1','silent_contract_mismatch','非恒等timestep_map未映射网络时间',
        'DiscreteVP接受timestep_map并用于drift系数，但GaussianBatch/ GaussianSampler给网络的time仍是压缩索引；迁移的SpacedDiffusion通过_WrappedModel映射回原始time。4步反例的第一个网络time应为15，当前是3。',
        '六种恒等map变体128步结果仍成立；更改respacing后不能沿用该等价结论。',
        '统一 path index、original diffusion time和network time，按原版wrapper实现映射并覆盖非恒等map。',
        [('umdiff/paths.py','class DiscreteVP:'),('umdiff/gaussian_objective.py','prediction=model(state,t,'),
         ('../multiscale_diffusion/diffusion/respace.py','new_ts = map_tensor[ts]')],['rdd_respaced_network_time'])
finding('RDD-02',['rdd'],'P1','missing_pipeline','RDD全阶段入口、DDIM/DPM/jump及CFG没有对应完整workflow',
        '当前训练和DDPM sampler已覆盖effective-epsilon、auxiliary drift和learned-range variance；fresh与历史posterior边界组件也存在。CLI不自动拼阶段；原迁移代码另有drift-aware DDIM、DPM1/2、jump sampling和CFG模型入口。',
        '不能把多个边界函数存在视为完整多尺度RDD训练/推理已由JSON一次跑通；generic stages只能通过显式API调用。',
        '封装真实阶段时间、网络/权重分配、clean/noise边界上下文和终态；为每个solver单独认证。',
        [('umdiff/engine.py','def sample_stages'),('umdiff/engine.py','Use sample_stages for explicit boundary'),
         ('../multiscale_diffusion/diffusion/gaussian_diffusion_with_drift.py','def dpm_solver_sample('),
         ('../multiscale_diffusion/DiT/sample_fid_jsonl.py','model_fn = model.forward_with_cfg')])
finding('PF-01',['pixelflow'],'P2','runtime_unverified','默认class-conditional算法覆盖较好；CUDA噪声stream有设备契约风险',
        'packed sequence、per-image token reduction、labels重排、network clock、2x2负相关噪声jitter、stage CFG ramp都已包含。当前sampler在CPU上生成boundary噪声却沿用传入stream；标准CLI创建的RandomStream在CUDA运行时持有CUDA generator，传给CPU torch.randn存在设备不匹配。此为调用链静态发现，未启动GPU验证。',
        'CPU/SDPA证据不认证GPU bf16/FlashAttention；不应把class dropout说成缺失，作者LabelEmbedding已在共享model中实现。',
        '为CPU boundary RNG提供独立且可恢复的CPU generator；验证跨设备draw顺序。GPU验证需新的有效授权。',
        [('umdiff/samplers.py',"torch.empty_like(state,device='cpu')"),('umdiff/randomness.py','self.generator=torch.Generator(device=device)'),
         ('umdiff/noise.py','stream.normal(torch.empty(4,device=x.device'),('sources/pixelflow/pixelflow/model.py','dropout_prob=0.1')])
finding('COMMON-01',['relay','ssd','pixelflow','dfm','lapflow','ppflow','dctdiff','mdm','dimr','diffusability','rdd'],'P1','training_recipe_scope','优化器配方、数据和预训练资产不在128步相等认证内',
        '普通Trainer固定AdamW、恒定LR/EMA、单batch更新，无AMP/GradScaler、累积、DDP/FSDP、LR warmup/scheduler和作者特定EMA策略。图像入口统一center crop/BICUBIC、顺序遍历；offline tensor只收BCHW。尚无通用本地pretrained importer、latent/VAE/text encoder完整配方或正式FID协议。RMD有独立两优化器，但也不是全原版分布式recipe。',
        '两个路径用共同controlled optimizer的逐位相等，不代表与作者完整train.py的默认更新轨迹相等，也不保证最终画质至少一样。decoder的缩放/shift、JPG或PNG和统计集会影响最终像素与FID。',
        '将初始化/参数分组/优化器/LR/EMA/精度/数据预处理列成可检验recipe；正式实验前完成小真实数据训练、native checkpoint import与guided decoded end-to-end对照。',
        [('umdiff/training.py','self.optimizer=torch.optim.AdamW'),('umdiff/training.py','self.decay=ema_decay'),
         ('umdiff/training.py','Image.Resampling.BICUBIC'),('umdiff/cli.py','Local tensor input must be floating BCHW'),
         ('sources/dctdiff/README.md','generated images must be saved into png')])
finding('COMMON-02',['relay','ssd','pixelflow','dfm','lapflow','ppflow','dctdiff','mdm','dimr','diffusability','rmd','rdd'],'P1','composition_scope','数学框架可描述不等于任意native组件可交换',
        'native batch会拥有并禁止覆盖path/noise/metric/target，保障原版语义；魔改需回到generic engine或新写batch。不同author repo共用models/libs等顶层模块名，factory明确要求不同进程。sample_stages要求调用者给boundary上下文，CLI无自动stage runner。表示/目标/损失/求解器之间没有完整自动兼容证明。',
        '原用户期望“任意组合配置即可复现所有论文”尚未完全达到。限制有合理数学原因：product-state、有损变换、非各向同性posterior和inverse可识别性不是换个upsampler名字就兼容。',
        '增加typed state/clock/conditioning/metric/guidance契约、capability manifest和组合校验；把native认证组合与controlled transplant分别暴露，不静默跳过组件。',
        [('umdiff/engine.py','Native batch owns'),('umdiff/engine.py','Native batch has fixed author prediction semantics'),
         ('umdiff/native_models.py','Native repository namespace collision'),('umdiff/engine.py','One boundary context per transition required')])


METHODS = {
    'relay':('高分辨率blur目标/完整后段sampler已测；欠双模型链与训练/采样共享clock。',['RELAY-01','RELAY-02']),
    'ssd':('累计bilinear算子、真实adjoint、非各向同性posterior/Lanczos/FlexiUNet已包含；默认batch策略与非默认clock漏接。',['SSD-01','SSD-02']),
    'pixelflow':('默认C2I pipeline核心覆盖相对完整；packed batch/per-image reduction/CFG ramp都在。',['PF-01']),
    'dfm':('release的product-state/path/drop mask/REPA已测；guidance缺失且release不同paper。',['DFM-01','DFM-02']),
    'lapflow':('两层native及三层组件已覆盖；不要把noise coupling/causal mask误列为缺失。',['LAP-01','LAP-02']),
    'ppflow':('2/3级permutation/packing/level embedding在；guided采样和微调初始化缺失。',['PPF-01','PPF-02','PPF-03']),
    'dctdiff':('UViT/codec/VP-SNR/DPM在；默认频率目标权重是明确缺项。',['DCT-01','DCT-02']),
    'mdm':('共享NestedUNet、gamma shift、v→epsilon坐标转换在；mixed batch/预训练curriculum/guidance缺失。',['MDM-01','MDM-02']),
    'dimr':('TD-LN、特征cascade、噪声尺度校正与DPM在；CFG数据和采样recipe缺失。',['DIMR-01','DIMR-02']),
    'diffusability':('共享VAE内部和单尺度MSE/KL已测；完整SE微调和下游生成链未完成。',['DIFF-01']),
    'rmd':('专用三网络/两优化器目标与两transition已测；归属及paper多backbone范围有限。',['RMD-01']),
    'rdd':('joint/aux drift/learned variance/DDPM已测；respacing网络clock与全stage workflow不足。',['RDD-01','RDD-02']),
}


def main():
    coverage=json.loads((ROOT/'PAPER_COVERAGE.json').read_text())
    by_id={p['id']:p for p in coverage['methods']}
    records=[]
    for key,(assessment,ids) in METHODS.items():
        paper=by_id[key]
        source=SOURCES.get(key)
        records.append(dict(id=key,name=paper['name'],paper_url=paper.get('paper'),
                            source=None if source is None else {k:source[k] for k in ('repo','commit')},
                            assessment=assessment,findings=ids+(['COMMON-01'] if key!='rmd' else [])+['COMMON-02'],
                            full_paper_recipe_coverage=False,
                            previous_parity_still_valid_in_tested_scope=True))
    extra=[dict(id='speed',assessment='仅FLUX推理workflow已验证；作者还发布PixelGen与Wan视频入口，当前workflow没有这些backend。训练目标未公开，不补造。'),
           dict(id='nami',assessment='作者vendored FLUX、BridgeFlow、pool/mask与decoder推理已测；bridge/denoiser训练未发布。未执行的callback/IP-adapter等管线可选项不认证。'),
           dict(id='mrflow',assessment='FLUX→RGB量化/PIL→真实RRDB tile→VAE重编码→refine已测；当前workflow不覆盖论文Qwen/Pi-Flow等全部backend，文本编码时间不在预计算embedding测试中。')]
    closed=[dict(id=p['id'],name=p['name'],status=p['status'],limitation=p.get('skip_reason'))
            for p in coverage['methods'] if p.get('status')=='paper_mapping_native_skipped']
    unchanged=[]
    changed=[]
    for key,s in SOURCES.items():
        for relative,digest in s['selected_files'].items():
            path=ROOT/'sources'/key/relative
            (unchanged if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest else changed).append(f'{key}/{relative}')
    assert not changed,changed
    before=json.loads((ROOT/'evidence/INTEGRITY_AUDIT.json').read_text()).get('deliverable_sha256',{})
    production={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'umdiff').glob('*.py'))}
    modified=[p for p,h in production.items() if p in before and before[p]!=h]
    assert not modified,modified
    pdf=Path('/tmp/rdd_ssd_audit.pdf')
    report=dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='11 previously trained papers plus local RDD; 3 inference-only releases and 8 mapping-only papers separated. Pinned source snapshot audit, not arbitrary upstream HEAD certification.',
                conclusion='现有native fixtures的逐层与gradient相等仍有效；完整论文recipe/所有source branches/任意组件组合尚未完整兼容。',
                findings=FINDINGS,methods=records,inference_only_scope=extra,mapping_only_scope=closed,
                probes_file='PROBES.json',counterexample_count=len(PROBES['checks']),gpu_used=False,
                model_or_data_downloaded=False,author_source_files_unchanged=len(unchanged),author_source_changed=changed,
                production_algorithm_files_modified=modified,production_sha256=production,
                ssd_paper_read=dict(url='https://arxiv.org/pdf/2603.08709',
                                    web_reader_limit='PDF exceeds web reader 38 MB limit; fetched PDF read-only to /tmp, pdftotext used.',
                                    sha256=hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else None),
                required_before_formal_comparison=['P1 recipe gaps and silent clock mismatches',
                    'native guided inference with full codec/text/label/null handling',
                    'offline pretrained conversion and real-data preprocessing/optimizer recipe',
                    'small real-data end-to-end training and checkpoint-resume/inference',
                    'shared data/compute/evaluation FID or FVD protocol; quality must be measured, never inferred from local tensor parity'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    esc=html.escape
    doc=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>Unified 兼容性细节审计</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:auto;padding:24px;color:#182333;background:#f6f8fc}a{color:#175daf}article,section{background:white;padding:20px;margin:18px 0;border:1px solid #d7dfed;border-radius:10px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #d7dfed;text-align:left;vertical-align:top}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}.p1{color:#9c3322}.p2{color:#755714}</style>',
         '<h1>论文—作者代码—Unified：兼容性细节审计</h1>',f'<p>{esc(report["audited_at_utc"])} · CPU · 固定source commit · 未修改算法/作者代码</p>',
         '<section><h2>结论与证据边界</h2><p><b>确有缺项，而且部分会改变loss、gradient或采样结果，另有合法配置的训练/采样clock错配。</b>现有逐层parity只证明执行过的native专用小配置；不能外推完整论文配方、所有flags、GPU kernels或最终生成质量。</p>',
         f'<p>本轮 {len(PROBES["checks"])} 个CPU反例包含10个使用明确toy prediction/dependency的隔离分支探针，以及真实缩小作者UViT的参数梯度、真实PPFlow两/三阶段网络与原版ODE sampler的guided采样对照。它们不认证完整recipe或生成画质。源码完整性：{len(SOURCES)}个仓库、{len(unchanged)}个固定作者文件未变；umdiff生产代码未改。</p>',
         '<p><a href="PROBES.json">反例原始数值</a> · <a href="AUDIT.json">逐项证据/固定commit/修复要求</a> · <a href="../activation_parity/REPORT_ZH.html">此前逐层parity</a></p>',
         '<p>P1：正式native配方/合法配置需补；P2：可选分支、runtime或辅助功能。source/paper差异与作者未发布功能独立标注，不能当作我们重写成功。</p></section>',
         '<h2>逐篇结论</h2><table><tr><th>方法</th><th>已有覆盖与剩余范围</th><th>发现</th></tr>']
    for item in records:
        name=esc(item['name'])
        link=f'<a href="{esc(item["paper_url"])}">{name}</a>' if item['paper_url'] else name
        ids=' '.join(f'<a href="#{id}">{id}</a>' for id in item['findings'])
        doc.append(f'<tr><td>{link}</td><td>{esc(item["assessment"])}</td><td>{ids}</td></tr>')
    doc.append('</table><h2>已跑的反例</h2><table><tr><th>反例</th><th>数值/结果</th></tr>')
    for item in PROBES['checks']:
        values={k:v for k,v in item.items() if k not in ('name','status','interpretation')}
        doc.append(f'<tr><td>{esc(item["name"])}</td><td><pre>{esc(json.dumps(values,ensure_ascii=False,indent=2))}</pre></td></tr>')
    doc.append('</table><h2>逐项代码证据与后续工作</h2>')
    for f in FINDINGS:
        doc.append(f'<article id="{f["id"]}"><h3 class="{f["priority"].lower()}">{f["id"]} · {f["priority"]} · {esc(f["title"])}</h3><p>{esc(f["detail"])}</p><p><b>影响：</b>{esc(f["impact"])}</p><p><b>需补：</b>{esc(f["required_work"])}</p><ul>')
        for ref in f['evidence']:
            label=f'{ref["file"]}:{ref["line"]}'
            local='../../'+ref['file']
            public=f' · <a href="{esc(ref["public_url"])}">固定commit代码</a>' if ref['public_url'] else ''
            doc.append(f'<li><a href="{esc(local)}">{esc(label)}</a>{public}</li>')
        doc.append('</ul></article>')
    doc.append('<section><h2>推理release与paper-only范围</h2>')
    for item in extra:doc.append(f'<p><b>{esc(item["id"])}：</b>{esc(item["assessment"])}</p>')
    doc.append('<p>以下仍只是文献映射/接口：'+esc('、'.join(p['name'] for p in closed))+'。本轮不把它们计入native复现。</p></section>')
    doc.append('<section><h2>哪些trick已经处理，哪些来自原仓库</h2><p>已处理：Relay重叠rolling noise与冗余draw；SSD真实adjoint和非各向同性posterior；PixelFlow packed序列、per-image reduction、noise jitter和CFG ramp；LapFlow两/三层不同noise coupling；PPFlow可逆channel permutation；MDM gamma shift与预测坐标转换；DiMR TD-LN与backward重算；RDD detached mean的variance VB。这些不能再误列为缺失。</p><p>原版限制：DFM release架构不同论文、某3stage schedule被拒；SSD BilinearOp缺少输出尺寸属性需fixture显式兼容；LapFlow某head使用遗留loop变量；PPFlow部分learn_sigma=False几何存在原版限制。与原版bug相同只叫source parity，不代表paper intended behavior。可选GAN不应无依据列作所有论文的必要功能。</p></section>')
    doc.append('<section><h2>统一数学表达需要哪些更明确的对象</h2><p>已有E/Pi/F/B/K/phi/rho/H/T/W/Q/S/O/D轴可描述大部分行为，但实现需把W从标量提升为含频率矩阵、per-example active mask与精确reduction的算子；rho包含跨尺度batch分配和curriculum；phi区分path index、original time与network time；S包含逐band/stage CFG、通道mask和threshold；O包含初值转换、多网络更新顺序及参数冻结。每个组件须声明state空间、可识别性和输入/输出约定。数学上能写入这些对象，不代表现有JSON里已经存在对应实现。</p><p>这些缺口大多可以工程补齐，没有发现“这类方法原则上无法放进同一抽象”的证据。但仅用一个普通diffusion loss、一个upsampler和一个通用Euler循环无法忠实涵盖全部方法；保留专用pipeline和强契约是必要的。</p></section></html>')
    (OUT/'REPORT_ZH.html').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(findings=len(FINDINGS),methods=len(records),cpu_counterexamples=len(PROBES['checks']),source_files_unchanged=len(unchanged),production_modified=modified,output=str(OUT/'REPORT_ZH.html'))))


if __name__=='__main__':main()
