#!/usr/bin/env python3
"""Fail-closed aggregation of the executed CPU parity matrix and every gap ID."""
import argparse,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KINDS=('forward_input','forward_output','activation_gradient','parameter_gradient')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def audits(r):
    a=r.get('activation',r.get('variants',[]))
    return [a] if isinstance(a,dict) else a
def strict(r,training=True):
    assert r['status']=='passed',r.get('profile',r.get('method'))
    for a in audits(r):
        assert not a['structural_mismatches'],a['name']
        for k in KINDS:
            v=a['totals'][k]
            assert v['pairs']==v['byte_exact'] and v['max_abs']==0,(a['name'],k,v)
            if training:assert v['pairs']>0,(a['name'],k)
    assert not any(c.get('bitwise') is False for c in r.get('checks',[]))
def totals(r):
    out={k:0 for k in KINDS}
    for a in audits(r):
        for k in KINDS:out[k]+=a['totals'][k]['pairs']
    return out

# Status names distinguish actual fixes from a verified scoped test and from
# unavailable prerequisites. No synthetic test closes a real-asset FID item.
NOTES={
 'C01':('partial_cpu','已直接执行作者 forward/loss 和原 backward 语句；没有调用共同 Trainer。完整官方 CLI 的数据、保存、评估生命周期仍未认证。'),
 'C02':('requires_assets_or_formal_run','33 个完整 native CLI 认证仍为 false；本报告认证的是源码核心逐层对照，不篡改 CLI 认证字段。'),
 'C03':('partial_cpu','修正 DFM-SiT 模型/exp-name、LapFlow 全部选中 flags、RDD ImageNet、Relay flags；实际执行 DFM/LapFlow/RDD parser。完整资产/lifecycle overlay 不等同于 parser 成功。'),
 'C04':('fixed_and_verified_cpu','两个 backend 的 33 份最终 resolved 配置已落盘，并验证入口引用及实际解析控制项。'),
 'C05':('requires_cuda_distributed','CPU 对照使用 micro1/2/3/4/8，逐报告记录；未冒充官方 8/16 rank 或全局 batch 的轨迹。'),
 'C06':('boundary_acknowledged','paper 源码合同和 fair-rdd 新实验仍分别保存；公共 AdamW 128 步检查明确标为组件回归。'),
 'C07':('requires_assets_or_formal_run','不能从没有发布的 best-run 日志推测 seed/batch/预算；未知项保留。'),
 'C08':('partial_cpu','修复 drop_last、epoch 尾部和恢复游标；实际两个 production reader 索引片段对齐单 rank 官方 DataLoader 的不足整 batch 样本顺序。'),
 'C09':('partial_cpu','单 rank shuffle/3 个 epoch/恢复下一批已逐样本比对；真实多 worker、shard、rank 分配未认证。'),
 'C10':('partial_cpu','独立执行 DCTdiff 13 配置作者 PNG loader、flip、编码和真实类别字段；其他真实数据布局/统计/reader 未完整端到端运行。'),
 'C11':('fixed_and_verified_cpu','真实条件输入缺 label、错误 dtype/shape/range 均拒绝；合法多个类别通过。仅显式 synthetic 模式生成夹具 label。'),
 'C12':('requires_assets_or_formal_run','各核心测试直接使用同一模型域输入；没有声称真实 VAE rank-local posterior 编码生命周期相同。'),
 'C13':('requires_assets_or_formal_run','源码、架构 JSON、配置和 harness 已绑定 hash；没有取得的 pretrained VAE/teacher/text/SR 内容 hash 不能补造。'),
 'C14':('boundary_acknowledged','两侧共用作者 backbone 实现，独立组装路径/目标/loss/backward；明确是 integration parity，没有声称独立重写网络架构。'),
 'C15':('boundary_acknowledged','CPU attention、AST 提取、接口适配和导入 stub 都在报告/source_executions 中披露；原 CUDA kernel 单列。'),
 'C16':('requires_assets_or_formal_run','复用只读现有环境，记录依赖版本；未安装/改造官方旧版环境，更未修改 dirty FastVideo。'),
 'C17':('fixed_and_verified_cpu','比较 shape、dtype 和 uint8 原始字节；negative controls 检查 signed zero、dtype、NaN/Inf。'),
 'C18':('partial_cpu','128 步loss/原始grad/权重/EMA逐字节相同；重读39对临时checkpoint，完整model/EMA/Adam状态、全部param_groups及typed scalars未过滤即相同。原正式CLI/scaler/文件序列化整体仍未认证。'),
 'C19':('fixed_and_verified_cpu','在任何 clip/repair/optimizer 操作前捕获参数 grad 和模块输出的 activation grad；没有用 nan_to_num 掩盖差异。CUDA unscale 另属 C29。'),
 'C20':('partial_cpu','导出每个执行模块、重复调用、输出 Tensor 路径、None-grad 原因和关键 functional 组件对照；没有声称捕获全部 ATen 操作。'),
 'C21':('fixed_and_verified_cpu','明确区分 ModuleList 等容器与未执行实体层；DiMR 相关 view 来自 checkpoint 首次 no_grad pass，重算路径实际捕获梯度。None 从未计作零梯度通过。'),
 'C22':('partial_cpu','33 份选中配置均完成当前 full-backbone 短测；合成模型域数据/随机共用非零权重、CPU FP32、RMD 缩网格等限制逐项保留。'),
 'C23':('fixed_and_verified_cpu','修补后 31 配置缩小网络、31 配置完整网络、12 方法各128步、12扩展各128步全部重新执行并做严格 byte gate。'),
 'C24':('fixed_and_verified_cpu','当前完整架构覆盖全部13 DCT、MDM256/1024、DiMR G/XL、RDD L/XL 等31配置，加完整 codec 和30层 Wan。旧16份历史 hash 不再承担本次结论。'),
 'C25':('partial_cpu','修 MDM setup RNG、FP32 累积 backward 次序、Relay 增强/loss 汇总、RMD warmup；真实初始化/callback/eval/原 AMP 完整生命周期未认证。'),
 'C26':('requires_assets_or_formal_run','临时随机检查 checkpoint 已验状态并清理；没有把它们当作各篇真实官方 CLI 的新进程 resume 证明。'),
 'C27':('partial_cpu','DCT 13 配置量化前 float decode、PixelFlow 浮点 API、三种 inference workflow 的浮点输出已比对；真实 pretrained 完整像素/质量仍缺资产。'),
 'C28':('partial_cpu','PixelFlow 新 ndarray BHWC/float32 adapter 经5种 Euler/Dopri5/CFG/shift 配置测试；NAMI bridge 显式接口映射。所有原文件 metadata 未统一认证。'),
 'C29':('requires_cuda_distributed','本次没有 GPU 授权且仅用 CPU；CUDA AMP/scaler/Flash/xformers/TF32 不能用 CPU 检查替代。'),
 'C30':('requires_cuda_distributed','原 8/16 卡 DDP/FSDP/all-reduce/rank RNG 未运行。'),
 'C31':('requires_assets_or_formal_run','没有启动百万步正式训练，没有重新训练所有论文主表和多 seed；forward/backward 零差本身不证明最佳 FID。'),
 'C32':('requires_assets_or_formal_run','没有下载真实数据/参考统计或执行50k FID/IS/视频主表评测。'),
 'C33':('requires_cuda_distributed','CPU 报告只记录检查耗时；不把它称为 GPU 速度、显存或 FLOPs/质量比较。'),
 'C34':('boundary_acknowledged','17 源快照1211个选定文件重新核 hash；CAT补充4个源码另有 hash清单。没有声称全部未知/未发布分支被审完。'),
 'C35':('fixed_and_verified_cpu','MDM256 原 FP32/1024 原 FP16 分开标注；RMD 为 prompt-only/image-free，模板元信息已修正。'),
 'C36':('requires_cuda_distributed','未来 GPU 计划不是执行证据；本次 CPU runner 可执行，未认证原 CUDA runner。'),
 'C37':('fixed_and_verified_cpu','33 profile 逐项绑定实际数据集/分辨率/任务；不再使用整篇论文数据集汇总替代 profile 数据。'),
 'REL01':('fixed_and_verified_cpu','恢复实际 AugmentPipe p=.12/原随机 draw 次序，包含完整网络当前对照。'),
 'REL02':('fixed_and_verified_cpu','两配置 augment_dim=9，当前完整 backbone/layer/grad 对照通过。'),
 'REL03':('partial_cpu','修 high EMA half-life 为500000、low为50M；完整真实多 rank image clock/rampup 另需验证。'),
 'REL04':('partial_cpu','CPU microbatch 直接执行作者 sum(loss)*loss_multiplier.backward，low25/32、high1/256；原8rank/FP16 loss scale 未执行。'),
 'REL05':('partial_cpu','参考 CLI 已补原选中 flags；未加载真实 ZIP/权重启动完整8rank CLI。'),
 'REL06':('requires_assets_or_formal_run','CPU AugmentPipe 已测；真实 ZIP/dataset.json reader/label/data order 尚缺同数据完整 CLI 对照。'),
 'REL07':('requires_assets_or_formal_run','最佳阶段预算/low-high checkpoint 对尚无作者可绑定 run 记录。'),
 'REL08':('requires_assets_or_formal_run','随机小 cascade 回归通过；真实64→256最佳权重 cascade 未测。'),
 'REL09':('requires_cuda_distributed','原 xformers/FP16/8rank training lifecycle 未执行。'),
 'REL10':('fixed_and_verified_cpu','FFT DCT 运算树对齐；84/84组件检查 byte exact，max_abs=0。'),
 'SSD01':('partial_cpu','当前所选完整 CelebA64 四层架构及全部7条训练分辨率路径通过；其他论文主表配置仍未形成最佳训练结果。'),
 'SSD02':('requires_assets_or_formal_run','最佳 checkpoint/停止预算与评测行未定位到正式 run。'),
 'SSD03':('requires_assets_or_formal_run','元信息保留官方 split 要求；没有真实 CelebA split 数据验证。'),
 'SSD04':('partial_cpu','受控 runner 独立 steps 上限已可用；作者无限循环 Hydra 原入口尚未建立正式 stop-only adapter。'),
 'SSD05':('boundary_acknowledged','CPU H_out/W_out 属性 shim 与 rank0 依赖明确列出；不是原 Hydra/DDP CLI 的认证。'),
 'SSD06':('partial_cpu','完整64架构/7路径和缩小128步采样通过；未下载其它主表权重/真实数据运行质量评测。'),
 'PIX01':('requires_assets_or_formal_run','公开示例不自动等于 best run；最佳 budget/batch 未补造。'),
 'PIX02':('requires_cuda_distributed','完整31配置含 PixelFlow backbone 和 packed CPU SDPA，原 Flash/BF16 未运行。'),
 'PIX03':('boundary_acknowledged','实际 NumPy RoPE shim 明确记录，未认证作者旧依赖的原 CUDA 环境。'),
 'PIX04':('partial_cpu','resolved contracts 已落盘并通过解析；epoch-only 原 CLI 的全部 fair overlay/lifecycle 未认证。'),
 'PIX05':('fixed_and_verified_cpu','ndarray BHWC float32 adapter 与原 pipeline 的5个配置逐像素、dtype/API 完全相同。'),
 'PIX06':('requires_assets_or_formal_run','没有真实 pretrained checkpoint/text-to-image完整训练/50kFID。'),
 'PIX07':('partial_cpu','完整随机256模型 forward/backward，Euler/Dopri5 小采样/API/RNG 回归通过；真实最佳权重 sampler 仍无证据。'),
 'DFM01':('unavailable_release','发布 README 的 SiT/REPA reimplementation 与论文 DiT/SE-FLUX 差异保留，未伪造私有架构复现。'),
 'DFM02':('boundary_acknowledged','分别保留公开模板与论文正文 optimizer 配方，不混写为同一 best run。'),
 'DFM03':('requires_assets_or_formal_run','仅已发布256 latent配置完整 backbone；论文512/1024/video主表未正式复现。'),
 'DFM04':('unavailable_release','私有 SE-FLUX 数据/确切 codec/run 未发布，不能取得原 best 资产。'),
 'DFM05':('fixed_and_verified_cpu','补 exp-name 与 DFM-SiT-B/2、DFM-SiT-XL/2，实际 parser/model 类型检查通过；不会误走普通 SiT baseline。'),
 'DFM06':('partial_cpu','local teacher_features 的成对入口接通，REPA scalar reduction 与作者 train.py 相同；完整 XL/projector raw grad 通过。真实 DINO/预处理仍缺资产。'),
 'DFM07':('requires_assets_or_formal_run','未以随机latent替代真实预计算VAE mean/std或 SD-VAE-ft-mse。'),
 'DFM08':('partial_cpu','当前实际 loss/backward/阶段两分支/REPA 和128步通过；原 AMP/warmup/full checkpoint/质量未认证。'),
 'LAP01':('fixed_and_verified_cpu','dataset 修为 imagenet_256；实际作者 parser 验证。'),
 'LAP02':('fixed_and_verified_cpu','所选完整模型/尺度/通道/mask/T/class/epoch flags 已补齐并核实 parser 结果。'),
 'LAP03':('boundary_acknowledged','CLI default 和 paper flags 两配置分别对照，原正式 recipe/FID 仍未认证。'),
 'LAP04':('requires_assets_or_formal_run','未取得真实EQVAE与posterior资产。'),
 'LAP05':('requires_assets_or_formal_run','CFG2 明确是测试选择，不宣称最优 FID。'),
 'LAP06':('unavailable_release','发布 square sampler 存在源码调用缺陷；没有静默修作者文件/把失败分支伪报认证。'),
 'LAP07':('partial_cpu','补 native 三层训练+CFG采样128步；仍是缩小网络/合成latent，非真实最佳三层训练实验。'),
 'LAP08':('requires_assets_or_formal_run','HQ512/1024主表recipe/完整训练与最佳权重未纳入正式实验。'),
 'LAP09':('requires_assets_or_formal_run','原完整epoch/data/cosine/resume/正式质量未运行。'),
 'LAP10':('mathematical_roundoff_explained','715项中仅两项是原图分解后重构恒等式的FP32舍入2.3841858e-7；native两侧所有逐层值/grad byte exact。'),
 'PPF01':('partial_cpu','SiT shape/权重转换小分支回归通过；未取得真实SiT预训练权重进行fine-tune。'),
 'PPF02':('requires_assets_or_formal_run','1400epoch与1M更新正式预算没有据真实数据量等同。'),
 'PPF03':('partial_cpu','CPU runner 使用官方packing micro8直接对照；原CLI完整accum/max-update/localVAE overlay未认证。'),
 'PPF04':('unavailable_release','公开2/3层均完整 backbone 和128步通过；未发布FLUX/4-5层及其他论文主结果不能由配置名补造。'),
 'PPF05':('partial_cpu','原2/3层packing与真实backward语句通过；发布learn_sigma=False unpatchify有源码缺陷，任意micro或资产组合不宣称通过。'),
 'PPF06':('requires_assets_or_formal_run','小solver/CFG分支通过不等于最佳checkpoint与完整评测。'),
 'PPF07':('fixed_and_verified_cpu','stage端点插值/velocity运算树修正，484/484 byte exact，prediction/grad/weight max_abs=0。'),
 'PPF08':('requires_assets_or_formal_run','未做正式成本计量；记录必须包含SiT pretraining。'),
 'DCT01':('unavailable_release','已发布13个UViT完整网络通过；论文未发布最强DiT分支没有作数值复现。'),
 'DCT02':('fixed_and_verified_cpu','必需真实label guard及作者PNG loader真实类别937已对照。'),
 'DCT03':('partial_cpu','13 profile 作者PNG loader/flip/DCT编码对照完成；真实JPEG导出、split/FID stats未认证。'),
 'DCT04':('requires_cuda_distributed','原4A100/FP16/Accelerate/DataLoader完整CLI未运行。'),
 'DCT05':('fixed_and_verified_cpu','全部13配置均重新运行当前完整UViT架构，逐层forward/activation grad/parameter grad全byte exact。'),
 'DCT06':('requires_assets_or_formal_run','配置统计/EBFR已保留，随机训练/采样回归不等于所有数据集真实权重与正式质量。'),
 'DCT07':('fixed_and_verified_cpu','13 profile作者decode仅去最终uint8 cast暴露float64边界；量化前与uint8两种逐像素bytes相同。'),
 'MDM01':('fixed_and_verified_cpu','拆分仅导入类与构建模型，setup不再构建丢弃网络消耗RNG；256/1024输入全提供时三类RNG保持不变。'),
 'MDM02':('unavailable_release','CC12M YAML不等于Flickr50M最佳模型/完整curriculum；公开资料缺口保留。'),
 'MDM03':('requires_assets_or_formal_run','最佳global batch/rank/阶段budget缺作者run不能推测。'),
 'MDM04':('requires_assets_or_formal_run','未下载vis_model_256x256.pth；随机网络检查不认证真实初始化。'),
 'MDM05':('requires_assets_or_formal_run','完整1024随机模型明确禁用inner pretrained；真实checkpoint/curriculum未测。'),
 'MDM06':('partial_cpu','mixed-resolution/VDM/scale/CFG等组件128步回归；在线T5/tokenizer/negative text/streaming真实reader缺资产。'),
 'MDM07':('boundary_acknowledged','reader/MLX/S3/torchinfo导入stub明确披露；未以stub替代真实数据调用。'),
 'MDM08':('partial_cpu','精度元信息已修，FP32 backward先于accum除法的原次序也已对照；1024原FP16 scaler尚未执行。'),
 'MDM09':('requires_assets_or_formal_run','未取得T5/vision/inner最佳权重进行正式1024像素/画质验证。'),
 'MDM10':('fixed_and_verified_cpu','当前256/1024 full backbone与loss/backward、128步和采样分支重新执行，替代历史不匹配hash证据。'),
 'DIM01':('requires_cuda_distributed','原16rank/FP16/Accelerate/worker整CLI未执行。'),
 'DIM02':('requires_assets_or_formal_run','未加载真实 CompVis mean/logvar/label npy 数据验证 posterior 生命周期。'),
 'DIM03':('partial_cpu','已发布 CFG dropout/guidance 组件当前回归通过；真实pipeline文本/资产质量未认证。'),
 'DIM04':('requires_assets_or_formal_run','800epoch与1M更新关系没有原数据量/run日志不能认证。'),
 'DIM05':('fixed_and_verified_cpu','完整DiMR G/XL两个当前模型层级forward/rawgrad检查均通过。'),
 'DIM06':('fixed_and_verified_cpu','逐Tensor无grad原因记录：reentrant checkpoint首次no_grad的ChannelFirst view；重算中真正损失路径的梯度均已捕获并相等。'),
 'DIM07':('requires_assets_or_formal_run','真实VAE/最佳G-XL weights和正式FID未执行。'),
 'DIF01':('unavailable_release','作者训练/eval/checkpoint说明缺完整best recipe；完整codec组件不能补出best run。'),
 'DIF02':('boundary_acknowledged','SE2/4 KL0-SE1属于明确组件实验，与source默认KL1-SE0分开；不冒称定位到最佳配置。'),
 'DIF03':('requires_assets_or_formal_run','完整Hydra/data/teacher训练合同未认证；本次直接作者codec/loss函数对照。'),
 'DIF04':('partial_cpu','补全SD式83.65M VAE/256grid +真实VGG19代码三尺度/冻结encoder，全部层及rawgrad相等；权重随机，非真实pretrained资产。'),
 'DIF05':('unavailable_release','时序/GAN/tokenizer/distill-AE等论文扩展整pipeline未全部接入；不把codec分支证据外推。'),
 'DIF06':('requires_assets_or_formal_run','没有codec换后下游正式生成器重训和主表质量。'),
 'RMD01':('requires_assets_or_formal_run','完整公开1.3B backbone对照不认证论文14B最佳实验。'),
 'RMD02':('boundary_acknowledged','发布global64和论文video batch4分别保留，未混成一种配置。'),
 'RMD03':('requires_assets_or_formal_run','JourneyDB/VidProM真实prompts、负条件和text模型未绑定/加载。'),
 'RMD04':('fixed_and_verified_cpu','修为prompt-only/image-free，不再要求真实video作为论文主实验必需数据。'),
 'RMD05':('partial_cpu','30层Wan完整1.3B结构、transition1/2各17步，覆盖15步warmup后的fake/student backward及双Adam；真实21x60x104网格/预训练初始化未测。'),
 'RMD06':('partial_cpu','CPU双optimizer/clip/state+floatdecode通过；原BF16/FSDP/AMP/正式checkpoint resume未认证。'),
 'RMD07':('unavailable_release','GAN和SDXL/PixArt/SD3.5各实验pipeline未完整接入/对照。'),
 'RMD08':('boundary_acknowledged','数值结论归于固定发布commit，作者归属限制仍明确保留。'),
 'RMD09':('requires_assets_or_formal_run','主表质量/加速/完整CLI未正式跑。'),
 'SPD01':('partial_cpu','FLUX DCT/DWT/FFT当前源码workflow逐层前向+浮点decode通过；PixelGen/Wan完整路线未认证。'),
 'SPD02':('requires_assets_or_formal_run','真实pretrained/text/tokenizer/VAE缺绑定，不下载替代。'),
 'SPD03':('partial_cpu','三个变换BF16 CPU随机fixture完整浮点/uint8像素及逐层forward相同；真实video/pretrained采样未测。'),
 'SPD04':('requires_assets_or_formal_run','正式质量/原GPU速度显存未测。'),
 'SPD05':('boundary_acknowledged','公开training-free流程无训练loss，backward标N/A，不把0个grad计为通过。'),
 'NAM01':('unavailable_release','公开只有推理，没有作者trainer，不能从batch768恢复训练。'),
 'NAM02':('requires_assets_or_formal_run','随机bridge已测，真实训练bridge和transformer权重未取得。'),
 'NAM03':('partial_cpu','实际使用作者vendored Diffusers，模型/VAE一起clone保持SiLU别名；真实text/tokenizer/VAE权重未测。'),
 'NAM04':('partial_cpu','三阶段固定16x32/64/128几何，CFG1/3、模块逐层与浮点像素相同；每stage3步/小宽度非最佳pipeline认证。'),
 'NAM05':('boundary_acknowledged','无发布作者训练器，backward N/A；不伪造一项作者训练验证。'),
 'MRF01':('partial_cpu','源码FLUX完整随机workflow逐层通过；Qwen分支尚未认证。'),
 'MRF02':('requires_assets_or_formal_run','FLUX/Qwen/T5/VAE/RRDB真实权重缺绑定，不下载。'),
 'MRF03':('partial_cpu','actual main完整VAE→PIL→tile SR→resize→posterior→refinement、sigma.12/RNG对照通过；小随机资产，非原真实分辨率最佳模型。'),
 'MRF04':('requires_assets_or_formal_run','正式质量/速度显存未执行。'),
 'MRF05':('boundary_acknowledged','training-free source没有作者loss，backward N/A。'),
 'RDD01':('boundary_acknowledged','旧8x32与fair预算不同如实保留；CPU检查不宣称复现历史492k/407.6k训练轨迹。'),
 'RDD02':('fixed_and_verified_cpu','original参考补data-source imagenet，实际parser正确。'),
 'RDD03':('requires_assets_or_formal_run','真实ImageNet/ADM/SD-VAE-ema/posterior rank-local生命周期未完整认证。'),
 'RDD04':('fixed_and_verified_cpu','当前完整L和XL逐层forward/原始backward、阶段汇总均完全相同。'),
 'RDD05':('partial_cpu','六历史目标变体+multires mapped-time原始loss/grad/三solver回归；所有可选REPA/minSNR/loss-aware组合未穷举。'),
 'RDD06':('fixed_and_verified_cpu','DiscreteVP effective-noise、FM clean重构、native stage mean汇总运算树对齐；412/412通用组件byte exact，native/扩展loss与grad也全byte exact。'),
 'RDD07':('requires_assets_or_formal_run','已删旧GPU payload不能恢复，未补造历史数据/权重；新临时检查产物验收后清理。'),
 'RDD08':('requires_assets_or_formal_run','未运行真实完整CLI正式训练/新进程resume/同预算quality。'),
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=ROOT/'evidence/forward_backward_20261012');a=p.parse_args();base=a.evidence.resolve()
    manifest=read(ROOT/'configs/paired_experiments/manifest.json');profiles=[v for v in manifest['profiles'] if v['paper'] not in ('diffusability','rmd')]
    current=read(base/'full_current/SUMMARY.json');assert not current['remaining'] and len(current['results'])==31
    reduced=read(base/'reduced_release/SUMMARY.json');assert not reduced['remaining'] and len(reduced['results'])==31
    for v in profiles:strict(read(base/'reduced_release'/(v['id']+'.json')))
    rows=[]
    for v in profiles:
        path=base/'full_current'/(v['id']+'.json');r=read(path);strict(r)
        assert r['full_released_architecture'] and r['full_input_grid'] and len(r['steps'])==2
        assert all(s['raw_loss_bytes_equal'] and s['loss_bytes_equal'] and s['rng_equal'] for s in r['steps'])
        assert not r['cuda_initialized']
        assert r['config_sha256']==sha(ROOT/'configs/paired_experiments'/(v['id']+'.json'))
        assert r['harness_sha256']==sha(ROOT/'tools/validate_full_forward_backward.py')
        for k in ('implementation_sha256','verification_sha256'):
            for rel,h in r[k].items():assert sha(ROOT/rel)==h,(v['id'],rel)
        rows.append(dict(profile=v['id'],method=v['paper'],report=str(path.relative_to(ROOT)),report_sha256=sha(path),parameters=r['parameters'],micro_batch=r['micro_batch'],counts=totals(r)))
    for profile,method,path in [('diffusability_se24','diffusability',base/'codec_current/diffusability.json'),('rmd_wan13b','rmd',base/'rmd/rmd_full_backbone.json')]:
        r=read(path);strict(r)
        rows.append(dict(profile=profile,method=method,report=str(path.relative_to(ROOT)),report_sha256=sha(path),counts=totals(r),parameters=r.get('details',{}).get('parameters') or r.get('details',{}).get('variants',[{}])[0].get('parameters'),micro_batch=2))
    for method in ('relay','ssd','pixelflow','dfm','lapflow','ppflow','dctdiff','mdm','dimr','diffusability','rmd','rdd'):
        r=read(base/'long_release'/(method+'.json'));strict(r);assert r['steps']==128 and not r['original_checks']['nonexact']
        assert r['harness_sha256']==sha(ROOT/'tools/validate_activations.py')
        for rel,h in r['implementation_sha256'].items():assert sha(ROOT/rel)==h
    for f in base.glob('extensions_current/*/*.json'):
        r=read(f);strict(r);assert r['harness_sha256']==sha(ROOT/'tools/validate_recipe_extensions.py')
    assert len(list(base.glob('extensions_current/*/*.json')))==12
    for method in ('speed','nami','mrflow'):
        r=read(base/'inference_layers'/(method+'_inference.json'));strict(r,False)
        assert r['harness_sha256']==sha(ROOT/'tools/validate_inference.py')
    for name, full, cases in [('cat_codec',False,16),('cat_codec_full',True,8)]:
        cat=read(base/name/'cat_codec.json');strict(cat)
        assert cat['full_released_architecture']==full and len(cat['cases'])==cases
        assert cat['configuration']['sample_size']==(512 if full else 64)
        assert cat['harness_sha256']==sha(ROOT/'tools/validate_cat_published_codec.py')
        assert cat['adapter_sha256']==sha(ROOT/'umdiff/compat/cat_codec.py')
        for rel,h in cat['source_sha256'].items():assert sha(ROOT/rel)==h
    cat_api=read(base/'cat_api/cat_api.json');strict(cat_api)
    assert len(cat_api['cases'])==8 and not cat_api['cuda_initialized']
    assert cat_api['harness_sha256']==sha(ROOT/'tools/validate_cat_api_contract.py')
    assert cat_api['adapter_sha256']==sha(ROOT/'umdiff/compat/cat_codec.py')
    assert read(base/'gap_regressions.json')['status']=='passed';assert read(base/'source_integrity.json')['status']=='passed'
    assert read(base/'ssd_resolution_branches.json')['status']=='passed'
    assert read(base/'checkpoint_state_bytes.json')['status']=='passed'
    for method in ('relay','ppflow','rdd'):
        assert all(c['bitwise'] for c in read(base/'components'/(method+'.json'))['checks'])
    lap_nonbyte=[c for c in read(base/'components/lapflow.json')['checks'] if not c['bitwise']]
    assert len(lap_nonbyte)==2 and all(c['name']=='lapflow/reconstruction' and c['max_abs']==2.384185791015625e-7 for c in lap_nonbyte)
    original=(ROOT/'ORIGINAL_UNIFIED_GAPS_TODO_ZH.md').read_text();items={}
    for line in original.splitlines():
        m=re.match(r'\| (C\d\d) \| (.*?) \|',line) or re.match(r'- \[[ x]\] ([A-Z]+\d\d)(.*)',line)
        if m:items[m[1]]=m[2].strip()
    assert len(items)==148 and set(items)==set(NOTES),(len(items),set(items)^set(NOTES))
    ledger=[dict(id=k,old_issue=v,status=NOTES[k][0],resolution_and_limit=NOTES[k][1]) for k,v in items.items()]
    (base/'GAP_CLOSURE_LEDGER.json').write_text(json.dumps(dict(date='2026-10-12',counts=Counter(x['status'] for x in ledger),items=ledger,all_148_fully_closed=False),ensure_ascii=False,indent=2)+'\n')
    lines=['# 2026-10-12：原清单148项逐项处理记录','', '此表区分已修复的CPU错误、已经实测的限定范围和仍缺少外部前提的验收。**148项并未全部关闭**；CUDA、多卡、私有代码、真实资产和正式主表实验没有被合成CPU测试代替。全部实测矩阵见 [完成报告](CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md)。','', '| 编号 | 本次状态 | 已做与保留范围 |','|---|---|---|']
    for x in ledger:lines.append('| '+x['id']+' | '+x['status']+' | '+x['resolution_and_limit'].replace('|','/')+' |')
    (ROOT/'CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md').write_text('\n'.join(lines)+'\n')
    profile_totals={k:sum(x['counts'][k] for x in rows) for k in KINDS}
    audit=dict(date='2026-10-12',status='passed',training_methods=12,training_profiles=33,strict_definition='equal shape,dtype,contiguous raw uint8 bytes; max_abs0; raw gradients before clipping/repair',profiles=rows,profile_tensor_pair_totals=profile_totals,
      full_profile_count=31,full_codec_count=1,full_wan_backbone_count=1,training_128_step_methods=12,extension_128_step_cases=12,inference_only_forward_methods=3,cat_additional_published_codec=True,cat_full_published_codec=True,cat_api_case_count=8,
      official_cli_certified=0,official_cli_total=33,paper_best_results_reproduced=0,all_148_gaps_closed=False,shared_author_backbones=True,pretrained_assets_used=False,real_data_used=False,gpu_used=False,
      gap_ledger='GAP_CLOSURE_LEDGER.json',source_integrity='source_integrity.json',cleanup='cleanup.json',environment='environment.json',checkpoint_pairs_verified=39,
      report_generator_sha256=sha(Path(__file__)),cpu_runner_sha256=sha(ROOT/'run_cpu_layer_parity.sh'),
      provenance_limits=['RMD full-backbone report binds source executions, recipe and local architecture config; its running wrapper did not record its own harness SHA at launch. This is not silently filled as run-time provenance.'])
    (base/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    coverage_path=ROOT/'PAPER_COVERAGE.json';coverage=read(coverage_path)
    coverage['cpu_verification_20261012']=dict(report='CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md',audit='evidence/forward_backward_20261012/AUDIT.json',training_methods=12,profiles=33,inference_only_methods=3,additional_published_codec='cat',full_paper_quality_verified=False)
    coverage['native_skipped_papers']=7
    coverage['partial_published_codec_releases']=1
    for row in coverage['methods']:
        if row['id']=='cat':
            row.update(status='published_codec_core_parity_partial',source_provenance='NeurIPS2025 official Supplemental-Conference.zip, SHA and4Python files recorded',
              skip_reason='Full diffusion trainer/full codec training objective and best assets absent; published codec forward/KL and diagnostic adjoint now verified.',
              implementation='Explicit CATCodecAdapter; independent encode/posterior/decode/KL assembly reuses published nested codec modules. Reduced and full classification.py512-grid1/2/3/mixed-rate mean/sample verification; no paper training or quality claim.',
              configured_components=[dict(file=str((base/'cat_codec_full/cat_codec.json').relative_to(ROOT)),configuration=read(base/'cat_codec_full/cat_codec.json')['configuration']),dict(file=str((base/'cat_codec/cat_codec.json').relative_to(ROOT)),configuration=read(base/'cat_codec/cat_codec.json')['configuration'])],
              verified_components=['published six-block nested codec','compression rates1/2/3 and mixed batches','masked posterior mean/sample','independent codec forward/KL composition'],
              source_release='https://papers.neurips.cc/paper_files/paper/2025/file/014898b821e3ce966ae1db68cb8d9966-Supplemental-Conference.zip',
              cpu_evidence='evidence/forward_backward_20261012/cat_codec/cat_codec.json',full_cpu_evidence='evidence/forward_backward_20261012/cat_codec_full/cat_codec.json',official_training_loss_verified=False)
    coverage_path.write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
    lines=['# CPU forward/backward逐层对照：2026-10-12','', '**12种有训练/codec/蒸馏核心的方法、33份选中配置，全部完成本轮逐层对照。所有实际捕获的模块输入、输出、activation gradient、处理前parameter gradient均逐字节相同，max_abs=0。** 31份image配置使用当前完整发布网络/输入尺寸，另有完整SD式VAE及30层Wan1.3B结构。','', '这不是所有论文最佳结果或完整官方CLI认证：两侧共用作者backbone，实现独立目标/loss/backward组装；CPU FP32、同一非零随机权重和合成模型域输入。RMD仍使用缩小视频网格、随机text/teacher。31配置各2次更新，RMD两transition各17次更新覆盖15步warmup，完整codec2次更新。','', '范围：Relay、Scale Space Diffusion、PixelFlow、DFM、LapFlow、PPFlow、DCTdiff、MDM、DiMR、Diffusability、RMD及本地RDD。RMD固定仓库自称官方且metadata作者匹配，但论文页未独立链接此仓库；归属限制仍保留，不能把仓库数值parity变成作者身份认证。', '', '## 每份完整网络配置的实际结果','', '| 配置 | 参数量 | micro | 输入Tensor对 | 输出Tensor对 | activation-grad对 | parameter-grad对 | 证据 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:
        c=x['counts'];lines.append(f"| {x['profile']} | {x.get('parameters')} | {x.get('micro_batch')} | {c['forward_input']} | {c['forward_output']} | {c['activation_gradient']} | {c['parameter_gradient']} | [JSON]({x['report']}) |")
    lines+=['', '对数是执行次数×模块边界Tensor数，不是独立网络层数。每个JSON的 `activation.coverage`/`layer_tensor_stats`/`gradient_coverage`/`absent_gradient_outputs` 给出模块路径、重复调用、首次差异与无梯度原因；RMD/codec用activation列表。','', '## 实际执行与一致性定义','', '- original侧执行固定源码的作者forward/loss函数或AST语句，以及作者backward语句；unified侧调用我们的Method。主31配置没有共用Trainer。','- 在相同初始state和输入下分别执行两张计算图，恢复相同Torch/NumPy/Python随机流；比较原始loss及标量objective，梯度先比对再更新。非零输出head避免全零网络假通过。','- 比较shape、dtype、uint8原始字节；signed-zero/dtype/nonfinite negative controls已通过。不是“误差小于某阈值就通过”。','- 12方法各128步完整模块回归，以及12个扩展case各128步已重新执行，loss/grad/weight/EMA/采样所有检查byte exact。复用官方网络、DPM solver、CPUattention及导入shim均显式记录。','- MDM256保留作者FP32 backward先于accum除法的次序；MDM1024原FP16控制并未在CPU模拟成CUDA认证。','- DiMR checkpoint首次no_grad pass的view没有grad_fn，真实重算输出梯度已捕获；冻结encoder/teacher和未调用容器不会冒充已测梯度。functional/ATen并非全部逐操作hook。','', '39对临时checkpoint另外实际重载：14,948个model/EMA/Adam张量以及全部保存的parameter groups、typed scalars和容器结构相同，没有过滤冻结参数组。文件ZIP的SHA单列，未宣称序列化文件相同。见 [checkpoint状态证据](evidence/forward_backward_20261012/checkpoint_state_bytes.json)。', '', '## 修正并重测的真实差异','', 'Relay完整增强/augment_dim/EMA/官方loss sum与DCT FFT；PPFlow stage端点/velocity运算树；RDD DDPM effective noise、FM clean和阶段先mean后sum；LapFlow官方whole-batch F.mse_loss；DFM REPA denoising.mean()+projection.mean()*weight及DFM-SiT CLI；PixelFlow packing标量归约/ndarray API；MDM setup RNG/FP32累积/1024文本夹具；DCT float64 decode/条件label；两个reader drop_last/epoch边界；Diffusability原B×1 reconstruction与B-vector KL广播及SE加法次序。修正未修改作者源文件。','', '通用组件：Relay84/84、PPFlow484/484、RDD412/412 byte exact。LapFlow713/715：仅两项是FP32分解→重构对原图恒等式的2.3841858e-7舍入；**native两侧训练/采样并无这个差异**，未把恒等式强行标byte exact。','', 'NAMI初次逐层检查还暴露了测试clone把模型/VAE共用SiLU别名打散的问题：现已一起deepcopy、保持别名图，并显式适配bridge参数/返回容器；数学和作者权重没有改。失败历史日志保留，最终证据使用修复后报告。','', '## 仅公开推理的方法与新发现的CAT源码','', 'SPEED、NAMI、MrFlow的作者推理workflow、执行模块输出及量化前浮点像素检查全部通过；它们的公开路径没有作者训练loss/backward，严格标记 **backward N/A**，没有用0个梯度冒充通过。真实pretrained资产不在本次范围。','', 'CAT的NeurIPS官方补充ZIP包含nested VAE两文件、scoring和classification。补充源码与独立组装codec在1/2/3和mixed compression、posterior mean/sample上通过：缩小六block网络2轮16case，以及classification.py原始512网格完整架构1轮8case，逐层输出和诊断adjoint均逐字节相同；但没有发布diffusion trainer或完整codec训练loss。诊断loss明确是pixel.square().mean()+作者KL，不冒称论文训练复现；不能据此把CAT加入上面的12种发布训练核心已验证方法。见 [CAT证据](evidence/forward_backward_20261012/cat_codec/cat_codec.json)、[CAT完整512配置](evidence/forward_backward_20261012/cat_codec_full/cat_codec.json)、[8个默认rate/返回tuple/显式generator API与诊断gradient case](evidence/forward_backward_20261012/cat_api/cat_api.json) 与 [发布复核](evidence/forward_backward_20261012/RELEASE_RECHECK_ZH.md)。','', 'CAT缩小夹具的通道全部设为32，改变了作者encoder中依赖channel差异的提前退出条件，因此该夹具的homogeneous-rate输出是16/32/64；这不是完整发布配置的行为。187,619,158参数的原classification.py配置已独立实测：8case全部输出512×512且逐层/梯度一致。完整配置的证据不能由缩小夹具代替。', '', '## 原清单的处理结果和不可混淆的结论','', '148项全部逐项登记了本次处理与剩余前提：[逐项记录](CPU_GAP_CLOSURE_LEDGER_20261012_ZH.md)。它包含真正不可能仅靠当前CPU任务关闭的事项：未发布的作者trainer/私有分支，缺失的真实VAE/T5/DINO/最佳权重/数据，原CUDA Flash/AMP和8/16rank分布式，百万步正式训练及50kFID。**这148项没有全部关闭；完整官方CLI仍为0/33认证，最佳主表仍为0行正式复现。**','', '逐层零差可以证明本次明确路径/配置下的实现一致性，不能证明随机模型具有论文最佳画质、所有输入分支均等价或真实训练能取得同样FID。未执行层、None grad、CPU替代、精度/网格偏离均保留明细。','', '## 证据绑定边界', '', '31份完整image配置绑定当前配置/harness/实现/tracer hash；12个128步方法和12个扩展绑定当前实现/harness。CAT绑定4份补充源码、adapter和harness。RMD完整报告绑定source_executions、recipe及本地架构JSON，其运行wrapper未在启动时记录自己的harness SHA；没有事后填成启动时证据。', '', '## 重跑与产物','', '重跑入口：[run_cpu_layer_parity.sh](run_cpu_layer_parity.sh)。最终机器可读索引：[AUDIT.json](evidence/forward_backward_20261012/AUDIT.json)，逐配置报告、逐层统计、当前源码/配置/harness hash和日志全部保留。测试使用现有只读Python环境，不下载模型/数据、不初始化CUDA、不启动正式长训练。临时checkpoint/像素Tensor在对照结束后按用户既定要求清理，清单与hash记录于cleanup.json；它们不是可恢复的正式训练模型。']
    (ROOT/'CPU_FORWARD_BACKWARD_COMPLETION_20261012_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',profiles=len(rows),gap_items=len(ledger),gap_statuses=Counter(x['status'] for x in ledger)),ensure_ascii=False))
if __name__=='__main__':main()
