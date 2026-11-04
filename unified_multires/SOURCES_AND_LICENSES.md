> **当前获取方式（2026-11-04）**：第三方源码不随Git提交。`sources.lock.json`现固定23个仓库、2172份源码文件，包含下文历史快照及Pyramid Flow。运行`python tools/fetch_sources.py --repo 名称`按需准备，`--all --verify`离线校验。以下本地` sources/ `链接在获取后可用；各仓库许可证仍分别适用。

# 源码、版本与许可证

本框架使用固定 commit 的作者网络类与明确标注的求解器 backend。算法对照通过原始函数体/模块执行；作者文件未改写。各项目适用各自许可证，未将作者源码统一改称本框架许可。

|来源|仓库|固定 commit|许可证文件|
|---|---|---|---|
|relay|[zai-org/RelayDiffusion](https://github.com/zai-org/RelayDiffusion)|`a97877fcee9e7515595e24f7d72bc5f6a722ce6f`|[LICENSE](sources/relay/LICENSE)|
|ssd|[prateksha/ScaleSpaceDiffusion](https://github.com/prateksha/ScaleSpaceDiffusion)|`68a6961983f0c2e19910b34f77ef16f7c6764130`|[LICENSE](sources/ssd/LICENSE)|
|speed|[howardhx/speed](https://github.com/howardhx/speed)|`ca7801c9bdffe681742e9592345bcf4885959be5`|[LICENSE](sources/speed/LICENSE)|
|pixelflow|[ShoufaChen/PixelFlow](https://github.com/ShoufaChen/PixelFlow)|`8805204d1be8df22382280959b3063974ff3d77d`|[LICENSE](sources/pixelflow/LICENSE)|
|dfm|[snap-research/dfm](https://github.com/snap-research/dfm)|`637fcab6b602686887873ed20e69d083efb26312`|[LICENSE](sources/dfm/LICENSE)|
|lapflow|[sjtuytc/ICLR26_LapFlow_Official](https://github.com/sjtuytc/ICLR26_LapFlow_Official)|`388610e8fd09f39f17c7bf5bd64135e6d1ce7d52`|[LICENSE](sources/lapflow/LICENSE); [old_baselines/Pyramid-Flow/LICENSE](sources/lapflow/old_baselines/Pyramid-Flow/LICENSE)|
|ppflow|[fudan-generative-vision/PPFlow](https://github.com/fudan-generative-vision/PPFlow)|`1f4dd1738a6da7feb7279ad9af12c7d93a07dd93`|No standalone license file in the selected release; see repository metadata/README.|
|nami|[360CVGroup/NAMI-T2I](https://github.com/360CVGroup/NAMI-T2I)|`4aa35bd7f2f76f1f86ce1241eeddc3297e14ef41`|[LICENSE](sources/nami/LICENSE)|
|mrflow|[Xingyu-Zheng/MrFlow](https://github.com/Xingyu-Zheng/MrFlow)|`108837b95e93fc1f69a69413109ba505dac0f9ab`|[LICENSE](sources/mrflow/LICENSE)|
|dctdiff|[forever208/DCTdiff](https://github.com/forever208/DCTdiff)|`b915fa1b8ba279e4b74efefe85bced681e3d8ace`|[LICENSE](sources/dctdiff/LICENSE)|
|fresco|[DarrenZheng303/Fresco](https://github.com/DarrenZheng303/Fresco)|`1e6cb624d87de0b6fd789ddbce9571d63a9f84d4`|[LICENSE](sources/fresco/LICENSE)|
|avits|[QHR69/AViTS](https://github.com/QHR69/AViTS)|`c5d4a4c96fbc8f5eff9e8a54294b03bdda59030b`|No standalone license file in the selected release; see repository metadata/README.|
|diffusability|[snap-research/diffusability](https://github.com/snap-research/diffusability)|`d5d51820a773d2eaddd365e71df52bdfa8d60050`|[LICENSE.txt](sources/diffusability/LICENSE.txt)|
|mdm|[apple/ml-mdm](https://github.com/apple/ml-mdm)|`633659d1db70afb9df9411e98b1b2147fc911d1e`|[LICENSE](sources/mdm/LICENSE)|
|dimr|[qihao067/DiMR](https://github.com/qihao067/DiMR)|`fa141ae401cde28fa0fb5d983fe00746e40b7cb1`|[LICENSE](sources/dimr/LICENSE)|
|realesrgan|[ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN)|`362a0316878f41dbdfbb23657b450c3353de5acf`|[LICENSE](sources/realesrgan/LICENSE)|
|rmd|[Skylar-aigc/RMD-Inference-Acceleration](https://github.com/Skylar-aigc/RMD-Inference-Acceleration)|`213a6691d19abe2463d1c91f4cb9a925a3a13d12`|[LICENSE](sources/rmd/LICENSE)|
|pixelgen（SPEED依赖）|[Zehong-Ma/PixelGen](https://github.com/Zehong-Ma/PixelGen)|`7b94115d6f14d1c4ce0fe7a17a06351934ef2d4a`|No standalone license file in the selected snapshot; see repository metadata/README.|
|wan21（SPEED依赖）|[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)|`9737cba9c1c3c4d04b33fcad41c111989865d315`|[LICENSE.txt](sources/wan21/LICENSE.txt)|

RDD 使用迁移后的 `multiscale_diffusion` 本地源码；其源文件 hash 见各 RDD oracle 证据。RealESRGAN 是 MrFlow 引用的确切 gitlink。RMD 仓库自称官方、metadata 作者名匹配，但论文页未链接它，来源归属未独立确认。

2026-10-12再次核对[RMD论文v2](https://arxiv.org/html/2603.06136v2)、[所选仓库](https://github.com/Skylar-aigc/RMD-Inference-Acceleration)和[owner页面](https://github.com/Skylar-aigc)，仍未找到能独立确认作者归属的对应链接。数值结论限定所选commit，不能将它升级为已独立确认的论文作者实现；[复核记录](evidence/unified_scope_20261012/rmd_provenance_recheck.json)保留。没有更新正在验收的源码，也没有联系他人或获取模型/数据。

最新保留核心算法的支持约定见 [SUPPORTED_REPRODUCTIONS_ZH.md](SUPPORTED_REPRODUCTIONS_ZH.md)。19份外部源码快照的2,077个选定文件由 `evidence/sources.json` 固定；新增依赖仅用于SPEED的公开PixelGen/Wan推理分支，不将它们另计为新的训练方法。临时CPU依赖安装在本目录 `_deps`，记录在 `requirements-authors.txt`；共享 FastVideo 仓库及其环境未修改。

预审记录：`evidence/PREIMPLEMENTATION_REVIEW.json`、`REALESRGAN_PREIMPLEMENTATION_REVIEW.json`、`RMD_PREIMPLEMENTATION_REVIEW.json`。早期 source-only 获取中额外 manifest/media 获取记录在 `evidence/acquisition_incident.json`；获取逻辑已修复。不包含模型或训练数据下载。

## 2026-10-12 CAT 官方补充源码

CAT 的 [NeurIPS2025官方补充ZIP](https://papers.neurips.cc/paper_files/paper/2025/file/014898b821e3ce966ae1db68cb8d9966-Supplemental-Conference.zip)包含4份Python源码。只保留源码与[ZIP目录/来源/SHA清单](evidence/forward_backward_20261012/cat_release/supplement_inventory.json)，没有保存ZIP/PDF/模型或数据。模型两文件用于codec对照；classification.py仅AST读取架构配置，没有执行下载/CUDA/下游训练脚本。补充材料未附单独许可证，不声称这些源码受本框架许可，不自动发布或推送。其训练器/完整训练objective未公开；codec诊断backward明确不能作为论文训练结果复现。

## 2026-10-13 共同网格实现前的源码审阅

本轮先获取并阅读下列source-only快照，版本及选中文件SHA256已追加至`evidence/sources.json`。它们属于候选审阅，不因下载就被计为已实现的方法。没有下载模型或数据，也没有修改作者文件。

|来源|固定commit|许可证与使用范围|
|---|---|---|
|[RALU](https://github.com/ignoww/RALU)|`3979f8cab54332ec2d16f8646b68fa134d258990`|快照未附独立LICENSE；保留外部源码身份，本轮没有复制其实现进公共引擎|
|[Just-in-Time](https://github.com/Wenhao-Sun77/Just-in-Time)|`818ed8de7333f0c83e3f88a006576036c0f931e8`|[LICENSE](sources/jit_spatial/LICENSE)，Apache-2.0；仅审阅其不规则token与Micro-Flow接口|
|[Inverse Heat Dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation)|`3f735cadf61609c30094e3aee5a37d82a301aecb`|[LICENSE](sources/inverse_heat/LICENSE)；仅审阅，本轮未把其真实数据模糊先验替换成Gaussian并冒称复现|

Pyramid Flow沿用已有固定作者checkout：`/scr/yuheng/bugswarm/official_video_baselines_20261004/reference/Pyramid-Flow`，commit `a012faa1dc4d71301a7a153c7f9554c081947ea2`，许可证为该库`LICENSE`。新测试执行原`add_pyramid_noise`函数体和原scheduler；共同模型独立于作者DiT。SPEED边界继续对照已固定的`speed/utils.py`。

f-DM、PyramidalWan、Edify、Non-Uniform、PDDPM及Bottleneck的新过程是独立论文实现/明确子族，不声称找到作者完整训练源码。论文URL、读取材料SHA、实现范围和本次数值报告在`evidence/cross_modal_20261013`；PyramidalWan另与迁移项目的独立paper math实现对照，该文件不能改称作者代码。所有模型只通过离线构造器创建随机参数。
