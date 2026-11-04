"""Freeze the pre-implementation review and source anchors."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = {
 'relay': (['blurring.py', 'training/loss.py', 'training/networks.py', 'generate_imagenet.py'],
  'Block DCT heat blur, EDM clean prediction and preconditioning, truncated lognormal sigma, rolling-window correlated noise, stochastic second-order blur sampler. Noise is overlapping cyclic rolls, NOT block-constant repetition. Preserve redundant RNG draws and fresh corrector noise for seeded parity.'),
 'ssd': (['guided_diffusion/scale_space_diffusion.py', 'guided_diffusion/ssd_math_util.py', 'guided_diffusion/ssd_utils.py', 'guided_diffusion/unet.py'],
  'Cumulative antialiased bilinear M, clean target at previous resolution rs, adjoint MT, anisotropic posterior covariance sampled with Lanczos and power-iteration PSD clamp. BilinearOp.MT references missing H_out/W_out attributes: explicit compatibility repair needed. DDP barriers need single-process gloo or isolated no-op communication, not math changes.'),
 'speed': (['utils.py', 'latent_image_gen.py', 'pixel_image_gen.py', 'comfyui/speed_sampler.py'],
  'Inference-only public release. DCT/Haar/FFT spectral expansion uses NumPy per-leading-index RNG order, float32, scipy/pywt, kappa=r/(1+(r-1)t), aligned t=t*kappa. Public time is noise-oriented. FFT takes real part of non-Hermitian coefficients; do not replace with a different Hermitian noise coupling under an exact-native claim.'),
 'pixelflow': (['pixelflow/data_in1k.py', 'pixelflow/model.py', 'pixelflow/pipeline_pixelflow.py', 'pixelflow/scheduling_pixelflow.py', 'train.py'],
  'Repeated bilinear D without antialias, nearest U, gamma=-1/3 corrected stage start, local tau endpoint difference velocity, packed per-image loss. Native 2x2 block Gaussian has diagonal epsilon jitter 1e-6; iterative sampling is CPU. Scheduler rounds state and dt to fp32. Packed training requires FlashAttention; provide isolated CPU SDPA varlen replacement and quantify kernel difference.'),
 'dfm': (['loss.py', 'dfm_utils/laplacian_decomposer.py', 'dfm_utils/samplers_dfm.py', 'models/sit_dfm.py', 'README.md'],
  'Released SiT/REPA reimplementation, NOT original paper architecture. Bilinear/bilinear Laplacian, product states with independent IID noise at ALL levels, t=1 noise, sampled last-active stage, earlier logit-normal(-2,1), unnormalized sum of active per-scale MSE. Sampler schedule stored fp32 despite intermediate fp64, fp64 Euler state with model dtype restoration. Optional REPA needs a teacher; no teacher downloaded.'),
 'lapflow': (['sampler/ours_sample.py', 'sampler/edify_image_sample.py', 'sampler/ours_sample_joint.py', 'models/MultiScaleDiT.py', 'feb12_train_ours.py', 'README.md'],
  'Main entry feb12_train_ours + ours_sample, not historical Pyramid-Flow. Bilinear D without AA and nearest U, fine-to-coarse residual state; 2-level noise is decomposed from one full Gaussian, 3-level public default generate_multiscale_noise instead uses full-resolution plus independent coarse IID. Delayed per-band data alpha=(t-a)/(1-a), sigma=1-t; global-time velocities. MoT forward final_layers uses the final loop i for all outputs (preserve native behavior explicitly, optional corrected-head transplant). CUDA-only time tensors need device compatibility.'),
 'ppflow': (['modules/invertible_sample.py', 'modules/patch_emb.py', 'models.py', 'transport/transport.py', 'transport/path.py', 'transport/integrators.py'],
  'Invertible channel-spatial permutation preserves all state dimensions; NOT lossy downsampling. 2/3 stages include width-only layout. No renoising. Ordinary FM in permuted coordinates with global-time target, native packed heterogeneous batches and separate patch heads. Native rejection time sampling may return too few samples for tiny batches; expose both exact-native sampler and an explicitly labeled direct-stratified replacement.'),
 'nami': (['src/infer.py', 'src/scheduling_flow_matching.py', 'src/pipeline_flux.py', 'src/transformer_flux.py'],
  'Inference source only. Learned BridgeFlow nearest upsample times scalar plus per-pixel learnable shift (16x64x64 and 16x128x128), not additive Gaussian. Separate sigma/local model clocks, correction gamma=+1/3, progressive Flux depth and stage position embeddings. Preserve public scheduler float promotion; no author training pipeline can be compared.'),
 'mrflow': (['flux1_mrflow.py', 'mrflow_utils.py', 'examples/direct_sigma_refine.py', 'README.md'],
  'Inference method, not new training objective. Low-resolution complete Flux sampling, VAE decode to PIL, RealESRGAN x2 then PIL resize, VAE encode and sigma=.12 direct linear refine. Exact pixel pipeline requires supplied VAE/SR weights; test orchestration with explicitly synthetic codec and never claim this is native perceptual parity.'),
 'dctdiff': (['DCT_utils.py', 'datasets.py', 'utils.py', 'sde.py', 'libs/uvit.py', 'train.py'],
  'Fixed block-DCT representation: RGB255 to YCbCr, 4:2:0 bilinear chroma, tokens ordered 4Y Cb Cr, divide frequency-specific eta, zigzag truncate, decode clip[-2,2] and uint8. VPSDE scales SNR through modified cumulative beta, U-ViT token architecture. Distinguish token, latent and pixel parity.'),
 'diffusability': (['src/training/loss.py', 'src/training/diffusers_ae.py', 'src/utils/training_utils.py', 'README.md'],
  'Codec training regularizer, not progressive sampler. Decode resized latent to resized RGB target, mean or align_corners=True bilinear, optional DCT cut. Compare native pixel-reconstruction branch without pretrained LPIPS; full GAN/perceptual training requires user-provided networks/assets. Repository setup references AlphaFlow.'),
 'mdm': (['ml-mdm-matryoshka/ml_mdm/diffusion.py', 'ml-mdm-matryoshka/ml_mdm/samplers.py', 'ml-mdm-matryoshka/ml_mdm/models/nested_unet.py', 'ml-mdm-matryoshka/configs/models/cc12m_256x256.yaml'],
  'Joint high-to-low product states, nested UNet, independent redrawn low-resolution noise despite initially pooled noise. Schedule-shift power configurable; model v prediction but released 256 config uses DDPM noise loss after conversion, double loss, no_use_residual=True. Obsolete residual branch references undefined ratio. mlx.data reader dependency can be isolated for synthetic fixtures, without altering model/sampler bodies.'),
 'dimr': (['sde.py', 'train.py', 'libs/core/blocks.py', 'libs/core/sigmoid/module.py', 'configs/DiMR-XL-2R_imagenet256.py'],
  'Multi-resolution backbone, full-resolution VP input; multi-head normalized downsampled epsilon targets, coefficients[.25,1], training log_snr=n/1000*8-4 (not actual logSNR). Native time-dependent normalization has a pure torch custom=False branch. Keep separate training/inference conditioning conventions.'),
 'fresco': (['README.md'], 'Public repository contains README and LICENSE only: no executable author source. Skip native reproduction, retain paper-level component mapping.'),
 'avits': (['README.md'], 'Public repository README says core inference code pending. Skip native reproduction, retain adaptive layout interface.'),
}

def main():
    sources = json.loads((ROOT/'evidence/sources.json').read_text())
    rows = {}
    for key, (paths, notes) in REVIEWS.items():
        anchors = {}
        for rel in paths:
            p = ROOT/'sources'/key/rel
            if not p.is_file():
                raise FileNotFoundError(p)
            anchors[rel] = {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                            'lines':len(p.read_text().splitlines())}
        rows[key] = {'commit':sources[key]['commit'], 'anchors':anchors,
                     'review':notes, 'status':'no_executable_release' if key in ('fresco','avits') else 'reviewed_before_framework_implementation'}
    closed = {
        'fdm':'https://jiataogu.me/fdm/', 'edify':'https://arxiv.org/abs/2411.07126',
        'pddpm':'https://arxiv.org/abs/2208.01864', 'rmd':'https://arxiv.org/abs/2603.06136',
        'ddit':'https://arxiv.org/abs/2602.16968', 'cat':'https://arxiv.org/abs/2501.03120',
        'lot':'https://georgenakayama.github.io/lotdiffusion/',
    }
    result = {'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'framework_core_implementation_has_started':False,
              'public_repositories':rows,
              'no_official_executable_repository_confirmed':closed,
              'new_papers':{'mdm':'https://arxiv.org/abs/2310.15111','dimr':'https://arxiv.org/abs/2406.09416','lot':'https://arxiv.org/abs/2610.05816'},
              'validation_contract':'Native function/architecture bodies, fixed synthetic inputs and random initial weights. Independent component ports compared to native operations, then training gradients/optimizer and sampling trajectories. Reusing native backbone alone does not count as independent algorithm parity. No quality/FID claim without assets and data.'}
    (ROOT/'evidence/PREIMPLEMENTATION_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(rows),'source reviews complete; core implementation may start.')

if __name__=='__main__':
    main()
