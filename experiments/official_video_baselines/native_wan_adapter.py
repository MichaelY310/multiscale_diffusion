"""Original Wan modules in FastVideo's modular training and comparison API.

Use this profile for source consistency. The separately tested FastVideo native
kernel/FSDP profile remains a performance option with disclosed BF16 drift.
No changes to either original repository or the shared dirty checkout.
"""
from typing import Any
import torch
import torch.distributed as dist
from rdd.comparison.training import ComparisonWanModel
from rdd.finetune import RDDWanModel


class _NativeWan:

    def __init__(self: Any, *, native_contract_version: Any = 2, **kwargs: Any) -> None:
        if native_contract_version != 2:
            raise ValueError('Original FP32 VAE and FP32 normalization require native contract v2')
        self.native_contract_version = native_contract_version
        super().__init__(**kwargs)

    def init_preprocessors(self: Any, training_config: Any) -> Any:
        parent: Any = super()
        parent.init_preprocessors(training_config)
        self.vae = self.load_original_vae(str(training_config.model_path))
        # Iterator reconstruction must not consume the model's global CPU RNG.
        # StatefulDataLoader checkpoints this dedicated generator with its data
        # snapshot. Do this after ComparisonWan's RGB-cache loader replacement.
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataloader.generator = torch.Generator().manual_seed(training_config.data.seed + 10000 + rank)

    def _get_training_dtype(self: Any) -> Any:
        # Normalize original raw FP32 VAE latents before the DiT's BF16 AMP
        # boundary. Quantizing raw latents first changes the FM objective.
        return torch.float32

    def load_original_vae(self: Any, path: Any) -> Any:
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(path, subfolder='vae', torch_dtype=torch.float32,
                                               local_files_only=True).to(self.device).eval().requires_grad_(False)
        vae.latents_mean = vae.config.latents_mean
        vae.latents_std = vae.config.latents_std
        return vae

    @torch.no_grad()
    def decode_latents(self: Any, latents_b_t_c_h_w: Any) -> Any:
        latent = latents_b_t_c_h_w.permute(0, 2, 1, 3, 4).float()
        mean = torch.tensor(self.vae.config.latents_mean, device=latent.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latent.device).view(1, -1, 1, 1, 1)
        # Preserve the original native pipeline's floating-point operation
        # order, including division by the reciprocal standard deviation.
        pixels = self.vae.decode(latent / (1.0 / std) + mean, return_dict=False)[0]
        return (pixels / 2 + .5).clamp(0, 1)

    def _load_transformer(self: Any,
                          *,
                          init_from: Any,
                          trainable: Any,
                          disable_custom_init_weights: Any,
                          enable_gradient_checkpointing_type: Any,
                          training_config: Any,
                          transformer_override_safetensor: Any = None,
                          attention_backend: Any = None) -> Any:
        if transformer_override_safetensor:
            raise ValueError('Strict original loading requires the pinned complete HF snapshot')
        if training_config.distributed.sp_size != 1 or training_config.distributed.tp_size != 1:
            raise ValueError('Original Wan adapter supports replicated DP; SP=TP=1')
        from diffusers import WanTransformer3DModel
        import os
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
        model = WanTransformer3DModel.from_pretrained(init_from,
                                                      subfolder='transformer',
                                                      torch_dtype=torch.float32 if trainable else torch.bfloat16,
                                                      local_files_only=True).to(device)
        model.requires_grad_(trainable)
        if trainable and (enable_gradient_checkpointing_type
                          or training_config.model.enable_gradient_checkpointing_type):
            model.enable_gradient_checkpointing()
        if trainable and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
        return model

    def _build_attention_metadata(self: Any, batch: Any) -> Any:
        batch.attn_metadata = None
        batch.attn_metadata_vsa = None
        return batch

    def predict_noise(self: Any,
                      noisy_latents: Any,
                      timestep: Any,
                      batch: Any,
                      *,
                      conditional: Any,
                      cfg_uncond: Any = None,
                      attn_kind: Any = 'dense',
                      clean_x: Any = None,
                      aug_t: Any = None) -> Any:
        if attn_kind != 'dense' or clean_x is not None or aug_t is not None:
            raise ValueError('Original Wan profile requires unmodified dense T2V conditioning')
        text = batch.conditional_dict if conditional else self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)
        latent = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16)
        # Native BF16 inference preserves Diffusers' FP32 time projections.
        # AMP belongs to the FP32-master training profile; applying it to the
        # already BF16-loaded inference model changes those original operators.
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self._trainable):
            result = self.transformer(latent,
                                      timestep=timestep,
                                      encoder_hidden_states=text['encoder_hidden_states'].to(dtype=torch.bfloat16),
                                      return_dict=False)[0]
        return result.permute(0, 2, 1, 3, 4)

    def backward(self: Any, loss: Any, ctx: Any, *, grad_accum_rounds: Any = 1) -> Any:
        (loss / max(1, grad_accum_rounds)).backward()


class NativeWanComparisonModel(_NativeWan, ComparisonWanModel):
    pass


class NativeRDDWanModel(_NativeWan, RDDWanModel):
    pass
