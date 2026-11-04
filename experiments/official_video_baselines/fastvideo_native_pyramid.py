"""Pinned native Pyramid Flow plugins for FastVideo's current train interfaces.

The official runner, DiT, optimizer factory, LR arrays and training inner loop
execute unchanged. This deliberately uses the managed-optimization interface:
the native high-loss skip policy and Accelerate accumulation cannot be replaced
by FastVideo's default backward/step loop. GPU/distributed parity is still a gate.
"""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from fastvideo.train.models.base import ModelBase
from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.callbacks.callback import Callback

from pyramid_smoke import PIN, check_source


class NativePyramidModel(ModelBase):
    def __init__(self, *, source, init_from, annotation, empty_prompt,
                 cache_fingerprint, recipe, training_config, trainable=True,
                 component_cpu=False):
        super().__init__(trainable=trainable)
        if not trainable:
            raise ValueError("The native training plugin requires all DiT parameters trainable")
        self.source = Path(source).resolve()
        self.source_hashes = check_source(self.source)
        sys.path.insert(0, str(self.source))
        from pyramid_dit import PyramidDiTForVideoGeneration
        from dataset import LengthGroupedVideoTextDataset
        from pyramid_data_state import build_stateful_native_loader
        if not Path(sys.modules[PyramidDiTForVideoGeneration.__module__].__file__).resolve().is_relative_to(self.source):
            raise ValueError("A different Pyramid Flow module was already imported")
        self.recipe = dict(recipe)
        if self.recipe.get("use_flash_attn", False):
            raise ValueError("Native AR parity requires masked SDPA; official Flash omits the causal mask")
        if self.recipe.get("load_vae", False) or self.recipe.get("load_text_encoder", False):
            raise ValueError("This entrypoint requires audited native cached latents/text")
        self.recipe.update(load_vae=False, load_text_encoder=False, use_flash_attn=False,
                           use_mixed_training=True)  # Native main always keeps FP32 master weights.
        self.component_cpu = bool(component_cpu)
        self.accelerator = Accelerator(
            cpu=self.component_cpu,
            mixed_precision="no" if self.component_cpu or training_config.dit_precision == "fp32" else training_config.dit_precision,
            gradient_accumulation_steps=training_config.loop.gradient_accumulation_steps,
        )
        # Native Accelerate owns replicated DDP. FastVideo only supplies the
        # outer method/callback runtime; never double-wrap this model in FSDP.
        if training_config.distributed.sp_size != 1 or training_config.distributed.tp_size != 1:
            raise ValueError("Native FastVideo bridge currently requires SP=TP=1")
        from accelerate.utils import set_seed
        from trainer_misc import setup_for_distributed
        setup_for_distributed(self.accelerator.is_main_process)
        set_seed(training_config.data.seed, device_specific=True)
        self.runner = PyramidDiTForVideoGeneration(str(init_from), **self.recipe)
        null = torch.load(empty_prompt, map_location="cpu", weights_only=True)
        if set(null) != {"prompt_embed", "prompt_attention_mask", "pooled_prompt_embed"}:
            raise ValueError("Invalid native empty-prompt cache keys")
        self.runner.null_text_embeds = null
        self.transformer = self.runner.dit
        self.noise_scheduler = self.runner.scheduler
        self.start_step = 0
        frames = self.runner.max_temporal_length
        dataset = LengthGroupedVideoTextDataset(str(annotation), max_frames=frames,
                    resolution="384p", load_vae_latent=True, load_text_fea=True)
        self.dataloader = build_stateful_native_loader(dataset,
            batch_size=training_config.data.train_batch_size, max_frames=frames,
            num_workers=training_config.data.dataloader_num_workers, rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes, epoch=training_config.data.seed,
            dataset_fingerprint=cache_fingerprint)
        from pyramid_dcp_state import NativeDCPDataLoader
        self.dataloader = NativeDCPDataLoader(self.dataloader)

    @property
    def device(self):
        return self.accelerator.device

    def prepare_batch(self, raw_batch, *, generator, latents_source="data"):
        if latents_source != "data":
            raise ValueError("Native training uses the official cached dataset")
        # Official code mutates cached text under CFG dropout and video during
        # native normalization. Keep values identical, with device placement only.
        return {"video": raw_batch["video"].to(self.device),
                "text": {k: v.to(self.device) for k, v in raw_batch["text"].items()},
                "identifier": raw_batch["identifier"]}

    def native_loss(self, raw_batch, *, temporal=True):
        batch = self.prepare_batch(raw_batch, generator=None)
        return self.runner(batch["video"], batch["text"], batch["identifier"],
                           use_temporal_pyramid=temporal, accelerator=self.accelerator)

    def add_noise(self, clean_latents, noise, timestep):
        raise NotImplementedError("Use the native runner's stage/history preparation")

    def predict_noise(self, noisy_latents, timestep, batch, **kwargs):
        raise NotImplementedError("Native PF requires nested stage/history inputs, not a single Wan-style tensor")

    def backward(self, loss, ctx, grad_accum_rounds=1):
        self.accelerator.backward(loss)


class _NativeControlState:
    def __init__(self, method):
        self.method = method

    def state_dict(self):
        m = self.method
        if not m.student.accelerator.sync_gradients:
            raise ValueError("Native checkpoints are supported only at accumulation boundaries")
        return {"schema": 1, "source_commit": PIN, "recipe": m.student.recipe,
                "lr_sha256": hashlib.sha256(m.lr_values.tobytes()).hexdigest(),
                "global_step": m.global_step,
                "accumulate_steps": m.student.runner.accumulate_steps,
                "gradient_accumulation_steps": m.native_args.gradient_accumulation_steps,
                "clip_grad": m.clip_grad, "temporal": m.temporal}

    def load_state_dict(self, state):
        reference = self.state_dict()
        for key in reference.keys() - {"global_step", "accumulate_steps"}:
            if state.get(key) != reference[key]:
                raise ValueError(f"Native checkpoint recipe mismatch: {key}")
        for key in ("global_step", "accumulate_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Invalid native checkpoint counter: {key}")
        if state["global_step"] > len(self.method.lr_values):
            raise ValueError("Native checkpoint exceeds LR schedule")
        self.method.global_step = state["global_step"]
        self.method.student.runner.accumulate_steps = state["accumulate_steps"]
        # Save/load is boundary-only; no unsaved gradients or partial microsteps.
        self.method.student.accelerator.step = 0


class NativePyramidMethod(TrainingMethod):
    def __init__(self, *, cfg, role_models):
        super().__init__(cfg=cfg, role_models=role_models)
        if not isinstance(self.student, NativePyramidModel):
            raise TypeError("NativePyramidMethod requires NativePyramidModel")
        from trainer_misc import create_optimizer, constant_scheduler, cosine_scheduler
        tc = cfg.training
        oc = tc.optimizer
        self.temporal = bool(cfg.method.get("use_temporal_pyramid", True))
        self.clip_grad = float(cfg.method.get("clip_grad", 1.0))
        self.native_args = SimpleNamespace(
            opt="adamw", lr=oc.learning_rate, weight_decay=oc.weight_decay,
            opt_eps=float(cfg.method.get("opt_eps", 1e-8)), opt_beta1=oc.betas[0],
            opt_beta2=oc.betas[1], gradient_accumulation_steps=tc.loop.gradient_accumulation_steps,
            batch_size=tc.data.train_batch_size)
        self.optimizer = create_optimizer(self.native_args, self.student.transformer)
        self.student.runner.dit, self.optimizer = self.student.accelerator.prepare(
            self.student.transformer, self.optimizer)
        self.student.transformer = self.student.runner.dit
        self.role_modules["student"]["transformer"] = self.student.transformer
        schedule = cfg.method.get("native_lr_schedule", "constant_with_warmup")
        schedule_args = dict(base_value=oc.learning_rate, epochs=1,
            niter_per_ep=tc.loop.max_train_steps, warmup_epochs=0, warmup_steps=oc.lr_warmup_steps)
        if schedule == "constant_with_warmup":
            self.lr_values = constant_scheduler(**schedule_args)
        elif schedule == "cosine":
            self.lr_values = cosine_scheduler(**schedule_args,
                final_value=float(cfg.method.get("min_lr", 0.0)))
        else:
            raise ValueError("Unknown native LR schedule")
        self.global_step = 0
        self.control_state = _NativeControlState(self)

    def on_train_start(self):
        # The model constructor seeds before model/data construction, as native main does.
        # Preserve native GLOBAL RNG; the shared FastVideo generator would change
        # dropout/noise/time draws. CheckpointManager snapshots both global streams.
        self.student.transformer.train()
        # FastVideo's generic checkpoint API expects a shared generator. Native
        # PF does not draw from it; construction does not advance global RNG.
        self.cuda_generator = torch.Generator(device=self.student.device).manual_seed(
            self.training_config.data.seed)

    def single_train_step(self, batch, iteration):
        loss, logs = self.student.native_loss(batch, temporal=self.temporal)
        return {"total_loss": loss}, {}, {}

    def manages_optimization(self):
        return True

    def managed_train_step(self, data_stream, iteration):
        if iteration != self.global_step + 1:
            raise ValueError("Native global step and FastVideo iteration differ")
        from trainer_misc import train_one_epoch_with_fsdp
        student = self.student
        def native_batches():
            while True:
                yield student.prepare_batch(next(data_stream), generator=None)
        # Exactly one native outer step; all native microsteps, backward, clipping,
        # high-loss skips, AdamW and global-step LR selection remain official code.
        stats = train_one_epoch_with_fsdp(student.runner, None, student.accelerator,
            "float32" if student.component_cpu else self.training_config.dit_precision,
            native_batches(), self.optimizer, self.lr_values, student.device,
            epoch=0, clip_grad=self.clip_grad, start_steps=self.global_step,
            args=self.native_args, print_freq=100000, iters_per_epoch=1,
            use_temporal_pyramid=self.temporal)
        self.global_step = iteration
        return {"total_loss": torch.tensor(stats["loss"], device=student.device)}, {}, {
            "native/lr": stats["lr"], "native/ar_counter": student.runner.accumulate_steps}

    def get_optimizers(self, iteration):
        return [self.optimizer]

    def get_lr_schedulers(self, iteration):
        return []  # Native LR is a global-step-indexed array, not scheduler.step().

    @property
    def _optimizer_dict(self):
        return {"student": self.optimizer}

    @property
    def _lr_scheduler_dict(self):
        return {}

    def checkpoint_state(self):
        result = super().checkpoint_state()
        result["native_pyramid_control"] = self.control_state
        return result


class NativePyramidEMACallback(Callback):
    """Official EMA initialization, fixed decay and preincrement 0/100 cadence."""
    def __init__(self, *, decay=0.9999):
        self.decay = float(decay)
        self.model_ema = None
        self.update_steps = []

    def on_train_start(self, method, iteration=0):
        self.model_ema = copy.deepcopy(method.student.transformer).eval()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            method.student.accelerator.mixed_precision, torch.float32)
        self.model_ema.to(dtype=dtype)  # Same non-trainable EMA dtype as native main.
        self.model_ema.requires_grad_(False)

    def on_training_step_end(self, method, loss_dict, iteration=0):
        if (iteration - 1) % 100 == 0:
            from trainer_misc.fsdp_trainer import update_ema_for_dit
            update_ema_for_dit(method.student.transformer, self.model_ema,
                             method.student.accelerator, self.decay)
            self.update_steps.append(iteration - 1)

    def state_dict(self):
        return {"decay": self.decay, "weights": self.model_ema.state_dict(),
                "update_steps": self.update_steps}

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("Native EMA decay changed")
        self.model_ema.load_state_dict(state["weights"], strict=True)
        self.update_steps = list(state["update_steps"])
