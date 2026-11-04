"""Original RDD modular method on CPU/Gloo with an explicitly tiny surrogate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fastvideo", type=Path, required=True)
    p.add_argument("--rdd", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from fastvideo_cpu_import import bootstrap
    compatibility = bootstrap(args.fastvideo)
    sys.path.append(str(args.rdd))
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import yaml
    from fastvideo.train.utils.config import load_run_config
    from rdd.finetune import RDDWanModel, RDDFineTuneMethod
    from rdd.flow import RDDPath
    torch.set_num_threads(2)
    torch.manual_seed(73)
    scratch = Path(tempfile.mkdtemp(prefix="rdd-original-cpu-"))
    cfg_dict = yaml.safe_load((args.rdd / "rdd/comparison/configs/rdd.yaml").read_text())
    cfg_dict.pop("pipeline", None)
    cfg_dict["method"].update(smoke_all_stages=True, deadline_grace_seconds=5.0, preview_every=0)
    cfg_dict["training"]["data"].update(num_latent_t=3, num_height=128, num_width=256)
    cfg_dict["training"]["distributed"].update(num_gpus=1, sp_size=1, tp_size=1)
    cfg_dict["training"]["loop"].update(max_train_steps=3, gradient_accumulation_steps=1)
    cfg_dict["training"]["optimizer"]["lr_warmup_steps"] = 0  # Same <=3-update smoke policy as the shared suite.
    config = scratch / "component.yaml"
    config.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_run_config(str(config))

    class Student(RDDWanModel):
        def __init__(self):
            self.training_config = cfg.training
            self._trainable = True
            self.attention_backend = None
            self.transformer = torch.nn.Conv3d(16, 16, 1)
            self.rdd_path = RDDPath()
            self.rdd_stage = 0
            self.rdd_fixed_t = None

        @property
        def device(self):
            return torch.device("cpu")

        def init_preprocessors(self, training_config):
            self.dataloader = None

        def prepare_batch(self, raw_batch, *, generator, latents_source="data"):
            self.negative_prompt_embeds = torch.zeros(1, 512, 8)
            self.negative_prompt_attention_mask = torch.ones(1, 512)
            batch = SimpleNamespace(latents=raw_batch["vae_latent"],
                encoder_hidden_states=torch.zeros(1, 512, 8), encoder_attention_mask=torch.ones(1, 512),
                attn_metadata=None, attn_metadata_vsa=None)
            return self._prepare_dit_inputs(batch, generator)

        def predict_noise(self, x, timestep, batch, **kwargs):
            return self.transformer(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

    dist.init_process_group("gloo", init_method=f"file://{scratch / 'gloo'}", rank=0, world_size=1)
    try:
        student = Student()
        method = RDDFineTuneMethod(cfg=cfg, role_models={"student": student})
        method.cuda_generator = torch.Generator(device="cpu").manual_seed(17)
        os.environ["RDD_DEADLINE_UNIX"] = str(time.time() + 120)
        raw = {"vae_latent": torch.randn(1, 16, 3, 16, 32)}
        results = []
        for step in range(1, 4):
            method.optimizers_zero_grad(step)
            before = {key: value.clone() for key, value in student.transformer.state_dict().items()}
            losses, outputs, metrics = method.single_train_step(raw, step)
            method.backward(losses, outputs, grad_accum_rounds=1)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.transformer.parameters())
            method.optimizers_schedulers_step(step)
            assert any(not torch.equal(value, before[key]) for key, value in student.transformer.state_dict().items())
            results.append({"step": step, "stage": metrics["rdd_stage"], "loss": float(losses["total_loss"].detach())})
        saved = {key: value.clone() for key, value in student.transformer.state_dict().items()}
        dcp.save(method.checkpoint_state(), checkpoint_id=str(scratch / "dcp"))
        with torch.no_grad():
            for value in student.transformer.parameters():
                value.zero_()
        dcp.load(method.checkpoint_state(), checkpoint_id=str(scratch / "dcp"))
        assert all(torch.equal(value, saved[key]) for key, value in student.transformer.state_dict().items())
        assert not torch.cuda.is_initialized()
        args.output.write_text(json.dumps({"status": "passed", "original_rdd_method": True,
            "surrogate_not_wan": True, "gpu_used": False, "cuda_initialized": False,
            "single_rank_cpu_gloo": True, "results": results, "dcp_weights": True,
            "120s_deadline_compatible": True, "compatibility": compatibility, "scratch": str(scratch),
            "source_sha256": {name: hashlib.sha256((args.rdd / name).read_bytes()).hexdigest()
                              for name in ("rdd/flow.py", "rdd/finetune.py")}}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
