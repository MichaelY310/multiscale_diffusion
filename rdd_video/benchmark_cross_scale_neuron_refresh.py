"""Matched resident 3-NFE RDD+VSA benchmark of cross-scale neuron refresh.

Approximate architecture pilot on FastWan, NOT proof of trained RDD quality.
Baseline preserves all VSA coarse branches and gates. Optional _sa14 arms
transport the complete previous projected self-attention (including its old
coarse branch/gate), not a current-step VSA evaluation. Cross-attention stays live.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sample_fastwan_native_vsa import add_fastvideo_paths, sample_three_call_dmd
from benchmark_csra_prefix_feasibility import aggregate, group_bootstrap_ci


CONFIGS = {
    "baseline": (1., ()),
    "fine_refresh50": (.5, (2,)),
    "fine_refresh25": (.25, (2,)),
    "cascade_refresh50": (.5, (1, 2)),
    "cascade_refresh25": (.25, (1, 2)),
    "cascade_refresh125": (.125, (1, 2)),
    "fine_refresh25_sa14": (.25, (2,)),
    "cascade_refresh25_sa14": (.25, (1, 2)),
    "cascade_refresh50_sa14": (.5, (1, 2)),
    "self_only_sa14": (1., (1, 2)),
}


def main(a):
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"metrics.json").exists():
        raise FileExistsError("Completed result exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise ValueError("Require exactly physical GPU 0 or 1")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0] < 35*2**30:
        raise RuntimeError("Need 35GiB free; never evict another process")
    torch.cuda.set_per_process_memory_fraction(.45, dev)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fv = root/"third_party/FastVideo-ref"
    add_fastvideo_paths(fv, fv/"fastvideo-kernel")
    from attention_analysis.cross_scale_neuron_refresh import CrossScaleNeuronRefresh
    from attention_analysis.fastwan_native_attention import install_projected_cache_class_patch, FastWanNativeAttentionController
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from sample_rdd_wan import save_video
    wan = root/"pretrained_models/Wan2.1-T2V-1.3B"
    text = T5EncoderModel(text_len=512, dtype=torch.float32, device=torch.device("cpu"),
        checkpoint_path=str(wan/"models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan/"google/umt5-xxl"))
    text.model.to(dev)
    with torch.inference_mode():
        tokens = text([a.prompt], dev)[0]
        context = torch.zeros((1, 512, tokens.shape[-1]), device=dev, dtype=torch.bfloat16)
        context[0, :len(tokens)] = tokens.to(torch.bfloat16)
    text.model.cpu()
    del text, tokens
    torch.cuda.empty_cache()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(a.port), RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    weights = root/"pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer"
    fa = FastVideoArgs(model_path=str(weights), num_gpus=1, dit_cpu_offload=False,
        dit_layerwise_offload=False, pipeline_config=FastWan2_1_T2V_480P_Config(
            dit_config=WanVideoConfig(), dit_precision="bf16"))
    fa.device = dev
    # Before construction: retain FastVideo's managed-forward hooks.
    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    model = TransformerLoader().load(str(weights), fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    vae = WanVAE(z_dim=16, vae_pth=str(wan/"Wan2.1_VAE.pth"), dtype=torch.bfloat16, device=dev)
    shape = (1, 16, (a.frames-1)//4+1, a.height//8, a.width//8)
    factors = [(4, 1), (2, 1), (1, 1)]
    if (a.frames-1)%4 or a.height%64 or a.width%64:
        raise ValueError("Nested spatial421 requires H/W divisible64, frames4n+1")
    controller = CrossScaleNeuronRefresh(model)
    self_controller = None
    class Combined:
        attention_mode = "vsa"
        def begin_model_forward(self, stage, shape):
            controller.begin_model_forward(stage, shape)
            if self_controller is not None:
                if stage == 0:
                    self_controller.reset()
                self_controller.begin_model_forward(stage, shape)
    combined = Combined()
    selected = list(range(1, len(model.blocks)-1))
    variants = ["baseline"]+a.variants.split(",")
    if len(set(variants)) != len(variants) or any(v not in CONFIGS for v in variants):
        raise ValueError("Invalid/duplicate variants")
    setups = {}
    def configure(v, backend="triton"):
        f, stages = CONFIGS[v]
        controller.configure(f, selected, stages, backend)
        if self_controller is not None:
            self_controller.configure("vsa", cross_stage_branch_reuse="self" if v.endswith("_sa14") else "none",
                cross_stage_branch_reuse_layers=list(range(1,len(model.blocks)-1,2)) if v.endswith("_sa14") else [],
                cross_stage_branch_reuse_stages=stages if v.endswith("_sa14") else [])
        setups[v] = controller.setup_seconds
    def sample(c="configured"):
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return sample_three_call_dmd(model, context, shape, factors, [1000, 757, 522],
                a.seed, .8, "noise_filling", dev, attention_controller=combined if c=="configured" else c,
                boundary_executor="native", return_breakdown=True)
    configure("baseline")
    native = sample(None)[0]
    if any(v.endswith("_sa14") for v in variants):
        self_controller = FastWanNativeAttentionController(model)
        self_controller.configure("vsa")
    no_op = sample()[0]
    torch.testing.assert_close(native, no_op, rtol=0, atol=0)
    del native, no_op
    # Same approximate model, two transports. This is kernel equivalence,
    # explicitly distinct from approximation error versus the dense FFN.
    first = variants[1]
    configure(first, "torch")
    eager = sample()[0]
    configure(first)
    fused = sample()[0]
    torch.testing.assert_close(eager, fused, rtol=0, atol=0)
    del eager, fused
    report = dict(complete=False, method="cross_scale_neuron_refresh", model=str(weights),
        model_is_rdd_trained=False, finetuned=False, physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=a.seed, prompt=a.prompt,
        shape=shape, factors=factors, NFE=3, timesteps=[1000,757,522], boundary="noise_filling",
        attention="native VSA keep20 including coarse output/gate in executed branches; _sa14 transports the entire old projected self output", offload=False, dtype="bf16",
        vae="identical resident BF16 WanVAE, decoded after each timed sample",
        selected_layers_zero_based=selected, first_last_layer_full=True,
        configs={v:dict(fresh_fraction=CONFIGS[v][0], stages=CONFIGS[v][1],
            self_reuse_layers_zero_based=list(range(1,len(model.blocks)-1,2)) if v.endswith("_sa14") else []) for v in variants},
        baseline_noop_max_abs=0., fused_vs_eager_max_abs=0., groups=a.groups,
        warning="Speed/approximation pilot on FastWan weights, not post-finetuning quality evidence",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), entries=[])
    latents = {}
    def run(v, export=False):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = sample()
        torch.cuda.synchronize()
        denoise = time.perf_counter()-start
        with torch.inference_mode():
            video = vae.decode([result[0][0].float()])[0]
        torch.cuda.synchronize()
        total = time.perf_counter()-start
        if not torch.isfinite(video).all() or not torch.isfinite(result[0]).all():
            raise AssertionError("Nonfinite output")
        expected = len(selected)*len(CONFIGS[v][1]) if CONFIGS[v][0] < 1 else 0
        if controller.audit.get("refreshes", 0) != expected or controller.cache:
            raise AssertionError(f"Cache lifecycle/hit mismatch: {controller.audit}")
        sa_hits = self_controller.stats["cross_stage_self_reuse_hits"] if self_controller is not None else 0
        expected_sa = len(range(1,len(model.blocks)-1,2))*len(CONFIGS[v][1]) if v.endswith("_sa14") else 0
        if sa_hits != expected_sa:
            raise AssertionError("Self-attention reuse hit mismatch")
        if export:
            save_video(video, str(out/f"{v}.mp4"), fps=16)
            from PIL import Image
            frame = video[:, video.shape[1]//2].float().cpu().clamp(-1,1)
            Image.fromarray(((frame.permute(1,2,0)+1)*127.5).round().byte().numpy()).save(out/f"{v}.png")
            latents[v] = result[0].float().cpu()
        return dict(variant=v, denoise_seconds=denoise, total_seconds=total,
            vae_seconds=total-denoise, stages=result[2], audit=dict(controller.audit, self_reuse_hits=sa_hits),
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
    for v in variants:
        configure(v)
        for j in range(a.warmups):
            row = run(v, export=j==a.warmups-1)
        print(json.dumps(dict(warmup=v, row=row)), flush=True)
    # Configure outside timer only packs static weights; every cache capture,
    # coarse subtraction and lift is counted inside all timed trajectories.
    for v in variants[1:]:
        groups, reductions = [], {"denoise":[], "total":[]}
        for g in range(a.groups):
            order = ["baseline", v, v, "baseline"] if g%2 == 0 else [v,"baseline","baseline",v]
            rows = []
            for name in order:
                configure(name)
                rows.append(run(name))
            groups.append(rows)
            for k in reductions:
                base = statistics.mean(r[k+"_seconds"] for r in rows if r["variant"]=="baseline")
                cand = statistics.mean(r[k+"_seconds"] for r in rows if r["variant"]==v)
                reductions[k].append(100*(1-cand/base))
            print(json.dumps(dict(variant=v, group=g, reductions={k:x[-1] for k,x in reductions.items()})), flush=True)
        br = [r for rows in groups for r in rows if r["variant"]=="baseline"]
        cr = [r for rows in groups for r in rows if r["variant"]==v]
        report["entries"].append(dict(variant=v, paired_groups=groups,
            baseline={k:aggregate([r[k+"_seconds"] for r in br]) for k in reductions},
            candidate={k:aggregate([r[k+"_seconds"] for r in cr]) for k in reductions},
            reductions={k:group_bootstrap_ci(x) for k,x in reductions.items()}))
        (out/"progress.json").write_text(json.dumps(report, indent=2))
    report["static_packing_seconds"] = setups
    report["latent_relative_l2"] = {v:float((z-latents["baseline"]).norm()/latents["baseline"].norm()) for v,z in latents.items()}
    report["complete"] = True
    (out/"metrics.json").write_text(json.dumps(report, indent=2))
    controller.restore()
    torch.distributed.destroy_process_group()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/scr/yuheng/DevOpsBench/multiscale_diffusion")
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--frames", type=int, default=29)
    p.add_argument("--groups", type=int, default=6)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--variants", default="fine_refresh50,fine_refresh25,cascade_refresh50,cascade_refresh25")
    p.add_argument("--port", type=int, default=29871)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--prompt", default="A red fox walking through a snowy forest, cinematic natural light.")
    args = p.parse_args()
    if args.groups < 4 or args.warmups < 2:
        p.error("At least four paired groups and two warmups")
    main(args)
