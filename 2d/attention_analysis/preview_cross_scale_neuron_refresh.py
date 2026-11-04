"""Eight paired true RDD-DMD images; no training and no video-quality claim."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"DiT"))
sys.path.insert(0, str(ROOT))
from models_multires import DiTMultiRes_models
from dmd_cascade_reuse import CascadeDMD, coupled_noises
from rdd_video.attention_analysis.cross_scale_neuron_refresh import CrossScaleNeuronRefresh


CONFIGS = {"baseline":(1.,()), "fine50":(.5,(2,)), "fine25":(.25,(2,)),
           "cascade50":(.5,(1,2)), "cascade25":(.25,(1,2)), "cascade125":(.125,(1,2))}


@torch.inference_mode()
def main(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 16*2**30:
        raise RuntimeError("Need 16GiB headroom")
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(a.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = DiTMultiRes_models["DiT-L/2"](num_classes=1000, base_size=16)
    model.load_state_dict(ck["model"], strict=True)
    step = int(ck["step"])
    del ck
    model = model.cuda().eval().requires_grad_(False)
    control = CrossScaleNeuronRefresh(model)
    class Rollout(CascadeDMD):
        def stage(self, x, t, y, stage, cache=None, **kwargs):
            control.begin_model_forward(stage, (1, x.shape[-2], x.shape[-1]))
            return super().stage(x,t,y,stage,cache,**kwargs)
    net = Rollout(model, "dense", False)
    plain = CascadeDMD(model, "dense", False)
    layers = list(range(1,len(model.blocks)-1))
    rng = torch.Generator(device="cuda").manual_seed(a.seed)
    noises, labels = coupled_noises(8, torch.device("cuda"), rng)
    signatures = hashlib.sha256(b"".join(t.cpu().numpy().tobytes() for t in [labels,*noises])).hexdigest()
    control.configure(1., stages=())
    with torch.autocast("cuda",dtype=torch.bfloat16):
        native = plain(noises, labels)["cleans"][-1]
        no_op = net(noises,labels)["cleans"][-1]
    torch.testing.assert_close(native,no_op,rtol=0,atol=0)
    zs, audit = {}, {}
    for name, (fraction, stages) in CONFIGS.items():
        control.configure(fraction, layers, stages)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            result = net(noises,labels)
        if not all(z.isfinite().all() for z in result["cleans"]):
            raise AssertionError("Nonfinite preview")
        zs[name] = result["cleans"][-1].float().cpu()
        audit[name] = dict(control.audit)
    control.restore()
    del model, net, plain, native, no_op, result, noises
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(a.vae,local_files_only=True).cuda().eval().requires_grad_(False)
    out.mkdir(parents=True)
    images = {}
    for name,z in zs.items():
        pictures = []
        for start in range(0,8,2):
            decoded = vae.decode(z[start:start+2].cuda()/0.18215).sample.float().clamp(-1,1)
            for array in ((decoded+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy():
                image = Image.fromarray(array)
                image.save(out/f"{name}_{len(pictures):02d}.png")
                pictures.append(image)
        images[name] = pictures
    sheets = []
    for pair in (("fine50","fine25"), ("cascade50","cascade25"), ("cascade25","cascade125")):
        names = ("baseline", *pair)
        for start in (0,4):
            canvas = Image.new("RGB", (3*256, 4*286), "white")
            draw = ImageDraw.Draw(canvas)
            for row,i in enumerate(range(start,start+4)):
                for col,name in enumerate(names):
                    draw.text((col*256+4,row*286+3), f"{name} / class {int(labels[i])}", fill="black")
                    canvas.paste(images[name][i],(col*256,row*286+30))
            file = f"comparison_{pair[0]}_{pair[1]}_{start}.png"
            canvas.save(out/file)
            sheets.append(file)
    manifest = dict(complete=True, model="true RDD Unified DiT-L/2", checkpoint=a.checkpoint,
        checkpoint_step=step, new_training=False, no_op_max_abs=0., selected_layers_zero_based=layers,
        configs=CONFIGS, seeds=[a.seed], labels=labels.tolist(), noise_sha256=signatures,
        NFE=3, timesteps=[999,599,299], stages=[8,16,32], boundary="noise_filling",
        sampler="existing RDD phased DMD3", attention="unchanged original DiT dense attention",
        audits=audit, sample_count=8, warning="Small approximation sanity only; not FID or trained student quality",
        latent_relative_l2={name:float((z-zs["baseline"]).norm()/zs["baseline"].norm()) for name,z in zs.items()})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    gallery = "".join(f'<h2>{html.escape(s)}</h2><img width="768" src="{s}">' for s in sheets)
    (out/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>CNR: true RDD-DMD image pilot</title><style>body{{max-width:1000px;margin:auto;font-family:system-ui}}img{{max-width:100%}}</style><h1>Cross-scale neuron refresh: true RDD-DMD</h1><p>Untrained approximation. Eight fixed paired samples, no cherry-picking. First/last block full; attention unchanged. Current MLP channels: 50%, 25%, 12.5%; other channels transported from the preceding coarse computation.</p><p><a href="manifest.json">Complete configuration</a></p>{gallery}',encoding="utf-8")
    print(json.dumps(manifest),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--vae",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=20261002)
    main(p.parse_args())
