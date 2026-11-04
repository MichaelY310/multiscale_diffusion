"""Render measured CNR pilots and all paired preview media, without inference."""
import argparse
import html
import json
import os
from pathlib import Path
from html.parser import HTMLParser


def main(a):
    root = Path(a.root).resolve()
    out = root/"eval_results/cnr_20261002_summary"
    out.mkdir(parents=True,exist_ok=True)
    folders = sorted((root/"eval_results").glob("cnr_20261002_*"))
    rows, galleries, summary = [], [], []
    def link(p):
        return html.escape(os.path.relpath(p,out).replace("\\","/"),quote=True)
    for folder in folders:
        if not (folder/"metrics.json").exists():
            continue
        data = json.loads((folder/"metrics.json").read_text())
        if not data.get("complete"):
            continue
        for entry in data["entries"]:
            name = entry["variant"]
            d, t = entry["reductions"]["denoise"], entry["reductions"]["total"]
            cfg = data["configs"][name]
            row = dict(run=folder.name,variant=name,configs=cfg,
                denoise_reduction=d,total_reduction=t,latent_relative_l2=data["latent_relative_l2"][name])
            summary.append(row)
            res = f'{data["shape"][-1]*8}×{data["shape"][-2]*8} / {(data["shape"][2]-1)*4+1}f'
            sa = len(cfg.get("self_reuse_layers_zero_based",[]))
            rows.append(f'<tr><td>{res}</td><td>{html.escape(name)}</td><td>{cfg["fresh_fraction"]*100:g}%</td><td>{cfg["stages"]}</td><td>{sa}</td><td>{d["mean"]:.2f}% [{d["lower"]:.2f},{d["upper"]:.2f}]</td><td>{t["mean"]:.2f}% [{t["lower"]:.2f},{t["upper"]:.2f}]</td><td>{entry["baseline"]["denoise"]["median"]:.3f} → {entry["candidate"]["denoise"]["median"]:.3f}s</td><td>{entry["baseline"]["total"]["median"]:.3f} → {entry["candidate"]["total"]["median"]:.3f}s</td><td>{row["latent_relative_l2"]:.3f}</td></tr>')
        cards = []
        for name in ("baseline",*[e["variant"] for e in data["entries"]]):
            cfg = data["configs"][name]
            cards.append(f'<article><h3>{html.escape(name)}</h3><video controls preload="none" poster="{link(folder/(name+".png"))}" src="{link(folder/(name+".mp4"))}"></video><table><tr><th>Fresh FFN neurons</th><td>{cfg["fresh_fraction"]*100:g}%</td></tr><tr><th>Partial stages (0=coarse)</th><td>{cfg["stages"]}</td></tr><tr><th>Transported self-attention layers</th><td>{cfg.get("self_reuse_layers_zero_based",[])}</td></tr><tr><th>Seed / prompt</th><td>{data["seed"]} / {html.escape(data["prompt"])}</td></tr></table></article>')
        galleries.append(f'<h2>{html.escape(folder.name)}</h2><p><a href="{link(folder/"metrics.json")}">Raw configuration and ABBA samples</a></p><div class="cards">{"".join(cards)}</div>')
    image = root/"eval_results/cnr_20261002_images/index.html"
    image_link = f'<p><a href="{link(image)}">True RDD-DMD: eight paired ImageNet samples</a></p>' if image.exists() else '<p>True RDD-DMD image pilot not yet downloaded.</p>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cross-scale neuron refresh — measured pilots</title><style>body{font:16px system-ui;line-height:1.6;max-width:1400px;margin:24px auto;padding:0 16px;color:#202735}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #ddd;text-align:left}section{overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}video{width:100%}article{border:1px solid #ddd;padding:12px}.warning{background:#fff1cd;padding:16px}pre{overflow:auto;background:#f5f5f5;padding:12px}</style>
<h1>Cross-scale neuron refresh (CNR)</h1><p class="warning">These are inference-only architecture pilots. No CNR finetuning has been performed. Video speed uses FastWan weights not trained for RDD; video quality is not evidence of a trained RDD+CNR model. Both teacher and candidate may be visibly broken. A speed threshold is not a quality threshold.</p>
<h2>Method</h2><p>Partition the FFN hidden neurons into fresh S and transported I. Native attention and timestep gates remain unchanged. Retain the first and last full block. At fine stages, recompute S from the current fine input, and lift only the inactive contribution from a smaller stage. Unlike whole-block reuse, fresh neurons can create within-cell differences in all output channels.</p><pre>coarse FFN: Yc = GELU(Xc Wi + bi) Wo + bo
cached inactive: CI = Yc - GELU(Xc Wi[S] + bi[S]) Wo[:,S]
fine FFN: Yf ≈ nearest(CI) + GELU(Xf Wi[S] + bi[S]) Wo[:,S]
combined (_sa14): additionally lift projected self-attention in 14/30 blocks
                 current cross-attention, gates and fresh FFN neurons remain active</pre>
<p>Static selection: largest ||Wi[j,:]||₂ ||Wo[:,j]||₂; round the fresh count to 64 neurons. This is a heuristic, not optimal error-based routing. Packing is one-time setup and is separately reported; every coarse capture, additional coarse projection, subtraction and transport is timed. The fused Triton kernel combines coarse gather and fresh addition; it does not make the approximation lossless.</p>
<h2>Protocol and measured results</h2><p>Same resident BF16 backbone and BF16 VAE, native VSA keep20 including its coarse output/gate, spatial factors (4,1)→(2,1)→(1,1), noise_filling, t=1000/757/522, exactly 3 NFE. Six alternating ABBA/BAAB groups; two warmups per arm; real VAE decode after every timed trajectory. Percentages are paired latency reduction. Intervals are group bootstrap 95% intervals from one prompt/seed, not population guarantees. Loading, text encoding, compilation and video encoding are excluded. No other GPU job was intentionally displaced.</p><section><table><tr><th>Output</th><th>Variant</th><th>Fresh FFN</th><th>Partial stages</th><th>Self reuse blocks</th><th>Denoise reduction [CI]</th><th>Sample+VAE reduction [CI]</th><th>Denoise median</th><th>Sample+VAE median</th><th>Latent relative L2</th></tr>ROWS</table></section>
<p>Latent L2 measures change from the same-seed baseline, not FID or perceptual quality. A full-resolution previous-step cache can imitate parts of this mechanism; the RDD-specific opportunity is that the original coarse computation is already much smaller and available without adding another network evaluation. Absolute exclusivity or novelty has not been established.</p>
<h2>Quality and next validation</h2><p><strong>The true RDD-DMD image pilot visibly fails without training, including the 50% final-stage refresh arm. This is not a usable acceleration method.</strong> The 1280×768 baseline middle frame shows a recognizable fox, whereas the 25% refresh + self-reuse middle frame loses its structure. The smaller 832×448 baseline already has significant distortion. Baseline limitations do not excuse the additional corruption. Inspect all fixed previews below. Do not select the fastest broken clip as a success. In the _sa14 arms, the entire old projected self-attention is transported, including its old VSA coarse branch/gate; only the outer current timestep gate, cross-attention and fresh FFN neurons are recomputed. This is an approximation, not deletion of the VSA coarse output to claim an equivalent speedup.</p><p>Possible training must use the actual three-call student rollout and differentiable coarse cache, current/final high-frequency supervision and a distribution/perceptual objective. This inference pilot uses packed frozen weights, and deliberately raises on training use. Recovery of baseline quality remains unproven. Given the severe distortion, a long training run is not justified by these speed measurements alone.</p>IMAGE
<p>Related prior art: <a href="https://arxiv.org/abs/2505.05829">Increment-calibrated caching (CVPR 2025)</a> also corrects cached features, using low-rank approximations across timesteps. CNR instead refreshes a subset of actual FFN neurons across resolution stages. <a href="https://arxiv.org/abs/2609.32197">SparSP</a> motivates sparse KV communication, a separate direction not benchmarked here.</p>
<h2>All paired video previews</h2>GALLERIES</html>'''
    page = page.replace("ROWS","".join(rows)).replace("IMAGE",image_link).replace("GALLERIES","".join(galleries))
    class CheckAssets(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src","poster","href"} and value and not value.startswith(("http://","https://","#")):
                    if not (out/value).resolve().is_file():
                        raise FileNotFoundError(f"Missing report asset: {value}")
    CheckAssets().feed(page)
    (out/"index.html").write_text(page,encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(out/"index.html")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=str(Path(__file__).resolve().parents[1]))
    main(p.parse_args())
