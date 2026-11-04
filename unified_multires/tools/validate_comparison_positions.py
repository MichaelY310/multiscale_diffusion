"""CPU differential checks against fixed author PE functions, no assets.

Author functions are extracted by AST to avoid importing CUDA training stacks.
Only Wan's CUDA autocast decorators are omitted for this CPU check. Backbone
parity below means common DiT + author PE vs common DiT + independent PE port,
not parity to the author's different full network.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from einops import rearrange

from umdiff import build
from umdiff.comparison_position import ComparisonPosition
from umdiff.position import rotary_2d

ROOT = Path(__file__).resolve().parents[1]
PF = ROOT/'sources/pyramid_flow/pyramid_dit/flux_modules'
COUNTS = {}
SOURCES = {}


def extract(path, names, cls=None):
    text = path.read_text()
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(text)
    nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    selected = [copy.deepcopy(n) for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    env = dict(torch=torch, np=np, nn=nn, List=List, Tuple=Tuple, Union=Union, rearrange=rearrange,
               F=torch.nn.functional)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def exact(a, b, kind):
    if a is None or b is None:
        assert a is b, kind
        return
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b), kind
        for x, y in zip(a, b):
            exact(x, y, kind)
        return
    assert a.shape == b.shape and a.dtype == b.dtype, kind
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), kind
    aa, bb = a.detach().cpu().contiguous().reshape(-1).clone(), b.detach().cpu().contiguous().reshape(-1).clone()
    assert torch.equal(aa.view(torch.uint8), bb.view(torch.uint8)), (kind, (a-b).abs().max().item())
    COUNTS[kind] = COUNTS.get(kind, 0)+1


class AuthorPosition(ComparisonPosition):
    def __init__(self, name):
        super().__init__(name)
        if name == 'pixelflow_rope':
            self.fn = extract(ROOT/'sources/pixelflow/pixelflow/model.py', {'apply_rotary_emb'})
            # Native snapshot pins the old NumPy-grid Diffusers API; avoid
            # silently exercising the newer meshgrid/coordinate convention.
            from diffusers.models.embeddings import get_2d_rotary_pos_embed_from_grid
            self.grid_fn = get_2d_rotary_pos_embed_from_grid
        elif name in ('dfm_sincos', 'lapflow_sincos'):
            path = ROOT/('sources/dfm/models/sit.py' if name == 'dfm_sincos' else 'sources/lapflow/models/DiT.py')
            self.fn = extract(path, {'get_2d_sincos_pos_embed', 'get_2d_sincos_pos_embed_from_grid', 'get_1d_sincos_pos_embed_from_grid'})
        elif name == 'pyramid_flow_rope':
            self.fn = extract(PF/'modeling_pyramid_flux.py', {'rope', 'EmbedND'})
            self.ids = extract(PF/'modeling_pyramid_flux.py', {'_prepare_image_ids'}, 'PyramidFluxTransformer')['_prepare_image_ids']
            self.apply = extract(PF/'modeling_flux_block.py', {'apply_rope'})['apply_rope']
        elif name == 'wan_rope':
            self.fn = extract(ROOT/'sources/wan21/wan/modules/model.py', {'rope_params', 'rope_apply'})
        self.grids = None

    def additive(self, dim, h, w, device):
        if self.name in ('dfm_sincos', 'lapflow_sincos'):
            assert h == w
            return torch.from_numpy(self.fn['get_2d_sincos_pos_embed'](dim, h)).float().to(device)
        return None

    def rotary(self, core, h, w, device):
        d = core.head_dim
        if self.name == 'legacy_rdd':
            return core.get_rope(h, w, device)
        if self.name in ('dfm_sincos', 'lapflow_sincos'):
            return None
        if self.name == 'pixelflow_rope':
            gh = np.linspace(0, h, h, endpoint=False, dtype=np.float32)
            gw = np.linspace(0, w, w, endpoint=False, dtype=np.float32)
            grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, h, w)
            return tuple(x.to(device) for x in self.grid_fn(d, grid, use_real=True))
        if self.name == 'pyramid_flow_rope':
            # Explicit dimension adaptation belongs to version3, not source.
            dt = max(4, 4*(d//16))
            ids = self.ids(None, 1, 1, h, w, h, w, device)
            return self.fn['EmbedND'](d, 10000, [dt, (d-dt)//2, (d-dt)//2])(ids)[0, :, 0]
        dims = [d-4*(d//6), 2*(d//6), 2*(d//6)]
        freqs = torch.cat([self.fn['rope_params'](max(h, w), size) for size in dims], 1).to(device)
        # Keep source grid/frequency representation; source applies the gather.
        self.grids = (h, w)
        return freqs

    def rotate(self, x, pos):
        if self.name == 'legacy_rdd':
            return super().rotate(x, pos)
        if pos is None:
            return x
        if self.name == 'pixelflow_rope':
            return self.fn['apply_rotary_emb'](x, torch.stack(pos, -1))
        if self.name == 'pyramid_flow_rope':
            return self.apply(x, x, pos[None, None])[0]
        h, w = self.grids
        out = self.fn['rope_apply'](x.transpose(1, 2), torch.tensor([[1, h, w]]*len(x)), pos)
        return out.transpose(1, 2)


def capture(net):
    outputs, grads, handles = {}, {}, []
    for name, module in net.named_modules():
        if not name:
            continue
        def hook(mod, args, output, name=name):
            if torch.is_tensor(output):
                outputs[name] = output.detach().clone()
                if output.requires_grad:
                    output.register_hook(lambda g, name=name: grads.__setitem__(name, g.detach().clone()))
        handles.append(module.register_forward_hook(hook))
    return outputs, grads, handles


def run():
    torch.set_num_threads(2)
    torch.manual_seed(867)
    policies = ['legacy_rdd', 'pixelflow_rope', 'dfm_sincos', 'lapflow_sincos', 'pyramid_flow_rope', 'wan_rope']
    # PE tables and Q/K rotations at actual head72 as well as small test heads.
    for name in policies[1:]:
        native, port = AuthorPosition(name), ComparisonPosition(name)
        for h, w in [(2, 2), (4, 4), (8, 8), (16, 16)]:
            for d in (8, 72):
                core = SimpleNamespace(head_dim=d)
                aa, ab = native.additive(d*4, h, w, 'cpu'), port.additive(d*4, h, w, 'cpu')
                exact(aa, ab, name+'/additive')
                pa, pb = native.rotary(core, h, w, 'cpu'), port.rotary(core, h, w, 'cpu')
                if name != 'wan_rope':
                    exact(pa, pb, name+'/rotary_table')
                for dtype in (torch.float32, torch.bfloat16):
                    x = torch.randn(2, 3, h*w, d).to(dtype).requires_grad_()
                    z = x.detach().clone().requires_grad_()
                    a, b = native.rotate(x, pa), port.rotate(z, pb)
                    exact(a, b, name+'/qk_forward_'+str(dtype))
                    probe = torch.randn_like(a)
                    (a*probe).sum().backward(); (b*probe).sum().backward()
                    exact(x.grad, z.grad, name+'/qk_backward_'+str(dtype))
    # Nonzero common DiT + independent source PE, source core.forward remains.
    for name in policies:
        arch = dict(patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=4,
                    class_dropout_prob=.1, num_classes=1000, learn_sigma=True, base_size=16)
        torch.manual_seed(77)
        net = build('model', {'name':'comparison_dit', 'args':{'architecture':arch, 'position':name}}).eval()
        for p in net.parameters():
            with torch.no_grad(): p.normal_(0, .03)
        ref = build('model', {'name':'native_rdd_unified', 'args':arch}).eval()
        ref.load_state_dict(net.core.state_dict())
        native = AuthorPosition(name)
        if name != 'legacy_rdd':
            # Source transformer blocks and forward; only PE is an adapter.
            ref.get_rope = MethodType(lambda core, h, w, device: (native.rotary(core, h, w, device), None), ref)
            def attn(self, x, position, unused):
                n, s, dim = x.shape
                q, k, v = self.qkv(x).reshape(n, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = native.rotate(q, position), native.rotate(k, position)
                a = ((q @ k.transpose(-2, -1))*self.scale).softmax(-1)
                return self.proj((a @ v).transpose(1, 2).reshape(n, s, dim))
            for block in ref.blocks:
                block.attn.forward = MethodType(attn, block.attn)
        for resolution in (4, 8, 16, 32):
            ah = None
            if name in ('dfm_sincos', 'lapflow_sincos'):
                def add(module, args, out):
                    return out+native.additive(32, resolution//2, resolution//2, out.device)
                ah = ref.x_embedder.register_forward_hook(add)
            va, ga, ha = capture(ref); vb, gb, hb = capture(net.core)
            ref.zero_grad(); net.zero_grad()
            x = torch.randn(2, 4, resolution, resolution, requires_grad=True)
            z = x.detach().clone().requires_grad_()
            y = torch.tensor([3, 8]); t = torch.tensor([.2, .73])
            a = ref(x, t*1000, y)[:, :4]; b = net(z, t, y=y)
            exact(a, b, name+'/model_output')
            probe = torch.randn_like(a)
            (a*probe).sum().backward(); (b*probe).sum().backward()
            exact(x.grad, z.grad, name+'/input_gradient')
            # Additive PE is outside the port's patch module but inside the
            # reference hook, so compare block outputs and all other modules.
            for k in set(va) & set(vb):
                if k == 'x_embedder' and name in ('dfm_sincos', 'lapflow_sincos'):
                    continue
                exact(va[k], vb[k], name+'/module_output')
                if k in ga and k in gb: exact(ga[k], gb[k], name+'/activation_gradient')
            for (ka, pa), (kb, pb) in zip(ref.named_parameters(), net.core.named_parameters()):
                assert ka == kb
                exact(pa.grad, pb.grad, name+'/parameter_gradient')
            for handle in ha+hb: handle.remove()
            if ah is not None: ah.remove()
    # Joint-band position is evaluated on each port's actual token grid.
    # This checks the PE port, not DFM's original same-grid patch fusion.
    for name, order in [('dfm_sincos', 'coarse_first'), ('lapflow_sincos', 'fine_first')]:
        net = build('model', {'name':'comparison_dit', 'args':{
            'architecture':arch, 'position':name, 'channels':[4, 4],
            'product_order':order, 'causal_scales':order == 'fine_first'}}).eval()
        for p in net.parameters():
            with torch.no_grad(): p.normal_(0, .03)
        ref = copy.deepcopy(net); ref.position = AuthorPosition(name)
        values = [torch.randn(2, 4, 8, 8), torch.randn(2, 4, 16, 16)]
        if order == 'fine_first': values.reverse()
        t = torch.tensor([[.2, .7], [.3, .9]]); y = torch.tensor([3, 8])
        a = ref(values, t, y=y); b = net(values, t, y=y)
        exact(a, b, name+'/product_output')
        sum(x.square().mean() for x in a).backward()
        sum(x.square().mean() for x in b).backward()
        for (ka, pa), (kb, pb) in zip(ref.named_parameters(), net.named_parameters()):
            assert ka == kb
            exact(pa.grad, pb.grad, name+'/product_parameter_gradient')
    # Position adapters consume no initialization RNG and preserve state keys;
    # compare an actual changed output to avoid a dormant-policy test.
    torch.manual_seed(119)
    legacy = build('model', {'name':'comparison_dit', 'args':{'architecture':arch}}).eval()
    state = torch.get_rng_state().clone()
    torch.manual_seed(119)
    changed = build('model', {'name':'comparison_dit', 'args':{'architecture':arch, 'position':'pixelflow_rope'}}).eval()
    exact(state, torch.get_rng_state(), 'adapter_initialization_RNG')
    assert legacy.state_dict().keys() == changed.state_dict().keys()
    for k, v in legacy.state_dict().items(): exact(v, changed.state_dict()[k], 'adapter_initial_weights')
    for p in legacy.parameters():
        with torch.no_grad(): p.normal_(0, .2)
    changed.load_state_dict(legacy.state_dict())
    x = torch.randn(2, 4, 8, 8); t = torch.tensor([.1, .8]); y = torch.tensor([3, 8])
    assert not torch.equal(legacy(x, t, y=y), changed(x, t, y=y)), 'Position must affect the nonzero model'
    return policies


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope':'Author PE functions and their injection into the common DiT, synthetic CPU; no author-full-model or quality claim'}
    try:
        report['policies'] = run(); report['status'] = 'passed'
    except Exception:
        import traceback
        report.update(status='failed', failure=traceback.format_exc())
    report.update(checks=COUNTS, source_hashes=SOURCES, cuda_initialized=torch.cuda.is_initialized())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report['status'] != 'passed'


if __name__ == '__main__': raise SystemExit(main())
