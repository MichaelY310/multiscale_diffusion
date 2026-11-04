"""Configurable common backbones, independent of diffusion method and grid rank.

These are controlled architectures, not renamed author networks. DiT uses
patch tokens, adaLN-Zero blocks and joint attention across frames/scales.
"""
import math
import torch
from torch import nn
from .registry import register, build
from .grids import as_video


def fourier(value, width):
    n = (width + 1)//2
    freq = torch.exp(-math.log(10000)*torch.arange(n, device=value.device, dtype=torch.float32)/max(n-1, 1))
    phase = value.float()[..., None]*freq
    return torch.cat((phase.sin(), phase.cos()), -1)[..., :width]


def grid_position(coords,width):
    # Distinct channel subspaces identify T,H,W. Summing the same encoding for
    # all axes would alias coordinate permutations such as (t,h,w) and (h,t,w).
    sizes=[width//3,width//3,width-2*(width//3)]
    return torch.cat([fourier(coords[:,axis],size) for axis,size in enumerate(sizes)],-1)


def patchify(x, patch):
    x = as_video(x)
    b, c, t, h, w = x.shape
    pt, ph, pw = patch
    if t % pt or h % ph or w % pw:
        raise ValueError(f'Grid {x.shape[2:]} is not divisible by patch {patch}')
    grid = (t//pt, h//ph, w//pw)
    return x.reshape(b, c, grid[0], pt, grid[1], ph, grid[2], pw).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, math.prod(grid), -1), grid


def unpatchify(tokens, grid, patch, channels, image=False):
    b = tokens.shape[0]
    t, h, w = grid
    pt, ph, pw = patch
    x = tokens.reshape(b, t, h, w, pt, ph, pw, channels).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, channels, t*pt, h*ph, w*pw)
    return x.squeeze(2) if image else x


def rotary(x, coords):
    # x: B, heads, N, head_dim; leave any remainder unrotated.
    pairs = x.shape[-1]//6
    if pairs == 0:
        raise ValueError('3D RoPE needs head_dim >= 6')
    chunks = []
    offset = 0
    for axis in range(3):
        freq = torch.exp(-math.log(10000)*torch.arange(pairs, device=x.device, dtype=torch.float32)/max(pairs-1, 1))
        phase = (coords[:, axis, None]*freq).to(x.dtype)[None, None]
        block = x[..., offset:offset+2*pairs].reshape(*x.shape[:-1], pairs, 2)
        a, b = block.unbind(-1)
        chunks.append(torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1).flatten(-2))
        offset += 2*pairs
    return torch.cat((*chunks, x[..., offset:]), -1)


class DiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, text_dim, position):
        super().__init__()
        self.heads, self.position = heads, position
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))
        self.mlp = nn.Sequential(nn.Linear(width, int(width*mlp_ratio)), nn.GELU(approximate='tanh'), nn.Linear(int(width*mlp_ratio), width))
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True, kdim=text_dim, vdim=text_dim) if text_dim else None
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond, coords, mask, text, text_mask):
        shift, scale, gate, shift2, scale2, gate2 = self.modulation(cond).chunk(6, -1)
        h = self.norm1(x)*(1+scale)+shift
        q, k, v = self.qkv(h).reshape(h.shape[0], h.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        if self.position == 'rope':
            q, k = rotary(q, coords), rotary(k, coords)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x+gate*self.proj(attn.transpose(1, 2).flatten(2))
        if self.cross is not None:
            if text is None:
                raise ValueError('Configured text cross-attention requires text embeddings')
            x = x+self.cross(x, text, text, key_padding_mask=None if text_mask is None else ~text_mask.bool(), need_weights=False)[0]
        return x+gate2*self.mlp(self.norm2(x)*(1+scale2)+shift2)


@register('model', 'grid_dit')
class GridDiT(nn.Module):
    def __init__(self, channels=3, width=192, depth=4, num_heads=6, patch=(1, 2, 2),
                 position='sincos', coordinate_scale='unit', max_stages=8, heads=None,
                 num_classes=0, text_dim=None, mlp_ratio=4., causal_scales=False,
                 gradient_checkpointing=False, product_order='coarse_first'):
        super().__init__()
        if width % num_heads or depth < 1 or max_stages < 1:
            raise ValueError('Invalid DiT width/depth/head count')
        if position not in ('sincos', 'rope') or coordinate_scale not in ('unit', 'index'):
            raise ValueError('Choose sincos/rope and unit/index coordinates')
        if position == 'rope' and width//num_heads < 6:
            raise ValueError('3D RoPE needs head_dim >= 6')
        if product_order not in ('coarse_first', 'fine_first'):
            raise ValueError('Unknown product-state order')
        self.product_order = product_order
        self.patch = tuple(patch) if len(patch) == 3 else (1, *patch)
        if len(self.patch) != 3 or min(self.patch) < 1:
            raise ValueError('Patch must contain positive T,H,W dimensions')
        self.channels = [channels] if isinstance(channels, int) else list(channels)
        self.width, self.position, self.coordinate_scale = width, position, coordinate_scale
        self.head_names, self.max_stages = heads, max_stages
        self.causal_scales, self.gradient_checkpointing = causal_scales, gradient_checkpointing
        if heads and len(heads) != len(set(heads)):
            raise ValueError('Duplicate prediction heads')
        count = len(heads) if heads else 1
        self.inputs = nn.ModuleList([nn.Linear(c*math.prod(self.patch), width) for c in self.channels])
        self.outputs = nn.ModuleList([nn.Linear(width, c*math.prod(self.patch)*count) for c in self.channels])
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.stage_embed = nn.Embedding(max_stages, width)
        self.clock_vector = nn.Linear(max_stages, width, bias=False)
        self.label = nn.Embedding(num_classes+1, width) if num_classes else None
        self.blocks = nn.ModuleList([DiTBlock(width, num_heads, mlp_ratio, text_dim, position) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))

    def forward(self, x, t, stage=0, y=None, text=None, text_mask=None, drop_mask=None):
        is_dict, is_list = isinstance(x, dict), isinstance(x, (list, tuple))
        keys = list(x) if is_dict else list(range(len(x))) if is_list else [0]
        leaves = [x[k] for k in keys] if is_dict or is_list else [x]
        if t.ndim==0:t=t.expand(len(leaves[0]))
        tokens, conditions, coordinates, layouts, ids = [], [], [], [], []
        for j, (key, value) in enumerate(zip(keys, leaves)):
            idx = int(key) if is_dict or is_list else int(stage)
            if is_list and self.product_order == 'fine_first':
                idx += self.max_stages-len(leaves)
            if not 0 <= idx < self.max_stages:
                raise ValueError('Stage index outside configured max_stages')
            port = idx if len(self.channels) > 1 else 0
            if port >= len(self.channels) or value.shape[1] != self.channels[port]:
                raise ValueError('Channel count differs from the stage input/output port')
            z, grid = patchify(value, self.patch)
            axes = [torch.arange(n, device=value.device, dtype=torch.float32) for n in grid]
            if self.coordinate_scale == 'unit':
                axes = [(a+.5)/n for a, n in zip(axes, grid)]
            coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
            local = t[:, idx] if t.ndim == 2 and (is_dict or is_list) and t.shape[1] == self.max_stages else t[:, j] if t.ndim == 2 and (is_dict or is_list) else t
            if local.ndim > 1:
                # B,T clocks (e.g. separate first/tail noise); align with tokens.
                if local.shape != (len(value), as_video(value).shape[2]):
                    raise ValueError('Tensor input expects B times or B,T frame times')
                local = local.reshape(len(value), grid[0], self.patch[0]).mean(-1)
                emb = self.time(fourier(local, self.width).to(z.dtype)).repeat_interleave(grid[1]*grid[2], 1)
            else:
                emb = self.time(fourier(local, self.width).to(z.dtype))[:, None].expand(-1, z.shape[1], -1)
            emb = emb+self.stage_embed.weight[idx]
            if t.ndim == 2 and (is_dict or is_list):
                if t.shape[1] > self.max_stages:
                    raise ValueError('Too many product-state clocks')
                clocks = torch.nn.functional.pad(t, (0, self.max_stages-t.shape[1])).to(z.dtype)
                emb = emb+self.clock_vector(clocks)[:, None]
            if self.label is not None:
                if y is None:
                    raise ValueError('Class-conditioned DiT requires y labels')
                emb = emb+self.label(y)[:, None]
            elif y is not None:
                raise ValueError('Labels supplied to an unconditional DiT')
            z = self.inputs[port](z)
            if self.position == 'sincos':
                z = z+grid_position(coords,self.width).to(z.dtype)[None]
            tokens.append(z); conditions.append(emb); coordinates.append(coords)
            layouts.append((grid, port, value.ndim == 4, z.shape[1]))
            ids.extend([idx]*z.shape[1])
        z, cond, coords = torch.cat(tokens, 1), torch.cat(conditions, 1), torch.cat(coordinates)
        mask = None
        if self.causal_scales:
            ids = torch.tensor(ids, device=z.device)
            mask = ids[None, :] <= ids[:, None] if self.product_order == 'coarse_first' else ids[None, :] >= ids[:, None]
        if drop_mask is not None:
            # Inactive scales cannot leak their state into active scales.
            allowed = torch.cat([torch.full((layout[-1],), bool(drop_mask[j]), device=z.device) for j, layout in enumerate(layouts)])
            mask = allowed[None].expand(z.shape[1], -1) if mask is None else mask & allowed[None]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                z = checkpoint(block, z, cond, coords, mask, text, text_mask, use_reentrant=False)
            else:
                z = block(z, cond, coords, mask, text, text_mask)
        shift, scale = self.final_modulation(cond).chunk(2, -1)
        z = self.norm(z)*(1+scale)+shift
        results = []
        for part, (grid, port, image, _) in zip(z.split([s[-1] for s in layouts], 1), layouts):
            out = self.outputs[port](part)
            if self.head_names:
                # Each named head owns a contiguous patch-sized output chunk.
                results.append({name: unpatchify(chunk, grid, self.patch, self.channels[port], image) for name, chunk in zip(self.head_names, out.chunk(len(self.head_names), -1))})
            else:
                results.append(unpatchify(out, grid, self.patch, self.channels[port], image))
        return dict(zip(keys, results)) if is_dict else results if is_list else results[0]


@register('model', 'grid_conv')
class GridConv(nn.Module):
    def __init__(self, channels=3, width=32, heads=None, max_stages=8):
        super().__init__()
        self.head_names = heads
        self.input = nn.Conv3d(channels, width, 3, padding=1)
        self.body = nn.Sequential(nn.GroupNorm(4, width), nn.SiLU(), nn.Conv3d(width, width, 3, padding=1), nn.SiLU())
        self.time = nn.Linear(32, width)
        self.stage = nn.Embedding(max_stages, width)
        self.output = nn.Conv3d(width, channels*(len(heads) if heads else 1), 3, padding=1)

    def forward(self, x, t, stage=0):
        image = x.ndim == 4
        x = as_video(x)
        emb = self.time(fourier(t, 32).to(x.dtype))
        emb = emb[:, :, None, None, None] if t.ndim == 1 else emb.transpose(1, 2)[:, :, :, None, None]
        h = self.input(x)+emb+self.stage.weight[stage][None, :, None, None, None]
        out = self.output(h+self.body(h))
        if image:
            out = out.squeeze(2)
        return dict(zip(self.head_names, out.chunk(len(self.head_names), 1))) if self.head_names else out


@register('model', 'stage_models')
class StageModels(nn.Module):
    """Same architecture is independent of parameter sharing across stages."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList([build('model', spec) for spec in models])

    def forward(self, x, t, stage=0, **kwargs):
        return self.models[stage](x, t, stage=stage, **kwargs)


@register('model', 'dit_grid')
class AuthorDiTGrid(nn.Module):
    """Original local DiT blocks with dynamic spatial PE and joint video tokens.

At its original 2D grid the wrapper uses the exact original embedding path.
Video adds temporal position vectors; patch embedding stays spatial (1,p,p).
"""
    def __init__(self, architecture, heads=None, time_scale=1000., temporal_position=True):
        super().__init__()
        args = dict(architecture)
        if args.get('d_prediction', False):
            raise ValueError('Select named heads through this adapter, not d_prediction')
        if heads is not None and len(heads) not in (1, 2):
            raise ValueError('Original DiT adapter supports one or two output heads')
        if heads:
            args['learn_sigma'] = len(heads) == 2
        elif args.get('learn_sigma', False):
            raise ValueError('Name both output heads explicitly when learn_sigma=True')
        else:
            args['learn_sigma'] = False
        self.model = build('model', {'name': 'native_rdd', 'args': args})
        self.head_names, self.time_scale, self.temporal_position = heads, time_scale, temporal_position

    def forward(self, x, t, stage=0, y=None):
        if t.ndim != 1:
            raise ValueError('Original DiT uses one time per example; use grid_dit/wan_grid for per-frame clocks')
        net = self.model; image = x.ndim == 4; video = as_video(x)
        b, c, frames, h, w = video.shape; p = net.patch_size
        if h % p or w % p:
            raise ValueError('DiT patch must divide the spatial grid')
        if y is None:
            y = torch.zeros(b, device=x.device, dtype=torch.long)
        original_grid = net.x_embedder.img_size
        if frames == 1 and (h, w) == tuple(original_grid):
            out = net(video.squeeze(2), t*self.time_scale, y)
            if not image:out=out.unsqueeze(2)
        else:
            import numpy as np
            helpers = net.initialize_weights.__func__.__globals__
            flat = video.permute(0, 2, 1, 3, 4).reshape(b*frames, c, h, w)
            tokens = net.x_embedder.proj(flat).flatten(2).transpose(1, 2)
            tokens = net.x_embedder.norm(tokens)
            coord = np.stack(np.meshgrid(np.arange(w//p, dtype=np.float32), np.arange(h//p, dtype=np.float32)), axis=0).reshape(2, 1, h//p, w//p)
            spatial = torch.from_numpy(helpers['get_2d_sincos_pos_embed_from_grid'](tokens.shape[-1], coord)).to(tokens)
            tokens = (tokens+spatial[None]).reshape(b, frames, -1, tokens.shape[-1])
            if self.temporal_position and frames > 1:
                temporal = helpers['get_1d_sincos_pos_embed_from_grid'](tokens.shape[-1], np.arange(frames, dtype=np.float32))
                temporal = torch.from_numpy(temporal-temporal[:1]).to(tokens)
                tokens = tokens+temporal[None, :, None]
            tokens = tokens.flatten(1, 2)
            cond = net.t_embedder(t*self.time_scale)+net.y_embedder(y, net.training)
            for block in net.blocks:
                tokens = block(tokens, cond)
            out = unpatchify(net.final_layer(tokens, cond), (frames, h//p, w//p), (1, p, p), net.out_channels, image)
        return dict(zip(self.head_names, out.chunk(len(self.head_names), 1))) if self.head_names else out


@register('model', 'wan_grid')
class WanGrid(nn.Module):
    """Offline native Diffusers Wan backbone; image inputs use T=1.

No VAE/text encoder/checkpoint is implicitly constructed or downloaded.
"""
    def __init__(self, architecture, heads=None, time_scale=1000.):
        super().__init__()
        args = dict(architecture)
        channels = args.get('in_channels', 16)
        count = len(heads) if heads else 1
        if args.get('out_channels', channels*count) != channels*count:
            raise ValueError('Wan output channels must match configured prediction heads')
        args['out_channels'] = channels*count
        self.model = build('model', {'name': 'wan_transformer', 'args': args})
        self.head_names, self.time_scale = heads, time_scale

    def forward(self, x, t, stage=0, text=None):
        if text is None:
            raise ValueError('Wan requires explicit local text embeddings (or an explicit zero fixture)')
        image = x.ndim == 4; x = as_video(x)
        if t.ndim == 2:
            pt, ph, pw = self.model.config.patch_size
            if t.shape != (len(x), x.shape[2]) or x.shape[2] % pt:
                raise ValueError('Wan frame times must match the video grid')
            t = t.reshape(len(x), -1, pt).mean(-1).repeat_interleave((x.shape[3]//ph)*(x.shape[4]//pw), -1)
        out = self.model(hidden_states=x, timestep=t*self.time_scale, encoder_hidden_states=text, return_dict=False)[0]
        if image:
            out = out.squeeze(2)
        return dict(zip(self.head_names, out.chunk(len(self.head_names), 1))) if self.head_names else out


@register('model', 'edm_grid')
class EDMGrid(nn.Module):
    """EDM preconditioning facade usable by the existing Relay/EDM samplers."""
    def __init__(self, model, sigma_data=.5, sigma_min=0., sigma_max=float('inf'), img_resolution=32):
        super().__init__(); self.model=build('model',model)
        self.sigma_data=sigma_data; self.sigma_min=sigma_min; self.sigma_max=sigma_max; self.img_resolution=img_resolution

    def round_sigma(self, sigma):return torch.as_tensor(sigma)

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        if augment_labels is not None:
            raise ValueError('Configure a backbone augmentation adapter to use nonzero Relay augmentation labels')
        x=x.float(); sigma=sigma.float().reshape(-1)
        if len(sigma)==1:sigma=sigma.expand(len(x))
        s=sigma.reshape(len(x),*([1]*(x.ndim-1))); d=self.sigma_data
        kwargs={} if class_labels is None else {'y':class_labels.argmax(-1) if class_labels.ndim==2 else class_labels}
        pred=self.model(x/(s*s+d*d).sqrt(),sigma.log()/4,**kwargs)
        return d*d/(s*s+d*d)*x+s*d/(s*s+d*d).sqrt()*pred


@register('model', 'ssd_grid')
class SSDGrid(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=build('model',model)

    def forward(self,x,t,cond_rs=None,**kwargs):
        from .operators import resize
        if cond_rs is not None:
            if not torch.all(cond_rs==cond_rs[0]):raise ValueError('SSD batch requires a common output grid')
            size=int(cond_rs[0]); x=resize(x,size=(size,size),mode='bilinear',antialias=True)
        return self.model(x,t,**kwargs)
