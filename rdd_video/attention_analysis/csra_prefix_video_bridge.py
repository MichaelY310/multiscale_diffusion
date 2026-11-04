"""Opt-in video analogue of the trained 2-D conditional prefix bridge.

This module has NO trained Wan bridge weights. It exists to count the actual
compute/memory of the candidate, not to claim video quality or checkpoint parity
when prefix replacement is enabled. The original FastVideo source is untouched.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from attention_analysis.fastwan_native_attention import FastWanNativeAttentionController


def lift_token_grid(source, source_grid, target_grid):
    if source.ndim != 3 or source.shape[1] != math.prod(source_grid):
        raise ValueError('Source hidden sequence does not match its T/H/W token grid')
    if any(b % a for a, b in zip(source_grid, target_grid)):
        raise ValueError('This feasibility test requires exactly nested token grids')
    b, _, h = source.shape
    volume = source.reshape(b, *source_grid, h).permute(0, 4, 1, 2, 3)
    lifted = F.interpolate(volume, size=target_grid, mode='nearest')
    return lifted.permute(0, 2, 3, 4, 1).reshape(b, -1, h).contiguous()


class VideoConditionalPrefixBridge(nn.Module):
    """Same operations as ConditionalPrefixStudent, generalized to native 3-D KV.

    FP32 full-rank affine + BF16 fine-to-coarse attention + BF16 pointwise FFN.
    Wan uses its actual unmodulated timestep embedding for c. Unlike 2-D there
    can be N/8 (spatiotemporal) as well as N/4 (spatial) memory tokens.
    """
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must divide heads')
        self.heads = heads
        self.proj = nn.Linear(3 * hidden, hidden, dtype=torch.float32)
        self.q_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.m_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 4 * hidden))
        self.query = nn.Linear(hidden, hidden)
        self.key_value = nn.Linear(hidden, 2 * hidden)
        self.read_out = nn.Linear(hidden, hidden)
        self.nonlinear = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                       nn.Linear(2 * hidden, hidden))

    def forward(self, h0, c, memory, source_grid, target_grid):
        b, n, h = h0.shape
        if n != math.prod(target_grid) or c.shape != (b, h):
            raise ValueError('Fine sequence / current timestep embedding mismatch')
        if memory.shape != (b, math.prod(source_grid), h):
            raise ValueError('Native coarse hidden shape mismatch')
        with torch.autocast(h0.device.type, enabled=False):
            source = lift_token_grid(memory.float(), source_grid, target_grid)
            joined = torch.cat((h0.float(), c[:, None].expand(-1, n, -1).float(), source), -1)
            base = h0.float() + self.proj(joined)
        with torch.autocast(h0.device.type, dtype=torch.bfloat16, enabled=h0.is_cuda):
            sq, aq, sm, am = self.condition(c).chunk(4, -1)
            q_input = self.q_norm(h0) * (1 + aq[:, None]) + sq[:, None]
            m_input = self.m_norm(memory) * (1 + am[:, None]) + sm[:, None]
            q = self.query(q_input).reshape(b, n, self.heads, h // self.heads).transpose(1, 2)
            kv = self.key_value(m_input).reshape(b, memory.shape[1], 2, self.heads, h // self.heads)
            k, v = kv.unbind(2)
            # SDPA must not materialize the Nfine x Ncoarse probability matrix
            # on video. The benchmark explicitly restricts the CUDA backend.
            read = F.scaled_dot_product_attention(q, k.transpose(1, 2),
                                                  v.transpose(1, 2), dropout_p=0.)
            read = self.read_out(read.transpose(1, 2).reshape(b, n, h))
            local = self.nonlinear(q_input)
        return (base + read.float() + local.float()).to(h0.dtype)


class _PrefixController(FastWanNativeAttentionController):
    def __init__(self, model, manager, role):
        self.manager, self.role = manager, role
        super().__init__(model)
        self.configure(attention_mode='vsa', reuse_every=1, small_stage_executor='native')

    def _dispatch_block(self, layer, block, hidden_states, encoder_hidden_states,
                        temb, freqs_cis, original_seq_len):
        m = self.manager
        h = hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states
        if (self.role == 'fine' and m.depth > 0 and self.current_stage == 2
                and m.front_keep <= layer < m.front_keep + m.depth):
            m.skipped.append(layer)
            if layer == m.front_keep:
                if m.memory is None or m.condition is None:
                    raise RuntimeError('Missing actual middle hidden / current fine timestep')
                start, end = m.new_events()
                if start is not None: start.record()
                h = m.bridge(h, m.condition, m.memory,
                             m.memory_grid, self._token_grid(self.current_shape))
                if end is not None: end.record()
                m.bridge_events = (start, end)
                m.bridge_calls += 1
            return h
        out = self._run_block_without_cross_stage_reuse(
            layer, block, hidden_states, encoder_hidden_states, temb, freqs_cis, original_seq_len)
        m.executed[self.current_stage] += 1
        if self.role == 'coarse' and m.depth > 0 and self.current_stage == 1 and layer == m.source_layer:
            start, end = m.new_events()
            if start is not None: start.record()
            m.memory = out.detach().clone()  # full hidden, NOT residual delta
            if end is not None: end.record()
            m.capture_events = (start, end)
            m.memory_grid = self._token_grid(self.current_shape)
        return out


class VideoPrefixSamplerProxy:
    """SP=1 only, separate original/coarse and potentially trainable fine weights.

    depth=0 is a no-op baseline. depth>0 replaces ONLY the last NFE's prefix;
    all executed blocks retain native VSA including coarse-output and gates.
    """
    attention_mode = 'vsa'

    def __init__(self, original, fine, bridge, source_layer=13, front_keep=0):
        if original is fine:
            raise ValueError('Separate fine weights must be counted, not silently shared')
        if len(original.blocks) != len(fine.blocks):
            raise ValueError('Model depths do not match')
        if not 0 <= source_layer < len(original.blocks):
            raise ValueError('Invalid source layer')
        if type(front_keep) is not int or not 0 <= front_keep < len(fine.blocks):
            raise ValueError('Invalid number of real fine blocks before bridge')
        self.front_keep = front_keep
        self.original, self.fine, self.bridge = original, fine, bridge
        self.patch_size, self.source_layer = original.patch_size, source_layer
        self.depth, self.stage = 0, -1
        self.original_controller = _PrefixController(original, self, 'coarse')
        self.fine_controller = _PrefixController(fine, self, 'fine')
        self.condition_hook = fine.condition_embedder.register_forward_hook(self._capture_condition)
        self.reset(0)

    def _capture_condition(self, module, args, output):
        self.condition = output[0]  # actual Wan unmodulated embedding, not mean(6H)

    def reset(self, depth):
        if not 0 <= depth < len(self.fine.blocks) - self.front_keep:
            raise ValueError('Invalid prefix depth')
        self.depth, self.stage = depth, -1
        self.memory = self.memory_grid = self.condition = None
        self.executed, self.skipped, self.bridge_calls = [0, 0, 0], [], 0
        self.bridge_events = self.capture_events = (None, None)
        self.original_controller.reset()
        self.fine_controller.reset()

    def begin_model_forward(self, stage, shape):
        if stage != self.stage + 1 or stage not in (0, 1, 2):
            raise ValueError('Expected exactly three ordered NFE')
        self.stage = stage
        self.original_controller.begin_model_forward(stage, shape)
        self.fine_controller.begin_model_forward(stage, shape)

    def new_events(self):
        if next(self.bridge.parameters()).is_cuda:
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        return None, None

    def __call__(self, **kwargs):
        model = self.fine if self.depth and self.stage == 2 else self.original
        result = model(**kwargs)
        if self.stage == 2:
            self.memory = self.condition = None  # free cache inside measured workload
        return result

    def audit(self):
        layers = len(self.original.blocks)
        expected = [layers, layers, layers - self.depth]
        skipped = list(range(self.front_keep, self.front_keep + self.depth))
        if self.executed != expected or self.skipped != skipped:
            raise AssertionError(f'Wrong executed/skipped blocks: {self.executed}/{self.skipped}')
        if self.bridge_calls != int(self.depth > 0):
            raise AssertionError('Bridge must execute exactly once when replacing prefix')
        def duration(events):
            a, b = events
            return float(a.elapsed_time(b)) / 1000 if a is not None else 0.
        return dict(executed=list(self.executed), skipped=list(self.skipped),
                    bridge_calls=self.bridge_calls, bridge_seconds=duration(self.bridge_events),
                    capture_seconds=duration(self.capture_events))
