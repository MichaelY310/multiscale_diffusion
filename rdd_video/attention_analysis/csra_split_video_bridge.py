"""Opt-in complete two-reader video cost analogue; NO trained video weights.

Exactly two full-channel readers replace fine spans [0,3), [6,9).
The real blocks in between remain native FastWan VSA with gates/coarse output.
"""
from torch import nn
from attention_analysis.fastwan_native_attention import FastWanNativeAttentionController
from attention_analysis.csra_prefix_video_bridge import VideoPrefixSamplerProxy


class _SpanController(FastWanNativeAttentionController):
    def __init__(self, model, manager, role):
        self.manager, self.role = manager, role
        super().__init__(model)
        self.configure(attention_mode='vsa', reuse_every=1, small_stage_executor='native')

    def _dispatch_block(self, layer, block, hidden_states, encoder_hidden_states,
                        temb, freqs_cis, original_seq_len):
        m = self.manager
        h = hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states
        if self.role == 'fine' and m.depth and self.current_stage == 2:
            for reader, (start_layer, end_layer) in enumerate(m.spans):
                if start_layer <= layer < end_layer:
                    m.skipped.append(layer)
                    if layer == start_layer:
                        if m.memory is None or m.condition is None:
                            raise RuntimeError('Missing actual middle hidden/current condition')
                        start, end = m.new_events()
                        if start is not None: start.record()
                        h = m.bridge[reader](h, m.condition, m.memory,
                            m.memory_grid, self._token_grid(self.current_shape))
                        if end is not None: end.record()
                        m.reader_events.append((start, end))
                        m.bridge_calls += 1
                    return h
        out = self._run_block_without_cross_stage_reuse(layer, block, hidden_states,
            encoder_hidden_states, temb, freqs_cis, original_seq_len)
        m.executed[self.current_stage] += 1
        if self.role == 'coarse' and m.depth and self.current_stage == 1 and layer == m.source_layer:
            start, end = m.new_events()
            if start is not None: start.record()
            m.memory = out.detach().clone()
            if end is not None: end.record()
            m.capture_events = (start, end)
            m.memory_grid = self._token_grid(self.current_shape)
        return out


class VideoSpanSamplerProxy(VideoPrefixSamplerProxy):
    """One installation per model: no nested old/new controller wrappers.

    reset('baseline'/'prefix6'/'split3_3') also clears all cross-sample memory.
    Both readers remain resident in ALL arms; inactive readers do not execute.
    """
    def __init__(self, original, fine, readers, source_layer=13):
        if original is fine or len(original.blocks) != len(fine.blocks):
            raise ValueError('Two distinct same-depth models required')
        if len(readers) != 2 or len(fine.blocks) < 10:
            raise ValueError('Exactly two readers and >=10 blocks required')
        if not 0 <= source_layer < len(original.blocks):
            raise ValueError('Invalid source layer')
        self.original, self.fine = original, fine
        self.bridge = nn.ModuleList(readers)
        self.patch_size, self.source_layer = original.patch_size, source_layer
        self.original_controller = _SpanController(original, self, 'coarse')
        self.fine_controller = _SpanController(fine, self, 'fine')
        self.condition_hook = fine.condition_embedder.register_forward_hook(self._capture_condition)
        self.reset('baseline')

    def reset(self, variant):
        layouts = {'baseline': (), 'prefix6': ((0,6),), 'split3_3': ((0,3),(6,9))}
        if variant not in layouts: raise ValueError('Unknown layout')
        self.variant, self.spans = variant, layouts[variant]
        self.depth, self.stage = sum(b-a for a,b in self.spans), -1
        self.memory = self.memory_grid = self.condition = None
        self.executed, self.skipped, self.bridge_calls = [0,0,0], [], 0
        self.reader_events, self.capture_events = [], (None,None)
        self.original_controller.reset(); self.fine_controller.reset()

    def audit(self):
        count = len(self.original.blocks)
        skipped = [i for a,b in self.spans for i in range(a,b)]
        if self.executed != [count,count,count-self.depth] or self.skipped != skipped:
            raise AssertionError(f'Wrong block scope: {self.executed}/{self.skipped}')
        if self.bridge_calls != len(self.spans): raise AssertionError('Wrong reader count')
        def seconds(pair):
            a,b = pair
            return a.elapsed_time(b)/1000 if a is not None else 0.
        return dict(executed=list(self.executed), skipped=list(self.skipped),
            bridge_calls=self.bridge_calls, reader_seconds=[seconds(p) for p in self.reader_events],
            bridge_seconds=sum(seconds(p) for p in self.reader_events),
            capture_seconds=seconds(self.capture_events))
