"""Exact cache-lifetime optimization for ONE model call per RDD stage only.

Never use this with250-step image sampling or multiple calls in one stage.
It changes storage lifetime, not the residual or native block arithmetic.
"""
from .fastwan_native_attention import FastWanNativeAttentionController


class OneCallCascadeController(FastWanNativeAttentionController):
    lifetime_optimized=True

    def reset(self):
        super().reset()
        self.seen_stages=set()

    def begin_model_forward(self, stage, shape):
        if stage in self.seen_stages:
            raise ValueError('OneCallCascadeController requires ONE NFE per stage')
        self.seen_stages.add(stage)
        if (self.attention_mode!='vsa' or self.cross_stage_reuse_layer_coefficients
            or self.cross_stage_reuse_layer_coefficients_by_stage
            or self.cross_stage_reuse_precompose):
            raise ValueError('Only direct previous-stage residuals with native VSA')
        super().begin_model_forward(stage,shape)

    def _should_capture_cross_stage_residual(self, layer):
        if not self.lifetime_optimized:
            return super()._should_capture_cross_stage_residual(layer)
        target=self.current_stage+1
        return target in self.cross_stage_reuse_stages and layer in self._reuse_layers_for_stage(target)

    def _dispatch_block(self, layer, *args, **kwargs):
        output=super()._dispatch_block(layer,*args,**kwargs)
        if self.lifetime_optimized:
            # Previous-stage layer was consumed; this stage runs only once.
            self.cross_stage_residual_cache.pop((self.current_stage-1,layer),None)
            # In particular, do not retain a huge full-resolution residual
            # cache through VAE decode if there is no following consumer.
            if not self._should_capture_cross_stage_residual(layer):
                self.cross_stage_residual_cache.pop((self.current_stage,layer),None)
        return output
