"""Training-free attention controls for the native FastWan VSA model.

The controller exposes two independent choices without modifying checkpoint
weights:

* ``vsa`` keeps FastVideo's learned online VSA router;
* ``c2f_route`` replaces the online router after the coarsest RDD stage with
  block routes captured at the preceding resolution, while retaining the
  checkpoint's current-resolution VSA coarse-output branch;
* ``c2f_route_only`` is the deployable RDD-native variant: after the first
  stage it lifts the previous sparse route and executes only sparse fine
  attention.  It requires RDD/C2F finetuning for final quality, but measures
  the method without silently paying VSA's extra online coarse attention.

``reuse_every > 1`` implements StepReuse-Out(k): within repeated calls at the
same latent shape, self-attention is recomputed once every k calls and reused
in between.  ``reuse_scope=backend`` preserves the original experiment path
which caches ``O=A*V`` but still executes QKV/output projections.
``reuse_scope=projected`` caches the post-``to_out`` self-attention tensor and
skips norm/QKV/gate/RoPE/tile/sparse-attention/output-projection work on a hit.
The two scopes produce the same self-attention residual input because all
skipped projections have fixed inference weights.  A three-call 4x->2x->1x
FastWan trajectory has one call per shape, hence intentionally records zero
hits.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as functional


@dataclass
class BlockRoute:
    block_grid: tuple[int, int, int]
    indices: torch.Tensor
    counts: torch.Tensor
    # Dense block-level probability from the seed (lowest-resolution) stage.
    # It is tiny compared with token attention and lets a later RDD stage keep
    # VSA's global coarse branch without recomputing fine-stage QcKc.
    probabilities: torch.Tensor | None = None
    probability_grid: tuple[int, int, int] | None = None


def _flat(coord: tuple[int, int, int], grid: tuple[int, int, int]) -> int:
    return (coord[0] * grid[1] + coord[1]) * grid[2] + coord[2]


def install_projected_cache_class_patch(block_cls: type) -> None:
    """Install dispatch before FastVideo constructs/wraps Wan VSA blocks.

    FastVideo may keep parameters as zero-sized placeholders outside its
    managed forward region.  Patching an individual block *after* loading can
    bypass that region.  A class-level dispatch installed before construction
    remains inside the loader/FSDP lifecycle and is a no-op until a controller
    attaches itself to a concrete block.
    """
    if hasattr(block_cls, "_rdd_original_forward"):
        return
    original_forward = block_cls.forward

    def dispatch(block, *args, **kwargs):
        controller = getattr(block, "_rdd_attention_controller", None)
        if controller is None:
            return original_forward(block, *args, **kwargs)
        return controller._dispatch_block(
            int(block._rdd_attention_layer), block, *args, **kwargs
        )

    block_cls._rdd_original_forward = original_forward
    block_cls.forward = dispatch


class FastWanNativeAttentionController:
    """Patch native FastWan blocks while preserving their learned weights."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.original_block_forwards: list[Callable] = []
        self.original_impl_forwards: list[Callable] = []
        self.attention_mode = "vsa"
        self.route_density = 0.2
        self.reuse_every = 1
        self.reuse_scope = "backend"
        self.small_stage_executor = "native"
        self.small_stage_max_padded_tokens = 0
        self.cross_stage_reuse_layers: frozenset[int] = frozenset()
        self.cross_stage_reuse_layers_by_stage: dict[int, frozenset[int]] = {}
        self.cross_stage_reuse_stages: frozenset[int] = frozenset()
        self.cross_stage_reuse_scale = 1.0
        self.cross_stage_reuse_scale_by_stage: dict[int, float] = {}
        self.cross_stage_reuse_layer_coefficients: dict[
            int, tuple[float, float, float]
        ] = {}
        self.cross_stage_reuse_layer_coefficients_by_stage: dict[
            int, dict[int, tuple[float, float, float]]
        ] = {}
        self.cross_stage_reuse_precompose = False
        self.cross_stage_reuse_resize = "nearest"
        self.cross_stage_branch_reuse = "none"
        self.cross_stage_branch_reuse_layers: frozenset[int] = frozenset()
        self.cross_stage_branch_reuse_stages: frozenset[int] = frozenset()
        self.cross_stage_branch_reuse_scale = 1.0
        self.current_stage = 0
        self.current_shape: tuple[int, int, int] = (0, 0, 0)
        self.current_shape_step = 0
        self._install()
        self.reset()

    def _install(self) -> None:
        for layer_id, block in enumerate(self.model.blocks):
            class_original = getattr(type(block), "_rdd_original_forward", None)
            original_block_forward = (
                class_original.__get__(block, type(block))
                if class_original is not None
                else block.forward
            )
            original_impl = block.attn1.attn_impl.forward
            self.original_block_forwards.append(original_block_forward)
            self.original_impl_forwards.append(original_impl)
            block._rdd_attention_controller = self
            block._rdd_attention_layer = layer_id

            if class_original is None:
                # Standalone tests or older FastVideo versions that did not
                # install the pre-construction class dispatch.
                def block_forward(
                    this,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    freqs_cis,
                    original_seq_len,
                    _layer=layer_id,
                ):
                    return self._dispatch_block(
                        _layer,
                        this,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        freqs_cis,
                        original_seq_len,
                    )

                block.forward = types.MethodType(block_forward, block)

            def impl_forward(
                this,
                query,
                key,
                value,
                gate_compress,
                attn_metadata,
                _layer=layer_id,
            ):
                return self._impl_forward(
                    _layer,
                    query,
                    key,
                    value,
                    gate_compress,
                    attn_metadata,
                )

            block.attn1.attn_impl.forward = types.MethodType(
                impl_forward, block.attn1.attn_impl
            )

    def configure(
        self,
        attention_mode: str,
        route_density: float = 0.2,
        reuse_every: int = 1,
        reuse_scope: str = "backend",
        small_stage_executor: str = "native",
        small_stage_max_padded_tokens: int = 0,
        cross_stage_reuse_layers: Sequence[int] | None = None,
        cross_stage_reuse_layers_by_stage: dict[
            int | str, Sequence[int]
        ] | None = None,
        cross_stage_reuse_stages: Sequence[int] | None = None,
        cross_stage_reuse_scale: float = 1.0,
        cross_stage_reuse_scale_by_stage: dict[int | str, float] | None = None,
        cross_stage_reuse_layer_coefficients: dict[
            int | str, Sequence[float]
        ] | None = None,
        cross_stage_reuse_layer_coefficients_by_stage: dict[
            int | str, dict[int | str, Sequence[float]]
        ] | None = None,
        cross_stage_reuse_precompose: bool = False,
        cross_stage_reuse_resize: str = "nearest",
        cross_stage_branch_reuse: str = "none",
        cross_stage_branch_reuse_layers: Sequence[int] | None = None,
        cross_stage_branch_reuse_stages: Sequence[int] | None = None,
        cross_stage_branch_reuse_scale: float = 1.0,
    ) -> None:
        if attention_mode not in {
            "vsa",
            "c2f_route",
            "c2f_route_only",
            "c2f_route_reuse_coarse",
        }:
            raise ValueError(f"unknown attention mode: {attention_mode}")
        if not 0.0 < route_density <= 1.0:
            raise ValueError("route_density must be in (0,1]")
        if reuse_every < 1:
            raise ValueError("reuse_every must be >=1")
        if reuse_scope not in {"backend", "projected"}:
            raise ValueError("reuse_scope must be 'backend' or 'projected'")
        if small_stage_executor not in {
            "native",
            "masked_sdpa",
            "dense_sdpa",
            "triton_q16",
            "triton_q32",
        }:
            raise ValueError(
                "small_stage_executor must be native, masked_sdpa, dense_sdpa, "
                "triton_q16 or triton_q32"
            )
        if small_stage_max_padded_tokens < 0:
            raise ValueError("small_stage_max_padded_tokens must be >= 0")
        reuse_layers = frozenset(
            int(value) for value in (cross_stage_reuse_layers or ())
        )
        reuse_layers_by_stage = {
            int(stage): frozenset(int(value) for value in values)
            for stage, values in (cross_stage_reuse_layers_by_stage or {}).items()
        }
        reuse_stages = frozenset(
            int(value) for value in (cross_stage_reuse_stages or ())
        ) | frozenset(reuse_layers_by_stage)
        all_reuse_layers = set(reuse_layers)
        for values in reuse_layers_by_stage.values():
            all_reuse_layers.update(values)
        if any(
            layer < 0 or layer >= len(self.model.blocks)
            for layer in all_reuse_layers
        ):
            raise ValueError(
                f"cross_stage_reuse_layers must be in [0,{len(self.model.blocks) - 1}]"
            )
        if any(stage <= 0 for stage in reuse_stages):
            raise ValueError("cross_stage_reuse_stages must contain positive stage indices")
        if not math.isfinite(cross_stage_reuse_scale):
            raise ValueError("cross_stage_reuse_scale must be finite")
        scale_by_stage = {
            int(stage): float(value)
            for stage, value in (cross_stage_reuse_scale_by_stage or {}).items()
        }
        if any(stage <= 0 for stage in scale_by_stage):
            raise ValueError("cross_stage_reuse_scale_by_stage keys must be positive")
        if any(not math.isfinite(value) for value in scale_by_stage.values()):
            raise ValueError("cross-stage per-stage scales must be finite")

        def parse_coefficients(
            raw: dict[int | str, Sequence[float]] | None,
        ) -> dict[int, tuple[float, float, float]]:
            parsed: dict[int, tuple[float, float, float]] = {}
            for raw_layer, raw_values in (raw or {}).items():
                layer = int(raw_layer)
                values = tuple(float(value) for value in raw_values)
                if len(values) not in {2, 3}:
                    raise ValueError(
                        "each cross-stage layer coefficient must be [a,b] or [a,b,beta]"
                    )
                if len(values) == 2:
                    values = (*values, 0.0)
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("cross-stage layer coefficients must be finite")
                parsed[layer] = values
            return parsed

        layer_coefficients = parse_coefficients(
            cross_stage_reuse_layer_coefficients
        )
        layer_coefficients_by_stage = {
            int(stage): parse_coefficients(values)
            for stage, values in (
                cross_stage_reuse_layer_coefficients_by_stage or {}
            ).items()
        }
        if not set(layer_coefficients).issubset(reuse_layers):
            raise ValueError(
                "cross-stage coefficient keys must be selected reuse layers"
            )
        if any(stage < 2 for stage in reuse_stages) and any(
            abs(values[1]) > 0.0 for values in layer_coefficients.values()
        ):
            raise ValueError(
                "the M-C trend term requires a target stage index of at least 2"
            )
        for stage, coefficients in layer_coefficients_by_stage.items():
            if stage <= 0:
                raise ValueError(
                    "cross_stage_reuse_layer_coefficients_by_stage keys must be positive"
                )
            selected = reuse_layers | reuse_layers_by_stage.get(stage, frozenset())
            if not set(coefficients).issubset(selected):
                raise ValueError(
                    f"stage {stage} coefficient keys must be selected reuse layers"
                )
            if stage < 2 and any(abs(values[1]) > 0.0 for values in coefficients.values()):
                raise ValueError("the M-C trend term requires target stage >= 2")
        if cross_stage_reuse_resize not in {"nearest", "trilinear"}:
            raise ValueError("cross_stage_reuse_resize must be nearest or trilinear")
        if cross_stage_branch_reuse not in {"none", "self", "mlp", "self_mlp"}:
            raise ValueError("cross_stage_branch_reuse must be none, self, mlp or self_mlp")
        branch_layers = frozenset(
            int(value) for value in (cross_stage_branch_reuse_layers or ())
        )
        branch_stages = frozenset(
            int(value) for value in (cross_stage_branch_reuse_stages or ())
        )
        if any(layer < 0 or layer >= len(self.model.blocks) for layer in branch_layers):
            raise ValueError("cross_stage_branch_reuse_layers contains an invalid layer")
        if any(stage <= 0 for stage in branch_stages):
            raise ValueError("cross_stage_branch_reuse_stages must be positive")
        if not math.isfinite(cross_stage_branch_reuse_scale):
            raise ValueError("cross_stage_branch_reuse_scale must be finite")
        self.attention_mode = attention_mode
        self.route_density = float(route_density)
        self.reuse_every = int(reuse_every)
        self.reuse_scope = reuse_scope
        self.small_stage_executor = small_stage_executor
        self.small_stage_max_padded_tokens = int(small_stage_max_padded_tokens)
        self.cross_stage_reuse_layers = reuse_layers
        self.cross_stage_reuse_layers_by_stage = reuse_layers_by_stage
        self.cross_stage_reuse_stages = reuse_stages
        self.cross_stage_reuse_scale = float(cross_stage_reuse_scale)
        self.cross_stage_reuse_scale_by_stage = scale_by_stage
        self.cross_stage_reuse_layer_coefficients = layer_coefficients
        self.cross_stage_reuse_layer_coefficients_by_stage = (
            layer_coefficients_by_stage
        )
        self.cross_stage_reuse_precompose = bool(cross_stage_reuse_precompose)
        self.cross_stage_reuse_resize = cross_stage_reuse_resize
        self.cross_stage_branch_reuse = cross_stage_branch_reuse
        self.cross_stage_branch_reuse_layers = branch_layers
        self.cross_stage_branch_reuse_stages = branch_stages
        self.cross_stage_branch_reuse_scale = float(cross_stage_branch_reuse_scale)
        self.reset()

    def reset(self) -> None:
        self.routes: dict[tuple[int, int], BlockRoute] = {}
        self.lifted_routes: dict[tuple[int, int, tuple[int, int, int]], BlockRoute] = {}
        self.output_cache: dict[tuple[tuple[int, int, int], int], torch.Tensor] = {}
        self.projected_output_cache: dict[
            tuple[tuple[int, int, int], int], torch.Tensor
        ] = {}
        self.cross_stage_residual_cache: dict[
            tuple[int, int], tuple[tuple[int, int, int], torch.Tensor]
        ] = {}
        self.cross_stage_precomposed_cache: dict[
            tuple[int, int], tuple[tuple[int, int, int], torch.Tensor]
        ] = {}
        self.cross_stage_self_cache: dict[
            tuple[int, int], tuple[tuple[int, int, int], torch.Tensor]
        ] = {}
        self.cross_stage_mlp_cache: dict[
            tuple[int, int], tuple[tuple[int, int, int], torch.Tensor]
        ] = {}
        # Geometry-only tensors survive per-video reset and are shared by all
        # layers.  Route values never live here.
        if not hasattr(self, "_lift_geometry_cache"):
            self._lift_geometry_cache: dict[
                tuple[tuple[int, int, int], tuple[int, int, int], torch.device],
                tuple[torch.Tensor, torch.Tensor, bool],
            ] = {}
        self.shape_steps: dict[tuple[int, int, int], int] = {}
        self.stats = {
            "online_vsa_calls": 0,
            "c2f_sparse_calls": 0,
            "c2f_fallback_calls": 0,
            "c2f_seed_vsa_calls": 0,
            "captured_routes": 0,
            "step_reuse_hits": 0,
            "step_reuse_recomputes": 0,
            "projected_reuse_hits": 0,
            "projected_reuse_recomputes": 0,
            "route_reuse_aliases": 0,
            "native_sparse_calls": 0,
            "masked_sdpa_sparse_calls": 0,
            "dense_sdpa_seed_calls": 0,
            "c2f_reused_coarse_calls": 0,
            "cross_stage_block_reuse_hits": 0,
            "cross_stage_block_reuse_captures": 0,
            "cross_stage_multiscale_predictor_hits": 0,
            "cross_stage_fine_highpass_hits": 0,
            "cross_stage_precompose_captures": 0,
            "cross_stage_precompose_hits": 0,
            "cross_stage_self_reuse_hits": 0,
            "cross_stage_self_captures": 0,
            "cross_stage_mlp_reuse_hits": 0,
            "cross_stage_mlp_captures": 0,
            "effective_density_sum": 0.0,
            "effective_density_count": 0,
        }

    def begin_model_forward(self, stage: int, shape: tuple[int, int, int]) -> None:
        shape = tuple(int(value) for value in shape)
        self.current_stage = int(stage)
        self.current_shape = shape
        self.current_shape_step = self.shape_steps.get(shape, 0)
        self.shape_steps[shape] = self.current_shape_step + 1

    def summary(self) -> dict[str, Any]:
        result = dict(self.stats)
        count = int(result.pop("effective_density_count"))
        total_raw = result.pop("effective_density_sum")
        total = float(total_raw.item()) if torch.is_tensor(total_raw) else float(total_raw)
        result.update(
            {
                "attention_mode": self.attention_mode,
                "route_density_requested": self.route_density,
                "reuse_every": self.reuse_every,
                "reuse_scope": self.reuse_scope,
                "small_stage_executor": self.small_stage_executor,
                "small_stage_max_padded_tokens": self.small_stage_max_padded_tokens,
                "cross_stage_reuse_layers": sorted(self.cross_stage_reuse_layers),
                "cross_stage_reuse_layers_by_stage": {
                    str(stage): sorted(layers)
                    for stage, layers in sorted(
                        self.cross_stage_reuse_layers_by_stage.items()
                    )
                },
                "cross_stage_reuse_stages": sorted(self.cross_stage_reuse_stages),
                "cross_stage_reuse_scale": self.cross_stage_reuse_scale,
                "cross_stage_reuse_scale_by_stage": {
                    str(stage): scale
                    for stage, scale in sorted(
                        self.cross_stage_reuse_scale_by_stage.items()
                    )
                },
                "cross_stage_reuse_layer_coefficients": {
                    str(layer): list(values)
                    for layer, values in sorted(
                        self.cross_stage_reuse_layer_coefficients.items()
                    )
                },
                "cross_stage_reuse_layer_coefficients_by_stage": {
                    str(stage): {
                        str(layer): list(values)
                        for layer, values in sorted(coefficients.items())
                    }
                    for stage, coefficients in sorted(
                        self.cross_stage_reuse_layer_coefficients_by_stage.items()
                    )
                },
                "cross_stage_reuse_precompose": self.cross_stage_reuse_precompose,
                "cross_stage_reuse_resize": self.cross_stage_reuse_resize,

                "cross_stage_branch_reuse": self.cross_stage_branch_reuse,
                "cross_stage_branch_reuse_layers": sorted(
                    self.cross_stage_branch_reuse_layers
                ),
                "cross_stage_branch_reuse_stages": sorted(
                    self.cross_stage_branch_reuse_stages
                ),
                "cross_stage_branch_reuse_scale": self.cross_stage_branch_reuse_scale,
                "mean_c2f_density": total / count if count else None,
                "route_records": len(self.routes),
                "shape_call_counts": {
                    "x".join(str(v) for v in shape): calls
                    for shape, calls in self.shape_steps.items()
                },
            }
        )
        return result

    def _token_grid(self, latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        patch = tuple(int(value) for value in self.model.patch_size)
        if any(size % stride for size, stride in zip(latent_shape, patch)):
            raise ValueError(
                f"latent shape {latent_shape} is not divisible by patch size {patch}"
            )
        return tuple(size // stride for size, stride in zip(latent_shape, patch))

    def _resize_sequence_between_stages(
        self,
        sequence: torch.Tensor,
        source_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Resize a flattened T/H/W block residual without mixing channels.

        Wan flattens the post-patch grid in T-major, then H, then W order.
        Sequence parallelism is one in these experiments, so the flattened
        length is exactly the product of the token grid.
        """
        source_grid = self._token_grid(source_shape)
        target_grid = self._token_grid(target_shape)
        if sequence.dim() != 3:
            raise ValueError(f"expected [B,N,C] residual, got {tuple(sequence.shape)}")
        batch, tokens, channels = sequence.shape
        if tokens != math.prod(source_grid):
            raise ValueError(
                f"residual tokens {tokens} do not match source token grid {source_grid}"
            )
        volume = sequence.view(batch, *source_grid, channels).permute(0, 4, 1, 2, 3)
        if self.cross_stage_reuse_resize == "nearest":
            resized = functional.interpolate(volume, size=target_grid, mode="nearest")
        else:
            resized = functional.interpolate(
                volume.float(), size=target_grid, mode="trilinear", align_corners=False
            ).to(sequence.dtype)
        return resized.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels).contiguous()

    def _project_sequence_to_stage_low(
        self,
        sequence: torch.Tensor,
        coarse_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Average inside coarse cells and lift back to the target token grid."""
        coarse_grid = self._token_grid(coarse_shape)
        target_grid = self._token_grid(target_shape)
        if any(
            fine % coarse
            for fine, coarse in zip(target_grid, coarse_grid)
        ):
            raise ValueError(
                f"non-nested token grids: {coarse_grid} -> {target_grid}"
            )
        factors = tuple(
            fine // coarse for fine, coarse in zip(target_grid, coarse_grid)
        )
        batch, tokens, channels = sequence.shape
        if tokens != math.prod(target_grid):
            raise ValueError(
                f"sequence tokens {tokens} do not match target grid {target_grid}"
            )
        volume = sequence.view(
            batch, *target_grid, channels
        ).permute(0, 4, 1, 2, 3)
        low = functional.avg_pool3d(
            volume.float(), kernel_size=factors, stride=factors
        )
        low = functional.interpolate(low, size=target_grid, mode="nearest")
        return low.to(sequence.dtype).permute(
            0, 2, 3, 4, 1
        ).reshape(batch, -1, channels).contiguous()

    def _reuse_layers_for_stage(self, stage: int) -> frozenset[int]:
        return self.cross_stage_reuse_layers | self.cross_stage_reuse_layers_by_stage.get(
            stage, frozenset()
        )

    def _reuse_coefficients_for_stage(
        self, stage: int, layer: int
    ) -> tuple[float, float, float] | None:
        stage_coefficients = self.cross_stage_reuse_layer_coefficients_by_stage.get(
            stage, {}
        )
        return stage_coefficients.get(
            layer, self.cross_stage_reuse_layer_coefficients.get(layer)
        )

    def _should_capture_cross_stage_residual(self, layer: int) -> bool:
        return any(
            stage > self.current_stage
            and layer in self._reuse_layers_for_stage(stage)
            for stage in self.cross_stage_reuse_stages
        )

    def _cross_stage_reuse_enabled(self, layer: int) -> bool:
        return (
            layer in self._reuse_layers_for_stage(self.current_stage)
            and self.current_stage in self.cross_stage_reuse_stages
            and (self.current_stage - 1, layer) in self.cross_stage_residual_cache
        )

    def _maybe_precompose_cross_stage_residual(
        self,
        layer: int,
        source_shape: tuple[int, int, int],
        source_delta: torch.Tensor,
    ) -> None:
        """Compose the next stage's linear predictor at the current resolution.

        Nearest-neighbour lifting is linear, so

            U(a M + b (M - U(C)))

        is algebraically identical to

            a U(M) + b (U(M) - U(C)).

        The first form performs the two-source blend on the smaller middle-stage
        tensor and needs only one lift in the expensive fine stage.  This cache
        is opt-in and does not alter legacy CSBR configurations.
        """
        if not self.cross_stage_reuse_precompose:
            return
        target_stage = self.current_stage + 1
        if target_stage not in self.cross_stage_reuse_stages:
            return
        coefficients = self._reuse_coefficients_for_stage(target_stage, layer)
        if coefficients is None:
            return
        middle_scale, trend_scale, _ = coefficients
        prepared = middle_scale * source_delta
        if abs(trend_scale) > 0.0:
            coarse_key = (self.current_stage - 1, layer)
            if coarse_key not in self.cross_stage_residual_cache:
                return
            coarse_shape, coarse_delta = self.cross_stage_residual_cache[coarse_key]
            coarse_at_source = self._resize_sequence_between_stages(
                coarse_delta, coarse_shape, source_shape
            )
            prepared = prepared + trend_scale * (source_delta - coarse_at_source)
        self.cross_stage_precomposed_cache[(self.current_stage, layer)] = (
            source_shape,
            prepared.detach(),
        )
        self.stats["cross_stage_precompose_captures"] += 1

    def _branch_reuse_enabled(self, layer: int, branch: str) -> bool:
        if layer not in self.cross_stage_branch_reuse_layers:
            return False
        if self.current_stage not in self.cross_stage_branch_reuse_stages:
            return False
        if branch == "self":
            enabled = self.cross_stage_branch_reuse in {"self", "self_mlp"}
            cache = self.cross_stage_self_cache
        else:
            enabled = self.cross_stage_branch_reuse in {"mlp", "self_mlp"}
            cache = self.cross_stage_mlp_cache
        return enabled and (self.current_stage - 1, layer) in cache

    def _block_forward_cross_stage_branches(
        self,
        layer: int,
        block: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        """Reuse coarse self-attention and/or MLP residual, but keep cross-attention.

        This is less aggressive than replacing a whole fine block: prompt
        cross-attention is always recomputed from the current fine hidden state.
        """
        original_owner = getattr(self.original_block_forwards[layer], "__self__", None)
        if original_owner is not None:
            block = original_owner
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        original_dtype = hidden_states.dtype
        modulation = block.scale_shift_table + temb.float()
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = modulation.chunk(6, dim=1)

        if self._branch_reuse_enabled(layer, "self"):
            source_shape, source_self = self.cross_stage_self_cache[
                (self.current_stage - 1, layer)
            ]
            projected_attention = self._resize_sequence_between_stages(
                source_self, source_shape, self.current_shape
            ) * self.cross_stage_branch_reuse_scale
            self.stats["cross_stage_self_reuse_hits"] += 1
        else:
            norm_hidden = (
                block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
            ).to(original_dtype)
            query, _ = block.to_q(norm_hidden)
            key, _ = block.to_k(norm_hidden)
            value, _ = block.to_v(norm_hidden)
            gate_compress, _ = block.to_gate_compress(norm_hidden)
            if block.norm_q is not None:
                query = block.norm_q(query)
            if block.norm_k is not None:
                key = block.norm_k(key)
            query = query.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            key = key.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            value = value.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            gate_compress = gate_compress.squeeze(1).unflatten(
                2, (block.num_attention_heads, -1)
            )
            attention, _ = block.attn1(
                query,
                key,
                value,
                original_seq_len,
                freqs_cis=freqs_cis,
                gate_compress=gate_compress,
            )
            attention = attention.flatten(2)
            attention, _ = block.to_out(attention)
            projected_attention = attention.squeeze(1)
            self.stats["cross_stage_self_captures"] += 1
        self.cross_stage_self_cache[(self.current_stage, layer)] = (
            self.current_shape,
            projected_attention.detach(),
        )

        zero = hidden_states.new_zeros(1)
        norm_hidden, hidden_states = block.self_attn_residual_norm(
            hidden_states, projected_attention, gate_msa, zero, zero
        )
        norm_hidden = norm_hidden.to(original_dtype)
        hidden_states = hidden_states.to(original_dtype)
        attention = block.attn2(
            norm_hidden, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden, hidden_states = block.cross_attn_residual_norm(
            hidden_states, attention, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden = norm_hidden.to(original_dtype)
        hidden_states = hidden_states.to(original_dtype)

        if self._branch_reuse_enabled(layer, "mlp"):
            source_shape, source_delta = self.cross_stage_mlp_cache[
                (self.current_stage - 1, layer)
            ]
            mlp_delta = self._resize_sequence_between_stages(
                source_delta, source_shape, self.current_shape
            ) * self.cross_stage_branch_reuse_scale
            output = hidden_states + mlp_delta
            self.stats["cross_stage_mlp_reuse_hits"] += 1
        else:
            ff_output = block.ffn(norm_hidden)
            output = block.mlp_residual(hidden_states, ff_output, c_gate_msa)
            mlp_delta = output - hidden_states
            self.stats["cross_stage_mlp_captures"] += 1
        self.cross_stage_mlp_cache[(self.current_stage, layer)] = (
            self.current_shape,
            mlp_delta.detach(),
        )
        return output.to(original_dtype)

    def _run_block_without_cross_stage_reuse(
        self,
        layer: int,
        block: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        if self.reuse_scope == "projected" and self.reuse_every > 1:
            return self._block_forward_projected_cache(
                layer,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                freqs_cis,
                original_seq_len,
            )
        return self.original_block_forwards[layer](
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cis,
            original_seq_len,
        )

    def _dispatch_block(
        self,
        layer: int,
        block: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        """Optionally replace a fine-stage block by an upsampled coarse residual.

        For a normal coarse call, cache Δ_l = h_{l+1}-h_l.  At a selected
        higher-resolution stage, apply h_l + s*U(Δ_l) and skip the whole block.
        This preserves the current fine hidden state and transfers only the
        transformation learned at the lower resolution.  It is an approximate
        architecture and requires finetuning for paper-quality use.
        """
        squeezed = hidden_states.squeeze(1) if hidden_states.dim() == 4 else hidden_states
        if self._cross_stage_reuse_enabled(layer):
            source_shape, source_delta = self.cross_stage_residual_cache[
                (self.current_stage - 1, layer)
            ]
            coefficients = self._reuse_coefficients_for_stage(
                self.current_stage, layer
            )
            precomposed = self.cross_stage_precomposed_cache.get(
                (self.current_stage - 1, layer)
            )
            if precomposed is not None:
                prepared_shape, prepared_delta = precomposed
                applied_delta = self._resize_sequence_between_stages(
                    prepared_delta, prepared_shape, self.current_shape
                )
                self.stats["cross_stage_precompose_hits"] += 1
            else:
                lifted_delta = self._resize_sequence_between_stages(
                    source_delta, source_shape, self.current_shape
                )
            if precomposed is None and coefficients is None:
                applied_delta = self.cross_stage_reuse_scale_by_stage.get(
                    self.current_stage, self.cross_stage_reuse_scale
                ) * lifted_delta
            elif precomposed is None:
                middle_scale, trend_scale, fine_high_scale = coefficients
                applied_delta = middle_scale * lifted_delta
                if abs(trend_scale) > 0.0:
                    coarse_key = (self.current_stage - 2, layer)
                    if coarse_key not in self.cross_stage_residual_cache:
                        raise RuntimeError(
                            f"missing two-stage residual cache for layer {layer}"
                        )
                    coarse_shape, coarse_delta = self.cross_stage_residual_cache[
                        coarse_key
                    ]
                    lifted_coarse = self._resize_sequence_between_stages(
                        coarse_delta, coarse_shape, self.current_shape
                    )
                    applied_delta = applied_delta + trend_scale * (
                        lifted_delta - lifted_coarse
                    )
                    self.stats["cross_stage_multiscale_predictor_hits"] += 1
            if coefficients is not None:
                _, _, fine_high_scale = coefficients
                if abs(fine_high_scale) > 0.0:
                    fine_low = self._project_sequence_to_stage_low(
                        squeezed, source_shape, self.current_shape
                    )
                    applied_delta = applied_delta + fine_high_scale * (
                        squeezed - fine_low
                    )
                    self.stats["cross_stage_fine_highpass_hits"] += 1
            output = squeezed + applied_delta
            self.cross_stage_residual_cache[(self.current_stage, layer)] = (
                self.current_shape,
                applied_delta.detach(),
            )
            self._maybe_precompose_cross_stage_residual(
                layer, self.current_shape, applied_delta
            )
            previous_route = self.routes.get((self.current_stage - 1, layer))
            if self.attention_mode.startswith("c2f_route") and previous_route is not None:
                self.routes[(self.current_stage, layer)] = previous_route
                self.stats["route_reuse_aliases"] += 1
            self.stats["cross_stage_block_reuse_hits"] += 1
            return output

        if layer in self.cross_stage_branch_reuse_layers:
            return self._block_forward_cross_stage_branches(
                layer,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                freqs_cis,
                original_seq_len,
            )

        output = self._run_block_without_cross_stage_reuse(
            layer,
            block,
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cis,
            original_seq_len,
        )
        if self._should_capture_cross_stage_residual(layer):
            output_squeezed = output.squeeze(1) if output.dim() == 4 else output
            if output_squeezed.shape != squeezed.shape:
                raise ValueError(
                    "cannot cache cross-stage residual with mismatched block shapes: "
                    f"input={tuple(squeezed.shape)}, output={tuple(output_squeezed.shape)}"
                )
            self.cross_stage_residual_cache[(self.current_stage, layer)] = (
                self.current_shape,
                (output_squeezed - squeezed).detach(),
            )
            self._maybe_precompose_cross_stage_residual(
                layer,
                self.current_shape,
                output_squeezed - squeezed,
            )
            self.stats["cross_stage_block_reuse_captures"] += 1
        return output

    def _should_reuse(self, layer: int) -> bool:
        key = (self.current_shape, layer)
        return (
            self.reuse_scope == "backend"
            and
            self.reuse_every > 1
            and self.current_shape_step % self.reuse_every != 0
            and key in self.output_cache
        )

    def _should_reuse_projected(self, layer: int) -> bool:
        key = (self.current_shape, layer)
        return (
            self.reuse_scope == "projected"
            and self.reuse_every > 1
            and self.current_shape_step % self.reuse_every != 0
            and key in self.projected_output_cache
        )

    def _block_forward_projected_cache(
        self,
        layer: int,
        block: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        """Wan VSA block with an exact post-projection self-attention cache.

        On a cache hit only the self-attention branch is reused.  Current-step
        modulation gates, the residual/norm update, cross-attention and MLP are
        still evaluated.  This matches the old backend-output cache exactly:
        ``to_out`` is an inference-time fixed affine map, so applying it again
        to the same cached backend output cannot change the result.
        """
        # FastVideo's loader may replace entries in ``model.blocks`` while
        # retaining the originally bound forward target.  Use the exact module
        # that the known-good original bound method would execute.
        original_owner = getattr(
            self.original_block_forwards[layer], "__self__", None
        )
        if original_owner is not None:
            block = original_owner
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype
        try:
            modulation = block.scale_shift_table + temb.float()
        except RuntimeError as error:
            raise RuntimeError(
                f"projected-cache modulation mismatch at layer {layer}: "
                f"scale_shift_table={tuple(block.scale_shift_table.shape)}, "
                f"temb={tuple(temb.shape)}, block={type(block).__name__}"
            ) from error
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = modulation.chunk(6, dim=1)
        cache_key = (self.current_shape, layer)

        if self._should_reuse_projected(layer):
            projected_attention = self.projected_output_cache[cache_key]
            self.stats["projected_reuse_hits"] += 1
            previous_route = self.routes.get((self.current_stage - 1, layer))
            if self.attention_mode.startswith("c2f_route") and previous_route is not None:
                self.routes[(self.current_stage, layer)] = previous_route
                self.stats["route_reuse_aliases"] += 1
        else:
            norm_hidden_states = (
                block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
            ).to(orig_dtype)
            query, _ = block.to_q(norm_hidden_states)
            key, _ = block.to_k(norm_hidden_states)
            value, _ = block.to_v(norm_hidden_states)
            gate_compress, _ = block.to_gate_compress(norm_hidden_states)

            if block.norm_q is not None:
                query = block.norm_q(query)
            if block.norm_k is not None:
                key = block.norm_k(key)

            query = query.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            key = key.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            value = value.squeeze(1).unflatten(2, (block.num_attention_heads, -1))
            gate_compress = gate_compress.squeeze(1).unflatten(
                2, (block.num_attention_heads, -1)
            )
            attention, _ = block.attn1(
                query,
                key,
                value,
                original_seq_len,
                freqs_cis=freqs_cis,
                gate_compress=gate_compress,
            )
            attention = attention.flatten(2)
            attention, _ = block.to_out(attention)
            projected_attention = attention.squeeze(1)
            self.projected_output_cache[cache_key] = projected_attention.detach()
            self.stats["projected_reuse_recomputes"] += 1

        zero = hidden_states.new_zeros(1)
        norm_hidden_states, hidden_states = block.self_attn_residual_norm(
            hidden_states,
            projected_attention,
            gate_msa,
            zero,
            zero,
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        attention = block.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
        )
        norm_hidden_states, hidden_states = block.cross_attn_residual_norm(
            hidden_states,
            attention,
            1,
            c_shift_msa,
            c_scale_msa,
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)

        ff_output = block.ffn(norm_hidden_states)
        hidden_states = block.mlp_residual(hidden_states, ff_output, c_gate_msa)
        return hidden_states.to(orig_dtype)

    @staticmethod
    def _centroids(
        x: torch.Tensor, variable_sizes: torch.Tensor
    ) -> torch.Tensor:
        # Input is tiled [B,H,num_blocks*64,D].
        batch, heads, sequence, dim = x.shape
        blocks = int(variable_sizes.numel())
        block = sequence // blocks
        denom = variable_sizes.to(torch.float32).clamp_min(1).view(1, 1, blocks, 1)
        return (
            x.view(batch, heads, blocks, block, dim).float().sum(dim=3) / denom
        ).to(x.dtype)

    def _capture_route(
        self,
        layer: int,
        scores: torch.Tensor,
        block_grid: tuple[int, int, int],
    ) -> BlockRoute:
        blocks = scores.shape[-1]
        keep = max(1, min(blocks, math.ceil(self.route_density * blocks)))
        indices = scores.topk(keep, dim=-1, sorted=True).indices
        self_index = torch.arange(blocks, device=scores.device).view(1, 1, blocks, 1)
        missing = ~(indices == self_index).any(dim=-1, keepdim=True)
        indices[..., -1:] = torch.where(missing, self_index, indices[..., -1:])
        indices = indices.to(torch.int32).contiguous()
        counts = torch.full(
            indices.shape[:-1], keep, dtype=torch.int32, device=scores.device
        )
        probabilities = torch.softmax(scores, dim=-1).detach()
        route = BlockRoute(
            block_grid,
            indices.detach(),
            counts.detach(),
            probabilities=probabilities,
            probability_grid=block_grid,
        )
        self.routes[(self.current_stage, layer)] = route
        self.stats["captured_routes"] += 1
        return route

    @staticmethod
    def _parent_children(
        coarse: tuple[int, int, int], fine: tuple[int, int, int]
    ) -> tuple[list[int], list[list[int]]]:
        children: list[list[int]] = [[] for _ in range(math.prod(coarse))]
        parents: list[int] = []
        for t in range(fine[0]):
            for h in range(fine[1]):
                for w in range(fine[2]):
                    parent_coord = (
                        min(coarse[0] - 1, t * coarse[0] // fine[0]),
                        min(coarse[1] - 1, h * coarse[1] // fine[1]),
                        min(coarse[2] - 1, w * coarse[2] // fine[2]),
                    )
                    parent = _flat(parent_coord, coarse)
                    child = _flat((t, h, w), fine)
                    parents.append(parent)
                    children[parent].append(child)
        return parents, children

    def _lift_route(
        self,
        layer: int,
        source: BlockRoute,
        target_grid: tuple[int, int, int],
    ) -> BlockRoute:
        cache_key = (self.current_stage, layer, target_grid)
        cached = self.lifted_routes.get(cache_key)
        if cached is not None:
            return cached
        if source.indices.is_cuda:
            try:
                from .rdd_c2f_route_triton import lift_c2f_route
            except (ImportError, ModuleNotFoundError):
                lift_c2f_route = None
            if lift_c2f_route is not None:
                indices, counts = lift_c2f_route(
                    source.indices,
                    source.counts,
                    source.block_grid,
                    target_grid,
                )
                route = BlockRoute(
                    target_grid,
                    indices.contiguous(),
                    counts.contiguous(),
                    probabilities=source.probabilities,
                    probability_grid=source.probability_grid,
                )
                self.lifted_routes[cache_key] = route
                return route
        coarse = source.block_grid
        geometry_key = (coarse, target_grid, source.indices.device)
        geometry = self._lift_geometry_cache.get(geometry_key)
        if geometry is None:
            device = source.indices.device
            coarse_count = math.prod(coarse)
            fine_count = math.prod(target_grid)
            exact_nested = all(
                fine % low == 0 for low, fine in zip(coarse, target_grid)
            )

            fine_ids = torch.arange(fine_count, device=device, dtype=torch.long)
            fine_t = fine_ids // (target_grid[1] * target_grid[2])
            fine_h = (fine_ids // target_grid[2]) % target_grid[1]
            fine_w = fine_ids % target_grid[2]
            parent_for_fine = (
                (fine_t * coarse[0] // target_grid[0]) * coarse[1]
                + (fine_h * coarse[1] // target_grid[1])
            ) * coarse[2] + (fine_w * coarse[2] // target_grid[2])

            coarse_ids = torch.arange(coarse_count, device=device, dtype=torch.long)
            coarse_t = coarse_ids // (coarse[1] * coarse[2])
            coarse_h = (coarse_ids // coarse[2]) % coarse[1]
            coarse_w = coarse_ids % coarse[2]
            if exact_nested:
                ratios = tuple(
                    fine // low for low, fine in zip(coarse, target_grid)
                )
                child_count = math.prod(ratios)
                offsets = torch.arange(child_count, device=device, dtype=torch.long)
                dt = offsets // (ratios[1] * ratios[2])
                dh = (offsets // ratios[2]) % ratios[1]
                dw = offsets % ratios[2]
                child_t = coarse_t[:, None] * ratios[0] + dt[None, :]
                child_h = coarse_h[:, None] * ratios[1] + dh[None, :]
                child_w = coarse_w[:, None] * ratios[2] + dw[None, :]
                children_for_parent = (
                    (child_t * target_grid[1] + child_h) * target_grid[2]
                    + child_w
                )
            else:
                # Ceil-created VSA grids such as (1,2,4)->(1,4,7) are nested
                # geometrically but do not have a constant child count. Build
                # a padded child table once on GPU; route values never leave
                # the device.
                max_children = math.prod(
                    math.ceil(fine / low)
                    for low, fine in zip(coarse, target_grid)
                )
                children_for_parent = torch.full(
                    (coarse_count, max_children),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                parent_t = fine_t * coarse[0] // target_grid[0]
                parent_h = fine_h * coarse[1] // target_grid[1]
                parent_w = fine_w * coarse[2] // target_grid[2]
                start_t = (
                    parent_t * target_grid[0] + coarse[0] - 1
                ) // coarse[0]
                start_h = (
                    parent_h * target_grid[1] + coarse[1] - 1
                ) // coarse[1]
                start_w = (
                    parent_w * target_grid[2] + coarse[2] - 1
                ) // coarse[2]
                end_h = (
                    (parent_h + 1) * target_grid[1] + coarse[1] - 1
                ) // coarse[1]
                end_w = (
                    (parent_w + 1) * target_grid[2] + coarse[2] - 1
                ) // coarse[2]
                size_h = end_h - start_h
                size_w = end_w - start_w
                child_slot = (
                    ((fine_t - start_t) * size_h + (fine_h - start_h))
                    * size_w
                    + (fine_w - start_w)
                )
                children_for_parent[parent_for_fine, child_slot] = fine_ids
            geometry = (parent_for_fine, children_for_parent, exact_nested)
            self._lift_geometry_cache[geometry_key] = geometry

        parent_for_fine, children_for_parent, exact_nested = geometry
        batch, heads, _, source_slots = source.indices.shape
        fine_count = int(parent_for_fine.numel())
        max_child_count = int(children_for_parent.shape[1])

        parent_rows = parent_for_fine.view(1, 1, fine_count, 1).expand(
            batch, heads, -1, source_slots
        )
        selected_parents = torch.gather(
            source.indices.to(torch.long), 2, parent_rows
        )
        selected_children = children_for_parent[selected_parents]
        selected_source_counts = torch.gather(
            source.counts.to(torch.int32),
            2,
            parent_for_fine.view(1, 1, -1).expand(batch, heads, -1),
        )
        if exact_nested:
            indices = selected_children.flatten(-2).to(torch.int32).contiguous()
            counts = selected_source_counts * max_child_count
        else:
            source_slot = torch.arange(source_slots, device=source.indices.device)
            source_valid = source_slot.view(1, 1, 1, -1, 1) < (
                selected_source_counts.unsqueeze(-1).unsqueeze(-1)
            )
            child_valid = selected_children >= 0
            valid_children = source_valid & child_valid
            flat_children = selected_children.flatten(-2)
            flat_valid = valid_children.flatten(-2)
            sentinel = torch.full_like(flat_children, fine_count)
            # Softmax is invariant to selected-key ordering. Sorting moves all
            # padded entries behind the valid prefix without CPU loops/sync.
            indices = torch.where(flat_valid, flat_children, sentinel).sort(
                dim=-1
            ).values.to(torch.int32)
            counts = flat_valid.sum(dim=-1, dtype=torch.int32)

        # `_capture_route` always includes the coarse self block, therefore
        # each fine query already occurs among its parent's children.  Keep a
        # defensive fully-GPU repair for externally supplied routes.
        slots = torch.arange(indices.shape[-1], device=indices.device)
        valid = slots.view(1, 1, 1, -1) < counts.unsqueeze(-1)
        query_ids = torch.arange(fine_count, device=indices.device, dtype=torch.int32)
        has_self = ((indices == query_ids.view(1, 1, -1, 1)) & valid).any(dim=-1)
        repaired = torch.zeros(
            (*indices.shape[:-1], indices.shape[-1] + 1),
            dtype=torch.int32,
            device=indices.device,
        )
        repaired[..., :-1] = indices
        repaired.scatter_(
            -1,
            counts.to(torch.long).unsqueeze(-1),
            query_ids.view(1, 1, -1, 1).expand(batch, heads, -1, -1),
        )
        counts = counts + (~has_self).to(torch.int32)
        indices = repaired
        route = BlockRoute(
            target_grid,
            indices.contiguous(),
            counts.contiguous(),
            probabilities=source.probabilities,
            probability_grid=source.probability_grid,
        )
        self.lifted_routes[cache_key] = route
        return route

    @staticmethod
    def _parent_for_fine(
        coarse: tuple[int, int, int],
        fine: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        fine_count = math.prod(fine)
        fine_ids = torch.arange(fine_count, device=device, dtype=torch.long)
        fine_t = fine_ids // (fine[1] * fine[2])
        fine_h = (fine_ids // fine[2]) % fine[1]
        fine_w = fine_ids % fine[2]
        return (
            (fine_t * coarse[0] // fine[0]) * coarse[1]
            + (fine_h * coarse[1] // fine[1])
        ) * coarse[2] + (fine_w * coarse[2] // fine[2])

    def _reuse_seed_coarse_output(
        self,
        source: BlockRoute,
        value_h: torch.Tensor,
        variable_sizes: torch.Tensor,
        target_grid: tuple[int, int, int],
    ) -> torch.Tensor | None:
        """Apply seed-stage block probabilities to current fine-stage values.

        Current V blocks are pooled back to the seed block grid, multiplied by
        the cached seed probability matrix, then broadcast to the seed block's
        fine children.  No current-resolution QcKc matrix is formed.
        """
        probabilities = source.probabilities
        coarse_grid = source.probability_grid
        if probabilities is None or coarse_grid is None:
            return None
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (
            fused_block_mean,
        )

        target_blocks = math.prod(target_grid)
        block_elements = value_h.shape[2] // target_blocks
        value_center = fused_block_mean(
            value_h, variable_sizes, block_elements
        ).float()
        parent = self._parent_for_fine(coarse_grid, target_grid, value_h.device)
        coarse_blocks = math.prod(coarse_grid)
        weights = variable_sizes.to(torch.float32)
        batch, heads, _, dim = value_center.shape
        pooled = torch.zeros(
            (batch, heads, coarse_blocks, dim),
            device=value_h.device,
            dtype=torch.float32,
        )
        parent_expanded = parent.view(1, 1, -1, 1).expand(batch, heads, -1, dim)
        pooled.scatter_add_(
            2,
            parent_expanded,
            value_center * weights.view(1, 1, -1, 1),
        )
        denominator = torch.zeros(
            coarse_blocks, device=value_h.device, dtype=torch.float32
        )
        denominator.scatter_add_(0, parent, weights)
        pooled = pooled / denominator.clamp_min(1).view(1, 1, -1, 1)
        coarse_output = torch.matmul(probabilities.float(), pooled)
        fine_output = coarse_output[:, :, parent, :].to(value_h.dtype)
        return (
            fine_output.unsqueeze(3)
            .expand(-1, -1, -1, block_elements, -1)
            .reshape_as(value_h)
        )

    def _execute_sparse(
        self,
        query_h: torch.Tensor,
        key_h: torch.Tensor,
        value_h: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        variable_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Execute exactly the supplied block route with a shape-aware backend."""
        padded_tokens = int(query_h.shape[2])
        use_alternative = (
            self.small_stage_executor != "native"
            and self.small_stage_max_padded_tokens > 0
            and padded_tokens <= self.small_stage_max_padded_tokens
        )
        if not use_alternative:
            from fastvideo_kernel.block_sparse_attn import (
                block_sparse_attn_from_indices,
            )

            self.stats["native_sparse_calls"] += 1
            return block_sparse_attn_from_indices(
                query_h,
                key_h,
                value_h,
                indices,
                counts,
                variable_sizes,
            )[0]

        if self.small_stage_executor in {"triton_q16", "triton_q32"}:
            from attention_analysis.rdd_sparse_attn_smallq_triton import (
                smallq_block_sparse_attention,
            )

            return smallq_block_sparse_attention(
                query_h,
                key_h,
                value_h,
                indices,
                counts,
                variable_sizes,
                query_tile=16 if self.small_stage_executor == "triton_q16" else 32,
            )

        import torch.nn.functional as functional

        batch, heads, q_blocks, slots = indices.shape
        k_blocks = int(variable_sizes.numel())
        block_elements = padded_tokens // q_blocks
        if self.small_stage_executor == "dense_sdpa":
            block_ids = torch.arange(k_blocks, device=indices.device)[:, None]
            offsets = torch.arange(block_elements, device=indices.device)[None, :]
            valid_indices = (
                block_ids * block_elements + offsets
            )[offsets < variable_sizes[:, None]]
            query_valid = query_h[:, :, valid_indices]
            key_valid = key_h[:, :, valid_indices]
            value_valid = value_h[:, :, valid_indices]
            output_valid = functional.scaled_dot_product_attention(
                query_valid, key_valid, value_valid
            )
            output = torch.zeros_like(query_h)
            output[:, :, valid_indices] = output_valid
            self.stats["dense_sdpa_seed_calls"] += 1
            return output

        slot = torch.arange(slots, device=indices.device).view(1, 1, 1, -1)
        valid_slot = slot < counts.unsqueeze(-1)
        block_mask = torch.zeros(
            (batch, heads, q_blocks, k_blocks),
            device=indices.device,
            dtype=torch.bool,
        )
        # FastVideo pads every route row with a sentinel index equal to
        # ``k_blocks``.  The native kernel ignores those slots via ``counts``,
        # but torch.scatter still validates their indices even when ``src`` is
        # False.  Redirect invalid slots to block zero; the False value leaves
        # that entry unchanged and preserves the exact counted route.
        safe_indices = torch.where(
            valid_slot,
            indices.to(torch.long),
            torch.zeros_like(indices, dtype=torch.long),
        )
        block_mask.scatter_(-1, safe_indices, valid_slot)
        token_mask = block_mask.repeat_interleave(
            block_elements, dim=-2
        ).repeat_interleave(block_elements, dim=-1)
        valid_key = (
            torch.arange(block_elements, device=indices.device)[None, :]
            < variable_sizes[:, None]
        ).reshape(-1)
        token_mask &= valid_key.view(1, 1, 1, -1)
        self.stats["masked_sdpa_sparse_calls"] += 1
        return functional.scaled_dot_product_attention(
            query_h, key_h, value_h, attn_mask=token_mask
        )

    def _impl_forward(
        self,
        layer: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        cache_key = (self.current_shape, layer)
        if self._should_reuse(layer):
            self.stats["step_reuse_hits"] += 1
            previous_route = self.routes.get((self.current_stage - 1, layer))
            if self.attention_mode.startswith("c2f_route") and previous_route is not None:
                self.routes[(self.current_stage, layer)] = previous_route
                self.stats["route_reuse_aliases"] += 1
            return self.output_cache[cache_key]

        if self.attention_mode == "vsa":
            self.stats["online_vsa_calls"] += 1
            output = self.original_impl_forwards[layer](
                query, key, value, gate_compress, attn_metadata
            )
        else:
            from fastvideo_kernel.triton_kernels.fused_compress_topk import (
                fused_block_mean,
                fused_topk_mask,
            )
            from fastvideo_kernel.triton_kernels.index import map_to_index

            query_h = query.transpose(1, 2).contiguous()
            key_h = key.transpose(1, 2).contiguous()
            value_h = value.transpose(1, 2).contiguous()
            variable_sizes = attn_metadata.variable_block_sizes.to(torch.int32).contiguous()
            block_grid = tuple(int(value) for value in attn_metadata.num_tiles)
            source = self.routes.get((self.current_stage - 1, layer))
            route_only_ready = (
                self.attention_mode in {
                    "c2f_route_only",
                    "c2f_route_reuse_coarse",
                }
                and self.current_stage > 0
                and source is not None
            )

            if route_only_ready:
                route = self._lift_route(layer, source, block_grid)
                # The used fine route becomes the source for the next stage.
                # This gives a true 8->16->32 cascade without an online dense
                # block-score matrix at 16 or 32.
                self.routes[(self.current_stage, layer)] = route
                sparse_output = self._execute_sparse(
                    query_h,
                    key_h,
                    value_h,
                    route.indices,
                    route.counts,
                    variable_sizes,
                )
                if self.attention_mode == "c2f_route_reuse_coarse":
                    reused_coarse = self._reuse_seed_coarse_output(
                        source, value_h, variable_sizes, block_grid
                    )
                    if reused_coarse is not None:
                        gate_h = gate_compress.transpose(1, 2).contiguous()
                        sparse_output = sparse_output + reused_coarse * gate_h
                        self.stats["c2f_reused_coarse_calls"] += 1
                output = sparse_output.transpose(1, 2).contiguous()
                density = route.counts.float().mean() / math.prod(block_grid)
                self.stats["c2f_sparse_calls"] += 1
                self.stats["effective_density_sum"] += density
                self.stats["effective_density_count"] += 1
            else:
                block_elements = query_h.shape[2] // math.prod(block_grid)
                q_center = fused_block_mean(
                    query_h, variable_sizes, block_elements
                )
                k_center = fused_block_mean(
                    key_h, variable_sizes, block_elements
                )
                scores = torch.matmul(
                    q_center, k_center.transpose(-2, -1)
                ) / math.sqrt(query_h.shape[-1])
                self._capture_route(layer, scores, block_grid)

            if not route_only_ready and (self.current_stage == 0 or source is None):
                # Seed the cascade with the checkpoint's *exact* VSA equation,
                # but reuse the already-computed centroids/scores instead of
                # calling the original backend and paying for compression,
                # dense block scores, sparse attention and the coarse branch a
                # second time.  `_capture_route` above remains the sole C2F
                # route definition; this branch only builds the VSA output.
                blocks = math.prod(block_grid)
                current_topk = max(
                    1,
                    min(
                        blocks,
                        math.ceil((1.0 - float(attn_metadata.VSA_sparsity)) * blocks),
                    ),
                )
                mask = fused_topk_mask(scores, current_topk)
                current_indices, current_counts = map_to_index(mask)
                sparse_output = self._execute_sparse(
                    query_h,
                    key_h,
                    value_h,
                    current_indices,
                    current_counts,
                    variable_sizes,
                )
                v_center = fused_block_mean(
                    value_h, variable_sizes, block_elements
                )
                coarse_attention = self.routes[(self.current_stage, layer)].probabilities
                assert coarse_attention is not None
                coarse_output = torch.matmul(coarse_attention, v_center)
                coarse_output = (
                    coarse_output.unsqueeze(3)
                    .repeat(1, 1, 1, block_elements, 1)
                    .reshape_as(query_h)
                )
                gate_h = gate_compress.transpose(1, 2).contiguous()
                output = (
                    sparse_output + coarse_output * gate_h
                ).transpose(1, 2).contiguous()
                self.stats["c2f_seed_vsa_calls"] += 1
            elif not route_only_ready:
                route = self._lift_route(layer, source, block_grid)
                sparse_output = self._execute_sparse(
                    query_h,
                    key_h,
                    value_h,
                    route.indices,
                    route.counts,
                    variable_sizes,
                )
                v_center = fused_block_mean(
                    value_h, variable_sizes, block_elements
                )
                coarse_attention = torch.softmax(scores, dim=-1)
                coarse_output = torch.matmul(coarse_attention, v_center)
                coarse_output = (
                    coarse_output.unsqueeze(3)
                    .expand(-1, -1, -1, block_elements, -1)
                    .reshape_as(query_h)
                )
                gate_h = gate_compress.transpose(1, 2).contiguous()
                output = (
                    sparse_output + coarse_output * gate_h
                ).transpose(1, 2).contiguous()
                density = route.counts.float().mean() / math.prod(block_grid)
                self.stats["c2f_sparse_calls"] += 1
                self.stats["effective_density_sum"] += density
                self.stats["effective_density_count"] += 1

        if self.reuse_every > 1:
            self.output_cache[cache_key] = output.detach()
            self.stats["step_reuse_recomputes"] += 1
        return output
