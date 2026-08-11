"""Compatibility adjustments for DeepSeek V4 Flash 0731 checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v4 import Model as PinnedDeepseekV4Model
from mlx_lm.models.deepseek_v4 import ModelArgs
from mlx_lm.models.deepseek_v4 import MoEGate as PinnedMoEGate


def normalize_deepseek_v4_0731_weights(
    weights: dict[str, mx.array], args: ModelArgs
) -> dict[str, mx.array]:
    """Apply only the 0731 checkpoint deltas that precede base sanitization."""
    normalized: dict[str, mx.array] = {}
    expected_leading_shape = (args.o_groups, args.o_lora_rank)

    for path, value in weights.items():
        normalized_path = path.replace(".attn_hc.", ".hc_attn.").replace(
            ".ffn_hc.", ".hc_ffn."
        )
        if normalized_path in normalized:
            raise ValueError(
                "DeepSeek V4 0731 hyper-connection key collision at "
                f"{normalized_path!r}"
            )

        if ".attn.wo_a." in normalized_path and value.ndim == 3:
            observed_shape = tuple(value.shape)
            if observed_shape[:2] != expected_leading_shape:
                raise ValueError(
                    "DeepSeek V4 0731 grouped wo_a has unexpected shape for "
                    f"{normalized_path}: observed {observed_shape}, expected leading "
                    f"shape {expected_leading_shape}"
                )
            value = value.reshape(-1, observed_shape[-1])

        normalized[normalized_path] = value

    return normalized


class _GateState(Protocol):
    layer_id: int
    hash: bool
    top_k: int
    score_func: str
    route_scale: float
    norm_topk_prob: bool
    weight: mx.array
    tid2eid: mx.array
    e_score_correction_bias: mx.array


class _QuantizedLinearState(Protocol):
    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    bias: mx.array
    group_size: int
    bits: int
    mode: str

    def get(self, key: str, default: mx.array | None = None) -> mx.array | None: ...

    def __contains__(self, key: str) -> bool: ...


class _PrefillAttention(Protocol):
    wqkv_a: object
    q_lora_rank: int


class _PrefillLayer(Protocol):
    attn: _PrefillAttention


class _PrefillModelBody(Protocol):
    layers: Sequence[_PrefillLayer]


class _PrefillModel(Protocol):
    model: _PrefillModelBody


def _score_func(scores: mx.array, score_func: str) -> mx.array:
    if score_func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if score_func == "sigmoid":
        return mx.sigmoid(scores)
    return mx.sqrt(mx.logaddexp(scores, 0))


class SplitOutputQuantizedLinear(nn.QuantizedLinear):
    """Use separate quantized output-row projections for multi-token prefill."""

    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    bias: mx.array
    group_size: int
    bits: int
    mode: str
    split: int

    @classmethod
    def from_quantized_linear(
        cls, source: nn.QuantizedLinear, *, split: int
    ) -> SplitOutputQuantizedLinear:
        state = cast(_QuantizedLinearState, cast(object, source))
        output_dims = int(state.weight.shape[0])
        if split <= 0 or split >= output_dims:
            raise ValueError(f"split must be in (0, {output_dims}), got {split}")

        # Calling QuantizedLinear.__init__ allocates and quantizes a full random
        # matrix. Construct the frozen module directly so the loaded parameters
        # remain the only allocation and retain their original tree paths.
        result = cast(SplitOutputQuantizedLinear, cls.__new__(cls))
        nn.Module.__init__(result)
        result.weight = state.weight
        result.scales = state.scales
        result.biases = state.biases
        if "bias" in state:
            result.bias = state.bias
        result.group_size = state.group_size
        result.bits = state.bits
        result.mode = state.mode
        result.split = split
        result.freeze()
        return result

    def _project_rows(self, hidden: mx.array, start: int, end: int) -> mx.array:
        return mx.quantized_matmul(
            hidden,
            self.weight[start:end],
            scales=self.scales[start:end],
            biases=None if self.biases is None else self.biases[start:end],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        # Decode already matches the reference with the fused kernel.  The
        # prefill-only split preserves OMLX's separate Q/KV quantized calls.
        if hidden.ndim < 2 or hidden.shape[-2] == 1:
            return super().__call__(hidden)
        output = mx.concatenate(
            [
                self._project_rows(hidden, 0, self.split),
                self._project_rows(hidden, self.split, self.weight.shape[0]),
            ],
            axis=-1,
        )
        return output if "bias" not in self else output + self.bias


def install_deepseek_v4_0731_attention_prefill_split(
    attention: _PrefillAttention,
) -> None:
    """Install the idempotent prefill split only on a loaded 0731 attention."""
    current = attention.wqkv_a
    if isinstance(current, SplitOutputQuantizedLinear):
        return
    if not isinstance(current, nn.QuantizedLinear):
        raise TypeError(
            "0731 attention wqkv_a must be quantized before installing split prefill"
        )
    state = cast(_QuantizedLinearState, cast(object, current))
    if state.mode != "mxfp8":
        return
    attention.wqkv_a = SplitOutputQuantizedLinear.from_quantized_linear(
        current, split=int(attention.q_lora_rank)
    )


def install_deepseek_v4_0731_prefill_attention(model: DeepseekV40731Model) -> None:
    """Install the 0731-only split prefill path after MLX has loaded weights."""
    state = cast(_PrefillModel, cast(object, model))
    for layer in state.model.layers:
        install_deepseek_v4_0731_attention_prefill_split(layer.attn)


def install_deepseek_v4_sdpa_float32() -> None:
    """Run the DeepSeek V4 attention softmax in float32.

    The bfloat16 ``scaled_dot_product_attention`` Metal kernel is
    non-deterministic across M4 Max and M5 Max GPUs: on identical random
    (q, k, v) inputs the attention output differs by ~2.9 in max-abs (the
    QuantizedLinear matmuls feeding it are bitwise-identical across the two
    devices, so the softmax kernel itself is the source of the divergence).
    The attention softmax amplifies that into a >3x output error which,
    under tensor-parallel sharding across the two physical nodes, desyncs
    rank 0 (M5 Max) and rank 1 (M4 Max) -- rank 1 sees a different attention
    matrix than rank 0, corrupting every decoded token.

    Computing the softmax in float32 (standard numerical-stability practice)
    restores cross-device agreement to within FP32 reduction-order noise
    (~2.6e-3) -- well inside the mxfp8 parity tolerance -- and keeps each
    GPU's softmax deterministic run-to-run, so 2-rank decode is reproducible
    across repeated requests.

    Idempotent: a no-op if already installed.
    """
    import mlx_lm.models.deepseek_v4 as _dsv4

    if getattr(_dsv4, "_exo_sdpa_float32_patched", False):
        return
    _orig = cast("Callable[..., mx.array]", _dsv4.scaled_dot_product_attention)  # pyright: ignore[reportAttributeAccessIssue]

    def _sdpa_float32(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        cache: object,
        scale: float,
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        return _orig(
            queries.astype(mx.float32),
            keys.astype(mx.float32),
            values.astype(mx.float32),
            cache,
            scale,
            mask,
            sinks,
        ).astype(queries.dtype)

    # Rebind the real module global; a misspelled target is a silent no-op and
    # pyright additionally cannot see the mlx_lm re-exported symbol, so the
    # rebind is a deliberate dynamic override of the module namespace.
    _dsv4.scaled_dot_product_attention = _sdpa_float32  # pyright: ignore[reportAttributeAccessIssue]
    _dsv4._exo_sdpa_float32_patched = True  # pyright: ignore[reportAttributeAccessIssue]


def install_deepseek_v4_0731_float32_backbone() -> None:
    """Keep the hidden state float32 across every 0731 decoder block.

    This is an opt-in experiment for measuring deployed-path quality. It does
    not dequantize checkpoint weights; it only prevents bfloat16 state from
    being carried across block boundaries, matching the broad parity harness.
    """
    import mlx_lm.models.deepseek_v4 as _dsv4

    if getattr(_dsv4, "_exo_dsv4_float32_backbone_patched", False):
        return
    block_type = _dsv4.DeepseekV4Block
    original = cast("Callable[..., mx.array]", block_type.__call__)

    def _float32_block(
        self: object,
        hidden: mx.array,
        cache: object,
        input_ids: mx.array,
    ) -> mx.array:
        return original(self, hidden.astype(mx.float32), cache, input_ids).astype(
            mx.float32
        )

    block_type.__call__ = _float32_block  # type: ignore[method-assign]
    _dsv4._exo_dsv4_float32_backbone_patched = True  # pyright: ignore[reportAttributeAccessIssue]


class DeepseekV40731MoEGate(PinnedMoEGate):
    """0731 gate with OMLX-equivalent routing arithmetic and selection."""

    def __call__(
        self, x: mx.array, input_ids: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        state = cast(_GateState, cast(object, self))
        logits = (x @ state.weight.T).astype(mx.float32)
        scores = _score_func(logits, state.score_func)
        if state.hash:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash routing requires input_ids.")
            indices = state.tid2eid[input_ids.reshape(-1)].reshape(
                *x.shape[:-1], state.top_k
            )
        else:
            biased_scores = scores + state.e_score_correction_bias
            indices = mx.stop_gradient(
                mx.argpartition(-biased_scores, kth=state.top_k - 1, axis=-1)[
                    ..., : state.top_k
                ]
            )
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if state.score_func != "softmax" and state.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, weights * state.route_scale


class DeepseekV40731Model(PinnedDeepseekV4Model):
    """Pinned DeepSeek V4 model with the verified 0731 checkpoint deltas."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        for layer in self.model.layers:
            layer.ffn.shared_experts.swiglu_limit = args.swiglu_limit
            pinned_gate = cast(_GateState, cast(object, layer.ffn.gate))
            gate = DeepseekV40731MoEGate(args, pinned_gate.layer_id)
            gate.weight = pinned_gate.weight
            if pinned_gate.hash:
                gate.tid2eid = pinned_gate.tid2eid
            else:
                gate.e_score_correction_bias = pinned_gate.e_score_correction_bias
            layer.ffn.gate = gate

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        normalized = normalize_deepseek_v4_0731_weights(weights, self.args)
        return super().sanitize(normalized)
