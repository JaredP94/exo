from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.deepseek_v4 import Model as PinnedDeepseekV4Model
from mlx_lm.models.deepseek_v4 import ModelArgs

from exo.worker.engines.mlx.deepseek_v4_0731_config import QuantizationSpec
from exo.worker.engines.mlx.deepseek_v4_0731_model import (
    DeepseekV40731Model,
    DeepseekV40731MoEGate,
    SplitOutputQuantizedLinear,
    install_deepseek_v4_0731_prefill_attention,
    normalize_deepseek_v4_0731_weights,
)


def _args(*, swiglu_limit: float = 10.0) -> ModelArgs:
    return ModelArgs(  # type: ignore[reportCallIssue]
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        q_lora_rank=8,
        o_lora_rank=1024,
        o_groups=8,
        head_dim=4,
        qk_rope_head_dim=2,
        moe_intermediate_size=32,
        n_routed_experts=1,
        n_shared_experts=1,
        num_experts_per_tok=1,
        hc_mult=4,
        swiglu_limit=swiglu_limit,
    )


class _Linear(Protocol):
    weight: mx.array


class _SharedExpert(Protocol):
    swiglu_limit: float
    gate_proj: _Linear
    up_proj: _Linear
    down_proj: _Linear

    def __call__(self, x: mx.array) -> mx.array: ...


class _RoutingArgs(Protocol):
    hidden_size: int
    num_hidden_layers: int
    n_routed_experts: int
    num_experts_per_tok: int
    num_hash_layers: int
    routed_scaling_factor: float


class _HashGate(Protocol):
    weight: mx.array
    tid2eid: mx.array

    def __call__(self, x: mx.array, input_ids: mx.array) -> tuple[mx.array, mx.array]: ...


class _QuantizedLinearState(Protocol):
    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    bias: mx.array | None
    group_size: int
    bits: int
    mode: str

    def __call__(self, hidden: mx.array) -> mx.array: ...

    def to_quantized(
        self, *, group_size: int, bits: int, mode: str
    ) -> nn.QuantizedLinear: ...

    def get(self, key: str) -> object: ...


class _QuantizedMatmul(Protocol):
    def __call__(
        self,
        x: mx.array,
        weight: mx.array,
        *,
        scales: mx.array,
        biases: mx.array | None,
        transpose: bool,
        group_size: int,
        bits: int,
        mode: str,
    ) -> mx.array: ...


class _MutableQuantizedAttention(Protocol):
    q_lora_rank: int
    wqkv_a: nn.QuantizedLinear


def test_normalize_renames_only_dotted_hyper_connection_keys() -> None:
    args = _args()
    weights = {
        "model.layers.0.attn_hc.fn": mx.zeros((1,)),
        "model.layers.0.ffn_hc.scale": mx.ones((1,)),
        "model.layers.1.hc_attn.base": mx.full((1,), 2.0),
        "model.layers.1.hc_ffn.fn": mx.full((1,), 3.0),
        "model.layers.0.hc_attn_fn": mx.full((1,), 4.0),
    }

    normalized = normalize_deepseek_v4_0731_weights(weights, args)

    assert set(normalized) == {
        "model.layers.0.hc_attn.fn",
        "model.layers.0.hc_ffn.scale",
        "model.layers.1.hc_attn.base",
        "model.layers.1.hc_ffn.fn",
        "model.layers.0.hc_attn_fn",
    }
    assert normalized["model.layers.0.hc_attn.fn"] is weights["model.layers.0.attn_hc.fn"]
    assert normalized["model.layers.0.hc_ffn.scale"] is weights["model.layers.0.ffn_hc.scale"]
    assert normalized["model.layers.1.hc_attn.base"] is weights["model.layers.1.hc_attn.base"]
    assert normalized["model.layers.1.hc_ffn.fn"] is weights["model.layers.1.hc_ffn.fn"]
    assert normalized["model.layers.0.hc_attn_fn"] is weights["model.layers.0.hc_attn_fn"]


def test_normalize_rejects_dotted_hyper_connection_destination_collisions() -> None:
    args = _args()
    weights = {
        "model.layers.0.attn_hc.fn": mx.zeros((1,)),
        "model.layers.0.hc_attn.fn": mx.ones((1,)),
    }

    with pytest.raises(ValueError, match="model.layers.0.hc_attn.fn"):
        normalize_deepseek_v4_0731_weights(weights, args)


def test_normalize_reshapes_grouped_wo_a_weights_and_scales() -> None:
    args = _args()
    weight = mx.zeros((8, 1024, 1024))
    scales = mx.zeros((8, 1024, 128))
    already_flat = mx.zeros((8192, 1024))
    weights = {
        "model.layers.0.attn.wo_a.weight": weight,
        "model.layers.0.attn.wo_a.scales": scales,
        "model.layers.1.attn.wo_a.weight": already_flat,
    }

    normalized = normalize_deepseek_v4_0731_weights(weights, args)

    assert normalized["model.layers.0.attn.wo_a.weight"].shape == (8192, 1024)
    assert normalized["model.layers.0.attn.wo_a.scales"].shape == (8192, 128)
    assert normalized["model.layers.0.attn.wo_a.weight"] is not weight
    assert normalized["model.layers.0.attn.wo_a.scales"] is not scales
    assert normalized["model.layers.1.attn.wo_a.weight"] is already_flat


def test_normalize_rejects_wo_a_with_unexpected_grouped_shape() -> None:
    args = _args()
    path = "model.layers.0.attn.wo_a.weight"

    with pytest.raises(ValueError) as error:
        normalize_deepseek_v4_0731_weights({path: mx.zeros((4, 1024, 1024))}, args)

    message = str(error.value)
    assert path in message
    assert "(4, 1024, 1024)" in message
    assert "(8, 1024)" in message


def test_shared_experts_inherit_and_apply_swiglu_limit() -> None:
    limited_args = _args(swiglu_limit=1.0)
    limited_model = DeepseekV40731Model(limited_args)
    unlimited_model = PinnedDeepseekV4Model(_args(swiglu_limit=0.0))

    for layer in limited_model.layers:
        shared = cast(_SharedExpert, cast(object, layer.ffn.shared_experts))
        assert shared.swiglu_limit == limited_args.swiglu_limit

    for model in (limited_model, unlimited_model):
        shared = cast(_SharedExpert, cast(object, model.layers[0].ffn.shared_experts))
        shared.gate_proj.weight = mx.eye(32)
        shared.up_proj.weight = mx.eye(32)
        shared.down_proj.weight = mx.eye(32)

    activations = mx.full((1, 1, 32), 50.0)
    limited_shared = cast(
        _SharedExpert, cast(object, limited_model.layers[0].ffn.shared_experts)
    )
    unlimited_shared = cast(
        _SharedExpert, cast(object, unlimited_model.layers[0].ffn.shared_experts)
    )
    limited = limited_shared(activations)
    unlimited = unlimited_shared(activations)
    mx.eval(limited, unlimited)

    assert limited[0, 0, 0].item() < unlimited[0, 0, 0].item()


def test_compatibility_prepass_leaves_base_sanitizer_transformations_untouched() -> None:
    args = _args()
    fused = mx.zeros((12, 32))
    stacked = mx.zeros((1, 32, 32))
    mtp = mx.zeros((32, 32))
    weights = {
        "model.layers.0.attn.wqkv_a.weight": fused,
        "model.layers.0.ffn.switch_mlp.gate_proj.weight": stacked,
        "mtp.0.markov_head.markov_w1.weight": mtp,
    }

    prepassed = normalize_deepseek_v4_0731_weights(weights, args)

    assert set(prepassed) == set(weights)
    assert prepassed["model.layers.0.attn.wqkv_a.weight"] is fused
    assert prepassed["model.layers.0.ffn.switch_mlp.gate_proj.weight"] is stacked
    assert prepassed["mtp.0.markov_head.markov_w1.weight"] is mtp

    sanitized = DeepseekV40731Model(args).sanitize(prepassed)

    assert "mtp.0.markov_head.markov_w1.weight" not in sanitized
    assert sanitized["model.layers.0.attn.wqkv_a.weight"] is fused
    assert sanitized["model.layers.0.ffn.switch_mlp.gate_proj.weight"] is stacked


def test_0731_hash_gate_uses_bf16_gemm_before_float32_scoring_and_keeps_paths() -> None:
    args = _args()
    routing_args = cast(_RoutingArgs, cast(object, args))
    routing_args.n_routed_experts = 8
    routing_args.num_experts_per_tok = 2
    routing_args.num_hash_layers = 1
    routing_args.hidden_size = 4096
    routing_args.num_hidden_layers = 1
    model = DeepseekV40731Model(args)
    gate = cast(_HashGate, cast(object, model.layers[0].ffn.gate))
    gate.weight = mx.array(
        [
            [((expert * 7919 + feature * 1543) % 997 - 498) / 137
             for feature in range(routing_args.hidden_size)]
            for expert in range(routing_args.n_routed_experts)
        ],
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    gate.tid2eid = mx.array([[0, 7], [2, 5]], dtype=mx.int32)
    hidden = mx.array(
        [
            [
                ((position * 1237 + feature * 3163) % 1019 - 509) / 17
                for feature in range(routing_args.hidden_size)
            ]
            for position in range(4)
        ],
        dtype=mx.float32,
    )[None].astype(mx.bfloat16)
    input_ids = mx.array([[0, 1, 0, 1]], dtype=mx.int32)

    logits = (hidden @ gate.weight.T).astype(mx.float32)
    scores = mx.sqrt(mx.logaddexp(logits, 0))
    selected = mx.take_along_axis(scores, gate.tid2eid[input_ids], axis=-1)
    expected = (
        selected / (selected.sum(axis=-1, keepdims=True) + 1e-20)
        * routing_args.routed_scaling_factor
    )
    indices, actual = gate(hidden, input_ids)
    mx.eval(indices, actual, expected)

    np.testing.assert_array_equal(np.array(indices), np.array(gate.tid2eid[input_ids]))
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.0,
        atol=0.0,
    )
    assert actual.dtype == mx.float32
    parameter_paths = {path for path, _ in tree_flatten(model.parameters())}
    assert "model.layers.0.ffn.gate.weight" in parameter_paths
    assert "model.layers.0.ffn.gate.tid2eid" in parameter_paths



def test_0731_non_hash_gate_uses_omlx_biased_plain_topk_and_float32_weights() -> None:
    args = _args()
    routing_args = cast(_RoutingArgs, cast(object, args))
    routing_args.n_routed_experts = 256
    routing_args.num_experts_per_tok = 6
    routing_args.num_hash_layers = 0
    routing_args.num_hidden_layers = 1
    gate = DeepseekV40731MoEGate(args, layer_id=0)
    gate.weight = mx.array(
        [
            [((expert * 73 + feature * 29) % 97 - 48) / 19
             for feature in range(routing_args.hidden_size)]
            for expert in range(routing_args.n_routed_experts)
        ],
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    gate.e_score_correction_bias = mx.array(
        [((expert * 31) % 89 - 44) / 7 for expert in range(routing_args.n_routed_experts)],
        dtype=mx.float32,
    )
    hidden = mx.array(
        [
            [((position * 47 + feature * 17) % 101 - 50) / 13
             for feature in range(routing_args.hidden_size)]
            for position in range(4)
        ],
        dtype=mx.float32,
    )[None].astype(mx.bfloat16)

    logits = (hidden @ gate.weight.T).astype(mx.float32)
    scores = mx.sqrt(mx.logaddexp(logits, 0))
    selected_indices = mx.argpartition(
        -(scores + gate.e_score_correction_bias), kth=routing_args.num_experts_per_tok - 1, axis=-1
    )[..., : routing_args.num_experts_per_tok]
    selected_scores = mx.take_along_axis(scores, selected_indices, axis=-1)
    expected_weights = (
        selected_scores / (selected_scores.sum(axis=-1, keepdims=True) + 1e-20)
        * routing_args.routed_scaling_factor
    )
    actual_indices, actual_weights = gate(hidden)
    mx.eval(selected_indices, expected_weights, actual_indices, actual_weights)

    np.testing.assert_array_equal(np.array(actual_indices), np.array(selected_indices))
    np.testing.assert_allclose(
        np.array(actual_weights), np.array(expected_weights), rtol=0.0, atol=0.0
    )
    assert actual_weights.dtype == mx.float32


def test_split_quantized_wqkv_uses_two_row_slices_for_prefill_and_fused_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = nn.QuantizedLinear(
        32, 12, bias=True, group_size=32, bits=8, mode="mxfp8"
    )
    source_state = cast(_QuantizedLinearState, cast(object, source))
    split = SplitOutputQuantizedLinear.from_quantized_linear(source, split=8)
    prefill = mx.arange(64, dtype=mx.float32).reshape(1, 2, 32) / 17
    decode = prefill[:, :1]
    calls: list[tuple[int, int]] = []
    original = cast(_QuantizedMatmul, cast(object, mx.quantized_matmul))

    def traced_quantized_matmul(
        hidden: mx.array,
        weight: mx.array,
        *,
        scales: mx.array,
        biases: mx.array | None,
        transpose: bool,
        group_size: int,
        bits: int,
        mode: str,
    ) -> mx.array:
        calls.append((weight.shape[0], bits))
        return original(
            hidden,
            weight,
            scales=scales,
            biases=biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    monkeypatch.setattr(mx, "quantized_matmul", traced_quantized_matmul)
    split_prefill = split(prefill)
    assert calls == [(8, 8), (4, 8)]
    calls.clear()
    split_decode = split(decode)
    assert calls == [(12, 8)]
    assert source_state.bias is not None
    expected_prefill: mx.array = mx.concatenate(
        [
            original(
                prefill,
                source_state.weight[:8],
                scales=source_state.scales[:8],
                biases=(
                    source_state.biases[:8] if source_state.biases is not None else None
                ),
                transpose=True,
                group_size=source_state.group_size,
                bits=source_state.bits,
                mode=source_state.mode,
            ),
            original(
                prefill,
                source_state.weight[8:],
                scales=source_state.scales[8:],
                biases=(
                    source_state.biases[8:] if source_state.biases is not None else None
                ),
                transpose=True,
                group_size=source_state.group_size,
                bits=source_state.bits,
                mode=source_state.mode,
            ),
        ],
        axis=-1,
    ) + source_state.bias
    expected_decode = source_state(decode)
    mx.eval(split_prefill, expected_prefill, split_decode, expected_decode)
    np.testing.assert_array_equal(np.array(split_prefill), np.array(expected_prefill))
    np.testing.assert_array_equal(np.array(split_decode), np.array(expected_decode))


def test_prefill_installer_preserves_direct_quantized_paths_and_audit() -> None:
    from exo.worker.engines.mlx.deepseek_v4_0731_loader import (
        audit_deepseek_v4_0731_quantization,
    )

    args = _args()
    args.num_hidden_layers = 1
    model = DeepseekV40731Model(args)
    attention = cast(
        _MutableQuantizedAttention,
        cast(object, model.model.layers[0].attn),
    )
    quantized: nn.QuantizedLinear = cast(
        _QuantizedLinearState, cast(object, attention.wqkv_a)
    ).to_quantized(
        group_size=32, bits=8, mode="mxfp8"
    )
    attention.wqkv_a = quantized
    before_paths = {path for path, _ in tree_flatten(model.parameters())}

    install_deepseek_v4_0731_prefill_attention(model)
    wrapped = attention.wqkv_a
    install_deepseek_v4_0731_prefill_attention(model)

    assert isinstance(wrapped, SplitOutputQuantizedLinear)
    assert attention.wqkv_a is wrapped
    assert isinstance(wrapped, nn.QuantizedLinear)
    assert wrapped.group_size == 32
    assert wrapped.bits == 8
    assert wrapped.mode == "mxfp8"
    assert "model.layers.0.attn.wqkv_a.weight" in {path for path, _ in tree_flatten(model.parameters())}
    assert "model.layers.0.attn.wqkv_a.scales" in {path for path, _ in tree_flatten(model.parameters())}
    wrapped_state = cast(_QuantizedLinearState, cast(object, wrapped))
    quantized_state = cast(_QuantizedLinearState, cast(object, quantized))
    assert wrapped_state.get("biases") is quantized_state.get("biases")
    assert ("bias" in wrapped) is ("bias" in quantized)
    assert not any("wqkv_a.original" in path for path, _ in tree_flatten(model.parameters()))
    assert before_paths == {path for path, _ in tree_flatten(model.parameters())}
    audit_deepseek_v4_0731_quantization(
        model,
        {
            "model.layers.0.attn.wqkv_a": QuantizationSpec(
                bits=8, group_size=32, mode="mxfp8"
            ),
            "model.layers.0.ffn.switch_mlp.gate_proj": QuantizationSpec(
                bits=4, group_size=32, mode="mxfp4"
            ),
            "model.layers.0.ffn.switch_mlp.up_proj": QuantizationSpec(
                bits=4, group_size=32, mode="mxfp4"
            ),
            "model.layers.0.ffn.switch_mlp.down_proj": QuantizationSpec(
                bits=4, group_size=32, mode="mxfp4"
            ),
        },
    )


@pytest.mark.parametrize(("mode", "bits"), [("affine", 4), ("mxfp4", 4)])
def test_prefill_installer_leaves_non_mxfp8_quantized_attention_unchanged(
    mode: str, bits: int
) -> None:
    class _Attention:
        q_lora_rank = 8

        def __init__(self) -> None:
            self.wqkv_a = nn.QuantizedLinear(
                32, 12, bias=False, group_size=32, bits=bits, mode=mode
            )

    attention = _Attention()
    original = attention.wqkv_a

    from exo.worker.engines.mlx.deepseek_v4_0731_model import (
        install_deepseek_v4_0731_attention_prefill_split,
    )

    install_deepseek_v4_0731_attention_prefill_split(attention)  # pyright: ignore[reportArgumentType]  # Test fixture supplies the production prefill protocol structurally.

    assert attention.wqkv_a is original
