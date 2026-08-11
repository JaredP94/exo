"""Prefill must preserve the prompt sequence length at attention."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx_lm.models.deepseek_v4 as dsv4
import pytest

SEQ_LEN = 7
SYNTHETIC: dict[str, int] = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "head_dim": 16,
    "qk_rope_head_dim": 8,
    "o_groups": 2,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "num_hidden_layers": 1,
}


def test_attention_sees_the_true_sequence_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    original: Callable[..., mx.array] = dsv4.scaled_dot_product_attention  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]

    def spy(queries: mx.array, *args: object, **kwargs: object) -> mx.array:
        seen.append(int(queries.shape[-2]))
        return original(queries, *args, **kwargs)  # pyright: ignore[reportUnknownVariableType]

    monkeypatch.setattr(dsv4, "scaled_dot_product_attention", spy)

    args = dsv4.ModelArgs(**SYNTHETIC)  # pyright: ignore[reportArgumentType]
    attention = dsv4.V4Attention(args, layer_id=0)
    hidden = mx.random.normal((1, SEQ_LEN, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(attention(hidden, cache=None))

    assert seen, "attention was never called"
    assert set(seen) == {SEQ_LEN}, (
        f"attention saw sequence lengths {sorted(set(seen))}, expected {SEQ_LEN}; "
        f"a value of {SEQ_LEN * 4} means the hc_mult axis was folded into the "
        "sequence axis and every token is presented four times"
    )
