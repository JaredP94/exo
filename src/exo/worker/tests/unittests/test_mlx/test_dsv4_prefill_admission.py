"""The prompt admission ceiling must stop unsafe work at every prefill boundary.

Break each test catches:

* test_guard_rejects_a_prompt_above_the_ceiling - the comparison being >=
  instead of >, or the guard being placed after the first model call rather
  than before it.
* test_guard_admits_a_prompt_exactly_at_the_ceiling - an off-by-one that
  refuses a legal 32,768-token prompt, silently shrinking the proven supported
  context.
* test_error_message_names_both_counts - a bare "prompt too long" with no
  numbers. Because the refusal reaches the client as an opaque 500, the
  message is the only diagnostic the caller gets.
* test_a_zero_or_negative_ceiling_disables_the_guard - a misconfigured ceiling
  of 0 refusing every request and taking the node out of service.
* test_default_ceiling_is_the_proven_supported_length - the default drifting
  off 32,768, which is the highest length measured to work.
* test_normal_generation_admits_before_cache_allocation - the normal path
  allocating a cache before checking the prompt.
* test_batch_generation_admits_before_cache_allocation - the batch path
  omitting the guard or allocating a cache before checking the prompt.
* test_disaggregated_prefill_admits_before_cache_allocation - the remote
  prefill server bypassing the normal generator guard.
"""

from __future__ import annotations

from typing import NoReturn, cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.disaggregated import serve as disaggregated_serve
from exo.worker.engines.mlx.generator import batch_generate, generate
from exo.worker.engines.mlx.generator.admission import (
    PROMPT_ADMISSION_CEILING_DEFAULT,
    PromptTooLongError,
    assert_prompt_within_ceiling,
)
from exo.worker.engines.mlx.types import Model


def test_guard_rejects_a_prompt_above_the_ceiling() -> None:
    """Break this catches: the comparison being >= instead of >, or the guard
    being placed after the first model call rather than before it."""
    with pytest.raises(PromptTooLongError) as excinfo:
        assert_prompt_within_ceiling(prompt_tokens=40_000, limit_tokens=32_768)
    assert excinfo.value.prompt_tokens == 40_000
    assert excinfo.value.limit_tokens == 32_768


def test_guard_admits_a_prompt_exactly_at_the_ceiling() -> None:
    """Break this catches: an off-by-one that refuses a legal 32,768-token
    prompt, silently shrinking the proven supported context."""
    assert_prompt_within_ceiling(prompt_tokens=32_768, limit_tokens=32_768)


def test_error_message_names_both_counts() -> None:
    """Break this catches: a bare 'prompt too long' with no numbers. Because the
    refusal reaches the client as an opaque 500, the message is the only
    diagnostic the caller gets."""
    error = PromptTooLongError(prompt_tokens=40_000, limit_tokens=32_768)
    assert "40000" in str(error) and "32768" in str(error)


def test_a_zero_or_negative_ceiling_disables_the_guard() -> None:
    """Break this catches: a misconfigured ceiling of 0 refusing every request
    and taking the node out of service."""
    assert_prompt_within_ceiling(prompt_tokens=100_000, limit_tokens=0)
    assert_prompt_within_ceiling(prompt_tokens=100_000, limit_tokens=-1)


def test_default_ceiling_is_the_proven_supported_length() -> None:
    """Break this catches: the default drifting off 32768, which is the highest
    length measured to work. 65536 has panicked a host."""
    assert PROMPT_ADMISSION_CEILING_DEFAULT == 32_768


def _over_limit_prompt() -> mx.array:
    return mx.zeros((4_000,), dtype=mx.int32)


def _generation_task() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("synthetic-model"),
        input=[
            InputMessage(
                role="user",
                content=InputMessageContent("synthetic safe prompt"),
            )
        ],
        use_prefix_cache=False,
        seed=42,
    )


def _synthetic_encode_prompt(
    _tokenizer: TokenizerWrapper,
    _prompt: str,
    *,
    raw_input_ids: list[int] | None = None,
) -> mx.array:
    del raw_input_ids
    return _over_limit_prompt()


def _identity_prompt_tokens(tokens: mx.array, _tokenizer: TokenizerWrapper) -> mx.array:
    return tokens


def _zero_system_prompt(
    _task: TextGenerationTaskParams, _tokenizer: TokenizerWrapper
) -> int:
    return 0


def _fail_if_cache_is_allocated(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("cache allocation happened before prompt admission")


def test_normal_generation_admits_before_cache_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break this catches: normal generation allocating a cache before admission.

    The 4,000-token synthetic prompt is intentionally safe; the test only
    exercises a temporary 2,048-token ceiling and never approaches the
    prohibited prompt size.
    """
    monkeypatch.setattr(generate, "EXO_PROMPT_ADMISSION_CEILING", 2_048)
    monkeypatch.setattr(generate, "encode_prompt", _synthetic_encode_prompt)
    monkeypatch.setattr(
        generate, "fix_unmatched_think_end_tokens", _identity_prompt_tokens
    )
    monkeypatch.setattr(generate, "system_prompt_token_count", _zero_system_prompt)
    monkeypatch.setattr(generate, "make_kv_cache", _fail_if_cache_is_allocated)

    with pytest.raises(PromptTooLongError):
        next(
            generate.mlx_generate(
                model=cast(Model, object()),
                tokenizer=cast(TokenizerWrapper, object()),
                task=_generation_task(),
                prompt="synthetic safe prompt",
                kv_prefix_cache=None,
                group=None,
            )
        )


def test_batch_generation_admits_before_cache_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break this catches: batch generation omitting admission before cache creation."""
    monkeypatch.setattr(batch_generate, "EXO_PROMPT_ADMISSION_CEILING", 2_048)
    monkeypatch.setattr(batch_generate, "encode_prompt", _synthetic_encode_prompt)
    monkeypatch.setattr(
        batch_generate,
        "fix_unmatched_think_end_tokens",
        _identity_prompt_tokens,
    )
    monkeypatch.setattr(batch_generate, "make_kv_cache", _fail_if_cache_is_allocated)
    generator = object.__new__(batch_generate.ExoBatchGenerator)
    generator.model = cast(Model, object())
    generator.tokenizer = cast(TokenizerWrapper, object())
    generator.group = None
    generator.kv_prefix_cache = None
    generator.vision_processor = None

    with pytest.raises(PromptTooLongError):
        generator.submit(
            task_params=_generation_task(),
            prompt="synthetic safe prompt",
        )


def test_disaggregated_prefill_admits_before_cache_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break this catches: the remote prefill server bypassing admission."""
    monkeypatch.setattr(disaggregated_serve, "EXO_PROMPT_ADMISSION_CEILING", 2_048)
    monkeypatch.setattr(
        disaggregated_serve,
        "fix_unmatched_think_end_tokens",
        _identity_prompt_tokens,
    )
    monkeypatch.setattr(
        disaggregated_serve, "make_kv_cache", _fail_if_cache_is_allocated
    )

    with pytest.raises(PromptTooLongError):
        disaggregated_serve.run_prefill_for_request(
            model=cast(Model, object()),
            tokenizer=cast(TokenizerWrapper, object()),
            group=None,
            kv_prefix_cache=None,
            request=PrefillRequest(
                model_id="synthetic-model",
                token_ids=[1] * 4_000,
                request_id="synthetic-request",
            ),
        )
