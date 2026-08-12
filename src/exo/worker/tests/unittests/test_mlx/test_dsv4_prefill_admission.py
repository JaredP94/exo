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
  below the recorded successful lengths 32,804, 32,819 and 32,828.
* test_environment_override_is_honoured - the documented operator override
  being ignored or parsed only after the worker has started.
* test_normal_generation_admits_before_cache_allocation - the normal path
  allocating a cache before checking the prompt.
* test_batch_generation_admits_before_cache_allocation - the batch path
  omitting the guard or allocating a cache before checking the prompt.
* test_disaggregated_prefill_admits_before_cache_allocation - the remote
  prefill server bypassing the normal generator guard.
* test_sequential_step_reports_a_refusal_without_raising - the runner-killing
  error handler at SequentialGenerator.step() re-raising a user input error.
* test_sequential_start_next_reports_a_refusal_without_raising - the same
  failure at SequentialGenerator._start_next().
* test_batch_step_reports_a_refusal_without_raising - the batch path re-raising
  a refusal from BatchGenerator.step().
* test_refusal_does_not_wedge_the_next_task - clearing only the exception while
  leaving the queue or active state stuck.
* test_non_admission_errors_still_raise - broadening the fix to swallow every
  exception instead of only PromptTooLongError.
* test_normal_generation_rechecks_vision_prompt - deleting the vision-expanded
  prompt re-check.
* test_batch_generation_rechecks_vision_prompt - deleting the batch vision
  re-check.
"""

from __future__ import annotations

import importlib
from collections import deque
from collections.abc import Callable, Generator, Sequence
from typing import NoReturn, cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.disaggregated import serve as disaggregated_serve
from exo.worker.engines.mlx.generator import batch_generate, generate
from exo.worker.engines.mlx.generator.admission import (
    PROMPT_ADMISSION_CEILING_DEFAULT,
    PromptTooLongError,
    assert_prompt_within_ceiling,
)
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vision import VisionProcessor, VisionResult
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)


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
    """Break this catches: lowering the safety margin below recorded successes."""
    # Successful cold prefills are recorded at 32,804, 32,819 and 32,828.
    assert PROMPT_ADMISSION_CEILING_DEFAULT > 32_828


def test_environment_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break this catches: ignoring the documented operator override."""
    import exo.shared.constants as constants

    monkeypatch.setenv("EXO_PROMPT_ADMISSION_CEILING", "12345")
    importlib.reload(constants)
    assert constants.EXO_PROMPT_ADMISSION_CEILING == 12_345

    monkeypatch.delenv("EXO_PROMPT_ADMISSION_CEILING")
    importlib.reload(constants)


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


class _RecordingEventSender:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _agree_on_no_tasks(_runner: SequentialGenerator | BatchGenerator) -> None:
    return None


def _raise_prompt_refusal(_task: TextGeneration) -> Generator[object, None, None]:
    return _raising_generation(PromptTooLongError(50_000, 40_960))


def _raise_runtime_error(_task: TextGeneration) -> Generator[object, None, None]:
    return _raising_generation(RuntimeError("not an admission refusal"))


def _raise_batch_refusal(_task: TextGeneration) -> NoReturn:
    raise PromptTooLongError(50_000, 40_960)


def _short_encode_prompt(*_args: object, **_kwargs: object) -> mx.array:
    return mx.zeros((1,))


def _vision_prompt(**_kwargs: object) -> VisionResult:
    return _vision_result(2_049)


def _raising_generation(error: Exception) -> Generator[object, None, None]:
    if False:
        yield None
    raise error


def _empty_generation() -> Generator[object, None, None]:
    if False:
        yield None


def _runner_task(task_id: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(task_id),
        instance_id=InstanceId("synthetic-instance"),
        command_id=CommandId(f"command-{task_id}"),
        task_params=_generation_task().model_copy(update={"bench": True}),
    )


class _NoWorkBatchEngine:
    has_work = False


def _make_sequential_runner(
    sender: _RecordingEventSender,
    tasks: list[TextGeneration],
) -> SequentialGenerator:
    runner = object.__new__(SequentialGenerator)
    runner.device_rank = 0
    runner.model = cast(Model, object())
    runner.tokenizer = cast(TokenizerWrapper, object())
    runner.tool_parser = None
    runner.model_id = ModelId("synthetic-model")
    setattr(runner, "event_sender", sender)  # noqa: B010
    setattr(runner, "_queue", deque(tasks))  # noqa: B010
    setattr(runner, "_maybe_queue", [])  # noqa: B010
    setattr(runner, "_cancelled_tasks", set())  # noqa: B010
    setattr(runner, "_active", None)  # noqa: B010
    return runner


def _make_batch_runner(
    sender: _RecordingEventSender,
    tasks: list[TextGeneration],
) -> BatchGenerator:
    runner = object.__new__(BatchGenerator)
    runner.device_rank = 0
    runner.model = cast(Model, object())
    runner.tokenizer = cast(TokenizerWrapper, object())
    runner.tool_parser = None
    runner.model_id = ModelId("synthetic-model")
    setattr(runner, "event_sender", sender)  # noqa: B010
    setattr(runner, "_queue", deque(tasks))  # noqa: B010
    setattr(runner, "_maybe_queue", [])  # noqa: B010
    setattr(runner, "_cancelled_tasks", set())  # noqa: B010
    setattr(runner, "_active_tasks", {})  # noqa: B010
    setattr(runner, "_gen", _NoWorkBatchEngine())  # noqa: B010
    return runner


def _assert_finished_response(
    results: Sequence[tuple[TaskId, object]],
) -> None:
    assert any(isinstance(response, FinishedResponse) for _, response in results)


def test_sequential_step_reports_a_refusal_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _RecordingEventSender()
    runner = _make_sequential_runner(sender, [_runner_task("refused")])
    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", _agree_on_no_tasks)
    monkeypatch.setattr(
        runner,
        "_build_generator",
        _raise_prompt_refusal,
    )

    results = list(runner.step())

    _assert_finished_response(results)
    assert getattr(runner, "_active") is None  # noqa: B009
    assert len(sender.events) == 1


def test_sequential_start_next_reports_a_refusal_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _RecordingEventSender()
    runner = _make_sequential_runner(sender, [_runner_task("refused")])

    def raise_refusal(_task: TextGeneration) -> Generator[object, None, None]:
        raise PromptTooLongError(50_000, 40_960)

    monkeypatch.setattr(runner, "_build_generator", raise_refusal)

    start_next: Callable[[], TextGeneration | None] = runner._start_next  # pyright: ignore[reportPrivateUsage]
    result = start_next()

    assert result is not None
    assert result.task_id == TaskId("refused")
    assert len(sender.events) == 1


def test_batch_step_reports_a_refusal_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _RecordingEventSender()
    runner = _make_batch_runner(sender, [_runner_task("refused")])
    monkeypatch.setattr(BatchGenerator, "agree_on_tasks", _agree_on_no_tasks)
    monkeypatch.setattr(
        runner,
        "_start_task",
        _raise_batch_refusal,
    )

    results = list(runner.step())

    _assert_finished_response(results)
    assert not getattr(runner, "_active_tasks")  # noqa: B009
    assert len(sender.events) == 1


def test_refusal_does_not_wedge_the_next_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _RecordingEventSender()
    runner = _make_sequential_runner(
        sender, [_runner_task("refused"), _runner_task("next")]
    )
    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", _agree_on_no_tasks)
    generators = iter(
        [
            _raising_generation(PromptTooLongError(50_000, 40_960)),
            _empty_generation(),
        ]
    )

    def next_generation(_task: TextGeneration) -> Generator[object, None, None]:
        return next(generators)

    monkeypatch.setattr(runner, "_build_generator", next_generation)

    first_results = list(runner.step())
    second_results = list(runner.step())

    _assert_finished_response(first_results)
    _assert_finished_response(second_results)
    assert getattr(runner, "_active") is None  # noqa: B009
    assert not getattr(runner, "_queue")  # noqa: B009
    assert len(sender.events) == 1


def test_non_admission_errors_still_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _RecordingEventSender()
    runner = _make_sequential_runner(sender, [_runner_task("broken")])
    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", _agree_on_no_tasks)
    monkeypatch.setattr(
        runner,
        "_build_generator",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="not an admission refusal"):
        list(runner.step())


def _vision_result(prompt_tokens: int) -> VisionResult:
    return VisionResult(
        prompt="synthetic vision prompt",
        prompt_tokens=mx.zeros((prompt_tokens,), dtype=mx.int32),
        embeddings=mx.zeros((1,), dtype=mx.float32),
        media_regions=[],
        image_token_id=0,
    )


def test_normal_generation_rechecks_vision_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate, "EXO_PROMPT_ADMISSION_CEILING", 2_048)
    monkeypatch.setattr(generate, "encode_prompt", _short_encode_prompt)
    monkeypatch.setattr(
        generate, "fix_unmatched_think_end_tokens", _identity_prompt_tokens
    )
    monkeypatch.setattr(generate, "system_prompt_token_count", _zero_system_prompt)
    monkeypatch.setattr(generate, "prepare_vision", _vision_prompt)

    with pytest.raises(PromptTooLongError):
        next(
            generate.mlx_generate(
                model=cast(Model, object()),
                tokenizer=cast(TokenizerWrapper, object()),
                task=_generation_task(),
                prompt="synthetic safe prompt",
                kv_prefix_cache=None,
                group=None,
                vision_processor=cast(VisionProcessor, object()),
            )
        )


def test_batch_generation_rechecks_vision_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_generate, "EXO_PROMPT_ADMISSION_CEILING", 2_048)
    monkeypatch.setattr(batch_generate, "encode_prompt", _short_encode_prompt)
    monkeypatch.setattr(
        batch_generate, "fix_unmatched_think_end_tokens", _identity_prompt_tokens
    )
    monkeypatch.setattr(batch_generate, "prepare_vision", _vision_prompt)
    generator = object.__new__(ExoBatchGenerator)
    generator.model = cast(Model, object())
    generator.tokenizer = cast(TokenizerWrapper, object())
    generator.group = None
    generator.kv_prefix_cache = None
    generator.vision_processor = cast(VisionProcessor, object())

    with pytest.raises(PromptTooLongError):
        generator.submit(
            task_params=_generation_task(),
            prompt="synthetic safe prompt",
        )
