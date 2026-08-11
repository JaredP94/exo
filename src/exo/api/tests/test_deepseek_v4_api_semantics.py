# pyright: reportAny=false
"""DeepSeek V4 0731 semantics through the Chat Completions and Responses APIs.

Every test drives raw V4 model tokens through `apply_all_parsers` — the exact
composition `batch_generator._start_task` uses — and then through the real API
adapters. That covers parser-to-wire composition rather than hand-built final
chunks, which is what plan Task 6 Step 1 asks for.

Honest labelling, per the Task 5 review lesson recorded in the ledger:

* These are **contract guards, not regression guards**. Plan 2 Tasks 1-5 are
  already committed, so every test here is green against this branch by
  construction. A red-green cycle is therefore not available and would be a
  lie to claim.
* Their value is proven by **mutation** instead: each test class names the
  implementation break that makes it fail. Those mutations were executed and
  the results recorded in
  `docs/superpowers/validation/2026-08-04-dsv4-0731-prompt-api.md`.

Token splits are the real ones. `｜DSML｜` is a single V4 token (128825), so
`<｜DSML｜tool_calls>` reaches the parser as six tokens
(`'<'`, `'｜DSML｜'`, `'tool'`, `'_c'`, `'alls'`, `'>'`). Split delimiters are
the only way these markers ever arrive for this checkpoint, which is why the
streams below are written token-by-token rather than as whole strings.
"""

import json
from collections.abc import AsyncGenerator, Generator, Iterator
from typing import Any, cast

import pytest
from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from exo.api.types.openai_responses import ResponseInputMessage, ResponsesRequest
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.text_generation import ReasoningEffort, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.types import Model as ExoModel
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import dsml_token
from exo.worker.runner.llm_inference.model_output_parsers import apply_all_parsers

V4_MODEL_ID = ModelId("Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp")
THINK_START = "<think>"
THINK_END = "</think>"

#: Every marker fragment that must never reach a user-visible channel. `dsml`
#: alone is deliberately included: Task 5 regressions surface as a lone
#: `｜DSML｜` token leaking without its surrounding angle brackets.
FORBIDDEN_MARKUP = (dsml_token, "tool_calls>", "invoke", "<parameter", THINK_END)


class _V4Tokenizer:
    """The V4 fields `apply_all_parsers` reads.

    `render_chat_template`'s V4 branch never touches the tokenizer, and
    `apply_all_parsers` reads only these three attributes plus
    `detect_thinking_prompt_suffix`, so a real 6 MiB `TokenizerWrapper` would
    add load time without adding coverage.
    """

    has_thinking = True
    think_start = THINK_START
    think_end = THINK_END


def _tokenizer() -> TokenizerWrapper:
    # Structural stub, not a `TokenizerWrapper` subclass, so basedpyright needs
    # the widening step through `object`.
    return cast("TokenizerWrapper", cast("object", _V4Tokenizer()))


def _usage(prompt_tokens: int = 11, completion_tokens: int = 7) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


def _model_tokens(
    texts: list[str], *, finish_reason: str | None
) -> Generator[GenerationResponse | None]:
    """Emit one `GenerationResponse` per token, usage on the last one only."""
    last_index = len(texts) - 1
    for index, text in enumerate(texts):
        is_last = index == last_index
        yield GenerationResponse(
            text=text,
            token=index,
            finish_reason=cast(Any, finish_reason) if is_last else None,
            usage=_usage() if is_last else None,
        )


def _parsed_chunks(
    texts: list[str],
    *,
    prompt: str,
    finish_reason: str | None = "stop",
) -> list[GenerationChunk]:
    """Run the production parser composition over a raw V4 token stream."""
    chunks: Iterator[GenerationChunk | None] = apply_all_parsers(
        _model_tokens(texts, finish_reason=finish_reason),
        prompt,
        None,
        _tokenizer(),
        # Production passes `type(self.model)`, which the loader types as exo's
        # structural `Model`. `DeepseekV4Model` is the concrete class it loads
        # for this checkpoint, and is what `apply_all_parsers` calls
        # `issubclass` against.
        cast("type[ExoModel]", DeepseekV4Model),
        V4_MODEL_ID,
        None,
    )
    return [chunk for chunk in chunks if chunk is not None]


async def _as_stream(
    chunks: list[GenerationChunk],
) -> AsyncGenerator[
    PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
]:
    for chunk in chunks:
        yield cast(
            PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, chunk
        )


async def _chat_stream_events(chunks: list[GenerationChunk]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for raw in generate_chat_stream(CommandId("cmd-chat"), _as_stream(chunks)):
        for line in raw.split("\n"):
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                events.append(cast(dict[str, Any], json.loads(line[len("data: ") :])))
    return events


async def _chat_collected(chunks: list[GenerationChunk]) -> dict[str, Any]:
    parts = [
        part
        async for part in collect_chat_response(
            CommandId("cmd-chat-collected"), _as_stream(chunks)
        )
    ]
    assert len(parts) == 1
    return cast(dict[str, Any], json.loads(parts[0]))


async def _responses_stream_events(
    chunks: list[GenerationChunk],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for raw in generate_responses_stream(
        CommandId("cmd-resp"), str(V4_MODEL_ID), _as_stream(chunks)
    ):
        for line in raw.split("\n"):
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                events.append(cast(dict[str, Any], json.loads(line[len("data: ") :])))
    return events


async def _responses_collected(chunks: list[GenerationChunk]) -> dict[str, Any]:
    parts = [
        part
        async for part in collect_responses_response(
            CommandId("cmd-resp-collected"), str(V4_MODEL_ID), _as_stream(chunks)
        )
    ]
    assert len(parts) == 1
    return cast(dict[str, Any], json.loads(parts[0]))


def _assert_no_markup(text: str, channel: str) -> None:
    for marker in FORBIDDEN_MARKUP:
        assert marker not in text, (
            f"{channel} leaked DeepSeek V4 markup {marker!r}; full text={text!r}"
        )


# --------------------------------------------------------------------------- #
# Raw V4 token streams, written with the checkpoint's real token splits.
# --------------------------------------------------------------------------- #

THINKING_THEN_ANSWER = [
    THINK_START,
    "The user",
    " wants the capital.",
    THINK_END,
    "The capital",
    " is Paris.",
]

_TOOL_CALL_BLOCK = [
    "<",
    dsml_token,
    "tool",
    "_c",
    "alls",
    ">",
    "\n<",
    dsml_token,
    "invoke",
    ' name="get_weather"',
    ">\n<",
    dsml_token,
    "parameter",
    ' name="city" string="true"',
    ">",
    "Paris",
    "</",
    dsml_token,
    "parameter",
    ">\n</",
    dsml_token,
    "invoke",
    ">\n</",
    dsml_token,
    "tool",
    "_c",
    "alls",
    ">",
]

THINKING_THEN_TOOL_CALL = [
    THINK_START,
    "I should check",
    " the weather.",
    THINK_END,
    *_TOOL_CALL_BLOCK,
]


def _text_of(chunks: list[GenerationChunk], *, thinking: bool) -> str:
    return "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and chunk.is_thinking is thinking
    )


# --------------------------------------------------------------------------- #
# Step 1: parsed chunks reaching the wire
# --------------------------------------------------------------------------- #


class TestParserComposition:
    """Breaks caught: routing every token to one channel, and any Task 5
    regression that lets a `｜DSML｜` fragment or `</think>` reach a chunk."""

    def test_reasoning_and_content_land_in_separate_channels(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")

        assert _text_of(chunks, thinking=True) == "The user wants the capital."
        assert _text_of(chunks, thinking=False) == "The capital is Paris."

    def test_no_marker_survives_into_any_chunk(self):
        chunks = _parsed_chunks(THINKING_THEN_TOOL_CALL, prompt="user turn")

        for chunk in chunks:
            if isinstance(chunk, TokenChunk):
                _assert_no_markup(chunk.text, "TokenChunk")

    def test_a_tool_call_becomes_structured_rather_than_text(self):
        chunks = _parsed_chunks(THINKING_THEN_TOOL_CALL, prompt="user turn")

        tool_chunks = [c for c in chunks if isinstance(c, ToolCallChunk)]
        assert len(tool_chunks) == 1
        calls = tool_chunks[0].tool_calls
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "Paris"}


class TestChatCompletionsWire:
    """Breaks caught: merging `reasoning_content` into `content`, emitting a
    finish reason other than `tool_calls` for a structured call, and emitting
    more or fewer than one terminal event."""

    async def test_reasoning_appears_only_in_reasoning_content(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        events = await _chat_stream_events(chunks)

        reasoning = "".join(
            e["choices"][0]["delta"].get("reasoning_content", "") for e in events
        )
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)

        assert reasoning == "The user wants the capital."
        assert content == "The capital is Paris."
        _assert_no_markup(reasoning, "reasoning_content")
        _assert_no_markup(content, "content")

    async def test_collected_response_separates_the_two_channels(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        payload = await _chat_collected(chunks)

        message = payload["choices"][0]["message"]
        assert message["content"] == "The capital is Paris."
        assert message["reasoning_content"] == "The user wants the capital."
        _assert_no_markup(message["content"], "message.content")
        _assert_no_markup(message["reasoning_content"], "message.reasoning_content")

    async def test_structured_tool_call_finishes_as_tool_calls(self):
        chunks = _parsed_chunks(
            THINKING_THEN_TOOL_CALL, prompt="user turn", finish_reason=None
        )
        events = await _chat_stream_events(chunks)

        finish_reasons = [
            e["choices"][0]["finish_reason"]
            for e in events
            if e["choices"][0].get("finish_reason") is not None
        ]
        assert finish_reasons == ["tool_calls"]

        tool_events = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
        assert len(tool_events) == 1
        function = tool_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
        assert function["name"] == "get_weather"
        assert json.loads(function["arguments"]) == {"city": "Paris"}

    async def test_exactly_one_terminal_event_is_emitted(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")

        raw: list[str] = []
        async for event in generate_chat_stream(
            CommandId("cmd-terminal"), _as_stream(chunks)
        ):
            raw.append(event)

        assert sum(part.count("data: [DONE]") for part in raw) == 1
        events = await _chat_stream_events(chunks)
        terminal = [
            e for e in events if e["choices"][0].get("finish_reason") is not None
        ]
        assert len(terminal) == 1

    async def test_usage_reports_reasoning_tokens(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        payload = await _chat_collected(chunks)

        details = payload["usage"]["completion_tokens_details"]
        assert details["reasoning_tokens"] == 2
        assert details["reasoning_tokens"] <= payload["usage"]["completion_tokens"]

    async def test_no_dsml_markup_reaches_the_serialized_stream(self):
        chunks = _parsed_chunks(
            THINKING_THEN_TOOL_CALL, prompt="user turn", finish_reason=None
        )
        events = await _chat_stream_events(chunks)

        for event in events:
            delta = event["choices"][0]["delta"]
            _assert_no_markup(delta.get("content") or "", "delta.content")
            _assert_no_markup(
                delta.get("reasoning_content") or "", "delta.reasoning_content"
            )


class TestResponsesWire:
    """Breaks caught: collapsing reasoning items into output text, losing a
    function call's name or JSON arguments, and dropping reasoning usage from
    `response.completed`."""

    async def test_reasoning_and_output_use_distinct_item_types(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        payload = await _responses_collected(chunks)

        item_types = [item["type"] for item in payload["output"]]
        assert "reasoning" in item_types
        assert "message" in item_types

    async def test_streaming_emits_reasoning_and_output_separately(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        events = await _responses_stream_events(chunks)

        event_types = {e["type"] for e in events}
        assert "response.reasoning_summary_text.delta" in event_types
        assert "response.output_text.delta" in event_types

        reasoning = "".join(
            e.get("delta", "")
            for e in events
            if e["type"] == "response.reasoning_summary_text.delta"
        )
        output = "".join(
            e.get("delta", "")
            for e in events
            if e["type"] == "response.output_text.delta"
        )
        assert reasoning == "The user wants the capital."
        assert output == "The capital is Paris."

    async def test_function_call_retains_name_and_json_arguments(self):
        chunks = _parsed_chunks(
            THINKING_THEN_TOOL_CALL, prompt="user turn", finish_reason=None
        )
        payload = await _responses_collected(chunks)

        calls = [item for item in payload["output"] if item["type"] == "function_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}

    async def test_response_completed_carries_reasoning_usage(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        events = await _responses_stream_events(chunks)

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        usage = completed[0]["response"]["usage"]
        assert usage["output_tokens_details"]["reasoning_tokens"] == 2

    async def test_exactly_one_terminal_event_is_emitted(self):
        chunks = _parsed_chunks(THINKING_THEN_ANSWER, prompt="user turn")
        events = await _responses_stream_events(chunks)

        terminal = [
            e for e in events if e["type"] in {"response.completed", "response.failed"}
        ]
        assert len(terminal) == 1

    async def test_no_dsml_markup_reaches_serialized_events(self):
        chunks = _parsed_chunks(
            THINKING_THEN_TOOL_CALL, prompt="user turn", finish_reason=None
        )
        events = await _responses_stream_events(chunks)

        for event in events:
            if event["type"] == "response.function_call_arguments.delta":
                continue
            _assert_no_markup(json.dumps(event), f"event {event['type']}")


# --------------------------------------------------------------------------- #
# Step 2: request-to-prompt adapters
# --------------------------------------------------------------------------- #


async def _chat_prompt(
    *,
    messages: list[ChatCompletionMessage],
    reasoning_effort: ReasoningEffort | None = None,
    enable_thinking: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    request = ChatCompletionRequest(
        model=V4_MODEL_ID,
        messages=messages,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    params = await chat_request_to_text_generation(request)
    return apply_chat_template(_tokenizer(), params)


async def _responses_prompt(
    *,
    input_messages: list[ResponseInputMessage],
    reasoning_effort: ReasoningEffort | None = None,
    enable_thinking: bool | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": str(V4_MODEL_ID),
        "input": [m.model_dump() for m in input_messages],
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    request = ResponsesRequest.model_validate(payload)
    params = await responses_request_to_text_generation(request)
    return apply_chat_template(_tokenizer(), params)


def _user(text: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="user", content=text)


WEATHER_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class TestReasoningEffortSelection:
    """Breaks caught: mapping `xhigh` onto the V4 `high` tier, which is the
    Task 2 defect and makes OMLX's `max` tier unreachable."""

    async def test_high_and_xhigh_select_different_v4_tiers(self):
        high = await _chat_prompt(
            messages=[_user("Explain gravity.")], reasoning_effort="high"
        )
        xhigh = await _chat_prompt(
            messages=[_user("Explain gravity.")], reasoning_effort="xhigh"
        )

        assert high != xhigh

    @pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
    async def test_each_effort_emits_a_nonempty_prefix(self, effort: str):
        prompt = await _chat_prompt(
            messages=[_user("Explain gravity.")],
            reasoning_effort=cast(ReasoningEffort, effort),
        )
        plain = await _chat_prompt(
            messages=[_user("Explain gravity.")], enable_thinking=False
        )

        assert prompt != plain

    async def test_thinking_disabled_selects_chat_mode(self):
        thinking_off = await _chat_prompt(
            messages=[_user("Explain gravity.")], enable_thinking=False
        )
        thinking_on = await _chat_prompt(
            messages=[_user("Explain gravity.")], enable_thinking=True
        )

        assert not thinking_off.rstrip().endswith(THINK_START)
        assert thinking_on.rstrip().endswith(THINK_START)


class TestPromptStructure:
    """Breaks caught: the Task 1 relocation (a mid-conversation system message
    consolidated into the leading system prompt instead of rendering as
    `<｜latest_reminder｜>`) and the Task 3 channel defect (a trailing
    assistant prefill rendered inside the reasoning channel)."""

    async def test_mid_conversation_reminder_becomes_latest_reminder(self):
        """The supported `user -> system -> (assistant | end)` placement, which
        is the shape Claude Code emits."""
        prompt = await _chat_prompt(
            messages=[
                ChatCompletionMessage(role="system", content="You are helpful."),
                _user("Hello."),
                ChatCompletionMessage(role="system", content="Plan mode"),
            ]
        )

        assert "｜latest_reminder｜" in prompt
        assert "Plan mode" in prompt
        # The reminder is relocated to immediately before the user turn it
        # qualifies, not left where it was sent and not merged into the system
        # prompt.
        assert prompt.index("｜latest_reminder｜") < prompt.index("<｜User｜>")
        assert "You are helpful.\nPlan mode" not in prompt

    async def test_unsupported_reminder_placement_falls_back_to_consolidation(self):
        """Pinned known limitation, not a defect: `relocate_mid_system_messages`
        rewrites only the `user -> system -> (assistant | end)` shape. An
        `assistant -> system -> user` reminder is out of contract, so the
        encoder consolidates it into the leading system prompt and warns that
        the prefix is rewritten. Asserted so a future contract widening is a
        deliberate, visible change rather than a silent one."""
        prompt = await _chat_prompt(
            messages=[
                ChatCompletionMessage(role="system", content="You are helpful."),
                _user("Hello."),
                ChatCompletionMessage(role="assistant", content="Hi!"),
                ChatCompletionMessage(role="system", content="Plan mode"),
                _user("Continue."),
            ]
        )

        assert "｜latest_reminder｜" not in prompt
        assert "You are helpful.\nPlan mode" in prompt

    async def test_trailing_prefill_lands_after_think_end_and_before_eos(self):
        prompt = await _chat_prompt(
            messages=[
                _user("Write a haiku."),
                ChatCompletionMessage(role="assistant", content="Silent pond,"),
            ],
            enable_thinking=True,
        )

        assert "Silent pond," in prompt
        prefill_at = prompt.rindex("Silent pond,")
        think_end_at = prompt.rindex(THINK_END)
        assert think_end_at < prefill_at, (
            "prefill must sit in the content channel, after </think>"
        )
        assert not prompt[prefill_at:].strip().endswith("｜end▁of▁sentence｜>")

    async def test_reminder_preserves_the_append_only_prefix(self):
        """The offline counterpart of plan Step 7. Reasoning is retained only
        in tool conversations (`encode_messages` flips `drop_thinking` off when
        any message carries tools), so the append-only prefix invariant is a
        tool-conversation property. A historical turn plus a newly appended
        user turn and supported reminder must extend the previous prompt
        byte-for-byte, or the KV cache cannot be reused."""
        history: list[ChatCompletionMessage] = [
            ChatCompletionMessage(role="system", content="You are helpful."),
            _user("Weather in Paris?"),
        ]
        first = await _chat_prompt(messages=history, tools=WEATHER_TOOL)
        extended = await _chat_prompt(
            messages=[
                *history,
                ChatCompletionMessage(
                    role="assistant",
                    content="It is sunny.",
                    reasoning_content="Check the tool.",
                ),
                _user("And Rome?"),
                ChatCompletionMessage(role="system", content="Plan mode"),
            ],
            tools=WEATHER_TOOL,
        )

        assert extended.startswith(first), (
            "appending a turn must not rewrite the cached prefix"
        )
        assert "｜latest_reminder｜" in extended


class TestCrossEndpointAgreement:
    """Break caught: either request adapter drifting so that the same logical
    conversation renders differently through the two endpoints."""

    async def test_identical_conversations_render_identically(self):
        chat = await _chat_prompt(
            messages=[
                ChatCompletionMessage(role="system", content="You are helpful."),
                _user("What is the capital of France?"),
            ],
            reasoning_effort="high",
        )
        responses = await _responses_prompt(
            input_messages=[
                ResponseInputMessage(role="system", content="You are helpful."),
                ResponseInputMessage(
                    role="user", content="What is the capital of France?"
                ),
            ],
            reasoning_effort="high",
        )

        assert chat == responses

    async def test_thinking_disabled_agrees_across_endpoints(self):
        chat = await _chat_prompt(messages=[_user("Hello.")], enable_thinking=False)
        responses = await _responses_prompt(
            input_messages=[ResponseInputMessage(role="user", content="Hello.")],
            enable_thinking=False,
        )

        assert chat == responses


def test_task_params_carry_the_target_model() -> None:
    """Guards the fixtures above: every prompt assertion is meaningless if the
    request adapter stops selecting the V4 encoder branch."""
    params = TextGenerationTaskParams(
        model=V4_MODEL_ID,
        input=[],
        enable_thinking=True,
    )
    assert "deepseek-v4" in params.model.normalize().lower()
