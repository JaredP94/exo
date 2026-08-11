import json
from collections.abc import Generator
from typing import Any

from exo.api.types import CompletionTokensDetails, PromptTokensDetails, Usage
from exo.shared.types.worker.runner_response import (
    FinishReason,
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
    dsml_token as v4_dsml_token,
)
from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
    tool_calls_block_name as v4_tool_calls_block,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import (
    DSML_TOKEN,
    THINKING_END,
    THINKING_START,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    count_reasoning_tokens,
    parse_deepseek_v32,
    parse_thinking_models,
    parse_tool_calls,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_response(
    text: str, token: int, finish_reason: FinishReason | None = None
) -> GenerationResponse:
    return GenerationResponse(
        text=text, token=token, finish_reason=finish_reason, usage=None
    )


def _queue_source(
    tokens: list[GenerationResponse],
) -> Generator[GenerationResponse | None]:
    for token in tokens:
        yield token
        yield None
    while True:
        yield None


def _step_until_finish(
    parser_gen: Generator[GenerationResponse | ToolCallResponse | None],
    max_steps: int = 200,
) -> list[GenerationResponse | ToolCallResponse]:
    results: list[GenerationResponse | ToolCallResponse] = []
    for _ in range(max_steps):
        try:
            result = next(parser_gen)
        except StopIteration:
            break
        if result is None:
            continue
        results.append(result)
        if isinstance(result, GenerationResponse) and result.finish_reason is not None:
            return results
        if isinstance(result, ToolCallResponse):
            return results
    return results


def _got_finish(results: list[GenerationResponse | ToolCallResponse]) -> bool:
    for r in results:
        if isinstance(r, ToolCallResponse):
            return True
        if r.finish_reason is not None:
            return True
    return False


# ── parse_deepseek_v32 ──────────────────────────────────────────


class TestDeepSeekV32FinishReason:
    def test_finish_reason_with_buffered_dsml_prefix(self):
        tokens = [
            _make_response("Hello! The answer is x", 0),
            _make_response("<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert "Hello" in full_text
        assert "<" in full_text

    def test_finish_reason_completes_tool_call_block(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 2),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n',
                3,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 4),
            _make_response(TOOL_CALLS_END, 5, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_mid_tool_call_before_close(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(
                f'<{DSML_TOKEN}invoke name="get_weather">\n', 2, finish_reason="stop"
            ),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_single_token_complete_dsml_block(self):
        dsml_block = (
            f"{TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="get_weather">\n'
            f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n'
            f"</{DSML_TOKEN}invoke>\n"
            f"{TOOL_CALLS_END}"
        )
        tokens = [_make_response(dsml_block, 0, finish_reason="stop")]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("I need to think about this", 1),
            _make_response(" carefully before responding", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_after_thinking_then_tool_call(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("Let me check the weather.", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n\n", 3),
            _make_response(TOOL_CALLS_START, 4),
            _make_response("\n", 5),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 6),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
                7,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 8),
            _make_response(TOOL_CALLS_END, 9, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_normal_text_no_buffering(self):
        tokens = [
            _make_response("Hello", 0),
            _make_response(" world", 1),
            _make_response("!", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert full_text == "Hello world!"

    def test_finish_reason_multiple_buffered_prefix_tokens(self):
        tokens = [
            _make_response("text ", 0),
            _make_response("<", 1),
            _make_response("not a tag", 2),
            _make_response(" more<", 3),
            _make_response("", 4, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)


# ── parse_thinking_models ────────────────────────────────────────


class TestThinkingModelsFinishReason:
    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("reasoning here", 1),
            _make_response("more reasoning", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)
        last_gen = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(last_gen) == 1
        assert last_gen[0].is_thinking is False

    def test_finish_reason_after_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("hmm", 1),
            _make_response("</think>", 2),
            _make_response("The answer is 42.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_starts_in_thinking(self):
        tokens = [
            _make_response("still thinking", 0),
            _make_response("</think>", 1),
            _make_response("done", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=True,
            )
        )
        assert _got_finish(results)

    def test_reasoning_tokens_counted(self):
        """reasoning_tokens in Usage reflects the number of thinking tokens."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("<think>", 0),
            _make_response("let me", 1),
            _make_response(" think", 2),
            _make_response("</think>", 3),
            GenerationResponse(text="42", token=4, finish_reason="stop", usage=usage),
        ]
        results = _step_until_finish(
            count_reasoning_tokens(
                parse_thinking_models(
                    _queue_source(tokens),
                    think_start="<think>",
                    think_end="</think>",
                    starts_in_thinking=False,
                )
            )
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2

    def test_reasoning_tokens_starts_in_thinking(self):
        """reasoning_tokens counts correctly when starts_in_thinking=True."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("hmm", 0),
            _make_response("ok", 1),
            _make_response("</think>", 2),
            GenerationResponse(
                text="answer", token=3, finish_reason="stop", usage=usage
            ),
        ]
        results = _step_until_finish(
            count_reasoning_tokens(
                parse_thinking_models(
                    _queue_source(tokens),
                    think_start="<think>",
                    think_end="</think>",
                    starts_in_thinking=True,
                )
            )
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2


# ── parse_tool_calls (generic) ──────────────────────────────────


def _dummy_parser_fn(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


_dummy_parser = make_mlx_parser("<tool_call>", "</tool_call>", _dummy_parser_fn)


class TestGenericToolCallsFinishReason:
    def test_finish_reason_after_complete_tool_call(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("body", 1),
            _make_response("</tool_call>", 2),
            _make_response("extra text", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1

    def test_finish_reason_mid_tool_call_unclosed(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("partial content", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_no_tool_calls(self):
        tokens = [
            _make_response("Just", 0),
            _make_response(" a", 1),
            _make_response(" normal", 2),
            _make_response(" response.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)


# ── Double parser chain (parse_thinking_models → parse_deepseek_v32) ──


class TestDeepSeekV32StartsInThinking:
    """Regression tests for deepseek v3.2 where the chat template appends
    <think> to the prompt so the model starts already inside a thinking block.
    """

    def test_reasoning_tagged_when_starts_in_thinking(self):
        tokens = [
            _make_response("let me", 0),
            _make_response(" think", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n", 3),
            _make_response("42", 4, finish_reason="stop"),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(parse_deepseek_v32(thinking))
        gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is None
        ]
        texts = [(r.text, r.is_thinking) for r in gens]
        assert texts == [("let me", True), (" think", True), ("\n", False)]
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].text == "42"
        assert final[0].is_thinking is False

    def test_starts_in_thinking_then_tool_call(self):
        tokens = [
            _make_response("need weather", 0),
            _make_response(THINKING_END, 1),
            _make_response("\n\n", 2),
            _make_response(TOOL_CALLS_START, 3),
            _make_response("\n", 4),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 5),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
                6,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 7),
            _make_response(TOOL_CALLS_END, 8, finish_reason="stop"),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(parse_deepseek_v32(thinking))
        reasoning_gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse)
            and r.finish_reason is None
            and r.is_thinking
        ]
        assert [r.text for r in reasoning_gens] == ["need weather"]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_reasoning_tokens_counted_starts_in_thinking(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("reasoning", 0),
            _make_response(" more", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n", 3),
            GenerationResponse(text="42", token=4, finish_reason="stop", usage=usage),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(
            count_reasoning_tokens(parse_deepseek_v32(thinking))
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2


class TestBatchGeneratorSingleNext:
    def test_finish_reason_with_buffered_tokens_drain_loop(self):
        from exo.worker.runner.llm_inference.batch_generator import GeneratorQueue

        queue: GeneratorQueue[GenerationResponse] = GeneratorQueue()
        parser = parse_deepseek_v32(queue.gen())

        tokens = [
            _make_response("Hello ", 0),
            _make_response(" `<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]

        collected: list[GenerationResponse | ToolCallResponse] = []
        for token in tokens:
            queue.push(token)
            while (parsed := next(parser, None)) is not None:
                collected.append(parsed)
            if token.finish_reason is not None:
                break

        assert _got_finish(collected), (
            f"No finish_reason in collected: {[(type(r).__name__, getattr(r, 'finish_reason', None) if isinstance(r, GenerationResponse) else 'tool') for r in collected]}"
        )


# ── parse_thinking_models prefix buffering ──────────────────────


def _drain_text(
    results: list[GenerationResponse | ToolCallResponse],
) -> str:
    return "".join(
        r.text
        for r in results
        if isinstance(r, GenerationResponse) and r.finish_reason is None
    )


class TestThinkingModelsPrefixBuffering:
    def test_lone_lt_is_preserved(self):
        tokens = [
            _make_response("<", 0),
            _make_response("function", 1),
            _make_response(">", 2),
            _make_response("", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "<function>"
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert all(not r.is_thinking for r in gens)

    def test_lone_lt_slash_is_preserved(self):
        tokens = [
            _make_response("</", 0),
            _make_response("parameter", 1),
            _make_response(">", 2),
            _make_response("", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "</parameter>"

    def test_partial_prefix_then_diverge(self):
        tokens = [
            _make_response("<", 0),
            _make_response("t", 1),
            _make_response("h", 2),
            _make_response("other", 3),
            _make_response("", 4, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "<thother"

    def test_real_think_tag_still_swallowed(self):
        tokens = [
            _make_response("<", 0),
            _make_response("think", 1),
            _make_response(">", 2),
            _make_response("body", 3),
            _make_response("</", 4),
            _make_response("think", 5),
            _make_response(">", 6),
            _make_response("after", 7),
            _make_response("", 8, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is None
        ]
        texts = [(r.text, r.is_thinking) for r in gens]
        assert texts == [("body", True), ("after", False)]

    def test_finish_reason_flushes_buffer(self):
        tokens = [
            _make_response("<", 0),
            _make_response("", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert len(gens) == 2
        assert gens[0].text == "<"
        assert gens[0].is_thinking is False
        assert gens[0].finish_reason is None
        assert gens[1].finish_reason == "stop"
        assert gens[1].is_thinking is False

    def test_tool_call_after_prefix_tokens_parses(self):
        def _capture_parser(text: str) -> dict[str, Any]:
            return {"name": "captured", "arguments": {"raw": text}}

        tool_parser = make_mlx_parser("<tool_call>", "</tool_call>", _capture_parser)

        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("\n", 1),
            _make_response("<", 2),
            _make_response("function", 3),
            _make_response("=glob", 4),
            _make_response(">", 5),
            _make_response("\n", 6),
            _make_response("<", 7),
            _make_response("parameter", 8),
            _make_response("=pattern", 9),
            _make_response(">", 10),
            _make_response("**/*", 11),
            _make_response("</", 12),
            _make_response("parameter", 13),
            _make_response(">", 14),
            _make_response("</", 15),
            _make_response("function", 16),
            _make_response(">", 17),
            _make_response("</tool_call>", 18, finish_reason="stop"),
        ]

        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start="<think>",
            think_end="</think>",
            starts_in_thinking=False,
        )
        results = _step_until_finish(
            parse_tool_calls(thinking, tool_parser, tools=None)
        )

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        raw = json.loads(tool_results[0].tool_calls[0].arguments)["raw"]  # pyright: ignore[reportAny]
        assert "<function=glob>" in raw
        assert "<parameter=pattern>" in raw
        assert "</parameter>" in raw
        assert "</function>" in raw


# ── Task 5: split delimiters ─────────────────────────────────────
#
# Which test catches which break (verified by running this file against the
# unfixed parser -- a test that stays green there is noted as a contract guard,
# not a regression guard):
#
#   * test_start_marker_split_across_the_final_two_responses - the finish branch
#     inspected only `response.text`, never `accumulated`, so a start marker
#     split across the penultimate and final response was flushed from the
#     pending buffer as visible text and the tool call was lost.
#   * test_visible_text_before_a_marker_in_the_same_chunk_survives - prose
#     arriving in the SAME response as the marker had no buffered response to
#     carry it and was dropped.
#   * test_visible_text_before_a_split_marker_in_one_window and
#     test_ordinary_content_containing_a_think_tag - `parse_thinking_models`
#     compared the WHOLE accumulated buffer against `marker[:len]`, so a marker
#     preceded by visible text in the same window was never recognised.
#     The `*_split_at_every_boundary_*` tests do NOT catch this: they always
#     end a chunk at the marker, which the old prefix comparison handled.
#   * test_marker_followed_by_prose_in_one_response and
#     test_two_markers_in_one_window - the transition test was `endswith`, so a
#     marker with anything after it in the same response was never recognised:
#     the marker leaked AND `is_thinking` stayed wedged for the whole stream.
#   * test_marker_completed_by_the_terminal_response,
#     test_whole_marker_carried_by_the_terminal_response and
#     test_terminal_response_carrying_prose_marker_and_prose - the thinking
#     finish branch flushed the held partial marker as visible text. These
#     assert over `_all_text`, not `_drain_text`: the leak lands ON the terminal
#     response, which `_drain_text` filters out.
#   * test_every_boundary_of_a_complete_block_yields_one_tool_call - the same
#     finish-branch break as the first entry, across every split point.
#   * test_metadata_is_emitted_once_per_response - the split and segment paths
#     copied the source response wholesale, so one token's logprob appeared on
#     several chunks and `Logprobs.content` outnumbered the tokens generated.
#   * test_invalidated_hold_does_not_split_a_response and
#     test_a_zero_length_response_is_still_forwarded - these guard regressions
#     introduced EARLIER IN THIS TASK, so they are green against the
#     pre-task parser as well: holding a marker candidate then re-emitting head
#     and tail duplicated a token id, and the emit loop exiting on
#     `remaining > 0` dropped an empty response queued behind the held text.
#   * complete_block_in_one_chunk, end_marker_split, two_tool_calls,
#     incomplete_block_at_finish, terminal_*, *_split_at_every_boundary_* -
#     contract guards, green before and after; they pin behaviour the rewrite
#     had to preserve.
#   * test_text_after_the_end_marker_is_dropped and
#     test_a_truncated_block_is_emitted_as_text - pinned known limitations, see
#     their docstrings.


# `dsml_encoding.TOOL_CALLS_START` is the V3.2 `function_calls` form. V4 uses
# `tool_calls`, so build the V4 markers from the shared V4 encoder constants —
# the same constants `parse_deepseek_v4` now uses.
V4_TOOL_CALLS_START = f"<{v4_dsml_token}{v4_tool_calls_block}>"
V4_TOOL_CALLS_END = f"</{v4_dsml_token}{v4_tool_calls_block}>"


def _boundary_partitions(text: str) -> list[list[str]]:
    """Every two-way split of `text`, plus the fully fragmented split.

    Every partition here ENDS a chunk at the end of `text`. Use
    `_partitions_with_trailing` for the harder shape where the marker's tail
    shares a response with what follows it.
    """
    partitions = [[text[:i], text[i:]] for i in range(1, len(text))]
    partitions.append(list(text))
    return partitions


def _all_text(results: list[GenerationResponse | ToolCallResponse]) -> str:
    """Every character the client sees, INCLUDING the terminal response's.

    `_drain_text` excludes the terminal response, so a marker leaking on the
    terminal chunk is invisible to it.
    """
    return "".join(r.text for r in results if isinstance(r, GenerationResponse))


def _partitions_with_trailing(marker: str, trailing: str) -> list[list[str]]:
    """Every split of `marker` where the tail carries `trailing` in one chunk.

    Includes the zero-split case, i.e. marker and trailing prose in a single
    response, which no `_boundary_partitions` case produces.
    """
    return [
        [marker[:i], marker[i:] + trailing] if i else [marker + trailing]
        for i in range(len(marker))
    ]


def _responses(chunks: list[str]):
    """Chunks followed by a separate empty terminal response."""
    tokens = [_make_response(chunk, i) for i, chunk in enumerate(chunks)]
    tokens.append(_make_response("", len(chunks), finish_reason="stop"))
    return tokens


def _responses_finishing_on_last(chunks: list[str]):
    """The final CONTENT chunk itself carries the finish reason.

    This is the shape that exercises the finish branch: with a separate empty
    terminal response the ordinary path has already consumed the markers.
    """
    return [
        _make_response(chunk, i, finish_reason="stop" if i == len(chunks) - 1 else None)
        for i, chunk in enumerate(chunks)
    ]


_ONE_CALL = (
    f"{V4_TOOL_CALLS_START}\n"
    f'<{DSML_TOKEN}invoke name="get_weather">\n'
    f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n'
    f"</{DSML_TOKEN}invoke>\n"
    f"{V4_TOOL_CALLS_END}"
)


def _v4_results(chunks: list[str], *, finish_on_last: bool = True):
    from exo.worker.runner.llm_inference.model_output_parsers import parse_deepseek_v4

    tokens = (
        _responses_finishing_on_last(chunks) if finish_on_last else _responses(chunks)
    )
    return _step_until_finish(parse_deepseek_v4(_queue_source(tokens)))


def _step_past_tool_calls(
    tokens: list[GenerationResponse], max_steps: int = 200
) -> list[GenerationResponse | ToolCallResponse]:
    """Drain to the terminal response rather than stopping at a tool call.

    `_step_until_finish` returns as soon as a `ToolCallResponse` appears, which
    hides anything emitted after it.
    """
    from exo.worker.runner.llm_inference.model_output_parsers import parse_deepseek_v4

    parser_gen = parse_deepseek_v4(_queue_source(tokens))
    results: list[GenerationResponse | ToolCallResponse] = []
    for _ in range(max_steps):
        try:
            result = next(parser_gen)
        except StopIteration:
            break
        if result is None:
            continue
        results.append(result)
        if isinstance(result, GenerationResponse) and result.finish_reason is not None:
            break
    return results


class TestDeepSeekV4SplitDelimiters:
    def test_complete_block_in_one_chunk(self):
        results = _v4_results([_ONE_CALL])

        calls = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(calls) == 1
        assert calls[0].tool_calls[0].name == "get_weather"
        assert json.loads(calls[0].tool_calls[0].arguments) == {"city": "Tokyo"}
        assert DSML_TOKEN not in _drain_text(results)

    def test_start_marker_split_across_the_final_two_responses(self):
        head, tail = V4_TOOL_CALLS_START[:6], V4_TOOL_CALLS_START[6:]
        body = _ONE_CALL[len(V4_TOOL_CALLS_START) :]

        results = _v4_results(chunks=[head, tail + body])

        calls = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(calls) == 1, f"expected one tool call, got {results}"
        assert DSML_TOKEN not in _drain_text(results)

    def test_visible_text_before_a_split_start_marker(self):
        head, tail = V4_TOOL_CALLS_START[:4], V4_TOOL_CALLS_START[4:]
        body = _ONE_CALL[len(V4_TOOL_CALLS_START) :]

        results = _v4_results(chunks=["Let me check. " + head, tail + body])

        calls = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(calls) == 1
        assert _drain_text(results) == "Let me check. "

    def test_visible_text_before_a_marker_in_the_same_chunk_survives(self):
        """Pre-existing defect found while writing these tests.

        The pre-marker flush walked only the pending buffer, so prose arriving in
        the SAME response as the marker had no buffered response to carry it and
        was dropped: the model says "Let me check." and the user sees nothing.
        """
        results = _v4_results(chunks=["Let me check. " + _ONE_CALL])

        assert _drain_text(results) == "Let me check. "
        assert len([r for r in results if isinstance(r, ToolCallResponse)]) == 1

    def test_end_marker_split_across_the_final_two_responses(self):
        cut = len(_ONE_CALL) - 5
        results = _v4_results(chunks=[_ONE_CALL[:cut], _ONE_CALL[cut:]])

        calls = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(calls) == 1
        assert DSML_TOKEN not in _drain_text(results)

    def test_two_tool_calls_in_a_split_block(self):
        two = (
            f"{V4_TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="first">\n</{DSML_TOKEN}invoke>\n'
            f'<{DSML_TOKEN}invoke name="second">\n</{DSML_TOKEN}invoke>\n'
            f"{V4_TOOL_CALLS_END}"
        )
        cut = len(two) // 2

        results = _v4_results(chunks=[two[:cut], two[cut:]])

        calls = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(calls) == 1
        assert [c.name for c in calls[0].tool_calls] == ["first", "second"]

    def test_incomplete_block_at_finish_emits_one_terminal_result(self):
        truncated = _ONE_CALL[: -len(V4_TOOL_CALLS_END) - 3]

        results = _v4_results(chunks=[truncated])

        assert _got_finish(results), "must not hang"
        assert len([r for r in results if isinstance(r, ToolCallResponse)]) == 0
        terminal = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(terminal) == 1

    def test_every_boundary_of_a_complete_block_yields_one_tool_call(self):
        for chunks in _boundary_partitions(_ONE_CALL):
            results = _v4_results(chunks)
            calls = [r for r in results if isinstance(r, ToolCallResponse)]
            assert len(calls) == 1, f"partition {chunks[:2]}... gave {results}"
            assert calls[0].tool_calls[0].name == "get_weather"
            assert DSML_TOKEN not in _drain_text(results), f"markup leaked: {chunks}"

    def test_every_boundary_with_a_separate_terminal_response(self):
        """The other terminal shape: content chunks, then an empty stop.

        Here the ordinary (non-finish) path consumes the markers, so this pins
        that the finish-branch rewrite did not change the common case.
        """
        for chunks in _boundary_partitions(_ONE_CALL):
            results = _v4_results(chunks, finish_on_last=False)
            calls = [r for r in results if isinstance(r, ToolCallResponse)]
            assert len(calls) == 1, f"partition {chunks[:2]}... gave {results}"
            assert DSML_TOKEN not in _drain_text(results), f"markup leaked: {chunks}"

    def test_prose_before_the_marker_survives_at_every_split(self):
        prose = "Let me check. "
        for chunks in _partitions_with_trailing(
            V4_TOOL_CALLS_START, _ONE_CALL[len(V4_TOOL_CALLS_START) :]
        ):
            chunks = [prose + chunks[0], *chunks[1:]]
            results = _v4_results(chunks)
            assert _drain_text(results) == prose, f"{chunks} lost prose"
            assert len([r for r in results if isinstance(r, ToolCallResponse)]) == 1

    def test_text_after_the_end_marker_is_dropped(self):
        """PINNED KNOWN LIMITATION, unchanged from before this task.

        Everything from the start marker onwards is handed to the DSML body
        parser, which tolerates and discards trailing prose, so text following
        `tool_calls_end` WITHIN THE SAME accumulated block never reaches the
        client. Text arriving in a later response does reach it, which the
        second half of this test pins. Left as-is because the fix belongs with
        the block-boundary rework, not the delimiter-splitting fix: DeepSeek
        emits the tool-call block last, so in practice this text is empty.
        """
        same_response = _v4_results([_ONE_CALL + "Done."])

        assert len([r for r in same_response if isinstance(r, ToolCallResponse)]) == 1
        assert "Done." not in _all_text(same_response), "the limitation"

        later_response = _step_past_tool_calls(_responses([_ONE_CALL, "Done."]))

        assert "Done." in _all_text(later_response), "only the same-block case drops"

    def test_a_truncated_block_is_emitted_as_text(self):
        """PINNED KNOWN LIMITATION, unchanged from before this task.

        A block that never closes is emitted verbatim on the terminal response
        rather than suppressed, so raw markup reaches the client. Deliberate:
        suppressing it would silently swallow whatever the model did produce.
        The rewrite only moved WHICH chunk carries it, so this is a contract
        guard, not a regression guard.
        """
        cut = len(V4_TOOL_CALLS_START) - 2
        results = _v4_results(
            [_ONE_CALL[:cut], _ONE_CALL[cut : len(V4_TOOL_CALLS_START)]]
        )

        assert _got_finish(results), "must not hang"
        assert len([r for r in results if isinstance(r, ToolCallResponse)]) == 0
        assert _all_text(results) == V4_TOOL_CALLS_START


class TestThinkingMarkerSplitBoundaries:
    def _thinking_results(self, chunks: list[str], *, starts_in_thinking: bool = False):
        return _step_until_finish(
            parse_thinking_models(
                _queue_source(_responses(chunks)),
                think_start=THINKING_START,
                think_end=THINKING_END,
                starts_in_thinking=starts_in_thinking,
            )
        )

    def _finishing_on_last(self, chunks: list[str], starts_in_thinking: bool):
        """The final CONTENT chunk carries the finish reason."""
        return _step_until_finish(
            parse_thinking_models(
                _queue_source(_responses_finishing_on_last(chunks)),
                think_start=THINKING_START,
                think_end=THINKING_END,
                starts_in_thinking=starts_in_thinking,
            )
        )

    @staticmethod
    def _terminal(
        results: list[GenerationResponse | ToolCallResponse],
    ) -> GenerationResponse:
        terminal = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(terminal) == 1, f"expected exactly one terminal: {results}"
        return terminal[0]

    @staticmethod
    def _reasoning_text(results: list[GenerationResponse | ToolCallResponse]) -> str:
        return "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and r.is_thinking
        )

    @staticmethod
    def _content_text(results: list[GenerationResponse | ToolCallResponse]) -> str:
        return "".join(
            r.text
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        )

    def test_think_start_split_at_every_boundary_is_swallowed(self):
        for chunks in _boundary_partitions(THINKING_START):
            results = self._thinking_results([*chunks, "reasoning"])
            text = _drain_text(results)
            assert THINKING_START not in text, f"marker leaked for {chunks}"
            assert text == "reasoning", f"{chunks} -> {text!r}"
            reasoning = [
                r
                for r in results
                if isinstance(r, GenerationResponse) and r.text == "reasoning"
            ]
            assert reasoning and all(r.is_thinking for r in reasoning)

    def test_think_end_split_at_every_boundary_is_swallowed(self):
        for chunks in _boundary_partitions(THINKING_END):
            results = self._thinking_results(
                [*chunks, "answer"], starts_in_thinking=True
            )
            text = _drain_text(results)
            assert THINKING_END not in text, f"marker leaked for {chunks}"
            assert text == "answer", f"{chunks} -> {text!r}"

    def test_visible_text_before_a_split_marker_in_one_window(self):
        results = self._thinking_results(["hello <th", "ink>", "reasoning"])

        assert _drain_text(results) == "hello reasoning"
        assert THINKING_START not in _drain_text(results)

    def test_terminal_while_inside_thinking_is_not_thinking(self):
        results = self._thinking_results(["deep"], starts_in_thinking=True)

        terminal = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(terminal) == 1
        assert terminal[0].is_thinking is False

    def test_terminal_immediately_after_think_end(self):
        results = self._thinking_results([*list(THINKING_END)], starts_in_thinking=True)

        terminal = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(terminal) == 1
        assert terminal[0].is_thinking is False
        assert THINKING_END not in _drain_text(results)

    def test_ordinary_content_containing_a_think_tag(self):
        """Documents the deliberate interpretation: EXO sees text, not token ids,
        so a literal `<think>` in content is indistinguishable from the marker
        and is treated as the marker.
        """
        results = self._thinking_results(["the tag <think>", "x"])

        assert "<think>" not in _drain_text(results)

    # ── marker sharing a response with the text that FOLLOWS it ──────────
    #
    # The transition test used to be `endswith`, so none of these were
    # recognised: the marker leaked into visible content and `is_thinking`
    # stayed wedged for the remainder of the stream.

    def test_marker_followed_by_prose_in_one_response(self):
        results = self._thinking_results(
            ["</think>\n\n", "The answer."], starts_in_thinking=True
        )

        assert THINKING_END not in _drain_text(results)
        assert _drain_text(results) == "\n\nThe answer."
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert all(not r.is_thinking for r in gens), "thinking must have ended"

    def test_prose_marker_and_prose_in_a_single_response(self):
        results = self._thinking_results(
            ["Let me think.</think>\n\nThe answer is 42."], starts_in_thinking=True
        )

        assert THINKING_END not in _drain_text(results)
        assert self._reasoning_text(results) == "Let me think."
        assert self._content_text(results) == "\n\nThe answer is 42."

    def test_marker_with_trailing_prose_at_every_split(self):
        for chunks in _partitions_with_trailing(THINKING_END, "answer"):
            results = self._thinking_results(chunks, starts_in_thinking=True)
            assert THINKING_END not in _drain_text(results), f"leaked: {chunks}"
            assert self._content_text(results) == "answer", f"{chunks}"

    def test_two_markers_in_one_window(self):
        """The end of one marker and the start of the next in one response."""
        results = self._thinking_results(
            ["</thin", "k><think>", "answer"], starts_in_thinking=True
        )

        text = _drain_text(results)
        assert THINKING_END not in text and THINKING_START not in text
        assert self._reasoning_text(results) == "answer"

    # ── marker completed by the terminal response ────────────────────────

    def test_marker_completed_by_the_terminal_response(self):
        results = self._finishing_on_last(["reason", "</thi", "nk>"], True)

        assert THINKING_END not in _all_text(results)
        assert self._reasoning_text(results) == "reason"
        assert _got_finish(results)

    def test_whole_marker_carried_by_the_terminal_response(self):
        results = self._finishing_on_last(["reason", "</think>"], True)

        assert THINKING_END not in _all_text(results), "the leak lands on the terminal"
        assert self._reasoning_text(results) == "reason"
        assert _got_finish(results)

    def test_terminal_response_carrying_prose_marker_and_prose(self):
        results = self._finishing_on_last(["reason", "more</think>done"], True)

        assert THINKING_END not in _all_text(results)
        assert self._reasoning_text(results) == "reasonmore"
        assert self._content_text(results) == "done"

    def test_terminal_with_no_marker_is_forwarded_with_its_text(self):
        """Guards the common case against the marker-aware finish branch."""
        results = self._finishing_on_last(["reason", "ing"], True)

        terminal = self._terminal(results)
        assert terminal.text == "ing", "terminal text must not be re-carried"
        assert terminal.is_thinking is False

    # ── metadata fidelity ───────────────────────────────────────────────

    def test_invalidated_hold_does_not_split_a_response(self):
        """A response held as a marker candidate then released must stay whole.

        Re-emitting its head and tail separately duplicates its token id and
        logprob, and `count_reasoning_tokens` counts responses, so the reported
        reasoning tokens exceed the tokens actually generated.
        """
        results = self._thinking_results(
            ["if", " a", " <", " b"], starts_in_thinking=True
        )

        gens = [r for r in results if isinstance(r, GenerationResponse)]
        tokens = [r.token for r in gens]
        assert len(tokens) == len(set(tokens)), f"token id emitted twice: {tokens}"
        assert self._reasoning_text(results) == "if a < b"
        assert sum(1 for r in gens if r.is_thinking) == 4

    def test_metadata_is_emitted_once_per_response(self):
        """One response is one token, so one logprob and one usage.

        Where a response's text has to be cut around a marker, the pieces after
        the first must drop the metadata: `collect_chat_response` appends a
        `Logprobs.content` entry for every chunk carrying a logprob, so
        repeating it reports more logprobs than there were tokens.
        """
        tokens = [
            GenerationResponse(
                text=" a</think>b<think>c", token=0, logprob=-0.5, usage=None
            ),
            _make_response("", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start=THINKING_START,
                think_end=THINKING_END,
                starts_in_thinking=True,
            )
        )

        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert _all_text(results) == " abc", "text must survive the cut"
        with_logprob = [r for r in gens if r.logprob is not None]
        assert len(with_logprob) == 1, f"one token, {len(with_logprob)} logprobs"
        assert with_logprob[0].token == 0

    def test_terminal_metadata_survives_a_marker_on_the_terminal_response(self):
        """The terminal chunk is where the adapter reads usage from."""
        usage = Usage(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=1),
        )
        tokens = [
            _make_response("reason", 0),
            GenerationResponse(
                text="more</think>done",
                token=1,
                logprob=-0.2,
                finish_reason="stop",
                usage=usage,
            ),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start=THINKING_START,
                think_end=THINKING_END,
                starts_in_thinking=True,
            )
        )

        gens = [r for r in results if isinstance(r, GenerationResponse)]
        terminal = self._terminal(results)
        assert terminal.usage == usage and terminal.logprob == -0.2
        assert len([r for r in gens if r.usage is not None]) == 1
        assert len([r for r in gens if r.logprob is not None]) == 1

    def test_a_zero_length_response_is_still_forwarded(self):
        """An empty response queued behind held text carries a logprobs entry."""
        tokens = [
            _make_response("a", 0),
            _make_response("", 1),
            _make_response("", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start=THINKING_START,
                think_end=THINKING_END,
                starts_in_thinking=True,
            )
        )

        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert [r.token for r in gens] == [0, 1, 2]

    def test_text_is_conserved_at_every_split_of_a_full_exchange(self):
        """No character is lost or duplicated, whatever the chunk boundaries.

        This is the buffer invariant stated in `parse_thinking_models` asserted
        from the outside: emitted text must equal the input minus whole markers.
        """
        stream = f"pre{THINKING_START}why{THINKING_END}post"
        expected = "prewhypost"
        for chunks in _boundary_partitions(stream):
            for finishing in (False, True):
                results = (
                    self._finishing_on_last(chunks, False)
                    if finishing
                    else self._thinking_results(chunks)
                )
                emitted = "".join(
                    r.text for r in results if isinstance(r, GenerationResponse)
                )
                assert emitted == expected, (
                    f"{chunks} (finish={finishing}) -> {emitted!r}"
                )

    # ── marker configuration edge cases ─────────────────────────────────

    def test_absent_markers_pass_everything_through(self):
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(_responses(["hello", " world"])),
                think_start=None,
                think_end=None,
                starts_in_thinking=False,
            )
        )

        assert _drain_text(results) == "hello world"
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert all(not r.is_thinking for r in gens)

    def test_one_marker_being_a_prefix_of_the_other(self):
        """`<t` is a prefix of `<te`, so the hold length must cover the longer."""
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(_responses(["a<", "t", "b<", "te", "c"])),
                think_start="<t",
                think_end="<te",
                starts_in_thinking=False,
            )
        )

        assert _drain_text(results) == "abc", _drain_text(results)
