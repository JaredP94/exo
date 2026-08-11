from collections.abc import Callable, Generator, Iterator
from functools import cache
from typing import Any

from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
    dsml_token as v4_dsml_token,
)
from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
    tool_calls_block_name as v4_tool_calls_block_name,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import parse_dsml_output
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def count_reasoning_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Count tokens with is_thinking=True and patch the total into Usage on the final response."""
    reasoning_tokens = 0
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, GenerationResponse) and response.is_thinking:
            reasoning_tokens += 1
        if response.usage is not None and reasoning_tokens > 0:
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={
                            "completion_tokens_details": response.usage.completion_tokens_details.model_copy(
                                update={"reasoning_tokens": reasoning_tokens}
                            )
                        }
                    )
                }
            )
        yield response


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
) -> Iterator[GenerationChunk | None]:
    generator = receiver

    normalized_id = model_id.normalize().lower()
    if issubclass(model_type, GptOssModel):
        generator = parse_gpt_oss(generator)
    elif issubclass(model_type, DeepseekV32Model) and "deepseek" in normalized_id:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )
        generator = parse_deepseek_v32(generator)
    elif issubclass(model_type, DeepseekV4Model) and "deepseek-v4" in normalized_id:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )
        generator = parse_deepseek_v4(generator)
    else:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )

        if tool_parser:
            generator = parse_tool_calls(generator, tool_parser, tools)

    generator = count_reasoning_tokens(generator)

    return map(lambda r: map_responses_to_chunks(r, model_id), generator)


def map_responses_to_chunks(
    response: GenerationResponse | ToolCallResponse | None, model_id: ModelId
) -> GenerationChunk | None:
    match response:
        case None:
            return None
        case GenerationResponse():
            if response.finish_reason == "error":
                return ErrorChunk(
                    error_message=response.text,
                    model=model_id,
                )
            else:
                finish_reason = response.finish_reason
                assert finish_reason not in (
                    "error",
                    "tool_calls",
                    "function_call",
                )
                return TokenChunk(
                    model=model_id,
                    text=response.text,
                    token_id=response.token,
                    usage=response.usage,
                    finish_reason=finish_reason,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                    is_thinking=response.is_thinking,
                )
        case ToolCallResponse():
            return ToolCallChunk(
                tool_calls=response.tool_calls,
                model=model_id,
                usage=response.usage,
                stats=response.stats,
            )


def parse_gpt_oss(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        if response is None:
            yield None
            continue
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Debug: log every token with state
        logger.debug(
            f"parse_gpt_oss token={response.token} text={response.text!r} "
            f"recipient={recipient!r} ch={ch!r} delta={delta!r} "
            f"state={stream.state} current_tool={current_tool_name!r}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    f"parse_gpt_oss yielding tool call: name={current_tool_name!r}"
                )
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        if delta:
            yield response.model_copy(
                update={"text": delta, "is_thinking": ch == "analysis"}
            )

        if response.finish_reason is not None:
            yield response


def parse_deepseek_v32(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Thinking tag handling is delegated to parse_thinking_models, which
    wraps this parser in apply_all_parsers.
    """
    from exo.worker.engines.mlx.vendor.dsml_encoding import (
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    return _parse_dsml_stream(
        responses, TOOL_CALLS_START, TOOL_CALLS_END, parse_dsml_output
    )


def parse_deepseek_v4(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    start = f"<{v4_dsml_token}{v4_tool_calls_block_name}>"
    end = f"</{v4_dsml_token}{v4_tool_calls_block_name}>"
    return _parse_dsml_stream(responses, start, end, parse_dsml_output)


def _parse_dsml_stream(
    responses: Generator[GenerationResponse | None],
    tool_calls_start: str,
    tool_calls_end: str,
    parse_body: Callable[[str], list[ToolCallItem] | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    accumulated = ""
    in_tool_call = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    def _flush_pre_marker_text(
        pre_text: str, response: GenerationResponse
    ) -> Generator[GenerationResponse]:
        """Emit the text preceding a tool-call marker, then drop the buffer.

        Buffered responses carry their own metadata, so they are reused oldest
        first. Text left over once the buffer is exhausted came from `response`
        itself — a marker arriving in the same chunk as visible prose — and is
        emitted under that response's metadata rather than dropped.
        """
        remaining = pre_text
        while remaining and pending_buffer:
            buffered = pending_buffer.pop(0)
            chunk = buffered.text
            if len(chunk) <= len(remaining):
                yield buffered
                remaining = remaining[len(chunk) :]
            else:
                yield buffered.model_copy(update={"text": remaining})
                remaining = ""
        pending_buffer.clear()
        if remaining:
            yield response.model_copy(
                update={
                    "text": remaining,
                    "finish_reason": None,
                    **_SECONDARY_PIECE,
                }
            )

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_body(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(f"DSML tool call parsing failed for: {text}")
        return response.model_copy(update={"text": text})

    for response in responses:
        if response is None:
            yield None
            continue

        if response.finish_reason is not None:
            if in_tool_call:
                yield from pending_buffer
                pending_buffer.clear()
                tool_call_text += response.text
                yield (
                    _try_parse_tool_call(tool_call_text, response)
                    if tool_calls_end in tool_call_text
                    else response.model_copy(update={"text": tool_call_text})
                )
            else:
                # `accumulated` holds the buffered text that has not been emitted
                # yet, which is where a start marker split across this response
                # and its predecessors lives. Inspecting only `response.text`
                # flushed that partial marker as visible text and lost the call.
                combined = accumulated + response.text
                if tool_calls_start in combined:
                    start_idx = combined.index(tool_calls_start)
                    yield from _flush_pre_marker_text(combined[:start_idx], response)
                    block = combined[start_idx:]
                    yield (
                        _try_parse_tool_call(block, response)
                        if tool_calls_end in block
                        else response.model_copy(update={"text": block})
                    )
                else:
                    yield from pending_buffer
                    pending_buffer.clear()
                    yield response
            # Every branch above has already emitted the buffer; clearing it
            # again keeps the tail flush after the loop from double-emitting if
            # one of them ever stops doing so.
            pending_buffer.clear()
            break

        if in_tool_call:
            tool_call_text += response.text
            if tool_calls_end in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                in_tool_call = False
                tool_call_text = ""
            continue

        accumulated += response.text

        if tool_calls_start in accumulated:
            start_idx = accumulated.index(tool_calls_start)
            yield from _flush_pre_marker_text(accumulated[:start_idx], response)
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            if tool_calls_end in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        if _could_be_marker_prefix(accumulated, tool_calls_start):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer


def _could_be_marker_prefix(text: str, marker: str) -> bool:
    max_check = len(marker)
    tail = text[-max_check:] if len(text) > max_check else text
    for i in range(len(tail)):
        suffix = tail[i:]
        if marker.startswith(suffix):
            return True
    return False


# Applied to the second and later pieces of one response's text. The response
# stands for a single token, so its logprob, stats and usage belong to exactly
# one emitted chunk: `collect_chat_response` appends a `Logprobs.content` entry
# per chunk carrying a logprob, and `count_reasoning_tokens` counts responses.
_SECONDARY_PIECE: dict[str, object] = {
    "logprob": None,
    "top_logprobs": None,
    "stats": None,
    "usage": None,
}


def _longest_marker_candidate_suffix(text: str, *markers: str | None) -> int:
    """Length of the longest suffix of `text` that could still become a marker.

    Zero when none can. Comparing the WHOLE buffer against `marker[:len(buffer)]`
    instead misses a marker preceded by visible text in the same window, so the
    marker leaks into visible content.
    """
    longest_marker = max((len(marker) for marker in markers if marker), default=0)
    for length in range(min(len(text), longest_marker), 0, -1):
        suffix = text[-length:]
        if any(marker and marker.startswith(suffix) for marker in markers):
            return length
    return 0


def parse_thinking_models(
    responses: Generator[GenerationResponse | None],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.

    A marker can arrive split across responses and can share a response with
    visible text on either side. Only the suffix of the buffer that could still
    become a marker is held back; everything before it is emitted immediately
    under its own response's metadata.

    Note that this sees text, not token ids, so a literal `<think>` in ordinary
    content is indistinguishable from the marker and is treated as the marker.

    One shape is deliberately not handled: any content carried by the terminal
    response itself is reported as `is_thinking=False`, even reasoning that
    began on an earlier response, because a terminal response is contractually
    not thinking.
    """
    is_thinking = starts_in_thinking
    accumulated = ""
    # Invariant: "".join(r.text for r in pending_buffer) == accumulated
    pending_buffer: list[GenerationResponse] = []

    def _next_marker(text: str, _is_thinking: bool) -> tuple[int, str, bool] | None:
        """Earliest marker occurrence in `text` that would change state."""
        found: tuple[int, str, bool] | None = None
        for marker, target in ((think_start, True), (think_end, False)):
            if not marker or _is_thinking == target:
                continue
            index = text.find(marker)
            if index != -1 and (found is None or index < found[0]):
                found = (index, marker, target)
        return found

    def _emit(
        char_count: int, _is_thinking: bool, *, split: bool = False
    ) -> Generator[GenerationResponse]:
        """Emit the first `char_count` buffered characters, keeping metadata.

        `split=False` stops at the last whole response that fits, so an ordinary
        hold-back never divides a response: `count_reasoning_tokens` counts
        responses, not tokens, and would over-report. Splitting is used only
        either side of a marker, where the pieces after the first are stripped
        of the metadata that must be emitted once (`_SECONDARY_PIECE`).
        """
        nonlocal accumulated
        remaining = char_count
        while pending_buffer:
            buffered = pending_buffer[0]
            chunk = buffered.text
            if len(chunk) <= remaining:
                _ = pending_buffer.pop(0)
                yield buffered.model_copy(update={"is_thinking": _is_thinking})
                remaining -= len(chunk)
            elif split and remaining > 0:
                pending_buffer[0] = buffered.model_copy(
                    update={"text": chunk[remaining:], **_SECONDARY_PIECE}
                )
                yield buffered.model_copy(
                    update={"text": chunk[:remaining], "is_thinking": _is_thinking}
                )
                remaining = 0
            else:
                break
        accumulated = "".join(buffered.text for buffered in pending_buffer)

    def _discard(char_count: int) -> None:
        """Drop the first `char_count` buffered characters without emitting."""
        nonlocal accumulated
        remaining = char_count
        while remaining > 0 and pending_buffer:
            buffered = pending_buffer[0]
            chunk = buffered.text
            if len(chunk) <= remaining:
                _ = pending_buffer.pop(0)
                remaining -= len(chunk)
            else:
                pending_buffer[0] = buffered.model_copy(
                    update={"text": chunk[remaining:]}
                )
                remaining = 0
        accumulated = "".join(buffered.text for buffered in pending_buffer)

    for response in responses:
        if response is None:
            yield None
            continue

        if response.finish_reason is not None:
            # The held buffer is a partial marker candidate; the terminal
            # response can be what completes it. Flushing the buffer without
            # looking leaks the marker into visible content.
            held = accumulated
            combined = held + response.text
            first = _next_marker(combined, is_thinking)
            if first is None:
                yield from _emit(len(accumulated), is_thinking)
                yield response.model_copy(update={"is_thinking": False})
                continue

            index, marker, target = first
            yield from _emit(min(index, len(held)), is_thinking, split=True)
            pending_buffer.clear()
            accumulated = ""
            # Text between markers that the terminal response contributed needs
            # its own non-terminal carrier, since the terminal yield below is
            # reserved for the text after the last marker. The carriers repeat
            # the terminal response's token id but not its logprob, stats or
            # usage, which the terminal yield keeps.
            segments: list[tuple[str, bool]] = []
            if index > len(held):
                segments.append((combined[len(held) : index], is_thinking))
            is_thinking = target
            rest = combined[index + len(marker) :]
            while (found := _next_marker(rest, is_thinking)) is not None:
                index, marker, target = found
                if index:
                    segments.append((rest[:index], is_thinking))
                is_thinking = target
                rest = rest[index + len(marker) :]
            for text, segment_is_thinking in segments:
                yield response.model_copy(
                    update={
                        "text": text,
                        "is_thinking": segment_is_thinking,
                        "finish_reason": None,
                        **_SECONDARY_PIECE,
                    }
                )
            yield response.model_copy(update={"text": rest, "is_thinking": False})
            continue

        pending_buffer.append(response)
        accumulated += response.text

        # `find`, not `endswith`: a marker followed by visible text in the same
        # response would otherwise never be recognised, leaking the marker and
        # wedging `is_thinking` for the rest of the stream. Looping handles a
        # window holding the end of one marker and the start of the next.
        while (found := _next_marker(accumulated, is_thinking)) is not None:
            index, marker, target = found
            yield from _emit(index, is_thinking, split=True)
            _discard(len(marker))
            is_thinking = target

        hold = _longest_marker_candidate_suffix(accumulated, think_start, think_end)
        yield from _emit(len(accumulated) - hold, is_thinking)


def parse_tool_calls(
    responses: Generator[GenerationResponse | None],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    accumulated_tool_calls: list[ToolCallItem] = []

    for response in responses:
        if response is None:
            yield None
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if (
            not in_tool_call
            and accumulated_tool_calls
            and (response.stats is not None or response.finish_reason is not None)
        ):
            yield ToolCallResponse(
                tool_calls=accumulated_tool_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_tool_calls.clear()
            continue

        if not in_tool_call:
            yield response
            continue

        tool_call_text_parts.append(response.text)
        if response.text.endswith(tool_parser.end_parsing):
            # parse the actual tool calls from the tool call text
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None:
                logger.warning(f"tool call parsing failed for text {combined}")
                yield response.model_copy(
                    update={"text": combined, "token": 0, "finish_reason": "error"}
                )
                break

            accumulated_tool_calls.extend(parsed)
            if accumulated_tool_calls and (
                response.finish_reason is not None or response.stats is not None
            ):
                yield ToolCallResponse(
                    tool_calls=accumulated_tool_calls,
                    usage=response.usage,
                    stats=response.stats,
                )
                accumulated_tool_calls.clear()
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": "".join(tool_call_text_parts),
                    "token": 0,
                    "finish_reason": "error",
                }
            )
            yield response

    if not accumulated_tool_calls:
        logger.warning("Tool calls should have all been emitted but were not")
