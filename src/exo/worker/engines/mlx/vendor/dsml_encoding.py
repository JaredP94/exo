import ast
import json
import re
from typing import Any, cast

from mlx_lm.chat_templates import deepseek_v32

from exo.api.types import ToolCallItem

BOS_TOKEN: str = deepseek_v32.bos_token
EOS_TOKEN: str = deepseek_v32.eos_token
DSML_TOKEN: str = deepseek_v32.dsml_token
THINKING_START: str = deepseek_v32.thinking_start_token
THINKING_END: str = deepseek_v32.thinking_end_token
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
TOOL_CALLS_START = f"<{DSML_TOKEN}function_calls>"
TOOL_CALLS_END = f"</{DSML_TOKEN}function_calls>"
_ORPHAN_THINK_END = ASSISTANT_TOKEN + THINKING_END
_FIXED_THINK_BLOCK = ASSISTANT_TOKEN + THINKING_START + "\n" + THINKING_END
_FUNCTION_RESULTS_CLOSE = "</function_results>"
_ORPHAN_TOOL_RESULT_SUFFIX = _FUNCTION_RESULTS_CLOSE + "\n\n" + THINKING_END
_EMPTY_THINK_BLOCKS = (
    THINKING_START + "\n\n" + THINKING_END,
    THINKING_START + "\n" + THINKING_END,
    THINKING_START + THINKING_END,
)


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str = "thinking",
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    tools: Any = None,  # pyright: ignore[reportAny]
) -> str:
    # V3.2 (like V4) is `tool_conditional`: when tools are in play, prior-turn
    # reasoning_content must be retained so multi-step tool chains stay
    # coherent.
    effective_drop_thinking = drop_thinking
    if tools:
        effective_drop_thinking = False
    prompt: str = deepseek_v32.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=effective_drop_thinking,
        add_default_bos_token=add_default_bos_token,
        tools=tools,
    )
    prompt = prompt.replace(_ORPHAN_TOOL_RESULT_SUFFIX, _FUNCTION_RESULTS_CLOSE)
    prompt = prompt.replace(_ORPHAN_THINK_END, _FIXED_THINK_BLOCK)
    for empty in _EMPTY_THINK_BLOCKS:
        prompt = prompt.replace(empty, "")
    return prompt


_INVOKE_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\s+name=\"([^\"]+)\"\s*>"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)

_PARAM_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=\"([^\"]+)\""
    rf"\s+string=\"(true|false)\"\s*>"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}parameter>",
    re.DOTALL,
)


def _decode_dsml_value(raw: str, is_string: bool) -> object:
    """Decode one DSML parameter value, mirroring OMLX's `_decode_value`.

    Models pad values with a single newline before the closing tag. Trim that
    and only that, so intentional surrounding whitespace inside a string value
    survives. String parameters are returned verbatim. Non-string parameters are
    JSON literals, with `ast.literal_eval` as a fallback for the Python-style
    literals models occasionally emit.

    Error handling rationale: none of the failures below are exceptional, so all
    are handled here rather than raised. Whatever this returns is re-serialized
    by the caller with `json.dumps` into `ToolCallItem.arguments`, and
    `parse_dsml_output` is called without a try/except from
    `model_output_parsers._try_parse_tool_call` — which is itself inside the
    runner's generation loop. An exception here would therefore not merely spoil
    one tool call, it would tear down generation. So any value that cannot be
    re-serialized as strict JSON is returned as text instead, which is the
    documented contract for output this parser cannot make sense of.

    Two distinct failures need that treatment: types `json.dumps` rejects
    outright (`ast.literal_eval` happily produces sets, bytes and complex
    numbers), and non-finite floats, which `json.loads` accepts via JSON's
    `Infinity`/`NaN` extensions and `json.dumps` would emit back as tokens no
    strict client can parse. `allow_nan=False` turns the second into an error we
    can catch alongside the first.

    The `ValueError` caught below is deliberately broader than the reference
    implementation's `json.JSONDecodeError`: a numeric literal above CPython's
    `int_max_str_digits` limit raises a bare `ValueError`, which would otherwise
    escape. Do not narrow it.
    """
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]

    if is_string:
        return raw

    decoded: object
    try:
        decoded = cast(object, json.loads(raw))
    except ValueError:  # includes json.JSONDecodeError
        try:
            decoded = cast(object, ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            return raw

    try:
        _ = json.dumps(decoded, allow_nan=False)
    except (TypeError, ValueError):
        return raw
    return decoded


def parse_dsml_output(text: str) -> list[ToolCallItem] | None:
    """Parse DSML function_calls block from model output text.

    Args:
        text: The text containing the DSML function_calls block
              (including the start/end markers).

    Returns:
        List of ToolCallItem, or None if parsing fails.
    """
    tool_calls: list[ToolCallItem] = []

    for invoke_match in _INVOKE_PATTERN.finditer(text):
        func_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)

        args: dict[str, object] = {}
        for param_match in _PARAM_PATTERN.finditer(invoke_body):
            param_name = param_match.group(1)
            is_string = param_match.group(2) == "true"
            param_value = param_match.group(3)

            args[param_name] = _decode_dsml_value(param_value, is_string)

        tool_calls.append(
            ToolCallItem(
                name=func_name,
                arguments=json.dumps(args),
            )
        )

    return tool_calls if tool_calls else None
