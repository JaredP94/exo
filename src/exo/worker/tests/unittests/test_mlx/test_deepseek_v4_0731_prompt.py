from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast, get_args

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import ReasoningEffort, TextGenerationTaskParams
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.vendor import deepseek_v4_encoding

MODEL_ID = ModelId("Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp")
MODEL_PATH = Path(
    "/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp"
)
FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures/deepseek_v4_0731_prompt_goldens.json"
)

# `get_args` is typed `tuple[Any, ...]`; bind it once behind an explicit cast so
# no `Any` leaks into the assertions below (basedpyright runs with
# `reportAny = "error"`).
PUBLIC_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = cast(
    "tuple[ReasoningEffort, ...]", get_args(ReasoningEffort)
)
ENCODER_TIERS: tuple[utils_mlx.V4EncoderEffort, ...] = cast(
    "tuple[utils_mlx.V4EncoderEffort, ...]", get_args(utils_mlx.V4EncoderEffort)
)


class GoldenCase(TypedDict):
    messages: list[dict[str, Any]]
    thinking_mode: Literal["chat", "thinking"]
    reasoning_effort: ReasoningEffort | None
    tools: list[dict[str, Any]] | None
    retain_reasoning: NotRequired[bool]
    continue_final_message: NotRequired[bool]
    expected_relocated_messages: list[dict[str, Any]] | None
    expected_prompt_text: str
    expected_token_ids: list[int]


def _goldens() -> dict[str, GoldenCase]:
    payload = cast(dict[str, object], json.loads(FIXTURE_PATH.read_text()))
    assert payload["generator_version"] == 1
    return cast(dict[str, GoldenCase], payload["cases"])


@functools.lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    """Parsing the 6 MiB tokenizer once keeps the parametrized cases cheap."""
    return Tokenizer.from_file(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        str(MODEL_PATH / "tokenizer.json")
    )


def _encode_without_special_tokens(prompt: str) -> list[int]:
    """Confine the untyped `tokenizers` boundary to one place.

    The installed `tokenizers` stubs annotate neither `Tokenizer.encode` nor
    `Encoding.ids`, so every call site would otherwise trip the repository's
    `reportUnknownMemberType = "error"`. Narrowing to `list[int]` here keeps the
    assertions themselves fully typed.
    """
    encoding = _tokenizer().encode(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        prompt, add_special_tokens=False
    )
    return cast("list[int]", encoding.ids)


def _assert_golden_matches(case_name: str) -> None:
    case = _goldens()[case_name]
    prompt = _render(case)

    assert prompt == case["expected_prompt_text"]
    assert _encode_without_special_tokens(prompt) == case["expected_token_ids"]


def _render(case: GoldenCase) -> str:
    params = TextGenerationTaskParams(
        model=MODEL_ID,
        input=[],
        tools=copy.deepcopy(case["tools"]),
        enable_thinking=case["thinking_mode"] == "thinking",
        reasoning_effort=case["reasoning_effort"],
    )
    unused_tokenizer = cast(TokenizerWrapper, object())
    return utils_mlx.render_chat_template(
        unused_tokenizer,
        copy.deepcopy(case["messages"]),
        params,
    )


def test_v4_declares_arbitrary_mid_system_messages_unsupported() -> None:
    assert deepseek_v4_encoding.supports_mid_system_messages is False


def test_raw_input_ids_bypass_chat_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_ENABLE_RAW_INPUT_IDS_DEBUG", "1")
    params = TextGenerationTaskParams(
        model=MODEL_ID,
        input=[],
        raw_input_ids=[0, 128803, 128821],
    )

    prompt = utils_mlx.apply_chat_template(
        cast(TokenizerWrapper, object()),
        params,
    )

    assert prompt == ""


def test_encode_prompt_accepts_raw_input_ids_without_tokenizer() -> None:
    tokens = encode_prompt(
        cast(TokenizerWrapper, object()),
        "ignored by raw input IDs",
        raw_input_ids=[0, 128803, 128821],
    )

    assert tokens.tolist() == [0, 128803, 128821]


def test_reminder_relocation_keeps_leading_system_and_does_not_mutate() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Plan mode"},
    ]
    original = copy.deepcopy(messages)

    relocated = deepseek_v4_encoding.relocate_mid_system_messages(messages)

    assert relocated == [
        {"role": "system", "content": "Be helpful."},
        {"role": "latest_reminder", "content": "Plan mode"},
        {"role": "user", "content": "Hello"},
    ]
    assert messages == original


def test_reminder_relocation_merges_consecutive_content_with_blank_line() -> None:
    relocated = deepseek_v4_encoding.relocate_mid_system_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Plan mode"},
            {"role": "system", "content": "Hook context"},
            {"role": "assistant", "content": "Understood."},
        ]
    )

    assert relocated == [
        {"role": "latest_reminder", "content": "Plan mode\n\nHook context"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Understood."},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "First"},
            {"role": "system", "content": "Ambiguous"},
            {"role": "user", "content": "Second"},
        ],
        [
            {"role": "user", "content": "Hello"},
            {
                "role": "system",
                "content": [{"type": "image_url", "image_url": "data:image/png"}],
            },
        ],
    ],
)
def test_reminder_relocation_rejects_ambiguous_or_non_text_content(
    messages: list[dict[str, Any]],
) -> None:
    assert deepseek_v4_encoding.relocate_mid_system_messages(messages) is None


def test_reminder_relocation_does_not_reclassify_nonleading_developer() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "developer", "content": "Developer note"},
    ]

    assert deepseek_v4_encoding.relocate_mid_system_messages(messages) == messages


def test_append_only_normal_follow_up_when_reasoning_is_retained() -> None:
    system = {"role": "system", "content": "You are a coding agent."}
    first_user = {"role": "user", "content": "Refactor the parser."}
    assistant = {
        "role": "assistant",
        "content": "Done.",
        "reasoning_content": "Plan the split, then extract.",
    }
    second_user = {"role": "user", "content": "Now add tests."}

    turn_1_prompt = deepseek_v4_encoding.encode_messages(
        [system, first_user], thinking_mode="thinking", drop_thinking=False
    )
    turn_2_prompt = deepseek_v4_encoding.encode_messages(
        [system, first_user, assistant, second_user],
        thinking_mode="thinking",
        drop_thinking=False,
    )

    assert turn_2_prompt.startswith(turn_1_prompt)


def test_append_only_tool_loop_when_reasoning_is_retained() -> None:
    system = {"role": "system", "content": "You are a coding agent."}
    first_user = {"role": "user", "content": "Inspect a.py."}
    tool_call = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I need the file.",
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "file contents",
    }
    assistant = {
        "role": "assistant",
        "content": "Done.",
        "reasoning_content": "Now explain it.",
    }
    second_user = {"role": "user", "content": "Continue."}

    turn_1_prompt = deepseek_v4_encoding.encode_messages(
        [system, first_user, tool_call, tool_result],
        thinking_mode="thinking",
        drop_thinking=False,
    )
    turn_2_prompt = deepseek_v4_encoding.encode_messages(
        [system, first_user, tool_call, tool_result, assistant, second_user],
        thinking_mode="thinking",
        drop_thinking=False,
    )

    assert turn_2_prompt.startswith(turn_1_prompt)


def test_reminder_keeps_leading_system_prefix_and_uses_marker() -> None:
    goldens = _goldens()
    before = _render(goldens["simple_thinking_on"])
    after = _render(goldens["reminder_after_user_at_end"])

    leading_system_prefix = "<｜begin▁of▁sentence｜>Be helpful."
    assert before.startswith(leading_system_prefix)
    assert after.startswith(leading_system_prefix)
    assert "<｜latest_reminder｜>Plan mode" in after


TASK_1_CASES = [
    "simple_thinking_on",
    "simple_thinking_off",
    "prior_assistant_reasoning_retained_in_tool_conversation",
    "one_tool_call_and_result",
    "two_tool_calls_reverse_results",
    "reminder_after_user_at_end",
    "reminder_after_user_before_assistant",
    "consecutive_mid_system_reminders",
    "ambiguous_user_system_user",
]


@pytest.mark.parametrize("case_name", TASK_1_CASES)
def test_task_1_prompt_text_and_token_ids_match_omlx_golden(
    case_name: str,
) -> None:
    _assert_golden_matches(case_name)


def test_v4_no_think_prompt_variant_omits_thinking_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_DSV4_PROMPT_VARIANT", "no_think")
    params = TextGenerationTaskParams(
        model=MODEL_ID,
        input=[],
        enable_thinking=False,
    )

    prompt = utils_mlx.render_chat_template(
        cast(TokenizerWrapper, object()),
        [{"role": "user", "content": "The capital of France is"}],
        params,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>The capital of France is"
        "<｜Assistant｜>"
    )


# --------------------------------------------------------------------------- #
# Task 2: reasoning-effort tiers
#
# Break each test catches:
#   * maps_every_public_value      - a wrong tier for any public effort.
#   * covers_every_public_effort   - a value added to ReasoningEffort with no
#     tier, which would raise KeyError at request time instead of failing here.
#   * chat_mode_emits_no_prefix    - losing the thinking-mode guard, leaking a
#     reasoning instruction into a thinking-disabled request. No golden covers
#     this: the only chat-mode golden uses the tier whose text is empty.
#   * high_and_max_differ          - both tiers pointing at one string, which
#     would silently collapse `xhigh` into `high`.
#   * invalid_tier_rejected        - dropping the tier validation, letting a
#     typo render a prefix-free prompt instead of failing.
# --------------------------------------------------------------------------- #

EFFORT_CASES = [
    "reasoning_effort_none",
    "reasoning_effort_minimal",
    "reasoning_effort_low",
    "reasoning_effort_medium",
    "reasoning_effort_high",
    "reasoning_effort_xhigh",
]


def _effort_params(
    reasoning_effort: ReasoningEffort | None,
    *,
    enable_thinking: bool,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=MODEL_ID,
        input=[],
        tools=None,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


def _thinking_prompt(reasoning_effort: ReasoningEffort | None) -> str:
    unused_tokenizer = cast(TokenizerWrapper, object())
    return utils_mlx.render_chat_template(
        unused_tokenizer,
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Solve 2 + 2."},
        ],
        _effort_params(reasoning_effort, enable_thinking=True),
    )


@pytest.mark.parametrize(
    ("public_effort", "expected_tier"),
    [
        ("none", None),
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "low"),
        ("high", "high"),
        ("xhigh", "max"),
    ],
)
def test_v4_reasoning_effort_maps_every_public_value(
    public_effort: ReasoningEffort, expected_tier: str | None
) -> None:
    params = _effort_params(public_effort, enable_thinking=True)
    assert utils_mlx._v4_reasoning_effort(params) == expected_tier  # pyright: ignore[reportPrivateUsage]


def test_v4_reasoning_effort_returns_none_when_unspecified() -> None:
    params = _effort_params(None, enable_thinking=True)
    assert utils_mlx._v4_reasoning_effort(params) is None  # pyright: ignore[reportPrivateUsage]


def test_v4_reasoning_effort_covers_every_public_effort() -> None:
    assert set(utils_mlx.V4_REASONING_EFFORT_MAP) == set(PUBLIC_REASONING_EFFORTS)


def test_encoder_table_and_caller_tiers_do_not_drift() -> None:
    """A tier named in `V4EncoderEffort` with no row in the vendored table (or
    the reverse) makes a live request fail the encoder's assert."""
    assert set(deepseek_v4_encoding.REASONING_EFFORT_PROMPTS) == set(ENCODER_TIERS)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium"])
def test_low_tier_efforts_add_no_prefix(effort: ReasoningEffort) -> None:
    assert _thinking_prompt(effort) == _thinking_prompt(None)


def test_high_effort_adds_a_prefix_before_the_system_content() -> None:
    prompt = _thinking_prompt("high")

    assert prompt != _thinking_prompt(None)
    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: ")
    assert "Be precise." in prompt


def test_high_and_max_tiers_emit_different_prefixes() -> None:
    assert _thinking_prompt("high") != _thinking_prompt("xhigh")


def test_chat_mode_emits_no_reasoning_effort_prefix() -> None:
    """A thinking-disabled request must not receive a reasoning instruction."""
    unused_tokenizer = cast(TokenizerWrapper, object())
    prompt = utils_mlx.render_chat_template(
        unused_tokenizer,
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Solve 2 + 2."},
        ],
        _effort_params("xhigh", enable_thinking=False),
    )

    assert "Reasoning Effort:" not in prompt


def test_invalid_encoder_tier_is_rejected_naming_the_allowed_tiers() -> None:
    with pytest.raises(AssertionError) as excinfo:
        deepseek_v4_encoding.encode_messages(
            [{"role": "user", "content": "Hello"}],
            thinking_mode="thinking",
            reasoning_effort="maximum",
        )

    message = str(excinfo.value)
    assert "maximum" in message
    for allowed in ("low", "high", "max"):
        assert allowed in message


@pytest.mark.parametrize("case_name", EFFORT_CASES)
def test_effort_prompt_text_and_token_ids_match_omlx_golden(case_name: str) -> None:
    _assert_golden_matches(case_name)


def test_effort_case_list_covers_every_public_effort() -> None:
    """Guards the parametrize list above against drifting from the fixture."""
    goldens = _goldens()
    assert {f"reasoning_effort_{value}" for value in PUBLIC_REASONING_EFFORTS} == set(
        EFFORT_CASES
    )
    for name in EFFORT_CASES:
        assert name in goldens


# --------------------------------------------------------------------------- #
# Task 3: a trailing assistant turn is a prefill, and belongs in the content
# channel rather than inside the reasoning block.
#
# Break each test catches:
#   * thinking_mode_prefill_lands_in_the_content_channel - the defect itself:
#     raw concatenation put PREFIX straight after the generation anchor, i.e.
#     inside `<think>`, feeding prefilled content to the model as its own hidden
#     reasoning. Golden-backed by `trailing_assistant_prefill`.
#   * chat_mode_prefill_after_closed_thinking - the chat-mode shape, which was
#     accidentally correct before this task because the anchor already closes the
#     block. A regression guard. No committed golden covers chat-mode prefill;
#     the expectation was cross-checked against the real OMLX module.
#   * prefill_prompt_does_not_end_with_eos - a missing `wo_eos` mark, which would
#     close the turn and leave the model nothing to continue from.
#   * prefill_content_is_not_inside_the_thinking_block - the channel invariant,
#     asserted structurally so it still fires when the surrounding prompt changes.
#   * prior_reasoning_and_prefill_keep_channel_order - the two channels being
#     swapped or merged.
#   * non_v4_models_still_pop_and_append - this task's regression risk: moving
#     the pop below the V4 branch must not remove it from the other families.
# --------------------------------------------------------------------------- #

PREFILL_CASES = ["trailing_assistant_prefill"]

NON_V4_MODEL = ModelId("mlx-community/DeepSeek-V3.2-4bit")

PREFILL_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Complete this sentence."},
    {"role": "assistant", "content": "PREFIX"},
]


def _render_messages(
    messages: list[dict[str, Any]],
    *,
    model: ModelId = MODEL_ID,
    enable_thinking: bool = True,
) -> str:
    params = TextGenerationTaskParams(
        model=model,
        input=[],
        tools=None,
        enable_thinking=enable_thinking,
        reasoning_effort=None,
    )
    unused_tokenizer = cast(TokenizerWrapper, object())
    return utils_mlx.render_chat_template(
        unused_tokenizer, copy.deepcopy(messages), params
    )


def test_thinking_mode_prefill_lands_in_the_content_channel() -> None:
    prompt = _render_messages(PREFILL_MESSAGES)

    assert "<｜Assistant｜><think></think>PREFIX" in prompt


def test_chat_mode_prefill_lands_after_the_closed_thinking_block() -> None:
    prompt = _render_messages(PREFILL_MESSAGES, enable_thinking=False)

    assert "<｜Assistant｜></think>PREFIX" in prompt


def test_prefill_prompt_does_not_end_with_eos() -> None:
    prompt = _render_messages(PREFILL_MESSAGES)

    assert not prompt.endswith(deepseek_v4_encoding.eos_token)
    assert prompt.endswith("PREFIX")


def test_prefill_content_is_not_inside_the_thinking_block() -> None:
    prompt = _render_messages(PREFILL_MESSAGES)

    opened = prompt.rindex(deepseek_v4_encoding.thinking_start_token)
    closed = prompt.rindex(deepseek_v4_encoding.thinking_end_token)
    prefix_at = prompt.rindex("PREFIX")

    assert opened < closed < prefix_at, (
        "PREFIX must follow the closed thinking block, not sit inside it"
    )


def test_prior_reasoning_and_prefill_keep_channel_order() -> None:
    prompt = _render_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "PREFIX", "reasoning_content": "RC"},
        ]
    )

    assert "<｜Assistant｜><think>RC</think>PREFIX" in prompt


def test_non_v4_models_still_pop_and_append_the_trailing_assistant() -> None:
    """No stub: the sibling encoder is pure, so compare against it directly."""
    from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

    expected = (
        encode_messages(
            messages=[PREFILL_MESSAGES[0], PREFILL_MESSAGES[1]],
            thinking_mode="thinking",
            tools=None,
        )
        + "PREFIX"
    )

    assert _render_messages(PREFILL_MESSAGES, model=NON_V4_MODEL) == expected


@pytest.mark.parametrize("case_name", PREFILL_CASES)
def test_prefill_prompt_text_and_token_ids_match_omlx_golden(case_name: str) -> None:
    _assert_golden_matches(case_name)


def test_every_committed_golden_case_is_asserted_somewhere() -> None:
    """A golden added to the fixture but never asserted protects nothing."""
    asserted = set(TASK_1_CASES) | set(EFFORT_CASES) | set(PREFILL_CASES)
    assert asserted == set(_goldens())


# --------------------------------------------------------------------------- #
# Review fix round for Task 3. Breaks these catch:
#
#   * empty_trailing_assistant_keeps_thinking_open - marking an empty assistant
#     turn `wo_eos` renders a pre-closed `<think></think>`, denying a
#     thinking-enabled request any reasoning. Reachable: the chat-completions
#     adapter only drops an assistant message when content, reasoning_content
#     AND tool_calls are all None, so `content: ""` survives.
#   * trailing_tool_calls_does_not_render_unterminated - the same mark on a turn
#     carrying tool_calls emits a tool-call block with no EOS and no fresh
#     anchor, whose likeliest continuation is an empty completion.
#   * prefill_containing_a_think_tag_is_preserved - the trailing turn passing
#     through _strip_v4_thinking_markers silently rewrites the caller's own text.
# --------------------------------------------------------------------------- #

OPEN_ANCHOR = "<｜Assistant｜><think>"


def test_empty_trailing_assistant_keeps_the_thinking_block_open() -> None:
    prompt = _render_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ]
    )

    assert prompt.endswith(OPEN_ANCHOR), (
        "an empty assistant turn is not a prefill; reasoning must stay open"
    )


def test_trailing_tool_calls_turn_does_not_render_unterminated() -> None:
    prompt = _render_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a":1}'},
                    }
                ],
            },
        ]
    )

    assert deepseek_v4_encoding.tool_calls_block_name not in prompt
    assert prompt.endswith(OPEN_ANCHOR)


def test_prefill_containing_a_think_tag_is_preserved_verbatim() -> None:
    prefill = "Use <think> tags like this"
    prompt = _render_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": prefill},
        ]
    )

    assert prompt.endswith(prefill)


def test_trailing_assistant_with_content_and_tool_calls_is_not_a_prefill() -> None:
    """Exercises the tool_calls guard specifically.

    The empty-content guard already rejects the `content: ""` tool-call shape, so
    only a turn with BOTH real content and tool_calls can prove the tool_calls
    guard exists. Marking that `wo_eos` would render the content followed by an
    unterminated tool-call block; the historical drop-and-append is preferable.
    """
    prompt = _render_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "PARTIAL",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a":1}'},
                    }
                ],
            },
        ]
    )

    assert deepseek_v4_encoding.tool_calls_block_name not in prompt
    assert prompt.endswith(OPEN_ANCHOR + "PARTIAL")
