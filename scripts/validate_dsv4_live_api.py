#!/usr/bin/env python3
"""Live DeepSeek V4 Flash 0731 API validation with assertions that can fail.

Replaces `run_live_matrix_validation.py` and `verify_live_prefix_caching.py`,
whose verdicts were `PASS` whenever the HTTP call did not raise. Nothing there
asserted channel separation, DSML leakage, finish reasons, terminal-event
counts, or cross-endpoint agreement, so a matrix in which no tool call ever
fired and every Responses stream reported zero terminal events was recorded as
36/36 passing.

Design rules, each a direct response to a way the previous harness misled:

* A check yields PASS, FAIL or INCONCLUSIVE. `PASS` is only ever returned by a
  predicate that could have returned `FAIL` on this input.
* `finish_reason == "length"` makes a case INCONCLUSIVE, never PASS. A truncated
  completion cannot evidence "no DSML markup in the content channel", because
  there is barely any content to inspect.
* Terminal events are counted per endpoint. Chat Completions ends with the
  `[DONE]` sentinel; the Responses API ends with a `response.completed` event.
  Counting `[DONE]` on `/v1/responses` always yields zero.
* `max_tokens` is per-case and large enough to emit a complete DSML tool-call
  block. `<｜DSML｜tool_calls>` alone is six tokens and a full invoke with one
  parameter is 25+; with reasoning consuming the budget first, 32 tokens cannot
  reach a closing marker.
* Prefix-cache reuse is evidenced by `prompt_tokens_details.cached_tokens` and
  paired with a NEGATIVE CONTROL — an unrelated prompt that must NOT show high
  cached tokens. Without the control, a always-high counter looks like success.

Run `--self-test` to exercise every check against synthetic payloads, including
deliberately broken ones. That needs no cluster and no MLX, so the harness can
be trusted before a cluster run is spent on it.

Usage:
    uv run python scripts/validate_dsv4_live_api.py --self-test
    uv run python scripts/validate_dsv4_live_api.py --base-url http://localhost:52415
    uv run python scripts/validate_dsv4_live_api.py --only 5,6,7   # tool cases only

Exit code is non-zero if any check FAILED or any case was INCONCLUSIVE, so this
is usable as a gate rather than something whose output must be eyeballed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

# Defined locally rather than imported: importing the vendored encoder pulls in
# `exo.worker.engines.mlx`, and this script must run as a plain HTTP client.
# `--self-test` cross-checks these against the vendored constants when the
# import happens to succeed, so drift is caught rather than assumed away.
DSML_TOKEN = "｜DSML｜"
V4_TOOL_CALLS_START = f"<{DSML_TOKEN}tool_calls>"
V4_TOOL_CALLS_END = f"</{DSML_TOKEN}tool_calls>"
THINK_START = "<think>"
THINK_END = "</think>"
LATEST_REMINDER = "<｜latest_reminder｜>"

# Any of these appearing in a user-visible channel is a leak.
LEAK_MARKERS = (DSML_TOKEN, V4_TOOL_CALLS_START, THINK_START, THINK_END)

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass
class Check:
    name: str
    verdict: Verdict
    detail: str


@dataclass
class CaseResult:
    case: str
    endpoint: str
    stream: bool
    checks: list[Check] = field(default_factory=list)
    error: str | None = None

    @property
    def verdict(self) -> Verdict:
        if self.error is not None:
            return "FAIL"
        if any(c.verdict == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c.verdict == "INCONCLUSIVE" for c in self.checks):
            return "INCONCLUSIVE"
        return "PASS" if self.checks else "FAIL"

    @property
    def summary(self) -> str:
        if self.error is not None:
            return f"request error: {self.error}"
        bad = [c for c in self.checks if c.verdict != "PASS"]
        if bad:
            return "; ".join(f"{c.name}: {c.detail}" for c in bad)
        return "; ".join(c.detail for c in self.checks if c.detail)


def ok(name: str, detail: str = "") -> Check:
    return Check(name, "PASS", detail)


def bad(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


def unknown(name: str, detail: str) -> Check:
    return Check(name, "INCONCLUSIVE", detail)


# --------------------------------------------------------------------------- #
# Checks. Each takes already-parsed payload data and returns Checks. Kept pure
# so --self-test can drive them with synthetic input.
# --------------------------------------------------------------------------- #


def check_no_leak(channel_name: str, text: str) -> Check:
    hits = [m for m in LEAK_MARKERS if m in text]
    if hits:
        return bad(f"no_markup_in_{channel_name}", f"leaked {hits} into {channel_name}")
    return ok(f"no_markup_in_{channel_name}", f"{channel_name}_len={len(text)}")


def check_channels_separated(
    content: str, reasoning: str, thinking: bool
) -> list[Check]:
    checks = [check_no_leak("content", content), check_no_leak("reasoning", reasoning)]
    if thinking:
        if not reasoning:
            checks.append(
                bad(
                    "reasoning_present",
                    "thinking requested but reasoning_content empty",
                )
            )
        else:
            checks.append(ok("reasoning_present", f"reasoning_len={len(reasoning)}"))
    else:
        if reasoning:
            checks.append(
                bad(
                    "no_reasoning_when_disabled",
                    f"got {len(reasoning)} reasoning chars",
                )
            )
        else:
            checks.append(ok("no_reasoning_when_disabled"))
    return checks


def check_finish_reason(finish: str | None, expect_tool_call: bool) -> Check:
    if finish == "length":
        return unknown(
            "finish_reason",
            "truncated at max_tokens — this case proves nothing, raise the budget",
        )
    if expect_tool_call:
        if finish == "tool_calls":
            return ok("finish_reason", "tool_calls")
        return bad("finish_reason", f"expected tool_calls, got {finish!r}")
    if finish == "stop":
        return ok("finish_reason", "stop")
    return bad("finish_reason", f"expected stop, got {finish!r}")


def check_tool_calls(calls: list[dict[str, Any]], expect_count: int) -> list[Check]:
    """The check the previous harness lacked entirely: it recorded
    `tool_calls=False` for all three tool cases and called them PASS."""
    if not calls:
        return [bad("tool_calls_present", "no tool calls returned")]
    checks: list[Check] = [ok("tool_calls_present", f"n={len(calls)}")]
    if len(calls) < expect_count:
        checks.append(
            bad("tool_call_count", f"expected >={expect_count}, got {len(calls)}")
        )
    else:
        checks.append(ok("tool_call_count", f"n={len(calls)}"))

    for i, call in enumerate(calls):
        fn = call.get("function")
        if not isinstance(fn, dict):
            checks.append(bad(f"tool_call[{i}]_shape", "missing function object"))
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            checks.append(bad(f"tool_call[{i}]_name", f"bad name {name!r}"))
        raw_args = fn.get("arguments")
        if not isinstance(raw_args, str):
            checks.append(
                bad(
                    f"tool_call[{i}]_arguments",
                    f"expected JSON string, got {type(raw_args).__name__}",
                )
            )
            continue
        try:
            parsed = json.loads(raw_args)
        except ValueError as e:
            checks.append(bad(f"tool_call[{i}]_arguments", f"not valid JSON: {e}"))
            continue
        if not isinstance(parsed, dict):
            checks.append(
                bad(
                    f"tool_call[{i}]_arguments",
                    f"expected object, got {type(parsed).__name__}",
                )
            )
            continue
        # `json.loads` accepts JSON's non-standard Infinity/-Infinity/NaN
        # extensions, so a permissive parse is not evidence of valid JSON. A
        # strict client's `JSON.parse` rejects those tokens. This is the exact
        # hole fixed in the DSML value decoder for Task 4, asserted end to end.
        try:
            _ = json.dumps(parsed, allow_nan=False)
        except (TypeError, ValueError) as e:
            checks.append(
                bad(
                    f"tool_call[{i}]_arguments",
                    f"not strict JSON, a client would reject it: {e}",
                )
            )
            continue
        # Inspect the DECODED values, not the wire string: `json.dumps` escapes
        # non-ASCII by default, so a leaked `｜DSML｜` arrives as `｜DSML｜`
        # and a substring test on the raw string misses it.
        leaked = _leaked_markers(parsed)
        if leaked:
            checks.append(
                bad(
                    f"tool_call[{i}]_arguments",
                    f"markup {leaked} inside argument values",
                )
            )
            continue
        checks.append(ok(f"tool_call[{i}]", f"{name}({', '.join(sorted(parsed))})"))
    return checks


def _leaked_markers(value: object) -> list[str]:
    """Markers found anywhere in a decoded JSON structure, keys included."""
    found: set[str] = set()
    if isinstance(value, str):
        found.update(m for m in LEAK_MARKERS if m in value)
    elif isinstance(value, dict):
        for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
            found.update(_leaked_markers(k))
            found.update(_leaked_markers(v))
    elif isinstance(value, list):
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            found.update(_leaked_markers(item))
    return sorted(found)


def check_chat_stream_terminals(events: list[str]) -> list[Check]:
    """Chat Completions terminates with the literal `[DONE]` sentinel."""
    done = sum(1 for e in events if e.strip() == "[DONE]")
    checks: list[Check] = []
    if done == 1:
        checks.append(ok("exactly_one_terminal", "[DONE] x1"))
    else:
        checks.append(bad("exactly_one_terminal", f"[DONE] x{done}"))

    finishes = 0
    leaked: list[str] = []
    for raw in events:
        if raw.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        for choice in obj.get("choices", []):
            if choice.get("finish_reason") is not None:
                finishes += 1
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content"):
                val = delta.get(key)
                if isinstance(val, str):
                    leaked += [m for m in LEAK_MARKERS if m in val]
    if finishes == 1:
        checks.append(ok("exactly_one_finish_reason", "1"))
    else:
        checks.append(
            bad("exactly_one_finish_reason", f"{finishes} chunks carried one")
        )
    if leaked:
        checks.append(bad("no_markup_in_stream", f"leaked {sorted(set(leaked))}"))
    else:
        checks.append(ok("no_markup_in_stream", f"events={len(events)}"))
    return checks


def check_responses_stream_terminals(events: list[str]) -> list[Check]:
    """The Responses API terminates with a `response.completed` event, NOT
    `[DONE]`. Counting `[DONE]` here always yields zero, which is why the
    previous record shows `done=0` on all nine Responses rows."""
    completed = 0
    leaked: list[str] = []
    kinds: list[str] = []
    for raw in events:
        if raw.strip() == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        kind = obj.get("type")
        if isinstance(kind, str):
            kinds.append(kind)
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                completed += 1
        for key in ("delta", "text"):
            val = obj.get(key)
            if isinstance(val, str):
                leaked += [m for m in LEAK_MARKERS if m in val]

    checks: list[Check] = []
    if completed == 1:
        checks.append(ok("exactly_one_terminal", "response.completed x1"))
    elif completed == 0:
        checks.append(
            bad(
                "exactly_one_terminal",
                f"no terminal event; saw types {sorted(set(kinds))[:6]}",
            )
        )
    else:
        checks.append(bad("exactly_one_terminal", f"{completed} terminal events"))
    if leaked:
        checks.append(bad("no_markup_in_stream", f"leaked {sorted(set(leaked))}"))
    else:
        checks.append(ok("no_markup_in_stream", f"events={len(events)}"))
    return checks


def check_effort_prompt_tokens(by_effort: dict[str, int]) -> list[Check]:
    """Live-observable evidence for Task 2's tiers.

    The tier injects a prompt prefix, so distinct tiers must yield distinct
    prompt_tokens on an otherwise identical request. `high` and `xhigh` map to
    different V4 tiers; equal counts mean they collapsed to one.
    """
    checks: list[Check] = []
    low, high, xhigh = (
        by_effort.get("low"),
        by_effort.get("high"),
        by_effort.get("xhigh"),
    )
    if None in (low, high, xhigh):
        return [unknown("effort_tiers", f"missing measurements: {by_effort}")]
    assert low is not None and high is not None and xhigh is not None
    if high > low:
        checks.append(ok("high_adds_prefix", f"low={low} high={high}"))
    else:
        checks.append(
            bad("high_adds_prefix", f"low={low} high={high} (no prefix added)")
        )
    if xhigh != high:
        checks.append(ok("high_and_xhigh_differ", f"high={high} xhigh={xhigh}"))
    else:
        checks.append(
            bad("high_and_xhigh_differ", f"both {high} — tiers collapsed to one")
        )
    return checks


def check_prefix_cache(
    turn1_prompt: int,
    turn2_prompt: int,
    turn2_cached: int | None,
    control_cached: int | None,
) -> list[Check]:
    """Append-only prefix evidence, with the negative control the plan implies.

    The plan forbids claiming reuse from timing. It also is not evidenced by
    prompt-token counts alone, which is all the previous script printed before
    unconditionally reporting SUCCESS.
    """
    checks: list[Check] = []
    if turn2_prompt <= turn1_prompt:
        checks.append(
            bad("turn2_extends_turn1", f"turn1={turn1_prompt} turn2={turn2_prompt}")
        )
    else:
        checks.append(
            ok("turn2_extends_turn1", f"turn1={turn1_prompt} turn2={turn2_prompt}")
        )

    if turn2_cached is None:
        checks.append(
            unknown(
                "cached_tokens_reported",
                "prompt_tokens_details.cached_tokens absent — cannot evidence reuse",
            )
        )
        return checks

    # Reuse should cover most of turn 1's prompt. Allow slack for block-aligned
    # cache granularity rather than demanding an exact figure.
    threshold = int(turn1_prompt * 0.8)
    if turn2_cached >= threshold:
        checks.append(
            ok("prefix_reused", f"cached={turn2_cached} of turn1={turn1_prompt}")
        )
    else:
        checks.append(
            bad(
                "prefix_reused",
                f"cached={turn2_cached} < {threshold} (turn1={turn1_prompt})",
            )
        )

    if control_cached is None:
        checks.append(unknown("negative_control", "control cached_tokens absent"))
    elif control_cached >= threshold:
        checks.append(
            bad(
                "negative_control",
                f"unrelated prompt also cached {control_cached} — counter is not evidence",
            )
        )
    else:
        checks.append(
            ok("negative_control", f"unrelated prompt cached={control_cached}")
        )
    return checks


def check_cross_endpoint(chat_text: str, responses_text: str) -> Check:
    """Wire envelopes differ legitimately; logical content must agree."""

    def norm(s: str) -> str:
        return " ".join(s.split()).strip().lower()

    a, b = norm(chat_text), norm(responses_text)
    if not a and not b:
        return unknown(
            "cross_endpoint_agreement", "both endpoints returned empty content"
        )
    if a == b:
        return ok("cross_endpoint_agreement", f"identical ({len(a)} chars)")
    shorter, longer = sorted((a, b), key=len)
    if shorter and longer.startswith(shorter):
        return ok("cross_endpoint_agreement", "one is a prefix of the other")
    return bad("cross_endpoint_agreement", f"chat={a[:60]!r} responses={b[:60]!r}")


# --------------------------------------------------------------------------- #
# Case definitions
# --------------------------------------------------------------------------- #

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current time for a timezone",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}


@dataclass
class Case:
    number: int
    name: str
    messages: list[dict[str, Any]]
    opts: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    expect_tool_calls: int = 0
    thinking: bool = True
    # Tool cases need room for reasoning AND a complete DSML block.
    max_tokens: int = 256
    # True when `responses_body` cannot express this conversation faithfully, so
    # the two endpoints do not receive equivalent prompts and comparing their
    # output is meaningless. The first version of this harness asserted the
    # comparison anyway for case 7, whose `tool` message is flattened into
    # prose, and reported a FAIL that was the harness's fault rather than the
    # product's.
    lossy_responses_conversion: bool = False


CASES: list[Case] = [
    Case(
        1,
        "thinking_off",
        [{"role": "user", "content": "Say hello in one sentence."}],
        {"enable_thinking": False},
        thinking=False,
    ),
    Case(
        2,
        "thinking_low",
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        {"reasoning_effort": "low"},
    ),
    Case(
        3,
        "thinking_high",
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        {"reasoning_effort": "high"},
    ),
    Case(
        4,
        "thinking_xhigh",
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        {"reasoning_effort": "xhigh"},
    ),
    Case(
        5,
        "single_tool",
        [{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        expect_tool_calls=1,
        max_tokens=512,
    ),
    Case(
        6,
        "multi_tool",
        [
            {
                "role": "user",
                "content": "Get both the weather and the current time in Tokyo.",
            }
        ],
        tools=[WEATHER_TOOL, TIME_TOOL],
        expect_tool_calls=2,
        max_tokens=768,
    ),
    Case(
        7,
        "tool_result_followup",
        [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the weather tool.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny 22C"},
        ],
        tools=[WEATHER_TOOL],
        max_tokens=512,
        lossy_responses_conversion=True,
    ),
    Case(
        8,
        "mid_system_reminder",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarise your instructions in one line."},
            {"role": "system", "content": "Plan mode: be terse."},
        ],
    ),
    Case(
        9,
        "trailing_prefill",
        [
            {"role": "user", "content": "Write a short greeting."},
            {"role": "assistant", "content": "Hello there,"},
        ],
        {"enable_thinking": True},
        thinking=False,
    ),  # prefill pre-closes reasoning
]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310 - fixed http scheme, operator supplied
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        parsed = json.loads(resp.read().decode())
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> list[str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310
        url, data=data, headers={"Content-Type": "application/json"}
    )
    events: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        for line in resp:
            text = line.decode().strip()
            if text.startswith("data: "):
                events.append(text[6:])
    return events


def chat_body(case: Case, *, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": case.messages,
        "stream": stream,
        "max_tokens": case.max_tokens,
        "temperature": 0.0,
    }
    body.update(case.opts)
    if case.tools:
        body["tools"] = case.tools
    return body


def responses_body(case: Case, *, stream: bool) -> dict[str, Any]:
    converted: list[dict[str, Any]] = []
    for msg in case.messages:
        role = msg["role"]
        if role == "tool":
            # The Responses schema has no tool role in this harness's input
            # shape. Flag it rather than silently flattening to user text, which
            # is what the previous script did — it meant case 7 never exercised
            # tool results on this endpoint.
            converted.append(
                {"role": "user", "content": f"[tool output] {msg.get('content', '')}"}
            )
        else:
            converted.append({"role": role, "content": msg.get("content") or ""})
    body: dict[str, Any] = {
        "input": converted,
        "stream": stream,
        "max_output_tokens": case.max_tokens,
        "temperature": 0.0,
    }
    opts = dict(case.opts)
    effort = opts.pop("reasoning_effort", None)
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    body.update(opts)
    if case.tools:
        body["tools"] = case.tools
    return body


def responses_text(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract (content, reasoning) from a Responses payload."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        for block in item.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            (reasoning_parts if kind == "reasoning" else content_parts).append(text)
        if kind == "reasoning":
            for sm in item.get("summary", []) or []:
                if isinstance(sm, dict) and isinstance(sm.get("text"), str):
                    reasoning_parts.append(sm["text"])
    return "".join(content_parts), "".join(reasoning_parts)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_live(
    base_url: str, model: str, only: set[int] | None, timeout: float
) -> list[CaseResult]:
    results: list[CaseResult] = []
    effort_prompt_tokens: dict[str, int] = {}
    chat_content: dict[int, str] = {}
    resp_content: dict[int, str] = {}

    for case in CASES:
        if only and case.number not in only:
            continue
        label = f"{case.number}. {case.name}"
        print(f"[{label}] running...", file=sys.stderr)

        # Chat Completions, non-streaming
        r = CaseResult(label, "ChatCompletions", False)
        try:
            body = chat_body(case, stream=False) | {"model": model}
            payload = post_json(f"{base_url}/v1/chat/completions", body, timeout)
            msg = payload["choices"][0]["message"]
            finish = payload["choices"][0].get("finish_reason")
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            calls = msg.get("tool_calls") or []
            chat_content[case.number] = content

            r.checks.append(check_finish_reason(finish, case.expect_tool_calls > 0))
            if case.expect_tool_calls:
                r.checks += check_tool_calls(calls, case.expect_tool_calls)
                r.checks.append(check_no_leak("content", content))
            else:
                r.checks += check_channels_separated(content, reasoning, case.thinking)

            usage = payload.get("usage") or {}
            pt = usage.get("prompt_tokens")
            effort = case.opts.get("reasoning_effort")
            if isinstance(effort, str) and isinstance(pt, int):
                effort_prompt_tokens[effort] = pt
            details = usage.get("completion_tokens_details") or {}
            rt = details.get("reasoning_tokens")
            if case.thinking and isinstance(rt, int):
                if rt > 0:
                    r.checks.append(ok("reasoning_tokens_counted", f"{rt}"))
                else:
                    r.checks.append(
                        bad(
                            "reasoning_tokens_counted",
                            "reasoning present but usage says 0",
                        )
                    )
        except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError) as e:
            r.error = f"{type(e).__name__}: {e}"
        results.append(r)

        # Chat Completions, streaming
        r = CaseResult(label, "ChatCompletions", True)
        try:
            body = chat_body(case, stream=True) | {"model": model}
            events = post_stream(f"{base_url}/v1/chat/completions", body, timeout)
            r.checks += check_chat_stream_terminals(events)
        except (urllib.error.URLError, OSError, ValueError) as e:
            r.error = f"{type(e).__name__}: {e}"
        results.append(r)

        # Responses, non-streaming
        r = CaseResult(label, "Responses", False)
        try:
            body = responses_body(case, stream=False) | {"model": model}
            payload = post_json(f"{base_url}/v1/responses", body, timeout)
            content, reasoning = responses_text(payload)
            resp_content[case.number] = content
            status = payload.get("status")
            if status == "incomplete":
                r.checks.append(
                    unknown("status", "incomplete — raise max_output_tokens")
                )
            elif status == "completed":
                r.checks.append(ok("status", "completed"))
            else:
                r.checks.append(bad("status", f"unexpected status {status!r}"))
            r.checks.append(check_no_leak("content", content))
            r.checks.append(check_no_leak("reasoning", reasoning))
            if case.lossy_responses_conversion:
                r.checks.append(
                    unknown(
                        "cross_endpoint_agreement",
                        "skipped: the Responses input schema cannot express this "
                        "conversation's tool message, so the endpoints receive "
                        "different prompts and comparing output proves nothing",
                    )
                )
            elif case.number in chat_content:
                r.checks.append(
                    check_cross_endpoint(chat_content[case.number], content)
                )
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            r.error = f"{type(e).__name__}: {e}"
        results.append(r)

        # Responses, streaming
        r = CaseResult(label, "Responses", True)
        try:
            body = responses_body(case, stream=True) | {"model": model}
            events = post_stream(f"{base_url}/v1/responses", body, timeout)
            r.checks += check_responses_stream_terminals(events)
        except (urllib.error.URLError, OSError, ValueError) as e:
            r.error = f"{type(e).__name__}: {e}"
        results.append(r)

    if effort_prompt_tokens:
        tiers = CaseResult("effort tiers (cross-case)", "ChatCompletions", False)
        tiers.checks += check_effort_prompt_tokens(effort_prompt_tokens)
        results.append(tiers)

    return results


def run_prefix_cache(base_url: str, model: str, timeout: float) -> CaseResult:
    result = CaseResult("prefix cache (append-only)", "ChatCompletions", False)
    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    extended: list[dict[str, Any]] = base_messages + [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Checking weather for Paris.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny 22C"},
        {"role": "user", "content": "What about Rome?"},
        {"role": "system", "content": "Plan mode: be terse."},
    ]
    control: list[dict[str, Any]] = [
        {"role": "system", "content": "You translate text into French."},
        {"role": "user", "content": "Translate: the harbour is quiet tonight."},
    ]

    def send(messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": messages,
            "tools": [WEATHER_TOOL],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        return post_json(f"{base_url}/v1/chat/completions", body, timeout)

    def cached_of(payload: dict[str, Any]) -> int | None:
        details = (payload.get("usage") or {}).get("prompt_tokens_details") or {}
        value = details.get("cached_tokens")
        return value if isinstance(value, int) else None

    try:
        turn1 = send(base_messages)
        turn2 = send(extended)
        ctrl = send(control)
        t1 = int((turn1.get("usage") or {})["prompt_tokens"])
        t2 = int((turn2.get("usage") or {})["prompt_tokens"])
        result.checks += check_prefix_cache(t1, t2, cached_of(turn2), cached_of(ctrl))
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


# --------------------------------------------------------------------------- #
# Self-test: prove every check can fail
# --------------------------------------------------------------------------- #


# Typographic characters a legitimate English reply may contain even though they
# sit above the Latin blocks. Everything else beyond Latin Extended-B counts as a
# different script: Cyrillic, Greek and CJK all appeared in the 2026-08-06 run,
# so a CJK-only test would have passed `ссе`.
_ALLOWED_PUNCTUATION = frozenset("–—‘’“”…·•°×÷≈≠≤≥€£¥™©®′″")


def _non_latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    foreign = sum(
        1 for ch in text if ord(ch) > 0x024F and ch not in _ALLOWED_PUNCTUATION
    )
    return foreign / len(text)


def run_determinism(
    base_url: str, model: str, repeats: int, timeout: float
) -> CaseResult:
    """Send one identical request N times and compare the completions.

    This is the bisection that splits the two causes of incoherent output:

    * `temperature=0.0` means argmax — mlx_lm's `make_sampler` documents "if 0
      the argmax is used" — and `batch_generate` pins `mx.random.seed(42)`. So
      identical requests MUST produce identical completions.
    * If they differ, the forward pass itself is non-deterministic: reduction
      order or a race in the distributed all-reduce. Sampling cannot be blamed.
    * If they are identical but incoherent, the weights or the quantization are
      wrong, and the next step is the OMLX reference comparison rather than more
      cluster runs.

    A bit-exact tensor-parallel unit test passing does not settle this: a race
    can stay hidden on short synthetic inputs and appear under real generation.
    """
    result = CaseResult(f"determinism (x{repeats}, greedy)", "ChatCompletions", False)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "enable_thinking": False,
        "max_tokens": 48,
        "temperature": 0.0,
    }
    try:
        completions: list[str] = []
        finishes: list[str | None] = []
        for _ in range(repeats):
            payload = post_json(f"{base_url}/v1/chat/completions", body, timeout)
            choice = payload["choices"][0]
            completions.append(choice["message"].get("content") or "")
            finishes.append(choice.get("finish_reason"))

        unique = set(completions)
        if len(unique) == 1:
            result.checks.append(
                ok("greedy_is_deterministic", f"all {repeats} runs identical")
            )
        else:
            result.checks.append(
                bad(
                    "greedy_is_deterministic",
                    f"{len(unique)} distinct completions at temperature 0 — the "
                    f"forward pass is non-deterministic, not the sampler",
                )
            )

        # Coherence is not machine-checkable, but two cheap proxies are, and both
        # were violated in the 2026-08-06 run.
        sample = completions[0]
        if all(f == "length" for f in finishes):
            result.checks.append(
                bad(
                    "emits_eos",
                    f"every run hit max_tokens; the model never emitted EOS "
                    f"(first completion: {sample[:60]!r})",
                )
            )
        else:
            result.checks.append(ok("emits_eos", f"finish_reasons={finishes}"))

        foreign = _non_latin_ratio(sample)
        if sample and foreign > 0.2:
            result.checks.append(
                bad(
                    "plausible_english_reply",
                    f"{foreign:.0%} non-Latin characters in a reply to an English "
                    f"prompt: {sample[:60]!r}",
                )
            )
        else:
            result.checks.append(
                ok("plausible_english_reply", f"completion={sample[:60]!r}")
            )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def self_test() -> int:
    """Drive every check with good and deliberately-bad input.

    A check that cannot return FAIL is worthless; this proves each one can, and
    it runs with no cluster and no MLX.
    """
    failures: list[str] = []

    def expect(label: str, got: Verdict, want: Verdict) -> None:
        flag = "ok " if got == want else "BAD"
        print(f"  [{flag}] {label}: got {got}, want {want}")
        if got != want:
            failures.append(label)

    print("no-leak check:")
    expect("clean content", check_no_leak("content", "Hello there.").verdict, "PASS")
    expect(
        "DSML token leak", check_no_leak("content", f"a{DSML_TOKEN}b").verdict, "FAIL"
    )
    expect("think tag leak", check_no_leak("content", "x</think>y").verdict, "FAIL")

    print("channel separation:")
    expect(
        "thinking with reasoning",
        CaseResult(
            "t", "e", False, check_channels_separated("hi", "because", True)
        ).verdict,
        "PASS",
    )
    expect(
        "thinking without reasoning",
        CaseResult("t", "e", False, check_channels_separated("hi", "", True)).verdict,
        "FAIL",
    )
    expect(
        "disabled but reasoning present",
        CaseResult(
            "t", "e", False, check_channels_separated("hi", "oops", False)
        ).verdict,
        "FAIL",
    )

    print("finish reason:")
    expect("stop when no tools", check_finish_reason("stop", False).verdict, "PASS")
    expect(
        "stop when tools expected", check_finish_reason("stop", True).verdict, "FAIL"
    )
    expect(
        "tool_calls when expected",
        check_finish_reason("tool_calls", True).verdict,
        "PASS",
    )
    expect(
        "truncated is inconclusive",
        check_finish_reason("length", False).verdict,
        "INCONCLUSIVE",
    )

    print("tool calls (the gap in the previous harness):")
    good_call = [
        {"function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}}
    ]
    expect(
        "valid single call",
        CaseResult("t", "e", False, check_tool_calls(good_call, 1)).verdict,
        "PASS",
    )
    expect(
        "EMPTY list — previously recorded as PASS",
        CaseResult("t", "e", False, check_tool_calls([], 1)).verdict,
        "FAIL",
    )
    expect(
        "one call when two expected",
        CaseResult("t", "e", False, check_tool_calls(good_call, 2)).verdict,
        "FAIL",
    )
    expect(
        "non-JSON arguments",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [{"function": {"name": "f", "arguments": "{not json"}}], 1
            ),
        ).verdict,
        "FAIL",
    )
    # json.dumps escapes non-ASCII by default, so the leak must be sought in the
    # DECODED values. The first version of this harness tested the raw wire
    # string and passed both of the next two cases.
    expect(
        "DSML in arguments, ASCII-escaped on the wire",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [
                    {
                        "function": {
                            "name": "f",
                            "arguments": json.dumps({"a": DSML_TOKEN}),
                        }
                    }
                ],
                1,
            ),
        ).verdict,
        "FAIL",
    )
    expect(
        "DSML in arguments, literal on the wire",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [
                    {
                        "function": {
                            "name": "f",
                            "arguments": json.dumps(
                                {"a": DSML_TOKEN}, ensure_ascii=False
                            ),
                        }
                    }
                ],
                1,
            ),
        ).verdict,
        "FAIL",
    )
    expect(
        "DSML nested in a list value",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [
                    {
                        "function": {
                            "name": "f",
                            "arguments": json.dumps(
                                {"a": [{"b": V4_TOOL_CALLS_START}]}, ensure_ascii=False
                            ),
                        }
                    }
                ],
                1,
            ),
        ).verdict,
        "FAIL",
    )
    expect(
        "Infinity in arguments (Task 4 regression)",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [{"function": {"name": "f", "arguments": '{"a": Infinity}'}}], 1
            ),
        ).verdict,
        "FAIL",
    )
    expect(
        "NaN in arguments",
        CaseResult(
            "t",
            "e",
            False,
            check_tool_calls(
                [{"function": {"name": "f", "arguments": '{"a": NaN}'}}], 1
            ),
        ).verdict,
        "FAIL",
    )

    print("chat stream terminals:")
    chat_good = [
        json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "[DONE]",
    ]
    expect(
        "one DONE and one finish",
        CaseResult("t", "e", True, check_chat_stream_terminals(chat_good)).verdict,
        "PASS",
    )
    expect(
        "missing DONE",
        CaseResult("t", "e", True, check_chat_stream_terminals(chat_good[:-1])).verdict,
        "FAIL",
    )
    expect(
        "marker leaked in a delta",
        CaseResult(
            "t",
            "e",
            True,
            check_chat_stream_terminals(
                [
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {"content": V4_TOOL_CALLS_START},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ),
                    "[DONE]",
                ]
            ),
        ).verdict,
        "FAIL",
    )

    print("responses stream terminals (previously always reported done=0):")
    resp_good = [
        json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
        json.dumps({"type": "response.completed"}),
    ]
    expect(
        "one response.completed",
        CaseResult("t", "e", True, check_responses_stream_terminals(resp_good)).verdict,
        "PASS",
    )
    expect(
        "no terminal event",
        CaseResult(
            "t", "e", True, check_responses_stream_terminals(resp_good[:1])
        ).verdict,
        "FAIL",
    )
    expect(
        "only [DONE], no completed event",
        CaseResult(
            "t", "e", True, check_responses_stream_terminals(["[DONE]"])
        ).verdict,
        "FAIL",
    )

    print("effort tiers:")
    expect(
        "distinct tiers",
        CaseResult(
            "t",
            "e",
            False,
            check_effort_prompt_tokens({"low": 20, "high": 28, "xhigh": 34}),
        ).verdict,
        "PASS",
    )
    expect(
        "high == xhigh (collapsed)",
        CaseResult(
            "t",
            "e",
            False,
            check_effort_prompt_tokens({"low": 20, "high": 28, "xhigh": 28}),
        ).verdict,
        "FAIL",
    )
    expect(
        "high adds nothing",
        CaseResult(
            "t",
            "e",
            False,
            check_effort_prompt_tokens({"low": 20, "high": 20, "xhigh": 34}),
        ).verdict,
        "FAIL",
    )

    print("prefix cache:")
    expect(
        "reuse with clean control",
        CaseResult("t", "e", False, check_prefix_cache(286, 363, 280, 0)).verdict,
        "PASS",
    )
    expect(
        "no reuse",
        CaseResult("t", "e", False, check_prefix_cache(286, 363, 0, 0)).verdict,
        "FAIL",
    )
    expect(
        "control also cached — counter proves nothing",
        CaseResult("t", "e", False, check_prefix_cache(286, 363, 280, 280)).verdict,
        "FAIL",
    )
    expect(
        "cached_tokens absent",
        CaseResult("t", "e", False, check_prefix_cache(286, 363, None, None)).verdict,
        "INCONCLUSIVE",
    )
    expect(
        "turn2 does not extend turn1",
        CaseResult("t", "e", False, check_prefix_cache(363, 286, 280, 0)).verdict,
        "FAIL",
    )

    print("cross-endpoint agreement:")
    expect(
        "identical",
        check_cross_endpoint("Hello there.", "hello   there.").verdict,
        "PASS",
    )
    expect("divergent", check_cross_endpoint("Hello.", "Goodbye.").verdict, "FAIL")
    expect(
        "both empty is inconclusive",
        check_cross_endpoint("", "").verdict,
        "INCONCLUSIVE",
    )

    print("constant drift vs the vendored encoder:")
    try:
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (  # noqa: PLC0415
            dsml_token,
            tool_calls_block_name,
        )

        same = (
            dsml_token == DSML_TOKEN
            and f"<{dsml_token}{tool_calls_block_name}>" == V4_TOOL_CALLS_START
        )
        expect("markers match vendored constants", "PASS" if same else "FAIL", "PASS")
    except ImportError as e:
        print(f"  [--] vendored encoder not importable here ({e}); skipped")

    print()
    if failures:
        print(
            f"SELF-TEST FAILED: {len(failures)} checks behaved unexpectedly: {failures}"
        )
        return 1
    print("SELF-TEST PASSED: every check returns PASS on good input and FAIL on bad.")
    return 0


# --------------------------------------------------------------------------- #


def print_report(results: list[CaseResult]) -> None:
    print("\n| # | Case | Endpoint | Stream | Verdict | Detail |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        stream = "Stream" if r.stream else "Non-Stream"
        detail = r.summary.replace("|", "\\|")
        print(f"| {i} | {r.case} | {r.endpoint} | {stream} | {r.verdict} | {detail} |")

    passed = sum(1 for r in results if r.verdict == "PASS")
    failed = sum(1 for r in results if r.verdict == "FAIL")
    unsure = sum(1 for r in results if r.verdict == "INCONCLUSIVE")
    print(
        f"\n{passed} PASS, {failed} FAIL, {unsure} INCONCLUSIVE, {len(results)} total"
    )
    if failed or unsure:
        print("\nNot a pass. Failing and inconclusive rows:")
        for r in results:
            if r.verdict != "PASS":
                print(
                    f"  [{r.verdict}] {r.case} / {r.endpoint} / "
                    f"{'stream' if r.stream else 'non-stream'}: {r.summary}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp")
    ap.add_argument(
        "--only", default="", help="comma-separated case numbers, e.g. 5,6,7"
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--skip-prefix-cache", action="store_true")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="verify every check can fail; needs no cluster",
    )
    ap.add_argument(
        "--determinism",
        type=int,
        default=0,
        metavar="N",
        help=(
            "send one identical greedy request N times and compare. Run this "
            "FIRST when output looks incoherent: differing completions mean a "
            "non-deterministic forward pass, identical ones mean wrong weights"
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.determinism:
        result = run_determinism(
            args.base_url, args.model, args.determinism, args.timeout
        )
        print_report([result])
        return 0 if result.verdict == "PASS" else 1

    only = {int(x) for x in args.only.split(",") if x.strip()} or None
    results = run_live(args.base_url, args.model, only, args.timeout)
    if not args.skip_prefix_cache and not only:
        results.append(run_prefix_cache(args.base_url, args.model, args.timeout))
    print_report(results)
    return 0 if all(r.verdict == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
