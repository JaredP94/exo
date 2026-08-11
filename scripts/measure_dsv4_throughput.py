#!/usr/bin/env python3
"""Measure streamed DeepSeek V4 prefill and decode throughput."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from typing import Any

Clock = Callable[[], float]
Post = Callable[[str, dict[str, Any]], Iterable[dict[str, Any]]]


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    direct = usage.get("cached_tokens")
    if isinstance(direct, int):
        return direct
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return details["cached_tokens"]
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict) and isinstance(
        input_details.get("cached_tokens"), int
    ):
        return input_details["cached_tokens"]
    return None


def _stream_post(base_url: str, body: dict[str, Any]) -> Iterable[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            value = json.loads(payload)
            if isinstance(value, dict):
                yield value


def measure(
    base_url: str,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    *,
    _clock: Clock = time.perf_counter,
    _post: Post | None = None,
) -> dict[str, float]:
    """Measure one unique, streamed prompt and return six numeric metrics."""

    if prompt_tokens <= 0 or max_tokens <= 0:
        raise ValueError("prompt_tokens and max_tokens must be positive")
    unique_prompt = f"{uuid.uuid4()} Throughput probe. " + ("probe " * prompt_tokens)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": unique_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    post = _post or _stream_post
    started = _clock()
    first_token: float | None = None
    finished = started
    completion_tokens = 0
    reported_prompt_tokens = prompt_tokens
    cached_tokens: int | None = None

    for event in post(base_url, body):
        finished = _clock()
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and (
                isinstance(delta.get("content"), str)
                or isinstance(delta.get("reasoning_content"), str)
                or delta.get("tool_calls")
            ):
                first_token = first_token or finished
        usage = event.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                reported_prompt_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
            cached_tokens = _cached_tokens(usage)

    if first_token is None:
        raise RuntimeError("stream contained no token event")
    if cached_tokens != 0:
        raise AssertionError(
            f"throughput prompt was not cold: cached_tokens={cached_tokens!r}"
        )
    if completion_tokens <= 0:
        raise RuntimeError("stream did not report a positive completion token count")

    ttft_s = first_token - started
    decode_s = finished - first_token
    if ttft_s <= 0 or decode_s <= 0:
        raise RuntimeError(f"non-positive timing interval: {ttft_s=} {decode_s=}")
    return {
        "ttft_s": ttft_s,
        "prefill_tokens_per_s": reported_prompt_tokens / ttft_s,
        "decode_tokens_per_s": completion_tokens / decode_s,
        "prompt_tokens": float(reported_prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "cached_tokens": float(cached_tokens or 0),
    }


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


def _stub_post(_base_url: str, _body: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield {"choices": [{"delta": {"content": "stub"}}]}
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": 32,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def _self_test() -> int:
    result = measure(
        "stub://",
        "stub",
        128,
        32,
        _clock=_FakeClock(),
        _post=_stub_post,
    )
    assert result["decode_tokens_per_s"] > 0, "decode rate not computed"
    assert result["ttft_s"] > 0, "time to first token not measured"
    assert result["cached_tokens"] == 0, (
        "the measurement prompt was not unique, so prefill time is a cache "
        "artifact rather than a throughput figure"
    )
    print("self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required unless --self-test is used")
    result = measure(
        args.base_url,
        args.model,
        args.prompt_tokens,
        args.max_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
