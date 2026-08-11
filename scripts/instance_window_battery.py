#!/usr/bin/env python3
"""Run every live measurement in one instance window, cold-first.

Instance windows have been expensive to obtain, so ordering matters more than
convenience: the cold single request must be the first request the runner ever
sees.  ``--self-test`` uses an in-process stub and does not touch a real
instance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

Post = Callable[[str, dict[str, Any]], dict[str, Any]]


def _post_json(base_url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object, got {type(payload).__name__}")
    return payload


def _nonce(prompt: str) -> str:
    return f"{uuid.uuid4()} {prompt}"


def _cached_tokens(payload: dict[str, Any]) -> int | None:
    direct = payload.get("cached_tokens")
    if isinstance(direct, int):
        return direct
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
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


def _completion(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _chat_body(
    model: str, prompt: str, *, max_tokens: int, temperature: float = 0.0
) -> dict[str, Any]:
    prompt = (
        f"{prompt} {uuid.uuid4()}"
        if prompt.startswith("COLD-PROBE")
        else _nonce(prompt)
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "use_prefix_cache": False,
    }


def _measurement(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "cached_tokens": _cached_tokens(payload),
        "completion": _completion(payload),
        "response": payload,
    }


def _run_subprocess(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_battery(
    base_url: str,
    model: str,
    out_path: Path,
    *,
    only: set[str] | None = None,
    _post: Post | None = None,
) -> dict[str, Any]:
    """Run the measurements in dependency order and write a JSON report."""

    post = _post or _post_json
    selected = only or {
        "cold_single",
        "factual_short",
        "long_form",
        "tool_cases",
        "throughput",
        "context_ceiling",
    }
    measurements: list[dict[str, Any]] = []

    if "cold_single" in selected:
        cold = post(
            base_url,
            "/v1/chat/completions",
            _chat_body(model, "COLD-PROBE: answer with one word.", max_tokens=128),
        )
        cached = _cached_tokens(cold)
        if cached != 0:
            raise AssertionError(
                f"cold_single expected cached_tokens=0, got {cached!r}"
            )
        measurements.append(_measurement("cold_single", cold))

    if "factual_short" in selected:
        response = post(
            base_url,
            "/v1/chat/completions",
            _chat_body(model, "The capital of France is", max_tokens=3),
        )
        measurements.append(_measurement("factual_short", response))

    if "long_form" in selected:
        response = post(
            base_url,
            "/v1/chat/completions",
            _chat_body(
                model,
                "Write a sustained explanation of how a two-node tensor-parallel "
                "inference request moves from prefill through decode.",
                max_tokens=8192,
                temperature=1.0,
            ),
        )
        completion = _completion(response)
        measurements.append(
            _measurement("long_form", response)
            | {
                "has_think_end": "</think>" in completion,
                "has_answer": bool(completion.split("</think>", 1)[-1].strip()),
            }
        )

    if "tool_cases" in selected:
        script = Path(__file__).with_name("validate_dsv4_live_api.py")
        result = _run_subprocess(
            [sys.executable, str(script), "--base-url", base_url, "--only", "5,6,7"]
        )
        measurements.append({"name": "tool_cases", "verdict": result})

    if "throughput" in selected:
        script = Path(__file__).with_name("measure_dsv4_throughput.py")
        result = _run_subprocess(
            [
                sys.executable,
                str(script),
                "--base-url",
                base_url,
                "--model",
                model,
            ]
        )
        measurements.append({"name": "throughput", "verdict": result})

    if "context_ceiling" in selected:
        for tokens in (16_000, 32_000, 64_000):
            prompt = f"Context ceiling probe at approximately {tokens} tokens. " + (
                "probe " * tokens
            )
            response = post(
                base_url,
                "/v1/chat/completions",
                _chat_body(model, prompt, max_tokens=8),
            )
            measurements.append(
                {"name": "context_ceiling", "prompt_tokens": tokens}
                | _measurement(str(tokens), response)
            )

    report = {"base_url": base_url, "model": model, "measurements": measurements}
    if out_path != Path("/dev/null"):
        out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _self_test() -> int:
    sent: list[str] = []

    def fake_post(_base_url: str, _path: str, body: dict[str, Any]) -> dict[str, Any]:
        sent.append(body["messages"][0]["content"][:24])
        return {
            "choices": [{"message": {"content": "stub"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cached_tokens": 0},
        }

    report = run_battery(
        "stub://",
        "stub-model",
        Path("/dev/null"),
        only={"cold_single"},
        _post=fake_post,
    )
    assert sent, "battery sent no requests"
    assert sent[0].startswith("COLD-PROBE"), (
        f"first request was {sent[0]!r}; the cold probe must be first or its "
        f"cached_tokens=0 guarantee is void"
    )
    assert report["measurements"][0]["name"] == "cold_single", (
        "the report does not lead with the cold single request"
    )
    print("self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, default=Path("dsv4-instance-window.json"))
    parser.add_argument("--only", help="comma-separated measurement names")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required unless --self-test is used")
    only = set(args.only.split(",")) if args.only else None
    run_battery(args.base_url, args.model, args.out, only=only)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
