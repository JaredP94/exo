#!/usr/bin/env python3
"""Generate immutable DeepSeek V4 0731 prompt goldens from oMLX."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

from tokenizers import Tokenizer

GENERATOR_VERSION = 1

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

FILE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _cases() -> dict[str, dict[str, Any]]:
    base_effort_messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Solve 2 + 2."},
    ]
    return {
        "simple_thinking_on": {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
        },
        "simple_thinking_off": {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "thinking_mode": "chat",
            "reasoning_effort": None,
            "tools": None,
        },
        "prior_assistant_reasoning_retained_in_tool_conversation": {
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Inspect a.py."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I need to inspect the file first.",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read",
                    "content": "def answer(): return 42",
                },
                {"role": "user", "content": "Explain it."},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": FILE_TOOL,
            "retain_reasoning": True,
        },
        "one_tool_call_and_result": {
            "messages": [
                {"role": "user", "content": "Weather in Seoul?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_seoul",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Seoul"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_seoul",
                    "content": "Sunny, 24 C",
                },
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": WEATHER_TOOL,
        },
        "two_tool_calls_reverse_results": {
            "messages": [
                {"role": "user", "content": "Compare Seoul and London."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_seoul",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Seoul"}',
                            },
                        },
                        {
                            "id": "call_london",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"London"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_london",
                    "content": "Rain, 16 C",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_seoul",
                    "content": "Sunny, 24 C",
                },
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": WEATHER_TOOL,
        },
        "reminder_after_user_at_end": {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Plan mode"},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
        },
        "reminder_after_user_before_assistant": {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Plan mode"},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "Continue."},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
        },
        "consecutive_mid_system_reminders": {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Plan mode"},
                {"role": "system", "content": "Hook context"},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
        },
        "ambiguous_user_system_user": {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "system", "content": "Ambiguous"},
                {"role": "user", "content": "Second"},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
        },
        "trailing_assistant_prefill": {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Complete this sentence."},
                {"role": "assistant", "content": "PREFIX"},
            ],
            "thinking_mode": "thinking",
            "reasoning_effort": None,
            "tools": None,
            "continue_final_message": True,
        },
        **{
            f"reasoning_effort_{effort}": {
                "messages": copy.deepcopy(base_effort_messages),
                "thinking_mode": "chat" if effort == "none" else "thinking",
                "reasoning_effort": effort,
                "tools": None,
            }
            for effort in ("none", "minimal", "low", "medium", "high", "xhigh")
        },
    }


def _load_omlx_template(omlx_root: Path) -> ModuleType:
    source = omlx_root / "omlx/patches/deepseek_v4/chat_template_v4.py"
    spec = importlib.util.spec_from_file_location("omlx_dsv4_golden_reference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load oMLX template from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for message in copy.deepcopy(messages):
        if message.get("role") in {"system", "developer"}:
            content = message.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
        else:
            non_system.append(message)
    if system_parts:
        non_system.insert(0, {"role": "system", "content": "\n".join(system_parts)})
    return non_system


def _encoder_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort in {"minimal", "low", "medium"}:
        return "low"
    if reasoning_effort == "high":
        return "high"
    if reasoning_effort == "xhigh":
        return "max"
    return None


def _render_case(reference: ModuleType, case: dict[str, Any]) -> tuple[str, Any]:
    messages = copy.deepcopy(case["messages"])
    relocated = reference.relocate_mid_system_messages(messages)
    prepared = relocated if relocated is not None else messages
    prepared = _consolidate_system_messages(prepared)
    continue_final = bool(case.get("continue_final_message", False))
    prompt = reference.apply_chat_template(
        prepared,
        continue_final_message=continue_final,
        add_generation_prompt=not continue_final,
        thinking_mode=case["thinking_mode"],
        reasoning_effort=_encoder_effort(case["reasoning_effort"]),
        tools=case["tools"],
        drop_thinking=not bool(case.get("retain_reasoning", False)),
    )
    return prompt, relocated


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generate(model_path: Path, omlx_root: Path) -> dict[str, Any]:
    tokenizer_config = model_path / "tokenizer_config.json"
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    reference = _load_omlx_template(omlx_root)
    rendered_cases: dict[str, dict[str, Any]] = {}
    for name, case in _cases().items():
        prompt, relocated = _render_case(reference, case)
        rendered_cases[name] = {
            **case,
            "expected_relocated_messages": relocated,
            "expected_prompt_text": prompt,
            "expected_token_ids": tokenizer.encode(
                prompt, add_special_tokens=False
            ).ids,
        }
    return {
        "generator_version": GENERATOR_VERSION,
        "omlx_git_sha": _git_sha(omlx_root),
        "tokenizer_config_sha256": _sha256(tokenizer_config),
        "cases": rendered_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--omlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = generate(args.model_path, args.omlx_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
