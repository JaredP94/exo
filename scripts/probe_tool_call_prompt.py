#!/usr/bin/env python3
"""Check that a DeepSeek V4 tool definition survives prompt rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

MODEL_PATH = Path(
    "/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp"
)

TOOL_FIXTURE: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def render_tool_prompt(
    fixture: list[dict[str, Any]],
    model_path: Path = MODEL_PATH,
) -> tuple[str, list[int], str]:
    """Render, tokenize, and decode one tool-bearing DeepSeek V4 prompt."""

    rendered = encode_messages(
        messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        thinking_mode="thinking",
        tools=fixture,
    )
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    ids = tokenizer.encode(rendered).ids
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return rendered, ids, decoded


def _self_test() -> int:
    rendered, ids, decoded = render_tool_prompt(TOOL_FIXTURE)
    assert "get_current_weather" in rendered, (
        "the tool name is absent from the rendered prompt; the model is never "
        "told the tool exists, so no amount of token budget will produce a call"
    )
    assert "get_current_weather" in decoded, (
        "the tool name survives rendering but not tokenisation"
    )
    assert ids, "tool prompt tokenised to an empty sequence"
    print("self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    rendered, ids, decoded = render_tool_prompt(TOOL_FIXTURE)
    if args.dump:
        print(rendered)
        print(f"\n[token_count={len(ids)}]\n")
        print(decoded)
    else:
        print(f"tool prompt rendered and tokenised ({len(ids)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
