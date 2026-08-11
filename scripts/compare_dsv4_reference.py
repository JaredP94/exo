#!/usr/bin/env python3
"""Single-node DeepSeek V4 0731 forward-pass comparison against the goldens.

Purpose: decide whether incoherent cluster output comes from the weights and
quantization or from the distributed path, by running the SAME checkpoint through
EXO's loader on one machine.

WHY NOT `scripts/test_single_node_gen.py` AS WRITTEN. That script prompts with
`[0, 128821, 15, 16, 17]` — BOS, then `<think>`, then three arbitrary token ids.
That is not a prompt the model was trained to continue: there is no `<｜User｜>`
turn and no `<｜Assistant｜>` anchor. A perfectly healthy model produces garbage
from it, so the test cannot distinguish a broken model from a nonsense prompt and
will appear to confirm the bug either way. This script uses the committed golden
token ids instead, which are the real rendered prompt.

NOT TESTED ON HARDWARE. Written in an environment without MLX. Treat the first
run as debugging the script as much as the model.

Usage:
    uv run python scripts/compare_dsv4_reference.py
    uv run python scripts/compare_dsv4_reference.py --case simple_thinking_off
    uv run python scripts/compare_dsv4_reference.py --max-tokens 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from exo.worker.engines.mlx.deepseek_v4_0731_loader import load_exo_model
from mlx_lm.generate import generate_step

DEFAULT_MODEL_PATH = Path(
    "/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp"
)
GOLDENS = Path("src/exo/worker/tests/fixtures/deepseek_v4_0731_prompt_goldens.json")

EOS_ID = 1


def load_golden(case: str) -> tuple[str, list[int]]:
    payload: dict[str, Any] = json.loads(GOLDENS.read_text())
    cases: dict[str, Any] = payload["cases"]
    if case not in cases:
        raise SystemExit(f"unknown case {case!r}; available: {sorted(cases)}")
    entry = cases[case]
    return entry["expected_prompt_text"], list(entry["expected_token_ids"])


def report_first_token(model: nn.Module, prompt_ids: list[int], top_k: int) -> None:
    """Top-k first-token logits.

    This is the comparison to run against OMLX, not generated strings. One
    divergent token cascades, so comparing completions tells you only THAT
    something is wrong. The first token's ranked logits tell you whether the
    forward pass is wrong at all, and bisecting by layer from there tells you
    where.
    """
    logits = model(mx.array(prompt_ids)[None])[:, -1, :]
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    order = mx.argsort(-logprobs[0])[:top_k]
    mx.eval(order, logprobs)

    print(f"\n  top {top_k} first-token candidates:")
    for rank, idx in enumerate(order.tolist(), 1):  # pyright: ignore[reportAny]
        token_id = int(idx)  # pyright: ignore[reportAny]
        lp = float(logprobs[0, token_id].item())
        print(f"    {rank}. id={token_id:<7} logprob={lp:9.4f}")

    finite = bool(mx.all(mx.isfinite(logits)).item())
    print(f"  all logits finite: {finite}")
    if not finite:
        print("  *** NaN or Inf in the logits — this alone explains incoherent output")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--case", default="simple_thinking_on")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "load with strict=True. Production uses strict=False, which silently "
            "tolerates missing or unexpected weights; a strict failure is itself "
            "a finding worth having"
        ),
    )
    args = ap.parse_args()

    prompt_text, prompt_ids = load_golden(args.case)
    print(f"case          : {args.case}")
    print(f"prompt text   : {prompt_text!r}")
    print(f"prompt ids    : {prompt_ids}")
    print(f"strict load   : {args.strict}")

    print("\nloading (single node, no JACCL)...")
    model, _config = load_exo_model(args.model_path, lazy=True, strict=args.strict)
    mx.eval(model)
    print("loaded.")

    report_first_token(model, prompt_ids, args.top_k)

    # generate_step defaults to argmax when no sampler is passed, so this is
    # greedy and must be reproducible run to run on one machine.
    print(f"\ngreedy continuation, up to {args.max_tokens} tokens:")
    generated: list[int] = []
    for (token, _logprobs), _ in zip(
        generate_step(mx.array(prompt_ids)[None], model),
        range(args.max_tokens),
        strict=False,
    ):
        token_id = int(token.item())  # pyright: ignore[reportAny]
        generated.append(token_id)
        if token_id == EOS_ID:
            break

    print(f"  ids     : {generated}")
    print(f"  hit EOS : {EOS_ID in generated}")

    # Detokenize through the same path EXO uses, so the text is comparable with
    # what the cluster returned.
    try:
        from exo.shared.types.common import ModelId  # noqa: PLC0415
        from exo.worker.engines.mlx.utils_mlx import (
            load_tokenizer_for_model_id,  # noqa: PLC0415
        )

        tokenizer = load_tokenizer_for_model_id(
            ModelId("Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp"), args.model_path
        )
        print(f"  text    : {tokenizer.decode(generated)!r}")
    except Exception as e:  # noqa: BLE001 - diagnostic script, any failure is informative
        print(f"  (detokenization unavailable: {type(e).__name__}: {e})")

    print(
        "\nInterpretation:\n"
        "  Coherent here but incoherent on the cluster -> the distributed path.\n"
        "  Incoherent here too -> weights or quantization. Run the same golden\n"
        "  prompt through OMLX next and compare the top-k above; if OMLX's top-1\n"
        "  differs, bisect by layer from the first divergence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
