#!/usr/bin/env python3
"""Compare EXO's loaded layer-0 tensors against the raw checkpoint.

The remaining suspect after sharding was eliminated is `sanitize()` in the pinned
`rltakashige/mlx-lm` fork, which maps checkpoint tensors onto the module tree.
This tests that directly and without OMLX: the same safetensors file, the same
quantization, so any tensor `sanitize()` passes through untouched must come back
bit-identical. A mismatch on a pass-through tensor names the mapping bug.

Why not compare against OMLX for this. OMLX would have to load and be trusted as
a reference; the checkpoint is the reference, and it is already on disk. Use OMLX
for the *hidden state* comparison (which layer first diverges), not for weights.

Why this fits in memory. The earlier single-node attempt OOMed evaluating all 43
layers. Loading is lazy and this touches layer 0 only, comparing one shard's
tensors at a time, so peak residency is one shard plus one layer.

Expected differences, which are NOT bugs:
  * `wqkv_a` — `sanitize()` concatenates the checkpoint's separate `attn.wq_a` and
    `attn.wkv` into one fused projection. Reported as EXPECTED-CONCAT.
  * MTP tensors — dropped at load. Absent from the module tree by design.

Usage:
    uv run python scripts/compare_loaded_vs_checkpoint.py
    uv run python scripts/compare_loaded_vs_checkpoint.py --layer 0 --verbose

Exit code is non-zero if any pass-through tensor differs, so this gates.

NOT TESTED ON HARDWARE: written without MLX available.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from exo.worker.engines.mlx.deepseek_v4_0731_loader import load_exo_model

DEFAULT_MODEL_PATH = Path(
    "/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp"
)

# Checkpoint tensors that `sanitize()` legitimately transforms. Anything else
# differing is a finding.
FUSED_INTO_WQKV = ("attn.wq_a", "attn.wkv")


def shard_index(model_path: Path) -> dict[str, str]:
    candidates = sorted(model_path.glob("*index.json"))
    if not candidates:
        raise SystemExit(f"no safetensors index in {model_path}")
    payload: dict[str, Any] = json.loads(candidates[0].read_text())
    weight_map: dict[str, str] = payload["weight_map"]
    return weight_map


def module_at(root: nn.Module, dotted: str) -> object | None:
    """Resolve `attn.wo_a.weight` against the module tree, or None if absent."""
    current: object = root
    for part in dotted.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if isinstance(current, nn.Module) and part in current:
            current = current[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        return None
    return current


def inner_layers(model: nn.Module) -> list[nn.Module]:
    for path in ("layers", "model.layers"):
        found = module_at(model, path)
        if isinstance(found, list) and found:
            return found  # pyright: ignore[reportUnknownVariableType]
    raise SystemExit("could not locate the decoder layer list on the loaded model")


def compare(a: mx.array, b: mx.array) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    # Packed quantized arrays are integer types; float tensors should also be
    # exact here, since a pass-through applies no arithmetic.
    equal = bool(mx.all(a == b).item())
    if equal:
        return True, "bit-identical"
    diff = mx.sum((a != b).astype(mx.int32))
    total = 1
    for dim in a.shape:
        total *= int(dim)
    return False, f"{int(diff.item())}/{total} elements differ"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prefix = f"model.layers.{args.layer}."
    weight_map = shard_index(args.model_path)
    wanted = sorted(k for k in weight_map if k.startswith(prefix))
    if not wanted:
        raise SystemExit(f"no checkpoint tensors under {prefix!r}")
    print(f"checkpoint tensors under {prefix}: {len(wanted)}")

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in wanted:
        by_shard[weight_map[key]].append(key)
    print(f"spanning {len(by_shard)} shard file(s)")

    print("\nloading model (single node, lazy)...")
    model, _config = load_exo_model(args.model_path, lazy=True, strict=False)
    layers = inner_layers(model)
    if args.layer >= len(layers):
        raise SystemExit(
            f"model has {len(layers)} layers; --layer {args.layer} is out of range"
        )
    layer = layers[args.layer]
    print(f"loaded; comparing layer {args.layer} of {len(layers)}")

    identical: list[str] = []
    differing: list[tuple[str, str]] = []
    absent: list[str] = []
    expected: list[str] = []

    for shard_file, keys in sorted(by_shard.items()):
        print(f"\n-- {shard_file} ({len(keys)} tensors)")
        raw: dict[str, mx.array] = mx.load(str(args.model_path / shard_file))  # pyright: ignore[reportAssignmentType]
        for key in keys:
            relative = key[len(prefix) :]
            if any(relative.startswith(f) for f in FUSED_INTO_WQKV):
                expected.append(relative)
                if args.verbose:
                    print(
                        f"   EXPECTED-CONCAT  {relative} (fused into wqkv_a by sanitize)"
                    )
                continue

            target = module_at(layer, relative)
            if not isinstance(target, mx.array):
                absent.append(relative)
                print(f"   ABSENT           {relative} (no such array on the module)")
                continue

            same, detail = compare(target, raw[key])
            if same:
                identical.append(relative)
                if args.verbose:
                    print(f"   identical        {relative} {tuple(target.shape)}")
            else:
                differing.append((relative, detail))
                print(f"   DIFFERS          {relative}: {detail}")
        del raw
        mx.clear_cache()

    print("\n" + "=" * 70)
    print(f"bit-identical        : {len(identical)}")
    print(f"expected transform   : {len(expected)}  {expected}")
    print(f"absent from module   : {len(absent)}  {absent}")
    print(f"UNEXPECTED DIFFERENCE: {len(differing)}")
    for name, detail in differing:
        print(f"   {name}: {detail}")

    if differing:
        print(
            "\nA pass-through tensor differing from the checkpoint is a sanitize()\n"
            "mapping defect. Check whether the tensor was written into the wrong\n"
            "slot, transposed, or reshaped."
        )
        return 1
    if absent:
        print(
            "\nNo unexpected value differences. Absences above may be benign (the\n"
            "module may store them under another name) — resolve each before\n"
            "concluding sanitize() is clean."
        )
        return 1
    print(
        "\nEvery pass-through tensor is bit-identical to the checkpoint. sanitize()\n"
        "maps layer weights faithfully, so the next step is a hidden-state\n"
        "comparison against OMLX layer by layer to find the first divergence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
