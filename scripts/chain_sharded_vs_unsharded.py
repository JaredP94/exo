#!/usr/bin/env python3
"""Measure sharded/unsharded divergence as layers are chained.

The one-layer parity harness answers whether a single block stays within a
numeric tolerance. This harness feeds each block's output into the next block
on both paths, so it measures the quantity that matters for conditioning:
whether a small per-layer difference compounds through depth.

It uses two local MLX processes and materializes one layer at a time. Previous
layers are released after their output has been consumed so this remains a
layer-scale experiment rather than a full-checkpoint load.

    mlx.launch -n 2 scripts/chain_sharded_vs_unsharded.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import mlx.core as mx
from parity_sharded_vs_unsharded import (
    DEFAULT_MODEL_PATH,
    _layers,
    _OneLayerModel,
    _strategy,
    load_exo_model,
)


def _max_abs(reference: mx.array, candidate: mx.array) -> tuple[float, float, float]:
    diff = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
    max_abs = float(mx.max(diff).item())
    mean_abs = float(mx.mean(diff).item())
    reference_scale = float(mx.mean(mx.abs(reference.astype(mx.float32))).item())
    return max_abs, mean_abs, reference_scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    requested_depths = sorted(set(args.depths))
    max_depth = max(requested_depths)
    group = mx.distributed.init()
    n, rank = group.size(), group.rank()
    if n != 2:
        print(
            f"[rank {rank}] group size {n}, not 2. Launch with mlx.launch -n 2.",
            file=sys.stderr,
        )
        return 1

    say = print if rank == 0 else (lambda *a, **k: None)  # noqa: ARG005
    say(
        f"chained depth through {max_depth} layers, world size {n}, "
        f"seq_len {args.seq_len}"
    )
    say("loading reference and shard copies (lazy)...")
    ref_model, _ = load_exo_model(args.model_path, lazy=True, strict=False)
    shard_model, _ = load_exo_model(args.model_path, lazy=True, strict=False)
    ref_layers = _layers(ref_model)
    shard_layers = _layers(shard_model)
    if max_depth > len(ref_layers):
        raise SystemExit(
            f"requested depth {max_depth}, but model has only {len(ref_layers)} layers"
        )

    mx.random.seed(args.seed)
    hidden = int(ref_layers[0].attn.dim)  # pyright: ignore[reportAny]
    hc_mult = (
        int(ref_layers[0].hc_attn.mult)
        if hasattr(ref_layers[0].hc_attn, "mult")
        else 4
    )  # pyright: ignore[reportAny]
    initial = mx.random.normal((1, args.seq_len, hc_mult, hidden)).astype(mx.bfloat16)
    ref_hidden = initial
    shard_hidden = initial
    input_ids = mx.arange(args.seq_len, dtype=mx.int32)[None] % 1000
    mx.eval(ref_hidden, shard_hidden, input_ids)

    for layer_index in range(max_depth):
        ref_layer = ref_layers[layer_index]
        shard_layer = shard_layers[layer_index]
        mx.eval(ref_layer.parameters())
        mx.eval(shard_layer.parameters())
        for _progress in _strategy(group).shard_model(_OneLayerModel(shard_layer)):
            pass
        mx.eval(shard_layer.parameters())

        ref_hidden = ref_layer(ref_hidden, None, input_ids)  # pyright: ignore[reportAny]
        shard_hidden = shard_layer(shard_hidden, None, input_ids)  # pyright: ignore[reportAny]
        mx.eval(ref_hidden, shard_hidden)

        if rank == 0:
            max_abs, mean_abs, reference_scale = _max_abs(ref_hidden, shard_hidden)
            marker = " *" if layer_index + 1 in requested_depths else ""
            print(
                f"depth {layer_index + 1:2d}{marker}  max_abs={max_abs:.6g} "
                f"mean_abs={mean_abs:.6g} ref_scale={reference_scale:.6g}"
            )

        # The model roots retain their layer lists, so remove the materialized
        # layer references before advancing to keep this a layer-scale run.
        ref_layers[layer_index] = None  # type: ignore[assignment]
        shard_layers[layer_index] = None  # type: ignore[assignment]
        del ref_layer, shard_layer
        gc.collect()
        mx.clear_cache()

    if rank == 0:
        print("* requested reporting depth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
