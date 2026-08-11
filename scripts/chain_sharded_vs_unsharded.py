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
    _force_float32_gate_inputs,
    _force_float32_hyper_connections,
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


def _install_gate_capture(
    ref_layers: list[object], shard_layers: list[object]
) -> dict[tuple[str, int], mx.array]:
    """Capture routed indices from the real gate calls inside each layer."""
    gate_ids: dict[int, tuple[str, int]] = {}
    for index, (ref_layer, shard_layer) in enumerate(
        zip(ref_layers, shard_layers, strict=True)
    ):
        gate_ids[id(ref_layer.ffn.gate)] = ("unsharded", index)  # type: ignore[attr-defined]
        gate_ids[id(shard_layer.ffn.gate)] = ("sharded", index)  # type: ignore[attr-defined]

    gate_type = type(ref_layers[0].ffn.gate)  # type: ignore[attr-defined]
    original = gate_type.__call__
    captured: dict[tuple[str, int], mx.array] = {}

    def capture_gate(self: object, x: mx.array, input_ids: mx.array | None = None):
        indices, weights = original(self, x, input_ids)
        label = gate_ids.get(id(self))
        if label is not None:
            captured[label] = indices
        return indices, weights

    gate_type.__call__ = capture_gate  # type: ignore[method-assign]
    return captured


def _report_expert_sets(
    layer_index: int, reference: mx.array, candidate: mx.array
) -> bool:
    reference_sets = mx.sort(reference.astype(mx.int32), axis=-1)
    candidate_sets = mx.sort(candidate.astype(mx.int32), axis=-1)
    mx.eval(reference_sets, candidate_sets)
    same = bool(mx.all(reference_sets == candidate_sets).item())
    if same:
        print(f"routing layer {layer_index:2d}  exact_set_agreement=True")
        return True

    reference_rows = reference_sets.tolist()
    candidate_rows = candidate_sets.tolist()
    mismatch_count = 0
    first_mismatch = ""
    for row_index, (reference_row, candidate_row) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        if reference_row != candidate_row:
            mismatch_count += 1
            if not first_mismatch:
                first_mismatch = (
                    f"token={row_index} unsharded={reference_row} "
                    f"sharded={candidate_row}"
                )
    print(
        f"routing layer {layer_index:2d}  exact_set_agreement=False "
        f"mismatched_tokens={mismatch_count} {first_mismatch}"
    )
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument(
        "--float32-all",
        action="store_true",
        help="keep hidden state float32 and use the full float32 hyper path",
    )
    ap.add_argument(
        "--float32-gate",
        action="store_true",
        help="cast MoE gate inputs to float32 before logits matmul",
    )
    ap.add_argument(
        "--expert-selection",
        action="store_true",
        help="capture exact top-k expert sets during every chained layer",
    )
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

    if args.float32_all:
        _force_float32_hyper_connections()
    if args.float32_gate:
        _force_float32_gate_inputs()

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

    captured = (
        _install_gate_capture(ref_layers, shard_layers)
        if args.expert_selection
        else None
    )
    routing_ok = True

    mx.random.seed(args.seed)
    hidden = int(ref_layers[0].attn.dim)  # pyright: ignore[reportAny]
    hc_mult = (
        int(ref_layers[0].hc_attn.mult)
        if hasattr(ref_layers[0].hc_attn, "mult")
        else 4
    )  # pyright: ignore[reportAny]
    initial = mx.random.normal((1, args.seq_len, hc_mult, hidden)).astype(
        mx.float32 if args.float32_all else mx.bfloat16
    )
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
        if args.float32_all:
            ref_hidden = ref_hidden.astype(mx.float32)
            shard_hidden = shard_hidden.astype(mx.float32)
        mx.eval(ref_hidden, shard_hidden)

        if captured is not None:
            reference_indices = captured.pop(("unsharded", layer_index), None)
            candidate_indices = captured.pop(("sharded", layer_index), None)
            if reference_indices is None or candidate_indices is None:
                raise RuntimeError(f"missing captured gate output at layer {layer_index}")
            mx.eval(reference_indices, candidate_indices)
            routing_ok &= _report_expert_sets(
                layer_index, reference_indices, candidate_indices
            )

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
    return 0 if routing_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
