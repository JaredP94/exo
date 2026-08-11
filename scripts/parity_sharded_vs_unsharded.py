#!/usr/bin/env python3
"""Does the SHARDED forward pass reproduce the unsharded one?

Everything verified so far concerns weights: shapes halve correctly, `scales`
tracks `weight`, the ranks keep different blocks, `sanitize()` maps the
checkpoint bit-exactly, and the unsharded layer math matches OMLX to within
quantization noise across all three layer regimes.

None of that tested the sharded *computation*. `DeepseekV4ShardingStrategy`
installs two collectives — `_AllSumLinear` around `attn.wo_b` and `ShardedMoEV4`
around the FFN — and by elimination the defect is now in that path or nowhere.

This compares EXO against EXO: one layer, same weights, same input, sharded
versus not. No OMLX, so nothing external has to be trusted or made to agree.

It bisects in one run:

  1. attention  — isolates head sharding, the wo_a input split and _AllSumLinear
  2. ffn        — isolates the expert sharding and ShardedMoEV4
  3. full block — catches anything in the hyper-connection wrapping around them

The pre-attention and pre-FFN inputs are computed once from the UNSHARDED block,
so both sides receive byte-identical input and any difference is caused by the
sharding alone.

Needs a group of size 2, not two Macs — two processes on one host. Memory is one
layer times two copies, roughly 8 GB per process, so this does not need the
cluster and is unaffected by the topology-discovery failure.

    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 0
    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 2   # Indexer
    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 3   # Compressor
    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 3 --float32-hyper
    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 3 --float32-sinkhorn
    mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer 3 --expert-selection

`compress_ratios` is [0, 0, 4, 128, 4, 128, ...], so layer 0 has no compressor,
layer 2 runs the Indexer, and layer 3 runs the Compressor without it. Cover all
three; they are different code paths. The harness also exposes the layer-4 and
layer-5 ratio comparison and reports the input/output deltas at both
hyper-connection boundaries. `--float32-hyper` is an experiment switch only; it
forces the hyper-connection collapse/residual path to float32 and does not alter
production code. `--float32-sinkhorn` uses the separate float32 Sinkhorn
normalization path but casts the collapse result back to the input dtype, leaving
downstream layers in their normal dtype. Both switches are experiment-only.

NOT TESTED ON HARDWARE: written without MLX available.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.deepseek_v4 as dsv4
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from exo.worker.engines.mlx.auto_parallel import DeepseekV4ShardingStrategy
from exo.worker.engines.mlx.deepseek_v4_0731_loader import load_exo_model
from mlx.nn.layers.distributed import shard_inplace, shard_linear

DEFAULT_MODEL_PATH = Path(
    "/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp"
)

# Loose enough to absorb quantized-matmul and reduction-order noise — the OMLX
# parity run saw up to 2**-7 on block output — but far tighter than a
# structurally wrong result, which is O(1) or larger.
DEFAULT_TOLERANCE = 0.05


def _force_float32_hyper_connections() -> None:
    """Replace the fused bf16 collapse with a fully float32 experiment path."""

    def float32_hc_sinkhorn(
        mixes: mx.array,
        scale: mx.array,
        base: mx.array,
        x: mx.array,
        hc_mult: int,
        sinkhorn_iters: int,
        eps: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        pre, post, comb = dsv4.hc_split_sinkhorn(
            mixes, scale, base, hc_mult, sinkhorn_iters, eps
        )
        collapsed = (pre[:, :, None, :] @ x.astype(mx.float32)).squeeze(2)
        return collapsed, post, comb

    dsv4.hc_sinkhorn_collapse = float32_hc_sinkhorn  # pyright: ignore[reportAttributeAccessIssue]


def _force_float32_sinkhorn_only() -> None:
    """Use float32 Sinkhorn normalization while preserving the input dtype."""

    def float32_sinkhorn_only(
        mixes: mx.array,
        scale: mx.array,
        base: mx.array,
        x: mx.array,
        hc_mult: int,
        sinkhorn_iters: int,
        eps: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        pre, post, comb = dsv4.hc_split_sinkhorn(
            mixes, scale, base, hc_mult, sinkhorn_iters, eps
        )
        collapsed = (pre[:, :, None, :] @ x.astype(mx.float32)).squeeze(2)
        return collapsed.astype(x.dtype), post, comb

    dsv4.hc_sinkhorn_collapse = float32_sinkhorn_only  # pyright: ignore[reportAttributeAccessIssue]


class _OneLayerModel(nn.Module):
    """`shard_model` only ever touches `model.layers`, so this is enough.

    Sharding the real 43-layer model would evaluate every layer and exhaust
    memory; this restricts the work to the layer under test.
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layers: list[nn.Module] = [layer]


def _strategy(group: mx.distributed.Group) -> DeepseekV4ShardingStrategy:
    """Rebuild the exact bindings `tensor_auto_parallel` uses.

    Hand-rolling the sharding instead would skip `_AllSumLinear` and
    `ShardedMoEV4`, which are the components actually under suspicion.
    """
    n = group.size()
    segments = 1

    def _all_to_sharded(path: str, weight: mx.array) -> tuple[int, int]:
        if path.endswith("bias"):
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    def _sharded_to_all(path: str, weight: mx.array) -> tuple[int, int] | None:
        if path.endswith("bias"):
            weight /= n
            return None
        return -1, segments

    return DeepseekV4ShardingStrategy(
        group=group,
        all_to_sharded_linear=partial(
            shard_linear, sharding="all-to-sharded", group=group
        ),
        sharded_to_all_linear=partial(
            shard_linear, sharding="sharded-to-all", group=group
        ),
        all_to_sharded_linear_in_place=partial(
            shard_inplace,
            sharding=_all_to_sharded,  # pyright: ignore[reportArgumentType]
            group=group,
        ),
        sharded_to_all_linear_in_place=partial(
            shard_inplace,
            sharding=_sharded_to_all,  # pyright: ignore[reportArgumentType]
            group=group,
        ),
    )


def _layers(model: nn.Module) -> list[nn.Module]:
    for holder in (model, getattr(model, "model", None)):
        found = getattr(holder, "layers", None) if holder is not None else None
        if isinstance(found, list) and found:
            return found  # pyright: ignore[reportUnknownVariableType]
    raise SystemExit("could not locate the decoder layer list on the loaded model")


def _report(label: str, reference: mx.array, candidate: mx.array, tol: float) -> bool:
    if reference.shape != candidate.shape:
        print(
            f"  {label:14} SHAPE MISMATCH {tuple(reference.shape)} vs {tuple(candidate.shape)}"
        )
        return False
    diff = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
    max_abs = float(mx.max(diff).item())
    mean_abs = float(mx.mean(diff).item())
    scale = float(mx.mean(mx.abs(reference.astype(mx.float32))).item())
    ok = max_abs <= tol
    # A doubled tensor is the signature of an all_sum over unsharded partials.
    ratio = (
        float(
            mx.mean(candidate.astype(mx.float32))
            / mx.mean(reference.astype(mx.float32))
        )
        if scale > 0
        else float("nan")
    )
    print(
        f"  {label:14} {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} ref_scale={scale:.6g} mean_ratio={ratio:.4f}"
    )
    if not ok and abs(ratio - 2.0) < 0.05:
        print("      mean_ratio ~= 2.0 — the sharded result is DOUBLED, which is what")
        print("      an all_sum over two unsharded partials produces.")
    return ok


def _report_with_input(
    label: str,
    reference_input: mx.array,
    candidate_input: mx.array,
    reference_output: mx.array,
    candidate_output: mx.array,
    tol: float,
) -> bool:
    """Report a stage delta and how much it amplifies its input delta."""
    if reference_output.shape != candidate_output.shape:
        print(
            f"  {label:18} SHAPE MISMATCH "
            f"{tuple(reference_output.shape)} vs {tuple(candidate_output.shape)}"
        )
        return False
    input_diff = mx.abs(
        reference_input.astype(mx.float32) - candidate_input.astype(mx.float32)
    )
    output_diff = mx.abs(
        reference_output.astype(mx.float32) - candidate_output.astype(mx.float32)
    )
    max_input = float(mx.max(input_diff).item())
    max_output = float(mx.max(output_diff).item())
    amplification = max_output / max_input if max_input else float("nan")
    ok = max_output <= tol
    print(
        f"  {label:18} {'PASS' if ok else 'FAIL'}  "
        f"max_abs_in={max_input:.6g} max_abs_out={max_output:.6g} "
        f"amplification={amplification:.6g}"
    )
    return ok


def _report_expert_selection(
    reference_indices: mx.array, candidate_indices: mx.array
) -> bool:
    """Compare routed expert sets exactly, ignoring only top-k ordering."""
    if reference_indices.shape != candidate_indices.shape:
        print(
            "  expert_selection SHAPE MISMATCH "
            f"{tuple(reference_indices.shape)} vs {tuple(candidate_indices.shape)}"
        )
        return False

    reference_sets = mx.sort(reference_indices.astype(mx.int32), axis=-1)
    candidate_sets = mx.sort(candidate_indices.astype(mx.int32), axis=-1)
    mx.eval(reference_sets, candidate_sets)
    same = bool(mx.all(reference_sets == candidate_sets).item())
    print(
        f"  expert_selection {'PASS' if same else 'FAIL'}  "
        f"shape={tuple(reference_indices.shape)}  exact_set_agreement={same}"
    )
    if not same:
        reference_rows = reference_sets.tolist()
        candidate_rows = candidate_sets.tolist()
        for row_index, (reference_row, candidate_row) in enumerate(
            zip(reference_rows, candidate_rows, strict=True)
        ):
            if reference_row != candidate_row:
                print(
                    f"      first mismatch at flattened token {row_index}: "
                    f"unsharded={reference_row} sharded={candidate_row}"
                )
                break
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    float32_group = ap.add_mutually_exclusive_group()
    float32_group.add_argument(
        "--float32-hyper",
        action="store_true",
        help="force the hyper-connection collapse and residual path to float32",
    )
    float32_group.add_argument(
        "--float32-sinkhorn",
        action="store_true",
        help="use float32 Sinkhorn normalization but preserve the input dtype",
    )
    ap.add_argument(
        "--expert-selection",
        action="store_true",
        help="compare exact top-k expert index sets and stop before FFN execution",
    )
    args = ap.parse_args()

    if args.float32_sinkhorn:
        _force_float32_sinkhorn_only()
    elif args.float32_hyper:
        _force_float32_hyper_connections()

    group = mx.distributed.init()
    n, rank = group.size(), group.rank()
    if n != 2:
        print(
            f"[rank {rank}] group size {n}, not 2. Sharding would be a no-op and "
            f"every check would pass trivially. Launch with mlx.launch -n 2.",
            file=sys.stderr,
        )
        return 1

    say = print if rank == 0 else (lambda *a, **k: None)  # noqa: ARG005
    say(f"layer {args.layer}, world size {n}, seq_len {args.seq_len}")

    # Two lazy loads: materialise only the layer under test in each.
    say("loading reference and shard copies (lazy)...")
    ref_model, _ = load_exo_model(args.model_path, lazy=True, strict=False)
    shard_model_root, _ = load_exo_model(args.model_path, lazy=True, strict=False)
    ref_layer = _layers(ref_model)[args.layer]
    shard_layer = _layers(shard_model_root)[args.layer]
    mx.eval(ref_layer.parameters())
    mx.eval(shard_layer.parameters())
    say("materialised.")

    say("sharding via the real DeepseekV4ShardingStrategy...")
    for _progress in _strategy(group).shard_model(_OneLayerModel(shard_layer)):
        pass
    mx.eval(shard_layer.parameters())
    say("sharded.")

    # Identical on both ranks: same seed, and neither the RNG nor these tensors
    # depend on rank.
    mx.random.seed(args.seed)
    hidden = int(ref_layer.attn.dim)  # pyright: ignore[reportAny]
    hc_mult = int(ref_layer.hc_attn.mult) if hasattr(ref_layer.hc_attn, "mult") else 4  # pyright: ignore[reportAny]
    h = mx.random.normal((1, args.seq_len, hc_mult, hidden)).astype(mx.bfloat16)
    input_ids = mx.arange(args.seq_len, dtype=mx.int32)[None] % 1000
    mx.eval(h, input_ids)

    ok = True

    # 1. Attention and its hyper-connection boundaries. The standalone attention
    # comparison remains useful for isolating sharding, while the boundary
    # reports show whether the full-block path amplifies an upstream delta.
    ref_hc_attn, ref_attn_post, ref_attn_comb = ref_layer.hc_attn.hc_pre(h)  # pyright: ignore[reportAny]
    shard_hc_attn, shard_attn_post, shard_attn_comb = shard_layer.hc_attn.hc_pre(h)  # pyright: ignore[reportAny]
    ref_y_in = ref_layer.attn_norm(ref_hc_attn)  # pyright: ignore[reportAny]
    shard_y_in = shard_layer.attn_norm(shard_hc_attn)  # pyright: ignore[reportAny]
    mx.eval(
        ref_hc_attn,
        shard_hc_attn,
        ref_y_in,
        shard_y_in,
    )
    ref_attn = ref_layer.attn(ref_y_in, cache=None)  # pyright: ignore[reportAny]
    shard_attn = shard_layer.attn(shard_y_in, cache=None)  # pyright: ignore[reportAny]
    ref_attn_isolated = ref_layer.attn(ref_y_in, cache=None)  # pyright: ignore[reportAny]
    shard_attn_isolated = shard_layer.attn(ref_y_in, cache=None)  # pyright: ignore[reportAny]
    mx.eval(ref_attn, shard_attn, ref_attn_isolated, shard_attn_isolated)
    ref_after_attn = ref_layer.hc_attn.hc_post(  # pyright: ignore[reportAny]
        ref_attn, h, ref_attn_post, ref_attn_comb
    )
    shard_after_attn = shard_layer.hc_attn.hc_post(  # pyright: ignore[reportAny]
        shard_attn, h, shard_attn_post, shard_attn_comb
    )
    mx.eval(ref_after_attn, shard_after_attn)
    say("\nattention (head shard + wo_a input split + _AllSumLinear):")
    if rank == 0:
        ok &= _report_with_input(
            "hc_attn_pre", h, h, ref_hc_attn, shard_hc_attn, args.tolerance
        )
        ok &= _report(
            "attn_output (isolated)",
            ref_attn_isolated,
            shard_attn_isolated,
            args.tolerance,
        )
        ok &= _report(
            "attn_output (chained)", ref_attn, shard_attn, args.tolerance
        )
        ok &= _report_with_input(
            "hc_attn_post",
            ref_attn,
            shard_attn,
            ref_after_attn,
            shard_after_attn,
            args.tolerance,
        )

    # 2. FFN and its hyper-connection boundaries.
    ref_hc_ffn, ref_ffn_post, ref_ffn_comb = ref_layer.hc_ffn.hc_pre(ref_after_attn)  # pyright: ignore[reportAny]
    shard_hc_ffn, shard_ffn_post, shard_ffn_comb = shard_layer.hc_ffn.hc_pre(shard_after_attn)  # pyright: ignore[reportAny]
    ref_y_ffn = ref_layer.ffn_norm(ref_hc_ffn)  # pyright: ignore[reportAny]
    shard_y_ffn = shard_layer.ffn_norm(shard_hc_ffn)  # pyright: ignore[reportAny]
    mx.eval(ref_hc_ffn, shard_hc_ffn, ref_y_ffn, shard_y_ffn)

    if args.expert_selection:
        ref_expert_indices, _ = ref_layer.ffn.gate(  # pyright: ignore[reportAny]
            ref_y_ffn, input_ids
        )
        shard_expert_indices, _ = shard_layer.ffn.gate(  # pyright: ignore[reportAny]
            shard_y_ffn, input_ids
        )
        mx.eval(ref_expert_indices, shard_expert_indices)
        say("\nexpert selection (exact top-k index sets):")
        if rank == 0:
            return 0 if _report_expert_selection(
                ref_expert_indices, shard_expert_indices
            ) else 1
        return 0

    ref_ffn = ref_layer.ffn(ref_y_ffn, input_ids)  # pyright: ignore[reportAny]
    shard_ffn = shard_layer.ffn(shard_y_ffn, input_ids)  # pyright: ignore[reportAny]
    ref_ffn_isolated = ref_layer.ffn(ref_y_ffn, input_ids)  # pyright: ignore[reportAny]
    shard_ffn_isolated = shard_layer.ffn(ref_y_ffn, input_ids)  # pyright: ignore[reportAny]
    mx.eval(ref_ffn, shard_ffn, ref_ffn_isolated, shard_ffn_isolated)
    ref_after_ffn = ref_layer.hc_ffn.hc_post(  # pyright: ignore[reportAny]
        ref_ffn, ref_after_attn, ref_ffn_post, ref_ffn_comb
    )
    shard_after_ffn = shard_layer.hc_ffn.hc_post(  # pyright: ignore[reportAny]
        shard_ffn, shard_after_attn, shard_ffn_post, shard_ffn_comb
    )
    mx.eval(ref_after_ffn, shard_after_ffn)
    say("\nffn (expert shard + ShardedMoEV4):")
    if rank == 0:
        ok &= _report_with_input(
            "hc_ffn_pre",
            ref_after_attn,
            shard_after_attn,
            ref_hc_ffn,
            shard_hc_ffn,
            args.tolerance,
        )
        ok &= _report(
            "ffn_output (isolated)",
            ref_ffn_isolated,
            shard_ffn_isolated,
            args.tolerance,
        )
        ok &= _report("ffn_output (chained)", ref_ffn, shard_ffn, args.tolerance)
        ok &= _report_with_input(
            "hc_ffn_post",
            ref_ffn,
            shard_ffn,
            ref_after_ffn,
            shard_after_ffn,
            args.tolerance,
        )

    # 3. Whole block, which adds the hyper-connection wrapping.
    ref_block = ref_layer(h, None, input_ids)
    shard_block = shard_layer(h, None, input_ids)
    mx.eval(ref_block, shard_block)
    say("\nfull block (adds hyper-connections):")
    if rank == 0:
        ok &= _report("block_output", ref_block, shard_block, args.tolerance)

    # 4. Both ranks must agree after the collectives.
    gathered = mx.distributed.all_gather(
        mx.mean(shard_block.astype(mx.float32)).reshape(1), group=group
    )
    mx.eval(gathered)
    if rank == 0:
        a, b = float(gathered[0].item()), float(gathered[1].item())
        agree = abs(a - b) <= 1e-4 * max(1.0, abs(a))
        print(
            f"\nranks agree after all_sum: {'yes' if agree else 'NO'} ({a:.6g} vs {b:.6g})"
        )
        ok &= agree

    if rank != 0:
        return 0

    print()
    if ok:
        print(
            "PASS: the sharded forward pass reproduces the unsharded one for this\n"
            "layer. Run the other two regimes (--layer 2 and --layer 3) before\n"
            "concluding; if all three pass, the defect is not in a single layer and\n"
            "the next suspects are cross-layer state — the KV cache and the\n"
            "hc_head/norm applied after the final layer."
        )
        return 0
    print(
        "FAIL: sharding changes this layer's output. The first failing stage above\n"
        "names the component: attn_output implicates head sharding, the wo_a input\n"
        "split or _AllSumLinear; ffn_output implicates the expert sharding or\n"
        "ShardedMoEV4; a passing attn and ffn with a failing block implicates the\n"
        "hyper-connection wrapping."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
