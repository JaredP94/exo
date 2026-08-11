#!/usr/bin/env python3
"""Do rank 0 and rank 1 keep DIFFERENT blocks when a weight is sharded?

Every probe so far compared shapes, and shapes are identical on both ranks by
construction: both end at `wo_a.weight=(8192, 512)` whether they hold different
halves or the same one. If `shard_inplace` is not applying `group.rank()` when
selecting the block, both ranks compute the identical partial, `_AllSumLinear`
doubles it, and half the heads contribute nothing at any of 43 layers — with
every shape and every arrays_agree check correct.

This needs a `mx.distributed` group of size 2, NOT two Macs. Two processes on one
host answer the question, because it is about `shard_inplace`'s use of
`group.rank()`, which is machine-agnostic. It uses a synthetic V4Attention rather
than the 154 GiB checkpoint, so it is fast and repeatable.

It replicates production's sharding callbacks from `tensor_auto_parallel`
verbatim, including passing the REAL group. Passing `group=None` is what made an
earlier probe report a false all-clear.

Launch (either should work; try both if one is unavailable):

    mpirun -np 2 uv run python scripts/probe_shard_rank_selection.py
    uv run mlx.launch -n 2 scripts/probe_shard_rank_selection.py

Exit code is non-zero if any sharded weight is identical across ranks, or if the
group is not size 2 — a size-1 group cannot answer the question and must fail
loudly rather than silently pass.

NOT TESTED ON HARDWARE: written without MLX available. Expect to debug the
launcher invocation on first run.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import mlx.nn as nn
from exo.worker.engines.mlx.auto_parallel import _shard_v4_attention_heads
from mlx.nn.layers.distributed import shard_inplace
from mlx.utils import tree_map
from mlx_lm.models.deepseek_v4 import ModelArgs, V4Attention

# Structurally identical to the real checkpoint: n_heads divisible by o_groups,
# heads_per_group divisible by 2. Real values are n_heads=64, head_dim=512,
# o_groups=8, o_lora_rank=1024, moe_intermediate_size=2048.
SYNTHETIC: dict[str, int] = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "head_dim": 16,
    "qk_rope_head_dim": 8,
    "o_groups": 2,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "num_hidden_layers": 1,
}

PROBE_ELEMENTS = 16


def _probe_vector(weight: mx.array) -> mx.array:
    """A float32 vector that exactly represents the first PROBE_ELEMENTS values.

    Quantized weights are uint32, whose full range is not exactly representable
    in float32 — a naive cast would make distinct slices compare equal for large
    values. Split into 16-bit halves, each of which is exact.
    """
    flat = weight.reshape(-1)[:PROBE_ELEMENTS]
    if flat.dtype in (mx.float32, mx.float16, mx.bfloat16):
        return flat.astype(mx.float32)
    packed = flat.astype(mx.uint32)
    low = (packed % 65536).astype(mx.float32)
    high = (packed // 65536).astype(mx.float32)
    return mx.concatenate([low, high])


def _ramp(shape: tuple[int, ...], dtype: mx.Dtype = mx.float32) -> mx.array:
    """Position-dependent values, so every element is distinguishable."""
    total = 1
    for dim in shape:
        total *= int(dim)
    return (mx.arange(total, dtype=mx.float32) % 4096).reshape(shape).astype(dtype)


def _deterministic_weights(module: nn.Module) -> None:
    """Replace every parameter with a position-derived ramp.

    Random initialisation would differ across ranks anyway, since each process
    seeds independently, so a post-shard difference would prove nothing. Deriving
    values from position makes both ranks start from an identical tensor, so any
    difference after sharding is caused by the shard and nothing else.
    """

    def replace(value: mx.array) -> mx.array:
        return _ramp(tuple(int(d) for d in value.shape), value.dtype)

    module.update(tree_map(replace, module.parameters()))  # pyright: ignore[reportAny]
    mx.eval(module.parameters())


def main() -> int:
    group = mx.distributed.init()
    n, rank = group.size(), group.rank()

    if n != 2:
        print(
            f"[rank {rank}] group size is {n}, not 2. A size-1 group cannot answer "
            f"this question — shard_inplace would be a no-op and every check would "
            f"trivially pass. Launch under mpirun -np 2 or mlx.launch -n 2.",
            file=sys.stderr,
        )
        return 1

    # Production's callbacks from tensor_auto_parallel, verbatim.
    segments = 1

    def _sharded_to_all(path: str, weight: mx.array) -> tuple[int, int] | None:
        if path.endswith("bias"):
            weight /= n
            return None
        return -1, segments

    def _all_to_sharded(path: str, weight: mx.array) -> tuple[int, int]:
        if path.endswith("bias"):
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    args = ModelArgs(**SYNTHETIC)  # pyright: ignore[reportArgumentType]
    attn = V4Attention(args, layer_id=0)
    _deterministic_weights(attn)
    nn.quantize(attn, group_size=32, bits=4)
    mx.eval(attn.parameters())

    dense = nn.Linear(64, 64, bias=False)
    dense.weight = _ramp((64, 64))
    mx.eval(dense.parameters())

    # The real sequence: interleaved head slice, then the input-dim split.
    _shard_v4_attention_heads(attn, world_size=n, rank=rank)
    shard_inplace(attn.wo_a, sharding=_sharded_to_all, group=group)  # pyright: ignore[reportArgumentType]
    # Control: an unquantized linear on the same path. If this differs across
    # ranks but the quantized one does not, the defect is quantization-specific.
    shard_inplace(dense, sharding=_all_to_sharded, group=group)  # pyright: ignore[reportArgumentType]
    mx.eval(attn.parameters(), dense.parameters())

    probes: dict[str, mx.array] = {
        "attn.wo_a.weight": attn.wo_a.weight,  # pyright: ignore[reportAny]
        "attn.wq_b.weight": attn.wq_b.weight,  # pyright: ignore[reportAny]
        "attn.attn_sink": attn.attn_sink,
        "control.dense.weight": dense.weight,
    }

    failures: list[str] = []
    for label, weight in probes.items():
        mine = _probe_vector(weight)
        gathered = mx.distributed.all_gather(mine[None], group=group)
        mx.eval(gathered)
        if rank == 0:
            a, b = gathered[0], gathered[1]
            same = bool(mx.all(a == b).item())
            verdict = "IDENTICAL — BUG" if same else "differ (correct)"
            print(f"  {label:24} shape={tuple(weight.shape)}  ranks {verdict}")
            print(f"    r0 first4: {[int(v) for v in a[:4].tolist()]}")  # pyright: ignore[reportAny]
            print(f"    r1 first4: {[int(v) for v in b[:4].tolist()]}")  # pyright: ignore[reportAny]
            if same:
                failures.append(label)

    if rank != 0:
        return 0

    print()
    if failures:
        print(
            f"FAIL: {failures} identical across ranks. Both ranks kept the same "
            f"block, so each computes the same partial and the all_sum doubles it. "
            f"shard_inplace is not applying group.rank() for these weights."
        )
        return 1
    print(
        "PASS: every sharded weight differs across ranks, so the block selection "
        "does use the rank. This eliminates the last structural candidate in the "
        "sharding path; move to the sanitize() weight-mapping comparison."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
