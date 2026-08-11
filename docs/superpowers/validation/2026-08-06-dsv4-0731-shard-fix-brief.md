# Instrumentation patch: observe the V4 attention shard on the live cluster

Apply to `src/exo/worker/engines/mlx/auto_parallel.py`. Deliberately in EXO and
not in the `mlx_lm` fork, so `uv sync` cannot wipe it.

This produces an **observation** of what sharding does to `wo_a`, replacing the
hand arithmetic in the 2026-08-06 report. It logs once, for layer 0 only, so it
costs nothing and does not spam 43 layers.

NOT TESTED — written without MLX available. Review before running.

## 1. Add the helper, above `class DeepseekV4ShardingStrategy`

```python
def _log_v4_attention_shard(
    attn: V4Attention,
    stage: Literal["before", "after"],
    world_size: int,
    rank: int,
) -> None:
    """One-shot diagnostic for the wo_a grouped-projection contract.

    `_grouped_output_projection` computes its input width from the SHARDED head
    count::

        group_feat = (self.n_heads * self.head_dim) // self.n_groups

    but takes the weight's per-group width from whatever `wo_a` happens to be::

        weight = woa_w.reshape(n_groups, o_lora_rank, -1)

    Those two numbers come from different sources and nothing asserts they
    agree. This logs both so a disagreement is visible rather than inferred.
    """
    wo_a = attn.wo_a
    weight = getattr(wo_a, "weight", None)
    shape = None if weight is None else tuple(weight.shape)  # pyright: ignore[reportAny]

    group_feat = (attn.n_heads * attn.head_dim) // attn.n_groups
    implied = None
    if weight is not None and attn.n_groups:
        # Mirror the reshape in _grouped_output_projection: the trailing -1 is
        # the per-group width, PACKED for a quantized weight.
        rows = int(weight.shape[0])  # pyright: ignore[reportAny]
        per_group_rows = rows // attn.n_groups
        implied = (
            per_group_rows,
            None if weight.ndim < 2 else int(weight.shape[-1]),  # pyright: ignore[reportAny]
        )

    logger.info(
        f"V4 shard diagnostic [{stage}] rank={rank}/{world_size} "
        f"n_heads={attn.n_heads} head_dim={attn.head_dim} "
        f"n_groups={attn.n_groups} o_lora_rank={attn.o_lora_rank} "
        f"quantized={isinstance(wo_a, nn.QuantizedLinear)} "
        f"wo_a.weight.shape={shape} "
        f"expected_group_feat={group_feat} "
        f"wo_a_per_group(rows, trailing)={implied}"
    )
```

## 2. Call it around the shard, layer 0 only

In `DeepseekV4ShardingStrategy.shard_model`, replace:

```python
            _shard_v4_attention_heads(layer.attn, self.N, self.group.rank())
            self.sharded_to_all_linear_in_place(layer.attn.wo_a)
```

with:

```python
            if i == 0:
                _log_v4_attention_shard(
                    layer.attn, "before", self.N, self.group.rank()
                )
            _shard_v4_attention_heads(layer.attn, self.N, self.group.rank())
            if i == 0:
                _log_v4_attention_shard(
                    layer.attn, "heads-sharded", self.N, self.group.rank()
                )
            self.sharded_to_all_linear_in_place(layer.attn.wo_a)
            if i == 0:
                _log_v4_attention_shard(
                    layer.attn, "after", self.N, self.group.rank()
                )
```

`stage` is typed `Literal["before", "after"]` above; widen it to include
`"heads-sharded"` or drop the annotation to `str`.

## 3. What to look for

Start the two-rank cluster and read the three log lines from **each** rank.

| Observation | Meaning |
|---|---|
| `rank=0/1` — world_size is 1 | You are not running two-rank TP. Nothing below is valid. This is also what made the previous single-process test a no-op. |
| `wo_a.weight.shape` identical `before` and `after`, with `world_size=2` | `sharded_to_all_linear_in_place` is a **no-op on this weight**. Sharding is failing to happen, not happening wrongly — the opposite fix from the one the report proposes. Suspect `shard_inplace` not handling a 3D or quantized `wo_a`. |
| trailing dim halves between `heads-sharded` and `after`, and `expected_group_feat` halves too | Both sides tracked the shard. Widths may well agree — attention would then be exonerated and the search moves to the MoE path. |
| trailing dim halves but `expected_group_feat` does not, or vice versa | Confirms the report's mismatch, and shows which side is wrong. |

Note `expected_group_feat` is an **unpacked** count while a quantized weight's
trailing dim is **packed**. Establish the packing factor for this build before
comparing them — mxfp4 stored as uint8 gives 2 values per element, as uint32
gives 8. Get that from the `before` line on an unsharded load, where
`expected_group_feat` is known correct.

## 4. Then check the MoE path the same way

`switch_mlp.down_proj` and `shared_experts.down_proj` go through the identical
`sharded_to_all_linear_in_place`. If `wo_a` turns out to be sharded wrongly or
not at all, these are candidates for the same defect, and a broken expert
down-projection also produces a single repeated token.

## 5. Remove before committing the fix

This is diagnostic scaffolding. Either delete it, or if the shape contract is
worth enforcing permanently, convert it into an assertion at shard time —
which is the real lesson, since the two widths currently originate from
different places with nothing checking they agree.


---

# Replacement pinning test

Save as `src/exo/worker/tests/unittests/test_mlx/test_v4_attention_sharding.py`.

```python
"""Head-sharding arithmetic for V4Attention, without the 154 GiB checkpoint.

Replaces `tests/test_v4_real_model_attn_mismatch.py`, which loaded the real
model in a single process. That could not pin the defect it claimed to: the
strategy takes `self.N = group.size()` (auto_parallel.py:619), so with no
two-rank distributed group the shard is a no-op — which is exactly why that test
printed identical `wo_a.weight` shapes before and after.

`_shard_v4_attention_heads` takes `world_size` and `rank` as plain ints, so its
arithmetic IS testable in one process. That is what this file covers, in
milliseconds, with a synthetic module small enough to reason about.

What is deliberately NOT covered: `sharded_to_all_linear_in_place`, which needs a
real `mx.distributed.Group` of size 2 and therefore two processes. The contract
between the two — that the post-shard `group_feat` matches `wo_a`'s per-group
width — is asserted here as arithmetic, and instrumented on the live cluster
separately.

Break each test catches:

* heads_reduce_per_rank - a world_size that is ignored, leaving every rank with
  the full head count while wo_a is narrowed.
* every_rank_owns_heads_from_every_group - the naive contiguous split the
  docstring warns about, which puts whole original groups on one rank so the
  per-rank "group g" holds heads that do not belong to group g. This is the
  subtle property the interleaved slice exists to preserve, and nothing tested
  it.
* ranks_partition_the_heads - overlapping or gapped slices, so a head is used
  twice or dropped.
* group_feat_halves_with_world_size - the arithmetic that must line up with
  wo_a's per-group input width. Documents the contract the live instrumentation
  checks against real weights.
* attn_sink_is_sliced_interleaved_by_group and
  projection_keeps_the_same_heads_as_the_sink - a sink left at full width, or
  sliced contiguously while the heads are sliced interleaved, which applies each
  sink to the wrong head. The expected head sets [0,1,4,5] and [2,3,6,7] were
  derived independently of this code rather than read off a passing run.
* heads_stay_in_group_major_order - a slice that survives the partition checks
  but leaves the heads unsorted, so the downstream reshape mixes groups.
* world_size_one_is_a_no_op - documents, rather than assumes, why the previous
  single-process test could not observe the transform.
* quantized_wo_a_survives_head_sharding - head sharding and the later input-dim
  split interfering with each other.
* indivisible_config_is_rejected - a silent wrong answer when heads_per_group is
  not divisible by world_size.

NOT TESTED ON HARDWARE: written without MLX available. Expect to debug the
harness on first run as much as the code.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from exo.worker.engines.mlx.auto_parallel import _shard_v4_attention_heads
from mlx_lm.models.deepseek_v4 import ModelArgs, V4Attention

# Small but structurally identical to the real checkpoint: n_heads divisible by
# o_groups, heads_per_group divisible by 2. Real values are n_heads=64,
# head_dim=512, o_groups=8, o_lora_rank=1024.
SYNTHETIC = {
    "hidden_size": 64,
    "num_attention_heads": 8,
    "head_dim": 16,
    "qk_rope_head_dim": 8,
    "o_groups": 2,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "num_hidden_layers": 1,
}


def _attention() -> V4Attention:
    args = ModelArgs(**SYNTHETIC)  # pyright: ignore[reportArgumentType]
    return V4Attention(args, layer_id=0)


def _head_rows(attn: V4Attention) -> mx.array:
    """First column of wq_b, one entry per (head, head_dim) output row."""
    weight = attn.wq_b.weight  # pyright: ignore[reportAny]
    return weight[:, 0]  # pyright: ignore[reportAny]


def _tag_heads(attn: V4Attention) -> V4Attention:
    """Overwrite wq_b so every output row records which head it came from.

    Row value == head index. After sharding, the surviving values name exactly
    which original heads this rank kept, which is what makes the group-identity
    property assertable.
    """
    n_heads, head_dim = attn.n_heads, attn.head_dim
    in_features = int(attn.wq_b.weight.shape[-1])  # pyright: ignore[reportAny]
    tags = mx.repeat(mx.arange(n_heads), head_dim).reshape(n_heads * head_dim, 1)
    # Materialised, not a broadcast view: `_slice_head_major_flat` reshapes and
    # slices, and a view would make the assertions depend on layout luck.
    attn.wq_b.weight = mx.contiguous(  # pyright: ignore[reportAttributeAccessIssue]
        tags.astype(mx.float32) + mx.zeros((1, in_features))
    )
    mx.eval(attn.wq_b.weight)  # pyright: ignore[reportAny]
    return attn


def _kept_heads(attn: V4Attention) -> list[int]:
    rows = _head_rows(attn)
    mx.eval(rows)
    seen: list[int] = []
    for value in rows.tolist():  # pyright: ignore[reportAny]
        head = int(value)  # pyright: ignore[reportAny]
        if head not in seen:
            seen.append(head)
    return seen


def test_heads_reduce_per_rank() -> None:
    attn = _attention()
    original = attn.n_heads

    _shard_v4_attention_heads(attn, world_size=2, rank=0)

    assert attn.n_heads == original // 2


def test_every_rank_owns_heads_from_every_group() -> None:
    """The property the interleaved slice exists to preserve.

    With n_heads=8 and o_groups=2, heads 0-3 are group 0 and heads 4-7 are
    group 1. A contiguous split would give rank 0 heads 0-3 — all of group 0 and
    none of group 1 — so rank 0's "group 1" would contain group 0's heads and the
    wo_a grouped weight mapping would be wrong.
    """
    heads_per_group = SYNTHETIC["num_attention_heads"] // SYNTHETIC["o_groups"]

    for rank in (0, 1):
        attn = _tag_heads(_attention())
        _shard_v4_attention_heads(attn, world_size=2, rank=rank)
        kept = _kept_heads(attn)

        by_group: dict[int, list[int]] = {}
        for head in kept:
            by_group.setdefault(head // heads_per_group, []).append(head)

        assert set(by_group) == set(range(SYNTHETIC["o_groups"])), (
            f"rank {rank} kept heads {kept}, missing groups "
            f"{set(range(SYNTHETIC['o_groups'])) - set(by_group)}"
        )
        for group, heads in by_group.items():
            assert len(heads) == heads_per_group // 2, (
                f"rank {rank} group {group} kept {heads}, expected "
                f"{heads_per_group // 2} heads"
            )


def test_ranks_partition_the_heads() -> None:
    kept: list[list[int]] = []
    for rank in (0, 1):
        attn = _tag_heads(_attention())
        _shard_v4_attention_heads(attn, world_size=2, rank=rank)
        kept.append(_kept_heads(attn))

    assert not set(kept[0]) & set(kept[1]), f"overlap: {kept}"
    assert sorted(kept[0] + kept[1]) == list(range(SYNTHETIC["num_attention_heads"])), (
        f"heads dropped or duplicated: {kept}"
    )


def test_heads_stay_in_group_major_order() -> None:
    """SDPA output feeds `reshape(B, S, n_groups, group_feat)`, so the surviving
    heads must remain sorted group-major or the reshape mixes groups."""
    attn = _tag_heads(_attention())
    _shard_v4_attention_heads(attn, world_size=2, rank=0)

    kept = _kept_heads(attn)
    assert kept == sorted(kept), f"not group-major: {kept}"


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        # n_heads=8, o_groups=2 -> heads_per_group=4, hpg_per_rank=2.
        # Group 0 is heads 0-3, group 1 is heads 4-7. Each rank takes the same
        # slice position from BOTH groups, which is the interleaving.
        (0, [0, 1, 4, 5]),
        (1, [2, 3, 6, 7]),
    ],
)
def test_attn_sink_is_sliced_interleaved_by_group(
    rank: int, expected: list[int]
) -> None:
    attn = _attention()
    attn.attn_sink = mx.arange(attn.n_heads).astype(mx.float32)  # pyright: ignore[reportAttributeAccessIssue]
    mx.eval(attn.attn_sink)

    _shard_v4_attention_heads(attn, world_size=2, rank=rank)

    sink = attn.attn_sink
    mx.eval(sink)
    assert int(sink.shape[0]) == attn.n_heads
    assert [int(v) for v in sink.tolist()] == expected  # pyright: ignore[reportAny]


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0, [0, 1, 4, 5]), (1, [2, 3, 6, 7])],
)
def test_projection_keeps_the_same_heads_as_the_sink(
    rank: int, expected: list[int]
) -> None:
    """wq_b and attn_sink must agree, or the sink is applied to the wrong head."""
    attn = _tag_heads(_attention())

    _shard_v4_attention_heads(attn, world_size=2, rank=rank)

    assert _kept_heads(attn) == expected


def test_group_feat_halves_with_world_size() -> None:
    """The contract `_grouped_output_projection` depends on.

    It computes `group_feat = (n_heads * head_dim) // n_groups` from the SHARDED
    head count, while taking wo_a's per-group width from the weight itself via
    `reshape(n_groups, o_lora_rank, -1)`. Nothing asserts the two agree; this
    pins the half the arithmetic controls, and the live instrumentation covers
    the other half against real weights.
    """
    unsharded = _attention()
    before = (unsharded.n_heads * unsharded.head_dim) // unsharded.n_groups

    sharded = _attention()
    _shard_v4_attention_heads(sharded, world_size=2, rank=0)
    after = (sharded.n_heads * sharded.head_dim) // sharded.n_groups

    assert after * 2 == before, f"group_feat went {before} -> {after}"


def test_world_size_one_is_a_no_op() -> None:
    """Documents why the previous single-process test could not pin anything."""
    attn = _tag_heads(_attention())
    original_heads = attn.n_heads
    original_shape = tuple(attn.wq_b.weight.shape)  # pyright: ignore[reportAny]

    _shard_v4_attention_heads(attn, world_size=1, rank=0)

    assert attn.n_heads == original_heads
    assert tuple(attn.wq_b.weight.shape) == original_shape  # pyright: ignore[reportAny]
    assert _kept_heads(attn) == list(range(original_heads))


def test_indivisible_config_is_rejected() -> None:
    args = ModelArgs(**{**SYNTHETIC, "num_attention_heads": 6, "o_groups": 2})  # pyright: ignore[reportArgumentType]
    attn = V4Attention(args, layer_id=0)

    # heads_per_group == 3, not divisible by 2.
    with pytest.raises(AssertionError, match="divisible by world_size"):
        _shard_v4_attention_heads(attn, world_size=2, rank=0)


def test_quantized_wo_a_survives_head_sharding() -> None:
    """Head sharding must not disturb wo_a; only the later in-place input-dim
    split touches it. If this fails, the two transforms are interfering."""
    attn = _attention()
    nn.quantize(attn, group_size=32, bits=4)
    before = tuple(attn.wo_a.weight.shape)  # pyright: ignore[reportAny]

    _shard_v4_attention_heads(attn, world_size=2, rank=0)

    assert tuple(attn.wo_a.weight.shape) == before  # pyright: ignore[reportAny]
```
