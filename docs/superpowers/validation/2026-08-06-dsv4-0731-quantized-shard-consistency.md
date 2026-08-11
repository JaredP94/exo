# Instrumentation round 2: is `scales` sharded consistently with `weight`?

The first round logged only `weight.shape`. That was my omission, and it left the
last structural candidate unchecked.

A quantized linear carries `weight`, `scales`, and sometimes `biases`. All three
index the same input axis. If `shard_inplace` slices `weight` but not `scales`,
then in `_grouped_output_projection`:

```python
weight = woa_w.reshape(n_groups, o_lora_rank, -1)
scales = woa_s.reshape(n_groups, o_lora_rank, -1)
```

both reshapes succeed with *different* trailing dimensions, and
`mx.quantized_matmul` dequantizes every group with the wrong scales. No
exception, every layer, every token, fully deterministic — which matches every
symptom on the cluster, including one token repeating forever.

## The invariant, which needs no assumptions

Do not compare against `group_feat`, packing factors, or `group_size`. Those all
require knowing how this build packs mxfp4/mxfp8, and getting that wrong is how
round 1 nearly went astray.

Instead: **sharding must not change the ratio between the trailing dimensions of
`weight` and `scales`.** Both describe the same input axis, so whatever the
packing is, halving one must halve the other.

```
ratio = weight.shape[-1] / scales.shape[-1]
```

Log it before and after. If it changes, the shard is inconsistent, and the
direction of the change says which array was missed. This holds for 2D attention
weights and 3D batched expert weights alike, because it only reads the last axis.

## Patch

Add to `src/exo/worker/engines/mlx/auto_parallel.py`, replacing the round-1
`_log_v4_attention_shard` helper.

```python
def _log_quantized_shard(label: str, module: nn.Module, stage: str) -> None:
    """Log the weight/scales/biases trailing dims of a quantized projection.

    The load-bearing number is `ratio`. `weight` and `scales` index the same
    input axis, so sharding must scale both by the same factor whatever the
    packing is. A ratio that changes across the shard means one array was
    sliced and the other was not, which silently feeds wrong scales to
    `mx.quantized_matmul` rather than raising.
    """
    weight = module.get("weight") if isinstance(module, nn.Module) else None
    if not isinstance(weight, mx.array):
        logger.info(f"shard-consistency [{stage}] {label}: not quantized, skipped")
        return
    scales = module.get("scales")
    biases = module.get("biases")

    w_trailing = int(weight.shape[-1])
    s_trailing = int(scales.shape[-1]) if isinstance(scales, mx.array) else None
    b_trailing = int(biases.shape[-1]) if isinstance(biases, mx.array) else None
    ratio = None if not s_trailing else w_trailing / s_trailing

    logger.info(
        f"shard-consistency [{stage}] {label}: "
        f"weight={tuple(weight.shape)} scales="
        f"{None if not isinstance(scales, mx.array) else tuple(scales.shape)} "
        f"biases={None if not isinstance(biases, mx.array) else tuple(biases.shape)} "
        f"w_trailing={w_trailing} s_trailing={s_trailing} b_trailing={b_trailing} "
        f"ratio={ratio} "
        f"group_size={getattr(module, 'group_size', None)} "
        f"bits={getattr(module, 'bits', None)} "
        f"mode={getattr(module, 'mode', None)}"
    )
```

Then in `DeepseekV4ShardingStrategy.shard_model`, for layer 0 only:

```python
            probe = i == 0

            if probe:
                _log_quantized_shard("attn.wo_a", layer.attn.wo_a, "before")
            _shard_v4_attention_heads(layer.attn, self.N, self.group.rank())
            self.sharded_to_all_linear_in_place(layer.attn.wo_a)
            if probe:
                _log_quantized_shard("attn.wo_a", layer.attn.wo_a, "after")
            layer.attn.wo_b = _AllSumLinear(layer.attn.wo_b, self.group)  # type: ignore

            ffn = layer.ffn
            if getattr(ffn, "shared_experts", None) is not None:
                if probe:
                    _log_quantized_shard(
                        "shared.down_proj", ffn.shared_experts.down_proj, "before"
                    )
                self.all_to_sharded_linear_in_place(ffn.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(ffn.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(ffn.shared_experts.up_proj)
                if probe:
                    _log_quantized_shard(
                        "shared.down_proj", ffn.shared_experts.down_proj, "after"
                    )
            if probe:
                _log_quantized_shard(
                    "switch.down_proj", ffn.switch_mlp.down_proj, "before"
                )
                _log_quantized_shard(
                    "switch.gate_proj", ffn.switch_mlp.gate_proj, "before"
                )
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.gate_proj)
            self.sharded_to_all_linear_in_place(ffn.switch_mlp.down_proj)
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.up_proj)
            if probe:
                _log_quantized_shard(
                    "switch.down_proj", ffn.switch_mlp.down_proj, "after"
                )
                _log_quantized_shard(
                    "switch.gate_proj", ffn.switch_mlp.gate_proj, "after"
                )
```

`gate_proj` is included as a **control**. It is sharded on the *output* axis
(`all_to_sharded`), so its trailing input dim should not change at all. If
`gate_proj`'s ratio also shifts, the problem is broader than the input-dim path.

## Reading the result

| Observation | Meaning |
|---|---|
| `ratio` identical before and after, on every weight | `scales` tracks `weight`. This candidate is eliminated and the structural search over sharding is exhausted — move to `sanitize()` in the pinned `mlx_lm` fork. |
| `attn.wo_a` ratio doubles (w_trailing halves, s_trailing does not) | **The bug.** `shard_inplace` sliced `weight` and left `scales` full width. Fix by slicing all quantized components together, as `_shard_quantized_rows` already does for `wq_b`. |
| `switch.down_proj` or `shared.down_proj` ratio changes but `wo_a` does not | Same defect, in the MoE path. A broken expert down-projection also yields one repeated token. |
| `scales=None` on a weight reported as quantized | The module stores scales elsewhere; report the shape and stop rather than guessing. |
| `gate_proj` trailing input dim changes | Unexpected. `all_to_sharded` should not touch the input axis. Report before proceeding. |

The ratio is `group_size / packing_factor`, not the packing factor itself —
`weight`'s trailing dim is `in / packing_factor` while `scales`' is
`in / group_size`. With `group_size=64` and the packing factor of 4 that round 1
established, expect **16.0**. But do not treat that number as the check: only the
*change* across the shard matters, and the whole point of the ratio is that it
needs no knowledge of either constant.

Verified against synthetic shapes before writing this: a consistent shard holds
the ratio (4.0 → 4.0), `weight` sliced with `scales` left full halves it
(4.0 → 2.0), and the reverse doubles it (4.0 → 8.0). So the direction of the
change identifies which array was missed.

## If this eliminates sharding entirely

The remaining candidate is weight transformation at load time —
`sanitize()` in the pinned fork `rltakashige/mlx-lm` branch `leo/deepseek-v4`,
which does most of the checkpoint-to-module mapping. That cannot be bisected on
one node because 154 GiB needs two-rank TP, so the approach is a per-tensor
comparison against OMLX after load: for one layer, compare `wo_a.weight`,
`wo_a.scales`, `wq_b.weight` and the hyper-connection tensors element-wise
between EXO's loaded module tree and OMLX's. The existing
`scripts/compare_exo_vs_omlx_logits.py` and `scripts/dump_exo_logits.py` from the
previous session are the right starting point; consider committing them, since
this comparison will be wanted again.

## Note on the round-1 assertion

`_assert_v4_attention_shard_contract` uses `getattr(wo_a, "bits", 4)`, giving
`ratio = 32 // 4 = 8`. The observed ratio is 4, so `bits` must be 8 and that
default is wrong — if the attribute were ever absent the assertion would fail
falsely on correct weights. Read `wo_a.bits` directly and let it raise. The
assertion is worth keeping otherwise; extend it to compare `scales` as well,
since that is the invariant this round exists to check.
