# OMLX cache and indexer profiling

**Date:** 2026-08-11
**Scope:** single node, synthetic tensors, no cluster, no instance, no model or checkpoint load

This is an isolated timing study of the two operations behind S4 and S3. It is
not a full-model performance measurement.

## Shapes and derivation

The checkpoint metadata was read from:

`/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp/config.json`

The relevant values were:

```json
{
  "num_hidden_layers": 43,
  "compress_ratios": [0, 0, 4, 128, 4, 128, "...", 4, 128, 4],
  "hidden_size": 4096,
  "head_dim": 512,
  "index_topk": 512,
  "index_n_heads": 64,
  "index_head_dim": 128,
  "num_attention_heads": 64
}
```

Counting only the first `num_hidden_layers` entries gives:

```text
layers: 43
counts: {0: 2, 4: 21, 128: 20}
```

The installed package was:

`.../.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py`

Its `Compressor.__init__` uses `coff = 2 if self.overlap else 1` at installed
lines 1484-1489. Ratio-4 layers therefore use `coff = 2`, while ratio-128
layers use `coff = 1`. The compressor input/intermediate row width is
`coff * head_dim` (`1024` for ratio 4); the pool row itself is `[B, rows,
head_dim]`, as shown by `_CompressorBranch` and `update_pool` at installed
lines 652-654 and 1148-1188. The pool width used here is therefore 512.

The single-node EXO path sets `prefill_step_size = 4096` at
`src/exo/worker/engines/mlx/generator/generate.py:336` and passes it to
`mlx_lm.generate.stream_generate` at `:363`. The installed `stream_generate`
loop processes `total - 1` tokens in chunks at lines 430-445. For 32,768
tokens, that is eight prefill chunks: seven of 4,096 tokens and one of 4,095.
The final one-token `_step` at installed line 453 completes the three-token
ratio-4 carry, so the exact pool-growth schedule used here was:

```python
SCHEDULE = [1024] * 7 + [1023, 1]
```

That is nine `update_pool` appends and exactly 8,192 ratio-4 pool rows.

## Evaluation and timing method

All timings used `time.perf_counter()` around the complete synthetic operation.
The timed loops explicitly forced MLX evaluation with the following code,
inside the timed block:

```python
pool = new_pooled if pool is None else mx.concatenate([pool, new_pooled], axis=1)
mx.eval(pool)
```

For the geometric-growth harness, the corresponding synchronization was:

```python
storage[:, length:needed] = new_pooled
length = needed
mx.eval(storage)
```

For indexer scoring and the identity shortcut, it was:

```python
out = mx.argpartition(-score, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
mx.eval(out)
```

and:

```python
out = mx.broadcast_to(mx.arange(T, dtype=mx.uint32), (B, S, T))
mx.eval(out)
```

The harness used `B=1`, `bfloat16`, seven repetitions per case, and a warm-up
pass before measurement. Benchmark A ran two warm-up runs per approach;
Benchmark B ran one warm-up run per pooled length and approach.
`mx.clear_cache()` ran between measured repetitions, outside the timed region.

## Benchmark A — pooled growth

The baseline reproduced repeated `mx.concatenate([pool, new_pooled], axis=1)`.
The comparison preallocated a geometric backing array, copied only when
capacity was exhausted, wrote each new block into spare capacity, and forced
evaluation after every append.

Raw samples are milliseconds:

| Approach | Samples | Median | Min–max | Spread |
|---|---:|---:|---:|---:|
| Repeated concatenate | 2.875, 2.880, 2.832, 2.885, 2.842, 2.851, 2.846 | 2.851 ms | 2.832–2.885 ms | 0.053 ms |
| Geometric growth | 2.065, 2.597, 2.582, 2.724, 2.628, 2.816, 2.605 | 2.605 ms | 2.065–2.816 ms | 0.751 ms |

The isolated median delta is `0.246 ms` per ratio-4 layer. Scaling that delta
by 21 ratio-4 layers gives `5.166 ms`, or:

```text
5.166 ms / 196,466 ms = 0.00263%
```

This is an **estimate**, not a measurement of the full model. It assumes the 21
layers are independent and execute sequentially, with no overlap or shared
allocator effects. Under that assumption, pooled growth is not a meaningful
share of the 196.466-second 32K prefill: **no**.

## Benchmark B — indexer scoring

The benchmark reproduced the installed `Indexer.__call__` scoring section at
lines 1783-1801: the einsum, ReLU, weighted head reduction, and
`argpartition`. The synthetic shapes were `q=[1,546,64,128]`,
`idx_kv=[1,pooled_len,128]`, and `per_head_weights=[1,546,64]`.

Raw samples are milliseconds:

| Pooled rows | Scoring samples | Scoring median | Shortcut samples | Shortcut median |
|---:|---:|---:|---:|---:|
| 136 | 1.013, 0.674, 0.811, 0.826, 0.859, 0.858, 0.867 | 0.858 ms | 0.257, 0.247, 0.230, 0.217, 0.231, 0.215, 0.216 | 0.230 ms |
| 512 | 2.199, 1.399, 1.380, 1.506, 1.527, 1.421, 1.407 | 1.421 ms | 0.251, 0.252, 0.250, 0.207, 0.218, 0.192, 0.160 | 0.218 ms |
| 1024 | 3.787, 1.879, 2.613, 2.301, 1.361, 1.366, 1.218 | 1.879 ms | 0.215, 0.176, 0.145, 0.144, 0.153, 0.152, 0.196 | 0.153 ms |

At 136 rows, the scoring delta is `0.628 ms` per ratio-4 layer. Scaling by 21
layers gives `13.188 ms`, or:

```text
13.188 ms / 5,305 ms = 0.249%
```

At 512 rows, the scoring delta is `1.203 ms` per ratio-4 layer. The equivalent
scaled estimate is `25.263 ms`, or `0.476%` of the 5.305-second TTFT.

At 1024 rows, the scoring delta is `1.726 ms` per ratio-4 layer. The equivalent
scaled estimate is `36.246 ms`, or `0.683%` of TTFT. The identity construction
at 1024 is shown only as a timing reference; it is not a valid S3 shortcut once
the pool is larger than `index_topk`.

All scaled percentages are **estimates** under the same independent,
sequential-layer assumption. At the target 546-token prompt (`pooled_len=136`),
indexer scoring is not a meaningful share of TTFT: **no**.

## Surprises and limitations

- The geometric harness was only `0.246 ms` faster per ratio-4 layer, much less
  than the full-prefill time would suggest from the source-level quadratic
  pattern.
- The 1024-row scoring samples had a 2.569 ms spread despite explicit
  synchronization; the median is reported, but that variability makes the
  1024 estimate especially weak.
- The packet's description of 512 as the point where the shortcut stops firing
  is off by one condition: the source condition is `pooled_len <= index_topk`,
  so 512 is still eligible and scoring resumes above 512.
- The packet's `generate.py:195` citation is also two lines early in this
  checkout: the single-node step-size division is at `:197`.
- These are synthetic operation timings, not end-to-end model timings. They do
  not include compressor work, projections, attention, allocator interactions
  across layers, or any distributed scheduling.
- No S3 or S4 implementation was used, and no harness file was placed under
  `src`; the benchmark ran inline with `uv run --no-sync`.
