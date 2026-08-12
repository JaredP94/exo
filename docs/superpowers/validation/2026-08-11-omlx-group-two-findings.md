# OMLX Group II findings

Checkpoint inspected directly:

`/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp`

## `c48e1a82` — block-config crossover

**Verdict: DOES NOT APPLY.**

Evidence from the pinned fork's `mlx_lm.models.switch_layers` source:

```text
grep -nE "MIN_ROUTES|_block_config|_mxfp4_block_config|native_kind|LARGE_BLOCK|block_bm|block_variant|block_meta" .../mlx_lm/models/switch_layers.py
(no output; grep exit 1)
```

The fork has no block-config mechanism, route-count crossover, or related
symbols. Its Switch layer calls `mx.gather_qmm` directly. The upstream
mxfp4/affine crossover defect therefore has no corresponding fork behavior.

This would change to **APPLIES** if the pinned fork gained a block-config
mechanism or a shared route-count threshold affecting affine quantization.

## `b6811ed6` — sub-4-bit rule

**Verdict: DOES NOT APPLY.**

### Specific upstream rule

The fork has no native ratio-128 sparse-attention dispatch:

```text
grep -nE "has_symbol|use_native|deepseek_v4_sparse_attention|ratio.?128" .../mlx_lm/models/deepseek_v4.py | head
(no output)
```

The command returned no matches. The fork instead uses the existing
`compress_ratio == 4` behavior; it has no `has_symbol` dispatch or
`use_native_ratio128_attention` flag. The specific upstream guard therefore
has nothing to disable.

This part would change to **APPLIES** if the fork added a native ratio-128
sparse-attention path whose enablement depended on declared sub-4-bit
quantization.

### Underlying fused-attention concern

The direct `config.json` histogram was:

```text
top-level scalars: {"bits": 4, "group_size": 64, "mode": "affine"}
(4, 32, 'mxfp4') 138
(8, 32, 'mxfp8') 389
(8, 64, 'affine') 147
```

EXO's realized, backbone-only view was:

```text
(4, 32, 'mxfp4') 129
(8, 32, 'mxfp8') 322
(8, 64, 'affine') 85
```

The histograms do not numerically agree, which is recorded as a finding.
The direct view counts raw explicit declarations, including 24 mxfp8 and 9
mxfp4 `mtp.` declarations. EXO drops `mtp.` entries and maps paired source
declarations onto realized fused destinations, so its histogram counts
realized modules rather than raw source entries. The direct backbone-only
raw histogram is `(4, 32, 'mxfp4') 129`, `(8, 32, 'mxfp8') 365`, and
`(8, 64, 'affine') 147`; the remaining count reduction is the expected
source-to-fused normalization, but the raw and realized totals must not be
presented as identical evidence.

No entry in either view declares `bits < 4`.

The pinned fork's fused guards are at
`mlx_lm/models/deepseek_v4.py:1588`, `:2082`, `:2097`, and `:2218-2219`.
They all require `nn.QuantizedLinear.mode == "mxfp4"`. The loader fuses
`wq_a + wkv` into `wqkv_a` and compressor `wkv + wgate` into `wkv_gate` at
`:2719-2734`.

The checkpoint declarations relevant to those guards are:

```text
wq_a       (8, 32, 'mxfp8') 46
wkv        (8, 32, 'mxfp8') 46
compressor.wgate (8, 64, 'affine') 62
compressor.wkv   (8, 64, 'affine') 62
wq_b       (8, 32, 'mxfp8') 46
wo_a       (8, 32, 'mxfp8') 46
wo_b       (8, 32, 'mxfp8') 46
```

Therefore the static guard results are:

| Guard | Declared source mode(s) | Engages? |
| --- | --- | --- |
| `wkv_gate` | compressor `wkv` + `wgate`: affine | No |
| `wqkv_a` | `wq_a` + `wkv`: mxfp8 | No |
| `wq_b` | mxfp8 | No |
| `wo_a` + `wo_b` chain | both mxfp8 | No |

All four paths are statically decidable. None engages for this checkpoint,
and no sub-4-bit module is present on an engaging fused path. The underlying
concern therefore **DOES NOT APPLY** to this checkpoint.

This would change to **APPLIES** if any relevant declaration became sub-4-bit
and its corresponding realized module mode engaged an mxfp4 fused guard.

Because both Group II verdicts are **DOES NOT APPLY**, no test was added.

## Raised, not fixed

This document records the investigation only. No code, checkpoint, config,
dependency, or test was changed.
