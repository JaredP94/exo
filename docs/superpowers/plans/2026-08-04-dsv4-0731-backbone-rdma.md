# DeepSeek V4 Flash 0731 Backbone and Tensor-RDMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load `Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp` strictly through EXO, execute its backbone with correct mixed quantization, and serve deterministic two-rank tensor-parallel inference over EXO's existing JACCL/RDMA runtime on the M5 Max and M4 Max Macs.

**Architecture:** Keep EXO's control plane, model placement, `DeepseekV4ShardingStrategy`, and generated JACCL device matrix. Add a narrowly detected 0731 compatibility boundary before tensor sharding: a pure-Python checkpoint/config inspector, a subclass of the pinned `mlx_lm` DeepSeek V4 model that applies only the four verified compatibility deltas, and an EXO-owned loader that enforces strict weights plus a realized-quantization audit. Validate the independent implementation against OMLX one real layer at a time before the full two-node run.

**Tech Stack:** Python 3.13, MLX 0.32.0, EXO's pinned `mlx-lm` DeepSeek V4 fork, pytest, basedpyright, Ruff, safetensors, OMLX as a local numeric reference, Zenoh discovery, JACCL/RDMA over Thunderbolt 5.

## Global Constraints

- Execute this plan before `2026-08-04-dsv4-0731-prompt-api.md`.
- Do not modify OMLX or the target checkpoint. OMLX is a read-only reference.
- Do not add DSpark/MTP execution. Recognize and report the 114 `mtp.*` tensors, then rely on the pinned sanitizer's existing filtering.
- Do not reimplement the pinned DeepSeek V4 sanitizer. The subclass pre-pass contains only D1, D2, and no-op compatibility handling; the base sanitizer retains FP8/FP4 conversion, fusion, expert stacking, top-level remapping, and MTP filtering.
- Do not change the existing DSV4 tensor-sharding architecture unless the repaired non-degenerate TP test proves a narrowly scoped correctness defect.
- Do not hard-code `en*` or `rdma_en*` interfaces. Use EXO discovery and its generated JACCL matrix.
- For the recognized target, strict weight loading and the post-quantization `(bits, group_size, mode)` audit are mandatory. Ordinary models retain their current requested strictness.
- The checkpoint's 46 `compress_ratios` entries are valid for 43 backbone layers. Reject only a short list or values outside `{0, 4, 128}`.
- Preserve the model card's `base_model = "DeepSeek V4 Flash"` prefix and do not advertise MTP.
- Run tests on a real Metal-capable macOS process. The current sandbox cannot initialize a Metal device.

---

## Task 1: Add pure checkpoint detection, quantization normalization, and shard preflight

**Files:**

- Create: `src/exo/worker/engines/mlx/deepseek_v4_0731_config.py`
- Create: `src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py`

### Step 1: Write failing detection and metadata tests

- [x] Add tests for these exact outcomes:

  - non-`deepseek_v4` config returns `None`;
  - ordinary DeepSeek V4 with neither DSpark field nor 0731 key evidence delegates and returns `None`;
  - `model_type == "deepseek_v4"`, `dspark_block_size > 0`, and a non-empty `dspark_target_layer_ids` returns a supported checkpoint record;
  - `num_nextn_predict_layers` alone is not a 0731 discriminator;
  - exactly one DSpark discriminator, or 0731 weight evidence combined with malformed DSpark fields, raises `DeepseekV40731CompatibilityError` with the model path and offending fields;
  - the weight index records the total and `mtp.*` tensor counts without opening tensor shards.

Use minimal `tmp_path` fixtures containing `config.json` and `model.safetensors.index.json`. The supported fixture must include dotted `attn_hc`, a `model.layers.0.attn.wo_a.weight` entry, and an `mtp.0.markov_head.markov_w1.weight` entry.

- [x] Run the tests and confirm the import fails because the module does not exist:

```bash
uv run pytest -q src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py
```

### Step 2: Implement the public types and inspector

- [x] Add these concrete types and entry point:

```python
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DeepseekV40731CompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizationSpec:
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class DeepseekV40731Checkpoint:
    model_path: Path
    config: dict[str, Any]
    weight_keys: frozenset[str]
    mtp_weight_count: int
    mtp_quantization_count: int
    realized_quantization: dict[str, QuantizationSpec]
```

Export `inspect_deepseek_v4_0731_checkpoint(model_path: Path) -> DeepseekV40731Checkpoint | None` as the public entry point.

The implementation must use `json.loads(path.read_text())`, copy the config before normalization, and read only index keys. Treat `*.attn_hc.*`, `*.ffn_hc.*`, three-dimensional `wo_a` evidence when available, and `mtp.<stage>.markov_head.*` as 0731 evidence. Do not use the repository ID as a discriminator.

### Step 3: Write failing quantization mapping tests

- [x] Add table-driven tests for the exact path map:

```python
[
    (
        "model.layers.3.attn.wq_a",
        "model.layers.3.attn.wqkv_a",
    ),
    (
        "model.layers.3.attn.wkv",
        "model.layers.3.attn.wqkv_a",
    ),
    (
        "model.layers.3.attn.compressor.wgate",
        "model.layers.3.attn.compressor.wkv_gate",
    ),
    (
        "model.layers.2.attn.indexer.compressor.wkv",
        "model.layers.2.attn.indexer.compressor.wkv_gate",
    ),
]
```

- [x] Cover all of the following:

  - identical source-pair specs collapse into one realized path;
  - a mismatch in any of `bits`, `group_size`, or `mode` raises and names both sources;
  - `mtp.*` entries are counted then omitted from the realized backbone map;
  - the global `bits/group_size/mode` fields remain present in the normalized config but no explicit backbone declaration is removed without a realized replacement;
  - a synthetic 641-entry source map with 105 compatible pairs produces 536 realized paths;
  - `quantization_config` is updated to the same normalized mapping when present.

### Step 4: Implement config normalization

- [x] Implement these pure functions:

Implement `realized_quantization_path(path: str) -> str` and `normalize_deepseek_v4_0731_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, QuantizationSpec]]`.

`realized_quantization_path()` must use two anchored regular expressions so unrelated `wkv` or `wgate` names are not rewritten:

```python
path = re.sub(r"(\.attn)\.(wq_a|wkv)$", r"\1.wqkv_a", path)
return re.sub(
    r"(\.attn(?:\.indexer)?\.compressor)\.(wkv|wgate)$",
    r"\1.wkv_gate",
    path,
)
```

The normalizer must validate `num_hidden_layers`, `compress_ratios`, all quantization triples, pair compatibility, and the absence of dangling non-MTP declarations. It must copy nested structures rather than mutate the caller's config.

### Step 5: Write failing geometry tests

- [x] Add tests for:

  - target world sizes 1, 2, 4, and 8 pass when `num_attention_heads=64` and `o_groups=8`;
  - world sizes 3 and 16 fail with a message containing `heads_per_group=8`;
  - `num_attention_heads % o_groups != 0` fails;
  - a quantized `attn.wo_a`, routed `down_proj`, or shared-expert `down_proj` fails when its logical input width is not divisible by `group_size * world_size`;
  - the real target dimensions pass for world size 2.

### Step 6: Implement geometry validation

- [x] Add:

Add `validate_deepseek_v4_0731_shard_geometry(config: Mapping[str, Any], realized_quantization: Mapping[str, QuantizationSpec], world_size: int) -> None`.

Calculate the sharded-to-all input widths from config, not packed tensor shapes:

- `attn.wo_a`: `(num_attention_heads // o_groups) * head_dim`;
- `ffn.switch_mlp.down_proj`: `moe_intermediate_size`;
- `ffn.shared_experts.down_proj`: `moe_intermediate_size * n_shared_experts`.

Reject `world_size < 1`; require `heads_per_group % world_size == 0`; and require each affected explicit quantized module's logical input width to be divisible by `spec.group_size * world_size`.

### Step 7: Run pure tests and static checks

- [x] Run:

```bash
uv run pytest -q src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py
uv run basedpyright src/exo/worker/engines/mlx/deepseek_v4_0731_config.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py
uv run ruff check src/exo/worker/engines/mlx/deepseek_v4_0731_config.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py
```

Expected result: all tests and both static checks pass without Metal initialization.

### Step 8: Commit Task 1

- [x] Commit only the two Task 1 files:

```bash
git add src/exo/worker/engines/mlx/deepseek_v4_0731_config.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py
git commit -m "feat: inspect DeepSeek V4 0731 checkpoints"
```

---

## Task 2: Add the 0731 compatibility model delta

**Files:**

- Create: `src/exo/worker/engines/mlx/deepseek_v4_0731_model.py`
- Create: `src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py`

### Step 1: Write failing hyper-connection normalization tests

- [x] Build a tiny `DeepseekV40731Model` fixture using the pinned `ModelArgs` with `o_groups=8`, `hc_mult=4`, and two layers.

- [x] Test the pre-sanitize helper independently with lazy MLX arrays:

  - `model.layers.0.attn_hc.fn` becomes `model.layers.0.hc_attn.fn`;
  - `model.layers.0.ffn_hc.scale` becomes `model.layers.0.hc_ffn.scale`;
  - already-correct `hc_attn.*` and `hc_ffn.*` keys are unchanged;
  - underscore raw-HF keys such as `hc_attn_fn` are left for the base sanitizer;
  - simultaneous old and new forms for the same destination raise instead of overwriting.

### Step 2: Write failing `wo_a` reshape tests

- [x] Cover weight `[8, 1024, 1024]` to `[8192, 1024]`, scales `[8, 1024, 128]` to `[8192, 128]`, and an already-2D no-op.

- [x] Reject a three-dimensional leading shape that does not equal `(args.o_groups, args.o_lora_rank)` and include the parameter path, observed shape, and expected leading shape in the error.

### Step 3: Write the failing D4 behavior test

- [x] Assert every `model.layers[*].ffn.shared_experts.swiglu_limit` equals `args.swiglu_limit` after construction.

- [x] Use gate/up activations whose magnitude exceeds the clamp boundary and compare the shared-expert output with a base pinned model constructed at `swiglu_limit=0.0`; assert the outputs differ. This prevents a constructor-only test that never exercises the correction.

### Step 4: Run the focused test and confirm red

- [x] Run outside the sandbox on the local Mac:

```bash
uv run pytest -q src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py
```

Expected result: import or assertion failures before implementation.

### Step 5: Implement the subclass and delta helper

- [x] Add these interfaces:

```python
from typing import Any

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import Model as PinnedDeepseekV4Model
from mlx_lm.models.deepseek_v4 import ModelArgs


class DeepseekV40731Model(PinnedDeepseekV4Model):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        for layer in self.model.layers:
            layer.ffn.shared_experts.swiglu_limit = args.swiglu_limit

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        normalized = normalize_deepseek_v4_0731_weights(weights, self.args)
        return super().sanitize(normalized)
```

The class calls `normalize_deepseek_v4_0731_weights(weights: dict[str, mx.array], args: ModelArgs) -> dict[str, mx.array]`, defined in the same module.

The helper must return a new dictionary but retain the original lazy array objects except for D2 `reshape`. It must not concatenate fusion pairs, stack experts, decode FP8/FP4, remap top-level names, or filter MTP; those remain the base sanitizer's responsibility.

### Step 6: Prove the base sanitizer still owns existing transformations

- [x] Add regression tests that pass already-fused `wqkv_a`, already-stacked `switch_mlp`, and `mtp.*` keys through the compatibility pre-pass. Assert the pre-pass does not duplicate or drop them. Then call the subclass sanitizer on a reduced compatible fixture and assert the base sanitizer filters `mtp.*` exactly once.

### Step 7: Run tests and static checks

- [x] Run:

```bash
uv run pytest -q src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py
uv run basedpyright src/exo/worker/engines/mlx/deepseek_v4_0731_model.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py
uv run ruff check src/exo/worker/engines/mlx/deepseek_v4_0731_model.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py
```

Expected result: all pass on the Metal-capable local Mac.

### Step 8: Commit Task 2

- [x] Commit:

```bash
git add src/exo/worker/engines/mlx/deepseek_v4_0731_model.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py
git commit -m "feat: normalize DeepSeek V4 0731 weights"
```

---

## Task 3: Route both EXO load paths through strict compatibility loading and audit quantization

**Files:**

- Create: `src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py`
- Create: `src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py`
- Modify: `src/exo/worker/engines/mlx/utils_mlx.py:38-42,163-284`

### Step 1: Write failing loader-routing tests

- [x] Monkeypatch the pinned `mlx_lm.utils.load_model` dependency at the loader module boundary and verify:

  - ordinary models preserve `lazy` and `strict` exactly and use default model-class dispatch;
  - the recognized target passes the normalized config through `model_config`, supplies a `get_model_classes` callback returning `(DeepseekV40731Model, ModelArgs)`, and forces `strict=True` even when the caller requested `False`;
  - malformed 0731 evidence raises before `mlx_lm` is called;
  - the supported path reports `deepseek_v4_0731_backbone`, `mtp_weight_count=114`, `dspark_block_size=5`, target layers `[40, 41, 42]`, and an explicit `DSpark/MTP speculative decoding is disabled` warning.

### Step 2: Write failing quantization-audit tests

- [x] Construct a small model tree containing `nn.QuantizedLinear`, `nn.QuantizedEmbedding`, and the quantized switch-linear type returned by MLX for `SwitchGLU`.

- [x] Test:

  - exact `(bits, group_size, mode)` matches pass;
  - matching bits/group size but `affine` versus `mxfp8` fails;
  - missing realized modules and unexpected realized quantized modules fail with representative paths;
  - a dangling `mtp.*` declaration is ignored;
  - all 536 expected target paths can be accounted for without a global fallback.

The failure text must name the model-tree path, expected triple, and realized triple.

### Step 3: Implement the compatibility loader

- [x] Add:

Add `audit_deepseek_v4_0731_quantization(model: nn.Module, expected: dict[str, QuantizationSpec]) -> None` and `load_exo_model(model_path: Path, *, lazy: bool = False, strict: bool = True) -> tuple[nn.Module, dict[str, Any]]`.

Import the underlying loader as a private alias so tests can patch one seam:

```python
from mlx_lm.utils import load_model as _mlx_load_model
```

For a recognized checkpoint:

1. inspect and normalize;
2. call `_mlx_load_model(model_path, lazy=lazy, strict=True, model_config=normalized_config, get_model_classes=_get_0731_classes)`;
3. assert the returned model is `DeepseekV40731Model`;
4. audit quantization using `checkpoint.realized_quantization`;
5. log the backbone-only status and return the model plus normalized config.

Use `mlx.utils.tree_flatten(model.leaf_modules())` for the audit and recognize the three concrete quantized module families. Do not accept an absent `mode` attribute by assuming `affine`.

### Step 4: Route single-node and distributed loading through the wrapper

- [x] In `utils_mlx.py`, replace the direct import and both calls:

```python
from exo.worker.engines.mlx.deepseek_v4_0731_loader import load_exo_model
```

Single-node:

```python
model, _ = load_exo_model(model_path, lazy=True, strict=False)
```

Distributed:

```python
model, config = load_exo_model(model_path, lazy=True, strict=False)
```

Immediately after distributed load and before tokenization or `tensor_auto_parallel`, re-inspect only when the returned config matches 0731 and call:

```python
validate_deepseek_v4_0731_shard_geometry(
    config,
    normalized_quantization_specs(config),
    group.size(),
)
```

Expose one helper from the config module to reconstruct `QuantizationSpec` values from an already-normalized config so `utils_mlx.py` does not duplicate parsing.

### Step 5: Add integration tests around both call sites

- [x] Patch `load_exo_model` in `utils_mlx.py` and assert `load_mlx_items()` and `shard_and_load()` call it. For the distributed test, use a fake group with `size() == 3` and assert geometry fails before `tensor_auto_parallel` is invoked.

- [x] Add an ordinary-model regression proving the existing `strict=False` single-node behavior remains unchanged.

### Step 6: Run focused tests and static checks

- [x] Run:

```bash
uv run pytest -q \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py
uv run basedpyright \
  src/exo/worker/engines/mlx/deepseek_v4_0731_config.py \
  src/exo/worker/engines/mlx/deepseek_v4_0731_model.py \
  src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py \
  src/exo/worker/engines/mlx/utils_mlx.py
uv run ruff check \
  src/exo/worker/engines/mlx/deepseek_v4_0731_config.py \
  src/exo/worker/engines/mlx/deepseek_v4_0731_model.py \
  src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py \
  src/exo/worker/engines/mlx/utils_mlx.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py
```

### Step 7: Run the real checkpoint preflight without materializing weights

- [x] Add a short test invocation that calls only the pure inspector against:

```text
/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp
```

Assert 2230 total keys, 114 MTP keys, 33 MTP quantization declarations, 641 backbone source declarations, 536 realized paths, `compress_ratios` length 46, and world size 2 geometry success. This command must not import MLX or open a `.safetensors` shard.

### Step 8: Commit Task 3

- [x] Commit:

```bash
git add \
  src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py \
  src/exo/worker/engines/mlx/utils_mlx.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py
git commit -m "feat: load DeepSeek V4 0731 strictly"
```

---

## Task 4: Repair and de-degenerate tensor-parallel correctness coverage

**Files:**

- Modify: `src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py`
- Modify only if proven necessary: `src/exo/worker/engines/mlx/auto_parallel.py:827-951`

### Step 1: Replace the stale, permanently skipped test shape

- [x] Remove the blanket `@pytest.mark.skip` and the stale third argument to `tensor_auto_parallel`.

- [x] Consume the loading generator correctly:

```python
def _consume_sharding(
    generator: Generator[ModelLoadingResponse, None, nn.Module],
) -> nn.Module:
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value
```

- [x] Narrow this file's runnable matrix to two purposeful DSV4 fixtures instead of retaining an all-architecture test that has never run successfully:

  - `deepseek_v4_bf16`;
  - `deepseek_v4_q4` using `affine`, 4 bits, group size 32.

Both fixtures must set `o_groups=8`, `hc_mult=4`, `num_attention_heads=64`, `head_dim=16`, `q_lora_rank=32`, `o_lora_rank=32`, at least four layers, ratios including `0`, `4`, and `128`, routed and shared experts, and input widths divisible by `group_size * 2`.

### Step 2: Define explicit acceptance math

- [x] Replace `max_diff == 0.0` with:

```python
TOLERANCES = {
    "deepseek_v4_bf16": {"rtol": 2e-2, "atol": 2e-2},
    "deepseek_v4_q4": {"rtol": 5e-2, "atol": 5e-2},
}
```

Use `np.testing.assert_allclose`. In the same comparison, require `np.argmax(logits, axis=-1)` to be identical for reference and rank 0. Generate at least four greedy steps and require the token sequence to match exactly, not only the first logits tensor.

### Step 3: Run the repaired test red

- [x] On the local Mac, run:

```bash
uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
```

If it passes without product changes, continue. If it fails, retain the complete max/mean diff, layer path, rank, and fixture in the test output before changing production code.

### Step 4: Fix only demonstrated DSV4 sharding defects

- [x] If the test proves a defect, make the smallest correction inside `_shard_v4_attention_heads` or `DeepseekV4ShardingStrategy`. Do not redesign collectives or touch unrelated sharding strategies.

- [x] Add a regression assertion that fails against the pre-fix behavior and passes after it. If the test already passes, make no production change in this step.

### Step 5: Run the TP test repeatedly

- [x] Run it three consecutive times to rule out process/port nondeterminism:

```bash
uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
```

All runs must execute rather than skip and must pass both numeric and greedy-token gates.

### Step 6: Commit Task 4

- [x] Commit the test and only a proven narrow product fix:

```bash
git add src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
git add src/exo/worker/engines/mlx/auto_parallel.py
git commit -m "test: validate DeepSeek V4 tensor parallelism"
```

If `auto_parallel.py` was unchanged, omit it from `git add`.

---

## Task 5: Add hardware-feasible real-layer parity against OMLX

**Files:**

- Create: `scripts/validate_dsv4_0731_layer_parity.py`
- Create: `src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py`

### Step 1: Define the command and immutable reference inputs

- [x] Implement this CLI contract with `argparse`:

```bash
uv run python scripts/validate_dsv4_0731_layer_parity.py \
  --model-path /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --omlx-root /Users/jared/aishit/exo-install/omlx \
  --layers 0,2,3 \
  --seed 42 \
  --output /private/tmp/dsv4-0731-layer-parity.json
```

The script must refuse to run if either git root is dirty in files that the script imports, record both git SHAs, and never write beneath the model directory or OMLX root.

### Step 2: Write failing helper tests

- [x] Add pure tests for:

  - selecting only a requested layer's keys from `model.safetensors.index.json`;
  - remapping `model.layers.<source>.` to `model.layers.0.` without touching top-level parameters;
  - preserving U32/U8/BF16/F32/I32 dtypes while reading selected safetensors keys;
  - refusing any request that would materialize all 43 layers;
  - computing max absolute error, mean absolute error, and relative error with the Task 4 tolerances.

### Step 3: Implement selective one-layer loading

- [x] Read only the files named by the requested layer's index entries and retrieve only the selected keys with `safetensors.safe_open`. Construct a reduced one-layer config from the real config with the selected layer's `compress_ratio`, correct hash/non-hash gate behavior, and the original quantization entries remapped to layer 0.

- [x] Instantiate, quantize, sanitize, and load one OMLX `DeepseekV4Block` and one EXO `DeepseekV40731Model` layer at a time. Release both models and call `mx.clear_cache()` before advancing to the next source layer. Never instantiate or load the complete checkpoint on one 128 GB Mac.

### Step 4: Compare representative real paths

- [x] Use these source layers and gates:

  - layer 0: local attention plus hash routing;
  - layer 2: overlap compression plus indexer (`compress_ratio=4`);
  - layer 3: compressed attention without an indexer (`compress_ratio=128`) and non-hash routing.

For each, use identical BF16 hidden states and token IDs. Compare:

1. gate expert indices exactly;
2. gate scores within BF16 tolerance;
3. attention output;
4. routed-expert output;
5. shared-expert output with inputs scaled high enough to trigger `swiglu_limit=10.0`;
6. complete block output.

Report the exact subpath and tensor index of the largest divergence. Use Task 4's BF16 tolerance for dense intermediates and quantized tolerance for quantized projection outputs.

### Step 5: Make the parity script a hard gate

- [x] Exit zero only when all three layers pass, routing decisions match, the shared-expert clamp is exercised, and no complete-model allocation occurred. Write a JSON record containing config facts, source layer, path metrics, routing equality, peak active memory from `mx.get_active_memory()`, EXO SHA, OMLX SHA, and checkpoint index SHA-256.

### Step 6: Run helper tests, static checks, and real parity

- [x] Run:

```bash
uv run pytest -q src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py
uv run ruff check scripts/validate_dsv4_0731_layer_parity.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py
uv run python scripts/validate_dsv4_0731_layer_parity.py \
  --model-path /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --omlx-root /Users/jared/aishit/exo-install/omlx \
  --layers 0,2,3 \
  --seed 42 \
  --output /private/tmp/dsv4-0731-layer-parity.json
jq '.passed, .layers, .checkpoint_index_sha256' /private/tmp/dsv4-0731-layer-parity.json
```

Expected result: `.passed` is `true`, all three layers are present, and routing equality is true for every MoE comparison.

### Step 7: Commit Task 5

- [x] Commit:

```bash
git add scripts/validate_dsv4_0731_layer_parity.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py
git commit -m "test: compare DeepSeek V4 0731 layers with OMLX"
```

---

## Task 6: Validate the real checkpoint and two-node JACCL/RDMA backbone

**Files:**

- Create: `docs/superpowers/validation/2026-08-04-dsv4-0731-backbone-rdma.md`
- Modify only if a failure proves it necessary: `src/exo/worker/engines/mlx/auto_parallel.py`
- Modify only if a failure proves it necessary: `src/exo/worker/engines/mlx/utils_mlx.py`

### Step 1: Run the complete local quality gate

- [x] Run on the local Mac:

```bash
uv run pytest -q \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py \
  src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py
uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py
uv run basedpyright
uv run ruff check
nix fmt
uv run pytest -q
```

Record command, exit code, and test counts in the validation document. If `nix fmt` changes task files, review and stage those formatting changes. Do not call a skipped TP test a pass.

### Step 2: Confirm identical revisions and checkpoint facts on both Macs

- [x] Locally record:

```bash
git rev-parse HEAD
uv --version
sw_vers
shasum -a 256 /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp/model.safetensors.index.json
rdma_ctl status
```

- [x] Remotely record through the approved passwordless target:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 jared@Retirement-Plan.local \
  'cd /Users/jared/aishit/exo && git rev-parse HEAD && uv --version && sw_vers && shasum -a 256 /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp/model.safetensors.index.json && rdma_ctl status'
```

The EXO SHA, dependency lock, macOS version/build, and checkpoint hash must match. If they do not, stop and reconcile them before starting EXO.

### Step 3: Start both nodes in an isolated namespace

- [x] Use the same values on both nodes and let EXO discover interfaces:

```bash
EXO_ZENOH_NAMESPACE=dsv4f-0731-validation \
EXO_DEFAULT_MODELS_DIR=/Users/jared/.cache/huggingface/hub/models \
uv run exo
```

Do not set `EXO_LIBP2P_NAMESPACE`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR`, or an interface name manually.

### Step 4: Gate topology before instance creation

- [x] From the elected API node, capture `/state` and the target placement preview. Require exactly two workers in an RDMA-connected cycle and a preview with tensor sharding plus `MlxJaccl`:

```bash
curl -s http://localhost:52415/state | jq .
curl -s 'http://localhost:52415/instance/previews?model_id=Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp' | jq .
```

If the only preview is ring or pipeline, stop. Record EXO's topology diagnostics; do not accept an automatic or manual downgrade as RDMA validation.

### Step 5: Create the exact target instance

- [x] Select the preview whose shard metadata is `Tensor` and backend is `MlxJaccl`, then submit that exact instance object with the documented `/instance` request shape from `README.md`. Save the request JSON and returned instance ID in the validation document.

- [x] Tail both logs through readiness. Require:

  - compatibility mode `deepseek_v4_0731_backbone`;
  - 114 MTP tensors and 33 MTP quantization declarations skipped;
  - DSpark block size 5 and target layers 40, 41, 42 reported;
  - explicit MTP-disabled warning;
  - strict loading success;
  - 536-path quantization audit success;
  - generated JACCL device matrix and coordinator on both ranks;
  - group size 2 with ranks 0 and 1;
  - completed post-sharding barrier and ready instance.

Any missing backbone key, quantization mismatch, rank disagreement, JACCL initialization failure, or barrier failure is fatal.

### Step 6: Run deterministic backbone smoke tests

- [x] With `temperature=0`, a fixed short prompt, and bounded output, call `/v1/chat/completions` twice and require identical token IDs and text. At this stage use a simple no-tool request; the full 0731 prompt/API matrix belongs to the second plan.

- [x] Increase input length through 1K, 4K, 8K, 16K, and 32K tokens with `max_tokens` bounded to 16. At each size record:

  - success/failure and exact token count;
  - peak process RSS and wired memory on both nodes;
  - time to first token and decode throughput;
  - rank health and instance state.

Stop increasing context if either node approaches unsafe memory pressure. A controlled stop is not a 32K acceptance pass; record the highest stable length.

### Step 7: Record and resolve only evidence-backed failures

- [x] If real TP exposes a sharding or wired-limit defect not covered by tests, first add a focused failing regression. Make the smallest change to the loader or DSV4 sharding strategy, rerun Tasks 3-6, and add a separate commit describing the proven defect.

### Step 8: Finalize the validation record

- [x] The validation document must contain:

  - local and remote hardware/OS/revision/checkpoint facts;
  - layer-parity JSON path and summary;
  - topology and selected preview;
  - compatibility, MTP, and quantization audit logs;
  - deterministic short-run result;
  - context-length table through the highest attempted size;
  - memory and basic performance observations;
  - any deviations and their associated regression test/commit.

### Step 9: Commit Task 6

- [x] Commit the completed evidence record and any separately tested corrective changes:

```bash
git add docs/superpowers/validation/2026-08-04-dsv4-0731-backbone-rdma.md
git commit -m "docs: validate DeepSeek V4 0731 over RDMA"
```

---

## Completion Gate

- [x] Confirm Tasks 1-6 are complete and every checkbox's evidence exists.
- [x] Confirm the target checkpoint was never modified.
- [x] Confirm no MTP execution was added or advertised.
- [x] Confirm the repaired DSV4 TP test executes without a skip marker.
- [x] Confirm representative real OMLX layer parity passed before cluster acceptance.
- [x] Confirm the live instance is exactly two-rank `Tensor`/`MlxJaccl`, not ring or pipeline.
- [x] Confirm ordinary model loading behavior is covered by a passing regression.
- [x] Run `git status --short` and verify only intentional task files are staged or committed; preserve the user's pre-existing `CONTRIBUTING.md`, `README.md`, `justfile`, model card, and Claude review changes.

After this gate passes, execute `2026-08-04-dsv4-0731-prompt-api.md` to complete latest-reminder caching, reasoning-effort tiers, assistant prefill, DSML tolerance, parser boundaries, and the Chat Completions/Responses acceptance matrix.
