# DeepSeek V4 Flash 0731 on EXO — status and handoff

Date: 2026-08-07. Branch `codex/dsv4-0731-backbone`, worktree
`exo/.worktrees/codex-dsv4-0731-backbone`, HEAD `e3d94e77`. Nothing pushed.

---

## 1. Where the project stands

**Plan 1 (backbone + tensor-RDMA)** complete at `f0658404`. Two-node JACCL over
Thunderbolt 5 validated, 16K prompt ceiling proven, 32K reproducibly SIGSEGVs in
pinned MLX prefill — do not retry.

**Plan 2 (prompt + API)** all six tasks committed:

| Task | Commit | Subject |
|---|---|---|
| 1 | `e19eebae` | latest_reminder relocation |
| 2 | `b2822771` | reasoning-effort tiers |
| 3 | `4d4409e7` | assistant prefill channel |
| 4 | `881a5e0d` | DSML parser tolerance |
| 5 | `3c010d37` | split streaming delimiters |
| 6 | `f5dbb050`, `ad26cd60` | API semantics tests, validation record |

**The local quality gate is discharged.** basedpyright 0 errors, ruff clean, 717
passed / 3 skipped across `src`, golden regeneration an empty diff, TP parity
matrix green for `deepseek_v4_bf16` and `deepseek_v4_q4`. That clears the
`VERIFICATION OUTSTANDING` notes carried on Tasks 1-5.

**But the model does not work.** At `temperature=0.0` the live cluster returns
incoherent text and never emits EOS — `沙发` for "Say hello in one sentence.",
`". The The The..."` on the most recent run. Phase one is not complete and cannot
be declared so.

---

## 2. What is proven, and can stop being re-litigated

* **Prompt construction is correct.** All 16 committed goldens match; effort tiers
  differentiate live (prompt prefix 14 → 93 → 106 tokens); the prefix cache reused
  284 of 286 tokens with a negative control at exactly 0.
* **The tokenizer is correct.** The startup "generic tokenizer" RuntimeWarning is
  benign, fully diagnosed: unknown `model_type: deepseek_v4` plus a `rope_scaling`
  key makes transformers read an attribute only concrete configs define. mlx_lm
  rebuilds from the checkpoint's own `tokenizer.json`, which reproduces all 16
  golden token-id arrays exactly. Do not edit `config.json`.
* **The forward pass is deterministic.** Three byte-identical greedy completions
  eliminate RDMA races and non-deterministic reduction order.
* **The sampler is correct.** `make_sampler` uses argmax at temp 0, `resolve()`
  preserves `0.0`, `mx.random.seed(42)` is pinned.
* **`wo_a`'s weight width shards correctly on the live cluster.** Round-1
  instrumentation, both ranks at `N=2`: `(8192, 1024)` → `(8192, 512)`, with the
  unsharded stage anchoring the packing ratio empirically.
* **The cross-rank collective fires.** Wrapping `wo_b` in `_AllSumLinear` makes
  `isinstance(self.wo_b, nn.QuantizedLinear)` false, so the fused
  `_attn_wo_chain_quant` path is skipped and control reaches `self.wo_b(o)`, which
  performs the `all_sum`.
* **Head-sharding arithmetic is correct.** 12 synthetic tests pass in
  `test_v4_attention_sharding.py`, covering group-major ordering, head
  partitioning, and interleaved sink slicing.

---

## 3. The open question, stated precisely

**Round 2's elimination is void, and I can show why without hardware.**

Its numbers came from `tests/test_v4_shard_consistency_real.py` (untracked in the
working tree), which is a **single-process** script calling:

```python
shard_inplace(layer.attn.wo_a, sharding=_sharded_to_all, group=None)
```

with `group=None` and a `segments` value of 1. There is no world size, so nothing
was divided. That is why all four weights came back unchanged — including
`switch.gate_proj`, which shards a different axis. It is the same trap that
invalidated the original `test_v4_real_model_attn_mismatch.py`: a probe that
cannot observe the transform it is measuring.

So hypothesis H2 from the round-3 brief — sharding not applied on the cluster
while the reducers are — is **not supported by that data**. It is untested, and
round 1's live observation actively argues against it, since `wo_a.weight` did
halve at `N=2` on both ranks.

**What has never been checked on the live cluster is `scales`.** And the numbers
now available make that the sharp question:

| | trailing dim | implied unpacked input |
|---|---|---|
| `wo_a.weight` unsharded | 1024 | 1024 × 4 = **4096** ✓ = unsharded `group_feat` |
| `wo_a.scales` unsharded | 128 | 128 × 32 = **4096** ✓ agrees |
| `wo_a.weight` after `N=2` shard (observed, round 1) | 512 | 512 × 4 = **2048** ✓ = sharded `group_feat` |
| `wo_a.scales` after `N=2` shard | **unknown** | 64 → 2048 ✓ consistent, or 128 → 4096 ✗ **the bug** |

`attn.wo_a` is mxfp8 (`bits=8, group_size=32`); `switch_mlp` is mxfp4
(`bits=4, group_size=32`).

If `scales` stays at 128 while `weight` goes to 512, then
`_grouped_output_projection` reshapes both with a trailing `-1`, both succeed with
different widths, and `mx.quantized_matmul` dequantizes every group with the
wrong scales. No exception, every layer, every token, deterministic — which
matches every symptom, including one token repeating forever.

**One number on the live cluster settles it.**

---

## 4. Raised, not fixed

* **`ModelCard.backends` is required with no default** (`model_cards.py:170`), so
  custom cards written before that field fail validation.
  `_load_cards_from_dir` catches `ValidationError` per card and only *warns*, so
  the card is silently skipped and the model vanishes from the cache. Worked
  around by hand-editing `~/.exo/custom_model_cards/*.toml` — undocumented machine
  state, defect unfixed. `fetch_from_hf` already defaults unknown models to every
  backend for exactly this reason. Predates this branch, affects every model.
* A trailing assistant turn carrying `tool_calls` is dropped with only its text
  re-appended, losing the calls.
* `_PARAM_PATTERN` is attribute-order sensitive: `string="true" name="city"`
  silently drops arguments. Returning `None` on zero parameters would break every
  legitimate no-arg tool call.
* The checkpoint's top-level `quantization_config.group_size` is 64 while these
  modules report 32 — quantization is per-module. Worth confirming the
  post-quantize audit compares per-module parameters, since `mode` mismatches are
  invisible to strict loading.

## 5. Open decisions for Jared

* **Commit conventions.** `e19eebae` and codex's earlier commits use lowercase
  conventional-commits; everything from `b2822771` follows RULES.md. All unpushed,
  so amendable. `f5dbb050` and `ad26cd60` also share an identical subject.
* **`scripts/` is outside basedpyright's `include`** (`src`, `bench`, `tools`), so
  `validate_dsv4_live_api.py` (93 untyped-JSON errors), `compare_dsv4_reference.py`
  and the comparison scripts are unchecked. Whether `scripts` joins the include or
  these move to `tools/` is your call.

## 6. Working-tree state to resolve

* `src/exo/worker/engines/mlx/auto_parallel.py` is **modified and uncommitted** —
  the round-2 diagnostic scaffolding is still in place. Decide whether it becomes
  the round-3 probe, a permanent assertion, or is reverted.
* `scripts/compare_exo_vs_omlx_weights.py` is **untracked**, despite the round-2
  report stating it was committed. Worth committing — the OMLX comparison will be
  wanted again.
* `tests/test_v4_shard_consistency_real.py` is untracked and misleading (see §3).
  Delete it, or move it under `src/.../unittests/` and give it a real group.
* `resources/inference_model_cards/Jundot--...toml` is deliberately untracked and
  needed for live runs — leave it.
* `.git/_stale_locks/` accumulates moved lock and `tmp_obj_*` files because the
  Cowork mount forbids `unlink`. `rm -rf .git/_stale_locks && git gc` natively.

---

## 7. Test-quality lessons this project keeps re-teaching

Three separate probes in this investigation reported a conclusion they could not
support, each for the same reason — the harness could not observe the thing it
claimed to measure:

1. `run_live_matrix_validation.py` appended `PASS` whenever HTTP did not raise,
   and reported 36/36 while its own table showed no tool call ever fired.
2. `test_v4_real_model_attn_mismatch.py` "pinned" a shard mismatch in a single
   process, where the shard is a no-op.
3. `test_v4_shard_consistency_real.py` checked shard consistency with
   `group=None`, so nothing was sharded.

And one of mine: the round-2 helper dropped `world_size` and `rank`, which is why
(3) could not be spotted from its output.

The discipline that catches all four: **before believing a green result, ask what
input would have made it red.** If you cannot name one, the check is not a check.
`validate_dsv4_live_api.py --self-test` exists to answer that question for the
live harness, and it found two real bugs in itself during authoring.
