# OMLX Upstream Sync — Baseline

**Date:** 2026-08-11
**Branch:** `codex/dsv4-omlx-upstream-sync`
**Purpose:** a health check and sanity range for the OMLX reconciliation.

## This is NOT a regression reference

Cross-window variance on this cluster is far larger than any effect Tasks 5 and 6
aim to detect. Measured against previously recorded figures:

| metric | this baseline | previously recorded | delta |
|---|---|---|---|
| `prefill_tokens_per_second` | 109.145 | 187.00 | −42% |
| `time_to_first_token_seconds` | 5.305 | 2.9197 | +82% |
| `decode_tokens_per_second` | 23.259 | 21.37 | +9% |
| `cold_prefill_32k_seconds` | 196.466 | 163.792 | +20% |

Decode improved while prefill collapsed. That is unexplained and no explanation
should be inferred from these numbers alone.

**Consequence:** performance claims in Tasks 5 and 6 come from **same-window
A/B/A** measurement, never from differencing against this document. See the
implementation plan's "Same-window A/B/A measurement" section.

## Measurements

All from instance `dd53379f-101e-42f0-b27e-e4c94d3e71d1` (the D2 window),
except where noted.

| metric | value | notes |
|---|---:|---|
| `decode_tokens_per_second` | 23.259 | median of 3 runs |
| `prefill_tokens_per_second` | 109.145 | median of 3 runs |
| `time_to_first_token_seconds` | 5.305 | median of 3 runs |
| `cold_prefill_16k_seconds` | 78.872 | 16,036 prompt tokens, single sample |
| `cold_prefill_32k_seconds` | 196.466 | 32,036 prompt tokens, single sample |

`cached_tokens = 0` on all three throughput runs and on both cold windows.

No 64K request was issued at any point.

## Environment

Identical on both nodes:

```
mlx         0.32.0.dev20260804+cc3f3e60
mlx_lm      0.31.3
core_sha256 8276864c00f3f22bb1f5ef65c7b7687f5df0346759d9f314b7c8e045ce1b1b48
```

Thunderbolt addresses at time of measurement: M5 `169.254.195.150`, M4
`169.254.26.204`. **These reassign** — the handover's `169.254.240.63` is stale.
Always re-derive.

## Gates

`src`-scoped: `746 passed, 3 skipped, 172 deselected`; basedpyright 0 errors, 0
warnings, 0 notes.

Repo-wide `ruff check`: **8** errors across **scripts/compare_dsv4_reference.py, scripts/compare_exo_vs_omlx_weights.py, scripts/compare_loaded_vs_checkpoint.py, scripts/parity_sharded_vs_unsharded.py, scripts/probe_shard_rank_selection.py**.
`scripts/` sits outside basedpyright's include (`pyproject.toml:138`) but inside
ruff's scope (`:243`), and every previous "clean" claim was `src`-scoped. Known,
raised, unresolved — recorded, not fixed.

## Determinism — UNRESOLVED, pre-existing

Two experiments, both confounded. Neither settles the question.

| window | endpoint | prefix cache | completion tokens | verdict |
|---|---|---|---|---|
| D2 `dd53379f` | `/v1/chat/completions` | on | 789 / 1093 / 1093 | DIVERGED |
| S0-final `5e4b1a09` | `/v1/chat/completions` | on | 1369 / 1369 / 1369 | IDENTICAL |
| S0-final `5e4b1a09` | `/bench/chat/completions` | off | 1369 / 8192 / 8192 | **INVALID** |

**The `/bench` arm is invalid as a determinism test.** `generate.py:612-615` and
`batch_generate.py:300-303` ban EOS tokens on the bench path — *"Only sample
length eos tokens"* — so generation runs to `max_tokens` by design.
`finish_reason=length` there is expected behaviour, not divergence.

**The two `/v1` rows contradict each other**: identical prompt, settings, build
and endpoint, divergent in one window and stable in the next. That is
intermittent, which is what a reduction-order or RDMA race looks like.

The cache hit in the stable window was only **17 tokens** of prompt prefix — far
too small to explain a 500-token output difference, so "cold versus warm
numerics" is a weak explanation too.

**Status:** pre-existing cluster property, not introduced by this workstream, and
deliberately not chased further — three instance windows had already gone to it.
Tasks 5 and 6 retain their offline unit-test correctness proofs, which compare
in-process attention output on fixed inputs and are unaffected by cluster
non-determinism. Only their *live regression* checks are weakened.

**Do not record this as resolved without new evidence.**

## Raised, not fixed

- `scripts/instance_window_battery.py:198` — `for tokens in (16_000, 32_000,
  64_000)` inside `context_ceiling`, and `--only` selects by measurement name, so
  there is no invocation yielding 16K and 32K without also firing a 64K prefill.
  A 64K prompt has caused a kernel watchdog host panic. Task 4's admission guard
  is the remedy.
- The native runbook's launch block at `:206-209` omits `EXO_BOOTSTRAP_PEERS` and
  `--no-sync`, both required in practice.
- The handover's Thunderbolt address for node 1 is stale.
- Node 2 expired mid-measurement in an earlier window, costing a throughput
  sample.
- `use_prefix_cache` is exposed only on `BenchChatCompletionRequest`, so the
  normal endpoint's cache cannot be controlled without also inheriting the bench
  path's EOS ban. This is what made both determinism experiments confounded.
- JACCL initialisation failed reproducibly until both nodes were rebooted;
  consistent with stale RDMA state after an unclean teardown, but a reboot also
  resets broader host state, so causality is not isolated.
