# OMLX Upstream Reconciliation for the DeepSeek-V4-Flash-0731 EXO Port

**Date:** 2026-08-11
**Status:** design, approved for planning
**Working revisions:** EXO branch `codex/dsv4-0731-backbone` (validated tip); OMLX pinned reference `50846648`; OMLX upstream head `2450a53c`
**Relates to:** `2026-08-04-deepseek-v4-flash-0731-exo-rdma-design.md`, `2026-08-11-dsv4-0731-consolidated-handover-v2.md`

## Purpose

The project's design document pins OMLX `50846648` (2026-08-04) as the reference
revision for checkpoint layout, 0731 numerics, and the DeepSeek-0731 prompt
protocol. `jundot/omlx` has since advanced 51 commits to `2450a53c`. This
document records what that delta contains, what of it is integrable, and how the
integrable part is to be implemented and verified.

It is written after the phase-one goal was met. Two-rank distributed inference is
validated end to end; this work is hardening and forward-compatibility, not
recovery.

## The constraint that shapes everything

EXO does not import `omlx`. It subclasses the pinned fork
`rltakashige/mlx-lm@leo/deepseek-v4` (`6a3df6c`) and layers a compatibility
module over it (`src/exo/worker/engines/mlx/deepseek_v4_0731_model.py`).

OMLX's `omlx/patches/deepseek_v4/` and that fork are **independent
implementations of the same architecture**. OMLX has three attention classes
(`LocalAttention`, `CompressedAttention`, `SparseCompressedAttention`) over a
`PoolingCache`; the fork has a single `V4Attention` over a `DeepseekV4Cache`,
with its own fused Metal kernels (`_make_hc_split_sinkhorn_kernel`,
`_make_overlap_emit_kernel`, `_make_moe_gate_kernel`).

Consequently most of the upstream delta optimises code shapes that do not exist
on EXO's side. **The reconciliation is mostly reference-only.** Treating it as a
merge would be a category error. What survives triage is small, and is valuable
precisely because it was selected against the port's real structure rather than
against a diff.

A second consequence: the three genuinely portable model-path items live inside
the pinned fork, not in EXO. They are applied by runtime installation over the
fork at load time, following the precedent already in the branch
(`install_deepseek_v4_sdpa_float32()`), not by changing the dependency pin. The
handover records that the MLX and `mlx_lm` pins are load-bearing for the float32
SDPA fix; a pin change would force re-validation of that fix as well.

## Triage of all 51 commits

### Group I — Integrate (5 commits, 3 work items)

| Commits | Item | Mechanism |
|---|---|---|
| `745d9c22`, `4dc9baab`, `8a00cfbf` | Prefill memory estimation and reclaim accounting | EXO-native reimplementation. OMLX's scheduler is not EXO's, so the idea transfers and the code does not. |
| `4c6b5931` | Skip indexer scoring when `pooled_len <= index_topk` | Runtime install over the fork |
| `b128b232` (cache portion only) | Pooled buffer append-in-place rather than per-step `concatenate` | Runtime install over the fork |

Evidence for the two fork-level items, verified against `6a3df6c`:

- `Indexer.__call__` (`mlx_lm/models/deepseek_v4.py:1799`) computes
  `k = min(self.index_topk, idx_kv.shape[1])` and always runs the scoring einsum
  and `argpartition`, including when `pooled_len <= index_topk`, where the
  selection is the identity set. With `index_topk = 512`, that condition holds
  for ratio-128 layers across the whole supported context range.
- `_CompressorBranch` (`:1182`) grows the pooled buffer with
  `mx.concatenate([pool, new_pooled], axis=1)` on every update — the quadratic
  pattern upstream replaced with geometric append-in-place. This is a candidate
  explanation for 163.792 s on a 32,819-token cold prefill.

The kernel half of `b128b232` (wsdpa fusion, `custom_kernels/glm_moe_dsa/csrc`)
is not portable; the fork has its own fused kernels for the same operations.

### Group II — Verify, then decide (2 commits)

| Commit | Question to settle |
|---|---|
| `b6811ed6` | Upstream's `_native_ratio128_attention_enabled` now **disables** the native ratio-128 sparse-attention path for sub-4-bit V4 quantization. Does that rule bite on `oQ4e`'s realized per-module quantization, and are the fork's equivalent fast paths reachable under this recipe at all? |
| `c48e1a82` | Upstream split one conflated block-config threshold into separate mxfp4 (`16384`) and affine (`8192`) crossovers. Does the fork's `switch_layers.py` conflate them the same way? This is a tuning-correctness defect, not a tuning number. |

Both are investigations. Per `RULES.md`, a hazard found here is **raised, not
fixed**.

### Group III — Phase-two inputs, recorded not integrated (3 commits)

`5d77302b` and `f3267488` implement MTPLX side-car import for `qwen3-next-mtp`
packaging, which does not describe `DeepSeek-V4-Flash-0731-oQ4e-mtp` — that
checkpoint carries its MTP tensors in-index under `mtp.*`. Two things transfer
regardless:

- **`mlx_lm`'s weight glob is `model*.safetensors`.** A `mtp.safetensors`
  side-car is never opened even when the index points at it. `f3267488` exists
  because `5d77302b` shipped exactly that mistake: detection flipped true while
  the text path could not bind the head.
- The fail-closed pattern — validate the contract and audit the payload before
  any mutation, write config and index atomically with timestamped backups.

`7fe62f9c` (admit late joins during singleton MTP decode via drain handoff) is
late-join semantics for the phase-two decode loop.

### Group IV — Reference-only, structurally different (7 commits)

`cbd7daa4`, `7efcc629` (per-member `CacheList` block storage — EXO has its own
cache-copy design in `engines/mlx/cache.py`), `cfec5021` (OMLX scheduler cache
block sizing), `267d5436`, `ded2bbe4` (GDN sidecars — not a V4 layer type),
`2c10f0fb` (grammar sampling), `2fcf8894` (kernel cleanup for `b128b232`).

### Group V — Not applicable (34 commits)

Ling 3.0 / bailing_hybrid ×5 (`c6244635`, `d4adcc35`, `97fdbad3`, `fe3101e3`,
`b75e1aa0`);
Muse Glimmer VLM ×4 (`6ee393d4`, `39bb1784`, `9a57d63d`, and `e1acb0bc`'s VLM MTP
thinking budget); Jina reranker ×3 (`876e1797`, `03a3120e`, `7b755b90`); Inkling
×2 (`5215d9b4`, `5306b733`); generic XML tool-calling ×3 (`cdeea4c5`, `12937527`,
`d5592aa0`); `13997cec` gemma4; `a714035f` Hermes; admin and engine ×4
(`198c5ce9`, `fe79b272`, `c10c5c5b`, `9b59e122`); i18n and mac-app ×3
(`76e13909`, `9aacf8d9`, `90277828`); `128615b7` codex CLI; `d2575b1d` deps;
version bumps ×3 (`49ec2716`, `ab95612a`, `350dc08b`); `2450a53c` web search;
`24e0d2b1` test stub; `95c38c13` VLM sampler typing.

The three `fix(tool-calling)` commits are OMLX's generic XML tool protocol, not
DSML. They are not applicable regardless of merit, since the checkpoint has no
trained tool-call capability under either declared protocol.

### Group VI — The provable non-change

`omlx/patches/deepseek_v4/chat_template_v4.py` and `tool_parser_v4.py` have
**zero commits** in `50846648..2450a53c`. The prompt protocol and the DSML
reference parser are unchanged, so re-pinning the reference revision cannot
invalidate the 16 committed goldens or Tasks 1 through 5. This is stated as a
claim to be proved in S1, not assumed.

## Segments

Work lands on `codex/dsv4-omlx-upstream-sync`, cut from the validated tip into a
fresh worktree. The `codex-dsv4-0731-backbone` worktree carries uncommitted user
work and is not to be touched; nor is `exo-dsv4-validation` on node 2 (which
retains an unapplied `stash@{0}`) or the detached `dsv4-0731-live-clean`.

Each segment is one commit and one codex packet. Packets are issued one at a
time, each written after the previous segment's report has been cross-checked.

### S0 — Baseline capture

No code change. On the new worktree: full offline gate
(`basedpyright && ruff check && nix fmt && pytest`), `--determinism 3`,
TTFT/prefill/decode via `scripts/measure_dsv4_throughput.py`, and cold 16K and
32K prefill wall time with `cached_tokens=0` confirmed.

This is a control, not a formality. The project's recorded failures include a
0.05 per-layer tolerance read for weeks as end-to-end evidence, and a harness
that reported 36/36 without a single assertion. No later segment may claim
"faster" or "unchanged" without differencing against S0.

### S1 — Re-pin the reference revision

Update the design document's `Working revisions` to OMLX `2450a53c` and add a
reconciliation appendix carrying this triage. Prove Group VI: the two reference
files byte-identical across the range, and golden regeneration producing an empty
diff. Offline only, no runtime change.

### S2 — Settle Group II

Investigation. Report the checkpoint's realized per-module quantization against
the sub-4-bit rule, whether the fork's fast paths are reachable under this
recipe, and whether `switch_layers.py` conflates the mxfp4 and affine crossovers.
Deliverable is a written finding plus, only if a hazard is real, a regression
test that pins the current behaviour. Fixes are raised for decision, not applied.

### S3 — Indexer all-pooled shortcut

Add `install_deepseek_v4_indexer_shortcut()` beside the existing float32 SDPA
installer. Two requirements carry the risk:

1. **A contract guard.** The installer patches private fork internals. It must
   assert the shapes and call signature it depends on and fail loudly if a fork
   bump changes them, rather than silently installing a no-op.
2. **An ordering proof.** The substituted index set must be shown to be the
   identity *and* the downstream `concatenate([window_kv, gathered])` layout
   shown to be preserved. Attention is permutation-invariant over keys only when
   the mask permutes with them; this is the one place the change could be quietly
   wrong.

Gate: offline suite, live `--determinism 3`, `parity_sharded_vs_unsharded.py`
unchanged, throughput differenced against S0.

### S4 — Pooled append-in-place

Higher risk than S3. It mutates cache growth, and EXO's `_copy_compressor_branch`
and `_copy_v4_cache` (`engines/mlx/cache.py:136`, `:150`) capture state from
exactly that buffer. Upstream's own note is that views survive appends because
committed rows are never rewritten, but long-lived consumers must copy —
snapshot and prefix-cache paths are long-lived consumers.

Gate: offline suite including the prefix-cache tests, live `--determinism 3`,
`instance_window_battery.py`, and 16K/32K cold prefill timings differenced
against S0.

### S5 — Prefill admission guard

Estimate prefill memory before allocating and refuse an oversized prompt with an
explicit API error. This closes the open prompt-cap question and removes a
host-availability risk: a 64K prompt caused a kernel watchdog panic with
`watchdogd` missing check-ins for 94 seconds.

**The gate must prove the guard fires without issuing a 64K prefill.** Verify at
the estimator and admission boundary with synthetic sizes; reproducing the panic
is forbidden.

The refusal must also be diagnosable. The handover raises that a request carrying
`tools` against a model with `has_tool_calling=False` is accepted, executed, and
returned as success with `tool_calls=null` while both ranks log the condition.
A silently-capped prompt would be the same defect class.

### Sequence

S0 → S1 → S2 → S5 → S3 → S4.

S0 is unconditionally first. S1 and S2 are offline and independent. S5 precedes
the perf pair because an unbounded prompt can currently take a host down, which
outranks prefill latency. S3 precedes S4 so each throughput delta is attributable
to one change.

## The codex packet contract

Each packet is a self-contained document. The agent receives no other context.

1. **Context and prohibitions** — branch, worktree, what is already true. Explicit
   do-not-touch list: the MLX and `mlx_lm` pins, `install_deepseek_v4_sdpa_float32`,
   the other three worktrees, and any live 64K prefill.
2. **Provenance** — the upstream commits the segment derives from, what is being
   taken, and what is deliberately not being taken and why.
3. **Required reading** — specific `file:line` references, not directory names.
4. **Steps** — numbered, each with its own completion condition.
5. **Verification gate** — exact commands and pass criteria. Every new test states
   what input would make it FAIL. `scripts/validate_dsv4_live_api.py` is run with
   `--self-test` first and no assertion in it may be softened.
6. **Claims prohibited without evidence** — an explicit list, so an unverified
   claim is a protocol violation rather than an oversight.
7. **Report format** — diff, verbatim verification output, and answers to a stated
   question list, returned for cross-check.
8. **Commit message form** — per `RULES.md`, matching the convention used from
   `b2822771` onward.

## Cross-check protocol

Two review rounds per segment. The project's record is that round two repeatedly
found more than round one, including self-introduced regressions.

Round one checks the diff against the packet: scope, `RULES.md` conformance
(exhaustive typing, `Literal` over enums, no three-letter acronyms, no new
dependencies, error-handling rationale in docstrings), and whether every claim in
the report has evidence behind it in the pasted output.

Round two is adversarial and independent of the report: what would make this
wrong, which new test would still pass on the unfixed code, and what did the
packet fail to ask for.

A segment is complete when both rounds pass and the gate output is in hand. Not
when the agent says it is.

## Risks

**A fork bump silently disables S3 or S4.** Mitigated by the contract guard;
accepted residual risk is that the guard itself drifts. Recorded in the runbook.

**S4 interacts with snapshot and prefix-cache paths.** The prefix cache is
already load-bearing (284 of 286 tokens reused, with a negative control at
exactly 0). A regression here is a correctness regression, not a performance one.

**S3's ordering claim.** Addressed by requiring a proof rather than an assertion.

**The perf work may not pay.** If S3 and S4 together do not move the 32K cold
prefill measurably against S0, both are reverted rather than kept on the argument
that they are theoretically better.

## Open decisions for Jared

- The prompt-length cap value that S5 enforces, and whether it is a hard refusal
  or a configurable ceiling.
- Whether to send the two fork-level fixes upstream to `rltakashige/mlx-lm` in
  parallel, with the OMLX commits as prior art, so the runtime installs can
  eventually be retired.
- Whether re-pinning the reference revision in S1 should also update the
  `PR-BODY-dsv4-0731-backbone.md` at the repository root.

## Explicitly out of scope

Vendoring OMLX's DeepSeek-V4 runtime or Metal kernels. Changing the MLX or
`mlx_lm` pins. Any phase-two MTP or DSpark implementation. Any tool-calling work,
which remains blocked on sourcing a tool-capable checkpoint rather than on code.
