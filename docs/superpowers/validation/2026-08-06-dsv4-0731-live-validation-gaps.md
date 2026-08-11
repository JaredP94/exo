# DeepSeek V4 0731: gaps in the live validation record, and one raised defect

**Date:** 2026-08-06
**Concerns:** `docs/superpowers/validation/2026-08-04-dsv4-0731-prompt-api.md`
(commit `ad26cd60`), and the two scripts it was produced by.

This document qualifies that record. It does not dispute the local quality gate,
which is sound and which discharged the outstanding verification on Plan 2
Tasks 1-5. It disputes the live half.

## Summary

The record states that all 36 live API combinations passed, that "content and
reasoning channels remained strictly separated, structured tool calls were
preserved, and no DSML markup leaked into output streams", and that append-only
prefix caching was verified. **None of those four claims is supported by the
harness that produced them.**

In `scripts/run_live_matrix_validation.py`, `PASS` is appended whenever the HTTP
request does not raise:

```python
results.append((case_name, "ChatCompletions", "non-stream", "PASS", f"content_len=..."))
except Exception as e:
    results.append((case_name, "ChatCompletions", "non-stream", "FAIL", str(e)))
```

There is no assertion on channel separation, DSML leakage, `finish_reason`,
terminal-event counts, tool-call structure, or cross-endpoint agreement. All 36
verdicts therefore mean only: the endpoint returned 200 with parseable JSON.
That is a liveness check.

`scripts/verify_live_prefix_caching.py` prints two prompt-token counts and then
prints `SUCCESS` unconditionally. It never renders or compares prompts and never
reads `cached_tokens`.

## What the recorded data itself shows

Read as data rather than as verdicts, the committed table already contains the
evidence that the live matrix did not exercise its subject:

1. **No tool call ever fired.** Cases 5 (`single_tool`), 6 (`multi_tool`) and 7
   (`tool_result_followup`) all record `tool_calls=False`, and all are marked
   PASS. Tasks 4 and 5 — the entire DSML tolerance and split-delimiter effort —
   are therefore **unvalidated live**. The claim that structured tool calls were
   preserved is contradicted by the table it appears above.

   Likely cause: `max_tokens=32`. `<｜DSML｜tool_calls>` alone is six tokens and
   a full invoke with one parameter is 25 or more; with reasoning consuming the
   budget first, the model cannot reach a closing marker. Cases 5 and 6 report
   *identical* metrics (`content_len=2, reasoning_len=62`), which fits truncation
   at the same point rather than two distinct tool behaviours.

2. **`done=0` on all nine Responses streaming rows.** The script counts the
   literal `[DONE]` sentinel, which belongs to Chat Completions; the Responses
   API terminates with a `response.completed` event. So "terminal events appear
   exactly once" is unverified across 18 of the 36 rows, and the harness would
   have reported PASS with zero terminal events either way.

3. **Cases 2-8 produced 1 to 3 characters of content.** At `max_tokens=32` with
   reasoning enabled, `finish_reason` was almost certainly `length` throughout —
   the script never records it. "No DSML markup in the content channel" is
   vacuously true when there is almost no content to inspect.

4. **Effort tiers are not differentiated.** Case 3 (`thinking_high`) reports
   `reasoning_len=31` against case 2 (`thinking_low`) at 139. Nothing asserts the
   tiers differ, which is Task 2's whole subject.

5. **Cross-endpoint agreement is never compared**, and the Responses conversion
   flattens `tool` messages into `user` prose, so case 7 does not exercise tool
   results on that endpoint at all.

6. **Step 8 was not performed.** `GET /` returning 200 is a liveness probe on a
   static asset. The step asks for three conversations *through the dashboard*,
   confirming the displayed answer excludes DSML markup and that reasoning is
   separated in the UI.

None of this implies the implementation is broken. The 27 offline tests in
`f5dbb050` cover most of these properties properly and were mutation-tested.
What is unproven is that the live two-node cluster exhibits them — which is the
one thing only Task 6 could establish.

## Remedy

`scripts/validate_dsv4_live_api.py` replaces both scripts. Differences that
matter:

* Every check returns PASS, FAIL or INCONCLUSIVE, and `PASS` is only returned by
  a predicate that could have returned FAIL on that input.
* `finish_reason == "length"` yields INCONCLUSIVE, never PASS.
* Terminal events are counted per endpoint: `[DONE]` for Chat Completions,
  `response.completed` for Responses.
* `max_tokens` is per-case, 512-768 for tool cases.
* Tool calls are asserted present, counted, name-checked, and their `arguments`
  must parse *and* survive `json.dumps(..., allow_nan=False)` — a permissive
  `json.loads` accepts `Infinity`/`NaN`, which a strict client's `JSON.parse`
  rejects. This is the Task 4 hole, asserted end to end.
* Marker leakage is sought in *decoded* argument values, recursively, because
  `json.dumps` escapes non-ASCII by default and a substring test on the wire
  string misses an escaped `｜DSML｜`.
* Effort tiers are checked by comparing `prompt_tokens` across otherwise
  identical requests: the tier injects a prompt prefix, so `high` and `xhigh`
  must differ. Collapsed tiers fail.
* Prefix-cache reuse is evidenced by `prompt_tokens_details.cached_tokens`, with
  a **negative control** — an unrelated prompt that must not show high cached
  tokens. Without the control, an always-high counter reads as success.
* Exit code is non-zero on any FAIL or INCONCLUSIVE, so it gates rather than
  needing to be eyeballed.

**`--self-test` drives every check with good and deliberately-bad synthetic
payloads and needs neither the cluster nor MLX.** It found two real bugs in the
harness during authoring — the raw-string leak test and the missing strict-JSON
guard, both listed above — and now passes. Run it before spending a cluster run:

```bash
uv run python scripts/validate_dsv4_live_api.py --self-test
uv run python scripts/validate_dsv4_live_api.py --only 5,6,7   # the tool cases first
uv run python scripts/validate_dsv4_live_api.py
```

Start with `--only 5,6,7`. That is where the risk concentrates and where the
previous run proved nothing.

`ruff check` and `ruff format` are clean. `basedpyright` reports 93 errors, all
from untyped JSON traversal under `reportAny` / `reportUnknownMemberType`.
`scripts/` is outside `[tool.basedpyright] include` (`src`, `bench`, `tools`), so
this is not CI-visible, and the two scripts it replaces are equally unchecked.
**Raising rather than deciding:** whether `scripts` should join the include, and
whether this harness should move to `tools/` and be typed to the strict standard
with narrowing accessors, is Jared's call. It is not claimed to be type-clean.

## Raised defect: a required `backends` field silently drops legacy custom cards

Found while reconciling the cluster environment; recorded in the validation
record as a setup fix. It is a product defect and is still live in the repo.

`ModelCard.backends` is declared `list[Backend]` with no default
(`src/exo/shared/models/model_cards.py:170`), so it is required. Cards written to
`~/.exo/custom_model_cards/` before that field existed fail validation.

`_CardCache._load_cards_from_dir` (`model_cards.py:75-87`) catches
`ValidationError` per card and logs a warning:

```python
except (ValidationError, TOMLKitError) as e:
    logger.opt(exception=e).warning(f"failed to validate model card at {toml_file}")
```

So the failure mode is **not** a hard startup crash — I overstated that
initially. It is worse in one respect: the card is skipped, the model silently
disappears from the cache, and the only signal is a warning in the log. A user
upgrading loses their custom cards without an error.

Observed on both nodes; worked around by hand-editing each legacy card to add
`backends = ["MlxMetal", "MlxCuda", "MlxCpu"]`. That edit lives in `~/.exo/` on
Jared's machines only — it is undocumented machine state, and the defect is
unfixed.

There is precedent in the codebase for the obvious fix. `ModelCard.fetch_from_hf`
already defaults unknown models to every backend, with the rationale in a comment
at `model_cards.py:259-261`:

> all backends — we don't know what an arbitrary HF model supports; let placement
> gate decide

Giving `backends` a default of `list(Backend)` would apply that same intent to
legacy cards. A migration that rewrites cards in place is the alternative.

Per `RULES.md` — "if you see code that violates these rules, raise it with me
rather than fixing" — this is raised, not fixed. It is out of scope for Plan 2
either way: it predates this branch and affects every model, not just V4.

## Status of the plan's Completion Gate

Of the ten items, the six that are prompt- and parser-level are met by the
committed goldens and the offline suites. These four are not yet evidenced:

* *All 36 live API combinations pass on the validated two-rank JACCL instance* —
  not evidenced; see above. Re-run with the new harness.
* *Latest reminders preserve the append-only prefix invariant when reasoning is
  retained* — asserted offline in `f5dbb050`; live evidence pending
  `cached_tokens`.
* *Arbitrary delimiter chunking never leaks markers or hangs* — proven offline
  and by 6,288 partition cases, but never exercised live, because no tool call
  fired.
* *The dashboard uses the same healthy instance and does not expose DSML markup*
  — not performed.

## 2026-08-10 SDPA float32 validation addendum

The DeepSeek V4 SDPA workaround in `eb3e6fd6` passed the focused local gate:
67 selected tests passed, including all seven guards in
`test_dsv4_sdpa_float32_patch.py`; basedpyright reported zero errors, warnings,
and notes for the patch and its test.

The two-device MLX probe used bit-identical `[-1, 1]` inputs and reported mean
absolute attention output `0.381306`. Float32 output agreed across the devices
within the gate (`max_abs=0.000787497`, tolerance `0.025`). The unpatched
bfloat16 comparison was `max_abs=1.14062`. Because attention output is a convex
combination of values in this probe, its magnitude cannot exceed one; that
bfloat16 disagreement is evidence of a wrong computation, not ordinary rounding.
The probe now prints each rank's bfloat16 min/max and finiteness for an upstream
MLX report.

The enhanced probe did not complete on its first rerun: rank 0 blocked in
`mlx::core::distributed::ring::RingGroup::accept_connections()` at
`TCPSocket::accept`, while rank 1 never connected. Treat this as a topology
failure and do not work around it in the live-server phase; preserve the exact
failure and re-establish a symmetric ring first.

Both hosts use MLX-LM 0.31.3 and MLX source commit
`cc3f3e60be1289506125f2fa19b73b05aa770df8`, but their dev wheels are dated one
day apart (`0.32.0.dev20260803` and `0.32.0.dev20260804`). That is sufficient to
proceed with the recorded result, but it is not binary parity, particularly
because this investigation concerns kernel selection. A future reproduction and
upstream report should pin one identical MLX wheel, not only a source commit.

The committed message has one terminal blank-line difference from the reference
file, caused by Git's `cleanup=strip` during the final message-only amend. It is
cosmetic; do not amend the commit again.

## 2026-08-11 acceptance addendum

The Phase 1 control-plane fix is now live-validated on the two Macs. After
rebuilding `exo_rs` on both hosts, both APIs reported the same two-node
topology with reciprocal direct TCP edges (`169.254.240.63` and
`169.254.233.2`) and reciprocal JACCL edges (`rdma_en6` and `rdma_en3`). A
`Tensor`/`MlxJaccl` instance reached `RunnerReady` on both ranks; the first
successful instance was `f89529d7-386b-4f8b-8215-72384ff3a98a`. The resilient
reachability and poll-loop fixes are commits `62b94c05` and `2289e119`.

The provisional cold-first Phase 2 probe ran on a fresh instance
(`4d80d5fe-be0b-4c0f-bd09-9761f45d07b5`) with `cached_tokens=0` and produced the
one-token completion `' word'`. That evidence was later rejected as too weak
to establish conditioning; the replacement nonce probe and current Task 5
verdict are recorded below. The `use_prefix_cache=false` fix is `abceba63`; its
focused regression passed 2/2 and the full runner unit directory passed 149/149
on the real Metal host. Two identical live normal-chat requests after the fix
both reported `cached_tokens=0`.

The offline tool prompt probe is a hard Phase 3 blocker. With the actual
user-only message shape, `scripts/probe_tool_call_prompt.py --self-test`
fails because `get_current_weather` is absent from the rendered prompt. The
pinned `deepseek_v32.render_message` implementation only renders the global
`tools` argument for system/developer messages, not a user-only request. A
live tool-call run was therefore not claimed: the model is not told that a
tool exists. This remains a templating defect, not a parser verdict.

The cold streamed throughput measurements were:

| Requested prompt | Actual prompt | Cache | TTFT (s) | Prefill tok/s | Decode tok/s | Completion |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 546 | 0 | 2.2712 | 240.41 | 33.87 | 256 |
| 16,384 | 16,419 | 0 | 69.4549 | 236.40 | 31.34 | 256 |
| 32,768 | 32,804 | 0 | 168.5824 | 194.59 | 30.39 | 256 |

The subsequent 65,536-token probe did not complete: the local node rebooted
while the request was in progress and both APIs became unavailable. The
preserved panic report classifies this event as a kernel watchdog panic after
94 seconds without `watchdogd` check-ins; it does not show OOM or GPU-fault
evidence. No further live test above 32K is permitted in this acceptance run.
Revisit context management later, including leveraging OMLX TurboQuant to
reduce long-prefill resource use before reconsidering 64K.

## 2026-08-11 evidence correction (historical intermediate record)

### Task 5 cold-single verdict (superseded and failed)

The earlier one-token completion `' word'` is not a valid conditioning verdict:
the record did not establish a multi-token answer, and the branch that skipped
Tasks 7 and 8 was therefore unsafe. The earlier
`74c8eac4-d5e3-4297-8ff6-d6cdbb7d3e1a` attempt returned `'RO'` at an 8-token
budget with `finish_reason=length`; it was also truncated.

The replacement probe used fresh instance
`08e66087-b50d-4b35-a19a-109299f1f405`, with this exact request:

```text
Return exactly this string and nothing else: WIRE-7F3C9A2D
```

The request set `max_tokens=256`, `temperature=0`, and
`enable_thinking=false`. It reported `prompt_tokens_details.cached_tokens=0`,
`finish_reason=length`, and `completion_tokens=256`. The complete assistant
completion, quoted with the captured whitespace and escaping preserved, was:

```text
'\n\nWe need to solve the equation:\n\n\\[\n\\log_2(x+1) + \\log_2(x+3) = 3\n\\]\n\n**Step 1: Combine the logarithms**\n\nUsing the property \\(\\log_b A + \\log_b B = \\log_b(AB)\\):\n\n\\[\n\\log_2[(x+1)(x+3)] = 3\n\\]\n\n**Step 2: Convert to exponential form**\n\n\\[\n(x+1)(x+3) = 2^3\n\\]\n\n\\[\n(x+1)(x+3) = 8\n\\]\n\n**Step 3: Expand and solve the quadratic**\n\n\\[\nx^2 + 4x + 3 = 8\n\\]\n\n\\[\nx^2 + 4x - 5 = 0\n\\]\n\nFactor:\n\n\\[\n(x+5)(x-1) = 0\n\\]\n\nSo:\n\n\\[\nx = -5 \\quad \\text{or} \\quad x = 1\n\\]\n\n**Step 4: Check for domain restrictions**\n\nThe original logarithms require:\n\n\\[\nx+1 > 0 \\quad \\Rightarrow \\quad x > -1\n\\]\n\\[\nx+3 > 0 \\quad \\Rightarrow'
```

The nonce did not appear in this earlier probe. This was a failed exact-output
probe, not proof that prompt text was absent from the model input; it is
superseded as the conditioning diagnosis by the later nonce completion quoted
in the absolute-quality addendum below. The prefix-cache fix remains
regression-tested, and Tasks 7 and 8 were run independently below.

### Full-suite gate

The local `uv run` path could not open the sandboxed uv cache, and the local
headless process cannot initialize Metal. The repository-local `tools/src`
package resolved the former collection problem on the real Metal validation
host. Running `PYTHONPATH=tools/src .venv/bin/pytest src -q` there produced:
`734 passed, 3 skipped, 172 deselected in 27.85s`. The full-source gate is
therefore met on the validation host; local headless MLX execution remains an
environment limitation.

### 64K failure classification

The local node rebooted at approximately 00:19 SAST on 2026-08-11 and preserved
`/Library/Logs/DiagnosticReports/panic-full-2026-08-11-002133.0002.panic`. Its
panic string is `watchdog timeout: no checkins from watchdogd in 94 seconds`;
the backtrace names `AppleARMWatchdogTimer`. The report does not show an OOM or
GPU fault. Post-boot `memory_pressure` showed 96% free memory, zero swapins and
zero swapouts; this is evidence for a kernel watchdog panic, not a confirmed
RAM exhaustion event. The other node did not reboot. Reproducibility (“every
time”) was not tested: the standing safety limit remains no live prompt above
32K, so the operational record is **≤32K accepted, 64K hazardous, one confirmed
watchdog panic**.

The supported operational ceiling is a hard **32K prompt limit** for this
acceptance record; no live prompt above 32K was attempted after the panic.
Decision for Jared: consider enforcing an API prompt-length cap at 32K (or a
slightly lower operational margin) so unbounded requests cannot reach this
watchdog-shaped denial-of-service hazard. This follow-up does not implement
that policy. Revisit longer-context management later, including OMLX TurboQuant
support, before reconsidering 64K.

### Topology root-cause qualification

The Phase 1 tests prove that transport exceptions and poll-task exceptions are
contained, but no live log captured `connect failed from`, `reachability probe
failed`, or a poll exception for the original node-2 gap. Topology became
symmetric after the native `exo_rs` libraries were rebuilt/reinstalled on both
hosts and the checkouts/runtimes were reconciled. Therefore the Python
resilience changes are defensible hardening and are deployed and
regression-tested, but their causal role in the original missing edge is
**not proven**; the triggering interface class is unknown. The prior
“resilience fixes validated” wording should be read as deployment validation,
not root-cause confirmation. The defect may recur.

### Tool-call status and TurboQuant scope

Tool calls remain **unvalidated live**. The offline user-only prompt render
drops the tool schema, so live tool-call cases 5–7 were not run and have zero
live evidence; no parser or endpoint verdict is claimed for them. OMLX
TurboQuant is a context-state quantization/compression path that may reduce
resource use during long prefills; it appeared as follow-up scope because the
64K attempt reached a kernel-watchdog failure, but it has not been integrated
or shown to prevent that failure.

### 2026-08-11 independent Task 7 and Task 8 results

Task 7 was run regardless of the invalid Task 5 branch with two local MLX
processes and no two-node instance:

| Layer regime | Attention | FFN | Full block | Verdict |
|---|---:|---:|---:|---|
| Layer 0 (no compressor) | PASS, `max_abs=0.03125` | PASS, `0.00195312` | PASS, `0.015625` | PASS |
| Layer 2 (Indexer) | PASS, `0.03125` | PASS, `0.00390625` | PASS, `0.015625` | PASS |
| Layer 3 (Compressor) | PASS, `0.03125` | PASS, `0.00195312` | **FAIL, `0.125`** | FAIL |

All rank-agreement checks passed. Layer 3 therefore fails first at the full
block while attention and FFN remain within the `0.05` tolerance; this
implicates hyper-connection wrapping, not the Indexer, Compressor, head
sharding, or expert sharding. Commands were:
`mlx.launch -n 2 scripts/parity_sharded_vs_unsharded.py --layer {0,2,3}`.

Task 8 added
`src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_mask.py` with the
planned synthetic dimensions and `SEQ_LEN=7`. On real Metal it passed (`1
passed`): the attention spy observed query length 7, not 28. This is unit
coverage only and does not establish live tool-call behavior.

### 2026-08-11 hyper-connection follow-up

The ratio-pattern check did not generalize. With the original parity script,
layer 4 (ratio 4) passed with full-block `max_abs=0.0234375`, and layer 5
(ratio 128) also passed with full-block `max_abs=0.03125`. The evidence is
therefore a layer-specific distributed numerical fragility, not a proven
ratio-128 rule.

The parity script now reports `hc_pre` and `hc_post` separately for both the
attention and FFN paths, including input/output deltas and an amplification
factor. On layer 3's default path:

| Stage | Input delta | Output delta | Amplification |
|---|---:|---:|---:|
| `hc_attn_pre` | `0` | `0` | n/a |
| attention | — | `0.03125` | — |
| `hc_attn_post` | `0.03125` | `0.015625` | `0.5x` |
| `hc_ffn_pre` | `0.015625` | `0.0234375` | `1.5x` |
| FFN (chained path) | — | `0.121094` | — |
| `hc_ffn_post` | `0.121094` | `0.125` | `1.03x` |

The standalone FFN isolation still passes; the chained FFN result fails because
it receives the already-diverged residual stream. This localizes the first
observable mismatch to distributed attention, with subsequent hyper-connection
and FFN stages preserving/amplifying it rather than creating an independent
`hc_pre` mismatch.

As a hypothesis test only, `--float32-hyper` forces the hyper-connection
collapse/residual path through float32 in the parity harness. Layer 3 then
reported isolated attention `max_abs=9.5e-7`, chained FFN `5.7e-7`, and
full-block `max_abs=0.000487`, all PASS. `hc_ffn_post` still had a large
relative amplification factor (`859x`) because its input delta was already
sub-micro; the absolute output remained inside tolerance. This is strong
evidence of numerical fragility exposed by distributed execution, but it is
not yet a production fix: the experiment changes the dtype seen by the
downstream attention/FFN path and needs a separately scoped implementation and
regression plan.

The requested full-block deltas across the measured layers are:

| Layer | Configuration | Full-block `max_abs` |
|---:|---|---:|
| 0 | no compressor | `0.015625` |
| 2 | ratio 4 / Indexer | `0.015625` |
| 3 | ratio 128 / Compressor | `0.125` |
| 4 | ratio 4 | `0.0234375` |
| 5 | ratio 128 | `0.03125` |

The layer-3 chain is one amplification sequence, not three independent
defects: attention seeds a `0.03125` delta, the chained FFN reaches `0.121094`,
and the block ends at `0.125`. The isolated FFN remains at `0.00195312`, so it
does not independently diverge. A follow-up `--float32-sinkhorn` experiment
used the separate float32 Sinkhorn normalization path but cast the collapse
back to bfloat16. It did not change the layer-3 result: chained FFN
`0.121094`, `hc_ffn_post` `0.125`, and full block `0.125`. The `0.000487`
result therefore requires a wider float32 path than Sinkhorn normalization
alone; no production cast has been enabled and no live nonce probe was run
with an unvalidated fix.

### 2026-08-11 systemic drift and routing follow-up

The one-layer results must not be read as an end-to-end pass. The measured
full-block deltas at layers 0, 2, 4, and 5 (`0.015625`, `0.015625`,
`0.0234375`, and `0.03125`) are all nonzero against reference output scales
around `0.38` in the earlier layer runs. The tolerance of `0.05` is therefore
only a single-layer diagnostic bound; it is not evidence that 43 chained layers
preserve the prompt.

The independent same-input expert-selection probe compared sorted top-k index
sets with no floating-point tolerance. It found a layer-3 mismatch at token 3:
the unsharded set was `[12, 21, 24, 30, 69, 111]` and the sharded set was
`[12, 21, 24, 30, 69, 109]`; the other five experts were identical. This was
an isolated-layer diagnostic with a fresh synthetic input, not the end-to-end
depth finding, and is superseded below by the genuinely chained audit.

The new chained two-process harness feeds each output into the next layer and
reports the accumulated full-state delta:

| Chained depth | `max_abs` | Mean absolute output scale |
|---:|---:|---:|
| 1 | `0.015625` | `0.730007` |
| 2 | `0.015625` | `0.695943` |
| 4 | `0.03125` | `0.671867` |
| 8 | `0.046875` | `0.642554` |
| 9 | `0.226562` | `0.641240` |
| 16 | `0.90625` | `0.561247` |

This is systemic accumulated drift with a sharp increase after depth 8, not a
layer-3-only exception. The independent layer-3 mismatch above is superseded
as the end-to-end finding: in the chained BF16 run, layers 0-7 agreed and layer
8 was the first routing mismatch. That chained result is the one aligned with
the depth-8/9 knee and is the accepted localization for the baseline path.
At the time of this local-chain entry, the live nonce result had been treated as
conditioning failure. The later nonce completion shows the narrower and more
accurate result: prompt text is reaching the model, but instruction following
and answer quality are defective. This local chain has not established the
cause of that behavior.

### 2026-08-11 precision policy and full-depth routing audit

The broad float32 experiment kept the hidden state and hyper-connection path in
float32 on both local ranks (the quantized weights were not dequantized). It
removed the depth-8/9 knee and reduced the depth-16 delta from `0.90625` to
`0.00965488`:

| Chained depth | BF16 `max_abs` | Broad float32 `max_abs` |
|---:|---:|---:|
| 1 | `0.015625` | `0.00047946` |
| 2 | `0.015625` | `0.00189805` |
| 4 | `0.03125` | `0.00196719` |
| 8 | `0.046875` | `0.00396991` |
| 9 | `0.226562` | `0.00569558` |
| 16 | `0.90625` | `0.00965488` |

This is strong evidence for precision-sensitive systemic drift, but not exact
parity. A narrower experiment that cast only MoE gate inputs to float32 was
insufficient and was worse than baseline: depths 1, 2, 4, 8, and 16 reported
`0.015625`, `0.0234375`, `0.160156`, `0.270508`, and `1.01562`. Gate-only
casting is therefore not a validated production fix.

The diagnostic criterion used for this local audit was exact expert-set
agreement, not a per-layer floating-point tolerance: for a fixed input, the
sorted top-k expert index sets must match at every layer. That criterion is not
a live-quality acceptance gate for a 43-layer discrete-routing MoE; the chained
results below are characterization. The chained 43-layer BF16 audit passed at layers
0-7; layer 8 was the first mismatch (one token, token 0), and every layer 9-42
also mismatched. The final depth-43 state reached `max_abs=81`, with mean
absolute output `1.92695` against a reference scale of `3.93115`. This gives a
reproducible local mechanism for distributed path divergence: accumulated
distributed numerical drift eventually flips a near-tied MoE routing decision,
after which the residual streams diverge qualitatively. It remains local
characterization rather than a live two-node fix, and it is not established as
the cause of the live instruction-following failure.

The broad-float32 43-layer routing audit did not meet the then-proposed
zero-mismatch criterion. It kept the hidden state and hyper-connection path in
float32 on both local ranks while leaving quantized weights quantized. Layers
0-24 agreed; layer 25 was the first mismatch, and layers 26-42 also mismatched.
Depth 43 was `max_abs=10.1784`, mean absolute delta `0.351607`, reference scale
`3.20986`. This result is retained as numerical characterization, not as a
failed live-quality fix: exact zero routing mismatches at depth 43 is not a
realistic acceptance gate for a discrete top-k MoE because any nonzero path
difference can eventually cross a routing boundary. Broad float32 delayed the
first mismatch from layer 8 to layer 25 and greatly reduced drift through depth
16, but it cannot establish bit-identical execution.

The gate-only result is intentionally counter-intuitive: casting MoE gate
inputs to float32 made depth-16 drift worse (`1.01562` versus BF16 `0.90625`).
BF16 can mask near-tied logits by rounding both paths to the same value; the
float32 gate exposes a disagreement already present upstream and can make the
routing decision diverge sooner. This means precision belongs upstream, where
the hidden-state disagreement is introduced, rather than only at the discrete
decision point. Sinkhorn-only failed for the same structural reason: it did not
remove the upstream disagreement before the residual path.

The earlier `--float32-hyper` result (`max_abs=0.000487` at layer 3) should be
read as a precision-stability result: the wider float32 path bought enough
precision to keep a routing decision stable. It did not prove that
hyper-connections were independently defective, and it is not enabled in
production.

## 2026-08-11 absolute quality and broad-float32 live validation

The previous routing gate was removed before live validation. The question is
whether the deployed path produces correct output, not whether a distributed
execution path remains identical to a single-node reference through all 43
discrete-routing layers.

The live instance used two Tensor/MlxJaccl ranks, with
`EXO_DSV4_FLOAT32_BACKBONE=1`, quantized weights left quantized, and
`use_prefix_cache=false` for every request. The runners were healthy and Ready
before and after the measurements.

### Absolute factual probe

Request:

```json
{"model":"Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":1,"temperature":0,"enable_thinking":false,"use_prefix_cache":false,"logprobs":true,"top_logprobs":5,"stream":false}
```

The response had `cached_tokens=0`, `completion_tokens=1`, and
`finish_reason="length"`. The top-5 were:

```text
The   -0.03501701354980469
1     -3.539815902709961
 The  -6.872257232666016
用户  -6.969638824666016
**    -7.276315689086914
```

`Paris` was absent. The deployed broad-float32 two-rank path therefore fails
this absolute factual-quality probe; the result is not merely a
sharded-versus-unsharded disagreement.

### Cold nonce probe

Request:

```text
Return exactly this string and nothing else: WIRE-4C9E7A1B
```

The response had `cached_tokens=0`, `completion_tokens=249`, and
`finish_reason="stop"`. The complete completion was:

```text
好的，用户要求将“Return exactly this string and nothing else: WIRE-4C9E7A1B”翻译成中文。这是一个非常直接的指令，核心是翻译这个特定的字符串。

我需要先理解这个字符串的性质。“WIRE-4C9E7A1B”看起来像是一个代码、序列号或标识符，由字母和数字组成。对于这类技术性字符串，翻译时通常需要保持原样，因为它们是专有名称或代码，不能意译。

用户明确要求“Return exactly this string and nothing else”，这意味着我的回复必须极其简洁，只输出翻译结果，不能有任何解释、补充或格式变化。翻译策略就是：将指令部分“Return exactly this string and nothing else”翻译成中文，而将后面的字符串“WIRE-4C9E7A1B”原样保留。

所以，最终的中文翻译应该是“只返回这个字符串，不要其他任何内容：WIRE-4C9E7A1B”。这样既完成了翻译指令，又严格遵循了“只输出这个”的要求。
</think>只返回这个字符串，不要其他任何内容：WIRE-4C9E7A1B
```

The literal nonce appears, satisfying the narrow substring check, but only
inside unrelated translation/explanation text; the exact-output instruction
failed. This proves that prompt text reaches the model sufficiently for it to
quote and discuss the nonce. The live defect is instruction following and
answer quality, not proven loss of prompt information. The routing drift is
therefore a real numerical characterization, but it is not established as the
cause of this live behavior.

## 2026-08-11 offline tokenizer provenance audit

No cluster, EXO instance, model load, or numerical experiment was used for this
audit. The checkpoint's `config.json` declares `model_type="deepseek_v4"` and
`vocab_size=129280`. Its `tokenizer_config.json` declares
`tokenizer_class="PreTrainedTokenizerFast"` but has `chat_template=null` and
does not carry an `added_tokens_decoder` mapping. The authoritative marker IDs
in `tokenizer.json` are:

| Marker | Checkpoint `tokenizer.json` | Runtime tokenizer |
|---|---:|---:|
| `<｜begin▁of▁sentence｜>` | 0 | 0 |
| `<｜User｜>` | 128803 | 128803 |
| `<｜Assistant｜>` | 128804 | 128804 |
| `<think>` | 128821 | 128821 |
| `</think>` | 128822 | 128822 |
| `｜DSML｜` | 128825 | 128825 |

The runtime object is `mlx_lm.tokenizer_utils.TokenizerWrapper` wrapping
`transformers.tokenization_utils_tokenizers.TokenizersBackend`, with
`name_or_path` pointing at this checkpoint. It emits the known warning that
Transformers does not recognize the checkpoint config and that MLX-LM is using
its generic tokenizer fallback. That fallback still reads this checkpoint's
`tokenizer.json`; the six runtime IDs above match exactly. The fallback is real,
but this audit finds no special-token ID substitution.

The checkpoint has no tokenizer-supplied chat template, and the runtime
`chat_template` is also `null`. For a model ID containing `deepseek-v4`, EXO
selects its vendored `deepseek_v4_encoding.encode_messages` path before calling
the generic tokenizer template. For the factual probe, that encoder rendered:

```text
<｜begin▁of▁sentence｜><｜User｜>The capital of France is<｜Assistant｜></think>
```

This is therefore not evidence of a generic chat template silently replacing
the V4 role format. The offline result rules out the proposed wrong-special-ID
diagnosis and the embedding-vocabulary mismatch: the embedding and `lm_head`
both have shape `[129280, 1024]`, matching `config.json`'s `vocab_size`.
The remaining quality/instruction-following defect needs a separate prompt or
model-behavior investigation; numerics work is stopped for this branch.

### Context window measurements

Both requests completed with `cached_tokens=0` and left both runners Ready:

| Prompt tokens | Completion tokens | TTFT | Prefill | Decode |
|---:|---:|---:|---:|---:|
| 16,037 | 8 | `72.0219 s` | `222.668 tok/s` | `34.6579 tok/s` |
| 32,034 | 8 | `181.4775 s` | `176.518 tok/s` | `32.7032 tok/s` |

The 32K request completed successfully under broad float32. Available RAM
reported by the API was 34.98 GB local and 46.25 GB remote before the context
ladder, and 38.99 GB local and 50.29 GB remote after the 32K request. These are
availability samples, not peak allocation measurements; Metal unified-memory
RSS and page-reclamation samples are not reliable peak-model-memory evidence.
No 64K request was attempted. The supported ceiling remains 32K, with the
previous 64K watchdog-panic report retained as an operational hazard.

Live tool calls remain unvalidated: the templating blocker means there is still
zero live evidence for tool-call behavior.

## 2026-08-11 prompt-arrangement A/B and raw-token localization

The pinned environment identifies its reference as
'rltakashige/mlx-lm', branch 'leo/deepseek-v4', commit
'6a3df6cd6b00a347ee40f12d97a182aaf86ea599' (uv.lock). The installed reference
file is 'mlx_lm/chat_templates/deepseek_v4.py'; it was compared directly with
EXO's 'src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py'. For ordinary
chat, thinking mode, and leading-system-plus-user messages, the rendered
strings match byte-for-byte, including whitespace and newline placement. The
reference itself emits <｜Assistant｜></think> for chat mode and opens
<think> for thinking mode, so the pre-closed marker is not by itself evidence
of an EXO-only mistake. EXO's "max" reasoning text is the remaining known
offline difference.

The checkpoint has no own chat template, so this comparison is against the
pinned V4 reference implementation, not a checkpoint-declared template. The
official DeepSeek V3 documentation likewise uses apply_chat_template with a
generation prompt, while the DeepSeek R1 documentation recommends the
checkpoint tokenizer's chat-template path and discusses the reasoning
<think> prefix. Those references establish the general convention, but do not
prove that this V4 checkpoint's arrangement is correct.

The live A/B used the exact same user message, greedy decoding, max_tokens=64,
and use_prefix_cache=false on a two-rank Tensor/MlxJaccl instance. Every
response reported cached_tokens=0; none followed the exact instruction:

~~~
Return exactly this string and nothing else: WIRE-PROMPT-AUDIT-20260811
~~~

| Arrangement | Completion prefix / first ten generated pieces | Result |
|---|---|---|
| Current EXO format, enable_thinking=false | I need to respond to the user's query...; I, need, to, respond, to, the, user, 's, query, . | Failed; meta-analysis loop, finish_reason=length, 64 tokens |
| Current EXO format, enable_thinking=true | reasoning begins The user is asking me...; The, user, is, asking, me, to, complete, a, sentence, that; visible content was ' -' | Failed; 63 reasoning tokens, 1 visible token, finish_reason=length |
| System turn plus user, thinking off | . The user's query is in Chinese...; ., The, user, 's, query, is, in, Chinese, and, asks | Failed; unrelated Chinese instruction, finish_reason=length, 64 tokens |
| Opt-in EXO_DSV4_PROMPT_VARIANT=no_think | 你提供的这段文字看起来像是...; 你, 提供的, 这段, 文字, 看起来, 像是, 某种, 代码, 或, 加密 | Failed; unrelated Chinese explanation, finish_reason=length, 64 tokens |

The fourth arrangement was an opt-in diagnostic only; the default encoder is
unchanged. The A/B therefore does not identify a working chat arrangement, and
it does not support a claim that the model is ignoring the prompt wholesale.
That conclusion is limited to the nonce exact-echo instruction: the later
factual raw-token bisect shows that the same complete chat arrangement can
produce a correct, terminating answer.

### Raw-token control

Because all four arrangements failed, the branch adds a test-covered,
debug-only raw_input_ids field to Chat Completions and the internal task
parameters. With EXO_ENABLE_RAW_INPUT_IDS_DEBUG=1 set on every rank, it
bypasses chat rendering and both sequential and batch string-tokenization
paths; the supplied IDs are passed directly to prefill. It rejects an empty
ID list and is disabled by default.

The live two-rank raw continuation used the checkpoint tokenizer's IDs for
The capital of France is without BOS or role scaffolding:

~~~
raw_input_ids=[671, 6102, 294, 8760, 344]
~~~

The request reported prompt_tokens=5, cached_tokens=0, and returned:

~~~
Paris. The capital of France is Paris
~~~

This is the decisive localization in the current record: the checkpoint,
tokenizer IDs, distributed model, collectives, and decode path can produce the
correct factual continuation when the chat encoder is bypassed. The remaining
question is therefore whether the model handles particular prompt content and
task types reliably; this result does not establish a broken chat encoder. The
numerical-routing work remains characterization and is not the demonstrated
cause of the live instruction-following failure.

### Raw-token scaffolding bisect

To localize the transition, the same two-rank instance received six sequential
raw-token requests with greedy decoding, `max_tokens=16`,
`use_prefix_cache=false`, and `cached_tokens=0` for every request. This is
cache-cold evidence on one loaded instance, not six process restarts. The
checkpoint tokenizer supplied the token pieces shown below.

| Step | Raw IDs | First ten token-text pieces (checkpoint-tokenizer re-encoding) | Completion / metadata |
|---:|---|---|---|
| 1 control | `[671, 6102, 294, 8760, 344]` | `Paris`, `.`, ` The`, ` capital`, ` of`, ` Spain`, ` is`, ` Madrid`, `.`, ` The` | `Paris. The capital of Spain is Madrid. The capital of Italy is Rome.\`; 16 tokens, `finish_reason=length` |
| 2 BOS | `[0, 671, 6102, 294, 8760, 344]` | `Paris`, `.`, ` It`, ` is`, ` located`, ` in`, ` the`, ` north`, `-central`, ` part` | `Paris. It is located in the north-central part of the country, on the`; 16 tokens, `finish_reason=length` |
| 3 User | `[0, 128803, 671, 6102, 294, 8760, 344]` | `Paris`, `.`, ` The`, ` capital`, ` of`, ` Germany`, ` is`, ` Berlin`, `.`, ` The` | `Paris. The capital of Germany is Berlin. The capital of Italy is Rome.`; 16 tokens, `finish_reason=length` |
| 4 Assistant | `[0, 128803, 671, 6102, 294, 8760, 344, 128804]` | `The`, ` capital`, ` of`, ` France`, ` is`, ` **`, `Paris`, `**.` | `The capital of France is **Paris**.`; 9 tokens, `finish_reason=stop` |
| 5 open think | `[0, 128803, 671, 6102, 294, 8760, 344, 128804, 128821]` | `1`, `.`, ` `, ` **`, `Analy`, `ze`, ` the`, ` Request`, `**`, `:` | `1.  **Analyze the Request**:\n    *   The user's`; 16 tokens, `finish_reason=length` |
| 6 closed think | `[0, 128803, 671, 6102, 294, 8760, 344, 128804, 128821, 128822]` | `The`, ` capital`, ` of`, ` France`, ` is`, ` **`, `Paris`, `**.` | `The capital of France is **Paris**.`; 9 tokens, `finish_reason=stop` |

The first transition is the addition of `<｜Assistant｜>`: BOS alone and
`<｜User｜>` still produce document-style continuation, while the Assistant
marker produces a direct answer. Opening `<think>` deliberately changes the
mode to analysis, and appending `</think>` returns to the direct answer. Thus
the live evidence does not support the earlier claim that the complete
`<｜Assistant｜><think></think>` arrangement is intrinsically broken. It does
show that the role boundary is load-bearing, and that the no-role raw control
and the fully closed chat form are materially different prompt modes.

This bisect is stronger than conformance to the pinned third-party V4 fork:
the fork is a useful implementation reference, not a checkpoint-declared
authoritative template. The raw five-token continuation also retires the
numerical-drift causal hypothesis for this symptom: depth-43 sharded-versus-
unsharded divergence remains a real characterization, but it did not prevent
the deployed sharded path from producing a conditioned factual continuation.

## 2026-08-11 corrected probe interpretation and content discriminator

The earlier factual API probe was misdesigned and misinterpreted. It used
`max_tokens=1` and treated top-1 ` Paris` as the pass condition. A chat-formatted
response correctly begins with `The`, not ` Paris`, so the observed top-1 `The`
was not evidence of failure. The nine-token response from the raw bisect,
`The capital of France is **Paris**.`, is the corrected factual verdict and
terminates normally.

The exact arrangement that produced that factual answer was then held fixed
through `raw_input_ids`, while the content was changed to an exact-echo nonce
instruction. The prompt content was:

~~~
Return only this exact string: WIRE-PROMPT-AUDIT-20260811
~~~

The raw IDs were:

~~~
[0, 128803, 25529, 1353, 566, 6319, 3418, 28, 448, 34549, 6351,
 3674, 6806, 54, 6526, 12876, 2992, 15, 939, 24877, 779,
 128804, 128821, 128822]
~~~

With `max_tokens=64`, greedy decoding, `use_prefix_cache=false`, and
`cached_tokens=0`, the completion was:

~~~
We need to return only the exact string "WIRE-PROMPT-AUDIT-20260811". The user said "Return only this exact string: WIRE-PROMPT-AUDIT-20260811". So we just output that string. No extra text.</think>WIRE-PROMP
~~~

It included the nonce in the model's explanation but did not complete the
exact-echo task before the 64-token limit. Because the same raw chat scaffolding
answers the factual task correctly, this is evidence against the encoder,
tokenizer, distributed stack, and numerical-routing work as the cause of the
nonce behavior. The remaining observation is model/task-content behavior.

### Normal chat instruction battery

Four ordinary requests were sent through the normal chat API on the same
two-rank instance, with `enable_thinking=false`, greedy decoding,
`max_tokens=128`, `use_prefix_cache=false`, and `cached_tokens=0` each time:

| Request | Completion | Verdict |
|---|---|---|
| `What is 2+2?` | `2+2 equals 4.`; 8 tokens, stop | Pass |
| `List three primary colours.` | `1. Red\n2. Blue\n3. Yellow`; 12 tokens, stop | Pass |
| `Write a haiku about rain.` | 128-token planning/meta-analysis response, finish length | Fails to produce the requested poem within budget |
| `Summarise this in one sentence: The server received the request. The worker returned the response.` | `The server received the request, and the worker returned the response.`; 14 tokens, stop | Pass |

The evidence now supports a narrower conclusion: the two-rank distributed path
and chat arrangement produce correct factual and ordinary instructional output,
while this checkpoint is unreliable on exact-echo/random-string instructions
and at least some creative-writing requests. The validation record must not
call that an inference-stack failure. Tool calls remain unvalidated live, and
the 32K ceiling/64K watchdog hazard remain unchanged.
