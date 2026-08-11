# Task 6 native-agent brief: prove DeepSeek-V4-Flash-0731 API semantics on the live cluster

**Audience:** an agent with a real macOS shell on Jared's M4 Max or M5 Max, with
`uv`, MLX and both cluster nodes reachable. Everything here was prepared in a
Linux container that cannot reach MLX or the nodes, so Steps 4-8 have never been
executed. Treat every claim marked PREPARED as untested on hardware.

**Read first:** `RULES.md` and `CLAUDE.md` (= `AGENTS.md`) at the repo root. They
override this brief. In particular: `basedpyright && ruff check && nix fmt &&
pytest` before every commit, `bugfix|feature|documentation|refactor|chore|test`
commit prefixes with a capitalised subject, and **if you find code that violates
the rules, raise it with Jared rather than fixing it**.

**Use the superpowers skills.** `superpowers:executing-plans` for the task
sequence, `superpowers:test-driven-development` for Steps 1-2,
`superpowers:systematic-debugging` for any failure, and
`superpowers:verification-before-completion` before any success claim.

**2026-08-10 operational safeguard.** If either node appears to fail at exec or
to hang immediately, inspect the size and mtime of the selected Python
interpreter before diagnosing MLX: the shared uv `cpython-3.13.11` binary on
node 1 was replaced by a 112-byte shell shim, breaking every project using that
interpreter and mimicking an import/Metal hang. The writer is unattributed;
Jared must be notified of the machine-wide incident.

**2026-08-10 ring note.** One enhanced cross-device SDPA launch stalled with
rank 0 blocked in `TCPSocket::accept`; a committed-script control and the next
enhanced run both formed a symmetric ring. Treat ring formation as intermittent,
not fixed. If the EXO topology is asymmetric, stop and record the exact error;
do not apply the stashed placement workaround or demote Thunderbolt.

**2026-08-10 native/topology qualification.** Rebuilding node 2's stale
`exo_rs` extension restored the missing namespace implementation and resolved
one observed split-control-plane incident. It did **not** close the broader
connection problem: `SocketConnection` asymmetry recurred after both hosts had
rebuilt from the same source. Treat fresh topology formation as an independent,
intermittent gate, not as proof that native-artifact reconciliation is complete.

---

## 0. Where things stand

Branch `codex/dsv4-0731-backbone`, worktree
`exo/.worktrees/codex-dsv4-0731-backbone`. Plan 2 Tasks 1-5 are committed:

| Task | Commit | Subject |
|---|---|---|
| 1 | `e19eebae` | preserve DeepSeek V4 reminder prefixes |
| 2 | `b2822771` | Align DeepSeek V4 reasoning effort |
| 3 | `4d4409e7` | Preserve DeepSeek V4 prefill channel |
| 4 | `881a5e0d` | Tolerate DeepSeek V4 DSML variants |
| 5 | `3c010d37` | Parse split DeepSeek V4 delimiters |

**All five landed without their native gates**, at Jared's explicit instruction,
because the authoring session had no macOS shell. This is recorded in each
commit body, in each `VERIFICATION OUTSTANDING` note in
`docs/superpowers/plans/2026-08-04-dsv4-0731-prompt-api.md`, and in
`.superpowers/sdd/2026-08-04-dsv4-0731-prompt-api/progress.md`.

**Your first job is therefore not Task 6 — it is discharging that debt.** Run
Step 1 below before writing any new code. If it fails, stop and report; do not
patch forward on top of a suspect commit.

## 1. Discharge the outstanding verification (do this first)

```bash
cd /Users/jared/aishit/exo-install/exo/.worktrees/codex-dsv4-0731-backbone
uv run basedpyright
uv run ruff check
nix fmt
uv run pytest -q
```

Then the golden-regeneration check, which is the single highest-risk item in the
whole branch. Task 1's committed goldens had drifted from their own generator and
were repaired by full regeneration; that repair changed a committed acceptance
artifact and has never been independently confirmed:

```bash
uv run python scripts/generate_dsv4_0731_prompt_goldens.py \
  --model-path /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp
git diff --stat src/exo/worker/tests/fixtures/deepseek_v4_0731_prompt_goldens.json
```

**The diff must be empty.** A non-empty diff means the Task 1 fixture repair was
wrong and Tasks 2-5 rest on a bad acceptance target. Stop and report.

Also run the Plan 1 BF16/q4 tensor-parallel parity matrix. Tasks 1-5 touch
neither the loader, the compatibility model, nor sharding, so the risk is low,
but it is unproven.

Record every exit code and test count — they go in the validation document.

## 2. Tokenizer pre-flight: expect a warning, do not fix it

On startup you will see, from `mlx_lm/tokenizer_utils.py`:

```
RuntimeWarning: Falling back to a generic tokenizer because Transformers does
not recognize this model config yet: 'PreTrainedConfig' object has no attribute
'max_position_embeddings'
```

This is benign and **must not be "fixed"**. Diagnosed in full offline:

* `config.json` declares `model_type: "deepseek_v4"`, unknown to the pinned
  transformers 5.6.2, so `AutoConfig` yields a bare `PreTrainedConfig`.
* `config.json` also contains `rope_scaling`, so transformers runs
  `standardize_rope_params()`, which at `modeling_rope_utils.py:758` reads
  `self.max_position_embeddings` — an attribute only concrete architecture
  configs define. Hence the `AttributeError`.
* Isolated by single-variable test: dropping **only** `rope_scaling` loads
  cleanly; dropping `quantization` still fails; dropping `config.json` entirely
  loads cleanly.
* `mlx_lm.tokenizer_utils.load` catches `(AttributeError, ValueError)` and
  retries with a `PretrainedConfig` stub built from six config keys. The
  tokenizer itself is still built from the checkpoint's own `tokenizer.json`.

Verified equivalent offline against the real checkpoint files:

* vocab mapping identical, 129,280 entries
* **all 16 committed golden `expected_token_ids` arrays reproduce exactly** —
  an independent cross-check, since the goldens were generated through OMLX
* detokenization round-trips exactly on the encoder's control-token prompts
* BOS/EOS resolve to 0/1, agreeing with `config.json` and `generation_config.json`
* the goldens' recorded `tokenizer_config_sha256`
  (`6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`) matches
  the checkpoint's `tokenizer_config.json`
* `<think>` = 128821 and `</think>` = 128822 are real single tokens, so
  `_infer_thinking` returns them, `has_thinking` is True, and
  `parse_thinking_models` receives correct markers
* `detect_thinking_prompt_suffix` gives the right `starts_in_thinking` for all
  16 goldens, including chat mode (False) and `trailing_assistant_prefill` (False)

Do **not** edit `config.json` to silence this. That would mutate the checkpoint
and invalidate every golden. If Jared wants the warning gone, raise it as a
proposal rather than acting.

### The one real consequence, and why it is harmless

The checkpoint has no `chat_template` — no key in `tokenizer_config.json` and no
`chat_template.jinja`. So `_infer_tool_parser(None)` returns `None`, the
`TokenizerWrapper` gets no tool parser, and startup logs
`has_tool_calling=False`. **This does not affect V4 tool calls.**
`apply_all_parsers` (`model_output_parsers.py:96`) dispatches on
`issubclass(model_type, DeepseekV4Model) and "deepseek-v4" in normalized_id` to
`parse_deepseek_v4`; the `tool_parser` argument is consulted only in the generic
`else` branch. Expect that log line and ignore it.

### If completions come out as garbage

The ledger previously claimed the generic tokenizer made "generated text not
semantically correct". That inference was wrong — the tokenizer is proven
equivalent above. If you nonetheless see semantically broken output at Step 5
case 1, the tokenizer is **eliminated** and the remaining suspects are the
quantization recipe and the sharding. Escalate to
`superpowers:systematic-debugging`, and note the recorded facts: the global
quantization fallback is correct for **zero** modules, `mode` mismatches are
invisible to strict loading (hence the mandatory post-quantize audit), and all
five fused mxfp4 attention fast paths fall back under this recipe.

## 3. Then execute plan Steps 1-9

Follow `docs/superpowers/plans/2026-08-04-dsv4-0731-prompt-api.md` Task 6 as
written. Notes that will save you time:

**Steps 1-2 (offline API tests).** Compose `apply_all_parsers()` ahead of the
adapters in `test_deepseek_v4_api_semantics.py`, per the plan, so the test covers
parser-to-wire composition rather than hand-built final chunks. Two hard-won
lessons from Tasks 1-5, both of which cost a review round:

* **Mutation-test your tests.** Break the implementation deliberately, one change
  at a time, and confirm a test fails. In Task 5 this caught two tests that did
  not exercise the guard they claimed to.
* **Also run new tests against the pristine pre-change code.** A test that stays
  green there is a contract guard, not a regression guard — label it honestly. In
  Task 5, 8 of 14 new tests were green against the unfixed parser while their
  comments claimed to catch the bug.

**Step 4 (start the instance).** Confirm the active instance is exactly two-rank
`Tensor`/`MlxJaccl` and that the post-quantize audit passes *before* any API
call. Use explicit Thunderbolt bootstrap peers — cross-host multicast does not
work between these two Macs. The M5's macOS firewall may still be disabled from
Plan 1; check it.

**Multi-host build-artifact reconciliation (required).** Matching Git HEAD and
Python source paths is insufficient when the environment contains a native
extension. Rebuild every local native package from the checked-out source on
every host, then verify the runtime behaviour that carries the contract. For
the Zenoh namespace contract, both `exo_rs` binaries must contain `namespaces/`
and the running peers must declare the same namespace-prefixed EXO topics.
Record the MLX and `mlx_lm` versions as well. Do not require native-binary
hashes to match: Rust builds embed host-specific paths and toolchain details.

```bash
EXO_ZENOH_NAMESPACE=dsv4f-0731-validation \
EXO_DEFAULT_MODELS_DIR=/Users/jared/.cache/huggingface/hub/models \
uv run exo
```

**Context ceiling is a live measurement, not an assumption.** Plan 1 recorded
a 16K ceiling and a 32K MLX-prefill SIGSEGV. After rebuilding `exo_rs` on both
hosts on 2026-08-10, one 32,773-prompt-token request completed successfully;
this removes the old crash as a current blocker but is not by itself a new
accepted ceiling. Record the exact prompt-token count, cache state, response,
and post-request runner health for every future 16K/32K probe; never silently
substitute a smaller context after a failure.

**Step 5-6 (the 9 × 2 endpoints × 2 stream modes = 36 combinations).** For each,
assert: structured tool calls rather than text, correct `finish_reason`,
reasoning and content in separate channels, **no DSML markup in either channel**,
and exactly one terminal event. Compare the two endpoints on normalised logical
content — wire envelopes legitimately differ, logical results must not.

Watch these specifically, because they are the shapes the recent commits changed
and the ones most likely to expose a defect:

* Case 9 (trailing assistant prefill) — the prefill must land after `</think>`
  and before EOS, never inside the reasoning channel. This was a real bug fixed
  in Task 3 (`4d4409e7`).
* Cases 5-7 (tool calls) — `｜DSML｜` is a single token (128825), so
  `<｜DSML｜tool_calls>` arrives as six tokens: `'<'`, `'｜DSML｜'`, `'tool'`,
  `'_c'`, `'alls'`, `'>'`. Split delimiters are the **only** way these markers
  ever arrive, so Task 5 (`3c010d37`) is directly load-bearing here. A marker
  fragment appearing in visible content is a Task 5 regression.
* Case 8 (mid-conversation reminder) — must render as `<｜latest_reminder｜>`
  (token 128828) and must preserve the append-only prefix invariant.
* Cases 3-4 (high / xhigh effort) — must produce *different* prefixes. `high`
  and `xhigh` map to distinct V4 tiers; collapsing them is a Task 2 regression.

**Step 7 (prefix caching).** Prove the second rendered prompt begins with the
first through the historical turn, by prompt hash or cache diagnostics. Do not
claim cache reuse from timing alone — the plan is explicit about this.

## 4. Known limitations — pinned deliberately, do not "fix" mid-matrix

If you hit one of these, it is expected behaviour with a test pinning it. Record
it and move on; raise with Jared if you think the trade-off is wrong.

* Text following `tool_calls_end` **inside the same accumulated block** is
  dropped by the DSML body parser. Text in a later response is delivered
  normally. Pinned by `test_text_after_the_end_marker_is_dropped`.
* A never-closed tool-call block is emitted verbatim on the terminal response
  rather than suppressed. Pinned by `test_a_truncated_block_is_emitted_as_text`.
* Any content carried by the terminal response is reported `is_thinking=False`,
  even reasoning that began earlier — a terminal response is contractually not
  thinking.
* A trailing assistant turn carrying `tool_calls` is dropped with only its text
  re-appended, losing the calls. Rendering it needs an anchor the encoder does
  not emit after an assistant turn.
* `_PARAM_PATTERN` is attribute-order sensitive: `string="true" name="city"`
  yields a tool call with arguments silently dropped. Returning `None` on zero
  parameters is **not** the fix — a legitimate zero-argument call produces
  byte-identical output, so that would break every no-arg tool.
* A zero-length response lying inside a marker's character span loses its
  logprobs entry, in both parsers.

## 5. Deliverables

1. `src/exo/api/tests/test_deepseek_v4_api_semantics.py` plus the additions to
   `test_chat_completions_stream.py` and `test_openai_responses_api.py`.
2. `docs/superpowers/validation/2026-08-04-dsv4-0731-prompt-api.md` containing
   everything plan Step 9 lists: local and remote EXO SHAs, active instance ID,
   the OMLX golden SHA (`50846648273591a621bb96cb1b3956c45d43efed`) and the
   checkpoint tokenizer hash, local test commands and counts, the full 36-row
   table, prefix-cache evidence, dashboard observations, and for any failure its
   regression test and corrective commit.
3. An update to `.superpowers/sdd/2026-08-04-dsv4-0731-prompt-api/progress.md`
   recording the Step 1 outcome, which is what clears the
   `VERIFICATION OUTSTANDING` notes on Tasks 1-5.
4. Commit per plan Step 10, but as `test: Validate DeepSeek V4 0731 API
   semantics` — RULES.md capitalisation, which supersedes the lowercase form
   written in the plan.

## 6. Open decision for Jared, do not settle it yourself

The branch mixes commit conventions. `e19eebae` and codex's earlier commits use
lowercase conventional-commits; `b2822771`, `4d4409e7`, `881a5e0d` and
`3c010d37` follow RULES.md. Everything is unpushed and therefore amendable. Ask
before normalising.

## 7. Housekeeping

The authoring sessions ran through a mount that forbids `unlink`, so git
stranded lock and temp files that had to be moved aside rather than deleted:

```bash
cd /Users/jared/aishit/exo-install/exo
rm -rf .git/_stale_locks                                    # ~95 moved lock/tmp_obj files
rm -rf .worktrees/codex-dsv4-0731-backbone/_to_delete       # scratch dirs
rm -rf tok_stage                                            # tokenizer files staged for the offline diagnosis
git gc
```

`resources/inference_model_cards/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp.toml`
is deliberately untracked and is needed for the live run — leave it. `git fsck`
reports one harmless dangling blob.

## 2026-08-11 acceptance continuation

The current context-management limit is 32K. Cold streamed measurements on the
two-rank `Tensor`/`MlxJaccl` instance completed at 16,419 and 32,804 actual
prompt tokens. A 65,536-token attempt rebooted the local node before
completion; the preserved panic report classifies it as a kernel watchdog panic
after 94 seconds without `watchdogd` check-ins, with no OOM or GPU-fault
evidence. The run was not repeated. Do not run a live prompt above 32K until
context management is improved. A follow-up should evaluate OMLX TurboQuant
for reducing context-state resource use.

Tool-call acceptance is blocked before inference: the offline probe found that
the user-only DeepSeek V4 render path drops the supplied tool schema. The
rendered prompt therefore contains no tool name, so live cases 5-7 were not
run or reported as parser failures. Throughput and cache evidence are recorded
in `2026-08-06-dsv4-0731-live-validation-gaps.md`.

## Evidence status correction

The earlier one-token Task 5 completion `' word'` was not sufficient to
establish conditioning, and the earlier `'RO'` result was an 8-token truncated
probe. The replacement cold probe used fresh instance
`08e66087-b50d-4b35-a19a-109299f1f405` and exact prompt
`Return exactly this string and nothing else: WIRE-7F3C9A2D` with
`max_tokens=256`, `temperature=0`, and `enable_thinking=false`. It reported
`cached_tokens=0`, `finish_reason=length`, and `completion_tokens=256`; the
complete completion was an unrelated logarithm derivation ending at the
truncated text `x+3 > 0 \\Rightarrow`, and it did not contain the nonce. Task 5
therefore failed the conditioning pass condition. The full quoted completion
is retained in the companion validation report.

The full source gate was run on the real Metal host with
`PYTHONPATH=tools/src .venv/bin/pytest src -q`: 734 passed, 3 skipped, and 172
deselected. The local headless checkout remains unable to initialize Metal.

The 64K event is now classified by the preserved local panic report as a
kernel watchdog panic (`watchdogd` missed check-ins for 94 seconds), not an
observed OOM or GPU fault. The other node did not reboot. Reproducibility was
not tested because the live safety ceiling remains 32K; record **≤32K accepted,
64K hazardous, one confirmed watchdog panic**.

32K is a hard supported prompt ceiling for this run. Decision for Jared:
consider enforcing an API prompt-length cap at 32K, or slightly below it for
operational margin; this task does not implement that policy. Revisit OMLX
TurboQuant context-state quantization/compression before reconsidering 64K.

No live per-interface exception was captured for the original node-2 edge gap.
The Python resilience changes are defensible hardening and are covered by
red-green tests and deployed, but their causal role is **not proven** because
topology became symmetric only after native `exo_rs` rebuild/reconciliation;
no per-interface exception was captured, and the triggering interface class is
unknown. The defect may recur.

Tool calls remain **unvalidated live**: the offline user-only render drops the
tool schema, so cases 5-7 have zero live evidence. OMLX TurboQuant is noted as
future context-state quantization/compression work because a lower-resource
long-prefill path is needed before reconsidering 64K; it is not part of this
patch or a validated remedy.

## 2026-08-11 Task 7 and Task 8 independent results

Task 7 ran regardless of the failed Task 5 branch with two local MLX processes:

| Layer | Attention | FFN | Full block | Verdict |
|---|---:|---:|---:|---|
| 0 (no compressor) | PASS `0.03125` | PASS `0.00195312` | PASS `0.015625` | PASS |
| 2 (Indexer) | PASS `0.03125` | PASS `0.00390625` | PASS `0.015625` | PASS |
| 3 (Compressor) | PASS `0.03125` | PASS `0.00195312` | **FAIL `0.125`** | FAIL |

The tolerance was `max_abs <= 0.05`; all rank-agreement checks passed. Layer 3
fails first at the full block, while attention and FFN pass, implicating the
hyper-connection wrapping rather than the Indexer, Compressor, or sharding
collectives.

Task 8 added
`src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_mask.py`. The
synthetic `SEQ_LEN=7` test passed on real Metal (`1 passed`), and its attention
spy observed query length 7 rather than 28. This is unit evidence only; live
tool calls remain unvalidated.

## 2026-08-11 hyper-connection follow-up

The ratio prediction was not confirmed: layer 4 (ratio 4) passed with full
block `max_abs=0.0234375`, but layer 5 (ratio 128) also passed with
`max_abs=0.03125`. Do not generalize the layer-3 failure to every ratio-128
layer.

The parity harness now reports `hc_pre` and `hc_post` input/output deltas and
amplification. Layer 3's default path showed: attention delta `0.03125`;
`hc_attn_post` reduced it to `0.015625` (`0.5x`); `hc_ffn_pre` produced
`0.0234375` (`1.5x`); isolated FFN remained at `0.00195312`, while the chained
FFN produced `0.121094`; and `hc_ffn_post` ended at `0.125` (`1.03x`).
`hc_attn_pre` itself was identical. The first mismatch is distributed attention
and the residual path compounds it before the chained FFN; the isolated FFN
does not independently diverge.

The experimental `--float32-hyper` harness mode reduced layer-3 attention delta
to `9.5e-7`, chained FFN delta to `5.7e-7`, and full-block delta to `0.000487`
(PASS). This supports a distributed numerical-fragility hypothesis, but it is
not a production fix: the experiment changes the dtype entering downstream
attention/FFN and needs its own implementation and regression plan.

The full-block `max_abs` values recorded across the measured layers are:

| Layer | Configuration | Full-block `max_abs` |
|---:|---|---:|
| 0 | no compressor | `0.015625` |
| 2 | ratio 4 / Indexer | `0.015625` |
| 3 | ratio 128 / Compressor | `0.125` |
| 4 | ratio 4 | `0.0234375` |
| 5 | ratio 128 | `0.03125` |

Layer 3 is one amplification chain: attention seeds `0.03125`, the chained
FFN reaches `0.121094`, and the block reaches `0.125`; the isolated FFN is
`0.00195312`. A sinkhorn-only experiment (`--float32-sinkhorn`) used the
separate float32 Sinkhorn path but cast its collapse back to bfloat16. It did
not improve layer 3: chained FFN `0.121094`, `hc_ffn_post` `0.125`, full block
`0.125`. The earlier `0.000487` full-path result therefore cannot be attributed
to Sinkhorn normalization alone, and no production change or live acceptance
probe has been run for it.

## 2026-08-11 systemic drift and routing follow-up

The one-layer tolerance is not an end-to-end acceptance criterion. Full-block
deltas at layers 0, 2, 4, and 5 were all nonzero (`0.015625`, `0.015625`,
`0.0234375`, and `0.03125`), so per-layer PASS only bounds local error.

The independent same-input expert-selection mode compares sorted top-k index
sets without a numeric tolerance. It found a layer-3 mismatch at token 3:
unsharded `[12, 21, 24, 30, 69, 111]`; sharded
`[12, 21, 24, 30, 69, 109]`. This is an isolated-layer diagnostic with a
fresh synthetic input, not the end-to-end depth finding, and is superseded by
the chained audit below.

The chained two-process run produced this accumulated full-state drift:

| Chained depth | `max_abs` | Mean absolute output scale |
|---:|---:|---:|
| 1 | `0.015625` | `0.730007` |
| 2 | `0.015625` | `0.695943` |
| 4 | `0.03125` | `0.671867` |
| 8 | `0.046875` | `0.642554` |
| 9 | `0.226562` | `0.641240` |
| 16 | `0.90625` | `0.561247` |

This supports systemic numerical drift with a depth-8/9 discontinuity, rather
than a layer-3-only defect. The isolated layer-3 mismatch is superseded as the
end-to-end finding: the chained BF16 run agrees through layers 0-7 and first
mismatches at layer 8, aligned with the depth-8/9 knee. That chained result is
the accepted baseline localization, not the isolated-layer result.

## 2026-08-11 precision policy and full-depth routing audit

The broad float32 experiment kept the hidden state and hyper-connection path in
float32 on both local ranks while leaving the quantized weights quantized. It
removed the depth-8/9 knee and reduced depth-16 `max_abs` from `0.90625` to
`0.00965488`:

| Depth | BF16 | Broad float32 |
|---:|---:|---:|
| 1 | `0.015625` | `0.00047946` |
| 2 | `0.015625` | `0.00189805` |
| 4 | `0.03125` | `0.00196719` |
| 8 | `0.046875` | `0.00396991` |
| 9 | `0.226562` | `0.00569558` |
| 16 | `0.90625` | `0.00965488` |

This supports a precision-sensitive systemic drift hypothesis but does not
establish exact parity. Casting only MoE gate inputs to float32 was not
sufficient: depths 1, 2, 4, 8, and 16 were `0.015625`, `0.0234375`,
`0.160156`, `0.270508`, and `1.01562`, respectively, so gate-only casting is
not a validated production fix.

The diagnostic criterion for this local audit was exact sorted top-k expert-set
agreement for a fixed input at every layer; the old single-layer float tolerance
is diagnostic only. Exact zero routing agreement is not a live-quality gate for
a 43-layer discrete-routing MoE, because any nonzero distributed difference can
eventually cross a routing boundary. In the chained 43-layer BF16 audit, layers 0-7 agreed exactly, layer 8 was the
first mismatch (one token, token 0), and layers 9-42 all mismatched. At depth
43, `max_abs=81`, mean absolute output was `1.92695`, and the reference scale
was `3.93115`. The local evidence therefore supports this mechanism for
distributed path divergence: small numerical drift accumulates until a
near-tied MoE top-k decision flips, after which the residual streams diverge
qualitatively. This is reproducible local characterization, not a live two-node
fix and not a proven explanation for the live instruction-following failure.

The earlier `--float32-hyper` layer-3 result (`max_abs=0.000487`) is a
precision-stability result, not evidence of an independently broken
hyper-connection. The wider float32 path kept that routing decision stable; no
production cast is enabled.

The broad-float32 43-layer routing audit did not satisfy the then-proposed
zero-mismatch criterion. Layers 0-24 agreed exactly; layer 25 was the first
mismatch and layers 26-42 also mismatched. At depth 43 the broad-float32
`max_abs` was `10.1784`, mean absolute delta `0.351607`, and reference scale
`3.20986`. It removes the early knee and substantially reduces drift through
depth 16. Exact zero routing mismatches at depth 43 is not a valid live-quality
gate for a discrete top-k MoE: any nonzero distributed numerical difference can
eventually cross a routing boundary. This is characterization of the system's
numerical behavior, not a failed fix, and the live evidence is recorded in the
absolute-quality addendum in the companion validation document.

The gate-only result is counter-intuitive but informative: casting MoE gate
inputs to float32 worsened depth-16 drift to `1.01562` versus BF16 `0.90625`.
BF16 can mask near-tied logits by rounding both paths to the same value; the
float32 gate exposes an upstream disagreement and can make routing diverge
sooner. Precision therefore needs to be applied upstream, where the hidden
state disagreement is introduced, not only at the discrete decision point.
Sinkhorn-only failed for the same structural reason because it did not remove
the upstream disagreement before the residual path.

The live broad-float32 run used the exact two-rank Tensor/MlxJaccl instance with
`use_prefix_cache=false`. The factual probe (`The capital of France is`, one
token, top-5 logprobs) returned `The` as top-1 with `cached_tokens=0`,
`finish_reason="length"`, and no `Paris` in the top-5. The cold nonce probe
returned 249 tokens with `finish_reason="stop"` and `cached_tokens=0`; the
nonce appeared only inside unrelated translation/explanation text rather than
as the exact requested output. Broad float32 therefore did not restore live
instruction following: the nonce was present, but only in unrelated
translation/explanation text. The 16K and 32K requests both completed; the 32K request
was the highest tested, and no 64K request was attempted. See the companion
validation document for the complete prompt, completion, throughput, and RAM
availability evidence.
