# Handoff — diagnose incoherent generation on the DeepSeek V4 0731 cluster

Task 6 is **blocked, not failed**. The re-validation was honest and useful: it
established that the prompt path is correct and that generation is not.

Paste the block below into a fresh session on the M4 or M5 with the cluster up.

```
Invoke superpowers:systematic-debugging and follow it. This is a diagnosis, not a
fix task. Do not propose or apply a fix until Phase 1 identifies a root cause.
You will be tempted to skip ahead because the symptom is dramatic; don't.

  cd /Users/jared/aishit/exo-install/exo/.worktrees/codex-dsv4-0731-backbone
  git log --oneline -3

Read docs/superpowers/validation/2026-08-06-dsv4-0731-live-validation-gaps.md
and the live sections of the validation record. Then RULES.md and CLAUDE.md.

THE SYMPTOM. At temperature 0.0 the model returns incoherent multilingual text
and never emits EOS. Observed content for "Say hello in one sentence." with
thinking disabled: 沙发 ("sofa"). Elsewhere: 强弱, 循环, ссе, 'rounding',
'visualization', 'upp'. finish_reason was `length` on all nine non-streaming
Chat cases, including thinking-off at max_tokens=256 — a working model says
"Hello!" in about five tokens.

WHAT IS ALREADY RULED OUT. Do not re-investigate these:

* The tokenizer. All 16 committed golden token-id arrays reproduce exactly under
  the tokenizer EXO actually builds, the vocab mapping is identical across
  129,280 entries, detokenization round-trips exactly, BOS/EOS resolve to 0/1,
  and the goldens' tokenizer_config hash matches the checkpoint. The startup
  "Falling back to a generic tokenizer" RuntimeWarning is benign and diagnosed
  in the runbook §2.
* Prompt construction. The goldens match, effort tiers differentiate correctly
  (prompt prefix 14 -> 93 -> 106 tokens for low/high/xhigh), and the prefix cache
  reused 284 of 286 tokens with the negative control at exactly 0. The prompt
  reaching the model is right.
* The sampler. mlx_lm's make_sampler documents "if 0 the argmax is used";
  resolve() at src/exo/shared/types/text_generation.py:150 preserves 0.0
  correctly because it tests `is not None`; batch_generate.py:186 passes it
  through; and mx.random.seed(42) is pinned. Generation is greedy and should be
  deterministic. "Non-deterministic sampling" is not available as an explanation.

STEP 1 — the bisection. Run this first; it is one request repeated and takes
seconds:

  uv run python scripts/validate_dsv4_live_api.py --determinism 3

Greedy decoding with a pinned seed MUST produce identical completions. The result
splits the investigation and you should not proceed until you know which branch
you are in:

  * COMPLETIONS DIFFER -> the forward pass is non-deterministic. Go to Step 2A.
    Sampling cannot be blamed. Suspect reduction order or a race in the
    distributed all-reduce. Note that test_tp_bit_exact.py passing does not rule
    this out: a race can stay hidden on short synthetic inputs and appear under
    real generation length.
  * COMPLETIONS IDENTICAL BUT INCOHERENT -> the weights or the quantization are
    wrong. Go to Step 2B. Do not spend further cluster runs on the API layer.

STEP 2A — non-deterministic forward pass.

  a. Run the same prompt on a SINGLE rank (world size 1, no JACCL). If single
     rank is deterministic and two ranks are not, the fault is in the distributed
     path, and that is a Plan 1 defect rather than anything in Plan 2.
  b. Capture the first generated token id for a fixed prompt across several runs.
     Comparing generated strings is weak — one divergent token cascades. The
     first token is the sharp signal.
  c. Inspect the JACCL all-reduce for order-dependent accumulation, and any place
     the reduction is split across the Thunderbolt link.

STEP 2B — wrong weights or quantization.

  a. Confirm the post-quantize audit actually ran and passed on this instance.
     The 2026-08-06 report does not mention it. It is mandatory because `mode`
     mismatches are invisible to strict loading and the global quantization
     fallback is correct for ZERO modules in this checkpoint.
  b. THE DECISIVE EXPERIMENT: run the same checkpoint through OMLX, which is the
     read-only reference implementation and runs single-machine. Same prompt,
     greedy, same max tokens.
       * OMLX coherent, EXO incoherent -> the fault is in EXO's loader or
         quantization path, not the checkpoint. This is the most likely outcome
         and the most actionable.
       * Both incoherent -> the checkpoint itself is suspect. Verify it against
         the published hashes before anything else.
  c. Compare the FIRST-TOKEN logits rather than generated text. EXO's API
     supports logprobs and top_logprobs, so request top-5 logprobs for the first
     generated token on a fixed prompt, and get the same from OMLX. If the top-1
     differs, the forward pass is wrong and you can bisect by layer. This is far
     stronger evidence than comparing strings, and it is how you find WHICH layer
     rather than just THAT something is wrong.
  d. If you get to layer bisection, these are recorded facts worth having in
     hand: hyper-connections are dotted attn_hc/ffn_hc; attn.wo_a is 3D;
     replicated hyper-connections are CORRECT and must not be "fixed"; all five
     fused mxfp4 attention fast paths fall back under this recipe. The sanitize()
     in the pinned fork rltakashige/mlx-lm branch leo/deepseek-v4 does most of
     the weight transformation, so it is a prime suspect.

STEP 3 — when you have a root cause, and not before.

Write a failing test first, at the smallest scope that reproduces it — ideally a
unit test over one layer or one weight transformation rather than an end-to-end
generation. Then one fix. Then re-run:

  uv run python scripts/validate_dsv4_live_api.py --determinism 3
  uv run python scripts/validate_dsv4_live_api.py --only 5,6,7
  uv run python scripts/validate_dsv4_live_api.py

If three fixes fail, STOP and question the architecture with Jared rather than
attempting a fourth. That is systematic-debugging Phase 4.5 and it exists for
exactly this shape of problem.

DO NOT:

* Do not raise max_tokens to make cases pass. Truncation is a symptom of never
  emitting EOS, not the cause of anything.
* Do not edit scripts/validate_dsv4_live_api.py to soften a check. Run
  `--self-test` if you doubt it; it proves every check can fail.
* Do not attempt further Task 6 API validation until generation is coherent. A
  model emitting soup will never produce a well-formed DSML block, which is why
  tool calls have never fired and why Tasks 4 and 5 remain unvalidated live.
* Do not fix the model-card `backends` defect, normalise commit conventions, or
  move the harness into basedpyright's include. All raised for Jared.

REPORT: the Step 1 branch and its evidence, what Step 2 established, the OMLX
comparison result, and the root cause with the failing test that pins it. If you
run out of session before a root cause, report the bisection state — which
branch, what is eliminated — so the next session resumes rather than restarts.
```

## Context you may want before handing this over

**This symptom predates Plan 2.** The project notes carried "generated text is
not yet semantically correct" from the Plan 1 era, attributed to the tokenizer.
That attribution was wrong, but the observation was real — so this has been
present since the backbone work and was never a Plan 2 regression. Worth
considering whether Plan 1's completion criteria ever asserted output coherence,
or only bit-exactness and throughput. "Two-node JACCL validated" and "16K ceiling
proven" are both consistent with a cluster that runs at speed and produces
nonsense.

**Harness changes since the last run** (`--self-test` still passes):

* `--determinism N` implements Step 1 directly, and also checks two proxies that
  the last run violated: that EOS is ever emitted, and that a reply to an English
  prompt is not predominantly non-Latin. The script-detection threshold covers
  Cyrillic and Greek as well as CJK — a CJK-only test passed `ссе`.
* Case 7's cross-endpoint check now reports INCONCLUSIVE with a reason instead of
  FAIL. That failure was mine: `responses_body` flattens the `tool` message into
  prose, so the two endpoints receive different prompts and the comparison could
  never have passed. The cases 1, 3 and 8 divergences are real signal and remain
  asserted.

**What the 25 PASS rows do and do not mean.** The streaming checks assert
terminal-event counts and marker leakage, not content coherence, so they are
silent on this defect. Do not read them as partial reassurance.

**What is genuinely proven and can be closed out.** The prompt protocol work —
Tasks 1, 2 and 3 — is validated by the goldens, the offline suites, the effort
tier prompt-token differentiation, and the prefix-cache evidence. Only Tasks 4
and 5, the DSML tool-call path, still need live proof, and they cannot get it
until generation works.
