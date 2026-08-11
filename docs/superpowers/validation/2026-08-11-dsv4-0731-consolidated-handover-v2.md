# DeepSeek V4 0731 on EXO — Consolidated Handover (v2)

**Date:** 2026-08-11
**Supersedes:** `2026-08-10-dsv4-0731-consolidated-handover.md`. Roughly half of v1 has been overtaken by evidence; the superseded conclusions are listed in the final section so they are not inherited from the older document.
**Model:** `Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp` (154.4 GiB)
**Target:** two-rank tensor-parallel decode over Thunderbolt 5 + JACCL/RDMA, M4 Max + M5 Max
**Branch:** `codex/dsv4-0731-backbone`, published to `JaredP94/exo`
**Status:** the project's goal is met. Two-rank distributed inference is validated end to end. One item — tool calling — is closed as unvalidatable on this checkpoint.

Every claim below is labelled observed or inferred. Nothing should be upgraded in a later summary without new evidence.

## Status in one paragraph

The two-rank distributed inference path works. `--determinism 3` passes with three byte-identical terminating completions, factual questions return correct answers and stop, and a haiku request closes its own reasoning trace and completes in 240 tokens. One genuine defect was found in the model path — bfloat16 attention diverging between the two GPUs — and it is fixed with a directly measured cross-device invariant behind it. Throughput is measured for the first time: 21.37 tok/s decode with a 2.9 s time-to-first-token. The supported context ceiling is 32K, with a host-panic hazard at 64K. Tool calling cannot be validated here because the checkpoint has no trained tool-call capability under either declared protocol, which is a checkpoint-sourcing question rather than an engineering one. What remains open is a short list of raised-not-fixed items and one un-root-caused control-plane defect that is currently hardened rather than understood.

## What is proven

**Distributed inference is correct end to end.** A raw five-token prompt (`[671, 6102, 294, 8760, 344]`) on the two-rank sharded instance returns `Paris. The capital of France is Paris`. The full chat form returns `The capital of France is **Paris**.` and terminates after nine tokens. `--determinism 3` at `max_tokens=8192` passes with three identical terminating completions. Together these clear the model weights, the sharding, the collectives, JACCL over Thunderbolt, decode, the tokenizer's text path and the chat encoder.

**The float32 attention fix rests on measurement, not inference.** On bitwise-identical q/k/v, cross-device max-abs is 1.14062 in bfloat16 and 0.000787497 in float32, against a mean absolute attention output of 0.381306. Both ranks' bfloat16 outputs are finite and bounded — rank 0 within [-0.507812, 0.511719], rank 1 within [-0.6875, 0.679688] — so this is neither overflow nor NaN. A 1.14 delta between operands bounded near 0.69 requires near-opposite signs at the worst element, and rank 1's range is consistently wider rather than randomly scattered. The inference, which I consider secure: the two devices take different kernel paths, so float32 is a workaround for an upstream MLX defect and the MLX version is load-bearing.

**Gates are green.** basedpyright over the whole `src` include: zero errors, warnings and notes. `ruff check src` and `ruff format --check src` clean. Full suite on the Metal host: 734 passed, 3 skipped, 172 deselected. Both hosts run MLX `0.32.0.dev20260804+cc3f3e60` and `mlx_lm 0.31.3`, with the wheel byte-copied between machines to eliminate build skew.

**Capacity and throughput are measured.** Cold 16,389 prompt tokens in 70.85 s; cold 32,819 in 163.792 s; both `cached_tokens=0`. On a cold unique 546-token prompt: 2.9197 s TTFT, 187.00 tok/s prefill, 21.37 tok/s decode. The practical envelope is roughly five seconds for a short factual answer, fifteen for a 240-token reasoning answer, ninety seconds for two thousand tokens, and 6m26s for a full 8,192-token generation. This cluster suits interactive short-form work; long reasoning chains are possible but slow.

## The single most useful behavioural finding

**`enable_thinking=false` does not suppress reasoning on this checkpoint.** The pre-closed `<think></think>` is reference-conformant, and the model produces a planning trace anyway for anything non-trivial. Almost every "incoherent generation" report in this project's history was this behaviour meeting a `max_tokens` that was too small: a 650-token run at `finish_reason=length`, case 5 spending 2,047 of 2,048 tokens reasoning, a 256-token nonce probe cut off mid-answer, a 128-token haiku cut off mid-plan. Raised to 8,192 tokens, the same prompts complete correctly and stop on their own.

The operational consequence: every client budget must accommodate a reasoning trace, whether thinking is nominally enabled or not. Any harness or integration that assumes a few hundred tokens is enough will observe what looks like model incoherence and is actually truncation.

## Known limitations

**Tool calling is unvalidatable on this checkpoint.** Asked to call a tool with the full `get_weather` schema and V4 DSML instructions present in the rendered prompt, the model emits ordinary ASCII text pieces — `30, 94, 72461, 65, 94, 32, 1`, which is `<|tool_|` followed by EOS — rather than either declared protocol. The tokenizer declares both the V3 tool family at `128806–128814` and DSML at `128825`. Forcing `128806` into the stream produced unrelated Chinese and JSON prose with no tokens in that range; forcing `128825` produced `128825` repeated for all 128 tokens, which is the signature of a vocabulary entry that exists but was never trained; a V3-convention prompt drew a refusal after 13 tokens. Both ranks independently log `model has_tool_calling=False using tokens None, None`.

Because the forced-token probes bypass EXO's configuration entirely, this conclusion is safe regardless of EXO's own detection — it is not self-fulfilling. DSML parsing itself carries 58 tests and a clean 6,288-case fuzz run; Tasks 4 and 5 are tested but cannot be exercised live until a tool-capable checkpoint is available. That is the one open item requiring a decision from Jared, and it is about sourcing a checkpoint rather than changing code.

**32K is the supported ceiling and 64K is a host hazard.** A 64K attempt caused a kernel watchdog panic, with `watchdogd` missing check-ins for 94 seconds. Not OOM, not a GPU fault. Reproducibility is unknown and was deliberately not retested. Treat this as a host-availability risk rather than a capacity limit — an unbounded prompt length can take the machine down, which is why the API-cap question is on Jared's list.

**The control-plane connection asymmetry is hardened, not root-caused.** A stale `exo_rs` native extension on node 2 caused a real and separate defect: mismatched Zenoh key prefixes produced a healthy cross-host transport with zero state replication and no error on either side. Rebuilding both extensions fixed that. The one-way `SocketConnection` remains unexplained. Across three staggered trials the surviving edge was always node 1 → node 2 regardless of which node was master, so direction is bound to host identity rather than role or start order; every network-layer explanation was eliminated by direct measurement (MTU 1500 both interfaces with DF probes succeeding both ways, listeners on `*:52415`, literal addresses advertised rather than `.local` names, application firewall disabled, TCP reachable both directions, no proxy variables, interfaces republished every 10 s and present in both `/state` responses, node IDs matching); and node 2 was observed never to attempt the connection at all, with no error logged. The poll-loop hardening in this branch may or may not be the fix. Any explanation must also account for the fact that placement succeeded once, immediately after the native rebuild, with reciprocal edges reported.

**Sharded and unsharded paths diverge with depth, and that is fine.** Per-layer full-block deltas: layer 0 `0.015625`, layer 2 `0.015625`, layer 3 `0.125`, layer 4 `0.0234375`, layer 5 `0.03125`. Chained: `0.015625` at depth 1 rising to `0.046875` at depth 8, then `0.226562` at depth 9 and `0.90625` at depth 16 — the knee coinciding with the first expert-routing mismatch at layer 8. Broad float32 reduces depth-16 divergence to `0.00965488` and moves the first mismatch to layer 25, but depth-43 divergence is `10.1784` regardless. Bit-parity with a single-node reference is unachievable for a discrete-routing MoE at this depth, because any nonzero difference eventually crosses a routing boundary. It is also unnecessary: the sharded path returns correct, conditioned, terminating answers. This is numerical characterisation, not a defect.

One counter-intuitive result worth preserving: float32 for the `MoEGate` logits alone made divergence **worse** (`1.01562` versus a bfloat16 baseline of `0.90625`), because bfloat16 was accidentally masking near-ties — both paths rounded to the same bits and picked the same expert. Precision must be added upstream where the inputs disagree, never at the decision point, which only sharpens the disagreement. The same structural reason explains why sinkhorn-only float32 failed.

## Raised, not fixed

A request carrying a `tools` array against a model where `has_tool_calling=False` is accepted, executed, and returned as success with `tool_calls=null` and no explanation, while both ranks log the condition. This is the same diagnosability class as the stale-extension split: the system knew and the API said nothing.

Whether the API should enforce a prompt-length cap, given the 64K host panic.

The `ModelCard.backends` default, which makes cards silently vanish because the error is caught per-card and only warned. Trailing-assistant `tool_calls` being dropped. `_PARAM_PATTERN` attribute-order sensitivity. Node identity regenerated per process via `os.urandom(16)`, so it does not survive a restart — odd for an event-sourced system. Exposing the native extension's build identity in `/state`, which would have turned several investigation passes into a five-second comparison. Whether `scripts/` joins basedpyright's include or the harnesses move to `tools/`. Commit-message normalisation across the unpushed commits.

## Process requirements learned the hard way

**Reconcile compiled artifacts, not just git HEAD.** Matching HEAD, macOS version, RDMA status and checkpoint hash was insufficient; the divergence was in a native extension. Gate on runtime behaviour — both nodes declaring the same key prefix — rather than binary hash equality, since Rust builds embed absolute paths and identical source can compile to non-identical binaries.

**Check the interpreter before diagnosing a hang.** Node 1's shared uv CPython 3.13.11 was replaced by a 112-byte bash shim that circularly re-exec'd the venv's own `python3`, breaking every project on the host using that interpreter, not only this one. A broken interpreter fails at `exec` and looks exactly like a Metal deadlock from outside; that misdiagnosis cost several cycles chasing a nonexistent GPU wedge. Evidence preserved at `/tmp/shim_evidence_python3.13`; repaired with `uv python install --reinstall` (`--force` alone is a no-op for an existing installation). Author unattributable; no concurrent writer found.

**A single-node bisect of this checkpoint is impossible.** 154.4 GiB does not fit in 128 GB of unified memory. That is why the project is two-rank.

**Round-trip consistency is not correctness.** The wire audit verified that token IDs decode back to the intended prompt — but encode and decode share the same table, so a wrong-but-self-consistent tokenizer would pass identically. The IDs were subsequently verified against the checkpoint's own declarations (User `128803`, Assistant `128804`, think `128821`/`128822`, DSML `128825`, `vocab_size=129280` matching embedding and `lm_head` rows `[129280, 1024]`), and the checkpoint ships no chat template, so EXO's vendored V4 encoder supplies the arrangement. An empirical bisect through `raw_input_ids` established that the `<｜Assistant｜>` marker is what establishes an answering turn, and that the full chat form works.

**Per-layer tolerances say nothing about behaviour at depth.** A 0.05 per-layer parity tolerance was chosen for quantization noise and read for weeks as evidence of end-to-end correctness. It never was.

## Superseded conclusions

These circulated as findings before being falsified. They are listed so they are not inherited from v1 of this document or from progress reports.

Conditioning was never lost — the model reads its prompt, as demonstrated by the nonce appearing in output and by correct factual answers. The prompt-irrelevant output was, in almost every case, a reasoning trace truncated by an insufficient token budget. The one-token factual probe reporting top-1 `The` was a **test design error**: `The capital of France is **Paris**.` correctly begins with `The`, so a passing result was misread as a failure and sent the investigation into a tokenizer audit that found nothing. A tokenizer special-token substitution was hypothesised and disproved. `--float32-hyper` was not fixing a hyper-connection defect; it was buying enough precision to keep a routing decision stable. Ratio-128 layers are not selectively affected — layers 4 and 5 pass. The isolated layer-3 routing mismatch is superseded by the chained layer-8 baseline. Zero routing mismatches at depth 43 was set as an acceptance gate and is unachievable in principle; that criterion was wrong and it blocked the measurements that mattered. The `scaled_dot_project_attention` typo was a bug inside the fix, never the cause of the original defect. The collectives were never implicated: `all_sum` is symmetric, so ranks resynchronise after every collective. The stale `exo_rs` binary explained the namespace split, not the connection asymmetry. Socket-formation direction does not track mastership. The 52-of-55 cache hit was legitimate prefix sharing between two audit prompts, not corruption. A 1.14 cross-device difference does not imply out-of-range output, since a difference between in-range operands can exceed either one's magnitude.

## Evidence index

**Published branch:** `codex/dsv4-0731-backbone` on `JaredP94/exo`, latest `51527641`. Published history consists of recreated snapshots rather than the local commit sequence, because local git authentication is unavailable; published trees were verified against local trees at each step. Four commits (design docs, the implementation plan, a worktree `.gitignore`) were pushed to fork `main` in error and left in place deliberately.

**Key local commits:** `eb3e6fd6` the float32 SDPA fix; `c8fb5a19` native-artifact reconciliation in the runbook; `ba5c0304` the qualification distinguishing namespace split from connection asymmetry.

**Worktrees:** `codex-dsv4-0731-backbone` on node 1, branch `codex/dsv4-0731-backbone`, carries uncommitted user work — do not disturb; `exo-dsv4-validation` on node 2, same branch, `stash@{0}` retained and unapplied; `dsv4-0731-live-clean` on node 1, detached at `eb3e6fd6`.

**Addresses and ports:** node 1 `169.254.240.63` on `en6`; node 2 `169.254.233.2` on `en3`; Zenoh `52414`, API `52415`.

**Debug flags, off by default:** `EXO_ENABLE_RAW_INPUT_IDS_DEBUG=1` for pre-tokenised input; `EXO_DEBUG_TOKEN_IDS=1` for generated token IDs. Both are test-covered and both were decisive.

**Harnesses:** `scripts/check_cross_device_sdpa.py`, `scripts/validate_dsv4_live_api.py` (run `--self-test` before trusting any result and do not soften an assertion — several exist because weaker versions passed on broken output), `scripts/parity_sharded_vs_unsharded.py` (now with chained-depth and expert-routing audit), `scripts/instance_window_battery.py`, `scripts/measure_dsv4_throughput.py`, `scripts/probe_tool_call_prompt.py`, `scripts/compare_dsv4_reference.py`, `scripts/compare_loaded_vs_checkpoint.py`, `scripts/probe_shard_rank_selection.py`.

**Preserved artifacts:** `/private/tmp/exo-dsv4-topology-trial{1,2,3}-node2.log`, `/private/tmp/exo-dsv4-node2-outbound-audit.log`, `/private/tmp/dsv4-dirty-backup-2026-08-10.diff` and `.status`, `/private/tmp/dsv4-detached-diagnostics-2026-08-10.patch`, `/private/tmp/dsv4-stale-locks-20260811`, `/tmp/shim_evidence_python3.13`.

**Still to file:** the upstream MLX report on bfloat16 `scaled_dot_product_attention` cross-device divergence. `scripts/check_cross_device_sdpa.py` reproduces it in one command. Disclose that both hosts run source commit `cc3f3e60` from wheels built one day apart, and that the measurement was taken after byte-copying one wheel to both hosts; a future pin should reference a single wheel rather than a source commit, since kernel selection is the defect under discussion.
