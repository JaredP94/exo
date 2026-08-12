# OMLX Upstream Reconciliation Implementation Plan

> **For agentic workers:** This plan is executed by an external codex agent, one task per prepared packet, with orchestrator cross-check between tasks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the DeepSeek-V4-Flash-0731 EXO port up to date with 51 commits of upstream OMLX by integrating the three items that are genuinely portable, settling two open numerics questions, and recording the rest as reference.

**Architecture:** EXO does not import `omlx`; it subclasses the pinned fork `rltakashige/mlx-lm@leo/deepseek-v4` and layers a compatibility module over it. OMLX's `patches/deepseek_v4/` and that fork are independent implementations, so the two fork-level changes are applied as runtime installs from EXO — following `install_deepseek_v4_sdpa_float32()` — and the prefill memory guard is reimplemented EXO-native rather than ported.

**Tech Stack:** Python 3.13, MLX `0.32.0.dev20260804+cc3f3e60`, `mlx_lm 0.31.3` (pinned fork), FastAPI, pytest, basedpyright, ruff, just, nix.

## Global Constraints

- Gate before every commit: `just lint && just fmt && just check && just test` (`uv run ruff check --fix`, `treefmt || nix fmt`, `uv run basedpyright --project pyproject.toml`, `uv run pytest src`).
- `basedpyright` must report **zero** errors, warnings and notes over the whole `src` include. Never run it per-file.
- `RULES.md`: strict exhaustive typing, `Literal` over enums, no three-letter acronyms, no new dependencies without asking, error-handling rationale stated in the docstring.
- `RULES.md`: **code that violates the rules is raised with Jared, not fixed.**
- Do not change the MLX or `mlx_lm` pins in `pyproject.toml`. They are load-bearing for the float32 SDPA fix.
- Do not modify or revert `install_deepseek_v4_sdpa_float32()`.
- Do not touch worktrees `codex-dsv4-0731-backbone` (uncommitted user work), `exo-dsv4-validation` on node 2 (unapplied `stash@{0}`), or `dsv4-0731-live-clean` (detached at `eb3e6fd6`).
- **Never issue a 64K-token prefill.** A 64K prompt caused a kernel watchdog host panic with `watchdogd` missing check-ins for 94 seconds. Reproduction is forbidden in every task.
- `scripts/validate_dsv4_live_api.py --self-test` runs before any live use of that harness, and no assertion in it may be softened or removed.
- Every `max_tokens` budget in a live check is **8192 or greater**. `enable_thinking=false` does not suppress reasoning on this checkpoint; a smaller budget produces truncation that reads as incoherence.
- Commit subjects follow `RULES.md:71-84` **literally**: imperative mood; prefixed with one of `documentation` / `feature` / `refactor` / `bugfix` / `chore` / `test`; capitalised after the prefix; **fifty characters or fewer including the prefix**; no trailing period. The branch's existing commits use conventional-commits (`docs:`, `fix:`, `style:` — none of which are RULES.md change types) and six of the last twenty-five exceed fifty characters. That divergence is known, raised and unresolved: do not imitate it, and do not rewrite it.

---

## Delegation model

This plan is the master document. Each task below becomes one self-contained packet handed to the codex agent, which runs on node 1 and can drive both nodes. Packets are issued **one at a time**: the next is written only after the previous task's report has passed two rounds of cross-check.

Each packet is assembled from its task here plus the Global Constraints section verbatim, the "Claims prohibited without evidence" list for that task, and the report format. The agent receives no other context.

## Branch and worktree

All work lands on `codex/dsv4-omlx-upstream-sync`, cut from the validated tip of `codex/dsv4-0731-backbone` into a **new** worktree. Creating it is Task 1, Step 1.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `docs/superpowers/specs/2026-08-11-omlx-upstream-reconciliation-design.md` | The approved design; committed and re-pinned | 1, 2 |
| `docs/superpowers/validation/2026-08-11-omlx-sync-baseline.md` | S0 control measurements, referenced by every later gate | 1 |
| `docs/superpowers/specs/2026-08-04-deepseek-v4-flash-0731-exo-rdma-design.md` | Reference revision line and reconciliation appendix | 2 |
| `docs/superpowers/validation/2026-08-11-omlx-group-two-findings.md` | Sub-4-bit rule and block-config findings | 3 |
| `src/exo/shared/models/model_cards.py` | `context_length`, already present, read by the guard | 4 |
| `src/exo/worker/engines/mlx/generator/generate.py` | `prefill()` at `:282`; admission guard inserted before allocation | 4 |
| `src/exo/api/main.py` | HTTP 400 refusal with an explanatory detail | 4 |
| `src/exo/worker/engines/mlx/deepseek_v4_0731_model.py` | Both runtime installers, beside the existing ones | 5, 6 |
| `src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py` | Installer call sites at `:89-92` | 5, 6 |
| `src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py` | Guard tests | 4 |
| `src/exo/worker/tests/unittests/test_mlx/test_dsv4_indexer_shortcut.py` | Shortcut contract and equivalence tests | 5 |
| `src/exo/worker/tests/unittests/test_mlx/test_dsv4_pooled_append.py` | Pool append contract and equivalence tests | 6 |

`src/exo/worker/tests/unittests/test_mlx/test_dsv4_sdpa_float32_patch.py` is the template for both new contract-guard test files. Read it before writing either. Its module docstring enumerates, per test, the exact break it catches; the new files must do the same.

---

## Task 1 (S0): Baseline capture

No production code changes. This task exists because the project's recorded failures include a harness that reported 36/36 with no assertions and a 0.05 per-layer tolerance read for weeks as end-to-end evidence. Every later task differences against the numbers produced here.

**Files:**
- Create: `docs/superpowers/validation/2026-08-11-omlx-sync-baseline.md`
- Move into the new branch: `docs/superpowers/specs/2026-08-11-omlx-upstream-reconciliation-design.md` (currently untracked in the `codex-dsv4-0731-backbone` worktree)

**Interfaces:**
- Consumes: nothing.
- Produces: `2026-08-11-omlx-sync-baseline.md` containing, as a markdown table with one row per measurement and a `median` column over three runs: `decode_tokens_per_second`, `prefill_tokens_per_second`, `time_to_first_token_seconds`, `cold_prefill_16k_seconds`, `cold_prefill_32k_seconds`. Tasks 5 and 6 cite these field names directly.

- [ ] **Step 1: Create the branch and worktree**

```bash
cd /Users/jared/aishit/exo-install/exo
git worktree add -b codex/dsv4-omlx-upstream-sync \
  .worktrees/codex-dsv4-omlx-upstream-sync codex/dsv4-0731-backbone
cd .worktrees/codex-dsv4-omlx-upstream-sync
git log --oneline -1
```

Expected: the worktree is created and HEAD matches the validated tip of `codex/dsv4-0731-backbone`.

- [ ] **Step 2: Commit the design and plan documents**

Copy both from the `codex-dsv4-0731-backbone` worktree, where they sit untracked. Copy them; do not `git mv`, and do not stage anything in that worktree.

- `docs/superpowers/specs/2026-08-11-omlx-upstream-reconciliation-design.md`
- `docs/superpowers/plans/2026-08-11-omlx-upstream-reconciliation.md`

```bash
git add docs/superpowers/specs/2026-08-11-omlx-upstream-reconciliation-design.md \
        docs/superpowers/plans/2026-08-11-omlx-upstream-reconciliation.md
git commit -m "documentation: Add OMLX sync design and plan"
```

- [ ] **Step 3: Confirm the environment matches the validated run**

```bash
uv run python -c "import mlx.core, mlx_lm; print(mlx.core.__version__, mlx_lm.__version__)"
```

Expected: `0.32.0.dev20260804+cc3f3e60` and `0.31.3`. **If either differs, stop and report.** The baseline is meaningless against a different kernel selection, and the float32 SDPA fix is pinned to these.

- [ ] **Step 4: Run the full offline gate**

```bash
just lint && just fmt && just check && just test
```

Expected: ruff clean, basedpyright zero errors/warnings/notes, pytest passing. Record the exact pass/skip/deselect counts — Tasks 4, 5 and 6 compare against them.

- [ ] **Step 5: Self-test the live harness before trusting it**

```bash
uv run python scripts/validate_dsv4_live_api.py --self-test
```

Expected: every check demonstrates it can fail. If any check cannot be made to fail, stop and report — that check is not a check.

- [ ] **Step 6: Bring up the two-rank cluster and run determinism**

```bash
uv run python scripts/validate_dsv4_live_api.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --determinism 3 --max-tokens 8192
```

Expected: three byte-identical terminating completions, exit code 0. A `finish_reason` of `length` is a FAIL, not a pass.

- [ ] **Step 7: Measure throughput, three runs**

```bash
for i in 1 2 3; do
  uv run python scripts/measure_dsv4_throughput.py \
    --base-url http://localhost:52415 \
    --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
    --prompt-tokens 546 --max-tokens 8192
done
```

Record all three runs and the median for TTFT, prefill tok/s and decode tok/s. Three runs, not one — a single sample cannot support a later regression claim.

- [ ] **Step 8: Measure cold prefill at 16K and 32K**

```bash
uv run python scripts/instance_window_battery.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --out /tmp/omlx-sync-baseline-windows.json
```

Each prompt must be unique so `cached_tokens=0`. **Confirm `cached_tokens=0` in the output and state it in the report** — a cached prefill is not a cold prefill, and the prefix cache demonstrably reuses 284 of 286 tokens when given the chance. Do not run a 64K window.

- [ ] **Step 9: Write and commit the baseline document**

Record every number with its run-to-run spread, the environment versions from Step 3, and the offline gate counts from Step 4.

```bash
git add docs/superpowers/validation/2026-08-11-omlx-sync-baseline.md
git commit -m "documentation: Record the OMLX sync baseline"
```

**Claims prohibited without evidence:** that the environment matches (paste Step 3 output); that determinism passed (paste the three completions); that any prefill was cold (paste `cached_tokens`).

---

## Task 2 (S1): Re-pin the reference revision

Offline only. No runtime change. Proves the Group VI claim rather than asserting it.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-deepseek-v4-flash-0731-exo-rdma-design.md:9` (the `Working revisions` line)
- Modify: same file, append a reconciliation appendix

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: the design document's `Working revisions` line reading `EXO a6cd2fce; OMLX 2450a53c`.

- [ ] **Step 1: Fetch upstream OMLX**

```bash
cd /Users/jared/aishit/exo-install/omlx
git fetch origin
git log --oneline -1 origin/main
```

Expected: `2450a53c`. If upstream has advanced past `2450a53c`, **stop and report** — the triage in the design document covers exactly `50846648..2450a53c`, and a longer range invalidates the commit counts.

- [ ] **Step 2: Prove the prompt protocol and DSML reference are unchanged**

```bash
cd /Users/jared/aishit/exo-install/omlx
for f in omlx/patches/deepseek_v4/chat_template_v4.py \
         omlx/patches/deepseek_v4/tool_parser_v4.py; do
  a=$(git show 50846648:$f | shasum -a 256 | cut -d' ' -f1)
  b=$(git show 2450a53c:$f | shasum -a 256 | cut -d' ' -f1)
  echo "$f  $a  $b  $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFERS)"
done
```

Expected: both `IDENTICAL`. **If either differs, stop and report** — the entire justification for a zero-risk re-pin is that these two files did not change, and a difference means the goldens must be re-derived rather than merely regenerated.

- [ ] **Step 3: Regenerate the goldens and confirm an empty diff**

```bash
cd /Users/jared/aishit/exo-install/exo/.worktrees/codex-dsv4-omlx-upstream-sync
uv run python scripts/generate_dsv4_0731_prompt_goldens.py
git diff --stat
```

Expected: empty. A non-empty diff contradicts Step 2 and must be reported, not committed.

- [ ] **Step 4: Update the reference revision line**

In `docs/superpowers/specs/2026-08-04-deepseek-v4-flash-0731-exo-rdma-design.md:9`, change `OMLX 50846648` to `OMLX 2450a53c`, and add immediately below it:

```markdown
**Reference reconciliation:** `50846648..2450a53c` (51 commits) triaged in
`2026-08-11-omlx-upstream-reconciliation-design.md`. `chat_template_v4.py` and
`tool_parser_v4.py` are byte-identical across the range, so the prompt protocol,
the DSML reference parser and the 16 committed goldens are unaffected.
```

- [ ] **Step 5: Append the reconciliation appendix**

Append an `## Appendix B — OMLX reconciliation 50846648..2450a53c` section carrying the Group I through VI verdict tables from the reconciliation design document, with the per-group commit counts (I: 5, II: 2, III: 3, IV: 7, V: 34, summing to 51).

- [ ] **Step 6: Report on the root PR body, do not edit it**

`/Users/jared/aishit/exo-install/PR-BODY-dsv4-0731-backbone.md` also names a reference revision. Report whether it does and what it says. **Do not edit it** — whether it is re-pinned is an open decision for Jared, and it sits outside both worktrees.

- [ ] **Step 7: Run the offline gate**

```bash
just lint && just fmt && just check && just test
```

Expected: identical counts to Task 1 Step 4.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-08-04-deepseek-v4-flash-0731-exo-rdma-design.md
git commit -m "documentation: Re-pin OMLX reference to 2450a53c"
```

**Claims prohibited without evidence:** that the two reference files are unchanged (paste the shasum comparison); that goldens regenerate cleanly (paste `git diff --stat`).

---

## Task 3 (S2): Settle the two Group II questions

Investigation. Produces a written finding and, only if a hazard is real, a regression test pinning current behaviour. **Fixes are raised for decision, not applied** — `RULES.md`.

**Files:**
- Create: `docs/superpowers/validation/2026-08-11-omlx-group-two-findings.md`
- Read: `/Users/jared/aishit/exo-install/omlx` at `2450a53c`, `omlx/patches/deepseek_v4/utils_patch.py` (`_native_ratio128_attention_enabled`) and `omlx/patches/deepseek_v4/switch_layers.py` (`_block_config`)
- Read: the pinned fork's `mlx_lm/models/switch_layers.py` and `mlx_lm/models/deepseek_v4.py`
- Read: `src/exo/worker/engines/mlx/deepseek_v4_0731_config.py` (`QuantizationSpec`, `realized_quantization`)

**Interfaces:**
- Consumes: nothing.
- Produces: a findings document with two verdict sections, each stated as `APPLIES`, `DOES NOT APPLY`, or `INCONCLUSIVE` with the evidence that decides it.

- [ ] **Step 1: Extract the checkpoint's realized quantization**

Read the checkpoint's declared quantization directly, so the answer does not depend on EXO's parsing being correct:

```bash
uv run python -c "
import collections, json
from pathlib import Path
root = Path.home() / '.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp'
config = json.loads((root / 'config.json').read_text())
quantization = config.get('quantization', {})
hist = collections.Counter()
for key, value in quantization.items():
    if isinstance(value, dict):
        hist[(value.get('bits'), value.get('group_size'), value.get('mode'))] += 1
    else:
        hist[('scalar:' + key, value, None)] += 1
for entry, count in sorted(hist.items(), key=lambda kv: str(kv[0])):
    print(entry, count)
"
```

Expected: a histogram of `(bits, group_size, mode)` over every declared module, plus the top-level scalar keys. Then cross-check it against EXO's own view via `normalized_quantization_specs` in `deepseek_v4_0731_config.py:90`, and report both. A disagreement between the two is itself a finding.

- [ ] **Step 2: Decide whether the sub-4-bit rule applies**

Upstream `b6811ed6` disables the native ratio-128 sparse-attention path when any declared `bits < 4`. Answer three questions in the findings document:

1. Does any module in the Step 1 histogram declare `bits < 4`?
2. Does the pinned fork have a ratio-128 native-attention path at all, and under what condition is it taken? Cite `deepseek_v4.py` line numbers.
3. If both are yes, is that path reachable for this checkpoint at runtime?

Verdict is `APPLIES` only if all three are yes.

- [ ] **Step 3: Compare the block-config crossovers**

Upstream `c48e1a82` split one threshold into mxfp4 (`16384`) and affine (`8192`). Determine whether the fork's `switch_layers.py` uses a single threshold for both kinds. Quote both implementations side by side in the findings document.

- [ ] **Step 4: Write the findings document**

Each section states its verdict, the evidence, and — if `APPLIES` — the proposed change **as a proposal for Jared, not as an edit**. Include what would change the verdict, so a later session can re-test rather than re-argue.

- [ ] **Step 5: Add a regression test only if a hazard is real**

If either verdict is `APPLIES`, add a test that pins the **current** behaviour and states in its docstring what break it catches, so the eventual fix has a control. If both verdicts are `DOES NOT APPLY`, add no test and say so explicitly.

- [ ] **Step 6: Run the offline gate and commit**

```bash
just lint && just fmt && just check && just test
git add docs/superpowers/validation/2026-08-11-omlx-group-two-findings.md
git commit -m "documentation: Record the Group II findings"
```

**Claims prohibited without evidence:** any verdict without the histogram and the cited line numbers; `DOES NOT APPLY` on the grounds that the path "probably falls back" — reachability must be shown, not assumed.

---

## Task 4 (S5): Prefill admission guard

Closes the open prompt-cap question and removes a host-availability risk. Ordered ahead of the performance work because an unbounded prompt can currently take a host down.

**Files:**
- Modify: `src/exo/worker/engines/mlx/generator/generate.py` — `prefill()` at `:282`, guard inserted after `num_tokens = len(prompt_tokens)` at `:300` and before any model call
- Modify: `src/exo/api/main.py` — reject at the request boundary with HTTP 400
- Create: `src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py`

**Interfaces:**
- Consumes: `ModelCard.context_length` (`src/exo/shared/models/model_cards.py:172`), already populated from `max_position_embeddings`.
- Produces: `class PromptTooLongError(ValueError)` in `generate.py`, carrying `prompt_tokens: int` and `limit_tokens: int` attributes. `main.py` catches it and maps it to HTTP 400.

- [ ] **Step 1: Write the failing tests**

```python
def test_guard_rejects_a_prompt_above_the_ceiling() -> None:
    """Break this catches: the ceiling being compared with > instead of >=,
    or the guard being placed after the first model call."""
    with pytest.raises(PromptTooLongError) as excinfo:
        assert_prompt_within_ceiling(prompt_tokens=40_000, limit_tokens=32_768)
    assert excinfo.value.prompt_tokens == 40_000
    assert excinfo.value.limit_tokens == 32_768


def test_guard_admits_a_prompt_at_the_ceiling() -> None:
    """Break this catches: an off-by-one that refuses a legal 32,768-token
    prompt, which would silently shrink the supported context."""
    assert_prompt_within_ceiling(prompt_tokens=32_768, limit_tokens=32_768)


def test_error_names_both_numbers() -> None:
    """Break this catches: a bare 'prompt too long' with no numbers, which is
    the same non-diagnosability defect as returning tool_calls=null in silence."""
    error = PromptTooLongError(prompt_tokens=40_000, limit_tokens=32_768)
    assert "40000" in str(error) and "32768" in str(error)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py -v
```

Expected: FAIL with `ImportError` / `NameError` on `PromptTooLongError` and `assert_prompt_within_ceiling`.

- [ ] **Step 3: Implement the guard**

```python
class PromptTooLongError(ValueError):
    """A prompt exceeded the admission ceiling and was refused before allocation.

    Raised rather than truncated: a 64K prompt has been observed to cause a
    kernel watchdog host panic, so silently proceeding risks host availability.
    Both counts are carried on the exception so the API layer can explain the
    refusal instead of returning an opaque failure.
    """

    def __init__(self, prompt_tokens: int, limit_tokens: int) -> None:
        self.prompt_tokens: int = prompt_tokens
        self.limit_tokens: int = limit_tokens
        super().__init__(
            f"prompt of {prompt_tokens} tokens exceeds the admission ceiling of "
            f"{limit_tokens} tokens"
        )


def assert_prompt_within_ceiling(prompt_tokens: int, limit_tokens: int) -> None:
    """Refuse an over-ceiling prompt before any memory is allocated."""
    if limit_tokens > 0 and prompt_tokens > limit_tokens:
        raise PromptTooLongError(prompt_tokens, limit_tokens)
```

Call it in `prefill()` immediately after `num_tokens = len(prompt_tokens)` at `:300`, before the first `model(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py -v
```

Expected: PASS.

- [ ] **Step 5: Map the error to HTTP 400**

In `src/exo/api/main.py`, catch `PromptTooLongError` and re-raise as `HTTPException(status_code=400, detail=str(exc)) from exc`, matching the existing pattern at `:495`. Add a test asserting the response body contains both token counts.

- [ ] **Step 6: Verify the guard fires without a 64K prefill**

The ceiling for verification is set to a small value so the guard is exercised at a safe size:

```bash
uv run python scripts/validate_dsv4_live_api.py --self-test
uv run python -c "
from exo.worker.engines.mlx.generator.generate import (
    PromptTooLongError,
    assert_prompt_within_ceiling,
)
try:
    assert_prompt_within_ceiling(prompt_tokens=600, limit_tokens=512)
except PromptTooLongError as exc:
    print('REFUSED:', exc)
else:
    raise SystemExit('guard did not fire')
"
```

Then send one live request whose prompt exceeds the configured ceiling and confirm HTTP 400 with both counts in the body. Choose a ceiling and a prompt size that are both well inside the supported 32K range for this check.

**Do not issue a 64K prefill to prove the guard.** The guard is proved at the boundary, not by reproducing the panic.

- [ ] **Step 7: Confirm the supported path is unaffected**

```bash
uv run python scripts/validate_dsv4_live_api.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --determinism 3 --max-tokens 8192
uv run python scripts/instance_window_battery.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --out /tmp/omlx-sync-s5-windows.json
```

Expected: determinism passes; the 16K and 32K windows still complete and are **not** refused. A guard that rejects a legal 32K prompt is a regression.

- [ ] **Step 8: Run the offline gate and commit**

```bash
just lint && just fmt && just check && just test
git add src/exo/worker/engines/mlx/generator/generate.py src/exo/api/main.py \
        src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py
git commit -m "feature: Refuse over-ceiling prompts at admission"
```

**Claims prohibited without evidence:** that the guard fires (paste the refusal); that 32K still works (paste the window battery output); any claim about 64K behaviour whatsoever.

**Open decision to surface in the report:** the ceiling's default value and whether it is a hard refusal or configurable. Propose a value with reasoning; do not settle it.

---

## Task 5 (S3): Indexer all-pooled shortcut

Upstream `4c6b5931`. The fork's `Indexer.__call__` (`mlx_lm/models/deepseek_v4.py:1799`) computes `k = min(self.index_topk, idx_kv.shape[1])` and always runs the scoring einsum and `argpartition`, including when the selection is the identity set.

**Files:**
- Modify: `src/exo/worker/engines/mlx/deepseek_v4_0731_model.py` — add `install_deepseek_v4_indexer_shortcut()` after `install_deepseek_v4_sdpa_float32()` at `:189-239`
- Modify: `src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py:21-23, 89-92` — import and call it
- Create: `src/exo/worker/tests/unittests/test_mlx/test_dsv4_indexer_shortcut.py`

**Interfaces:**
- Consumes: Task 1's `decode_tokens_per_second`, `prefill_tokens_per_second`, `cold_prefill_16k_seconds`, `cold_prefill_32k_seconds`.
- Produces: `install_deepseek_v4_indexer_shortcut() -> None`, idempotent, guarded by module flag `_exo_dsv4_indexer_shortcut_patched`.

**The two conditions that carry the risk.** The shortcut is only valid when the identity set is genuinely equivalent:

1. **Uniform pool lengths.** `cache.pooled_lengths(_K_IDX)` (`deepseek_v4.py:1192`) returns a per-row list for variable-length batches. When rows have different valid lengths, an identity set would include invalid pool slots that the scoring path masks out with `-1e30`. **The shortcut must be skipped unless `pooled_lengths` is `None` or every entry equals `idx_kv.shape[1]`.**
2. **Ordering.** The returned indices feed `mx.take_along_axis` and then `mx.concatenate([window_kv, gathered], axis=1)` at `:2172`. Attention is permutation-invariant over keys only if the mask permutes with them. The equivalence test must compare final attention output, not just the index set.

- [ ] **Step 1: Read the template**

Read `src/exo/worker/tests/unittests/test_mlx/test_dsv4_sdpa_float32_patch.py` in full, including its module docstring listing the break each test catches. The new file follows that structure.

**Local test helpers.** Write these in the same file before the tests. They are test scaffolding, not production code:

- `_fake_cache(pool_lengths: list[int] | None) -> dsv4.DeepseekV4Cache` — a cache whose `_K_IDX` branch has `pool_lengths` set as given and a `pool` sized to `max(pool_lengths)` when the list is ragged.
- `_call_indexer(cache, pooled_len: int, index_topk: int) -> _IndexerResult` — invokes the patched `Indexer.__call__` against `cache` with a pool of `pooled_len` rows and the given `index_topk`, returning a small dataclass `_IndexerResult` with fields `indices: mx.array` and `scoring_path_was_taken: bool`. Detect which path ran by wrapping `mx.argpartition` with a counter for the duration of the call, not by re-deriving the condition — a test that recomputes the branch condition proves nothing.
- `_attention_output(shortcut: bool, pooled_len: int, index_topk: int) -> mx.array` — runs one `V4Attention.__call__` on fixed pseudo-random inputs seeded with `mx.random.seed(42)`, with the installer applied or not applied, and returns the attention output.

- [ ] **Step 2: Write the failing tests**

```python
def test_upstream_symbol_still_exists() -> None:
    """Break this catches: an mlx_lm rename. The install would raise at load
    time, which is loud, but this fails in CI first and says why."""
    assert hasattr(dsv4, "Indexer")
    assert callable(getattr(dsv4.Indexer, "__call__", None))


def test_patch_replaces_the_bound_method(unpatched: Callable[..., Any]) -> None:
    """Break this catches: patching a misspelled attribute, which creates a new
    unused name and leaves the original scoring path running."""
    install_deepseek_v4_indexer_shortcut()
    assert dsv4.Indexer.__call__ is not unpatched


def test_shortcut_is_skipped_for_ragged_pool_lengths() -> None:
    """Break this catches: the ragged-batch guard being dropped. An identity set
    over a ragged pool selects slots the scoring path masks with -1e30, so short
    rows would attend to uninitialised pool entries."""
    cache = _fake_cache(pool_lengths=[4, 7])
    result = _call_indexer(cache, pooled_len=7, index_topk=512)
    assert result.scoring_path_was_taken


def test_shortcut_matches_the_scoring_path_on_attention_output() -> None:
    """Break this catches: an ordering divergence. The indices feed
    take_along_axis then concatenate([window_kv, gathered]); comparing index
    sets alone would pass while the attention output differed."""
    reference = _attention_output(shortcut=False, pooled_len=64, index_topk=512)
    patched = _attention_output(shortcut=True, pooled_len=64, index_topk=512)
    assert mx.allclose(reference, patched, atol=0.0, rtol=0.0)


def test_scoring_path_is_kept_above_the_threshold() -> None:
    """Break this catches: the condition being inverted, which would replace a
    real top-k selection with the identity and change what the model attends to."""
    cache = _fake_cache(pool_lengths=None)
    result = _call_indexer(cache, pooled_len=1024, index_topk=512)
    assert result.scoring_path_was_taken


def test_patch_is_idempotent() -> None:
    """Break this catches: double application wrapping the wrapper."""
    install_deepseek_v4_indexer_shortcut()
    first = dsv4.Indexer.__call__
    install_deepseek_v4_indexer_shortcut()
    assert dsv4.Indexer.__call__ is first
```

`test_shortcut_matches_the_scoring_path_on_attention_output` uses `atol=0.0, rtol=0.0` deliberately: the claim is that the shortcut is **lossless**, not close. If exact equality does not hold, the shortcut is not lossless and the task stops for a decision.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_indexer_shortcut.py -v
```

Expected: FAIL on `ImportError` for `install_deepseek_v4_indexer_shortcut`.

- [ ] **Step 4: Implement the installer**

Wrap `dsv4.Indexer.__call__`. Inside the wrapper, compute `idx_kv` exactly as the original does, and take the shortcut only when both conditions hold, otherwise delegate to the original. State the ragged-pool and ordering reasoning in the docstring per `RULES.md`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_indexer_shortcut.py -v
```

Expected: PASS, including the exact-equality test.

- [ ] **Step 6: Wire it into the loader**

Add the import beside the others at `:21-23` and the call after `install_deepseek_v4_sdpa_float32()` at `:90`. Add a source-level test that the call site exists, mirroring `test_dsv4_sdpa_float32_patch.py`'s `loader_installs_the_patch`.

- [ ] **Step 7: Run the offline gate**

```bash
just lint && just fmt && just check && just test
```

Expected: counts equal to Task 1 Step 4 plus the new tests. No pre-existing test may change status.

- [ ] **Step 8: Live verification**

```bash
uv run python scripts/validate_dsv4_live_api.py --self-test
uv run python scripts/validate_dsv4_live_api.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --determinism 3 --max-tokens 8192
uv run python scripts/parity_sharded_vs_unsharded.py \
  --model-path "$HOME/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp" \
  --layer 8 --seq-len 512 --seed 42
for i in 1 2 3; do
  uv run python scripts/measure_dsv4_throughput.py \
    --base-url http://localhost:52415 \
    --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
    --prompt-tokens 546 --max-tokens 8192
done
uv run python scripts/instance_window_battery.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --out /tmp/omlx-sync-s3-windows.json
```

Pass criteria, each stated against Task 1's baseline document:
- determinism: three identical terminating completions
- `parity_sharded_vs_unsharded` at layer 8: unchanged from baseline
- `decode_tokens_per_second`: within 3% of baseline median, or better
- `cold_prefill_16k_seconds` and `cold_prefill_32k_seconds`: improved, or within 3% — with `cached_tokens=0` confirmed

- [ ] **Step 9: Commit**

```bash
git add src/exo/worker/engines/mlx/deepseek_v4_0731_model.py \
        src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py \
        src/exo/worker/tests/unittests/test_mlx/test_dsv4_indexer_shortcut.py
git commit -m "refactor: Skip indexer scoring on identity topk"
```

**Claims prohibited without evidence:** that the shortcut is lossless (the exact-equality test must pass, not `allclose` with a tolerance); any throughput claim not differenced against Task 1's recorded median across three runs.

**Revert condition:** if the 32K cold prefill does not improve and no test justifies the change on correctness grounds, revert rather than keep it.

---

## Task 6 (S4): Pooled append-in-place

Upstream `b128b232`, cache portion only. The fork's `DeepseekV4Cache.update_pool` (`mlx_lm/models/deepseek_v4.py:1152-1190`) grows the pool with `mx.concatenate([pool, new_pooled], axis=1)` on every update.

**This is the highest-risk task and is ordered last.**

**Files:**
- Modify: `src/exo/worker/engines/mlx/deepseek_v4_0731_model.py` — add `install_deepseek_v4_pooled_append()`
- Modify: `src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py:21-23, 89-92`
- Create: `src/exo/worker/tests/unittests/test_mlx/test_dsv4_pooled_append.py`

**Interfaces:**
- Consumes: Task 1's `cold_prefill_16k_seconds` and `cold_prefill_32k_seconds`.
- Produces: `install_deepseek_v4_pooled_append() -> None`, idempotent, guarded by `_exo_dsv4_pooled_append_patched`.

**Three obstacles that must be resolved before implementing.** Report on all three in Step 1; if any cannot be resolved cleanly, stop and report rather than working around it.

1. **`_CompressorBranch` uses `__slots__`** (`deepseek_v4.py:661-672`) with no spare slot, so a capacity/length pair cannot simply be added as attributes.
2. **`branch.pool.shape[1]` is read as the logical length** elsewhere — `update_pool` itself at `:1163`, and `V4Attention` at `:2138` via `get_branch(_K_COMP).pool`. A capacity-backed buffer breaks that invariant unless `pool` is exposed as a view of the first `_pool_len` rows.
3. **`_branch_tuple` / `_set_branch_tuple`** (`:765`, `:775`) serialise branch state, and EXO's `_copy_compressor_branch` and `_copy_v4_cache` (`src/exo/worker/engines/mlx/cache.py:136`, `:150`) snapshot it. Snapshots are long-lived consumers and **must copy**, not hold a view onto a buffer that later regrows.

- [ ] **Step 1: Resolve the three obstacles and report before writing code**

Produce a short written design covering: where the capacity and logical length live given `__slots__`; how `pool` stays a correct view; and how the snapshot and prefix-cache paths are kept copy-semantic. **Send this to the orchestrator and wait for approval before Step 2.** This is the one mid-task checkpoint in the plan.

**Local test helpers.** Write these in the same file before the tests. `_copy_compressor_branch` is **not** a helper — import the real one from `src/exo/worker/engines/mlx/cache.py:136`, because the point is to test the production snapshot path.

- `_new_cache() -> dsv4.DeepseekV4Cache` — a cache constructed with the checkpoint's `sliding_window`, with both branches empty.
- `_rows(n: int, batch: int = 1) -> mx.array` — a `[batch, n, head_dim]` array whose values are `mx.arange`-derived and therefore distinguishable per row, so a wrong write window is visible rather than masked by zeros.
- `_rows_expected(n: int) -> mx.array` — the `[1, n, head_dim]` array that `_rows(n)` produces, for asserting a snapshot's contents.
- `_pool_via_concatenate(chunks: list[mx.array]) -> mx.array` — appends every chunk using the original unpatched `update_pool`, returning the final pool.
- `_pool_via_append(chunks: list[mx.array]) -> mx.array` — the same with the installer applied.
- `_count_allocations(appends: int, rows_each: int) -> int` — counts backing-buffer allocations across `appends` calls by wrapping `mx.zeros` with a counter for the duration.

- [ ] **Step 2: Write the failing tests**

```python
def test_pool_view_reports_the_logical_length() -> None:
    """Break this catches: exposing raw capacity as pool.shape[1], which would
    make V4Attention at deepseek_v4.py:2138 attend to uninitialised rows."""
    cache = _new_cache()
    cache.update_pool(_rows(3), key="compressor")
    cache.update_pool(_rows(2), key="compressor")
    assert cache.get_branch("compressor").pool.shape[1] == 5


def test_append_matches_concatenate_exactly() -> None:
    """Break this catches: an off-by-one in the write window, or growth copying
    the wrong region. The appended pool must be bit-identical to the original."""
    reference = _pool_via_concatenate([_rows(3), _rows(2), _rows(7)])
    patched = _pool_via_append([_rows(3), _rows(2), _rows(7)])
    assert mx.array_equal(reference, patched)


def test_a_snapshot_survives_a_later_append() -> None:
    """Break this catches: the snapshot holding a view onto a buffer that later
    regrows or is written past. The prefix cache reuses 284 of 286 tokens in
    production, so a stale or mutated snapshot is a correctness defect."""
    cache = _new_cache()
    cache.update_pool(_rows(4), key="compressor")
    snapshot = _copy_compressor_branch(cache.get_branch("compressor"))
    cache.update_pool(_rows(64), key="compressor")
    assert mx.array_equal(snapshot.pool, _rows_expected(4))


def test_ragged_pool_lengths_use_the_original_path() -> None:
    """Break this catches: the append path swallowing the variable-length branch,
    which builds a per-row merged buffer with different semantics."""
    cache = _new_cache()
    cache.get_branch("indexer")._new_pool_lengths = [2, 5]
    cache.update_pool(_rows(5, batch=2), key="indexer")
    assert cache.pooled_lengths("indexer") == [2, 5]


def test_growth_is_geometric_not_linear() -> None:
    """Break this catches: reverting to per-append reallocation, which restores
    the quadratic behaviour this task exists to remove."""
    allocations = _count_allocations(appends=64, rows_each=1)
    assert allocations <= 8


def test_patch_is_idempotent() -> None:
    """Break this catches: double application wrapping the wrapper."""
    install_deepseek_v4_pooled_append()
    first = dsv4.DeepseekV4Cache.update_pool
    install_deepseek_v4_pooled_append()
    assert dsv4.DeepseekV4Cache.update_pool is first
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_pooled_append.py -v
```

Expected: FAIL on `ImportError` for `install_deepseek_v4_pooled_append`.

- [ ] **Step 4: Implement the installer per the approved Step 1 design**

Only the uniform path (`new_lengths is None`) changes. The ragged path is left exactly as upstream wrote it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_pooled_append.py -v
```

Expected: PASS.

- [ ] **Step 6: Mutation-test the new tests against pristine code**

Temporarily disable the installer call and confirm which of the Step 2 tests fail. `test_append_matches_concatenate_exactly` and `test_ragged_pool_lengths_use_the_original_path` are expected to still pass unpatched — they assert equivalence. `test_growth_is_geometric_not_linear` **must** fail. Record the result. In Task 5 of the prompt-protocol work, 8 of 14 tests were green on unfixed code while claiming to catch the bug; this step exists so that cannot recur silently.

- [ ] **Step 7: Wire it into the loader and run the offline gate**

Import and call after `install_deepseek_v4_indexer_shortcut()`. Add the source-level call-site test.

```bash
just lint && just fmt && just check && just test
```

Expected: the prefix-cache tests in particular must be unchanged. Name them explicitly in the report.

- [ ] **Step 8: Live verification**

```bash
uv run python scripts/validate_dsv4_live_api.py --self-test
uv run python scripts/validate_dsv4_live_api.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --determinism 3 --max-tokens 8192
uv run python scripts/instance_window_battery.py \
  --base-url http://localhost:52415 \
  --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
  --out /tmp/omlx-sync-s4-windows.json
for i in 1 2 3; do
  uv run python scripts/measure_dsv4_throughput.py \
    --base-url http://localhost:52415 \
    --model Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp \
    --prompt-tokens 546 --max-tokens 8192
done
```

Additionally, run a prefix-cache reuse check with a negative control: two requests sharing a long prefix must report a high `cached_tokens`, and an unrelated prompt must report exactly 0.

Pass criteria against Task 1's baseline:
- determinism: three identical terminating completions
- `cold_prefill_32k_seconds`: measurably improved with `cached_tokens=0`
- `decode_tokens_per_second`: within 3% of baseline median
- prefix-cache reuse: high on the shared prefix, exactly 0 on the negative control

- [ ] **Step 9: Commit**

```bash
git add src/exo/worker/engines/mlx/deepseek_v4_0731_model.py \
        src/exo/worker/engines/mlx/deepseek_v4_0731_loader.py \
        src/exo/worker/tests/unittests/test_mlx/test_dsv4_pooled_append.py
git commit -m "refactor: Grow the V4 pooled cache in place"
```

**Claims prohibited without evidence:** that snapshots are unaffected (paste the prefix-cache reuse figures with the negative control); that prefill improved (difference against Task 1's recorded seconds); that the new tests catch anything (paste the Step 6 mutation result).

**Revert condition:** if `cold_prefill_32k_seconds` does not improve measurably, revert. This change buys complexity in a load-bearing cache and is only worth it if it pays.

---

## Cross-check protocol

Two rounds per task, run by the orchestrator, before the next packet is written.

**Round one — against the packet.** Scope: did the diff do only what was asked? `RULES.md` conformance: exhaustive typing, `Literal` over enums, no three-letter acronyms, no new dependencies, error-handling rationale in docstrings. Evidence: does every claim in the report have pasted output behind it?

**Round two — adversarial, independent of the report.** What would make this wrong? Which new test would still pass on unfixed code? What did the packet fail to ask for? The project's record is that round two repeatedly found more than round one, including self-introduced regressions.

A task is complete when both rounds pass and the gate output is in hand — not when the agent says it is.

## Not in this plan

Vendoring OMLX's DeepSeek-V4 runtime or Metal kernels. Changing the MLX or `mlx_lm` pins. Phase-two MTP or DSpark. Tool calling, which is blocked on sourcing a tool-capable checkpoint rather than on code. The Group III phase-two facts are recorded in the design document and are not implemented here.
