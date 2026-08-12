# Prefill Admission Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refuse prompts above the configured admission ceiling before model or cache allocation on every MLX prefill path.

**Architecture:** Keep the environment-backed ceiling in `exo.shared.constants`, and keep the typed error and pure comparison helper in a small MLX generator admission module. Call the helper at the three earliest complete-token boundaries: normal generation, batch generation, and the disaggregated prefill server.

**Tech Stack:** Python, MLX, pytest, Ruff, BasedPyright, uv with `--no-sync`.

## Global Constraints

- Default ceiling is 32,768 tokens and `EXO_PROMPT_ADMISSION_CEILING <= 0` disables the guard.
- Refusal is a hard error; do not truncate prompts or change HTTP status handling.
- Never issue a prompt of 64K tokens or more and do not run `scripts/instance_window_battery.py`.
- Use `uv run --no-sync` for every uv command.
- Do not edit `scripts/`, MLX pins, `mlx_lm` pins, or `install_deepseek_v4_sdpa_float32()`.
- Do not expose `use_prefix_cache` on `ChatCompletionRequest`.
- Preserve the pre-existing untracked model card and all unrelated worktree changes.
- Follow `RULES.md`: strict typing, descriptive names, no new dependencies, and error-handling rationale in docstrings.

---

### Task 1: Add the failing admission tests

**Files:**
- Create: `src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py`

**Interfaces:**
- Consumes: `PromptTooLongError`, `assert_prompt_within_ceiling`, and `PROMPT_ADMISSION_CEILING_DEFAULT` from the admission module.
- Produces: focused unit coverage for comparison semantics, diagnostics, disabled configuration, normal generation, batch generation, and the disaggregated prefill server.

- [ ] **Step 1: Write the module docstring and pure-helper tests**

  Include the exact break each test catches, then add tests for rejection above the ceiling, admission exactly at the ceiling, both numeric diagnostics, disabled non-positive ceilings, and the 32,768 default.

- [ ] **Step 2: Add one synthetic call-site test for each of the three admission boundaries**

  Use synthetic token arrays and monkeypatch model/cache/prefill operations so an over-limit request raises before any model invocation or cache allocation. Keep all synthetic values well below 64K.

- [ ] **Step 3: Run the focused test file and verify the expected missing-symbol failures**

  Run `uv run --no-sync pytest src/exo/worker/tests/unittests/test_mlx/test_dsv4_prefill_admission.py -v` and confirm it fails because the production admission symbols and call sites do not yet exist.

### Task 2: Implement the admission boundary

**Files:**
- Create: `src/exo/worker/engines/mlx/generator/admission.py`
- Modify: `src/exo/shared/constants.py`
- Modify: `src/exo/worker/engines/mlx/generator/generate.py`
- Modify: `src/exo/worker/engines/mlx/generator/batch_generate.py`
- Modify: `src/exo/worker/engines/mlx/disaggregated/serve.py`

**Interfaces:**
- Consumes: `EXO_PROMPT_ADMISSION_CEILING` from shared constants and prompt token counts at each boundary.
- Produces: `PromptTooLongError(prompt_tokens: int, limit_tokens: int)`, `assert_prompt_within_ceiling(prompt_tokens: int, limit_tokens: int) -> None`, and `PROMPT_ADMISSION_CEILING_DEFAULT == 32_768`.

- [ ] **Step 1: Add the environment-backed ceiling**

  Define `PROMPT_ADMISSION_CEILING_DEFAULT = 32_768` and parse `EXO_PROMPT_ADMISSION_CEILING` at module scope with that default. A value at or below zero disables the guard.

- [ ] **Step 2: Add the typed error and pure assertion helper**

  Carry both counts as typed attributes, repeat both in the message, and document why raising is required: the refusal protects host availability and must not silently alter the caller’s prompt; the client receives only an opaque HTTP 500 error message.

- [ ] **Step 3: Add the three guards before allocation or model execution**

  Guard the final normalized prompt in `mlx_generate()` and `ExoBatchGenerator.submit()` before `make_kv_cache`, and guard normalized request tokens in `run_prefill_for_request()` before prefix-cache lookup or cache creation. Do not add redundant downstream guards in `prefill()` or `remote_prefill()`.

- [ ] **Step 4: Run the focused tests and confirm green**

  Run the same focused pytest command and record the exact passing count.

### Task 3: Mutation-test the regression suite

**Files:**
- Modify temporarily: the implementation and selected call-site guard lines only; restore all mutations before continuing.

- [ ] **Step 1: Mutate `>` to `>=` and run the focused test file**

  Confirm the exact-at-ceiling test fails.

- [ ] **Step 2: Mutate the default to `65_536` and run the focused test file**

  Confirm the default-ceiling test fails.

- [ ] **Step 3: Remove each chosen guard in turn and run the focused test file**

  Confirm the corresponding path test fails before restoring the guard.

- [ ] **Step 4: Restore the implementation and rerun the focused test file**

  Confirm all focused tests pass after all mutations are removed.

### Task 4: Run the offline verification gate

**Files:**
- No additional files unless formatting reports a changed-file issue.

- [ ] **Step 1: Run Ruff**

  Run `uv run --no-sync ruff check src` and `uv run --no-sync ruff format --check src`. If format check fails only on changed files, run `uv run --no-sync ruff format src` and review the resulting diff.

- [ ] **Step 2: Run strict BasedPyright**

  Run `uv run --no-sync basedpyright --project pyproject.toml` and record the error, warning, and note counts.

- [ ] **Step 3: Run the full repository tests**

  Run `uv run --no-sync pytest src`, record the new total and skipped/deselected counts, and verify no pre-existing test changes status.

### Task 5: Perform safe live verification and commit

- [ ] **Step 1: Re-verify both cluster nodes before any request**

  Use freshly derived Thunderbolt addresses, explicit `EXO_BOOTSTRAP_PEERS`, and verify MLX version and core SHA on both nodes. Do not touch the prohibited worktrees.

- [ ] **Step 2: Exercise only the safe small-ceiling refusal**

  Start with `EXO_PROMPT_ADMISSION_CEILING=2048`, send roughly 4,000 tokens, and retain the HTTP 500 response body with both counts visible. Never send 64K or larger.

- [ ] **Step 3: Exercise supported 16K and 32K cold-prefill paths**

  Restart with the default ceiling and record timings plus `cached_tokens=0` for both requests. If the cluster or runbook is unavailable, report the live-verification blocker rather than inferring evidence.

- [ ] **Step 4: Review status and commit only the requested files**

  Run `git diff --check` and `git status --short`, preserve the untracked model card, then stage the changed source/test files and commit with `feature: Refuse over-ceiling prompts at admission`.
