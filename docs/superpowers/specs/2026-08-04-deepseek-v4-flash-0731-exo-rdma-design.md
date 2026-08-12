# DeepSeek-V4-Flash-0731 OMLX Checkpoint Support in EXO

**Status:** Revised design (revision 4, 2026-08-04) — supersedes revisions 1 through 3. See [Revision history](#revision-history).

**Date:** 2026-08-04

**Target checkpoint:** `Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp` (confirmed)

**Working revisions:** EXO `a6cd2fce`; OMLX `2450a53c`

**Reference reconciliation:** `50846648..2450a53c` (51 commits), triaged in
`2026-08-11-omlx-upstream-reconciliation-design.md`. `chat_template_v4.py` and
`tool_parser_v4.py` are byte-identical across the range, so the prompt protocol
and the DSML reference parser are unaffected.

The 16 committed goldens in
`src/exo/worker/tests/fixtures/deepseek_v4_0731_prompt_goldens.json` still record
`omlx_git_sha: 50846648…` and are deliberately **not** regenerated.
`generate_dsv4_0731_prompt_goldens.py:239-241` loads exactly one file from the
OMLX tree — `chat_template_v4.py` — which is byte-identical across the range, so
regeneration can change nothing but the recorded sha (written at `:324`). Keep
the local `omlx` working tree at `50846648`: checking it out at `2450a53c` and
regenerating would rewrite all 16 goldens in that single metadata field and
nothing else.

**Checkpoint inspected:** yes — see [Appendix A](#appendix-a-checkpoint-facts-verified). All layout and quantization claims below are measured, not inferred.

## Summary

Add a narrowly scoped DeepSeek-V4-Flash-0731 compatibility layer to EXO's MLX engine so that OMLX-produced, pre-quantized checkpoints can use EXO's existing tensor-parallel JACCL/RDMA cluster runtime. The first milestone supports backbone inference, DeepSeek-0731 prompting, thinking, tool calls, the dashboard, and OpenAI-compatible APIs across two directly connected Macs.

The implementation must recognize but intentionally not execute the checkpoint's DSpark/MTP tensors in phase one. It must report that limitation explicitly. Distributed MTP is a separate phase-two project.

OMLX remains unchanged and acts as the reference for the checkpoint layout, the numerics of the 0731 architecture, and the DeepSeek-0731 prompt protocol. EXO remains the authority for discovery, placement, worker lifecycle, tensor sharding, JACCL initialization, serving, and cluster observability.

## Context

The target checkpoint is a roughly 154 GiB (`storage_size = 165828909712`) DeepSeek-V4-Flash-0731 model with 43 backbone layers and additional `mtp.*` weights. Its quantization is **mixed, not uniform** — see [Quantization reality](#quantization-reality). A complete local copy exists on each cluster node under:

```text
/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp
```

EXO resolves that layout when launched with:

```bash
EXO_DEFAULT_MODELS_DIR=/Users/jared/.cache/huggingface/hub/models uv run exo
```

A model card already exists at `resources/inference_model_cards/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp.toml` declaring `n_layers = 43`, `hidden_size = 4096`, `num_key_value_heads = 1`, `supports_tensor = true`, `base_model = "DeepSeek V4 Flash"`, and `capabilities = ["text", "thinking", "thinking_toggle"]`. It does not advertise MTP acceleration and requires no change for phase one — see [Model card](#model-card-no-change-required).

### What EXO's pinned mlx-lm already does

EXO does **not** depend on upstream `mlx-lm`. It pins a purpose-built fork:

```
mlx-lm = { git = "https://github.com/rltakashige/mlx-lm?branch=leo/deepseek-v4" }
```

(`pyproject.toml`, resolved at `uv.lock:543`.) That fork's `Model.sanitize()` at `.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py:2604-2760` already implements, deliberately and with explanatory comments:

- dropping `mtp.*` and any `layers.N` beyond `num_hidden_layers` (`:2619-2634`) — an intentional backbone-only decision in that fork, not an accident;
- F8_E8M0 scale decode, FP8 block dequantization to bf16, and FP4-packed routed-expert reinterpretation as MLX mxfp4 (`:2636-2688`);
- top-level remapping of `embed.weight`, `norm.weight`, `head.weight`, and `hc_head_{fn,base,scale}` (`:2690-2699`);
- `layers.` → `model.layers.` prefixing, `ffn.gate.bias` → `e_score_correction_bias`, `shared_experts.w{1,2,3}` → `{gate,up,down}_proj`, and underscore-form hyper-connection remapping `.hc_{attn,ffn}_{fn,base,scale}` → `.hc_{attn,ffn}.{...}` (`:2701-2716`);
- fusing `attn.wq_a` + `attn.wkv` → `attn.wqkv_a` and, for **both** `attn.compressor` and `attn.indexer.compressor`, `wkv` + `wgate` → `wkv_gate`, concatenating `weight`/`scales`/`biases` on axis 0 (`:2719-2734`);
- stacking per-expert `w{1,2,3}` weights and scales into the `SwitchGLU` layout (`:2736-2759`).

It also implements hash-routed MoE layers (`tid2eid`, `num_hash_layers = 3`, `:2352-2363` and `:2410`), attention sinks, the compressor's absolute positional embedding `ape`, and the `sqrtsoftplus` scoring path — so those are not compatibility concerns.

The consequence for this project is that the required weight transformation is a **small delta** against that function, not a fresh implementation. Revision 1 of this design specified four mapping rows of which three were already present upstream, while omitting the one transformation that actually breaks the load. Section [3](#3-weight-and-quantization-normalization) is now written as a delta.

### Why the checkpoint's key space is what it is

`Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp` is an oQ-requantized repository. oQ runs the source model's `sanitize()` *before* writing shards (`omlx/omlx/oq.py:3479-3499`, `:2107-2117`), so the checkpoint on disk is in OMLX **post-sanitize** space rather than raw DeepSeek HF space. Every difference this project must absorb follows from that one fact, and the differences are asymmetric because OMLX and EXO's fork are independent implementations of the same architecture — OMLX's is a 1:1 port of `mlx-lm` PR #1192 (`omlx/omlx/patches/deepseek_v4/README.md:1-11`); EXO's fork is separate work.

All rows below are confirmed against the checkpoint on disk (Appendix A).

| Aspect | On disk (measured) | EXO pinned fork expects | Action |
| --- | --- | --- | --- |
| Key prefix | `model.layers.N.*`, `model.embed_tokens.*`, `lm_head.*` | same | none — EXO's top-level remap and prefixing no-op harmlessly |
| Hyper-connections | dotted `attn_hc.{fn,base,scale}` / `ffn_hc.*` | `hc_attn.*` / `hc_ffn.*` | **rename required** (EXO rewrites only the underscore form) |
| `attn.wq_a` / `attn.wkv` | separate, unfused | fused `attn.wqkv_a` | none — EXO's `_fuse_pair` handles it; arithmetic verified in Appendix A |
| compressor / indexer.compressor `wkv` / `wgate` | separate, unfused | fused `wkv_gate` | none — EXO's `_fuse_pair` handles both, including `biases` |
| `attn.wo_a` | 3D `[8, 1024, 1024]` weight, `[8, 1024, 128]` scales | 2D `[8192, 1024]` / `[8192, 128]` | **reshape required** |
| Routed experts | already stacked into `ffn.switch_mlp.*` | stacks from `ffn.experts.N.*` | none — EXO's stacking loop no-ops, pre-stacked keys pass through |
| `ffn.gate.bias` | already `e_score_correction_bias` | same | none |
| `ffn.gate.tid2eid` | `I32 [129280, 6]` | `mx.int32` | none — dtype already matches |
| `quantization` dict keys | unfused module paths (`...attn.wq_a`, `...compressor.wkv`) | fused model-tree paths (`...attn.wqkv_a`, `...compressor.wkv_gate`) | **remap required** |

### Quantization reality

Revisions 1 and 2 described the checkpoint as "4-bit affine" and "mixed" respectively. The measured recipe is mixed and — this is the important part — **the global setting is correct for zero modules**:

| Setting | Count | Modules |
| --- | --- | --- |
| global fallback `{mode: affine, bits: 4, group_size: 64}` | **0** | nothing. It is a decoy. |
| `{mode: mxfp8, bits: 8, group_size: 32}` | 389 | `attn.{wq_a,wq_b,wkv,wo_a,wo_b}`, `attn.indexer.wq_b`, `ffn.shared_experts.*_proj` |
| `{mode: affine, bits: 8, group_size: 64}` | 147 | `attn.compressor.{wkv,wgate}`, `attn.indexer.compressor.{wkv,wgate}`, `attn.indexer.weights_proj`, `lm_head`, `model.embed_tokens` |
| `{mode: mxfp4, bits: 4, group_size: 32}` | 138 | `ffn.switch_mlp.{gate,up,down}_proj` |

All 674 quantized modules carry an explicit per-module override, and every module with a `.scales` tensor has one — the two sets match exactly, with no gaps in either direction (Appendix A). So the global `affine/4/64` is never the right answer for anything. Any module whose per-module key fails to match the model tree silently receives the wrong **mode and the wrong bit width**.

Note also that three modes coexist *within a single attention module*: `attn.wqkv_a` must be mxfp8/32 while `attn.compressor.wkv_gate` must be affine/8/64 (the latter carrying `biases`, which the mx formats do not have). No single global setting can serve both, so the per-module dict is load-bearing rather than an optimisation.

EXO's fork hard-quantizes routed experts to mxfp4/32/4 in `DeepseekV4MoE.__init__` (`deepseek_v4.py:2441-2453`), unconditionally, before `load_model`'s `nn.quantize` runs. That happens to match the checkpoint exactly, so experts are safe by construction. Everything else depends on the per-module `quantization` dict surviving the same path rewrite as the tensors.

The trap is visible in `mlx_lm/utils.py:429-436`: the per-module lookup is `if p in config["quantization"]` keyed on the **model-tree** path, with a silent fallback to `f"{p}.scales" in weights` plus the global mode. A `wqkv_a` that should be mxfp8/8/32 will be quantized affine/4/64 with no error raised anywhere.

Critically — and this is why strict loading alone is insufficient — **`bits` and `group_size` mismatches change tensor shapes and are caught by strict loading; `mode` mismatches do not.** mxfp8 and affine-8 at the same group size produce identically shaped `weight` and `scales`, load cleanly, and compute wrong results silently. Phase one therefore requires an explicit post-quantization mode audit (§3).

### Consequence: EXO's fused attention fast paths will all be disabled

EXO's fork carries five hand-fused kernels for the attention hot path, each gated on `mode == "mxfp4"` and falling back to the generic path otherwise: the `wqkv_a` matmul-plus-split-plus-two-RMSNorms (`deepseek_v4.py:2082`), the `wq_b` matmul-plus-per-head-norm (`:2097`), the chained `wo_a`+`wo_b` projection (`:2216-2219`), and the compressor `wkv_gate` split matmul (`:1588`).

Under this checkpoint's recipe those modules are mxfp8 and affine-8, so **every one of those fast paths falls back**. The guards are correct — this is a throughput consequence, not a correctness bug — but it means EXO's DSV4 performance characteristics were tuned for a uniformly-mxfp4 checkpoint and will not be realised here. Phase one should record this explicitly alongside its throughput observations rather than treating the numbers as representative, and adding mxfp8/affine variants of these kernels is a candidate follow-up independent of MTP.

### Why the compatibility boundary belongs in EXO

Two structural facts, both worth recording so they are not relitigated:

1. **OMLX has model-level tensor-parallel primitives, but not EXO's cluster control plane.** Its current DSV4 `Model.shard()` (`omlx/.../deepseek_v4_model.py:2151-2179`) shards attention, shared experts, and routed experts and assigns the relevant `sharding_group`. OMLX does not provide EXO's automatic discovery, placement, distributed worker lifecycle, JACCL device-matrix construction, dashboard, or multi-API orchestration. "Add clustering to OMLX" therefore means integrating and validating those systems around its model-level TP implementation, not writing DSV4 TP from scratch.
2. **EXO's `DeepseekV4ShardingStrategy` identifies work by literal attribute names walked off the live module tree**, not by config fields and not by isinstance checks below the top level (`src/exo/worker/engines/mlx/auto_parallel.py:827-939`; dispatch at `:516-517`). It reads `attn.n_heads`, `attn.head_dim`, `attn.n_groups`, `attn.wq_b`, `attn.attn_sink`, `attn.wo_a`, `attn.wo_b`, `layer.ffn`, `ffn.shared_experts.*_proj`, and `ffn.switch_mlp.*_proj`, and raises `ValueError(f"Unsupported model type: {type(model)}")` at `:599` for anything else.

Subclassing EXO's pinned `Model` is the narrowest approach that preserves EXO's existing distributed integration and module-tree contract. Vendoring OMLX's runtime is possible, but it would require routing EXO around `DeepseekV4ShardingStrategy` to OMLX's `Model.shard()`, adapting lifecycle and dependency boundaries, and validating OMLX's custom kernels within EXO. That remains a larger phase-one change than the measured compatibility delta.

## Hardware and deployment constraints

The initial cluster consists of:

- An M5 Max MacBook Pro with 128 GiB RAM, using `/Users/jared/aishit/exo-install/exo`.
- An M4 Max MacBook Pro with 128 GiB RAM, reachable as `jared@Retirement-Plan.local`, using `/Users/jared/aishit/exo`.
- A direct Thunderbolt 5 connection.
- macOS 26.5.2 build 25F84 on both nodes.
- `rdma_ctl` enabled on both nodes.
- The same EXO base revision and local model copy on both nodes.

Thunderbolt and RDMA interface names differ across the machines. The implementation and launch procedure must use EXO's topology discovery and generated JACCL device matrix. They must not hard-code `en*` or `rdma_en*` names.

### Memory expectations

Safetensors-header accounting puts the phase-one backbone at 144.26 GiB: 140.87 GiB of tensors handled by the existing sharding strategy and 3.39 GiB replicated. At two ranks this is approximately **73.83 GiB of stored weights per node** before caches, compiled kernels, and transient loading allocations. The skipped MTP tensors add another 10.18 GiB on disk. The backbone alone therefore cannot be used as a credible full-model single-node reference on either 128 GB Mac.

Steady state should land near 75–80 GiB per node. Three caveats must be monitored rather than designed away:

- `get_weights_size` divides `storage_size` by world size for tensor shards (`utils_mlx.py:71-81`) and feeds `set_wired_limit_for_model`. Replicated tensors are not divided, so the wired limit is understated.
- Sharding forces each **full unsharded layer** resident before slicing (`mx.eval(layer.parameters())` at `auto_parallel.py:916`), dominated by the stacked routed experts at roughly 3 GiB per layer at 4 bits. `mx.eval(layer)` plus `mx.clear_cache()` at `:935-936` bound the transient to about one layer.
- The KV cache is not sharded (`cache.py:145-166`), so tensor parallelism buys nothing on KV memory. DSV4's `sliding_window = 128` plus the compressed pool keeps this small, but it is the term that grows with the 32K target.

## Goals

1. Strictly load the target's complete backbone from its existing OMLX checkpoint layout without creating a converted checkpoint copy.
2. Reuse EXO's existing `DeepseekV4ShardingStrategy` and JACCL/RDMA runtime, permitting only narrowly scoped correctness fixes demonstrated by the repaired tensor-parallel tests.
3. Serve deterministic two-rank tensor-parallel inference through the dashboard, Chat Completions API, and Responses API.
4. Demonstrate numeric parity against OMLX on representative layers loaded from the real checkpoint before trusting cluster output, with an optional full-model two-rank differential run after cluster bring-up.
5. Match OMLX's DeepSeek-V4-Flash-0731 prompt rendering for thinking and tool-use conversations, including the append-only invariant that keeps prompt caches valid.
6. Validate stable generation through a 32K-token prompt.
7. Fail early and diagnostically for incompatible checkpoints, quantization mappings and modes, missing backbone weights, invalid shard geometry, or non-RDMA placement.
8. Preserve a clear future path to distributed DSpark/MTP without pretending it is active in phase one.

## Non-goals

- Executing or distributing DSpark/MTP in phase one.
- Adding EXO-style clustering to OMLX.
- Replacing EXO's MLX engine with OMLX's server runtime.
- Rewriting the checkpoint on disk or requiring a second 154 GiB converted copy.
- Hard-coding support to only one Hugging Face repository ID.
- Replacing `DeepseekV4ShardingStrategy` or the JACCL initialization architecture. Narrow correctness fixes are allowed only when a non-degenerate parity test demonstrates the need.
- Treating ring transport or pipeline sharding as successful RDMA validation.
- Guaranteeing a throughput target on the heterogeneous M4/M5 pair.

## Recorded invariants

Two properties make this plan sound and both look like bugs on casual reading. They are recorded here so they are not "fixed".

**Replicated hyper-connections are correct.** `hc_attn`/`hc_ffn` (and `hc_head`) are left fully replicated in fp32 and no collective touches them. That is right, because the residual stream is full-hidden on every rank by construction: `_AllSumLinear` all-sums the partial `wo_a` output before applying the replicated `wo_b` (`auto_parallel.py:822-824`), and `ShardedMoEV4` all-sums the MoE output (`:778-784`). Two collectives per layer, both restoring the full hidden state, so `hc_pre`/`hc_post` observe identical inputs on all ranks. Adding a gather for hyper-connection parameters would be wrong and would cost bandwidth.

**Compressor and indexer are intentionally replicated.** Neither `attn.compressor` nor `attn.indexer` (including `indexer.wq_b`, `indexer.weights_proj`, `indexer.compressor`) is sharded; the compute is duplicated on every rank. Likewise `wqkv_a`, `q_norm`, `kv_norm`, `rope`, `embed_tokens`, `lm_head`, `norm`, and `ffn.gate`. This is the existing strategy's design, phase one does not change it, and the replicated router is what guarantees identical `inds`/`weights` across ranks.

## Selected approach

Introduce an EXO-native model loader wrapper and a DeepSeek-V4-0731 compatibility model subclass. The wrapper detects compatible 0731 checkpoints, supplies normalized configuration to `mlx_lm`, and selects the compatibility subclass through `load_model`'s `get_model_classes` callback. The subclass performs the weight-layout delta through `sanitize()` before strict weight loading, then the wrapper audits the resulting quantization state.

All other models delegate to the current `mlx_lm.utils.load_model` behavior unchanged. This avoids a process-wide monkey patch and confines the compatibility risk to positively identified DeepSeek-V4-0731 checkpoints.

The rejected alternatives were:

1. Vendoring OMLX's full DeepSeek-V4 runtime and Metal kernels into EXO. Beyond the size of the transplant, this requires bypassing EXO's current DSV4 strategy in favor of OMLX's `Model.shard()` and integrating a second runtime boundary. It remains a candidate for phase-two MTP.
2. Adding clustering to OMLX. OMLX has model-level DSV4 sharding, but this still requires integrating EXO-equivalent discovery, placement, JACCL orchestration, worker lifecycle, dashboard, and API serving.

## Component design

### 0. Checkpoint verification — complete

Done. See [Appendix A](#appendix-a-checkpoint-facts-verified) for measured layout, shapes, dtypes, and the full quantization recipe. D1 and D2 are confirmed necessary; D3 is confirmed necessary and more consequential than initially assumed; D4 captures the independently verified shared-expert activation divergence; and the proposed `tid2eid` cast, now numbered D5, is confirmed unnecessary. Re-run Appendix A's script if the checkpoint is ever re-quantized or replaced, and treat any drift as a design change rather than an implementation detail.

### 1. Compatibility detection

Add a small model-specific compatibility module under `src/exo/worker/engines/mlx/`. It will inspect a checkpoint's configuration and weight index before model construction.

Detection requires `model_type == "deepseek_v4"` plus 0731/DSpark structural evidence. The correct discriminator is OMLX's: `dspark_block_size > 0` **and** `dspark_target_layer_ids` non-empty (`omlx/.../deepseek_v4_model.py:90-94`; also `deepseek_v4_dspark.py:28-32`). Do **not** key off `num_nextn_predict_layers` — OMLX's own comment at `:158-160` records that 0731 deliberately leaves that legacy field set for backward compatibility. 0731 weight-path evidence (dotted `attn_hc`, 3D `wo_a`, `mtp.<s>.markov_head.*`) is an acceptable secondary signal. The repository ID is useful in diagnostics but is not a detection requirement.

Note that `BaseModelArgs.from_dict` filters unknown keys (`mlx_lm/models/base.py:13-21`), so the `dspark_*` fields are silently dropped before `ModelArgs` construction and are invisible downstream. The detection module is therefore the only place that can observe and report "this checkpoint carries a DSpark drafter that we are not executing".

Detection returns one of:

- Not applicable: delegate to normal loading.
- Supported 0731 backbone layout: normalize and load.
- DeepSeek-V4-like but incompatible: fail with a targeted explanation rather than delegating to permissive loading.

The detection and normalization functions should be pure Python wherever possible so they can be tested without initializing Metal or loading tensor shards.

### 2. Loader integration

Add `load_exo_model()` alongside EXO's MLX loading utilities and replace both direct `load_model()` calls — `utils_mlx.py:175` (single-node) and `utils_mlx.py:238` (distributed), both currently `strict=False`.

For an ordinary model, the wrapper passes through the existing arguments and preserves current strictness behavior. For a recognized 0731 model, it will:

1. Load and copy the configuration.
2. Normalize configuration and quantization paths in memory (§3), asserting `len(compress_ratios) >= num_hidden_layers`. The checkpoint carries 46 entries for 43 backbone layers (the extra three are the DSpark stages) and EXO's fork bounds-checks its indexing at `deepseek_v4.py:1984` — `ratios[layer_id] if layer_id < len(ratios) else 0` — so a longer list is safe. The assertion exists to catch a *short* list, which would silently degrade compressed layers to local attention.
3. Call `mlx_lm.utils.load_model()` with the normalized configuration and a `get_model_classes` callback that returns the compatibility model.
4. Require strict weight loading after the intentional transformations.
5. Audit realized quantization state against the checkpoint's declared per-module settings (§3), because strict loading cannot see `mode`.
6. Run the shard-geometry pre-flight checks (§5) before sharding is attempted.
7. Attach or log a compatibility status identifying backbone-only mode and the skipped MTP tensor count.

This hook occurs before tensor sharding. Each node loads the compatible model from its local checkpoint, after which EXO applies its existing tensor-parallel strategy.

### 3. Weight and quantization normalization

The compatibility model subclasses EXO's pinned DeepSeek-V4 `Model` so existing `isinstance` dispatch into `DeepseekV4ShardingStrategy` continues to work (`auto_parallel.py:516-517`).

Its `sanitize()` performs a 0731 pre-normalization pass and then calls the existing DeepSeek-V4 sanitizer. **The pre-pass is a delta against `deepseek_v4.py:2604-2760`, not a reimplementation.** Everything that function already does — the fusions, expert stacking, top-level remapping, MTP filtering, FP8/FP4 handling — must not be duplicated.

The required delta, as verified against the checkpoint and both model implementations, is three weight/config transformations plus one constructor correction:

| # | Transformation | Status | Rationale |
| --- | --- | --- | --- |
| D1 | `*.attn_hc.{fn,base,scale}` → `*.hc_attn.{...}`; `*.ffn_hc.*` → `*.hc_ffn.*` | **required** | EXO rewrites only the underscore form `.hc_attn_fn` (`:2711-2713`), so the dotted form on disk passes through untouched and becomes an unexpected key under strict loading. The checkpoint already stores these tensors as `F32`; `cast_predicate` is not part of this active load path, so strict key matching is the relevant protection. |
| D2 | `attn.wo_a.{weight,scales}` reshaped `[8, 1024, 1024]` → `[8192, 1024]` and `[8, 1024, 128]` → `[8192, 128]`, i.e. `reshape(o_groups * o_lora_rank, -1)` | **required** | EXO declares `wo_a` as a 2D `nn.Linear(group_feat, n_groups * o_lora_rank)` (`:1999`) and reshapes back to 3D inside the forward pass (`:2049-2054`) — the exact inverse. No `biases` exist on this module (mxfp8). |
| D3 | Per-module `quantization` keys rewritten through the same fusion table EXO applies to tensors: `attn.wq_a` + `attn.wkv` → `attn.wqkv_a`, `{compressor,indexer.compressor}.wkv` + `.wgate` → `.wkv_gate` | **required, and the highest-risk item** | `mlx_lm/utils.py:432` looks per-module settings up by model-tree path. Unrewritten keys never match, and the fallback is a global `affine/4/64` that is correct for **no module in this checkpoint**. Note D1's rename does not affect this table — hyper-connections are unquantized. |
| D4 | Set every backbone `ffn.shared_experts.swiglu_limit` from the checkpoint's `ModelArgs.swiglu_limit` (`10.0`) after base-model construction | **required** | EXO's routed experts use `args.swiglu_limit`, but its shared experts hard-code `0.0` (`:2456-2460`). OMLX uses `config.swiglu_limit` for both (`omlx/.../deepseek_v4_model.py:827-833`). Leaving this divergence in place changes the shared-expert activation on every layer. |
| ~~D5~~ | ~~`tid2eid` cast to `int32`~~ | **not needed** | Measured as `I32 [129280, 6]`, already matching EXO's `mx.int32` declaration (`:2362`). Keep a tolerant assertion rather than a cast. |

Write D1 and D2 to tolerate the already-correct form as a no-op and to log which form was observed, so the adapter survives a future checkpoint written in raw-HF space.

Implement D4 in the compatibility subclass after `super().__init__()` so unrelated DeepSeek-V4 checkpoints retain the pinned fork's existing behavior. Unit and parity tests must exercise inputs large enough for the limited-SwiGLU clamp to affect the result; otherwise `0.0` and `10.0` can appear spuriously equivalent.

Fusion-pair compatibility is satisfied by this checkpoint and the arithmetic has been verified end to end (Appendix A): `wq_a` and `wkv` are both mxfp8/8/32 and concatenate to the `[1536, 1024]` weight EXO declares; both compressor pairs are both affine/8/64 and concatenate to `[1024, 1024]` with `biases` carried through by `_fuse_pair`. The adapter must still **assert** compatible `bits`, `group_size`, and `mode` on each source pair before fusing, and must not fall back to the global setting when explicit child settings disagree — that assertion is what turns a future re-quantization into a clear failure instead of silent corruption.

**Post-quantization mode audit.** After `nn.quantize` and `load_weights`, walk the model tree and assert that every `QuantizedLinear`, `QuantizedSwitchLinear`, and `QuantizedEmbedding`'s realized `(bits, group_size, mode)` triple matches the checkpoint's declared value for that module, with unfused source paths mapped through the D3 table. There are **641 backbone source declarations**, which collapse through 105 compatible fusion pairs into **536 realized quantized module paths**. The remaining 33 per-module entries are under `mtp.*` and are expected to dangle unmatched after MTP filtering, so the audit must ignore that prefix rather than reporting it as a mismatch. Report any divergence as a fatal error naming the module, the expected triple, and the realized one. This is a mandatory gate, not a diagnostic: it is the only mechanism that catches a `mode` mismatch, which loads cleanly and produces wrong numbers.

Transformations should preserve MLX lazy arrays. Apart from the concatenations already required by the pinned fused architecture and D2's reshape, the adapter should avoid materializing or duplicating full tensors.

After normalization, strict loading is mandatory. Recognized MTP keys are the only intentional omissions. Missing model parameters or remaining unexpected non-MTP keys are fatal and the error should include counts and representative paths.

### 4. MTP phase boundary

Phase one identifies `mtp.*` tensors before filtering them. Structured logs must include:

- Compatibility mode, for example `deepseek_v4_0731_backbone`.
- The number of MTP tensors recognized and skipped.
- The observed DSpark configuration (`dspark_block_size`, `dspark_target_layer_ids`), since `from_dict` discards it before `ModelArgs` and nothing downstream can report it.
- A clear statement that DSpark/MTP speculative decoding is disabled.

The model card must not advertise MTP acceleration; it currently does not. Public inference response schemas remain unchanged because MTP affects execution strategy, not the response protocol.

No destructive transformation is applied to the checkpoint, so phase two can reuse the original MTP tensors. Phase two will separately evaluate porting OMLX's DSpark model components, caches, custom kernels, and distributed synchronization into EXO.

### 5. Tensor-parallel and RDMA flow

The distributed data flow is:

```text
model directory resolution
  -> 0731 compatibility inspection (detection + config normalization)
  -> compatibility sanitize() delta
  -> nn.quantize with rewritten per-module quantization paths
  -> strict load_weights on each node
  -> post-quantization mode audit
  -> shard-geometry pre-flight checks
  -> DeepseekV4ShardingStrategy
  -> EXO-generated two-rank JACCL device matrix
  -> JACCL initialization and rank barrier
  -> mx.eval(model) and final barrier
  -> ready model instance
  -> shared dashboard and API serving
```

The two steps revision 1 omitted — `nn.quantize` between `sanitize()` and `load_weights` (`mlx_lm/utils.py:426-496`), and `mx.eval(model)` plus the barrier after sharding (`utils_mlx.py:266-284`) — are exactly where the quantization-mode and shard-geometry failures land, so they are named explicitly.

Placement must select exactly two workers, tensor sharding, and `MlxJaccl`. Both nodes must appear in an RDMA-connected cycle and report `rdma_ctl` enabled. The generated device matrix and coordinator should be logged by the existing distributed initialization.

**Preserve, do not build, the no-downgrade behavior.** EXO already refuses to degrade: `master/placement.py:214-219` raises `"Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"`, `placement_utils.py:321-323` raises `"Current jaccl backend requires all-to-all RDMA connections"` on an incomplete device matrix, and `placement.py:133-137` raises when Tensor is requested for a model with `supports_tensor = false`. The only automatic coercion is single-node → ring/pipeline (`placement.py:246-252`). Phase one must not weaken any of this.

**Shard-geometry pre-flight checks (new).** Two constraints exist in the sharding code but not in placement, so today they fail late and opaquely. Add explicit checks in the loader with clear messages:

- `heads_per_group % world_size == 0`, asserted at `auto_parallel.py:856-859`. For the real config (`num_attention_heads = 64`, `o_groups = 8`) `heads_per_group = 8`, so **world size must divide 8** — tensor parallelism is capped at eight ranks. Placement checks only `hidden_size % len(cycle)` (`placement.py:144-149`), so a larger cluster passes preview and then asserts mid-load.
- Quantized shards on the input dimension go through `mx.split` on both the packed `weight` and the `scales` (`mlx/nn/layers/distributed.py:57-77`), which effectively requires `in_features % (group_size * world_size) == 0`. This is nowhere asserted and currently surfaces as an opaque `mx.split` error inside MLX. Affected modules are the sharded-to-all cases: `attn.wo_a`, `ffn.switch_mlp.down_proj`, `ffn.shared_experts.down_proj`.

Also note that the DSV4 kv-head bypass in placement is a string match on the model card — `base_model.startswith("DeepSeek V4")` at `placement.py:142`. The target card says `"DeepSeek V4 Flash"` and passes, but renaming it would break TP placement with a misleading kv-head error. Any card edit must preserve that prefix.

The validation namespace should be isolated. Note that `EXO_LIBP2P_NAMESPACE` was removed by commit `09f9ea31` (libp2p → zenoh) and now raises `ValueError` at `src/exo/main.py:343-346`:

```bash
EXO_ZENOH_NAMESPACE=dsv4f-0731-validation \
EXO_DEFAULT_MODELS_DIR=/Users/jared/.cache/huggingface/hub/models \
uv run exo
```

Before creating the instance, `/state` and the placement preview must show a valid two-node tensor/JACCL placement. If only ring or pipeline choices are available, validation stops and reports topology diagnostics.

Each node loads its own checkpoint copy and retains its assigned tensor shard. Readiness requires strict loading, the quantization audit, sharding, JACCL initialization, and the final distributed barrier to complete on both ranks.

### 6. DeepSeek-0731 prompt protocol

EXO's vendored encoder (`vendor/deepseek_v4_encoding.py:1-655`) and OMLX's reference (`omlx/omlx/patches/deepseek_v4/chat_template_v4.py:1-681`) share a common vendored core. All special tokens are byte-identical — `<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<think>` / `</think>`, `<｜latest_reminder｜>`, `<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="...">`, `<｜DSML｜parameter name="..." string="...">`, `<tool_result>` — and multi-turn rendering, tool-call block layout, tool-result merging, and out-of-order tool-result sorting already match.

**The work is therefore in EXO's consumer layer (`utils_mlx.py`) and parsers, not primarily in the vendored encoder.** Revision 1 pointed at the wrong file. The required changes, in priority order:

**P1 — mid-conversation system reminders and the append-only invariant.** OMLX implements `relocate_mid_system_messages` (`chat_template_v4.py:742-802`), which reclassifies a mid-conversation `system` run as a `latest_reminder` placed immediately before its associated user turn, merging consecutive entries with `"\n\n"` and returning `None` for shapes it cannot safely rewrite. This preserves the invariant that `omlx/tests/test_deepseek_v4_template_append_only.py:1-8` exists to protect: with reasoning retained, `render(msgs + new_turn).startswith(render(msgs))`, so prompt caches stay valid.

EXO instead hoists **every** system message into a single joined block at index 0 (`consolidate_system_messages`, `utils_mlx.py:515-538`). Each new reminder therefore rewrites the head of the prompt and invalidates the entire KV prefix. `<｜latest_reminder｜>` is referenced only inside the vendored encoder; nothing in `src/exo` ever emits `role: "latest_reminder"`. For agentic workloads this dominates time-to-first-token, so it is the highest-impact item and must be implemented, not merely aligned. Port the relocation logic and its `supports_mid_system_messages` capability probe, and add a regression test asserting the append-only prefix property across successive turns.

**P2 — reasoning effort.** EXO exposes a single `REASONING_EFFORT_MAX` constant keyed to `"max"` and asserts on `"low"` (`deepseek_v4_encoding.py:60-64`, `:276-280`). OMLX has a three-tier `REASONING_EFFORT_PROMPTS` dict with `"low"` as the documented default and a distinct, stronger `"max"` tier (`chat_template_v4.py:71-84`). Combined with EXO's mapping at `utils_mlx.py:493-499`, the live behavior is that an API request for `high` emits **nothing at all**, `xhigh` emits OMLX's `high` text, and OMLX's `max` tier is unreachable. Adopt OMLX's dict and default, and define the supported API value set explicitly; Goal 5 cannot otherwise be stated as met.

**P3 — assistant prefill channel.** EXO pops a trailing assistant message and appends its content raw after the generation anchor, which is `<｜Assistant｜><think>` (`utils_mlx.py:558-561`, `:606-607`), so prefilled *content* is fed to the model as *reasoning*. OMLX's `continue_final_message` path strips the anchor correctly (`chat_template_v4.py:845-849`). Adopt OMLX's behavior.

**P4 — DSML parser tolerance.** `dsml_encoding.py:58-63` hardcodes `name=\"([^\"]+)\">` with no `\s*`, so `<｜DSML｜invoke name="f" >` yields `None` and the whole tool-call block leaks into visible content; OMLX's `_INVOKE_RE` uses `\s*>` (`tool_parser_v4.py:49-54`). OMLX also strips one padding newline from `string="true"` values and applies `ast.literal_eval` to `string="false"` values (`:76-92`); EXO does neither, so `True` arrives as the string `"True"`. Align all three.

**P5 — streaming parser gaps.** `parse_thinking_models` uses a whole-buffer prefix comparison (`model_output_parsers.py:404-406`) rather than the suffix-aware `_could_be_marker_prefix` used on the DSML path (`:351-358`), so a `<think>` fragment arriving after text in the same accumulation window is not held back. Separately, in the `finish_reason` branch, `pending_buffer` is flushed as plain text and the marker test inspects only `response.text`, ignoring `accumulated` (`:281-299`), so a tool-call start marker split across the final two tokens is emitted verbatim as assistant content. Fix both and cover them by test.

**P6 — scoped cleanup only.** Align the V4 marker constants directly exercised by the V4 parser and add regression coverage for literal `<think>` text before changing `_strip_v4_thinking_markers`. Preserve the current generic `developer`-to-`system` consolidation unless a 0731 golden test demonstrates an incompatible rendering. Do not remove or wire the unused `context` plumbing or upstream reference output parser as part of phase one; broad dead-code cleanup is independent work. Any cleanup in this section must be required by a failing phase-one prompt or parser test.

**Correction to revision 1.** Revision 1 required removing EXO's synthetic-system-message tool injection. That is wrong: both sides insert an empty system message when none exists and append the identical `TOOLS_TEMPLATE` to its body (EXO `utils_mlx.py:589-597`; OMLX `chat_template_v4.py:688-718`). Removing it would *introduce* a divergence. OMLX also guards against double-injection when a supplied `context` prefix already carries the schema (`:702-708`); because EXO does not currently use that context path, this is documented rather than expanded into phase-one cleanup. Generic API adapters remain unchanged.

Generated output continues through EXO's streaming reasoning and DSML tool parsers. Both streaming and non-streaming Chat Completions and Responses requests must separate reasoning from visible content, return structured tool calls, set the correct finish reason, and avoid exposing DSML markup as ordinary text. The dashboard uses the same inference path.

### Model card: no change required

`capabilities = ["text", "thinking", "thinking_toggle"]` does not gate tool calling. No `"tools"` capability exists anywhere in EXO; `capabilities` is descriptive, loaded at `shared/models/model_cards.py:169`, echoed on `/v1/models`, and used only for dashboard filtering and badges. Server-side, tool support is gated on `_needs_v4_encoding` — a `"deepseek-v4" in task_params.model.lower()` substring match at `utils_mlx.py:489-490`, which does match `Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp` — plus non-empty `task_params.tools`. The three sibling V4/V3.2 cards declare the same capability set.

`reasoning_dialect = "tool_conditional"` drives only client-config emission for the dashboard (`dashboard/src/routes/integrations/+page.svelte:158-164`). It is never read by the encoder; the tool-conditional retain-reasoning behavior is hardcoded independently at `deepseek_v4_encoding.py:606-607`. The card value and the encoder behavior agree today but nothing enforces that, which is worth a comment rather than a code change in phase one.

## Failure handling

The following conditions fail before generation:

- An unsupported or ambiguous DeepSeek-V4 checkpoint revision.
- `compress_ratios` shorter than `num_hidden_layers`, or containing values outside `{0, 4, 128}`. A longer list is expected and must not fail.
- Incompatible explicit quantization settings for modules that must be fused.
- A realized `(bits, group_size, mode)` triple that disagrees with the checkpoint's declared per-module setting, per the §3 audit.
- Missing required backbone tensors after normalization.
- Unexpected non-MTP tensors after normalization.
- Shard geometry violations: `heads_per_group % world_size != 0`, or a quantized input-dimension shard where `in_features % (group_size * world_size) != 0`.
- Inability to construct a two-way RDMA device matrix.
- JACCL initialization or distributed barrier failure.
- Rank disagreement or failure while loading and sharding.

If either rank fails, the instance must enter EXO's error path and release both workers instead of remaining indefinitely in a loading state. Existing instance error surfaces should carry the root exception. Structured worker logs provide the model compatibility, quantization audit, and MTP status; no new public inference schema is required for phase one.

EXO must not automatically downgrade the validation instance from JACCL to ring or from tensor to pipeline. Such modes may be run deliberately for comparison but never count as successful RDMA acceptance.

## Test strategy

### Pure compatibility tests

Add unit tests for:

- Positive and negative checkpoint detection, including the DSpark discriminator and explicit rejection of `num_nextn_predict_layers` as a signal.
- D1 hyper-connection renaming, in both dotted and underscore input forms.
- D2 `wo_a` 3D→2D reshape on `weight` and `scales`, and the no-op path when the input is already 2D.
- D3 quantization-path rewriting through the fusion table, asserting that all 641 backbone source declarations map to 536 explicit realized module paths and that **none** falls back to the global `affine/4/64`. A fallback is a bug, not a default.
- D4 shared-expert `swiglu_limit` correction, including an input that distinguishes `0.0` from `10.0`.
- Rejection of incompatible fusion settings.
- The post-quantization mode audit, including a case where `bits` and `group_size` match but `mode` does not — the failure that strict loading cannot see — and a case where an `mtp.*` per-module entry dangles, which must be ignored rather than reported.
- Shard-geometry pre-flight rejection for an out-of-range world size and a group-size-misaligned input dimension.
- Exact MTP filtering and reporting.
- Rejection of unexpected backbone keys.
- Preservation of unrelated model loading behavior.
- No duplication of transformations already performed by `deepseek_v4.py:2604-2760` — assert that the pre-pass leaves already-fused and already-stacked keys untouched.

These tests use configuration dictionaries and weight-index key sets without importing or initializing Metal where possible.

### Synthetic model tests

Create a small DeepSeek-V4 fixture with OMLX post-sanitize 0731 names and the checkpoint's real mixed quantization metadata. Verify that it loads strictly through the compatibility wrapper, that every declared backbone parameter is populated, and that the mode audit passes.

### Repair the tensor-parallel correctness test, then extend it

`src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py` cannot be extended as it stands, and this is real scope rather than a footnote:

- It is skipped at `:425` — `@pytest.mark.skip("TP=2 is currently very different to TP=1. This test will not pass")`.
- It is stale: `:349` calls `tensor_auto_parallel(m, g, on_layer_loaded=None)`, but the current signature is `tensor_auto_parallel(model, group)` returning a generator (`auto_parallel.py:456-459`). It would raise `TypeError` even un-skipped.
- Its `deepseek_v4` fixture (`:237-272`) sets `o_groups = 1` and `hc_mult = 1` with no quantization, which degenerates the interleaved-per-group head slice to a contiguous slice, degenerates hyper-connections, and never exercises `_shard_quantized_rows`. It covers none of the paths that matter here.
- Nothing anywhere in the suite runs `backend="jaccl"`; every distributed test uses ring.

Work items: repair the call signature, diagnose the TP=1 vs TP=2 divergence that motivated the skip, change the fixture to `o_groups = 8` and `hc_mult = 4`, add a quantized variant, and then extend it so a tiny compatible 0731 model produces numerically equivalent logits and identical greedy tokens before and after two-rank sharding. Use a documented tolerance appropriate to bf16 and quantized distributed reductions; require literal bit-exactness only for subpaths shown to preserve operation order. If this non-degenerate test demonstrates a sharding bug, a narrow correctness fix is in scope. "Unit and integration checks pass" is not satisfied by retaining the skip marker.

### Backbone numeric parity against OMLX (gating and hardware-feasible)

Every other test in this plan compares EXO against EXO. Because OMLX and EXO's fork are independent implementations, self-consistency proves the cluster is stable but not that the model is right. D4 addresses one confirmed divergence: EXO constructs `shared_experts` with `swiglu_limit = 0.0` (`deepseek_v4.py:2456-2460`) while OMLX passes the checkpoint's `config.swiglu_limit = 10.0` (`omlx/.../deepseek_v4_model.py:829-833`).

A full single-node comparison is not feasible on this hardware: the phase-one backbone contains 144.26 GiB of stored tensors before runtime overhead, exceeding either Mac's 128 GB. The mandatory pre-cluster reference is therefore **representative real-layer parity**. Load one layer at a time from the target checkpoint into the corresponding OMLX and EXO compatibility blocks and compare identical inputs for representative paths: local/hash attention, overlap compression, compression plus indexer, routed experts, and shared experts with clamp-triggering activations. Compare intermediate outputs and block outputs within documented bf16/quantized tolerances and require matching routing decisions.

After the two-node infrastructure works, a full-model differential reference may be run with a small manual two-rank OMLX harness using OMLX's existing `Model.shard()` and the same local checkpoints, with DSpark/MTP execution disabled for parity. This is valuable additional evidence but is not a prerequisite for beginning cluster work or a phase-one acceptance requirement unless the layer-level comparisons leave an unresolved divergence.

### Prompt and parser tests

Create golden cases from the OMLX 0731 reference for:

- Simple chat with thinking on and off.
- Reasoning-effort variants across the full supported set, including `low` and `max`.
- A single tool call.
- Multiple tool calls and out-of-order tool results.
- Multi-turn tool continuation with prior reasoning.
- Mid-conversation system reminders, asserting `<｜latest_reminder｜>` placement.
- Assistant prefill, asserting the prefill lands in the content channel rather than inside the thinking block.
- The append-only prefix property across successive turns with reasoning retained.

Compare rendered prompt text and token IDs. Streaming parser tests must split delimiters across arbitrary chunk boundaries — including the final-token case — and cover malformed or incomplete tool output, specifically whitespace before `>` in an invoke tag, padded `string="true"` values, and Python-literal `string="false"` values.

### Real two-node acceptance

Run the following gates in order:

1. Record the checkpoint verification appendix from §0.
2. Pass representative real-layer OMLX parity on one Mac without loading the complete model.
3. Confirm matching EXO revision, dependency lock, macOS build, `rdma_ctl` state, and model index on both nodes.
4. Start both nodes in the isolated namespace using `EXO_ZENOH_NAMESPACE`.
5. Confirm a two-node RDMA topology and preview a `Tensor`/`MlxJaccl` placement.
6. Create the target instance and confirm strict load, a clean quantization mode audit, and MTP-disabled diagnostics on both ranks.
7. Run deterministic short-prompt generation twice and compare token output.
8. Exercise thinking-off and thinking-on requests.
9. Exercise a forced tool call and a follow-up containing the tool result.
10. Exercise a mid-conversation system reminder and confirm the prompt prefix is preserved across the turn.
11. Repeat through streaming and non-streaming Chat Completions and Responses endpoints.
12. Increase prompt sizes incrementally through 32K tokens with bounded output while monitoring memory, wired-limit headroom, and rank stability.
13. If practical, run the optional two-rank OMLX full-model differential harness after EXO cluster bring-up.
14. Record time-to-first-token and generation throughput as observations.

A deliberate ring/tensor run may be compared against JACCL with the same seed and prompt. Only the JACCL run satisfies final acceptance.

## Acceptance criteria

Phase one is complete when:

- The target checkpoint loads without modifying or duplicating its files.
- No backbone weights are missing or silently ignored.
- The quantization mode audit passes with every module's realized `(bits, group_size, mode)` matching the checkpoint.
- Representative layers loaded from the real checkpoint match OMLX within documented numeric tolerances, including identical routing decisions and the corrected shared-expert clamp behavior.
- Logs clearly state that recognized MTP tensors are skipped, the observed DSpark configuration, and that MTP execution is disabled.
- EXO creates a two-node `Tensor`/`MlxJaccl` instance over the direct Thunderbolt connection.
- Deterministic generation completes on both short and 32K prompts without rank failure.
- Thinking and tool-use behavior matches the 0731 prompt goldens, and the append-only prefix property holds for mid-conversation reminders.
- Dashboard, Chat Completions, and Responses paths all work.
- The repaired tensor-parallel correctness test runs — not skipped — with a non-degenerate fixture and passes its documented numeric and greedy-token criteria.
- Relevant unit, integration, type, lint, and formatting checks pass, with no skip markers newly added or retained on tests this project depends on.
- The live validation record includes topology, compatibility mode, quantization audit output, memory observations, and basic performance measurements.

## Out of scope, noted for later

`_sharded_to_all`'s bias branch does `weight /= n` on an immutable `mx.array` and returns `None`, so the division is discarded (`auto_parallel.py:487-492`). Harmless for DSV4 — `attention_bias = false` and there are no sharded-to-all biases — but it will produce wrong results for an architecture that has them. File separately.

## Future phase: distributed DSpark/MTP

Phase two will begin only after backbone correctness and serving are stable. It will evaluate the minimum OMLX components needed for DSpark/MTP, including auxiliary model layers, custom Metal kernels, cache semantics, prompt-state handling, and distributed coordination. Note that DSpark's checkpoint layout differs from legacy Lightning MTP — DSpark keeps stages at `mtp.<s>.*` directly, whereas the legacy path nests them under `mtp.<s>.block.*` (`omlx/omlx/patches/mlx_lm_mtp/deepseek_v4_model.py:752-772`; tests at `omlx/tests/test_deepseek_v4_dspark.py:216-253`) — and the 0731 checkpoint uses the DSpark layout. The phase-two design must define how speculative candidates and verification are partitioned or replicated across ranks and must demonstrate output equivalence with backbone-only decoding before performance claims are made.

## Appendix A: checkpoint facts (verified)

Measured 2026-08-04 from `/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp` — 30 safetensors shards, `config.json` 156 KB, plus `model.safetensors.index.json`, `oq_imatrix_report.json`, and tokenizer files. Shapes and dtypes were read from safetensors headers without materializing tensors.

### Architecture config

`num_hidden_layers 43`, `hidden_size 4096`, `num_attention_heads 64`, `num_key_value_heads 1`, `head_dim 512`, `q_lora_rank 1024`, `o_lora_rank 1024`, `o_groups 8`, `qk_rope_head_dim 64`, `sliding_window 128`, `n_routed_experts 256`, `n_shared_experts 1`, `num_experts_per_tok 6`, `moe_intermediate_size 2048`, `num_hash_layers 3`, `scoring_func "sqrtsoftplus"`, `topk_method "noaux_tc"`, `routed_scaling_factor 1.5`, `swiglu_limit 10.0`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `index_n_heads 64`, `index_head_dim 128`, `index_topk 512`, `compress_rope_theta 160000`, `vocab_size 129280`, `expert_dtype "fp4"`, `attention_bias false`, `tie_word_embeddings false`.

`rope_scaling = {type: "yarn", factor: 16, beta_fast: 32, beta_slow: 1, original_max_position_embeddings: 65536}` with `max_position_embeddings 1048576`. Note the 32K validation prompt sits *below* `original_max_position_embeddings`, so it does not exercise the YaRN extrapolation regime; confirm separately whether EXO's `patches/standard_yarn_rope.py` applies to this path.

`compress_ratios`: **46 entries** for 43 backbone layers, values in `{0, 4, 128}`, pattern `[0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0, 0, 0]`. Layers 0–1 are local attention (ratio 0), 41 layers carry a compressor, of which 21 also carry an indexer (ratio 128). The trailing three zeros are the DSpark stages, which are local-attention only.

DSpark fields present: `dspark_block_size 5`, `dspark_target_layer_ids [40, 41, 42]`, `dspark_markov_rank 256`, `dspark_noise_token_id 128799`. `num_nextn_predict_layers 1` is also present — confirming that it must not be used as the discriminator. No `n_mtp_layers` field, so the stage count derives from `len(dspark_target_layer_ids)` = 3.

### Weight layout

2230 tensors total, of which **114 are `mtp.*`** and 674 modules are quantized.

Key space is OMLX post-sanitize throughout: `model.embed_tokens.*`, `model.norm.weight`, `lm_head.*`, `model.hc_head.{fn,base,scale}`, `model.layers.N.*`. Hyper-connections are **dotted**: `model.layers.N.attn_hc.{fn,base,scale}` and `ffn_hc.*`, all `F32`, `fn` shaped `[24, 16384]` = `[(2+hc)*hc, hc*hidden]` and `model.hc_head.fn` shaped `[4, 16384]` — both exactly matching EXO's declarations. Routed experts are pre-stacked as `ffn.switch_mlp.{gate,up,down}_proj` and shared experts already use `{gate,up,down}_proj` names. `ffn.gate.e_score_correction_bias` `F32 [256]` on routed layers; `ffn.gate.tid2eid` `I32 [129280, 6]` on the three hash layers. `attn.attn_sink` `F32 [64]`. `attn.compressor.ape` is `BF16 [128, 512]` where EXO declares `float32` — shape matches and EXO casts at use (`:1621`), so this is benign, but it is the one dtype divergence found.

Representative shapes from layer 3 (a compressor + indexer layer):

| Tensor | dtype | shape | implied |
| --- | --- | --- | --- |
| `attn.wq_a.weight` / `.scales` | U32 / U8 | `[1024, 1024]` / `[1024, 128]` | mxfp8, in 4096 |
| `attn.wkv.weight` / `.scales` | U32 / U8 | `[512, 1024]` / `[512, 128]` | mxfp8, in 4096 |
| `attn.wq_b.weight` / `.scales` | U32 / U8 | `[32768, 256]` / `[32768, 32]` | mxfp8, in 1024 |
| `attn.wo_a.weight` / `.scales` | U32 / U8 | `[8, 1024, 1024]` / `[8, 1024, 128]` | **3D**, mxfp8, in 4096 per group |
| `attn.wo_b.weight` / `.scales` | U32 / U8 | `[4096, 2048]` / `[4096, 256]` | mxfp8, in 8192 |
| `attn.compressor.wkv.{weight,scales,biases}` | U32 / BF16 / BF16 | `[512, 1024]` / `[512, 64]` / `[512, 64]` | affine-8, group 64, in 4096 |
| `ffn.switch_mlp.gate_proj.weight` / `.scales` | U32 / U8 | `[256, 2048, 512]` / `[256, 2048, 128]` | mxfp4, in 4096 |
| `ffn.switch_mlp.down_proj.weight` / `.scales` | U32 / U8 | `[256, 4096, 256]` / `[256, 4096, 64]` | mxfp4, in 2048 |
| `lm_head.{weight,scales,biases}` | U32 / BF16 / BF16 | `[129280, 1024]` / `[129280, 64]` / `[129280, 64]` | affine-8, group 64 |
| `model.embed_tokens.{weight,scales,biases}` | U32 / BF16 / BF16 | `[129280, 1024]` / `[129280, 64]` / `[129280, 64]` | affine-8, group 64 |

### Fusion arithmetic (verified)

`wqkv_a` = concat(`wq_a` `[1024, 1024]`, `wkv` `[512, 1024]`, axis 0) = `[1536, 1024]`, matching EXO's `nn.Linear(4096, q_lora_rank + head_dim = 1536)` at mxfp8. Scales concatenate to `[1536, 128]`. Neither source has `biases`.

`compressor.wkv_gate` = concat(`wkv` `[512, 1024]`, `wgate` `[512, 1024]`) = `[1024, 1024]`, matching EXO's `nn.Linear(4096, 2 * coff * head_dim = 1024)` at affine-8/64. Scales and `biases` both concatenate to `[1024, 64]`; `_fuse_pair` handles all three suffixes.

`wo_a` D2 reshape: `[8, 1024, 1024]` → `[8192, 1024]` and `[8, 1024, 128]` → `[8192, 128]`, matching EXO's `nn.Linear(group_feat = 4096, n_groups * o_lora_rank = 8192)` at mxfp8. EXO's forward reshapes back to `[n_groups, o_lora_rank, -1]`, so the transformation round-trips exactly.

### Quantization coverage

674 modules have `.scales`; 674 per-module entries exist in `config["quantization"]`; the two source sets are **identical** — zero modules relying on the global setting and zero dangling entries before phase-one transformation. Of these, 641 entries are backbone and 33 are `mtp.*`. The 641 backbone declarations map to 536 realized model paths after 105 compatible fusion pairs; no fusion pair has conflicting quantization settings. Distinct triples and their module families are tabulated in [Quantization reality](#quantization-reality). `quantization_config` mirrors `quantization`.

### MTP / DSpark layout (for phase two)

Three stages in flat `mtp.<s>.*` form with **no** `.block.` nesting, confirming the DSpark layout rather than legacy Lightning MTP. Stage 0 carries `main_proj.weight` `BF16 [4096, 12288]` and `main_norm.weight`; stage 1 is the bare block; stage 2 additionally carries `norm.weight`, `hc_head.{fn,base,scale}`, `markov_head.{markov_w1,markov_w2}.weight`, and `confidence_head.proj.weight`. Every stage has its own `attn_hc`/`ffn_hc`, `attn.*`, `ffn.*`, and `attn_norm`/`ffn_norm`, quantized on the same mxfp8/mxfp4 split as the backbone. No stage carries a compressor or indexer, consistent with the trailing `compress_ratios` zeros.

### Reproduction

```bash
cd /Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp
python3 - <<'PY'
import json, struct, collections, re
c = json.load(open('config.json'))
q = c['quantization']
per = {k: v for k, v in q.items() if isinstance(v, dict)}
wm = json.load(open('model.safetensors.index.json'))['weight_map']
scaled = {k[:-7] for k in wm if k.endswith('.scales')}
def mapped_quant_path(k):
    k = re.sub(r'(\.attn)\.(wq_a|wkv)$', r'\1.wqkv_a', k)
    return re.sub(
        r'(\.attn(?:\.indexer)?\.compressor)\.(wkv|wgate)$',
        r'\1.wkv_gate',
        k,
    )
backbone_per = {k: v for k, v in per.items() if not k.startswith('mtp.')}
mapped_backbone = {mapped_quant_path(k) for k in backbone_per}
print('tensors', len(wm), 'mtp', sum(k.startswith('mtp.') for k in wm))
print('global', {k: v for k, v in q.items() if not isinstance(v, dict)})
print('quantized modules', len(scaled), 'per-module entries', len(per))
print('backbone source entries', len(backbone_per), 'realized paths', len(mapped_backbone))
print('no-entry (falls back to global):', sorted(scaled - per)[:10])
print('dangling entries:', sorted(per - scaled)[:10])
print(collections.Counter((v['mode'], v['bits'], v['group_size']) for v in per.values()))
print('compress_ratios', len(c['compress_ratios']), 'layers', c['num_hidden_layers'])
hdr = {}
def info(k):
    s = wm[k]
    if s not in hdr:
        with open(s, 'rb') as f:
            hdr[s] = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    e = hdr[s][k]; return e['dtype'], e['shape']
for k in sorted(x for x in wm if x.startswith('model.layers.3.')):
    print(f'{k:70s}', info(k))
PY
```

## Revision history

**Revision 4 (2026-08-04)** — amended after independent verification of revision 3 against EXO `a6cd2fce`, OMLX `50846648`, and the local checkpoint. Changes:

1. Corrected the claim that OMLX has no DSV4 tensor parallelism: current OMLX defines `Model.shard()`. The EXO approach remains selected because OMLX lacks EXO's discovery, placement, JACCL orchestration, lifecycle, dashboard, and API control plane.
2. Replaced the impossible full-model single-node parity gate. The phase-one backbone is 144.26 GiB before runtime overhead, so parity now uses representative real checkpoint layers on one Mac, with an optional full two-rank OMLX differential harness after cluster bring-up.
3. Added D4 to correct EXO's shared-expert `swiglu_limit` from its hard-coded `0.0` to the checkpoint value `10.0` within the compatibility subclass.
4. Corrected the quantization audit cardinality: 641 backbone source declarations collapse through 105 compatible fusion pairs into 536 realized quantized module paths; 33 MTP declarations remain intentionally excluded.
5. Replaced unconditional bit-exact tensor-parallel acceptance with documented bf16/quantized numeric tolerances plus identical greedy-token behavior, while permitting narrow sharding correctness fixes demonstrated by a non-degenerate test.
6. Scoped P6 to changes required by failing phase-one prompt/parser tests; broad dead-code, context-plumbing, and developer-role refactors are deferred.
7. Removed the active-load claim that D1 would silently downcast HC tensors through `cast_predicate`; the checkpoint tensors are already F32 and strict key matching is the relevant gate.
8. Added exact stored-weight accounting: 140.87 GiB sharded plus 3.39 GiB replicated yields approximately 73.83 GiB per rank at two nodes, before runtime overhead.

**Revision 3 (2026-08-04, superseded)** — checkpoint inspected; all inferred claims replaced with measured ones. Changes:

1. Added [Appendix A](#appendix-a-checkpoint-facts-verified) with measured config, weight layout, shapes, dtypes, quantization coverage, MTP layout, and a reproduction script. §0 is now closed.
2. Confirmed D1 (dotted `attn_hc`/`ffn_hc`) and D2 (3D `wo_a`) are both required, with exact shapes; added the observation that skipping D1 would also defeat `cast_predicate`'s fp32 protection, not merely fail the key match.
3. Dropped D4 — `tid2eid` is already `I32`, matching EXO's declaration. Downgraded to a tolerant assertion.
4. Sharpened D3 from "necessary" to "highest-risk": the global fallback is `affine/4/64` and is correct for **zero** modules, all 674 quantized modules carry explicit overrides with exact coverage, and three modes coexist within a single attention module. Set the audit target at 641 backbone modules with 33 `mtp.*` entries expected to dangle.
5. Verified the fusion arithmetic end to end and confirmed both fusion pairs are internally mode-compatible; retained the pre-fusion assertion as future-proofing rather than a live need.
6. **New finding:** all five of EXO's fused mxfp4 attention fast paths are gated on `mode == "mxfp4"` and will fall back under this checkpoint's mxfp8/affine recipe. Correct, but a throughput consequence that phase one must record rather than treat as representative.
7. Softened the `compress_ratios` requirement — 46 entries for 43 layers is expected and safe, since EXO bounds-checks its indexing; the assertion now guards only against a short list.
8. Noted the single dtype divergence found (`compressor.ape` BF16 vs EXO's float32 declaration, benign) and the YaRN observation that a 32K prompt sits below `original_max_position_embeddings = 65536`.

**Revision 2 (2026-08-04)** — amended after review against the working trees at EXO `a6cd2fce`. Changes:

1. Corrected the central premise: EXO pins the `rltakashige/mlx-lm` fork branch `leo/deepseek-v4`, whose `sanitize()` already implements three of revision 1's four mapping rows plus expert stacking, FP8/FP4 handling, and an intentional `mtp.*` drop. Section 3 is now a delta against that function.
2. Added D2 — the `wo_a` 3D→2D reshape — which revision 1 omitted and which would have hard-failed the load.
3. Replaced "4-bit affine" with the actual mixed recipe, and added the mandatory post-quantization mode audit, noting that strict loading catches `bits`/`group_size` but not `mode`.
4. Added the §0 checkpoint verification prerequisite; revision 1's mapping table was inferred from oQ's source rather than the artifact.
5. Added a gating single-node OMLX numeric parity test; revision 1's tests were all EXO-vs-EXO.
6. Reframed the tensor-parallel bit-exactness work as repair, de-degenerate, then extend — the existing test is skipped, stale against the current signature, and its fixture degenerates `o_groups` and `hc_mult`.
7. Replaced `EXO_LIBP2P_NAMESPACE` with `EXO_ZENOH_NAMESPACE`; the former was removed by commit `09f9ea31` and now raises.
8. Rewrote §6 to target EXO's consumer layer rather than the vendored encoder, led by the append-only / `latest_reminder` prompt-cache issue, and **withdrew** revision 1's instruction to remove synthetic-system-message tool injection, which was incorrect.
9. Recorded the two invariants that make the plan sound — replicated hyper-connections, and literal-attribute-name sharding as the reason subclassing is mandatory — so neither is "fixed" later.
10. Added shard-geometry pre-flight checks for the world-size-divides-8 constraint and quantized input-dimension group-size alignment.
11. Added the DSpark discriminator (`dspark_block_size` plus `dspark_target_layer_ids`, explicitly not `num_nextn_predict_layers`), `compress_ratios` validation, the `tid2eid` dtype note, and the observation that `from_dict` silently discards `dspark_*`.
12. Documented that the model card requires no change, correcting an open question about tool-capability gating.
13. Added memory expectations, and noted the understated wired limit and unsharded KV cache.

## Appendix B — OMLX reconciliation 50846648..2450a53c

The 51 commits in `50846648..2450a53c` partition as follows: Group I — 5;
Group II — 2; Group III — 3; Group IV — 7; Group V — 34; Group VI — the
provable non-change. The counted groups total **51 commits**.

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
`b75e1aa0`); Muse Glimmer VLM ×4 (`6ee393d4`, `39bb1784`, `9a57d63d`, and
`e1acb0bc`'s VLM MTP thinking budget); Jina reranker ×3 (`876e1797`, `03a3120e`,
`7b755b90`); Inkling ×2 (`5215d9b4`, `5306b733`); generic XML tool-calling ×3
(`cdeea4c5`, `12937527`, `d5592aa0`); `13997cec` gemma4; `a714035f` Hermes;
admin and engine ×4 (`198c5ce9`, `fe79b272`, `c10c5c5b`, `9b59e122`); i18n and
mac-app ×3 (`76e13909`, `9aacf8d9`, `90277828`); `128615b7` codex CLI;
`d2575b1d` deps; version bumps ×3 (`49ec2716`, `ab95612a`, `350dc08b`);
`2450a53c` web search; `24e0d2b1` test stub; `95c38c13` VLM sampler typing.

The three `fix(tool-calling)` commits are OMLX's generic XML tool protocol, not
DSML. They are not applicable regardless of merit, since the checkpoint has no
trained tool-call capability under either declared protocol.

### Group VI — The provable non-change

`omlx/patches/deepseek_v4/chat_template_v4.py` and
`tool_parser_v4.py` have **zero commits** in `50846648..2450a53c`. The prompt
protocol and the DSML reference parser are unchanged, so re-pinning the
reference revision cannot invalidate the 16 committed goldens or Tasks 1
through 5. This is the non-change proved in S1 by byte-identical hashes and the
generator-input audit above.
