# DeepSeek V4 Flash 0731 JACCL/RDMA validation

Status: accepted. Two-rank Tensor/JACCL/RDMA loading and inference passed through 16K context at HEAD `937ae049`. After discovery hardening, the exact two-rank placement and deterministic short request were recaptured at HEAD `9ddab629` with literal request, response, token, topology, TTFT, and memory artifacts. A 32K prefill crashed rank 1 and is explicitly not accepted or retried.

## Checkpoint and parity gate

- Checkpoint: `Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp`.
- Index SHA-256: `a7e7421c2789f0435eef9c47ec58fea412c5da095b7f55377c5740161cb54512`.
- Representative real-layer parity: `/private/tmp/dsv4-0731-layer-parity.json`, `passed: true` at EXO SHA `c36c982967d05da09137987dc604d10d863c1a26`.
- Layers `0`, `2`, and `3` passed, with the expected compression/cache topology. MTP execution was disabled and no complete model allocation occurred.

## Quality gates

| Gate | Exact command | Result |
| --- | --- | --- |
| DSV4 + bootstrap focused tests | `uv run pytest -q src/exo/worker/tests/unittests/test_main.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_config.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_model.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_loader.py src/exo/worker/tests/unittests/test_mlx/test_deepseek_v4_0731_parity_helpers.py` | Exit 0; 85 passed. |
| Complete unit tree | `uv run pytest -q src/exo/worker/tests/unittests --import-mode=importlib` | Exit 0; 230 passed, 166 deselected. |
| TP parity | `uv run pytest -v -m slow src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py` | Exit 0; 2 passed in 7.06s; BF16 and Q4 executed, neither skipped. |
| Networking | `cargo test -p networking` | Exit 0 outside the filesystem sandbox; 5 passed. This includes real loopback Zenoh bootstrap, same-namespace liveliness/pub-sub, and cross-namespace isolation. The sandboxed multicast-socket test alone received macOS `EPERM`. |
| Rust binding | `cargo check -p exo_rs` | Exit 0. |
| Static typing | `.venv/bin/basedpyright` | Exit 0; 0 errors, 0 warnings, 0 notes after test-only typing repairs. |
| Lint | `uv run ruff check` | Exit 0; all checks passed. |

Known non-green, non-DSV4 deviations: `nix fmt` cannot run because `nix` is absent. The repository-root `tests/` directory is a live cluster integration harness; plain top-level `uv run pytest -q` stops while importing its non-root `exo_tools` workspace package. The complete isolated unit tree above is green.

## Isolated validation worktrees and matching facts

| Fact | Local | Remote |
| --- | --- | --- |
| Worktree | `/Users/jared/aishit/exo-install/exo/.worktrees/codex-dsv4-0731-backbone` | `/Users/jared/aishit/exo-dsv4-validation` |
| Branch / HEAD | `codex/dsv4-0731-backbone` / `9ddab629917ad717b75ffda8087301d162f41008` | `codex/dsv4-0731-validation` / `9ddab629917ad717b75ffda8087301d162f41008` |
| Hardware | M5 Max MacBook Pro, 128 GB unified memory | M4 Max MacBook Pro, 128 GB unified memory |
| `uv.lock` SHA-256 | `d2b803bb48355ae128cc97cfac9b3e5ed7e0463f2de71f64b8646966475d301d` | `d2b803bb48355ae128cc97cfac9b3e5ed7e0463f2de71f64b8646966475d301d` |
| uv | `0.9.24 (0fda1525e 2026-01-09)` | `0.9.24 (0fda1525e 2026-01-09)` via `/Users/jared/.local/bin/uv` |
| macOS | `26.5.2 (25F84)` | `26.5.2 (25F84)` |
| RDMA | `rdma_ctl status`: `enabled` | `rdma_ctl status`: `enabled` |

The remote dirty main worktree (`/Users/jared/aishit/exo`) was preserved. A complete Git bundle was fetched only into `codex/dsv4-0731-validation`, then a new remote worktree was created.

## Model card

The source model cards in the local and remote original checkouts both had SHA-256 `75cf99cf330ecf2da2b1c58c41ace6905f4b27bc2d3cf680307fe3db466c54c2`. That exact card was copied as an untracked file into each isolated validation worktree and was never added to Git. The remote validation worktree retained only that untracked card, and the local validation record is tracked separately without staging the card.

## Final live validation

- Final live HEAD: `9ddab629917ad717b75ffda8087301d162f41008`. The earlier model fixes `f0d52516` (MLX keyword callback) and `937ae049` (do not re-quantize the three prebuilt switch projections) remained in the tested history.
- Two discovery defects were proven and fixed with focused regressions. `b7a9cf18` keeps transiently unreachable multicast routes eligible for later sends, leaving interface removal to netwatcher. Because macOS still rejected multicast routing over the direct link even with both firewalls temporarily disabled, `9ddab629` added the deterministic, optional `EXO_BOOTSTRAP_PEERS` path through EXO's existing Zenoh configuration. Ordinary no-bootstrap startup remains the default.
- Post-live review found that direct Zenoh peers could bypass multicast namespace matching. Commit `d603559fca71b250a809f5483a3dbfd587e45338` scopes all EXO topic and liveliness keys with the same 64-bit BLAKE3 namespace hash used by discovery, trims empty/whitespace bootstrap entries, and updates the CLI contract. A real three-session loopback test passed three consecutive times: same-namespace peers discovered each other and exchanged topic data, while a directly bootstrapped different-namespace peer exchanged no EXO traffic in either direction. This transport-only review fix did not require reloading the checkpoint; the prompt/API plan will deploy this revision on both Macs before its live matrix.
- The successful launch used namespace `dsv4f-0731-validation`, local `EXO_BOOTSTRAP_PEERS=tcp/169.254.233.2:52414`, and remote `EXO_BOOTSTRAP_PEERS=tcp/169.254.240.63:52414`. Direct TCP to the remote Thunderbolt address and port `52414` succeeded before launch.
- The live topology reported reciprocal 80 Gb/s RDMA: local M5 `rdma_en6` to remote M4 `rdma_en3`, and remote M4 `rdma_en3` to local M5 `rdma_en6`. Remote M4 rank 0 and local M5 rank 1 formed world size 2 on the required `Tensor`/`MlxJaccl` placement.
- Both logs confirmed JACCL initialization, strict `deepseek_v4_0731_backbone` loading, `mtp_weight_count=114`, DSpark `block_size=5` and targets `[40, 41, 42]`, with DSpark/MTP speculative decoding disabled. Rank 0 sharded/loaded in 25.29s and became ready after the 50-token warmup; rank 1 sharded/loaded in 25.66s and reported ready after 35.79s total initialization.
- The model emitted a pinned-transformers compatibility warning and used EXO's generic tokenizer fallback. The backbone output is therefore useful for execution determinism, but not for judging natural-language or chat-template correctness; that remains the prompt/API follow-on plan.

### Retained Step 5/6/8 evidence

- Selected preview: `/private/tmp/dsv4-recapture-selected-preview.json`. It is the required `Tensor`/`MlxJaccl` placement with instance ID `9ffdd8e8-fe3c-4b98-9ef1-53da1c1b7121`, world size 2, remote M4 rank 0, local M5 rank 1, JACCL devices `[[null, "rdma_en3"], ["rdma_en6", null]]`, rank-0 coordinator `0.0.0.0:55485`, and rank-1 coordinator `192.168.1.62:55485`.
- Exact instance submission: `/private/tmp/dsv4-recapture-instance-request.json`; response: `/private/tmp/dsv4-recapture-instance-response.json`; create command ID `4cdf4b61-0af6-443e-8ed1-ef405c13d71c`. `/private/tmp/dsv4-recapture-instance-await.sse` retained the await stream and `/private/tmp/dsv4-recapture-state-ready.json` retained both ready runners.
- The checkpoint contained 114 MTP tensors. The loader skipped 33 MTP quantization declarations, retained 536 non-MTP realized quantization paths for strict audit, and reported DSpark/MTP execution disabled. The audit passed before the instance was accepted.
- Both ranks emitted the generated JACCL matrix/coordinator evidence, initialized JACCL, completed post-sharding barrier synchronization, then loaded and became ready. This is retained as log evidence, not inferred from topology alone.
- Exact deterministic request: `/private/tmp/dsv4-recapture-short-request.json`. It used `temperature=0`, `seed=12345`, `max_tokens=16`, `use_prefix_cache=false`, and prompt `Say hello in one short sentence.` Responses are `/private/tmp/dsv4-recapture-short-response-1.json` and `/private/tmp/dsv4-recapture-short-response-2.json` with command IDs `282bce9d-4087-44a1-aba9-1d80dcec3846` and `5e2fdb57-0ee7-4f2e-8864-1c4229f5fd02`.
- Both runs produced byte-identical text and the identical token-ID array `[343, 77291, 343, 77291, 797, 671, 671, 671, 671, 671, 16, 343, 77291, 110137, 110137, 110137]`, retained in `/private/tmp/dsv4-recapture-token-ids.json`. The generic-tokenizer text was `(thinking (thinking).TheTheTheTheThe. (thinkingThinkingThinkingThinking` and is not treated as a semantic-quality result.
- Short-run prompt/decode throughput was 17.629/33.245 tok/s and 19.915/33.162 tok/s. Both reported peak model memory `85,173,786,505` bytes. The token-aware streaming probe `/private/tmp/dsv4_measure_ttft.py` recorded first-token latency `8.2494s` and total response time `8.7065s`; the SSE is `/private/tmp/dsv4-recapture-stream-response.sse`. HTTP keepalive arrival was explicitly excluded from TTFT.
- Before generation, total EXO-process RSS was `5,433,442,304` bytes local and `5,740,871,680` bytes remote; wired memory was `5,581,946,880` bytes local and `3,926,360,064` bytes remote. After generation, RSS was `5,439,160,320` bytes local and `5,742,133,248` bytes remote; wired memory was `5,572,231,168` bytes local and `3,924,508,672` bytes remote. MLX's reported 85.17 GB peak is the more representative model-allocation measure because process RSS and wired-page samples do not fully describe Metal unified-memory allocation.
- `/private/tmp/dsv4-recapture-state-after-generation.json` retained both runners `Ready`, reciprocal RDMA, local available memory `25,056,526,336` bytes, and remote available memory `44,661,784,576` bytes after the short runs.
- Health after the earlier successful context ladder was retained as a live two-rank ready state through 16K. After the rejected 32K run, EXO deleted the instance; local state was empty, M5 available memory was 111,660,630,016, and M4 ping/SSH recovered with roughly 108 GB free, but the remote validation daemon no longer advertised.

The exact request, response, token, placement, and state fields needed for acceptance are embedded above. These SHA-256 values make the original capture files independently identifiable even after `/private/tmp` is cleared:

| Capture | SHA-256 |
| --- | --- |
| selected preview | `a31b70d9d7eda39da676c393da8348ac9ecb8e3d444425df175da527ea3ea949` |
| instance request | `4179be0043d83f15088084e15ba8c32a6c324b73077cbe9570f47ef88a4a6721` |
| instance response | `ad1357d3371a9da31f1b512a6e87972981413c009a620d305eccb9c4377d3326` |
| deterministic request | `264be8071daabcd9c61b034f8fca3e06a7054c8e7300de033b923bf1aab23431` |
| deterministic response 1 | `f5ac99d1ac46c0164a4fb93c50ac3b93e293e8415dcf6c66d1f4213b4a868226` |
| deterministic response 2 | `ac6f2c050ff71ba88a53fa12083f9ec08813738b42d916cf62bef93c52331b2a` |
| token IDs | `7e5e8846f46132828d4f6753ba68e4106eb68692d70ee0c33262cb646962ee5b` |
| streaming SSE | `12b6bef62360cd6404fc3602b42f86e996a104820c0dc7ffcac335bcf315f7e2` |
| ready state | `d698677b5cf277e3ef42a1ab241cd94cac7bd3c2d4c4c815b93aefb66ae576fd` |
| post-generation state | `cea46e3e519a17f34de47f7faf64814b4f660d823f2a8f61777c9fa13324b00a` |

| Context | Prompt tok/s | Generation tok/s | Peak memory |
| --- | ---: | ---: | ---: |
| 1,032 | 104.612 | 27.385 | 86,717,189,487 |
| 4,104 | 192.337 | 25.689 | 90,194,775,647 |
| 8,200 | 292.806 | 26.946 | 93,338,020,153 |
| 16,392 | 212.133 | 26.385 | 99,495,391,010 |

The accepted ceiling is 16K. At roughly 32,776 expected tokens, M5 rank 1 segfaulted in `mlx_lm.generate.generate_step` during prefill (15:51:29); API returned HTTP 500, auto-retry hit JACCL queue-pair RTR errno 22, and EXO deleted the instance. Do not retry 32K without separate systematic debugging. Cleanup left local instance empty and M5 available memory 111,660,630,016; M4 recovered ping/SSH and roughly 108 GB free but its validation daemon no longer advertised. The exact short-run recapture at `9ddab629` ended with graceful shutdown on both nodes; no EXO or runner process remained.

## Known non-live deviations

- `nix fmt` could not run locally because `nix` is absent.
- Plain top-level pytest includes the live cluster integration harness and stops during `tests/conftest.py` import because `exo_tools` is a separate workspace package not installed into the root environment. The complete `src/exo/worker/tests/unittests` tree passed independently.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` is not a green repository gate: it reports the existing removed-lint, package-metadata, and pedantic-lint backlog outside these changes. `cargo check -p exo_rs` and the focused networking test suite are the applicable Rust gates for the discovery/bootstrap changes.
- macOS multicast discovery did not converge on the direct link even with both firewalls temporarily disabled. The accepted run therefore used the explicit Thunderbolt bootstrap endpoints above. Both firewalls were requested to be re-enabled immediately after the validation daemons were stopped.
