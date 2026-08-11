# type: ignore
"""Metal-backed DeepSeek V4 tensor-parallel parity tests."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import tempfile
import traceback
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from exo.shared.types.worker.runner_response import ModelLoadingResponse

if TYPE_CHECKING:
    import mlx.nn as nn


_DEEPSEEK_V4_ARGS = dict(
    model_type="deepseek_v4",
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=64,
    num_key_value_heads=1,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=8,
    head_dim=16,
    qk_rope_head_dim=8,
    sliding_window=16,
    compress_ratios=[0, 4, 128, 0],
    index_n_heads=8,
    index_head_dim=16,
    index_topk=16,
    moe_intermediate_size=128,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    num_hash_layers=1,
    hc_mult=4,
    num_nextn_predict_layers=0,
    max_position_embeddings=256,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 2,
        "original_max_position_embeddings": 128,
        "type": "yarn",
    },
)

MODEL_CONFIGS = {
    "deepseek_v4_bf16": dict(
        module="mlx_lm.models.deepseek_v4", args=dict(_DEEPSEEK_V4_ARGS)
    ),
    "deepseek_v4_q4": dict(
        module="mlx_lm.models.deepseek_v4",
        quantize=dict(group_size=32, bits=4, mode="affine"),
        args=dict(_DEEPSEEK_V4_ARGS),
    ),
}

TOLERANCES = {
    "deepseek_v4_bf16": {"rtol": 2e-2, "atol": 2e-2},
    "deepseek_v4_q4": {"rtol": 5e-2, "atol": 5e-2},
}

_PROMPT = [[1, 23, 45, 67, 89, 12, 34, 56]]
_GREEDY_STEPS = 4
_GREEDY_TOKEN = 255
_HEAD_SCALE = 1e-3
_TARGET_PROJECTION_SCALE = 1e-2
_GREEDY_BIAS = 2.0


def _consume_sharding(
    generator: Generator[ModelLoadingResponse, None, nn.Module],
) -> nn.Module:
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def _backbone_direction(probe: Any, mx: Any) -> Any:
    """Make a deterministic decode margin without altering the V4 backbone.

    MLX's Metal SDPA reduces a different number of heads in the unsharded and
    TP paths.  A normal random output head can turn that bounded numerical
    difference into a different greedy token.  We derive one output row from
    real, cache-backed backbone states for a fixed continuation, leaving every
    attention and MoE parameter and call path intact.  The row has a positive
    margin on every state in that continuation.  Every vocabulary row retains
    a small, nonzero projection so the logit distribution remains live.
    """
    prompt = mx.array(_PROMPT, dtype=mx.int32)
    continuation = mx.full((1, 1), _GREEDY_TOKEN, dtype=mx.int32)
    cache = probe.make_cache()
    prompt_state = probe.model(prompt, cache=cache)[:, -1, :]
    mx.eval(prompt_state)
    states = [prompt_state]
    for _ in range(_GREEDY_STEPS):
        state = probe.model(continuation, cache=cache)[:, -1, :]
        mx.eval(state)
        states.append(state)
    return mx.sum(mx.concatenate(states, axis=0), axis=0)


def _condition_greedy_head(model: Any, direction: Any, mx: Any) -> None:
    import mlx.nn as nn

    mx.eval(direction)

    conditioned = mx.concatenate(
        (
            model.lm_head.weight[:_GREEDY_TOKEN].astype(mx.float32) * _HEAD_SCALE,
            (direction.astype(mx.float32) * _TARGET_PROJECTION_SCALE)[None, :],
        ),
        axis=0,
    )
    conditioned = mx.contiguous(conditioned)
    mx.eval(conditioned)
    bias = mx.concatenate(
        (
            mx.zeros((_GREEDY_TOKEN,), dtype=mx.float32),
            mx.array([_GREEDY_BIAS], dtype=mx.float32),
        )
    )
    head = nn.Linear(model.lm_head.weight.shape[-1], model.lm_head.weight.shape[0])
    head.update({"weight": conditioned, "bias": bias})
    model.lm_head = head
    mx.eval(model.lm_head.weight)


def _build(name: str) -> tuple[Any, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path

    import exo.worker.engines.mlx.auto_parallel  # noqa: F401

    cfg = MODEL_CONFIGS[name]
    module = importlib.import_module(cfg["module"])
    mx.random.seed(0)
    model = module.Model(module.ModelArgs(**cfg["args"]))

    def _to_bf16(_path: str, value: Any) -> Any:
        if hasattr(value, "dtype") and value.dtype in (
            mx.float16,
            mx.float32,
            mx.bfloat16,
        ):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(_to_bf16, model.parameters()))
    probe = module.Model(module.ModelArgs(**cfg["args"]))
    probe.update(model.parameters())
    _condition_greedy_head(model, _backbone_direction(probe, mx), mx)
    if "quantize" in cfg:
        nn.quantize(model, **cfg["quantize"])
    mx.eval(model.parameters())
    return mx, model


def _branch_activity(model: Any, mx: Any) -> tuple[list[float], list[float]]:
    """Record real attention and MoE output norms without changing their work."""
    import mlx.nn as nn

    attention_norms: list[float] = []
    moe_norms: list[float] = []

    class _Trace(nn.Module):
        def __init__(self, inner: Any, norms: list[float]) -> None:
            super().__init__()
            self.inner = inner
            self.norms = norms

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            output = self.inner(*args, **kwargs)
            mx.eval(output)
            self.norms.append(float(mx.sqrt(mx.sum(output.astype(mx.float32) ** 2))))
            return output

    for layer in model.model.layers:
        layer.attn = _Trace(layer.attn, attention_norms)
        layer.ffn = _Trace(layer.ffn, moe_norms)
    return attention_norms, moe_norms


def _generate(model: Any, mx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompt = mx.array(_PROMPT, dtype=mx.int32)
    cache = model.make_cache()
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    initial_logits = np.asarray(logits.astype(mx.float32))
    tokens: list[int] = []
    greedy_logits: list[np.ndarray] = []

    for _ in range(_GREEDY_STEPS):
        greedy_logits.append(np.asarray(logits[:, -1:, :].astype(mx.float32)))
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(
            mx.int32
        )
        mx.eval(next_token)
        tokens.extend(np.asarray(next_token).reshape(-1).tolist())
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    greedy_logits.append(np.asarray(logits[:, -1:, :].astype(mx.float32)))

    return initial_logits, np.asarray(tokens, dtype=np.int32), np.concatenate(
        greedy_logits, axis=1
    )


def _shard_geometry(
    model: Any,
) -> tuple[list[int], list[int], list[int], list[int], list[str]]:
    return (
        [layer.attn.n_heads for layer in model.model.layers],
        [layer.attn.wq_b.weight.shape[0] for layer in model.model.layers],
        [layer.attn.wq_b.weight.shape[-1] for layer in model.model.layers],
        [layer.ffn._v4_inner.switch_mlp.gate_proj.weight.shape[-2] for layer in model.model.layers],
        [type(layer.attn.wq_b).__name__ for layer in model.model.layers],
    )


def _run(name: str, out_path: str, shard: bool) -> None:
    import mlx.core as mx

    group = mx.distributed.init(backend="ring", strict=True) if shard else None
    mx_, model = _build(name)
    if group is not None:
        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        model = _consume_sharding(tensor_auto_parallel(model, group))
        mx_.eval(model.parameters())
    heads, wq_rows, wq_cols, moe_widths, wq_types = (
        _shard_geometry(model) if shard else ([], [], [], [], [])
    )
    attention_norms, moe_norms = _branch_activity(model, mx_)
    logits, tokens, greedy_logits = _generate(model, mx_)
    np.savez(
        out_path,
        logits=logits,
        tokens=tokens,
        greedy_logits=greedy_logits,
        attention_norms=np.asarray(attention_norms),
        moe_norms=np.asarray(moe_norms),
        heads=np.asarray(heads),
        wq_rows=np.asarray(wq_rows),
        wq_cols=np.asarray(wq_cols),
        moe_widths=np.asarray(moe_widths),
        wq_types=np.asarray(wq_types),
        greedy_bias=np.asarray(model.lm_head["bias"][_GREEDY_TOKEN]),
    )


def _ref_worker(name: str, out_path: str, queue: Any) -> None:
    try:
        _run(name, out_path, shard=False)
        queue.put(True)
    except BaseException as error:
        queue.put(f"{error}\n{traceback.format_exc()}")


def _tp_worker(name: str, rank: int, hostfile: str, out_path: str, queue: Any) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        path = out_path if rank == 0 else f"{out_path}.r{rank}"
        _run(name, path, shard=True)
        queue.put((rank, True, None))
    except BaseException as error:
        queue.put((rank, False, f"{error}\n{traceback.format_exc()}"))


def _diagnostics(name: str, reference: np.ndarray, sharded: np.ndarray, path: str) -> str:
    diff = np.abs(reference - sharded)
    return (
        f"[fixture={name} rank=0 layer_path={path} TP=2] "
        f"max_diff={float(diff.max()):.9g} mean_diff={float(diff.mean()):.9g}"
    )


def _assert_live_fixture(name: str, output: Any, rank: int) -> None:
    attention_norms = output["attention_norms"]
    moe_norms = output["moe_norms"]
    assert attention_norms.size == moe_norms.size == 4 * (_GREEDY_STEPS + 1), (
        f"[fixture={name} rank={rank}] expected all attention/MoE calls to execute"
    )
    assert np.all(attention_norms > 1e-6), (
        f"[fixture={name} rank={rank}] attention branch was inactive"
    )
    assert np.all(moe_norms > 1e-6), (
        f"[fixture={name} rank={rank}] routed/shared MoE branch was inactive"
    )
    assert np.all(np.max(np.abs(output["logits"]), axis=(0, 1)) > 1e-8), (
        f"[fixture={name} rank={rank}] a vocabulary projection row is inactive"
    )
    assert float(output["greedy_bias"]) == _GREEDY_BIAS, (
        f"[fixture={name} rank={rank}] greedy bias was not {_GREEDY_BIAS}"
    )
    decode_logits = output["greedy_logits"]
    target_states = decode_logits[..., _GREEDY_TOKEN].reshape(-1)
    assert np.all(np.diff(np.sort(target_states)) > 1e-4), (
        f"[fixture={name} rank={rank}] target logits are not distinct across five real backbone states"
    )
    non_target_ranges = np.ptp(decode_logits[..., :_GREEDY_TOKEN], axis=(0, 1))
    assert np.all(non_target_ranges > 1e-7), (
        f"[fixture={name} rank={rank}] a non-target projection row is constant across five real backbone states"
    )


def _run_compare(name: str, port_base: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        ref_path = f"{directory}/ref.npz"
        tp_path = f"{directory}/tp.npz"
        context = mp.get_context("spawn")
        queue = context.Queue()

        reference_process = context.Process(target=_ref_worker, args=(name, ref_path, queue))
        reference_process.start()
        reference_process.join(300)
        result = queue.get(timeout=10)
        if result is not True:
            pytest.fail(f"[fixture={name} rank=reference] FAIL: {result}")

        hosts = [f"127.0.0.1:{port_base + rank}" for rank in range(2)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
            json.dump(hosts, file)
            hostfile = file.name
        try:
            processes = [
                context.Process(target=_tp_worker, args=(name, rank, hostfile, tp_path, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=300) for _ in processes]
            for process in processes:
                process.join(60)
            for rank, ok, payload in results:
                if not ok:
                    pytest.fail(f"[fixture={name} rank={rank}] FAIL: {payload}")
        finally:
            os.unlink(hostfile)

        reference = np.load(ref_path)
        sharded = np.load(tp_path)
        sharded_rank_1 = np.load(f"{tp_path}.r1.npz")
        _assert_live_fixture(name, reference, rank=-1)
        _assert_live_fixture(name, sharded, rank=0)
        _assert_live_fixture(name, sharded_rank_1, rank=1)
        for _rank, output in ((0, sharded), (1, sharded_rank_1)):
            np.testing.assert_array_equal(output["heads"], np.full(4, 32))
            np.testing.assert_array_equal(output["wq_rows"], np.full(4, 512))
            np.testing.assert_array_equal(output["moe_widths"], np.full(4, 64))
            np.testing.assert_array_equal(
                output["wq_cols"], np.full(4, 4 if "q4" in name else 32)
            )
            np.testing.assert_array_equal(
                output["wq_types"],
                np.full(4, "QuantizedLinear" if "q4" in name else "Linear"),
            )

        diagnostics = _diagnostics(name, reference["logits"], sharded["logits"], "model.logits")
        np.testing.assert_allclose(
            reference["logits"], sharded["logits"], **TOLERANCES[name], err_msg=diagnostics
        )
        np.testing.assert_array_equal(
            np.argmax(reference["logits"], axis=-1),
            np.argmax(sharded["logits"], axis=-1),
            err_msg=diagnostics,
        )
        np.testing.assert_array_equal(
            sharded["tokens"],
            reference["tokens"],
            err_msg=_diagnostics(
                name, reference["greedy_logits"], sharded["greedy_logits"], "model.greedy_logits"
            ),
        )


pytestmark = [pytest.mark.slow]


@pytest.mark.parametrize("name", list(MODEL_CONFIGS))
def test_deepseek_v4_tensor_parallel_parity(name: str) -> None:
    port = 32100 + list(MODEL_CONFIGS).index(name) * 10
    _run_compare(name, port)
