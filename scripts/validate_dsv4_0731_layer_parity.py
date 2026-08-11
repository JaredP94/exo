#!/usr/bin/env python3
"""Hard-gate one real DeepSeek V4 Flash 0731 layer against read-only OMLX.

This command deliberately never calls either project's full checkpoint loader:
each source layer is selected from the safetensors index, remapped to layer 0,
loaded into one block at a time, then released before the next layer begins.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

BACKBONE_LAYER_COUNT = 43
REPRESENTATIVE_LAYERS = (0, 2, 3)
EXPECTED_REPRESENTATIVE_TOPOLOGY = {
    0: {"compress_ratio": 0, "indexer_present": False, "omlx_cache_branch_count": 1},
    2: {"compress_ratio": 4, "indexer_present": True, "omlx_cache_branch_count": 3},
    3: {"compress_ratio": 128, "indexer_present": False, "omlx_cache_branch_count": 2},
}
BF16_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}
QUANTIZED_TOLERANCE = {"rtol": 5e-2, "atol": 5e-2}
TOLERANCES = {"bf16": BF16_TOLERANCE, "quantized": QUANTIZED_TOLERANCE}
EXO_IMPORTED_PATHS = (
    "src/exo/worker/engines/mlx/deepseek_v4_0731_config.py",
    "src/exo/worker/engines/mlx/deepseek_v4_0731_model.py",
)
OMLX_IMPORTED_PACKAGE_TREE = "omlx"


class _SafeOpen(Protocol):
    def __enter__(self) -> _SafeOpen: ...

    def __exit__(self, *args: object) -> None: ...

    def get_tensor(self, key: str) -> Any: ...


SafeOpenFactory = Callable[[Path, str], _SafeOpen]


def _identity(value: Any) -> Any:
    return value


def select_layer_weight_map(index: Mapping[str, Any], source_layer: int) -> dict[str, str]:
    """Return only safetensor keys belonging to one requested backbone layer."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("model.safetensors.index.json has no object weight_map")
    prefix = f"model.layers.{source_layer}."
    selected = {
        key: shard
        for key, shard in weight_map.items()
        if isinstance(key, str) and key.startswith(prefix) and isinstance(shard, str)
    }
    if not selected:
        raise ValueError(f"no checkpoint tensors found for source layer {source_layer}")
    return selected


def remap_layer_weights(weights: Mapping[str, Any], source_layer: int) -> dict[str, Any]:
    """Move exactly one source-layer prefix to layer 0 without changing top keys."""
    source = f"model.layers.{source_layer}."
    target = "model.layers.0."
    remapped: dict[str, Any] = {}
    for path, value in weights.items():
        destination = target + path.removeprefix(source) if path.startswith(source) else path
        if destination in remapped:
            raise ValueError(f"layer remap would overwrite {destination}")
        remapped[destination] = value
    return remapped


def read_selected_tensors(
    model_path: Path,
    selected_weight_map: Mapping[str, str],
    *,
    safe_open_factory: SafeOpenFactory | None = None,
) -> dict[str, Any]:
    """Read selected keys only; safetensors supplies arrays in their native dtype."""
    convert_tensor: Callable[[Any], Any] = _identity
    if safe_open_factory is None:
        import mlx.core as mx
        from safetensors import safe_open

        safe_open_factory = safe_open
        convert_tensor = mx.array

    keys_by_shard: dict[str, list[str]] = {}
    for key, shard in selected_weight_map.items():
        keys_by_shard.setdefault(shard, []).append(key)
    tensors: dict[str, Any] = {}
    for shard, keys in keys_by_shard.items():
        # safetensors' MLX backend cannot decode BF16.  Its PyTorch backend
        # preserves all checkpoint dtypes, and each selected tensor is then
        # converted directly to MLX without materializing the shard.
        with safe_open_factory(model_path / shard, framework="pt") as handle:
            for key in keys:
                tensors[key] = convert_tensor(handle.get_tensor(key))
    return tensors


def validate_requested_layers(layers: Sequence[int]) -> list[int]:
    """Require exactly the fixed, representative layers used by the hard gate."""
    normalized = sorted(layers)
    if tuple(normalized) != REPRESENTATIVE_LAYERS:
        raise ValueError(
            "hard gate requires exactly representative layers "
            f"{list(REPRESENTATIVE_LAYERS)}"
        )
    return normalized


def tensor_metrics(
    expected: Any, actual: Any, *, tolerance_kind: str
) -> dict[str, Any]:
    """Compute numeric parity facts without depending on MLX at import time."""
    if tolerance_kind not in TOLERANCES:
        raise ValueError(f"unknown tolerance kind {tolerance_kind!r}")
    reference = np.asarray(expected, dtype=np.float64)
    candidate = np.asarray(actual, dtype=np.float64)
    if reference.shape != candidate.shape:
        return {
            "shape_expected": list(reference.shape),
            "shape_actual": list(candidate.shape),
            "passed": False,
            **TOLERANCES[tolerance_kind],
        }
    absolute = np.abs(reference - candidate)
    largest_flat_index = int(np.argmax(absolute)) if absolute.size else 0
    largest_index = (
        [int(index) for index in np.unravel_index(largest_flat_index, absolute.shape)]
        if absolute.size
        else []
    )
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    tolerance = TOLERANCES[tolerance_kind]
    tolerance_limit = tolerance["atol"] + tolerance["rtol"] * np.abs(reference)
    within_tolerance = absolute <= tolerance_limit
    finite_absolute = absolute[np.isfinite(absolute)]
    return {
        "shape": list(reference.shape),
        "max_abs_error": float(np.max(absolute, initial=0.0)),
        "mean_abs_error": float(np.mean(absolute)) if absolute.size else 0.0,
        "max_relative_error": float(np.max(relative, initial=0.0)),
        "largest_error_index": largest_index,
        "largest_expected": float(reference.flat[largest_flat_index]) if absolute.size else None,
        "largest_actual": float(candidate.flat[largest_flat_index]) if absolute.size else None,
        "largest_tolerance_limit": float(tolerance_limit.flat[largest_flat_index])
        if absolute.size
        else None,
        "allclose_violation_count": int(np.count_nonzero(~within_tolerance)),
        "allclose_violation_fraction": float(np.mean(~within_tolerance))
        if absolute.size
        else 0.0,
        "p99_abs_error": float(np.percentile(finite_absolute, 99))
        if finite_absolute.size
        else 0.0,
        "p999_abs_error": float(np.percentile(finite_absolute, 99.9))
        if finite_absolute.size
        else 0.0,
        "p9999_abs_error": float(np.percentile(finite_absolute, 99.99))
        if finite_absolute.size
        else 0.0,
        "passed": bool(
            np.allclose(reference, candidate, rtol=tolerance["rtol"], atol=tolerance["atol"])
        ),
        **tolerance,
    }


def _git_root(path: Path) -> Path:
    directory = path if path.is_dir() else path.parent
    return Path(
        subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_imported_files_clean(root: Path, paths: Sequence[str]) -> None:
    """Fail closed when a file imported by this comparison is locally changed."""
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty_paths = [line[3:] for line in result.stdout.splitlines() if line]
    if dirty_paths:
        raise RuntimeError(
            f"read-only reference guard failed for {root}: " + ", ".join(dirty_paths)
        )


def omlx_package_guard_paths(root: Path) -> tuple[str, ...]:
    """Guard every OMLX package file, including transitive custom kernels."""
    tree = root / OMLX_IMPORTED_PACKAGE_TREE
    if not tree.is_dir():
        raise RuntimeError(f"OMLX package tree is missing: {tree}")
    return tuple(
        str(path.relative_to(root))
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    )


def _checkpoint_sha256(model_path: Path) -> str:
    index_path = model_path / "model.safetensors.index.json"
    return hashlib.sha256(index_path.read_bytes()).hexdigest()


def _parse_layers(value: str) -> list[int]:
    try:
        return validate_requested_layers([int(item) for item in value.split(",") if item])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _reduced_configs(config: Mapping[str, Any], source_layer: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build OMLX raw and EXO-normalized one-layer configurations."""
    raw = copy.deepcopy(dict(config))
    ratios = raw.get("compress_ratios")
    if not isinstance(ratios, list) or source_layer >= len(ratios):
        raise ValueError(f"compress_ratios does not include source layer {source_layer}")
    raw["num_hidden_layers"] = 1
    raw["compress_ratios"] = [ratios[source_layer]]
    raw["num_hash_layers"] = 1 if source_layer < int(config["num_hash_layers"]) else 0
    source_prefix = f"model.layers.{source_layer}."
    target_prefix = "model.layers.0."
    quantization = raw.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("checkpoint config lacks a quantization mapping")
    raw["quantization"] = {
        (target_prefix + path.removeprefix(source_prefix)) if path.startswith(source_prefix) else path: value
        for path, value in quantization.items()
        if path in {"bits", "group_size", "mode"} or path.startswith(source_prefix)
    }
    raw["quantization_config"] = copy.deepcopy(raw["quantization"])

    from exo.worker.engines.mlx.deepseek_v4_0731_config import (
        normalize_deepseek_v4_0731_config,
    )

    exo_config, _ = normalize_deepseek_v4_0731_config(raw)
    return raw, exo_config


def quantization_request(
    path: str, module: object, quantization: Mapping[str, Any], prefix: str
) -> bool | object:
    """Return an explicit spec unless OMLX already owns the quantized module."""
    if type(module).__name__.startswith("Quantized"):
        return False
    return quantization.get(prefix + path, False)


def _quantize_layer(module: Any, quantization: Mapping[str, Any], prefix: str) -> None:
    """Quantize only paths present in the selected layer's original mapping."""
    import mlx.nn as nn

    defaults = {field: quantization[field] for field in ("bits", "group_size", "mode")}

    def predicate(path: str, candidate: Any) -> bool | object:
        return quantization_request(path, candidate, quantization, prefix)

    nn.quantize(
        module,
        group_size=cast(int, defaults["group_size"]),
        bits=cast(int, defaults["bits"]),
        mode=cast(str, defaults["mode"]),
        class_predicate=predicate,
    )


def _layer_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "model.layers.0."
    selected = {
        path.removeprefix(prefix): value for path, value in weights.items() if path.startswith(prefix)
    }
    if not selected:
        raise ValueError("sanitization produced no selected layer weights")
    return selected


def _load_runtime(omlx_root: Path) -> tuple[Any, Any, Any]:
    """Import EXO first, then OMLX's versioned module without editing either."""
    from mlx_lm.models.deepseek_v4 import ModelArgs as ExoModelArgs

    from exo.worker.engines.mlx.deepseek_v4_0731_model import DeepseekV40731Model

    prepare_omlx_import(omlx_root)
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    # Keep EXO's class objects above, but release these registry names so OMLX
    # can install its independent reference implementation in this process.
    # The operation is process-local and does not mutate either checkout.
    sys.modules.pop("mlx_lm.models.deepseek_v4", None)
    sys.modules.pop("mlx_lm.models.hyper_connection", None)
    apply_deepseek_v4_patch()
    reference_module = importlib.import_module("mlx_lm.models.deepseek_v4")
    if not str(reference_module.__file__).startswith(str(omlx_root)):
        raise RuntimeError("OMLX DeepSeek V4 reference module was not installed")
    return DeepseekV40731Model, ExoModelArgs, reference_module


def prepare_omlx_import(omlx_root: Path) -> None:
    """Prevent OMLX imports from writing ignored bytecode into the reference tree."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(omlx_root))


def new_omlx_prefill_cache(container: Any, *, compress_ratio: int) -> Any:
    """Create one fresh OMLX layer cache, retaining compressed pool branches."""
    caches = container.make_cache()
    if len(caches) != 1:
        raise ValueError("reduced OMLX comparison model must have exactly one layer cache")
    cache = caches[0]
    cache_branches = getattr(cache, "caches", cache)
    if compress_ratio and len(cache_branches) < 2:
        raise ValueError("compressed OMLX prefill requires a local and pool cache")
    return cache


def omlx_attention_cache_facts(attention: Any, cache: Any) -> dict[str, int | bool]:
    """Report whether this reference attention layer owns an indexer pool."""
    branches = getattr(cache, "caches", None)
    return {
        "indexer_present": hasattr(attention, "indexer"),
        "omlx_cache_branch_count": len(branches) if branches is not None else 1,
    }


def topology_matches_expected(
    source_layer: int, compress_ratio: int, cache_facts: Mapping[str, int | bool]
) -> bool:
    """Require each selected layer's known compression/cache topology."""
    expected = EXPECTED_REPRESENTATIVE_TOPOLOGY.get(source_layer)
    return expected == {
        "compress_ratio": compress_ratio,
        "indexer_present": cache_facts.get("indexer_present"),
        "omlx_cache_branch_count": cache_facts.get("omlx_cache_branch_count"),
    }


def _load_one_layer(
    source_layer: int,
    config: Mapping[str, Any],
    model_path: Path,
    reference_module: Any,
    exo_model_class: Any,
    exo_args_class: Any,
) -> tuple[Any, Any, dict[str, Any], Any, Any, Any]:
    """Select, sanitize, quantize, and load exactly one reference and EXO layer."""
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    selected = select_layer_weight_map(index, source_layer)
    raw_weights = remap_layer_weights(read_selected_tensors(model_path, selected), source_layer)
    omlx_config, exo_config = _reduced_configs(config, source_layer)

    omlx_args = reference_module.ModelArgs.from_dict(omlx_config)
    omlx_container = reference_module.Model(omlx_args)
    omlx_weights = omlx_container.sanitize(dict(raw_weights))
    omlx_block = reference_module.DeepseekV4Block(omlx_args, 0)
    _quantize_layer(omlx_block, omlx_config["quantization"], "model.layers.0.")
    omlx_block.load_weights(list(_layer_weights(omlx_weights).items()), strict=True)
    omlx_attention_cache = new_omlx_prefill_cache(
        omlx_container, compress_ratio=int(omlx_config["compress_ratios"][0])
    )
    omlx_block_cache = new_omlx_prefill_cache(
        omlx_container, compress_ratio=int(omlx_config["compress_ratios"][0])
    )
    omlx_intermediate_cache = new_omlx_prefill_cache(
        omlx_container, compress_ratio=int(omlx_config["compress_ratios"][0])
    )
    del omlx_container

    exo_args = exo_args_class.from_dict(exo_config)
    exo_container = exo_model_class(exo_args)
    exo_weights = exo_container.sanitize(dict(raw_weights))
    exo_block = exo_container.model.layers[0]
    _quantize_layer(exo_block, exo_config["quantization"], "model.layers.0.")
    exo_block.load_weights(list(_layer_weights(exo_weights).items()), strict=True)
    from exo.worker.engines.mlx.deepseek_v4_0731_model import (
        install_deepseek_v4_0731_attention_prefill_split,
    )

    install_deepseek_v4_0731_attention_prefill_split(exo_block.attn)
    del exo_container
    return (
        omlx_block,
        exo_block,
        exo_config,
        omlx_attention_cache,
        omlx_block_cache,
        omlx_intermediate_cache,
    )


def _to_numpy(array: Any) -> np.ndarray:
    import mlx.core as mx

    # NumPy cannot consume MLX BF16's PEP 3118 buffer directly.  Converting
    # only the already-computed comparison result to F32 preserves the tensor
    # values while keeping selected checkpoint parameters in their native dtype.
    materialized = array.astype(mx.float32) if array.dtype == mx.bfloat16 else array
    mx.eval(materialized)
    return np.array(materialized)


def _weighted_routed_output(block: Any, hidden: Any, indices: Any, scores: Any) -> Any:
    routed = block.ffn.switch_mlp(hidden, indices)
    if routed.ndim == scores.ndim + 1:
        return (routed * scores[..., None].astype(routed.dtype)).sum(axis=-2)
    return routed


def actual_moe_output(block: Any, hidden: Any, input_ids: Any) -> Any:
    """Exercise the implementation's production MoE aggregation path."""
    return block.ffn(hidden, input_ids)


def remapped_layer_uses_hash_routing(config: Mapping[str, Any]) -> bool:
    """A selected source becomes layer 0, so the reduced gate owns the mode."""
    return int(config["num_hash_layers"]) > 0


def build_omlx_prefill_mask(
    batch: int,
    sequence: int,
    window: int,
    build_window_mask: Callable[[int, int, int, int, int], Any],
) -> Any:
    """Mirror the pinned prefill window mask for OMLX's caller-owned API."""
    return build_window_mask(batch, sequence, 0, window, sequence)


def _path_metric(name: str, expected: Any, actual: Any, tolerance: str) -> tuple[str, dict[str, Any]]:
    return name, tensor_metrics(_to_numpy(expected), _to_numpy(actual), tolerance_kind=tolerance)


def trace_local_attention_boundaries(
    omlx_attention: Any, exo_attention: Any, hidden: Any, mask: Any | None = None
) -> dict[str, dict[str, Any]]:
    """Test-only layer-0 trace across each local-attention data boundary."""
    import mlx.core as mx

    exo_globals = exo_attention.__call__.__globals__
    batch, sequence, _ = hidden.shape
    pre_omlx = omlx_attention.config  # Preserve the source-specific eps below.
    del pre_omlx
    o_input = hidden
    e_input = hidden
    o_q_a = omlx_attention.wq_a(o_input)
    e_qkv = exo_attention.wqkv_a(e_input)
    e_q_a = e_qkv[..., : exo_attention.q_lora_rank]
    o_kv_raw = omlx_attention.wkv(o_input)
    e_kv_raw = e_qkv[..., exo_attention.q_lora_rank :]
    o_q_norm = omlx_attention.q_norm(o_q_a)
    e_q_norm = exo_attention.q_norm(e_q_a)
    o_kv_norm = omlx_attention.kv_norm(o_kv_raw)
    e_kv_norm = exo_attention.kv_norm(e_kv_raw)
    o_q_b = omlx_attention.wq_b(o_q_norm)
    e_q_b = exo_attention.wq_b(e_q_norm)
    o_q = mx.fast.rms_norm(
        o_q_b.reshape(batch, sequence, omlx_attention.n_heads, omlx_attention.head_dim),
        None,
        omlx_attention.config.rms_norm_eps,
    ).transpose(0, 2, 1, 3)
    e_q = exo_globals["_attn_q_proj_norm"](
        e_q_b,
        exo_attention.n_heads,
        exo_attention.head_dim,
        exo_attention.eps,
    )
    o_q_rope = omlx_attention.rope(o_q, 0)
    o_kv_rope = omlx_attention.rope(
        o_kv_norm.reshape(batch, 1, sequence, omlx_attention.head_dim), 0
    )
    e_q_rope, e_kv_rope = exo_globals["_attn_qkv_partial_rope"](
        e_q,
        e_kv_norm,
        0,
        exo_attention.rope_head_dim,
        exo_attention.rope.freqs,
    )
    e_kv_rope = e_kv_rope[:, None]
    o_sdpa = exo_globals["scaled_dot_product_attention"](
        o_q_rope,
        o_kv_rope,
        o_kv_rope,
        cache=None,
        scale=omlx_attention.scale,
        mask=mask,
        sinks=omlx_attention.attn_sink.astype(o_q_rope.dtype),
    )
    e_sdpa = exo_globals["scaled_dot_product_attention"](
        e_q_rope,
        e_kv_rope,
        e_kv_rope,
        cache=None,
        scale=exo_attention.scale,
        mask=mask,
        sinks=exo_attention._sink_for(e_q_rope.dtype),
    )
    o_flat = omlx_attention.rope(o_sdpa, 0, inverse=True).transpose(0, 2, 1, 3).reshape(
        batch, sequence, -1
    )
    e_flat = exo_globals["_attn_inv_rope_flatten"](
        e_sdpa,
        0,
        exo_attention.rope_head_dim,
        exo_attention.rope.freqs,
        exo_attention.n_heads * exo_attention.head_dim,
    )
    o_grouped_input = o_flat.reshape(batch, sequence, omlx_attention.o_groups, -1).transpose(
        0, 2, 1, 3
    )
    e_grouped_input = e_flat.reshape(batch, sequence, exo_attention.n_groups, -1).transpose(
        0, 2, 1, 3
    )
    o_wo_a = omlx_attention.wo_a(o_grouped_input).transpose(0, 2, 1, 3)
    e_wo_a = exo_attention._grouped_output_projection(e_flat).reshape(
        batch, sequence, exo_attention.n_groups, exo_attention.o_lora_rank
    )
    o_wo_b = omlx_attention.wo_b(o_wo_a.reshape(batch, sequence, -1))
    e_wo_b = exo_attention.wo_b(e_wo_a.reshape(batch, sequence, -1))
    o_call = omlx_attention(hidden, mask=mask, cache=None)
    e_call = exo_attention(hidden, cache=None)
    boundaries = (
        ("pre_norm_input", o_input, e_input, "bf16"),
        ("q_a", o_q_a, e_q_a, "quantized"),
        ("wkv", o_kv_raw, e_kv_raw, "quantized"),
        ("q_norm", o_q_norm, e_q_norm, "bf16"),
        ("kv_norm", o_kv_norm, e_kv_norm, "bf16"),
        ("q_b", o_q_b, e_q_b, "quantized"),
        ("q_head_norm", o_q, e_q, "bf16"),
        ("rope_q", o_q_rope, e_q_rope, "bf16"),
        ("rope_kv", o_kv_rope, e_kv_rope, "bf16"),
        ("sdpa", o_sdpa, e_sdpa, "bf16"),
        ("grouped_wo_a_input", o_grouped_input, e_grouped_input, "bf16"),
        ("grouped_wo_a_output", o_wo_a, e_wo_a, "quantized"),
        ("wo_b", o_wo_b, e_wo_b, "quantized"),
        ("omlx_call_vs_boundaries", o_call, o_wo_b, "quantized"),
        ("exo_call_vs_boundaries", e_call, e_wo_b, "quantized"),
        ("attention_call", o_call, e_call, "quantized"),
    )
    return dict(_path_metric(name, expected, actual, tolerance) for name, expected, actual, tolerance in boundaries)


def trace_gate_and_routed_boundaries(
    omlx_block: Any, exo_block: Any, hidden: Any, input_ids: Any
) -> dict[str, Any]:
    """Test-only gate/MoE trace; layer 0 has no correction bias by design."""
    omlx_gate = omlx_block.ffn.gate
    exo_gate = exo_block.ffn.gate
    omlx_globals = omlx_gate.__call__.__globals__
    exo_globals = exo_gate.__call__.__globals__
    omlx_logits = omlx_globals["decode_matmul"](hidden, omlx_gate.weight.T)
    exo_logits = (hidden @ exo_gate.weight.T).astype(exo_globals["mx"].float32)
    omlx_scores = omlx_globals["_score_func"](
        omlx_logits.astype(omlx_globals["mx"].float32), omlx_gate.scoring_func
    )
    exo_scores = exo_globals["_score_func"](exo_logits, exo_gate.score_func)
    omlx_indices, omlx_returned = omlx_gate(hidden, input_ids)
    exo_indices, exo_returned = exo_gate(hidden, input_ids)
    omlx_topk = omlx_globals["mx"].take_along_axis(omlx_scores, omlx_indices, axis=-1)
    exo_topk = exo_globals["mx"].take_along_axis(exo_scores, exo_indices, axis=-1)
    omlx_normalized = omlx_topk / (omlx_topk.sum(axis=-1, keepdims=True) + 1e-20)
    exo_normalized = exo_topk / (exo_topk.sum(axis=-1, keepdims=True) + 1e-20)
    omlx_scaled = omlx_normalized * omlx_gate.routed_scaling_factor
    exo_scaled = exo_normalized * exo_gate.route_scale
    omlx_raw = omlx_block.ffn.switch_mlp(hidden, omlx_indices)
    exo_raw = exo_block.ffn.switch_mlp(hidden, exo_indices)
    omlx_weighted = (omlx_raw * omlx_returned[..., None].astype(omlx_raw.dtype)).sum(axis=-2)
    exo_weighted = (exo_raw * exo_returned[..., None].astype(exo_raw.dtype)).sum(axis=-2)
    boundaries = (
        ("raw_logits", omlx_logits, exo_logits, "bf16"),
        ("score_function", omlx_scores, exo_scores, "bf16"),
        ("topk_values", omlx_topk, exo_topk, "bf16"),
        ("normalized_weights", omlx_normalized, exo_normalized, "bf16"),
        ("route_scaled_weights", omlx_scaled, exo_scaled, "bf16"),
        ("returned_gate_weights", omlx_returned, exo_returned, "bf16"),
        ("raw_selected_expert_outputs", omlx_raw, exo_raw, "quantized"),
        ("weighted_routed_sum", omlx_weighted, exo_weighted, "quantized"),
    )
    return {
        "hash_routing": bool(omlx_gate.hash),
        "correction_bias": "not applicable for hash routing"
        if omlx_gate.hash
        else "compared by the non-hash gate path",
        "indices_equal": bool(
            np.array_equal(_to_numpy(omlx_indices), _to_numpy(exo_indices))
        ),
        "boundaries": dict(
            _path_metric(name, expected, actual, tolerance)
            for name, expected, actual, tolerance in boundaries
        ),
    }


def _compare_layer(
    source_layer: int,
    omlx_block: Any,
    exo_block: Any,
    exo_config: Mapping[str, Any],
    omlx_attention_cache: Any,
    omlx_block_cache: Any,
    omlx_intermediate_cache: Any,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx

    ratio = int(exo_config["compress_ratios"][0])
    length = 128 if ratio == 128 else 8 if ratio == 4 else 4
    mx.random.seed(seed + source_layer)
    dense_hidden = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(1, length, int(exo_config["hidden_size"])),
    ).astype(mx.bfloat16)
    hidden = mx.broadcast_to(
        dense_hidden[:, :, None, :],
        (1, length, int(exo_config["hc_mult"]), int(exo_config["hidden_size"])),
    )
    input_ids = mx.arange(length, dtype=mx.int32)[None] + 17
    comparison_mask = build_omlx_prefill_mask(
        1,
        length,
        exo_block.attn.window,
        exo_block.attn.__call__.__globals__["_build_window_mask"],
    )

    omlx_indices, omlx_scores = omlx_block.ffn.gate(dense_hidden, input_ids)
    exo_indices, exo_scores = exo_block.ffn.gate(dense_hidden, input_ids)
    routing_equal = bool(np.array_equal(_to_numpy(omlx_indices), _to_numpy(exo_indices)))

    omlx_attention = omlx_block.attn(
        omlx_block.attn_norm(dense_hidden), mask=comparison_mask, cache=omlx_attention_cache
    )
    exo_attention = exo_block.attn(exo_block.attn_norm(dense_hidden), cache=None)
    omlx_routed = _weighted_routed_output(omlx_block, dense_hidden, omlx_indices, omlx_scores)
    exo_routed = _weighted_routed_output(exo_block, dense_hidden, exo_indices, exo_scores)
    omlx_moe_direct = actual_moe_output(omlx_block, dense_hidden, input_ids)
    exo_moe_direct = actual_moe_output(exo_block, dense_hidden, input_ids)

    omlx_residual = hidden
    omlx_attn_input, omlx_attn_post, omlx_attn_comb = omlx_block.attn_hc(hidden)
    omlx_attn_actual = omlx_block.attn(
        omlx_block.attn_norm(omlx_attn_input),
        mask=comparison_mask,
        cache=omlx_intermediate_cache,
    )
    omlx_hc_expand = omlx_block.__call__.__globals__["hc_expand"]
    omlx_after_attn = omlx_hc_expand(
        omlx_attn_actual, omlx_residual, omlx_attn_post, omlx_attn_comb
    )
    omlx_ffn_input, _, _ = omlx_block.ffn_hc(omlx_after_attn)
    omlx_ffn_normalized = omlx_block.ffn_norm(omlx_ffn_input)

    exo_residual = hidden
    exo_attn_input, exo_attn_post, exo_attn_comb = exo_block.hc_attn.hc_pre(hidden)
    exo_attn_actual = exo_block.attn(exo_block.attn_norm(exo_attn_input), cache=None)
    exo_after_attn = exo_block.hc_attn.hc_post(
        exo_attn_actual, exo_residual, exo_attn_post, exo_attn_comb
    )
    exo_ffn_input, _, _ = exo_block.hc_ffn.hc_pre(exo_after_attn)
    exo_ffn_normalized = exo_block.ffn_norm(exo_ffn_input)
    omlx_moe_hc_ffn = actual_moe_output(omlx_block, omlx_ffn_normalized, input_ids)
    exo_moe_hc_ffn = actual_moe_output(exo_block, exo_ffn_normalized, input_ids)

    clamp_hidden = dense_hidden * 64.0
    omlx_gate = omlx_block.ffn.shared_experts.gate_proj(clamp_hidden)
    omlx_up = omlx_block.ffn.shared_experts.up_proj(clamp_hidden)
    exo_gate = exo_block.ffn.shared_experts.gate_proj(clamp_hidden)
    exo_up = exo_block.ffn.shared_experts.up_proj(clamp_hidden)
    clamp_exercised = bool(
        max(
            float(np.max(np.abs(_to_numpy(omlx_gate)))),
            float(np.max(np.abs(_to_numpy(omlx_up)))),
            float(np.max(np.abs(_to_numpy(exo_gate)))),
            float(np.max(np.abs(_to_numpy(exo_up)))),
        )
        > 10.0
    )
    omlx_shared = omlx_block.ffn.shared_experts(clamp_hidden)
    exo_shared = exo_block.ffn.shared_experts(clamp_hidden)
    omlx_block_out = omlx_block(
        hidden, mask=comparison_mask, cache=omlx_block_cache, input_ids=input_ids
    )
    exo_block_out = exo_block(hidden, cache=None, input_ids=input_ids)

    metrics = dict(
        [
            _path_metric("gate_scores", omlx_scores, exo_scores, "bf16"),
            _path_metric("attention_output", omlx_attention, exo_attention, "quantized"),
            _path_metric("routed_expert_output", omlx_routed, exo_routed, "quantized"),
            _path_metric("moe_output_direct", omlx_moe_direct, exo_moe_direct, "quantized"),
            _path_metric(
                "moe_output_hc_ffn_normalized",
                omlx_moe_hc_ffn,
                exo_moe_hc_ffn,
                "quantized",
            ),
            _path_metric("shared_expert_output", omlx_shared, exo_shared, "quantized"),
            _path_metric("block_output", omlx_block_out, exo_block_out, "quantized"),
        ]
    )
    divergences = [
        (name, metric)
        for name, metric in metrics.items()
        if "max_abs_error" in metric
    ]
    largest_path, largest = max(divergences, key=lambda item: item[1]["max_abs_error"])
    topology_facts = omlx_attention_cache_facts(omlx_block.attn, omlx_attention_cache)
    topology_ok = topology_matches_expected(source_layer, ratio, topology_facts)
    return {
        "source_layer": source_layer,
        "compress_ratio": ratio,
        **topology_facts,
        "topology_matches_expected": topology_ok,
        "hash_routing": remapped_layer_uses_hash_routing(exo_config),
        "routing_indices_equal": routing_equal,
        "shared_expert_clamp_exercised": clamp_exercised,
        "paths": metrics,
        "largest_divergence": {
            "path": largest_path,
            "index": largest["largest_error_index"],
            "max_abs_error": largest["max_abs_error"],
        },
        "passed": topology_ok
        and routing_equal
        and clamp_exercised
        and all(metric["passed"] for metric in metrics.values()),
    }


def run_parity(
    model_path: Path, omlx_root: Path, layers: Sequence[int], seed: int
) -> dict[str, Any]:
    """Run sequential, fixed-layer parity and return a JSON-safe hard-gate record."""
    import mlx.core as mx

    layers = validate_requested_layers(layers)
    exo_root = _git_root(Path(__file__).resolve())
    omlx_root = _git_root(omlx_root)
    assert_imported_files_clean(exo_root, EXO_IMPORTED_PATHS)
    assert_imported_files_clean(omlx_root, omlx_package_guard_paths(omlx_root))
    config = json.loads((model_path / "config.json").read_text())
    exo_model_class, exo_args_class, reference_module = _load_runtime(omlx_root)
    records: list[dict[str, Any]] = []
    peak_active_memory = 0
    for source_layer in layers:
        (
            omlx_block,
            exo_block,
            exo_config,
            omlx_attention_cache,
            omlx_block_cache,
            omlx_intermediate_cache,
        ) = _load_one_layer(
            source_layer,
            config,
            model_path,
            reference_module,
            exo_model_class,
            exo_args_class,
        )
        try:
            record = _compare_layer(
                source_layer,
                omlx_block,
                exo_block,
                exo_config,
                omlx_attention_cache,
                omlx_block_cache,
                omlx_intermediate_cache,
                seed,
            )
            records.append(record)
            peak_active_memory = max(peak_active_memory, int(mx.get_active_memory()))
        finally:
            del omlx_block
            del exo_block
            mx.clear_cache()
    config_facts = {
        "num_hidden_layers": config.get("num_hidden_layers"),
        "requested_layers": list(layers),
        "representative_layers_exact": tuple(layers) == REPRESENTATIVE_LAYERS,
        "complete_model_allocated": False,
        "mtp_execution_enabled": False,
    }
    no_full_model = (
        config_facts["complete_model_allocated"] is False
        and config_facts["mtp_execution_enabled"] is False
    )
    passed = (
        bool(records)
        and config_facts["representative_layers_exact"] is True
        and no_full_model
        and all(record["passed"] for record in records)
    )
    return {
        "passed": passed,
        "layers": records,
        "config_facts": config_facts,
        "peak_active_memory": peak_active_memory,
        "exo_sha": _git_sha(exo_root),
        "omlx_sha": _git_sha(omlx_root),
        "checkpoint_index_sha256": _checkpoint_sha256(model_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--omlx-root", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model_path = args.model_path.resolve()
    omlx_root = args.omlx_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(model_path) or output.is_relative_to(omlx_root):
        parser.error("--output must not be beneath the read-only model or OMLX roots")
    try:
        result = run_parity(model_path, omlx_root, args.layers, args.seed)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
