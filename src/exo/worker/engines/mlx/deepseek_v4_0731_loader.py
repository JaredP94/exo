"""Strict EXO loading for the DeepSeek V4 Flash 0731 backbone."""

import copy
from pathlib import Path
from typing import Any, cast

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.deepseek_v4 import ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load_model as _mlx_load_model

from exo.worker.engines.mlx.deepseek_v4_0731_config import (
    DeepseekV40731CompatibilityError,
    QuantizationSpec,
    inspect_deepseek_v4_0731_checkpoint,
)
from exo.worker.engines.mlx.deepseek_v4_0731_model import (
    DeepseekV40731Model,
    install_deepseek_v4_0731_prefill_attention,
    install_deepseek_v4_sdpa_float32,
)
from exo.worker.runner.bootstrap import logger

_PREQUANTIZED_SWITCH_PROJECTIONS = (
    ".ffn.switch_mlp.gate_proj",
    ".ffn.switch_mlp.up_proj",
    ".ffn.switch_mlp.down_proj",
)


def _get_0731_classes(config: dict[str, Any]) -> tuple[type[nn.Module], type[Any]]:
    return DeepseekV40731Model, ModelArgs


def _mlx_loader_config(config: dict[str, Any]) -> dict[str, Any]:
    """Avoid re-quantizing the routed-expert FP4 modules built by DeepSeek V4."""
    loader_config = copy.deepcopy(config)
    quantization_value = cast(object, loader_config["quantization"])
    if not isinstance(quantization_value, dict):
        raise DeepseekV40731CompatibilityError("normalized quantization must be a mapping")
    quantization: dict[str, object] = {}
    for path, spec in cast(dict[object, object], quantization_value).items():
        if not isinstance(path, str):
            raise DeepseekV40731CompatibilityError(
                "normalized quantization keys must be strings"
            )
        quantization[path] = spec
    loader_config["quantization"] = {
        path: spec
        for path, spec in quantization.items()
        if not path.endswith(_PREQUANTIZED_SWITCH_PROJECTIONS)
    }
    if "quantization_config" in loader_config:
        loader_config["quantization_config"] = copy.deepcopy(
            loader_config["quantization"]
        )
    return loader_config


def load_exo_model(
    model_path: Path, *, lazy: bool = False, strict: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model, strictly auditing recognized 0731 backbone checkpoints."""
    checkpoint = inspect_deepseek_v4_0731_checkpoint(model_path)
    if checkpoint is None:
        return _mlx_load_model(model_path, lazy=lazy, strict=strict)

    model, normalized_config = _mlx_load_model(
        model_path,
        lazy=lazy,
        strict=True,
        model_config=_mlx_loader_config(checkpoint.config),
        get_model_classes=_get_0731_classes,
    )
    for field in ("quantization", "quantization_config"):
        if field in checkpoint.config:
            normalized_config[field] = copy.deepcopy(
                cast(object, checkpoint.config[field])
            )
    if not isinstance(model, DeepseekV40731Model):
        raise DeepseekV40731CompatibilityError(
            f"{model_path}: expected DeepseekV40731Model, got {type(model).__name__}"
        )
    install_deepseek_v4_0731_prefill_attention(model)
    install_deepseek_v4_sdpa_float32()
    audit_deepseek_v4_0731_quantization(model, checkpoint.realized_quantization)
    logger.warning(
        "deepseek_v4_0731_backbone loaded; "
        f"mtp_weight_count={checkpoint.mtp_weight_count}; "
        f"dspark_block_size={checkpoint.config['dspark_block_size']}; "
        f"target_layers={checkpoint.config['dspark_target_layer_ids']}; "
        "DSpark/MTP speculative decoding is disabled"
    )
    return model, normalized_config


def audit_deepseek_v4_0731_quantization(
    model: nn.Module, expected: dict[str, QuantizationSpec]
) -> None:
    """Ensure loaded concrete quantized modules exactly match checkpoint specs."""
    concrete_quantized_types = (
        nn.QuantizedLinear,
        nn.QuantizedEmbedding,
        QuantizedSwitchLinear,
    )
    realized: dict[str, QuantizationSpec] = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=_is_module):
        if isinstance(module, concrete_quantized_types):
            realized[path] = _realized_spec(path, module)

    expected_backbone = {
        path: spec for path, spec in expected.items() if not path.startswith("mtp.")
    }
    missing_paths = sorted(set(expected_backbone) - set(realized))
    unexpected_paths = sorted(set(realized) - set(expected_backbone))
    if missing_paths or unexpected_paths:
        details: list[str] = []
        for path in missing_paths:
            details.append(
                f"{path}: expected {_format_spec(expected_backbone[path])}, realized <missing>"
            )
        for path in unexpected_paths:
            details.append(
                f"{path}: expected <missing>, realized {_format_spec(realized[path])}"
            )
        raise DeepseekV40731CompatibilityError(
            "0731 quantization module paths differ: " + "; ".join(details)
        )

    for path, expected_spec in expected_backbone.items():
        realized_spec = realized[path]
        if realized_spec != expected_spec:
            raise DeepseekV40731CompatibilityError(
                f"{path}: expected {_format_spec(expected_spec)}, "
                f"realized {_format_spec(realized_spec)}"
            )


def _realized_spec(path: str, module: object) -> QuantizationSpec:
    bits = getattr(module, "bits", None)
    group_size = getattr(module, "group_size", None)
    mode = getattr(module, "mode", None)
    if (
        not isinstance(bits, int)
        or isinstance(bits, bool)
        or bits <= 0
        or not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
        or not isinstance(mode, str)
        or not mode
    ):
        raise DeepseekV40731CompatibilityError(
            f"{path}: expected a concrete (bits, group_size, mode) triple, "
            f"realized (bits={bits!r}, group_size={group_size!r}, mode={mode!r})"
        )
    return QuantizationSpec(bits=bits, group_size=group_size, mode=mode)


def _is_module(value: object) -> bool:
    return isinstance(value, nn.Module)


def _format_spec(spec: QuantizationSpec) -> str:
    return (
        f"(bits={spec.bits}, group_size={spec.group_size}, mode={spec.mode!r})"
    )
