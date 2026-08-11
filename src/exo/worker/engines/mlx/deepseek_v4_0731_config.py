"""Pure compatibility checks for DeepSeek V4 Flash 0731 checkpoints."""

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class DeepseekV40731CompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizationSpec:
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class DeepseekV40731Checkpoint:
    model_path: Path
    config: dict[str, Any]
    weight_keys: frozenset[str]
    mtp_weight_count: int
    mtp_quantization_count: int
    realized_quantization: dict[str, QuantizationSpec]


def realized_quantization_path(path: str) -> str:
    """Map source module names to the fused paths realized by DeepSeek V4."""
    path = re.sub(r"(\.attn)\.(wq_a|wkv)$", r"\1.wqkv_a", path)
    return re.sub(
        r"(\.attn(?:\.indexer)?\.compressor)\.(wkv|wgate)$",
        r"\1.wkv_gate",
        path,
    )


def normalize_deepseek_v4_0731_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, QuantizationSpec]]:
    """Copy and normalize 0731's source quantization declarations."""
    normalized: dict[str, Any] = copy.deepcopy(dict(config))
    _validate_compress_ratios(normalized)
    _validate_global_quantization(normalized)

    quantization_value = cast(object, normalized.get("quantization", {}))
    quantization = _require_string_key_mapping(quantization_value, "quantization")

    global_quantization = {
        field: quantization[field]
        for field in ("bits", "group_size", "mode")
        if field in quantization
    }
    realized: dict[str, QuantizationSpec] = {}
    for source_path, source_spec in quantization.items():
        if source_path in {"bits", "group_size", "mode"}:
            continue
        spec = _parse_quantization_spec(source_path, source_spec)
        if source_path.startswith("mtp."):
            continue

        destination_path = realized_quantization_path(source_path)
        existing = realized.get(destination_path)
        if existing is not None and existing != spec:
            raise DeepseekV40731CompatibilityError(
                "incompatible quantization declarations for "
                f"{_first_source_for_destination(quantization, destination_path)} and {source_path}"
            )
        realized[destination_path] = spec

    normalized_quantization = global_quantization | {
        path: _spec_as_dict(spec) for path, spec in realized.items()
    }
    if len(normalized_quantization) != len(realized) + len(global_quantization):
        raise DeepseekV40731CompatibilityError(
            "a non-MTP quantization declaration has no realized replacement"
        )
    normalized["quantization"] = normalized_quantization
    if "quantization_config" in normalized:
        normalized["quantization_config"] = copy.deepcopy(normalized_quantization)

    return normalized, realized


def normalized_quantization_specs(
    normalized_config: Mapping[str, Any],
) -> dict[str, QuantizationSpec]:
    """Reconstruct explicit quantization specs from a normalized 0731 config."""
    quantization = _require_string_key_mapping(
        cast(object, normalized_config.get("quantization", {})), "quantization"
    )
    return {
        path: _parse_quantization_spec(path, value)
        for path, value in quantization.items()
        if not path.startswith("mtp.") and path not in {"bits", "group_size", "mode"}
    }


def inspect_deepseek_v4_0731_checkpoint(
    model_path: Path,
) -> DeepseekV40731Checkpoint | None:
    """Inspect config and index JSON only; never import MLX or open a shard."""
    config_value = cast(object, json.loads((model_path / "config.json").read_text()))
    raw_config = _require_string_key_mapping(
        config_value,
        f"{model_path}: config.json",
    )
    if raw_config.get("model_type") != "deepseek_v4":
        return None

    dspark_fields_present = {
        field
        for field in ("dspark_block_size", "dspark_target_layer_ids")
        if field in raw_config
    }
    valid_dspark_block_size = _is_positive_int(raw_config.get("dspark_block_size"))
    valid_dspark_targets = _is_nonempty_integer_list(
        raw_config.get("dspark_target_layer_ids")
    )
    if dspark_fields_present and not (valid_dspark_block_size and valid_dspark_targets):
        offending_fields = _malformed_dspark_fields(
            raw_config,
            valid_dspark_block_size,
            valid_dspark_targets,
        )
        raise DeepseekV40731CompatibilityError(
            f"{model_path}: malformed DSpark fields: {', '.join(offending_fields)}"
        )

    has_dspark_discriminators = valid_dspark_block_size and valid_dspark_targets
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        if has_dspark_discriminators:
            raise DeepseekV40731CompatibilityError(
                f"{model_path}: a 0731 checkpoint requires model.safetensors.index.json"
            )
        return None

    index_value = cast(object, json.loads(index_path.read_text()))
    index = _require_string_key_mapping(
        index_value,
        f"{model_path}: model.safetensors.index.json",
    )
    weight_map_value: object = index.get("weight_map")
    weight_map = _require_string_key_mapping(
        weight_map_value,
        f"{model_path}: weight_map",
    )
    weight_keys = frozenset(weight_map)

    has_0731_weight_evidence = _has_0731_weight_evidence(weight_keys)
    if not has_dspark_discriminators and not has_0731_weight_evidence:
        return None
    if not has_dspark_discriminators:
        raise DeepseekV40731CompatibilityError(
            f"{model_path}: 0731 weight evidence requires DSpark fields: "
            "dspark_block_size, dspark_target_layer_ids"
        )

    normalized_config, realized_quantization = normalize_deepseek_v4_0731_config(
        raw_config
    )
    quantization_value = raw_config.get("quantization", {})
    quantization = _require_string_key_mapping(
        quantization_value,
        f"{model_path}: quantization",
    )
    return DeepseekV40731Checkpoint(
        model_path=model_path,
        config=normalized_config,
        weight_keys=weight_keys,
        mtp_weight_count=sum(key.startswith("mtp.") for key in weight_keys),
        mtp_quantization_count=sum(path.startswith("mtp.") for path in quantization),
        realized_quantization=realized_quantization,
    )


def validate_deepseek_v4_0731_shard_geometry(
    config: Mapping[str, Any],
    realized_quantization: Mapping[str, QuantizationSpec],
    world_size: int,
) -> None:
    """Reject tensor-parallel layouts incompatible with explicit quantization."""
    if not _is_positive_int(world_size):
        raise DeepseekV40731CompatibilityError("world_size must be at least 1")

    num_attention_heads = _required_positive_int(config, "num_attention_heads")
    o_groups = _required_positive_int(config, "o_groups")
    head_dim = _required_positive_int(config, "head_dim")
    moe_intermediate_size = _required_positive_int(config, "moe_intermediate_size")
    n_shared_experts = _required_positive_int(config, "n_shared_experts")
    if num_attention_heads % o_groups != 0:
        raise DeepseekV40731CompatibilityError(
            "num_attention_heads must be divisible by o_groups"
        )
    heads_per_group = num_attention_heads // o_groups
    if heads_per_group % world_size != 0:
        raise DeepseekV40731CompatibilityError(
            f"heads_per_group={heads_per_group} must be divisible by world_size={world_size}"
        )

    widths_by_suffix = {
        ".attn.wo_a": heads_per_group * head_dim,
        ".ffn.switch_mlp.down_proj": moe_intermediate_size,
        ".ffn.shared_experts.down_proj": moe_intermediate_size * n_shared_experts,
    }
    for path, spec in realized_quantization.items():
        if path.startswith("mtp."):
            continue
        logical_input_width = next(
            (
                width
                for suffix, width in widths_by_suffix.items()
                if path.endswith(suffix)
            ),
            None,
        )
        if logical_input_width is None:
            continue
        divisor = spec.group_size * world_size
        if logical_input_width % divisor != 0:
            raise DeepseekV40731CompatibilityError(
                f"{path}: logical input width {logical_input_width} is not divisible by "
                f"group_size * world_size ({spec.group_size} * {world_size})"
            )


def _validate_compress_ratios(config: Mapping[str, Any]) -> None:
    num_hidden_layers = _required_positive_int(config, "num_hidden_layers")
    compress_ratios: object = config.get("compress_ratios")
    if not isinstance(compress_ratios, Sequence) or isinstance(compress_ratios, str):
        raise DeepseekV40731CompatibilityError("compress_ratios must be a sequence")
    ratios = cast(Sequence[object], compress_ratios)
    if len(ratios) < num_hidden_layers:
        raise DeepseekV40731CompatibilityError(
            "compress_ratios must cover every hidden layer"
        )
    if any(
        not isinstance(ratio, int)
        or isinstance(ratio, bool)
        or ratio not in {0, 4, 128}
        for ratio in ratios
    ):
        raise DeepseekV40731CompatibilityError(
            "compress_ratios may contain only 0, 4, or 128"
        )


def _validate_global_quantization(config: Mapping[str, Any]) -> None:
    fields = ("bits", "group_size", "mode")
    present_fields = [field for field in fields if field in config]
    if present_fields and len(present_fields) != len(fields):
        raise DeepseekV40731CompatibilityError(
            "global quantization must contain bits, group_size, and mode"
        )
    if present_fields:
        _parse_quantization_spec(
            "global quantization",
            {field: config[field] for field in fields},
        )


def _parse_quantization_spec(path: str, value: object) -> QuantizationSpec:
    declaration = _require_string_key_mapping(
        value,
        f"quantization declaration for {path}",
    )
    bits: object = declaration.get("bits")
    group_size: object = declaration.get("group_size")
    mode: object = declaration.get("mode")
    if (
        not _is_positive_int(bits)
        or not _is_positive_int(group_size)
        or not isinstance(mode, str)
        or not mode
    ):
        raise DeepseekV40731CompatibilityError(
            f"quantization declaration for {path} must contain positive bits/group_size and mode"
        )
    if not isinstance(bits, int) or not isinstance(group_size, int):
        raise AssertionError("positive integer validation must narrow to int")
    return QuantizationSpec(bits=bits, group_size=group_size, mode=mode)


def _spec_as_dict(spec: QuantizationSpec) -> dict[str, Any]:
    return {"bits": spec.bits, "group_size": spec.group_size, "mode": spec.mode}


def _first_source_for_destination(
    quantization: Mapping[str, object], destination_path: str
) -> str:
    for source_path in quantization:
        if (
            not source_path.startswith("mtp.")
            and realized_quantization_path(source_path) == destination_path
        ):
            return source_path
    return destination_path


def _has_0731_weight_evidence(weight_keys: frozenset[str]) -> bool:
    return any(
        ".attn_hc." in key
        or ".ffn_hc." in key
        or re.match(r"^mtp\.\d+\.markov_head\.", key) is not None
        for key in weight_keys
    )


def _malformed_dspark_fields(
    config: Mapping[str, Any], valid_block_size: bool, valid_targets: bool
) -> list[str]:
    fields: list[str] = []
    if "dspark_block_size" not in config or not valid_block_size:
        fields.append("dspark_block_size")
    if "dspark_target_layer_ids" not in config or not valid_targets:
        fields.append("dspark_target_layer_ids")
    return fields


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonempty_integer_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    layer_ids = cast(list[object], value)
    return bool(layer_ids) and all(
        isinstance(layer_id, int) and not isinstance(layer_id, bool)
        for layer_id in layer_ids
    )


def _required_positive_int(config: Mapping[str, Any], field: str) -> int:
    value: object = config.get(field)
    if not _is_positive_int(value):
        raise DeepseekV40731CompatibilityError(f"{field} must be a positive integer")
    if not isinstance(value, int):
        raise AssertionError("positive integer validation must narrow to int")
    return value


def _require_string_key_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeepseekV40731CompatibilityError(f"{description} must contain an object")
    mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise DeepseekV40731CompatibilityError(
                f"{description} keys must be strings"
            )
        result[key] = item
    return result
