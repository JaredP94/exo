import json
from pathlib import Path
from typing import Any, cast

import pytest

from exo.worker.engines.mlx.deepseek_v4_0731_config import (
    DeepseekV40731CompatibilityError,
    QuantizationSpec,
    inspect_deepseek_v4_0731_checkpoint,
    normalize_deepseek_v4_0731_config,
    normalized_quantization_specs,
    realized_quantization_path,
    validate_deepseek_v4_0731_shard_geometry,
)


def _config(**overrides: object) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "compress_ratios": [0],
        "num_attention_heads": 64,
        "o_groups": 8,
        "head_dim": 512,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "quantization": {},
    }
    config.update(overrides)
    return config


def _spec(bits: int = 8, group_size: int = 32, mode: str = "mxfp8") -> dict[str, Any]:
    return {"bits": bits, "group_size": group_size, "mode": mode}


def _write_checkpoint(
    model_path: Path,
    config: dict[str, Any],
    weight_keys: list[str],
) -> None:
    (model_path / "config.json").write_text(json.dumps(config))
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model-00001-of-00001.safetensors" for key in weight_keys}})
    )


def test_non_deepseek_v4_config_delegates_without_reading_index(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    assert inspect_deepseek_v4_0731_checkpoint(tmp_path) is None


def test_ordinary_single_file_deepseek_v4_delegates_without_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()))

    assert inspect_deepseek_v4_0731_checkpoint(tmp_path) is None


def test_ordinary_deepseek_v4_delegates(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, _config(), ["model.layers.0.attn.wo_a.weight"])

    assert inspect_deepseek_v4_0731_checkpoint(tmp_path) is None


def test_0731_weight_evidence_without_dspark_fields_is_rejected(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path, _config(), ["model.layers.0.attn_hc.fn"])

    with pytest.raises(DeepseekV40731CompatibilityError) as error:
        inspect_deepseek_v4_0731_checkpoint(tmp_path)

    assert "dspark_block_size" in str(error.value)
    assert "dspark_target_layer_ids" in str(error.value)


def test_num_nextn_predict_layers_is_not_a_0731_discriminator(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        _config(num_nextn_predict_layers=1),
        ["model.layers.0.attn.wo_a.weight"],
    )

    assert inspect_deepseek_v4_0731_checkpoint(tmp_path) is None


def test_supported_dspark_checkpoint_records_index_metadata(tmp_path: Path) -> None:
    config = _config(
        dspark_block_size=5,
        dspark_target_layer_ids=[0],
        quantization={
            "model.layers.0.attn.wq_a": _spec(),
            "mtp.0.markov_head.markov_w1": _spec(),
        },
    )
    weight_keys = [
        "model.layers.0.attn_hc.fn",
        "model.layers.0.attn.wo_a.weight",
        "mtp.0.markov_head.markov_w1.weight",
    ]
    _write_checkpoint(tmp_path, config, weight_keys)

    checkpoint = inspect_deepseek_v4_0731_checkpoint(tmp_path)

    assert checkpoint is not None
    assert checkpoint.model_path == tmp_path
    assert checkpoint.weight_keys == frozenset(weight_keys)
    assert checkpoint.mtp_weight_count == 1
    assert checkpoint.mtp_quantization_count == 1
    assert checkpoint.realized_quantization == {
        "model.layers.0.attn.wqkv_a": QuantizationSpec(8, 32, "mxfp8")
    }


@pytest.mark.parametrize(
    ("config_overrides", "weight_keys", "offending_field"),
    [
        ({"dspark_block_size": 5}, [], "dspark_target_layer_ids"),
        ({"dspark_target_layer_ids": [0]}, [], "dspark_block_size"),
        (
            {"dspark_block_size": 0, "dspark_target_layer_ids": [0]},
            ["model.layers.0.attn_hc.fn"],
            "dspark_block_size",
        ),
    ],
)
def test_partial_or_malformed_dspark_evidence_is_rejected(
    tmp_path: Path,
    config_overrides: dict[str, object],
    weight_keys: list[str],
    offending_field: str,
) -> None:
    _write_checkpoint(tmp_path, _config(**config_overrides), weight_keys)

    with pytest.raises(DeepseekV40731CompatibilityError) as error:
        inspect_deepseek_v4_0731_checkpoint(tmp_path)

    assert str(tmp_path) in str(error.value)
    assert offending_field in str(error.value)


@pytest.mark.parametrize(
    ("source", "realized"),
    [
        ("model.layers.3.attn.wq_a", "model.layers.3.attn.wqkv_a"),
        ("model.layers.3.attn.wkv", "model.layers.3.attn.wqkv_a"),
        (
            "model.layers.3.attn.compressor.wgate",
            "model.layers.3.attn.compressor.wkv_gate",
        ),
        (
            "model.layers.3.attn.indexer.compressor.wkv",
            "model.layers.3.attn.indexer.compressor.wkv_gate",
        ),
        ("model.layers.3.attn.unrelated_wkv", "model.layers.3.attn.unrelated_wkv"),
    ],
)
def test_realized_quantization_path_rewrites_only_expected_paths(source: str, realized: str) -> None:
    assert realized_quantization_path(source) == realized


def test_normalization_collapses_compatible_pairs_and_preserves_global_fields() -> None:
    original = _config(
        quantization={
            "model.layers.0.attn.wq_a": _spec(),
            "model.layers.0.attn.wkv": _spec(),
            "model.layers.0.ffn.switch_mlp.gate_proj": _spec(4, 64, "mxfp4"),
        }
    )

    normalized, realized = normalize_deepseek_v4_0731_config(original)

    assert original["quantization"] == {
        "model.layers.0.attn.wq_a": _spec(),
        "model.layers.0.attn.wkv": _spec(),
        "model.layers.0.ffn.switch_mlp.gate_proj": _spec(4, 64, "mxfp4"),
    }
    assert normalized["bits"] == 4
    assert normalized["group_size"] == 64
    assert normalized["mode"] == "affine"
    assert normalized["quantization"] == {
        "model.layers.0.attn.wqkv_a": _spec(),
        "model.layers.0.ffn.switch_mlp.gate_proj": _spec(4, 64, "mxfp4"),
    }
    assert realized == {
        "model.layers.0.attn.wqkv_a": QuantizationSpec(8, 32, "mxfp8"),
        "model.layers.0.ffn.switch_mlp.gate_proj": QuantizationSpec(4, 64, "mxfp4"),
    }


@pytest.mark.parametrize("field", ["bits", "group_size", "mode"])
def test_normalization_rejects_incompatible_fusion_pairs(field: str) -> None:
    mismatched = _spec()
    mismatched[field] = 4 if field != "mode" else "affine"
    config = _config(
        quantization={
            "model.layers.0.attn.wq_a": _spec(),
            "model.layers.0.attn.wkv": mismatched,
        }
    )

    with pytest.raises(DeepseekV40731CompatibilityError) as error:
        normalize_deepseek_v4_0731_config(config)

    message = str(error.value)
    assert "model.layers.0.attn.wq_a" in message
    assert "model.layers.0.attn.wkv" in message


def test_normalization_omits_mtp_quantization_and_updates_mirrored_config() -> None:
    config = _config(
        quantization={
            "model.layers.0.attn.wq_a": _spec(),
            "mtp.0.markov_head.markov_w1": _spec(),
        },
        quantization_config={"will": "be replaced"},
    )

    normalized, realized = normalize_deepseek_v4_0731_config(config)

    assert "mtp.0.markov_head.markov_w1" not in normalized["quantization"]
    assert "mtp.0.markov_head.markov_w1" not in realized
    assert normalized["quantization_config"] == normalized["quantization"]


def test_normalization_preserves_global_quantization_scalars_without_parsing_them() -> None:
    config = _config(
        quantization={
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "model.layers.0.attn.wq_a": _spec(),
        },
        quantization_config={
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "model.layers.0.attn.wq_a": _spec(),
        },
    )

    normalized, realized = normalize_deepseek_v4_0731_config(config)

    assert normalized["quantization"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "model.layers.0.attn.wqkv_a": _spec(),
    }
    assert normalized["quantization_config"] == normalized["quantization"]
    assert realized == {
        "model.layers.0.attn.wqkv_a": QuantizationSpec(8, 32, "mxfp8")
    }


def test_normalized_quantization_specs_reconstructs_explicit_triples() -> None:
    specs = normalized_quantization_specs(
        _config(
            quantization={
                "model.layers.0.attn.wqkv_a": _spec(),
                "model.layers.0.ffn.switch_mlp.gate_proj": _spec(4, 64, "mxfp4"),
            }
        )
    )

    assert specs == {
        "model.layers.0.attn.wqkv_a": QuantizationSpec(8, 32, "mxfp8"),
        "model.layers.0.ffn.switch_mlp.gate_proj": QuantizationSpec(4, 64, "mxfp4"),
    }


def test_normalization_rejects_an_invalid_global_quantization_triple() -> None:
    config = _config(bits=0)

    with pytest.raises(DeepseekV40731CompatibilityError, match="global quantization"):
        normalize_deepseek_v4_0731_config(config)


def test_normalization_collapses_641_sources_to_536_realized_paths() -> None:
    quantization: dict[str, Any] = {}
    for layer_id in range(105):
        quantization[f"model.layers.{layer_id}.attn.wq_a"] = _spec()
        quantization[f"model.layers.{layer_id}.attn.wkv"] = _spec()
    for declaration_id in range(431):
        quantization[f"model.layers.{declaration_id}.ffn.switch_mlp.gate_proj"] = _spec(4, 64, "mxfp4")
    config = _config(num_hidden_layers=536, compress_ratios=[0] * 536, quantization=quantization)

    normalized, realized = normalize_deepseek_v4_0731_config(config)

    assert len(quantization) == 641
    normalized_quantization = cast(dict[str, object], normalized["quantization"])
    assert len(normalized_quantization) == 536
    assert len(realized) == 536


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_geometry_accepts_world_sizes_that_divide_heads_per_group(world_size: int) -> None:
    validate_deepseek_v4_0731_shard_geometry(_config(), {}, world_size)


@pytest.mark.parametrize("world_size", [3, 16])
def test_geometry_rejects_world_sizes_that_do_not_divide_heads_per_group(world_size: int) -> None:
    with pytest.raises(DeepseekV40731CompatibilityError, match="heads_per_group=8"):
        validate_deepseek_v4_0731_shard_geometry(_config(), {}, world_size)


def test_geometry_requires_heads_to_divide_evenly_into_groups() -> None:
    with pytest.raises(DeepseekV40731CompatibilityError, match="num_attention_heads"):
        validate_deepseek_v4_0731_shard_geometry(_config(num_attention_heads=65), {}, 1)


@pytest.mark.parametrize(
    ("path", "config"),
    [
        ("model.layers.0.attn.wo_a", _config(head_dim=10)),
        (
            "model.layers.0.ffn.switch_mlp.down_proj",
            _config(moe_intermediate_size=100),
        ),
        (
            "model.layers.0.ffn.shared_experts.down_proj",
            _config(moe_intermediate_size=100, n_shared_experts=3),
        ),
    ],
)
def test_geometry_rejects_quantized_sharded_modules_with_incompatible_width(
    path: str, config: dict[str, Any]
) -> None:
    with pytest.raises(DeepseekV40731CompatibilityError, match=path):
        validate_deepseek_v4_0731_shard_geometry(
            config,
            {path: QuantizationSpec(8, 32, "mxfp8")},
            2,
        )


def test_geometry_accepts_real_target_dimensions_for_two_ranks() -> None:
    config = _config()
    specs = {
        "model.layers.0.attn.wo_a": QuantizationSpec(8, 32, "mxfp8"),
        "model.layers.0.ffn.switch_mlp.down_proj": QuantizationSpec(4, 32, "mxfp4"),
        "model.layers.0.ffn.shared_experts.down_proj": QuantizationSpec(4, 32, "mxfp4"),
    }

    validate_deepseek_v4_0731_shard_geometry(config, specs, 2)
