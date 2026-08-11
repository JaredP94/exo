import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.deepseek_v4 import ModelArgs
from mlx_lm.models.switch_layers import SwitchGLU

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.worker.engines.mlx.deepseek_v4_0731_config import (
    DeepseekV40731CompatibilityError,
    QuantizationSpec,
)


class _QuantizableProjection(Protocol):
    def to_quantized(
        self, *, group_size: int, bits: int, mode: str
    ) -> "_QuantizableProjection": ...


class _QuantizableSwitch(Protocol):
    gate_proj: _QuantizableProjection
    up_proj: _QuantizableProjection
    down_proj: _QuantizableProjection


class _ModelClassesResolver(Protocol):
    def __call__(
        self, *, config: dict[str, Any]
    ) -> tuple[type[nn.Module], type[Any]]: ...


def _config(**overrides: object) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "compress_ratios": [0] * 46,
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


def _write_checkpoint(
    model_path: Path, config: dict[str, Any], weight_keys: list[str]
) -> None:
    (model_path / "config.json").write_text(json.dumps(config))
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in weight_keys
                }
            }
        )
    )


def _supported_checkpoint(model_path: Path) -> dict[str, Any]:
    config = _config(
        dspark_block_size=5,
        dspark_target_layer_ids=[40, 41, 42],
        quantization={
            "model.layers.0.attn.wq_a": {
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
            },
            "model.layers.0.ffn.switch_mlp.gate_proj": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
            },
            "model.layers.0.ffn.switch_mlp.up_proj": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
            },
            "model.layers.0.ffn.switch_mlp.down_proj": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
            },
        },
    )
    _write_checkpoint(
        model_path,
        config,
        [
            "model.layers.0.attn_hc.fn",
            "model.layers.0.attn.wo_a.weight",
            "mtp.0.markov_head.markov_w1.weight",
        ],
    )
    return config


def test_ordinary_model_preserves_lazy_strict_and_default_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    _write_checkpoint(tmp_path, {"model_type": "llama"}, [])

    class CompatibleModel(nn.Module):
        pass

    model = CompatibleModel()
    returned_config = {"model_type": "llama"}
    captured: dict[str, object] = {}

    def fake_load_model(
        *args: object, **kwargs: object
    ) -> tuple[object, dict[str, Any]]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return model, returned_config

    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)

    loaded_model, config = loader.load_exo_model(tmp_path, lazy=True, strict=False)

    assert loaded_model is model
    assert config is returned_config
    assert captured["args"] == (tmp_path,)
    assert captured["kwargs"] == {"lazy": True, "strict": False}


def test_ordinary_single_file_deepseek_v4_uses_default_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    model = nn.Module()
    returned_config = _config()
    captured: dict[str, object] = {}

    def fake_load_model(
        *args: object, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return model, returned_config

    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)

    loaded_model, config = loader.load_exo_model(tmp_path, lazy=True, strict=False)

    assert loaded_model is model
    assert config is returned_config
    assert captured["args"] == (tmp_path,)
    assert captured["kwargs"] == {"lazy": True, "strict": False}


def test_0731_weight_evidence_without_dspark_fields_fails_before_mlx_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    _write_checkpoint(tmp_path, _config(), ["model.layers.0.attn_hc.fn"])

    def fail_if_called(*_: object, **__: object) -> tuple[nn.Module, dict[str, Any]]:
        pytest.fail("MLX loader must not run for an incomplete 0731 checkpoint")

    monkeypatch.setattr(loader, "_mlx_load_model", fail_if_called)

    with pytest.raises(loader.DeepseekV40731CompatibilityError) as error:
        loader.load_exo_model(tmp_path)

    assert "dspark_block_size" in str(error.value)
    assert "dspark_target_layer_ids" in str(error.value)


def test_0731_checkpoint_forces_strict_and_supplies_compatibility_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    source_config = _supported_checkpoint(tmp_path)

    class CompatibleModel(nn.Module):
        pass

    model = CompatibleModel()
    captured: dict[str, object] = {}

    def fake_load_model(
        *args: object, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        normalized = cast(dict[str, Any], kwargs["model_config"])
        captured["model_config_at_load"] = copy.deepcopy(normalized)
        resolver = cast(_ModelClassesResolver, kwargs["get_model_classes"])
        assert resolver(config=normalized) == (loader.DeepseekV40731Model, ModelArgs)
        return model, normalized

    def skip_prefill(_: nn.Module) -> None:
        pass

    def skip_audit(_: nn.Module, __: dict[str, QuantizationSpec]) -> None:
        pass

    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)
    monkeypatch.setattr(
        loader, "install_deepseek_v4_0731_prefill_attention", skip_prefill
    )
    monkeypatch.setattr(loader, "audit_deepseek_v4_0731_quantization", skip_audit)
    monkeypatch.setattr(loader, "DeepseekV40731Model", CompatibleModel)

    loaded_model, normalized = loader.load_exo_model(tmp_path, lazy=True, strict=False)

    assert loaded_model is model
    assert captured["args"] == (tmp_path,)
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["lazy"] is True
    assert kwargs["strict"] is True
    assert kwargs["model_config"] is normalized
    assert normalized is not source_config
    model_config_at_load = cast(dict[str, Any], captured["model_config_at_load"])
    switch_projections = (
        "model.layers.0.ffn.switch_mlp.gate_proj",
        "model.layers.0.ffn.switch_mlp.up_proj",
        "model.layers.0.ffn.switch_mlp.down_proj",
    )
    assert all(
        path not in model_config_at_load["quantization"] for path in switch_projections
    )
    assert all(path in normalized["quantization"] for path in switch_projections)


def test_malformed_0731_evidence_fails_before_mlx_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    _write_checkpoint(
        tmp_path,
        _config(dspark_block_size=5),
        ["model.layers.0.attn_hc.fn"],
    )
    called = False

    def fake_load_model(*_: object, **__: object) -> tuple[object, dict[str, Any]]:
        nonlocal called
        called = True
        return SimpleNamespace(), {}

    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)

    with pytest.raises(
        DeepseekV40731CompatibilityError, match="dspark_target_layer_ids"
    ):
        loader.load_exo_model(tmp_path)

    assert not called


def test_0731_loader_reports_backbone_only_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    _supported_checkpoint(tmp_path)

    class CompatibleModel(nn.Module):
        pass

    model = CompatibleModel()

    def fake_load_model(*_: object, **__: object) -> tuple[nn.Module, dict[str, Any]]:
        return model, {}

    def skip_prefill(_: nn.Module) -> None:
        pass

    def skip_audit(_: nn.Module, __: dict[str, QuantizationSpec]) -> None:
        pass

    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)
    monkeypatch.setattr(
        loader, "install_deepseek_v4_0731_prefill_attention", skip_prefill
    )
    monkeypatch.setattr(loader, "audit_deepseek_v4_0731_quantization", skip_audit)
    monkeypatch.setattr(loader, "DeepseekV40731Model", CompatibleModel)
    messages: list[str] = []
    monkeypatch.setattr(loader, "logger", SimpleNamespace(warning=messages.append))

    loader.load_exo_model(tmp_path)

    message = "\n".join(messages)
    assert "deepseek_v4_0731_backbone" in message
    assert "mtp_weight_count=1" in message
    assert "dspark_block_size=5" in message
    assert "target_layers=[40, 41, 42]" in message
    assert "DSpark/MTP speculative decoding is disabled" in message


def test_0731_loader_installs_prefill_split_before_quantization_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exo.worker.engines.mlx.deepseek_v4_0731_loader as loader

    _supported_checkpoint(tmp_path)
    events: list[str] = []

    class CompatibleModel(nn.Module):
        pass

    model = CompatibleModel()

    def fake_load_model(*_: object, **__: object) -> tuple[nn.Module, dict[str, Any]]:
        return model, {}

    def record_prefill(installed: nn.Module) -> None:
        events.append(f"install:{installed is model}")

    def record_audit(audited: nn.Module, _: dict[str, QuantizationSpec]) -> None:
        events.append(f"audit:{audited is model}")

    monkeypatch.setattr(loader, "DeepseekV40731Model", CompatibleModel)
    monkeypatch.setattr(loader, "_mlx_load_model", fake_load_model)
    monkeypatch.setattr(
        loader,
        "install_deepseek_v4_0731_prefill_attention",
        record_prefill,
    )
    monkeypatch.setattr(
        loader,
        "audit_deepseek_v4_0731_quantization",
        record_audit,
    )

    loader.load_exo_model(tmp_path)

    assert events == ["install:True", "audit:True"]


class _QuantizedAuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.QuantizedLinear(32, 32, group_size=32, bits=4, mode="affine")
        self.embedding = nn.QuantizedEmbedding(
            32, 32, group_size=32, bits=4, mode="affine"
        )
        self.switch = SwitchGLU(32, 32, 1)
        switch = cast(_QuantizableSwitch, cast(object, self.switch))
        switch.gate_proj = switch.gate_proj.to_quantized(
            group_size=32, bits=4, mode="affine"
        )
        switch.up_proj = switch.up_proj.to_quantized(
            group_size=32, bits=4, mode="affine"
        )
        switch.down_proj = switch.down_proj.to_quantized(
            group_size=32, bits=4, mode="affine"
        )


def _audit_specs(*, mode: str = "affine") -> dict[str, QuantizationSpec]:
    spec = QuantizationSpec(bits=4, group_size=32, mode=mode)
    return {
        "linear": spec,
        "embedding": spec,
        "switch.gate_proj": spec,
        "switch.up_proj": spec,
        "switch.down_proj": spec,
    }


def test_quantization_audit_accepts_all_concrete_quantized_module_families() -> None:
    from exo.worker.engines.mlx.deepseek_v4_0731_loader import (
        audit_deepseek_v4_0731_quantization,
    )

    audit_deepseek_v4_0731_quantization(_QuantizedAuditModel(), _audit_specs())


def test_quantization_audit_rejects_mode_mismatch_with_triples() -> None:
    from exo.worker.engines.mlx.deepseek_v4_0731_loader import (
        audit_deepseek_v4_0731_quantization,
    )

    model = _QuantizedAuditModel()
    model.linear.mode = "mxfp8"

    with pytest.raises(DeepseekV40731CompatibilityError) as error:
        audit_deepseek_v4_0731_quantization(model, _audit_specs())

    message = str(error.value)
    assert "linear" in message
    assert "(bits=4, group_size=32, mode='affine')" in message
    assert "(bits=4, group_size=32, mode='mxfp8')" in message


@pytest.mark.parametrize(
    ("expected", "path"),
    [
        (_audit_specs() | {"missing": QuantizationSpec(4, 32, "affine")}, "missing"),
        ({"linear": QuantizationSpec(4, 32, "affine")}, "embedding"),
    ],
)
def test_quantization_audit_rejects_missing_or_unexpected_modules(
    expected: dict[str, QuantizationSpec], path: str
) -> None:
    from exo.worker.engines.mlx.deepseek_v4_0731_loader import (
        audit_deepseek_v4_0731_quantization,
    )

    with pytest.raises(DeepseekV40731CompatibilityError, match=path):
        audit_deepseek_v4_0731_quantization(_QuantizedAuditModel(), expected)


def test_quantization_audit_ignores_dangling_mtp_declaration() -> None:
    from exo.worker.engines.mlx.deepseek_v4_0731_loader import (
        audit_deepseek_v4_0731_quantization,
    )

    expected = _audit_specs() | {"mtp.0.markov_head": QuantizationSpec(4, 32, "affine")}

    audit_deepseek_v4_0731_quantization(_QuantizedAuditModel(), expected)


def _tensor_metadata() -> TensorShardMetadata:
    return TensorShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test/model"),
            storage_size=Memory.from_gb(1),
            n_layers=1,
            hidden_size=32,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=3,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def test_single_node_load_routes_through_compatibility_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exo.worker.engines.mlx import utils_mlx

    metadata = _tensor_metadata()
    calls: list[tuple[Path, bool, bool]] = []

    def ignore_wired_limit(_: object) -> None:
        pass

    def model_path(_: object) -> Path:
        return tmp_path

    def fake_load_model(
        path: Path, *, lazy: bool, strict: bool
    ) -> tuple[nn.Module, dict[str, Any]]:
        calls.append((path, lazy, strict))
        return nn.Module(), {}

    def tokenizer(*_: object) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(utils_mlx, "set_wired_limit_for_model", ignore_wired_limit)
    monkeypatch.setattr(utils_mlx, "build_model_path", model_path)
    monkeypatch.setattr(
        utils_mlx,
        "load_exo_model",
        fake_load_model,
    )
    monkeypatch.setattr(utils_mlx, "get_tokenizer", tokenizer)
    bound_instance = cast(
        BoundInstance,
        cast(
            object,
            SimpleNamespace(
                bound_shard=metadata,
                instance=SimpleNamespace(bound_shard=metadata),
            ),
        ),
    )

    generator = utils_mlx.load_mlx_items(bound_instance, group=None)
    with pytest.raises(StopIteration):
        next(generator)

    assert calls == [(tmp_path, True, False)]


def test_distributed_geometry_failure_precedes_tensor_sharding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exo.worker.engines.mlx import utils_mlx

    metadata = _tensor_metadata()
    calls: list[tuple[Path, bool, bool]] = []
    tensor_sharding_started = False
    group = cast(
        mx.distributed.Group,
        cast(object, SimpleNamespace(size=lambda: 3, rank=lambda: 0)),
    )

    def model_path(_: object) -> Path:
        return tmp_path

    def fake_load_model(
        path: Path, *, lazy: bool, strict: bool
    ) -> tuple[nn.Module, dict[str, Any]]:
        calls.append((path, lazy, strict))
        return nn.Module(), {
            "model_type": "deepseek_v4",
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [40],
        }

    def fake_checkpoint(_: Path) -> object:
        return object()

    def empty_quantization_specs(_: dict[str, Any]) -> dict[str, QuantizationSpec]:
        return {}

    monkeypatch.setattr(utils_mlx, "build_model_path", model_path)
    monkeypatch.setattr(
        utils_mlx,
        "load_exo_model",
        fake_load_model,
    )
    monkeypatch.setattr(
        utils_mlx, "inspect_deepseek_v4_0731_checkpoint", fake_checkpoint
    )
    monkeypatch.setattr(
        utils_mlx, "normalized_quantization_specs", empty_quantization_specs
    )

    def geometry_failure(*_: object) -> None:
        raise DeepseekV40731CompatibilityError("heads_per_group=8")

    def tensor_sharding(*_: object) -> object:
        nonlocal tensor_sharding_started
        tensor_sharding_started = True
        return iter(())

    monkeypatch.setattr(
        utils_mlx, "validate_deepseek_v4_0731_shard_geometry", geometry_failure
    )
    monkeypatch.setattr(utils_mlx, "tensor_auto_parallel", tensor_sharding)

    generator = utils_mlx.shard_and_load(metadata, group)
    with pytest.raises(DeepseekV40731CompatibilityError, match="heads_per_group=8"):
        next(generator)

    assert calls == [(tmp_path, True, False)]
    assert not tensor_sharding_started
