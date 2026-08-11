"""Pure safety and reporting tests for the real-layer parity command."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest


class _SafeOpen(Protocol):
    def __enter__(self) -> "_SafeOpen": ...

    def __exit__(self, *args: object) -> None: ...

    def get_tensor(self, key: str) -> "_Tensor": ...


SafeOpenFactory = Callable[[Path, str], _SafeOpen]


class _ParityModule(Protocol):
    __file__: str
    OMLX_IMPORTED_PACKAGE_TREE: str

    def select_layer_weight_map(
        self, index: Mapping[str, object], source_layer: int
    ) -> dict[str, str]: ...

    def remap_layer_weights(
        self, weights: Mapping[str, object], source_layer: int
    ) -> dict[str, object]: ...

    def read_selected_tensors(
        self,
        model_path: Path,
        selected_weight_map: Mapping[str, str],
        *,
        safe_open_factory: SafeOpenFactory,
    ) -> dict[str, "_Tensor"]: ...

    def validate_requested_layers(self, layers: Sequence[int]) -> list[int]: ...

    def tensor_metrics(
        self, expected: object, actual: object, *, tolerance_kind: str
    ) -> dict[str, object]: ...

    def _git_root(self, path: Path) -> Path: ...

    def quantization_request(
        self,
        path: str,
        module: object,
        quantization: Mapping[str, object],
        prefix: str,
    ) -> bool | object: ...

    def remapped_layer_uses_hash_routing(
        self, config: Mapping[str, object]
    ) -> bool: ...

    def build_omlx_prefill_mask(
        self,
        batch: int,
        sequence: int,
        window: int,
        build_mask: Callable[[int, int, int, int, int], object],
    ) -> object: ...

    def new_omlx_prefill_cache(
        self, container: object, *, compress_ratio: int
    ) -> object: ...

    def omlx_attention_cache_facts(
        self, attention: object, cache: object
    ) -> dict[str, int | bool]: ...

    def topology_matches_expected(
        self,
        source_layer: int,
        compress_ratio: int,
        cache_facts: Mapping[str, int | bool],
    ) -> bool: ...

    def prepare_omlx_import(self, omlx_root: Path) -> None: ...

    def omlx_package_guard_paths(self, root: Path) -> tuple[str, ...]: ...

    def actual_moe_output(
        self, block: object, hidden: object, input_ids: object
    ) -> object: ...


def _parity_module() -> _ParityModule:
    script = (
        Path(__file__).resolve().parents[6]
        / "scripts"
        / "validate_dsv4_0731_layer_parity.py"
    )
    spec = importlib.util.spec_from_file_location("dsv4_0731_layer_parity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_ParityModule, cast(object, module))


def _approx(value: float) -> object:
    # pytest's shipped stub leaves approx's return type unknown.
    return pytest.approx(value)  # pyright: ignore[reportUnknownMemberType]


def test_select_layer_weight_map_reads_only_the_requested_backbone_layer() -> None:
    parity = _parity_module()
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "model.layers.2.attn.wq_a.weight": "model-00002.safetensors",
            "model.layers.2.ffn.gate.weight": "model-00003.safetensors",
            "model.layers.3.attn.wq_a.weight": "model-00004.safetensors",
        }
    }

    assert parity.select_layer_weight_map(index, 2) == {
        "model.layers.2.attn.wq_a.weight": "model-00002.safetensors",
        "model.layers.2.ffn.gate.weight": "model-00003.safetensors",
    }


def test_remap_layer_weights_changes_only_the_selected_layer_prefix() -> None:
    parity = _parity_module()
    weights = {
        "model.embed_tokens.weight": object(),
        "model.layers.3.attn.wq_a.weight": object(),
    }

    remapped = parity.remap_layer_weights(weights, 3)

    assert remapped["model.embed_tokens.weight"] is weights["model.embed_tokens.weight"]
    assert (
        remapped["model.layers.0.attn.wq_a.weight"]
        is weights["model.layers.3.attn.wq_a.weight"]
    )
    assert "model.layers.3.attn.wq_a.weight" not in remapped


def test_read_selected_tensors_preserves_every_checkpoint_dtype() -> None:
    parity = _parity_module()
    tensors = {
        "u32": _Tensor("uint32"),
        "u8": _Tensor("uint8"),
        "bf16": _Tensor("bfloat16"),
        "f32": _Tensor("float32"),
        "i32": _Tensor("int32"),
    }
    opened: list[Path] = []

    class _SafeOpen:
        def __init__(self, path: Path, framework: str) -> None:
            assert framework == "pt"
            opened.append(path)

        def __enter__(self) -> _SafeOpen:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get_tensor(self, key: str) -> _Tensor:
            return tensors[key]

    loaded = parity.read_selected_tensors(
        Path("/checkpoint"),
        {key: "layer.safetensors" for key in tensors},
        safe_open_factory=_SafeOpen,
    )

    assert loaded == tensors
    assert opened == [Path("/checkpoint/layer.safetensors")]
    assert [tensor.dtype for tensor in loaded.values()] == [
        "uint32",
        "uint8",
        "bfloat16",
        "float32",
        "int32",
    ]


@pytest.mark.parametrize("layers", ([0], [0, 2], [0, 2, 4], [0, 2, 2]))
def test_validate_requested_layers_requires_exact_representative_set(
    layers: list[int],
) -> None:
    parity = _parity_module()

    with pytest.raises(ValueError, match="exactly representative layers"):
        parity.validate_requested_layers(layers)


def test_validate_requested_layers_normalizes_representative_order() -> None:
    parity = _parity_module()

    assert parity.validate_requested_layers([3, 0, 2]) == [0, 2, 3]


def test_tensor_metrics_reports_task_four_tolerances_and_largest_index() -> None:
    parity = _parity_module()

    metrics = parity.tensor_metrics(
        [1.0, 2.0, 3.0], [1.0, 2.02, 2.8], tolerance_kind="bf16"
    )

    assert metrics["max_abs_error"] == _approx(0.2)
    assert metrics["mean_abs_error"] == _approx((0.0 + 0.02 + 0.2) / 3)
    assert metrics["max_relative_error"] == _approx(0.2 / 3)
    assert metrics["largest_error_index"] == [2]
    assert metrics["largest_expected"] == _approx(3.0)
    assert metrics["largest_actual"] == _approx(2.8)
    assert metrics["largest_tolerance_limit"] == _approx(0.08)
    assert metrics["allclose_violation_count"] == 1
    assert metrics["allclose_violation_fraction"] == _approx(1 / 3)
    assert cast(float, metrics["p99_abs_error"]) <= cast(
        float, metrics["p999_abs_error"]
    )
    assert cast(float, metrics["p999_abs_error"]) <= cast(
        float, metrics["p9999_abs_error"]
    )
    assert metrics["rtol"] == _approx(2e-2)
    assert metrics["atol"] == _approx(2e-2)
    assert metrics["passed"] is False
    json.dumps(metrics)


def test_git_root_accepts_a_script_file_path() -> None:
    parity = _parity_module()

    assert parity._git_root(Path(parity.__file__)) == Path.cwd()  # pyright: ignore[reportPrivateUsage]  # The command's private helper is intentionally unit-tested.


def test_quantization_request_leaves_prequantized_modules_unchanged() -> None:
    parity = _parity_module()
    spec = {"bits": 4}
    quantization = {"model.layers.0.ffn.switch_mlp.gate_proj": spec}

    assert (
        parity.quantization_request(
            "ffn.switch_mlp.gate_proj",
            QuantizedSwitchLinear(),
            quantization,
            "model.layers.0.",
        )
        is False
    )
    assert (
        parity.quantization_request(
            "ffn.switch_mlp.gate_proj",
            _Linear(),
            quantization,
            "model.layers.0.",
        )
        == spec
    )


def test_reduced_layer_reports_hash_routing_from_its_remapped_gate() -> None:
    parity = _parity_module()

    assert parity.remapped_layer_uses_hash_routing({"num_hash_layers": 1}) is True
    assert parity.remapped_layer_uses_hash_routing({"num_hash_layers": 0}) is False


def test_multi_token_omlx_attention_uses_exo_window_mask() -> None:
    parity = _parity_module()
    calls: list[tuple[int, int, int, int, int]] = []
    expected_mask = object()

    def build_mask(
        query_length: int,
        key_length: int,
        cache_offset: int,
        window_size: int,
        current_length: int,
    ) -> object:
        calls.append(
            (query_length, key_length, cache_offset, window_size, current_length)
        )
        return expected_mask

    assert parity.build_omlx_prefill_mask(2, 7, 128, build_mask) is expected_mask
    assert calls == [(2, 7, 0, 128, 7)]


def test_compressed_prefill_uses_the_omlx_pool_cache_contract() -> None:
    parity = _parity_module()
    local_cache = object()
    compressor_pool = object()
    indexer_pool = object()

    class _CacheList:
        def __init__(self) -> None:
            self.caches = (local_cache, compressor_pool, indexer_pool)

    class _ReferenceModel:
        def make_cache(self) -> list[_CacheList]:
            return [_CacheList()]

    cache = cast(
        _CacheList, parity.new_omlx_prefill_cache(_ReferenceModel(), compress_ratio=4)
    )

    assert cache.caches[0] is local_cache
    assert cache.caches[1] is compressor_pool
    assert cache.caches[2] is indexer_pool


def test_omlx_attention_cache_facts_distinguish_indexed_and_plain_compression() -> None:
    parity = _parity_module()

    class _CacheList:
        def __init__(self, branches: int) -> None:
            self.caches = tuple(object() for _ in range(branches))

    class _IndexedAttention:
        indexer = object()

    class _PlainAttention:
        pass

    assert parity.omlx_attention_cache_facts(_IndexedAttention(), _CacheList(3)) == {
        "indexer_present": True,
        "omlx_cache_branch_count": 3,
    }
    assert parity.omlx_attention_cache_facts(_PlainAttention(), _CacheList(2)) == {
        "indexer_present": False,
        "omlx_cache_branch_count": 2,
    }


@pytest.mark.parametrize(
    ("source_layer", "ratio", "indexer_present", "branch_count"),
    [
        (0, 0, False, 1),
        (2, 4, True, 3),
        (3, 128, False, 2),
    ],
)
def test_representative_topology_matches_expected(
    source_layer: int, ratio: int, indexer_present: bool, branch_count: int
) -> None:
    parity = _parity_module()

    assert parity.topology_matches_expected(
        source_layer,
        ratio,
        {"indexer_present": indexer_present, "omlx_cache_branch_count": branch_count},
    )


def test_representative_topology_rejects_incorrect_cache_facts() -> None:
    parity = _parity_module()

    assert not parity.topology_matches_expected(
        2, 4, {"indexer_present": False, "omlx_cache_branch_count": 3}
    )


def test_omlx_import_preparation_disables_bytecode_before_path_insertion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    parity = _parity_module()
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    parity.prepare_omlx_import(tmp_path)

    assert sys.dont_write_bytecode is True
    assert sys.path[0] == str(tmp_path)


def test_omlx_package_guard_covers_patch_and_custom_kernel_files(
    tmp_path: Path,
) -> None:
    parity = _parity_module()
    root = tmp_path / "omlx"
    patch_tree = root / parity.OMLX_IMPORTED_PACKAGE_TREE / "patches" / "deepseek_v4"
    patch_tree.mkdir(parents=True)
    (patch_tree / "deepseek_v4_model.py").touch()
    custom_kernels = root / parity.OMLX_IMPORTED_PACKAGE_TREE / "custom_kernels"
    custom_kernels.mkdir()
    (custom_kernels / "glm_moe_dsa.py").touch()
    (custom_kernels / "nax.py").touch()

    assert parity.omlx_package_guard_paths(root) == (
        "omlx/custom_kernels/glm_moe_dsa.py",
        "omlx/custom_kernels/nax.py",
        "omlx/patches/deepseek_v4/deepseek_v4_model.py",
    )


def test_actual_moe_output_uses_the_block_ffn_call() -> None:
    parity = _parity_module()
    calls: list[tuple[object, object]] = []

    class _FFN:
        def __call__(self, hidden: object, input_ids: object) -> object:
            calls.append((hidden, input_ids))
            return "production-moe-output"

    class _Block:
        ffn = _FFN()

    assert (
        parity.actual_moe_output(_Block(), "hidden", "input-ids")
        == "production-moe-output"
    )
    assert calls == [("hidden", "input-ids")]


class _Tensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class QuantizedSwitchLinear:
    pass


class _Linear:
    pass
