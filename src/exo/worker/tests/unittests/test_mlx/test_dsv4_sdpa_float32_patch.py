"""The float32 SDPA patch must take effect, not silently miss.

`install_deepseek_v4_sdpa_float32` rebinds a module global by name. While it was
being written it named `scaled_dot_project_attention` — "project", not "product" —
an attribute that does not exist, so the patch created a new unused attribute and
the model kept running bfloat16 attention. That is a whole class of failure:
a monkeypatch whose target name is wrong does nothing and says nothing.

These tests assert the EFFECTIVE state rather than the spelling, so they also
catch the patch being dropped from the loader, the upstream symbol being renamed,
and the cast being weakened.

Break each test catches:

* patch_replaces_an_existing_attribute - the original typo, and any future one.
  A patch must rebind a name that already exists; inventing an attribute is
  always a silent no-op.
* patch_introduces_no_new_public_names - the same defect from the other side: if
  a misspelled target were used, `dir()` would gain an entry.
* patched_function_casts_inputs_to_float32 - the cast being removed or narrowed
  to only some of q/k/v, which would leave part of attention in bfloat16.
* patched_function_restores_the_input_dtype - returning float32 to a bfloat16
  graph, which changes every downstream dtype and memory figure.
* patch_is_idempotent - double application wrapping the wrapper, doubling the
  cast cost per layer.
* upstream_symbol_still_exists - an mlx_lm rename. The install would raise
  AttributeError at load time, which is loud, but this fails in CI first and
  says why.
* loader_installs_the_patch - the call being deleted from the loader. Source-level
  and therefore weaker than the others: it proves the call site exists, not that
  it executes. The cross-device script is the end-to-end guard.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from typing import Any

import mlx.core as mx
import mlx_lm.models.deepseek_v4 as dsv4
import pytest

from exo.worker.engines.mlx import deepseek_v4_0731_loader
from exo.worker.engines.mlx.deepseek_v4_0731_model import (
    install_deepseek_v4_sdpa_float32,
)

PATCH_FLAG = "_exo_sdpa_float32_patched"
TARGET = "scaled_dot_product_attention"


@pytest.fixture
def unpatched() -> Generator[Callable[..., Any]]:
    """Hand back a pristine module global and restore it afterwards.

    The loader patches this process-wide, so without the restore a test ordering
    change would silently alter what every later test measures.
    """
    original: Callable[..., Any] = getattr(dsv4, TARGET)  # pyright: ignore[reportAny]
    had_flag = hasattr(dsv4, PATCH_FLAG)
    if had_flag:
        delattr(dsv4, PATCH_FLAG)
    try:
        yield original
    finally:
        setattr(dsv4, TARGET, original)
        if had_flag:
            setattr(dsv4, PATCH_FLAG, True)
        elif hasattr(dsv4, PATCH_FLAG):
            delattr(dsv4, PATCH_FLAG)


def test_upstream_symbol_still_exists() -> None:
    assert hasattr(dsv4, TARGET), (
        f"mlx_lm.models.deepseek_v4.{TARGET} is gone; the patch target was renamed "
        f"upstream and install_deepseek_v4_sdpa_float32 will raise at load time"
    )


def test_patch_replaces_an_existing_attribute(
    unpatched: Callable[..., Any],
) -> None:
    install_deepseek_v4_sdpa_float32()

    assert getattr(dsv4, TARGET) is not unpatched, (
        "the module global was not rebound — the patch is a no-op"
    )


def test_patch_introduces_no_new_public_names(
    unpatched: Callable[..., Any],
) -> None:
    """A misspelled target shows up as an extra attribute."""
    del unpatched
    before = set(dir(dsv4))

    install_deepseek_v4_sdpa_float32()

    added = set(dir(dsv4)) - before
    assert added <= {PATCH_FLAG}, (
        f"the patch created {sorted(added - {PATCH_FLAG})}; a patch that adds a "
        f"name rather than rebinding one has missed its target"
    )


def test_patched_function_casts_inputs_to_float32(
    unpatched: Callable[..., Any],
) -> None:
    del unpatched
    seen: dict[str, mx.Dtype] = {}

    def spy(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        cache: object,
        scale: float,
        mask: object,
        sinks: object = None,
    ) -> mx.array:
        del cache, scale, mask, sinks
        seen["queries"] = queries.dtype
        seen["keys"] = keys.dtype
        seen["values"] = values.dtype
        return queries

    setattr(dsv4, TARGET, spy)
    install_deepseek_v4_sdpa_float32()

    bf16 = mx.zeros((1, 1, 2, 4), dtype=mx.bfloat16)
    _: mx.array = getattr(dsv4, TARGET)(bf16, bf16, bf16, None, 1.0, None)  # pyright: ignore[reportAny]

    assert seen == {
        "queries": mx.float32,
        "keys": mx.float32,
        "values": mx.float32,
    }, f"attention did not run in float32: {seen}"


def test_patched_function_restores_the_input_dtype(
    unpatched: Callable[..., Any],
) -> None:
    del unpatched

    def spy(queries: mx.array, *args: object, **kwargs: object) -> mx.array:
        del args, kwargs
        return queries

    setattr(dsv4, TARGET, spy)
    install_deepseek_v4_sdpa_float32()

    bf16 = mx.zeros((1, 1, 2, 4), dtype=mx.bfloat16)
    out: mx.array = getattr(dsv4, TARGET)(bf16, bf16, bf16, None, 1.0, None)  # pyright: ignore[reportAny]

    assert out.dtype == mx.bfloat16, (
        f"returned {out.dtype}; leaking float32 into a bfloat16 graph changes "
        f"every downstream dtype and the attention memory footprint"
    )


def test_patch_is_idempotent(unpatched: Callable[..., Any]) -> None:
    del unpatched
    install_deepseek_v4_sdpa_float32()
    once: Callable[..., Any] = getattr(dsv4, TARGET)  # pyright: ignore[reportAny]

    install_deepseek_v4_sdpa_float32()

    assert getattr(dsv4, TARGET) is once, (
        "a second install wrapped the wrapper, so every layer pays the cast twice"
    )


def test_loader_installs_the_patch() -> None:
    """Source-level, and weaker than the rest: it proves the call site exists,
    not that it runs. Executing the loader needs the 154 GiB checkpoint, so the
    end-to-end guard is scripts/check_cross_device_sdpa.py on the cluster.
    """
    assert hasattr(deepseek_v4_0731_loader, "install_deepseek_v4_sdpa_float32"), (
        "the loader no longer imports the patch installer"
    )
    source = inspect.getsource(deepseek_v4_0731_loader)
    assert "install_deepseek_v4_sdpa_float32()" in source, (
        "the loader imports the installer but never calls it"
    )
