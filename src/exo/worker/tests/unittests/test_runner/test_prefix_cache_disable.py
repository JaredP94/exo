"""The explicit prefix-cache flag must control reuse outside benchmarks."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import mlx.core as mx

from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.types import Model


class _Model:
    layers: list[object] = []


def _cache() -> KVPrefixCache:
    cache = KVPrefixCache(None)
    cache.add_kv_cache(mx.array([1, 2, 3, 4, 5]), [])
    return cache


def test_disabled_flag_yields_no_reuse() -> None:
    model = cast(Model, cast(object, _Model()))
    with patch("exo.worker.engines.mlx.cache.make_kv_cache", return_value=[]):
        cache, remaining, matched_index, exact = _cache().get_kv_cache(
            model, mx.array([1, 2, 3, 4, 5]), use_prefix_cache=False
        )

    assert cache == []
    assert mx.array_equal(remaining, mx.array([1, 2, 3, 4, 5]))
    assert matched_index is None
    assert not exact


def test_enabled_flag_still_reuses() -> None:
    model = cast(Model, cast(object, _Model()))
    with patch("exo.worker.engines.mlx.cache.make_kv_cache", return_value=[]):
        _, _, matched_index, exact = _cache().get_kv_cache(
            model, mx.array([1, 2, 3, 4, 5]), use_prefix_cache=True
        )

    assert matched_index == 0
    assert exact
