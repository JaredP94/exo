"""Regression coverage for the cross-device SDPA diagnostic output."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, cast


def _load_script() -> ModuleType:
    script_path = Path(__file__).parents[6] / "scripts" / "check_cross_device_sdpa.py"
    spec = importlib.util.spec_from_file_location(
        "check_cross_device_sdpa", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bfloat16_rank_summary_includes_bounds_and_finiteness() -> None:
    script = _load_script()
    formatter = cast(
        Callable[[int, float, float, bool], str],
        vars(script)["_format_bfloat16_rank_summary"],
    )

    assert formatter(1, -0.75, 0.875, True) == (
        "rank 1 bfloat16 output: min=-0.75 max=0.875 all_finite=True"
    )
