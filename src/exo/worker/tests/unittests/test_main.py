from __future__ import annotations

import sys

import pytest

from exo.main import Args


def test_namespace_defaults_to_zenoh_namespace_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ZENOH_NAMESPACE", "validation-namespace")
    monkeypatch.setattr(sys, "argv", ["exo"])

    assert Args.parse().namespace == "validation-namespace"


def test_explicit_namespace_overrides_zenoh_namespace_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ZENOH_NAMESPACE", "validation-namespace")
    monkeypatch.setattr(sys, "argv", ["exo", "--namespace", "explicit-namespace"])

    assert Args.parse().namespace == "explicit-namespace"


def test_bootstrap_peers_parse_zenoh_endpoints_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "EXO_BOOTSTRAP_PEERS",
        " tcp/169.254.233.2:52414, ,tcp/[fe80::1%en6]:52414,",
    )
    monkeypatch.setattr(sys, "argv", ["exo"])

    assert Args.parse().bootstrap_peers == [
        "tcp/169.254.233.2:52414",
        "tcp/[fe80::1%en6]:52414",
    ]
