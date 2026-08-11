"""Transport failures must not abort the reachability scan."""

from __future__ import annotations

import socket

import httpx
import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.utils.info_gatherer.net_profile import check_reachability, check_reachable


@pytest.mark.parametrize(
    "raised",
    [
        OSError("No route to host"),
        socket.timeout("timed out"),
        socket.gaierror("nodename nor servname provided"),
        ConnectionRefusedError("connection refused"),
    ],
)
async def test_transport_failures_produce_no_reachability_result(
    monkeypatch: pytest.MonkeyPatch, raised: Exception
) -> None:
    self_node_id = NodeId("self")
    peer_node_id = NodeId("peer")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(peer_node_id)

    async def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise raised

    async def no_sleep(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr("httpx.AsyncClient.get", boom)
    monkeypatch.setattr("exo.utils.info_gatherer.net_profile.anyio.sleep", no_sleep)

    del topology, self_node_id
    out: dict[NodeId, set[str]] = {}
    async with httpx.AsyncClient() as client:
        await check_reachability(
            "169.254.99.99", peer_node_id, out, client, api_port=52415
        )

    assert out == {}


def _topology_and_network() -> tuple[
    Topology, NodeId, NodeId, dict[NodeId, NodeNetworkInfo]
]:
    self_node_id = NodeId("self")
    peer_node_id = NodeId("peer")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(peer_node_id)
    node_network = {
        peer_node_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="169.254.99.99")]
        )
    }
    return topology, self_node_id, peer_node_id, node_network


async def test_matching_node_id_produces_a_reachability_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, self_node_id, peer_node_id, _node_network = _topology_and_network()

    class Response:
        status_code = 200
        text = '"peer"'

    async def get(*args: object, **kwargs: object) -> Response:
        del args, kwargs
        return Response()

    monkeypatch.setattr("httpx.AsyncClient.get", get)

    del topology, self_node_id
    out: dict[NodeId, set[str]] = {}
    async with httpx.AsyncClient() as client:
        await check_reachability(
            "169.254.99.99", peer_node_id, out, client, api_port=52415
        )

    assert out == {peer_node_id: {"169.254.99.99"}}


async def test_mismatched_node_id_produces_no_reachability_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, self_node_id, _peer_node_id, _node_network = _topology_and_network()

    class Response:
        status_code = 200
        text = '"someone-else"'

    async def get(*args: object, **kwargs: object) -> Response:
        del args, kwargs
        return Response()

    monkeypatch.setattr("httpx.AsyncClient.get", get)

    del topology, self_node_id
    out: dict[NodeId, set[str]] = {}
    async with httpx.AsyncClient() as client:
        await check_reachability(
            "169.254.99.99", _peer_node_id, out, client, api_port=52415
        )

    assert out == {}


async def test_programming_errors_are_not_silenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, self_node_id, peer_node_id, node_network = _topology_and_network()

    async def programming_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("programming failure")

    monkeypatch.setattr(
        "exo.utils.info_gatherer.net_profile.check_reachability",
        programming_failure,
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        async for _item in check_reachable(
            topology, self_node_id, node_network, api_port=52415
        ):
            pass

    assert isinstance(exc_info.value.exceptions[0], TypeError)
    del peer_node_id
