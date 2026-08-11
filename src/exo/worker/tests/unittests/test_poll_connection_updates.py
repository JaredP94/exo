"""Connection polling must preserve healthy peers when one probe fails."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.utils.channels import channel
from exo.utils.info_gatherer import net_profile


async def test_bad_peer_does_not_abort_other_peer_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_node_id = NodeId("self")
    bad_peer = NodeId("bad-peer")
    good_peer = NodeId("good-peer")
    topology = Topology()
    for node_id in (self_node_id, bad_peer, good_peer):
        topology.add_node(node_id)
    node_network = {
        bad_peer: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="bad")]
        ),
        good_peer: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="good")]
        ),
    }

    async def fake_check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        client: object,
        api_port: int,
    ) -> None:
        del client, api_port
        if target_ip == "bad":
            raise OSError("this peer's interface is pathological")
        out[expected_node_id].add(target_ip)

    monkeypatch.setattr(net_profile, "check_reachability", fake_check_reachability)

    results = [
        result
        async for result in net_profile.check_reachable(
            topology, self_node_id, node_network, api_port=52415
        )
    ]

    assert results == [("good", good_peer)]


@pytest.mark.asyncio
async def test_poll_cycle_publishes_reachable_peer_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.worker import main as worker_main

    self_node_id = NodeId("self")
    peer_node_id = NodeId("peer")
    topology = Topology()
    topology.add_node(self_node_id)
    topology.add_node(peer_node_id)
    worker = worker_main.Worker.__new__(worker_main.Worker)
    worker.node_id = self_node_id
    worker.api_port = 52415
    worker.state = State(
        topology=topology,
        node_network={
            peer_node_id: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(name="en0", ip_address="169.254.240.63")
                ]
            )
        },
    )
    event_sender, event_receiver = channel[Event]()
    worker.event_sender = event_sender
    debug_messages: list[str] = []

    def debug(message: str) -> None:
        debug_messages.append(message)

    async def fake_check_reachable(
        *args: object, **kwargs: object
    ) -> AsyncIterator[tuple[str, NodeId]]:
        del args, kwargs
        yield ("169.254.240.63", peer_node_id)

    monkeypatch.setattr(worker_main, "check_reachable", fake_check_reachable)
    monkeypatch.setattr(worker_main.logger, "debug", debug)

    await worker._poll_connection_updates_once()  # pyright: ignore[reportPrivateUsage]

    published = event_receiver.collect()
    assert len(published) == 1
    event = published[0]
    assert isinstance(event, worker_main.TopologyEdgeCreated)
    assert event.conn == Connection(
        source=self_node_id,
        sink=peer_node_id,
        edge=SocketConnection(
            sink_multiaddr=worker_main.Multiaddr(
                address="/ip4/169.254.240.63/tcp/52415"
            )
        ),
    )
    assert any("connection poll" in message for message in debug_messages)
