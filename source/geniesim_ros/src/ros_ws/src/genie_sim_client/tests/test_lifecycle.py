# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lifecycle service responses enqueue work; state confirmation is the barrier."""

from types import SimpleNamespace

import pytest

from genie_sim_client._ros_bridge import RosEngineBridge


class Future:
    def __init__(self, response):
        self.response = response

    def done(self):
        return True

    def result(self):
        return self.response


@pytest.fixture
def lifecycle(monkeypatch):
    bridge = RosEngineBridge()
    bridge._node = object()
    bridge._EmptyRequest = bridge._GetStateRequest = object
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("genie_sim_client._ros_bridge.time.monotonic", lambda: clock.now)
    monkeypatch.setattr(
        "genie_sim_client._ros_bridge.time.sleep", lambda delta: setattr(clock, "now", clock.now + delta)
    )
    engine = SimpleNamespace(state=3, pending=None, commands=[], reads=0, apply=True, ready=True)

    def command(verb):
        engine.commands.append(verb)
        engine.pending = 2 if verb == "pause" else 3
        engine.reads = 0
        return Future(SimpleNamespace())

    def state():
        engine.reads += 1
        if engine.apply and engine.reads >= 3:
            engine.state = engine.pending
        return Future(SimpleNamespace(current_state=SimpleNamespace(id=engine.state)))

    bridge._lifecycle_clients = {
        verb: SimpleNamespace(service_is_ready=lambda: engine.ready, call_async=lambda _, verb=verb: command(verb))
        for verb in ("pause", "resume")
    }
    bridge._lifecycle_clients["get_state"] = SimpleNamespace(
        service_is_ready=lambda: engine.ready, call_async=lambda _: state()
    )
    return bridge, engine


def test_empty_response_is_not_pause_or_resume_confirmation(lifecycle):
    bridge, engine = lifecycle
    bridge.set_paused(True, 0.01)
    assert engine.state == 2
    assert engine.reads == 3
    bridge.set_paused(False, 0.01)
    assert engine.state == 3
    assert engine.reads == 3
    assert engine.commands == ["pause", "resume"]


def test_unapplied_pause_times_out_instead_of_reporting_success(lifecycle):
    bridge, engine = lifecycle
    engine.apply = False
    with pytest.raises(TimeoutError, match="state confirmation"):
        bridge.set_paused(True, 0.003)
    assert engine.state == 3
    assert engine.commands == ["pause"]


def test_missing_services_never_enqueue_pause(lifecycle):
    bridge, engine = lifecycle
    engine.ready = False
    with pytest.raises(TimeoutError, match="service discovery"):
        bridge.set_paused(True, 0.003)
    assert engine.commands == []


def test_lifecycle_service_construction_failure_keeps_node_owned_for_close(monkeypatch):
    import sys

    service = SimpleNamespace(Request=object)
    monkeypatch.setitem(sys.modules, "lifecycle_msgs.srv", SimpleNamespace(GetState=service))
    monkeypatch.setitem(sys.modules, "std_srvs.srv", SimpleNamespace(Empty=service))
    events = []

    def create_client(_, path):
        events.append(path)
        if path.endswith("resume"):
            raise RuntimeError("client construction failed")
        return object()

    bridge = RosEngineBridge()
    bridge._node = SimpleNamespace(create_client=create_client, destroy_node=lambda: events.append("destroy"))
    with pytest.raises(RuntimeError, match="construction failed"):
        bridge.set_paused(True, 0.01)
    bridge.close()
    assert events == ["/genie_sim_engine_newton/pause", "/genie_sim_engine_newton/resume", "destroy"]
    assert not bridge._lifecycle_clients
