#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``/rlinf/body_aabb_local`` is latched by a writer that has to stay alive.

The bug
-------
``publish_body_aabb_local`` used to create an rclpy node, publish, and call
``node.destroy_node()`` on the next line, on the stated reasoning that
"TRANSIENT_LOCAL keeps the sample alive for late subscribers even after the
node that published it is gone".

It does not.  ROS 2 durability is **writer-side**: TRANSIENT_LOCAL makes the
*publisher* responsible for replaying its history to late-joining
subscriptions, so the latch dies with the writer.  DDS ``TRANSIENT`` /
``PERSISTENT`` would outlive it; rmw exposes neither.

Measured on Humble + Fast DDS, matching QoS on both ends, subscriber
attaching 0.5 s after the publish:

    publisher destroyed after publish  ->  received nothing in 2 s
    publisher kept alive               ->  received the sample

The only consumer is rlinf's wrapper, in another process, attaching after
engine build.  So this channel listed in ``ros2 topic list``, errored
nowhere, and had never once carried data — the same failure shape as the
``/joint_states_scene`` silence found earlier in this merge.

What is pinned here
-------------------
The lifetime, not the delivery.  A real end-to-end check needs two DDS
participants and a couple of seconds of discovery, which is neither fast nor
reliable enough for this suite; the measurement above stands in for it, and
what these tests guard is the property that measurement identified: the
writer created by a publish is still there afterwards, and only
``shutdown()`` takes it away.

``rclpy`` / ``std_msgs`` are faked rather than imported.  Both are real on a
ROS host, but creating actual DDS participants in a unit test is exactly the
slowness this avoids — and the fakes let the QoS the publisher *asks for* be
asserted, which a real node would swallow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_INNER_PKG = _HERE.parent


def _load_by_path(mod_name: str, rel_path: str):
    mod_path = _INNER_PKG / rel_path
    assert mod_path.is_file(), f"{rel_path} not found at {mod_path}"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakePublisher:
    def __init__(self, topic, qos):
        self.topic = topic
        self.qos = qos
        self.published = []

    def publish(self, msg):
        self.published.append(msg.data)


class _FakeNode:
    def __init__(self, name, namespace=None):
        self.name = name
        self.namespace = namespace
        self.publishers = []
        self.destroyed = 0

    def create_publisher(self, msg_type, topic, qos):
        pub = _FakePublisher(topic, qos)
        self.publishers.append(pub)
        return pub

    def destroy_node(self):
        self.destroyed += 1


class _FakeRclpy(types.ModuleType):
    def __init__(self):
        super().__init__("rclpy")
        self.nodes = []
        self.inits = 0
        self.shutdowns = 0
        self._ok = True

    def ok(self):
        return self._ok

    def init(self):
        self.inits += 1
        self._ok = True

    def shutdown(self):
        self.shutdowns += 1
        self._ok = False

    def create_node(self, name, namespace=None):
        node = _FakeNode(name, namespace)
        self.nodes.append(node)
        return node


class _Log:
    def __init__(self):
        self.infos = []
        self.warns = []

    def info(self, m):
        self.infos.append(m)

    def warn(self, m):
        self.warns.append(m)


@pytest.fixture
def bp(monkeypatch):
    """A freshly loaded publisher module over a fake rclpy.

    Reloaded per test because the retained writer is module state — sharing
    it across tests would make each one depend on the last one's teardown.
    """
    rclpy = _FakeRclpy()
    qos_mod = types.ModuleType("rclpy.qos")
    for enum_name, members in (
        ("DurabilityPolicy", ("TRANSIENT_LOCAL", "VOLATILE")),
        ("ReliabilityPolicy", ("RELIABLE", "BEST_EFFORT")),
        ("HistoryPolicy", ("KEEP_LAST", "KEEP_ALL")),
    ):
        setattr(qos_mod, enum_name, type(enum_name, (), {m: f"{enum_name}.{m}" for m in members}))

    class _QoSProfile:
        def __init__(self, depth=None, reliability=None, history=None, durability=None):
            self.depth = depth
            self.reliability = reliability
            self.history = history
            self.durability = durability

    qos_mod.QoSProfile = _QoSProfile
    rclpy.qos = qos_mod

    std_msgs = types.ModuleType("std_msgs")
    msg_mod = types.ModuleType("std_msgs.msg")

    class _String:
        def __init__(self):
            self.data = ""

    msg_mod.String = _String
    std_msgs.msg = msg_mod

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_mod)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", msg_mod)
    mod = _load_by_path("benchmark_publisher_latch", "engine/newton/benchmark_publisher.py")
    mod._fake_rclpy = rclpy
    yield mod
    mod.shutdown()


_NAMES = ["robot/base_link", "Objects/cup_000/body"]
_BOXES = [[0.0, 0.0, 0.1, 0.05, 0.05, 0.1], [0.0, 0.0, 0.05, 0.03, 0.03, 0.05]]


def test_static_world_writer_is_separate_and_clears_on_reload(bp):
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    bp.publish_static_prop_aabb(['bin'], [[0, 0, 0, 1, 1, 1]])
    node = bp._retained()
    assert len(node.publishers) == 2
    writer = next(p for p in node.publishers if p.topic == '/rlinf/static_prop_aabb')
    assert json.loads(writer.published[-1])['aabb_world'] == [[0, 0, 0, 1, 1, 1]]
    bp.publish_static_prop_aabb([], [])
    assert json.loads(writer.published[-1]) == {'names': [], 'aabb_world': []}
    assert node.destroyed == 0
    bp.shutdown()
    assert node.destroyed == 1


# ----------------------------------------------------------------------
# The lifetime contract — this is the regression
# ----------------------------------------------------------------------


def test_an_empty_first_static_publish_creates_nothing(bp):
    """Engines with no eval_static_props must not get a node or the topic at all."""
    bp.publish_static_prop_aabb([], [])
    assert bp._retained() is None


def test_a_failing_static_publish_only_warns(bp):
    def _boom(*_a, **_k):
        raise RuntimeError("no ROS context")
    bp._fake_rclpy.create_node = _boom
    log = _Log()
    bp.publish_static_prop_aabb(["bin"], [[0, 0, 0, 1, 1, 1]], logger=log)
    assert log.warns, "a failed publish must be reported"


def test_the_writer_outlives_the_publish(bp):
    """The whole bug in one assertion: destroying it here un-latches."""
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    assert bp._retained() is not None, "the writer was destroyed; the latch is gone"
    assert bp._fake_rclpy.nodes[0].destroyed == 0


def test_shutdown_is_what_takes_it_away(bp):
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    node = bp._retained()
    bp.shutdown()
    assert bp._retained() is None
    assert node.destroyed == 1


def test_shutdown_is_idempotent(bp):
    """Called from ``PhysicsEngine.shutdown``, which runs on every exit path."""
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    bp.shutdown()
    bp.shutdown()
    assert bp._retained() is None


def test_shutdown_without_a_publish_is_silent(bp):
    bp.shutdown()  # no adapter exposed a table; nothing was ever created
    assert bp._retained() is None


def test_shutdown_leaves_the_rclpy_context_alone(bp):
    """Markers, the C++ bridge and the wrapper may all still be using it."""
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    bp.shutdown()
    assert bp._fake_rclpy.shutdowns == 0


def test_a_second_publish_reuses_the_one_writer(bp):
    """Two writers on one latched topic would leave a reader two samples to
    reconcile with no way to tell which is current."""
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    assert len(bp._fake_rclpy.nodes) == 1
    assert len(bp._fake_rclpy.nodes[0].publishers) == 1


# ----------------------------------------------------------------------
# The wire contract, unchanged by the fix
# ----------------------------------------------------------------------


def test_the_qos_is_the_latched_profile(bp):
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    qos = bp._fake_rclpy.nodes[0].publishers[0].qos
    assert qos.durability == "DurabilityPolicy.TRANSIENT_LOCAL"
    assert qos.reliability == "ReliabilityPolicy.RELIABLE"
    assert qos.history == "HistoryPolicy.KEEP_LAST"
    assert qos.depth == 1


def test_the_topic_is_absolute(bp):
    """A relative name would land under ``/genie_sim_engine`` and the wrapper
    subscribes to the absolute path."""
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    assert bp._fake_rclpy.nodes[0].publishers[0].topic == "/rlinf/body_aabb_local"


def test_the_payload_shape_is_unchanged(bp):
    bp.publish_body_aabb_local(_NAMES, _BOXES)
    body = json.loads(bp._fake_rclpy.nodes[0].publishers[0].published[0])
    assert body["names"] == _NAMES
    assert len(body["aabb_local"]) == len(_NAMES)
    assert all(len(row) == 6 for row in body["aabb_local"])
    assert body["aabb_local"][0][2] == pytest.approx(0.1)


def test_an_empty_table_publishes_nothing_at_all(bp):
    """No adapter table is the ordinary case for every adapter except mjwarp
    multi-env; it must not leave an empty latched sample behind."""
    bp.publish_body_aabb_local([], [])
    assert bp._retained() is None
    assert bp._fake_rclpy.nodes == []


def test_a_broken_publish_warns_and_does_not_raise(bp):
    """Engine startup is never worth failing over eval-checker support."""
    log = _Log()
    bp.publish_body_aabb_local(_NAMES, [[0.0, 0.0]], logger=log)  # not reshapeable to (2, 6)
    assert log.warns and "body_aabb_local publish failed" in log.warns[0]


def test_the_log_line_says_the_writer_is_held(bp):
    """An operator reading the log should be able to tell this is the fixed
    behaviour, not the version that published into a void."""
    log = _Log()
    bp.publish_body_aabb_local(_NAMES, _BOXES, logger=log)
    assert log.infos and "latched" in log.infos[0]
    assert "shutdown" in log.infos[0]
