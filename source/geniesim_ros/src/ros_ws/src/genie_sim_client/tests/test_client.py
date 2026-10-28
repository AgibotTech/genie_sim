# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Behavioral tests with a deterministic transport; no ROS/GPU runtime needed."""

from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest
import yaml

from genie_sim_client import ClientConfig, SimulationClient
from genie_sim_client import client as implementation
from genie_sim_client._ros_bridge import RosEngineBridge

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    scene = tmp_path / "scene.yaml"
    scene.write_text("robot: {}\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scene_yaml": str(scene)}))
    launcher = tmp_path / "launcher.yaml"
    launcher.write_text((PACKAGE / "config/launcher_client_newton.yaml").read_text())
    return ClientConfig.from_files(launcher, manifest)


class Transport:
    def __init__(self, **kwargs):
        self.state = dict(
            positions=np.array([1.0, 2.0]),
            velocities=np.array([0.0, 0.0]),
            joint_names=("j1", "j2"),
            body_poses=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            body_names=("/World/object",),
            joint_sequence=0,
            body_sequence=1,
            joint_stamp=1.0,
            body_stamp=1.0,
            reset_epoch=0,
            reset_stamp=1.0,
        )
        self.sent = []
        self.closed = False
        self.ready = True
        self.ack_step = True
        self.ack_reset = True
        self.delayed_tf = 0
        self.tf_waits = 0
        self.readers = 1

    def start(self):
        pass

    def close(self):
        self.closed = True

    def check_health(self):
        pass

    def read_state(self):
        if self.ready:
            self.state["joint_sequence"] += 1
        if self.delayed_tf:
            self.tf_waits += 1
            self.delayed_tf -= 1
            if not self.delayed_tf:
                self.state["body_stamp"] = self.state["joint_stamp"]
                self.state["body_sequence"] += 1
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.state.items()}

    def set_joint_layout(self, names):
        self.names = names
        self.ready = False

    def command_readers(self):
        return self.readers

    def prepare_reset(self, seeded):
        self.seeded_prepared = seeded

    def reset_readers(self, seeded):
        return 1

    def send_action(self, target):
        self.sent.append(target.copy())
        if self.ack_step:
            self.state["positions"] = target.copy()
            self.state["joint_sequence"] += 1
            self.state["joint_stamp"] += 1
            self.delayed_tf = 3

    def send_reset(self, ids, seeds, instances):
        self.reset_args = (ids, seeds, instances)
        if self.ack_reset:
            self.state["reset_epoch"] += 1
            self.state["positions"][:] = 0
            self.state["joint_sequence"] += 1
            self.state["joint_stamp"] += 1
            self.state["reset_stamp"] = self.state["joint_stamp"]
            self.delayed_tf = 3


@pytest.fixture
def session(config, monkeypatch):
    bridge = Transport()
    monkeypatch.setattr(implementation, "RosEngineBridge", lambda **kwargs: bridge)
    client = SimulationClient(config, spawn=False, ready_timeout_s=0.03, step_timeout_s=0.01)
    client.start()
    yield client, bridge
    client.close()


def test_step_and_readonly_snapshots_wait_for_tf(session):
    client, bridge = session
    old = client.snapshot()
    targets = np.array([[3.0, 4.0]])
    result = client.step(targets)
    np.testing.assert_array_equal(result.snapshot.joint_positions, targets)
    assert bridge.tf_waits == 3
    assert result.snapshot.body_stamp >= result.snapshot.joint_stamp
    assert result.snapshot.completion == "arrival_v1"
    assert client.config.step_duration == 0.04
    targets[:] = 100
    np.testing.assert_array_equal(result.snapshot.joint_positions, [[3.0, 4.0]])
    np.testing.assert_array_equal(old.joint_positions, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        result.snapshot.joint_positions[0, 0] = 99
    with pytest.raises(ValueError):
        result.snapshot.joint_positions.setflags(write=True)


def test_reset_waits_for_epoch_and_body_and_preserves_selection(session):
    client, bridge = session
    result = client.reset([0], seeds=[42], instances=[3])
    assert bridge.reset_args == ([0], [42], [3])
    assert bridge.seeded_prepared
    assert result.reset_epoch == 1
    assert result.body_stamp >= bridge.state["reset_stamp"]
    assert bridge.tf_waits == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"env_ids": []}, {"env_ids": [-1]}, {"env_ids": [0, 0]}, {"env_ids": [1]}, {"seeds": [1, 2]}, {"instances": [-2]}],
)
def test_bad_reset_never_sends(session, kwargs):
    client, bridge = session
    with pytest.raises(ValueError):
        client.reset(**kwargs)
    assert not hasattr(bridge, "reset_args")


@pytest.mark.parametrize("operation", ["step", "reset"])
def test_timeout_poisons_session_without_replaying(session, operation):
    client, bridge = session
    bridge.ack_step = bridge.ack_reset = False
    with pytest.raises(TimeoutError):
        client.step(np.zeros((1, 2))) if operation == "step" else client.reset()
    with pytest.raises(RuntimeError, match="failed"):
        client.step(np.zeros((1, 2)))
    assert len(bridge.sent) == (1 if operation == "step" else 0)


def test_missing_reset_epoch_is_not_reported_success(session):
    client, bridge = session
    bridge.state["reset_epoch"] = -1
    with pytest.raises(NotImplementedError, match="reset_epoch"):
        client.reset()
    assert not hasattr(bridge, "reset_args")


def test_bad_action_or_completion_never_sends(session):
    client, bridge = session
    for action in (np.zeros((2,)), np.array([[np.nan, 1]])):
        with pytest.raises(ValueError):
            client.step(action)
    with pytest.raises(NotImplementedError):
        client.step(np.zeros((1, 2)), completion="committed_v2")
    with pytest.raises(ValueError):
        client.step(np.zeros((1, 2)), render=True)
    assert not bridge.sent


def test_concurrent_call_is_rejected(session):
    client, bridge = session
    with client._exclusive():
        with pytest.raises(RuntimeError, match="in progress"):
            client.reset()
    assert not hasattr(bridge, "reset_args")


def test_render_runs_once_after_physics(session):
    client, bridge = session
    calls = []
    client._renderer = SimpleNamespace(render=lambda snapshot: calls.append(snapshot) or "pixels")
    result = client.step(np.zeros((1, 2)), render=True)
    assert calls == [result.snapshot]
    assert result.rendered == "pixels"
    assert len(bridge.sent) == 1


def test_layout_reorders_batched_actions_and_state(config, monkeypatch):
    from dataclasses import replace

    bridge = Transport()
    bridge.state.update(
        joint_names=("env_1/j2", "env_0/j1", "env_1/j1", "env_0/j2"),
        positions=np.array([4.0, 1.0, 3.0, 2.0]),
        velocities=np.zeros(4),
        body_names=("env_1/box", "env_0/box"),
        body_poses=np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(implementation, "RosEngineBridge", lambda **kwargs: bridge)
    with SimulationClient(replace(config, num_envs=2), spawn=False) as client:
        np.testing.assert_array_equal(client.snapshot().joint_positions, [[1, 2], [3, 4]])
        result = client.step(np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(bridge.sent[0], [8, 5, 7, 6])
        np.testing.assert_array_equal(result.snapshot.joint_positions, [[5, 6], [7, 8]])


def test_topology_change_is_an_error(session):
    client, bridge = session
    bridge.state["body_names"] = ("/World/other",)
    with pytest.raises(RuntimeError, match="topology changed"):
        client.step(np.zeros((1, 2)))


def test_start_failure_cleans_transport(config, monkeypatch):
    bridge = Transport()
    bridge.start = lambda: (_ for _ in ()).throw(RuntimeError("transport initialization"))
    monkeypatch.setattr(implementation, "RosEngineBridge", lambda **kwargs: bridge)
    client = SimulationClient(config, spawn=False)
    with pytest.raises(RuntimeError, match="transport initialization"):
        client.start()
    assert bridge.closed
    client.close()


def test_single_cached_message_does_not_satisfy_readiness(config, monkeypatch):
    bridge = Transport()
    bridge.ready = False
    monkeypatch.setattr(implementation, "RosEngineBridge", lambda **kwargs: bridge)
    client = SimulationClient(config, spawn=False, ready_timeout_s=0.003)
    with pytest.raises(TimeoutError):
        client.start()
    assert bridge.closed


def test_state_sequence_is_advanced_with_committed_cache_only():
    bridge = RosEngineBridge()
    stamp = SimpleNamespace(sec=2, nanosec=0)
    message = SimpleNamespace(
        name=["joint"], position=[3.0], velocity=[4.0], header=SimpleNamespace(stamp=stamp, frame_id="reset_epoch:7")
    )
    bridge._on_joint_states(message)
    state = bridge.read_state()
    assert state["joint_sequence"] == 1
    assert state["reset_epoch"] == 7
    assert state["positions"][0] == 3
    state["positions"][0] = 99
    assert bridge.read_state()["positions"][0] == 3
    message.name = []
    bridge._on_joint_states(message)
    assert bridge.read_state()["joint_sequence"] == 1


def test_bridge_shutdown_precedes_thread_join_without_global_shutdown():
    bridge = RosEngineBridge()
    events = []
    bridge._executor = SimpleNamespace(shutdown=lambda: events.append("shutdown"))
    bridge._spinner = SimpleNamespace(is_alive=lambda: True, join=lambda **kwargs: events.append("join"))
    bridge._node = SimpleNamespace(destroy_node=lambda: events.append("destroy"))
    bridge._rclpy = SimpleNamespace(shutdown=lambda: pytest.fail("global ROS shutdown"))
    bridge.close()
    bridge.close()
    assert events == ["shutdown", "join", "destroy"]


def test_config_keeps_launcher_params_and_nested_scene_defaults(config):
    doc = yaml.safe_load(config.launcher_yaml.read_text())
    params = doc["genie_sim_engine"]["ros__parameters"]
    params["scene_yaml_defaults"] = {"newton": {"mjc_contact": {"robot": {"condim": 3}}}}
    config.launcher_yaml.write_text(yaml.safe_dump(doc))
    updated = ClientConfig.from_files(config.launcher_yaml, config.stage_manifest)
    values = dict(item.split(":=", 1) for item in updated.command if ":=" in item)
    assert yaml.safe_load(values["scene_yaml_defaults"]) == params["scene_yaml_defaults"]
    assert values["step_lockstep_ticks"] == "4"
    assert values["physics_params_file"] == "physics_params.yaml"
    assert "rlinf.yaml" not in str(updated.command)


@pytest.mark.parametrize(
    "key,value",
    [
        ("step_mode", "realtime"),
        ("step_lockstep_ticks", 0),
        ("step_lockstep_ticks", 1.5),
        ("use_tf_render", False),
        ("physics_hz", float("nan")),
        ("num_envs", 0),
    ],
)
def test_config_rejects_incompatible_engine_modes(config, key, value):
    doc = yaml.safe_load(config.launcher_yaml.read_text())
    doc["genie_sim_engine"]["ros__parameters"][key] = value
    config.launcher_yaml.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
        ClientConfig.from_files(config.launcher_yaml, config.stage_manifest)


def test_numpy_reset_indices_and_int64_seed_are_accepted(session):
    client, bridge = session
    client.reset(np.array([0]), seeds=np.array([42], dtype=np.int64))
    assert bridge.reset_args == ([0], [42], None)


def test_cleanup_failure_retains_process_and_prevents_replacement(session, monkeypatch):
    client, bridge = session
    sentinel = object()
    client._process = sentinel
    monkeypatch.setattr(
        implementation, "_terminate_process_group", lambda proc: (_ for _ in ()).throw(RuntimeError("group survived"))
    )
    with pytest.raises(RuntimeError, match="group survived"):
        client.close()
    assert client._process is sentinel
    assert not client.is_started
    with pytest.raises(RuntimeError, match="group survived"):
        client.start()
    client._process = None  # synthetic process has no resources; allow fixture cleanup


@pytest.mark.parametrize(
    "names,worlds",
    [
        (("env_0/a", "a"), 1),
        (("env_0/a", "env_1/b"), 2),
        (("env_0/a",), 2),
        (("a", "a"), 1),
    ],
)
def test_nonuniform_or_ambiguous_world_names_are_rejected(names, worlds):
    with pytest.raises(ValueError):
        implementation._layout(names, worlds)


def test_body_wait_cannot_return_behind_a_newer_joint_cache(session, monkeypatch):
    client, bridge = session
    original = bridge.read_state
    advanced = False

    def read():
        nonlocal advanced
        state = original()
        if bridge.sent and bridge.delayed_tf == 0 and not advanced:
            # A later joint sample arrives just as the first target TF does.
            # The returned cache must wait for THAT sample's body state too.
            advanced = True
            bridge.state["joint_sequence"] += 1
            bridge.state["joint_stamp"] += 1
            bridge.delayed_tf = 2
            state["joint_sequence"] = bridge.state["joint_sequence"]
            state["joint_stamp"] = bridge.state["joint_stamp"]
        return state

    monkeypatch.setattr(bridge, "read_state", read)
    result = client.step(np.zeros((1, 2)))
    assert advanced
    assert result.snapshot.body_stamp >= result.snapshot.joint_stamp
