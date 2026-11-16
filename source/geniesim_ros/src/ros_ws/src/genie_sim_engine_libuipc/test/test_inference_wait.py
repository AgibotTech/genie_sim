# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU-only checks for an owner-held step and unchanged committed archives."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest

from genie_sim_engine_libuipc.action_sequence import ActionSequence
from genie_sim_engine_libuipc.engine import LibuipcEngine
from genie_sim_engine_libuipc.episode_sequence import EpisodeRequests
from genie_sim_engine_libuipc.observation_snapshot import SnapshotRequests
from genie_sim_engine_libuipc.physical_state_recording import PhysicalStateArchive
from test_recorded_reset import owner as recorded_owner


def packet(engine, token="action", ticks=2):
    return dict(
        request_id=token,
        source_id=engine._source_id,
        epoch=engine._epoch,
        names=["arm"],
        positions=[0.2],
        duration_s=ticks / engine._physics_hz,
    )


def episode(engine, operation, index=0):
    return dict(
        version=1,
        request_id=operation + str(index),
        source_id=engine._source_id,
        epoch=engine._epoch,
        operation=operation,
        index=index,
    )


@pytest.fixture
def engine(tmp_path):
    e, _ = recorded_owner(tmp_path)
    e._pause_when_idle = True
    e._dt = 1 / e._physics_hz
    e._control = e._ctx = None
    e._body_offsets = {}
    e._scalar = {"arm": NS(q_start=0)}
    e._static_joint_q = {}
    e._driver._cpu_q_indices = np.array([0])
    e._driver._cpu_config = NS(synchronization="barrier", sync_substeps=3, viewer=NS(enabled=False))
    e._driver._cpu_snapshot = {"steps": 0}
    e._driver._cpu_names = ["arm"]
    e._scene = NS(objects=[], _ranges={}, on_post_step=lambda *args: None)
    e._lifecycle = NS(_node=NS(context=NS(ok=lambda: True)))
    e._actions = ActionSequence(["arm"], [0.2], physics_hz=e._physics_hz, source_id=e._source_id)
    e._episode_source = None
    e._replies = []
    e._action_source = NS(publish=e._replies.append)
    e.get_body_transforms = lambda: (e._state_0.body_q.numpy(), ["/body"])
    for state in (e._state_0, e._state_1):
        state.clear_forces = lambda: None

    def substep(model, source, target, control, dt, **kwargs):
        assert kwargs["completed_sim_time"] == e._sim_time
        e._driver._cpu_snapshot["steps"] += 3
        for name in ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd"):
            getattr(target, name).assign(getattr(source, name))
        target.particle_q.data[:, 2] -= 0.001

    e._driver.substep = substep
    e._snapshot_requests = SnapshotRequests(e)
    e._captured = []
    e._capture_event = threading.Event()

    def capture():
        replies = e._snapshot_requests.on_tick()
        e._captured.extend(replies)
        if replies:
            e._capture_event.set()
        return replies

    e._snapshots = NS(on_tick=capture, cancel=e._snapshot_requests.cancel, reset_epoch=e._snapshot_requests.reset_epoch)
    yield e
    e._closed = True
    if e._state_recorder is not None:
        e._state_recorder.close(clean=e._execution_failure is None)


@contextmanager
def step_call(engine):
    done = threading.Event()
    errors = []

    def run():
        try:
            engine.step(engine._dt, time.monotonic())
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield done, errors
    finally:
        context = engine._lifecycle._node.context
        was_ok = context.ok
        if not done.is_set():
            context.ok = lambda: False
        thread.join(timeout=2)
        context.ok = was_ok
        assert not thread.is_alive(), "owner wait did not observe stopped ROS context"


def snapshot(engine, token):
    return engine._snapshot_requests.request(dict(request_id=token, source_id=engine._source_id, epoch=engine._epoch))


def test_runnable_tracks_pending_active_and_completed_action():
    actions = ActionSequence(["arm"], [0.2], physics_hz=60, source_id="world")
    e = NS(_source_id="world", _epoch=0, _physics_hz=60)
    assert actions.runnable is False
    actions.submit(packet(e))
    assert actions.runnable is True
    actions.begin_step()
    actions.complete_step(tick=1, sim_time=1 / 60)
    assert actions.runnable is True
    actions.begin_step()
    actions.complete_step(tick=2, sim_time=2 / 60)
    assert actions.runnable is False


def test_idle_capture_and_inference_wait_leave_cpu_native_time_and_archive_fixed(engine):
    e = engine
    root = e._state_recorder.directory
    before = {p.name: p.stat().st_size for p in Path(root).iterdir()}
    assert snapshot(e, "initial") is None
    with step_call(e) as (done, errors):
        assert e._capture_event.wait(1)
        assert not done.is_set(), "idle step returned without a committed native solve"
        time.sleep(0.03)
        assert e._tick == 1 and e._sim_time == 1 / 60
        assert e._driver._cpu_snapshot["steps"] == 0
        assert {p.name: p.stat().st_size for p in Path(root).iterdir()} == before
        assert e._captured[0]["tick"] == 1
        e._actions.submit(packet(e))
        assert done.wait(1) and not errors
    e._closed = False
    e.step(e._dt, time.monotonic())
    assert e._tick == 3 and e._driver._cpu_snapshot["steps"] == 6
    assert e._replies[-1]["ipc_steps"] == 2 and e._replies[-1]["cpu_steps"] == 6
    e._capture_event.clear()
    snapshot(e, "after")
    with step_call(e) as (done, errors):
        assert e._capture_event.wait(1)
        assert not done.is_set()
        time.sleep(0.03)
        assert e._tick == 3 and e._driver._cpu_snapshot["steps"] == 6
        assert e._captured[-1]["tick"] == 3
    e._state_recorder.close(clean=True)
    frames = list(PhysicalStateArchive(root).iter_frames())
    assert [frame["tick"] for frame in frames] == [0, 1, 2, 3]


def test_wait_serves_finish_reset_and_begin_without_extra_integration(engine):
    e = engine
    queue = EpisodeRequests(e)
    replies = []
    e._episode_source = NS(drain=lambda: queue.drain(replies.append))
    with step_call(e) as (done, errors):
        queue.request(episode(e, "finish"))
        deadline = time.monotonic() + 1
        while not replies and time.monotonic() < deadline:
            time.sleep(0.001)
        assert replies and replies[-1]["status"] == "completed"
        assert not done.is_set() and e._tick == 1 and e._paused
        queue.request(episode(e, "reset", 1))
        deadline = time.monotonic() + 1
        while len(replies) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(replies) == 2 and replies[-1]["status"] == "completed"
        assert e._epoch == 1 and e._tick == 0 and e._paused and not done.is_set()
        queue.request(episode(e, "begin", 1))
        deadline = time.monotonic() + 1
        while len(replies) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(replies) == 3 and replies[-1]["status"] == "completed"
        assert not e._paused and not done.is_set()
        e._actions.submit(packet(e, ticks=1))
        assert done.wait(1) and not errors
        assert e._tick == 1 and e._sim_time == 1 / 60


def test_wait_propagates_transport_failure_without_another_cpu_batch(engine):
    with step_call(engine) as (done, errors):
        time.sleep(0.02)
        assert not done.is_set()
        engine.report_transport_failure(RuntimeError("transport stopped"))
        assert done.wait(1)
        assert isinstance(errors[0], RuntimeError)
        assert engine._tick == 1 and engine._driver._cpu_snapshot["steps"] == 0
        assert engine._failure_finalized


def test_stopped_ros_context_interrupts_owner_wait_without_fault_latching(engine):
    engine._lifecycle._node.context.ok = lambda: False
    with step_call(engine) as (done, errors):
        assert done.wait(1)
        assert isinstance(errors[0], KeyboardInterrupt)
        assert engine._execution_failure is None
        assert engine._tick == 1 and engine._driver._cpu_snapshot["steps"] == 0


def test_disabled_scene_retains_free_running_and_legacy_paused_behavior(engine):
    engine._pause_when_idle = False
    engine.step(engine._dt, time.monotonic())
    assert engine._tick == 2 and engine._driver._cpu_snapshot["steps"] == 3
    engine.pause()
    assert engine.step(engine._dt, time.monotonic()) == 0.0
    assert engine._tick == 2 and engine._driver._cpu_snapshot["steps"] == 3


@pytest.mark.parametrize("value", [None, 0, 1, "false", {}, []])
def test_pause_when_idle_requires_a_boolean(engine, value):
    engine._scene_cfg = {"libuipc": {"pause_when_idle": value}}
    with pytest.raises(ValueError, match="pause_when_idle.*boolean"):
        engine._initialize_state_services()


@pytest.mark.parametrize("missing", ["action_sequence", "observation_snapshot"])
def test_pause_when_idle_requires_action_and_snapshot_channels(engine, missing):
    engine._scene_cfg = {
        "libuipc": dict(
            pause_when_idle=True, action_sequence=dict(topic="/action"), observation_snapshot=dict(topic="/observation")
        )
    }
    engine._scene_cfg["libuipc"].pop(missing)
    with pytest.raises(ValueError, match="pause_when_idle.*action_sequence.*observation_snapshot"):
        engine._initialize_state_services()


def test_pause_when_idle_rejects_a_live_cpu_viewer(engine):
    engine._scene_cfg = {
        "libuipc": dict(
            pause_when_idle=True, action_sequence=dict(topic="/action"), observation_snapshot=dict(topic="/observation")
        )
    }
    engine._driver._cpu_config.viewer.enabled = True
    with pytest.raises(ValueError, match="pause_when_idle.*viewer"):
        engine._initialize_state_services()


@pytest.mark.parametrize("settings", [{}, {"pause_when_idle": False}])
def test_default_or_false_setting_accepts_a_scene_without_action_channels(engine, settings):
    engine._scene_cfg = {"libuipc": settings}
    engine._sequence_entries = None
    engine._actions = engine._snapshots = engine._snapshot_requests = None
    engine._initialize_state_services()
    assert engine._pause_when_idle is False
    engine.step(engine._dt, time.monotonic())
    assert engine._tick == 2


@pytest.mark.parametrize("cpu", [None, NS(synchronization="latest")])
def test_pause_when_idle_requires_cpu_barrier_synchronization(engine, cpu):
    engine._scene_cfg = {
        "libuipc": dict(
            pause_when_idle=True, action_sequence=dict(topic="/action"), observation_snapshot=dict(topic="/observation")
        )
    }
    engine._driver._cpu_config = cpu
    with pytest.raises(ValueError, match="CPU barrier"):
        engine._initialize_state_services()


def test_engine_close_releases_an_idle_owner_without_integrating(engine):
    with step_call(engine) as (done, errors):
        time.sleep(0.02)
        assert not done.is_set()
        engine._closed = True
        assert done.wait(1)
        assert isinstance(errors[0], RuntimeError) and "closed" in str(errors[0])
        assert engine._tick == 1 and engine._driver._cpu_snapshot["steps"] == 0


def test_automatic_hold_preserves_explicit_episode_pause_resume_and_finish(engine):
    e = engine
    e._paused = True
    e.sequence_operation("begin", 0)
    e.sequence_operation("pause", 0)
    assert e._paused and e._inference_paused
    recorder = e._state_recorder
    root = recorder.directory
    before = {p.name: p.stat().st_size for p in root.iterdir()}
    e._actions.submit(packet(e, ticks=1))
    queue = EpisodeRequests(e)
    replies = []
    e._episode_source = NS(drain=lambda: queue.drain(replies.append))
    snapshot(e, "explicit-pause")
    with step_call(e) as (done, errors):
        assert e._capture_event.wait(1)
        assert not done.is_set(), "a pending action bypassed explicit inference pause"
        time.sleep(0.02)
        assert e._tick == 1 and e._driver._cpu_snapshot["steps"] == 0
        assert e._state_recorder is recorder
        assert {p.name: p.stat().st_size for p in root.iterdir()} == before
        queue.request(episode(e, "resume"))
        assert done.wait(1) and not errors
        assert replies[-1]["status"] == "completed"
        assert not e._paused and not e._inference_paused
        assert e._tick == 2 and e._driver._cpu_snapshot["steps"] == 3
    with step_call(e) as (done, errors):
        queue.request(episode(e, "finish"))
        deadline = time.monotonic() + 1
        while len(replies) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(replies) == 2 and replies[-1]["status"] == "completed"
        assert e._paused and e._sequence_finished and not e._inference_paused
        assert not done.is_set() and e._state_recorder is None
    frames = list(PhysicalStateArchive(root).iter_frames())
    assert [frame["tick"] for frame in frames] == [0, 1, 2]
