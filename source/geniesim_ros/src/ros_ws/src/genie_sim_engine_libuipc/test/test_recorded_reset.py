# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Real archive rotation around a constructor-free engine owner, without CUDA."""

from types import SimpleNamespace as NS
import threading
import numpy as np
import pytest
from genie_sim_engine_libuipc.engine import LibuipcEngine
from genie_sim_engine_libuipc.physical_state_recording import (
    PhysicalStateRecorder,
    PhysicalStateRecordingConfig,
    PhysicalStateArchive,
)


class Array:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def numpy(self):
        return self.data.copy()

    def assign(self, data):
        self.data[:] = data.numpy() if hasattr(data, "numpy") else data

    def zero_(self):
        self.data.fill(0)


def owner(tmp_path):
    e = object.__new__(LibuipcEngine)
    e._source_id = "world"
    e._epoch = 0
    e._tick = 1
    e._sim_time = 1 / 60
    e._paused = False
    e._closed = False
    e._sequence_index = 0
    e._sequence_finished = False
    e._execution_failure = None
    e._failure_lock = threading.Lock()
    e._failure_version = 0
    e._failure_finalized = False
    e._actions = e._snapshots = e._snapshot_requests = e._identity_source = None
    e._initial_q = np.array([0.2], dtype=np.float32)

    def state():
        return NS(
            body_q=Array([[0, 0, 1, 0, 0, 0, 1]]),
            body_qd=Array([[0] * 6]),
            particle_q=Array([[0, 0, 1]]),
            particle_qd=Array([[0, 0, 0]]),
            joint_q=Array([0.2]),
            joint_qd=Array([0]),
        )

    e._state_0, e._state_1 = state(), state()
    e._model = NS(joint_q=Array([0.2]))
    e._sequence_entries = [
        dict(
            state_recording=dict(enabled=True, output_dir=str(tmp_path / str(i))),
            material=dict(episode_plan={}, receipt_path=str(tmp_path / f"r{i}")),
        )
        for i in range(2)
    ]
    e._scene_cfg = {
        "libuipc": {"state_recording": e._sequence_entries[0]["state_recording"]},
        "generalization": {"material": e._sequence_entries[0]["material"]},
    }
    e._physical_metadata = lambda: dict(source_id=e._source_id, epoch=e._epoch)
    calls = []

    def recover(state):
        calls.append("native")
        state.particle_q.assign([[0, 0, 1]])
        state.particle_qd.zero_()

    e._driver = NS(
        _feedback_node=None,
        _cpu_started=True,
        _solver=NS(reset=recover),
        stop_feedback=lambda **kw: calls.append(kw),
        reset_inputs=lambda: calls.append("inputs"),
        start_cpu_worker=lambda: None,
    )
    e._state_recorder = PhysicalStateRecorder(
        PhysicalStateRecordingConfig.from_scene(e._scene_cfg), metadata=e._physical_metadata()
    )
    e._state_recorder.record(e._state_0, tick=0, sim_time=0, dt=0, initial=True)
    e._state_0.particle_q.assign([[0, 0, 0.9]])
    e._state_recorder.record(e._state_0, tick=1, sim_time=1 / 60, dt=1 / 60)
    e._joint_names = ["arm"]
    e._scalar = {"arm": None}
    e._physics_hz = 60
    return e, calls


def test_finish_then_reset_seals_previous_and_records_fresh_initial_state(tmp_path):
    e, calls = owner(tmp_path)
    first = e.sequence_operation("finish", 0)
    assert e._paused and e._state_recorder is None
    old = PhysicalStateArchive(tmp_path / "0")
    assert len(list(old.iter_frames())) == 2
    e.sequence_operation("reset", 1)
    assert calls == [{"close_worker": False}, "native", "inputs"]
    assert e._epoch == 1 and e._tick == 0 and e._sim_time == 0 and e._paused
    assert PhysicalStateArchive(tmp_path / "0").manifest["metadata"]["epoch"] == 0
    e._state_recorder.close(clean=True)
    new = PhysicalStateArchive(tmp_path / "1")
    assert new.manifest["metadata"]["epoch"] == 1
    assert np.array_equal(next(new.iter_frames())["state"]["particle_q"], [[0, 0, 1]])


def test_open_recording_blocks_reset_and_native_failure_never_advances_epoch(tmp_path):
    e, calls = owner(tmp_path)
    with pytest.raises(RuntimeError, match="finish"):
        e.episode_reset(instances={0: 1})
    assert not calls and e._epoch == 0
    e.sequence_operation("finish", 0)

    def fail(*args):
        raise RuntimeError("native recover failed")

    e._driver._solver.reset = fail
    with pytest.raises(RuntimeError, match="native recover failed"):
        e.episode_reset(instances={0: 1})
    assert e._epoch == 0 and e._execution_failure is not None
    assert not (tmp_path / "1").exists()
