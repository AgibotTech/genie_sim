# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Actual MuJoCo worker resets twice with the same process and native model."""

from dataclasses import replace
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from genie_sim_engine_libuipc.cpu.worker import CpuJointWorker, CpuMuJoCoConfig


@pytest.mark.parametrize("execution", ["thread", "process"])
def test_reset_restores_state_and_reuses_worker(execution, tmp_path):
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string(
        """<mujoco><option gravity="0 0 0"/><worldbody><body name="link"><joint name="arm" type="hinge"/><geom type="sphere" size="0.1" mass="1"/></body></worldbody><actuator><position joint="arm" kp="10"/></actuator></mujoco>"""
    )
    path = tmp_path / "model.mjb"
    mj.mj_saveModel(model, str(path), None)
    cfg = CpuMuJoCoConfig(
        execution=execution,
        synchronization="barrier",
        sync_substeps=10,
        physics_hz=1000,
        sync_timeout_s=5,
        drives=({"pattern": "arm", "kp": 100, "kd": 1, "force_limit": 100},),
    )
    worker = CpuJointWorker(path.read_bytes(), cfg, ("arm",), np.array([0.2]), body_names=("link",))
    try:
        worker.start()
        owner = worker._worker
        initial = []
        for repeat in range(3):
            first = worker.advance_sync(targets=[0.2], completed_sim_time=0)
            initial.append(first["positions"].copy())
            for i in range(1, 10):
                worker.advance_sync(targets=[1.0], completed_sim_time=i * 0.01)
            assert abs(worker.snapshot()["positions"][0] - 0.2) > 0.01
            if repeat < 2:
                worker.reset_sync()
                assert worker._worker is owner and owner.is_alive()
                assert worker.snapshot()["steps"] == 0
                assert np.allclose(worker.snapshot()["positions"], [0.2])
                assert np.allclose(worker.snapshot()["velocities"], [0])
        assert all(np.array_equal(a, initial[0]) for a in initial)
    finally:
        worker.close()


@pytest.mark.parametrize("execution", ["thread", "process"])
def test_reset_reloads_authored_base_pose_without_respawning_worker(execution, tmp_path):
    """A newly authored weld pose must reset contacts/controls in the existing PID."""
    mj = pytest.importorskip("mujoco")

    def model_bytes(x):
        model = mj.MjModel.from_xml_string(
            f'<mujoco><option gravity="0 0 0"/><worldbody><body name="base" pos="{x} 0 0">'
            '<body name="link"><joint name="arm" type="hinge"/><geom type="sphere" size="0.1" mass="1"/>'
            '</body></body></worldbody><actuator><position joint="arm" kp="10"/></actuator></mujoco>'
        )
        path = tmp_path / f"model-{x}.mjb"
        mj.mj_saveModel(model, str(path), None)
        return path.read_bytes()

    cfg = CpuMuJoCoConfig(
        execution=execution,
        synchronization="barrier",
        physics_hz=1000,
        sync_timeout_s=5,
        drives=({"pattern": "arm", "kp": 100, "kd": 1, "force_limit": 100},),
    )
    worker = CpuJointWorker(model_bytes(0), cfg, ("arm",), np.array([0.2]), body_names=("link",))
    try:
        worker.start()
        owner = worker._worker
        owner_pid = getattr(owner, "pid", None)
        for x in [0.25, -0.4]:
            worker.advance_sync(targets=[1.0], completed_sim_time=0)
            worker.reset_sync(model_bytes=model_bytes(x))
            snapshot = worker.snapshot()
            assert np.allclose(snapshot["body_poses"][0, :3], [x, 0, 0])
            assert np.allclose(snapshot["positions"], [0.2])
            assert np.allclose(snapshot["velocities"], [0])
            assert snapshot["steps"] == 0 and snapshot["sim_time"] == 0
            assert worker._worker is owner and owner.is_alive()
            assert getattr(owner, "pid", None) == owner_pid
    finally:
        worker.close()
