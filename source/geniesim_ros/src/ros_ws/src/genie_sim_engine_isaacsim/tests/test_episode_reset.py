# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from types import SimpleNamespace

import numpy as np
import pytest

from genie_sim_engine_isaacsim.kit.episode_reset import EpisodeSnapshot, validate_reset
from genie_sim_engine_isaacsim.kit.isaac_physx import IsaacPhysXEngine


class View:
    def __init__(self, paths, **values):
        self.prim_paths = paths
        self.values = {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}

    def __getattr__(self, name):
        if name.startswith("get_"):
            return lambda: self.values[name[4:]]
        if name.startswith("set_"):

            def write(value, indices):
                self.values[name[4:]] = np.asarray(value).copy()

            return write
        raise AttributeError(name)


def snapshot():
    robot = View(
        ["/Robot"], root_transforms=[[1, 2, 3, 0, 0, 0, 1]], dof_positions=[[9, 9]], dof_position_targets=[[9, 9]]
    )
    door = View(["/Door"], root_transforms=[[4, 5, 6, 0, 0, 0, 1]], dof_positions=[[0.2]], dof_position_targets=[[0.3]])
    inventory = SimpleNamespace(prim_paths=["/Robot", "/Door"], link_paths=[["/Robot/Link"], ["/Door/Link"]])
    bodies = View(["/Robot/Link", "/Door/Link", "/Prop"])
    prop = View(["/Prop"], transforms=[[7, 8, 9, 0, 0, 0, 1]])
    sim = SimpleNamespace(
        create_articulation_view=lambda path: (
            inventory if isinstance(path, list) else {"/*": inventory, "/Robot": robot, "/Door": door}[path]
        ),
        create_rigid_body_view=lambda path: bodies if path == "/*" else prop,
    )
    state = EpisodeSnapshot(
        sim,
        lambda a: a,
        lambda a: a,
        robot,
        np.array([0.5, -0.5]),
        ["/Robot", "/Door"],
        ["/Robot/Link", "/Door/Link", "/Prop"],
    )
    return state, robot, door, prop


def test_two_resets_restore_robot_background_and_free_bodies_from_owned_arrays():
    state, robot, door, prop = snapshot()
    for _ in range(2):
        for view in (robot, door, prop):
            for value in view.values.values():
                value.fill(42)
        state.restore()
        np.testing.assert_allclose(robot.values["dof_positions"], [[0.5, -0.5]])
        np.testing.assert_allclose(robot.values["dof_position_targets"], [[0.5, -0.5]])
        np.testing.assert_allclose(robot.values["root_transforms"], [[1, 2, 3, 0, 0, 0, 1]])
        np.testing.assert_allclose(door.values["dof_positions"], [[0.2]])
        np.testing.assert_allclose(door.values["dof_position_targets"], [[0.3]])
        np.testing.assert_allclose(prop.values["transforms"], [[7, 8, 9, 0, 0, 0, 1]])
        for view in (robot, door):
            for key in ("dof_velocities", "dof_velocity_targets", "dof_actuation_forces", "root_velocities"):
                assert not np.any(view.values[key])
        assert not np.any(prop.values["velocities"])


def test_reset_epoch_advances_only_after_successful_writes():
    engine = IsaacPhysXEngine.__new__(IsaacPhysXEngine)
    engine._reset_epoch = 0
    state, *_ = snapshot()
    engine._stage_obj = SimpleNamespace(_episode_snapshot=state)
    engine.episode_reset()
    engine.episode_reset([0], seeds={0: -1}, instances={0: -1})
    assert engine.reset_epoch == 2
    engine.episode_reset([])
    assert engine.reset_epoch == 2

    def fail():
        raise RuntimeError("write failed")

    engine._stage_obj._episode_snapshot = SimpleNamespace(restore=fail)
    with pytest.raises(RuntimeError, match="write failed"):
        engine.episode_reset()
    assert engine.reset_epoch == 2


@pytest.mark.parametrize(
    "args,error",
    [
        (([1], None, None), ValueError),
        (([0], {0: 7}, None), NotImplementedError),
        (([0], None, {0: 2}), NotImplementedError),
    ],
)
def test_unsupported_reset_requests_fail_before_writes(args, error):
    with pytest.raises(error):
        validate_reset(*args)


def test_newton_clears_free_joint_velocity_and_commits_both_buffers_before_epoch(monkeypatch):
    import sys
    from genie_sim_engine_isaacsim.kit.isaac_newton import IsaacNewtonEngine

    events = []

    def array(label):
        return SimpleNamespace(zero_=lambda: events.append(label))

    first = SimpleNamespace(
        joint_q="q",
        joint_qd=array("joint velocities"),
        body_qd=array("body velocities"),
        clear_forces=lambda: events.append("forces"),
    )
    second = SimpleNamespace(assign=lambda state: events.append("copy both"))
    ns = SimpleNamespace(
        model=SimpleNamespace(device="cuda:0"),
        state_0=first,
        state_1=second,
        update_fabric_attrs=lambda: events.append("fabric"),
    )
    engine = IsaacNewtonEngine.__new__(IsaacNewtonEngine)
    engine._reset_epoch = 0
    engine._get_newton_stage = lambda: ns
    engine._stage_obj = SimpleNamespace(_episode_snapshot=SimpleNamespace(restore=lambda: events.append("restore")))
    monkeypatch.setitem(sys.modules, "newton", SimpleNamespace(eval_fk=lambda *args: events.append("fk")))

    def sync(device):
        assert device == "cuda:0"
        assert engine.reset_epoch == 0
        events.append("sync")

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(synchronize_device=sync))
    engine.episode_reset()
    assert events == ["restore", "forces", "joint velocities", "body velocities", "fk", "copy both", "fabric", "sync"]
    assert engine.reset_epoch == 1
    assert not engine._cmd_pending
