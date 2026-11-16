# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Insertion evidence belongs to the episode, even before its scoring stage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.plugins.ader.action import action_parsing
from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask

CONFIG = Path(action_parsing.__file__).resolve().parents[3] / "benchmark/config"
TASKS = ("insert_ethernet_cable", "plug_charging_cable_into_phone", "put_the_cap_on_the_pen_new")
SCENARIOS = sorted(path for task in TASKS for path in (CONFIG / "llm_task" / task).glob("*/problems.json"))


@pytest.fixture(params=SCENARIOS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def episode(request, monkeypatch):
    path = request.param
    monkeypatch.setattr(action_parsing, "system_utils", SimpleNamespace(benchmark_conf_path=lambda: str(CONFIG)))
    state = SimpleNamespace(attached=True, z=0.7, reads=0)

    def attribute(path, name, default=None):
        state.reads += 1
        # Pen assets attach on /entity, cables on /entity/body.
        suffix = "/entity/PenAttachJoint" if "PenAttachJoint" in path else "/entity/body/" + path.rsplit("/", 1)[1]
        return state.attached if path.endswith(suffix) else default

    def pose(path):
        matrix = np.eye(4)
        matrix[2, 3] = state.z
        return matrix

    api = SimpleNamespace(
        get_prim_attribute=attribute,
        get_obj_world_pose_matrix=pose,
        get_obj_world_pose=lambda path: (np.array([0.0, 0.0, state.z]), np.array([1.0, 0.0, 0.0, 0.0])),
        get_obj_aabb=lambda path: [0, 0, 0.6, 0.1, 0.1, 0.7],
    )
    env = AderEnv(api, AderParams())
    env.init_task_config = {"sub_task_name": path.parent.parent.name, "scene": {"scene_instance_id": int(path.parent.name)}}
    env.current_step = env.policy_dispatch_count = 0
    env.task = AderTask(env, CONFIG)
    env.task.current_episode_id = 0
    env.task.instructions = [""]
    env.task.do_action_parsing(env)
    checker = next(row["acion_obj"] for row in env.task.task_progress if row["class_name"] == "PlugInserted")
    # Parsing happens before the physical reset; it must not sample that scene.
    assert state.reads == 0
    return env, state, checker


def advance(env, state, *, lift=True):
    env.current_step += 30
    env.policy_dispatch_count += 1
    if lift:
        state.z += 0.1
    env.action_update()


def enter_insertion_stage(env, state, checker):
    for _ in range(5):
        if not checker.is_init():
            break
        advance(env, state)
    assert not checker.is_init()


def test_insertion_before_its_stage_receives_credit(episode):
    env, state, checker = episode
    state.attached = False  # Physical reset completed after parsing.
    env.do_eval_action()
    state.attached = True  # Insert during the preceding pickup stage.
    assert checker.is_init()
    assert "SCORE" not in checker.progress_info  # Observation must not score early.
    enter_insertion_stage(env, state, checker)
    assert checker.is_finished()
    assert checker.progress_info["SCORE"] == 1


def test_normal_insertion_after_stage_start_still_passes(episode):
    env, state, checker = episode
    state.attached = False
    env.do_eval_action()
    enter_insertion_stage(env, state, checker)
    assert not checker.is_finished()
    state.attached = True
    advance(env, state)
    assert checker.progress_info["SCORE"] == 1


def test_stale_enabled_joint_never_earns_credit(episode):
    env, state, checker = episode
    env.do_eval_action()  # A surviving enabled joint was never seen detached.
    enter_insertion_stage(env, state, checker)
    for _ in range(3):
        advance(env, state)
    assert not checker.is_finished()
    assert checker.progress_info.get("SCORE", 0) == 0


def test_delayed_detach_is_observed_before_stage_start(episode):
    env, state, checker = episode
    env.do_eval_action()
    state.attached = False  # Attractor reset arrives after evaluation starts.
    advance(env, state, lift=False)
    assert checker.is_init()
    state.attached = True
    enter_insertion_stage(env, state, checker)
    assert checker.progress_info["SCORE"] == 1


def test_baseline_does_not_leak_into_next_episode(episode):
    env, state, checker = episode
    state.attached = False
    env.do_eval_action()
    env.cancel_action("eval")
    env.task.reset(env)
    replacement = next(row["acion_obj"] for row in env.task.task_progress if row["class_name"] == "PlugInserted")
    assert replacement is not checker
    state.attached = True
    env.do_eval_action()
    enter_insertion_stage(env, state, replacement)
    assert not replacement.is_finished()
    assert replacement.progress_info.get("SCORE", 0) == 0
