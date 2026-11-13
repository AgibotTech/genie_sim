# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Door approach tolerance must preserve its boundaries and traversal goals."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.plugins.ader.action import action_parsing
from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask
from geniesim_benchmark.plugins.output_system.eval_utils import TaskEvaluation

CONFIG = Path(action_parsing.__file__).resolve().parents[3] / "benchmark" / "config"


SCENARIOS = sorted(
    (p.parent.parent.name, int(p.parent.name)) for p in (CONFIG / "llm_task").glob("open_the_door*/*/problems.json")
)


@pytest.mark.parametrize("task,instance", SCENARIOS)
def test_door_approach_tolerance_preserves_crossing(monkeypatch, task, instance):
    monkeypatch.setattr(action_parsing, "system_utils", SimpleNamespace(benchmark_conf_path=lambda: str(CONFIG)))
    env = AderEnv(SimpleNamespace(), AderParams())
    env.init_task_config = {"sub_task_name": task, "scene": {"scene_instance_id": instance}}
    env.current_step = 0
    env.policy_dispatch_count = 0
    env.task = AderTask(env, CONFIG)
    env.task.current_episode_id = 0
    env.task.instructions = [""]
    env.task.do_action_parsing(env)
    # Pi05 batch 20261111T065505Z: the first stage timed out at this approach.
    state = {"position": (-0.059, -1.354, 0), "yaw": 0.0, "door": 0.0}

    def pose(path):
        half = np.deg2rad(state["yaw"]) / 2
        return state["position"], (np.cos(half), 0, 0, np.sin(half))

    def advance():
        env.current_step += 1
        env.policy_dispatch_count += 1
        env.execute_action.update(0.01)

    def scores():
        evaluation = TaskEvaluation(sub_task_name=env.task.problem_name, score_layout_name=env.task.score_layout_name)
        evaluation.update_progress(env.task.task_progress)
        evaluation.summarize_scores()
        return evaluation.result["scores"]

    env.api_core.get_obj_world_pose = pose
    env.api_core.get_obj_joint = lambda path: {"joint_positions": [state["door"]]}
    env.execute_action.start()
    approach = next(row["acion_obj"] for row in env.task.task_progress if row["class_name"] == "ChassisAtTarget")
    assert not approach.is_finished()  # Spawn must not satisfy the approach.
    state["position"] = (0.27, -0.967, 0)
    advance()
    assert not approach.is_finished()  # Just outside the current +/-0.95 m range.
    assert approach.x_thresh == 0.95
    assert not approach.check_position(0.27, -0.9537)[0]
    assert not approach.check_position(2.19, -0.9537)[0]
    state["position"] = (0.332, -0.967, 0)
    advance()
    assert approach.is_finished()
    assert not env.has_done
    assert [s["score"] for s in scores()["STEPS"]] == [1, 0, 0, 0]

    state["door"] = 0.7
    for _ in range(3):
        advance()
    assert [s["score"] for s in scores()["STEPS"]] == [1, 1, 0, 0]
    state["position"] = (1.9, -0.9537, 0)
    advance()
    assert scores()["E2E"] == 0  # Turning is still required.
    state["yaw"] = 180
    for _ in range(3):
        advance()
    state["door"] = 0.0
    for _ in range(3):
        advance()
    assert scores()["E2E"] == 1
