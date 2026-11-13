# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Replay shelf-edge placement evidence against authored conveyor checkers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.plugins.ader.action import action_parsing
from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams, AderTask


CONFIG = Path(action_parsing.__file__).resolve().parents[3] / "benchmark" / "config"
PROBLEMS = sorted(CONFIG.glob("**/llm_task/match_and_pick_from_conveyor/*/problems.json"))


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda path: str(path.relative_to(CONFIG)))
def test_shelf_edge_placement_passes_but_off_shelf_fails(monkeypatch, problem):
    config_root = problem.parents[3]
    monkeypatch.setattr(action_parsing, "system_utils", SimpleNamespace(benchmark_conf_path=lambda: str(config_root)))
    env = AderEnv(SimpleNamespace(), AderParams())
    env.init_task_config = {
        "sub_task_name": "match_and_pick_from_conveyor",
        "scene": {"scene_instance_id": int(problem.parent.name)},
    }
    env.current_step = env.policy_dispatch_count = 0
    env.task = AderTask(env, config_root)
    env.task.current_episode_id = 0
    env.task.instructions = [""]
    env.task.do_action_parsing(env)
    checker = next(row["acion_obj"] for row in env.task.task_progress if row["class_name"] == "InBBox")
    carton = "benchmark_carton_022_03"
    env.update_place_holder("placeholder_str1", carton)
    pose = np.eye(4)
    env.api_core.get_obj_world_pose_matrix = lambda path: pose

    for outside in [(-1.7188, 0.95, 1.293), (-1.7188, 1.0755, 0.8), (-2.8, 1.0755, 1.293)]:
        pose[:3, 3] = outside
        assert not checker._check_candidate(carton)

    # Pi05 episode 1: the carton origin settled 15 mm past the former front bound.
    pose[:3, 3] = (-1.7188, 1.0755, 1.293)
    checker.start()
    assert not checker.is_finished()
    checker.update(0.01)
    assert checker.is_finished()
