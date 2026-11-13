# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Conveyor layouts share a fixed robot start and one episode per instance."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from geniesim_benchmark.utils.instance_base_pose import instance_base_pose

PACKAGE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"
CONFIG = PACKAGE / "benchmark/config"
TASK = "match_and_pick_from_conveyor_g2_90d"


def test_conveyor_layouts_use_one_fixed_base_without_extra_episodes():
    public = json.loads((CONFIG / "eval_tasks" / (TASK + ".json")).read_text())
    command = yaml.safe_load((PACKAGE / "config/g2crsb90d_wbc_match_and_pick_from_conveyor.yaml").read_text())
    assert command["benchmark"]["num_instances"] in (10, 20)
    assert command["benchmark"]["num_episode"] == 1
    fixed_pose = public["robot"]["robot_init_pose"]
    for path in CONFIG.rglob(TASK + ".json"):
        task = json.loads(path.read_text())
        base = task["generalization"]["init_base"]
        assert base["enable"] is False, str(path)
        assert base["num"] == 0
        assert "poses_by_instance" not in base
        assert "pose_seed" not in base
        assert task["robot"]["robot_init_pose"] == fixed_pose
        for i in range(20):
            selected = deepcopy(task)
            selected["scene"]["scene_instance_id"] = i
            assert instance_base_pose(selected) is None


@pytest.mark.skipif(importlib.util.find_spec("geniesim_benchmark.hidden") is None, reason="Hidden suite lives on _hide")
def test_generated_hidden_layouts_keep_the_fixed_base_and_replay_exactly(tmp_path):
    from geniesim_benchmark.hidden.isaac import Foregrounds

    def generate(dest, seed, replay=None):
        dest.mkdir()
        # Conveyor foreground translation does not load the asset library.
        source = Foregrounds(CONFIG, "match_and_pick_from_conveyor", TASK, "wbc", tmp_path)
        params = source.generate(dest, np.random.default_rng(seed), replay)
        return params, json.loads((dest / "eval_tasks" / (TASK + ".json")).read_text())

    params, task = generate(tmp_path / "generated", 0)
    assert len(params) == 5
    base = task["generalization"]["init_base"]
    assert base["enable"] is False
    assert base["num"] == 0
    assert "poses_by_instance" not in base
    public = json.loads((CONFIG / "eval_tasks" / (TASK + ".json")).read_text())
    assert task["robot"]["robot_init_pose"] == public["robot"]["robot_init_pose"]
    for i, row in enumerate(params):
        assert "base_pose" not in row
        selected = deepcopy(task)
        selected["scene"]["scene_instance_id"] = i
        assert instance_base_pose(selected) is None
    replay_params, replay_task = generate(tmp_path / "replayed", 1234, params)
    assert replay_params == params
    assert replay_task == task
