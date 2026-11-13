# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from types import SimpleNamespace

import pytest
import yaml

from geniesim_benchmark.newton.shared_layouts import shared_scene


def plans(tmp_path):
    result = []
    for i, xy in [(5, [-0.0482, 0.0181]), (6, [0.025, -0.015])]:
        scene = tmp_path / f"scene{i}.yaml"
        scene.write_text(
            yaml.safe_dump(
                {
                    "robot": {
                        "init_base_pose": {"x": xy[0], "y": xy[1], "z": 0.04, "theta": 0},
                        "init_joint_pos": {"arm": 2},
                    },
                    "newton": {"scene_plugin": "chef.py"},
                }
            )
        )
        launcher = tmp_path / f"launcher{i}.yaml"
        launcher.write_text(
            yaml.safe_dump(
                {"genie_sim_engine": {"ros__parameters": {"isaac_render_log_path": str(i), "physics_solver": "mjxpbd"}}}
            )
        )
        evaluation = tmp_path / f"eval{i}.yaml"
        evaluation.write_text(
            yaml.safe_dump(
                {
                    "launcher": str(launcher),
                    "episodes": 1,
                    "max_steps": 600,
                    "benchmark": {"instance_id": i, "num_instances": 1},
                }
            )
        )
        result.append(SimpleNamespace(scene=scene, launcher=launcher, evaluation=evaluation, layout_id=i))
    return result


def test_layout_table_offsets_are_relative_to_build_and_never_resampled(tmp_path):
    source = plans(tmp_path)
    result = shared_scene(source)
    switch = result["instance_switch"]
    assert switch["build_instance"] == 5
    assert switch["group_instances"] == [5, 6]
    assert switch["robot_base_xy"]["5"] == [0, 0]
    assert switch["robot_base_xy"]["6"] == pytest.approx([0.0732, -0.0331])
    assert result["robot"]["init_base_pose"]["x"] == -0.0482
    assert "instance_switch" not in yaml.safe_load(source[0].scene.read_text())


@pytest.mark.parametrize("change", ["topology", "height", "scoring"])
def test_reuse_rejects_contract_changes_before_launch(tmp_path, change):
    source = plans(tmp_path)
    p = source[1].evaluation if change == "scoring" else source[1].scene
    doc = yaml.safe_load(p.read_text())
    if change == "scoring":
        doc["max_steps"] = 601
    elif change == "height":
        doc["robot"]["init_base_pose"]["z"] = 0.05
    else:
        doc["newton"]["food_count"] = 12
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
        shared_scene(source)


def test_published_base_readback_rejects_reset_that_stayed_on_previous_layout():
    import numpy as np
    from geniesim_benchmark.newton.runner import verify_layout_reset

    scene = {
        "robot": {"init_base_pose": {"x": -0.0482, "y": 0.0181, "z": 0.04}},
        "instance_switch": {"robot_base_xy": {"5": [0, 0], "6": [0.0732, -0.0331]}},
    }
    snapshot = SimpleNamespace(
        schema=SimpleNamespace(body_paths=["/genie/Geometry/base_link"]),
        body_poses=np.array([[[-0.0482, 0.0181, 0.04, 1, 0, 0, 0]]]),
        reset_epoch=1,
    )
    assert verify_layout_reset(snapshot, scene, 5)["instance_id"] == 5
    with pytest.raises(RuntimeError, match="base mismatch"):
        verify_layout_reset(snapshot, scene, 6)
    snapshot.body_poses[0, 0, :3] = [0.025, -0.015, 0.04]
    assert verify_layout_reset(snapshot, scene, 6)["base_xyz"] == pytest.approx([0.025, -0.015, 0.04])


def test_attached_evaluation_reuses_paused_client_across_layouts_and_native_results(tmp_path, monkeypatch):
    import numpy as np
    from geniesim_benchmark.newton import runner

    class Client:
        epoch = 0
        paused = True
        instances = []

        def resume(self):
            self.paused = False

        def reset(self, instances):
            assert not self.paused, "reset must resume the existing paused session"
            self.instances.append(instances[0])
            self.epoch += 1
            return SimpleNamespace(reset_epoch=self.epoch)

    client = Client()
    resets = []
    policy = SimpleNamespace(reset=lambda: resets.append(True))
    monkeypatch.setattr(
        runner,
        "verify_layout_reset",
        lambda snapshot, scene, instance: {"instance_id": instance, "reset_epoch": snapshot.reset_epoch},
    )

    def episode(c, images, policy, **kw):
        assert c is client
        c.paused = True
        return {
            "episode_idx": kw["episode_idx"],
            "issued_steps": 600,
            "policy_dispatch_count": 12,
            "task_success": False,
            "score": 0,
            "max_score": 2,
            "milestones": {"stable_lift": False, "food_toss": False},
            "evaluation": {"session_id": "one-session", "reset_epoch": c.epoch},
        }

    monkeypatch.setattr(runner, "run_episode", episode)
    epochs = []
    for instance in [5, 6]:
        output = tmp_path / str(instance)
        output.mkdir()
        config = {
            "episodes": 2,
            "max_steps": 600,
            "instruction": "flip",
            "task_name": "wok_flip",
            "observation_timeout_s": 180,
            "benchmark": {"instance_id": instance},
        }
        result = runner.evaluate_attached(
            client, None, policy, config=config, scene={}, output_dir=output, initial_positions={}
        )
        assert [r["instance_id"] for r in result] == [instance, instance]
        epochs.extend(r["evaluation"]["reset_epoch"] for r in result)
        import json

        native = json.loads((output / "evaluate_ret_00.json").read_text())
        assert len(native["details"]) == 2
        native = json.loads((output / "results.json").read_text())
        assert len(native) == 2
        assert [row["score"] for row in native] == [0, 0]
        assert [row["evaluation"]["reset_epoch"] for row in native] == [client.epoch - 1, client.epoch]
    assert client.instances == [5, 5, 6, 6]
    assert epochs == [1, 2, 3, 4]
    assert len(resets) == 4


def test_repeated_episodes_declare_each_physical_layout_once(tmp_path):
    source = plans(tmp_path)
    repeated = [source[0], source[0], source[1]]
    scene = shared_scene(repeated)
    assert scene["instance_switch"]["group_instances"] == [5, 6]
    assert set(scene["instance_switch"]["instance_body_poses"]) == {"5", "6"}
    assert scene["instance_switch"]["robot_base_xy"]["6"] == pytest.approx([0.0732, -0.0331])
