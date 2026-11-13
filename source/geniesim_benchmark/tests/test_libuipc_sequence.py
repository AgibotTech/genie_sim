# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Sequence YAML preserves physical settings and gives every episode fresh outputs."""

import copy
from pathlib import Path
import pytest
import yaml
from geniesim_benchmark.libuipc.config import materialize_sequence
from geniesim_benchmark.libuipc import config as config_module


def test_all_layout_plans_are_preloaded_once_and_geometry_changes_rejected(tmp_path, monkeypatch):
    cfg = {
        "benchmark": {"num_instances": 2, "num_episode": 2},
        "generalization": {"material": {"selection": "enumerate"}},
    }
    calls = []

    def scene(config, directory, *, assets_root, episode_index, episode_id):
        calls.append(episode_index)
        value = {
            "robot": {"name": "same"},
            "libuipc": {
                "solver": "ipc",
                "state_recording": {"enabled": True, "output_dir": str(directory / "physical_state")},
            },
            "generalization": {
                "material": {
                    "episode_plan": {"episode_id": episode_id, "episode_index": episode_index},
                    "receipt_path": str(directory / "receipt.json"),
                }
            },
        }
        p = directory / "scene.yaml"
        p.write_text(yaml.safe_dump(value))
        return p

    monkeypatch.setattr(config_module, "materialize_scene", scene)
    root = tmp_path / "run"
    root.mkdir()
    path, episodes = materialize_sequence(cfg, root, assets_root=tmp_path)
    document = yaml.safe_load(path.read_text())
    entries = document["libuipc"]["episode_sequence"]
    assert calls == [0, 0, 1, 1]
    assert len(entries) == 4 and len({e["state_recording"]["output_dir"] for e in entries}) == 4
    assert len({e["material"]["episode_plan"]["episode_id"] for e in entries}) == 4
    assert [(e["instance_id"], e["attempt_index"]) for e in episodes] == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def changed(*args, **kwargs):
        p = scene(*args, **kwargs)
        d = yaml.safe_load(p.read_text())
        d["robot"]["name"] = str(len(calls))
        p.write_text(yaml.safe_dump(d))
        return p

    monkeypatch.setattr(config_module, "materialize_scene", changed)
    root = tmp_path / "bad"
    root.mkdir()
    with pytest.raises(ValueError, match="identical physical geometry"):
        materialize_sequence(cfg, root, assets_root=tmp_path)


def test_cli_creates_one_session_for_every_configured_episode(tmp_path, monkeypatch):
    from geniesim_benchmark.libuipc import runner
    from genie_sim_client import IPCClientConfig
    from types import SimpleNamespace

    root = tmp_path / "repo"
    (root / "devel").mkdir(parents=True)
    (root / "devel/setup.bash").write_text("")
    config = {
        "repo_root": str(root),
        "benchmark": {"num_instances": 2, "num_episode": 2},
        "libuipc": {"fps": 30, "output_dir": "output"},
    }
    monkeypatch.setattr(runner, "load_config", lambda *a, **kw: config)
    monkeypatch.setattr(runner, "validate_settings", lambda c: None)
    config["libuipc"].update(evaluation_scene="scene", launcher="launcher")
    monkeypatch.setattr(IPCClientConfig, "from_files", lambda *a: SimpleNamespace(physics_hz=60))
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.episodes = [None] * 4
            calls.append("construct")

        def __enter__(self):
            calls.append("start")
            return self

        def __exit__(self, *args):
            calls.append("close")

    monkeypatch.setattr(runner, "LibuipcRunSession", Session)
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *args: {"test": "sequence"})

    def episode(session, index, *, evaluations):
        calls.append(index)
        result = {"execution_status": "completed", "scoring_status": "scored"}
        evaluations.append({"execution": result})
        return result

    monkeypatch.setattr(runner, "run_one_episode", episode)
    monkeypatch.setattr(runner, "summarize", lambda results: {"count": len(results)})
    monkeypatch.setattr(runner, "write_summary", lambda directory, rows: runner.write_json(
        directory / "evaluate_ret_00.json", {"details": rows}))
    assert runner.main(["--config", str(tmp_path / "task.yaml"), "--infer-host", "example:9000"]) == 0
    assert calls == ["construct", "start", 0, 1, 2, 3, "close"]


def test_gpu_probe_pose_comparison_accepts_antipodes_but_rejects_physical_motion():
    import importlib.util
    import numpy as np

    script = Path(__file__).with_name("native_libuipc_sequence_smoke.py")
    spec = importlib.util.spec_from_file_location("sequence_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = {"body_q": np.array([[1.0, 2, 3, 0, 0, 0, 1]]), "particle_q": np.array([[0.0, 0, 1]])}
    b = {k: v.copy() for k, v in a.items()}
    b["body_q"][:, 3:] *= -1
    assert module.state_differences(a, b)["body_rotation_max_error_rad"] == 0
    b["body_q"][0, 3:] = [0, 0, np.sin(0.01), np.cos(0.01)]
    with pytest.raises(AssertionError, match="rotation changed"):
        module.state_differences(a, b)
    b["body_q"][:] = a["body_q"]
    b["particle_q"][0, 2] += 0.01
    with pytest.raises(AssertionError, match="particle_q"):
        module.state_differences(a, b)


def test_skipping_scored_layouts_keeps_physics_and_real_renderer_cursors_aligned(tmp_path, monkeypatch):
    """The third layout must capture after zero-action reuse of layouts zero/one."""
    import importlib.util
    import json
    from types import SimpleNamespace
    from geniesim_benchmark.libuipc import runner, materials

    render_path = (
        Path(__file__).parents[2] / "geniesim_ros/src/ros_ws/src/genie_sim_render"
        "/scripts/_material_generalization.py"
    )
    spec = importlib.util.spec_from_file_location("reuse_material_episodes", render_path)
    renderer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer_module)
    entries = [{"episode_plan": {"episode_index": i}, "receipt_path": str(i)} for i in range(3)]
    renderer = renderer_module.MaterialEpisodes({"episode_sequence": entries})
    calls = []

    class Client:
        index = 0
        phase = "ready"

        @property
        def identity(self):
            return {"epoch": self.index}

        def reset(self, index):
            assert self.phase == "finished" and index == self.index + 1
            calls.append(("reset", index))
            self.index, self.phase = index, "ready"
            return self.identity

        def observe(self, **kwargs):
            assert self.phase == "ready"  # No physics step or policy action during reuse.
            calls.append(("observe", self.index))
            snapshot = {"episode_index": self.index, "epoch": self.index}
            renderer.select(snapshot)  # Actual renderer contract, including skip rejection.
            return {"snapshot": snapshot, "epoch": self.index}

        def begin_episode(self):
            assert self.phase == "ready"
            calls.append(("begin", self.index))
            self.phase = "running"

        def finish_episode(self):
            assert self.phase == "running"
            calls.append(("finish", self.index))
            self.phase = "finished"
            return self.identity

    session = runner.LibuipcRunSession.__new__(runner.LibuipcRunSession)
    session.client, session.config = Client(), {"libuipc": {"wait_timeout": 5}}
    session.assets, session.manifest = {}, tmp_path / "manifest.json"
    session.episodes = []
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "material_generalization.json").write_text(json.dumps(entries[index]["episode_plan"]))
        session.episodes.append({"directory": directory})
    monkeypatch.setattr(runner, "validate_identity", lambda identity: None)
    monkeypatch.setattr(runner, "validate_task_snapshot", lambda packet, assets: None)

    def validate(plan, receipt_path, packet, **kwargs):
        assert plan["episode_index"] == packet["snapshot"]["episode_index"]
        return {"episode_index": plan["episode_index"]}

    monkeypatch.setattr(materials, "validate_material_capture", validate)
    for index in [0, 1]:
        session.skip_completed_episode(index)
        assert (session.episodes[index]["directory"] / "reuse-material-capture.json").exists()
        assert not (session.episodes[index]["directory"] / "result.json").exists()
    session.client.reset(2)
    assert session.client.observe()["snapshot"]["episode_index"] == 2
    assert calls == [
        ("observe", 0),
        ("begin", 0),
        ("finish", 0),
        ("reset", 1),
        ("observe", 1),
        ("begin", 1),
        ("finish", 1),
        ("reset", 2),
        ("observe", 2),
    ]


def test_preloaded_base_xy_changes_remain_distinct_without_relaxing_geometry(tmp_path, monkeypatch):
    cfg = {"benchmark": {"num_instances": 2, "num_episode": 1}}

    def materialize(config, directory, *, assets_root, episode_index, episode_id):
        value = dict(
            robot=dict(name="same", init_base_pose=dict(x=episode_index * 0.25, y=0.0, z=0.0, theta=0.0)),
            libuipc=dict(solver="ipc", state_recording=dict(enabled=True, output_dir=str(directory / "states"))),
            generalization=dict(
                material=dict(episode_plan=dict(episode_id=episode_id), receipt_path=str(directory / "receipt.json"))
            ),
        )
        p = directory / "scene.yaml"
        p.write_text(yaml.safe_dump(value))
        return p

    monkeypatch.setattr(config_module, "materialize_scene", materialize)
    root = tmp_path / "run"
    root.mkdir()
    path, _ = materialize_sequence(cfg, root, assets_root=tmp_path)
    scene = yaml.safe_load(path.read_text())
    plans = scene["libuipc"]["episode_sequence"]
    assert [p["robot_base_pose"]["x"] for p in plans] == [0.0, 0.25]
    assert scene["robot"]["init_base_pose"]["x"] == 0
