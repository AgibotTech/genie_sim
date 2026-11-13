# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Rubik platform overrides and score/recording recovery without a GPU renderer."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geniesim_benchmark.backends.mujoco.rubik import config, runner
from geniesim_benchmark.plugins.output_system import checkpoint as common
from geniesim_benchmark.plugins.output_system.dexhand2 import RUBIK_STAGES, RUBIK_TASK, make_evaluation


def test_platform_resume_and_record_overrides_are_supported(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("{}")
    cfg = config.load_config(
        path,
        [
            "--benchmark.resume=true",
            "--benchmark.resume_id=job-42",
            "--benchmark.record=true",
            "--benchmark.infer_host=localhost:8999",
            "--rubik.package_root=" + str(tmp_path),
        ],
    )
    assert cfg.benchmark.resume and cfg.benchmark.record
    assert cfg.benchmark.resume_id == "job-42"
    with pytest.raises(ValueError, match="type/value"):
        config.load_config(path, ["--benchmark.resume=1"])


def test_resume_uses_stable_output_but_nonresume_still_requires_fresh_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config.Config()
    cfg.benchmark.resume = True
    cfg.benchmark.resume_id = "job-42"
    first = runner.create_output_directory(cfg)
    assert runner.create_output_directory(cfg) == first
    cfg.benchmark.resume = False
    with pytest.raises(FileExistsError):
        runner.create_output_directory(cfg)


@pytest.mark.parametrize("resume_at_start", [True, False])
def test_failed_episode_is_retried_without_duplicate_scores_or_recordings(tmp_path, monkeypatch, resume_at_start):
    from geniesim_benchmark.backends.mujoco.rubik import checkpoint
    from geniesim_benchmark.backends.mujoco.rubik.scenes import SceneCatalog

    package = tmp_path / "package"
    package.mkdir()
    (package / "MANIFEST.json").write_text("{}")
    monkeypatch.setattr(
        checkpoint, "load_package", lambda root: (Path(root), {}, common.file_digest(Path(root) / "MANIFEST.json"))
    )
    cfg = config.Config()
    cfg.benchmark.task_name = RUBIK_TASK
    cfg.benchmark.num_instances = 3
    cfg.benchmark.record = True
    cfg.benchmark.resume = resume_at_start
    cfg.benchmark.output_dir = str(tmp_path / "results")
    cfg.rubik.package_root = str(package)
    entry = Path(runner.__file__).parents[3] / "config/g2crsbwuji_dexmanip_rubik_cube_solve.yaml"
    cfg.rubik.scene_archive = config.load_config(
        entry, ["--rubik.package_root=" + str(package), "--benchmark.infer_host=localhost:8999"]
    ).rubik.scene_archive
    cfg.rubik.render = False
    starts, executed, render_directories = [], [], []
    failing = True

    class Environment:
        def __init__(self, settings):
            starts.append(True)
            self.scene_catalog = SceneCatalog(settings.scene_archive)
            self.graphics = self.viewer = None
            self.manifest_hash = "package"
            self.model = SimpleNamespace(body_mass=np.ones(2), nu=54, neq=0, npair=0)
            self.cube = SimpleNamespace(root=1, items=list(range(26)))

        def reset(self, *args):
            pass

        def configure_renderer(self, headless, directory):
            render_directories.append(directory)
            # The real renderer requires a fresh export directory on every retry.
            (directory / "isaac-render").mkdir()

        def action_contract(self):
            return {}

        def checker_contract(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(runner, "NativeEnvironment", Environment)
    monkeypatch.setattr(runner.mujoco, "mj_saveModel", lambda *args: None)
    monkeypatch.setattr(runner, "WebsocketPolicy", lambda *args: SimpleNamespace(close=lambda: None))

    def episode(env, cfg, policy, index, scramble, directory, scene=None, *, evaluations):
        executed.append(index)
        directory.mkdir()
        invalid = failing and index == 1
        native = {
            "episode_index": index,
            "status": "invalid_inference" if invalid else "task_failure",
            "reason": "inference_error" if invalid else "timeout",
            "valid": not invalid,
            "score": None if invalid else 0,
            "stage_points": dict.fromkeys(RUBIK_STAGES, 0),
            "result": {"scores": dict.fromkeys((*RUBIK_STAGES, "E2E"), 0)},
            "scene": scene.receipt(),
        }
        for name in ("result.json", "samples.json", "actions.json"):
            (directory / name).write_text("{}")
        np.savez_compressed(directory / "raw-1000hz.npz", qpos=[index])
        evaluations.append(
            make_evaluation(
                RUBIK_TASK,
                "Solve",
                native,
                stages=RUBIK_STAGES,
                stage_scores=native["result"]["scores"],
                e2e=0,
                count=1,
                step_out=1800,
                steps=1,
                duration=0.01,
                valid=not invalid,
            )
        )
        return native

    monkeypatch.setattr(runner, "run_episode", episode)
    assert runner.run(cfg) == 1
    assert executed == ([0, 1] if resume_at_start else [0, 1, 2])
    output = Path(cfg.benchmark.output_dir)
    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    assert [row["episode_idx"] for row in manifest["completed"]] == ([0] if resume_at_start else [0, 2])
    cfg.benchmark.resume = True
    executed.clear()
    starts_before = len(starts)
    cfg.benchmark.seed += 1
    assert runner.run(cfg) == common.CHECKPOINT_REJECTED_EXIT_CODE
    assert len(starts) == starts_before
    cfg.benchmark.seed -= 1
    failing = False
    assert runner.run(cfg) == 0
    assert executed == ([1, 2] if resume_at_start else [1])
    assert render_directories[0] != render_directories[1]
    details = json.loads((output / "evaluate_ret_00.json").read_text())["details"]
    assert [row["episode_coord"]["instance_id"] for row in details] == [0, 1, 2]
    assert json.loads((output / "summary.json").read_text())["total_episodes"] == 3
    for row in details:
        assert any(artifact["path"].endswith("raw-1000hz.npz") for artifact in row["episode_artifacts"])
    monkeypatch.setattr(
        runner, "NativeEnvironment", lambda *args: pytest.fail("finished retries must not start a simulator")
    )
    assert runner.run(cfg) == 0
    raw = output / next(
        artifact["path"] for artifact in details[0]["episode_artifacts"] if artifact["path"].endswith("raw-1000hz.npz")
    )
    raw.unlink()
    assert runner.run(cfg) == common.CHECKPOINT_REJECTED_EXIT_CODE
