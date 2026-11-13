# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Chef retry keeps each sealed episode, including multiple attempts per layout."""

import json

import pytest
import yaml

from geniesim_benchmark.newton.checkpoint import run_checkpointed
from geniesim_benchmark.newton.layout_batch import LayoutRun
from geniesim_benchmark.plugins.output_system.checkpoint import CheckpointRejected, atomic_write_json
from geniesim_benchmark.plugins.output_system.dexhand2 import WOK_STAGES, WOK_TASK, make_evaluation, write_summary


def test_normal_entry_accepts_platform_flags_and_preserves_effective_settings(tmp_path, monkeypatch):
    from pathlib import Path
    from geniesim_benchmark.newton import runner, instances

    entry = Path(runner.__file__).parents[1] / "config/g2crsbwuji_dexmanip_wok_flip.yaml"

    def run(options, path, config):
        assert config["headless"] is True
        assert config["benchmark"]["record"] is True
        assert config["benchmark"]["resume"] is True
        assert config["episodes"] == 2
        assert options.output_dir == tmp_path
        return 0

    monkeypatch.setattr(instances, "run_instances", run)
    assert (
        runner.main(
            [
                "--config",
                str(entry),
                "--app.headless=true",
                "--benchmark.record=true",
                "--benchmark.resume=true",
                "--benchmark.num_episode=2",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_resume_retries_only_uncommitted_episodes_and_keeps_score_order(tmp_path):
    config = {
        "runtime": "newton",
        "task_name": WOK_TASK,
        "episodes": 2,
        "headless": True,
        "benchmark": {"num_instances": 2, "resume": True, "seed": 1},
    }
    plans = []
    for i in range(2):
        scene = tmp_path / f"scene{i}.yaml"
        scene.write_text(yaml.safe_dump({"newton": {"scene_plugin": "chef.py"}}))
        (tmp_path / "chef.py").write_text("# native plugin")
        evaluation = tmp_path / f"eval{i}.yaml"
        evaluation.write_text(yaml.safe_dump(config))
        launcher = tmp_path / f"launcher{i}.yaml"
        launcher.write_text("launcher: original")
        run = tmp_path / "instances" / str(i)
        plans.append(
            LayoutRun(
                i,
                scene,
                evaluation,
                launcher,
                tmp_path,
                tmp_path / "manifest",
                run,
                run / "launcher.yaml",
                run / "evaluation.yaml",
                [f"launcher_config:={run / 'launcher.yaml'}", "headless:=false"],
                [
                    "python",
                    "--config",
                    str(run / "evaluation.yaml"),
                    "--output-dir",
                    str(run),
                    "--infer-host",
                    "model:8999",
                ],
            )
        )
    calls, fail = [], True

    def batch(runs, *, completed, **kwargs):
        for run in runs:
            doc = yaml.safe_load(run.evaluation.read_text())
            index = doc["benchmark"]["episode_offset"]
            assert "headless:=false" in run.engine_command
            assert doc["headless"] is True
            assert doc["episodes"] == 1
            calls.append(index)
            run.run_dir.mkdir(parents=True)
            if fail and index == 1:
                break
            native = {
                "episode_idx": index,
                "milestones": {stage: False for stage in WOK_STAGES},
                "task_success": False,
                "issued_steps": 10,
                "evaluation": {"session_id": str(index)},
            }
            atomic_write_json(run.run_dir / "results.json", [native])
            detail = make_evaluation(
                WOK_TASK,
                "Pan tossing",
                native,
                stages=WOK_STAGES,
                stage_scores=native["milestones"],
                e2e=0,
                count=1,
                step_out=600,
                steps=10,
                duration=1,
                valid=True,
            )
            write_summary(run.run_dir, [detail])
            completed(run, {"status": "evaluated"})

    output = tmp_path / "results"
    assert run_checkpointed(plans, output, config, "model:8999", batch) == 1
    assert calls == [0, 1]
    fail = False
    calls.clear()
    assert run_checkpointed(plans, output, config, "model:8999", batch) == 0
    assert calls == [1, 2, 3]
    details = json.loads((output / "evaluate_ret_00.json").read_text())["details"]
    assert [row["native_result"]["episode_idx"] for row in details] == [0, 1, 2, 3]
    assert [row["episode_coord"] for row in details] == [
        {"instance_id": i, "file_idx": 0, "episode_id": j} for i in range(2) for j in range(2)
    ]
    calls.clear()
    assert run_checkpointed(plans, output, config, "model:8999", batch) == 0
    assert not calls
    config["benchmark"]["seed"] = 2
    report = (output / "evaluate_ret_00.json").read_bytes()
    with pytest.raises(CheckpointRejected):
        run_checkpointed(plans, output, config, "model:8999", batch)
    assert (output / "evaluate_ret_00.json").read_bytes() == report
