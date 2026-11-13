# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Commit chef episodes independently of the managed engine lifecycle."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import uuid

import yaml

from geniesim_benchmark.plugins.output_system.checkpoint import (
    CheckpointRejected,
    EvaluationCheckpoint,
    atomic_write_json,
    digest,
    file_digest,
)
from geniesim_benchmark.plugins.output_system.dexhand2 import read_summary_details, write_summary


def stable_config(config):
    result = deepcopy(config)
    result.pop("launcher", None)
    for key in ("resume", "output_dir", "record_video_dir"):
        result.get("benchmark", {}).pop(key, None)
    return result


def run_checkpointed(plans, output, config, endpoint, run_batch, *, startup_timeout_s=600, shutdown_timeout_s=60):
    """A sealed native result and camera bag precede each atomic episode commit."""
    output = Path(output).resolve()
    benchmark = config.get("benchmark", {})
    count = config.get("episodes", 1)
    plan = [
        dict(episode_idx=i * count + episode, instance_id=p.layout_id, file_idx=0, episode_id=episode)
        for i, p in enumerate(plans)
        for episode in range(count)
    ]
    source = Path(__file__).parent
    identity = dict(
        runtime="newton",
        config=stable_config(config),
        endpoint=None if benchmark.get("resume_id") else endpoint,
        inputs=[
            dict(
                scene=file_digest(p.scene),
                launcher=file_digest(p.launcher),
                evaluation=stable_config(yaml.safe_load(p.evaluation.read_text())),
                plugin=file_digest(p.scene.parent / yaml.safe_load(p.scene.read_text())["newton"]["scene_plugin"]),
            )
            for p in plans
        ],
        sources={p.name: file_digest(p) for p in source.glob("*.py")},
    )
    with EvaluationCheckpoint(output, resume=benchmark.get("resume", False)) as checkpoint:
        if checkpoint.state is None:
            if (output / "evaluate_ret_00.json").exists():
                raise CheckpointRejected("Chef output already exists; resume it or choose a fresh output root")
            checkpoint._result_file = "evaluate_ret_00.json"
            checkpoint.prepare(identity, plan, [], config)
            evaluations = []
        else:
            evaluations = checkpoint.restore(identity)
            if checkpoint.state["plan"] != plan:
                raise CheckpointRejected("Chef episode plan changed")
            for row in evaluations:
                for artifact in row["episode_artifacts"]:
                    path = (output / artifact["path"]).resolve()
                    if not path.is_relative_to(output) or not path.is_file() or file_digest(path) != artifact["sha256"]:
                        raise CheckpointRejected("Committed chef episode artifacts are missing or changed")
            checkpoint.recording_paths()

        def publish():
            evaluations.sort(key=lambda row: row["native_result"]["episode_idx"])
            if evaluations:
                write_summary(output, evaluations)
                if digest(read_summary_details(output)) != digest(evaluations):
                    raise RuntimeError("Chef summary publication failed")
            else:
                atomic_write_json(output / "evaluate_ret_00.json", {"details": [], "statistics": {}})
            atomic_write_json(output / "results.json", [row["native_result"] for row in evaluations])
            atomic_write_json(output / "summary.json", checkpoint.progress)

        publish()
        pending = [entry for entry in plan if not checkpoint.is_completed(entry)]
        if pending:
            attempt = output / "_attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            by_id = {p.layout_id: p for p in plans}
            runs, coordinates = [], {}
            for entry in pending:
                original = by_id[entry["instance_id"]]
                directory = attempt / "runs" / f"episode-{entry['episode_idx']:04d}"
                evaluation = yaml.safe_load(original.evaluation.read_text())
                evaluation["episodes"] = 1
                if "max_steps" in config:
                    evaluation["max_steps"] = config["max_steps"]
                evaluation.setdefault("benchmark", {}).update(
                    episode_offset=entry["episode_idx"],
                    recording_root=str(output),
                    record=benchmark.get("record", False),
                    record_state=benchmark.get("record_state", True),
                    fps=benchmark.get("fps", 30),
                )
                if "headless" in config:
                    evaluation["headless"] = config["headless"]
                path = attempt / f"episode-{entry['episode_idx']:04d}.yaml"
                path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
                launcher = directory / "launcher.yaml"
                runtime_evaluation = directory / "evaluation.yaml"
                command = list(original.runner_command)
                command[command.index("--config") + 1] = str(runtime_evaluation)
                command[command.index("--output-dir") + 1] = str(directory)
                command.append("--benchmark.resume=false")
                engine = [
                    f"launcher_config:={launcher}" if arg.startswith("launcher_config:=") else arg
                    for arg in original.engine_command
                ]
                # Bringup gates RGB on this flag; its Isaac snapshot worker always runs offscreen.
                engine = ["headless:=false" if arg.startswith("headless:=") else arg for arg in engine]
                overrides = {"evaluation_overrides": {}} if hasattr(original, "evaluation_overrides") else {}
                runs.append(
                    replace(
                        original,
                        evaluation=path,
                        run_dir=directory,
                        runtime_launcher=launcher,
                        runtime_evaluation=runtime_evaluation,
                        runner_command=command,
                        engine_command=engine,
                        **overrides,
                    )
                )
                coordinates[directory] = entry

            def completed(run, row):
                details = read_summary_details(run.run_dir)
                if len(details) != 1 or not details[0].get("valid"):
                    raise RuntimeError("Chef episode has no valid canonical score")
                detail = details[0]
                entry = coordinates[run.run_dir]
                detail["episode_coord"] = {key: entry[key] for key in ("instance_id", "file_idx", "episode_id")}
                detail["episode_artifacts"] = [
                    dict(
                        path=str((run.run_dir / "results.json").relative_to(output)),
                        sha256=file_digest(run.run_dir / "results.json"),
                    )
                ]
                bag = detail["native_result"].get("recording")
                if benchmark.get("record", False) and not bag:
                    raise RuntimeError("Chef episode finished without a camera recording")
                evaluations.append(detail)
                publish()
                checkpoint.commit(detail, bag=bag)
                publish()

            run_batch(
                runs, startup_timeout_s=startup_timeout_s, shutdown_timeout_s=shutdown_timeout_s, completed=completed
            )
        if benchmark.get("record", False) and checkpoint.finished:
            from geniesim_benchmark.plugins.output_system.observations import finalize_recordings

            artifacts = finalize_recordings(
                output,
                checkpoint.recording_paths(),
                benchmark.get("record_video_dir"),
                reports=[checkpoint.result_path],
            )
            checkpoint.finalize_recordings(artifacts)
        publish()
        return 0 if checkpoint.finished else 1
