# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One Newton/RTX session for topology-identical chef layouts, using causal resets."""

from __future__ import annotations

from contextlib import ExitStack
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .layout_batch import OwnedProcessTree, _assert_engine_idle, _prepare, _wait_for_manifest


def shared_scene(plans):
    """Reject structural differences; author only the existing per-instance reset tables."""
    scenes = [yaml.safe_load(p.scene.read_text()) for p in plans]
    normalized = copy.deepcopy(scenes)
    origin = scenes[0]["robot"]["init_base_pose"]
    for scene in normalized:
        pose = scene["robot"]["init_base_pose"]
        if any(
            isinstance(pose.get(k), bool) or not isinstance(pose.get(k), (int, float)) or not math.isfinite(pose[k])
            for k in ("x", "y", "z", "theta")
        ):
            raise ValueError("shared chef layouts require finite base poses")
        if any(pose.get(k, 0) != origin.get(k, 0) for k in ("z", "theta")):
            raise ValueError("shared chef layouts require identical base z and orientation")
        pose["x"] = pose["y"] = 0.0
    if any(scene != normalized[0] for scene in normalized[1:]):
        raise ValueError("shared chef layouts may differ only in initial robot XY")
    launchers = []
    evaluations = []
    for plan in plans:
        launch = yaml.safe_load(plan.launcher.read_text())
        params = launch["genie_sim_engine"]["ros__parameters"]
        params.pop("isaac_render_log_path", None)
        launchers.append(launch)
        evaluation = yaml.safe_load(plan.evaluation.read_text())
        evaluation.pop("launcher", None)
        evaluation.get("benchmark", {}).pop("instance_id", None)
        evaluations.append(evaluation)
    if any(x != launchers[0] for x in launchers[1:]) or any(x != evaluations[0] for x in evaluations[1:]):
        raise ValueError("shared layouts require identical launch, scoring and episode contracts")
    scene = copy.deepcopy(scenes[0])
    if scene.get("instance_switch"):
        raise ValueError("chef shared layout scenes must not already contain instance reset tables")
    scene["instance_switch"] = {
        "body_pose_version": 1,
        "build_instance": plans[0].layout_id,
        "group_instances": [p.layout_id for p in plans],
        # The chef plugin owns its free wok and particles. They restore from
        # the engine's complete initial joint/particle snapshot on every reset.
        # No /World/Objects bodies exist in the supported chef scene.
        "instance_body_poses": {str(p.layout_id): {} for p in plans},
        "robot_base_xy": {
            str(p.layout_id): [s["robot"]["init_base_pose"][k] - origin[k] for k in ("x", "y")]
            for p, s in zip(plans, scenes)
        },
    }
    return scene


def run_shared_batch(plans, *, startup_timeout_s=600, shutdown_timeout_s=60):
    """Keep engine, RTX and client alive; reset(instances=...) selects each layout."""
    if not plans:
        raise ValueError("at least one layout is required")
    scene = shared_scene(plans)  # Validate before any process starts or output is created.
    output = plans[0].run_dir.parent
    output.mkdir(parents=True, exist_ok=False)
    shared = output / "shared"
    shared.mkdir()
    encoded = yaml.safe_dump(scene, sort_keys=False)
    scene_path = shared / ("scene_wuji_chef_reset_" + hashlib.sha256(encoded.encode()).hexdigest()[:12] + ".yaml")
    scene_path.write_text(encoded)
    plugin = scene["newton"]["scene_plugin"]
    shutil.copyfile(plans[0].scene.parent / plugin, shared / plugin)
    manifest = plans[0].assets_root / "scenes" / scene_path.stem / "manifest.json"
    first = plans[0]
    engine_command = [f"scene:={scene_path}" if x.startswith("scene:=") else x for x in first.engine_command]
    engine_command = [
        f"launcher_config:={shared / 'launcher.yaml'}" if x.startswith("launcher_config:=") else x
        for x in engine_command
    ]
    launch = yaml.safe_load(first.launcher.read_text())
    params = launch["genie_sim_engine"]["ros__parameters"]
    params.update(newton_video_path=str(shared / "observer.mp4"), isaac_render_log_path=str(shared / "isaac.log"))
    (shared / "launcher.yaml").write_text(yaml.safe_dump(launch, sort_keys=False))
    engine_plan = replace(first, scene=scene_path, manifest=manifest, run_dir=shared, engine_command=engine_command)
    rows = [
        {"layout_id": p.layout_id, "status": "not_run", "score": None, "task_success": None, "run_dir": str(p.run_dir)}
        for p in plans
    ]
    summary = {
        "version": 1,
        "task_name": "wok_flip",
        "lifecycle": "in_process_reset",
        "layouts": rows,
        "shared_video": str(shared / "observer.mp4"),
    }
    engine = tree = None
    previous_epoch = -1
    session_id = None

    def save():
        summary["evaluated_count"] = sum(r["status"] == "evaluated" for r in rows)
        summary["task_success_count"] = sum(r["task_success"] is True for r in rows)
        tmp = output / ".batch_summary.json.tmp"
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(output / "batch_summary.json")

    save()
    try:
        _assert_engine_idle()
        with ExitStack() as files:
            log = files.enter_context((shared / "engine.log").open("w"))
            started_ns = time.time_ns()
            engine = subprocess.Popen(
                engine_command,
                cwd=shared,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            tree = OwnedProcessTree(engine)
            engine.stdin.write(b"\n")
            engine.stdin.close()
            _wait_for_manifest(engine_plan, engine, started_ns, startup_timeout_s)
            summary["engine_pid"] = engine.pid
            save()
            # Keep transport subscriptions alive while the engine is paused.
            # A new client cannot obtain three fresh state messages from a
            # paused volatile publisher; reconnecting per layout is incorrect.
            from genie_sim_client import ClientConfig, SimulationClient
            from .runner import evaluate_attached
            from .corobot import CorobotClient, JointGroups
            from .images import RosImages
            from .progress import RosTaskProgress

            config = yaml.safe_load(first.evaluation.read_text())
            client_config = ClientConfig.from_files(shared / "launcher.yaml", manifest)
            images = files.enter_context(RosImages(config["image_topics"]))
            topic = config.get("task_progress_topic")
            progress = files.enter_context(RosTaskProgress(topic)) if topic else None
            client = SimulationClient(
                client_config,
                spawn=False,
                ready_timeout_s=config["ready_timeout_s"],
                step_timeout_s=config["step_timeout_s"],
            )
            files.callback(client.close)
            client.start()
            policy = CorobotClient(
                first.runner_command[first.runner_command.index("--infer-host") + 1],
                timeout_s=config["inference_timeout_s"],
            )
            files.callback(policy.close)
            initial = dict.fromkeys(JointGroups().all_joints, 0.0)
            initial.update(
                {name: math.radians(value) for name, value in scene["robot"].get("init_joint_pos", {}).items()}
            )
            summary["runner_pid"] = os.getpid()
            for plan, row in zip(plans, rows):
                try:
                    _prepare(plan)
                    if engine.poll() is not None:
                        raise RuntimeError("shared engine exited")
                    row["status"] = "running"
                    save()
                    config = yaml.safe_load(plan.runtime_evaluation.read_text())
                    with (plan.run_dir / "events.jsonl").open("x", buffering=1) as events:

                        def record(value):
                            events.write(
                                json.dumps({"wall_time_ns": time.time_ns(), **value}, ensure_ascii=False) + "\n"
                            )

                        results = evaluate_attached(
                            client,
                            images,
                            policy,
                            config=config,
                            scene=scene,
                            output_dir=plan.run_dir,
                            initial_positions=initial,
                            progress=progress,
                            event=record,
                        )
                    (plan.run_dir / "runner.log").write_text(
                        json.dumps({"runner_pid": os.getpid(), "session": "persistent", "episodes": len(results)})
                    )
                    expected = yaml.safe_load(plan.evaluation.read_text())["episodes"]
                    if len(results) != expected:
                        raise RuntimeError("layout episode count mismatch")
                    for result in results:
                        ev = result["evaluation"]
                        if session_id is None:
                            session_id = ev["session_id"]
                        if ev["session_id"] != session_id or ev["reset_epoch"] <= previous_epoch:
                            raise RuntimeError("shared session/reset epoch mismatch")
                        if result.get("instance_id") != plan.layout_id or result.get("score") not in (0, 1, 2):
                            raise RuntimeError("layout identity or native score mismatch")
                        previous_epoch = ev["reset_epoch"]
                    result = results[-1]
                    row.update(
                        status="evaluated",
                        episode_count=len(results),
                        episode_results=results,
                        result=result,
                        score=result["score"],
                        max_score=result["max_score"],
                        task_success=result["task_success"],
                        failed=result["failed"],
                        session_id=session_id,
                        reset_epoch=previous_epoch,
                        issued_steps=result["issued_steps"],
                    )
                except KeyboardInterrupt:
                    row.update(status="interrupted", error="interrupted by user")
                    raise
                except Exception as exc:
                    (plan.run_dir / "runner.log").write_text(traceback.format_exc())
                    row.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
                    # Do not continue a poisoned engine or silently rebuild it.
                    break
                finally:
                    save()
    finally:
        if tree is not None:
            summary["cleanup"] = tree.close(shutdown_timeout_s + 10)
        if summary.get("cleanup", {}).get("forced"):
            raise RuntimeError("shared engine required forced shutdown; video finalization unconfirmed")
        frames_path = shared / "observer.frames.jsonl"
        frames = [json.loads(line) for line in frames_path.read_text().splitlines()] if frames_path.exists() else []
        for plan, row in zip(plans, rows):
            epochs = {r["evaluation"]["reset_epoch"] for r in row.get("episode_results", [])}
            selected = [f["frame"] for f in frames if f["reset_epoch"] in epochs]
            if row["status"] == "evaluated" and not selected:
                raise RuntimeError(f"layout {plan.layout_id} has no recorded observer frames")
            if selected:
                row["video_frames"] = [min(selected), max(selected) + 1]
                clip = plan.run_dir / "observer.mp4"
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-n",
                        "-i",
                        str(shared / "observer.mp4"),
                        "-vf",
                        f"trim=start_frame={min(selected)}:end_frame={max(selected)+1},setpts=PTS-STARTPTS",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "yuv420p",
                        str(clip),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode:
                    raise RuntimeError(f"layout video extraction failed: {result.stderr[-1000:]}")
                (plan.run_dir / "video.json").write_text(
                    json.dumps(
                        {
                            "source": str(shared / "observer.mp4"),
                            "frame_range_exclusive": row["video_frames"],
                            "reset_epochs": sorted(epochs),
                        },
                        indent=2,
                    )
                )
        save()
    return summary
