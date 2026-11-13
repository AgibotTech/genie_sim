# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Synchronous Wuji chef inference on a managed or externally launched Newton engine."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from geniesim_benchmark.plugins.output_system.dexhand2 import (
    WOK_STAGES,
    WOK_TASK,
    make_evaluation,
    write_summary,
)

from .corobot import CorobotClient, JointGroups, build_payload
from .images import RosImages
from .progress import RosTaskProgress


def _positions(snapshot) -> dict[str, float]:
    if snapshot.schema.num_envs != 1:
        raise ValueError("Wuji chef evaluation currently requires exactly one world")
    return dict(zip(snapshot.schema.joint_names, map(float, snapshot.joint_positions[0])))


def pause_observation(client, images, *, timeout_s: float):
    """Freeze, then refresh again so in-flight running ticks cannot satisfy freshness."""
    client.pause()
    before = client.snapshot()
    requested_ns = time.time_ns()
    client.pause()
    deadline = time.monotonic() + timeout_s
    while True:
        state = client.snapshot()
        if (
            state.joint_sequence > before.joint_sequence
            and state.body_sequence > before.body_sequence
            and state.joint_stamp >= requested_ns / 1e9
            and state.body_stamp >= state.joint_stamp
        ):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("no final joint/body state after physics pause")
        time.sleep(0.001)
    stamp_ns, frames = images.wait(after_ns=requested_ns, timeout_s=max(0.001, deadline - time.monotonic()))
    return state, stamp_ns, frames


def run_episode(
    client,
    images,
    policy,
    *,
    instruction: str,
    initial_positions: dict[str, float],
    max_steps: int,
    task_name: str = "wok_flip",
    task_segments=(),
    episode_idx: int = 0,
    observation_timeout_s: float = 30,
    groups: JointGroups = JointGroups(),
    progress=None,
    evaluation_timeout_s: float = 30,
    event=None,
    recording=None,
) -> dict:
    """Run bounded action chunks; every model request occurs behind a pause barrier."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    record = event if event is not None else lambda value: None
    state, stamp_ns, frames = pause_observation(client, images, timeout_s=observation_timeout_s)
    evaluation = None

    def capture():
        if recording is not None:
            from geniesim_benchmark.plugins.output_system.observations import snapshot_state

            recording.capture(frames, steps, snapshot_state(state))

    def score_after(*, after_tick=None, after_ns=0):
        if progress is None:
            return None
        item = progress.wait(
            reset_epoch=state.reset_epoch, after_tick=after_tick, after_ns=after_ns, timeout_s=evaluation_timeout_s
        )
        record({"event": "task_progress", "steps": steps, **item.as_dict()})
        return item

    steps = requests = dispatches = 0
    capture()
    evaluation = score_after(after_ns=stamp_ns)
    held = _positions(state)
    active = set(groups.left_arm + groups.right_arm + groups.left_effector + groups.right_effector)
    controllable = tuple(name for name in held if name in active)
    initial = {**held, **initial_positions}
    held = {name: value if name in active else initial[name] for name, value in held.items()}
    history_interval = 0
    history = []
    reason = "step_limit"
    while steps < max_steps:
        if evaluation is not None and evaluation.failed:
            reason = evaluation.failure_reason
            break
        measured = {**initial, **{name: value for name, value in _positions(state).items() if name in active}}
        payload = build_payload(
            measured,
            frames,
            instruction,
            groups=groups,
            task_name=task_name,
            task_segments=task_segments,
            episode_idx=episode_idx,
            history_interval=history_interval,
            history_frames=history,
            timestamp_ns=stamp_ns,
            task_progress=evaluation.wire_progress() if evaluation is not None else (),
        )
        record({"event": "inference_start", "steps": steps, "observation_stamp_ns": stamp_ns})
        chunk = policy.infer(
            payload,
            groups=groups,
            controllable_joints=controllable,
            held_positions=held,
            initial_positions=initial,
        )
        requests += 1
        record({"event": "inference_end", "steps": steps, "chunk_steps": len(chunk.targets)})
        if chunk.server_episode_done:
            reason = "model_episode_done"
            break
        if not chunk.targets:
            raise ValueError("model returned an empty action chunk without episode_done")
        history_interval = chunk.history_interval
        history = []
        client.resume()
        for index, target in enumerate(chunk.targets):
            if steps >= max_steps:
                break
            target = {**held, **target}
            values = np.asarray([[target[name] for name in state.schema.joint_names]], dtype=np.float32)
            state = client.step(values).snapshot
            if index == 0:
                dispatches += 1
            held = dict(target)
            steps += 1
            record({"event": "action_issued", "steps": steps})
            # TF/joints can arrive before the independently published task
            # score. Wait before the next target, even in the same chunk.
            evaluation = score_after(after_tick=evaluation.tick if evaluation is not None else None)
            if evaluation is not None and evaluation.failed:
                reason = evaluation.failure_reason
                client.pause()
                break
            collect = history_interval > 0 and index > 0 and index % history_interval == 0
            last = index == len(chunk.targets) - 1 or steps == max_steps
            if collect or last:
                state, stamp_ns, frames = pause_observation(client, images, timeout_s=observation_timeout_s)
                capture()
                evaluation = score_after(after_ns=stamp_ns)
                if evaluation is not None and evaluation.failed:
                    reason = evaluation.failure_reason
                    break
                if collect:
                    encoded = build_payload(
                        {**initial, **{name: value for name, value in _positions(state).items() if name in active}},
                        frames,
                        instruction,
                        groups=groups,
                    )["params"]["images"]["head"]
                    history.append({"head": encoded})
                if not last:
                    client.resume()
    result = {
        "episode_idx": episode_idx,
        "issued_steps": steps,
        "inference_requests": requests,
        "stop_reason": reason,
        "task_success": None,
        "completion": "arrival_v1",
    }
    if evaluation is not None:
        if task_name == WOK_TASK:
            result["policy_dispatch_count"] = dispatches
        result.update(
            score=evaluation.score,
            max_score=evaluation.max_score,
            milestones=dict(evaluation.milestones),
            task_success=evaluation.task_success,
            failed=evaluation.failed,
            failure_reason=evaluation.failure_reason,
            evaluation=evaluation.as_dict(),
        )
    return result


def _resolve_launcher(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        local = config_path.parent / path
        if local.is_file():
            return local.resolve()
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("genie_sim_bringup")) / "config" / path
    return path.resolve(strict=True)


def verify_layout_reset(snapshot, scene, instance):
    """Independently verify fixed-root placement from published body poses."""
    switch = scene.get("instance_switch") or {}
    if instance is None or not switch.get("robot_base_xy"):
        return None
    offset = switch["robot_base_xy"][str(instance)]
    origin = scene["robot"]["init_base_pose"]
    expected = np.array([origin["x"] + offset[0], origin["y"] + offset[1], origin["z"]])
    indices = [i for i, path in enumerate(snapshot.schema.body_paths) if path.rsplit("/", 1)[-1] == "base_link"]
    if len(indices) != 1:
        raise RuntimeError("layout reset requires one published base_link body")
    actual = snapshot.body_poses[0, indices[0], :3]
    if not np.allclose(actual, expected, rtol=0, atol=1e-6):
        raise RuntimeError(f"layout {instance} reset base mismatch: {actual.tolist()} != {expected.tolist()}")
    return dict(
        instance_id=instance,
        reset_epoch=snapshot.reset_epoch,
        base_xyz=actual.tolist(),
        expected_base_xyz=expected.tolist(),
    )


def evaluate_attached(
    client,
    images,
    policy,
    *,
    config,
    scene,
    output_dir,
    initial_positions,
    episodes=None,
    max_steps=None,
    progress=None,
    event=None,
):
    """Evaluate layouts on an already-started client; retain native scoring in one place."""
    episodes = config["episodes"] if episodes is None else episodes
    max_steps = config["max_steps"] if max_steps is None else max_steps
    evaluation_timeout_s = config.get("evaluation_timeout_s", 30)
    record = event if event is not None else lambda value: None
    results, evaluations = [], []
    for episode in range(episodes):
        client.resume()
        policy.reset()
        instance = config.get("benchmark", {}).get("instance_id")
        reset_snapshot = client.reset(instances=[instance] if instance is not None else None)
        boundary = verify_layout_reset(reset_snapshot, scene, instance)
        if boundary is not None:
            record({"event": "layout_reset_verified", **boundary})
        started = time.monotonic()
        benchmark = config.get("benchmark", {})
        episode_index = benchmark.get("episode_offset", 0) + episode
        recording = None
        if benchmark.get("record", False):
            from geniesim_benchmark.plugins.output_system.observations import ObservationRecording

            recording = ObservationRecording(
                benchmark.get("recording_root", output_dir),
                episode_index,
                config["instruction"],
                fps=benchmark.get("fps", 30),
                record_state=benchmark.get("record_state", True),
            )
        try:
            result = run_episode(
                client,
                images,
                policy,
                instruction=config["instruction"],
                task_segments=config.get("segments", []),
                initial_positions=initial_positions,
                max_steps=max_steps,
                task_name=config["task_name"],
                episode_idx=episode_index,
                observation_timeout_s=config["observation_timeout_s"],
                progress=progress,
                evaluation_timeout_s=evaluation_timeout_s,
                event=record,
                **({"recording": recording} if recording is not None else {}),
            )
        finally:
            bag = recording.close() if recording is not None else None
        if bag is not None:
            result["recording"] = bag
        results.append(result)
        if "instance_id" in config.get("benchmark", {}):
            results[-1]["instance_id"] = config["benchmark"]["instance_id"]
        (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        if config["task_name"] == WOK_TASK:
            native = results[-1]
            evaluation = make_evaluation(
                WOK_TASK,
                config["instruction"],
                native,
                stages=WOK_STAGES,
                stage_scores=({k: int(v) for k, v in native["milestones"].items()} if "milestones" in native else None),
                e2e=int(native["task_success"]) if native["task_success"] is not None else None,
                count=native.get("policy_dispatch_count"),
                step_out=max_steps,
                steps=native["issued_steps"],
                duration=time.monotonic() - started,
                valid="evaluation" in native,
            )
            evaluations.append(evaluation)
            write_summary(output_dir, evaluations)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="attach to matching normal ROS bringup; omitted starts it")
    parser.add_argument("--infer-host", "--benchmark.infer_host", dest="infer_host", default="localhost:8999")
    parser.add_argument("--output-dir", "--benchmark.output_dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episodes", "--benchmark.num_episode", type=int)
    parser.add_argument("--benchmark.num_instances", dest="instances", type=int)
    parser.add_argument("--assets-root", type=Path)
    from geniesim_benchmark.execution import boolean

    for key in ("record", "record_state", "resume"):
        parser.add_argument(f"--benchmark.{key}", dest=key, type=boolean)
    parser.add_argument("--app.headless", dest="headless", type=boolean)
    parser.add_argument("--benchmark.seed", dest="seed", type=int)
    parser.add_argument("--benchmark.fps", dest="fps", type=int)
    parser.add_argument("--benchmark.resume_id", dest="resume_id")
    parser.add_argument("--benchmark.record_video_dir", dest="record_video_dir")
    parser.add_argument("--dry-run", action="store_true", help="print startup plan without launching ROS or physics")
    options = parser.parse_args(argv)
    config_path = options.config.expanduser().resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    from geniesim_benchmark.execution import output_root, task_output_directory

    if options.output_dir is None:
        options.output_dir = task_output_directory(config, output_root(config_path))
    benchmark = config.setdefault("benchmark", {})
    for key in ("record", "record_state", "resume", "resume_id", "seed", "record_video_dir", "fps"):
        value = getattr(options, key)
        if value is not None:
            benchmark[key] = value
    if options.headless is not None:
        config["headless"] = options.headless
    if config.get("runtime") != "newton":
        parser.error("this runner requires runtime: newton")
    max_steps = options.max_steps if options.max_steps is not None else config["max_steps"]
    episodes = options.episodes if options.episodes is not None else config.get("episodes", 1)
    if options.instances is not None:
        config.setdefault("benchmark", {})["num_instances"] = options.instances
    if type(max_steps) is not int or max_steps < 1 or type(episodes) is not int or episodes < 1:
        parser.error("max_steps and episodes must be positive integers")
    scoring_topic = config.get("task_progress_topic")
    evaluation_timeout_s = config.get("evaluation_timeout_s", 30)
    if (
        isinstance(evaluation_timeout_s, bool)
        or not math.isfinite(float(evaluation_timeout_s))
        or evaluation_timeout_s <= 0
    ):
        parser.error("evaluation_timeout_s must be finite and positive")
    for name in ("ready_timeout_s", "step_timeout_s", "observation_timeout_s", "inference_timeout_s"):
        if not math.isfinite(float(config[name])) or config[name] <= 0:
            parser.error(f"{name} must be finite and positive")
    if options.manifest is None and "instance_id" not in config["benchmark"]:
        config["benchmark"].setdefault("num_instances", 1)
        from .instances import run_instances

        config.update(max_steps=max_steps, episodes=episodes)
        try:
            return run_instances(options, config_path, config)
        except ValueError as error:
            from geniesim_benchmark.plugins.output_system.checkpoint import (
                CheckpointRejected,
                CHECKPOINT_REJECTED_EXIT_CODE,
            )

            if isinstance(error, CheckpointRejected):
                print(str(error), file=sys.stderr)
                return CHECKPOINT_REJECTED_EXIT_CODE
            parser.error(str(error))
        except OSError as error:
            parser.error(str(error))
    if options.manifest is None or options.dry_run:
        from geniesim_benchmark.rt_engine.session import run_managed

        return run_managed(options, config_path, config, main)

    from genie_sim_client import ClientConfig, SimulationClient

    launcher = _resolve_launcher(config["launcher"], config_path)
    client_config = ClientConfig.from_files(launcher, options.manifest)
    if client_config.num_envs != 1:
        parser.error("Wuji chef evaluation requires num_envs: 1")
    scene = yaml.safe_load(client_config.scene_yaml.read_text())
    if scene["robot"]["robot_name"] != "G2_crsB_wuji":
        parser.error("manifest must reference the G2_crsB_wuji chef scene")
    groups = JointGroups()
    initial = dict.fromkeys(groups.all_joints, 0.0)
    initial.update({name: math.radians(value) for name, value in scene["robot"].get("init_joint_pos", {}).items()})
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if (options.output_dir / "results.json").exists():
        parser.error("output directory already contains results.json; choose a new --output-dir")
    results = []
    with (options.output_dir / "events.jsonl").open("a", buffering=1) as events:

        def record(value):
            events.write(json.dumps({"wall_time_ns": time.time_ns(), **value}, ensure_ascii=False) + "\n")

        try:
            with ExitStack() as stack:
                images = stack.enter_context(RosImages(config["image_topics"]))
                progress = stack.enter_context(RosTaskProgress(scoring_topic)) if scoring_topic else None
                client = SimulationClient(
                    client_config,
                    spawn=False,
                    ready_timeout_s=config["ready_timeout_s"],
                    step_timeout_s=config["step_timeout_s"],
                )
                stack.callback(client.close)
                client.start()
                policy = CorobotClient(options.infer_host, timeout_s=config["inference_timeout_s"])
                stack.callback(policy.close)
                try:
                    results = evaluate_attached(
                        client,
                        images,
                        policy,
                        config=config,
                        scene=scene,
                        output_dir=options.output_dir,
                        initial_positions=initial,
                        episodes=episodes,
                        max_steps=max_steps,
                        progress=progress,
                        event=record,
                    )
                except BaseException:
                    try:
                        client.pause()
                    except BaseException as pause_error:
                        record({"event": "emergency_pause_failure", "error": str(pause_error)})
                    raise
                else:
                    client.pause()
        except BaseException as error:
            record({"event": "execution_failure", "type": type(error).__name__, "error": str(error)})
            raise
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
