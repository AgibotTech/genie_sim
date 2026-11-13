# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single-process native MuJoCo Rubik evaluation, with auditable outcomes."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import msgpack
import mujoco
import numpy as np
from PIL import Image

from geniesim_benchmark.plugins.output_system.dexhand2 import (
    EVALUATION_FILENAME,
    RUBIK_STAGES,
    RUBIK_TASK,
    make_evaluation,
    write_summary,
)
from geniesim_benchmark.plugins.output_system.checkpoint import (
    CHECKPOINT_REJECTED_EXIT_CODE,
    CheckpointRejected,
    atomic_write_json,
    digest,
)

from .config import load_config, config_dict
from .cameras import calibration
from .native import NativeEnvironment, PhysicsError, TaskFailure
from .policy import HoldPolicy, WebsocketPolicy, PolicyError, InferenceError, parse_actions, payload_for
from .kinematics import NativeIKFKSolver
from .scoring import RubikScorer, summarize
from .viewer import ViewerClosed
from .scenes import episode_scenes
from .checkpoint import RubikCheckpoint


def write_json(path, value):
    atomic_write_json(path, value)


def run_episode(env, cfg, policy, index, scramble, directory, scene=None, *, evaluations=None):
    directory.mkdir()
    started = time.monotonic()
    initial, last_scored_sample = None, None
    scorer = RubikScorer(
        cfg.rubik.hold_seconds,
        cfg.rubik.minimum_support_fingers,
        cfg.rubik.pickup_clearance,
    )
    inference_count, inference_seconds, action_number = 0, 0.0, 0
    status, reason, error = "task_failure", "timeout", None
    samples, action_log = [], []
    projection_count, maximum_projection = 0, 0.0
    arm_projection_count, arm_maximum_projection = 0, 0.0
    raw_times = []
    raw = {key: [] for key in ("qpos", "qvel", "ctrl", "qfrc_applied", "actuator_force")}

    def record_raw():
        if cfg.benchmark.record and (not raw_times or env.data.time > raw_times[-1]):
            raw_times.append(float(env.data.time))
            for key in raw:
                raw[key].append(getattr(env.data, key).copy())

    recording = None
    bag = None
    if cfg.benchmark.record and cfg.rubik.render:
        from geniesim_benchmark.plugins.output_system.observations import ObservationRecording

        recording = ObservationRecording(
            cfg.benchmark.output_dir,
            index,
            cfg.rubik.prompt,
            fps=cfg.benchmark.fps,
            record_state=cfg.benchmark.record_state,
        )

    def capture(images):
        if recording is not None:
            base = env.model.body("base_link").id
            quat = env.data.xquat[base]
            state = {
                "joints": {
                    "name": [env.model.joint(j).name for j in env.motor_joints],
                    "position": env.data.qpos[env.motor_q].tolist(),
                },
                "base_pose": {
                    "position": env.data.xpos[base].tolist(),
                    "orientation_xyzw": [*map(float, quat[1:]), float(quat[0])],
                },
            }
            recording.capture(images, env.steps, state, sim_time=float(env.data.time))

    max_steps = int(np.ceil(cfg.rubik.timeout_seconds / env.model.opt.timestep))
    try:
        env.hdr_seed = cfg.benchmark.seed
        env.hdr_episode = index
        initial = env.reset(scramble, scene)
        if env.graphics is not None:
            env.graphics.sync()
        scorer.initial_height = env.initial_height
        if scorer.update(initial):
            status, reason = "success", "cube_solved"
        last_scored_sample = initial
        samples.append(initial)
        record_raw()
        if recording is not None:
            capture(env.images())
        if not cfg.app.headless:
            env.open_viewer()
        policy.reset()
        native_ik = None
        while env.steps < max_steps and not scorer.success:
            if env.viewer is not None:
                env.viewer.sync()
            images = env.images() if cfg.rubik.render else {}
            capture(images)
            payload = payload_for(env, index, images, cfg.rubik.prompt, cfg.rubik.segments)
            if inference_count == 0:
                (directory / "initial_observation.msgpack").write_bytes(msgpack.packb(payload, use_bin_type=True))
                for name, image in images.items():
                    Image.fromarray(image).save(directory / (name + ".png"))
            if env.viewer is not None:
                print(
                    f"Episode {index}, sim={env.data.time:.3f}s: waiting for policy reply {inference_count + 1}",
                    flush=True,
                )
            sent_at = time.monotonic()
            reply = policy.infer(payload)
            reply_seconds = time.monotonic() - sent_at
            inference_seconds += reply_seconds
            inference_count += 1
            (directory / f"reply-{inference_count:04d}.msgpack").write_bytes(msgpack.packb(reply, use_bin_type=True))
            projection = {}
            result = reply.get("result") if isinstance(reply, dict) else None
            if (
                native_ik is None
                and isinstance(result, dict)
                and any(
                    isinstance(result.get(key), dict) and reply["result"][key].get("kind") == "EEF_ABS"
                    for key in ("left_arm", "right_arm")
                )
            ):
                native_ik = NativeIKFKSolver(env.model, env.data, env.arm_q, env.action_bounds[:14])
            actions, done = parse_actions(
                reply,
                env.action_contract(),
                cfg.rubik.max_chunk_size,
                cfg.rubik.hand_target_tolerance,
                projection,
                measured_state=payload["params"]["states"],
                ik_solver=native_ik,
            )
            if env.viewer is not None:
                print(f"Policy replied in {reply_seconds:.2f}s; executing {len(actions)} actions", flush=True)
            arm_projection_count += projection.get("arm_changed_values", 0)
            arm_maximum_projection = max(arm_maximum_projection, projection.get("arm_maximum_projection_radians", 0.0))
            projection_count += projection.get("changed_values", 0)
            maximum_projection = max(maximum_projection, projection.get("maximum_projection_radians", 0.0))
            for reply_row, action in enumerate(actions):
                action_started = time.monotonic()
                action_start_step = env.steps
                nominal_start = round(action_number / (env.model.opt.timestep * cfg.rubik.control_hz))
                action_number += 1
                nominal_end = round(action_number / (env.model.opt.timestep * cfg.rubik.control_hz))
                nominal_steps = nominal_end - nominal_start
                execution_steps = env.set_action(action, nominal_steps)
                action_log.append(
                    {
                        "at_step": env.steps,
                        "targets": action.tolist(),
                        "reply_index": inference_count,
                        "reply_row": reply_row,
                        "nominal_physics_ticks": nominal_steps,
                        "planned_physics_ticks": execution_steps,
                    }
                )
                next_step = min(action_start_step + execution_steps, max_steps)
                while env.steps < next_step:
                    try:
                        sample = env.tick()
                    finally:
                        record_raw()
                    last_scored_sample = sample
                    if scorer.update(sample):
                        status, reason = "success", "solved_and_held" if cfg.rubik.hold_seconds else "cube_solved"
                        break
                samples.append(sample if env.steps else initial)
                if recording is not None:
                    capture(env.images())
                if env.viewer is not None:
                    env.viewer.pace(action_started, (env.steps - action_start_step) * env.model.opt.timestep)
                if scorer.success or env.steps >= max_steps:
                    break
            if scorer.success:
                break
            if done:
                reason = "policy_finished"
                break
    except ViewerClosed as exc:
        status, reason, error = "interrupted", "viewer_closed", str(exc)
    except TaskFailure as exc:
        status, reason, error = "task_failure", str(exc), str(exc)
    except PhysicsError as exc:
        status, reason, error = "invalid_physics", "physics_error", str(exc)
    except InferenceError as exc:
        status, reason, error = "invalid_inference", "inference_error", str(exc)
    except PolicyError as exc:
        status, reason, error = "invalid_policy", "policy_error", str(exc)
    except Exception as exc:
        status, reason, error = "invalid_runtime", "runtime_error", f"{type(exc).__name__}: {exc}"
    finally:
        if recording is not None:
            bag = recording.close()
    valid = status in ("success", "task_failure")
    scores = scorer.scores()
    result = {
        "schema_version": 3,
        "scoring_contract": scorer.contract(),
        "checker_contract": env.checker_contract(),
        "task_type": "benchmark",
        "task_name": cfg.benchmark.task_name,
        "robot_type": "g2_crsb_wuji",
        "backend": "mujoco",
        "policy": cfg.benchmark.policy,
        "episode_index": index,
        "runtime_identity": {
            "pid": os.getpid(),
            "environment_id": id(env),
            "model_id": id(env.model),
            "renderer_pid": getattr(getattr(env.graphics, "process", None), "pid", None),
        },
        "render_background": getattr(env.graphics, "hdr_background", None),
        "scramble": scramble if scene is None else None,
        "scene": scene.receipt() if scene is not None else None,
        "status": status,
        "reason": reason,
        "valid": valid,
        "score": scorer.score if valid else None,
        "stage_points": scorer.stage_points(),
        "stage_events": scorer.stage_events(),
        "result": {
            "code": 0 if status == "success" else 2 if valid else 1,
            "step": env.steps,
            "msg": reason,
            "scores": scores,
        },
        "simulation_seconds": float(env.data.time),
        "wall_seconds": time.monotonic() - started,
        "inference_count": inference_count,
        "inference_wall_seconds": inference_seconds,
        "maximum_integration_error": env.maximum_integration_error,
        "arm_speed": env.arm_speed_diagnostics(),
        "robot_collisions": env.robot_collision_diagnostics(),
        "actuation": env.controller.diagnostics() if hasattr(env.controller, "diagnostics") else None,
        "maximum_hold_seconds": scorer.maximum_hold_seconds,
        "action_projection": {
            "arm_target_limit_policy": "clip_to_joint_limits",
            "arm_changed_values_in_received_chunks": arm_projection_count,
            "arm_maximum_projection_radians": arm_maximum_projection,
            "hand_target_tolerance": cfg.rubik.hand_target_tolerance,
            "changed_values_in_received_chunks": projection_count,
            "maximum_projection_radians": maximum_projection,
        },
        "error": error,
        "initial": initial,
        "last_scored_sample": last_scored_sample,
        "recorded_raw": cfg.benchmark.record,
        "recording": bag,
    }
    write_json(directory / "result.json", result)
    write_json(directory / "samples.json", samples)
    write_json(directory / "actions.json", action_log)
    if cfg.benchmark.record:
        np.savez_compressed(
            directory / "raw-1000hz.npz",
            timestep=env.model.opt.timestep,
            time=np.asarray(raw_times),
            **{k: np.asarray(v) for k, v in raw.items()},
        )
    if result["task_name"] == RUBIK_TASK:
        evaluation = make_evaluation(
            RUBIK_TASK,
            cfg.rubik.prompt,
            result,
            stages=RUBIK_STAGES,
            stage_scores=scores,
            e2e=int(scores["E2E"]),
            count=len({action["reply_index"] for action in action_log}),
            step_out=int(np.floor(cfg.rubik.timeout_seconds * cfg.rubik.control_hz)),
            steps=env.steps,
            duration=result["wall_seconds"],
            valid=valid,
        )
        if evaluations is None:
            write_summary(directory, [evaluation])
        else:
            evaluations.append(evaluation)
    return result


def create_output_directory(cfg):
    """Create an isolated result directory and record its resolved path."""
    resume = cfg.benchmark.resume and not cfg.benchmark.preview
    if cfg.benchmark.output_dir:
        out = Path(cfg.benchmark.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=resume)
    else:
        parent = Path("output") / cfg.benchmark.task_name
        parent.mkdir(parents=True, exist_ok=True)
        if resume:
            out = (parent / cfg.benchmark.resume_id if cfg.benchmark.resume_id else parent).resolve()
            out.mkdir(parents=True, exist_ok=True)
        else:
            prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            out = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
    cfg.benchmark.output_dir = str(out)
    return out


def run(cfg):
    try:
        out = create_output_directory(cfg)
    except OSError as error:
        print(f"Cannot create fresh output directory: {error}", file=sys.stderr)
        return 1
    episodes, env, policy = [], None, None
    evaluations = []
    checkpoint = None
    restored = False
    dexhand2 = cfg.benchmark.task_name == RUBIK_TASK
    report_path = out / (EVALUATION_FILENAME if dexhand2 else "evaluate_ret_rubik.json")

    def finalize():
        if cfg.benchmark.record and cfg.rubik.render and checkpoint.finished:
            from geniesim_benchmark.plugins.output_system.observations import finalize_recordings

            artifacts = finalize_recordings(
                out, checkpoint.recording_paths(), cfg.benchmark.record_video_dir, reports=[report_path]
            )
            checkpoint.finalize_recordings(artifacts)

    def publish():
        episodes.sort(key=lambda row: row["episode_index"])
        evaluations.sort(key=lambda row: row["native_result"]["episode_index"])
        statistics = summarize(episodes)
        if episodes and all(episode.get("scene") for episode in episodes):
            statistics["by_turns"] = {
                f"{turns}turn": summarize([episode for episode in episodes if episode["scene"]["turns"] == turns])
                for turns in sorted({episode["scene"]["turns"] for episode in episodes})
            }
        details = evaluations if dexhand2 else episodes
        if dexhand2 and evaluations:
            write_summary(out, evaluations)
            if digest(json.loads(report_path.read_text())["details"]) != digest(details):
                raise RuntimeError("Rubik evaluation summary was not published completely")
        else:
            write_json(report_path, {"details": details, "statistics": statistics})
        write_json(out / "summary.json", statistics)

    try:
        if not cfg.benchmark.preview:
            checkpoint = RubikCheckpoint(out, cfg)
            checkpoint.__enter__()
            rows = checkpoint.initialize()
            episodes = [row["native_result"] for row in rows] if dexhand2 else rows
            evaluations = rows if dexhand2 else []
            restored = True
            publish()  # Remove uncommitted result tails before any restart.
            if checkpoint.finished:
                finalize()
                print(f"All Rubik episodes already completed. Results: {report_path}", flush=True)
                return 0
        write_json(out / "config.json", config_dict(cfg))
        runtime_dir = out
        if checkpoint is not None:
            runtime_dir = out / "_attempts" / uuid.uuid4().hex
            runtime_dir.mkdir(parents=True)
        env = NativeEnvironment(cfg.rubik)
        env.task_name = cfg.benchmark.task_name
        env.hdr_seed = cfg.benchmark.seed
        env.hdr_episode = 0
        scenes = (
            checkpoint.scenes
            if checkpoint is not None
            else episode_scenes(
                env.scene_catalog,
                cfg.benchmark.task_name,
                cfg.benchmark.seed,
                cfg.benchmark.num_instances,
                cfg.benchmark.num_episode,
            )
        )
        first_index = (
            next((entry["episode_idx"] for entry in checkpoint.plan if not checkpoint.is_completed(entry)), 0)
            if checkpoint
            else 0
        )
        env.hdr_episode = first_index
        env.reset(
            checkpoint.scramble(first_index) if checkpoint else cfg.rubik.scramble or "R'",
            scenes[first_index] if scenes else None,
        )
        env.configure_renderer(cfg.app.headless, runtime_dir)
        metadata = {
            "backend": "mujoco",
            "live_visualization": not cfg.app.headless,
            "renderer": cfg.rubik.renderer,
            "scene_archive_sha256": env.scene_catalog.archive_sha256 if scenes else None,
            "rendering": env.graphics.metadata if env.graphics is not None else {"renderer": "mujoco"},
            "python": sys.version,
            "executable": sys.executable,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "package_manifest_sha256": env.manifest_hash,
            "action_contract": env.action_contract(),
            "physics": {
                "cube_mass_kg": float(env.model.body_mass[env.cube.root :].sum()),
                "cubies": len(env.cube.items),
                "robot_actuators": env.model.nu,
                "cube_actuators": 0,
                "equality_constraints": env.model.neq,
                "internal_contact_pairs": env.model.npair,
                "robot_collision_termination": cfg.rubik.terminate_on_robot_collision,
            },
            "checker_contract": env.checker_contract(),
            "scoring_contract": RubikScorer(
                cfg.rubik.hold_seconds, cfg.rubik.minimum_support_fingers, cfg.rubik.pickup_clearance
            ).contract(),
            "camera_calibration": calibration()["provenance"],
            "end_pose_reference": "arm_l_end_link / arm_r_end_link, expressed in base_link and arm_base_link; xyzw quaternion",
            "cameras": cfg.rubik.cameras,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(out / "run.json", metadata)
        mujoco.mj_saveModel(env.model, str(out / "model.mjb"))
        if cfg.benchmark.preview:
            if not cfg.app.headless:
                env.open_viewer()
            images = env.images()
            for name, image in images.items():
                Image.fromarray(image).save(out / (name + ".png"))
            payload = payload_for(env, 0, images, cfg.rubik.prompt, cfg.rubik.segments)
            (out / "observation.msgpack").write_bytes(msgpack.packb(payload, use_bin_type=True))
            write_json(out / "summary.json", {"status": "preview", "scored": False, "score": None})
            print(f"Camera preview: {out}", flush=True)
            if env.viewer is not None:
                env.viewer.wait_until_closed()
            return 0
        policy = (
            HoldPolicy()
            if cfg.benchmark.policy == "hold"
            else WebsocketPolicy(cfg.benchmark.infer_host, cfg.rubik.inference_timeout_seconds)
        )
        tokens = np.random.default_rng(cfg.benchmark.seed).permutation([f + s for f in "URFDLB" for s in ("", "'")])
        for index in range(cfg.benchmark.num_instances * cfg.benchmark.num_episode):
            entry = checkpoint.plan[index]
            if checkpoint.is_completed(entry):
                continue
            scramble = cfg.rubik.scramble or str(tokens[index % len(tokens)])
            scene = scenes[index] if scenes else None
            episode_dir = runtime_dir / f"episode-{index:04d}"
            result = run_episode(env, cfg, policy, index, scramble, episode_dir, scene, evaluations=evaluations)
            result.update(
                instance_id=index // cfg.benchmark.num_episode, attempt_index=index % cfg.benchmark.num_episode
            )
            write_json(episode_dir / "result.json", result)
            episodes.append(result)
            detail = evaluations[-1] if dexhand2 else result
            detail["episode_coord"] = {key: entry[key] for key in ("instance_id", "file_idx", "episode_id")}
            detail["episode_artifacts"] = checkpoint.artifacts(episode_dir)
            publish()
            if result["valid"]:
                if cfg.benchmark.record and cfg.rubik.render and not result.get("recording"):
                    raise RuntimeError("Rubik episode finished without a camera recording")
                checkpoint.commit(detail, bag=result.get("recording"))
            print(f"Episode {index}: {result['status']} ({result['reason']}), score={result['score']}", flush=True)
            if cfg.benchmark.resume and not result["valid"]:
                break  # Inference/runtime interruptions remain retryable.
        finalize()
        print(f"Results: {report_path}", flush=True)
        if env.viewer is not None:
            if env.viewer.is_running():
                env.viewer.sync()
            print("Evaluation finished; the window holds the final state. Close it to exit.", flush=True)
            env.viewer.wait_until_closed()
        return 1 if any(not e["valid"] for e in episodes) else 0
    except Exception as error:
        # Rejected checkpoints must never overwrite the previously committed report.
        if checkpoint is None or restored:
            write_json(out / "error.json", {"status": "invalid_setup", "error": f"{type(error).__name__}: {error}"})
            write_json(out / "summary.json", {**summarize(episodes), "status": "invalid_setup"})
        print(f"MuJoCo benchmark setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return CHECKPOINT_REJECTED_EXIT_CODE if isinstance(error, CheckpointRejected) else 1
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if checkpoint is not None:
                    checkpoint.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args(argv)
    try:
        cfg = load_config(args.config, overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
