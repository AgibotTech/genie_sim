# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Three Wuji tasks with one libuipc bringup and synchronized renderer per task.

Physics remains real time during camera capture and inference. The shared client
only attaches; this benchmark owns bringup, policy, scorer and archive lifetime.
"""

from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import uuid

from .adapter import (
    InferenceClient,
    build_payload,
    load_fixed_joints,
    parse_result,
    validate_identity,
    validate_task_snapshot,
    wire_images,
)
from .config import load_config, validate_settings, write_json
from .reporting import common_result, merge_score, summarize
from geniesim_benchmark.plugins.output_system.dexhand2 import write_summary
from geniesim_benchmark.plugins.output_system.checkpoint import (
    CHECKPOINT_REJECTED_EXIT_CODE,
    CheckpointRejected,
    EvaluationCheckpoint,
    atomic_write_json,
    digest,
    file_digest,
)

STOP_LADDER = ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0))


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(_plain(value), allow_nan=False) + "\n")


def archive_observation(directory, packet, tag):
    """Keep full source/camera identity and byte-identical model camera inputs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plain = _plain(packet)
    images = wire_images(plain["images"])
    for name, image in images.items():
        (directory / f"{tag}_{name}.jpg").write_bytes(image["image_data"])
    write_json(directory / (tag + ".json"), plain)


def run_episode(
    client,
    policy,
    *,
    task,
    instruction,
    fixed_joints,
    episode_dir,
    task_segments=(),
    episode_idx=0,
    max_steps=2400,
    action_chunk_steps=0,
    fps=30.0,
    wait_timeout=600.0,
    success=None,
    snapshot_validator=None,
    frame_bindings=None,
    record=False,
    record_state=True,
    recording_root=None,
    recording_run_id=None,
    robot_usd_path=None,
    ik_solver=None,
):
    """Validate whole chunks, count accepted dispatches and stop only at ACK boundaries."""
    root = Path(episode_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = dict(
        task_name=task,
        episode_idx=episode_idx,
        execution_status="failed",
        completed=False,
        steps=0,
        inference_requests=0,
        policy_dispatch_count=0,
        executed_sim_time_s=0.0,
        scoring_status="pending",
        score=None,
        success=None,
        task_tree_finished=None,
        termination_reason=None,
    )
    history = []
    history_metadata = []
    interval = 0
    started = time.monotonic()
    recording = None

    def stopped():
        evidence = success() if success is not None else None
        if evidence:
            result.update(termination_reason="success", early_stopped=True, success_stop_evidence=_plain(evidence))
            return True
        return False

    def capture(tag, *, archive=True):
        packet = client.observe(timeout_s=wait_timeout)
        if archive:
            archive_observation(root / "observations", packet, tag)
        if snapshot_validator is not None:
            snapshot_validator(packet)
        if recording is not None:
            recording.capture(packet, result["steps"])
        return packet

    try:
        if type(max_steps) is not int or max_steps < 1 or type(action_chunk_steps) is not int or action_chunk_steps < 0:
            raise ValueError("action cap must be positive and chunk prefix must be nonnegative")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps < float("inf"):
            raise ValueError("fps must be finite and positive")
        if record:
            from .recording import EpisodeRecording

            recording = EpisodeRecording(
                recording_root or root,
                episode_idx,
                instruction,
                fps=fps,
                record_state=record_state,
                fixed_joints=fixed_joints,
                frame_bindings=frame_bindings,
                run_id=recording_run_id,
            )
        packet = capture("initial")
        while result["steps"] < max_steps:
            if stopped():
                break
            request_index = result["inference_requests"]
            payload = build_payload(
                packet,
                fixed_joints,
                instruction=instruction,
                task_name=task,
                task_segments=task_segments,
                episode_idx=episode_idx,
                history=history,
                history_interval=interval,
                frame_bindings=frame_bindings,
            )
            request_dir = root / "inference" / f"request_{request_index:06d}"
            result["inference_requests"] += 1

            def metadata(value):
                snapshot = value["snapshot"]
                keys = (
                    "request_id",
                    "source_id",
                    "epoch",
                    "snapshot_id",
                    "tick",
                    "cloth_tick",
                    "sim_time",
                    "captured_wall_time_ns",
                )
                images = wire_images(value["images"])
                return {
                    "snapshot": {key: snapshot[key] for key in keys},
                    "render": {
                        key: _plain(value.get(key))
                        for key in ("render_frame", "render_reference_time", "render_complete_wall_time_ns")
                    },
                    "images": {
                        name: {
                            "sha256": hashlib.sha256(image["image_data"]).hexdigest(),
                            "bytes": len(image["image_data"]),
                            "width": image["width"],
                            "height": image["height"],
                        }
                        for name, image in images.items()
                    },
                }

            append_json(
                root / "inference_requests.jsonl",
                {
                    "request_index": request_index,
                    "steps": result["steps"],
                    "observation": metadata(packet),
                    "history_observations": history_metadata,
                    "history_interval": interval,
                },
            )
            response = policy.infer(payload, archive=request_dir)
            # Preserve optional welded channels and every tail row before prefix
            # selection or an early stop received while inference was pending.
            import msgpack

            if not request_dir.exists():
                request_dir.mkdir(parents=True)
            (request_dir / "result.msgpack").write_bytes(msgpack.packb(response, use_bin_type=True))
            chunk = parse_result(
                response,
                measured_positions={**packet["snapshot"]["joints"], **fixed_joints},
                ik_solver=ik_solver,
                robot_usd_path=robot_usd_path,
            )
            write_json(
                request_dir / "validated_chunk.json",
                {
                    "rows": len(chunk.targets),
                    "history_interval": chunk.history_interval,
                    "optional_held_channels": _plain(chunk.optional_channels),
                },
            )
            interval = chunk.history_interval
            history = []
            history_metadata = []
            if stopped():
                break
            count = min(len(chunk.targets), action_chunk_steps or len(chunk.targets), max_steps - result["steps"])
            dispatched = False

            def accepted(ack):
                nonlocal dispatched
                if not dispatched:
                    result["policy_dispatch_count"] += 1
                    dispatched = True
                    append_json(
                        root / "policy_dispatches.jsonl",
                        {"request_index": request_index, "steps": result["steps"], "ack": ack},
                    )

            for offset, target in enumerate(chunk.targets[:count]):
                acknowledgement = client.step(target, 1.0 / fps, timeout_s=wait_timeout, on_accepted=accepted)
                result["steps"] += 1
                result["executed_sim_time_s"] += acknowledgement["executed_sim_time_s"]
                append_json(
                    root / "action_execution.jsonl",
                    {
                        "request_index": request_index,
                        "chunk_offset": offset,
                        "step": result["steps"],
                        "targets": target,
                        "ack": acknowledgement,
                        "completed_wall_time_ns": time.time_ns(),
                    },
                )
                if record:
                    # Video/state sampling must not duplicate the large cloth
                    # geometry JSON on every control tick.
                    packet = capture(f"record_{result['steps']:06d}", archive=False)
                if stopped():
                    break
                if interval and (offset + 1) % interval == 0:
                    packet = capture(f"history_{result['steps']:06d}")
                    history.append({"head": wire_images(packet["images"])["head"]})
                    history_metadata.append(metadata(packet))
            packet = capture(
                "final"
                if result.get("early_stopped") or result["steps"] >= max_steps
                else f"request_{result['steps']:06d}"
            )
            if result.get("early_stopped"):
                break
        if result.get("early_stopped"):
            capture("terminal")
        result.update(
            execution_status="completed", completed=True, termination_reason=result["termination_reason"] or "max_steps"
        )
    except BaseException as exc:
        result.update(
            execution_status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            completed=False,
            error=str(exc) or "cancelled",
            error_type=type(exc).__name__,
        )
    finally:
        if recording is not None:
            try:
                result["recording"] = recording.close()
            except Exception as exc:
                result.update(execution_status="failed", completed=False, error=f"recording: {exc}")
        result["wall_seconds"] = time.monotonic() - started
        write_json(root / "execution.json", result)
    return result


def build_bringup_command(config, scene, *, headless):
    return [
        "ros2",
        "launch",
        "genie_sim_bringup",
        "app.launch.py",
        f"scene:={scene}",
        f"launcher_config:={config['libuipc']['launcher']}",
        f"headless:={str(headless).lower()}",
        "rviz:=false",
    ]


def _session_member(path, session_id):
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    if int(fields[3]) != session_id or fields[0] in ("Z", "X"):
        return None
    if path.stat().st_uid != os.getuid():
        raise RuntimeError("bringup session member ownership cannot be verified")
    return int(fields[19])  # Linux starttime distinguishes a reused PID.


def _session_members(process):
    """Inspect our private SID, including children that made another PGID."""
    process.poll()
    live = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = _session_member(path, process.pid)
            if identity is not None:
                live[int(path.name)] = identity
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Restricted /proc may hide other users. Every member of our own
            # session must remain inspectable before we signal or claim cleanup.
            try:
                if path.stat().st_uid == os.getuid():
                    raise RuntimeError("bringup session cannot be inspected")
            except (FileNotFoundError, ProcessLookupError):
                pass
    return live


def _live_group(process):
    """Compatibility name; ownership is the complete private session."""
    return list(_session_members(process))


def _signal_session_member(pid, starttime, session_id, sig):
    """Pin process identity before the final UID/SID check to avoid PID reuse."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if _session_member(Path("/proc") / str(pid), session_id) != starttime:
            return
        signal.pidfd_send_signal(descriptor, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(descriptor)


def stop_process_group(process):
    if process is None:
        return
    for sig, grace in STOP_LADDER:
        sent = set()
        deadline = time.monotonic() + grace
        while True:
            members = _session_members(process)
            if not members:
                process.wait(timeout=1)
                return
            for pid, starttime in members.items():
                identity = (pid, starttime)
                if identity not in sent:
                    _signal_session_member(pid, starttime, process.pid, sig)
                    sent.add(identity)
            if time.monotonic() >= deadline:
                break
            # A surviving parent can fork or change PGIDs during graceful
            # shutdown. Rescan and signal new verified members in this stage.
            time.sleep(0.05)
    if _live_group(process):
        raise RuntimeError("owned bringup session survived cleanup")
    process.wait(timeout=1)


def episode_domain(environment):
    """Preserve an explicit empty domain, otherwise select an unused private one."""
    explicit = environment.get("ROS_DOMAIN_ID")
    if explicit is not None:
        if not explicit.isdecimal() or not 0 <= int(explicit) <= 232:
            raise ValueError("ROS_DOMAIN_ID must be an integer in [0,232]")
        candidates = [int(explicit)]
    else:
        # DDS participant ports above domain 232 can exceed the UDP port range.
        candidates = [secrets.randbelow(200) + 1 for _ in range(8)]
    for domain in candidates:
        probe = {**environment, "ROS_DOMAIN_ID": str(domain)}
        response = subprocess.run(
            ["ros2", "node", "list", "--no-daemon", "--spin-time", "2"],
            env=probe,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        nodes = [line for line in response.stdout.splitlines() if line.strip().startswith("/")]
        if not nodes:
            return domain
        if explicit is not None:
            raise RuntimeError(f"explicit ROS domain {domain} is occupied: {nodes}")
    raise RuntimeError("could not discover an unused private ROS domain")


def shutdown_recorded_scene(episode_dir, *, timeout_s=30.0, environment=None, discovery_timeout_s=10.0):
    """Enqueue the existing lifecycle shutdown and confirm finalized archive flags."""
    if isinstance(discovery_timeout_s, bool) or not 0 < discovery_timeout_s < float("inf"):
        raise ValueError("shutdown discovery timeout must be finite and positive")
    discovery_deadline = time.monotonic() + discovery_timeout_s
    while True:
        remaining = discovery_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("cannot discover an existing IPC engine lifecycle shutdown service")
        try:
            # --no-daemon creates a fresh DDS participant. A single one-second
            # graph snapshot can miss an already-running service; discovery is
            # read-only and bounded, while the transition below is never retried.
            response = subprocess.run(
                ["ros2", "service", "list", "--no-daemon", "--spin-time", "2"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=remaining,
                check=True,
            )
        except subprocess.TimeoutExpired:
            continue
        candidates = [
            line.strip()
            for line in response.stdout.splitlines()
            if line.strip() in ("/genie_sim_engine_libuipc/change_state", "/genie_sim_engine/change_state")
        ]
        if len(candidates) > 1:
            raise RuntimeError("cannot identify one unambiguous IPC engine lifecycle shutdown service")
        if candidates:
            service = candidates[0]
            break
        time.sleep(min(0.05, max(0.0, discovery_deadline - time.monotonic())))
    reply = subprocess.run(
        ["ros2", "service", "call", service, "lifecycle_msgs/srv/ChangeState", "{transition: {id: 7}}"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if re.search(r"success\s*[=:]\s*(?:True|true)", reply.stdout) is None:
        raise RuntimeError("engine lifecycle shutdown was not accepted")
    deadline = time.monotonic() + timeout_s
    manifest = Path(episode_dir) / "physical_state/manifest.json"
    while time.monotonic() < deadline:
        if manifest.is_file():
            state = json.loads(manifest.read_text())
            if state.get("recording_closed") is True:
                if state.get("solver_shutdown_clean") is not True or state.get("recording_clean") is not True:
                    raise RuntimeError("engine closed a dirty physical archive")
                return service
        time.sleep(0.05)
    raise TimeoutError("engine lifecycle shutdown did not close a clean physical archive")


def verify_physical_archive(episode_dir):
    from genie_sim_engine_libuipc.physical_state_recording import PhysicalStateArchiveFollower

    follower = PhysicalStateArchiveFollower(Path(episode_dir) / "physical_state", timeout_s=30.0).initialize()
    for _ in follower.iter_frames():
        pass
    verification = follower.verification
    if not verification or verification["physical_frames"] < 2:
        raise ValueError("episode requires initialized and advanced verified physical states")
    write_json(Path(episode_dir) / "recording_verified.json", verification)
    return verification


class LibuipcRunSession:
    """One owned bringup, renderer, CPU worker and IPC client for all layouts."""

    def __init__(self, config, endpoint, repo_root, run_dir, *, headless, assets_root=None):
        from .config import materialize_sequence
        from .scoring import prepare_assets

        self.config, self.endpoint = config, endpoint
        self.repo_root, self.directory = Path(repo_root), Path(run_dir)
        self.assets = prepare_assets(config, repo_root, assets_root=assets_root)
        # A new engine needs fresh archives for every epoch, including layouts
        # skipped for renderer cursor alignment. Never overwrite committed data.
        attempt = self.directory / "_attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        self.log_path = attempt / "scene.log"
        self.scene, self.episodes = materialize_sequence(config, attempt, assets_root=self.assets["assets_root"])
        self.headless = headless
        self.process = self.client = self.log = None
        self.environment = None
        self.current = self.episodes[0]["directory"]
        self.original_domain = os.environ.get("ROS_DOMAIN_ID")

    def start(self):
        from genie_sim_client import IPCClient, IPCClientConfig
        from .scoring import assembled_geometry

        settings = self.config["libuipc"]
        try:
            self.environment = {**os.environ, **self.assets["process_environment"]}
            self.environment["ROS_DOMAIN_ID"] = str(episode_domain(self.environment))
            os.environ["ROS_DOMAIN_ID"] = self.environment["ROS_DOMAIN_ID"]
            command = build_bringup_command(self.config, self.scene, headless=self.headless)
            write_json(
                self.directory / "launch.json",
                dict(
                    command=command,
                    cwd=str(self.repo_root),
                    endpoint=self.endpoint,
                    runtime="libuipc",
                    episode_count=len(self.episodes),
                    scene_log=str(self.log_path),
                    process_environment={
                        **self.assets["process_environment"],
                        "ROS_DOMAIN_ID": self.environment["ROS_DOMAIN_ID"],
                    },
                ),
            )
            self.log = self.log_path.open("xb")
            self.process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            protocol = IPCClientConfig.from_files(self.scene, settings["launcher"])

            def event(packet):
                if self.process.poll() is not None:
                    raise RuntimeError(f"bringup exited {self.process.returncode}; inspect {self.log_path}")
                append_json(self.current / "protocol.jsonl", packet)

            self.client = IPCClient(
                protocol, ready_timeout_s=settings["wait_timeout"], wait_timeout_s=settings["wait_timeout"], event=event
            )
            self.client.start()
            validate_identity(_plain(self.client.identity))
            self.manifest = Path(self.assets["assets_root"]) / "scenes" / self.scene.stem / "manifest.json"
            manifest = json.loads(self.manifest.read_text())
            robot_usd = Path(manifest["robot_usda"])
            self.robot_usd_path = str(
                robot_usd if robot_usd.is_absolute() else Path(self.assets["assets_root"]) / robot_usd
            )
            self.assembled = assembled_geometry(
                self.manifest, self.assets["assets_root"], settings["evaluation_scene"], expected_geometry=self.assets
            )
            return self
        except BaseException:
            try:
                self.close()
            except BaseException as cleanup:
                write_json(self.directory / "session-cleanup-error.json", {"error": str(cleanup)})
            raise

    def skip_completed_episode(self, index):
        """Advance a previously scored layout without policy actions or rescoring.

        Physical resets and renderer material selection have separate cursors.
        Observe every skipped epoch while paused so the next unscored layout
        cannot jump over a renderer material plan.
        """
        from .materials import validate_material_capture

        episode = self.episodes[index]
        self.current = episode["directory"]
        if index:
            write_json(self.current / "reuse-reset.json", _plain(self.client.reset(index)))
        identity = _plain(self.client.identity)
        validate_identity(identity)
        packet = self.client.observe(timeout_s=self.config["libuipc"]["wait_timeout"])
        if packet["snapshot"].get("episode_index") != index or packet["epoch"] != index:
            raise ValueError("reuse observation belongs to another configured episode")
        validate_task_snapshot(packet, self.assets)
        plan = json.loads((self.current / "material_generalization.json").read_text())
        receipt = validate_material_capture(
            plan, self.current / "material_generalization_applied.json", packet, manifest_path=self.manifest
        )
        write_json(self.current / "reuse-material-capture.json", receipt)
        self.client.begin_episode()
        write_json(self.current / "reuse-finish.json", _plain(self.client.finish_episode()))

    def close(self):
        errors = []
        for name, close in (
            ("client", lambda: self.client.close() if self.client else None),
            (
                "engine shutdown",
                lambda: shutdown_recorded_scene(self.current, environment=self.environment) if self.process else None,
            ),
            ("bringup", lambda: stop_process_group(self.process)),
            ("scene log", lambda: self.log.close() if self.log else None),
        ):
            try:
                close()
            except BaseException as exc:
                errors.append(f"{name}: {exc}")
        if self.original_domain is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = self.original_domain
        self.process = self.client = self.log = None
        if errors:
            write_json(self.directory / "session-cleanup-error.json", {"errors": errors})
            raise RuntimeError("; ".join(errors))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise


def run_one_episode(session, index, *, evaluations=None):
    """Run, seal and score one epoch while the same simulation processes stay alive."""
    from .scoring import score_physical_episode
    from .streaming import StreamingScoringJob
    from .materials import validate_material_capture, wait_for_material_receipt

    episode = session.episodes[index]
    directory, token = episode["directory"], episode["token"]
    settings = session.config["libuipc"]
    client, process = session.client, session.process
    policy = scorer = None
    result = dict(
        task_name=settings["task"],
        episode_idx=index,
        execution_status="failed",
        completed=False,
        score=None,
        success=False,
        scoring_status="pending",
        steps=0,
        inference_requests=0,
        policy_dispatch_count=0,
    )
    identity, errors, begun = None, [], False
    session.current = directory
    try:
        if index:
            receipt = client.reset(index)
            write_json(directory / "reset.json", _plain(receipt))
        identity = _plain(client.identity)
        validate_identity(identity)
        write_json(directory / "runtime_identity.json", identity)
        write_json(directory / "asset_identity.json", session.assets)
        write_json(directory / "assembled_asset_identity.json", session.assembled)
        write_json(
            directory / "session.json",
            dict(
                bringup_pid=process.pid,
                scene=str(session.scene),
                manifest=str(session.manifest),
                sequence_index=index,
                epoch=identity["epoch"],
                source_id=identity["source_id"],
                success_token=token,
            ),
        )
        plan = json.loads((directory / "material_generalization.json").read_text())
        receipt_path = directory / "material_generalization_applied.json"

        def validate_snapshot_material(packet):
            if packet["snapshot"].get("episode_index") != index or packet["epoch"] != index:
                raise ValueError("observation belongs to another configured episode")
            validate_task_snapshot(packet, session.assets)
            receipt = validate_material_capture(plan, receipt_path, packet, manifest_path=session.manifest)
            append_json(directory / "material_generalization_checks.jsonl", receipt)

        # The native world stays at its recovered state during warmup/material
        # switching and score draining. This capture also commits the next plan
        # on the existing renderer before the episode is allowed to advance.
        packet = client.observe(timeout_s=settings["wait_timeout"])
        validate_snapshot_material(packet)
        wait_for_material_receipt(
            plan,
            receipt_path,
            manifest_path=session.manifest,
            process=process,
            members=_session_members,
            timeout_s=settings["wait_timeout"],
        )
        if settings["scoring_execution"] == "streaming":
            scorer = StreamingScoringJob(
                settings["task"],
                directory,
                session.repo_root,
                drain_timeout_s=settings["scoring_drain_timeout_s"],
                success_token=token,
            )
            scorer.start()
        policy = InferenceClient(session.endpoint, timeout_s=settings["infer_timeout"])

        def success():
            if process.poll() is not None:
                raise RuntimeError("bringup exited during evaluation")
            if scorer is not None:
                scorer.check()
                return scorer.success_notification() if settings["stop_on_success"] else None
            return None

        client.begin_episode()
        begun = True
        result = run_episode(
            client,
            policy,
            task=settings["task"],
            instruction=settings["instruction"],
            task_segments=settings.get("segments", []),
            fixed_joints=load_fixed_joints(session.scene),
            episode_dir=directory,
            episode_idx=index,
            max_steps=settings["max_steps"],
            action_chunk_steps=settings["action_chunk_steps"],
            fps=settings["fps"],
            wait_timeout=settings["wait_timeout"],
            success=success,
            snapshot_validator=validate_snapshot_material,
            frame_bindings=session.assembled["policy_frame_bindings"],
            robot_usd_path=session.robot_usd_path,
            record=session.config["benchmark"].get("record", False),
            record_state=session.config["benchmark"].get("record_state", True),
            recording_root=session.directory,
            recording_run_id=getattr(session, "checkpoint_run_id", None),
        )
    except BaseException as exc:
        result.update(
            execution_status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            completed=False,
            error=str(exc) or "cancelled",
            error_type=type(exc).__name__,
        )
    finally:
        for name, close in (
            ("policy", lambda: policy.close() if policy else None),
            (
                "episode finish",
                lambda: write_json(directory / "finish.json", _plain(client.finish_episode())) if begun else None,
            ),
        ):
            try:
                close()
            except BaseException as exc:
                errors.append(f"{name}: {exc}")
        try:
            if not begun or errors:
                raise RuntimeError("episode did not reach a verified clean finish boundary")
            verification = verify_physical_archive(directory)
            if result.get("execution_status") == "cancelled":
                result.update(scoring_status="cancelled", success=False)
            else:
                report = (
                    scorer.finish(verification)
                    if scorer
                    else score_physical_episode(
                        settings["task"],
                        directory,
                        directory / "scoring",
                        session.repo_root,
                        timeout_s=settings["scoring_drain_timeout_s"],
                    )
                )
                result = merge_score(result, report)
        except BaseException as exc:
            errors.append(f"archive/scoring: {exc}")
        finally:
            if scorer is not None:
                try:
                    scorer.stop()
                except BaseException as exc:
                    errors.append(f"scorer cleanup: {exc}")
        if errors:
            result.update(execution_status="failed", completed=False, success=False, cleanup_errors=errors)
            result.setdefault("error", "; ".join(errors))
        result.update(
            source_identity=identity,
            runtime="libuipc",
            endpoint=session.endpoint,
            instance_id=episode["instance_id"],
            attempt_index=episode["attempt_index"],
        )
        write_json(directory / "result.json", result)
        evaluation = common_result(
            settings["task"],
            result,
            result.get("scoring", {}),
            instruction=settings["instruction"],
            endpoint=session.endpoint,
            step_out=settings["max_steps"],
        )
        if evaluations is None:
            write_summary(directory, [evaluation])
        else:
            evaluations.append(evaluation)
    return result


def _boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() == "true"


def checkpoint_identity(config, endpoint, assets_root):
    """Bind reuse to the task inputs, native geometry, implementation and model."""
    from .scoring import prepare_assets, checker_sources

    value = copy.deepcopy(config)
    value.pop("repo_root", None)
    value.pop("config_path", None)
    benchmark = value["benchmark"]
    for key in ("resume", "output_dir", "record_video_dir"):
        benchmark.pop(key, None)
    for key, default in (("record", False), ("record_state", True), ("resume_id", "")):
        benchmark.setdefault(key, default)
    value["app"] = {k: v for k, v in value.get("app", {}).items() if k != "headless"}
    value["libuipc"].pop("output_dir", None)
    for key in ("evaluation_scene", "launcher", "cameras"):
        value["libuipc"][key] = file_digest(config["libuipc"][key])
    package = Path(__file__).resolve().parents[1]
    sources = [
        *Path(__file__).parent.glob("*.py"),
        *(package / "utils" / name for name in ("eef_control.py", "hard_ik.py", "usd_ik.py")),
        *(
            package / "plugins/output_system" / name
            for name in ("checkpoint.py", "local_recorder.py", "dexhand2.py", "policy_budget.py")
        ),
    ]
    return {
        "runtime": "libuipc",
        "config": value,
        "endpoint": endpoint,
        "assets": prepare_assets(config, config["repo_root"], assets_root=assets_root),
        "sources": {str(p.relative_to(package)): file_digest(p) for p in sources},
        "checkers": checker_sources(config["libuipc"]["task"]),
    }


def evaluate(config, endpoint, root, run_dir, *, headless, assets_root=None):
    """Commit only sealed/scored episodes; retries get independent native archives."""
    benchmark = config["benchmark"]
    plan = [
        dict(
            episode_idx=i,
            instance_id=i // benchmark["num_episode"],
            file_idx=0,
            episode_id=i % benchmark["num_episode"],
        )
        for i in range(benchmark.get("num_instances", 1) * benchmark["num_episode"])
    ]
    with EvaluationCheckpoint(run_dir, resume=benchmark.get("resume", False)) as checkpoint:
        identity = checkpoint_identity(config, endpoint, assets_root)
        if checkpoint.state is None:
            # Native DexHand reports have one canonical name, unlike Isaac's
            # optional sequential result files. Never overwrite another run.
            if (run_dir / "evaluate_ret_00.json").exists():
                raise CheckpointRejected(
                    "Evaluation output exists without a reusable checkpoint; choose a new directory"
                )
            checkpoint._result_file = "evaluate_ret_00.json"
            checkpoint.prepare(identity, plan, [], config)
            evaluations = []
        else:
            evaluations = checkpoint.restore(identity)
            if checkpoint.state["plan"] != plan:
                raise CheckpointRejected("Native episode plan changed")
        atomic_write_json(run_dir / "config.json", config)

        def publish(error=None):
            evaluations.sort(key=lambda row: (row["episode_coord"]["instance_id"], row["episode_coord"]["episode_id"]))
            summary = summarize([row["execution"] for row in evaluations])
            if error is not None:
                summary["error"] = str(error) or "cancelled"
            atomic_write_json(run_dir / "summary.json", summary)
            if evaluations:
                write_summary(run_dir, evaluations)
                saved = json.loads(checkpoint.result_path.read_text())["details"]
                if digest(saved) != digest(evaluations):
                    raise RuntimeError("Native evaluation report was not published completely")
            else:
                atomic_write_json(checkpoint.result_path, {"details": [], "statistics": {}})

        def recording_paths():
            try:
                paths = checkpoint.recording_paths()
            except ValueError as exc:
                raise CheckpointRejected(str(exc)) from exc
            if benchmark.get("record_state", True) and not checkpoint.recordings_ready:
                if any(not (Path(p) / "state.jsonl").is_file() for p in paths):
                    raise CheckpointRejected("A committed joint-state recording is missing")
            return paths

        if benchmark.get("record", False):
            recording_paths()  # Reject missing committed artifacts before bringup.
        checkpoint.cleanup_recordings()
        publish()  # Discard any result tail without a manifest commit.
        try:
            if not checkpoint.finished:
                with LibuipcRunSession(
                    config, endpoint, root, run_dir, headless=headless, assets_root=assets_root
                ) as session:
                    session.checkpoint_run_id = checkpoint.run_id
                    for index, entry in enumerate(plan):
                        if checkpoint.is_completed(entry):
                            session.skip_completed_episode(index)
                            continue
                        result = run_one_episode(session, index, evaluations=evaluations)
                        evaluation = evaluations[-1]
                        evaluation["episode_coord"] = {
                            key: entry[key] for key in ("instance_id", "file_idx", "episode_id")
                        }
                        publish()
                        if result["execution_status"] != "completed" or result.get("scoring_status") != "scored":
                            raise RuntimeError(result.get("error", "episode execution/scoring failed"))
                        bag = result.get("recording")
                        if benchmark.get("record", False) and bag is None:
                            raise RuntimeError("Completed episode has no video recording")
                        checkpoint.commit(evaluation, bag=bag)
            if benchmark.get("record", False) and not checkpoint.recordings_ready:
                from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder

                recorder = LocalRecorder()
                recorder.restore_checkpoint(checkpoint.run_id, recording_paths())
                artifacts = recorder.concat_all(preserve_bags=True)
                checkpoint.finalize_recordings(artifacts)
            if benchmark.get("record", False):
                from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder

                artifacts = [checkpoint.out_dir / item["name"] for item in checkpoint.state["artifacts"] or []]
                LocalRecorder.publish_checkpoint_artifacts(
                    artifacts + [checkpoint.result_path], run_dir, benchmark.get("record_video_dir")
                )
            publish()
        except BaseException as exc:
            publish(exc)
            raise


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--infer-host", "--benchmark.infer_host", dest="infer_host")
    parser.add_argument("--benchmark.num_episode", "--episodes", dest="episodes", type=int)
    parser.add_argument("--benchmark.num_instances", dest="instances", type=int)
    parser.add_argument("--app.headless", dest="headless", type=_boolean)
    parser.add_argument("--benchmark.seed", "--seed", dest="seed", type=int)
    parser.add_argument("--generalization.material.enable", dest="material_enable", type=_boolean)
    parser.add_argument(
        "--material-generalization", dest="material_enable", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--output-dir", "--benchmark.output_dir", dest="output_dir", type=Path)
    parser.add_argument("--benchmark.record", dest="record", type=_boolean)
    parser.add_argument("--benchmark.record_state", dest="record_state", type=_boolean)
    parser.add_argument("--benchmark.record_video_dir", dest="record_video_dir")
    parser.add_argument("--benchmark.resume", dest="resume", type=_boolean)
    parser.add_argument("--benchmark.resume_id", dest="resume_id")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--save-infer-images", action="store_true", help="compatibility flag; exact inputs are always archived"
    )
    parser.add_argument("--scoring-execution", choices=("offline", "streaming"))
    parser.add_argument("--scoring-drain-timeout-s", type=float)
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=None)
    for key, kind in (
        ("max_steps", int),
        ("action_chunk_steps", int),
        ("fps", float),
        ("wait_timeout", float),
        ("infer_timeout", float),
    ):
        flags = ["--" + key.replace("_", "-")]
        if key == "fps":
            flags.append("--benchmark.fps")
        parser.add_argument(*flags, dest=key, type=kind)
    return parser


def main(argv=None):
    parser = make_parser()
    options = parser.parse_args(argv)
    run_dir = None
    try:
        config = load_config(options.config, repo_root=options.repo_root)
        for key in (
            "max_steps",
            "action_chunk_steps",
            "fps",
            "wait_timeout",
            "infer_timeout",
            "scoring_execution",
            "scoring_drain_timeout_s",
            "stop_on_success",
        ):
            value = getattr(options, key)
            if value is not None:
                config["libuipc"][key] = value
        if options.episodes is not None:
            config["benchmark"]["num_episode"] = options.episodes
        if options.instances is not None:
            config["benchmark"]["num_instances"] = options.instances
            offsets = config["libuipc"].get("instance_base_xy_offsets")
            if offsets is not None and 1 <= options.instances <= len(offsets):
                config["libuipc"]["instance_base_xy_offsets"] = offsets[: options.instances]
        if options.seed is not None:
            config["benchmark"]["seed"] = options.seed
        if options.material_enable is not None:
            config.setdefault("generalization", {}).setdefault("material", {})["enable"] = options.material_enable
        for key in ("record", "record_state", "resume", "resume_id", "record_video_dir"):
            value = getattr(options, key)
            if value is not None:
                config["benchmark"][key] = value
        validate_settings(config)
        root = Path(config["repo_root"])
        # Revalidate request duration after application-only CLI overrides.
        from genie_sim_client import IPCClientConfig

        rate = IPCClientConfig.from_files(
            config["libuipc"]["evaluation_scene"], config["libuipc"]["launcher"]
        ).physics_hz
        import math

        if rate / config["libuipc"]["fps"] < 1 or not math.isclose(
            rate / config["libuipc"]["fps"], round(rate / config["libuipc"]["fps"]), abs_tol=1e-7, rel_tol=0
        ):
            raise ValueError("fps requires an integer count of YAML IPC steps")
        if options.check_only:
            from .scoring import prepare_assets

            from .materials import create_material_plan
            import yaml

            assets = prepare_assets(config, root, assets_root=options.assets_root)
            assets["material_generalization"] = create_material_plan(
                config,
                yaml.safe_load(Path(config["libuipc"]["evaluation_scene"]).read_text()),
                assets["assets_root"],
                episode_index=0,
                episode_id="check-only",
            )
            print(json.dumps(assets, indent=2))
            return 0
        if not options.infer_host:
            raise ValueError("--infer-host is required for downstream evaluation")
        if not (root / "devel/setup.bash").is_file():
            raise ValueError("prepared ROS overlay missing; build and source it before evaluation")
        benchmark = config["benchmark"]
        explicit_output = options.output_dir or benchmark.get("output_dir")
        output = Path(explicit_output) if explicit_output else root / config["libuipc"]["output_dir"]
        if explicit_output:
            run_dir = output.expanduser().resolve()
        elif benchmark.get("resume_id"):
            run_dir = output.resolve() / benchmark["resume_id"]
        elif benchmark.get("resume", False):
            run_dir = output.resolve()
        else:
            run_dir = output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex)
        run_dir.mkdir(parents=True, exist_ok=True)
        headless = options.headless if options.headless is not None else config.get("app", {}).get("headless", True)
        evaluate(config, options.infer_host, root, run_dir, headless=headless, assets_root=options.assets_root)
        print(f"results: {run_dir}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"libuipc evaluation failed: {exc}", file=sys.stderr)
        return CHECKPOINT_REJECTED_EXIT_CODE if isinstance(exc, CheckpointRejected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
