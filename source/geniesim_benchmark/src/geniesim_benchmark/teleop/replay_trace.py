# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Adapt compact source-sample records to the existing replay state file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


_SOURCE_SCHEMA = "genie-master.geniesim-source-samples.v1"
_ADAPTER_REVISION = "geniesim.compact-state-to-replay-state.v1"


class ReplayTraceError(ValueError):
    """The selected source trace cannot be reconstructed without invention."""


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length, name):
    if not isinstance(value, list) or len(value) != length or not all(_finite_number(item) for item in value):
        raise ReplayTraceError(f"{name} must contain {length} finite numbers")
    return [float(item) for item in value]


def _matrix(value, name):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
        or any(not _finite_number(item) for row in value for item in row)
    ):
        raise ReplayTraceError(f"{name} must be one finite 4x4 matrix")
    return [[float(item) for item in row] for row in value]


def _pose_matrix(pose, name):
    if not isinstance(pose, dict):
        raise ReplayTraceError(f"{name} must be an object")
    position = _vector(pose.get("position"), 3, f"{name}.position")
    x, y, z, w = _vector(pose.get("orientation_xyzw"), 4, f"{name}.orientation_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ReplayTraceError(f"{name}.orientation_xyzw must be nonzero")
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), position[0]],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), position[1]],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _robot_joints(value, profile_revision):
    if not isinstance(value, dict) or value.get("joint_basis") != "articulation_q_qdot":
        raise ReplayTraceError("robot_joints must use articulation_q_qdot")
    groups = value.get("groups")
    manifest = value.get("controlled_joints")
    expected = manifest.get("groups") if isinstance(manifest, dict) else None
    if (
        not isinstance(groups, dict)
        or not isinstance(expected, dict)
        or set(groups) != set(expected)
        or manifest.get("profile_revision") != profile_revision
    ):
        raise ReplayTraceError("robot_joints groups or profile do not match controlled_joints")
    names, positions, velocities, units = [], [], [], []
    for group_name, declaration in expected.items():
        group = groups[group_name]
        if not isinstance(group, dict) or not isinstance(declaration, dict):
            raise ReplayTraceError(f"robot_joints group {group_name} is invalid")
        group_names = group.get("names")
        group_positions = group.get("positions")
        group_velocities = group.get("velocities")
        group_units = group.get("units")
        if (
            not isinstance(group_names, list)
            or not group_names
            or group_names != declaration.get("names")
            or group_units != declaration.get("units")
            or len(group_positions or ()) != len(group_names)
            or len(group_velocities or ()) != len(group_names)
            or any(not isinstance(item, str) or not item for item in group_names)
            or any(not _finite_number(item) for item in group_positions or ())
            or any(not _finite_number(item) for item in group_velocities or ())
        ):
            raise ReplayTraceError(f"robot_joints group {group_name} is incomplete")
        names.extend(group_names)
        positions.extend(float(item) for item in group_positions)
        velocities.extend(float(item) for item in group_velocities)
        units.extend(group_units)
    if len(set(names)) != len(names):
        raise ReplayTraceError("robot_joints contain duplicate names")
    return {
        "joint_name": names,
        "joint_position": positions,
        "joint_velocity": velocities,
        "units": units,
        "joint_basis": "articulation_q_qdot",
    }


def _object_frame(objects):
    if not isinstance(objects, list):
        raise ReplayTraceError("sample.objects must be a list")
    result = {}
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ReplayTraceError("sample object identity is invalid")
        object_id = item["id"]
        if object_id in result:
            raise ReplayTraceError(f"duplicate sample object {object_id}")
        diagnostic = item.get("contact_diagnostic")
        body_path = diagnostic.get("body_path") if isinstance(diagnostic, dict) else None
        prim_path = (
            body_path.removesuffix("/entity")
            if isinstance(body_path, str) and body_path.endswith("/entity")
            else f"/Workspace/Objects/{object_id}"
        )
        result[object_id] = {
            "pose": _matrix(item.get("world_pose"), f"object {object_id} world_pose"),
            "pose_frame": "world",
            "prim_path": prim_path,
            **({"pose_prim_path": body_path} if isinstance(body_path, str) else {}),
        }
    return result


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_runtime(runtime, task_file, source_root):
    if not isinstance(runtime, dict):
        raise ReplayTraceError("runtime scenario must be an object")
    required = (
        "scene",
        "scene_sha256",
        "task_config",
        "task_config_sha256",
        "robot_cfg",
        "robot_config",
        "robot_config_sha256",
    )
    if (
        any(not isinstance(runtime.get(key), str) or not runtime[key] for key in required)
        or runtime.get("free_base") is not True
        or runtime.get("robot_cfg") != "G2_90d.json"
    ):
        raise ReplayTraceError("runtime scenario identity is incomplete")
    expected_task = (source_root / runtime["task_config"]).resolve()
    expected_scene = (source_root / runtime["scene"]).resolve()
    expected_robot = (source_root / runtime["robot_config"]).resolve()
    if any(not path.is_relative_to(source_root) for path in (expected_task, expected_scene, expected_robot)):
        raise ReplayTraceError("runtime scenario files escape the player source")
    if task_file.resolve() != expected_task:
        raise ReplayTraceError("task_file does not match the runtime scenario")
    if not expected_task.is_file() or _sha256(expected_task) != runtime["task_config_sha256"]:
        raise ReplayTraceError("task configuration bytes do not match the runtime scenario")
    if not expected_scene.is_file() or _sha256(expected_scene) != runtime["scene_sha256"]:
        raise ReplayTraceError("scene bytes do not match the runtime scenario")
    if not expected_robot.is_file() or _sha256(expected_robot) != runtime["robot_config_sha256"]:
        raise ReplayTraceError("robot configuration bytes do not match the runtime scenario")
    scene_info = runtime.get("scene_info")
    scene_info_sha256 = runtime.get("scene_info_sha256")
    if (scene_info is None) != (scene_info_sha256 is None):
        raise ReplayTraceError("scene_info identity is incomplete")
    scene_object_ids = None
    if scene_info is not None:
        expected_info = (source_root / scene_info).resolve()
        if not expected_info.is_relative_to(source_root):
            raise ReplayTraceError("scene_info escapes the player source")
        if not expected_info.is_file() or _sha256(expected_info) != scene_info_sha256:
            raise ReplayTraceError("scene_info bytes do not match the runtime scenario")
        try:
            scene_layout = json.loads(expected_info.read_text(encoding="utf-8")).get("layout")
        except (OSError, ValueError, AttributeError) as error:
            raise ReplayTraceError("scene_info is not readable JSON") from error
        if not isinstance(scene_layout, dict):
            raise ReplayTraceError("scene_info has no object layout")
        scene_object_ids = set(scene_layout)
    return expected_scene, scene_object_ids


def load_trace_entries(trace_root):
    """Read committed segmented JSONL records without changing their source order."""

    trace_root = Path(trace_root)
    directory_input = trace_root.is_dir()
    paths = [trace_root] if trace_root.is_file() else sorted(trace_root.glob("*.jsonl"))
    if not paths:
        raise ReplayTraceError("source trace has no JSONL segments")
    entries = []
    previous = None
    for path in paths:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        source_ref = None
        reference_path = path.with_suffix(".ref.json")
        if reference_path.is_file():
            try:
                source_ref = json.loads(reference_path.read_text(encoding="utf-8"))
            except ValueError as error:
                raise ReplayTraceError(f"{reference_path} is not JSON") from error
            if (
                not isinstance(source_ref, dict)
                or source_ref.get("sha256") != hashlib.sha256(raw).hexdigest()
                or source_ref.get("count") != len(lines)
            ):
                raise ReplayTraceError(f"{path} differs from its segment reference")
        elif directory_input:
            raise ReplayTraceError(f"{path} has no committed segment reference")
        segment_entries = []
        for line_number, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)
            except ValueError as error:
                raise ReplayTraceError(f"{path}:{line_number} is not JSON") from error
            ordinal = entry.get("ordinal") if isinstance(entry, dict) else None
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
                raise ReplayTraceError(f"{path}:{line_number} has no valid source ordinal")
            if previous is not None and ordinal <= previous:
                raise ReplayTraceError("source ordinals are duplicated or out of order")
            entry["_source_ref"] = source_ref
            entries.append(entry)
            segment_entries.append(entry)
            previous = ordinal
        if (
            source_ref is not None
            and segment_entries
            and (
                source_ref.get("first_ordinal") != segment_entries[0]["ordinal"]
                or source_ref.get("last_ordinal") != segment_entries[-1]["ordinal"]
            )
        ):
            raise ReplayTraceError(f"{path} ordinal range differs from its segment reference")
    return entries


def adapt_trace(
    entries,
    runtime,
    *,
    task_file,
    source_root,
    player_source_sha,
    start_sequence=None,
    end_sequence=None,
    task_attempt_id=None,
):
    """Project one same-binding clip into the existing replay state structure."""

    if (
        not isinstance(player_source_sha, str)
        or len(player_source_sha) != 40
        or any(character not in "0123456789abcdef" for character in player_source_sha)
    ):
        raise ReplayTraceError("player_source_sha must be one lowercase Git SHA")
    source_root = Path(source_root).resolve()
    task_file = Path(task_file).resolve()
    scene_file, scene_object_ids = _verified_runtime(runtime, task_file, source_root)
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReplayTraceError("task configuration is not readable JSON") from error
    if not isinstance(task, dict) or not isinstance(task.get("robot"), dict):
        raise ReplayTraceError("task configuration has no robot")
    raw_samples = []
    for entry in entries:
        record = entry.get("record") if isinstance(entry, dict) else None
        if not isinstance(record, dict) or record.get("schema") != _SOURCE_SCHEMA:
            continue
        if task_attempt_id is not None and record.get("task_attempt_id") != task_attempt_id:
            continue
        sample = record.get("sample")
        sequence = sample.get("sequence") if isinstance(sample, dict) else None
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            continue
        if start_sequence is not None and sequence < start_sequence:
            continue
        if end_sequence is not None and sequence > end_sequence:
            continue
        raw_samples.append((entry["ordinal"], record, entry.get("_source_ref")))
    if not raw_samples:
        raise ReplayTraceError("selected clip has no source samples")

    frames = []
    gaps = []
    selected_start_ordinal = raw_samples[0][0]
    selected_end_ordinal = raw_samples[-1][0]
    clip_ordinals = [
        entry["ordinal"] for entry in entries if selected_start_ordinal <= entry["ordinal"] <= selected_end_ordinal
    ]
    for before, after in zip(clip_ordinals, clip_ordinals[1:]):
        if after > before + 1:
            gaps.append(
                {
                    "kind": "source_ordinal_gap",
                    "after_ordinal": before,
                    "before_ordinal": after,
                    "lost_records": after - before - 1,
                }
            )
    binding = None
    physical_binding = None
    control_bindings = []
    scenario = None
    previous_sequence = None
    previous_time = None
    previous_loss = None
    object_names = None
    record_context = None
    for ordinal, record, source_ref in raw_samples:
        sample = record.get("sample")
        loss = record.get("source_gap_count")
        if isinstance(loss, int) and not isinstance(loss, bool):
            if previous_loss is not None and loss > previous_loss:
                gaps.append(
                    {
                        "kind": "declared_source_loss",
                        "ordinal": ordinal,
                        "lost_records": loss - previous_loss,
                    }
                )
            previous_loss = loss
        if record.get("sampling_error") is True or not isinstance(sample, dict) or sample.get("valid") is not True:
            gaps.append({"kind": "invalid_sample", "ordinal": ordinal})
            continue
        current_binding = sample.get("binding")
        current_scenario = record.get("scenario")
        if not isinstance(current_binding, dict) or not current_binding:
            gaps.append({"kind": "unbound_sample", "ordinal": ordinal, "sequence": sample.get("sequence")})
            continue
        physical_source = sample.get("physical_source")
        binding_text = (
            "robot_id",
            "service_id",
            "run_id",
            "runtime_image_id",
            "runtime_receipt_id",
            "sample_source_id",
            "physical_profile_revision",
            "source_sha",
            "assets_sha",
        )
        generation = current_binding.get("generation")
        if (
            any(not isinstance(current_binding.get(key), str) or not current_binding[key] for key in binding_text)
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
            or len(current_binding["source_sha"]) != 40
            or len(current_binding["assets_sha"]) != 40
            or not current_binding["physical_profile_revision"].startswith("sha256:")
        ):
            raise ReplayTraceError("sample binding identity is incomplete")
        source_fields = (
            "runtime_image_id",
            "runtime_receipt_id",
            "sample_source_id",
            "physical_profile_revision",
            "source_sha",
            "assets_sha",
        )
        if not isinstance(physical_source, dict) or any(
            physical_source.get(key) != current_binding.get(key) for key in source_fields
        ):
            raise ReplayTraceError("sample physical source differs from its binding")
        control_context = sample.get("control_context")
        if not isinstance(control_context, dict) or any(
            control_context.get(key) != current_binding.get(key) for key in ("service_id", "run_id", "generation")
        ):
            raise ReplayTraceError("sample control context differs from its binding")
        capture = physical_source.get("private_robot_capture")
        capture_sample = capture.get("sample") if isinstance(capture, dict) else None
        stage_id = capture.get("stage_id") if isinstance(capture, dict) else None
        if (
            not isinstance(stage_id, int)
            or isinstance(stage_id, bool)
            or stage_id <= 0
            or not isinstance(capture_sample, dict)
            or capture_sample.get("source_id") != current_binding.get("sample_source_id")
            or capture_sample.get("sequence") != sample.get("sequence")
            or capture_sample.get("simulation_time_s") != sample.get("simulation_time_s")
            or capture_sample.get("captured_at_ns") != sample.get("captured_at_ns")
        ):
            raise ReplayTraceError("sample timing differs from its physical capture")
        current_record_context = {
            "task_attempt_id": record.get("task_attempt_id"),
            "evidence_use": record.get("evidence_use"),
            "evaluator_revision": record.get("evaluator_revision"),
            "source_phase": record.get("source_phase"),
        }
        current_physical_binding = {
            key: current_binding.get(key)
            for key in (
                "robot_id",
                "service_id",
                "run_id",
                "runtime_image_id",
                "runtime_receipt_id",
                "sample_source_id",
                "physical_profile_revision",
                "source_sha",
                "assets_sha",
            )
        }
        current_physical_binding["stage_id"] = capture.get("stage_id")
        if binding is None:
            binding = current_binding
            physical_binding = current_physical_binding
            scenario = current_scenario
            record_context = current_record_context
        elif current_physical_binding != physical_binding:
            raise ReplayTraceError("selected clip crosses a run, physical source, stage, or Reset")
        if current_binding not in control_bindings:
            control_bindings.append(current_binding)
        if current_scenario != scenario:
            raise ReplayTraceError("selected clip crosses a scenario identity")
        if current_record_context != record_context:
            raise ReplayTraceError("selected clip crosses a task or recording context")

        sequence = sample["sequence"]
        simulation_time = sample.get("simulation_time_s")
        physics_dt = sample.get("physics_dt_s")
        if (
            not _finite_number(simulation_time)
            or not _finite_number(physics_dt)
            or simulation_time < 0
            or physics_dt <= 0
        ):
            raise ReplayTraceError("sample timing is incomplete")
        if previous_sequence is not None and (sequence <= previous_sequence or simulation_time <= previous_time):
            raise ReplayTraceError("sample sequence or simulation time is not increasing")
        previous_sequence, previous_time = sequence, float(simulation_time)

        objects = _object_frame(sample.get("objects"))
        names = sorted(objects)
        if scene_object_ids is not None and not set(names) <= scene_object_ids:
            raise ReplayTraceError("sample objects are absent from scene_info")
        if object_names is None:
            object_names = names
        elif names != object_names:
            raise ReplayTraceError("sample object membership changed inside the clip")
        root_pose = sample.get("base_pose")
        frame = {
            "source_ordinal": ordinal,
            **({"source_ref": source_ref} if source_ref is not None else {}),
            "sequence": sequence,
            "simulation_time_s": float(simulation_time),
            "physics_dt_s": float(physics_dt),
            "control_context": {
                key: control_context.get(key)
                for key in (
                    "actuation_contract_revision",
                    "service_id",
                    "run_id",
                    "generation",
                    "event_seq",
                    "write_revision",
                    "applied_action",
                )
            },
            "robot": {
                "pose": _pose_matrix(root_pose, "base_pose"),
                "root_pose": {
                    "prim_path": capture.get("frame_paths", {}).get("base"),
                    "position": _vector(root_pose.get("position"), 3, "base_pose.position"),
                    "orientation_xyzw": _vector(
                        root_pose.get("orientation_xyzw"),
                        4,
                        "base_pose.orientation_xyzw",
                    ),
                },
                "joints": _robot_joints(
                    sample.get("robot_joints"),
                    current_binding.get("physical_profile_revision"),
                ),
            },
            "objects": objects,
            "articulated_object": {},
        }
        frames.append(frame)
    if not frames or binding is None:
        raise ReplayTraceError("selected clip has no valid bound state frame")

    actual_start = frames[0]
    actual_end = frames[-1]
    return {
        "scene": {"scene_usd": str(scene_file)},
        "robot": {"name": Path(runtime["robot_cfg"]).stem},
        "objects": [{"name": name} for name in object_names or ()],
        "articulated_objects": [],
        "frames": frames,
        "replay": {
            "purpose": "replay",
            "adapter_revision": _ADAPTER_REVISION,
            "player": {
                "source_sha": player_source_sha,
                "adapter_revision": _ADAPTER_REVISION,
            },
            "record_source": physical_binding,
            "control_bindings": control_bindings,
            "source_schema": _SOURCE_SCHEMA,
            "source_segments": list(
                {
                    json.dumps(frame["source_ref"], sort_keys=True): frame["source_ref"]
                    for frame in frames
                    if "source_ref" in frame
                }.values()
            ),
            "scenario": scenario,
            "record_context": record_context,
            "runtime_scenario": {
                key: runtime.get(key)
                for key in (
                    "task_name",
                    "sub_task_name",
                    "instance_id",
                    "benchmark_seed",
                    "scene",
                    "scene_sha256",
                    "scene_info",
                    "scene_info_sha256",
                    "task_config",
                    "task_config_sha256",
                    "robot_cfg",
                    "robot_config_sha256",
                    "free_base",
                )
                if key in runtime
            },
            "binding": binding,
            "initial_state": {
                "source_ordinal": actual_start["source_ordinal"],
                "sequence": actual_start["sequence"],
                "simulation_time_s": actual_start["simulation_time_s"],
            },
            "clip": {
                "requested_start_sequence": start_sequence,
                "requested_end_sequence": end_sequence,
                "actual_start_sequence": actual_start["sequence"],
                "actual_end_sequence": actual_end["sequence"],
                "start_boundary": "selected-clip",
                "end_boundary": "selected-clip",
            },
            "gaps": gaps,
            "completeness": "partial" if gaps else "bounded-no-internal-gap-observed",
        },
    }


def robot_root_command(frame):
    """Build the existing SetObjectPose robot-root command from one replay frame."""

    robot = frame.get("robot") if isinstance(frame, dict) else None
    root = robot.get("root_pose") if isinstance(robot, dict) else None
    if not isinstance(root, dict):
        return None
    position = _vector(root.get("position"), 3, "root_pose.position")
    xyzw = _vector(root.get("orientation_xyzw"), 4, "root_pose.orientation_xyzw")
    return {
        "prim_path": "robot",
        "position": position,
        "rotation": [xyzw[3], *xyzw[:3]],
    }


def ordered_joint_positions(frame, live_joint_names):
    """Map recorded named q values to the loaded articulation order."""

    robot = frame.get("robot") if isinstance(frame, dict) else None
    joints = robot.get("joints") if isinstance(robot, dict) else None
    names = joints.get("joint_name") if isinstance(joints, dict) else None
    positions = joints.get("joint_position") if isinstance(joints, dict) else None
    if (
        not isinstance(names, list)
        or not isinstance(positions, list)
        or len(names) != len(positions)
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
        or any(not _finite_number(value) for value in positions)
    ):
        raise ReplayTraceError("replay robot joint state is incomplete")
    missing = sorted(set(names) - set(live_joint_names))
    if missing:
        raise ReplayTraceError("replay robot joints are absent from the loaded articulation: " + ", ".join(missing))
    result = [0.0] * len(live_joint_names)
    for name, value in zip(names, positions):
        result[live_joint_names.index(name)] = float(value)
    return result


def frame_delay(previous, current):
    """Return a compact-trace delay, or None for an all-legacy frame pair."""

    after = current.get("simulation_time_s")
    if previous is None:
        if after is None:
            return None
        if not _finite_number(after):
            raise ReplayTraceError("replay frame simulation time is invalid")
        return 0.0
    before = previous.get("simulation_time_s")
    if before is None and after is None:
        return None
    if not _finite_number(before) or not _finite_number(after) or after <= before:
        raise ReplayTraceError("replay frame simulation time is not increasing")
    return float(after - before)


def main():
    parser = argparse.ArgumentParser(description="Adapt compact GenieSim state samples for replay_state")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--runtime-scenario", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--player-source-sha", required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--start-sequence", type=int)
    parser.add_argument("--end-sequence", type=int)
    parser.add_argument("--task-attempt-id")
    args = parser.parse_args()
    entries = load_trace_entries(args.trace_root)
    runtime = json.loads(args.runtime_scenario.read_text(encoding="utf-8"))
    state = adapt_trace(
        entries,
        runtime,
        task_file=args.task_file,
        source_root=args.source_root,
        player_source_sha=args.player_source_sha,
        start_sequence=args.start_sequence,
        end_sequence=args.end_sequence,
        task_attempt_id=args.task_attempt_id,
    )
    args.output_state.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
