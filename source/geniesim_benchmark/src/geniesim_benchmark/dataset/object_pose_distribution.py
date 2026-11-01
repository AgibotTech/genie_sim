#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Download episode metadata and summarize frame-zero object positions.

The input is the JSON exported by the dataset review service. Only
``state.json`` and ``scene/background.usda`` are downloaded, making it possible
to inspect a large collection without first copying videos and joint recordings.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import urlsplit

# The source mesh for every benchmark scissors points from handle to blade tip
# along local -X. These endpoint coordinates are derived from the authored USD
# bounds (equivalently: lowpoly bbox center X - size X / 2).
SCISSORS_BLADE_TIP_LOCAL_X = {
    "benchmark_scissors_001": -0.123103223741,
    "benchmark_scissors_002": -0.160078257322,
    "benchmark_scissors_003": -0.124167785048,
    "benchmark_scissors_004": -0.113818317652,
    "benchmark_scissors_005": -0.168814823031,
    "benchmark_scissors_006": -0.168814823031,
}


class WorkflowError(RuntimeError):
    """Raised when an input or downloaded episode violates the workflow contract."""


def load_episodes(manifest_path: Path) -> list[dict]:
    """Load episode dictionaries from a review export JSON."""

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read manifest {manifest_path}: {exc}") from exc

    episodes = document.get("episodes") if isinstance(document, dict) else document
    if not isinstance(episodes, list):
        raise WorkflowError("manifest must be a list or contain an 'episodes' list")
    if not all(isinstance(episode, dict) for episode in episodes):
        raise WorkflowError("every manifest episode must be an object")
    return episodes


def rclone_source(uri: str, remote: str) -> str:
    """Map an agibot-style URI to an explicitly configured rclone remote."""

    parsed = urlsplit(uri)
    if parsed.scheme not in {"agibot-raw", "agibot-frame", "agibot-review", "zhoupu"}:
        raise WorkflowError(f"unsupported episode URI scheme: {parsed.scheme!r}")
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        raise WorkflowError(f"invalid episode URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise WorkflowError(f"episode URI must not contain query or fragment: {uri!r}")
    if not remote or ":" in remote or "/" in remote:
        raise WorkflowError(f"invalid rclone remote name: {remote!r}")
    return f"{remote}:{parsed.netloc}{parsed.path}"


def episode_key(episode: dict, path_field: str) -> str:
    """Return a stable local directory name for an episode."""

    uri = episode.get(path_field)
    if not isinstance(uri, str) or not uri:
        raise WorkflowError(f"episode has no non-empty {path_field!r}")
    key = Path(urlsplit(uri).path).name
    if not key or key in {".", ".."}:
        raise WorkflowError(f"cannot derive episode key from {uri!r}")
    return key


def download_state_json(
    episode: dict,
    output_root: Path,
    *,
    path_field: str,
    remote: str,
    rclone_config: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    """Download one episode's metadata and return its local ``state.json`` path."""

    uri = episode.get(path_field)
    if not isinstance(uri, str) or not uri:
        raise WorkflowError(f"episode has no non-empty {path_field!r}")
    destination = output_root / "episodes" / episode_key(episode, path_field)
    state_path = destination / "state.json"
    scene_path = destination / "scene" / "background.usda"
    if state_path.is_file() and scene_path.is_file():
        return state_path

    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "rclone",
        "copy",
        "--config",
        str(rclone_config),
        "--include",
        "state.json",
        "--include",
        "scene/background.usda",
        rclone_source(uri, remote),
        str(destination),
    ]
    try:
        runner(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkflowError(f"rclone failed for {uri}: {exc}") from exc
    if not state_path.is_file():
        raise WorkflowError(f"rclone completed but {state_path} is missing")
    return state_path


def _matrix_from_pose(name: str, value: object) -> list[list[float]]:
    if not isinstance(value, dict):
        raise WorkflowError(f"frame-zero object {name!r} is not an object")
    pose = value.get("pose")
    if not (isinstance(pose, list) and len(pose) == 4 and all(isinstance(row, list) and len(row) == 4 for row in pose)):
        raise WorkflowError(f"frame-zero object {name!r} has no 4x4 pose")
    try:
        matrix = [[float(cell) for cell in row] for row in pose]
    except (TypeError, ValueError) as exc:
        raise WorkflowError(f"frame-zero object {name!r} has a non-numeric pose") from exc
    if not all(math.isfinite(cell) for row in matrix for cell in row):
        raise WorkflowError(f"frame-zero object {name!r} has a non-finite pose")
    return matrix


def _position(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [matrix[index][3] for index in range(3)]


def _transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> list[float]:
    return [sum(matrix[row][column] * point[column] for column in range(3)) + matrix[row][3] for row in range(3)]


def _point_in_frame(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> list[float]:
    delta = [point[index] - matrix[index][3] for index in range(3)]
    return [sum(matrix[row][column] * delta[row] for row in range(3)) for column in range(3)]


def _normalize_xy(vector: Sequence[float], label: str) -> list[float]:
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1e-12:
        raise WorkflowError(f"{label} has no XY direction")
    return [vector[0] / norm, vector[1] / norm]


def _angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    return math.degrees(math.acos(dot))


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    if len(quaternion) != 4:
        raise WorkflowError("benchmark quaternion must contain four values")
    x, y, z, w = (float(value) for value in quaternion)
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return [sum(rotation[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _canonical_name(recording_name: str) -> str | None:
    prefix = "World_objects_"
    suffix = "_entity"
    if recording_name.startswith(prefix) and recording_name.endswith(suffix):
        return recording_name[len(prefix) : -len(suffix)]
    return None


def read_frame_zero(state_path: Path) -> dict:
    """Read canonical object, robot and hand poses from frame zero."""

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read state file {state_path}: {exc}") from exc
    frames = state.get("frames") if isinstance(state, dict) else None
    if not isinstance(frames, list) or not frames:
        raise WorkflowError(f"state file {state_path} has no frames")
    objects = frames[0].get("objects") if isinstance(frames[0], dict) else None
    if not isinstance(objects, dict):
        raise WorkflowError(f"state file {state_path} frame zero has no objects map")

    object_poses = {}
    for recording_name, value in objects.items():
        name = _canonical_name(recording_name)
        if name is not None:
            object_poses[name] = _matrix_from_pose(recording_name, value)
    if not object_poses:
        raise WorkflowError(f"state file {state_path} has no canonical objects in frame zero")

    robot = frames[0].get("robot")
    if not isinstance(robot, dict):
        raise WorkflowError(f"state file {state_path} frame zero has no robot map")
    robot_pose = _matrix_from_pose("robot", {"pose": robot.get("pose")})
    ee = frames[0].get("ee")
    if not isinstance(ee, dict):
        raise WorkflowError(f"state file {state_path} frame zero has no ee map")
    hand_poses = {}
    for hand in ("left", "right"):
        hand_poses[hand] = _matrix_from_pose(f"ee.{hand}", ee.get(hand))
    return {"objects": object_poses, "robot_pose": robot_pose, "hand_poses": hand_poses}


def read_frame_zero_positions(state_path: Path) -> dict[str, list[float]]:
    """Compatibility helper returning canonical frame-zero object translations."""

    return {name: _position(pose) for name, pose in read_frame_zero(state_path)["objects"].items()}


def first_pick_hand(episode: dict) -> str:
    """Read the arm used by the first action annotation."""

    steps = episode.get("action_steps")
    text = steps[0].get("action_text", "") if isinstance(steps, list) and steps and isinstance(steps[0], dict) else ""
    lowered = str(text).lower()
    left = "左" in lowered or "left" in lowered
    right = "右" in lowered or "right" in lowered
    if left == right:
        raise WorkflowError(f"cannot determine first pickup hand from action text: {text!r}")
    return "left" if left else "right"


def analyze_scissors_episode(frame_zero: dict, pickup_hand: str) -> dict:
    """Extract scissors, blade, support and selected-hand geometry."""

    poses = frame_zero["objects"]
    scissors = sorted(
        ({"name": name, "position": _position(pose)} for name, pose in poses.items() if "scissors" in name),
        key=lambda item: item["name"],
    )
    supports = sorted(
        ({"name": name, "position": _position(pose)} for name, pose in poses.items() if "building_blocks" in name),
        key=lambda item: item["name"],
    )
    if not scissors:
        raise WorkflowError("frame zero contains no canonical scissors")
    if len(supports) != 2:
        raise WorkflowError(f"expected exactly two support blocks, found {len(supports)}")

    midpoint = [statistics.fmean([item["position"][axis] for item in supports]) for axis in range(3)]
    target = min(
        scissors,
        key=lambda item: math.hypot(item["position"][0] - midpoint[0], item["position"][1] - midpoint[1]),
    )
    offset = [target["position"][axis] - midpoint[axis] for axis in range(3)]
    support_distance = math.dist(supports[0]["position"], supports[1]["position"])
    target_name = target["name"]
    if target_name not in SCISSORS_BLADE_TIP_LOCAL_X:
        raise WorkflowError(f"unknown scissors blade geometry: {target_name}")
    target_pose = poses[target_name]
    blade_direction = _normalize_xy([-target_pose[0][0], -target_pose[1][0]], "scissors blade")
    blade_tip = _transform_point(target_pose, [SCISSORS_BLADE_TIP_LOCAL_X[target_name], 0.0, 0.0])
    support_direction = _normalize_xy(
        [
            supports[1]["position"][0] - supports[0]["position"][0],
            supports[1]["position"][1] - supports[0]["position"][1],
        ],
        "support blocks",
    )
    robot_pose = frame_zero["robot_pose"]
    hand_pose = frame_zero["hand_poses"][pickup_hand]
    hand_position = _position(hand_pose)
    center_to_hand = [hand_position[index] - target["position"][index] for index in range(3)]
    center_to_hand_xy = _normalize_xy(center_to_hand, "center-to-hand vector")
    blade_in_robot = [sum(robot_pose[row][column] * blade_direction[row] for row in range(2)) for column in range(2)]
    blade_in_robot = _normalize_xy(blade_in_robot, "robot-frame blade")
    selected_side_alignment = blade_in_robot[1] if pickup_hand == "left" else -blade_in_robot[1]
    return {
        "first_pick_hand": pickup_hand,
        "scissors": scissors,
        "support_blocks": supports,
        "support_midpoint": midpoint,
        "support_distance": support_distance,
        "target_scissors": target,
        "target_offset_from_support_midpoint": offset,
        "blade_direction_xy": blade_direction,
        "blade_direction_robot_xy": blade_in_robot,
        "blade_heading_robot_deg": math.degrees(math.atan2(blade_in_robot[1], blade_in_robot[0])),
        "blade_tip_position": blade_tip,
        "blade_tip_robot": _point_in_frame(robot_pose, blade_tip),
        "target_center_robot": _point_in_frame(robot_pose, target["position"]),
        "selected_hand_initial_position": hand_position,
        "target_center_relative_to_initial_hand": [
            target["position"][index] - hand_position[index] for index in range(3)
        ],
        "blade_tip_relative_to_initial_hand": [blade_tip[index] - hand_position[index] for index in range(3)],
        "target_center_to_initial_hand_distance": math.dist(target["position"], hand_position),
        "blade_tip_to_initial_hand_distance": math.dist(blade_tip, hand_position),
        "blade_to_selected_hand_angle_deg": _angle_deg(blade_direction, center_to_hand_xy),
        "blade_to_support_00_to_01_angle_deg": _angle_deg(blade_direction, support_direction),
        "blade_toward_selected_side": selected_side_alignment,
    }


def _axis_summary(vectors: Sequence[Sequence[float]]) -> dict:
    axes = ("x", "y", "z")
    result = {"count": len(vectors)}
    for axis_index, axis_name in enumerate(axes):
        values = [vector[axis_index] for vector in vectors]
        result[axis_name] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "pstdev": statistics.pstdev(values),
            "percentiles": _percentiles(values),
        }
    return result


def _percentiles(values: Sequence[float]) -> dict:
    ordered = sorted(values)

    def interpolate(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {f"p{percentile:02d}": interpolate(percentile / 100.0) for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)}


def _scalar_summary(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "pstdev": statistics.pstdev(values),
        "percentiles": _percentiles(values),
    }


def _hand_summary(records: Sequence[dict]) -> dict:
    analyses = [record["analysis"] for record in records]
    return {
        "count": len(analyses),
        "target_scissors_counts": dict(
            sorted(collections.Counter(analysis["target_scissors"]["name"] for analysis in analyses).items())
        ),
        "target_center_robot": _axis_summary([analysis["target_center_robot"] for analysis in analyses]),
        "blade_tip_robot": _axis_summary([analysis["blade_tip_robot"] for analysis in analyses]),
        "blade_direction_robot_xy": _axis_summary(
            [[*analysis["blade_direction_robot_xy"], 0.0] for analysis in analyses]
        ),
        "target_center_relative_to_initial_hand": _axis_summary(
            [analysis["target_center_relative_to_initial_hand"] for analysis in analyses]
        ),
        "blade_tip_relative_to_initial_hand": _axis_summary(
            [analysis["blade_tip_relative_to_initial_hand"] for analysis in analyses]
        ),
        "blade_heading_robot_deg": _scalar_summary([analysis["blade_heading_robot_deg"] for analysis in analyses]),
        "target_center_to_initial_hand_distance": _scalar_summary(
            [analysis["target_center_to_initial_hand_distance"] for analysis in analyses]
        ),
        "blade_tip_to_initial_hand_distance": _scalar_summary(
            [analysis["blade_tip_to_initial_hand_distance"] for analysis in analyses]
        ),
        "blade_to_selected_hand_angle_deg": _scalar_summary(
            [analysis["blade_to_selected_hand_angle_deg"] for analysis in analyses]
        ),
        "blade_to_support_00_to_01_angle_deg": _scalar_summary(
            [analysis["blade_to_support_00_to_01_angle_deg"] for analysis in analyses]
        ),
        "blade_toward_selected_side": _scalar_summary(
            [analysis["blade_toward_selected_side"] for analysis in analyses]
        ),
    }


def build_report(records: Sequence[dict], failures: Sequence[dict], requested_count: int) -> dict:
    """Build a deterministic aggregate report from successful episode records."""

    report = {
        "requested_episode_count": requested_count,
        "analyzed_episode_count": len(records),
        "failed_episode_count": len(failures),
        "failures": sorted(failures, key=lambda item: item["episode_key"]),
        "episodes": sorted(records, key=lambda item: item["episode_key"]),
        "distributions": {},
    }
    if not records:
        return report

    report["distributions"] = {
        "target_scissors_counts": dict(
            sorted(collections.Counter(record["analysis"]["target_scissors"]["name"] for record in records).items())
        ),
        "target_scissors_position": _axis_summary(
            [record["analysis"]["target_scissors"]["position"] for record in records]
        ),
        "by_first_pick_hand": {},
        "object_positions": {},
        "support_midpoint": _axis_summary([record["analysis"]["support_midpoint"] for record in records]),
        "target_offset_from_support_midpoint": _axis_summary(
            [record["analysis"]["target_offset_from_support_midpoint"] for record in records]
        ),
        "support_distance": {
            "count": len(records),
            "min": min(record["analysis"]["support_distance"] for record in records),
            "max": max(record["analysis"]["support_distance"] for record in records),
            "mean": statistics.fmean(record["analysis"]["support_distance"] for record in records),
            "pstdev": statistics.pstdev(record["analysis"]["support_distance"] for record in records),
        },
    }
    object_vectors: dict[str, list[list[float]]] = {}
    for record in records:
        analysis = record["analysis"]
        for item in analysis["scissors"] + analysis["support_blocks"]:
            object_vectors.setdefault(item["name"], []).append(item["position"])
    report["distributions"]["object_positions"] = {
        name: _axis_summary(object_vectors[name]) for name in sorted(object_vectors)
    }
    hand_records = {
        hand: [record for record in records if record["analysis"]["first_pick_hand"] == hand]
        for hand in ("left", "right")
    }
    report["distributions"]["by_first_pick_hand"] = {
        hand: _hand_summary(selected) for hand, selected in hand_records.items() if selected
    }
    return report


def _load_benchmark_geometry(instance_dir: Path) -> dict:
    try:
        scene = json.loads((instance_dir / "scene_info.json").read_text(encoding="utf-8"))
        instructions = json.loads((instance_dir / "instructions.json").read_text(encoding="utf-8"))
        instruction = instructions["instructions"][0]["instruction"]
        layout = scene["layout"]
        model = scene["selected_scissors_id"]
        center = [float(value) for value in layout[model]["xyz"]]
        quaternion = layout[model]["xyzw"]
        support_00 = [float(value) for value in layout["benchmark_building_blocks_006_00"]["xyz"]]
        support_01 = [float(value) for value in layout["benchmark_building_blocks_006_01"]["xyz"]]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise WorkflowError(f"cannot read benchmark instance {instance_dir}: {exc}") from exc
    if model not in SCISSORS_BLADE_TIP_LOCAL_X:
        raise WorkflowError(f"benchmark instance {instance_dir.name} uses unknown scissors {model!r}")
    pickup_hand = first_pick_hand({"action_steps": [{"action_text": instruction.split(",", 1)[0]}]})
    tip_delta = _rotate_xyzw(quaternion, [SCISSORS_BLADE_TIP_LOCAL_X[model], 0.0, 0.0])
    blade_tip = [center[index] + tip_delta[index] for index in range(3)]
    blade_direction = _normalize_xy(tip_delta, "benchmark scissors blade")
    return {
        "instance_id": int(instance_dir.name),
        "first_pick_hand": pickup_hand,
        "scissors_model": model,
        "target_center": center,
        "blade_tip": blade_tip,
        "blade_direction_xy": blade_direction,
        "support_00": support_00,
        "support_01": support_01,
    }


def _assembly_points(analysis: dict) -> list[list[float]]:
    supports = {item["name"]: item["position"] for item in analysis["support_blocks"]}
    return [
        analysis["target_scissors"]["position"],
        analysis["blade_tip_position"],
        supports["benchmark_building_blocks_006_00"],
        supports["benchmark_building_blocks_006_01"],
    ]


def _geometry_rms(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    return math.sqrt(statistics.fmean(math.dist(left, right) ** 2 for left, right in zip(first, second)))


def _percentile_value(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _angle_toward_robot(center: Sequence[float], blade_direction: Sequence[float]) -> float:
    robot_direction = _normalize_xy([-center[0], -center[1]], "target-to-robot vector")
    return _angle_deg(blade_direction, robot_direction)


def compare_benchmark_variants(report: dict, variants_root: Path, nearest_count: int = 5) -> dict:
    """Find real episodes nearest to each authored benchmark assembly.

    Candidates must use the same first pickup hand and scissors model. Distance
    is the RMS of four 3D point distances: scissors center, blade tip and both
    support blocks. The blade-tip point makes orientation part of the metric
    without mixing metres and arbitrarily weighted angular units.
    """

    if nearest_count < 1:
        raise WorkflowError("nearest_count must be at least 1")
    instance_dirs = sorted(
        (path for path in variants_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not instance_dirs:
        raise WorkflowError(f"no numeric benchmark instances found in {variants_root}")

    grouped_records: dict[tuple[str, str], list[dict]] = {}
    for record in report.get("episodes", []):
        analysis = record["analysis"]
        key = (analysis["first_pick_hand"], analysis["target_scissors"]["name"])
        grouped_records.setdefault(key, []).append(record)

    group_thresholds = {}
    for key, records in grouped_records.items():
        if len(records) < 2:
            group_thresholds[key] = None
            continue
        nearest_neighbor_distances = []
        for index, record in enumerate(records):
            own_points = _assembly_points(record["analysis"])
            nearest_neighbor_distances.append(
                min(
                    _geometry_rms(own_points, _assembly_points(other["analysis"]))
                    for other_index, other in enumerate(records)
                    if other_index != index
                )
            )
        group_thresholds[key] = _percentile_value(nearest_neighbor_distances, 0.95)

    comparisons = []
    for instance_dir in instance_dirs:
        benchmark = _load_benchmark_geometry(instance_dir)
        key = (benchmark["first_pick_hand"], benchmark["scissors_model"])
        candidates = []
        for record in grouped_records.get(key, []):
            analysis = record["analysis"]
            supports = {item["name"]: item["position"] for item in analysis["support_blocks"]}
            point_distances = {
                "center_distance_m": math.dist(benchmark["target_center"], analysis["target_scissors"]["position"]),
                "blade_tip_distance_m": math.dist(benchmark["blade_tip"], analysis["blade_tip_position"]),
                "support_00_distance_m": math.dist(
                    benchmark["support_00"], supports["benchmark_building_blocks_006_00"]
                ),
                "support_01_distance_m": math.dist(
                    benchmark["support_01"], supports["benchmark_building_blocks_006_01"]
                ),
            }
            geometry_rms = math.sqrt(statistics.fmean(value * value for value in point_distances.values()))
            candidates.append(
                {
                    "episode_index": record["episode_index"],
                    "episode_key": record["episode_key"],
                    "geometry_rms_m": geometry_rms,
                    **point_distances,
                    "blade_angle_deg": _angle_deg(benchmark["blade_direction_xy"], analysis["blade_direction_xy"]),
                }
            )
        candidates.sort(key=lambda item: (item["geometry_rms_m"], item["episode_index"]))
        center_robot = [
            benchmark["target_center"][0] - 0.22738,
            benchmark["target_center"][1],
            benchmark["target_center"][2],
        ]
        angle_to_robot = _angle_toward_robot(center_robot, benchmark["blade_direction_xy"])
        data_angles = [
            _angle_toward_robot(
                record["analysis"]["target_center_robot"], record["analysis"]["blade_direction_robot_xy"]
            )
            for record in grouped_records.get(key, [])
        ]
        geometry_threshold = group_thresholds.get(key)
        reasons = []
        # Nearest-neighbour assembly distance is a similarity diagnostic, not
        # an OOD condition. A valid arrangement can be sparse in the finite
        # sample while its blade direction and selected-hand reach remain
        # supported (instance 00 is one concrete example).
        if data_angles and angle_to_robot < min(data_angles):
            reasons.append("blade_points_more_toward_robot_than_any_matching_episode")
        matching_hand_positions = [
            record["analysis"]["selected_hand_initial_position"] for record in grouped_records.get(key, [])
        ]
        representative_hand_position = (
            [statistics.median(position[axis] for position in matching_hand_positions) for axis in range(3)]
            if matching_hand_positions
            else None
        )
        center_reach = (
            math.dist(benchmark["target_center"], representative_hand_position)
            if representative_hand_position is not None
            else None
        )
        tip_reach = (
            math.dist(benchmark["blade_tip"], representative_hand_position)
            if representative_hand_position is not None
            else None
        )
        data_center_reaches = [
            record["analysis"]["target_center_to_initial_hand_distance"] for record in grouped_records.get(key, [])
        ]
        data_tip_reaches = [
            record["analysis"]["blade_tip_to_initial_hand_distance"] for record in grouped_records.get(key, [])
        ]
        if (
            data_center_reaches
            and center_reach is not None
            and not (min(data_center_reaches) <= center_reach <= max(data_center_reaches))
        ):
            reasons.append("target_center_outside_matching_hand_empirical_reach")
        if (
            data_tip_reaches
            and tip_reach is not None
            and not (min(data_tip_reaches) <= tip_reach <= max(data_tip_reaches))
        ):
            reasons.append("blade_tip_outside_matching_hand_empirical_reach")
        comparisons.append(
            {
                "instance_id": benchmark["instance_id"],
                "first_pick_hand": benchmark["first_pick_hand"],
                "scissors_model": benchmark["scissors_model"],
                "candidate_count": len(candidates),
                "ood": bool(reasons),
                "ood_reasons": reasons,
                "geometry_ood_threshold_m": geometry_threshold,
                "blade_angle_toward_robot_deg": angle_to_robot,
                "matching_data_blade_angle_toward_robot_min_deg": min(data_angles) if data_angles else None,
                "target_center_to_representative_hand_distance_m": center_reach,
                "matching_data_target_center_reach_range_m": (
                    [min(data_center_reaches), max(data_center_reaches)] if data_center_reaches else None
                ),
                "blade_tip_to_representative_hand_distance_m": tip_reach,
                "matching_data_blade_tip_reach_range_m": (
                    [min(data_tip_reaches), max(data_tip_reaches)] if data_tip_reaches else None
                ),
                "nearest": candidates[:nearest_count],
            }
        )
    return {
        "metric": (
            "RMS of 3D distances for scissors center, blade tip, support block 00 and support block 01; "
            "candidates require identical first-pick hand and scissors model"
        ),
        "instances": comparisons,
    }


def run_workflow(
    manifest_path: Path,
    output_root: Path,
    *,
    path_field: str,
    remote: str,
    rclone_config: Path,
    limit: int | None = None,
    benchmark_variants: Path | None = None,
    nearest_count: int = 5,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    """Download selected metadata, analyze it, and persist ``pose_distribution.json``."""

    episodes = load_episodes(manifest_path)
    if limit is not None:
        if limit < 1:
            raise WorkflowError("limit must be at least 1")
        episodes = episodes[:limit]

    records = []
    failures = []
    for index, episode in enumerate(episodes):
        try:
            key = episode_key(episode, path_field)
            state_path = download_state_json(
                episode,
                output_root,
                path_field=path_field,
                remote=remote,
                rclone_config=rclone_config,
                runner=runner,
            )
            analysis = analyze_scissors_episode(read_frame_zero(state_path), first_pick_hand(episode))
            records.append({"episode_index": index, "episode_key": key, "analysis": analysis})
        except WorkflowError as exc:
            raw_key = str(episode.get("episode_id", index))
            try:
                raw_key = episode_key(episode, path_field)
            except WorkflowError:
                pass
            failures.append({"episode_index": index, "episode_key": raw_key, "error": str(exc)})

    report = build_report(records, failures, len(episodes))
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "pose_distribution.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if benchmark_variants is not None:
        comparison = compare_benchmark_variants(report, benchmark_variants, nearest_count)
        comparison_path = output_root / "current_benchmark_nearest.json"
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="dataset review export JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--remote", required=True, help="rclone remote name, for example zhoupu")
    parser.add_argument("--path-field", default="raw_data_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark-variants", type=Path)
    parser.add_argument("--nearest-count", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if shutil.which("rclone") is None:
        print("ERROR: rclone is not on PATH", file=sys.stderr)
        return 2
    if not args.rclone_config.is_file():
        print(f"ERROR: rclone config does not exist: {args.rclone_config}", file=sys.stderr)
        return 2
    try:
        report = run_workflow(
            args.manifest,
            args.output_dir,
            path_field=args.path_field,
            remote=args.remote,
            rclone_config=args.rclone_config,
            limit=args.limit,
            benchmark_variants=args.benchmark_variants,
            nearest_count=args.nearest_count,
        )
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Analyzed {report['analyzed_episode_count']}/{report['requested_episode_count']} episodes; "
        f"report: {args.output_dir / 'pose_distribution.json'}"
    )
    return 1 if report["failed_episode_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
