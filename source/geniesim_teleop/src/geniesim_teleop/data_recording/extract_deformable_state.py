# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Convert deformable-state ROS topics into replay ``state.json`` + NPZ."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np

from geniesim_teleop.data_recording.deformable_state import (
    DEFAULT_TOPIC_PREFIX,
    DeformableStateDecoder,
    stamp_to_nanoseconds,
    topic_names,
)

JOINT_TOPIC = "/joint_states"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"


def _sample_epoch(records, fps: float):
    """Select records nearest a uniform bag-time grid, allowing repeats."""
    if not records:
        return []
    if fps <= 0.0:
        raise ValueError("output fps must be positive")
    stamps = [int(record[0]) for record in records]
    period_ns = 1_000_000_000.0 / float(fps)
    target_ns = float(stamps[0])
    last_target_ns = float(stamps[-1]) + 0.5 * period_ns
    sampled = []
    while target_ns <= last_target_ns:
        insertion = bisect.bisect_left(stamps, target_ns)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(records)]
        if not candidates:
            break
        index = min(candidates, key=lambda value: abs(stamps[value] - target_ns))
        sampled.append((int(round(target_ns)), records[index]))
        target_ns += period_ns
    return sampled


def _nearest_by_bag_stamp(records, target_bag_stamp: int, label: str, max_delta_ns=100_000_000):
    """Return the message nearest ``target_bag_stamp`` in receive-time order."""
    if not records:
        raise ValueError(f"bag has no {label} messages")
    bag_stamps = [int(record[0]) for record in records]
    insertion = bisect.bisect_left(bag_stamps, int(target_bag_stamp))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(records)]
    index = min(candidates, key=lambda value: abs(bag_stamps[value] - target_bag_stamp))
    delta_ns = abs(bag_stamps[index] - int(target_bag_stamp))
    if delta_ns > int(max_delta_ns):
        raise ValueError(f"nearest {label} message is {delta_ns / 1.0e6:.3f} ms from deformable state")
    return records[index][2]


def _matrix_from_transform(transform):
    """Convert a ROS Transform into a homogeneous parent-to-child matrix."""
    translation = transform.translation
    orientation = transform.rotation
    w, x, y, z = (
        float(orientation.w),
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
    )
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("transform contains an invalid quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation.z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _robot_frame(joint_message, odom_message, odom_child_to_robot_root=None):
    """Build the joint-mode robot payload consumed by KinematicReplay."""
    pose_message = odom_message.pose.pose
    position = pose_message.position
    orientation = pose_message.orientation
    w, x, y, z = (
        float(orientation.w),
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
    )
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("odometry contains an invalid robot quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    pose = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(position.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(position.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(position.z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if odom_child_to_robot_root is not None:
        pose = pose @ np.asarray(odom_child_to_robot_root, dtype=np.float64)
    return {
        "pose": pose.tolist(),
        "joints": {
            "joint_name": [str(name) for name in joint_message.name],
            "joint_position": np.asarray(joint_message.position, dtype=np.float64).tolist(),
            "joint_velocity": np.asarray(joint_message.velocity, dtype=np.float64).tolist(),
            "joint_effort": np.asarray(joint_message.effort, dtype=np.float64).tolist(),
        },
    }


def _normalize_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


def _robot_root_offset(
    odom_message,
    target_bag_stamp: int,
    dynamic_transforms,
    static_transforms,
    robot_root_frame: str = "base_link",
):
    """Resolve odometry child frame to the root frame used by the replay USD."""
    odom_child = _normalize_frame_id(getattr(odom_message, "child_frame_id", ""))
    robot_root = _normalize_frame_id(robot_root_frame)
    if not odom_child or odom_child == robot_root:
        return np.eye(4, dtype=np.float64)
    edge = (odom_child, robot_root)
    reverse_edge = (robot_root, odom_child)
    if edge in static_transforms:
        return _matrix_from_transform(static_transforms[edge])
    if reverse_edge in static_transforms:
        return np.linalg.inv(_matrix_from_transform(static_transforms[reverse_edge]))
    if edge in dynamic_transforms:
        transform = _nearest_by_bag_stamp(dynamic_transforms[edge], target_bag_stamp, f"TF {odom_child}->{robot_root}")
        return _matrix_from_transform(transform)
    if reverse_edge in dynamic_transforms:
        transform = _nearest_by_bag_stamp(
            dynamic_transforms[reverse_edge], target_bag_stamp, f"TF {robot_root}->{odom_child}"
        )
        return np.linalg.inv(_matrix_from_transform(transform))
    raise ValueError(
        f"bag has no direct TF between odometry child '{odom_child}' " f"and replay robot root '{robot_root}'"
    )


def _split_epochs(records):
    """Group timestamped records whenever simulation time moves backwards."""
    epochs = [[]]
    last_stamp = None
    for record in records:
        stamp_ns = record[1]
        if last_stamp is not None and stamp_ns < last_stamp:
            epochs.append([])
        epochs[-1].append(record)
        last_stamp = stamp_ns
    return [epoch for epoch in epochs if epoch]


def _state_document(
    decoder,
    frames,
    scene_name: str,
    scene_usd: str,
    robot_name: str,
    output_fps: float,
    sim_raw_data_version: str,
):
    stamps = np.asarray([frame["time_stamp"] for frame in frames], dtype=np.float64)
    positive_dt = np.diff(stamps)
    positive_dt = positive_dt[positive_dt > 0.0]
    fps = float(1.0 / np.median(positive_dt)) if len(positive_dt) else 0.0
    paths = list(decoder.info.get("body_paths") or [])
    paths.extend(str(item["parent_prim_path"]) for item in decoder.instances)
    paths = list(dict.fromkeys(paths))
    return {
        "schema": "simubotix.agibot.com/episode/v6",
        "scene": {
            "name": scene_name,
            "metadata": None,
            "scene_usd": scene_usd,
            "scene_glb": "",
        },
        "lights": [],
        "objects": [{"name": path.rsplit("/", 1)[-1], "prim_path": path, "metadata": None} for path in paths],
        "articulated_objects": [],
        "cameras": {},
        "robot": {"name": robot_name, "metadata": None},
        "frames": frames,
        "fps": float(output_fps) if output_fps > 0.0 else fps,
        "sim_raw_data_version": sim_raw_data_version,
    }


def extract_deformable_state(
    bag_path: str | Path,
    output_dir: str | Path,
    *,
    scene_name: str,
    scene_usd: str,
    robot_name: str,
    topic_prefix: str = DEFAULT_TOPIC_PREFIX,
    joint_topic: str = JOINT_TOPIC,
    odom_topic: str = ODOM_TOPIC,
    tf_topic: str = TF_TOPIC,
    tf_static_topic: str = TF_STATIC_TOPIC,
    robot_root_frame: str = "base_link",
    output_fps: float = 30.0,
    sim_raw_data_version: str = "deformable_v1_world",
) -> list[Path]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    bag_path = Path(bag_path).resolve()
    output_dir = Path(output_dir).resolve()
    stream_topics = topic_names(topic_prefix)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    info_message = None
    mapping_message = None
    states = []
    rigids = []
    joints = []
    odometry = []
    dynamic_transforms = {}
    static_transforms = {}
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        for connection, bag_stamp_ns, raw in reader.messages():
            if connection.topic == stream_topics["info"]:
                info_message = reader.deserialize(raw, connection.msgtype)
            elif connection.topic == stream_topics["mapping"] and mapping_message is None:
                mapping_message = reader.deserialize(raw, connection.msgtype)
            elif connection.topic == stream_topics["state"]:
                message = reader.deserialize(raw, connection.msgtype)
                states.append((int(bag_stamp_ns), stamp_to_nanoseconds(message.header.stamp), message))
            elif connection.topic == stream_topics["rigid_state"]:
                message = reader.deserialize(raw, connection.msgtype)
                rigids.append((int(bag_stamp_ns), stamp_to_nanoseconds(message.header.stamp), message))
            elif connection.topic == joint_topic:
                message = reader.deserialize(raw, connection.msgtype)
                joints.append((int(bag_stamp_ns), stamp_to_nanoseconds(message.header.stamp), message))
            elif connection.topic == odom_topic:
                message = reader.deserialize(raw, connection.msgtype)
                odometry.append((int(bag_stamp_ns), stamp_to_nanoseconds(message.header.stamp), message))
            elif connection.topic in (tf_topic, tf_static_topic):
                message = reader.deserialize(raw, connection.msgtype)
                for stamped_transform in message.transforms:
                    edge = (
                        _normalize_frame_id(stamped_transform.header.frame_id),
                        _normalize_frame_id(stamped_transform.child_frame_id),
                    )
                    if connection.topic == tf_static_topic:
                        static_transforms[edge] = stamped_transform.transform
                    else:
                        dynamic_transforms.setdefault(edge, []).append(
                            (int(bag_stamp_ns), int(bag_stamp_ns), stamped_transform.transform)
                        )

    required_streams = {
        stream_topics["info"]: info_message is not None,
        stream_topics["mapping"]: mapping_message is not None,
        stream_topics["state"]: bool(states),
        stream_topics["rigid_state"]: bool(rigids),
        joint_topic: bool(joints),
        odom_topic: bool(odometry),
    }
    missing = [topic for topic, present in required_streams.items() if not present]
    if missing:
        raise ValueError(f"bag is missing required topics: {', '.join(missing)}")
    decoder = DeformableStateDecoder.from_messages(info_message, mapping_message)
    state_epochs = _split_epochs(states)
    rigid_epochs = _split_epochs(rigids)
    if len(state_epochs) != len(rigid_epochs):
        raise ValueError("deformable rigid and soft streams contain different reset epochs")

    outputs = []
    for epoch_index, (state_epoch, rigid_epoch) in enumerate(zip(state_epochs, rigid_epochs)):
        epoch_dir = output_dir if len(state_epochs) == 1 else output_dir / f"episode_{epoch_index:03d}"
        rigid_by_stamp = {}
        for _bag_stamp, sim_stamp, message in rigid_epoch:
            if sim_stamp in rigid_by_stamp:
                raise ValueError(f"duplicate deformable rigid timestamp in epoch {epoch_index}: {sim_stamp}")
            rigid_by_stamp[sim_stamp] = message
        state_stamps = [record[1] for record in state_epoch]
        if set(state_stamps) != set(rigid_by_stamp):
            missing_rigid = sorted(set(state_stamps) - set(rigid_by_stamp))
            missing_soft = sorted(set(rigid_by_stamp) - set(state_stamps))
            raise ValueError(
                f"deformable rigid/soft timestamp sets differ in epoch {epoch_index}: "
                f"missing_rigid={missing_rigid[:3]} missing_soft={missing_soft[:3]}"
            )

        frames = []
        sampled_state_epoch = _sample_epoch(state_epoch, output_fps)
        for frame_number, (target_bag_stamp, source_record) in enumerate(
            sampled_state_epoch,
            start=1,
        ):
            _bag_stamp, sim_stamp, state_message = source_record
            written_stamp, objects, deformable = decoder.write_frame(
                state_message,
                rigid_by_stamp[sim_stamp],
                epoch_dir,
                frame_number,
            )
            if written_stamp != sim_stamp:
                raise AssertionError("decoder changed the source simulation timestamp")
            joint_message = _nearest_by_bag_stamp(joints, target_bag_stamp, "joint state")
            odom_message = _nearest_by_bag_stamp(odometry, target_bag_stamp, "odometry")
            robot_root_offset = _robot_root_offset(
                odom_message,
                target_bag_stamp,
                dynamic_transforms,
                static_transforms,
                robot_root_frame,
            )
            playback_stamp = state_epoch[0][1] + (target_bag_stamp - state_epoch[0][0])
            frames.append(
                {
                    "objects": objects,
                    "articulated_object": {},
                    "cameras": {},
                    "ee": {},
                    "robot": _robot_frame(
                        joint_message,
                        odom_message,
                        robot_root_offset,
                    ),
                    "time_stamp": playback_stamp / 1.0e9,
                    "frame_state": 0,
                    "deformable": deformable,
                }
            )

        epoch_dir.mkdir(parents=True, exist_ok=True)
        state_path = epoch_dir / "state.json"
        state_path.write_text(
            json.dumps(
                _state_document(
                    decoder,
                    frames,
                    scene_name,
                    scene_usd,
                    robot_name,
                    output_fps,
                    sim_raw_data_version,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output_info = dict(decoder.info)
        output_info["npz_points_coordinate_space"] = "world"
        (epoch_dir / "deformable_info.json").write_text(
            json.dumps(output_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outputs.append(state_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--topic-prefix", default=DEFAULT_TOPIC_PREFIX)
    parser.add_argument("--joint-topic", default=JOINT_TOPIC)
    parser.add_argument("--odom-topic", default=ODOM_TOPIC)
    parser.add_argument("--tf-topic", default=TF_TOPIC)
    parser.add_argument("--tf-static-topic", default=TF_STATIC_TOPIC)
    parser.add_argument("--robot-root-frame", default="base_link")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sim-raw-data-version", default="deformable_v1_world")
    args = parser.parse_args()
    outputs = extract_deformable_state(
        args.bag,
        args.output,
        scene_name=args.scene_name,
        scene_usd=args.scene_usd,
        robot_name=args.robot_name,
        topic_prefix=args.topic_prefix,
        joint_topic=args.joint_topic,
        odom_topic=args.odom_topic,
        tf_topic=args.tf_topic,
        tf_static_topic=args.tf_static_topic,
        robot_root_frame=args.robot_root_frame,
        output_fps=args.fps,
        sim_raw_data_version=args.sim_raw_data_version,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
